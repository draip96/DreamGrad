import math
import re

import jax
import jax.numpy as jnp
import ninjax as nj
import ninjax.ninjax as njx
import optax

from . import internal
from . import nets

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


def _grad_with_inputs(
    fun, targets, input_argnums, has_aux=False, separate_input_loss=False):
  """Ninjax state and explicit-input gradients from one stochastic primal.

  Ninjax 3.6 exposes either state/module gradients or argument gradients, but
  saved recurrent adjoints require both from the exact same stochastic
  forward. When ``separate_input_loss`` is true, ``fun`` returns two scalars:
  the parameter objective and the literal state-adjoint objective. Their VJPs
  share one primal evaluation and can use distinct cotangent seeds.
  """
  targets = targets if hasattr(targets, '__len__') else (targets,)
  input_argnums = tuple(input_argnums)
  if not input_argnums:
    raise ValueError('At least one explicit input gradient is required.')

  ctx = nj.context()
  if not has_aux:
    fun = lambda *args, _fun=fun, **kwargs: (_fun(*args, **kwargs), {})
  fun = nj.pure(fun, nested=True)

  def wrapper(*args, **kwargs):
    accessed, modified = njx._prerun(fun, *args, **kwargs)
    strings = []
    for target in targets:
      if isinstance(target, njx.Module):
        prefix = target.path + '/'
        matches = [key for key in ctx if key.startswith(prefix)]
      elif isinstance(target, str):
        pattern = re.compile(f'^{target}(/.*|$)')
        matches = [key for key in ctx if pattern.match(key)]
      else:
        raise TypeError(type(target))
      if not matches:
        existing = ', '.join(ctx.keys())
        raise KeyError(
            f"Gradient target '{target}' did not match any state entries. "
            f'Existing state entries: {existing}')
      strings += matches

    selected = {key: value for key, value in ctx.items() if key in strings}
    other = {key: value for key, value in ctx.items() if key not in strings}
    for key in selected:
      if key not in accessed:
        raise RuntimeError(
            f"Trying to compute gradient with respect to key '{key}' but the "
            f'differentiated function does not access it. Accessed keys: '
            f'{list(accessed)}')
    selected = {key: value for key, value in selected.items()
                if key in accessed}
    other = {key: value for key, value in other.items() if key in accessed}

    def forward(differentiated, nondifferentiated, *inner_args, **inner_kw):
      before = {**differentiated, **nondifferentiated}
      state, (value, aux) = fun(
          before, *inner_args, create=False, **inner_kw)
      changes = {key: value for key, value in state.items()
                 if key in modified}
      return value, (changes, aux)

    seed = nj.seed(None, True)
    if separate_input_loss:
      input_values = tuple(args[index] for index in input_argnums)

      def selected_forward(differentiated, *inner_inputs):
        inner_args = list(args)
        for index, value in zip(input_argnums, inner_inputs):
          inner_args[index] = value
        return forward(
            differentiated, other, *inner_args, seed=seed, **kwargs)

      values, pullback, (changes, aux) = jax.vjp(
          selected_forward, selected, *input_values, has_aux=True)
      assert isinstance(values, (tuple, list)) and len(values) == 2
      param_cotangents = pullback((
          jnp.ones_like(values[0]), jnp.zeros_like(values[1])))
      input_cotangents = pullback((
          jnp.zeros_like(values[0]), jnp.ones_like(values[1])))
      value = values[0]
      param_grads = param_cotangents[0]
      input_grads = input_cotangents[1:]
    else:
      # ``forward`` receives two state dictionaries before user arguments.
      argnums = (0,) + tuple(2 + index for index in input_argnums)
      backward = jax.value_and_grad(
          forward, argnums=argnums, has_aux=True)
      (value, (changes, aux)), gradients = backward(
          selected, other, *args, seed=seed, **kwargs)
      param_grads, *input_grads = gradients
    if ctx.modify:
      ctx.update(changes)
      selected = {key: ctx[key] for key in selected}
    return value, selected, param_grads, aux, tuple(input_grads)

  return wrapper


class Optimizer(nj.Module):

  summary_depth: int = 2

  def __init__(self, modules, opt):
    modules = modules if isinstance(modules, (list, tuple)) else (modules,)
    self.modules = modules
    self.opt = opt
    self.step = nj.Variable(jnp.array, 0, i32, name='step')
    self.scaling = (nets.COMPUTE_DTYPE == jnp.float16)
    if self.scaling:
      self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name='grad_scale')
      self.good_steps = nj.Variable(jnp.array, 0, i32, name='good_steps')

  def __call__(
      self, lossfn, *args, has_aux=False, input_grad_argnums=(),
      separate_input_loss=False, **kwargs):
    metrics = {}

    def lossfn2(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      objectives = loss if separate_input_loss else (loss,)
      assert len(objectives) == (2 if separate_input_loss else 1)
      for objective in objectives:
        assert objective.dtype == f32, (self.name, objective.dtype)
        assert objective.shape == (), (self.name, objective.shape)
      if self.scaling:
        objectives = tuple(
            value * sg(self.grad_scale.read()) for value in objectives)
      loss = objectives if separate_input_loss else objectives[0]
      return loss, aux

    input_grad_argnums = tuple(input_grad_argnums)
    if input_grad_argnums:
      loss, params, grads, aux, input_grads = _grad_with_inputs(
          lossfn2, self.modules, input_grad_argnums, has_aux=True,
          separate_input_loss=separate_input_loss)(
              *args, **kwargs)
    else:
      loss, params, grads, aux = nj.grad(
          lossfn2, self.modules, has_aux=True)(*args, **kwargs)
      input_grads = ()
    if self.scaling:
      loss *= 1 / self.grad_scale.read()

    counts = {k: math.prod(v.shape) for k, v in params.items()}
    if nj.creating():
      print(self._summarize_params(counts, self.summary_depth))

    axes = internal.get_data_axes()
    if axes:
      grads = jax.tree.map(lambda x: jax.lax.pmean(x, axes), grads)
      # Input cotangents belong to distinct replay rows on each device. They
      # remain local and are consumed through a batch mean on a later update;
      # averaging them across devices would mix unrelated cache entries.

    if self.scaling:
      invscale = 1 / self.grad_scale.read()
      grads = jax.tree.map(lambda x: x * invscale, grads)
      input_grads = jax.tree.map(lambda x: x * invscale, input_grads)

    state = self.sub('state', nj.Tree, self.opt.init, params)
    updates, new_state = self.opt.update(grads, state.read(), params)
    nj.context().update(optax.apply_updates(params, updates))
    state.write(new_state)
    grad_norm = optax.global_norm(grads)
    if self.scaling:
      self._update_scale(grads, jnp.isfinite(grad_norm))
      grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
      self.step.write(self.step.read() + i32(jnp.isfinite(grad_norm)))
      metrics['grad_scale'] = self.grad_scale.read()
      metrics['grad_overflow'] = f32(~jnp.isfinite(grad_norm))
    else:
      self.step.write(self.step.read() + 1)
    metrics['loss'] = loss.mean()
    metrics['updates'] = self.step.read()
    metrics['grad_norm'] = grad_norm
    metrics['grad_rms'] = nets.rms(grads)
    metrics['update_rms'] = nets.rms(updates)
    metrics['param_rms'] = nets.rms([x.values for x in self.modules])
    metrics['param_count'] = jnp.array(list(counts.values()), f32).sum()
    metrics = {f'{self.name}/{k}': v for k, v in metrics.items()}
    if input_grad_argnums:
      return ((metrics, aux, input_grads) if has_aux
              else (metrics, input_grads))
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads, finite):
    keep = (finite & (self.good_steps.read() < 1000))
    incr = (finite & (self.good_steps.read() >= 1000))
    decr = ~finite
    self.good_steps.write(i32(keep) * (self.good_steps.read() + 1))
    self.grad_scale.write(jnp.clip(
        f32(keep) * self.grad_scale.read() +
        f32(incr) * self.grad_scale.read() * 2 +
        f32(decr) * self.grad_scale.read() / 2, 1e-4, 1e5))
    return finite

  def _summarize_params(self, counts, depth):
    lines = []
    pfxs = []
    for key in counts:
      parts = key.split('/')
      pfxs += ['/'.join(parts[: i + 1]) for i in range(min(len(parts), depth))]
    subcounts = {
        prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
        for prefix in set(pfxs)}
    lines = [f'Optimizer {self.name} has {sum(counts.values()):,} params:']
    for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
      lines.append(f'{count:>14,} {prefix}')
    return '\n'.join(lines)


def clip_by_agc(clip=0.3, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = jax.tree.map(fn, params, updates) if clip else updates
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    nu = jax.tree.map(
        lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = jax.tree.map(
        lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step)
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)
