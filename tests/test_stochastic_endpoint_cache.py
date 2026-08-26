"""Exact categorical oracle for saved-gradient endpoint consistency."""

import itertools

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import elements
import embodied
from dreamerv3.agent import _gradient_cache_payloads
from dreamerv3.rssm import RSSM
from embodied.jax import nets


class _ExplicitCategoryRSSM(RSSM):
  """Actual RSSM internals with only the posterior category made explicit."""

  # Ninjax module fields are collected from the concrete class rather than
  # inherited annotations, so mirror RSSM's declarations for this test shim.
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0

  def observe_categories(
      self, carry, tokens, action, reset, categories, state_taps=None):
    carry, tokens, action = nets.cast((carry, tokens, action))
    entries, features = [], []
    for index in range(tokens.shape[1]):
      step_action = jax.tree.map(lambda x: x[:, index], action)
      tap = None if state_taps is None else jax.tree.map(
          lambda x: x[:, index], state_taps)
      carry, (entry, feat) = self._observe_category(
          carry, tokens[:, index], step_action, reset[:, index],
          categories[:, index], tap)
      entries.append(entry)
      features.append(feat)
    stack = lambda values: jax.tree.map(
        lambda *xs: jnp.stack(xs, 1), *values)
    return carry, stack(entries), stack(features)

  def _observe_category(
      self, carry, tokens, action, reset, category, state_tap=None):
    deter, stoch, action = nets.mask(
        ((carry['deter'] + state_tap['deter']) if state_tap is not None
         else carry['deter'],
         (carry['stoch'] + state_tap['stoch']) if state_tap is not None
         else carry['stoch'],
         action), ~reset)
    action = nets.DictConcat(self.act_space, 1)(action)
    action = nets.mask(action, ~reset)
    deter = self._core(deter, stoch, action)
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for index in range(self.obslayers):
      x = self.sub(f'obs{index}', nets.Linear, self.hidden, **self.kw)(x)
      x = nets.act(self.act)(
          self.sub(f'obs{index}norm', nets.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    # This is the production OneHot straight-through implementation; only its
    # random categorical index is supplied explicitly for finite enumeration.
    stoch = nets.cast(
        self._dist(logit).output._onehot_with_grad(category))
    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nets.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)


def _quality(reference, candidate):
  reference = np.asarray(reference).reshape(-1)
  candidate = np.asarray(candidate).reshape(-1)
  relative = np.linalg.norm(candidate - reference) / max(
      np.linalg.norm(reference), 1e-30)
  cosine = float(np.dot(reference, candidate) / max(
      np.linalg.norm(reference) * np.linalg.norm(candidate), 1e-30))
  return relative, cosine


def _sequence_at(replay, start):
  for chunkid, index in replay.items.values():
    sequence = replay._getseq(chunkid, index)
    if int(sequence['marker'][0]) == start:
      return {key: value[None].copy() for key, value in sequence.items()}
  raise KeyError(start)


def test_independent_categorical_endpoint_is_not_full_bptt():
  """Independent overlap draws differ; coherent physical-row draws are exact."""
  previous_dtype = nets.COMPUTE_DTYPE
  nets.COMPUTE_DTYPE = jnp.float32
  try:
    batch, span, total = 1, 3, 4
    action_space = {
        'action': elements.Space(np.float32, (2,), -1.0, 1.0),
    }
    rssm = _ExplicitCategoryRSSM(
        action_space, deter=4, hidden=5, stoch=1, classes=2,
        blocks=1, norm='none', act='tanh', imglayers=1, obslayers=1,
        dynlayers=1, unimix=0.0, free_nats=0.0, name='rssm')
    initial = {
        'deter': jnp.asarray([[0.20, -0.15, 0.10, -0.05]], jnp.float32),
        'stoch': jax.nn.one_hot(
            jnp.asarray([[0]], jnp.int32), 2, dtype=jnp.float32),
    }
    tokens = jnp.asarray([[
        [0.20, -0.10, 0.05],
        [-0.15, 0.25, 0.10],
        [0.30, 0.05, -0.20],
        [-0.10, -0.20, 0.35],
    ]], jnp.float32)
    actions = {'action': jnp.asarray([[
        [0.10, -0.20],
        [-0.25, 0.15],
        [0.20, 0.30],
        [-0.30, 0.10],
    ]], jnp.float32)}
    resets = jnp.zeros((batch, total), bool)
    pure_rssm = nj.pure(lambda carry, token, action, reset, category, taps=None:
        rssm.observe_categories(
            carry, token, action, reset, category, state_taps=taps))
    create_categories = jnp.zeros((batch, 1, 1), jnp.int32)
    params, _ = pure_rssm(
        {}, initial, tokens[:, :1],
        {key: value[:, :1] for key, value in actions.items()},
        resets[:, :1], create_categories, None,
        seed=jnp.asarray([17, 29], jnp.uint32), create=True)
    params = dict(params)
    # Uniform, genuinely stochastic posterior categories make all paths exactly
    # equiprobable while the hard category still changes the nonlinear core.
    params['rssm/obslogit/kernel'] = jnp.zeros_like(
        params['rssm/obslogit/kernel'])
    params['rssm/obslogit/bias'] = jnp.zeros_like(
        params['rssm/obslogit/bias'])
    # Freeze a one-dimensional nonlinear recurrence with strong dependence on
    # the previous hard category. This makes the covariance destroyed by an
    # independent overlapping-window draw observable rather than numerical
    # noise, while every operation still goes through the actual RSSM core.
    for key in ('rssm/dynin0/kernel', 'rssm/dynin2/kernel'):
      params[key] = jnp.zeros_like(params[key])
    params['rssm/dynin1/kernel'] = jnp.zeros_like(
        params['rssm/dynin1/kernel']).at[0, 0].set(2.0).at[1, 0].set(-2.0)
    params['rssm/dynhid0/kernel'] = jnp.zeros_like(
        params['rssm/dynhid0/kernel'])
    params['rssm/dynhid0/kernel'] = params['rssm/dynhid0/kernel'].at[
        0, 0, 0].set(0.9).at[0, 9, 0].set(1.2)
    params['rssm/dyngru/kernel'] = jnp.zeros_like(
        params['rssm/dyngru/kernel']).at[0, 0, 4].set(1.7)
    params['rssm/dyngru/bias'] = jnp.zeros_like(
        params['rssm/dyngru/bias']).at[:4].set(4.0).at[8:].set(2.4)

    deter_weight = jnp.asarray([0.8, -0.6, 0.4, 1.0], jnp.float32)
    stoch_weight = jnp.asarray([[-0.7, 0.9]], jnp.float32)

    def terminal_loss(feat):
      prediction = (
          (feat['deter'][:, -1] * deter_weight).sum(-1) +
          (feat['stoch'][:, -1] * stoch_weight).sum((-1, -2)))
      return (((prediction - 0.35) ** 2).mean() / span)

    def run(carry, start, stop, categories, taps=None):
      categories = jnp.asarray(categories, jnp.int32).reshape(
          (batch, stop - start, 1))
      _, result = pure_rssm(
          params, carry, tokens[:, start:stop],
          {key: value[:, start:stop] for key, value in actions.items()},
          resets[:, start:stop], categories, taps,
          seed=jnp.asarray([31, 43], jnp.uint32), create=False)
      return result

    _, _, uniform_feat = run(initial, 0, total, (0, 0, 0, 0))
    np.testing.assert_array_equal(uniform_feat['logit'], 0.0)

    def make_replay(states):
      replay = embodied.replay.Replay(
          length=span + 1, capacity=16, chunksize=2, online=False,
          atomic_updates=True, seed=53)
      for index in range(total + 1):
        replay.add({
            'marker': np.asarray(index, np.int32),
            'is_first': np.asarray(index == 0, bool),
            'is_last': np.asarray(index == total, bool),
            'is_terminal': np.asarray(index == total, bool),
            'dyn/deter': np.asarray(states['deter'][0, index]),
            'dyn/stoch': np.asarray(states['stoch'][0, index]),
            'grad/deter': np.zeros(initial['deter'].shape[1:], np.float32),
            'grad/stoch': np.zeros(initial['stoch'].shape[1:], np.float32),
            'grad/valid': np.asarray(index == total, bool),
        })
      return replay

    def backup(replay, start, categories, has_terminal_loss):
      data = _sequence_at(replay, start)
      np.testing.assert_array_equal(
          data['marker'][0], np.arange(start, start + span + 1))
      carry = {
          key: jnp.asarray(data[f'dyn/{key}'][:, 0])
          for key in ('deter', 'stoch')}
      future_valid = (
          data['grad/valid'][:, -1].astype(bool) &
          ~data['is_last'][:, -1].astype(bool))
      future = {
          key: jnp.where(
              future_valid.reshape(
                  future_valid.shape +
                  (1,) * (data[f'grad/{key}'].ndim - 2)),
              jnp.asarray(data[f'grad/{key}'][:, -1]),
              jnp.zeros_like(jnp.asarray(data[f'grad/{key}'][:, -1])))
          for key in ('deter', 'stoch')}
      taps = jax.tree.map(
          lambda value: jnp.zeros(
              (batch, span, *value.shape[1:]), jnp.float32), carry)

      def objective(value):
        final, entries, feat = run(
            carry, start, start + span, categories, value)
        local = terminal_loss(feat) if has_terminal_loss else jnp.asarray(
            0.0, jnp.float32)
        centered = jax.tree.map(
            lambda item: item - jax.lax.stop_gradient(item), final)
        boundary = sum(
            (jax.lax.stop_gradient(future[key]) * centered[key])
            .reshape((batch, -1)).sum(-1)
            for key in ('deter', 'stoch')).mean()
        return local + boundary, (final, entries)

      (_, (_, entries)), incoming = jax.value_and_grad(
          objective, has_aux=True)(taps)
      incoming = jax.tree.map(lambda value: batch * value, incoming)
      terminal = data['is_last'][:, :-1].astype(bool)
      incoming_finite = np.ones(terminal.shape, bool)
      for value in incoming.values():
        incoming_finite &= np.isfinite(np.asarray(value)).reshape(
            (batch, span, -1)).all(-1)
      incoming = jax.tree.map(
          lambda value: jnp.where(
              (terminal | ~incoming_finite).reshape(
                  terminal.shape + (1,) * (value.ndim - 2)),
              jnp.zeros_like(value), value), incoming)
      state_updates = {
          'stepid': data['stepid'][:, 1:],
          'dyn/deter': entries['deter'],
          'dyn/stoch': entries['stoch'],
      }
      grad_updates = {
          'stepid': data['stepid'][:, :-1],
          'grad/deter': incoming['deter'],
          'grad/stoch': incoming['stoch'],
          'grad/valid': incoming_finite | terminal,
      }
      payloads = _gradient_cache_payloads(
          data['stepid'], state_updates, grad_updates)
      payloads = tuple({
          key: np.asarray(value) for key, value in payload.items()
      } for payload in payloads)
      return payloads, incoming, entries

    full_messages = []
    independent_messages = []
    coherent_paths = 0
    for c1, a2, a3, b2, b3, c4 in itertools.product((0, 1), repeat=6):
      categories = (c1, a2, a3, c4)
      _, full_entries, _ = run(initial, 0, total, categories)
      states = jax.tree.map(
          lambda first, rest: jnp.concatenate([first[:, None], rest], 1),
          initial, full_entries)
      replay = make_replay(states)

      reference = jax.grad(lambda carry: terminal_loss(
          run(carry, 0, total, categories)[2]))(initial)
      producer, producer_incoming, producer_entries = backup(
          replay, 1, (b2, b3, c4), True)

      # A pending producer result is not visible until Replay.update().
      before = _sequence_at(replay, 0)
      assert not bool(before['grad/valid'][0, -1])
      replay.update(producer)
      after = _sequence_at(replay, 0)
      assert bool(after['grad/valid'][0, -1])
      for leaf in ('deter', 'stoch'):
        np.testing.assert_array_equal(
            after[f'grad/{leaf}'][0, -1],
            np.asarray(producer_incoming[leaf][0, span - 1]))
        np.testing.assert_array_equal(
            after[f'dyn/{leaf}'][0, -1],
            np.asarray(producer_entries[leaf][0, span - 2]))

      consumer, _, _ = backup(replay, 0, (c1, a2, a3), False)
      replay.update(consumer)
      cached = _sequence_at(replay, 0)
      candidate = {
          leaf: cached[f'grad/{leaf}'][:, 0]
          for leaf in ('deter', 'stoch')}
      full_messages.append(reference)
      independent_messages.append(candidate)

      if a2 == b2 and a3 == b3:
        coherent_paths += 1
        for leaf in ('deter', 'stoch'):
          assert np.linalg.norm(np.asarray(reference[leaf])) > 1e-7
          relative, cosine = _quality(reference[leaf], candidate[leaf])
          assert relative < 1e-5, (leaf, categories, relative)
          assert cosine > 0.99999, (leaf, categories, cosine)

    assert coherent_paths == 16
    mean_full = jax.tree.map(
        lambda *values: sum(values) / len(values), *full_messages)
    mean_independent = jax.tree.map(
        lambda *values: sum(values) / len(values), *independent_messages)
    full_flat = np.concatenate([
        np.asarray(value).reshape(-1) for value in jax.tree.leaves(mean_full)])
    independent_flat = np.concatenate([
        np.asarray(value).reshape(-1)
        for value in jax.tree.leaves(mean_independent)])
    mismatch = np.linalg.norm(independent_flat - full_flat) / max(
        np.linalg.norm(full_flat), 1e-30)
    # Fixed before inspecting the fixture: a five-percent expected-adjoint error
    # is already material for a mechanism intended to recover full BPTT.
    assert mismatch > 0.05, mismatch
  finally:
    nets.COMPUTE_DTYPE = previous_dtype
