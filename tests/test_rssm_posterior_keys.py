"""CPU regressions for replay-stable RSSM posterior sampling keys."""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamerv3.rssm import RSSM
from embodied.jax import nets


@pytest.fixture
def fp32():
  previous = nets.COMPUTE_DTYPE
  nets.COMPUTE_DTYPE = jnp.float32
  try:
    yield
  finally:
    nets.COMPUTE_DTYPE = previous


def _tree_equal(left, right):
  jax.tree.map(
      lambda x, y: np.testing.assert_array_equal(np.asarray(x), np.asarray(y)),
      left, right)


def _tree_take(tree, indices):
  return jax.tree.map(lambda x: x[indices], tree)


def _make_case():
  model = RSSM(
      {'action': elements.Space(np.float32, (2,), -1.0, 1.0)},
      deter=6, hidden=5, stoch=2, classes=3, blocks=2,
      norm='none', act='tanh', imglayers=1, obslayers=1,
      dynlayers=1, unimix=0.01, free_nats=0.0, name='rssm')
  batch, length = 3, 6
  initial = {
      'deter': jnp.asarray(np.linspace(
          -0.2, 0.3, batch * model.deter).reshape(batch, model.deter),
          jnp.float32),
      'stoch': jax.nn.one_hot(
          jnp.asarray([[0, 1], [1, 2], [2, 0]], jnp.int32),
          model.classes, dtype=jnp.float32),
  }
  tokens = jnp.asarray(np.linspace(
      -0.4, 0.5, batch * length * 4).reshape(batch, length, 4),
      jnp.float32)
  actions = {'action': jnp.asarray(np.linspace(
      -0.3, 0.2, batch * length * 2).reshape(batch, length, 2),
      jnp.float32)}
  resets = jnp.zeros((batch, length), bool)
  posterior_keys = jax.random.split(
      jax.random.PRNGKey(17), batch * length).reshape(batch, length, 2)
  state_taps = {
      'deter': jnp.asarray(np.linspace(
          -0.01, 0.01, batch * length * model.deter).reshape(
              batch, length, model.deter), jnp.float32),
      'stoch': jnp.asarray(np.linspace(
          -0.005, 0.005,
          batch * length * model.stoch * model.classes).reshape(
              batch, length, model.stoch, model.classes), jnp.float32),
  }

  def program(carry, token, action, reset, keys, taps=None):
    return model.observe(
        carry, token, action, reset, training=False,
        state_taps=taps, posterior_keys=keys)

  pure = nj.pure(program)
  params, _ = pure(
      {}, initial, tokens, actions, resets, posterior_keys, state_taps,
      seed=jnp.asarray([101, 203], jnp.uint32), create=True)

  def single_program(carry, token, action, reset, keys):
    return model.observe(
        carry, token, action, reset, training=False, single=True,
        posterior_keys=keys)

  pure_single = nj.pure(single_program)

  def run(
      seed, carry=initial, token=tokens, action=actions, reset=resets,
      keys=posterior_keys, taps=None):
    _, outputs = pure(
        params, carry, token, action, reset, keys, taps,
        seed=jnp.asarray(seed, jnp.uint32), create=False)
    return outputs

  def run_single(token, keys):
    _, outputs = pure_single(
        params, initial, token,
        jax.tree.map(lambda x: x[:, 0], actions), resets[:, 0], keys,
        seed=jnp.asarray([1409, 1511], jnp.uint32), create=False)
    return outputs

  return {
      'model': model,
      'initial': initial,
      'tokens': tokens,
      'actions': actions,
      'resets': resets,
      'posterior_keys': posterior_keys,
      'state_taps': state_taps,
      'run': run,
      'run_single': run_single,
  }


def test_same_posterior_keys_ignore_outer_seed_with_state_taps(fp32):
  case = _make_case()
  first = case['run']([301, 401], taps=case['state_taps'])
  second = case['run']([503, 607], taps=case['state_taps'])
  _tree_equal(first, second)


def test_posterior_keys_are_batch_permutation_equivariant(fp32):
  case = _make_case()
  reference = case['run']([701, 809], taps=case['state_taps'])
  order = jnp.asarray([2, 0, 1], jnp.int32)
  permuted = case['run'](
      [811, 907],
      carry=_tree_take(case['initial'], order),
      token=_tree_take(case['tokens'], order),
      action=_tree_take(case['actions'], order),
      reset=_tree_take(case['resets'], order),
      keys=_tree_take(case['posterior_keys'], order),
      taps=_tree_take(case['state_taps'], order))
  _tree_equal(permuted, _tree_take(reference, order))


def test_matching_boundary_and_row_keys_make_suffix_coherent(fp32):
  case = _make_case()
  final, entries, feat = case['run']([1009, 1103])
  split = 2
  boundary = jax.tree.map(lambda x: x[:, split - 1], entries)
  suffix = case['run'](
      [1201, 1301],
      carry=boundary,
      token=case['tokens'][:, split:],
      action=jax.tree.map(lambda x: x[:, split:], case['actions']),
      reset=case['resets'][:, split:],
      keys=case['posterior_keys'][:, split:])
  suffix_final, suffix_entries, suffix_feat = suffix
  _tree_equal(suffix_final, final)
  _tree_equal(suffix_entries, jax.tree.map(lambda x: x[:, split:], entries))
  _tree_equal(suffix_feat, jax.tree.map(lambda x: x[:, split:], feat))


def test_overlapping_keyed_windows_recover_full_bptt_endpoint_message(fp32):
  """Reconstructed overlap uses one physical stochastic realization."""
  case = _make_case()
  total, span = 4, 3
  batch = case['tokens'].shape[0]
  tokens = case['tokens'][:, :total]
  actions = jax.tree.map(lambda x: x[:, :total], case['actions'])
  resets = case['resets'][:, :total]
  keys = case['posterior_keys'][:, :total]

  def run(carry, start, stop, taps=None):
    return case['run'](
        [1601, 1709], carry=carry, token=tokens[:, start:stop],
        action=jax.tree.map(lambda x: x[:, start:stop], actions),
        reset=resets[:, start:stop], keys=keys[:, start:stop], taps=taps)

  deter_weight = jnp.linspace(
      0.2, 0.9, case['model'].deter, dtype=jnp.float32)
  stoch_weight = jnp.asarray([
      [[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2]],
      [[-0.2, 0.5, 0.3], [0.6, -0.4, 0.1]],
      [[0.7, 0.1, -0.5], [-0.3, 0.2, 0.4]],
  ], jnp.float32)

  def terminal_loss(feat):
    prediction = (
        (feat['deter'][:, -1] * deter_weight).sum(-1) +
        (feat['stoch'][:, -1] * stoch_weight).sum((-1, -2)))
    return ((prediction - 0.25) ** 2).mean()

  _, full_entries, _ = run(case['initial'], 0, total)
  boundary_q1 = jax.tree.map(
      lambda x: jax.lax.stop_gradient(x[:, 0]), full_entries)
  reference = jax.grad(
      lambda carry: terminal_loss(run(carry, 0, total)[2]))(
          case['initial'])

  taps = jax.tree.map(
      lambda x: jnp.zeros((batch, span, *x.shape[1:]), jnp.float32),
      boundary_q1)

  def producer_objective(value):
    final, entries, feat = run(boundary_q1, 1, total, value)
    return terminal_loss(feat), (final, entries)

  (_, (_, producer_entries)), incoming = jax.value_and_grad(
      producer_objective, has_aux=True)(taps)
  # The producer's last input tap is physical q3. Cache storage multiplies
  # input gradients by B because both producer and consumer losses are means.
  future_q3 = jax.tree.map(lambda x: batch * x[:, -1], incoming)

  def consumer_objective(carry):
    final, _, _ = run(carry, 0, span)
    centered = jax.tree.map(
        lambda x: x - jax.lax.stop_gradient(x), final)
    terms = [
        (jax.lax.stop_gradient(future_q3[key]) * centered[key])
        .reshape((batch, -1)).sum(-1)
        for key in ('deter', 'stoch')]
    return jnp.stack(terms).sum(0).mean()

  consumer_q3 = run(case['initial'], 0, span)[0]
  producer_q3 = jax.tree.map(lambda x: x[:, -2], producer_entries)
  _tree_equal(consumer_q3, producer_q3)
  candidate = jax.grad(consumer_objective)(case['initial'])
  for key in ('deter', 'stoch'):
    assert np.linalg.norm(np.asarray(reference[key])) > 1e-7
    np.testing.assert_allclose(
        np.asarray(candidate[key]), np.asarray(reference[key]),
        rtol=1e-5, atol=1e-6)


def test_keyed_onehot_sampling_keeps_straight_through_gradients(fp32):
  case = _make_case()
  tokens = case['tokens'][:, 0]
  weights = jnp.asarray([
      [[0.3, -0.2, 0.7], [-0.5, 0.4, 0.1]],
      [[-0.4, 0.8, 0.2], [0.6, -0.1, 0.3]],
      [[0.9, -0.3, 0.2], [0.1, 0.5, -0.7]],
  ], jnp.float32)
  keys1 = jax.random.split(jax.random.PRNGKey(37), len(tokens))
  keys2 = jax.random.split(jax.random.PRNGKey(43), len(tokens))

  def sample(values, keys):
    return case['run_single'](values, keys)[2]['stoch']

  sample1 = sample(tokens, keys1)
  sample2 = sample(tokens, keys2)
  assert np.any(np.asarray(sample1).argmax(-1) != np.asarray(sample2).argmax(-1))
  grad1 = jax.grad(lambda values: (sample(values, keys1) * weights).sum())(
      tokens)
  grad2 = jax.grad(lambda values: (sample(values, keys2) * weights).sum())(
      tokens)
  assert float(jnp.linalg.norm(grad1)) > 0.0
  np.testing.assert_array_equal(np.asarray(grad1), np.asarray(grad2))
