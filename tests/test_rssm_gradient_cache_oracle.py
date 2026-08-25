"""Frozen FP32 oracles over the actual Dreamer RSSM implementation."""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamerv3.rssm import RSSM
from embodied.jax import nets


def _quality(reference, candidate):
  ref = np.concatenate([
      np.asarray(x).reshape(-1) for x in jax.tree.leaves(reference)])
  got = np.concatenate([
      np.asarray(x).reshape(-1) for x in jax.tree.leaves(candidate)])
  relative = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)
  cosine = float(np.dot(ref, got) / max(
      np.linalg.norm(ref) * np.linalg.norm(got), 1e-30))
  return relative, cosine


def test_actual_rssm_over_256_steps_matches_full_bptt():
  previous_dtype = nets.COMPUTE_DTYPE
  nets.COMPUTE_DTYPE = jnp.float32
  try:
    action_space = {
        'action': elements.Space(np.float32, (2,), -1.0, 1.0),
    }
    rssm = RSSM(
        action_space, deter=8, hidden=6, stoch=2, classes=3,
        blocks=2, norm='none', act='tanh', imglayers=1, obslayers=1,
        dynlayers=1, unimix=0.01, free_nats=0.0, name='rssm')
    # Five 65-step learner segments exercise 325 recurrent transitions, beyond
    # the scientific ToyMemory target, while keeping the RSSM deliberately tiny.
    batch, length, segments = 1, 65, 5
    keys = jax.random.split(jax.random.PRNGKey(41), 2 * segments)
    tokens = [jax.random.normal(keys[i], (batch, length, 3))
              for i in range(segments)]
    actions = [dict(action=0.3 * jax.random.normal(
        keys[segments + i], (batch, length, 2))) for i in range(segments)]
    resets = [jnp.zeros((batch, length), bool) for _ in range(segments)]
    seeds = [jnp.asarray([101 + i, 211 + i], jnp.uint32)
             for i in range(segments)]
    initial = {
        'deter': jnp.asarray(
            [[0.10, -0.20, 0.05, 0.12, -0.07, 0.03, 0.09, -0.11]],
            jnp.float32),
        'stoch': jax.nn.one_hot(
            jnp.asarray([[0, 2]], jnp.int32), 3, dtype=jnp.float32),
    }

    def objective(feat):
      deter_weight = jnp.linspace(0.2, 0.9, 8, dtype=jnp.float32)
      stoch_weight = jnp.asarray(
          [[[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2]]], jnp.float32)
      deter = (feat['deter'] * deter_weight).sum(-1) ** 2
      stoch = (feat['stoch'] * stoch_weight).sum((-1, -2))
      return deter.mean() + 0.2 * stoch.mean()

    def segment_program(carry, token, action, reset, taps=None):
      final, _, feat = rssm.observe(
          carry, token, action, reset, training=False, state_taps=taps)
      return objective(feat), final

    pure_segment = nj.pure(segment_program)
    params, _ = pure_segment(
        {}, initial, tokens[0], actions[0], resets[0],
        seed=seeds[0], create=True)

    def run_segment(params, carry, index, taps=None):
      _, (loss, final) = pure_segment(
          params, carry, tokens[index], actions[index], resets[index], taps,
          seed=seeds[index], create=False)
      return loss, final

    taps = {
        key: jnp.zeros((batch, length, *value.shape[1:]), jnp.float32)
        for key, value in initial.items()}

    # Enabling the tap path with a zero future message must be an identity for
    # the native RSSM primal and parameter gradient under an identical seed.
    plain_loss, plain_final = run_segment(params, initial, 0)
    tapped_loss, tapped_final = run_segment(params, initial, 0, taps)
    np.testing.assert_array_equal(plain_loss, tapped_loss)
    for key in ('deter', 'stoch'):
      np.testing.assert_array_equal(plain_final[key], tapped_final[key])
    plain_params = jax.grad(
        lambda values: run_segment(values, initial, 0)[0])(params)
    tapped_params = jax.grad(
        lambda values: run_segment(values, initial, 0, taps)[0])(params)
    relative, cosine = _quality(plain_params, tapped_params)
    assert relative == 0.0
    assert cosine > 0.99999

    # The first actual RSSM tap is an additive view of the incoming boundary.
    # Its cotangent must match direct differentiation of both carry leaves.
    direct = jax.grad(lambda carry: run_segment(params, carry, 0)[0])(initial)
    tapped = jax.grad(
        lambda value: run_segment(params, initial, 0, value)[0])(taps)
    for key in ('deter', 'stoch'):
      np.testing.assert_allclose(
          tapped[key][:, 0], direct[key], rtol=1e-5, atol=1e-6)
      assert np.linalg.norm(np.asarray(direct[key])) > 0

    def rollout(params, carry, start=0):
      total = jnp.array(0.0, jnp.float32)
      for index in range(start, segments):
        local, carry = run_segment(params, carry, index)
        total += local
      return total, carry

    boundaries = [initial]
    for index in range(segments):
      _, state = run_segment(params, boundaries[-1], index)
      boundaries.append(jax.tree.map(jax.lax.stop_gradient, state))

    messages = []
    for index in range(segments):
      if index + 1 < segments:
        message = jax.grad(
            lambda state, start=index + 1: rollout(
                params, state, start)[0])(boundaries[index + 1])
      else:
        message = jax.tree.map(jnp.zeros_like, boundaries[index + 1])
      messages.append(message)

    contributions = []
    for index in range(segments):
      def local_with_boundary(values, index=index):
        local, final = run_segment(values, boundaries[index], index)
        future = sum(
            (messages[index][key] * final[key]).sum()
            for key in ('deter', 'stoch'))
        return local + future
      contributions.append(jax.grad(local_with_boundary)(params))

    segmented = jax.tree.map(lambda *values: sum(values), *contributions)
    reference = jax.grad(lambda values: rollout(
        values, initial)[0])(params)
    relative, cosine = _quality(reference, segmented)
    assert relative < 1e-5
    assert cosine > 0.99999
  finally:
    nets.COMPUTE_DTYPE = previous_dtype
