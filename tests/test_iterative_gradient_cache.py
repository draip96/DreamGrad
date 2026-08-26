"""Replay-backed iterative transport over a literal 258-transition path."""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import elements
import embodied
from dreamerv3.agent import _gradient_cache_payloads
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


def _sequence_at(replay, start):
  for chunkid, index in replay.items.values():
    sequence = replay._getseq(chunkid, index)
    if int(sequence['marker'][0]) == start:
      return {key: value[None].copy() for key, value in sequence.items()}
  raise KeyError(start)


def _physical_rows(replay, total):
  rows = {}
  for chunk in replay.chunks.values():
    if not chunk.length:
      continue
    for index, marker in enumerate(chunk.data['marker'][:chunk.length]):
      rows[int(marker)] = {
          key: np.asarray(value[index]).copy()
          for key, value in chunk.data.items()}
  assert sorted(rows) == list(range(total + 1))
  return {
      key: np.stack([rows[index][key] for index in range(total + 1)])
      for key in rows[0]}


def test_actual_rssm_adjoint_iterates_through_physical_replay_rows():
  previous_dtype = nets.COMPUTE_DTYPE
  nets.COMPUTE_DTYPE = jnp.float32
  try:
    batch, span, total = 1, 64, 258
    action_space = {
        'action': elements.Space(np.float32, (2,), -1.0, 1.0),
    }
    rssm = RSSM(
        action_space, deter=8, hidden=6, stoch=2, classes=2,
        blocks=1, norm='none', act='tanh', imglayers=1, obslayers=1,
        dynlayers=1, unimix=0.0, free_nats=0.0, name='rssm')
    initial = {
        'deter': jnp.asarray(
            [[0.10, -0.20, 0.05, 0.12, -0.07, 0.03, 0.09, -0.11]],
            jnp.float32),
        'stoch': jax.nn.one_hot(
            jnp.asarray([[1, 0]], jnp.int32), 2, dtype=jnp.float32),
    }
    keys = jax.random.split(jax.random.PRNGKey(73), 2)
    tokens = 0.05 * jax.random.normal(keys[0], (batch, total, 3))
    actions = {'action': 0.05 * jax.random.normal(
        keys[1], (batch, total, 2))}
    seed = jnp.asarray([109, 211], jnp.uint32)

    def rssm_program(carry, token, action, reset, taps=None):
      return rssm.observe(
          carry, token, action, reset, training=False, state_taps=taps)

    pure_rssm = nj.pure(rssm_program)
    params, _ = pure_rssm(
        {}, initial, tokens[:, :1],
        {key: value[:, :1] for key, value in actions.items()},
        jnp.zeros((batch, 1), bool), seed=seed, create=True)

    # Make posterior samples deterministic across overlapping windows while
    # retaining both categorical state leaves. Keep the GRU update gate small
    # enough that a 258-transition sensitivity remains numerically resolvable.
    params = dict(params)
    params['rssm/obslogit/kernel'] = jnp.zeros_like(
        params['rssm/obslogit/kernel'])
    params['rssm/obslogit/bias'] = jnp.tile(
        jnp.asarray([30.0, -30.0], jnp.float32), 2)
    update_start = 2 * initial['deter'].shape[-1]
    params['rssm/dyngru/kernel'] = params['rssm/dyngru/kernel'].at[
        ..., update_start:].set(0.0)
    params['rssm/dyngru/bias'] = params['rssm/dyngru/bias'].at[
        update_start:].set(-4.0)

    deter_weight = jnp.linspace(0.2, 0.9, 8, dtype=jnp.float32)

    def terminal_loss(feat, weight):
      prediction = (feat['deter'] * deter_weight).sum(-1)
      return (((prediction - 0.35) ** 2) * weight).sum() / span

    def run_case(reset_at=None):
      resets = jnp.zeros((batch, total), bool)
      if reset_at is not None:
        resets = resets.at[:, reset_at - 1].set(True)
      weights = jnp.zeros((batch, total), jnp.float32).at[:, -1].set(1.0)

      def run(carry, start, stop, taps=None):
        _, result = pure_rssm(
            params, carry, tokens[:, start:stop],
            {key: value[:, start:stop] for key, value in actions.items()},
            resets[:, start:stop], taps, seed=seed, create=False)
        return result

      full_final, full_entries, _ = run(initial, 0, total)
      del full_final
      states = jax.tree.map(
          lambda first, rest: jnp.concatenate([first[:, None], rest], 1),
          initial, full_entries)

      replay = embodied.replay.Replay(
          length=span + 1, capacity=512, chunksize=37, online=False,
          atomic_updates=True, seed=31)
      for index in range(total + 1):
        preceding_reset = reset_at is not None and index == reset_at - 1
        is_first = index == 0 or (
            reset_at is not None and index == reset_at)
        replay.add({
            'marker': np.asarray(index, np.int32),
            'is_first': np.asarray(is_first, bool),
            'is_last': np.asarray(index == total or preceding_reset, bool),
            'is_terminal': np.asarray(index == total, bool),
            'dyn/deter': np.asarray(states['deter'][0, index]),
            'dyn/stoch': np.asarray(states['stoch'][0, index]),
            'grad/deter': np.zeros(initial['deter'].shape[1:], np.float32),
            'grad/stoch': np.zeros(initial['stoch'].shape[1:], np.float32),
            'grad/valid': np.asarray(index == total, bool),
        })

      def backup(start):
        stop = start + span
        data = _sequence_at(replay, start)
        np.testing.assert_array_equal(
            data['marker'][0], np.arange(start, stop + 1))
        carry = {
            key: jnp.asarray(data[f'dyn/{key}'][:, 0])
            for key in ('deter', 'stoch')}
        future_valid = (
            data['grad/valid'][:, -1].astype(bool) &
            ~data['is_last'][:, -1].astype(bool))
        future = {
            key: jnp.where(
                future_valid.reshape(
                    future_valid.shape + (1,) * (data[f'grad/{key}'].ndim - 2)),
                jnp.asarray(data[f'grad/{key}'][:, -1]),
                jnp.zeros_like(jnp.asarray(data[f'grad/{key}'][:, -1])))
            for key in ('deter', 'stoch')}
        taps = jax.tree.map(
            lambda value: jnp.zeros(
                (batch, span, *value.shape[1:]), jnp.float32), carry)

        def objective(value):
          final, entries, feat = run(carry, start, stop, value)
          local = terminal_loss(feat, weights[:, start:stop])
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

        np.testing.assert_array_equal(
            payloads[0]['stepid'], data['stepid'][:, :1])
        np.testing.assert_array_equal(
            payloads[1]['stepid'], data['stepid'][:, -1:])
        np.testing.assert_array_equal(
            payloads[2]['stepid'], data['stepid'][:, 1:-1])
        np.testing.assert_array_equal(
            payloads[2]['dyn/deter'], entries['deter'][:, :-1])
        np.testing.assert_array_equal(
            payloads[2]['grad/deter'], incoming['deter'][:, 1:])

        before = _physical_rows(replay, total)
        payloads = tuple({
            key: np.asarray(value) for key, value in payload.items()
        } for payload in payloads)
        replay.update(payloads)
        after = _physical_rows(replay, total)

        for leaf in ('deter', 'stoch'):
          key = f'dyn/{leaf}'
          np.testing.assert_array_equal(
              after[key][:start + 1], before[key][:start + 1])
          np.testing.assert_array_equal(
              after[key][stop + 1:], before[key][stop + 1:])
          np.testing.assert_array_equal(
              after[key][start + 1:stop + 1], np.asarray(entries[leaf][0]))
        for leaf in ('deter', 'stoch'):
          key = f'grad/{leaf}'
          np.testing.assert_array_equal(after[key][:start], before[key][:start])
          np.testing.assert_array_equal(after[key][stop:], before[key][stop:])
          np.testing.assert_array_equal(
              after[key][start:stop], np.asarray(incoming[leaf][0]))
        np.testing.assert_array_equal(
            after['grad/valid'][:start], before['grad/valid'][:start])
        np.testing.assert_array_equal(
            after['grad/valid'][stop:], before['grad/valid'][stop:])
        np.testing.assert_array_equal(
            after['grad/valid'][start:stop],
            np.asarray(incoming_finite | terminal)[0])

      # This exact sequence requires the dense interior G_q64 produced by the
      # q2..q66 backup; fixed-boundary-only tests do not exercise that seam.
      for start in (194, 130, 66, 2, 0):
        backup(start)

      cached = _physical_rows(replay, total)
      candidate = {
          key: jnp.asarray(cached[f'grad/{key}'][0:1])
          for key in ('deter', 'stoch')}
      reference = jax.grad(lambda carry: terminal_loss(
          run(carry, 0, total)[2], weights))(initial)
      return reference, candidate, cached

    reference, candidate, cached = run_case()
    relative, cosine = _quality(reference, candidate)
    for value in reference.values():
      assert np.linalg.norm(np.asarray(value)) > 1e-8
    assert relative < 1e-5
    assert cosine > 0.99999
    assert cached['grad/valid'][0]
    assert cached['grad/valid'][64]

    reset_reference, reset_candidate, reset_cached = run_case(reset_at=130)
    for key in ('deter', 'stoch'):
      np.testing.assert_array_equal(reset_reference[key], 0.0)
      np.testing.assert_array_equal(reset_candidate[key], 0.0)
      np.testing.assert_array_equal(reset_cached[f'grad/{key}'][129], 0.0)
  finally:
    nets.COMPUTE_DTYPE = previous_dtype
