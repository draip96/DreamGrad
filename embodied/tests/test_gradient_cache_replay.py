"""Physical replay alignment and sampler-parity tests for saved gradients."""

import copy
import tempfile
import threading

import numpy as np

import embodied


def _step(index, with_grad=True, with_rng=False):
  step = {
      'marker': np.asarray(index, np.int32),
      'is_first': np.asarray(index == 0, bool),
      'is_last': np.asarray(False, bool),
      'is_terminal': np.asarray(False, bool),
      'dyn/deter': np.asarray([index], np.float32),
      'dyn/stoch': np.asarray([[index]], np.float32),
  }
  if with_grad:
    step.update({
        'grad/deter': np.asarray([-index], np.float32),
        'grad/stoch': np.asarray([[-index]], np.float32),
        'grad/valid': np.asarray(False, bool),
    })
  if with_rng:
    step['rng/posterior'] = np.asarray([index, index + 1000], np.uint32)
  return step


def _first_sequence(replay):
  chunkid, index = next(iter(replay.items.values()))
  parts = replay._getseq(chunkid, index, concat=False)
  return replay._assemble_batch([parts], 0, replay.length)


def _sequence_starting_at(replay, marker):
  for chunkid, index in replay.items.values():
    parts = replay._getseq(chunkid, index, concat=False)
    sequence = replay._assemble_batch([parts], 0, replay.length)
    if sequence['marker'][0, 0] == marker:
      return sequence
  raise AssertionError(f'No replay sequence starts at marker {marker}.')


class TestGradientCacheReplay:

  def test_state_and_adjoint_writes_use_distinct_stepid_ranges(self):
    replay = embodied.replay.Replay(
        length=5, capacity=32, chunksize=3, online=False, seed=7)
    for index in range(9):
      replay.add(_step(index))
    sampled = _first_sequence(replay)
    original_state = sampled['dyn/deter'].copy()
    original_grad = sampled['grad/deter'].copy()

    state_payload = {
        'stepid': sampled['stepid'][:, 1:].copy(),
        'dyn/deter': np.arange(101, 105, dtype=np.float32)[None, :, None],
        'dyn/stoch': np.arange(
            111, 115, dtype=np.float32)[None, :, None, None],
    }
    grad_payload = {
        'stepid': sampled['stepid'][:, :-1].copy(),
        'grad/deter': np.arange(201, 205, dtype=np.float32)[None, :, None],
        'grad/stoch': np.arange(
            211, 215, dtype=np.float32)[None, :, None, None],
        'grad/valid': np.ones((1, 4), bool),
    }
    replay.update((state_payload, grad_payload))

    updated = _first_sequence(replay)
    np.testing.assert_array_equal(
        updated['dyn/deter'][:, :1], original_state[:, :1])
    np.testing.assert_array_equal(
        updated['dyn/deter'][:, 1:, 0], [[101, 102, 103, 104]])
    np.testing.assert_array_equal(
        updated['grad/deter'][:, :-1, 0], [[201, 202, 203, 204]])
    np.testing.assert_array_equal(
        updated['grad/deter'][:, -1:], original_grad[:, -1:])
    np.testing.assert_array_equal(
        updated['grad/valid'], [[True, True, True, True, False]])

  def test_duplicate_rows_resolve_last_for_both_paired_payloads(self):
    replay = embodied.replay.Replay(
        length=3, capacity=16, chunksize=8, online=False, seed=3)
    for index in range(6):
      replay.add(_step(index))
    sampled = _first_sequence(replay)
    repeated = np.repeat(sampled['stepid'], 2, axis=0)
    state_payload = {
        'stepid': repeated[:, 1:].copy(),
        'dyn/deter': np.asarray([[[10], [11]], [[20], [21]]], np.float32),
    }
    grad_payload = {
        'stepid': repeated[:, :-1].copy(),
        'grad/deter': np.asarray([[[30], [31]], [[40], [41]]], np.float32),
        'grad/valid': np.ones((2, 2), bool),
    }
    replay.update((state_payload, grad_payload))
    updated = _first_sequence(replay)
    np.testing.assert_array_equal(updated['dyn/deter'][:, 1:, 0], [[20, 21]])
    np.testing.assert_array_equal(updated['grad/deter'][:, :-1, 0], [[40, 41]])

  def test_interior_pair_wins_shifted_endpoint_duplicates(self):
    replay = embodied.replay.Replay(
        length=4, capacity=16, chunksize=8, online=False, seed=4)
    for index in range(7):
      replay.add(_step(index))
    chunkid, _ = next(iter(replay.items.values()))
    chunk = replay.chunks[chunkid]
    ids = chunk.data['stepid']

    # Two overlapping three-transition windows have q1 as an interior/leading
    # collision and q2 as a trailing/interior collision. Endpoint writes happen
    # first; the paired payload must set both fields last from one occurrence.
    leading_grad = {
        'stepid': np.stack((ids[0:1], ids[1:2])),
        'grad/deter': np.asarray([[[10]], [[20]]], np.float32),
    }
    trailing_state = {
        'stepid': np.stack((ids[2:3], ids[3:4])),
        'dyn/deter': np.asarray([[[30]], [[40]]], np.float32),
    }
    paired = {
        'stepid': np.stack((ids[1:2], ids[2:3])),
        'dyn/deter': np.asarray([[[50]], [[60]]], np.float32),
        'grad/deter': np.asarray([[[70]], [[80]]], np.float32),
    }
    replay.update((leading_grad, trailing_state, paired))
    np.testing.assert_array_equal(chunk.data['dyn/deter'][1:3, 0], [50, 60])
    np.testing.assert_array_equal(chunk.data['grad/deter'][1:3, 0], [70, 80])

  def test_cache_fields_do_not_change_uniform_sampling(self):
    baseline = embodied.replay.Replay(
        length=4, capacity=64, chunksize=7, online=False, seed=19)
    cached = embodied.replay.Replay(
        length=4, capacity=64, chunksize=7, online=False, seed=19,
        atomic_updates=True)
    for index in range(24):
      baseline.add(_step(index, with_grad=False))
      cached.add(_step(index, with_grad=True))
    for _ in range(30):
      left = baseline.sample(3)['marker']
      right = cached.sample(3)['marker']
      np.testing.assert_array_equal(left, right)

  def test_multi_payload_update_does_not_mutate_caller_container(self):
    replay = embodied.replay.Replay(
        length=3, capacity=16, chunksize=8, online=False, seed=5)
    for index in range(6):
      replay.add(_step(index))
    sampled = _first_sequence(replay)
    payloads = ({
        'stepid': sampled['stepid'][:, 1:].copy(),
        'dyn/deter': np.ones((1, 2, 1), np.float32),
    }, {
        'stepid': sampled['stepid'][:, :-1].copy(),
        'grad/deter': np.ones((1, 2, 1), np.float32),
    })
    keys = tuple(tuple(value.keys()) for value in copy.deepcopy(payloads))
    replay.update(payloads)
    # Individual dictionaries intentionally follow upstream's pop-based API;
    # the tuple itself must remain structurally usable by delayed-output code.
    assert len(payloads) == 2
    assert keys[0][1:] == tuple(payloads[0].keys())
    assert keys[1][1:] == tuple(payloads[1].keys())

  def test_updated_state_and_adjoint_survive_save_reload(self):
    with tempfile.TemporaryDirectory() as directory:
      replay = embodied.replay.Replay(
          length=4, capacity=32, chunksize=16, directory=directory,
          save_wait=True, persist_updates=True, atomic_updates=True,
          online=False, seed=23)
      for index in range(8):
        replay.add(_step(index))
      replay.save()
      sampled = _first_sequence(replay)
      replay.update(({
          'stepid': sampled['stepid'][:, 1:].copy(),
          'dyn/deter': np.asarray([[[51], [52], [53]]], np.float32),
      }, {
          'stepid': sampled['stepid'][:, :-1].copy(),
          'grad/deter': np.asarray([[[61], [62], [63]]], np.float32),
          'grad/valid': np.ones((1, 3), bool),
      }))
      replay.save()

      restored = embodied.replay.Replay(
          length=4, capacity=32, chunksize=16, directory=directory,
          save_wait=True, persist_updates=True, atomic_updates=True,
          online=False, seed=23)
      restored.load()
      loaded = _first_sequence(restored)
      np.testing.assert_array_equal(loaded['dyn/deter'][:, 1:, 0], [[51, 52, 53]])
      np.testing.assert_array_equal(loaded['grad/deter'][:, :-1, 0], [[61, 62, 63]])
      np.testing.assert_array_equal(loaded['grad/valid'], [[True, True, True, False]])

  def test_posterior_rng_survives_overlapping_updates_and_save_reload(self):
    with tempfile.TemporaryDirectory() as directory:
      replay = embodied.replay.Replay(
          length=4, capacity=32, chunksize=16, directory=directory,
          save_wait=True, persist_updates=True, atomic_updates=True,
          online=False, seed=27)
      for index in range(8):
        replay.add(_step(index, with_rng=True))
      replay.save()

      first = _sequence_starting_at(replay, 0)
      second = _sequence_starting_at(replay, 1)
      first_keys = first['rng/posterior'].copy()
      second_keys = second['rng/posterior'].copy()
      stepids = np.concatenate((first['stepid'], second['stepid']), 0)
      state_payload = {
          'stepid': stepids[:, 1:].copy(),
          'dyn/deter': np.arange(6, dtype=np.float32).reshape(2, 3, 1) + 51,
      }
      grad_payload = {
          'stepid': stepids[:, :-1].copy(),
          'grad/deter': np.arange(6, dtype=np.float32).reshape(2, 3, 1) + 71,
          'grad/valid': np.ones((2, 3), bool),
      }
      assert all(
          'rng/posterior' not in payload
          for payload in (state_payload, grad_payload))
      replay.update((state_payload, grad_payload))

      updated_first = _sequence_starting_at(replay, 0)
      updated_second = _sequence_starting_at(replay, 1)
      np.testing.assert_array_equal(
          updated_first['rng/posterior'], first_keys)
      np.testing.assert_array_equal(
          updated_second['rng/posterior'], second_keys)
      np.testing.assert_array_equal(
          updated_first['rng/posterior'][:, 1:],
          updated_second['rng/posterior'][:, :-1])
      assert np.any(updated_first['dyn/deter'] != first['dyn/deter'])
      assert np.any(updated_first['grad/deter'] != first['grad/deter'])
      replay.save()

      restored = embodied.replay.Replay(
          length=4, capacity=32, chunksize=16, directory=directory,
          save_wait=True, persist_updates=True, atomic_updates=True,
          online=False, seed=27)
      restored.load()
      np.testing.assert_array_equal(
          _sequence_starting_at(restored, 0)['rng/posterior'], first_keys)
      np.testing.assert_array_equal(
          _sequence_starting_at(restored, 1)['rng/posterior'], second_keys)

  def test_sampler_cannot_observe_half_of_paired_update(self):
    replay = embodied.replay.Replay(
        length=3, capacity=8, chunksize=8, online=False, seed=29,
        atomic_updates=True)
    for index in range(3):
      replay.add(_step(index))
    sampled = _first_sequence(replay)
    payloads = ({
        'stepid': sampled['stepid'].copy(),
        'dyn/deter': np.full((1, 3, 1), 71.0, np.float32),
    }, {
        'stepid': sampled['stepid'].copy(),
        'grad/deter': np.full((1, 3, 1), 83.0, np.float32),
        'grad/valid': np.ones((1, 3), bool),
    })

    state_written = threading.Event()
    allow_grad = threading.Event()
    sample_entered = threading.Event()
    original_setseq = replay._setseq
    original_sample = replay._sample
    calls = {'count': 0}

    def paused_setseq(*args, **kwargs):
      result = original_setseq(*args, **kwargs)
      calls['count'] += 1
      if calls['count'] == 1:
        state_written.set()
        assert allow_grad.wait(5)
      return result

    def marked_sample(*args, **kwargs):
      sample_entered.set()
      return original_sample(*args, **kwargs)

    replay._setseq = paused_setseq
    replay._sample = marked_sample
    update_thread = threading.Thread(target=replay.update, args=(payloads,))
    update_thread.start()
    assert state_written.wait(5)
    result = {}
    sample_thread = threading.Thread(
        target=lambda: result.update(data=replay.sample(1)))
    sample_thread.start()
    # A sampler may not enter its critical section while only S is visible.
    assert not sample_entered.wait(0.2)
    allow_grad.set()
    update_thread.join(5)
    sample_thread.join(5)
    assert not update_thread.is_alive()
    assert not sample_thread.is_alive()
    assert np.all(result['data']['dyn/deter'] == 71.0)
    assert np.all(result['data']['grad/deter'] == 83.0)
    assert np.all(result['data']['grad/valid'])

  def test_default_save_reload_retains_upstream_snapshot_semantics(self):
    with tempfile.TemporaryDirectory() as directory:
      replay = embodied.replay.Replay(
          length=3, capacity=8, chunksize=8, directory=directory,
          save_wait=True, online=False, seed=31)
      for index in range(4):
        replay.add(_step(index))
      replay.save()
      sampled = _first_sequence(replay)
      original = sampled['dyn/deter'].copy()
      replay.update({
          'stepid': sampled['stepid'].copy(),
          'dyn/deter': np.full((1, 3, 1), 97.0, np.float32),
      })
      replay.save()

      restored = embodied.replay.Replay(
          length=3, capacity=8, chunksize=8, directory=directory,
          save_wait=True, online=False, seed=31)
      restored.load()
      np.testing.assert_array_equal(
          _first_sequence(restored)['dyn/deter'], original)

  def test_persist_updates_requires_synchronous_saves(self):
    with tempfile.TemporaryDirectory() as directory:
      with np.testing.assert_raises_regex(ValueError, 'save_wait'):
        embodied.replay.Replay(
            length=3, capacity=8, chunksize=8, directory=directory,
            save_wait=False, persist_updates=True, atomic_updates=True)

  def test_persist_updates_requires_atomic_visibility(self):
    with tempfile.TemporaryDirectory() as directory:
      with np.testing.assert_raises_regex(ValueError, 'atomic_updates'):
        embodied.replay.Replay(
            length=3, capacity=8, chunksize=8, directory=directory,
            save_wait=True, persist_updates=True, atomic_updates=False)
