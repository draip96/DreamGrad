#!/usr/bin/env python3
"""Validate a completed end-to-end DreamGrad integration smoke."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml

import embodied


def load_metrics(logdir):
  path = logdir / 'metrics.jsonl'
  assert path.exists(), path
  rows = [json.loads(line) for line in path.read_text().splitlines() if line]
  assert rows, path
  return rows


def load_config(logdir):
  loader = yaml.YAML(typ='safe')
  config = loader.load((logdir / 'config.yaml').read_text())
  rssm = config['agent']['dyn']['rssm']
  assert (rssm['deter'], rssm['hidden'], rssm['stoch'], rssm['classes']) == (
      2048, 256, 32, 16)
  for section in ('enc', 'dec'):
    simple = config['agent'][section]['simple']
    assert (simple['depth'], simple['units']) == (16, 256)
  assert config['jax']['platform'] == 'cuda'
  assert config['replay_context'] == 1
  return config


def load_update_count(logdir, expected_step):
  checkpoint = logdir / 'ckpt'
  latest = (checkpoint / 'latest').read_text().strip()
  final = checkpoint / latest
  assert (final / 'done').exists(), final
  with (final / 'step.pkl').open('rb') as handle:
    step = int(pickle.load(handle))
  assert step == int(expected_step), (step, expected_step)
  with (final / 'agent.pkl').open('rb') as handle:
    state = pickle.load(handle)
  updates = int(state['counters']['updates'])
  assert updates >= 1, updates
  assert int(np.asarray(state['params']['opt/step/value'])) == updates
  return updates


def values(rows, key):
  return [float(row[key]) for row in rows if key in row]


def load_replay(logdir, cache_enabled, config):
  directory = logdir / 'replay'
  assert list(directory.glob('*.npz')), directory
  length = (
      config['consec_train'] * config['batch_length'] +
      config['replay_context'])
  replay = embodied.replay.Replay(
      length=length, capacity=int(config['replay']['size']),
      chunksize=int(config['replay']['chunksize']), directory=directory,
      online=bool(config['replay']['online']),
      save_wait=True, persist_updates=cache_enabled,
      atomic_updates=cache_enabled, seed=0)
  replay.load()
  assert len(replay) > 0
  chunks = [chunk for chunk in replay.chunks.values() if chunk.length]
  assert chunks
  batch = replay.sample(1)
  assert batch['stepid'].shape[:2] == (1, length), batch['stepid'].shape
  for key in ('dyn/deter', 'dyn/stoch'):
    assert np.isfinite(batch[key]).all(), key
  if cache_enabled:
    for key in ('grad/deter', 'grad/stoch'):
      assert np.isfinite(batch[key]).all(), key
  return replay, chunks


def validate(logdir, cache_enabled):
  assert logdir.exists(), logdir
  config = load_config(logdir)
  posterior_rng_keys = bool(
      config['agent']['gradient_cache'].get('posterior_rng_keys', False))
  assert not posterior_rng_keys or cache_enabled
  assert any(logdir.glob('ckpt*')), f'Missing checkpoint in {logdir}'
  update_count = load_update_count(logdir, config['run']['steps'])
  rows = load_metrics(logdir)
  updates = values(rows, 'train/opt/updates')
  assert updates and max(updates) >= 1, updates
  replay_updates = values(rows, 'replay/updates')
  assert replay_updates and max(replay_updates) > 0, replay_updates
  for key in ('train/opt/loss', 'train/opt/grad_norm'):
    observed = values(rows, key)
    assert observed and np.isfinite(observed).all(), (key, observed)

  replay, chunks = load_replay(logdir, cache_enabled, config)
  common_keys = set.intersection(*(set(chunk.data) for chunk in chunks))
  all_keys = set.union(*(set(chunk.data) for chunk in chunks))
  for key in ('dyn/deter', 'dyn/stoch'):
    assert key in common_keys
    assert all(np.isfinite(chunk.data[key][:chunk.length]).all()
               for chunk in chunks)

  grad_keys = {'grad/deter', 'grad/stoch', 'grad/valid'}
  posterior_key = 'rng/posterior'
  if posterior_rng_keys:
    assert posterior_key in common_keys
    arrays = np.concatenate([
        chunk.data[posterior_key][:chunk.length] for chunk in chunks])
    assert arrays.dtype == np.uint32, arrays.dtype
    assert arrays.shape[-1] == 2, arrays.shape
  else:
    assert posterior_key not in all_keys
  if not cache_enabled:
    assert not (all_keys & grad_keys), all_keys & grad_keys
    assert not any(
        key.startswith('train/cache/') for row in rows for key in row)
    return {
        'cache_enabled': False,
        'posterior_rng_keys': False,
        'updates': update_count,
        'replay_items_after_reload': len(replay),
    }

  assert grad_keys <= common_keys, grad_keys - common_keys
  valid = np.concatenate([
      chunk.data['grad/valid'][:chunk.length] for chunk in chunks])
  assert valid.any(), 'No valid saved adjoints survived replay persistence.'
  nonzero = False
  for key in ('grad/deter', 'grad/stoch'):
    arrays = np.concatenate([
        chunk.data[key][:chunk.length] for chunk in chunks])
    assert np.isfinite(arrays).all(), key
    nonzero |= bool(np.any(arrays[valid] != 0))
  assert nonzero, 'All persisted valid adjoints are zero.'

  used = values(rows, 'train/cache/future_used_rate')
  finite = values(rows, 'train/cache/adjoint_finite_fraction')
  assert used and max(used) > 0, used
  assert finite and min(finite) == 1.0, finite
  return {
      'cache_enabled': True,
      'posterior_rng_keys': posterior_rng_keys,
      'updates': update_count,
      'replay_items_after_reload': len(replay),
      'valid_adjoint_rows': int(valid.sum()),
      'future_used_rate_max': max(used),
      'adjoint_finite_fraction_min': min(finite),
  }


def compare_configs(left, right):
  loader = yaml.YAML(typ='safe')
  lhs = loader.load((left / 'config.yaml').read_text())
  rhs = loader.load((right / 'config.yaml').read_text())
  lhs.pop('logdir')
  rhs.pop('logdir')
  lhs['agent']['gradient_cache']['enabled'] = '<cache-arm>'
  rhs['agent']['gradient_cache']['enabled'] = '<cache-arm>'
  assert lhs == rhs, 'Cache-on/off smoke configurations differ beyond cache flag.'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('logdir', type=Path)
  parser.add_argument('--cache-enabled', action='store_true')
  parser.add_argument('--compare-config', type=Path)
  args = parser.parse_args()
  result = validate(args.logdir, args.cache_enabled)
  if args.compare_config:
    compare_configs(args.logdir, args.compare_config)
    result['matched_control_config'] = True
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
