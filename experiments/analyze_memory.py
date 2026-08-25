#!/usr/bin/env python3
"""Analyze ToyMemory or BSuite score artifacts with explicit learning gates."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml


def read_jsonl(path):
  with path.open() as handle:
    return [json.loads(line) for line in handle if line.strip()]


def wilson_lower(successes, total, z=1.959963984540054):
  if not total:
    return 0.0
  rate = successes / total
  denom = 1 + z ** 2 / total
  center = rate + z ** 2 / (2 * total)
  radius = z * math.sqrt(
      rate * (1 - rate) / total + z ** 2 / (4 * total ** 2))
  return (center - radius) / denom


def finite_or_none(value):
  value = float(value)
  return value if math.isfinite(value) else None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('logdir', type=Path)
  parser.add_argument('--kind', choices=('toy', 'bsuite'), required=True)
  parser.add_argument('--length', type=int, required=True)
  parser.add_argument('--tail', type=int, default=1000)
  args = parser.parse_args()

  scores = read_jsonl(args.logdir / 'scores.jsonl')
  metrics = read_jsonl(args.logdir / 'metrics.jsonl')
  config = yaml.YAML(typ='safe').load(
      (args.logdir / 'config.yaml').read_text())
  expected_steps = int(config['run']['steps'])
  returns = np.asarray([
      row['episode/score'] for row in scores if 'episode/score' in row],
      np.float64)
  lengths = np.asarray([
      row['episode/length'] for row in metrics if 'episode/length' in row],
      np.int64)
  if not len(returns):
    raise RuntimeError('No completed episode scores found.')
  expected_length = args.length + 2
  if len(lengths) != len(returns):
    raise RuntimeError(
        f'Episode score/length count mismatch: {len(returns)} vs {len(lengths)}')
  if not np.all(lengths == expected_length):
    unique = np.unique(lengths).tolist()
    raise RuntimeError(
        f'Unexpected episode lengths {unique}; expected {expected_length}.')
  if int(scores[-1]['step']) != expected_steps:
    raise RuntimeError(
        f'Final complete episode ended at {scores[-1]["step"]}, but the run '
        f'budget was {expected_steps}; refusing a partial final episode.')
  expected_episodes = expected_steps // expected_length
  if expected_episodes * expected_length != expected_steps:
    raise RuntimeError(
        f'Run budget {expected_steps} is not divisible by episode length '
        f'{expected_length}.')
  if len(returns) != expected_episodes:
    raise RuntimeError(
        f'Expected {expected_episodes} complete episodes from the exact step '
        f'budget, got {len(returns)}.')

  correct = returns > 0
  tail_count = min(args.tail, len(returns))
  tail_correct = correct[-tail_count:]
  tail_returns = returns[-tail_count:]
  result = {
      'kind': args.kind,
      'configured_length': args.length,
      'physical_rows_per_episode': expected_length,
      'episodes': int(len(returns)),
      'expected_steps': expected_steps,
      'final_score_step': int(scores[-1]['step']),
      'cumulative_accuracy': float(correct.mean()),
      'cumulative_mean_return': float(returns.mean()),
      'tail_episodes': int(tail_count),
      'tail_accuracy': float(tail_correct.mean()),
      'tail_mean_return': float(tail_returns.mean()),
      'tail_wilson_lower_95': float(wilson_lower(
          int(tail_correct.sum()), tail_count)),
  }

  health_names = (
      'train/cache/future_hit_rate',
      'train/cache/future_used_rate',
      'train/cache/future_adjoint_rms',
      'train/cache/adjoint_rms',
      'train/cache/adjoint_finite_fraction',
      'train/model/terminal_fraction',
      'train/model/terminal_reward_sign_accuracy',
      'train/model/terminal_reward_mae',
      'train/rew',
      'train/ret',
  )
  for name in health_names:
    values = [row[name] for row in metrics if name in row]
    if values:
      result[f'final_{name.replace("/", "_")}'] = finite_or_none(values[-1])

  if args.kind == 'toy':
    result.update({
        'cue_to_query_distance': args.length,
        'cue_to_reward_dependency': args.length + 1,
        'gate_tail_accuracy_ge_0_95': result['tail_accuracy'] >= 0.95,
        'gate_tail_return_ge_0_90': result['tail_mean_return'] >= 0.90,
        'gate_wilson_lower_gt_0_90': result['tail_wilson_lower_95'] > 0.90,
    })
    passed = all(result[key] for key in (
        'gate_tail_accuracy_ge_0_95',
        'gate_tail_return_ge_0_90',
        'gate_wilson_lower_gt_0_90'))
  else:
    if len(returns) != 10_000:
      raise RuntimeError(
          f'BSuite requires exactly 10000 complete episodes, got {len(returns)}.')
    result.update({
        'official_actions_per_episode': args.length + 1,
        'first_cue_to_query_distance': args.length,
        'last_cue_to_query_distance': args.length - 1,
        'gate_official_accuracy_gt_0_625': (
            result['cumulative_accuracy'] > 0.625),
        'gate_tail_accuracy_ge_0_90': result['tail_accuracy'] >= 0.90,
    })
    passed = (
        result['gate_official_accuracy_gt_0_625'] and
        result['gate_tail_accuracy_ge_0_90'])

  result['passed'] = bool(passed)
  print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
  raise SystemExit(0 if passed else 3)


if __name__ == '__main__':
  main()
