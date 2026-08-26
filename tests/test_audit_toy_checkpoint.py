"""Deterministic helper contracts for the read-only ToyMemory audit."""

from pathlib import Path

import numpy as np
import pytest

from experiments import audit_toy_checkpoint as audit


def test_replay_chain_uses_uuid_successors_not_filename_order():
  paths = [
      Path('20260101T000002-C-D-3.npz'),
      Path('20260101T000000-A-B-3.npz'),
      Path('20260101T000001-B-C-3.npz'),
  ]
  assert [item.uuid for item in audit.order_chunk_chain(paths)] == [
      'A', 'B', 'C']


def test_replay_chain_rejects_disconnected_chunks():
  paths = [
      Path('20260101T000000-A-B-3.npz'),
      Path('20260101T000001-C-D-3.npz'),
  ]
  with pytest.raises(RuntimeError, match='one replay-chain head'):
    audit.order_chunk_chain(paths)


def test_geometry_and_action_reward_alignment():
  distance = 3
  episodes = 4
  length = distance + 2
  observation = np.zeros((episodes, length, 4), np.float32)
  observation[:, :, 0] = np.asarray([-1, -1 / 3, 1 / 3, 1, 1])
  observation[:, :, 3] = 1
  cues = np.asarray([0, 1, 0, 1], np.int32)
  observation[:, 0, 1] = 2 * cues - 1
  observation[:, distance, 2] = 1
  actions = np.zeros((episodes, length), np.int32)
  actions[:, distance] = np.asarray([0, 0, 1, 1])
  rewards = np.zeros((episodes, length), np.float32)
  rewards[:, -1] = np.where(actions[:, distance] == cues, 1, -1)
  first = np.zeros((episodes, length), bool)
  first[:, 0] = True
  last = np.zeros((episodes, length), bool)
  last[:, -1] = True
  stepids = np.arange(episodes * 20, dtype=np.uint8).reshape(episodes, 20)
  states = {
      'deter': np.zeros((episodes, 2), np.float32),
      'stoch': np.zeros((episodes, 1, 2), np.float32),
  }
  artifact = audit.ToyArtifact(
      distance, length, observation, actions, rewards, first, last, last,
      stepids, states, states)
  audit.validate_toy_geometry(artifact)
  artifact.rewards[0, -1] *= -1
  with pytest.raises(RuntimeError, match='alignment'):
    audit.validate_toy_geometry(artifact)


def test_nearest_centroid_proxy_is_held_out_and_labeled_as_proxy():
  labels = np.asarray([0, 1] * 30, np.int32)
  features = np.stack([2 * labels - 1, labels], -1).astype(np.float32)
  stepids = np.arange(len(labels) * 20, dtype=np.uint8).reshape(len(labels), 20)
  folds = audit.fixed_probe_folds(stepids)
  result = audit.nearest_centroid_probe(features, labels, folds)
  assert result['balanced_accuracy'] == 1.0
  assert 'not exact decodability' in result['interpretation']


def test_counterfactual_estimators_keep_map_and_mc_labels_separate():
  cues = np.asarray([0, 1], np.int32)
  actions = np.asarray([0, 1], np.int32)
  samples = np.asarray([
      [[1.0, 0.8], [-1.0, -0.8]],
      [[-1.0, -0.8], [1.0, 0.8]],
  ])
  terminals = np.ones_like(samples)
  mc = audit.counterfactual_metrics(samples, terminals, cues, actions)
  mapped = audit.map_counterfactual_metrics(
      samples.mean(-1), terminals.mean(-1), cues, actions)
  assert mc['model_greedy_accuracy'] == 1.0
  assert mc['estimator'] == 'monte_carlo_prior_expectation'
  assert 'not_expected_reward' in mapped['estimator']
