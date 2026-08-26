#!/usr/bin/env python3
"""Read-only functional audit of a completed ToyMemory checkpoint.

This utility never calls ``Agent.train`` or ``Replay.update``. It compares
saved replay states with full-episode states reconstructed under the final
checkpoint, and keeps deterministic MAP, Monte Carlo, and learned-proxy
quantities explicitly separate in its JSON output.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ChunkMeta:
  path: Path
  timestamp: str
  uuid: str
  successor: str
  length: int


@dataclass(frozen=True)
class ToyArtifact:
  distance: int
  episode_length: int
  observations: np.ndarray
  actions: np.ndarray
  rewards: np.ndarray
  is_first: np.ndarray
  is_last: np.ndarray
  is_terminal: np.ndarray
  query_stepids: np.ndarray
  cached_query: dict
  cached_terminal: dict

  @property
  def episodes(self):
    return len(self.observations)

  @property
  def cues(self):
    return (self.observations[:, 0, 1] > 0).astype(np.int32)

  @property
  def query_actions(self):
    return self.actions[:, self.distance].astype(np.int32)

  @property
  def terminal_rewards(self):
    return self.rewards[:, self.distance + 1].astype(np.float32)


def positive_int(value):
  value = int(value)
  if value < 1:
    raise argparse.ArgumentTypeError('value must be a positive integer')
  return value


def parse_chunk_filename(path):
  path = Path(path)
  parts = path.stem.split('-')
  if len(parts) != 4:
    raise ValueError(f'Invalid replay chunk filename: {path.name}')
  timestamp, uuid, successor, length = parts
  try:
    length = int(length)
  except ValueError as exc:
    raise ValueError(f'Invalid replay chunk length: {path.name}') from exc
  if length < 1:
    raise ValueError(f'Empty replay chunk is not auditable: {path.name}')
  return ChunkMeta(path, timestamp, uuid, successor, length)


def order_chunk_chain(paths):
  """Return the unique UUID-successor chain, with structural assertions."""
  metas = [parse_chunk_filename(path) for path in paths]
  if not metas:
    raise RuntimeError('No persisted replay chunks found.')
  by_uuid = {}
  for meta in metas:
    if meta.uuid in by_uuid:
      raise RuntimeError(f'Duplicate replay chunk UUID: {meta.uuid}')
    by_uuid[meta.uuid] = meta
  referenced = {meta.successor for meta in metas if meta.successor in by_uuid}
  heads = set(by_uuid) - referenced
  if len(heads) != 1:
    raise RuntimeError(f'Expected one replay-chain head, found {sorted(heads)}')
  ordered = []
  seen = set()
  current = next(iter(heads))
  while current in by_uuid:
    if current in seen:
      raise RuntimeError(f'Replay successor cycle at chunk {current}')
    seen.add(current)
    meta = by_uuid[current]
    ordered.append(meta)
    current = meta.successor
  if seen != set(by_uuid):
    missing = sorted(set(by_uuid) - seen)
    raise RuntimeError(f'Disconnected replay chunks: {missing[:8]}')
  return ordered


def _concat(parts, name):
  if not parts:
    raise RuntimeError(f'No replay values collected for {name}.')
  return np.concatenate(parts, axis=0)


def load_toy_artifact(logdir, config):
  """Stream a persisted replay and retain only query/terminal RSSM states."""
  replay_dir = logdir / 'replay'
  chain = order_chunk_chain(replay_dir.glob('*.npz'))
  distance = int(config['env']['toymemory']['distance'])
  episode_length = distance + 2
  expected_steps = int(config['run']['steps'])
  if expected_steps % episode_length:
    raise RuntimeError(
        f'Run steps {expected_steps} are not divisible by {episode_length}.')
  if sum(meta.length for meta in chain) != expected_steps:
    raise RuntimeError(
        f'Replay chain contains {sum(x.length for x in chain)} rows, expected '
        f'{expected_steps}.')

  base_names = (
      'observation', 'action', 'reward', 'is_first', 'is_last',
      'is_terminal')
  base = {name: [] for name in base_names}
  query_stepids = []
  cached_query = {'deter': [], 'stoch': []}
  cached_terminal = {'deter': [], 'stoch': []}
  for meta in chain:
    with np.load(meta.path, allow_pickle=False) as chunk:
      required = set(base_names) | {
          'stepid', 'dyn/deter', 'dyn/stoch'}
      missing = required - set(chunk.files)
      if missing:
        raise RuntimeError(f'{meta.path.name} is missing {sorted(missing)}')
      if any(len(chunk[name]) != meta.length for name in required):
        raise RuntimeError(f'Filename/data length mismatch in {meta.path.name}')
      for name in base_names:
        base[name].append(np.asarray(chunk[name]))
      observation = np.asarray(chunk['observation'])
      query = observation[:, 2] == 1
      terminal = np.asarray(chunk['is_terminal']).astype(bool)
      query_stepids.append(np.asarray(chunk['stepid'])[query].copy())
      deter = np.asarray(chunk['dyn/deter'])
      stoch = np.asarray(chunk['dyn/stoch'])
      cached_query['deter'].append(deter[query].copy())
      cached_query['stoch'].append(stoch[query].copy())
      cached_terminal['deter'].append(deter[terminal].copy())
      cached_terminal['stoch'].append(stoch[terminal].copy())

  flat = {name: _concat(parts, name) for name, parts in base.items()}
  episodes = expected_steps // episode_length
  shaped = {
      name: value.reshape((episodes, episode_length, *value.shape[1:]))
      for name, value in flat.items()}
  artifact = ToyArtifact(
      distance=distance,
      episode_length=episode_length,
      observations=shaped['observation'],
      actions=shaped['action'],
      rewards=shaped['reward'],
      is_first=shaped['is_first'],
      is_last=shaped['is_last'],
      is_terminal=shaped['is_terminal'],
      query_stepids=_concat(query_stepids, 'query step IDs'),
      cached_query={
          key: _concat(value, f'cached query {key}')
          for key, value in cached_query.items()},
      cached_terminal={
          key: _concat(value, f'cached terminal {key}')
          for key, value in cached_terminal.items()},
  )
  validate_toy_geometry(artifact)
  return artifact, chain


def validate_toy_geometry(artifact):
  """Assert literal q0 cue, q_distance query, q_(distance+1) reward."""
  n, length = artifact.observations.shape[:2]
  distance = artifact.distance
  if length != distance + 2:
    raise RuntimeError((length, distance))
  expected_first = np.zeros((n, length), bool)
  expected_first[:, 0] = True
  expected_last = np.zeros((n, length), bool)
  expected_last[:, -1] = True
  if not np.array_equal(artifact.is_first, expected_first):
    raise RuntimeError('Toy replay does not reset exactly at q0.')
  if not np.array_equal(artifact.is_last, expected_last):
    raise RuntimeError('Toy replay does not terminate exactly at q_(d+1).')
  if not np.array_equal(artifact.is_terminal, expected_last):
    raise RuntimeError('Toy terminal flags do not match episode ends.')

  cue_component = artifact.observations[:, :, 1]
  if not np.all(np.isin(cue_component[:, 0], (-1.0, 1.0))):
    raise RuntimeError('Toy cues are not balanced-bit values at q0.')
  if np.any(cue_component[:, 1:] != 0):
    raise RuntimeError('Cue signal appears after q0.')
  query_component = artifact.observations[:, :, 2]
  expected_query = np.zeros((n, length), query_component.dtype)
  expected_query[:, distance] = 1
  if not np.array_equal(query_component, expected_query):
    raise RuntimeError('Query marker is not exactly at q_distance.')
  progress = np.concatenate([
      np.linspace(-1, 1, distance + 1, dtype=np.float32),
      np.ones(1, np.float32)])
  if not np.allclose(artifact.observations[:, :, 0], progress, atol=1e-6):
    raise RuntimeError('Toy progress channel has unexpected geometry.')
  if not np.all(artifact.observations[:, :, 3] == 1):
    raise RuntimeError('Toy constant observation channel changed.')

  if np.any(artifact.rewards[:, :-1] != 0):
    raise RuntimeError('Toy reward appears before q_(distance+1).')
  if not np.all(np.isin(artifact.terminal_rewards, (-1.0, 1.0))):
    raise RuntimeError('Toy terminal rewards are not +/-1.')
  expected_reward = np.where(
      artifact.query_actions == artifact.cues, 1.0, -1.0)
  if not np.array_equal(artifact.terminal_rewards, expected_reward):
    raise RuntimeError('Cue/query action/reward alignment is inconsistent.')
  if artifact.query_stepids.shape != (n, 20):
    raise RuntimeError(
        f'Expected {n} query step IDs of width 20, got '
        f'{artifact.query_stepids.shape}.')
  if len({bytes(value) for value in artifact.query_stepids}) != n:
    raise RuntimeError('Query step IDs are not unique.')
  for name, states in (
      ('query', artifact.cached_query),
      ('terminal', artifact.cached_terminal)):
    if any(len(value) != n for value in states.values()):
      raise RuntimeError(f'Cached {name} state count does not match episodes.')
    if not all(np.isfinite(value).all() for value in states.values()):
      raise RuntimeError(f'Cached {name} states contain nonfinite values.')


def fixed_probe_folds(stepids, folds=3):
  """Stable split independent of Python hashing and episode chronology."""
  result = []
  for stepid in np.asarray(stepids):
    digest = hashlib.sha256(np.asarray(stepid, np.uint8).tobytes()).digest()
    result.append(int.from_bytes(digest[:4], 'big') % folds)
  return np.asarray(result, np.int32)


def balanced_accuracy(labels, predictions):
  labels = np.asarray(labels, np.int32)
  predictions = np.asarray(predictions, np.int32)
  recalls = []
  for label in (0, 1):
    mask = labels == label
    if not mask.any():
      raise RuntimeError(f'No examples for probe class {label}.')
    recalls.append(float((predictions[mask] == label).mean()))
  return float(np.mean(recalls))


def nearest_centroid_probe(features, labels, folds, test_fold=2):
  """Fixed held-out linear proxy; not an intrinsic decodability measure."""
  features = np.asarray(features, np.float32)
  labels = np.asarray(labels, np.int32)
  folds = np.asarray(folds, np.int32)
  train = folds != test_fold
  test = folds == test_fold
  if not train.any() or not test.any():
    raise RuntimeError('Probe split has an empty train or test partition.')
  mean = features[train].mean(0, dtype=np.float64).astype(np.float32)
  scale = features[train].std(0, dtype=np.float64).astype(np.float32)
  scale = np.maximum(scale, np.float32(1e-6))
  normalized = (features - mean) / scale

  def fit_predict(train_labels):
    centers = []
    for label in (0, 1):
      mask = train & (train_labels == label)
      if not mask.any():
        raise RuntimeError(f'Probe train split lacks class {label}.')
      centers.append(normalized[mask].mean(0, dtype=np.float64))
    centers = np.asarray(centers, np.float32)
    distances = np.stack([
        np.square(normalized[test] - center).sum(-1)
        for center in centers], -1)
    return distances.argmin(-1).astype(np.int32)

  prediction = fit_predict(labels)
  shuffled = labels.copy()
  rng = np.random.default_rng(0)
  shuffled_train = shuffled[train].copy()
  rng.shuffle(shuffled_train)
  shuffled[train] = shuffled_train
  shuffled_prediction = fit_predict(shuffled)
  return {
      'estimator': 'standardized_nearest_centroid_linear_proxy',
      'interpretation': 'held-out linear separability, not exact decodability',
      'train_examples': int(train.sum()),
      'test_examples': int(test.sum()),
      'accuracy': float((prediction == labels[test]).mean()),
      'balanced_accuracy': balanced_accuracy(labels[test], prediction),
      'fixed_label_shuffle_balanced_accuracy': balanced_accuracy(
          labels[test], shuffled_prediction),
  }


def policy_metrics(logits, probs, cues, estimator):
  logits = np.asarray(logits, np.float64)
  probs = np.asarray(probs, np.float64)
  cues = np.asarray(cues, np.int32)
  if logits.shape != (len(cues), 2) or probs.shape != logits.shape:
    raise RuntimeError((logits.shape, probs.shape, cues.shape))
  chosen = probs.argmax(-1)
  true_prob = probs[np.arange(len(cues)), cues]
  other_prob = probs[np.arange(len(cues)), 1 - cues]
  true_logit = logits[np.arange(len(cues)), cues]
  other_logit = logits[np.arange(len(cues)), 1 - cues]
  safe = np.clip(probs, 1e-12, 1.0)
  entropy = -(safe * np.log(safe)).sum(-1)
  return {
      'estimator': estimator,
      'argmax_accuracy': float((chosen == cues).mean()),
      'expected_sampled_accuracy': float(true_prob.mean()),
      'mean_true_action_probability': float(true_prob.mean()),
      'mean_true_minus_other_probability': float(
          (true_prob - other_prob).mean()),
      'mean_true_minus_other_logit': float(
          (true_logit - other_logit).mean()),
      'mean_entropy_nats': float(entropy.mean()),
  }


def reward_metrics(prediction, target, estimator):
  prediction = np.asarray(prediction, np.float64)
  target = np.asarray(target, np.float64)
  return {
      'estimator': estimator,
      'sign_accuracy': float(((prediction > 0) == (target > 0)).mean()),
      'mean_absolute_error': float(np.abs(prediction - target).mean()),
      'mean_prediction': float(prediction.mean()),
  }


def counterfactual_metrics(reward_samples, terminal_samples, cues, actions):
  """Summarize sampled p(S'|S,a); samples have shape [N, 2, S]."""
  rewards = np.asarray(reward_samples, np.float64)
  terminals = np.asarray(terminal_samples, np.float64)
  cues = np.asarray(cues, np.int32)
  actions = np.asarray(actions, np.int32)
  if rewards.ndim != 3 or rewards.shape[1] != 2:
    raise RuntimeError(f'Expected [N, 2, S] rewards, got {rewards.shape}.')
  if terminals.shape != rewards.shape:
    raise RuntimeError((terminals.shape, rewards.shape))
  means = rewards.mean(-1)
  greedy = means.argmax(-1)
  row = np.arange(len(cues))
  true_samples = rewards[row, cues]
  other_samples = rewards[row, 1 - cues]
  differences = true_samples - other_samples
  diff_mean = differences.mean(-1)
  if rewards.shape[-1] > 1:
    diff_se = differences.std(-1, ddof=1) / math.sqrt(rewards.shape[-1])
  else:
    diff_se = np.full(len(cues), np.nan)
  observed = means[row, actions]
  target = np.where(actions == cues, 1.0, -1.0)
  return {
      'estimator': 'monte_carlo_prior_expectation',
      'samples_per_action_state': int(rewards.shape[-1]),
      'model_greedy_accuracy': float((greedy == cues).mean()),
      'mean_correct_minus_wrong_reward': float(diff_mean.mean()),
      'median_correct_minus_wrong_reward': float(np.median(diff_mean)),
      'ambiguous_margin_fraction_95pct_normal_ci': float(
          (np.abs(diff_mean) <= 1.959963984540054 * diff_se).mean()),
      'observed_action_reward': reward_metrics(
          observed, target, 'monte_carlo_prior_mean'),
      'mean_predicted_terminal_probability': float(terminals.mean()),
      'mean_predicted_terminal_probability_by_action': [
          float(terminals[:, action].mean()) for action in (0, 1)],
  }


def map_counterfactual_metrics(rewards, terminals, cues, actions):
  rewards = np.asarray(rewards, np.float64)
  terminals = np.asarray(terminals, np.float64)
  cues = np.asarray(cues, np.int32)
  actions = np.asarray(actions, np.int32)
  if rewards.shape != (len(cues), 2):
    raise RuntimeError((rewards.shape, cues.shape))
  row = np.arange(len(cues))
  target = np.where(actions == cues, 1.0, -1.0)
  return {
      'estimator': 'deterministic_map_prior_state_not_expected_reward',
      'model_greedy_accuracy': float((rewards.argmax(-1) == cues).mean()),
      'mean_correct_minus_wrong_reward': float(
          (rewards[row, cues] - rewards[row, 1 - cues]).mean()),
      'observed_action_reward': reward_metrics(
          rewards[row, actions], target, 'deterministic_map_prior_state'),
      'mean_predicted_terminal_probability': float(terminals.mean()),
  }


def query_action_coverage(cues, actions):
  cues = np.asarray(cues, np.int32)
  actions = np.asarray(actions, np.int32)
  table = [[int(((cues == cue) & (actions == action)).sum())
            for action in (0, 1)] for cue in (0, 1)]
  return {
      'estimator': 'exact_replay_counts',
      'rows_are_cues_columns_are_actions': table,
      'cue_counts': [int((cues == cue).sum()) for cue in (0, 1)],
      'action_counts': [int((actions == action).sum()) for action in (0, 1)],
      'minimum_cell_count': int(np.min(table)),
  }


def _pad(value, size):
  value = np.asarray(value)
  missing = size - len(value)
  if missing < 0:
    raise ValueError((len(value), size))
  if not missing:
    return value
  padding = np.repeat(value[-1:], missing, axis=0)
  return np.concatenate([value, padding], axis=0)


def _tree_slice(tree, start, stop, size):
  return {key: _pad(value[start:stop], size) for key, value in tree.items()}


class CheckpointRunner:

  def __init__(self, logdir, config, batch_size, mc_samples, seed):
    # Heavy imports and device discovery happen only after CLI validation, so
    # --help and helper tests remain safe on login nodes.
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from embodied.jax import internal, nets, transform
    from dreamerv3 import main as dreamer_main

    raw = json.loads(json.dumps(config))
    raw['jax']['precompile'] = False
    raw['jax']['profiler'] = False
    raw['jax']['expect_devices'] = 1
    raw['logdir'] = str(logdir)
    agent_config = elements.Config(raw)
    self.agent = dreamer_main.make_agent(agent_config)
    latest_name = (logdir / 'ckpt' / 'latest').read_text().strip()
    self.checkpoint = logdir / 'ckpt' / latest_name
    if not (self.checkpoint / 'done').exists():
      raise RuntimeError(f'Incomplete checkpoint: {self.checkpoint}')
    checkpoint = elements.Checkpoint()
    checkpoint.agent = self.agent
    checkpoint.load(self.checkpoint, keys=['agent'])

    self.batch_size = batch_size
    self.mc_samples = mc_samples
    self.seed = seed
    self.jax = jax
    self.jnp = jnp
    self.internal = internal
    model = self.agent.model

    def heads(feat):
      tensor = model.feat2tensor(feat)
      policy = model.pol(tensor, 1)['action']
      reward = model.rew(tensor, 1).pred()
      continuation = model.con(tensor, 1).prob(
          jnp.ones(reward.shape, jnp.float32))
      return {
          'policy_logits': policy.logits,
          'policy_probs': jax.nn.softmax(policy.logits, -1),
          'reward': reward,
          'continuation': continuation,
      }

    def prior_sample(state, action):
      _, (feat, _) = model.dyn.imagine(
          state, {'action': action}, 1, False, single=True)
      output = heads(feat)
      return {
          'reward': output['reward'],
          'terminal_probability': 1 - output['continuation'],
      }

    def prior_map(state, action):
      action_embed = nets.DictConcat(model.dyn.act_space, 1)(
          {'action': action})
      deter = model.dyn._core(state['deter'], state['stoch'], action_embed)
      logit = model.dyn._prior(deter)
      stoch = model.dyn._dist(logit).pred()
      output = heads({'deter': deter, 'stoch': stoch, 'logit': logit})
      return {
          'reward': output['reward'],
          'terminal_probability': 1 - output['continuation'],
      }

    distance = int(config['env']['toymemory']['distance'])

    def full_sequence(observation, action, is_first):
      batch = len(observation)
      enc_carry = model.enc.initial(batch)
      dyn_carry = model.dyn.initial(batch)
      obs = {'observation': observation}
      enc_carry, _, tokens = model.enc(
          enc_carry, obs, is_first, training=False)
      previous = jnp.concatenate([
          jnp.zeros_like(action[:, :1]), action[:, :-1]], axis=1)
      _, _, feat = model.dyn.observe(
          dyn_carry, tokens, {'action': previous}, is_first,
          training=False)
      return {
          'query': {
              'deter': feat['deter'][:, distance],
              'stoch': feat['stoch'][:, distance],
          },
          'terminal': {
              'deter': feat['deter'][:, distance + 1],
              'stoch': feat['stoch'][:, distance + 1],
          },
      }

    def compile_hook(fn, inputs):
      _, activation_rules = self.agent.partition_rules
      return transform.apply(
          nj.pure(fn), self.agent.train_mesh,
          (self.agent.train_params_sharding, self.agent.train_mirrored) +
          (self.agent.train_sharded,) * inputs,
          (self.agent.train_sharded,), activation_rules,
          single_output=True, use_shardmap=self.agent.jaxcfg.use_shardmap)

    self.heads_hook = compile_hook(heads, 1)
    self.prior_sample_hook = compile_hook(prior_sample, 2)
    self.prior_map_hook = compile_hook(prior_map, 2)
    self.full_sequence_hook = compile_hook(full_sequence, 3)

  def _call(self, hook, seed, *inputs):
    inputs = [self.internal.device_put(x, self.agent.train_sharded)
              for x in inputs]
    output = hook(
        self.agent.params,
        self.agent._seeds(seed, self.agent.train_mirrored),
        *inputs)
    output = self.internal.fetch_async(output)
    return self.agent._take_outs(output)

  def heads(self, states, seed_offset=0):
    outputs = []
    total = len(states['deter'])
    for index, start in enumerate(range(0, total, self.batch_size)):
      stop = min(start + self.batch_size, total)
      batch = _tree_slice(states, start, stop, self.batch_size)
      output = self._call(self.heads_hook, self.seed + seed_offset + index,
                          batch)
      outputs.append({key: value[:stop - start]
                      for key, value in output.items()})
    return {key: np.concatenate([part[key] for part in outputs], 0)
            for key in outputs[0]}

  def counterfactual(self, states, seed_offset=10_000):
    sampled_parts = []
    map_parts = []
    total = len(states['deter'])
    bsize = self.batch_size
    samples = self.mc_samples
    for index, start in enumerate(range(0, total, bsize)):
      stop = min(start + bsize, total)
      state = _tree_slice(states, start, stop, bsize)
      expanded = {
          key: np.repeat(value[:, None, None], 2, axis=1)
              .repeat(samples, axis=2)
              .reshape((bsize * 2 * samples, *value.shape[1:]))
          for key, value in state.items()}
      actions = np.broadcast_to(
          np.arange(2, dtype=np.int32)[None, :, None],
          (bsize, 2, samples)).reshape(-1)
      sampled = self._call(
          self.prior_sample_hook, self.seed + seed_offset + index,
          expanded, actions)
      sampled_parts.append({
          key: value.reshape((bsize, 2, samples))[:stop - start]
          for key, value in sampled.items()})

      map_state = {
          key: np.repeat(value[:, None], 2, axis=1)
              .reshape((bsize * 2, *value.shape[1:]))
          for key, value in state.items()}
      map_actions = np.broadcast_to(
          np.arange(2, dtype=np.int32)[None], (bsize, 2)).reshape(-1)
      mapped = self._call(
          self.prior_map_hook, self.seed + seed_offset + 5000 + index,
          map_state, map_actions)
      map_parts.append({
          key: value.reshape((bsize, 2))[:stop - start]
          for key, value in mapped.items()})
    sampled = {
        key: np.concatenate([part[key] for part in sampled_parts], 0)
        for key in sampled_parts[0]}
    mapped = {
        key: np.concatenate([part[key] for part in map_parts], 0)
        for key in map_parts[0]}
    return sampled, mapped

  def reconstruct(self, artifact, posterior_sample):
    query_parts = {'deter': [], 'stoch': []}
    terminal_parts = {'deter': [], 'stoch': []}
    total = artifact.episodes
    for index, start in enumerate(range(0, total, self.batch_size)):
      stop = min(start + self.batch_size, total)
      observation = _pad(
          artifact.observations[start:stop], self.batch_size)
      action = _pad(artifact.actions[start:stop], self.batch_size)
      is_first = _pad(artifact.is_first[start:stop], self.batch_size)
      output = self._call(
          self.full_sequence_hook,
          self.seed + 100_000 * (posterior_sample + 1) + index,
          observation, action, is_first)
      for key in query_parts:
        query_parts[key].append(output['query'][key][:stop - start])
        terminal_parts[key].append(output['terminal'][key][:stop - start])
    query = {key: np.concatenate(value, 0)
             for key, value in query_parts.items()}
    terminal = {key: np.concatenate(value, 0)
                for key, value in terminal_parts.items()}
    return query, terminal


def _feature_sets(states):
  deter = np.asarray(states['deter'], np.float32)
  stoch = np.asarray(states['stoch'], np.float32).reshape((len(deter), -1))
  return {
      'deter': deter,
      'stoch': stoch,
      'deter_and_stoch': np.concatenate([deter, stoch], -1),
  }


def state_audit(states, terminal_states, runner, artifact, seed_offset=0):
  heads = runner.heads(states, seed_offset)
  terminal_heads = runner.heads(terminal_states, seed_offset + 1000)
  sampled, mapped = runner.counterfactual(states, seed_offset + 2000)
  return {
      'query_policy': policy_metrics(
          heads['policy_logits'], heads['policy_probs'], artifact.cues,
          'exact_current_policy_given_cached_hard_state'),
      'posterior_terminal_reward': reward_metrics(
          terminal_heads['reward'], artifact.terminal_rewards,
          'exact_head_readout_given_hard_posterior_state'),
      'prior_counterfactual_mc': counterfactual_metrics(
          sampled['reward'], sampled['terminal_probability'], artifact.cues,
          artifact.query_actions),
      'prior_counterfactual_map': map_counterfactual_metrics(
          mapped['reward'], mapped['terminal_probability'], artifact.cues,
          artifact.query_actions),
      '_policy_probs': heads['policy_probs'],
      '_policy_logits': heads['policy_logits'],
      '_terminal_reward': terminal_heads['reward'],
      '_sampled_reward': sampled['reward'],
      '_sampled_terminal': sampled['terminal_probability'],
      '_map_reward': mapped['reward'],
      '_map_terminal': mapped['terminal_probability'],
  }


def remove_private(mapping):
  return {key: value for key, value in mapping.items()
          if not key.startswith('_')}


def git_revision(root):
  try:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def audit(args):
  logdir = args.logdir.resolve()
  output = args.output.resolve()
  if not logdir.is_dir():
    raise RuntimeError(f'Missing ToyMemory artifact directory: {logdir}')
  try:
    output.relative_to(logdir)
  except ValueError:
    pass
  else:
    raise RuntimeError('Audit output must be outside the source artifact.')
  if output.exists():
    raise RuntimeError(f'Refusing to overwrite audit output: {output}')
  config = yaml.YAML(typ='safe').load((logdir / 'config.yaml').read_text())
  if config.get('task') != 'toymemory_onebit':
    raise RuntimeError(f'Expected task toymemory_onebit, got {config.get("task")}')
  artifact, chain = load_toy_artifact(logdir, config)
  runner = CheckpointRunner(
      logdir, config, args.batch_size, args.mc_samples, args.seed)

  cached = state_audit(
      artifact.cached_query, artifact.cached_terminal, runner, artifact)
  folds = fixed_probe_folds(artifact.query_stepids)
  cached_probes = {
      name: nearest_centroid_probe(features, artifact.cues, folds)
      for name, features in _feature_sets(artifact.cached_query).items()}

  full_audits = []
  first_query = None
  for sample in range(args.posterior_samples):
    query, terminal = runner.reconstruct(artifact, sample)
    first_query = query if first_query is None else first_query
    full_audits.append(state_audit(
        query, terminal, runner, artifact,
        seed_offset=1_000_000 * (sample + 1)))

  policy_probs = np.mean(
      [value['_policy_probs'] for value in full_audits], axis=0)
  policy_logits = np.mean(
      [value['_policy_logits'] for value in full_audits], axis=0)
  terminal_reward = np.mean(
      [value['_terminal_reward'] for value in full_audits], axis=0)
  sampled_reward = np.concatenate(
      [value['_sampled_reward'] for value in full_audits], axis=-1)
  sampled_terminal = np.concatenate(
      [value['_sampled_terminal'] for value in full_audits], axis=-1)
  map_reward = np.mean(
      [value['_map_reward'] for value in full_audits], axis=0)
  map_terminal = np.mean(
      [value['_map_terminal'] for value in full_audits], axis=0)
  full = {
      'aggregation': {
          'estimator': 'mean_over_fixed_seed_full_posterior_reconstructions',
          'posterior_samples': args.posterior_samples,
          'note': 'Posterior states are stochastic; this is not historical '
                  'training-state reconstruction.'},
      'query_policy': policy_metrics(
          policy_logits, policy_probs, artifact.cues,
          'mean_over_fixed_seed_full_posterior_reconstructions'),
      'posterior_terminal_reward': reward_metrics(
          terminal_reward, artifact.terminal_rewards,
          'mean_over_fixed_seed_full_posterior_reconstructions'),
      'prior_counterfactual_mc': counterfactual_metrics(
          sampled_reward, sampled_terminal, artifact.cues,
          artifact.query_actions),
      'prior_counterfactual_map': map_counterfactual_metrics(
          map_reward, map_terminal, artifact.cues, artifact.query_actions),
  }
  full_probes = {
      name: nearest_centroid_probe(features, artifact.cues, folds)
      for name, features in _feature_sets(first_query).items()}

  latest = runner.checkpoint
  with (latest / 'step.pkl').open('rb') as handle:
    checkpoint_step = int(pickle.load(handle))
  root = Path(__file__).resolve().parents[1]
  result = {
      'schema_version': SCHEMA_VERSION,
      'kind': 'toymemory_checkpoint_functional_audit',
      'read_only_contract': {
          'agent_train_called': False,
          'replay_update_called': False,
          'source_artifact_modified': False,
      },
      'estimator_boundaries': {
          'policy_probs': 'exact conditional on each supplied hard RSSM state',
          'map': 'deterministic MAP next state; not expected reward',
          'mc': 'Monte Carlo estimate of stochastic prior expected reward',
          'linear_probe': 'fixed held-out linear separability proxy; not '
                          'intrinsic or exact cue decodability',
          'historical_reconstruction': 'unavailable because replay does not '
              'store parameter versions or stochastic RNG keys',
      },
      'artifact': {
          'logdir': str(logdir),
          'checkpoint': str(latest),
          'checkpoint_step': checkpoint_step,
          'checkpoint_updates': int(runner.agent.n_updates),
          'configured_steps': int(config['run']['steps']),
          'distance': artifact.distance,
          'episode_length': artifact.episode_length,
          'episodes': artifact.episodes,
          'replay_chunks': len(chain),
          'replay_rows': int(sum(value.length for value in chain)),
          'training_git_revision': (
              (logdir / 'provenance' / 'git-revision.txt').read_text().strip()
              if (logdir / 'provenance' / 'git-revision.txt').exists()
              else None),
          'audit_git_revision': git_revision(root),
      },
      'audit_parameters': {
          'batch_size': args.batch_size,
          'mc_samples_per_action_state': args.mc_samples,
          'full_posterior_samples': args.posterior_samples,
          'seed': args.seed,
      },
      'exact_replay_evidence': {
          'query_action_coverage': query_action_coverage(
              artifact.cues, artifact.query_actions),
          'observed_terminal_reward': {
              'positive': int((artifact.terminal_rewards > 0).sum()),
              'negative': int((artifact.terminal_rewards < 0).sum()),
              'accuracy': float((artifact.terminal_rewards > 0).mean()),
          },
      },
      'cached_replay_state': remove_private(cached),
      'full_current_posterior_state': full,
      'linear_separability_proxy': {
          'split': 'SHA256(query_stepid) modulo 3; fold 2 held out',
          'cached_replay_state': cached_probes,
          'full_current_first_fixed_seed_state': full_probes,
      },
  }
  text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(f'.{output.name}.tmp-{os.getpid()}')
  temporary.write_text(text)
  os.replace(temporary, output)
  print(text, end='')


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Read-only functional audit of a ToyMemory checkpoint.')
  parser.add_argument('logdir', type=Path,
                      help='Completed ToyMemory artifact directory.')
  parser.add_argument('--batch-size', type=positive_int, default=16,
                      help='Fixed JAX episode batch size (default: 16).')
  parser.add_argument('--mc-samples', type=positive_int, default=64,
                      help='Prior samples per state and action (default: 64).')
  parser.add_argument('--posterior-samples', type=positive_int, default=4,
                      help='Full-current posterior reconstructions (default: 4).')
  parser.add_argument('--seed', type=int, default=0,
                      help='Deterministic audit RNG seed (default: 0).')
  parser.add_argument('--output', type=Path, required=True,
                      help='New JSON path outside the source artifact.')
  return parser.parse_args(argv)


def main(argv=None):
  audit(parse_args(argv))


if __name__ == '__main__':
  main()
