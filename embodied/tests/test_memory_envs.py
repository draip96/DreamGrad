"""Literal cue/query and BSuite MemoryChain geometry tests."""

import numpy as np
import pytest
import ruamel.yaml as yaml

import elements

from embodied.envs.toy_memory import ToyMemory


@pytest.mark.parametrize('distance', (1, 8, 257))
def test_toy_memory_uses_literal_cue_query_distance(distance):
  env = ToyMemory('onebit', distance=distance, seed=11)
  obs = env.step({'reset': True, 'action': np.int32(0)})
  cue = int(obs['observation'][1] > 0)
  assert obs['is_first']
  assert not obs['observation'][2]

  for index in range(1, distance + 1):
    obs = env.step({'reset': False, 'action': np.int32(1 - cue)})
    assert not obs['is_last']
    assert bool(obs['observation'][2]) == (index == distance)

  obs = env.step({'reset': False, 'action': np.int32(cue)})
  assert obs['is_last']
  assert obs['is_terminal']
  assert obs['reward'] == np.float32(1.0)
  assert obs['log/correct'] == np.float32(1.0)


@pytest.mark.parametrize('memory_length', (11, 17, 25, 31, 71))
def test_bsuite_adapter_uses_literal_memory_length(memory_length):
  pytest.importorskip('bsuite')
  from embodied.envs.bsuite import BSuite

  env = BSuite('memory', memory_length=memory_length, seed=13)
  assert env.env.env._env._memory_length == memory_length
  first = env.step({'reset': True, 'action': np.int32(0)})
  assert first['is_first']
  assert first['observation'].shape == (3,)
  assert first['observation'][0] == np.float32(1.0)
  cue_action = np.int32(first['observation'][2] > 0)

  # BSuite repeats its t=0 observation on the first action, unlike ToyMemory.
  obs = env.step({'reset': False, 'action': np.int32(1 - cue_action)})
  np.testing.assert_array_equal(obs['observation'], first['observation'])
  assert not obs['is_first']
  assert not obs['is_last']

  # Physical row q_m is the decision observation. For num_bits=1 the query
  # index itself is zero, so timing is identified by time-to-live 1 / m.
  for physical_row in range(2, memory_length + 1):
    obs = env.step({'reset': False, 'action': np.int32(1 - cue_action)})
    expected = 1.0 - (physical_row - 1) / memory_length
    np.testing.assert_allclose(obs['observation'][0], expected, atol=1e-7)
    assert not obs['is_last']
  np.testing.assert_allclose(
      obs['observation'][0], 1.0 / memory_length, atol=1e-7)

  terminal = env.step({'reset': False, 'action': cue_action})
  assert terminal['is_last']
  assert terminal['is_terminal']
  assert terminal['reward'] == np.float32(1.0)
  assert terminal['observation'][0] == np.float32(0.0)
  assert env.max_episodes == 10_000


@pytest.mark.parametrize('profile', ('toy_memory', 'bsuite'))
def test_memory_profiles_resolve_to_upstream_size12m(profile):
  configs = yaml.YAML(typ='safe').load(
      elements.Path('dreamerv3/configs.yaml').read())
  config = elements.Config(configs['defaults']).update(configs[profile])
  rssm = config.agent.dyn.rssm
  assert (rssm.deter, rssm.hidden, rssm.stoch, rssm.classes) == (
      2048, 256, 32, 16)
  assert config.agent.enc.simple.depth == 16
  assert config.agent.enc.simple.units == 256
  assert config.agent.dec.simple.depth == 16
  assert config.agent.dec.simple.units == 256
  assert config.agent.policy.units == 256
  assert config.agent.value.units == 256
  assert config.batch_size == 16
  assert config.batch_length == 64
  assert config.replay_context == 1
  assert config.replay.size == 5e6
  assert config.run.train_ratio == 1024
  assert config.agent.opt.lr == 4e-5
  assert config.agent.gradient_cache.enabled is True
  if profile == 'bsuite':
    assert config.env.bsuite.memory_length == 11
    assert config.run.steps == 130_000
