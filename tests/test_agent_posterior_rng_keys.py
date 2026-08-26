"""Agent wiring regressions for immutable replay posterior RNG keys."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamerv3 import agent as agentlib


def _config(*, cache=True, keys=False, replay_context=1, report=True):
  return SimpleNamespace(
      gradient_cache=SimpleNamespace(
          enabled=cache, posterior_rng_keys=keys),
      replay_context=replay_context,
      report=report,
  )


def _raw_data(batch=2, length=4):
  return {
      'is_first': np.zeros((batch, length), bool),
      'is_last': np.zeros((batch, length), bool),
      'is_terminal': np.zeros((batch, length), bool),
      'stepid': np.arange(
          batch * length * 20, dtype=np.uint8).reshape(batch, length, 20),
      'rng/posterior': np.arange(
          batch * length * 2, dtype=np.uint32).reshape(batch, length, 2),
      'grad/deter': np.zeros((batch, length, 3), np.float32),
      'grad/stoch': np.zeros((batch, length, 2, 4), np.float32),
      'grad/valid': np.zeros((batch, length), bool),
      'action': np.zeros((batch, length, 1), np.float32),
  }


def test_replay_keys_validate_raw_shape_dtype_and_slice_physical_prefix():
  data = _raw_data()
  keys = agentlib._replay_posterior_keys(data, enabled=True)
  np.testing.assert_array_equal(keys, data['rng/posterior'][:, 1:])

  assert agentlib._replay_posterior_keys(
      {'is_first': data['is_first']}, enabled=False) is None
  with pytest.raises(ValueError, match="Missing required replay field"):
    agentlib._replay_posterior_keys(
        {'is_first': data['is_first']}, enabled=True)

  malformed = dict(data)
  malformed['rng/posterior'] = np.zeros((2, 4, 3), np.uint32)
  with pytest.raises(ValueError, match=r"must have shape \(2, 4, 2\)"):
    agentlib._replay_posterior_keys(malformed, enabled=True)

  malformed = dict(data)
  malformed['rng/posterior'] = np.zeros((2, 4, 2), np.int32)
  with pytest.raises(ValueError, match='must have dtype uint32'):
    agentlib._replay_posterior_keys(malformed, enabled=True)


def test_subflag_requires_saved_gradients_and_one_context_row():
  instance = object.__new__(agentlib.Agent)
  with pytest.raises(ValueError, match='requires saved gradients'):
    agentlib.Agent.__init__(
        instance, {}, {}, _config(cache=False, keys=True))
  with pytest.raises(ValueError, match='requires replay_context=1'):
    agentlib.Agent.__init__(
        instance, {}, {}, _config(cache=True, keys=True, replay_context=0))


def test_ext_space_adds_uint32_pair_only_when_enabled():
  base = SimpleNamespace(
      config=_config(cache=False, keys=False, replay_context=0),
      posterior_rng_keys=False,
      enc=SimpleNamespace(entry_space={}),
      dyn=SimpleNamespace(entry_space={}, deter=3, stoch=2, classes=4),
      dec=SimpleNamespace(entry_space={}),
  )
  assert agentlib._POSTERIOR_KEY not in agentlib.Agent.ext_space.fget(base)

  keyed = SimpleNamespace(
      config=_config(cache=True, keys=True, replay_context=1),
      posterior_rng_keys=True,
      enc=SimpleNamespace(entry_space={}),
      dyn=SimpleNamespace(entry_space={}, deter=3, stoch=2, classes=4),
      dec=SimpleNamespace(entry_space={}),
  )
  space = agentlib.Agent.ext_space.fget(keyed)[agentlib._POSTERIOR_KEY]
  assert space.shape == (2,)
  assert np.dtype(space.dtype) == np.dtype(np.uint32)


class _PolicyDynamics:

  deter = 3
  stoch = 2
  classes = 4

  def __init__(self):
    self.posterior_keys = 'unset'

  def observe(
      self, carry, tokens, prevact, reset, *, posterior_keys='absent', **kw):
    self.posterior_keys = posterior_keys
    batch = len(reset)
    entry = {
        'deter': jnp.zeros((batch, self.deter), jnp.float32),
        'stoch': jnp.zeros(
            (batch, self.stoch, self.classes), jnp.float32),
    }
    feat = {**entry, 'logit': jnp.zeros_like(entry['stoch'])}
    return entry, entry, feat


def _policy_agent(keys):
  dynamics = _PolicyDynamics()
  agent = SimpleNamespace(
      posterior_rng_keys=keys,
      config=_config(cache=True, keys=keys, replay_context=1),
      enc=lambda carry, obs, reset, **kw: (
          carry, {}, jnp.zeros((len(reset), 5), jnp.float32)),
      dyn=dynamics,
      dec=None,
      pol=lambda feat, bdims: None,
      feat2tensor=lambda feat: feat['deter'],
  )
  batch = 3
  carry = (
      {},
      {
          'deter': jnp.zeros((batch, dynamics.deter), jnp.float32),
          'stoch': jnp.zeros(
              (batch, dynamics.stoch, dynamics.classes), jnp.float32),
      },
      {},
      {'action': jnp.zeros((batch, 1), jnp.float32)},
  )
  obs = {
      'is_first': jnp.zeros((batch,), bool),
      'observation': jnp.zeros((batch, 2), jnp.float32),
  }
  return agent, dynamics, carry, obs


def test_policy_uses_one_posterior_rng_slot_and_persists_split_row_keys(
    monkeypatch):
  agent, dynamics, carry, obs = _policy_agent(keys=True)
  slot = jnp.asarray([123, 456], jnp.uint32)
  calls = []

  def seed():
    calls.append(None)
    return slot

  monkeypatch.setattr(agentlib.nj, 'seed', seed)
  monkeypatch.setattr(
      agentlib, 'sample',
      lambda unused: {'action': jnp.ones((len(obs['is_first']), 1), jnp.float32)})
  _, _, out = agentlib.Agent.policy(agent, carry, obs)

  expected = jax.random.split(slot, len(obs['is_first']))
  assert len(calls) == 1
  np.testing.assert_array_equal(out[agentlib._POSTERIOR_KEY], expected)
  np.testing.assert_array_equal(dynamics.posterior_keys, expected)
  assert out[agentlib._POSTERIOR_KEY].dtype == jnp.uint32


def test_default_policy_does_not_allocate_store_or_forward_posterior_keys(
    monkeypatch):
  agent, dynamics, carry, obs = _policy_agent(keys=False)
  monkeypatch.setattr(
      agentlib.nj, 'seed',
      lambda: pytest.fail('default policy unexpectedly consumed an RNG slot'))
  monkeypatch.setattr(
      agentlib, 'sample',
      lambda unused: {'action': jnp.ones((len(obs['is_first']), 1), jnp.float32)})
  _, _, out = agentlib.Agent.policy(agent, carry, obs)

  assert dynamics.posterior_keys == 'absent'
  assert agentlib._POSTERIOR_KEY not in out


def test_train_passes_raw_rows_one_onward_and_excludes_keys_from_updates():
  data = _raw_data()
  batch, raw_length = data['is_first'].shape
  length = raw_length - 1
  dyn_carry = {
      'deter': jnp.zeros((batch, 3), jnp.float32),
      'stoch': jnp.zeros((batch, 2, 4), jnp.float32),
  }
  carry3 = ({}, dyn_carry, {})
  obs = {
      'is_first': jnp.zeros((batch, length), bool),
      'is_last': jnp.zeros((batch, length), bool),
      'is_terminal': jnp.zeros((batch, length), bool),
  }
  prevact = {'action': jnp.zeros((batch, length, 1), jnp.float32)}
  entries = ({}, {
      'deter': jnp.zeros((batch, length, 3), jnp.float32),
      'stoch': jnp.zeros((batch, length, 2, 4), jnp.float32),
  }, {})
  seen = {}

  def optimizer(*args, **kwargs):
    seen['args'] = args
    seen['kwargs'] = kwargs
    incoming = ({
        'deter': jnp.ones((batch, length, 3), jnp.float32),
        'stoch': jnp.ones((batch, length, 2, 4), jnp.float32),
    },)
    return {}, (carry3, entries, {}, {}), incoming

  agent = SimpleNamespace(
      config=_config(cache=True, keys=True, replay_context=1),
      posterior_rng_keys=True,
      act_space={'action': None},
      loss=lambda *args, **kwargs: None,
      opt=optimizer,
      slowval=SimpleNamespace(update=lambda: None),
      _apply_replay_context=lambda carry, raw: (
          carry3, obs, prevact, raw['stepid'][:, 1:]),
  )
  initial = (*carry3, {'action': jnp.zeros((batch, 1), jnp.float32)})
  _, outs, _ = agentlib.Agent.train(agent, initial, data)

  assert seen['args'][0] is agent.loss
  np.testing.assert_array_equal(
      seen['args'][6], data[agentlib._POSTERIOR_KEY][:, 1:])
  assert seen['kwargs']['training'] is True
  assert all(
      agentlib._POSTERIOR_KEY not in payload
      for payload in outs['replay'])


def test_report_and_loss_forward_the_sliced_keys_to_rssm():
  data = _raw_data()
  expected = data[agentlib._POSTERIOR_KEY][:, 1:]
  seen = {}

  class StopReport(Exception):
    pass

  def report_loss(carry, obs, prevact, **kwargs):
    seen['report'] = kwargs['posterior_keys']
    raise StopReport

  report_agent = SimpleNamespace(
      config=_config(cache=True, keys=True, replay_context=1, report=True),
      posterior_rng_keys=True,
      loss=report_loss,
      _apply_replay_context=lambda carry, raw: (
          ({}, {}, {}),
          {'is_first': raw['is_first'][:, 1:]},
          {'action': raw['action'][:, 1:]},
          raw['stepid'][:, 1:]),
  )
  with pytest.raises(StopReport):
    agentlib.Agent.report(report_agent, None, data)
  np.testing.assert_array_equal(seen['report'], expected)

  class StopRSSM(Exception):
    pass

  class Dynamics:
    def loss(self, *args, **kwargs):
      seen['rssm'] = kwargs['posterior_keys']
      raise StopRSSM

  loss_agent = SimpleNamespace(
      enc=lambda carry, obs, reset, training: (carry, {}, jnp.zeros(
          (*reset.shape, 5), jnp.float32)),
      dyn=Dynamics(),
  )
  obs = {'is_first': jnp.zeros(expected.shape[:2], bool)}
  with pytest.raises(StopRSSM):
    agentlib.Agent.loss(
        loss_agent, ({}, {}, {}), obs, {'action': jnp.zeros(
            (*expected.shape[:2], 1), jnp.float32)},
        posterior_keys=expected)
  np.testing.assert_array_equal(seen['rssm'], expected)


def test_cache_payload_builder_never_emits_posterior_keys():
  data = _raw_data()
  state = {
      'stepid': data['stepid'][:, 1:],
      'dyn/deter': np.ones((2, 3, 3), np.float32),
      agentlib._POSTERIOR_KEY: data[agentlib._POSTERIOR_KEY][:, 1:],
  }
  grad = {
      'stepid': data['stepid'][:, :-1],
      'grad/deter': np.ones((2, 3, 3), np.float32),
      agentlib._POSTERIOR_KEY: data[agentlib._POSTERIOR_KEY][:, :-1],
  }
  payloads = agentlib._gradient_cache_payloads(
      data['stepid'], state, grad)
  assert len(payloads) == 3
  assert all(agentlib._POSTERIOR_KEY not in payload for payload in payloads)
  assert all('stepid' in payload for payload in payloads)
