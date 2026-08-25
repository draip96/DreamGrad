"""Frozen-parameter numerical oracles for saved recurrent-state gradients."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np


def _params():
  return {
      'dd': jnp.array([[0.61, -0.17], [0.12, 0.53]], jnp.float32),
      'ds': jnp.array([[0.21, 0.34], [-0.28, 0.19]], jnp.float32),
      'di': jnp.array([[0.37], [-0.41]], jnp.float32),
      'sd': jnp.array([[-0.14, 0.46], [0.31, 0.22]], jnp.float32),
      'ss': jnp.array([[0.52, 0.08], [-0.11, 0.64]], jnp.float32),
      'si': jnp.array([[-0.26], [0.33]], jnp.float32),
      'out': jnp.array([[0.73, -0.36, 0.28, 0.41]], jnp.float32),
  }


def _segment(
    params, inputs, targets, initial, *, taps=None, resets=None, weights=None,
    future=None):
  """Two-leaf recurrent chunk with DreamGrad's tap/reset/surrogate order."""
  time, batch = inputs.shape[:2]
  if taps is None:
    taps = {
        key: jnp.zeros((time, *value.shape), jnp.float32)
        for key, value in initial.items()}
  if resets is None:
    resets = jnp.zeros((time, batch), bool)
  if weights is None:
    weights = jnp.ones((time, batch), jnp.float32)

  def step(state, args):
    inp, tap, reset = args
    incoming = {
        key: state[key] + tap[key] for key in ('deter', 'stoch')}
    incoming = {
        key: jnp.where(reset[:, None], 0.0, value)
        for key, value in incoming.items()}
    deter = jnp.tanh(
        incoming['deter'] @ params['dd'].T +
        incoming['stoch'] @ params['ds'].T + inp @ params['di'].T)
    stoch = jnp.tanh(
        incoming['deter'] @ params['sd'].T +
        incoming['stoch'] @ params['ss'].T + inp @ params['si'].T)
    state = {'deter': deter, 'stoch': stoch}
    feature = jnp.concatenate([deter, stoch], -1)
    pred = feature @ params['out'].T
    return state, pred[..., 0]

  final, preds = jax.lax.scan(step, initial, (inputs, taps, resets))
  squared = (preds - targets) ** 2
  loss = (squared * weights).sum() / jnp.maximum(weights.sum(), 1.0)
  if future is not None:
    per_example = sum(
        (jax.lax.stop_gradient(future[key]) *
         (final[key] - jax.lax.stop_gradient(final[key]))).sum(-1)
        for key in ('deter', 'stoch'))
    loss += per_example.mean()
  return loss, final


def _quality(reference, candidate):
  ref = np.concatenate([
      np.asarray(x).reshape(-1) for x in jax.tree.leaves(reference)])
  got = np.concatenate([
      np.asarray(x).reshape(-1) for x in jax.tree.leaves(candidate)])
  relative = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)
  cosine = float(np.dot(ref, got) / max(
      np.linalg.norm(ref) * np.linalg.norm(got), 1e-30))
  return relative, cosine


class GradientCacheOracleTest(unittest.TestCase):

  def setUp(self):
    self.params = _params()
    self.future_params = jax.tree.map(lambda x: 0.93 * x + 0.01, self.params)
    self.initial = {
        'deter': jnp.array([[0.17, -0.23]], jnp.float32),
        'stoch': jnp.array([[0.31, 0.09]], jnp.float32),
    }
    self.x1 = jnp.array(
        [[[0.2]], [[-0.4]], [[0.1]], [[0.6]]], jnp.float32)
    self.x2 = jnp.array(
        [[[-0.3]], [[0.5]], [[0.7]], [[-0.2]]], jnp.float32)
    self.y1 = jnp.array([[0.2], [-0.1], [0.4], [-0.3]], jnp.float32)
    self.y2 = -self.y1

  def test_cached_boundary_matches_full_bptt_parameter_contribution(self):
    _, boundary = _segment(
        self.params, self.x1, self.y1, self.initial)
    future_loss = lambda state: _segment(
        self.future_params, self.x2, self.y2, state)[0]
    future = jax.grad(future_loss)(boundary)

    def full_loss(params):
      local, middle = _segment(params, self.x1, self.y1, self.initial)
      return local + future_loss(middle)

    def cached_loss(params):
      return _segment(
          params, self.x1, self.y1, self.initial, future=future)[0]

    reference = jax.grad(full_loss)(self.params)
    cached = jax.grad(cached_loss)(self.params)
    relative, cosine = _quality(reference, cached)
    self.assertLess(relative, 1e-5)
    self.assertGreater(cosine, 0.99999)

    deter_only = dict(future)
    deter_only['stoch'] = jnp.zeros_like(deter_only['stoch'])
    incomplete = jax.grad(lambda params: _segment(
        params, self.x1, self.y1, self.initial,
        future=deter_only)[0])(self.params)
    relative, _ = _quality(reference, incomplete)
    self.assertGreater(relative, 1e-3)

  def test_shared_parameter_segment_contributions_sum_to_full_bptt(self):
    segments = (
        (self.x1, self.y1),
        (self.x2, self.y2),
        (0.7 * self.x1 - 0.1, 0.4 * self.y1 + 0.2),
    )

    def rollout(params, initial, values):
      total = jnp.array(0.0, jnp.float32)
      state = initial
      for inputs, targets in values:
        local, state = _segment(params, inputs, targets, state)
        total += local
      return total, state

    states = [self.initial]
    for inputs, targets in segments:
      _, state = _segment(self.params, inputs, targets, states[-1])
      states.append(jax.tree.map(jax.lax.stop_gradient, state))

    messages = []
    for index in range(len(segments)):
      suffix = segments[index + 1:]
      if suffix:
        message = jax.grad(
            lambda state, values=suffix: rollout(
                self.params, state, values)[0])(states[index + 1])
      else:
        message = jax.tree.map(jnp.zeros_like, states[index + 1])
      messages.append(message)

    contributions = []
    for index, (inputs, targets) in enumerate(segments):
      contributions.append(jax.grad(lambda params, index=index: _segment(
          params, inputs, targets, states[index],
          future=messages[index])[0])(self.params))
    segmented = jax.tree.map(lambda *xs: sum(xs), *contributions)
    reference = jax.grad(
        lambda params: rollout(params, self.initial, segments)[0])(self.params)
    relative, cosine = _quality(reference, segmented)
    self.assertLess(relative, 1e-5)
    self.assertGreater(cosine, 0.99999)

  def test_zero_primal_surrogate_preserves_reported_loss(self):
    _, boundary = _segment(
        self.params, self.x1, self.y1, self.initial)
    future = jax.grad(lambda state: _segment(
        self.future_params, self.x2, self.y2, state)[0])(boundary)
    native = _segment(self.params, self.x1, self.y1, self.initial)[0]
    cached = _segment(
        self.params, self.x1, self.y1, self.initial, future=future)[0]
    self.assertEqual(float(native), float(cached))

  def test_per_example_messages_are_batch_size_invariant(self):
    def message(batch):
      initial = jax.tree.map(lambda x: jnp.repeat(x, batch, 0), self.initial)
      inputs = jnp.repeat(self.x1, batch, 1)
      targets = jnp.repeat(self.y1, batch, 1)
      taps = {
          key: jnp.zeros((len(inputs), *value.shape), jnp.float32)
          for key, value in initial.items()}
      gradient = jax.grad(lambda value: _segment(
          self.params, inputs, targets, initial, taps=value)[0])(taps)
      return jax.tree.map(lambda x: batch * x[:, 0], gradient)

    one = message(1)
    eight = message(8)
    for expected, actual in zip(jax.tree.leaves(one), jax.tree.leaves(eight)):
      np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

  def test_reset_severs_predecessor_adjoint(self):
    taps = {
        key: jnp.zeros((len(self.x1), *value.shape), jnp.float32)
        for key, value in self.initial.items()}
    resets = jnp.array([[False], [False], [True], [False]])
    weights = jnp.array([[0.0], [0.0], [1.0], [1.0]], jnp.float32)
    gradient = jax.grad(lambda value: _segment(
        self.params, self.x1, self.y1, self.initial,
        taps=value, resets=resets, weights=weights)[0])(taps)
    for value in gradient.values():
      np.testing.assert_array_equal(np.asarray(value[:3]), 0.0)
      self.assertGreater(float(jnp.linalg.norm(value[3])), 0.0)

  def test_joint_parameter_and_tap_reverse_has_one_primal_trace(self):
    calls = {'count': 0}
    taps = {
        key: jnp.zeros((len(self.x1), *value.shape), jnp.float32)
        for key, value in self.initial.items()}

    def objective(params, value):
      calls['count'] += 1
      return _segment(
          params, self.x1, self.y1, self.initial, taps=value)[0]

    _, gradients = jax.value_and_grad(objective, argnums=(0, 1))(
        self.params, taps)
    self.assertEqual(calls['count'], 1)
    self.assertEqual(len(gradients), 2)


if __name__ == '__main__':
  unittest.main()
