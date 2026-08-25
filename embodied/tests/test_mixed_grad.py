"""Joint Ninjax module/input gradient tests for saved adjoints."""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from embodied.jax.opt import Optimizer, _grad_with_inputs


class Scale(nj.Module):

  def loss(self, inputs, taps):
    weight = self.value('weight', jnp.asarray, 0.7, jnp.float32)
    noise = 0.05 * jax.random.normal(nj.seed(), inputs.shape)
    residual = weight * inputs + taps + noise
    return jnp.mean(residual ** 2), residual

  def split_loss(self, inputs, taps):
    weight = self.value('weight', jnp.asarray, 0.7, jnp.float32)
    noise = 0.05 * jax.random.normal(nj.seed(), inputs.shape)
    param_residual = weight * inputs + taps + noise
    input_residual = weight * inputs + 2.0 * taps + noise
    objectives = (
        jnp.mean(param_residual ** 2),
        jnp.mean(input_residual ** 2),
    )
    return objectives, (param_residual, input_residual)


class TestMixedGrad:

  def test_module_and_input_gradients_share_one_stochastic_forward(self):
    module = Scale(name='scale')

    def program(inputs, taps):
      return _grad_with_inputs(
          module.loss, [module], input_argnums=(1,), has_aux=True)(
              inputs, taps)

    inputs = jnp.asarray([0.2, -0.4, 0.8], jnp.float32)
    taps = jnp.asarray([0.1, 0.3, -0.2], jnp.float32)
    state, result = nj.pure(program)(
        {}, inputs, taps, seed=jnp.asarray([3, 7], jnp.uint32), create=True)
    loss, params, param_grads, residual, input_grads = result
    (tap_grads,) = input_grads

    assert np.isfinite(float(loss))
    assert set(params) == {'scale/weight'}
    assert set(param_grads) == {'scale/weight'}
    expected_taps = 2 * np.asarray(residual) / len(inputs)
    expected_weight = np.sum(expected_taps * np.asarray(inputs))
    np.testing.assert_allclose(tap_grads, expected_taps, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        param_grads['scale/weight'], expected_weight, rtol=1e-6, atol=1e-7)
    assert state['scale/weight'] == jnp.float32(0.7)

    def native_program(inputs, taps):
      return nj.grad(module.loss, module, has_aux=True)(inputs, taps)

    native_state, native = nj.pure(native_program)(
        {}, inputs, taps, seed=jnp.asarray([3, 7], jnp.uint32), create=True)
    native_loss, native_params, native_grads, native_residual = native
    np.testing.assert_array_equal(native_loss, loss)
    np.testing.assert_array_equal(native_residual, residual)
    np.testing.assert_array_equal(
        native_params['scale/weight'], params['scale/weight'])
    np.testing.assert_array_equal(
        native_grads['scale/weight'], param_grads['scale/weight'])
    assert native_state.keys() == state.keys()
    for key in state:
      np.testing.assert_array_equal(native_state[key], state[key])

  def test_distinct_objective_seeds_share_stochastic_primal(self):
    module = Scale(name='scale')

    def program(inputs, taps):
      return _grad_with_inputs(
          module.split_loss, [module], input_argnums=(1,), has_aux=True,
          separate_input_loss=True)(inputs, taps)

    inputs = jnp.asarray([0.2, -0.4, 0.8], jnp.float32)
    taps = jnp.asarray([0.1, 0.3, -0.2], jnp.float32)
    state, result = nj.pure(program)(
        {}, inputs, taps, seed=jnp.asarray([5, 11], jnp.uint32), create=True)
    loss, params, param_grads, residuals, input_grads = result
    param_residual, input_residual = residuals
    (tap_grads,) = input_grads

    expected_params = np.sum(
        2 * np.asarray(param_residual) / len(inputs) * np.asarray(inputs))
    expected_taps = 4 * np.asarray(input_residual) / len(inputs)
    np.testing.assert_allclose(
        param_grads['scale/weight'], expected_params, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(tap_grads, expected_taps, rtol=1e-6, atol=1e-7)
    assert np.isfinite(float(loss))
    assert set(params) == {'scale/weight'}
    assert state['scale/weight'] == jnp.float32(0.7)

  def test_optimizer_routes_distinct_objectives_end_to_end(self):
    module = Scale(name='scale')
    optimizer = Optimizer(module, optax.sgd(0.1), name='optimizer')

    def program(inputs, taps):
      return optimizer(
          module.split_loss, inputs, taps, has_aux=True,
          input_grad_argnums=(1,), separate_input_loss=True)

    inputs = jnp.asarray([0.2, -0.4, 0.8], jnp.float32)
    taps = jnp.asarray([0.1, 0.3, -0.2], jnp.float32)
    state, (metrics, residuals, input_grads) = nj.pure(program)(
        {}, inputs, taps, seed=jnp.asarray([13, 17], jnp.uint32), create=True)
    param_residual, input_residual = residuals
    (tap_grads,) = input_grads

    expected_param_grad = np.sum(
        2 * np.asarray(param_residual) / len(inputs) * np.asarray(inputs))
    expected_tap_grad = 4 * np.asarray(input_residual) / len(inputs)
    np.testing.assert_allclose(
        state['scale/weight'], 0.7 - 0.1 * expected_param_grad,
        rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        tap_grads, expected_tap_grad, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(metrics['optimizer/updates'], 1)
    np.testing.assert_allclose(
        metrics['optimizer/loss'], np.mean(np.asarray(param_residual) ** 2),
        rtol=1e-6, atol=1e-7)
