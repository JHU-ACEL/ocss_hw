# gradients.py

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt

class GradientsEval:
    def __init__(self) -> None:
        return

    def function_eval(self, x0: jax.Array) -> jax.Array:
      """
      Input:
        x0 (jax.Array): 2D vector consisting of [x, y]

      Output:
        function eval (float): returns the function evaluated at x0.
      """
      raise NotImplementedError("GradientsEval.function_eval has yet to be implemented")

    def analytical_grad(self, x0: jax.Array) -> jax.Array:
      """
      Input:
        x0 (jax.Array): 2D vector consisting of [x, y]

      Output:
        grad (jax.Array): 2D gradient vector at [x, y] evaulated analytically
      """
      raise NotImplementedError("GradientsEval.analytical_grad has yet to be implemented")

    def numerical_grad(self, x0: jax.Array, epsilon: float) -> jax.Array:
      """
      Input:
        x0 (jax.Array]: 2D vector consisting of [x, y]
        epsilon (float): "step" used to numerically approximate derivative

      Output:
        grad (jax.Array): 2D gradient vector at [x, y] evaluated using numerical approximation
      """
      raise NotImplementedError("GradientsEval.numerical_grad has yet to be implemented")

    def jax_grad(self, x0: jax.Array) -> jax.Array:
      """
      Input:
        x0 (jax.Array): 2D vector consisting of [x, y]

      Output:
        grad (jax.Array): 2D gradient vector at [x, y] evaluated using numerical approximation
      """
      raise NotImplementedError("GradientsEval.jax_grad has yet to be implemented")
