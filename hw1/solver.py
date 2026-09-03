from __future__ import annotations
from typing import Callable, Iterable, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def function_to_minimize(x: jnp.array) -> float:
    raise NotImplementedError("Function has not been implemented yet")

class Solver():
  def __init__(self):
    """
    Attributes:
        tol (float): tolerance check for norm of gradient
        max_iter (int): maximum number of iterations

    """
    self.tol = 1e-2
    self.max_iter = 100

  def compute_step_size_ls(self, x0: jax.Array, grad_at_x0: jax.Array, f_eval: Callable) -> float:
    """
    Inputs:
      x0 (jax.Array): Initial point from which descent direction is being determined
      grad_at_x0 (jax.Array): Gradient at the initial point
      f_eval: function handle for evaluating objective

    Outputs:
      alpha (float): Step size to take in the direction of the gradient
    """
    raise NotImplementedError("Solver.compute_step_size has yet to be implemented")

  def solve_with_gradient_descent(self, x0: jax.Array, f_eval: Callable) -> Tuple[jax.Array, jax.Array]:
    """
    Inputs:
      x0 (jax.Array): Initial point for solver
      f_eval: function handle for evaluating objective

    Outputs:
      x_soln (jax.Array): Optimal solution for this problem
      x_traj (jax.Array): The "trajectory" of the optimizer passed out as an array of size
            (iteration count, x0 dimension)
    """
    raise NotImplementedError("Solver.solve_with_gradient_descent has yet to be implemented")

  def solve_with_newton_method(self, x0: jax.Array, f_eval: Callable) -> Tuple[jax.Array, jax.Array]:
    """
    Inputs:
      x0: Initial point for solver
      f_eval: function handle for evaluating objective

    Outputs:
      x_soln (jax.Array): Optimal solution for this problem
      x_traj (jax.Array): The "trajectory" of the optimizer passed out as an array of size
            (iteration count, x0 dimension)
    """
    raise NotImplementedError("Solver.solve_with_newton_method has yet to be implemented")