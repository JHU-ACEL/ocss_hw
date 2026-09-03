from __future__ import annotations
from typing import Callable, Iterable, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from solver import Solver

class GaussNewtonSolver(Solver):
  def __init__(self, lm_damping: float = 1e-3):
    """
    Attributes:
        lm_damping (float): Levenberg-Marquardt damping term. 
    """
    super().__init__()
    self.lm_damping = lm_damping

  def solve_with_gauss_newton_method(self, x0: jax.Array, b: jax.Array, f_eval: Callable) -> Tuple[jax.Array, jax.Array]:
    """
    Inputs:
      x0: Initial point for solver
      b: Target output value
      f_eval: Function handle for evaluating objective.

    Outputs:
      x_soln (jax.Array): Optimal solution for this problem
    """
    raise NotImplementedError("GaussNewtonSolver.solve_with_gauss_newton_method has yet to be implemented")