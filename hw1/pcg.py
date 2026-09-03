import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

class PCG():
  def __init__(self, max_iter:int = 10000, tol:float = 1e-8):
    """
    Preconditioned conjugate gradient (PCG) solver for linear
      systems of the form $Ax=b$, where $A$ is positive definite.
      Implementation based on guide from "Templates for the Solution of
      Linear Systems: Building Blocks for Iterative Methods" by R. Barrett
      et al.

    Attributes:
      max_iter (int): maximum number of PCG iterations
      tol (float): positive value for measuring residual convergence
    """
    self.max_iter = max_iter
    self.tol = tol
    return

  def solve(self, A: jax.Array, b: jax.Array):
    """
    Solves linear system A x = b and returns x.

    Returns:
      A (jnp.array): PD matrix for linear system.
      b (jnp.array): target for linear system.

    Returns:
      xk (jnp.array): solution for linear system.
    """
    raise NotImplementedError("PCG.solve has yet to be implemented")
