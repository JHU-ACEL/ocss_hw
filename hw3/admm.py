import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from typing import Tuple

class ADMMSolver():
    def __init__(self, max_iter: int = 500, tol: float = 1e-2, newton_iter: int = 10, newton_tol: float = 1e-4, rho: float = 0.5):
        """
        Attributes:
            max_iter (int): maximum number of QP iterations
            tol (float): positive value for measuring QP residual convergence
            newton_iter (int): maximum number of Newton iterations to solve equality constrained system.
            newton_tol (float): positive value for measuring convergence of equality constrained system.

            nx (int): number of primal variables
            n_eq (int): number of equality constraints
            n_ineq (int): number of inequality constraints

            xk (jnp.array): primal solution x
            zk (jnp.array): splitting variable
            lk (jnp.array): Lagrange variable
        """
        self.max_iter = max_iter
        self.tol = tol
        self.newton_iter = newton_iter
        self.newton_tol = newton_tol

        self.nx = None
        self.n_eq = None
        self.n_ineq = None

        self.rho = rho

    def init_problem(self, P: jax.Array, p: jax.Array, A: jax.Array, b: jax.Array, G: jax.Array, h: jax.Array) -> None:
        self._P = 0.5*(P + P.T)
        self._p = p
        self.nx = p.size

        self._A = A
        self._b = b
        self.n_eq = b.size

        self._G = G
        self._h = h
        self.n_ineq = h.size

    def solve_linear_eq(self, xk: jax.Array, uk: jax.Array, zk: jax.Array):
        """
        Solves the equality constrained subproblem to update x^{k+1}.

        Parameters:
            xk (jnp.array): Current primal variable
            uk (jnp.array): Current Lagrange variable
            zk (jnp.array): Current splitting variable

        Returns:
            xk (jnp.array): Updated solution to equality constrained problem
        """
        raise NotImplementedError("ADMMSolver.solve_linear_eq has yet to be implemented")

    def solve_qp(self, verbose: bool = False):
        """
        Solves the quadratic program by computing the
            primal, Lagrange variable, and splitting variable updates.
        """
        raise NotImplementedError("ADMMSolver.solve_qp has yet to be implemented")
