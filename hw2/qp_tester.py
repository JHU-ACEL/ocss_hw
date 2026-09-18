import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpy as np
import cvxpy as cp

from typing import Callable, Iterable, Tuple

from solvers import PDIPSolver

class PDIPTester():
    def __init__(self, P: jax.Array, p: jax.Array, A: jax.Array, b: jax.Array, G: jax.Array, h: jax.Array, cost_tol: float = 0.05, primal_tol: float = 0.05):
        nx = P.shape[1]
        n_eq = A.shape[0]
        n_ineq = G.shape[0]
        assert nx == p.size
        assert n_eq == b.size
        assert n_ineq == h.size

        self.P = P
        self.p = p
        self.A = A
        self.b = b
        self.G = G
        self.h = h

        self.cost_tol = cost_tol
        self.primal_tol = primal_tol

        # Initialize JAX solver
        self.solver = PDIPSolver()
        self.solver.init_problem(self.P, self.p, self.A, self.b, self.G, self.h)

    def compare_solutions(self) -> bool:
        costs = self.solver.solve_qp(verbose=False)
        nx = self.P.shape[1]

        # Create equivalent QP inside CVXPY
        Z = cp.Variable(shape=(nx))
        cvx_cost = cp.quad_form(Z[:], 0.5*self.P) + Z[:].T @ self.p
        constraints = []
        constraints += [self.G @ Z <= self.h]
        constraints += [self.A @ Z == self.b]
        prob = cp.Problem(cp.Minimize(cvx_cost), constraints)
        prob.solve()

        if prob.status in ["infeasible", "unbounded"]:
          print('[WARN] This problem is infeasible!')
          return False
        elif jnp.abs(prob.value - costs[-1]) < self.cost_tol and \
            jnp.linalg.norm(Z.value - self.solver.x0) < self.primal_tol:
          print('QP solutions match!')
          return True
        else:
          return False


