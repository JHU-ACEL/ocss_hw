import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from typing import Tuple

class PDIPSolver():
    """
    Follows the implementation of the primal-dual interior point algorithm as found in
        "CVXGEN: a code generator for embedded convex optimization", J. Mattingley and S. Boyd (2012).
    """
    def __init__(self, max_qp_iter: int = 50, tol: float = 1e-2):
        """
        Attributes:
            max_qp_iter (int): maximum number of QP iterations
            tol (float): positive value for measuring QP residual convergence

            nx (int): number of primal variables
            n_eq (int): number of equality constraints
            n_ineq (int): number of inequality constraints

            x0 (jnp.array): primal solution x
            s0 (jnp.array): slack variable s
            z0 (jnp.array): dual variables for inequality constraints z
            y0 (jnp.array): dual variables for equality constraints y
        """
        self.max_qp_iter = max_qp_iter
        self.tol = tol

        self.nx = None
        self.n_eq = None
        self.n_ineq = None

        self.x0 = None
        self.s0 = None
        self.z0 = None
        self.y0 = None

    def init_problem(self, P: jax.Array, p: jax.Array, A: jax.Array, b: jax.Array, G: jax.Array, h: jax.Array) -> None:

        raise NotImplementedError("PDIPSolver.init_problem has yet to be implemented")

    def init_soln(self):
        """
        Follows initialization procedure from Section 5.2 in CVXGEN paper

        Returns:
            x0 (jnp.array): init. soln. for primary variable x
            s0 (jnp.array): init. soln. for slack variable s
            z0 (jnp.array): init. soln. for dual variables associated with inequality constraints
            y0 (jnp.array): init. soln. for dual variables associated with equality constraints
        """
        raise NotImplementedError("PDIPSolver.init_soln has yet to be implemented")

    def compute_residuals(self, xbar: jax.Array, sbar: jax.Array, zbar: jax.Array, ybar: jax.Array):
        """
        Implements the residuals associated with the right hand side of the KKT system.

        Parameters:
            xbar (jnp.array): Current primal variable x
            sbar (jnp.array): Current slack variable s
            zbar: (jnp.array): Current dual variable z for inequality constraints
            sbar: (jnp.array): Current dual variable s for equality constraints

        Returns:
            r1 (jnp.array): Residual for KKT derivative
            r2 (jnp.array): Residual for complementary slackness
            r3 (jnp.array): Residual for inequality constraint
            r4 (jnp.array): Residual for equality constraint
        """
        raise NotImplementedError("PDIPSolver.compute_residuals has yet to be implemented")

    def compute_centering_plus_corrector(self, s0: jax.Array, ds: jax.Array, z0: jax.Array, dz: jax.Array) -> None:
        """
	Computes the sigma and mu terms used for the centering-plus-corrector step from Section 5.2 in CVXGEN paper.
	Hint: you should be able to reuse self.compute_line_search here.

        Parameters:
            s0 (jnp.array): current iterate for slack variable s.
            ds (jnp.array): descent direction for slack variable s.
            z0 (jnp.array): current iterate for dual variable z.
            dz (jnp.array): descent direction for dual variable s.

        Returns:
            mu (float): mu term for centering-corrector step
            sigma (float): sigma term for centering-corrector step
        """

        raise NotImplementedError("PDIPSolver.compute_centering_plus_corrector has yet to be implemented")

    def compute_line_search(self, s0: jax.Array, ds: jax.Array, z0: jax.Array, dz: jax.Array, n_steps: int = 500, tol: float = 1e-4):
        """
        Parameters:
            s0 (jnp.array): current iterate for slack variable s.
            ds (jnp.array): descent direction for slack variable s.
            z0 (jnp.array): current iterate for dual variable z.
            dz (jnp.array): descent direction for dual variable s.
            n_steps (int): max number of iterations for line search.
            tol (float): positive number tolerance for passing line search (i.e., s+ds >= tol)

        Returns:
            alpha (float): step size value
            line_search_successful (bool): indicates if line search found valid solution
        """

        raise NotImplementedError("PDIPSolver.compute_line_search has yet to be implemented")

    def has_converged(self) -> bool:
        """
        Given current solution iterate (x0, s0, z0, y0), computes whether the
        current QP residuals have been sufficiently minimized.

        Returns:
            converged (bool): true if QP solver has minimized residuals.
        """

        raise NotImplementedError("PDIPSolver.has_converged has yet to be implemented")

    def factorize_kkt_system(self, sbar: jax.Array, zbar: jax.Array):

        raise NotImplementedError("PDIPSolver.factorize_kkt_system has yet to be implemented")  

    def solve_kkt_system(self, r1: jax.Array, r2: jax.Array, r3: jax.Array, r4: jax.Array, sbar: jax.Array, zbar: jax.Array):
        """
        Solves the KKT system at the current iteration using the residual values.

        Parameters:
            r1 (jnp.array): Residual for KKT derivative
            r2 (jnp.array): Residual for complementary slackness
            r3 (jnp.array): Residual for inequality constraint
            r4 (jnp.array): Residual for equality constraint
            sbar (jnp.array)
            zbar (jnp.array)

        Returns:
            dx (jnp.array): descent direction for primal x
            ds (jnp.array): descent direction for slack variable s
            dz (jnp.array): descent direction for dual variable z
            dy (jnp.array): descent direction for dual variable y
        """
        raise NotImplementedError("PDIPSolver.solve_kkt_system has yet to be implemented")

    def solve_qp(self, verbose: bool = False):
        """
        Solves the QP by:
        1. Initialize the primal, slack, and dual variables using self.init_soln
        2. Carry out max_qp_iter number of iterations of solving the KKT system
            2a. At each iteration, use self.compute_residuals and self.solve_kkt_system
            2b. Use self.compute_centering_plus_corrector for correcting dz and ds corrections
            2c. Use self.compute_line_search to compute the step size
            2d. Check for convergence

        Returns
            costs (list): a list of the objective function across iterations
        """

        raise NotImplementedError("PDIPSolver.solve_qp has yet to be implemented")
