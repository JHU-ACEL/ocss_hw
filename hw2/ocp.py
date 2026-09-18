import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

class OCPSolver():
    def compute_cost_term(self, Q: jax.Array, R: jax.Array, N: int) -> jax.Array:
        """
        Parameters:
            Q (jax.Array): (nx, nx) matrix for a single time step (i.e., x_k.T @ Q @ x_k)
            R (jax.Array): (nu, nu) matrix for a single time step (i.e., u_k.T @ R @ u_k)
            N (int): number of time steps for optimal control problem.

        Returns:
            P (jax.Array): large matrix for evaluating quadratic (i.e. x @ P @ x)
        """
        raise NotImplementedError("OCPSolver.compute_cost_term has yet to be implemented")

    def compute_equality_matrix(self, Ak: jax.Array, Bk: jax.Array, N: int) -> jax.Array:
        """
        Computes Abar for the consolidated linear equality constraint Abar @ x = bvec.

        Parameters:
            Ak (jax.Array): discretized double integrator Ak matrix.
            Bk (jax.Array): discretized double integrator Bk matrix.
            N (int): number of time steps for optimal control problem.

        Returns:
            Amat (jax.Array): consolidated equality constraint matrix.
        """
        raise NotImplementedError("OCPSolver.compute_equality_matrix has yet to be implemented")

    def compute_equality_bc(self, x0: jax.Array, Ak: jax.Array, N:int) -> jax.Array:
        """
        Computes bvec for the consolidated linear equality constraint Abar @ x = bvec.

        Parameters:
            x0 (jax.Array): initial state for problem.
            Ak (jax.Array): discretized double integrator Ak matrix.
            N (int): number of time steps for optimal control problem.

        Returns:
            Amat (jax.Array): consolidated equality constraint matrix.
        """
        raise NotImplementedError("OCPSolver.compute_equality_bc has yet to be implemented")

    
    def compute_ineq_con(self, x_min: jax.Array, x_max: jax.Array, u_min: jax.Array, u_max: jax.Array, N: int) -> tuple[jax.Array, jax.Array]:
        """
        Computes Gmat and hvec for the consolidated linear inequality constraint Gmat @ x <= hvec.

        Parameters:
            x_min (jax.Array): lower bound for state vector.
            x_max (jax.Array): upper bound for state vector.
            u_min (jax.Array): lower bound for control vector.
            u_max (jax.Array): upper bound for control vector.
            N (int): number of time steps for optimal control problem.

        Returns:
            Gmat (jax.Array): consolidated inequality constraint matrix.
            hvec (jax.Array): consolidated inequality constraint vector.
        """
        raise NotImplementedError("OCPSolver.compute_ineq_con has yet to be implemented")
