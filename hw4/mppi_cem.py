import equinox as eqx
import jax
import jax.numpy as jnp


class MppiCemSolver(eqx.Module):
    x_goal: jax.Array
    control_limits: jax.Array
    Q: jax.Array
    Qf: jax.Array
    R: jax.Array
    obstacle_centers: jax.Array
    obstacle_radii: jax.Array

    dt: float
    T: int
    num_samples: int
    num_iter: int
    kappa: float
    L: float
    v_lim: tuple
    delta_lim: tuple
    safety_margin: float
    w_barrier: float # may or may not use in obstacle_cost
    w_collision: float # may or may not use in obstacle_cost

    def __init__(
        self,
        x_goal,
        dt=0.1,
        T=30,
        num_samples=500,
        num_iter=6,
        kappa=1.0,
        L=2.5,
        v_lim=(0.0, 2.0),
        delta_lim=(-0.4, 0.4),
        acc_lim=(-1.0, 1.0),
        delta_dot_lim=(-1.0, 1.0),
        obstacles=(( (2.0, 6.0), 1.0),),
        safety_margin=0.1,
        Q=None,
        Qf=None,
        R=None,
        w_barrier=None, # these are costs that might be helpful to use in obstacle_cost. Possible to just use one or the other
        w_collision=None,
    ):
        state_dim = 5
        control_dim = 2

        centers = [jnp.asarray(center, dtype=float) for center, _ in obstacles]
        radii = [radius for _, radius in obstacles]

        # Keep empty-obstacle inputs valid and shape-stable.
        obstacle_centers = (
            jnp.stack(centers)
            if centers
            else jnp.zeros((0, 2), dtype=float)
        )
        obstacle_radii = jnp.asarray(radii, dtype=float)

        object.__setattr__(self, "x_goal", jnp.asarray(x_goal, dtype=float))
        object.__setattr__(
            self,
            "control_limits",
            jnp.asarray(
                [
                    [acc_lim[0], acc_lim[1]],
                    [delta_dot_lim[0], delta_dot_lim[1]],
                ],
                dtype=float,
            ),
        )
        object.__setattr__(
            self, "Q", jnp.eye(state_dim) if Q is None else jnp.asarray(Q)
        )
        object.__setattr__(
            self, "Qf", jnp.eye(state_dim) if Qf is None else jnp.asarray(Qf)
        )
        object.__setattr__(
            self, "R", jnp.eye(control_dim) if R is None else jnp.asarray(R)
        )
        object.__setattr__(self, "obstacle_centers", obstacle_centers)
        object.__setattr__(self, "obstacle_radii", obstacle_radii)

        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "num_samples", num_samples)
        object.__setattr__(self, "num_iter", num_iter)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "L", L)
        object.__setattr__(self, "v_lim", v_lim)
        object.__setattr__(self, "delta_lim", delta_lim)
        object.__setattr__(self, "safety_margin", safety_margin)
        object.__setattr__(
            self, "w_barrier", 0.1 if w_barrier is None else w_barrier
        )
        object.__setattr__(
            self, "w_collision", 10.0 if w_collision is None else w_collision
        )

    def dynamics_step(self, state, control):
        """ return x_k+1 given x_k, u_k"""
        raise NotImplementedError("MppiCemSolver.dynamics_step has yet to be implemented")

    @eqx.filter_jit
    def trajectory_rollout(self, x0, controls):
        """Roll out controls with shape (T, 2); return states with shape (T+1, 5)."""
        raise NotImplementedError("MppiCemSolver.trajectory_rollout has yet to be implemented")

    def obstacle_cost(self, state):
        raise NotImplementedError("MppiCemSolver.obstacle_cost has yet to be implemented")

    @eqx.filter_jit
    def cost(self, states, controls):
        """Cost for states shaped (T+1, 5) and controls shaped (T, 2)."""
        raise NotImplementedError("MppiCemSolver.cost has yet to be implemented")

    def solve_mppi(self, x0, key):
        """MPPI; returns (states, controls). Pass in a JAX key for randomization."""
        raise NotImplementedError("MppiCemSolver.solve_mppi has yet to be implemented")

    def solve_cem(self, x0, key):
        """CEM; returns (states, controls). Pass in a JAX key for randomization."""
        raise NotImplementedError("MppiCemSolver.solve_cem has yet to be implemented")