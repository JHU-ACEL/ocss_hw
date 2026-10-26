import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrandom
import numpy as np
from typing import Tuple

from mppi_cem import MppiCemSolver

class MPC():

    def __init__(self, cem_solver: MppiCemSolver, replan_freq=20):
        self.solver = cem_solver
        self.replan_freq = replan_freq
        self.cached_controls = None # controls from a previous solver call to use in between replanning steps
        self.control_idx = 0
        self.delta = 0.0 # tracked steering angle
        self.cem_states = None

    def step_mujoco(self, mj_data, mass, inertia_z, wheelbase, gravity):
    
        # Extract current MuJoCo state into bicycle state vector
        x_pos = float(mj_data.qpos[0])
        y_pos = float(mj_data.qpos[1])
    
        qw, qx, qy, qz = mj_data.qpos[3:7]
        theta = float(jnp.arctan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz)))
    
        vx_w, vy_w = float(mj_data.qvel[0]), float(mj_data.qvel[1])
        v     = float(vx_w * jnp.cos(theta) + vy_w * jnp.sin(theta))
        delta = self.delta   # tracked internally — not observable from freejoint
    
        x_current = np.array([x_pos, y_pos, theta, v, delta])
    
        # Replan or use cached trajectory
        needs_replan = (
            self.cached_controls is None
            or self.control_idx % self.replan_freq == 0
            or self.control_idx >= len(self.cached_controls)
        )
    
        if needs_replan:
            x0 = jnp.array(x_current)
            print("start cem")
            states, controls = self.solver.solve_cem(x0, jrandom.key(1234))
            self.cached_controls = np.asarray(controls)   # shape (N, 2)
            self.control_idx     = 0
            self.cem_states= states
    
        # Get current control
        u_cmd     = self.cached_controls[self.control_idx]
        self.control_idx += 1
    
        px_new, py_new, theta_new, v_new, delta_new = self.cem_states[self.control_idx]
    
        # Write state directly into MuJoCo (kinematic override)
        # Position
        mj_data.qpos[0] = px_new
        mj_data.qpos[1] = py_new
        mj_data.qpos[2] = float(mj_data.qpos[2])   # z unchanged — gravcomp holds it
    
        # Heading -> quaternion (pure yaw, rotation about world Z)
        mj_data.qpos[3] = float(jnp.cos(theta_new / 2.0))   # qw
        mj_data.qpos[4] = 0.0                                 # qx
        mj_data.qpos[5] = 0.0                                 # qy
        mj_data.qpos[6] = float(jnp.sin(theta_new / 2.0))   # qz
    
        # Velocity - project longitudinal speed back to world frame
        mj_data.qvel[0] = float(v_new * jnp.cos(theta_new))   # world vx
        mj_data.qvel[1] = float(v_new * jnp.sin(theta_new))   # world vy
        mj_data.qvel[2] = 0.0                                   # vz
        mj_data.qvel[3] = 0.0                                   # wx
        mj_data.qvel[4] = 0.0                                   # wy
        # Yaw rate from bicycle kinematics: θ̇'= v*tan(delta)/L
        mj_data.qvel[5] = float((v_new / self.solver.L) * jnp.tan(delta_new))
    
        # Update tracked steering angle
        self.delta = float(delta_new)
    
        # Zero out applied forces - MuJoCo is purely a renderer now
        force  = jnp.zeros(3)
        torque = jnp.zeros(3)
    
        return force, torque