import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt




from solvers import PDIPSolver
from ocp import OCPSolver
from mpc import MPC

from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.resolve()

MODEL_PATH = "models/mhs/mhs_scene.xml"  
MODEL_PATH = os.path.join(ROOT, MODEL_PATH)

NX, NU = 4, 2  
N_HORIZON = 20
DT = 0.1


Q_POS, Q_VEL, R_ACCEL = 10.0, 4.0, 0.1


X_MIN = jnp.array([-10.0, -10.0, -5.0, -5.0])
X_MAX = jnp.array([10.0, 10.0, 5.0, 5.0])

U_MIN = jnp.array([-15.0, -15])
U_MAX = jnp.array([15.0, 15])


X_GOAL = jnp.array([0.0, 0.0, 0.0, 0.0])  # hover at origin

POS_TOL = 0.5  # m
VEL_TOL = 0.5  # m/s


SPAWN_ALTITUDE = 5.0  
INITIAL_STATE = jnp.array([9.4, 7.2, -0.5, 1.0])


VIEWER_SPAN = 20.0  
DRAW_ALTITUDE = 8.0  


SAFETY_MAX_SIM_TIME = 40.0




def build_double_integrator(dt: float):
    I2 = jnp.eye(2)
    Ak = jnp.block([[I2, dt * I2], [jnp.zeros((2, 2)), I2]])
    Bk = jnp.block([[0.5 * dt**2 * I2], [dt * I2]])
    return Ak, Bk





Q_mat = jnp.diag(jnp.array([Q_POS] * 2 + [Q_VEL] * 2))
R_mat = R_ACCEL * jnp.eye(NU)


def setup_camera(viewer_ctx):
    """Position camera to view the JHU path from above."""
    # xy = np.array([[float(p[0]), float(p[1])] for p in WAYPOINTS])
    center = np.array([0.0, 0.0])  # center of JHU path
    span = VIEWER_SPAN
    viewer_ctx.cam.lookat[:] = np.array([center[0], center[1], DRAW_ALTITUDE])
    viewer_ctx.cam.distance = span * 1.4 + 5.0
    viewer_ctx.cam.azimuth = 90
    viewer_ctx.cam.elevation = -89


def add_marker(viewer_ctx, pos, rgba=(1, 0, 0, 0.6), size=0.3):
    """Add a persistent visual-only sphere marker to the passive viewer."""
    scn = viewer_ctx.user_scn
    if scn.ngeom >= scn.maxgeom:
        return  
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, 0, 0]),
        pos=np.array(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scn.ngeom += 1



def run(use_viewer: bool):
    mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    mj_data = mujoco.MjData(mj_model)

    # Get helicopter body
    body = mj_model.body("Body_MHS_MainBody_v16")
    mass = body.mass[0]
    g = mj_model.opt.gravity[2]  


    Ak, Bk = build_double_integrator(DT)
    ocp_solver = OCPSolver()
    solver = PDIPSolver()
    mpc = MPC(ocp_solver, solver)
    mpc.init_mpc(Q_mat, R_mat, N_HORIZON, Ak, Bk, X_MIN, X_MAX, U_MIN, U_MAX)

    



    mj_data.qpos[:2] = INITIAL_STATE[:2]
    mj_data.qpos[2] = SPAWN_ALTITUDE
    mj_data.qpos[3:7] = np.array([0.0, 1.0, 0.0, 0.0])  
    mj_data.qvel[:2] = INITIAL_STATE[2:4]
    mj_data.qvel[2:] = np.zeros(4)
    mujoco.mj_forward(mj_model, mj_data)

    physics_steps_per_control = max(1, round(DT / mj_model.opt.timestep))

    times, positions, velocities = [], [], []

    viewer_ctx = None
    if use_viewer:
        viewer_ctx = mujoco.viewer.launch_passive(mj_model, mj_data)
        setup_camera(viewer_ctx)
        add_marker(viewer_ctx, [X_GOAL[0], X_GOAL[1], SPAWN_ALTITUDE], rgba=(0, 1, 0, 0.6), size=0.3)



    x_goal = X_GOAL
    arrived = False
    t_sim = 0.0

    try:
        while True:
            if viewer_ctx is not None and not viewer_ctx.is_running():
                break
            if arrived:
                break
            if viewer_ctx is None and t_sim >= SAFETY_MAX_SIM_TIME:
                print("[WARN] Hit safety time limit, stopping.")
                break

            step_start = time.time()

            # Find force for step using mpc
            force = mpc.step_mujoco(mj_data, mass, g, x_goal)

            mj_data.xfrc_applied[body.id, :3] = force
            mj_data.xfrc_applied[body.id, 3:] = 0.0  # no torque

            # Step physics
            for _ in range(physics_steps_per_control):
                mujoco.mj_step(mj_model, mj_data)

            # Sync viewer
            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    break
                viewer_ctx.sync()
                sleep_time = DT - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            t_sim += DT
            times.append(t_sim)
            positions.append(mj_data.qpos[:2].copy())
            velocities.append(mj_data.qvel[:2].copy())

            # Check waypoint arrival
            pos_err = float(np.linalg.norm(np.asarray(mj_data.qpos[:2]) - x_goal[:2]))
            speed = float(np.linalg.norm(mj_data.qvel[:2] - x_goal[2:4]))
            if pos_err < POS_TOL and speed < VEL_TOL:
                # Have arrived at goal state
                arrived = True
                print(f"Reached goal state at t = {t_sim:.2f}s")

    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()

    positions = np.array(positions)
    velocities = np.array(velocities)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helicopter MPC waypoint tracking demo")
    parser.add_argument("--viewer", action="store_true", help="Render live in the MuJoCo passive viewer")
    args = parser.parse_args()
    run(use_viewer=args.viewer)
