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

from mpc import MPC
from mppi_cem import MppiCemSolver

from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.resolve()

MODEL_PATH = "models/rover/rover_viewer.xml"  
MODEL_PATH = os.path.join(ROOT, MODEL_PATH)

DT = 0.2
N_HORIZON = 70
REPLAN_FREQ = 70

X_GOAL = jnp.array([6., 14., 0.0, 0., 0.])  # get to target w/ zero velocity, 0 heading, and 0 steering angle
OBSTACLES = [(np.array([2.0, 6.0]), 1),]

POS_TOL = 1.0  # m
VEL_TOL = 0.5  # m/s
SAFETY_MAX_TIME = 120.0

SPAWN_ALTITUDE = 0.51          # rover CoM height above ground [m]
WHEELBASE    = 2.5           # bicycle model wheelbase L [m]
INITIAL_STATE = jnp.array([0., 0., 0., 2., 0.])


VIEWER_SPAN = 20.0  
DRAW_ALTITUDE = 8.0  

# Initialize MPC
solver = MppiCemSolver(
    x_goal=X_GOAL, dt=DT, T=N_HORIZON,
    num_samples=400, num_iter=15, kappa=2.0, obstacles=OBSTACLES
)

mpc = MPC(solver, REPLAN_FREQ)



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

def download_mars_texture(save_dir: str) -> str | None:
    """
    Downloads a free Mars-like terrain texture from NASA's publicly
    available image library. Falls back to None if offline.
    Returns the saved file path or None.
    """
    import urllib.request
    import os

    # NASA Mars surface image (public domain)
    url      = "https://mars.nasa.gov/msl-raw-images/proj/msl/redops/ods/surface/sol/01000/opgs/edr/fcam/FLB_486265257EDR_F0481570FHAZ00323M_.JPG"
    out_path = os.path.join(save_dir, "mars_surface.png")

    if os.path.exists(out_path):
        return out_path
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"[INFO] Mars texture saved to {out_path}")
        return out_path
    except Exception as e:
        print(f"[WARN] Could not download mars texture: {e}")
        return None


def add_obstacles_to_xml(xml_path: str, obstacles: list) -> str:
    import xml.etree.ElementTree as ET
    import os

    tree    = ET.parse(xml_path)
    root    = tree.getroot()
    xml_dir = os.path.dirname(os.path.abspath(xml_path))

    # ── Mesh path ─────────────────────────────────────────────────────────────
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", xml_dir)

    # ── Visual ────────────────────────────────────────────────────────────────
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")

    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient",  "0.3 0.25 0.2")
    headlight.set("diffuse",  "0.6 0.5 0.4")
    headlight.set("specular", "0.1 0.1 0.1")

    # ── Floor ─────────────────────────────────────────────────────────────────
    worldbody = root.find("worldbody")
    floor = worldbody.find(".//geom[@name='floor']")
    if floor is not None:
        floor.set("material", "mars_ground")
        floor.set("size",     "100 100 0.1")

    # ── Assets ────────────────────────────────────────────────────────────────
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    grid_tex = next((e for e in asset.findall("texture") if e.get("name") == "grid"), None)
    grid_mat = next((e for e in asset.findall("material") if e.get("name") == "grid"), None)

    if grid_tex is not None:
        grid_tex.attrib.clear()
        grid_tex.set("name", "mars_ground")
        grid_tex.set("type", "2d")
        grid_tex.set("file", "/workspace/models/rover/gravelly_sand_disp_4k.png")
    else:
        ET.SubElement(asset, "texture", {
            "name" : "mars_ground",
            "type" : "2d",
            "file" : "/workspace/models/rover/gravelly_sand_disp_4k.png",
        })

    if grid_mat is not None:
        grid_mat.attrib.clear()
        grid_mat.set("name",        "mars_ground")
        grid_mat.set("texture",     "mars_ground")
        grid_mat.set("texrepeat",   "2 2")
        grid_mat.set("texuniform",  "true")
        grid_mat.set("reflectance", "0.05")
    else:
        ET.SubElement(asset, "material", {
            "name"        : "mars_ground",
            "texture"     : "mars_ground",
            "texrepeat"   : "0.2 0.2",
            "texuniform"  : "true",
            "reflectance" : "0.05",
        })

    ET.SubElement(asset, "texture", {
        "name"    : "mars_sky",
        "type"    : "skybox",
        "builtin" : "gradient",
        "rgb1"    : "0.62 0.42 0.28",
        "rgb2"    : "0.38 0.22 0.12",
        "width"   : "512",
        "height"  : "512",
    })

    # Rock material — use same ground texture for realistic surface
    ET.SubElement(asset, "material", {
        "name"        : "rock",
        "texture"     : "mars_ground",   # reuse the same loaded texture
        "texrepeat"   : "0.5 0.5",       # larger texels → coarser grain on rock
        "texuniform"  : "true",
        "reflectance" : "0.02",
        "shininess"   : "0.0",
    })

    # ── Lights ────────────────────────────────────────────────────────────────
    for light in worldbody.findall("light"):
        worldbody.remove(light)

    ET.SubElement(worldbody, "light", {
        "name"        : "sun",
        "directional" : "true",
        "pos"         : "20 -20 30",
        "dir"         : "-0.4 0.4 -1",
        "diffuse"     : "0.8 0.65 0.5",
        "specular"    : "0.2 0.15 0.1",
        "castshadow"  : "true",
    })

    ET.SubElement(worldbody, "light", {
        "name"        : "fill",
        "directional" : "true",
        "pos"         : "-20 20 10",
        "dir"         : "0.4 -0.4 -1",
        "diffuse"     : "0.2 0.18 0.15",
        "specular"    : "0.0 0.0 0.0",
        "castshadow"  : "false",
    })


    # ── Obstacles as rocks ────────────────────────────────────────────────────
    for i, (center, radius) in enumerate(obstacles):
        cx, cy = float(center[0]), float(center[1])
        r      = float(radius)

        # Invisible cylinder — bounding box for collision/planning purposes
        # contype/conaffinity=0 since rover is kinematic, but kept for reference
        bounding = ET.SubElement(worldbody, "geom")
        bounding.set("name",        f"obstacle_{i}_bbox")
        bounding.set("type",        "cylinder")
        bounding.set("pos",         f"{cx} {cy} {r}")
        bounding.set("size",        f"{r} {r}")
        bounding.set("rgba",        "0 0 0 0")      # fully invisible
        bounding.set("contype",     "0")
        bounding.set("conaffinity", "0")

        # Rock visual: several overlapping ellipsoids with offsets and tilts
        # to break symmetry and produce a craggy boulder appearance.
        #
        # Each entry: (dx, dy, dz, rx, ry, rz, sx, sy, sz)
        #   dx,dy,dz  — offset from rock centre as fraction of radius
        #   rx,ry,rz  — euler angles [deg] for tilt
        #   sx,sy,sz  — ellipsoid radii as fraction of r
        #
        import random
        rng = random.Random(i)   # seed per obstacle so rocks are deterministic
                                 # but different from each other

        def rp(lo, hi): return rng.uniform(lo, hi)   # shorthand

        # Generate many small overlapping ellipsoids at random positions/tilts
        # within the cylinder bounding box to produce a craggy irregular boulder
        rock_pieces = []

        # ── Core: one large central mass ──────────────────────────────────────
        rock_pieces.append((
            rp(-0.05, 0.05),   # dx — nearly centred
            rp(-0.05, 0.05),   # dy
            rp(0.45,  0.60),   # dz — sits above ground
            rp(-5,    5),      # rx tilt
            rp(-8,    8),      # ry tilt
            rp(-15,  15),      # rz tilt
            rp(0.85, 0.95),    # sx — wide
            rp(0.80, 0.92),    # sy
            rp(0.50, 0.65),    # sz — flattish
        ))

        # ── Mid layer: 4-6 medium lumps around the sides ──────────────────────
        n_mid = rng.randint(4, 6)
        for k in range(n_mid):
            angle  = (k / n_mid) * 2 * 3.14159 + rp(-0.3, 0.3)
            dist   = rp(0.15, 0.40)
            rock_pieces.append((
                dist * rp(0.8, 1.0) * (1 if rp(0,1)>0.5 else -1) * abs(rp(0.1, 0.5)),
                dist * rp(0.8, 1.0) * (1 if rp(0,1)>0.5 else -1) * abs(rp(0.1, 0.5)),
                rp(0.25, 0.70),        # dz
                rp(-20,  20),          # rx
                rp(-20,  20),          # ry
                rp(-30,  30),          # rz
                rp(0.35, 0.60),        # sx
                rp(0.30, 0.55),        # sy
                rp(0.22, 0.42),        # sz
            ))

        # ── Top layer: 3-5 small craggy protrusions on top ───────────────────
        n_top = rng.randint(3, 5)
        for k in range(n_top):
            rock_pieces.append((
                rp(-0.35, 0.35),       # dx
                rp(-0.35, 0.35),       # dy
                rp(0.70,  0.95),       # dz — high up
                rp(-25,   25),         # rx — strongly tilted
                rp(-25,   25),         # ry
                rp(-45,   45),         # rz
                rp(0.18,  0.40),       # sx — small
                rp(0.15,  0.38),       # sy
                rp(0.12,  0.28),       # sz
            ))

        # ── Base: 3-4 flat pieces flush with ground to hide the cylinder base ─
        n_base = rng.randint(3, 4)
        for k in range(n_base):
            rock_pieces.append((
                rp(-0.30, 0.30),       # dx
                rp(-0.30, 0.30),       # dy
                rp(0.08,  0.18),       # dz — very low
                rp(-10,   10),
                rp(-10,   10),
                rp(-45,   45),
                rp(0.55,  0.80),       # sx — wide flat slab
                rp(0.50,  0.75),       # sy
                rp(0.08,  0.15),       # sz — very thin
            ))

        for j, (dx, dy, dz, rx, ry, rz, sx, sy, sz) in enumerate(rock_pieces):
            geom = ET.SubElement(worldbody, "geom")
            geom.set("name",        f"obstacle_{i}_rock_{j}")
            geom.set("type",        "ellipsoid")
            geom.set("pos",         f"{cx + dx*r:.4f} {cy + dy*r:.4f} {dz*r:.4f}")
            geom.set("size",        f"{sx*r:.4f} {sy*r:.4f} {sz*r:.4f}")
            geom.set("euler",       f"{rx:.2f} {ry:.2f} {rz:.2f}")
            geom.set("material",    "rock")
            geom.set("contype",     "0")
            geom.set("conaffinity", "0")

    return ET.tostring(root, encoding="unicode")


def run(use_viewer: bool):
    xml_string = add_obstacles_to_xml(MODEL_PATH, OBSTACLES)
    mj_model   = mujoco.MjModel.from_xml_string(xml_string)
    #mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    mj_data = mujoco.MjData(mj_model)

    # Get rover body
    body = mj_model.body("rover")
    body_id = body.id
    mass = float(mj_model.body_mass[body_id])
    g = mj_model.opt.gravity[2]  
    inertia_z = float(mj_model.body_inertia[body_id][2])

    # Initial conditions
    init_x, init_y, init_theta, init_v, init_delta = INITIAL_STATE

    mj_data.qpos[0] = float(init_x)
    mj_data.qpos[1] = float(init_y)
    mj_data.qpos[2] = SPAWN_ALTITUDE

    # Convert initial heading to quaternion [qw, qx, qy, qz]
    # Pure yaw rotation about world Z
    mj_data.qpos[3] = float(jnp.cos(init_theta / 2.0))   # qw
    mj_data.qpos[4] = 0.0                                  # qx
    mj_data.qpos[5] = 0.0                                  # qy
    mj_data.qpos[6] = float(jnp.sin(init_theta / 2.0))   # qz

    # Initial velocity: project longitudinal speed onto world frame
    mj_data.qvel[0] = float(init_v * jnp.cos(init_theta))  # world vx
    mj_data.qvel[1] = float(init_v * jnp.sin(init_theta))  # world vy
    mj_data.qvel[2] = 0.0                                   # vz
    mj_data.qvel[3] = 0.0                                   # wx
    mj_data.qvel[4] = 0.0                                   # wy
    mj_data.qvel[5] = 0.0                                   # wz (yaw rate)

    # Initialise solver's tracked steering angle to match initial state
    mpc.delta = float(init_delta)

  
    mujoco.mj_forward(mj_model, mj_data)

    physics_steps_per_control = max(1, round(DT / mj_model.opt.timestep))

    times, positions, velocities = [], [], []

    viewer_ctx = None
    if use_viewer:
        viewer_ctx = mujoco.viewer.launch_passive(mj_model, mj_data)
        setup_camera(viewer_ctx)
        add_marker(viewer_ctx, [X_GOAL[0], X_GOAL[1], SPAWN_ALTITUDE], rgba=(0, 1, 0, 0.6), size=0.3)
        add_marker(viewer_ctx, [INITIAL_STATE[0], INITIAL_STATE[1], SPAWN_ALTITUDE], rgba=(1, 0, 0, 0.6), size=0.3)

    arrived = False
    t_sim = 0.0

    try:
        while True:
            if viewer_ctx is not None and not viewer_ctx.is_running():
                break
            if arrived:
                time.sleep(20)
                break
            if viewer_ctx is None and t_sim >= SAFETY_MAX_SIM_TIME:
                print("[WARN] Hit safety time limit, stopping.")
                break

            step_start = time.time()

            # Find force for step using mpc
            force, torque = mpc.step_mujoco(mj_data, mass, inertia_z, WHEELBASE, g)

            # mj_data.xfrc_applied[body.id, :3] = force
            # mj_data.xfrc_applied[body.id, 3:] = torque

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
            pos_err = float(np.linalg.norm(np.asarray(mj_data.qpos[:2]) - X_GOAL[:2]))
            speed = float(np.linalg.norm(mj_data.qvel[:2] - X_GOAL[2:4]))
            if pos_err < POS_TOL and speed < VEL_TOL:
                # Have arrived at goal state
                arrived = True
                print(f"Reached goal state at t = {t_sim:.2f}s")

    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()

    positions = np.array(positions)
    velocities = np.array(velocities)

    plot_trajectory(times, positions, velocities)


def plot_trajectory(times, positions, velocities):
    os.makedirs("outputs", exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    labels = ["x", "y", "z"]
    for i in range(2):
        axes[0].plot(times, positions[:, i], label=labels[i])
        axes[1].plot(times, velocities[:, i], label=labels[i])
    axes[0].set_ylabel("position (m)")
    axes[0].set_title("Rover CEM Waypoint Tracking")
    axes[1].set_ylabel("velocity (m/s)")
    axes[1].set_xlabel("time (s)")
    axes[0].legend()
    axes[1].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/rover_mpc_trajectory.png", dpi=150)
    print("Saved plot to outputs/rover_mpc_trajectory.png")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(positions[:, 0], positions[:, 1], "b-", linewidth=1.5)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rover MPC waypoint tracking demo")
    parser.add_argument("--viewer", action="store_true", help="Render live in the MuJoCo passive viewer")
    args = parser.parse_args()
    run(use_viewer=args.viewer)
