#!/usr/bin/env python3
"""Render the B1 articulation and report how well it holds its home pose.

Two jobs. It gives you PICTURES of the robot as Isaac Lab actually loads it (the
MJCF-import path is new for this repo, so "does it even look like a humanoid" is worth
answering visually), and it reports PER-JOINT SAG -- target minus achieved angle after
settling -- which is how you tell a gains problem from a pose problem.

Writes PNGs, so it works over SSH with no display. Run it with Isaac Sim's python:

    IsaacLab/_isaac_sim/python.sh gear_sonic/scripts/preview_b1.py --enable_cameras

``--enable_cameras`` is REQUIRED: without it Isaac Lab creates no render products and the
camera yields nothing. Do NOT pass ``--headless`` -- it is deprecated in Isaac Lab 3.x
(headless is the default; ``--viz`` opts into visualization instead). Run from the repo
root, since B1_CFG's asset_path is repo-relative.

Add --gravity-off to see the commanded pose with no load, which isolates whether the
kinematics/home pose are right before you argue about stiffness.
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out-dir", default="/tmp/b1_preview", help="directory for PNGs")
parser.add_argument(
    "--settle-seconds",
    type=float,
    default=3.0,
    help="hold the home pose this long before rendering",
)
parser.add_argument(
    "--gravity-off",
    action="store_true",
    help="disable gravity (isolates pose from load)",
)
parser.add_argument(
    "--hold-seconds",
    type=float,
    default=0.0,
    help="keep stepping after the renders so an interactive viewer stays live "
    "(use with --viz viser); 0 exits immediately, negative runs until Ctrl-C",
)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
try:
    from isaaclab.app import AppLauncher
except ImportError:
    sys.exit("ERROR: Isaac Lab is required. Run via IsaacLab/_isaac_sim/python.sh.")
AppLauncher.add_app_launcher_args(parser)
# parse_known_args (not parse_args) so raw Kit settings can be passed straight through,
# e.g. --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998
args, kit_args = parser.parse_known_args()
sys.argv[1:] = [a for a in sys.argv[1:] if a not in kit_args] + kit_args

# AppLauncher contributes --livestream; >=1 means a WebRTC client is the display.
live_stream = getattr(args, "livestream", 0) >= 1
if live_stream:
    # The WebRTC client fixes a resolution when it connects, and the client shows a black
    # window if the server then emits frames at any other size ("Cannot stream video frame
    # with resolution AxB that differs from CxD established when the client connected").
    # Isaac Sim's own streaming experience sets allowDynamicResize for this, but Isaac
    # Lab's isaaclab.python*.kit files do not, so set it here.
    resize = "--/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true"
    if resize not in sys.argv:
        sys.argv.append(resize)
    if getattr(args, "enable_cameras", False):
        print(
            "NOTE: --enable_cameras is redundant with --livestream (the offscreen camera is"
            "\n      skipped either way). Harmless, but you can drop it."
        )
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

from gear_sonic.envs.manager_env.robots.b1 import B1_CFG  # noqa: E402

ASSET_MESH_DIR = "gear_sonic/data/assets/robot_description/meshes/b1"

# Eye positions and look-at targets, chosen to frame a ~1.5 m tall robot at the origin.
# B1 stands ~1.5 m with its torso root near 0.85 m, so frame on z ~= 0.8 from ~3.5 m out.
VIEWS = {
    "front": ((3.6, 0.0, 1.0), (0.0, 0.0, 0.8)),
    "side": ((0.0, 3.6, 1.0), (0.0, 0.0, 0.8)),
    "three_quarter": ((2.6, 2.6, 1.7), (0.0, 0.0, 0.8)),
    "closeup_legs": ((2.0, 0.6, 0.55), (0.0, 0.0, 0.35)),
}


def _stl_vertices(path, scale=0.001):
    """Return an (N, 3) array of vertices from a binary or ASCII STL, in metres.

    The B1 MJCF references its meshes with scale="0.001 0.001 0.001" and the foot geoms
    carry no pos/quat offset, so these vertices are already in the foot BODY frame once
    scaled -- which is what makes the ground-clearance sum below exact.
    """
    with open(path, "rb") as f:
        blob = f.read()
    # An ASCII STL starts with "solid", but so can a binary one, so confirm on content.
    if blob[:5] == b"solid" and b"facet normal" in blob[:2048]:
        verts = [
            [float(x) for x in line.split()[1:4]]
            for line in blob.decode("ascii", "ignore").splitlines()
            if line.strip().startswith("vertex")
        ]
        return np.asarray(verts, dtype=np.float64) * scale
    # Binary: 80-byte header, uint32 triangle count, then 50 bytes per triangle
    # (12 float32 = normal + 3 vertices, plus a 2-byte attribute word).
    count = int(np.frombuffer(blob[80:84], dtype="<u4")[0])
    tri = np.frombuffer(blob[84 : 84 + count * 50], dtype=np.uint8).reshape(count, 50)
    floats = tri[:, :48].copy().view("<f4").reshape(count, 4, 3)
    return floats[:, 1:, :].reshape(-1, 3).astype(np.float64) * scale


def report_ground_clearance(robot, mesh_dir):
    """Print the true lowest point of each foot, from mesh vertices, not body origins.

    init_state.pos z in b1.py is the MJCF's nominal 0.9 and was never derived from
    geometry, so the soles can sit below z=0. mjlab solves this with
    hdb1_constants.compute_home_z(); this is the same computation done here.
    """
    from isaaclab.utils.math import quat_apply

    print("\n=== ground clearance (mesh vertices, not body origins) ===")
    lowest_overall = None
    for name, stl in (
        ("hdb1_left_foot", "hdb1_left_foot.stl"),
        ("hdb1_right_foot", "hdb1_right_foot.stl"),
    ):
        idx = robot.body_names.index(name)
        pos = robot.data.body_pos_w.torch[0, idx]
        quat = robot.data.body_quat_w.torch[0, idx]
        v = torch.as_tensor(
            _stl_vertices(os.path.join(mesh_dir, stl)), dtype=pos.dtype, device=pos.device
        )
        world = quat_apply(quat.unsqueeze(0).expand(v.shape[0], 4), v) + pos
        low = world[:, 2].min().item()
        lowest_overall = low if lowest_overall is None else min(lowest_overall, low)
        print(f"  {name}: origin z = {pos[2].item():+.4f}, lowest vertex z = {low:+.4f}")
    print(f"  lowest point of either foot: {lowest_overall:+.4f} m")
    if lowest_overall < -0.001:
        fix = B1_CFG.init_state.pos[2] - lowest_overall
        print(f"  -> SOLES ARE {abs(lowest_overall) * 100:.1f} cm BELOW the ground plane.")
        print(f"     Set init_state.pos z = {fix:.4f} in robots/b1.py to rest them on z=0.")
    elif lowest_overall > 0.005:
        print(f"  -> floating {lowest_overall * 100:.1f} cm above the ground.")
        print(f"     Set init_state.pos z = {B1_CFG.init_state.pos[2] - lowest_overall:.4f}.")
    else:
        print("  -> resting on the ground within 5 mm; no change needed.")
    return lowest_overall


def place_at_default_state(robot):
    """Put the articulation in its configured init_state.

    ``sim.reset()`` alone does NOT do this: the articulation comes up in the pose
    authored in the USD (all joints at zero for a converted MJCF), and only
    ``default_joint_pos`` on the data object reflects ``init_state``. Without this the
    robot spawns with straight legs and the PD then drags it into the home squat while
    gravity acts -- which looks exactly like a gains failure but is not.
    """
    root = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    )
    robot.reset()


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1 / 200.0,
            gravity=(0.0, 0.0, 0.0) if args.gravity_off else (0.0, 0.0, -9.81),
        )
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    for i, cfg in enumerate(
        (
            sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 1.0)),
            sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0),
        )
    ):
        cfg.func(f"/World/light_{i}", cfg)

    # NOTE: deliberately NOT calling sim.set_camera_view() here. It forwards to registered
    # visualizers and was never confirmed to move the Kit viewport that the WebRTC stream
    # captures; adding it correlated with the stream going black every time. If the opening
    # view is wrong, reframe inside the client (select the robot, press F) rather than
    # re-adding this.
    robot = Articulation(B1_CFG.replace(prim_path="/World/Robot"))
    # An offscreen Camera sensor defines its own render product at args.width x
    # args.height. Under --livestream that RESIZES the render target after the WebRTC
    # client has already negotiated a resolution, and the client just shows black. So
    # when streaming we skip the camera entirely and let the viewport be the only
    # consumer -- you are watching it live, the PNGs would be redundant anyway.
    camera = None
    if not live_stream:
        camera = Camera(
            CameraCfg(
                prim_path="/World/cam",
                update_period=0,
                width=args.width,
                height=args.height,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.05, 1e5)
                ),
            )
        )
    sim.reset()
    place_at_default_state(robot)

    targets = robot.data.default_joint_pos.clone()
    steps = int(args.settle_seconds / sim.get_physics_dt())
    for _ in range(steps):
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
    # Render needs at least one step even when settling is skipped.
    if steps == 0:
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    # --- per-joint sag ---------------------------------------------------------
    achieved = robot.data.joint_pos[0]
    commanded = targets[0]
    err = (achieved - commanded).cpu().numpy()
    deg = 180.0 / np.pi
    print(f"\n=== home-pose tracking after {args.settle_seconds:g} s"
          f"{' (gravity OFF)' if args.gravity_off else ''} ===")
    print(f"{'joint':<18} {'target':>9} {'actual':>9} {'error':>9}  {'error':>8}")
    print(f"{'':<18} {'(deg)':>9} {'(deg)':>9} {'(deg)':>9}")
    order = np.argsort(-np.abs(err))
    for i in order:
        n = robot.joint_names[i]
        print(f"{n:<18} {commanded[i].item()*deg:>9.2f} {achieved[i].item()*deg:>9.2f} "
              f"{err[i]*deg:>9.2f}  {'<-- worst' if i == order[0] else ''}")
    print(f"\nmax |error| = {np.abs(err).max()*deg:.2f} deg on "
          f"{robot.joint_names[int(np.argmax(np.abs(err)))]}, "
          f"RMS = {np.sqrt((err**2).mean())*deg:.2f} deg")

    # --- CoM over support polygon ----------------------------------------------
    # A humanoid held by a passive PD has no balance controller, so it WILL topple
    # eventually; that alone is not a bug. What matters for training is whether the
    # spawn pose starts balanced. Measure the horizontal CoM against the feet.
    # These are warp-backed ProxyArrays; .torch unwraps to real tensors. Indexing a
    # ProxyArray works, but .device on one returns a warp Device that torch rejects, so
    # unwrap both before doing tensor math. (body_mass supersedes the deprecated
    # default_mass, which Isaac Lab 4.0 removes.)
    pos = robot.data.body_pos_w.torch[0]
    masses = robot.data.body_mass.torch[0].to(pos.device)
    com = (pos * masses.unsqueeze(-1)).sum(dim=0) / masses.sum()
    lf = pos[robot.body_names.index("hdb1_left_foot")]
    rf = pos[robot.body_names.index("hdb1_right_foot")]
    mid = (lf + rf) / 2
    print(f"\ntotal mass = {masses.sum().item():.2f} kg")
    print(f"CoM (x, y, z) = ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f})")
    print(f"feet midpoint (x, y) = ({mid[0]:.4f}, {mid[1]:.4f})")
    print(f"CoM offset from feet midpoint: dx = {(com[0]-mid[0]).item():+.4f} m, "
          f"dy = {(com[1]-mid[1]).item():+.4f} m")
    lever = (com[0] - mid[0]).item()
    tau = masses.sum().item() * 9.81 * abs(lever)
    print(f"  -> static ankle-pitch torque to resist that lean: {tau:.1f} Nm "
          f"(the pair of ankles supply up to 62 Nm)")
    print("  NOTE B1 has NO ankle roll, so lateral (dy) balance has zero ankle authority")
    print("       and must come from hip roll alone.")

    report_ground_clearance(
        robot, os.path.join(ASSET_MESH_DIR, "collisions")
    )

    root_z = robot.data.root_pos_w[0, 2].item()
    feet_z = [
        robot.data.body_pos_w[0, robot.body_names.index(n), 2].item()
        for n in ("hdb1_left_foot", "hdb1_right_foot")
    ]
    print(f"\nroot (TORSO) z = {root_z:.4f}, feet origin z = {feet_z[0]:.4f} / {feet_z[1]:.4f}")
    if root_z < min(feet_z):
        print("  ^ torso is BELOW the feet: the robot is not standing.")

    # --- renders --------------------------------------------------------------
    if camera is None:
        print(
            "\nlivestreaming: skipping the offscreen PNG renders (an extra render product"
            "\nwould resize the stream and black out the client). Connect the Isaac Sim"
            "\nWebRTC Streaming Client to this host now."
        )
        views = {}
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        views = VIEWS
    written = []
    for name, (eye, target) in views.items():
        camera.set_world_poses_from_view(
            torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device)
        )
        # A few render ticks let the RTX renderer converge before the grab.
        for _ in range(8):
            sim.render()
        camera.update(dt=0.0)
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        path = os.path.join(args.out_dir, f"b1_{name}.png")
        import imageio.v3 as iio

        iio.imwrite(path, rgb)
        written.append(path)
        print(f"wrote {path}")

    if args.hold_seconds != 0.0:
        # Keep the physics loop alive so an attached viewer (e.g. --viz viser on
        # http://<host>:8080) has something to show and stays connected.
        print(
            f"\nholding for {'ever' if args.hold_seconds < 0 else f'{args.hold_seconds:g} s'}"
            " -- attach a viewer now (Ctrl-C to stop)"
        )
        held = 0.0
        try:
            while args.hold_seconds < 0 or held < args.hold_seconds:
                robot.set_joint_position_target(targets)
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.get_physics_dt())
                held += sim.get_physics_dt()
        except KeyboardInterrupt:
            print("\ninterrupted -- shutting down")
    return 0


if __name__ == "__main__":
    # simulation_app.close() force-exits the process, which swallows a pending exception
    # and still reports status 0 -- a silent failure. Print and set the code ourselves.
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
    finally:
        simulation_app.close()
    sys.exit(code)
