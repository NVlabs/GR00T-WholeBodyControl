#!/usr/bin/env python3
"""Verify the B1 port's hardcoded orderings against a real Isaac Lab articulation.

Why this exists: ``robots/b1.py``'s IsaacLab body/joint order was derived by assuming
PhysX enumerates the articulation breadth-first (level order) -- an assumption checked
only against G1's committed list, not against a live B1 load. A wrong ordering does not
crash; it silently scrambles observations and actions, so confirm it before spending GPU
hours. This also reports the exact spawn height, which ``b1.py`` leaves at the MJCF's
nominal 0.9.

    IsaacLab/_isaac_sim/python.sh gear_sonic/scripts/verify_b1_orders.py

Run from the repo root (B1_CFG's asset_path is repo-relative). Do NOT pass ``--headless``:
it is deprecated in Isaac Lab 3.x, where headless is already the default and ``--viz`` is
how you opt into visualization. Exits non-zero on any mismatch, so it is safe to wire
into CI.
"""

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__)
try:
    from isaaclab.app import AppLauncher
except ImportError:
    sys.exit(
        "ERROR: Isaac Lab is required. Activate the Isaac Lab environment first.\n"
        "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html"
    )
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.assets import Articulation  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

from gear_sonic.envs.manager_env.robots.b1 import (  # noqa: E402
    B1_CFG,
    B1_ISAACLAB_DOF_NAMES,
    B1_ISAACLAB_JOINTS,
    B1_ISAACLAB_TO_MUJOCO_BODY,
    B1_ISAACLAB_TO_MUJOCO_DOF,
    B1_MUJOCO_BODIES,
    B1_MUJOCO_DOF_NAMES,
    B1_MUJOCO_TO_ISAACLAB_BODY,
    B1_MUJOCO_TO_ISAACLAB_DOF,
)


def report(label, expected, actual):
    """Print a name-list comparison and return True when it matches."""
    if list(expected) == list(actual):
        print(f"  OK   {label}: {len(actual)} entries match")
        return True
    print(f"  FAIL {label}:")
    for i in range(max(len(expected), len(actual))):
        e = expected[i] if i < len(expected) else "<missing>"
        a = actual[i] if i < len(actual) else "<missing>"
        print(f"    {i:3d}  expected {e:<28} actual {a}{'' if e == a else '   <-- differs'}")
    return False


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
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 200.0))
    # A ground plane is required for the spawn-height check below to mean anything --
    # without one the robot free-falls and the "settled" height is just 0.5*g*t^2.
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2000.0)
    )
    robot = Articulation(B1_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    place_at_default_state(robot)

    print("\n=== IsaacLab ordering, as reported by the loaded articulation ===")
    ok = report("body order (B1_ISAACLAB_JOINTS)", B1_ISAACLAB_JOINTS, robot.body_names)
    ok &= report("joint order (B1_ISAACLAB_DOF_NAMES)", B1_ISAACLAB_DOF_NAMES, robot.joint_names)

    print("\n=== index mappings ===")
    # out[i] = in[mapping[i]], so applying a mapping to the source NAME list must
    # reproduce the destination name list exactly.
    for label, src, mapping, dst in (
        ("isaaclab->mujoco dof", robot.joint_names, B1_ISAACLAB_TO_MUJOCO_DOF, B1_MUJOCO_DOF_NAMES),
        ("mujoco->isaaclab dof", B1_MUJOCO_DOF_NAMES, B1_MUJOCO_TO_ISAACLAB_DOF, robot.joint_names),
        ("isaaclab->mujoco body", robot.body_names, B1_ISAACLAB_TO_MUJOCO_BODY, B1_MUJOCO_BODIES),
        ("mujoco->isaaclab body", B1_MUJOCO_BODIES, B1_MUJOCO_TO_ISAACLAB_BODY, robot.body_names),
    ):
        if len(src) != len(mapping):
            print(f"  FAIL {label}: mapping has {len(mapping)} entries, source has {len(src)}")
            ok = False
            continue
        ok &= report(label, dst, [src[i] for i in mapping])

    print("\n=== actuator coverage ===")
    # A regex in an actuator group that matches nothing means those joints silently run
    # with PhysX defaults instead of the gains in b1.py.
    driven = {n for a in robot.actuators.values() for n in a.joint_names}
    missing = [n for n in robot.joint_names if n not in driven]
    if missing:
        print(f"  FAIL joints with no actuator group: {missing}")
        ok = False
    else:
        print(f"  OK   all {len(robot.joint_names)} joints are covered by an actuator group")

    print("\n=== spawn height (drop test onto the ground plane) ===")
    # Hold the configured home pose with the PD gains from b1.py and let it settle, so
    # the reported height reflects the pose the policy will actually start from.
    targets = robot.data.default_joint_pos.clone()
    for _ in range(600):  # 3 s at 200 Hz
        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    root_z = robot.data.root_pos_w[0, 2].item()
    root_speed = robot.data.root_lin_vel_w[0].norm().item()
    feet = {
        n: robot.data.body_pos_w[0, robot.body_names.index(n), 2].item()
        for n in ("hdb1_left_foot", "hdb1_right_foot")
    }
    print(f"  configured init_state.pos z = {B1_CFG.init_state.pos[2]:.4f}")
    print(f"  after 3 s: root z = {root_z:.4f}  (|root lin vel| = {root_speed:.4f} m/s)")
    for n, z in feet.items():
        print(f"    {n} origin z = {z:.4f}")
    if root_speed >= 0.05:
        print("  WARN still moving after 3 s -- it has not settled; treat the numbers above")
        print("       as a snapshot, and suspect the PD gains or the home pose.")
    if root_z < 0.02 and min(feet.values()) < 0.0:
        # Through the floor is a real defect: bad collision geometry or a broken asset.
        print("  FAIL root and feet are below the ground plane -- the robot fell THROUGH it.")
        ok = False
    elif root_z < 0.5 * B1_CFG.init_state.pos[2]:
        # Toppling is NOT a failure. Holding a humanoid at a fixed pose with a passive PD
        # has no balance controller at all, so it falls over by construction; that is what
        # the SONIC policy is trained to do. Report it, but do not gate the exit code on
        # it -- this script's contract is ordering correctness.
        print(f"  WARN root fell to {root_z:.3f} m, well under the {B1_CFG.init_state.pos[2]:.2f} m"
              " spawn: it toppled. Expected for a passive PD")
        print("       hold with no balance policy -- NOT a failure of the port. Use")
        print("       preview_b1.py to see the pose and check the CoM over the feet.")
    print(
        "  NOTE body-frame origins are not mesh extremes, so this brackets rather than\n"
        "  replaces hdb1_constants.compute_home_z(). Use that for the exact value."
    )

    print("\nRESULT:", "PASS" if ok else "FAIL -- fix robots/b1.py before training")
    return 0 if ok else 1


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
