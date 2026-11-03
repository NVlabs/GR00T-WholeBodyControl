"""Replay a qpos trajectory or visualize one qpos state in MuJoCo."""

import argparse
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from decoupled_wbc.control.main.planner.simulation.robot import (
    G1Up,
    JOINT_NAMES_UP,
)
from decoupled_wbc.control.robot_model.supplemental_info.g1.g1_supplemental_info import (
    G1SupplementalInfo,
)


PLANNER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_XML = PLANNER_DIR / "simulation" / "g1_up.xml"


def load_qpos(path: Path) -> np.ndarray:
    """Load qpos from a .npy file or an NPZ containing qpos."""
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if path.suffix != ".npz":
        raise ValueError("Expected a .npy or .npz file")

    with np.load(path, allow_pickle=False) as data:
        if "qpos" in data:
            qpos = np.asarray(data["qpos"], dtype=np.float32)
            if "joint_names" not in data:
                return qpos
            joint_names = [str(name) for name in data["joint_names"].tolist()]
        elif "body_proprioception" in data:
            qpos = np.asarray(data["body_proprioception"], dtype=np.float32)
            joint_names = G1SupplementalInfo().body_actuated_joints
        else:
            raise ValueError(f"{path} contains neither 'qpos' nor 'body_proprioception'")

    name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    missing = [name for name in JOINT_NAMES_UP if name not in name_to_index]
    if missing:
        raise ValueError(f"Missing planner joints: {missing}")
    return qpos[..., [name_to_index[name] for name in JOINT_NAMES_UP]]


def validate_qpos(qpos: np.ndarray, expected_ndim: int) -> None:
    if qpos.ndim != expected_ndim or qpos.shape[-1] != len(JOINT_NAMES_UP):
        expected = "(17,)" if expected_ndim == 1 else "(N, 17)"
        raise ValueError(f"Expected qpos shape {expected}, got {qpos.shape}")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("qpos contains non-finite values")


def load_model(xml_path: Path) -> mujoco.MjModel:
    """Load includes and meshes relative to the planner directory."""
    previous_cwd = Path.cwd()
    os.chdir(PLANNER_DIR)
    try:
        return mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
    finally:
        os.chdir(previous_cwd)


def hold_viewer(robot: G1Up) -> None:
    while robot.viewer.is_running():
        time.sleep(0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trajectory", type=Path, help="Replay an (N, 17) NPZ/NPY")
    mode.add_argument("--state", type=Path, help="Display one (17,) NPY/NPZ state")
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    args = parser.parse_args()

    if args.frequency <= 0:
        raise ValueError("frequency must be positive")

    source = args.trajectory if args.trajectory is not None else args.state
    qpos = load_qpos(source.resolve())
    validate_qpos(qpos, expected_ndim=2 if args.trajectory is not None else 1)

    robot = G1Up(load_model(args.xml), visualize=True)
    try:
        if args.state is not None:
            robot.set_joint_qpos(qpos)
            robot.viewer.sync()
        else:
            period = 1.0 / args.frequency
            for state in qpos:
                if not robot.viewer.is_running():
                    return
                started = time.monotonic()
                robot.set_joint_qpos(state)
                robot.viewer.sync()
                time.sleep(max(0.0, period - (time.monotonic() - started)))
        hold_viewer(robot)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
