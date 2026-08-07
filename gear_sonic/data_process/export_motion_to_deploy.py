#!/usr/bin/env python3
"""Export motion_lib motions to the ``gear_sonic_deploy`` reference CSV format."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import sys

import easydict
import joblib
import numpy as np
import torch

from gear_sonic.utils.motion_lib import motion_lib_robot

_G1_MAPPING_NAMES = (
    "G1_ISAACLAB_JOINTS",
    "G1_ISAACLAB_TO_MUJOCO_DOF",
    "G1_MUJOCO_TO_ISAACLAB_DOF",
    "G1_ISAACLAB_TO_MUJOCO_BODY",
    "G1_MUJOCO_TO_ISAACLAB_BODY",
)


def _load_g1_mappings() -> dict[str, list]:
    """Return the G1 IsaacLab<->MuJoCo ordering tables.

    ``robots/g1.py`` defines these as plain list literals but imports isaaclab at
    module scope, which drags in the Isaac Sim runtime (``carb``).  Export is a
    kinematics-only operation, so fall back to reading the literals straight out
    of the source when Isaac Sim is unavailable.
    """
    try:
        from gear_sonic.envs.manager_env.robots import g1  # noqa: PLC0415

        return {name: getattr(g1, name) for name in _G1_MAPPING_NAMES}
    except Exception:  # noqa: BLE001 - any isaaclab/carb import failure
        g1_path = (
            Path(__file__).resolve().parents[1]
            / "envs"
            / "manager_env"
            / "robots"
            / "g1.py"
        )
        tree = ast.parse(g1_path.read_text())
        found: dict[str, list] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _G1_MAPPING_NAMES:
                    found[target.id] = ast.literal_eval(node.value)
        missing = set(_G1_MAPPING_NAMES) - set(found)
        if missing:
            raise RuntimeError(f"Could not read {sorted(missing)} from {g1_path}") from None
        return found

# The 14 bodies the deploy reference format carries, in order.  Indices are into
# the IsaacLab body ordering (post ``mujoco_to_isaaclab_body`` remap).  Verified
# against the shipped ``reference/example/*/metadata.txt``.
DEPLOY_BODY_INDEXES = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]

DEFAULT_ASSET_ROOT = "gear_sonic/data/assets/robot_description/mjcf/"
DEFAULT_ASSET_FILE = "g1_29dof_rev_1_0.xml"

# Keys convert_motions.py consumes.
_EXPORT_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def build_motion_lib_cfg(
    motion_file: str,
    target_fps: int = 50,
    body_indexes: list[int] | None = None,
    asset_root: str = DEFAULT_ASSET_ROOT,
    asset_file: str = DEFAULT_ASSET_FILE,
    zero_root_xy: bool = False,
) -> easydict.EasyDict:
    """Assemble the motion_lib config, mirroring ``MotionCommand.create_offline``."""
    mapping = _load_g1_mappings()
    if body_indexes is None:
        body_indexes = list(DEPLOY_BODY_INDEXES)

    cfg = easydict.EasyDict(
        {
            "motion_file": motion_file,
            "smpl_motion_file": None,
            "asset": {
                "assetRoot": asset_root,
                "assetFileName": asset_file,
                "urdfFileName": "",
            },
            "extend_config": [],
            "target_fps": target_fps,
            "multi_thread": False,
            "use_parallel_fk": False,
            "zero_root_xy": zero_root_xy,
            "body_indexes_data": body_indexes,
            "mujoco_to_isaaclab_body": mapping["G1_MUJOCO_TO_ISAACLAB_BODY"],
            "mujoco_to_isaaclab_dof": mapping["G1_MUJOCO_TO_ISAACLAB_DOF"],
            "isaaclab_to_mujoco_body": mapping["G1_ISAACLAB_TO_MUJOCO_BODY"],
            "isaaclab_to_mujoco_dof": mapping["G1_ISAACLAB_TO_MUJOCO_DOF"],
        }
    )
    return cfg


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def export_motions(
    motion_file: str,
    target_fps: int = 50,
    body_indexes: list[int] | None = None,
    device: str = "cpu",
    zero_root_xy: bool = False,
    asset_root: str = DEFAULT_ASSET_ROOT,
    asset_file: str = DEFAULT_ASSET_FILE,
) -> dict:
    """Load ``motion_file`` through MotionLibRobot and return a deploy-format dict."""
    if body_indexes is None:
        body_indexes = list(DEPLOY_BODY_INDEXES)

    cfg = build_motion_lib_cfg(
        motion_file=motion_file,
        target_fps=target_fps,
        body_indexes=body_indexes,
        asset_root=asset_root,
        asset_file=asset_file,
        zero_root_xy=zero_root_xy,
    )

    motion_lib = motion_lib_robot.MotionLibRobot(cfg, num_envs=1, device=device)
    motion_lib.load_motions_for_training()

    if not hasattr(motion_lib, "dof_pos"):
        raise RuntimeError(
            "motion_lib has no 'dof_pos' -- the source PKL lacks per-frame DOF data."
        )

    exported: dict[str, dict] = {}
    num_motions = len(motion_lib.curr_motion_keys)
    for i in range(num_motions):
        name = str(motion_lib.curr_motion_keys[i])
        start = int(motion_lib.length_starts[i])
        n_frames = int(motion_lib._motion_num_frames[i])  # noqa: SLF001
        end = start + n_frames

        entry = {
            "joint_pos": _to_numpy(motion_lib.dof_pos[start:end]),
            "joint_vel": _to_numpy(motion_lib.dof_vel[start:end]),
            "body_pos_w": _to_numpy(motion_lib.body_pos_w[start:end]),
            "body_quat_w": _to_numpy(motion_lib.body_quat_w[start:end]),
            "body_lin_vel_w": _to_numpy(motion_lib.body_lin_vel_w[start:end]),
            "body_ang_vel_w": _to_numpy(motion_lib.body_ang_vel_w[start:end]),
            "_body_indexes": np.asarray(body_indexes, dtype=np.int64),
            "time_step_total": np.int64(n_frames),
        }

        for key in _EXPORT_KEYS:
            if entry[key].shape[0] != n_frames:
                raise RuntimeError(
                    f"{name}: '{key}' has {entry[key].shape[0]} frames, expected {n_frames}"
                )
        if entry["body_pos_w"].shape[1] != len(body_indexes):
            raise RuntimeError(
                f"{name}: body_pos_w has {entry['body_pos_w'].shape[1]} bodies, "
                f"expected {len(body_indexes)}"
            )

        exported[name] = entry
        print(  # noqa: T201
            f"  {name}: {n_frames} frames @ {target_fps} Hz, "
            f"{entry['joint_pos'].shape[1]} dof, {entry['body_pos_w'].shape[1]} bodies"
        )

    return exported


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export motion_lib motions to gear_sonic_deploy reference CSVs.",
    )
    parser.add_argument("motion_file", help="Input motion_lib PKL.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output dir; one subfolder per motion is created inside it.",
    )
    parser.add_argument("--target-fps", type=int, default=50, help="Deploy control rate.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--zero-root-xy",
        action="store_true",
        help="Shift each motion so root XY starts at the origin.",
    )
    parser.add_argument("--asset-root", default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--asset-file", default=DEFAULT_ASSET_FILE)
    parser.add_argument(
        "--body-indexes",
        default=None,
        help="Comma-separated body indexes; defaults to the 14 deploy bodies.",
    )
    parser.add_argument(
        "--pkl-out",
        default=None,
        help="Also write the intermediate deploy-format PKL here.",
    )
    args = parser.parse_args()

    body_indexes = (
        [int(x) for x in args.body_indexes.split(",")] if args.body_indexes else None
    )

    print(f"Loading {args.motion_file} at target_fps={args.target_fps}")  # noqa: T201
    exported = export_motions(
        motion_file=args.motion_file,
        target_fps=args.target_fps,
        body_indexes=body_indexes,
        device=args.device,
        zero_root_xy=args.zero_root_xy,
        asset_root=args.asset_root,
        asset_file=args.asset_file,
    )
    if not exported:
        print("No motions exported.", file=sys.stderr)  # noqa: T201
        return 1

    pkl_path = args.pkl_out
    if pkl_path is None:
        os.makedirs(args.output_dir, exist_ok=True)
        pkl_path = os.path.join(args.output_dir, "_deploy_export.pkl")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(pkl_path)), exist_ok=True)
    joblib.dump(exported, pkl_path)
    print(f"Wrote intermediate PKL: {pkl_path}")  # noqa: T201

    # Reuse the deploy-side CSV writer verbatim rather than duplicating it.
    repo_root = Path(__file__).resolve().parents[2]
    converter_dir = repo_root / "gear_sonic_deploy" / "reference"
    sys.path.insert(0, str(converter_dir))
    import convert_motions  # noqa: PLC0415

    convert_motions.convert_motion_data(pkl_path, args.output_dir)
    print(f"\nDone. Reference CSVs under: {args.output_dir}/")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
