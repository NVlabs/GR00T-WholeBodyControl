#!/usr/bin/env python3
"""
Convert a GMR robot motion pickle (e.g. output of GMR/scripts/gvhmr_to_robot.py with --save_path)
into SONIC reference motion CSVs for g1_deploy / MotionDataReader.

Prerequisites
-------------
1. Human pose -> GMR retarget:
   cd GMR && python scripts/gvhmr_to_robot.py \\
     --gvhmr_pred_file /path/to/hmr4d_results.pt \\
     --robot unitree_g1 \\
     --save_path /path/to/clip.pkl \\
     --rate_limit

2. Run this script on the resulting .pkl.

Output (per clip subfolder)
---------------------------
- joint_pos.csv, joint_vel.csv  (29 joints, IsaacLab column order)
- body_pos.csv, body_quat.csv   (root only; quaternion w,x,y,z)
- metadata.txt                  (Body part indexes: [0])
- info.txt                      (human-readable summary; not read by C++)

Validation
----------
  cd gear_sonic_deploy
  python visualize_motion.py --motion_dir reference/<parent>/<clip_name>/

  bash deploy.sh --motion-data reference/<parent>/ sim

References
----------
- Motion format: https://nvlabs.github.io/GR00T-WholeBodyControl/references/motion_reference.html
- dof layout matches visualize_motion.py joint_pos -> dof remap.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Same ordering as gear_sonic_deploy/visualize_motion.py / policy_parameters.hpp
# dof[k] = joint_pos[:, ISAACLAB_TO_MUJOCO[k]]  =>  joint_pos[j] = dof[INV_MUJOCO_SLOT[j]]
ISAACLAB_TO_MUJOCO = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
]


def _build_mujoco_slot_for_isaaclab() -> list[int]:
    inv = [0] * 29
    for mj_slot in range(29):
        j_isaac = ISAACLAB_TO_MUJOCO[mj_slot]
        inv[j_isaac] = mj_slot
    return inv


MUJOCO_SLOT_FOR_ISAACLAB = _build_mujoco_slot_for_isaaclab()


def _slerp_quat_xyzw(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation; quaternions xyzw, shape (4,)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    s0 = np.sin((1.0 - t) * theta_0) / sin_theta_0
    s1 = np.sin(t * theta_0) / sin_theta_0
    out = s0 * q0 + s1 * q1
    return out / (np.linalg.norm(out) + 1e-12)


def resample_root_rot_slerp(
    t_src: np.ndarray, t_tgt: np.ndarray, quat_xyzw: np.ndarray
) -> np.ndarray:
    """quat_xyzw: (T, 4) in x,y,z,w. Returns (T_tgt, 4)."""
    try:
        from scipy.spatial.transform import Rotation, Slerp

        rots = Rotation.from_quat(quat_xyzw)
        slerp = Slerp(t_src, rots)
        return slerp(t_tgt).as_quat()
    except ImportError:
        pass
    out = np.zeros((len(t_tgt), 4), dtype=np.float64)
    idx = np.searchsorted(t_src, t_tgt, side="right") - 1
    idx = np.clip(idx, 0, len(t_src) - 2)
    alpha = (t_tgt - t_src[idx]) / (t_src[idx + 1] - t_src[idx] + 1e-12)
    for i in range(len(t_tgt)):
        out[i] = _slerp_quat_xyzw(quat_xyzw[idx[i]], quat_xyzw[idx[i] + 1], float(alpha[i]))
    return out.astype(np.float64)


def resample_linear(t_src: np.ndarray, t_tgt: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y: (T, D) -> (T_tgt, D)"""
    if y.ndim == 1:
        y = y[:, None]
    out = np.zeros((len(t_tgt), y.shape[1]), dtype=np.float64)
    for d in range(y.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, y[:, d].astype(np.float64))
    return out


def joint_velocity_from_positions(q: np.ndarray, dt: float) -> np.ndarray:
    """Central differences; q shape (T, 29)."""
    v = np.zeros_like(q, dtype=np.float64)
    if q.shape[0] < 2:
        return v
    v[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    v[0] = (q[1] - q[0]) / dt
    v[-1] = (q[-1] - q[-2]) / dt
    return v


def write_csv_matrix(path: str, data: np.ndarray, headers: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in data:
            f.write(",".join(f"{float(x):.6f}" for x in row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GMR .pkl (unitree_g1 retarget) -> SONIC reference motion folder (CSV)."
    )
    parser.add_argument("--pkl", type=str, required=True, help="Path to GMR robot motion .pkl")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Base directory; creates <out>/<clip-name>/ with CSVs",
    )
    parser.add_argument(
        "--clip-name",
        type=str,
        default=None,
        help="Subfolder name (default: stem of --pkl)",
    )
    parser.add_argument("--target-fps", type=float, default=50.0, help="Output rate (default 50)")
    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Override fps from pickle (if detector / file fps is wrong)",
    )
    parser.add_argument(
        "--root-pos-source",
        choices=("pkl", "zeros"),
        default="pkl",
        help="body_pos.csv: use GMR root_pos or zeros (minimal motion ref)",
    )
    args = parser.parse_args()

    pkl_path = Path(args.pkl).resolve()
    if not pkl_path.is_file():
        print(f"Error: pickle not found: {pkl_path}", file=sys.stderr)
        return 1

    with open(pkl_path, "rb") as f:
        motion = pickle.load(f)

    required = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [k for k in required if k not in motion]
    if missing:
        print(f"Error: pickle missing keys {missing}", file=sys.stderr)
        return 1

    fps_in = float(args.source_fps) if args.source_fps is not None else float(motion["fps"])
    if fps_in <= 0:
        print("Error: fps must be positive", file=sys.stderr)
        return 1

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)  # xyzw in GMR pickle
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        print("Error: root_pos must be (T, 3)", file=sys.stderr)
        return 1
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        print("Error: root_rot must be (T, 4) xyzw", file=sys.stderr)
        return 1
    if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
        print("Error: dof_pos must be (T, 29)", file=sys.stderr)
        return 1

    t = root_pos.shape[0]
    if dof_pos.shape[0] != t or root_rot.shape[0] != t:
        print("Error: root_pos, root_rot, dof_pos must have same number of rows", file=sys.stderr)
        return 1

    # IsaacLab-order joint positions (columns joint_0..joint_28)
    joint_isaac = np.zeros((t, 29), dtype=np.float64)
    for j in range(29):
        mj_slot = MUJOCO_SLOT_FOR_ISAACLAB[j]
        joint_isaac[:, j] = dof_pos[:, mj_slot]

    t_src = np.arange(t, dtype=np.float64) / fps_in
    t_end = float(t_src[-1])
    target_fps = float(args.target_fps)
    if target_fps <= 0:
        print("Error: --target-fps must be positive", file=sys.stderr)
        return 1
    dt_tgt = 1.0 / target_fps
    t_tgt = np.arange(0.0, t_end + 1e-9, dt_tgt, dtype=np.float64)
    if len(t_tgt) < 2:
        print("Error: trajectory too short after resampling", file=sys.stderr)
        return 1

    joint_tgt = resample_linear(t_src, t_tgt, joint_isaac)
    if args.root_pos_source == "pkl":
        root_pos_tgt = resample_linear(t_src, t_tgt, root_pos)
    else:
        root_pos_tgt = np.zeros((len(t_tgt), 3), dtype=np.float64)

    root_quat_tgt = resample_root_rot_slerp(t_src, t_tgt, root_rot)

    joint_vel_tgt = joint_velocity_from_positions(joint_tgt, dt_tgt)

    # body_quat: wxyz for SONIC CSV
    x, y, z, w = (
        root_quat_tgt[:, 0],
        root_quat_tgt[:, 1],
        root_quat_tgt[:, 2],
        root_quat_tgt[:, 3],
    )
    body_quat = np.stack([w, x, y, z], axis=1)

    clip = args.clip_name if args.clip_name else pkl_path.stem
    out_dir = Path(args.out).resolve() / clip
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_headers = [f"joint_{i}" for i in range(29)]
    vel_headers = [f"joint_vel_{i}" for i in range(29)]
    body_pos_headers = ["body_0_x", "body_0_y", "body_0_z"]
    body_quat_headers = ["body_0_w", "body_0_x", "body_0_y", "body_0_z"]

    write_csv_matrix(str(out_dir / "joint_pos.csv"), joint_tgt, joint_headers)
    write_csv_matrix(str(out_dir / "joint_vel.csv"), joint_vel_tgt, vel_headers)
    write_csv_matrix(str(out_dir / "body_pos.csv"), root_pos_tgt, body_pos_headers)
    write_csv_matrix(str(out_dir / "body_quat.csv"), body_quat, body_quat_headers)

    meta_path = out_dir / "metadata.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"Metadata for: {clip}\n")
        f.write("=" * 30 + "\n\n")
        f.write("Body part indexes:\n")
        f.write("[0]\n\n")
        f.write(f"Total timesteps: {joint_tgt.shape[0]}\n")

    info_path = out_dir / "info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Source pkl: {pkl_path}\n")
        f.write(f"Source fps (used): {fps_in}\n")
        f.write(f"Target fps: {target_fps}\n")
        f.write(f"Frames out: {joint_tgt.shape[0]}\n")
        f.write(f"root_pos source: {args.root_pos_source}\n")
        f.write("root_rot convention in pkl: xyzw (GMR gvhmr_to_robot.py)\n")
        f.write("body_quat.csv: wxyz (SONIC MotionDataReader)\n")

    print(f"Wrote motion to: {out_dir}")
    print(f"  frames={joint_tgt.shape[0]}  dt={dt_tgt:.6f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
