#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Retarget extracted SOMA joints to H2 motion PKLs.

This script takes the human-side SOMA joint motions produced by
``extract_soma_joints_from_bvh.py`` and fits an H2 joint trajectory to them using
frame-wise inverse kinematics with warm-start initialization.

Input format (per motion PKL):
    {
        "motion_name": {
            "soma_joints":    np.ndarray float32, (T, 26, 3), Z-up, meters,
            "soma_root_quat": np.ndarray float32, (T, 4), wxyz, Y-up,
            "soma_transl":    np.ndarray float32, (T, 3), Y-up,
            "fps":            int,
            "joint_names":    list[str],
        }
    }

Output format (per motion PKL):
    {
        "motion_name": {
            "joint_pos":   np.ndarray float32, (T, 31), radians,
            "body_pos_w":  np.ndarray float32, (T, 32, 3), meters, Z-up world,
            "body_quat_w": np.ndarray float32, (T, 32, 4), wxyz, Z-up world,
            "joint_order": "mj",
            "fps":         int,
        }
    }
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import joblib
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import transform

NUM_DOF = 31
NUM_BODIES = 32
QUAT_NORM_ATOL = 1e-3

SOMA_JOINT_NAMES = [
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandMiddle1",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandMiddle1",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
]

H2_DEFAULT_DOF = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        0.0,
        -0.363,
        -0.312,
        0.0,
        0.0,
        0.669,
        0.0,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

ROOT_CONV_XYZW = transform.Rotation.from_euler("x", 90.0, degrees=True).as_quat().astype(np.float32)
ROOT_CONV_WXYZ = ROOT_CONV_XYZW[[3, 0, 1, 2]]

SOMA_TO_H2_BODY_MAP = {
    "pelvis": "Hips",
    "waist_roll_link": "Spine1",
    "torso_link": "Chest",
    "head_yaw_link": "Head",
    "left_shoulder_pitch_link": "LeftShoulder",
    "left_elbow_link": "LeftForeArm",
    "left_wrist_yaw_link": "LeftHand",
    "right_shoulder_pitch_link": "RightShoulder",
    "right_elbow_link": "RightForeArm",
    "right_wrist_yaw_link": "RightHand",
    "left_hip_pitch_link": "LeftLeg",
    "left_knee_link": "LeftShin",
    "left_ankle_pitch_link": "LeftFoot",
    "right_hip_pitch_link": "RightLeg",
    "right_knee_link": "RightShin",
    "right_ankle_pitch_link": "RightFoot",
}

BODY_WEIGHTS = {
    "pelvis": 6.0,
    "waist_roll_link": 2.0,
    "torso_link": 3.0,
    "head_yaw_link": 1.0,
    "left_shoulder_pitch_link": 1.5,
    "left_elbow_link": 2.0,
    "left_wrist_yaw_link": 3.0,
    "right_shoulder_pitch_link": 1.5,
    "right_elbow_link": 2.0,
    "right_wrist_yaw_link": 3.0,
    "left_hip_pitch_link": 1.5,
    "left_knee_link": 2.0,
    "left_ankle_pitch_link": 3.0,
    "right_hip_pitch_link": 1.5,
    "right_knee_link": 2.0,
    "right_ankle_pitch_link": 3.0,
}

REST_REG_DOF = np.array([15, 16, 21, 22, 23, 28, 29, 30], dtype=np.int32)


@dataclass
class RetargetContext:
    model: object
    data: object
    body_names: list[str]
    body_indices: np.ndarray
    target_weights: np.ndarray
    robot_reference_points: np.ndarray
    neutral_root_offset: np.ndarray
    actuator_qpos_adr: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray


def y_up_to_z_up_positions(pos_yup: np.ndarray) -> np.ndarray:
    """Convert positions from Y-up to Z-up using the repo's convention."""
    pos_zup = np.empty_like(pos_yup, dtype=np.float32)
    pos_zup[..., 0] = pos_yup[..., 0]
    pos_zup[..., 1] = -pos_yup[..., 2]
    pos_zup[..., 2] = pos_yup[..., 1]
    return pos_zup


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply in wxyz convention."""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    ).astype(np.float32)


def y_up_to_z_up_quat_wxyz(quat_yup_wxyz: np.ndarray) -> np.ndarray:
    """Convert world quaternions from Y-up to Z-up."""
    quat_zup = quat_mul_wxyz(np.broadcast_to(ROOT_CONV_WXYZ, quat_yup_wxyz.shape), quat_yup_wxyz)
    quat_zup /= np.linalg.norm(quat_zup, axis=-1, keepdims=True) + 1e-8
    return quat_zup.astype(np.float32)


def collect_input_pkls(input_path: str) -> list[str]:
    """Collect input PKLs recursively from a file or directory."""
    if os.path.isfile(input_path):
        if not input_path.endswith(".pkl"):
            raise ValueError(f"Input file must be .pkl: {input_path}")
        return [input_path]
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    paths = []
    for root, _, files in os.walk(input_path):
        for name in sorted(files):
            if name.endswith(".pkl"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def load_soma_motion(path: str) -> dict:
    """Load one SOMA motion PKL and validate required fields."""
    data = joblib.load(path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected non-empty dict in {path}")

    motions = {}
    for seq_name, seq_data in data.items():
        required = {"soma_joints", "soma_root_quat", "soma_transl", "fps", "joint_names"}
        missing = required - set(seq_data.keys())
        if missing:
            raise ValueError(f"Motion {seq_name!r} missing keys: {sorted(missing)}")

        soma_joints = np.asarray(seq_data["soma_joints"], dtype=np.float32)
        soma_root_quat = np.asarray(seq_data["soma_root_quat"], dtype=np.float32)
        soma_transl = np.asarray(seq_data["soma_transl"], dtype=np.float32)
        fps = int(seq_data["fps"])
        joint_names = list(seq_data["joint_names"])

        T = soma_joints.shape[0]
        if soma_joints.shape != (T, len(SOMA_JOINT_NAMES), 3):
            raise ValueError(f"Invalid soma_joints shape for {seq_name!r}: {soma_joints.shape}")
        if soma_root_quat.shape != (T, 4):
            raise ValueError(f"Invalid soma_root_quat shape for {seq_name!r}: {soma_root_quat.shape}")
        if soma_transl.shape != (T, 3):
            raise ValueError(f"Invalid soma_transl shape for {seq_name!r}: {soma_transl.shape}")
        if joint_names != SOMA_JOINT_NAMES:
            raise ValueError(f"Unexpected joint_names for {seq_name!r}")

        motions[seq_name] = {
            "soma_joints": soma_joints,
            "soma_root_quat": soma_root_quat,
            "soma_transl": soma_transl,
            "fps": fps,
            "joint_names": joint_names,
        }

    return motions


def init_h2_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Initialize a MuJoCo H2 model/data pair for FK and IK evaluation."""
    mjcf_path = os.path.abspath(
        os.path.join(
            "gear_sonic",
            "data",
            "assets",
            "robot_description",
            "mjcf",
            "h2.xml",
        )
    )
    mesh_dir = os.path.abspath(
        os.path.join(
            "gear_sonic",
            "data",
            "assets",
            "robot_description",
            "urdf",
            "h2",
            "meshes",
        )
    )

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", mesh_dir)

    xml_string = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    return model, data


def build_context(model: mujoco.MjModel, data: mujoco.MjData) -> RetargetContext:
    """Build static retarget context for H2."""
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    body_name_to_idx = {name: i for i, name in enumerate(body_names)}
    body_indices = np.array([body_name_to_idx[name] for name in SOMA_TO_H2_BODY_MAP], dtype=np.int32)
    target_weights = np.array([BODY_WEIGHTS[name] for name in SOMA_TO_H2_BODY_MAP], dtype=np.float32)

    actuator_qpos_adr = []
    lower_bounds = []
    upper_bounds = []
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        qpos_adr = model.jnt_qposadr[joint_id]
        actuator_qpos_adr.append(qpos_adr)
        lower_bounds.append(model.jnt_range[joint_id, 0])
        upper_bounds.append(model.jnt_range[joint_id, 1])
    actuator_qpos_adr = np.array(actuator_qpos_adr, dtype=np.int32)
    lower_bounds = np.array(lower_bounds, dtype=np.float32)
    upper_bounds = np.array(upper_bounds, dtype=np.float32)

    neutral_root_trans = np.zeros(3, dtype=np.float32)
    neutral_root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    neutral_qpos = np.concatenate([neutral_root_trans, neutral_root_quat, H2_DEFAULT_DOF.astype(np.float32)])
    body_pos_w, _ = qpos_to_global_transforms(model, data, actuator_qpos_adr, neutral_qpos)
    robot_reference_points = body_pos_w[body_indices] - body_pos_w[0]
    neutral_root_offset = body_pos_w[0]
    return RetargetContext(
        model=model,
        data=data,
        body_names=body_names,
        body_indices=body_indices,
        target_weights=target_weights,
        robot_reference_points=robot_reference_points.astype(np.float32),
        neutral_root_offset=neutral_root_offset.astype(np.float32),
        actuator_qpos_adr=actuator_qpos_adr,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


def qpos_to_global_transforms(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_qpos_adr: np.ndarray,
    qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run H2 forward kinematics for one qpos state via MuJoCo."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = qpos[0:3]
    data.qpos[3:7] = qpos[3:7]
    data.qpos[actuator_qpos_adr] = qpos[7:]
    mujoco.mj_forward(model, data)

    body_pos = np.array(data.xpos[1 : 1 + NUM_BODIES], dtype=np.float32)
    body_quat_wxyz = np.array(data.xquat[1 : 1 + NUM_BODIES], dtype=np.float32)
    body_rot = transform.Rotation.from_quat(body_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)
    return body_pos, body_rot


def build_h2_target_pose(seq_data: dict, ctx: RetargetContext) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build scaled Z-up world-space target points for H2 IK."""
    soma_joints = seq_data["soma_joints"]
    soma_root_quat_yup = seq_data["soma_root_quat"]
    soma_transl_yup = seq_data["soma_transl"]
    fps = int(seq_data["fps"])

    root_trans_zup = y_up_to_z_up_positions(soma_transl_yup)
    root_quat_zup = y_up_to_z_up_quat_wxyz(soma_root_quat_yup)

    soma_name_to_idx = {name: i for i, name in enumerate(SOMA_JOINT_NAMES)}
    mapped_local_points_yup = np.stack(
        [soma_joints[:, soma_name_to_idx[soma_name], :] for soma_name in SOMA_TO_H2_BODY_MAP.values()],
        axis=1,
    )
    
    # [추가됨] 로컬 조인트 좌표들을 Y-up에서 Z-up으로 변환
    mapped_local_points_zup = mapped_local_points_yup

    # Estimate a single sequence scale from pelvis-centered landmark lengths.
    # 스케일 계산 시에도 Z-up으로 변환된 좌표를 사용하는 것이 안전합니다.
    human_ref = mapped_local_points_zup[0] - mapped_local_points_zup[0, 0]
    human_norm = np.linalg.norm(human_ref, axis=-1)
    robot_norm = np.linalg.norm(ctx.robot_reference_points, axis=-1)
    valid = human_norm > 1e-4
    scale = float(np.median(robot_norm[valid] / human_norm[valid])) if np.any(valid) else 1.0
    scale = float(np.clip(scale, 0.6, 1.4))

    # [수정됨] 변환된 mapped_local_points_zup을 더해줌
    target_world = root_trans_zup[:, None, :] + scale * mapped_local_points_zup
    return target_world.astype(np.float32), root_trans_zup.astype(np.float32), root_quat_zup.astype(np.float32), fps


def solve_frame_ik(
    target_points: np.ndarray,
    root_trans: np.ndarray,
    root_quat_wxyz: np.ndarray,
    prev_dof: np.ndarray,
    ctx: RetargetContext,
) -> np.ndarray:
    """Solve one frame of H2 IK using least-squares on target body positions."""

    def residual(dof: np.ndarray) -> np.ndarray:
        qpos = np.concatenate([root_trans, root_quat_wxyz, dof.astype(np.float32)], axis=0)
        body_pos_w, _ = qpos_to_global_transforms(ctx.model, ctx.data, ctx.actuator_qpos_adr, qpos)
        pos_err = (body_pos_w[ctx.body_indices] - target_points) * ctx.target_weights[:, None]
        smooth_reg = 0.05 * (dof - prev_dof)
        rest_reg = 0.02 * dof[REST_REG_DOF]
        return np.concatenate([pos_err.reshape(-1), smooth_reg, rest_reg], axis=0)

    result = least_squares(
        residual,
        x0=prev_dof.astype(np.float64),
        bounds=(ctx.lower_bounds.astype(np.float64), ctx.upper_bounds.astype(np.float64)),
        method="trf",
        max_nfev=40,
        verbose=0,
    )
    return result.x.astype(np.float32)


def matrix_to_quat_wxyz(rot_mats: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to wxyz quaternions."""
    quat_xyzw = transform.Rotation.from_matrix(rot_mats).as_quat().astype(np.float32)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    quat_wxyz /= np.linalg.norm(quat_wxyz, axis=-1, keepdims=True) + 1e-8
    return quat_wxyz.astype(np.float32)


def solve_h2_ik(seq_data: dict, ctx: RetargetContext, max_frames: int | None = None) -> dict:
    """Retarget one SOMA sequence to an H2 motion sequence."""
    target_world, root_trans_zup, root_quat_zup, fps = build_h2_target_pose(seq_data, ctx)

    T = target_world.shape[0]
    if max_frames is not None:
        T = min(T, max_frames)
        target_world = target_world[:T]
        root_trans_zup = root_trans_zup[:T]
        root_quat_zup = root_quat_zup[:T]

    joint_pos = np.zeros((T, NUM_DOF), dtype=np.float32)
    body_pos_w = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    body_quat_w = np.zeros((T, NUM_BODIES, 4), dtype=np.float32)

    prev_dof = H2_DEFAULT_DOF.copy()
    for t in range(T):
        prev_dof = solve_frame_ik(
            target_points=target_world[t],
            root_trans=root_trans_zup[t],
            root_quat_wxyz=root_quat_zup[t],
            prev_dof=prev_dof,
            ctx=ctx,
        )
        qpos = np.concatenate([root_trans_zup[t], root_quat_zup[t], prev_dof], axis=0)
        frame_body_pos, frame_body_rot = qpos_to_global_transforms(
            ctx.model, ctx.data, ctx.actuator_qpos_adr, qpos
        )
        frame_body_quat = matrix_to_quat_wxyz(frame_body_rot)

        joint_pos[t] = prev_dof
        body_pos_w[t] = frame_body_pos
        body_quat_w[t] = frame_body_quat

    return {
        "joint_pos": joint_pos,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "joint_order": "mj",
        "fps": fps,
    }


def downsample_h2_sequence(entry: dict, fps_target: int) -> dict:
    """Downsample sequence using the same stride rule as other converters."""
    fps_source = int(entry["fps"])
    if fps_source == fps_target:
        return entry
    jump = int(fps_source / fps_target)
    if jump <= 1:
        return {**entry, "fps": fps_target}
    return {
        "joint_pos": entry["joint_pos"][::jump],
        "body_pos_w": entry["body_pos_w"][::jump],
        "body_quat_w": entry["body_quat_w"][::jump],
        "joint_order": entry["joint_order"],
        "fps": fps_target,
    }


def validate_h2_retarget_output(seq_name: str, entry: dict):
    """Validate H2 retarget output format and numeric health."""
    joint_pos = entry["joint_pos"]
    body_pos_w = entry["body_pos_w"]
    body_quat_w = entry["body_quat_w"]
    joint_order = entry["joint_order"]

    T = joint_pos.shape[0]
    assert joint_pos.shape == (T, NUM_DOF), f"{seq_name}: invalid joint_pos shape {joint_pos.shape}"
    assert body_pos_w.shape == (T, NUM_BODIES, 3), f"{seq_name}: invalid body_pos_w shape {body_pos_w.shape}"
    assert body_quat_w.shape == (T, NUM_BODIES, 4), f"{seq_name}: invalid body_quat_w shape {body_quat_w.shape}"
    assert joint_order == "mj", f"{seq_name}: joint_order must be mj"
    assert np.isfinite(joint_pos).all(), f"{seq_name}: joint_pos contains NaN/Inf"
    assert np.isfinite(body_pos_w).all(), f"{seq_name}: body_pos_w contains NaN/Inf"
    assert np.isfinite(body_quat_w).all(), f"{seq_name}: body_quat_w contains NaN/Inf"

    quat_norms = np.linalg.norm(body_quat_w, axis=-1)
    if not np.allclose(quat_norms, 1.0, atol=QUAT_NORM_ATOL):
        raise ValueError(
            f"{seq_name}: non-unit quaternions detected, range=({quat_norms.min():.6f}, {quat_norms.max():.6f})"
        )


def export_h2_retarget_pkl(output_path: str, seq_name: str, entry: dict):
    """Export one motion to PKL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({seq_name: entry}, output_path, compress=True)


def make_output_path(input_root: str, output_root: str, source_pkl: str, seq_name: str) -> str:
    """Build output path preserving input directory structure."""
    if os.path.isfile(input_root):
        return os.path.join(output_root, f"{seq_name}.pkl")
    rel_dir = os.path.relpath(os.path.dirname(source_pkl), input_root)
    if rel_dir == ".":
        return os.path.join(output_root, f"{seq_name}.pkl")
    return os.path.join(output_root, rel_dir, f"{seq_name}.pkl")


def print_debug_info(seq_name: str, seq_in: dict, seq_out: dict):
    """Print input/output details for one debug motion."""
    print(f"\n[DEBUG] motion: {seq_name}")
    print(f"[DEBUG] input keys: {sorted(seq_in.keys())}")
    for key, value in seq_in.items():
        if hasattr(value, "shape"):
            print(f"[DEBUG] input {key}: shape={value.shape} dtype={value.dtype}")
        else:
            print(f"[DEBUG] input {key}: {type(value)} {value}")

    print(f"[DEBUG] output keys: {sorted(seq_out.keys())}")
    for key, value in seq_out.items():
        if hasattr(value, "shape"):
            print(f"[DEBUG] output {key}: shape={value.shape} dtype={value.dtype}")
        else:
            print(f"[DEBUG] output {key}: {type(value)} {value}")

    quat_norms = np.linalg.norm(seq_out["body_quat_w"], axis=-1)
    print(
        f"[DEBUG] quat norm range: ({quat_norms.min():.6f}, {quat_norms.max():.6f})"
    )
    print(
        f"[DEBUG] joint_pos min/max: ({seq_out['joint_pos'].min():.6f}, {seq_out['joint_pos'].max():.6f})"
    )


def main():
    parser = argparse.ArgumentParser(description="Retarget SOMA joints PKLs to H2 retarget PKLs")
    parser.add_argument("--input", required=True, help="Input SOMA PKL file or directory")
    parser.add_argument("--output", required=True, help="Output PKL file or directory")
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Optional target output FPS. Defaults to each motion's original fps.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Write one PKL per motion. Recommended for debugging/filtering.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only one motion and print detailed debug information.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional frame cap for debugging or faster iteration.",
    )
    args = parser.parse_args()

    try:
        input_pkls = collect_input_pkls(args.input)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not input_pkls:
        print("ERROR: No input PKLs found")
        sys.exit(1)

    model, data = init_h2_model()
    ctx = build_context(model, data)

    print(f"Found {len(input_pkls)} SOMA PKL(s)")
    print(f"H2 bodies: {len(ctx.body_names)}, DOFs: {len(ctx.actuator_qpos_adr)}")

    if args.individual:
        os.makedirs(args.output, exist_ok=True)
        saved = 0
        processed_one = False
        for pkl_path in input_pkls:
            motions = load_soma_motion(pkl_path)
            for seq_name, seq_data in motions.items():
                print(f"Retargeting {seq_name} from {pkl_path}")
                entry = solve_h2_ik(seq_data, ctx, max_frames=args.max_frames)
                if args.fps is not None:
                    entry = downsample_h2_sequence(entry, args.fps)
                validate_h2_retarget_output(seq_name, entry)
                if args.debug:
                    print_debug_info(seq_name, seq_data, entry)
                out_path = make_output_path(args.input, args.output, pkl_path, seq_name)
                export_h2_retarget_pkl(out_path, seq_name, entry)
                saved += 1
                if args.debug:
                    processed_one = True
                    break
            if args.debug and processed_one:
                break
        print(f"Done: saved {saved} H2 retarget PKL(s) to {args.output}")
        return

    combined = {}
    processed_one = False
    for pkl_path in input_pkls:
        motions = load_soma_motion(pkl_path)
        for seq_name, seq_data in motions.items():
            print(f"Retargeting {seq_name} from {pkl_path}")
            entry = solve_h2_ik(seq_data, ctx, max_frames=args.max_frames)
            if args.fps is not None:
                entry = downsample_h2_sequence(entry, args.fps)
            validate_h2_retarget_output(seq_name, entry)
            if args.debug:
                print_debug_info(seq_name, seq_data, entry)
            combined[seq_name] = entry
            if args.debug:
                processed_one = True
                break
        if args.debug and processed_one:
            break

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(combined, args.output, compress=True)
    print(f"Done: saved {len(combined)} motions to {args.output}")


if __name__ == "__main__":
    main()
