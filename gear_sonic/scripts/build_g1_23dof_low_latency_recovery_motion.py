"""Build an FK-consistent true23 stand/dance recovery reference.

This creates new simulator training material only.  It does not overwrite the
released dance sample, export a deployment policy, or access robot transport.
Every changed pose is replayed through the pinned G1-23 MuJoCo model so joint,
body, and velocity arrays describe one coherent kinematic trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gear_sonic.envs.mjlab.sonic_true23 import SONIC_HARDWARE_DEFAULT_Q

FPS = 50.0
DT = 1.0 / FPS
DEFAULT_XML = (
    Path(__file__).resolve().parents[2]
    / "external_dependencies"
    / "unitree_rl_mjlab"
    / "src"
    / "assets"
    / "robots"
    / "unitree_g1"
    / "xmls"
    / "g1_23dof.xml"
)
REQUIRED_KEYS = {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quintic(count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("transition needs at least two frames")
    phase = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)


def _wxyz_to_xyzw(value: np.ndarray) -> np.ndarray:
    return value[..., (1, 2, 3, 0)]


def _xyzw_to_wxyz(value: np.ndarray) -> np.ndarray:
    return value[..., (3, 0, 1, 2)]


def _slerp_wxyz(start: np.ndarray, stop: np.ndarray, blend: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_quat(
        _wxyz_to_xyzw(np.stack((start, stop), axis=0))
    )
    result = Slerp((0.0, 1.0), rotations)(blend).as_quat()
    return _xyzw_to_wxyz(result)


def _yaw_only_wxyz(quaternion: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(_wxyz_to_xyzw(quaternion))
    yaw = rotation.as_euler("xyz")[2]
    return _xyzw_to_wxyz(Rotation.from_euler("z", yaw).as_quat())


def _continuous_quaternions(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    for frame in range(1, result.shape[0]):
        flip = np.sum(result[frame - 1] * result[frame], axis=-1) < 0.0
        result[frame, flip] *= -1.0
    return result


def _gradient(value: np.ndarray) -> np.ndarray:
    return np.gradient(value, DT, axis=0, edge_order=2)


def _angular_velocity_w(quaternion_wxyz: np.ndarray) -> np.ndarray:
    frames, bodies, width = quaternion_wxyz.shape
    if frames < 3 or width != 4:
        raise ValueError("body quaternion trajectory must be [frames>=3,bodies,4]")
    rotations = Rotation.from_quat(
        _wxyz_to_xyzw(quaternion_wxyz.reshape(-1, 4))
    ).as_matrix().reshape(frames, bodies, 3, 3)
    result = np.empty((frames, bodies, 3), dtype=np.float64)
    relative = rotations[2:] @ np.swapaxes(rotations[:-2], -1, -2)
    result[1:-1] = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(
        frames - 2, bodies, 3
    ) / (2.0 * DT)
    first_relative = rotations[1] @ np.swapaxes(rotations[0], -1, -2)
    last_relative = rotations[-1] @ np.swapaxes(rotations[-2], -1, -2)
    result[0] = Rotation.from_matrix(first_relative).as_rotvec() / DT
    result[-1] = Rotation.from_matrix(last_relative).as_rotvec() / DT
    return result


def _forward_kinematics(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frames = joint_pos.shape[0]
    if model.nbody != 25 or model.nq != 30 or model.nv != 29:
        raise ValueError("recovery builder requires exact free-base G1-23 model")
    data = mujoco.MjData(model)
    body_pos = np.empty((frames, model.nbody - 1, 3), dtype=np.float64)
    body_quat = np.empty((frames, model.nbody - 1, 4), dtype=np.float64)
    for frame in range(frames):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[7:] = joint_pos[frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[1:]
        body_quat[frame] = data.xquat[1:]
    return body_pos, _continuous_quaternions(body_quat)


def build_recovery_motion(
    *,
    source_path: Path,
    output_path: Path,
    metadata_path: Path,
    xml_path: Path = DEFAULT_XML,
    stand_frames_each: int = 600,
    transition_in_frames: int = 150,
    transition_out_frames: int = 200,
    safe_margin_fraction: float = 0.025,
) -> dict[str, object]:
    source = source_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    metadata = metadata_path.expanduser().resolve()
    xml = xml_path.expanduser().resolve()
    for path, label in ((source, "source motion"), (xml, "G1-23 XML")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing: {path}")
    for path in (output, metadata):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite recovery material: {path}")
    if stand_frames_each < 500:
        raise ValueError("each neutral stand segment must cover one 10 s episode")
    if not 0.0 < safe_margin_fraction < 0.1:
        raise ValueError("safe_margin_fraction must be in (0, 0.1)")

    with np.load(source, allow_pickle=False) as data:
        if not REQUIRED_KEYS.issubset(data.files):
            raise ValueError("source motion lacks required true23 arrays")
        if float(data["fps"].reshape(-1)[0]) != FPS:
            raise ValueError("source motion must be exactly 50 Hz")
        source_joint = np.asarray(data["joint_pos"], dtype=np.float64)
        source_root_pos = np.asarray(data["body_pos_w"][:, 0], dtype=np.float64)
        source_root_quat = np.asarray(data["body_quat_w"][:, 0], dtype=np.float64)
    if source_joint.ndim != 2 or source_joint.shape[1] != 23:
        raise ValueError("source joint_pos must have shape [frames,23]")

    model = mujoco.MjModel.from_xml_path(str(xml))
    hard_limits = np.asarray(model.jnt_range[1:], dtype=np.float64)
    centers = hard_limits.mean(axis=1)
    half_ranges = (hard_limits[:, 1] - hard_limits[:, 0]) * 0.5
    soft_low = centers - 0.9 * half_ranges
    soft_high = centers + 0.9 * half_ranges
    soft_span = soft_high - soft_low
    safe_low = soft_low + safe_margin_fraction * soft_span
    safe_high = soft_high - safe_margin_fraction * soft_span
    safe_joint = np.clip(source_joint, safe_low, safe_high)
    clipped_mask = np.abs(safe_joint - source_joint) > 1.0e-9

    neutral_joint = np.asarray(SONIC_HARDWARE_DEFAULT_Q, dtype=np.float64)
    if np.any(neutral_joint < safe_low) or np.any(neutral_joint > safe_high):
        raise ValueError("SONIC neutral pose lies outside recovery safe limits")
    neutral_root_pos = source_root_pos[0].copy()
    neutral_root_pos[2] = 0.78
    neutral_root_quat = _yaw_only_wxyz(source_root_quat[0])

    stand_root_pos = np.repeat(neutral_root_pos[None], stand_frames_each, axis=0)
    stand_root_quat = np.repeat(neutral_root_quat[None], stand_frames_each, axis=0)
    stand_joint = np.repeat(neutral_joint[None], stand_frames_each, axis=0)

    blend_in = _quintic(transition_in_frames)
    in_root_pos = neutral_root_pos + blend_in[:, None] * (
        source_root_pos[0] - neutral_root_pos
    )
    in_root_quat = _slerp_wxyz(neutral_root_quat, source_root_quat[0], blend_in)
    in_joint = neutral_joint + blend_in[:, None] * (safe_joint[0] - neutral_joint)

    blend_out = _quintic(transition_out_frames)
    out_root_pos = source_root_pos[-1] + blend_out[:, None] * (
        neutral_root_pos - source_root_pos[-1]
    )
    out_root_quat = _slerp_wxyz(source_root_quat[-1], neutral_root_quat, blend_out)
    out_joint = safe_joint[-1] + blend_out[:, None] * (
        neutral_joint - safe_joint[-1]
    )

    root_pos = np.concatenate(
        (stand_root_pos, in_root_pos[1:], source_root_pos[1:], out_root_pos[1:], stand_root_pos),
        axis=0,
    )
    root_quat = _continuous_quaternions(
        np.concatenate(
            (
                stand_root_quat,
                in_root_quat[1:],
                source_root_quat[1:],
                out_root_quat[1:],
                stand_root_quat,
            ),
            axis=0,
        )[:, None, :]
    )[:, 0]
    joint_pos = np.concatenate(
        (stand_joint, in_joint[1:], safe_joint[1:], out_joint[1:], stand_joint),
        axis=0,
    )
    body_pos, body_quat = _forward_kinematics(
        model, root_pos, root_quat, joint_pos
    )
    joint_vel = _gradient(joint_pos)
    body_lin_vel = _gradient(body_pos)
    body_ang_vel = _angular_velocity_w(body_quat)

    arrays = {
        "fps": np.asarray([FPS], dtype=np.float64),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin_vel.astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
    }
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("recovery motion contains NaN or Inf")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)

    segments = {
        "neutral_pre": [0, stand_frames_each - 1],
        "transition_into_safe_dance": [
            stand_frames_each,
            stand_frames_each + transition_in_frames - 2,
        ],
        "safe_dance": [
            stand_frames_each + transition_in_frames - 2,
            stand_frames_each + transition_in_frames + source_joint.shape[0] - 3,
        ],
        "transition_to_neutral": [
            stand_frames_each + transition_in_frames + source_joint.shape[0] - 2,
            stand_frames_each
            + transition_in_frames
            + source_joint.shape[0]
            + transition_out_frames
            - 4,
        ],
        "neutral_post": [joint_pos.shape[0] - stand_frames_each, joint_pos.shape[0] - 1],
    }
    report: dict[str, object] = {
        "schema": "g1_true23_low_latency_recovery_motion_v1",
        "simulator_only": True,
        "deployment_ready": False,
        "source": {
            "filename": source.name,
            "sha256": _sha256(source),
            "frames": int(source_joint.shape[0]),
        },
        "model": {"filename": xml.name, "sha256": _sha256(xml)},
        "output": {
            "filename": output.name,
            "sha256": _sha256(output),
            "frames": int(joint_pos.shape[0]),
            "fps": FPS,
            "duration_s": float((joint_pos.shape[0] - 1) / FPS),
        },
        "segments_inclusive": segments,
        "safe_joint_projection": {
            "soft_limit_factor": 0.9,
            "inner_margin_fraction_of_soft_span": safe_margin_fraction,
            "changed_frame_joint_pairs": int(np.count_nonzero(clipped_mask)),
            "changed_frames": int(np.count_nonzero(np.any(clipped_mask, axis=1))),
            "max_abs_joint_change_rad": float(np.max(np.abs(safe_joint - source_joint))),
        },
        "kinematics": {
            "all_body_poses_recomputed_by_mujoco_fk": True,
            "joint_and_body_velocities_recomputed_at_50hz": True,
            "loop_starts_and_ends_at_same_neutral_pose": True,
        },
    }
    metadata.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--stand-frames-each", type=int, default=600)
    parser.add_argument("--transition-in-frames", type=int, default=150)
    parser.add_argument("--transition-out-frames", type=int, default=200)
    parser.add_argument("--safe-margin-fraction", type=float, default=0.025)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_recovery_motion(
        source_path=args.source,
        output_path=args.output,
        metadata_path=args.metadata,
        xml_path=args.xml,
        stand_frames_each=args.stand_frames_each,
        transition_in_frames=args.transition_in_frames,
        transition_out_frames=args.transition_out_frames,
        safe_margin_fraction=args.safe_margin_fraction,
    )
    print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
