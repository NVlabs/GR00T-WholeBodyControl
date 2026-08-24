"""Build native true23 full-body motion arrays from causal PICO packets.

The causal packet already contains exact q0..q10 lower-body history and a
contiguous native-23 q9/q10 proof.  This module preserves those samples,
reconstructs a complete joint trajectory, and uses the pinned rev-1.0 MuJoCo
model for body FK.  Root height follows the lower ankle so crouching remains a
real full-body crouch instead of being hidden behind a standing balance policy.

No transport, robot, DDS, or actuation path exists here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_contract import (
    CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES,
    HARDWARE_23_JOINT_NAMES,
    ISAACLAB_TO_MUJOCO_DOF,
    NATIVE_IL23_TO_CANONICAL_IL29,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_RAW_ACTION_CLIP,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import validate_reference_terms
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

FPS = 50.0
DT = 1.0 / FPS
BODY_COUNT = 24
TARGET_ANKLE_BODY_HEIGHT_M = 0.06
DEFAULT_MINIMUM_FRAMES = 1024
PACKET_BUNDLE_KEYS = {"robot_independent_reference_packets", "semantic_packets"}

NATIVE_TO_HARDWARE = np.asarray(ISAACLAB_TO_MUJOCO_DOF, dtype=np.int64)
NATIVE_TO_IL29 = np.asarray(NATIVE_IL23_TO_CANONICAL_IL29, dtype=np.int64)
LOWER_BODY_IL29 = np.asarray(CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES, dtype=np.int64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _packet_list(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if set(bundle) != PACKET_BUNDLE_KEYS:
        raise ValueError("PICO packet bundle keys mismatch")
    packets = bundle["robot_independent_reference_packets"]
    semantic = bundle["semantic_packets"]
    if (
        not isinstance(packets, list)
        or len(packets) < 2
        or not isinstance(semantic, list)
        or len(semantic) != len(packets)
        or any(not isinstance(packet, Mapping) for packet in packets)
    ):
        raise ValueError("PICO packet bundle rows mismatch")
    return packets


def _complete_native_joint_path(packets: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Reconstruct q0..qN+9 while preserving every packet q9/q10 proof."""

    summaries = [validate_reference_terms(packet) for packet in packets]
    for previous, current in zip(summaries[:-1], summaries[1:], strict=True):
        if (
            current["anchor_index"] != previous["anchor_index"] + 1
            or current["anchor_monotonic_ns"] != previous["anchor_monotonic_ns"] + 20_000_000
        ):
            raise ValueError("PICO packet sequence is not contiguous 50 Hz")

    first = packets[0]
    anchor_il29 = np.asarray(first["anchor_joint_pos_il29"], dtype=np.float64)
    history = np.asarray(first["causal_history_lower_body"], dtype=np.float64)[:120].reshape(10, 12)
    il29_prefix = np.repeat(anchor_il29[None], 10, axis=0)
    il29_prefix[:, LOWER_BODY_IL29] = history
    prefix_native = il29_prefix[:, NATIVE_TO_IL29]

    anchors = np.asarray([packet["q_ref23_native"] for packet in packets], dtype=np.float64)
    velocities = np.asarray([packet["qd_ref23_native"] for packet in packets], dtype=np.float64)
    if anchors.shape != (len(packets), 23) or velocities.shape != anchors.shape:
        raise ValueError("PICO native joint packet shape mismatch")
    proof = anchors + velocities * DT
    if not np.array_equal(proof[:-1], anchors[1:]):
        raise ValueError("PICO packet q9/q10 proof differs from next anchor")
    if not np.allclose(prefix_native[-1], anchors[0], rtol=0.0, atol=1.0e-9):
        raise ValueError("PICO history anchor differs from native q9")

    result = np.concatenate((prefix_native, anchors[1:], proof[-1:]), axis=0)
    if result.shape != (len(packets) + 10, 23) or not np.isfinite(result).all():
        raise ValueError("PICO reconstructed native trajectory is invalid")
    for index, packet in enumerate(packets):
        q9 = result[9 + index]
        q10 = result[10 + index]
        if not np.array_equal(q9, np.asarray(packet["q_ref23_native"], dtype=np.float64)):
            raise ValueError("PICO reconstructed q9 drift")
        expected_q10 = q9 + np.asarray(packet["qd_ref23_native"], dtype=np.float64) * DT
        if not np.array_equal(q10, expected_q10):
            raise ValueError("PICO reconstructed q10 drift")
    return result


def _forward_difference(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    result[:-1] = np.diff(values, axis=0) / DT
    result[-1] = result[-2]
    return result


def _body_angular_velocity(quaternions_wxyz: np.ndarray) -> np.ndarray:
    frames, bodies, width = quaternions_wxyz.shape
    if width != 4:
        raise ValueError("body quaternion width mismatch")
    result = np.zeros((frames, bodies, 3), dtype=np.float64)
    for body in range(bodies):
        rotations = Rotation.from_quat(quaternions_wxyz[:, body, [1, 2, 3, 0]])
        relative = rotations[1:] * rotations[:-1].inv()
        result[:-1, body] = relative.as_rotvec() / DT
        result[-1, body] = result[-2, body]
    return result


def _terminal_hold(arrays: Mapping[str, np.ndarray], minimum_frames: int) -> dict[str, np.ndarray]:
    frames = int(arrays["joint_pos"].shape[0])
    if minimum_frames < frames:
        raise ValueError("minimum frames cannot truncate a PICO motion")
    result = {name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()}
    if minimum_frames == frames:
        return result
    repeat = minimum_frames - frames
    pose_names = {"joint_pos", "body_pos_w", "body_quat_w"}
    for name, value in tuple(result.items()):
        tail = (
            np.repeat(value[-1:], repeat, axis=0)
            if name in pose_names
            else np.zeros((repeat, *value.shape[1:]), dtype=np.float32)
        )
        result[name] = np.concatenate((value, tail), axis=0)
    return result


def build_pico_fullbody_motion(
    *,
    repository_root: Path,
    packet_bundle: Mapping[str, Any],
    source_path: Path | None = None,
    minimum_frames: int = DEFAULT_MINIMUM_FRAMES,
    reachable_raw_abs: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return a full-body true23 motion and non-promotional build report."""

    root = repository_root.resolve(strict=True)
    packets = _packet_list(packet_bundle)
    native = _complete_native_joint_path(packets)
    hardware = native[:, NATIVE_TO_HARDWARE]
    original_hardware = hardware.copy()
    if reachable_raw_abs is not None:
        if not 0.0 < reachable_raw_abs < SAFE_TARGET_RAW_ACTION_CLIP:
            raise ValueError("reachable raw bound must be within (0,10)")
        raw = np.full(23, np.float32(reachable_raw_abs), dtype=np.float32)
        _, positive_target = safe_target_transform_numpy(raw)
        _, negative_target = safe_target_transform_numpy(-raw)
        reachable_low = np.minimum(negative_target, positive_target).astype(np.float64)
        reachable_high = np.maximum(negative_target, positive_target).astype(np.float64)
        hardware = np.clip(hardware, reachable_low, reachable_high)

    module, model, _ = prepare_true23_model(
        root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
    )
    if model.nq != 30 or model.nv != 29 or model.nbody - 1 != BODY_COUNT:
        raise ValueError("pinned true23 MuJoCo model layout changed")
    joint_ranges = np.asarray(model.jnt_range[1:], dtype=np.float64)
    if (
        joint_ranges.shape != (23, 2)
        or np.any(hardware < joint_ranges[:, 0] - 1.0e-6)
        or np.any(hardware > joint_ranges[:, 1] + 1.0e-6)
    ):
        raise ValueError("PICO full-body reference exceeds true23 joint range")

    left_ankle = module.mj_name2id(model, module.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_ankle = module.mj_name2id(model, module.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    if left_ankle < 1 or right_ankle < 1:
        raise ValueError("pinned true23 ankle bodies missing")

    data = module.MjData(model)
    body_pos = np.empty((len(hardware), BODY_COUNT, 3), dtype=np.float64)
    body_quat = np.empty((len(hardware), BODY_COUNT, 4), dtype=np.float64)
    root_height = np.empty(len(hardware), dtype=np.float64)
    for frame, joints in enumerate(hardware):
        data.qpos[:3] = (0.0, 0.0, 0.0)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[7:] = joints
        module.mj_forward(model, data)
        ankle_at_zero = min(float(data.xpos[left_ankle, 2]), float(data.xpos[right_ankle, 2]))
        height = TARGET_ANKLE_BODY_HEIGHT_M - ankle_at_zero
        if not 0.20 <= height <= 0.95:
            raise ValueError(f"PICO kinematic root height outside full-body envelope: {height:.6f}")
        root_height[frame] = height
        data.qpos[2] = height
        module.mj_forward(model, data)
        body_pos[frame] = data.xpos[1:]
        body_quat[frame] = data.xquat[1:]

    raw_arrays = {
        "joint_pos": hardware.astype(np.float32),
        "joint_vel": _forward_difference(hardware).astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": _forward_difference(body_pos).astype(np.float32),
        "body_ang_vel_w": _body_angular_velocity(body_quat).astype(np.float32),
    }
    arrays = _terminal_hold(raw_arrays, minimum_frames)
    arrays = {"fps": np.asarray([FPS], dtype=np.float64), **arrays}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("PICO full-body motion contains NaN or Inf")

    report = {
        "schema_version": 1,
        "kind": "g1_true23_pico_fullbody_kinematic_motion_v1",
        "source_packet_count": len(packets),
        "source_motion_frames": len(hardware),
        "stored_frames": minimum_frames,
        "terminal_hold_frames": minimum_frames - len(hardware),
        "fps": FPS,
        "physical_dof": 23,
        "source_29dof_physics_used": False,
        "root_orientation": "upright_identity",
        "root_height_rule": "minimum_ankle_roll_body_z_equals_0p06m",
        "minimum_root_height_m": float(np.min(root_height)),
        "maximum_root_height_m": float(np.max(root_height)),
        "maximum_absolute_joint_position_rad": float(np.max(np.abs(hardware))),
        "maximum_absolute_joint_velocity_rad_s": float(np.max(np.abs(raw_arrays["joint_vel"]))),
        "reference_projection": {
            "enabled": reachable_raw_abs is not None,
            "source_packet_values_preserved": reachable_raw_abs is None,
            "reachable_raw_abs": reachable_raw_abs,
            "changed_frame_joint_pairs": int(np.count_nonzero(np.abs(hardware - original_hardware) > 1.0e-9)),
            "changed_frames": int(np.count_nonzero(np.any(np.abs(hardware - original_hardware) > 1.0e-9, axis=1))),
            "maximum_absolute_change_rad": float(np.max(np.abs(hardware - original_hardware))),
            "changed_joint_names": [
                HARDWARE_23_JOINT_NAMES[index]
                for index in np.flatnonzero(np.any(np.abs(hardware - original_hardware) > 1.0e-9, axis=0))
            ],
            "reason": (
                None
                if reachable_raw_abs is None
                else "project only unreachable packet targets into exact native23 safe-action image"
            ),
        },
        "source_path": None if source_path is None else str(source_path.resolve()),
        "source_sha256": None if source_path is None else sha256_file(source_path.resolve(strict=True)),
        "authorization": {
            "simulator_dataset_only": True,
            "training_input_candidate": True,
            "deployment_ready": False,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }
    return arrays, report


def load_and_build_pico_fullbody_motion(
    *,
    repository_root: Path,
    packet_path: Path,
    minimum_frames: int = DEFAULT_MINIMUM_FRAMES,
    reachable_raw_abs: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = packet_path.resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("PICO packet bundle must be a JSON object")
    return build_pico_fullbody_motion(
        repository_root=repository_root,
        packet_bundle=payload,
        source_path=path,
        minimum_frames=minimum_frames,
        reachable_raw_abs=reachable_raw_abs,
    )


__all__ = (
    "DEFAULT_MINIMUM_FRAMES",
    "FPS",
    "TARGET_ANKLE_BODY_HEIGHT_M",
    "build_pico_fullbody_motion",
    "load_and_build_pico_fullbody_motion",
    "sha256_file",
)
