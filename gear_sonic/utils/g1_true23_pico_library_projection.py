"""Project a true23 library motion into the exact SONIC V2 action image.

This module is simulator/dataset-only.  It never opens DDS or publishes robot
commands.  Root motion is preserved; joint targets are clipped only where the
bounded raw SONIC action cannot represent the source target.  All body-space
arrays are then recomputed from the pinned physical 23-DoF MuJoCo model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_sonic_library_replay import FPS, validate_library_motion
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

DT = 1.0 / FPS
BODY_COUNT = 24


def reachable_hardware_bounds(raw_abs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return hardware-order joint bounds reachable by native raw +/-raw_abs."""

    if type(raw_abs) is not float or not 0.0 < raw_abs < 10.0:
        raise ValueError("raw_abs must be a float within (0,10)")
    raw = np.full(23, np.float32(raw_abs), dtype=np.float32)
    _, positive = safe_target_transform_numpy(raw)
    _, negative = safe_target_transform_numpy(-raw)
    low = np.minimum(negative, positive).astype(np.float64)
    high = np.maximum(negative, positive).astype(np.float64)
    if low.shape != (23,) or high.shape != (23,) or np.any(low >= high):
        raise RuntimeError("true23 reachable joint bounds are invalid")
    return low, high


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


def project_library_motion_to_safe_image(
    *,
    repository_root: Path,
    motion: Mapping[str, np.ndarray],
    reachable_raw_abs: float = 9.0,
    root_mode: str = "preserve",
    minimum_frames: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Project one validated true23 motion and rebuild exact body kinematics."""

    frame_count = validate_library_motion(motion)
    stored_frames = frame_count if minimum_frames is None else int(minimum_frames)
    if stored_frames < frame_count:
        raise ValueError("minimum_frames cannot truncate source motion")
    if root_mode not in {"preserve", "canonical_upright_ankle"}:
        raise ValueError("root_mode must be preserve or canonical_upright_ankle")
    root = repository_root.resolve(strict=True)
    low, high = reachable_hardware_bounds(reachable_raw_abs)
    source_joint = np.asarray(motion["joint_pos"], dtype=np.float64)
    projected_joint = np.clip(source_joint, low, high)

    module, model, _ = prepare_true23_model(
        root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
    )
    if model.nq != 30 or model.nv != 29 or model.nu != 23 or model.nbody - 1 != BODY_COUNT:
        raise ValueError("pinned true23 MuJoCo model layout changed")

    source_root_position = np.asarray(motion["body_pos_w"][:, 0], dtype=np.float64)
    source_root_quaternion = np.asarray(motion["body_quat_w"][:, 0], dtype=np.float64)
    root_position = source_root_position.copy()
    root_quaternion = source_root_quaternion.copy()
    left_ankle = module.mj_name2id(model, module.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_ankle = module.mj_name2id(model, module.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    if min(left_ankle, right_ankle) < 1:
        raise ValueError("pinned true23 ankle bodies missing")
    data = module.MjData(model)
    body_pos = np.empty((frame_count, BODY_COUNT, 3), dtype=np.float64)
    body_quat = np.empty((frame_count, BODY_COUNT, 4), dtype=np.float64)
    for frame in range(frame_count):
        if root_mode == "canonical_upright_ankle":
            data.qpos[:3] = 0.0
            data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            data.qpos[7:] = projected_joint[frame]
            data.qvel[:] = 0.0
            module.mj_forward(model, data)
            height = 0.06 - min(float(data.xpos[left_ankle, 2]), float(data.xpos[right_ankle, 2]))
            if not 0.20 <= height <= 1.10:
                raise ValueError("canonical true23 root height outside physical envelope")
            root_position[frame] = (0.0, 0.0, height)
            root_quaternion[frame] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[:3] = root_position[frame]
        data.qpos[3:7] = root_quaternion[frame]
        data.qpos[7:] = projected_joint[frame]
        data.qvel[:] = 0.0
        module.mj_forward(model, data)
        body_pos[frame] = data.xpos[1:]
        body_quat[frame] = data.xquat[1:]

    arrays = {
        "fps": np.asarray([FPS], dtype=np.float64),
        "joint_pos": projected_joint.astype(np.float32),
        "joint_vel": _forward_difference(projected_joint).astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": _forward_difference(body_pos).astype(np.float32),
        "body_ang_vel_w": _body_angular_velocity(body_quat).astype(np.float32),
    }
    if stored_frames > frame_count:
        repeat = stored_frames - frame_count
        for name in ("joint_pos", "body_pos_w", "body_quat_w"):
            value = arrays[name]
            arrays[name] = np.concatenate((value, np.repeat(value[-1:], repeat, axis=0)), axis=0)
        for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            value = arrays[name]
            arrays[name] = np.concatenate(
                (value, np.zeros((repeat, *value.shape[1:]), dtype=value.dtype)),
                axis=0,
            )
    validate_library_motion(arrays)
    changed = np.abs(projected_joint - source_joint) > 1.0e-9
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "g1_true23_pico_sonic_safe_image_projection_v1",
        "physical_model": "g1_23dof_rev_1_0",
        "physical_dof": 23,
        "source_29dof_physics_used": False,
        "source_frame_count": frame_count,
        "stored_frame_count": stored_frames,
        "terminal_hold_frames": stored_frames - frame_count,
        "reachable_raw_abs": reachable_raw_abs,
        "root_mode": root_mode,
        "root_position_preserved": root_mode == "preserve",
        "root_orientation_preserved": root_mode == "preserve",
        "maximum_root_position_change_m": float(
            np.max(np.linalg.norm(root_position - source_root_position, axis=1))
        ),
        "maximum_root_orientation_component_change": float(
            np.max(np.abs(root_quaternion - source_root_quaternion))
        ),
        "body_kinematics_recomputed": True,
        "changed_frame_joint_pairs": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(np.any(changed, axis=1))),
        "maximum_absolute_joint_change_rad": float(np.max(np.abs(projected_joint - source_joint))),
        "changed_joint_names": [
            HARDWARE_23_JOINT_NAMES[index]
            for index in np.flatnonzero(np.any(changed, axis=0))
        ],
        "authorization": {
            "simulator_dataset_only": True,
            "hardware_authorized": False,
            "dds_opened": False,
            "robot_commands_published": False,
        },
    }
    return arrays, report


__all__ = ["project_library_motion_to_safe_image", "reachable_hardware_bounds"]
