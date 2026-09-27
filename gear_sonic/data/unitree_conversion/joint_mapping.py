"""Pure semantic joint-order mappings for Unitree G1 and Dex3 data."""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

import numpy as np

G1_MUJOCO_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_ISAACLAB_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# Absolute joint targets from the deployed policy, in G1_MUJOCO_NAMES order.
NOMINAL_G1_MUJOCO = np.array(
    (
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
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
    ),
    dtype=np.float64,
)
NOMINAL_G1_MUJOCO.setflags(write=False)

DEX3_DDS_NAMES = MappingProxyType(
    {
        "left": ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"),
        "right": ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"),
    }
)


def _duplicate_names(names: tuple[str, ...]) -> list[str]:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return sorted(name for name, count in counts.items() if count > 1)


def reorder_by_name(
    values: object,
    source_names: Sequence[str],
    target_names: Sequence[str],
) -> np.ndarray:
    """Reorder the final array dimension from source-name order to target order."""
    array = np.asarray(values)
    source = tuple(source_names)
    target = tuple(target_names)

    if array.ndim == 0 or array.shape[-1] != len(source):
        raise ValueError(f"values must have final dimension {len(source)}; got shape {array.shape}")

    duplicate_source = _duplicate_names(source)
    if duplicate_source:
        raise ValueError(f"duplicate source names: {duplicate_source}")
    duplicate_target = _duplicate_names(target)
    if duplicate_target:
        raise ValueError(f"duplicate target names: {duplicate_target}")

    source_set = set(source)
    target_set = set(target)
    if source_set != target_set:
        missing = sorted(target_set - source_set)
        extra = sorted(source_set - target_set)
        raise ValueError(
            f"source and target name sets differ: missing from source={missing}, extra in source={extra}"
        )

    source_indices = {name: index for index, name in enumerate(source)}
    indices = [source_indices[name] for name in target]
    return array[..., indices]


def dex3_dds_to_robot_model(side: str, raw: object) -> dict[str, object]:
    """Map one side's seven DDS values to semantic RobotModel joint names."""
    if not isinstance(side, str) or side not in DEX3_DDS_NAMES:
        raise ValueError(f"side must be exactly 'left' or 'right'; got {side!r}")

    values = np.asarray(raw)
    if values.shape != (7,):
        raise ValueError(f"raw must have shape (7,); got {values.shape}")

    return {
        f"{side}_hand_{semantic}_joint": value
        for semantic, value in zip(DEX3_DDS_NAMES[side], values, strict=True)
    }
