"""Joint utility functions and constants for G1 robot.

This module provides joint ordering constants and helper functions for mapping
between motion library data and robot joints.
"""

import torch

from gear_sonic.utils import inspire_hand_spec

# G1 body joint names in IsaacLab order (29 DOF)
G1_ISAACLab_ORDER = [
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
]

# G1 hand joint names (12 DOF) - Inspire RH56, 6 drives per hand in SDK
# angle_set order (pinky, ring, middle, index, thumb_bend, thumb_rot).
G1_HAND_JOINTS = inspire_hand_spec.joint_names(is_left=True) + inspire_hand_spec.joint_names(
    is_left=False
)

# Caches for joint indices
_body_joint_indices_cache = {}
_hand_joint_indices_cache = {}


def _get_joint_indices_by_names(asset, joint_names: list, cache: dict) -> torch.Tensor:
    """Get indices of specified joints in the robot's joint list."""
    cache_key = (id(asset), tuple(joint_names))
    if cache_key in cache:
        return cache[cache_key]

    robot_joint_names = asset.joint_names
    indices = [robot_joint_names.index(n) for n in joint_names if n in robot_joint_names]
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=asset.device)
    cache[cache_key] = indices_tensor
    return indices_tensor


def get_body_joint_indices(asset) -> torch.Tensor:
    """Get indices of body joints (29 DOF) using G1_ISAACLab_ORDER."""
    return _get_joint_indices_by_names(asset, G1_ISAACLab_ORDER, _body_joint_indices_cache)


def get_hand_joint_indices(asset) -> torch.Tensor:
    """Get indices of hand joints (12 DOF) using G1_HAND_JOINTS."""
    return _get_joint_indices_by_names(asset, G1_HAND_JOINTS, _hand_joint_indices_cache)
