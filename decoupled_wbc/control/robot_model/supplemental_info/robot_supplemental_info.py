from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np


@dataclass
class RobotSupplementalInfo:
    """
    Base class for robot-specific information that is not easily extractable from URDF.
    This includes information about actuated joints, joint hierarchies, etc.
    """

    name: str

    # List of body actuated joint names (excluding hands)
    body_actuated_joints: List[str]

    # List of left hand actuated joint names
    left_hand_actuated_joints: List[str]

    # List of right hand actuated joint names
    right_hand_actuated_joints: List[str]

    # Dictionary of joint groups, where each group is a dictionary with:
    # - "joints": list of joint names
    # - "groups": list of subgroup names (optional)
    # Example: {
    #   "right_arm": {
    #       "joints": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint"],
    #       "groups": []
    #   },
    #   "left_arm": {
    #       "joints": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint"],
    #       "groups": []
    #   },
    #   "upper_body": {
    #       "joints": ["torso_pitch_joint", "torso_yaw_joint", "torso_roll_joint"],
    #       "groups": ["right_arm", "left_arm"]
    #   }
    # }
    joint_groups: Dict[str, Dict[str, List[str]]]

    # Name of the root frame
    root_frame_name: str

    # Dictionary of hand frame names
    # Example: {
    #   "left": "left_hand_frame",
    #   "right": "right_hand_frame"
    # }
    hand_frame_names: Dict[str, str]

    # Dictionary of joint limits
    # Example: {
    #   "left_shoulder_pitch_joint": [-np.pi / 2, np.pi / 2],
    #   "right_shoulder_pitch_joint": [-np.pi / 2, np.pi / 2]
    # }
    joint_limits: Dict[str, List[float]]

    # Dictionary of elbow calibration joint angles
    # Example: {
    #   "left": -90.0,
    #   "right": -90.0
    # }
    elbow_calibration_joint_angles: Dict[str, float]

    # Dictionary of joint name mapping from generic types to robot-specific names
    # Example: {
    #   "waist_pitch": "waist_pitch_joint",
    #   "shoulder_pitch": {
    #       "left": "left_shoulder_pitch_joint",
    #       "right": "right_shoulder_pitch_joint"
    #   },
    #   "elbow_pitch": {
    #       "left": "left_elbow_pitch_joint",
    #       "right": "right_elbow_pitch_joint"
    #   }
    # }
    joint_name_mapping: Dict[str, Union[str, Dict[str, str]]]

    # Maps from generic joint names to robot-specific joint values
    # Example: {
    #   "waist_roll": 0.2,
    #   "elbow_pitch": {"left": 1.0, "right": 1.0}
    # }
    default_joint_q: Dict[str, Union[float, Dict[str, float]]]

    hand_rotation_correction: np.ndarray

    teleop_upper_body_motion_scale: float
