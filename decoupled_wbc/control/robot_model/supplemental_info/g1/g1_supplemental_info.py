from dataclasses import dataclass
from enum import Enum

import numpy as np

from decoupled_wbc.control.robot_model.supplemental_info.robot_supplemental_info import (
    RobotSupplementalInfo,
)


class WaistLocation(Enum):
    """Enum for waist location configuration."""

    LOWER_BODY = "lower_body"
    UPPER_BODY = "upper_body"
    LOWER_AND_UPPER_BODY = "lower_and_upper_body"


class ElbowPose(Enum):
    """Enum for elbow pose configuration."""

    LOW = "low"
    HIGH = "high"


@dataclass
class G1SupplementalInfo(RobotSupplementalInfo):
    """
    Supplemental information for the G1 robot.

    Args:
        waist_location: Where to place waist joints in the joint groups
        elbow_pose: Which elbow pose configuration to use for default joint positions
    """

    def __init__(
        self,
        waist_location: WaistLocation = WaistLocation.LOWER_BODY,
        elbow_pose: ElbowPose = ElbowPose.LOW,
        hand_type: str = "dex3",
    ):
        if hand_type not in ("dex3", "inspire"):
            raise ValueError(f"Unsupported hand_type: {hand_type}. Must be 'dex3' or 'inspire'.")
        self.hand_type = hand_type
        name = "G1_G1ThreeFinger" if hand_type == "dex3" else "G1_G1Inspire"

        # Define all actuated joints
        body_actuated_joints = [
            # Left leg
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            # Right leg
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            # Waist
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            # Left arm
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            # Right arm
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        if hand_type == "dex3":
            left_hand_actuated_joints = [
                # Left hand
                "left_hand_thumb_0_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "left_hand_index_0_joint",
                "left_hand_index_1_joint",
                "left_hand_middle_0_joint",
                "left_hand_middle_1_joint",
            ]

            right_hand_actuated_joints = [
                # Right hand
                "right_hand_thumb_0_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
                "right_hand_index_0_joint",
                "right_hand_index_1_joint",
                "right_hand_middle_0_joint",
                "right_hand_middle_1_joint",
            ]
        else:  # inspire: 6 DOF, ordered to match the Inspire SDK angle_set array
            # (pinky, ring, middle, index, thumb_bend, thumb_rot). Keep in sync
            # with decoupled_wbc/control/envs/g1/utils/inspire_hand_spec.py and
            # g1_29dof_with_inspire_hand.urdf.
            left_hand_actuated_joints = [
                "left_hand_pinky_joint",
                "left_hand_ring_joint",
                "left_hand_middle_joint",
                "left_hand_index_joint",
                "left_hand_thumb_bend_joint",
                "left_hand_thumb_rot_joint",
            ]

            right_hand_actuated_joints = [
                "right_hand_pinky_joint",
                "right_hand_ring_joint",
                "right_hand_middle_joint",
                "right_hand_index_joint",
                "right_hand_thumb_bend_joint",
                "right_hand_thumb_rot_joint",
            ]

        # Define joint limits from URDF
        joint_limits = {
            # Left leg
            "left_hip_pitch_joint": [-2.5307, 2.8798],
            "left_hip_roll_joint": [-0.5236, 2.9671],
            "left_hip_yaw_joint": [-2.7576, 2.7576],
            "left_knee_joint": [-0.087267, 2.8798],
            "left_ankle_pitch_joint": [-0.87267, 0.5236],
            "left_ankle_roll_joint": [-0.2618, 0.2618],
            # Right leg
            "right_hip_pitch_joint": [-2.5307, 2.8798],
            "right_hip_roll_joint": [-2.9671, 0.5236],
            "right_hip_yaw_joint": [-2.7576, 2.7576],
            "right_knee_joint": [-0.087267, 2.8798],
            "right_ankle_pitch_joint": [-0.87267, 0.5236],
            "right_ankle_roll_joint": [-0.2618, 0.2618],
            # Waist
            "waist_yaw_joint": [-2.618, 2.618],
            "waist_roll_joint": [-0.52, 0.52],
            "waist_pitch_joint": [-0.52, 0.52],
            # Left arm
            "left_shoulder_pitch_joint": [-3.0892, 2.6704],
            "left_shoulder_roll_joint": [0.19, 2.2515],
            "left_shoulder_yaw_joint": [-2.618, 2.618],
            "left_elbow_joint": [-1.0472, 2.0944],
            "left_wrist_roll_joint": [-1.972222054, 1.972222054],
            "left_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "left_wrist_yaw_joint": [-1.614429558, 1.614429558],
            # Right arm
            "right_shoulder_pitch_joint": [-3.0892, 2.6704],
            "right_shoulder_roll_joint": [-2.2515, -0.19],
            "right_shoulder_yaw_joint": [-2.618, 2.618],
            "right_elbow_joint": [-1.0472, 2.0944],
            "right_wrist_roll_joint": [-1.972222054, 1.972222054],
            "right_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "right_wrist_yaw_joint": [-1.614429558, 1.614429558],
        }

        # Hand joint limits depend on the hand type (must match the loaded URDF).
        if hand_type == "dex3":
            joint_limits.update(
                {
                    # Left hand
                    "left_hand_thumb_0_joint": [-1.04719755, 1.04719755],
                    "left_hand_thumb_1_joint": [-0.72431163, 1.04719755],
                    "left_hand_thumb_2_joint": [0, 1.74532925],
                    "left_hand_index_0_joint": [-1.57079632, 0],
                    "left_hand_index_1_joint": [-1.74532925, 0],
                    "left_hand_middle_0_joint": [-1.57079632, 0],
                    "left_hand_middle_1_joint": [-1.74532925, 0],
                    # Right hand
                    "right_hand_thumb_0_joint": [-1.04719755, 1.04719755],
                    "right_hand_thumb_1_joint": [-0.72431163, 1.04719755],
                    "right_hand_thumb_2_joint": [0, 1.74532925],
                    "right_hand_index_0_joint": [-1.57079632, 0],
                    "right_hand_index_1_joint": [-1.74532925, 0],
                    "right_hand_middle_0_joint": [-1.57079632, 0],
                    "right_hand_middle_1_joint": [-1.74532925, 0],
                }
            )
        else:  # inspire
            # Limits are the physical joint travel from the Inspire FTP manual
            # (fingers 20-176deg, thumb bend -13-70deg, thumb rot 90-165deg), with
            # 0 = open. Keep in sync with inspire_hand_spec.Q_CLOSED and the URDF.
            joint_limits.update(
                {
                    # Left hand
                    "left_hand_pinky_joint": [0.0, 2.72],
                    "left_hand_ring_joint": [0.0, 2.72],
                    "left_hand_middle_joint": [0.0, 2.72],
                    "left_hand_index_joint": [0.0, 2.72],
                    "left_hand_thumb_bend_joint": [0.0, 1.45],
                    "left_hand_thumb_rot_joint": [0.0, 1.31],
                    # Right hand
                    "right_hand_pinky_joint": [0.0, 2.72],
                    "right_hand_ring_joint": [0.0, 2.72],
                    "right_hand_middle_joint": [0.0, 2.72],
                    "right_hand_index_joint": [0.0, 2.72],
                    "right_hand_thumb_bend_joint": [0.0, 1.45],
                    "right_hand_thumb_rot_joint": [0.0, 1.31],
                }
            )

        # Define joint groups
        joint_groups = {
            # Body groups
            "waist": {
                "joints": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                "groups": [],
            },
            # Leg groups
            "left_leg": {
                "joints": [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                ],
                "groups": [],
            },
            "right_leg": {
                "joints": [
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                ],
                "groups": [],
            },
            "legs": {"joints": [], "groups": ["left_leg", "right_leg"]},
            # Arm groups
            "left_arm": {
                "joints": [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "right_arm": {
                "joints": [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "arms": {"joints": [], "groups": ["left_arm", "right_arm"]},
            # Hand groups (populated below based on hand_type)
            "left_hand": {"joints": list(left_hand_actuated_joints), "groups": []},
            "right_hand": {"joints": list(right_hand_actuated_joints), "groups": []},
            "hands": {"joints": [], "groups": ["left_hand", "right_hand"]},
            # Full body groups
            "lower_body": {"joints": [], "groups": ["waist", "legs"]},
            "upper_body_no_hands": {"joints": [], "groups": ["arms"]},
            "body": {"joints": [], "groups": ["lower_body", "upper_body_no_hands"]},
            "upper_body": {"joints": [], "groups": ["upper_body_no_hands", "hands"]},
        }

        # Define joint name mapping from generic types to robot-specific names
        joint_name_mapping = {
            # Waist joints
            "waist_pitch": "waist_pitch_joint",
            "waist_roll": "waist_roll_joint",
            "waist_yaw": "waist_yaw_joint",
            # Shoulder joints
            "shoulder_pitch": {
                "left": "left_shoulder_pitch_joint",
                "right": "right_shoulder_pitch_joint",
            },
            "shoulder_roll": {
                "left": "left_shoulder_roll_joint",
                "right": "right_shoulder_roll_joint",
            },
            "shoulder_yaw": {
                "left": "left_shoulder_yaw_joint",
                "right": "right_shoulder_yaw_joint",
            },
            # Elbow joints
            "elbow_pitch": {"left": "left_elbow_joint", "right": "right_elbow_joint"},
            # Wrist joints
            "wrist_pitch": {"left": "left_wrist_pitch_joint", "right": "right_wrist_pitch_joint"},
            "wrist_roll": {"left": "left_wrist_roll_joint", "right": "right_wrist_roll_joint"},
            "wrist_yaw": {"left": "left_wrist_yaw_joint", "right": "right_wrist_yaw_joint"},
        }

        root_frame_name = "pelvis"

        hand_frame_names = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

        elbow_calibration_joint_angles = {"left": 0.0, "right": 0.0}

        hand_rotation_correction = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        # Configure default joint positions based on elbow pose
        if elbow_pose == ElbowPose.HIGH:
            default_joint_q = {
                "shoulder_roll": {"left": 0.5, "right": -0.5},
                "shoulder_pitch": {"left": -0.2, "right": -0.2},
                "shoulder_yaw": {"left": -0.5, "right": 0.5},
                "wrist_roll": {"left": -0.5, "right": 0.5},
                "wrist_yaw": {"left": 0.5, "right": -0.5},
                "wrist_pitch": {"left": -0.2, "right": -0.2},
            }
        else:  # ElbowPose.LOW
            default_joint_q = {
                "shoulder_roll": {"left": 0.2, "right": -0.2},
            }

        teleop_upper_body_motion_scale = 1.0

        # Configure joint groups based on waist location
        modified_joint_groups = joint_groups.copy()
        if waist_location == WaistLocation.UPPER_BODY:
            # Move waist from lower_body to upper_body_no_hands
            modified_joint_groups["lower_body"] = {"joints": [], "groups": ["legs"]}
            modified_joint_groups["upper_body_no_hands"] = {
                "joints": [],
                "groups": ["arms", "waist"],
            }
        elif waist_location == WaistLocation.LOWER_AND_UPPER_BODY:
            # Add waist to upper_body_no_hands while keeping it in lower_body
            modified_joint_groups["upper_body_no_hands"] = {
                "joints": [],
                "groups": ["arms", "waist"],
            }
        # For LOWER_BODY, keep default joint_groups as is

        super().__init__(
            name=name,
            body_actuated_joints=body_actuated_joints,
            left_hand_actuated_joints=left_hand_actuated_joints,
            right_hand_actuated_joints=right_hand_actuated_joints,
            joint_limits=joint_limits,
            joint_groups=modified_joint_groups,
            root_frame_name=root_frame_name,
            hand_frame_names=hand_frame_names,
            elbow_calibration_joint_angles=elbow_calibration_joint_angles,
            joint_name_mapping=joint_name_mapping,
            hand_rotation_correction=hand_rotation_correction,
            default_joint_q=default_joint_q,
            teleop_upper_body_motion_scale=teleop_upper_body_motion_scale,
        )
