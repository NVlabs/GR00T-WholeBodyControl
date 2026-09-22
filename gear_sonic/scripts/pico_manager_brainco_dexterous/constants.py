"""Constants and state enums for Pico/BrainCo teleoperation."""

from enum import Enum, IntEnum

import numpy as np
from scipy.spatial.transform import Rotation as sRot

class LocomotionMode(IntEnum):
    """Locomotion mode enum for robot movement."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19


class StreamMode(Enum):
    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5


### Parse 3 point pose from SMPL
#
# OFFSETS: Rotation corrections applied to each keypoint to align SMPL joint frames
# with the desired robot/visualization coordinate convention.
#
# Index mapping (based on [0, 22, 23, 12].index(joint_id)):
#   - OFFSETS[0]: Root/Pelvis (joint 0)
#   - OFFSETS[1]: Left Wrist (joint 22)
#   - OFFSETS[2]: Right Wrist (joint 23)
#   - OFFSETS[3]: Neck (joint 12) - more stable than Head (joint 15) for body tracking
#
# Scipy euler rotation convention:
#   - Lowercase "xyz" = EXTRINSIC rotations (about the FIXED/ORIGINAL frame's axes)
#   - Uppercase "XYZ" = INTRINSIC rotations (about the ROTATING body's axes)
#
# For EXTRINSIC "xyz" with angles [a, b, c]:
#   All rotations are about the ORIGINAL frame's axes (before any rotation):
#     R_total = R_z(c) @ R_y(b) @ R_x(a)   (matrix multiplication order)
#   Applied as: first rotate 'a' about original X, then 'b' about original Y, then 'c' about original Z
#
# For INTRINSIC "XYZ" with angles [a, b, c]:
#   Each rotation is about the CURRENT (rotated) frame's axis:
#     R_total = R_x(a) @ R_y(b) @ R_z(c)   (matrix multiplication order)
#   Applied as: first rotate 'a' about X, then 'b' about NEW Y, then 'c' about NEW Z
#
OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Root: yaw -90° about fixed Z
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),  # L-Wrist: roll +90° about fixed X
    sRot.from_euler(
        "xyz", [-90, 0, 180], degrees=True
    ),  # R-Wrist: roll -90° about fixed X, then yaw 180° about fixed Z
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Neck: yaw -90° about fixed Z
]

BRAINCO_NUM_MOTORS = 6
BRAINCO_LEFT_COMMAND_TOPIC = "rt/brainco/left/cmd"
BRAINCO_RIGHT_COMMAND_TOPIC = "rt/brainco/right/cmd"
BRAINCO_LEFT_STATE_TOPIC = "rt/brainco/left/state"
BRAINCO_RIGHT_STATE_TOPIC = "rt/brainco/right/state"
BRAINCO_TRIGGER_THRESHOLD = 0.5
BRAINCO_TRIGGER_RANGE_CHOICES = ("auto", "normal_0_to_1", "legacy_10_to_0")
BRAINCO_MOTOR_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
BRAINCO_EXCLUDED_FINGER_CHOICES = tuple(
    f"{side}_{motor_name}"
    for side in ("left", "right")
    for motor_name in BRAINCO_MOTOR_NAMES
)
BRAINCO_DEFAULT_EXCLUDED_FINGERS = ("right_thumb", "right_thumb_aux")

G1_UPPER_BODY_JOINTS = (
    (12, "waist_yaw"),
    (13, "waist_roll"),
    (14, "waist_pitch"),
    (15, "left_shoulder_pitch"),
    (16, "left_shoulder_roll"),
    (17, "left_shoulder_yaw"),
    (18, "left_elbow"),
    (19, "left_wrist_roll"),
    (20, "left_wrist_pitch"),
    (21, "left_wrist_yaw"),
    (22, "right_shoulder_pitch"),
    (23, "right_shoulder_roll"),
    (24, "right_shoulder_yaw"),
    (25, "right_elbow"),
    (26, "right_wrist_roll"),
    (27, "right_wrist_pitch"),
    (28, "right_wrist_yaw"),
)

# BrainCo command range is normalized: 0.0=open, 1.0=closed.
BRAINCO_OPEN_Q = np.zeros(BRAINCO_NUM_MOTORS, dtype=np.float32)
BRAINCO_CLOSED_Q = np.array([0.98, 0.70, 0.98, 0.98, 0.98, 0.98], dtype=np.float32)
BRAINCO_TAU_THRESHOLDS = np.array([0.7, 0.4, 0.7, 0.7, 0.7, 0.7], dtype=np.float32)


# Joystick deadzone threshold
JOYSTICK_DEADZONE = 0.15


