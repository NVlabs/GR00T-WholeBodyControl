"""
Utility functions and constants for co-training with real robot.
"""

from robosuite.models.robots.manipulators.gr1_robot import GR1

# Initial pose configuration for co-training with real robot
COTRAIN_REAL_MATCHED_ROBOT_INITIAL_POSE = {
    GR1: {
        "robot0_r_shoulder_pitch": -0.22963779,
        "robot0_r_shoulder_roll": -0.38363408,
        "robot0_r_shoulder_yaw": 0.14360377,
        "robot0_r_elbow_pitch": -1.5289252,
        "robot0_r_wrist_yaw": -0.2897802,
        "robot0_r_wrist_roll": -0.07134621,
        "robot0_r_wrist_pitch": -0.04550289,
        "robot0_l_shoulder_pitch": -0.10933163,
        "robot0_l_shoulder_roll": 0.43292055,
        "robot0_l_shoulder_yaw": -0.15983289,
        "robot0_l_elbow_pitch": -1.48233023,
        "robot0_l_wrist_yaw": 0.2359135,
        "robot0_l_wrist_roll": 0.26184522,
        "robot0_l_wrist_pitch": 0.00830735,
    }
}
