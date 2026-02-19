import time

import numpy as np
import pytest

from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.robot_model.robot_model import RobotModel
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK


@pytest.fixture(params=["lower_body", "lower_and_upper_body"])
def retargeting_ik(request):
    waist_location = request.param
    robot_model = instantiate_g1_robot_model(waist_location=waist_location)
    left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
    return TeleopRetargetingIK(
        robot_model=robot_model,
        left_hand_ik_solver=left_hand_ik_solver,
        right_hand_ik_solver=right_hand_ik_solver,
        enable_visualization=False,  # Change to true to visualize movements
        body_active_joint_groups=["upper_body"],
    )


def generate_target_wrist_poses(mode: str, side: str, full_robot: RobotModel) -> dict:
    """
    Args:
        mode: One of "rotation" or "translation"
        side: One of "left" or "right" - specifies which side to animate
    Returns:
        Dictionary mapping link names to target poses for both wrists
    """

    assert mode in ["rotation", "translation", "both"]
    assert side in ["left", "right", "both"]

    # Set up initial state
    full_robot.cache_forward_kinematics(full_robot.q_zero)

    # Get both wrist link names
    left_wrist_link = full_robot.supplemental_info.hand_frame_names["left"]
    right_wrist_link = full_robot.supplemental_info.hand_frame_names["right"]

    # Initialize default poses for both sides
    left_default_pose = full_robot.frame_placement(left_wrist_link).np
    right_default_pose = full_robot.frame_placement(right_wrist_link).np

    left_initial_pose_matrix = full_robot.frame_placement(left_wrist_link).np
    right_initial_pose_matrix = full_robot.frame_placement(right_wrist_link).np

    # Constants
    translation_cycle_duration = 4.0
    rotation_cycle_duration = 4.0
    total_duration = 12.0
    translation_amplitude = 0.1
    rotation_amplitude = np.deg2rad(60)  # 30 degrees

    body_data_list = []
    for t in np.linspace(0, total_duration, 100):
        rotation_matrix = np.eye(3)
        current_left_translation_vector = left_initial_pose_matrix[:3, 3].copy()
        current_right_translation_vector = right_initial_pose_matrix[:3, 3].copy()

        if mode == "rotation" or mode == "both":
            # For rotation-only mode, start rotating immediately
            rotation_axis_index = int(t // rotation_cycle_duration) % 3
            time_within_cycle = t % rotation_cycle_duration
            angle = rotation_amplitude * np.sin(
                (2 * np.pi / rotation_cycle_duration) * time_within_cycle
            )

            if rotation_axis_index == 0:  # Roll
                rotation_matrix = np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)],
                    ]
                )
            elif rotation_axis_index == 2:  # Pitch
                rotation_matrix = np.array(
                    [
                        [np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)],
                    ]
                )
            else:  # Yaw
                rotation_matrix = np.array(
                    [
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1],
                    ]
                )

        if mode == "translation" or mode == "both":
            translation_axis_index = int(t // translation_cycle_duration) % 3
            time_within_cycle = t % translation_cycle_duration
            offset = translation_amplitude * np.sin(
                (2 * np.pi / translation_cycle_duration) * time_within_cycle
            )
            current_left_translation_vector[translation_axis_index] += offset
            current_right_translation_vector[translation_axis_index] += offset

        # Construct the 4x4 pose matrix for the animated side
        left_animated_pose = np.eye(4)
        left_animated_pose[:3, :3] = rotation_matrix
        left_animated_pose[:3, 3] = current_left_translation_vector

        right_animated_pose = np.eye(4)
        right_animated_pose[:3, :3] = rotation_matrix
        right_animated_pose[:3, 3] = current_right_translation_vector

        # Create body_data dictionary with both wrists
        body_data = {}
        if side == "left":
            body_data[left_wrist_link] = left_animated_pose
            body_data[right_wrist_link] = right_default_pose
        elif side == "right":
            body_data[left_wrist_link] = left_default_pose
            body_data[right_wrist_link] = right_animated_pose
        elif side == "both":
            body_data[left_wrist_link] = left_animated_pose
            body_data[right_wrist_link] = right_animated_pose

        body_data_list.append(body_data)

    return body_data_list


@pytest.mark.parametrize("mode", ["translation", "rotation"])
@pytest.mark.parametrize("side", ["both", "left", "right"])
def test_ik_matches_fk(retargeting_ik, mode, side):
    full_robot = retargeting_ik.full_robot

    # Generate target wrist poses
    body_data_list = generate_target_wrist_poses(mode, side, full_robot)

    max_pos_error = 0
    max_rot_error = 0

    for body_data in body_data_list:

        time_start = time.time()

        # Run IK to get joint angles
        q = retargeting_ik.compute_joint_positions(
            body_data,
            left_hand_data=None,  # Hand IK not tested
            right_hand_data=None,  # Hand IK not tested
        )

        time_end = time.time()
        ik_time = time_end - time_start
        print(f"IK time: {ik_time} s")
        # Test commented out because of inconsistency in CI/CD computation time
        # assert ik_time < 0.05, f"IK time too high for 20Hz loop: {ik_time} s"

        # Run FK to compute where the wrists actually ended up
        full_robot.cache_forward_kinematics(q, auto_clip=False)
        left_wrist_link = full_robot.supplemental_info.hand_frame_names["left"]
        right_wrist_link = full_robot.supplemental_info.hand_frame_names["right"]
        T_fk_left = full_robot.frame_placement(left_wrist_link).np
        T_fk_right = full_robot.frame_placement(right_wrist_link).np
        T_target_left = body_data[left_wrist_link]
        T_target_right = body_data[right_wrist_link]

        # Check that FK translation matches target translation
        pos_fk_left = T_fk_left[:3, 3]
        pos_target_left = T_target_left[:3, 3]
        pos_fk_right = T_fk_right[:3, 3]
        pos_target_right = T_target_right[:3, 3]

        max_pos_error = max(max_pos_error, np.linalg.norm(pos_fk_left - pos_target_left))
        max_pos_error = max(max_pos_error, np.linalg.norm(pos_fk_right - pos_target_right))

        # Check that FK rotation matches target rotation
        rot_fk_left = T_fk_left[:3, :3]
        rot_target_left = T_target_left[:3, :3]
        rot_diff_left = rot_fk_left @ rot_target_left.T
        rot_error_left = np.arccos(np.clip((np.trace(rot_diff_left) - 1) / 2, -1, 1))
        rot_fk_right = T_fk_right[:3, :3]
        rot_target_right = T_target_right[:3, :3]
        rot_diff_right = rot_fk_right @ rot_target_right.T
        rot_error_right = np.arccos(np.clip((np.trace(rot_diff_right) - 1) / 2, -1, 1))

        max_rot_error = max(max_rot_error, rot_error_left)
        max_rot_error = max(max_rot_error, rot_error_right)

    assert max_pos_error < 0.01 and max_rot_error < np.deg2rad(
        1
    ), f"Max position error: {max_pos_error}, Max rotation error: {np.rad2deg(max_rot_error)} deg"
