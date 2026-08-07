from copy import deepcopy

import numpy as np

from decoupled_wbc.control.teleop.pre_processor.pre_processor import PreProcessor

RIGHT_HAND_ROTATION = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])


class WristsPreProcessor(PreProcessor):
    def __init__(self, motion_scale: float, **kwargs):
        super().__init__(**kwargs)
        self.motion_scale = motion_scale  # scale factor for robot motion
        self.calibration_ee_pose = {}  # poses to calibrate the robot
        self.ee_name_list = (
            []
        )  # name of the end-effector "link_head_pitch", "link_LArm7", "link_RArm7"
        self.robot_world_T_init_ee = {}  # initial end-effector pose of the robot
        self.init_teleop_T_teleop_world = {}  # initial transformation matrix
        self.init_teleop_T_init_ee = {}  # alignment of end effector and local teleop frames

        self.latest_data = None

    def calibrate(self, data, control_device):
        left_elbow_joint_name = self.robot.supplemental_info.joint_name_mapping["elbow_pitch"][
            "left"
        ]
        right_elbow_joint_name = self.robot.supplemental_info.joint_name_mapping["elbow_pitch"][
            "right"
        ]
        if "left_wrist" in data:
            self.ee_name_list.append(self.robot.supplemental_info.hand_frame_names["left"])
            self.calibration_ee_pose[left_elbow_joint_name] = (
                self.robot.supplemental_info.elbow_calibration_joint_angles["left"]
            )
        if "right_wrist" in data:
            self.ee_name_list.append(self.robot.supplemental_info.hand_frame_names["right"])
            self.calibration_ee_pose[right_elbow_joint_name] = (
                self.robot.supplemental_info.elbow_calibration_joint_angles["right"]
            )

        if self.calibration_ee_pose:
            q = deepcopy(self.robot.q_zero)
            # set pose
            for joint, degree in self.calibration_ee_pose.items():
                joint_idx = self.robot.joint_to_dof_index[joint]
                q[joint_idx] = np.deg2rad(degree)
            self.robot.cache_forward_kinematics(q, auto_clip=False)
            calibration_ee_poses = [
                self.robot.frame_placement(ee_name).np for ee_name in self.ee_name_list
            ]
            self.robot.reset_forward_kinematics()
        else:
            calibration_ee_poses = [
                self.robot.frame_placement(ee_name).np for ee_name in self.ee_name_list
            ]

        for ee_name in self.ee_name_list:
            self.robot_world_T_init_ee[ee_name] = deepcopy(
                calibration_ee_poses[self.ee_name_list.index(ee_name)]
            )
            self.init_teleop_T_teleop_world[ee_name] = (
                np.linalg.inv(deepcopy(data["left_wrist"]))
                if ee_name == self.robot.supplemental_info.hand_frame_names["left"]
                else np.linalg.inv(deepcopy(data["right_wrist"]))
            )

            # Initial teleop local frame is aligned with the initial end effector
            # local frame with hardcoded rotations since we don't have a common
            # reference frame for teleop and robot.
            self.init_teleop_T_init_ee[ee_name] = np.eye(4)
            if control_device == "pico":
                # TODO: add pico wrist calibration respect to the headset frame
                pass
            else:
                if ee_name == self.robot.supplemental_info.hand_frame_names["left"]:
                    self.init_teleop_T_init_ee[ee_name][
                        :3, :3
                    ] = self.robot.supplemental_info.hand_rotation_correction
                else:
                    self.init_teleop_T_init_ee[ee_name][:3, :3] = (
                        RIGHT_HAND_ROTATION @ self.robot.supplemental_info.hand_rotation_correction
                    )

    def __call__(self, data) -> dict:
        processed_data = {}
        for ee_name in self.ee_name_list:
            # Select wrist data based on ee_name
            teleop_world_T_final_teleop = (
                data["left_wrist"]
                if ee_name == self.robot.supplemental_info.hand_frame_names["left"]
                else data["right_wrist"]
            )
            init_teleop_T_final_teleop = (
                self.init_teleop_T_teleop_world[ee_name] @ teleop_world_T_final_teleop
            )
            # End effector differential transform is teleop differential transform expressed
            # in the end effector frame.
            init_ee_T_final_ee = (
                np.linalg.inv(self.init_teleop_T_init_ee[ee_name])
                @ init_teleop_T_final_teleop
                @ self.init_teleop_T_init_ee[ee_name]
            )
            # Translation scaling
            init_ee_T_final_ee[:3, 3] = self.motion_scale * init_ee_T_final_ee[:3, 3]
            robot_world_T_final_ee = self.robot_world_T_init_ee[ee_name] @ init_ee_T_final_ee
            processed_data[ee_name] = robot_world_T_final_ee

        self.latest_data = processed_data
        return processed_data
