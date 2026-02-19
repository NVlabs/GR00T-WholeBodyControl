from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from robocasa.models.robots import remove_mimic_joints
from robosuite.models.robots import RobotModel as RobosuiteRobotModel

from decoupled_wbc.control.robot_model import RobotModel


class Gr00tJointInfo:
    """
    Mapping from decoupled_wbc actuated joint names to robocasa joint names.
    """

    def __init__(self, robot_model: RobosuiteRobotModel):
        self.robocasa_body_prefix = "robot0_"
        self.robocasa_gripper_prefix = "gripper0_"

        self.robot_model: RobotModel = robot_model
        self.body_actuated_joint_names: List[str] = (
            self.robot_model.supplemental_info.body_actuated_joints
        )
        self.left_hand_actuated_joint_names: List[str] = (
            self.robot_model.supplemental_info.left_hand_actuated_joints
        )
        self.right_hand_actuated_joint_names: List[str] = (
            self.robot_model.supplemental_info.right_hand_actuated_joints
        )

        self.actuated_joint_names: List[str] = self._get_gr00t_actuated_joint_names()
        self.body_actuated_joint_to_index: Dict[str, int] = (
            self._get_gr00t_body_actuated_joint_name_to_index()
        )
        self.gripper_actuated_joint_to_index: Tuple[Dict[str, int], Dict[str, int]] = (
            self._get_gr00t_gripper_actuated_joint_name_to_index()
        )
        self.actuated_joint_name_to_index: Dict[str, int] = (
            self._get_gr00t_actuated_joint_name_to_index()
        )

    def _get_gr00t_actuated_joint_names(self) -> List[str]:
        """Get list of gr00t actuated joint names ordered by their indices."""
        if self.robot_model.supplemental_info is None:
            raise ValueError("Robot model must have supplemental_info")

        # Get joint names and indices
        body_names = self.robot_model.supplemental_info.body_actuated_joints
        left_hand_names = self.robot_model.supplemental_info.left_hand_actuated_joints
        right_hand_names = self.robot_model.supplemental_info.right_hand_actuated_joints

        body_indices = self.robot_model.get_joint_group_indices("body")
        left_hand_indices = self.robot_model.get_joint_group_indices("left_hand")
        right_hand_indices = self.robot_model.get_joint_group_indices("right_hand")

        # Create a dictionary mapping index to name
        index_to_name = {}
        for name, idx in zip(body_names, body_indices):
            index_to_name[idx] = self.robocasa_body_prefix + name
        for name, idx in zip(left_hand_names, left_hand_indices):
            index_to_name[idx] = self.robocasa_gripper_prefix + "left_" + name
        for name, idx in zip(right_hand_names, right_hand_indices):
            index_to_name[idx] = self.robocasa_gripper_prefix + "right_" + name
        sorted_indices = sorted(index_to_name.keys())
        all_actuated_joint_names = [index_to_name[idx] for idx in sorted_indices]
        return all_actuated_joint_names

    def _get_gr00t_body_actuated_joint_name_to_index(self) -> Dict[str, int]:
        """Get dictionary mapping gr00t actuated joint names to indices."""
        if self.robot_model.supplemental_info is None:
            raise ValueError("Robot model must have supplemental_info")
        body_names = self.robot_model.supplemental_info.body_actuated_joints
        body_indices = self.robot_model.get_joint_group_indices("body")
        sorted_indices = np.argsort(body_indices)
        sorted_names = [body_names[i] for i in sorted_indices]
        return {self.robocasa_body_prefix + name: ii for ii, name in enumerate(sorted_names)}

    def _get_gr00t_gripper_actuated_joint_name_to_index(
        self,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Get dictionary mapping gr00t actuated joint names to indices."""
        if self.robot_model.supplemental_info is None:
            raise ValueError("Robot model must have supplemental_info")
        left_hand_names = self.robot_model.supplemental_info.left_hand_actuated_joints
        right_hand_names = self.robot_model.supplemental_info.right_hand_actuated_joints
        left_hand_indices = self.robot_model.get_joint_group_indices("left_hand")
        right_hand_indices = self.robot_model.get_joint_group_indices("right_hand")
        sorted_left_hand_indices = np.argsort(left_hand_indices)
        sorted_right_hand_indices = np.argsort(right_hand_indices)
        sorted_left_hand_names = [left_hand_names[i] for i in sorted_left_hand_indices]
        sorted_right_hand_names = [right_hand_names[i] for i in sorted_right_hand_indices]
        return (
            {
                self.robocasa_gripper_prefix + "left_" + name: ii
                for ii, name in enumerate(sorted_left_hand_names)
            },
            {
                self.robocasa_gripper_prefix + "right_" + name: ii
                for ii, name in enumerate(sorted_right_hand_names)
            },
        )

    def _get_gr00t_actuated_joint_name_to_index(self) -> Dict[str, int]:
        """Get dictionary mapping gr00t actuated joint names to indices."""
        return {name: ii for ii, name in enumerate(self.actuated_joint_names)}


@dataclass
class Gr00tObsActionConverter:
    """
    Converter to align simulation environment joint action space with real environment joint action space.
    Handles joint order and range conversion.
    """

    robot_model: RobotModel
    robosuite_robot_model: RobosuiteRobotModel
    robocasa_body_prefix: str = "robot0_"
    robocasa_gripper_prefix: str = "gripper0_"

    def __post_init__(self):
        """Initialize converter with robot configuration."""

        self.robot_key = self.robot_model.supplemental_info.name
        self.gr00t_joint_info = Gr00tJointInfo(self.robot_model)
        self.robocasa_joint_names_for_each_part: Dict[str, List[str]] = (
            self._get_robocasa_joint_names_for_each_part()
        )
        self.robocasa_actuator_names_for_each_part: Dict[str, List[str]] = (
            self._get_robotcasa_actuator_names_for_each_part()
        )

        # Store mappings directly as class attributes
        self.gr00t_joint_name_to_index = self.gr00t_joint_info.actuated_joint_name_to_index
        self.gr00t_body_joint_name_to_index = self.gr00t_joint_info.body_actuated_joint_to_index
        self.gr00t_gripper_joint_name_to_index = {
            "left": self.gr00t_joint_info.gripper_actuated_joint_to_index[0],
            "right": self.gr00t_joint_info.gripper_actuated_joint_to_index[1],
        }
        self.gr00t_to_robocasa_actuator_indices = self._get_actuator_mapping()

        if self.robot_key == "GR1_Fourier":
            self.joint_multiplier = (
                lambda x: np.array([-1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]) * x
            )
            self.actuator_multiplier = (
                lambda x: np.array([-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1]) * x
            )
        else:
            self.joint_multiplier = lambda x: x
            self.actuator_multiplier = lambda x: x

        # Store DOF counts directly
        self.body_dof = len(self.gr00t_joint_info.body_actuated_joint_names)
        self.gripper_dof = len(self.gr00t_joint_info.left_hand_actuated_joint_names) + len(
            self.gr00t_joint_info.right_hand_actuated_joint_names
        )
        self.whole_dof = self.body_dof + self.gripper_dof
        self.body_nu = len(self.gr00t_joint_info.body_actuated_joint_names)
        self.gripper_nu = len(self.gr00t_joint_info.left_hand_actuated_joint_names) + len(
            self.gr00t_joint_info.right_hand_actuated_joint_names
        )
        self.whole_nu = self.body_nu + self.gripper_nu

    def _get_robocasa_joint_names_for_each_part(self) -> Dict[str, List[str]]:
        part_names = self.robosuite_robot_model._ref_joints_indexes_dict.keys()
        robocasa_joint_names_for_each_part = {}
        for part_name in part_names:
            joint_indices = self.robosuite_robot_model._ref_joints_indexes_dict[part_name]
            joint_names = [
                self.robosuite_robot_model.sim.model.joint_id2name(j) for j in joint_indices
            ]
            robocasa_joint_names_for_each_part[part_name] = joint_names
        return robocasa_joint_names_for_each_part

    def _get_robotcasa_actuator_names_for_each_part(self) -> Dict[str, List[str]]:
        part_names = self.robosuite_robot_model._ref_actuators_indexes_dict.keys()
        robocasa_actuator_names_for_each_part = {}
        for part_name in part_names:
            if part_name == "base":
                continue
            actuator_indices = self.robosuite_robot_model._ref_actuators_indexes_dict[part_name]
            actuator_names = [
                self.robosuite_robot_model.sim.model.actuator_id2name(j) for j in actuator_indices
            ]
            robocasa_actuator_names_for_each_part[part_name] = actuator_names
        return robocasa_actuator_names_for_each_part

    def _get_actuator_mapping(self) -> Dict[str, List[int]]:
        """Get mapping from decoupled_wbc actuatored joint order to robocasa actuatored joint order for whole body."""
        return {
            part_name: [
                self.gr00t_joint_info.actuated_joint_name_to_index[j]
                for j in self.robocasa_actuator_names_for_each_part[part_name]
            ]
            for part_name in self.robocasa_actuator_names_for_each_part.keys()
        }

    def check_action_dim_match(self, vec_dim: int) -> bool:
        """
        Check if input vector dimension matches expected dimension.

        Args:
            vec_dim: Dimension of input vector

        Returns:
            bool: True if dimensions match
        """
        return vec_dim == self.whole_dof

    def gr00t_to_robocasa_action_dict(self, action_vec: np.ndarray) -> Dict[str, Any]:
        """
        Convert gr00t flat action vector to robocasa dictionary mapping part names to actions.

        Args:
            robot: Robocasa robot model instance
            action_vec: Full action vector array in gr00t actuated joint order

        Returns:
            dict: Mapping from part names to action vectors for robocasa
        """
        if not self.check_action_dim_match(len(action_vec)):
            raise ValueError(
                f"Action vector dimension mismatch: {len(action_vec)} != {self.whole_dof}"
            )

        action_dict = {}
        cc = self.robosuite_robot_model.composite_controller

        for part_name, controller in cc.part_controllers.items():
            if "gripper" in part_name:
                robocasa_action = action_vec[self.gr00t_to_robocasa_actuator_indices[part_name]]
                if self.actuator_multiplier is not None:
                    robocasa_action = self.actuator_multiplier(robocasa_action)
                action_dict[part_name] = remove_mimic_joints(
                    cc.grippers[part_name], robocasa_action
                )
            elif "base" in part_name:
                assert (
                    len(self.gr00t_to_robocasa_actuator_indices.get(part_name, [])) == 0
                    or self.robosuite_robot_model.default_base == "FloatingLeggedBase"
                )
            else:
                action_dict[part_name] = action_vec[
                    self.gr00t_to_robocasa_actuator_indices[part_name]
                ]

        return action_dict

    def robocasa_to_gr00t_actuated_order(
        self, joint_names: List[str], q: np.ndarray, obs_type: str = "body"
    ) -> np.ndarray:
        """
        Convert observation from robocasa joint order to gr00t actuated joint order.

        Args:
            joint_names: List of joint names in robocasa order (with prefixes)
            q: Joint positions corresponding to joint_names
            obs_type: Type of observation ("body", "left_gripper", "right_gripper", or "whole")

        Returns:
            Joint positions in gr00t actuated joint order
        """
        assert len(joint_names) == len(q), "Joint names and q must have the same length"

        if obs_type == "body":
            actuated_q = np.zeros(self.body_dof)
            for i, jn in enumerate(joint_names):
                actuated_q[self.gr00t_body_joint_name_to_index[jn]] = q[i]
        elif obs_type == "left_gripper":
            actuated_q = np.zeros(self.gripper_dof // 2)
            for i, jn in enumerate(joint_names):
                actuated_q[self.gr00t_gripper_joint_name_to_index["left"][jn]] = q[i]
        elif obs_type == "right_gripper":
            actuated_q = np.zeros(self.gripper_dof // 2)
            for i, jn in enumerate(joint_names):
                actuated_q[self.gr00t_gripper_joint_name_to_index["right"][jn]] = q[i]
        elif obs_type == "whole":
            actuated_q = np.zeros(self.whole_dof)
            for i, jn in enumerate(joint_names):
                actuated_q[self.gr00t_joint_name_to_index[jn]] = q[i]
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")
        return actuated_q

    def gr00t_to_robocasa_joint_order(
        self, joint_names: List[str], q_in_actuated_order: np.ndarray
    ) -> np.ndarray:
        """
        Convert gr00t actuated joint order to robocasa joint order.

        Args:
            joint_names: List of joint names in robocasa order (with prefixes)
            q_in_actuated_order: Joint positions corresponding to joint_names in gr00t actuated joint order

        Returns:
            Joint positions in robocasa joint order
        """
        q = np.zeros(len(joint_names))
        for i, jn in enumerate(joint_names):
            q[i] = q_in_actuated_order[self.gr00t_joint_name_to_index[jn]]
        return q
