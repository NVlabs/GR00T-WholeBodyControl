import time
from typing import List, Optional

import numpy as np

from decoupled_wbc.control.base.policy import Policy
from decoupled_wbc.control.robot_model.robot_model import ReducedRobotModel, RobotModel
from decoupled_wbc.control.teleop.solver.body.body_ik_solver import BodyIKSolver
from decoupled_wbc.control.teleop.solver.body.body_ik_solver_settings import BodyIKSolverSettings
from decoupled_wbc.control.teleop.solver.solver import Solver
from decoupled_wbc.control.visualization.humanoid_visualizer import RobotVisualizer


class TeleopRetargetingIK(Policy):
    """
    Robot-agnostic teleop retargeting inverse kinematics code.
    Focus only on IK processing, ignore commands.
    """

    def __init__(
        self,
        robot_model: RobotModel,
        left_hand_ik_solver: Solver,
        right_hand_ik_solver: Solver,
        enable_visualization=False,
        body_active_joint_groups: Optional[List[str]] = None,
        body_ik_solver_settings_type: str = "default",
    ):
        # initialize the body
        if body_active_joint_groups is not None:
            self.body = ReducedRobotModel.from_active_groups(robot_model, body_active_joint_groups)
            self.full_robot = self.body.full_robot
            self.using_reduced_robot_model = True
        else:
            self.body = robot_model
            self.full_robot = self.body
            self.using_reduced_robot_model = False
        if body_ik_solver_settings_type == "default":
            body_ik_solver_settings = BodyIKSolverSettings()
        else:
            raise ValueError(
                f"Unknown body_ik_solver_settings_type: {body_ik_solver_settings_type}"
            )
        self.body_ik_solver = BodyIKSolver(body_ik_solver_settings)

        # We register the specific robot model to the robot-agnostic body IK solver class
        self.body_ik_solver.register_robot(self.body)

        # Hand IK solvers are hand specific, so we pass them in the constructor
        self.left_hand_ik_solver = left_hand_ik_solver
        self.right_hand_ik_solver = right_hand_ik_solver

        # enable visualizer
        self.enable_visualization = enable_visualization
        if self.enable_visualization:
            self.visualizer = RobotVisualizer(self.full_robot)
            self.visualizer.visualize(self.full_robot.q_zero)
            time.sleep(1)  # wait for the visualizer to start

        self.in_warmup = True
        self._most_recent_ik_data = None
        self._most_recent_q = self.full_robot.default_body_pose.copy()

    def compute_joint_positions(
        self, body_data: dict, left_hand_data: dict, right_hand_data: dict
    ) -> np.ndarray:
        """Process only IK-related data, return joint positions"""
        if self.in_warmup:
            # TODO: Warmup is not necessary if we start IK from the current robot qpos, rather than the zero qpos
            for _ in range(50):
                target_robot_joints = self._inverse_kinematics(
                    body_data, left_hand_data, right_hand_data
                )
            self.in_warmup = False
        else:
            target_robot_joints = self._inverse_kinematics(
                body_data, left_hand_data, right_hand_data
            )

        return target_robot_joints

    def _inverse_kinematics(
        self,
        body_target_pose,
        left_hand_target_pose,
        right_hand_target_pose,
    ):
        """
        Solve the inverse kinematics problem for the given target poses.
        Args:
            body_target_pose: Dictionary of link names and their corresponding target pose.
            left_hand_target_pose: Dictionary with key "position" mapping to a (25, 4, 4) np.ndarray from AVP data
            right_hand_target_pose: Dictionary with key "position" mapping to a (25, 4, 4) np.ndarray from AVP data
            q: Initial configuration vector.
        Returns:
            Configuration vector that achieves the target poses.
        """
        if body_target_pose:
            if self.using_reduced_robot_model:
                body_q = self.body.reduced_to_full_configuration(
                    self.body_ik_solver(body_target_pose)
                )
            else:
                body_q = self.body_ik_solver(body_target_pose)
        else:
            # If no body target pose is provided, set the body to the default pose
            body_q = self.full_robot.default_body_pose.copy()

        if left_hand_target_pose is not None:
            left_hand_actuated_q = self.left_hand_ik_solver(left_hand_target_pose)
            body_q[self.full_robot.get_hand_actuated_joint_indices(side="left")] = (
                left_hand_actuated_q
            )

        if right_hand_target_pose is not None:
            right_hand_actuated_q = self.right_hand_ik_solver(right_hand_target_pose)
            body_q[self.full_robot.get_hand_actuated_joint_indices(side="right")] = (
                right_hand_actuated_q
            )

        if self.enable_visualization:
            self.visualizer.visualize(np.array(body_q))

        return body_q

    def reset(self):
        """Reset the robot model and IK solvers to the initial state, and re-activate the warmup procedure."""
        self.body.reset_forward_kinematics()  # self.body is the same one as self.body_ik_solver.robot
        self.full_robot.reset_forward_kinematics()
        self.body_ik_solver.initialize()
        # If in the future, the hand IK solver has initialize method, call it
        self._most_recent_ik_data = None
        self._most_recent_q = self.full_robot.default_body_pose.copy()
        self.in_warmup = True

    def set_goal(self, ik_data: dict):
        self._most_recent_ik_data = ik_data

    def get_action(self) -> dict[str, any]:
        # Process IK if active
        if self._most_recent_ik_data is not None:
            body_data = self._most_recent_ik_data["body_data"]
            left_hand_data = self._most_recent_ik_data["left_hand_data"]
            right_hand_data = self._most_recent_ik_data["right_hand_data"]
            target_joints = self.compute_joint_positions(body_data, left_hand_data, right_hand_data)
            self._most_recent_q = target_joints

        return self._most_recent_q[self.full_robot.get_joint_group_indices("upper_body")]
