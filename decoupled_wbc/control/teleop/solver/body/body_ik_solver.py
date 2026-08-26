from typing import Dict

import numpy as np
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
import pinocchio as pin
import qpsolvers

from decoupled_wbc.control.teleop.solver.body.body_ik_solver_settings import BodyIKSolverSettings
from decoupled_wbc.control.teleop.solver.solver import Solver


class WeightedPostureTask(PostureTask):
    def __init__(
        self, cost: float, weights: np.ndarray, lm_damping: float = 0.0, gain: float = 1.0
    ) -> None:
        r"""Create weighted posture task.

        Args:
            cost: value used to cast joint angle differences to a homogeneous
                cost, in :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
            weights: vector of weights for each joint.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when
                the error is large) regularization term, which helps when
                targets are unfeasible. Increase this value if the task is too
                jerky under unfeasible targets, but beware that too large a
                damping can slow down the task.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering) for dead-beat
                control.
        """
        super().__init__(cost=cost, lm_damping=lm_damping, gain=gain)
        self.weights = weights

    def compute_error(self, configuration):
        error = super().compute_error(configuration)
        return self.weights * error

    def compute_jacobian(self, configuration):
        J = super().compute_jacobian(configuration)
        # breakpoint()
        return self.weights[:, np.newaxis] * J

    def __repr__(self):
        """Human-readable representation of the weighted posture task."""
        return (
            "WeightedPostureTask("
            f"cost={self.cost}, "
            f"weights={self.weights}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )


class BodyIKSolver(Solver):
    def __init__(self, ik_solver_settings: BodyIKSolverSettings):
        self.dt = ik_solver_settings.dt
        self.num_step_per_frame = ik_solver_settings.num_step_per_frame
        self.amplify_factor = ik_solver_settings.amplify_factor
        self.link_costs = ik_solver_settings.link_costs
        self.posture_weight = ik_solver_settings.posture_weight
        self.posture_cost = ik_solver_settings.posture_cost
        self.posture_lm_damping = ik_solver_settings.posture_lm_damping
        self.robot = None

    def register_robot(self, robot):
        self.robot = robot
        self.initialize()

    def initialize(self):
        self.solver = qpsolvers.available_solvers[0]
        if "quadprog" in qpsolvers.available_solvers:
            self.solver = "quadprog"
        else:
            self.solver = qpsolvers.available_solvers[0]

        q_default = self.robot.q_zero.copy()
        q_default[self.robot.joint_to_dof_index["left_shoulder_roll_joint"]] = 0.2
        q_default[self.robot.joint_to_dof_index["right_shoulder_roll_joint"]] = -0.2

        self.configuration = pink.Configuration(
            self.robot.pinocchio_wrapper.model,
            self.robot.pinocchio_wrapper.data,
            q_default,
        )
        self.configuration.model.lowerPositionLimit = self.robot.lower_joint_limits
        self.configuration.model.upperPositionLimit = self.robot.upper_joint_limits

        # initialize tasks
        self.tasks = {}
        for link_name, weight in self.link_costs.items():
            assert link_name != "posture", "posture is a reserved task name"

            # Map robot-agnostic link names to robot-specific names
            if link_name == "hand":
                # Use hand_frame_names from supplemental info
                for side in ["left", "right"]:
                    frame_name = self.robot.supplemental_info.hand_frame_names[side]
                    task = FrameTask(
                        frame_name,
                        **weight,
                    )
                    self.tasks[frame_name] = task
            else:
                # For other links, use the name directly
                task = FrameTask(
                    link_name,
                    **weight,
                )
                self.tasks[link_name] = task

        # add posture task
        if self.posture_weight is not None:
            weight = np.ones(self.robot.num_dofs)

            # Map robot-agnostic joint types to specific robot joint names using supplemental info
            for joint_type, posture_weight in self.posture_weight.items():
                if joint_type not in self.robot.supplemental_info.joint_name_mapping:
                    print(f"Warning: Unknown joint type {joint_type}")
                    continue

                # Get the joint name mapping for this type
                joint_mapping = self.robot.supplemental_info.joint_name_mapping[joint_type]

                # Handle both single joint names and left/right mappings
                if isinstance(joint_mapping, str):
                    # Single joint (e.g., waist joints)
                    if joint_mapping in self.robot.joint_to_dof_index:
                        joint_idx = self.robot.joint_to_dof_index[joint_mapping]
                        weight[joint_idx] = posture_weight
                else:
                    # Left/right mapping (e.g., arm joints)
                    for side in ["left", "right"]:
                        joint_name = joint_mapping[side]
                        if joint_name in self.robot.joint_to_dof_index:
                            joint_idx = self.robot.joint_to_dof_index[joint_name]
                            weight[joint_idx] = posture_weight

            self.tasks["posture"] = WeightedPostureTask(
                cost=self.posture_cost,
                weights=weight,
                lm_damping=self.posture_lm_damping,
            )
        else:
            self.tasks["posture"] = PostureTask(
                cost=self.posture_cost, lm_damping=self.posture_lm_damping
            )
        for task in self.tasks.values():
            task.set_target_from_configuration(self.configuration)

    def __call__(self, target_pose: Dict):
        for link_name, pose in target_pose.items():
            if link_name not in self.tasks:
                continue
            pose = pin.SE3(pose[:3, :3], pose[:3, 3])
            self.tasks[link_name].set_target(pose)

        for _ in range(self.num_step_per_frame):
            velocity = solve_ik(
                self.configuration,
                self.tasks.values(),
                dt=self.dt,
                solver=self.solver,
            )
            self.configuration.q = self.robot.clip_configuration(
                self.configuration.q + velocity * self.dt * self.amplify_factor
            )
            self.configuration.update()
            self.robot.cache_forward_kinematics(self.configuration.q)

        return self.configuration.q.copy()

    def update_weights(self, weights):
        for link_name, weight in weights.items():
            if "position_cost" in weight:
                self.tasks[link_name].set_position_cost(weight["position_cost"])
            if "orientation_cost" in weight:
                self.tasks[link_name].set_orientation_cost(weight["orientation_cost"])
