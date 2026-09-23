import time as time_module
from typing import Optional

import numpy as np
from pinocchio import rpy

from decoupled_wbc.control.base.policy import Policy
from decoupled_wbc.control.main.constants import DEFAULT_NAV_CMD


class G1DecoupledWholeBodyPolicy(Policy):
    """
    This class implements a whole-body policy for the G1 robot by combining an upper-body
    policy and a lower-body RL-based policy.
    It is designed to work with the G1 robot's specific configuration and control requirements.
    """

    def __init__(
        self,
        robot_model,
        lower_body_policy: Policy,
        upper_body_policy: Policy,
    ):
        self.robot_model = robot_model
        self.lower_body_policy = lower_body_policy
        self.upper_body_policy = upper_body_policy
        self.last_goal_time = time_module.monotonic()
        self.is_in_teleop_mode = False  # Track if lower body is in teleop mode

    def set_observation(self, observation):
        # Upper body policy is open loop (just interpolation), so we don't need to set the observation
        self.lower_body_policy.set_observation(observation)

    def set_goal(self, goal):
        """
        Set the goal for both upper and lower body policies.

        Args:
            goal: Command from the planners
            goal["target_upper_body_pose"]: Target pose for the upper body policy
            goal["target_time"]: Target goal time
            goal["interpolation_garbage_collection_time"]: Waypoints earlier than this time are removed
            goal["navigate_cmd"]: Target navigation velocities for the lower body policy
            goal["base_height_command"]: Target base height for both upper and lower body policies
        """
        # Update goal timestamp for timeout safety
        self.last_goal_time = time_module.monotonic()

        upper_body_goal = {}
        lower_body_goal = {}

        # Upper body goal keys
        upper_body_keys = [
            "target_upper_body_pose",
            "base_height_command",
            "target_time",
            "interpolation_garbage_collection_time",
            "navigate_cmd",
        ]
        for key in upper_body_keys:
            if key in goal:
                upper_body_goal[key] = goal[key]

        # Always ensure navigate_cmd is present to prevent interpolation from old dangerous values
        if "navigate_cmd" not in goal:
            # Safety: Inject safe default navigate_cmd to ensure interpolation goes to stop
            if "target_time" in goal and isinstance(goal["target_time"], list):
                upper_body_goal["navigate_cmd"] = [np.array(DEFAULT_NAV_CMD)] * len(
                    goal["target_time"]
                )
            else:
                upper_body_goal["navigate_cmd"] = np.array(DEFAULT_NAV_CMD)

        # Set teleop policy command flag
        has_teleop_commands = ("navigate_cmd" in goal) or ("base_height_command" in goal)
        self.is_in_teleop_mode = has_teleop_commands  # Track teleop state for timeout safety
        self.lower_body_policy.set_use_teleop_policy_cmd(has_teleop_commands)

        # Lower body goal keys
        lower_body_keys = [
            "toggle_stand_command",
            "toggle_policy_action",
        ]
        for key in lower_body_keys:
            if key in goal:
                lower_body_goal[key] = goal[key]

        self.upper_body_policy.set_goal(upper_body_goal)
        self.lower_body_policy.set_goal(lower_body_goal)

    def get_action(self, time: Optional[float] = None):
        current_time = time if time is not None else time_module.monotonic()

        # Safety timeout: Only apply when in teleop mode (communication loss dangerous)
        # When in keyboard mode, no timeout needed (user controls directly)
        if self.is_in_teleop_mode:
            time_since_goal = current_time - self.last_goal_time
            if time_since_goal > 1.0:  # 1 second timeout
                print(
                    f"SAFETY: Teleop mode timeout after {time_since_goal:.1f}s, injecting safe goal"
                )
                # Inject safe goal to trigger all safety mechanisms (gear_wbc reset + interpolation reset)
                safe_goal = {
                    "target_time": current_time + 0.1,
                    "interpolation_garbage_collection_time": current_time - 1.0,
                }
                self.set_goal(
                    safe_goal
                )  # This will reset is_in_teleop_mode to False and trigger all safety

        # Get indices for groups
        lower_body_indices = self.robot_model.get_joint_group_indices("lower_body")
        upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")

        # Initialize full configuration with zeros
        q = np.zeros(self.robot_model.num_dofs)

        upper_body_action = self.upper_body_policy.get_action(time)
        q[upper_body_indices] = upper_body_action["target_upper_body_pose"]
        q_arms = q[self.robot_model.get_joint_group_indices("arms")]
        base_height_command = upper_body_action.get("base_height_command", None)
        interpolated_navigate_cmd = upper_body_action.get("navigate_cmd", None)

        # Compute torso orientation relative to waist, to pass to lower body policy
        self.robot_model.cache_forward_kinematics(q, auto_clip=False)
        torso_orientation = self.robot_model.frame_placement("torso_link").rotation
        waist_orientation = self.robot_model.frame_placement("pelvis").rotation
        # Extract yaw from rotation matrix and create a rotation with only yaw
        # The rotation property is a 3x3 numpy array
        waist_yaw = np.arctan2(waist_orientation[1, 0], waist_orientation[0, 0])
        # Create a rotation matrix with only yaw using Pinocchio's rpy functions
        waist_yaw_only_rotation = rpy.rpyToMatrix(0, 0, waist_yaw)
        yaw_only_waist_from_torso = waist_yaw_only_rotation.T @ torso_orientation
        torso_orientation_rpy = rpy.matrixToRpy(yaw_only_waist_from_torso)

        lower_body_action = self.lower_body_policy.get_action(
            time, q_arms, base_height_command, torso_orientation_rpy, interpolated_navigate_cmd
        )

        # If pelvis is both in upper and lower body, lower body policy takes preference
        q[lower_body_indices] = lower_body_action["body_action"][0][
            : len(lower_body_indices)
        ]  # lower body (legs + waist)

        self.last_action = {"q": q}

        return {"q": q}

    def handle_keyboard_button(self, key):
        try:
            self.lower_body_policy.locomotion_policy.handle_keyboard_button(key)
        except AttributeError:
            # Only catch AttributeError, let other exceptions propagate
            self.lower_body_policy.handle_keyboard_button(key)

    def activate_policy(self):
        self.handle_keyboard_button("]")
