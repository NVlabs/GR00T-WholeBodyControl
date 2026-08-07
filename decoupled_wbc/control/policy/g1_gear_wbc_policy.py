import collections
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort
import torch

from decoupled_wbc.control.base.policy import Policy
from decoupled_wbc.control.utils.gear_wbc_utils import get_gravity_orientation, load_config


class G1GearWbcPolicy(Policy):
    """Simple G1 robot policy using OpenGearWbc trained neural network."""

    def __init__(self, robot_model, config: str, model_path: str):
        """Initialize G1GearWbcPolicy.

        Args:
            config_path: Path to gear_wbc YAML configuration file
        """
        self.config, self.LEGGED_GYM_ROOT_DIR = load_config(config)
        self.robot_model = robot_model
        self.use_teleop_policy_cmd = False

        package_root = Path(__file__).resolve().parents[2]
        self.sim2mujoco_root_dir = str(package_root / "sim2mujoco")
        model_path_1, model_path_2 = model_path.split(",")

        self.policy_1 = self.load_onnx_policy(
            self.sim2mujoco_root_dir + "/resources/robots/g1/" + model_path_1
        )
        self.policy_2 = self.load_onnx_policy(
            self.sim2mujoco_root_dir + "/resources/robots/g1/" + model_path_2
        )

        # Initialize observation history buffer
        self.observation = None
        self.obs_history = collections.deque(maxlen=self.config["obs_history_len"])
        self.obs_buffer = np.zeros(self.config["num_obs"], dtype=np.float32)
        self.counter = 0

        # Initialize state variables
        self.use_policy_action = False
        self.action = np.zeros(self.config["num_actions"], dtype=np.float32)
        self.target_dof_pos = self.config["default_angles"].copy()
        self.cmd = self.config["cmd_init"].copy()
        self.height_cmd = self.config["height_cmd"]
        self.freq_cmd = self.config["freq_cmd"]
        self.roll_cmd = self.config["rpy_cmd"][0]
        self.pitch_cmd = self.config["rpy_cmd"][1]
        self.yaw_cmd = self.config["rpy_cmd"][2]
        self.gait_indices = torch.zeros((1), dtype=torch.float32)

    def load_onnx_policy(self, model_path: str):
        print(f"Loading ONNX policy from {model_path}")
        model = ort.InferenceSession(model_path)

        def run_inference(input_tensor):
            ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0], device="cpu")

        print(f"Successfully loaded ONNX policy from {model_path}")

        return run_inference

    def compute_observation(self, observation: Dict[str, Any]) -> tuple[np.ndarray, int]:
        """Compute the observation vector from current state"""
        # Get body joint indices (excluding waist roll and pitch)
        self.gait_indices = torch.remainder(self.gait_indices + 0.02 * self.freq_cmd, 1.0)
        durations = torch.full_like(self.gait_indices, 0.5)
        phases = 0.5
        foot_indices = [
            self.gait_indices + phases,  # FL
            self.gait_indices,  # FR
        ]
        self.foot_indices = torch.remainder(
            torch.cat([foot_indices[i].unsqueeze(1) for i in range(2)], dim=1), 1.0
        )
        for fi in foot_indices:
            stance = fi < durations
            swing = fi >= durations
            fi[stance] = fi[stance] * (0.5 / durations[stance])
            fi[swing] = 0.5 + (fi[swing] - durations[swing]) * (0.5 / (1 - durations[swing]))

        self.clock_inputs = torch.stack([torch.sin(2 * np.pi * fi) for fi in foot_indices], dim=1)

        body_indices = self.robot_model.get_joint_group_indices("body")
        body_indices = [idx for idx in body_indices]

        n_joints = len(body_indices)

        # Extract joint data
        qj = observation["q"][body_indices].copy()
        dqj = observation["dq"][body_indices].copy()

        # Extract floating base data
        quat = observation["floating_base_pose"][3:7].copy()  # quaternion
        omega = observation["floating_base_vel"][3:6].copy()  # angular velocity

        # Handle default angles padding
        if len(self.config["default_angles"]) < n_joints:
            padded_defaults = np.zeros(n_joints, dtype=np.float32)
            padded_defaults[: len(self.config["default_angles"])] = self.config["default_angles"]
        else:
            padded_defaults = self.config["default_angles"][:n_joints]

        # Scale the values
        qj_scaled = (qj - padded_defaults) * self.config["dof_pos_scale"]
        dqj_scaled = dqj * self.config["dof_vel_scale"]
        gravity_orientation = get_gravity_orientation(quat)
        omega_scaled = omega * self.config["ang_vel_scale"]

        # Calculate single observation dimension
        single_obs_dim = 86  # 3 + 1 + 3 + 3 + 3 + n_joints + n_joints + 15, n_joints = 29

        # Create single observation
        single_obs = np.zeros(single_obs_dim, dtype=np.float32)
        single_obs[0:3] = self.cmd[:3] * self.config["cmd_scale"]
        single_obs[3:4] = np.array([self.height_cmd])
        single_obs[4:7] = np.array([self.roll_cmd, self.pitch_cmd, self.yaw_cmd])
        single_obs[7:10] = omega_scaled
        single_obs[10:13] = gravity_orientation
        # single_obs[14:17] = omega_scaled_torso
        # single_obs[17:20] = gravity_torso
        single_obs[13 : 13 + n_joints] = qj_scaled
        single_obs[13 + n_joints : 13 + 2 * n_joints] = dqj_scaled
        single_obs[13 + 2 * n_joints : 13 + 2 * n_joints + 15] = self.action
        # single_obs[13 + 2 * n_joints + 15 : 13 + 2 * n_joints + 15 + 2] = (
        #     processed_clock_inputs.detach().cpu().numpy()
        # )
        return single_obs, single_obs_dim

    def set_observation(self, observation: Dict[str, Any]):
        """Update the policy's current observation of the environment.

        Args:
            observation: Dictionary containing single observation from current state
                        Should include 'obs' key with current single observation
        """

        # Extract the single observation
        self.observation = observation
        single_obs, single_obs_dim = self.compute_observation(observation)

        # Update observation history every control_decimation steps
        # if self.counter % self.config['control_decimation'] == 0:
        # Add current observation to history
        self.obs_history.append(single_obs)

        # Fill history with zeros if not enough observations yet
        while len(self.obs_history) < self.config["obs_history_len"]:
            self.obs_history.appendleft(np.zeros_like(single_obs))

        # Construct full observation with history
        single_obs_dim = len(single_obs)
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * single_obs_dim
            end_idx = start_idx + single_obs_dim
            self.obs_buffer[start_idx:end_idx] = hist_obs

        # Convert to tensor for policy
        self.obs_tensor = torch.from_numpy(self.obs_buffer).unsqueeze(0)
        # self.counter += 1

        assert self.obs_tensor.shape[1] == self.config["num_obs"]

    def set_use_teleop_policy_cmd(self, use_teleop_policy_cmd: bool):
        self.use_teleop_policy_cmd = use_teleop_policy_cmd
        # Safety: When teleop is disabled, reset navigation to stop
        if not use_teleop_policy_cmd:
            self.nav_cmd = self.config["cmd_init"].copy()  # Reset to safe default

    def set_goal(self, goal: Dict[str, Any]):
        """Set the goal for the policy.

        Args:
            goal: Dictionary containing the goal for the policy
        """

        if "toggle_policy_action" in goal:
            if goal["toggle_policy_action"]:
                self.use_policy_action = not self.use_policy_action

    def get_action(
        self,
        time: Optional[float] = None,
        arms_target_pose: Optional[np.ndarray] = None,
        base_height_command: Optional[np.ndarray] = None,
        torso_orientation_rpy: Optional[np.ndarray] = None,
        interpolated_navigate_cmd: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute and return the next action based on current observation.

        Args:
            time: Optional "monotonic time" for time-dependent policies (unused)

        Returns:
            Dictionary containing the action to be executed
        """
        if self.obs_tensor is None:
            raise ValueError("No observation set. Call set_observation() first.")

        if base_height_command is not None and self.use_teleop_policy_cmd:
            self.height_cmd = (
                base_height_command[0]
                if isinstance(base_height_command, list)
                else base_height_command
            )

        if interpolated_navigate_cmd is not None and self.use_teleop_policy_cmd:
            self.cmd = interpolated_navigate_cmd

        if torso_orientation_rpy is not None and self.use_teleop_policy_cmd:
            self.roll_cmd = torso_orientation_rpy[0]
            self.pitch_cmd = torso_orientation_rpy[1]
            self.yaw_cmd = torso_orientation_rpy[2]

        # Run policy inference
        with torch.no_grad():
            # Select appropriate policy based on command magnitude
            if np.linalg.norm(self.cmd) < 0.05:
                # Use standing policy for small commands
                policy = self.policy_1
            else:
                # Use walking policy for movement commands
                policy = self.policy_2

            self.action = policy(self.obs_tensor).detach().numpy().squeeze()

        # Transform action to target_dof_pos
        if self.use_policy_action:
            cmd_q = self.action * self.config["action_scale"] + self.config["default_angles"]
        else:
            cmd_q = self.observation["q"][self.robot_model.get_joint_group_indices("lower_body")]

        cmd_dq = np.zeros(self.config["num_actions"])
        cmd_tau = np.zeros(self.config["num_actions"])

        return {"body_action": (cmd_q, cmd_dq, cmd_tau)}

    def handle_keyboard_button(self, key):
        if key == "]":
            self.use_policy_action = True
        elif key == "o":
            self.use_policy_action = False
        elif key == "w":
            self.cmd[0] += 0.2
        elif key == "s":
            self.cmd[0] -= 0.2
        elif key == "a":
            self.cmd[1] += 0.2
        elif key == "d":
            self.cmd[1] -= 0.2
        elif key == "q":
            self.cmd[2] += 0.2
        elif key == "e":
            self.cmd[2] -= 0.2
        elif key == "z":
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0
        elif key == "1":
            self.height_cmd += 0.1
        elif key == "2":
            self.height_cmd -= 0.1
        elif key == "n":
            self.freq_cmd -= 0.1
            self.freq_cmd = max(1.0, self.freq_cmd)
        elif key == "m":
            self.freq_cmd += 0.1
            self.freq_cmd = min(2.0, self.freq_cmd)
        elif key == "3":
            self.roll_cmd -= np.deg2rad(10)
        elif key == "4":
            self.roll_cmd += np.deg2rad(10)
        elif key == "5":
            self.pitch_cmd -= np.deg2rad(10)
        elif key == "6":
            self.pitch_cmd += np.deg2rad(10)
        elif key == "7":
            self.yaw_cmd -= np.deg2rad(10)
        elif key == "8":
            self.yaw_cmd += np.deg2rad(10)

        if key:
            print("--------------------------------")
            print(f"Linear velocity command: {self.cmd}")
            print(f"Base height command: {self.height_cmd}")
            print(f"Use policy action: {self.use_policy_action}")
            print(f"roll deg angle: {np.rad2deg(self.roll_cmd)}")
            print(f"pitch deg angle: {np.rad2deg(self.pitch_cmd)}")
            print(f"yaw deg angle: {np.rad2deg(self.yaw_cmd)}")
            print(f"Gait frequency: {self.freq_cmd}")
