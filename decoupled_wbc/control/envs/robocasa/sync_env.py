import sys
from typing import Any, Dict, Tuple

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from robocasa.environments.locomanipulation import REGISTERED_LOCOMANIPULATION_ENVS
from robocasa.models.robots import GR00T_LOCOMANIP_ENVS_ROBOTS
from robosuite.environments.robot_env import RobotEnv
from scipy.spatial.transform import Rotation as R

from decoupled_wbc.control.envs.g1.utils.joint_safety import JointSafetyMonitor
from decoupled_wbc.control.envs.robocasa.utils.controller_utils import (
    update_robosuite_controller_configs,
)
from decoupled_wbc.control.envs.robocasa.utils.robocasa_env import (  # noqa: F401
    ALLOWED_LANGUAGE_CHARSET,
    Gr00tLocomanipRoboCasaEnv,
)
from decoupled_wbc.control.robot_model.instantiation import get_robot_type_and_model
from decoupled_wbc.control.utils.n1_utils import (
    prepare_gym_space_for_eval,
    prepare_observation_for_eval,
)
from decoupled_wbc.data.constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH


class SyncEnv(gym.Env):
    MAX_MUJOCO_STATE_LEN = 800

    def __init__(self, env_name, robot_name, **kwargs):
        self.env_name = env_name
        self.robot_name = robot_name
        self.onscreen = kwargs.get("onscreen", True)
        self.enable_gravity_compensation = kwargs.pop("enable_gravity_compensation", False)
        self.gravity_compensation_joints = kwargs.pop("gravity_compensation_joints", ["arms"])
        _, self.robot_model = get_robot_type_and_model(
            robot_name, enable_waist_ik=kwargs.pop("enable_waist", False)
        )

        env_kwargs = {
            "onscreen": kwargs.get("onscreen", True),
            "offscreen": kwargs.get("offscreen", False),
            "renderer": kwargs.get("renderer", "mjviewer"),
            "render_camera": kwargs.get("render_camera", "frontview"),
            "camera_names": kwargs.get("camera_names", ["frontview"]),
            "camera_heights": kwargs.get("camera_heights", None),
            "camera_widths": kwargs.get("camera_widths", None),
            "controller_configs": kwargs["controller_configs"],
            "control_freq": kwargs.get("control_freq", 50),
            "translucent_robot": kwargs.get("translucent_robot", True),
            "ik_indicator": kwargs.get("ik_indicator", False),
            "randomize_cameras": kwargs.get("randomize_cameras", True),
        }
        self.env = Gr00tLocomanipRoboCasaEnv(
            env_name, robot_name, robot_model=self.robot_model, **env_kwargs
        )
        self.init_cache()

        self.reset()

    @property
    def base_env(self) -> RobotEnv:
        return self.env.env

    def overwrite_floating_base_action(self, navigate_cmd):
        if self.base_env.robots[0].robot_model.default_base == "FloatingLeggedBase":
            self.env.unwrapped.overridden_floating_base_action = navigate_cmd

    def get_mujoco_state_info(self):
        mujoco_state = self.base_env.sim.get_state().flatten()
        assert len(mujoco_state) < SyncEnv.MAX_MUJOCO_STATE_LEN
        padding_width = SyncEnv.MAX_MUJOCO_STATE_LEN - len(mujoco_state)
        padded_mujoco_state = np.pad(
            mujoco_state, (0, padding_width), mode="constant", constant_values=0
        )
        max_mujoco_state_len = SyncEnv.MAX_MUJOCO_STATE_LEN
        mujoco_state_len = len(mujoco_state)
        mujoco_state = padded_mujoco_state.copy()
        return max_mujoco_state_len, mujoco_state_len, mujoco_state

    def reset_to(self, state: Dict[str, Any]) -> Dict[str, Any] | None:
        if hasattr(self.base_env, "reset_to"):
            self.base_env.reset_to(state)
        else:
            # todo: maybe update robosuite to have reset_to()
            env = self.base_env
            if "model_file" in state:
                xml = env.edit_model_xml(state["model_file"])
                env.reset_from_xml_string(xml)
                env.sim.reset()
            if "states" in state:
                env.sim.set_state_from_flattened(state["states"])
                env.sim.forward()

        obs = self.env.force_update_observation(timestep=0)
        self.cache["obs"] = obs
        return

    def get_state(self) -> Dict[str, Any]:
        return self.base_env.get_state()

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        # First, try to use the base environment's is_success method if it exists
        if hasattr(self.base_env, "is_success"):
            return self.base_env.is_success()

        # Fall back to using _check_success if available
        elif hasattr(self.base_env, "_check_success"):
            succ = self.base_env._check_success()
            if isinstance(succ, dict):
                assert "task" in succ
                return succ
            return {"task": succ}

        # If neither method exists, return failure
        else:
            return {"task": False}

    def init_cache(self):
        self.cache = {
            "obs": None,
            "reward": None,
            "terminated": None,
            "truncated": None,
            "info": None,
        }

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, any], Dict[str, any]]:
        self.init_cache()
        obs, info = self.env.reset(seed=seed, options=options)
        self.cache["obs"] = obs
        self.cache["reward"] = 0
        self.cache["terminated"] = False
        self.cache["truncated"] = False
        self.cache["info"] = info
        return self.observe(), info

    def observe(self) -> Dict[str, any]:
        # Get observations from body and hands
        assert (
            self.cache["obs"] is not None
        ), "Observation cache is not initialized, please reset the environment first"
        raw_obs = self.cache["obs"]

        # Body and hand joint measurements come in actuator order, so we need to convert them to joint order
        whole_q = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=raw_obs["body_q"],
            left_hand_actuated_joint_values=raw_obs["left_hand_q"],
            right_hand_actuated_joint_values=raw_obs["right_hand_q"],
        )
        whole_dq = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=raw_obs["body_dq"],
            left_hand_actuated_joint_values=raw_obs["left_hand_dq"],
            right_hand_actuated_joint_values=raw_obs["right_hand_dq"],
        )
        whole_ddq = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=raw_obs["body_ddq"],
            left_hand_actuated_joint_values=raw_obs["left_hand_ddq"],
            right_hand_actuated_joint_values=raw_obs["right_hand_ddq"],
        )
        whole_tau_est = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=raw_obs["body_tau_est"],
            left_hand_actuated_joint_values=raw_obs["left_hand_tau_est"],
            right_hand_actuated_joint_values=raw_obs["right_hand_tau_est"],
        )
        eef_obs = self.get_eef_obs(whole_q)

        obs = {
            "q": whole_q,
            "dq": whole_dq,
            "ddq": whole_ddq,
            "tau_est": whole_tau_est,
            "floating_base_pose": raw_obs["floating_base_pose"],
            "floating_base_vel": raw_obs["floating_base_vel"],
            "floating_base_acc": raw_obs["floating_base_acc"],
            "wrist_pose": np.concatenate([eef_obs["left_wrist_pose"], eef_obs["right_wrist_pose"]]),
        }

        # Add state keys for model input
        obs = prepare_observation_for_eval(self.robot_model, obs)

        obs["annotation.human.task_description"] = raw_obs["language.language_instruction"]

        if hasattr(self.base_env, "get_privileged_obs_keys"):
            for key in self.base_env.get_privileged_obs_keys():
                obs[key] = raw_obs[key]

        for key in raw_obs.keys():
            if key.endswith("_image"):
                obs[key] = raw_obs[key]
                # TODO: add video.key without _image suffix for evaluation, convert to uint8, remove later
                obs[f"video.{key.replace('_image', '')}"] = raw_obs[key]
        return obs

    def step(
        self, action: Dict[str, any]
    ) -> Tuple[Dict[str, any], float, bool, bool, Dict[str, any]]:
        self.queue_action(action)
        return self.get_step_info()

    def get_observation(self):
        return self.base_env._get_observations()  # assumes base env is robosuite

    def get_step_info(self) -> Tuple[Dict[str, any], float, bool, bool, Dict[str, any]]:
        return (
            self.observe(),
            self.cache["reward"],
            self.cache["terminated"],
            self.cache["truncated"],
            self.cache["info"],
        )

    def convert_q_to_actuated_joint_order(self, q: np.ndarray) -> np.ndarray:
        body_q = self.robot_model.get_body_actuated_joints(q)
        left_hand_q = self.robot_model.get_hand_actuated_joints(q, side="left")
        right_hand_q = self.robot_model.get_hand_actuated_joints(q, side="right")
        whole_q = np.zeros_like(q)
        whole_q[self.robot_model.get_joint_group_indices("body")] = body_q
        whole_q[self.robot_model.get_joint_group_indices("left_hand")] = left_hand_q
        whole_q[self.robot_model.get_joint_group_indices("right_hand")] = right_hand_q

        return whole_q

    def set_ik_indicator(self, teleop_cmd):
        """Set the IK indicators for the simulator"""
        if "left_wrist" in teleop_cmd and "right_wrist" in teleop_cmd:
            left_wrist_input_pose = teleop_cmd["left_wrist"]
            right_wrist_input_pose = teleop_cmd["right_wrist"]
            ik_wrapper = self.base_env
            ik_wrapper.set_target_poses_outside_env([left_wrist_input_pose, right_wrist_input_pose])

    def render(self):
        if self.base_env.viewer is not None:
            self.base_env.viewer.update()
        if self.onscreen:
            self.base_env.render()

    def queue_action(self, action: Dict[str, any]):
        # action is in pinocchio joint order, we need to convert it to actuator order
        action_q = self.convert_q_to_actuated_joint_order(action["q"])

        # Compute gravity compensation torques if enabled
        tau_q = np.zeros_like(action_q)
        if self.enable_gravity_compensation and self.robot_model is not None:
            try:
                # Get current robot configuration from cache (more efficient than observe())
                raw_obs = self.cache["obs"]

                # Convert from actuated joint order to joint order for Pinocchio
                current_q_joint_order = self.robot_model.get_configuration_from_actuated_joints(
                    body_actuated_joint_values=raw_obs["body_q"],
                    left_hand_actuated_joint_values=raw_obs["left_hand_q"],
                    right_hand_actuated_joint_values=raw_obs["right_hand_q"],
                )

                # Compute gravity compensation in joint order using current robot configuration
                gravity_torques_joint_order = self.robot_model.compute_gravity_compensation_torques(
                    current_q_joint_order, joint_groups=self.gravity_compensation_joints
                )

                # Convert gravity torques to actuated joint order
                gravity_torques_actuated = self.convert_q_to_actuated_joint_order(
                    gravity_torques_joint_order
                )

                # Add gravity compensation to torques
                tau_q += gravity_torques_actuated

            except Exception as e:
                print(f"Error applying gravity compensation in sync_env: {e}")

        obs, reward, terminated, truncated, info = self.env.step({"q": action_q, "tau": tau_q})
        self.cache["obs"] = obs
        self.cache["reward"] = reward
        self.cache["terminated"] = terminated
        self.cache["truncated"] = truncated
        self.cache["info"] = info

    def queue_state(self, state: Dict[str, any]):
        # This function is for debugging or cross-playback between sim and real only.
        state_q = self.convert_q_to_actuated_joint_order(state["q"])
        obs, reward, terminated, truncated, info = self.env.unwrapped.step_only_kinematics(
            {"q": state_q}
        )
        self.cache["obs"] = obs
        self.cache["reward"] = reward
        self.cache["terminated"] = terminated
        self.cache["truncated"] = truncated
        self.cache["info"] = info

    @property
    def observation_space(self) -> gym.Space:
        # @todo: check if the low and high bounds are correct for body_obs.
        q_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        dq_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        ddq_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        tau_est_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        floating_base_pose_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        floating_base_vel_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        floating_base_acc_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        wrist_pose_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7 + 7,))

        obs_space = gym.spaces.Dict(
            {
                "floating_base_pose": floating_base_pose_space,
                "floating_base_vel": floating_base_vel_space,
                "floating_base_acc": floating_base_acc_space,
                "q": q_space,
                "dq": dq_space,
                "ddq": ddq_space,
                "tau_est": tau_est_space,
                "wrist_pose": wrist_pose_space,
            }
        )

        obs_space = prepare_gym_space_for_eval(self.robot_model, obs_space)

        obs_space["annotation.human.task_description"] = gym.spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )

        if hasattr(self.base_env, "get_privileged_obs_keys"):
            for key, shape in self.base_env.get_privileged_obs_keys().items():
                space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)
                obs_space[key] = space

        robocasa_obs_space = self.env.observation_space
        for key in robocasa_obs_space.keys():
            if key.endswith("_image"):
                space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=robocasa_obs_space[key].shape
                )
                obs_space[key] = space
                # TODO: add video.key without _image suffix for evaluation, remove later
                space_uint = gym.spaces.Box(low=0, high=255, shape=space.shape, dtype=np.uint8)
                obs_space[f"video.{key.replace('_image', '')}"] = space_uint

        return obs_space

    def reset_obj_pos(self):
        # For Tairan's goal-reaching task, a hacky way to reset the object position is needed.
        if hasattr(self.base_env, "reset_obj_pos"):
            self.base_env.reset_obj_pos()

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    def close(self):
        self.env.close()

    def __repr__(self):
        return (
            f"SyncEnv(env_name={self.env_name}, \n"
            f"            observation_space={self.observation_space}, \n"
            f"            action_space={self.action_space})"
        )

    def get_joint_gains(self):
        controller = self.base_env.robots[0].composite_controller

        gains = {}
        key_mapping = {
            "left": "left_arm",
            "right": "right_arm",
            "legs": "legs",
            "torso": "waist",
            "head": "neck",
        }
        for k in controller.part_controllers.keys():
            if hasattr(controller.part_controllers[k], "kp"):
                if k in key_mapping:
                    gains[key_mapping[k]] = controller.part_controllers[k].kp
                else:
                    gains[k] = controller.part_controllers[k].kp
        gains.update(
            {
                "left_hand": self.base_env.sim.model.actuator_gainprm[
                    self.base_env.robots[0]._ref_actuators_indexes_dict["left_gripper"], 0
                ],
                "right_hand": self.base_env.sim.model.actuator_gainprm[
                    self.base_env.robots[0]._ref_actuators_indexes_dict["right_gripper"], 0
                ],
            }
        )
        joint_gains = np.zeros(self.robot_model.num_dofs)
        for k in gains.keys():
            joint_gains[self.robot_model.get_joint_group_indices(k)] = gains[k]
        return joint_gains

    def get_joint_damping(self):
        controller = self.base_env.robots[0].composite_controller
        damping = {}
        key_mapping = {
            "left": "left_arm",
            "right": "right_arm",
            "legs": "legs",
            "torso": "waist",
            "head": "neck",
        }
        for k in controller.part_controllers.keys():
            if hasattr(controller.part_controllers[k], "kd"):
                if k in key_mapping:
                    damping[key_mapping[k]] = controller.part_controllers[k].kd
                else:
                    damping[k] = controller.part_controllers[k].kd
        damping.update(
            {
                "left_hand": -self.base_env.sim.model.actuator_biasprm[
                    self.base_env.robots[0]._ref_actuators_indexes_dict["left_gripper"], 2
                ],
                "right_hand": -self.base_env.sim.model.actuator_biasprm[
                    self.base_env.robots[0]._ref_actuators_indexes_dict["right_gripper"], 2
                ],
            }
        )
        joint_damping = np.zeros(self.robot_model.num_dofs)
        for k in damping.keys():
            joint_damping[self.robot_model.get_joint_group_indices(k)] = damping[k]
        return joint_damping

    def get_eef_obs(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        self.robot_model.cache_forward_kinematics(q)
        eef_obs = {}
        for side in ["left", "right"]:
            wrist_placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            wrist_pos, wrist_quat = wrist_placement.translation[:3], R.from_matrix(
                wrist_placement.rotation
            ).as_quat(scalar_first=True)
            eef_obs[f"{side}_wrist_pose"] = np.concatenate([wrist_pos, wrist_quat])

        return eef_obs


class G1SyncEnv(SyncEnv):
    def __init__(
        self,
        env_name,
        robot_name,
        **kwargs,
    ):
        renderer = kwargs.get("renderer", "mjviewer")
        if renderer == "mjviewer":
            default_render_camera = ["robot0_oak_egoview"]
        elif renderer in ["mujoco", "rerun"]:
            default_render_camera = [
                "robot0_oak_egoview",
                "robot0_oak_left_monoview",
                "robot0_oak_right_monoview",
            ]
        else:
            raise NotImplementedError
        default_camera_names = [
            "robot0_oak_egoview",
            "robot0_oak_left_monoview",
            "robot0_oak_right_monoview",
        ]
        default_camera_heights = [
            RS_VIEW_CAMERA_HEIGHT,
            RS_VIEW_CAMERA_HEIGHT,
            RS_VIEW_CAMERA_HEIGHT,
        ]
        default_camera_widths = [
            RS_VIEW_CAMERA_WIDTH,
            RS_VIEW_CAMERA_WIDTH,
            RS_VIEW_CAMERA_WIDTH,
        ]

        kwargs.update(
            {
                "onscreen": kwargs.get("onscreen", True),
                "offscreen": kwargs.get("offscreen", False),
                "render_camera": kwargs.get("render_camera", default_render_camera),
                "camera_names": kwargs.get("camera_names", default_camera_names),
                "camera_heights": kwargs.get("camera_heights", default_camera_heights),
                "camera_widths": kwargs.get("camera_widths", default_camera_widths),
                "translucent_robot": kwargs.get("translucent_robot", False),
            }
        )
        super().__init__(env_name=env_name, robot_name=robot_name, **kwargs)

        # Initialize safety monitor (visualization disabled) - G1 specific
        self.safety_monitor = JointSafetyMonitor(
            self.robot_model,
            enable_viz=False,
            env_type="sim",  # G1SyncEnv is always simulation
        )
        self.safety_monitor.ramp_duration_steps = 0
        self.safety_monitor.startup_complete = True
        self.safety_monitor.LOWER_BODY_VELOCITY_LIMIT = (
            1e10  # disable lower body velocity limits since it impacts WBC
        )
        self.last_safety_ok = True  # Track last safety status from queue_action

    @property
    def observation_space(self):
        obs_space = super().observation_space
        obs_space["torso_quat"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        obs_space["torso_ang_vel"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        return obs_space

    def observe(self):
        obs = super().observe()
        obs["torso_quat"] = self.cache["obs"]["secondary_imu_quat"]
        obs["torso_ang_vel"] = self.cache["obs"]["secondary_imu_vel"][3:6]
        return obs

    def queue_action(self, action: Dict[str, any]):
        # Safety check before queuing action
        obs = self.observe()
        safety_result = self.safety_monitor.handle_violations(obs, action)
        action = safety_result["action"]
        # Save safety status for efficient access
        self.last_safety_ok = not safety_result.get("shutdown_required", False)
        # Check if shutdown is required
        if safety_result["shutdown_required"]:
            self.safety_monitor.trigger_system_shutdown()

        # Call parent queue_action with potentially modified action
        super().queue_action(action)

    def get_joint_safety_status(self) -> bool:
        """Get current joint safety status from the last queue_action safety check.

        Returns:
            bool: True if joints are safe (no shutdown required), False if unsafe
        """
        return self.last_safety_ok


def create_gym_sync_env_class(env, robot, robot_alias, wbc_version):
    class_name = f"{env}_{robot}_{wbc_version}"
    id_name = f"gr00tlocomanip_{robot_alias}/{class_name}"

    if robot_alias.startswith("g1"):
        env_class_type = G1SyncEnv
    elif robot_alias.startswith("gr1"):
        env_class_type = globals().get("GR1SyncEnv", SyncEnv)
    else:
        env_class_type = SyncEnv

    controller_configs = update_robosuite_controller_configs(
        robot=robot,
        wbc_version=wbc_version,
    )

    env_class_type = type(
        class_name,
        (env_class_type,),
        {
            "__init__": lambda self, **kwargs: super(self.__class__, self).__init__(
                env_name=env,
                robot_name=robot,
                controller_configs=controller_configs,
                **kwargs,
            )
        },
    )

    current_module = sys.modules["decoupled_wbc.control.envs.robocasa.sync_env"]
    setattr(current_module, class_name, env_class_type)
    register(
        id=id_name,  # Unique ID for the environment
        entry_point=f"decoupled_wbc.control.envs.robocasa.sync_env:{class_name}",
    )


WBC_VERSION = "gear_wbc"

for ENV in REGISTERED_LOCOMANIPULATION_ENVS:
    for ROBOT, ROBOT_ALIAS in GR00T_LOCOMANIP_ENVS_ROBOTS.items():
        create_gym_sync_env_class(ENV, ROBOT, ROBOT_ALIAS, WBC_VERSION)


if __name__ == "__main__":

    env = gym.make("gr00tlocomanip_g1_sim/PnPBottle_g1_gear_wbc")
    print(env.observation_space)
