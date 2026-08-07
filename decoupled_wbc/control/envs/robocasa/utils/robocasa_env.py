import os
from typing import Any, Dict, List, Tuple

from gymnasium import spaces
import mujoco
import numpy as np
import robocasa
from robocasa.utils.gym_utils.gymnasium_basic import (
    RoboCasaEnv,
    create_env_robosuite,
)
from robocasa.wrappers.ik_wrapper import IKWrapper
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

from decoupled_wbc.control.envs.robocasa.utils.cam_key_converter import CameraKeyMapper
from decoupled_wbc.control.envs.robocasa.utils.robot_key_converter import Gr00tObsActionConverter
from decoupled_wbc.control.robot_model.robot_model import RobotModel

ALLOWED_LANGUAGE_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
)


class Gr00tLocomanipRoboCasaEnv(RoboCasaEnv):
    def __init__(
        self,
        env_name: str,
        robots_name: str,
        robot_model: RobotModel,  # gr00t robot model
        input_space: str = "JOINT_SPACE",  # either "JOINT_SPACE" or "EEF_SPACE"
        camera_names: List[str] = ["egoview"],
        camera_heights: List[int] | None = None,
        camera_widths: List[int] | None = None,
        onscreen: bool = False,
        offscreen: bool = False,
        dump_rollout_dataset_dir: str | None = None,
        rollout_hdf5: str | None = None,
        rollout_trainset: int | None = None,
        controller_configs: str | None = None,
        ik_indicator: bool = False,
        **kwargs,
    ):
        # ========= Create env =========
        if controller_configs is None:
            if "G1" in robots_name:
                controller_configs = (
                    "robocasa/examples/third_party_controller/default_mink_ik_g1_wbc.json"
                )
            elif "GR1" in robots_name:
                controller_configs = (
                    "robocasa/examples/third_party_controller/default_mink_ik_gr1_smallkd.json"
                )
            else:
                assert False, f"Unsupported robot name: {robots_name}"
        controller_configs = os.path.join(
            os.path.dirname(robocasa.__file__),
            "../",
            controller_configs,
        )
        controller_configs = load_composite_controller_config(
            controller=controller_configs,
            robot=robots_name.split("_")[0],
        )
        if input_space == "JOINT_SPACE":
            controller_configs["type"] = "BASIC"
            controller_configs["composite_controller_specific_configs"] = {}
            controller_configs["control_delta"] = False

        self.camera_key_mapper = CameraKeyMapper()
        self.camera_names = camera_names

        if camera_widths is None:
            self.camera_widths = [
                self.camera_key_mapper.get_camera_config(name)[1] for name in camera_names
            ]
        else:
            self.camera_widths = camera_widths
        if camera_heights is None:
            self.camera_heights = [
                self.camera_key_mapper.get_camera_config(name)[2] for name in camera_names
            ]
        else:
            self.camera_heights = camera_heights

        self.env, self.env_kwargs = create_env_robosuite(
            env_name=env_name,
            robots=robots_name.split("_"),
            controller_configs=controller_configs,
            camera_names=camera_names,
            camera_widths=self.camera_widths,
            camera_heights=self.camera_heights,
            enable_render=offscreen,
            onscreen=onscreen,
            **kwargs,  # Forward kwargs to create_env_robosuite
        )

        if ik_indicator:
            self.env = IKWrapper(self.env, ik_indicator=True)

        # ========= create converters first to get total DOFs =========
        # For now, assume single robot (multi-robot support can be added later)
        self.obs_action_converter: List[Gr00tObsActionConverter] = [
            Gr00tObsActionConverter(
                robot_model=robot_model,
                robosuite_robot_model=self.env.robots[i],
            )
            for i in range(len(self.env.robots))
        ]

        self.body_dofs = sum(converter.body_dof for converter in self.obs_action_converter)
        self.gripper_dofs = sum(converter.gripper_dof for converter in self.obs_action_converter)
        self.total_dofs = self.body_dofs + self.gripper_dofs
        self.body_nu = sum(converter.body_nu for converter in self.obs_action_converter)
        self.gripper_nu = sum(converter.gripper_nu for converter in self.obs_action_converter)
        self.total_nu = self.body_nu + self.gripper_nu

        # ========= create spaces to match total DOFs =========
        self.get_observation_space()
        self.get_action_space()

        self.enable_render = offscreen
        self.render_obs_key = f"{camera_names[0]}_image"
        self.render_cache = None

        self.dump_rollout_dataset_dir = dump_rollout_dataset_dir
        self.gr00t_exporter = None
        self.np_exporter = None

        self.rollout_hdf5 = rollout_hdf5
        self.rollout_trainset = rollout_trainset
        self.rollout_initial_state = {}

        self.verbose = False
        for k, v in self.observation_space.items():
            self.verbose and print("{OBS}", k, v)
        for k, v in self.action_space.items():
            self.verbose and print("{ACTION}", k, v)

        self.overridden_floating_base_action = None

    def get_observation_space(self):
        self.observation_space = spaces.Dict({})

        # Add all the observation spaces
        self.observation_space["time"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.observation_space["floating_base_pose"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.observation_space["floating_base_vel"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.observation_space["floating_base_acc"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.observation_space["body_q"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.body_dofs,), dtype=np.float32
        )
        self.observation_space["body_dq"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.body_dofs,), dtype=np.float32
        )
        self.observation_space["body_ddq"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.body_dofs,), dtype=np.float32
        )
        self.observation_space["body_tau_est"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.body_nu,), dtype=np.float32
        )
        self.observation_space["left_hand_q"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.gripper_dofs // 2,), dtype=np.float32
        )
        self.observation_space["left_hand_dq"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.gripper_dofs // 2,), dtype=np.float32
        )
        self.observation_space["left_hand_ddq"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.gripper_dofs // 2,), dtype=np.float32
        )
        self.observation_space["left_hand_tau_est"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.gripper_nu // 2,), dtype=np.float32
        )
        self.observation_space["right_hand_q"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.gripper_dofs // 2,), dtype=np.float32
        )
        self.observation_space["right_hand_dq"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.gripper_dofs // 2,), dtype=np.float32
        )
        self.observation_space["right_hand_ddq"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.gripper_dofs // 2,), dtype=np.float32
        )
        self.observation_space["right_hand_tau_est"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.gripper_nu // 2,), dtype=np.float32
        )

        self.observation_space["language.language_instruction"] = spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )

        # Add camera observation spaces
        for camera_name, w, h in zip(self.camera_names, self.camera_widths, self.camera_heights):
            k = self.camera_key_mapper.get_camera_config(camera_name)[0]
            self.observation_space[f"{k}_image"] = spaces.Box(
                low=0, high=255, shape=(h, w, 3), dtype=np.uint8
            )

        # Add extra privileged observation spaces
        if hasattr(self.env, "get_privileged_obs_keys"):
            for key, shape in self.env.get_privileged_obs_keys().items():
                self.observation_space[key] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
                )

        # Add robot-specific observation spaces
        if hasattr(self.env.robots[0].robot_model, "torso_body"):
            self.observation_space["secondary_imu_quat"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            )
            self.observation_space["secondary_imu_vel"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
            )

    def get_action_space(self):
        self.action_space = spaces.Dict(
            {"q": spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_dofs,), dtype=np.float32)}
        )

    def reset(self, seed=None, options=None):
        raw_obs, info = super().reset(seed=seed, options=options)
        obs = self.get_gr00t_observation(raw_obs)

        lang = self.env.get_ep_meta().get("lang", "")
        ROBOSUITE_DEFAULT_LOGGER.info(f"Instruction: {lang}")

        return obs, info

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # action={"q": xxx, "tau": xxx}
        for k, v in action.items():
            self.verbose and print("<ACTION>", k, v)

        joint_actoin_vec = action["q"]
        action_dict = {}
        for ii, robot in enumerate(self.env.robots):
            pf = robot.robot_model.naming_prefix
            _action_dict = self.obs_action_converter[ii].gr00t_to_robocasa_action_dict(
                joint_actoin_vec
            )
            action_dict.update({f"{pf}{k}": v for k, v in _action_dict.items()})
            if action.get("tau", None) is not None:
                _torque_dict = self.obs_action_converter[ii].gr00t_to_robocasa_action_dict(
                    action["tau"]
                )
                action_dict.update({f"{pf}{k}_tau": v for k, v in _torque_dict.items()})
            if self.overridden_floating_base_action is not None:
                action_dict["robot0_base"] = self.overridden_floating_base_action
        raw_obs, reward, terminated, truncated, info = super().step(action_dict)
        obs = self.get_gr00t_observation(raw_obs)

        for k, v in obs.items():
            self.verbose and print("<OBS>", k, v.shape if k.startswith("video.") else v)
        self.verbose = False

        return obs, reward, terminated, truncated, info

    def step_only_kinematics(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        joint_actoin_vec = action["q"]
        for ii, robot in enumerate(self.env.robots):
            joint_names = np.array(self.env.sim.model.joint_names)[robot._ref_joint_indexes]
            body_q = self.obs_action_converter[ii].gr00t_to_robocasa_joint_order(
                joint_names, joint_actoin_vec
            )
            self.env.sim.data.qpos[robot._ref_joint_pos_indexes] = body_q

            for side in ["left", "right"]:
                joint_names = np.array(self.env.sim.model.joint_names)[
                    robot._ref_joints_indexes_dict[side + "_gripper"]
                ]
                gripper_q = self.obs_action_converter[ii].gr00t_to_robocasa_joint_order(
                    joint_names, joint_actoin_vec
                )
                self.env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes[side]] = gripper_q

        mujoco.mj_forward(self.env.sim.model._model, self.env.sim.data._data)

        obs = self.force_update_observation()
        return obs, 0, False, False, {"success": False}

    def force_update_observation(self, timestep=0):
        raw_obs = self.env._get_observations(force_update=True, timestep=timestep)
        obs = self.get_basic_observation(raw_obs)
        obs = self.get_gr00t_observation(obs)
        return obs

    def get_basic_observation(self, raw_obs):
        # this function takes a lot of time, so we disable it for now
        # raw_obs.update(gather_robot_observations(self.env, format_gripper_space=False))

        # Image are in (H, W, C), flip it upside down
        def process_img(img):
            return np.copy(img[::-1, :, :])

        for obs_name, obs_value in raw_obs.items():
            if obs_name.endswith("_image"):
                # image observations
                raw_obs[obs_name] = process_img(obs_value)
            else:
                # non-image observations
                raw_obs[obs_name] = obs_value.astype(np.float32)

        # Return black image if rendering is disabled
        if not self.enable_render:
            for ii, name in enumerate(self.camera_names):
                raw_obs[f"{name}_image"] = np.zeros(
                    (self.camera_heights[ii], self.camera_widths[ii], 3), dtype=np.uint8
                )

        self.render_cache = raw_obs[self.render_obs_key]
        raw_obs["language"] = self.env.get_ep_meta().get("lang", "")

        return raw_obs

    def convert_body_q(self, q: np.ndarray) -> np.ndarray:
        # q is in the order of the joints
        robot = self.env.robots[0]
        joint_names = np.array(self.env.sim.model.joint_names)[robot._ref_joint_indexes]
        # this joint names are in the order of the obs_vec
        actuated_q = self.obs_action_converter[0].robocasa_to_gr00t_actuated_order(
            joint_names, q, "body"
        )
        return actuated_q

    def convert_gripper_q(self, q: np.ndarray, side: str = "left") -> np.ndarray:
        # q is in the order of the joints
        robot = self.env.robots[0]
        joint_names = np.array(self.env.sim.model.joint_names)[
            robot._ref_joints_indexes_dict[side + "_gripper"]
        ]
        actuated_q = self.obs_action_converter[0].robocasa_to_gr00t_actuated_order(
            joint_names, q, side + "_gripper"
        )
        return actuated_q

    def convert_gripper_tau(self, tau: np.ndarray, side: str = "left") -> np.ndarray:
        # tau is in the order of the actuators
        robot = self.env.robots[0]
        actuator_idx = robot._ref_actuators_indexes_dict[side + "_gripper"]
        actuated_joint_names = [
            self.env.sim.model.joint_id2name(self.env.sim.model.actuator_trnid[i][0])
            for i in actuator_idx
        ]
        actuated_tau = self.obs_action_converter[0].robocasa_to_gr00t_actuated_order(
            actuated_joint_names, tau, side + "_gripper"
        )
        return actuated_tau

    def get_gr00t_observation(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        obs = {}

        if self.env.sim.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            # If the first joint is a free joint, use this way to get the floating base data
            obs["floating_base_pose"] = self.env.sim.data.qpos[:7]
            obs["floating_base_vel"] = self.env.sim.data.qvel[:6]
            obs["floating_base_acc"] = self.env.sim.data.qacc[:6]
        else:
            # Otherwise, use self.env.sim.model to fetch the floating base pose
            root_body_id = self.env.sim.model.body_name2id("robot0_base")

            # Get position and orientation from body state
            root_pos = self.env.sim.data.body_xpos[root_body_id]
            root_quat = self.env.sim.data.body_xquat[root_body_id]  # quaternion in wxyz format

            # Combine position and quaternion to form 7-DOF pose
            obs["floating_base_pose"] = np.concatenate([root_pos, root_quat])
            # set vel and acc to 0
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["body_q"] = self.convert_body_q(raw_obs["robot0_joint_pos"])
        obs["body_dq"] = self.convert_body_q(raw_obs["robot0_joint_vel"])
        obs["body_ddq"] = self.convert_body_q(raw_obs["robot0_joint_acc"])

        obs["left_hand_q"] = self.convert_gripper_q(raw_obs["robot0_left_gripper_qpos"], "left")
        obs["left_hand_dq"] = self.convert_gripper_q(raw_obs["robot0_left_gripper_qvel"], "left")
        obs["left_hand_ddq"] = self.convert_gripper_q(raw_obs["robot0_left_gripper_qacc"], "left")
        obs["right_hand_q"] = self.convert_gripper_q(raw_obs["robot0_right_gripper_qpos"], "right")
        obs["right_hand_dq"] = self.convert_gripper_q(raw_obs["robot0_right_gripper_qvel"], "right")
        obs["right_hand_ddq"] = self.convert_gripper_q(
            raw_obs["robot0_right_gripper_qacc"], "right"
        )

        robot = self.env.robots[0]
        body_tau_idx_list = []
        left_gripper_tau_idx_list = []
        right_gripper_tau_idx_list = []
        for part_name, actuator_idx in robot._ref_actuators_indexes_dict.items():
            if "left_gripper" in part_name:
                left_gripper_tau_idx_list.extend(actuator_idx)
            elif "right_gripper" in part_name:
                right_gripper_tau_idx_list.extend(actuator_idx)
            elif "base" in part_name:
                assert (
                    len(actuator_idx) == 0 or robot.robot_model.default_base == "FloatingLeggedBase"
                )
            else:
                body_tau_idx_list.extend(actuator_idx)

        body_tau_idx_list = sorted(body_tau_idx_list)
        left_gripper_tau_idx_list = sorted(left_gripper_tau_idx_list)
        right_gripper_tau_idx_list = sorted(right_gripper_tau_idx_list)
        obs["body_tau_est"] = self.convert_body_q(
            self.env.sim.data.actuator_force[body_tau_idx_list]
        )
        obs["right_hand_tau_est"] = self.convert_gripper_tau(
            self.env.sim.data.actuator_force[right_gripper_tau_idx_list], "right"
        )
        obs["left_hand_tau_est"] = self.convert_gripper_tau(
            self.env.sim.data.actuator_force[left_gripper_tau_idx_list], "left"
        )

        obs["time"] = self.env.sim.data.time

        # Add camera images
        for ii, camera_name in enumerate(self.camera_names):
            mapped_camera_name = self.camera_key_mapper.get_camera_config(camera_name)[0]
            obs[f"{mapped_camera_name}_image"] = raw_obs[f"{camera_name}_image"]

        # Add privileged observations
        if hasattr(self.env, "get_privileged_obs_keys"):
            for key in self.env.get_privileged_obs_keys():
                obs[key] = raw_obs[key]

        # Add robot-specific observations
        if hasattr(self.env.robots[0].robot_model, "torso_body"):
            obs["secondary_imu_quat"] = raw_obs["robot0_torso_link_imu_quat"]
            obs["secondary_imu_vel"] = raw_obs["robot0_torso_link_imu_vel"]

        obs["language.language_instruction"] = raw_obs["language"]

        return obs
