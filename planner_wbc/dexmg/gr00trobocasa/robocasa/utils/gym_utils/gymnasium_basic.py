import datetime, uuid
from copy import deepcopy
import gymnasium as gym
import h5py
import numpy as np
import os
import robocasa.utils.transform_utils as T
import robosuite
from gymnasium import spaces
from robocasa.models.robots import (
    GR00T_ROBOCASA_ENVS_GR1_ARMS_ONLY,
    GR00T_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST,
    GR00T_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY,
    gather_robot_observations,
    make_key_converter,
)
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import HybridMobileBase
from robocasa.wrappers.ik_wrapper import IKWrapper


ALLOWED_LANGUAGE_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
)


def create_env_robosuite(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    controller_configs=None,
    camera_names=[
        "egoview",
        "robot0_eye_in_left_hand",
        "robot0_eye_in_right_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    render_camera=None,
    enable_render=True,
    seed=None,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
    onscreen=False,
    renderer="mujoco",
    translucent_robot=False,
    control_freq=20,
):
    if controller_configs is None:
        controller_configs = load_composite_controller_config(
            controller=None,
            robot=robots if isinstance(robots, str) else robots[0],
        )
    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=onscreen,
        has_offscreen_renderer=enable_render,
        renderer=renderer,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=enable_render,
        camera_depths=False,
        seed=seed,
        translucent_robot=translucent_robot,
        control_freq=control_freq,
        render_camera=render_camera,
        randomize_cameras=randomize_cameras,
    )

    env = robosuite.make(**env_kwargs)
    return env, env_kwargs


class RoboCasaEnv(gym.Env):
    def __init__(
        self,
        env_name=None,
        robots_name=None,
        camera_names=None,
        camera_widths=None,
        camera_heights=None,
        enable_render=True,
        render_camera=None,
        onscreen=False,
        dump_rollout_dataset_dir=None,
        rollout_hdf5=None,
        rollout_trainset=None,
        controller_configs=None,
        ik_indicator=False,
        **kwargs,  # Accept additional kwargs
    ):
        self.key_converter = make_key_converter(robots_name)
        (
            _,
            camera_names,
            default_camera_widths,
            default_camera_heights,
        ) = self.key_converter.get_camera_config()

        if camera_widths is None:
            camera_widths = default_camera_widths
        if camera_heights is None:
            camera_heights = default_camera_heights

        if controller_configs is not None:
            controller_configs = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../",
                controller_configs,
            )
        controller_configs = load_composite_controller_config(
            controller=controller_configs,
            robot=robots_name.split("_")[0],
        )
        if (
            robots_name in GR00T_ROBOCASA_ENVS_GR1_ARMS_ONLY
            or robots_name in GR00T_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST
            or robots_name in GR00T_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY
        ):
            controller_configs["type"] = "BASIC"
            controller_configs["composite_controller_specific_configs"] = {}
            controller_configs["control_delta"] = False

        self.env, self.env_kwargs = create_env_robosuite(
            env_name=env_name,
            robots=robots_name.split("_"),
            controller_configs=controller_configs,
            camera_names=camera_names,
            camera_widths=camera_widths,
            camera_heights=camera_heights,
            render_camera=render_camera,
            enable_render=enable_render,
            onscreen=onscreen,
            **kwargs,  # Forward kwargs to create_env_robosuite
        )

        if ik_indicator:
            self.env = IKWrapper(self.env, ik_indicator=True)

        # TODO: the following info should be output by gr00trobocasa
        self.camera_names = camera_names
        self.camera_widths = camera_widths
        self.camera_heights = camera_heights
        self.enable_render = enable_render and not onscreen
        self.render_obs_key = f"{camera_names[0]}_image"
        self.render_cache = None

        # setup spaces
        action_space = spaces.Dict()
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            for part_name, controller in cc.part_controllers.items():
                min_value, max_value = -1, 1
                start_idx, end_idx = cc._action_split_indexes[part_name]
                shape = [end_idx - start_idx]
                this_space = spaces.Box(
                    low=min_value, high=max_value, shape=shape, dtype=np.float32
                )
                action_space[f"{pf}{part_name}"] = this_space
            if isinstance(cc, HybridMobileBase):
                this_space = spaces.Discrete(2)
                action_space[f"{pf}base_mode"] = this_space

            action_space = spaces.Dict(action_space)
            self.action_space = action_space

        # Calling env._get_observations(force_update=True) will pollute the observables.
        # This is safe because it is only called from the constructor and will be overwritten later in reset().
        obs = (
            self.env.viewer._get_observations(force_update=True)
            if self.env.viewer_get_obs
            else self.env._get_observations(force_update=True)
        )
        obs.update(gather_robot_observations(self.env))
        observation_space = spaces.Dict()
        for obs_name, obs_value in obs.items():
            shape = list(obs_value.shape)
            if obs_name.endswith("_image"):
                continue
            min_value, max_value = -1, 1
            this_space = spaces.Box(low=min_value, high=max_value, shape=shape, dtype=np.float32)
            observation_space[obs_name] = this_space

        for camera_name in camera_names:
            shape = [camera_heights, camera_widths, 3]
            this_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
            observation_space[f"{camera_name}_image"] = this_space

        observation_space["language"] = spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )

        self.observation_space = observation_space

        self.dump_rollout_dataset_dir = dump_rollout_dataset_dir
        self.gr00t_exporter = None
        self.np_exporter = None

        self.rollout_hdf5 = rollout_hdf5
        self.rollout_trainset = rollout_trainset
        self.rollout_initial_state = {}

    def begin_rollout_dataset_dump(self):
        if self.dump_rollout_dataset_dir is not None:
            gr00t_env_meta = dict(
                env_name=self.env_kwargs["env_name"],
                env_version=robosuite.__version__,
                type=1,
                env_kwargs=deepcopy(self.env_kwargs),
            )
            gr00t_dir = os.path.join(
                self.dump_rollout_dataset_dir,
                self.env_kwargs["env_name"]
                + "_"
                + "_".join(robot.name for robot in self.env.robots)
                + "_Env",
                f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{str(uuid.uuid4())[:8]}",
            )
            if not os.path.exists(gr00t_dir):
                os.makedirs(gr00t_dir, exist_ok=True)
            self.gr00t_exporter = Gr00tExporter(
                gr00t_dir,
                self.env,
                gr00t_env_meta,
            )
            self.np_exporter = {}
            self.np_exporter["success"] = []
            for signal in self.env.get_subtask_term_signals():
                self.np_exporter[signal] = []

    def process_rollout_dataset_dump_before_step(self, env_action):
        if self.gr00t_exporter is not None:
            self.gr00t_exporter.add_record_before_step(self.env, env_action)
            self.np_exporter["success"].append(self.env.reward())
            for signal, value in self.env.get_subtask_term_signals().items():
                self.np_exporter[signal].append(value)

    def process_rollout_dataset_dump_after_step(self, env_action):
        if self.gr00t_exporter is not None:
            self.gr00t_exporter.add_record_after_step(self.env, env_action)

    def complete_rollout_dataset_dump(self):
        if self.gr00t_exporter is not None and self.np_exporter is not None:
            data = {k: np.array(v, dtype=np.float32) for k, v in self.np_exporter.items()}
            np.savez(os.path.join(self.gr00t_exporter.gr00t_dir, "rewards.npz"), **data)
            self.gr00t_exporter.finish()
            self.np_exporter = None
            self.gr00t_exporter = None

    def get_basic_observation(self, raw_obs):
        raw_obs.update(gather_robot_observations(self.env))

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
            for name in self.camera_names:
                raw_obs[f"{name}_image"] = np.zeros(
                    (self.camera_heights, self.camera_widths, 3), dtype=np.uint8
                )

        self.render_cache = raw_obs[self.render_obs_key]
        raw_obs["language"] = self.env.get_ep_meta().get("lang", "")

        return raw_obs

    def reset(self, seed=None, options=None):
        if seed is not None and self.rollout_trainset is not None and seed < self.rollout_trainset:
            if seed not in self.rollout_initial_state:
                f = h5py.File(self.rollout_hdf5, "r")
                demos = list(f["data"].keys())
                inds = np.argsort([int(elem[5:]) for elem in demos])
                demos = [demos[i] for i in inds]
                ep = demos[seed]
                self.rollout_initial_state[seed] = {
                    "states": f["data/{}/states".format(ep)][0],
                    "ep_meta": f["data/{}".format(ep)].attrs.get("ep_meta", None),
                    "model": f["data/{}".format(ep)].attrs["model_file"],
                }
            reset_to(self.env, self.rollout_initial_state[seed])
            obs = None
        else:
            # NOTE: self.env can be either a robosuite environment or ik wrapper
            if isinstance(self.env, IKWrapper):
                self.env.unwrapped.seed = seed
                self.env.unwrapped.rng = np.random.default_rng(seed=seed)
            else:
                self.env.seed = seed
                self.env.rng = np.random.default_rng(seed=seed)
            raw_obs = self.env.reset()
            obs = self.get_basic_observation(raw_obs)

        info = {}
        info["success"] = False
        info["intermediate_signals"] = {}

        self.complete_rollout_dataset_dump()
        self.begin_rollout_dataset_dump()

        return obs, info

    def step(self, action_dict):
        env_action = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            action = np.zeros(cc.action_limits[0].shape)
            for part_name, controller in cc.part_controllers.items():
                start_idx, end_idx = cc._action_split_indexes[part_name]
                act = action_dict.pop(f"{pf}{part_name}")
                action[start_idx:end_idx] = act

                # set external torque compensation if available
                if "gripper" not in part_name and controller.use_external_torque_compensation:
                    controller.external_torque_compensation = action_dict.pop(
                        f"{pf}{part_name}_tau"
                    )
            if isinstance(cc, HybridMobileBase):
                action[-1] = action_dict.pop(f"{pf}base_mode")
            env_action.append(action)

        env_action = np.concatenate(env_action)

        self.process_rollout_dataset_dump_before_step(env_action)
        raw_obs, reward, done, info = self.env.step(env_action)
        self.process_rollout_dataset_dump_after_step(env_action)

        obs = self.get_basic_observation(raw_obs)

        truncated = False

        info["success"] = reward > 0
        info["intermediate_signals"] = {}
        if hasattr(self.env, "_get_intermediate_signals"):
            info["intermediate_signals"] = self.env._get_intermediate_signals()

        if hasattr(self.env, "_get_auxiliary_states"):
            info["auxiliary_states"] = self.env._get_auxiliary_states()

        if hasattr(self.env, "obj_body_id"):
            for obj_name in self.env.obj_body_id.keys():
                info[f"{obj_name}_pos"] = list(
                    self.env.sim.data.body_xpos[self.env.obj_body_id[obj_name]]
                )
                info[f"{obj_name}_quat_xyzw"] = list(
                    T.convert_quat(
                        np.array(self.env.sim.data.body_xquat[self.env.obj_body_id[obj_name]]),
                        to="xyzw",
                    )
                )

        return obs, reward, done, truncated, info

    def render(self):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        return self.render_cache

    def close(self):
        self.complete_rollout_dataset_dump()
        self.env.close()

    def get_actuator_names(self):
        joint_names = self.get_joint_names()

        model = self.env.sim.model

        actuated_joint_names = []
        for ii in range(model.nu):
            joint_id = model.actuator_trnid[ii, 0]
            actuated_joint_names.append(model.joint(joint_id).name)

        actuator_names = {}
        actuator_indices = {}
        for part_name, _joint_names in joint_names.items():
            actuator_names[part_name] = []
            actuator_indices[part_name] = []
            for _jn in _joint_names:
                if _jn in actuated_joint_names:
                    actuator_names[part_name].append(_jn)
                    actuator_indices[part_name].append(actuated_joint_names.index(_jn))
            actuator_names[part_name] = (
                np.array(actuator_names[part_name])[np.argsort(actuator_indices[part_name])]
            ).tolist()
        return actuator_names

    def get_joint_names(self):
        joint_names = {}
        for _robot in self.env.robots:
            cc = _robot.composite_controller
            for part_name, part_controller in cc.part_controllers.items():
                if "gripper" in part_name:
                    joint_names[part_name] = part_controller.joint_index
                else:
                    joint_names[part_name] = part_controller.joint_names
        return joint_names
