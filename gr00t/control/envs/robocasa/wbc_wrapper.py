import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gr00t.control.main.teleop.run_sync_sim_data_collection import SyncSimDataCollectionConfig
from gr00t.control.policy.wbc_policy_factory import get_wbc_policy
from gr00t.control.robot_model.instantiation import get_robot_type_and_model
from gr00t.control.utils.sync_sim_utils import get_wbc_config


class WholeBodyControlWrapper(gym.Wrapper):
    def __init__(self, env, script_config):
        super().__init__(env)
        self.script_config = script_config
        self.wbc_policy = self.setup_wbc_policy()

        self._action_space = self._wbc_action_space()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.wbc_policy = self.setup_wbc_policy()
        self.wbc_policy.set_observation(obs)
        return obs, info

    def step(self, action):
        assert "target_upper_body_pose" in action, "Upper body joints are required for WBC"
        # TODO: make sure navigate cmd is always delta
        # if "navigate_command" not in action:
        #     action["navigate_command"] = np.zeros(3)
        wbc_goal = {}
        if "navigate_cmd" in action:
            wbc_goal["navigate_cmd"] = action["navigate_cmd"]

        if "base_height_command" in action:
            wbc_goal["base_height_command"] = action["base_height_command"]

        if "target_upper_body_pose" in action:
            wbc_goal["target_upper_body_pose"] = action["target_upper_body_pose"]

        self.wbc_policy.set_goal(wbc_goal)
        wbc_action = self.wbc_policy.get_action()

        # Hack due to env only takes q as action
        # this only effective when robot model is FloatingLeggedBase
        # cc: @Yu Fang
        self.unwrapped.overwrite_floating_base_action(wbc_goal["navigate_cmd"])
        result = super().step(wbc_action)
        self.wbc_policy.set_observation(result[0])
        return result

    def setup_wbc_policy(self):
        robot_type, robot_model = get_robot_type_and_model(
            self.script_config["robot"],
            enable_waist_ik=self.script_config.get("enable_waist", False),
        )
        config = SyncSimDataCollectionConfig.from_dict(self.script_config)
        config.update(
            {
                "save_img_obs": False,
                "ik_indicator": False,
                "enable_real_device": False,
                "replay_data_path": None,
            }
        )
        wbc_config = get_wbc_config(config)
        wbc_policy = get_wbc_policy(robot_type, robot_model, wbc_config, init_time=0.0)
        self.total_dofs = len(robot_model.get_joint_group_indices("upper_body"))
        wbc_policy.activate_policy()
        return wbc_policy

    def _wbc_action_space(self):
        # TODO: make this the same with WBC policy action space
        return spaces.Dict(
            {
                "target_upper_body_pose": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.total_dofs,), dtype=np.float32
                ),
                "navigate_cmd": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "base_height_command": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )
