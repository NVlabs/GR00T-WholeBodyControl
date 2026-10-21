from typing import Any, Dict, SupportsFloat, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from decoupled_wbc.control.main.constants import DEFAULT_BASE_HEIGHT, DEFAULT_NAV_CMD
from decoupled_wbc.control.main.teleop.configs.configs import SyncSimDataCollectionConfig
from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from decoupled_wbc.control.robot_model import RobotModel
from decoupled_wbc.control.robot_model.instantiation import get_robot_type_and_model


class WholeBodyControlWrapper(gym.Wrapper):
    """Gymnasium wrapper to integrate whole-body control for locomotion/manipulation sims."""

    def __init__(self, env, script_config):
        super().__init__(env)
        self.script_config = script_config
        self.script_config["robot"] = env.unwrapped.robot_name
        self.wbc_policy = self.setup_wbc_policy()
        self._action_space = self._wbc_action_space()

    @property
    def robot_model(self) -> RobotModel:
        """Return the robot model from the wrapped environment."""
        return self.env.unwrapped.robot_model  # type: ignore

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.wbc_policy = self.setup_wbc_policy()
        self.wbc_policy.set_observation(obs)
        return obs, info

    def step(self, action: Dict[str, Any]) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        action_dict = concat_action(self.robot_model, action)

        wbc_goal = {}
        for key in ["navigate_cmd", "base_height_command", "target_upper_body_pose"]:
            if key in action_dict:
                wbc_goal[key] = action_dict[key]

        self.wbc_policy.set_goal(wbc_goal)
        wbc_action = self.wbc_policy.get_action()

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
        wbc_config = config.load_wbc_yaml()
        wbc_config["upper_body_policy_type"] = "identity"
        wbc_policy = get_wbc_policy(robot_type, robot_model, wbc_config, init_time=0.0)
        self.total_dofs = len(robot_model.get_joint_group_indices("upper_body"))
        wbc_policy.activate_policy()
        return wbc_policy

    def _get_joint_group_size(self, group_name: str) -> int:
        """Return the number of joints in a group, cached since lookup is static."""
        if not hasattr(self, "_joint_group_size_cache"):
            self._joint_group_size_cache = {}
        if group_name not in self._joint_group_size_cache:
            self._joint_group_size_cache[group_name] = len(
                self.robot_model.get_joint_group_indices(group_name)
            )
        return self._joint_group_size_cache[group_name]

    def _wbc_action_space(self) -> spaces.Dict:
        action_space: Dict[str, spaces.Space] = {
            "action.navigate_command": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "action.base_height_command": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "action.left_hand": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._get_joint_group_size("left_hand"),),
                dtype=np.float32,
            ),
            "action.right_hand": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._get_joint_group_size("right_hand"),),
                dtype=np.float32,
            ),
            "action.left_arm": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._get_joint_group_size("left_arm"),),
                dtype=np.float32,
            ),
            "action.right_arm": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._get_joint_group_size("right_arm"),),
                dtype=np.float32,
            ),
        }
        if (
            "waist"
            in self.robot_model.supplemental_info.joint_groups["upper_body_no_hands"]["groups"]  # type: ignore[attr-defined]
        ):
            action_space["action.waist"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._get_joint_group_size("waist"),),
                dtype=np.float32,
            )
        return spaces.Dict(action_space)


def concat_action(robot_model: RobotModel, goal: Dict[str, Any]) -> Dict[str, Any]:
    """Combine individual joint-group targets into the upper-body action vector."""
    processed_goal = {}
    for key, value in goal.items():
        processed_goal[key.replace("action.", "")] = value

    first_value = next(iter(processed_goal.values()))
    action = np.zeros(first_value.shape[:-1] + (robot_model.num_dofs,))

    action_dict = {}
    action_dict["navigate_cmd"] = processed_goal.pop("navigate_command", DEFAULT_NAV_CMD)
    action_dict["base_height_command"] = np.array(
        processed_goal.pop("base_height_command", DEFAULT_BASE_HEIGHT)
    )

    for joint_group, value in processed_goal.items():
        indices = robot_model.get_joint_group_indices(joint_group)
        action[..., indices] = value

    upper_body_indices = robot_model.get_joint_group_indices("upper_body")
    action = action[..., upper_body_indices]
    action_dict["target_upper_body_pose"] = action
    return action_dict


def prepare_observation_for_eval(robot_model: RobotModel, obs: dict) -> dict:
    """Add joint-group slices to an observation dict (real + sim evaluation helper)."""
    assert "q" in obs, "q is not in the observation"

    whole_q = obs["q"]
    assert whole_q.shape[-1] == robot_model.num_joints, "q has wrong shape"

    left_arm_q = whole_q[..., robot_model.get_joint_group_indices("left_arm")]
    right_arm_q = whole_q[..., robot_model.get_joint_group_indices("right_arm")]
    waist_q = whole_q[..., robot_model.get_joint_group_indices("waist")]
    left_leg_q = whole_q[..., robot_model.get_joint_group_indices("left_leg")]
    right_leg_q = whole_q[..., robot_model.get_joint_group_indices("right_leg")]
    left_hand_q = whole_q[..., robot_model.get_joint_group_indices("left_hand")]
    right_hand_q = whole_q[..., robot_model.get_joint_group_indices("right_hand")]

    obs["state.left_arm"] = left_arm_q
    obs["state.right_arm"] = right_arm_q
    obs["state.waist"] = waist_q
    obs["state.left_leg"] = left_leg_q
    obs["state.right_leg"] = right_leg_q
    obs["state.left_hand"] = left_hand_q
    obs["state.right_hand"] = right_hand_q

    return obs


def prepare_gym_space_for_eval(
    robot_model: RobotModel, gym_space: gym.spaces.Dict
) -> gym.spaces.Dict:
    """Extend a gym Dict space with the joint-group keys used during evaluation."""
    left_arm_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("left_arm")),),
    )
    right_arm_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("right_arm")),),
    )
    waist_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("waist")),),
    )
    left_leg_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("left_leg")),),
    )
    right_leg_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("right_leg")),),
    )
    left_hand_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("left_hand")),),
    )
    right_hand_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(robot_model.get_joint_group_indices("right_hand")),),
    )

    gym_space["state.left_arm"] = left_arm_space
    gym_space["state.right_arm"] = right_arm_space
    gym_space["state.waist"] = waist_space
    gym_space["state.left_leg"] = left_leg_space
    gym_space["state.right_leg"] = right_leg_space
    gym_space["state.left_hand"] = left_hand_space
    gym_space["state.right_hand"] = right_hand_space

    return gym_space
