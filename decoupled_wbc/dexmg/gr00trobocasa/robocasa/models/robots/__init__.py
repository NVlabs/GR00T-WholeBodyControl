from numpy import dtype, ndarray


from typing import Any


import mujoco
import numpy as np

from robosuite.controllers.parts.generic.joint_pos import JointPositionController
from robosuite.controllers.parts.gripper.simple_grip import SimpleGripController
from robosuite.controllers.composite.composite_controller import HybridMobileBase
from robosuite.models.grippers import (
    PandaGripper,
    InspireLeftHand,
    InspireRightHand,
    FourierLeftHand,
    FourierRightHand,
)

from robocasa.models.grippers import G1ThreeFingerLeftHand, G1ThreeFingerRightHand

from .manipulators import *


# This function returns in the order of the gripper XML's actuators.
def unformat_gripper_space(gripper, formatted_action):
    formatted_action = np.array(formatted_action)
    if isinstance(gripper, InspireLeftHand) or isinstance(gripper, InspireRightHand):
        action = formatted_action[[0, 2, 4, 6, 8, 11]]
    elif isinstance(gripper, FourierLeftHand) or isinstance(gripper, FourierRightHand):
        action = formatted_action[[0, 1, 4, 6, 8, 10]]
    elif isinstance(gripper, PandaGripper):
        action = formatted_action
    else:
        raise TypeError
    return action


def remove_mimic_joints(gripper, formatted_action) -> ndarray[Any, dtype[Any]]:
    """_summary_

    This is to remove the mimic joints from the actuator-ordered action
    """
    formatted_action = np.array(formatted_action)
    if isinstance(gripper, InspireLeftHand) or isinstance(gripper, InspireRightHand):
        raise NotImplementedError
    elif isinstance(gripper, FourierLeftHand) or isinstance(gripper, FourierRightHand):
        action = formatted_action[[0, 2, 4, 6, 8, 10]]
    elif isinstance(gripper, G1ThreeFingerLeftHand) or isinstance(gripper, G1ThreeFingerRightHand):
        action = formatted_action
    elif isinstance(gripper, PandaGripper):
        action = formatted_action
    return action


def _reconstruct_latest_action_gr1_impl(robot, verbose=False):
    action_dict = {}
    cc = robot.composite_controller
    pf = robot.robot_model.naming_prefix

    for part_name, controller in cc.part_controllers.items():
        if isinstance(controller, JointPositionController):
            assert controller.input_type == "absolute"
            act = controller.goal_qpos
        elif isinstance(controller, SimpleGripController):
            assert not controller.use_action_scaling
            act = unformat_gripper_space(cc.grippers[part_name], controller.goal_qvel)
        else:
            raise TypeError
        action_dict[f"{pf}{part_name}"] = act

    if verbose:
        print("Actions:", [(k, len(action_dict[k])) for k in action_dict])

    return action_dict


def _reconstruct_latest_action_panda_impl(robot, action, verbose=False):
    action_dict = {}
    cc = robot.composite_controller
    pf = robot.robot_model.naming_prefix

    for part_name, controller in cc.part_controllers.items():
        start_idx, end_idx = cc._action_split_indexes[part_name]
        act = action[start_idx:end_idx]
        # split action deeper for OSC
        action_dict[f"{pf}{part_name}"] = act

    if verbose:
        print("Actions:", [(k, len(action_dict[k])) for k in action_dict])

    return action_dict


# This function tries to get all joints value, but sometimes we want to keep original action
# The function must be called after robot.control()
def reconstruct_latest_actions(env, actions=None, verbose=False):
    if actions is None:
        actions = np.zeros(env.action_dim)
    action_dict = {}
    cutoff = 0
    for robot in env.robots:
        cc = robot.composite_controller
        pf = robot.robot_model.naming_prefix
        robot_action = actions[cutoff : cutoff + robot.action_dim]
        cutoff += robot.action_dim
        if "GR1" in robot.name:
            action_dict.update(_reconstruct_latest_action_gr1_impl(robot))
        elif "Panda" in robot.name:
            action_dict.update(_reconstruct_latest_action_panda_impl(robot, robot_action))
        else:
            raise ValueError(f"Unknown robot name: {robot.name}")
        if isinstance(cc, HybridMobileBase):
            action_dict[f"{pf}base_mode"] = robot_action[-1]

    if verbose:
        print("Actions:", [(k, len(action_dict[k])) for k in action_dict])

    return action_dict


def gather_robot_observations(env, verbose=False, format_gripper_space=True):
    observations = {}

    for robot_id, robot in enumerate(env.robots):
        sim = robot.sim
        gripper_names = {robot.get_gripper_name(arm): robot.gripper[arm] for arm in robot.arms}
        for part_name, indexes in robot._ref_joints_indexes_dict.items():
            qpos_values = []
            for joint_id in indexes:
                qpos_addr = sim.model.jnt_qposadr[joint_id]
                # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtjoint
                joint_type = sim.model.jnt_type[joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_size = 7  # Free joint has 7 DOFs
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    qpos_size = 4  # Ball joint has 4 DOFs (quaternion)
                else:
                    qpos_size = 1  # Revolute or prismatic joint has 1 DOF
                qpos_values = np.append(
                    qpos_values, sim.data.qpos[qpos_addr : qpos_addr + qpos_size]
                )
            if part_name in gripper_names.keys():
                gripper = gripper_names[part_name]
                # Reverse the order to match the real robot
                # TODO: how to adapt to other grippers?
                if format_gripper_space:
                    qpos_values = unformat_gripper_space(gripper, qpos_values)[::-1]
            if len(qpos_values) > 0:
                observations[f"robot{robot_id}_{part_name}"] = qpos_values

    if verbose:
        print("States:", [(k, len(observations[k])) for k in observations])

    return observations


from gymnasium import spaces


class RobotKeyConverter:
    @classmethod
    def get_camera_config(cls):
        raise NotImplementedError

    @classmethod
    def map_obs(cls, input_obs):
        raise NotImplementedError

    @classmethod
    def map_action(cls, input_action):
        raise NotImplementedError

    @classmethod
    def unmap_action(cls, input_action):
        raise NotImplementedError

    @classmethod
    def get_metadata(cls, name):
        raise NotImplementedError

    @classmethod
    def map_obs_in_eval(cls, input_obs):
        output_obs = {}
        mapped_obs = cls.map_obs(input_obs)
        for k, v in mapped_obs.items():
            assert k.startswith("hand.") or k.startswith("body.")
            output_obs["state." + k[5:]] = v
        return output_obs

    @classmethod
    def get_missing_keys_in_dumping_dataset(cls):
        return {}

    @classmethod
    def convert_to_float64(cls, input):
        for k, v in input.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float32:
                input[k] = v.astype(np.float64)
        return input

    @classmethod
    def deduce_observation_space(cls, env):
        # Calling env._get_observations(force_update=True) will pollute the observables.
        # This is safe because it is only called from the constructor and will be overwritten later in reset().
        obs = (
            env.viewer._get_observations(force_update=True)
            if env.viewer_get_obs
            else env._get_observations(force_update=True)
        )
        obs.update(gather_robot_observations(env))
        obs = cls.map_obs(obs)
        observation_space = spaces.Dict()

        for k, v in obs.items():
            if k.startswith("hand.") or k.startswith("body."):
                observation_space["state." + k[5:]] = spaces.Box(
                    low=-1, high=1, shape=(len(v),), dtype=np.float32
                )
            else:
                raise ValueError(f"Unknown key: {k}")

        return observation_space

    @classmethod
    def deduce_action_space(cls, env):
        action = cls.map_action(reconstruct_latest_actions(env))
        action_space = spaces.Dict()
        for k, v in action.items():
            if isinstance(v, np.int64):
                action_space["action." + k[5:]] = spaces.Discrete(2)
            elif isinstance(v, np.ndarray):
                action_space["action." + k[5:]] = spaces.Box(
                    low=-1, high=1, shape=(len(v),), dtype=np.float32
                )
            else:
                raise ValueError(f"Unknown type: {type(v)}")
        return action_space


class GR1ArmsOnlyKeyConverter(RobotKeyConverter):
    @classmethod
    def get_camera_config(cls):
        mapped_names = ["video.ego_view_pad_res256_freq20"]
        camera_names = [
            "egoview",
        ]
        camera_widths, camera_heights = 1280, 800
        return mapped_names, camera_names, camera_widths, camera_heights

    @classmethod
    def map_obs(cls, input_obs):
        output_obs = type(input_obs)()
        output_obs = {
            "hand.right_hand": input_obs["robot0_right_gripper"],
            "hand.left_hand": input_obs["robot0_left_gripper"],
            "body.right_arm": input_obs["robot0_right"],
            "body.left_arm": input_obs["robot0_left"],
        }
        return output_obs

    @classmethod
    def map_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "hand.left_hand": input_action["robot0_left_gripper"],
            "hand.right_hand": input_action["robot0_right_gripper"],
            "body.left_arm": input_action["robot0_left"],
            "body.right_arm": input_action["robot0_right"],
        }
        return output_action

    @classmethod
    def unmap_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "robot0_left_gripper": input_action["action.left_hand"],
            "robot0_right_gripper": input_action["action.right_hand"],
            "robot0_left": input_action["action.left_arm"],
            "robot0_right": input_action["action.right_arm"],
        }
        return output_action

    @classmethod
    def get_missing_keys_in_dumping_dataset(cls):
        return {
            "body.waist": np.zeros(3, dtype=np.float64),
            "body.neck": np.zeros(3, dtype=np.float64),
            "body.right_leg": np.zeros(6, dtype=np.float64),
            "body.left_leg": np.zeros(6, dtype=np.float64),
        }

    @classmethod
    def get_metadata(cls, name):
        return {
            "absolute": True,
            "rotation_type": None,
        }


class GR1ArmsAndWaistKeyConverter(RobotKeyConverter):
    @classmethod
    def get_camera_config(cls):
        mapped_names = ["video.ego_view_pad_res256_freq20"]
        camera_names = [
            "egoview",
        ]
        camera_widths, camera_heights = 1280, 800
        return mapped_names, camera_names, camera_widths, camera_heights

    @classmethod
    def map_obs(cls, input_obs):
        output_obs = type(input_obs)()
        output_obs = {
            "hand.right_hand": input_obs["robot0_right_gripper"],
            "hand.left_hand": input_obs["robot0_left_gripper"],
            "body.right_arm": input_obs["robot0_right"],
            "body.left_arm": input_obs["robot0_left"],
            "body.waist": input_obs["robot0_torso"],
        }
        return output_obs

    @classmethod
    def map_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "hand.left_hand": input_action["robot0_left_gripper"],
            "hand.right_hand": input_action["robot0_right_gripper"],
            "body.left_arm": input_action["robot0_left"],
            "body.right_arm": input_action["robot0_right"],
            "body.waist": input_action["robot0_torso"],
        }
        return output_action

    @classmethod
    def unmap_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "robot0_left_gripper": input_action["action.left_hand"],
            "robot0_right_gripper": input_action["action.right_hand"],
            "robot0_left": input_action["action.left_arm"],
            "robot0_right": input_action["action.right_arm"],
            "robot0_torso": input_action["action.waist"],
        }
        return output_action

    @classmethod
    def get_missing_keys_in_dumping_dataset(cls):
        return {
            "body.neck": np.zeros(3, dtype=np.float64),
            "body.right_leg": np.zeros(6, dtype=np.float64),
            "body.left_leg": np.zeros(6, dtype=np.float64),
        }

    @classmethod
    def get_metadata(cls, name):
        return {
            "absolute": True,
            "rotation_type": None,
        }


class GR1FixedLowerBodyKeyConverter(RobotKeyConverter):
    @classmethod
    def get_camera_config(cls):
        mapped_names = ["video.agentview_pad_res256_freq20"]
        camera_names = [
            "agentview",
        ]
        camera_widths, camera_heights = 1280, 800
        return mapped_names, camera_names, camera_widths, camera_heights

    @classmethod
    def map_obs(cls, input_obs):
        output_obs = type(input_obs)()
        output_obs = {
            "hand.right_hand": input_obs["robot0_right_gripper"],
            "hand.left_hand": input_obs["robot0_left_gripper"],
            "body.right_arm": input_obs["robot0_right"],
            "body.left_arm": input_obs["robot0_left"],
            "body.waist": input_obs["robot0_torso"],
            "body.neck": input_obs["robot0_head"],
        }
        return output_obs

    @classmethod
    def map_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "hand.left_hand": input_action["robot0_left_gripper"],
            "hand.right_hand": input_action["robot0_right_gripper"],
            "body.left_arm": input_action["robot0_left"],
            "body.right_arm": input_action["robot0_right"],
            "body.waist": input_action["robot0_torso"],
            "body.neck": input_action["robot0_head"],
        }
        return output_action

    @classmethod
    def unmap_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "robot0_left_gripper": input_action["action.left_hand"],
            "robot0_right_gripper": input_action["action.right_hand"],
            "robot0_left": input_action["action.left_arm"],
            "robot0_right": input_action["action.right_arm"],
            "robot0_torso": input_action["action.waist"],
            "robot0_head": input_action["action.neck"],
        }
        return output_action

    @classmethod
    def get_missing_keys_in_dumping_dataset(cls):
        return {
            "body.right_leg": np.zeros(6, dtype=np.float64),
            "body.left_leg": np.zeros(6, dtype=np.float64),
        }

    @classmethod
    def get_metadata(cls, name):
        return {
            "absolute": True,
            "rotation_type": None,
        }


class PandaOmronKeyConverter(RobotKeyConverter):
    @classmethod
    def get_camera_config(cls):
        mapped_names = [
            "video.res256_image_side_0",
            "video.res256_image_side_1",
            "video.res256_image_wrist_0",
        ]
        camera_names = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]
        camera_widths, camera_heights = 512, 512
        return mapped_names, camera_names, camera_widths, camera_heights

    @classmethod
    def map_obs(cls, input_obs):
        output_obs = type(input_obs)()
        output_obs = {
            "hand.gripper_qpos": input_obs["robot0_gripper_qpos"],
            "body.base_position": input_obs["robot0_base_pos"],
            "body.base_rotation": input_obs["robot0_base_quat"],
            "body.end_effector_position_relative": input_obs["robot0_base_to_eef_pos"],
            "body.end_effector_rotation_relative": input_obs["robot0_base_to_eef_quat"],
        }
        return output_obs

    @classmethod
    def map_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "hand.gripper_close": (
                np.int64(0) if input_action["robot0_right_gripper"] < 0 else np.int64(1)
            ),
            "body.end_effector_position": input_action["robot0_right"][..., 0:3],
            "body.end_effector_rotation": input_action["robot0_right"][..., 3:6],
            "body.base_motion": np.concatenate(
                (
                    input_action["robot0_base"],
                    input_action["robot0_torso"],
                ),
                axis=-1,
            ),
            "body.control_mode": (
                np.int64(0) if input_action["robot0_base_mode"] < 0 else np.int64(1)
            ),
        }
        return output_action

    @classmethod
    def unmap_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "robot0_right_gripper": (-1.0 if input_action["action.gripper_close"] < 0.5 else 1.0),
            "robot0_right": np.concatenate(
                (
                    input_action["action.end_effector_position"],
                    input_action["action.end_effector_rotation"],
                ),
                axis=-1,
            ),
            "robot0_base": input_action["action.base_motion"][..., 0:3],
            "robot0_torso": input_action["action.base_motion"][..., 3:4],
            "robot0_base_mode": (-1.0 if input_action["action.control_mode"] < 0.5 else 1.0),
        }
        return output_action

    @classmethod
    def get_metadata(cls, name):
        # This function and import is only enabled in gr00t exporter
        from decoupled_wbc.data.schema.raw.metadata import RotationType

        if name in [
            "body.base_position",
            "body.end_effector_position_relative",
            "body.end_effector_position",
        ]:
            return {
                "absolute": False,
                "rotation_type": None,
            }
        elif name in ["body.base_rotation", "body.end_effector_rotation_relative"]:
            return {
                "absolute": False,
                "rotation_type": RotationType.QUATERNION,
            }
        elif name in ["body.end_effector_rotation"]:
            return {
                "absolute": False,
                "rotation_type": RotationType.AXIS_ANGLE,
            }
        else:
            return {
                "absolute": True,
                "rotation_type": None,
            }


class PandaPandaKeyConverter(RobotKeyConverter):
    @classmethod
    def get_camera_config(cls):
        mapped_names = [
            "video.agentview_pad_res256_freq20",
            "video.robot0_eye_in_hand_pad_res256_freq20",
            "video.robot1_eye_in_hand_pad_res256_freq20",
        ]
        camera_names = [
            "agentview",
            "robot0_eye_in_hand",
            "robot1_eye_in_hand",
        ]
        camera_widths, camera_heights = 1280, 800
        return mapped_names, camera_names, camera_widths, camera_heights

    @classmethod
    def map_obs(cls, input_obs):
        output_obs = type(input_obs)()
        output_obs = {
            "hand.right_gripper_qpos": input_obs["robot0_right_gripper"],
            "hand.left_gripper_qpos": input_obs["robot1_right_gripper"],
            "body.right_arm_eef_pos": input_obs["robot0_eef_pos"],
            "body.right_arm_eef_quat": input_obs["robot0_eef_quat"],
            "body.right_arm_joint_pos": input_obs["robot0_joint_pos"],
            "body.left_arm_eef_pos": input_obs["robot1_eef_pos"],
            "body.left_arm_eef_quat": input_obs["robot1_eef_quat"],
            "body.left_arm_joint_pos": input_obs["robot1_joint_pos"],
        }
        return output_obs

    @classmethod
    def map_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "hand.right_gripper_close": (
                np.int64(0) if input_action["robot0_right_gripper"] < 0 else np.int64(1)
            ),
            "hand.left_gripper_close": (
                np.int64(0) if input_action["robot1_right_gripper"] < 0 else np.int64(1)
            ),
            "body.right_arm_eef_pos": input_action["robot0_right"][..., 0:3],
            "body.right_arm_eef_rot": input_action["robot0_right"][..., 3:6],
            "body.left_arm_eef_pos": input_action["robot1_right"][..., 0:3],
            "body.left_arm_eef_rot": input_action["robot1_right"][..., 3:6],
        }
        return output_action

    @classmethod
    def unmap_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "robot0_right_gripper": (
                -1.0 if input_action["action.right_gripper_close"] < 0.5 else 1.0
            ),
            "robot1_right_gripper": (
                -1.0 if input_action["action.left_gripper_close"] < 0.5 else 1.0
            ),
            "robot0_right": np.concatenate(
                (
                    input_action["action.right_arm_eef_pos"],
                    input_action["action.right_arm_eef_rot"],
                ),
                axis=-1,
            ),
            "robot1_right": np.concatenate(
                (
                    input_action["action.left_arm_eef_pos"],
                    input_action["action.left_arm_eef_rot"],
                ),
                axis=-1,
            ),
        }
        return output_action

    @classmethod
    def get_metadata(cls, name):
        # This function and import is only enabled in gr00t exporter
        from decoupled_wbc.data.schema.raw.metadata import RotationType

        if name in ["body.right_arm_eef_pos", "body.left_arm_eef_pos"]:
            return {
                "absolute": False,
                "rotation_type": None,
            }
        elif name in ["body.right_arm_eef_quat", "body.right_arm_eef_quat"]:
            return {
                "absolute": False,
                "rotation_type": RotationType.QUATERNION,
            }
        elif name in ["body.right_arm_eef_rot", "body.left_arm_eef_rot"]:
            return {
                "absolute": False,
                "rotation_type": RotationType.AXIS_ANGLE,
            }
        else:
            return {
                "absolute": True,
                "rotation_type": None,
            }


class PandaDexRHPandaDexRHKeyConverter(RobotKeyConverter):
    @classmethod
    def get_camera_config(cls):
        mapped_names = [
            "video.agentview_pad_res256_freq20",
            "video.robot0_eye_in_hand_pad_res256_freq20",
            "video.robot1_eye_in_hand_pad_res256_freq20",
        ]
        camera_names = [
            "agentview",
            "robot0_eye_in_hand",
            "robot1_eye_in_hand",
        ]
        camera_widths, camera_heights = 1280, 800
        return mapped_names, camera_names, camera_widths, camera_heights

    @classmethod
    def map_obs(cls, input_obs):
        output_obs = type(input_obs)()
        output_obs = {
            "hand.right_hand": input_obs["robot0_right_gripper"],
            "hand.left_hand": input_obs["robot1_right_gripper"],
            "body.right_arm_eef_pos": input_obs["robot0_eef_pos"],
            "body.right_arm_eef_quat": input_obs["robot0_eef_quat"],
            "body.right_arm_joint_pos": input_obs["robot0_joint_pos"],
            "body.left_arm_eef_pos": input_obs["robot1_eef_pos"],
            "body.left_arm_eef_quat": input_obs["robot1_eef_quat"],
            "body.left_arm_joint_pos": input_obs["robot1_joint_pos"],
        }
        return output_obs

    @classmethod
    def map_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "hand.right_hand": input_action["robot0_right_gripper"],
            "hand.left_hand": input_action["robot1_right_gripper"],
            "body.right_arm_eef_pos": input_action["robot0_right"][0:3],
            "body.right_arm_eef_rot": input_action["robot0_right"][3:6],
            "body.left_arm_eef_pos": input_action["robot1_right"][0:3],
            "body.left_arm_eef_rot": input_action["robot1_right"][3:6],
        }
        return output_action

    @classmethod
    def unmap_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "robot0_right_gripper": input_action["action.right_hand"],
            "robot1_right_gripper": input_action["action.left_hand"],
            "robot0_right": np.concatenate(
                (
                    input_action["action.right_arm_eef_pos"],
                    input_action["action.right_arm_eef_rot"],
                ),
                axis=-1,
            ),
            "robot1_right": np.concatenate(
                (
                    input_action["action.left_arm_eef_pos"],
                    input_action["action.left_arm_eef_rot"],
                ),
                axis=-1,
            ),
        }
        return output_action

    @classmethod
    def get_metadata(cls, name):
        # This function and import is only enabled in gr00t exporter
        from decoupled_wbc.data.schema.raw.metadata import RotationType

        if name in ["body.right_arm_eef_pos", "body.left_arm_eef_pos"]:
            return {
                "absolute": False,
                "rotation_type": None,
            }
        elif name in ["body.right_arm_eef_quat", "body.right_arm_eef_quat"]:
            return {
                "absolute": False,
                "rotation_type": RotationType.QUATERNION,
            }
        elif name in ["body.right_arm_eef_rot", "body.left_arm_eef_rot"]:
            return {
                "absolute": False,
                "rotation_type": RotationType.AXIS_ANGLE,
            }
        else:
            return {
                "absolute": True,
                "rotation_type": None,
            }


# The values are only used in gr00t dataset embodiment tag and env name
GR00T_ROBOCASA_ENVS_GR1_ARMS_ONLY = {
    "GR1ArmsOnly": "gr1_arms_only_fourier_hands",
    "GR1ArmsOnlyInspireHands": "gr1_arms_only_inspire_hands",
    "GR1ArmsOnlyFourierHands": "gr1_arms_only_fourier_hands",
}
GR00T_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST = {
    "GR1ArmsAndWaistFourierHands": "gr1_arms_waist_fourier_hands",
}
GR00T_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY = {
    "GR1FixedLowerBody": "gr1_fixed_lower_body_fourier_hands",
    "GR1FixedLowerBodyInspireHands": "gr1_fixed_lower_body_inspire_hands",
    "GR1FixedLowerBodyFourierHands": "gr1_fixed_lower_body_fourier_hands",
}
GR00T_ROBOCASA_ENVS_PANDA = {
    "PandaMobile": "panda_mobile",
    "PandaOmron": "panda_omron",
}
GR00T_ROBOCASA_ENVS_BIMANUAL_GRIPPER = {
    "Panda_Panda": "bimanual_panda_parallel_gripper",
}
GR00T_ROBOCASA_ENVS_BIMANUAL_HAND = {
    "PandaDexRH_PandaDexLH": "bimanual_panda_inspire_hand",
}
GR00T_ROBOCASA_ENVS_G1 = {
    "G1": "g1_sim",
    "G1ArmsOnly": "g1_sim",
    "G1FixedLowerBody": "g1_sim",
    "G1FixedBase": "g1_sim",
}
GR00T_ROBOCASA_ENVS_ROBOTS = {
    **GR00T_ROBOCASA_ENVS_GR1_ARMS_ONLY,
    **GR00T_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST,
    **GR00T_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY,
    **GR00T_ROBOCASA_ENVS_PANDA,
    **GR00T_ROBOCASA_ENVS_BIMANUAL_GRIPPER,
    **GR00T_ROBOCASA_ENVS_BIMANUAL_HAND,
    **GR00T_ROBOCASA_ENVS_G1,
}
GR00T_LOCOMANIP_ENVS_ROBOTS = {**GR00T_ROBOCASA_ENVS_G1}


def make_key_converter(robots_name):
    if robots_name in GR00T_ROBOCASA_ENVS_GR1_ARMS_ONLY:
        return GR1ArmsOnlyKeyConverter
    elif robots_name in GR00T_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST:
        return GR1ArmsAndWaistKeyConverter
    elif robots_name in GR00T_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY:
        return GR1FixedLowerBodyKeyConverter
    elif robots_name in GR00T_ROBOCASA_ENVS_PANDA:
        return PandaOmronKeyConverter
    elif robots_name in GR00T_ROBOCASA_ENVS_BIMANUAL_GRIPPER:
        return PandaPandaKeyConverter
    elif robots_name in GR00T_ROBOCASA_ENVS_BIMANUAL_HAND:
        return PandaDexRHPandaDexRHKeyConverter
    else:
        raise ValueError(f"Unknown robot name: {robots_name}")
