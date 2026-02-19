"""
Dexterous hands for GR1 robot.
"""

import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import find_parent, xml_path_completion
from robosuite.models.grippers import register_gripper

import robocasa.models


@register_gripper
class G1ThreeFingerLeftHand(GripperModel):
    """
    Dexterous left hand of G1 robot
    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(
            xml_path_completion(
                "robots/unitree_g1/g1_threefinger_left_hand.xml",
                root=robocasa.models.assets_root,
            ),
            idn=idn,
        )

    def format_action(self, action):
        # grasp = action[0]
        # action = [0] * 7
        # if grasp > 0:
        #     # note that the sign of the action is opposite of the right hand
        #     action[1] = action[2] = grasp
        #     action[3] = action[4] = action[5] = action[6] = -grasp

        return action

    @property
    def init_qpos(self):
        return np.array([0.0] * 7)

    @property
    def grasp_qpos(self):
        return {
            -1: np.array([0.0] * 7),
            1: np.array([0, 1, 1.25, -1.2, -1.2, -1.2, -1.2]),
        }

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 7  # 12

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_hand_thumb_0_link_col",
                "left_hand_thumb_1_link_col",
                "left_hand_thumb_2_link_col",
                "left_hand_middle_0_link_col",
                "left_hand_middle_1_link_col",
                "left_hand_index_0_link_col",
                "left_hand_index_1_link_col",
            ],
            "right_finger": [
                "left_hand_thumb_0_link_col",
                "left_hand_thumb_1_link_col",
                "left_hand_thumb_2_link_col",
                "left_hand_middle_0_link_col",
                "left_hand_middle_1_link_col",
                "left_hand_index_0_link_col",
                "left_hand_index_1_link_col",
            ],
            "left_fingerpad": [
                "left_hand_thumb_2_link_col",
                "left_hand_middle_1_link_col",
                "left_hand_index_1_link_col",
            ],
            "right_fingerpad": [
                "left_hand_thumb_2_link_col",
                "left_hand_middle_1_link_col",
                "left_hand_index_1_link_col",
            ],
        }


@register_gripper
class G1ThreeFingerRightHand(GripperModel):
    """
    Dexterous right hand of G1 robot
    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(
            xml_path_completion(
                "robots/unitree_g1/g1_threefinger_right_hand.xml",
                root=robocasa.models.assets_root,
            ),
            idn=idn,
        )

    def format_action(self, action):
        # grasp = action[0]
        # action = [0] * 7
        # if grasp > 0:
        #     action[1] = action[2] = -grasp
        #     action[3] = action[4] = action[5] = action[6] = grasp
        return action

    @property
    def init_qpos(self):
        return np.array([0.0] * 7)

    @property
    def grasp_qpos(self):
        return {
            -1: np.array([0.0] * 7),
            1: np.array([0, -1, -1.25, 1.2, 1.2, 1.2, 1.2]),
        }

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 7

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "right_hand_thumb_0_link_col",
                "right_hand_thumb_1_link_col",
                "right_hand_thumb_2_link_col",
                "right_hand_middle_0_link_col",
                "right_hand_middle_1_link_col",
                "right_hand_index_0_link_col",
                "right_hand_index_1_link_col",
            ],
            "right_finger": [
                "right_hand_thumb_0_link_col",
                "right_hand_thumb_1_link_col",
                "right_hand_thumb_2_link_col",
                "right_hand_middle_0_link_col",
                "right_hand_middle_1_link_col",
                "right_hand_index_0_link_col",
                "right_hand_index_1_link_col",
            ],
            "left_fingerpad": [
                "right_hand_thumb_2_link_col",
                "right_hand_middle_1_link_col",
                "right_hand_index_1_link_col",
            ],
            "right_fingerpad": [
                "right_hand_thumb_2_link_col",
                "right_hand_middle_1_link_col",
                "right_hand_index_1_link_col",
            ],
        }
