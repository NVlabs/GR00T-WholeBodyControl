import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np

from robocasa.models.grippers import G1ThreeFingerLeftHand, G1ThreeFingerRightHand
from robosuite.examples.third_party_controller.mink_controller import IKSolverMink
from robosuite.models.grippers import InspireLeftHand, InspireRightHand
from robosuite.models.grippers import FourierLeftHand, FourierRightHand
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site
from robosuite.wrappers import VisualizationWrapper
import robosuite.utils.transform_utils as T


class IKWrapper(VisualizationWrapper):
    def __init__(self, env, indicator_configs=None, ik_indicator=False):
        self.ik_indicator = ik_indicator
        if ik_indicator:
            indicator_configs = []
            for i in range(2):
                indicator_configs.append(
                    {
                        "name": f"indicator_x_{i}",
                        "type": "capsule",
                        "pos": [0.012, 0, 0],
                        "size": [0.003, 0.012],
                        "rgba": [1, 0.5, 0.5, 0.8],
                        "quat": [0, 0.7071, 0, 0.7071],
                    }
                )
                indicator_configs.append(
                    {
                        "name": f"indicator_y_{i}",
                        "type": "capsule",
                        "pos": [0, 0.012, 0],
                        "size": [0.003, 0.012],
                        "rgba": [0.5, 1, 0.5, 0.8],
                        "quat": [0, 0, 0.7071, 0.7071],
                    }
                )
                indicator_configs.append(
                    {
                        "name": f"indicator_z_{i}",
                        "type": "capsule",
                        "pos": [0, 0, 0.012],
                        "size": [0.003, 0.012],
                        "rgba": [0.5, 0.5, 1, 0.8],
                    }
                )
            super().__init__(env, indicator_configs=indicator_configs)
        else:
            # Fall back to VisualizationWrapper
            super().__init__env, indicator_configs()

    def step(self, action):
        if self.ik_indicator:
            composite_controller = self.robots[0].composite_controller
            if composite_controller.name == "BASIC":
                # indicators will be set by calling `set_target_poses_outside_env`
                pass
            elif composite_controller.name in [
                "WHOLE_BODY_MINK_IK",
                "WHOLE_BODY_EXTERNAL_IK",
                "HYBRID_WHOLE_BODY_MINK_IK",
            ]:
                assert composite_controller.joint_action_policy.input_ref_frame == "base"
                input_action = action[: composite_controller.joint_action_policy.control_dim]
                input_action = input_action.reshape(
                    len(composite_controller.joint_action_policy.site_names), -1
                )
                input_pos = input_action[:, :3]
                input_ori = input_action[:, 3:]
                input_quat_wxyz = np.array(
                    [np.roll(T.axisangle2quat(input_ori[i]), 1) for i in range(len(input_ori))]
                )
                base_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("robot0_base")]
                base_ori = self.sim.data.body_xmat[
                    self.sim.model.body_name2id("robot0_base")
                ].reshape(3, 3)
                base_pose = T.make_pose(base_pos, base_ori)
                for _, name in enumerate(self.get_indicator_names()):
                    # indicators for x,y,z-axis share the same pose
                    i = int(name.split("_")[-1])
                    input_pose = T.make_pose(
                        input_pos[i], T.quat2mat(np.roll(input_quat_wxyz[i], -1))
                    )
                    target_pose = np.dot(base_pose, input_pose)
                    self.set_indicator_pos(name, target_pose[:3, 3])
                    self.set_indicator_ori(name, target_pose[:3, :3])
            else:
                assert (
                    False
                ), f"Unsupported composite controller {composite_controller.name} for IKWrapper"

        ret = super().step(action)

        return ret

    def set_target_poses_outside_env(self, input_poses):
        if self.env.robots[0].robot_model.default_base in ["NullBase", "NoActuationBase"]:
            base_name = "robot0_base"
        elif self.env.robots[0].robot_model.default_base == "FloatingLeggedBase":
            base_name = "mobilebase0_support"
        else:
            assert False, f"Unsupported base type: {self.env.robots[0].robot_model.default_base}"
        base_id = self.sim.model.body_name2id(base_name)
        base_pos = self.sim.data.body_xpos[base_id]
        base_ori = self.sim.data.body_xmat[base_id].reshape(3, 3)
        base_pose = T.make_pose(base_pos, base_ori)
        for _, name in enumerate(self.get_indicator_names()):
            i = int(name.split("_")[-1])
            input_pose = input_poses[i].copy()
            input_pose[3, 3] = 1
            target_pose = np.dot(base_pose, input_pose)
            self.set_indicator_pos(name, target_pose[:3, 3])
            self.set_indicator_ori(name, target_pose[:3, :3])

    # TODO: this will fix the duplicated body issue in the public robosuite repo
    def _add_indicators_to_model(self, xml):
        """
        Adds indicators to the mujoco simulation model

        Args:
            xml (string): MJCF model in xml format, for the current simulation to be loaded
        """
        if self.indicator_configs is not None:
            root = ET.fromstring(xml)
            worldbody = root.find("worldbody")

            from robosuite.utils.mjcf_utils import (
                find_elements,
                find_parent,
                new_element,
            )

            for arm in ["right", "left"]:
                gripper = self.env.robots[0].gripper[arm]
                if (
                    not isinstance(gripper, InspireLeftHand)
                    and not isinstance(gripper, InspireRightHand)
                    and not isinstance(gripper, FourierLeftHand)
                    and not isinstance(gripper, FourierRightHand)
                    and not isinstance(gripper, G1ThreeFingerLeftHand)
                    and not isinstance(gripper, G1ThreeFingerRightHand)
                ):
                    continue

                eef = find_elements(
                    root=worldbody,
                    tags="body",
                    attribs={"name": f"gripper0_{arm}_eef"},
                    return_first=True,
                )
                mount = find_parent(worldbody, eef)
                for i in range(4):
                    elem = find_elements(
                        root=worldbody,
                        tags="site",
                        attribs={"name": f"robot0_{arm}_pinch_spheres_{i}"},
                        return_first=True,
                    )
                    if elem is not None:
                        continue
                    if (
                        isinstance(gripper, InspireLeftHand)
                        or isinstance(gripper, InspireRightHand)
                        or isinstance(gripper, FourierLeftHand)
                        or isinstance(gripper, FourierRightHand)
                    ):
                        mount.append(
                            new_element(
                                tag="site",
                                name=f"robot0_{arm}_pinch_spheres_{i}",
                                pos=f"-0.09 -0.10 {-0.03 + i * 0.02}",
                                size="0.01 0.01 0.01",
                                rgba="1 0.5 0.5 1",
                                type="sphere",
                                group="1",
                            )
                        )
                    else:
                        if arm == "right":
                            mount.append(
                                new_element(
                                    tag="site",
                                    name=f"robot0_{arm}_pinch_spheres_{i}",
                                    pos=f"0.15 0.05 {-0.03 + i * 0.02}",
                                    size="0.01 0.01 0.01",
                                    rgba="1 0.5 0.5 1",
                                    type="sphere",
                                    group="1",
                                )
                            )
                        else:
                            mount.append(
                                new_element(
                                    tag="site",
                                    name=f"robot0_{arm}_pinch_spheres_{i}",
                                    pos=f"0.15 -0.05 {-0.03 + i * 0.02}",
                                    size="0.01 0.01 0.01",
                                    rgba="1 0.5 0.5 1",
                                    type="sphere",
                                    group="1",
                                )
                            )

            for indicator_config in self.indicator_configs:
                config = deepcopy(indicator_config)

                from robosuite.utils.mjcf_utils import find_elements

                body = find_elements(
                    root=worldbody,
                    tags="body",
                    attribs={"name": config["name"] + "_body"},
                )
                if body is not None:
                    continue

                indicator_body = new_body(name=config["name"] + "_body")
                indicator_body.append(new_site(**config))
                worldbody.append(indicator_body)

            xml = ET.tostring(root, encoding="utf8").decode("utf8")

        return xml
