import numpy as np
from robosuite.models.robots.manipulators.legged_manipulator_model import (
    LeggedManipulatorModel,
)
from robosuite.robots import register_robot_class
from robosuite.utils.mjcf_utils import find_elements, xml_path_completion

import robocasa.models


@register_robot_class("LeggedRobot")
class G1(LeggedManipulatorModel):
    """
    G1 is a mobile manipulator robot created by Unitree.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right", "left"]

    def __init__(self, idn: int = 0):
        super().__init__(
            xml_path_completion(
                "robots/unitree_g1/g1_29dof_rev_1_0.xml",
                root=robocasa.models.assets_root,
            ),
            idn=idn,
        )

    def _print_joint_info(self):
        """Print formatted information about joints"""
        # ANSI color codes
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"

        print(f"\n{BOLD}{BLUE}================ joints info ================{ENDC}")

        # Print joints by category
        categories = {
            "Arms": self._arms_joints,
            "Base": self._base_joints,
            "Torso": self._torso_joints,
            "Head": self._head_joints,
            "Legs": self._legs_joints,
        }

        for category, joints in categories.items():
            if joints:
                print(f"{GREEN}{category} ({len(joints)}){ENDC}: {', '.join(joints)}")
            else:
                print(f"{RED}{category} (disabled){ENDC}")
        print()

    def _print_actuator_info(self):
        """Print formatted information about actuators"""
        # ANSI color codes
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"

        print(f"\n{BOLD}{BLUE}================ actuators info ================{ENDC}")

        # Print actuators by category
        categories = {
            "Arms": self._arms_actuators,
            "Base": self._base_actuators,
            "Torso": self._torso_actuators,
            "Head": self._head_actuators,
            "Legs": self._legs_actuators,
        }

        for category, actuators in categories.items():
            if actuators:
                print(f"{GREEN}{category} ({len(actuators)}){ENDC}: {', '.join(actuators)}")
            else:
                print(f"{RED}{category} (disabled){ENDC}")
        print()

    def update_joints(self):
        """internal function to update joint lists"""
        for joint in self.all_joints:
            if "waist" in joint:
                self.torso_joints.append(joint)
            elif "base" in joint:
                self.base_joints.append(joint)
            elif "knee" in joint or "hip" in joint or "ankle" in joint:
                self.legs_joints.append(joint)

        for joint in self.all_joints:
            if (
                joint not in self._base_joints
                and joint not in self._torso_joints
                and joint not in self._head_joints
                and joint not in self._legs_joints
            ):
                self._arms_joints.append(joint)

        # adjust the order of the joints, make sure the right arm joints are before the left arm joints
        self._arms_joints = [joint for joint in self._arms_joints if "right" in joint] + [
            joint for joint in self._arms_joints if "left" in joint
        ]

        # self._print_joint_info()

    def update_actuators(self):
        """internal function to update actuator lists"""
        for actuator in self.all_actuators:
            if "waist" in actuator:
                self.torso_actuators.append(actuator)
            elif "base" in actuator:
                self.base_actuators.append(actuator)
            elif "knee" in actuator or "hip" in actuator or "ankle" in actuator:
                self.legs_actuators.append(actuator)

        for actuator in self.all_actuators:
            if (
                actuator not in self._base_actuators
                and actuator not in self._torso_actuators
                and actuator not in self._head_actuators
                and actuator not in self._legs_actuators
            ):
                self._arms_actuators.append(actuator)

        # adjust the order of the actuators, make sure the right arm actuators are before the left arm actuators
        self._arms_actuators = [
            actuator for actuator in self._arms_actuators if "right" in actuator
        ] + [actuator for actuator in self._arms_actuators if "left" in actuator]
        # self._print_actuator_info()

    @property
    def default_base(self):
        return "NullBase"
        # return "NoActuationBase"

    @property
    def default_gripper(self):
        # return {"right": None, "left": None}
        return {"right": "G1ThreeFingerRightHand", "left": "G1ThreeFingerLeftHand"}

    @property
    def init_qpos(self):
        DEFAULT_DOF_ANGLES = {
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,  # 0.3,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,  # 1.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,  # -0.3,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,  # 1.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

        joints = self.worldbody.findall(".//joint")
        joints = [joint for joint in joints if joint.get("name").startswith("robot")]
        init_qpos = np.array([0.0] * len(joints))
        for joint_id, joint in enumerate(joints):
            init_qpos[joint_id] = DEFAULT_DOF_ANGLES[
                joint.get("name")[joint.get("name").find("_") + 1 :]
            ]
        return init_qpos

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.30, -0.1, 0.95),
            "empty": (-0.29, 0, 0.95),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.95),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        return {"right": "right_eef", "left": "left_eef"}

    @property
    def torso_body(self):
        return ["torso_link"]

    def get_camera_configs(self):
        _cam_config = {}

        # egoview camera
        egoview_mount_body = find_elements(
            root=self.root,
            tags="body",
            attribs={"name": f"{self.naming_prefix}torso_link"},
        )
        if egoview_mount_body is not None:
            # realsense D435i, 1920 * 1080, 69° × 42°
            _cam_config[f"{self.naming_prefix}rs_egoview"] = dict(
                pos=[0.07555294, 0.02579919, 0.49719054],
                # quat=[0.69263989,  0.16169103, -0.19008497,  -0.67673754],
                quat=[0.67876761, 0.20864099, -0.20567003, -0.67338199],
                camera_attribs=dict(fovy="40"),
                parent_body=f"{self.naming_prefix}torso_link",
            )

            # OAK-D W, OV9782, 640*480
            _cam_config[f"{self.naming_prefix}oak_egoview"] = dict(
                pos=[0.10209156, -0.00937542, 0.42446595],
                quat=[0.64367383, 0.26523914, -0.27106013, -0.66472446],
                camera_attribs=dict(fovy="79.5"),
                parent_body=f"{self.naming_prefix}torso_link",
            )

            _cam_config[f"{self.naming_prefix}oak_left_monoview"] = dict(
                pos=[0.10209156, 0.02857542, 0.42446595],
                quat=[0.64367383, 0.26523914, -0.27106013, -0.66472446],
                camera_attribs=dict(fovy="79.5"),
                parent_body=f"{self.naming_prefix}torso_link",
            )

            _cam_config[f"{self.naming_prefix}oak_right_monoview"] = dict(
                pos=[0.10209156, -0.04657542, 0.42446595],
                quat=[0.64367383, 0.26523914, -0.27106013, -0.66472446],
                camera_attribs=dict(fovy="79.5"),
                parent_body=f"{self.naming_prefix}torso_link",
            )

        # tppview camera
        tpp_mount_body = find_elements(
            root=self.root,
            tags="body",
            attribs={"name": f"{self.naming_prefix}pelvis"},
        )
        if tpp_mount_body is not None:
            _cam_config[f"{self.naming_prefix}rs_tppview"] = dict(
                pos=[-1.131, -0.626, 1.247 - 0.793],
                quat=[0.67953146, 0.46872971, -0.3204774, -0.46456828],
                camera_attribs=dict(fovy="60"),
                parent_body=f"{self.naming_prefix}pelvis",
            )

        return _cam_config

    def _disable_collisions(self, body_parts):
        """
        Disable collision detection for specified body parts by setting
        contype="0" and conaffinity="0" for all collision geometries in those parts.

        Args:
            body_parts (list): List of strings identifying body parts to disable collisions for.
                              Examples: ["hip", "knee", "ankle"], ["waist"], ["shoulder"], etc.
        """
        # Find all body elements in the worldbody
        for body in self.worldbody.findall(".//body"):
            body_name = body.get("name", "")
            # Check if this body belongs to specified body parts
            if any(part in body_name for part in body_parts):
                # Find all geom elements in this body
                for geom in body.findall(".//geom"):
                    # Check if this is a collision geom (doesn't have contype="0" and conaffinity="0")
                    contype = geom.get("contype")
                    conaffinity = geom.get("conaffinity")

                    # If contype and conaffinity are not already "0", set them to disable collision
                    if contype != "0" or conaffinity != "0":
                        geom.set("contype", "0")
                        geom.set("conaffinity", "0")


@register_robot_class("LeggedRobot")
class G1FixedBase(G1):
    def __init__(self, idn: int = 0):
        super().__init__(idn=idn)

        # Remove base actuation
        self._remove_free_joint()


@register_robot_class("LeggedRobot")
class G1FixedLowerBody(G1):
    def __init__(self, idn: int = 0):
        super().__init__(idn=idn)

        # Remove lower body actuation
        self._remove_joint_actuation("knee")
        self._remove_joint_actuation("hip")
        self._remove_joint_actuation("ankle")

        self._remove_free_joint()


@register_robot_class("LeggedRobot")
class G1ArmsOnly(G1):
    def __init__(self, idn: int = 0):
        super().__init__(idn=idn)

        # Remove lower body actuation
        self._remove_joint_actuation("knee")
        self._remove_joint_actuation("hip")
        self._remove_joint_actuation("ankle")

        # Remove spine & head actuation
        self._remove_joint_actuation("waist")

        self._remove_free_joint()


@register_robot_class("LeggedRobot")
class G1ArmsOnlyFloating(G1):
    def __init__(self, idn: int = 0):
        super().__init__(idn=idn)

        # Remove lower body actuation
        self._remove_joint_actuation("knee")
        self._remove_joint_actuation("hip")
        self._remove_joint_actuation("ankle")

        # Remove spine & head actuation
        self._remove_joint_actuation("waist")

        self._remove_free_joint()

    @property
    def default_base(self):
        return "FloatingLeggedBase"

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.30, -0.1, 0.97),
            "empty": (-0.29, 0, 0.97),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.97),
        }


@register_robot_class("LeggedRobot")
class G1FloatingBody(G1):
    def __init__(self, idn: int = 0):
        super().__init__(idn=idn)

        # Remove lower body actuation
        self._remove_joint_actuation("knee")
        self._remove_joint_actuation("hip")
        self._remove_joint_actuation("ankle")

        # Disable collision detection for lower body parts
        self._disable_collisions(["hip", "knee", "ankle"])

        self._remove_free_joint()

    @property
    def default_base(self):
        return "FloatingLeggedBase"

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.30, -0.1, 0.97),
            "empty": (-0.29, 0, 0.97),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.97),
        }
