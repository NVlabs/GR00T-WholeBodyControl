from copy import deepcopy
import os
from typing import Optional, Type
import warnings
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import robosuite
from robosuite.environments.base import EnvMeta
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion
from robosuite.utils.observables import Observable, sensor

import robocasa
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes import GroundArena
import robocasa.utils.camera_utils as CamUtils
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robocasa.utils.object_utils import check_obj_upright
from robocasa.utils.visuals_utls import Gradient, randomize_materials_rgba

REGISTERED_LOCOMANIPULATION_ENVS = {}


def register_locomanipulation_env(target_class):
    REGISTERED_LOCOMANIPULATION_ENVS[target_class.__name__] = target_class


class LocoManipulationEnvMeta(EnvMeta):
    """Metaclass for registering robocasa environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        register_locomanipulation_env(cls)
        return cls


class CameraPoseRandomizer:
    @staticmethod
    def randomize_cameras(
        env: "LocoManipulationEnv",
        cam_names: list[str],
        pos_range: tuple[np.ndarray, np.ndarray],
        euler_range: tuple[np.ndarray, np.ndarray],
    ):
        """
        Randomize camera poses while maintaining their relative transforms.

        Args:
            env: The environment instance
            cam_names: List of camera names to randomize together
            pos_range: Tuple of (min_pos, max_pos) as 3D arrays for position randomization
            euler_range: Tuple of (min_euler, max_euler) as 3D arrays for euler angle randomization (in radians)
        """
        if len(cam_names) == 0:
            return

        # Sample random transform offset
        random_pos_offset = env.rng.uniform(pos_range[0], pos_range[1])
        random_euler_offset = env.rng.uniform(euler_range[0], euler_range[1])

        # Convert euler offset to quaternion
        quat_offset = np.zeros(4, dtype=float)
        mujoco.mju_euler2Quat(quat_offset, random_euler_offset, "xyz")

        # Apply the same transform to all specified cameras
        for cam_name in cam_names:
            if cam_name not in env._cam_configs:
                warnings.warn(f"Camera {cam_name} not found in camera configs. Skipping.")
                continue

            cam_config = env._cam_configs[cam_name]

            # Get original position and quaternion
            original_pos = np.array(cam_config["pos"], dtype=float)
            original_quat = np.array(cam_config["quat"], dtype=float)

            # Apply rotation offset to position (rotate position offset by the random rotation)
            rotated_offset = np.zeros(3, dtype=float)
            mujoco.mju_rotVecQuat(rotated_offset, random_pos_offset, original_quat)
            new_pos = original_pos + rotated_offset

            # Compose quaternions: new_quat = quat_offset * original_quat
            new_quat = np.zeros(4, dtype=float)
            mujoco.mju_mulQuat(new_quat, quat_offset, original_quat)

            # Update camera config
            cam_config["pos"] = new_pos.tolist()
            cam_config["quat"] = new_quat.tolist()

            # Update in simulation if already created
            if hasattr(env, "sim") and env.sim is not None:
                try:
                    cam_id = env.sim.model.camera_name2id(cam_name)
                    env.sim.model.cam_pos[cam_id] = new_pos
                    env.sim.model.cam_quat[cam_id] = new_quat
                except:
                    # Camera might not be in the model yet
                    pass


class RobotPoseRandomizer:
    @staticmethod
    def set_pose(
        env: "LocoManipulationEnv",
        x_range: [tuple[float, float]],
        y_range: [tuple[float, float]],
        yaw_range: [tuple[float, float]],
    ):
        new_x = env.rng.uniform(*x_range)
        new_y = env.rng.uniform(*y_range)
        new_yaw = env.rng.uniform(*yaw_range)

        if env.robots[0].name == "G1":
            base_offset = env.ROBOT_POS_OFFSETS[env.robots[0].robot_model.__class__.__name__]
            target_pos = np.array([new_x, new_y, base_offset[2]], dtype=float)
            quat = np.zeros(4, dtype=float)
            mujoco.mju_euler2Quat(quat, np.array([0.0, 0.0, new_yaw]), "xyz")
            base_freejoint = f"{env.robots[0].robot_model.naming_prefix}base"
            if base_freejoint in env.sim.model.joint_names:
                env.sim.data.set_joint_qpos(base_freejoint, np.concatenate([target_pos, quat]))
            else:
                warnings.warn(f"Base joint {base_freejoint} not found in the model.")
        else:
            base_joint_pos = np.array([new_x, new_y, new_yaw])
            base_joint_names = [
                "mobilebase0_joint_mobile_forward",
                "mobilebase0_joint_mobile_side",
                "mobilebase0_joint_mobile_yaw",
            ]
            for i, base_joint_name in enumerate(base_joint_names):
                if base_joint_name not in env.sim.model.joint_names:
                    warnings.warn(
                        f"Base joint {base_joint_name} not found in the model. "
                        f"Skipping randomization of {base_joint_name}."
                    )
                else:
                    env.sim.data.set_joint_qpos(base_joint_name, base_joint_pos[i])

    @staticmethod
    def set_arm(env: ManipulationEnv, elbow_qpos: float, shoulder_pitch_qpos: float):
        """Helper function to reinitialize G1 robot arm configuration."""
        robot = env.robots[0]
        if "G1" not in robot.name:
            # avoid reinitializing arm configuration for non-G1 robots
            return

        joint_names = robot.robot_joints
        joint_pos_indices = robot._ref_joint_pos_indexes
        for joint_name, pos_idx in zip(joint_names, joint_pos_indices):
            if "elbow" in joint_name:
                print(f"reinitializing G1 {joint_name} with idx {pos_idx} to {elbow_qpos}")
                env.sim.data.qpos[pos_idx] = elbow_qpos
            elif "shoulder_pitch" in joint_name:
                print(f"reinitializing G1 {joint_name} with idx {pos_idx} to {shoulder_pitch_qpos}")
                env.sim.data.qpos[pos_idx] = shoulder_pitch_qpos


class LocoManipulationEnv(ManipulationEnv, metaclass=LocoManipulationEnvMeta):
    """
    Initialized a Base Ground Standing environment.
    """

    MUJOCO_ARENA_CLS: Type[Arena] = GroundArena

    ROBOT_POS_OFFSETS: dict[str, list[float]] = {
        "PandaOmron": [0, 0, 0],
        "GR1FloatingBody": [0, 0, 0.97],
        "GR1": [0, 0, 0.97],
        "GR1FixedLowerBody": [0, 0, 0.97],
        "GR1FixedLowerBodyInspireHands": [0, 0, 0.97],
        "GR1FixedLowerBodyFourierHands": [0, 0, 0.97],
        "GR1ArmsOnly": [0, 0, 0.97],
        "GR1ArmsOnlyInspireHands": [0, 0, 0.97],
        "GR1ArmsOnlyFourierHands": [0, 0, 0.97],
        "GR1ArmsAndWaistFourierHands": [0, 0, 0.97],
        "G1": [0, 0, 0.793],
        "G1FixedBase": [0, 0, 0.793],
        "G1FixedLowerBody": [0, 0, 0.793],
        "G1ArmsOnly": [0, 0, 0.793],
        "G1ArmsOnlyFloating": [0, 0, 0.793],
        "G1FloatingBody": [0, 0, 0.793],
        "G1FloatingBodyWithVertical": [0, 0, 0.793],
    }

    def __init__(
        self,
        translucent_robot: bool = False,
        use_object_obs: bool = False,
        randomize_cameras: bool = False,
        *args,
        **kwargs,
    ):
        self.mujoco_objects = []
        self.randomize_cameras = randomize_cameras

        super().__init__(
            *args,
            **kwargs,
        )

        self.translucent_robot = translucent_robot

    def _load_model(self):
        super()._load_model()

        self.mujoco_arena = self.MUJOCO_ARENA_CLS()
        self.mujoco_arena.set_origin([0, 0, 0])
        self.set_cameras()

        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.mujoco_objects,
        )

        robot_base_pos = self.ROBOT_POS_OFFSETS[self.robots[0].robot_model.__class__.__name__]
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(robot_base_pos)
        # robot_model.set_base_ori(robot_base_ori)

    def set_cameras(self):
        """
        Adds new tabletop-relevant cameras to the environment. Will randomize cameras if specified.
        """

        self._cam_configs = deepcopy(CamUtils.CAM_CONFIGS)

        for robot in self.robots:
            if hasattr(robot.robot_model, "get_camera_configs"):
                self._cam_configs.update(robot.robot_model.get_camera_configs())

        for cam_name, cam_cfg in self._cam_configs.items():
            if cam_cfg.get("parent_body", None) is not None:
                continue

            self.mujoco_arena.set_camera(
                camera_name=cam_name,
                pos=cam_cfg["pos"],
                quat=cam_cfg["quat"],
                camera_attribs=cam_cfg.get("camera_attribs", None),
            )

        self.mujoco_arena.set_camera(
            camera_name="egoview",
            pos=[0.078, 0, 1.308],
            quat=[0.66491268, 0.24112495, -0.24112507, -0.66453637],
            camera_attribs=dict(fovy="90"),
        )

    def visualize(self, vis_settings):
        """
        In addition to super call, make the robot semi-transparent

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        visual_geom_names = []

        for robot in self.robots:
            robot_model = robot.robot_model
            visual_geom_names += robot_model.visual_geoms

        for name in visual_geom_names:
            rgba = self.sim.model.geom_rgba[self.sim.model.geom_name2id(name)]
            if self.translucent_robot:
                rgba[-1] = 0.10
            else:
                rgba[-1] = 1.0

    def reward(self, action=None):
        """
        Reward function for the task. The reward function is based on the task
        and to be implemented in the subclasses. Returns 0 by default.

        Returns:
            float: Reward for the task
        """
        reward = 0
        if self._check_success():
            reward = 1.0
        return reward

    def _check_success(self):
        """
        Checks if the task has been successfully completed.
        Success condition is based on the task and to be implemented in the
        subclasses. Returns False by default.

        Returns:
            bool: True if the task is successfully completed, False otherwise
        """
        return False

    def edit_model_xml(self, xml_str):
        """
        This function postprocesses the model.xml collected from a MuJoCo demonstration
        for retrospective model changes.

        Args:
            xml_str (str): Mujoco sim demonstration XML file as string

        Returns:
            str: Post-processed xml file as string
        """
        xml_str = super().edit_model_xml(xml_str)

        tree = ET.fromstring(xml_str)
        root = tree
        worldbody = root.find("worldbody")
        actuator = root.find("actuator")
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        robosuite_path_split = os.path.split(robosuite.__file__)[0].split("/")
        robocasa_path_split = os.path.split(robocasa.__file__)[0].split("/")

        # replace robocasa-specific asset paths
        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue

            old_path_split = old_path.split("/")
            # maybe replace all paths to robosuite assets
            if "models/assets" in old_path:
                if "/robosuite/" in old_path:
                    check_lst = [
                        loc for loc, val in enumerate(old_path_split) if val == "robosuite"
                    ]
                    ind = max(check_lst)  # last occurrence index
                    new_path_split = robosuite_path_split + old_path_split[ind + 1 :]
                elif "/robocasa/" in old_path:
                    check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robocasa"]
                    ind = max(check_lst)  # last occurrence index
                    new_path_split = robocasa_path_split + old_path_split[ind + 1 :]
                else:
                    raise ValueError

                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        # set cameras
        for cam_name, cam_config in self._cam_configs.items():
            parent_body = cam_config.get("parent_body", None)

            cam_root = worldbody
            if parent_body is not None:
                cam_root = find_elements(root=worldbody, tags="body", attribs={"name": parent_body})
                if cam_root is None:
                    # camera config refers to body that doesnt exist on the robot
                    continue

            cam = find_elements(root=cam_root, tags="camera", attribs={"name": cam_name})

            if cam is None:
                old_cam = find_elements(root=worldbody, tags="camera", attribs={"name": cam_name})
                if old_cam is not None:
                    # old camera associated with different body
                    continue

                cam = ET.Element("camera")
                cam.set("mode", "fixed")
                cam.set("name", cam_name)
                cam_root.append(cam)

            cam.set("pos", array_to_string(cam_config["pos"]))
            cam.set("quat", array_to_string(cam_config["quat"]))
            for k, v in cam_config.get("camera_attribs", {}).items():
                cam.set(k, v)

        # replace base -> mobilebase (this is needed for old PandaOmron demos)
        for elem in find_elements(
            root=worldbody, tags=["geom", "site", "body", "joint"], return_first=False
        ):
            if elem.get("name") is None:
                continue
            if elem.get("name").startswith("base0_"):
                old_name = elem.get("name")
                new_name = "mobilebase0_" + old_name[6:]
                elem.set("name", new_name)
        for elem in find_elements(
            root=actuator,
            tags=["velocity", "position", "motor", "general"],
            return_first=False,
        ):
            if elem.get("name") is None:
                continue
            if elem.get("name").startswith("base0_"):
                old_name = elem.get("name")
                new_name = "mobilebase0_" + old_name[6:]
                elem.set("name", new_name)
        for elem in find_elements(
            root=actuator,
            tags=["velocity", "position", "motor", "general"],
            return_first=False,
        ):
            if elem.get("joint") is None:
                continue
            if elem.get("joint").startswith("base0_"):
                old_joint = elem.get("joint")
                new_joint = "mobilebase0_" + old_joint[6:]
                elem.set("joint", new_joint)

        # result = ET.tostring(root, encoding="utf8").decode("utf8")
        result = ET.tostring(root).decode("utf8")

        # # replace with generative textures
        # if (self.generative_textures is not None) and (
        #     self.generative_textures is not False
        # ):
        #     # sample textures
        #     assert self.generative_textures == "100p"
        #     self._curr_gen_fixtures = get_random_textures(self.rng)

        #     cab_tex = self._curr_gen_fixtures["cab_tex"]
        #     counter_tex = self._curr_gen_fixtures["counter_tex"]
        #     wall_tex = self._curr_gen_fixtures["wall_tex"]
        #     floor_tex = self._curr_gen_fixtures["floor_tex"]

        #     result = replace_cab_textures(
        #         self.rng, result, new_cab_texture_file=cab_tex
        #     )
        #     result = replace_counter_top_texture(
        #         self.rng, result, new_counter_top_texture_file=counter_tex
        #     )
        #     result = replace_wall_texture(
        #         self.rng, result, new_wall_texture_file=wall_tex
        #     )
        #     result = replace_floor_texture(
        #         self.rng, result, new_floor_texture_file=floor_tex
        #     )

        return result

    def _setup_references(self):
        super()._setup_references()

        self.obj_body_id = {}

    def _randomize_robot_cameras(self):
        """Randomize the poses of robot-mounted cameras while preserving their relative transforms."""
        cam_names = ["robot0_oak_egoview", "robot0_oak_left_monoview", "robot0_oak_right_monoview"]

        # Define randomization ranges
        pos_range = (
            np.array([-0.02, -0.02, -0.02]),  # min position offset [x, y, z] in meters
            np.array([0.02, 0.02, 0.02]),  # max position offset [x, y, z] in meters
        )
        euler_range = (
            np.array([-0.1, -0.1, -0.1]),  # min euler angles [roll, pitch, yaw] in radians
            np.array([0.1, 0.1, 0.1]),  # max euler angles [roll, pitch, yaw] in radians
        )

        CameraPoseRandomizer.randomize_cameras(
            env=self, cam_names=cam_names, pos_range=pos_range, euler_range=euler_range
        )

    def _reset_internal(self):
        super()._reset_internal()

        if self.randomize_cameras:
            self._randomize_robot_cameras()

    def _reset_observables(self):
        if self.hard_reset:
            self._observables = self._setup_observables()

        # these sensors need a lot of computation, so we disable them by default for speed up simulation
        disabled_sensors = [
            "base_to_left_eef_pos",
            "base_to_left_eef_quat",
            "base_to_left_eef_quat_site",
            "base_to_right_eef_pos",
            "base_to_right_eef_quat",
            "base_to_right_eef_quat_site",
        ]
        for name in disabled_sensors:
            for robot in self.robots:
                robot_name_prefix = robot.robot_model.naming_prefix
                if f"{robot_name_prefix}{name}" in self._observables:
                    self._observables[f"{robot_name_prefix}{name}"].set_enabled(False)
                    self._observables[f"{robot_name_prefix}{name}"].set_active(False)

    def get_state(self):
        return {"states": self.sim.get_state().flatten()}


class GroundOnly(LocoManipulationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PrimitiveBottle:
    DEFAULT_RGB = [0.3, 0.7, 0.8]

    def __init__(
        self,
        name="bottle",
        radius: float = 0.03,
        half_height: float = 0.075,
        rgb: Optional[list[float]] = None,
    ):
        self.name = name
        self.assets = [
            ET.Element(
                "texture",
                type="2d",
                name=f"{name}_tex",
                builtin="flat",
                rgb1=" ".join(map(str, self.DEFAULT_RGB if rgb is None else rgb)),
                width="512",
                height="512",
            ),
            ET.Element(
                "material",
                name=f"{name}_mat",
                texture=f"{name}_tex",
                texuniform="true",
                reflectance="0.1",
            ),
        ]

        self.body = ET.Element("body", name=f"{self.name}_body", pos="0.35 0 0.8")
        bottle_vis_geom = ET.Element(
            "geom",
            name=f"{name}_vis",
            pos="0 0 0",
            size=f"{radius} {half_height}",
            type="cylinder",
            material=f"{name}_mat",
            group="1",
            conaffinity="0",
            contype="0",
        )
        self.body.append(bottle_vis_geom)

        # Cylinder collider approximation for stable contacts
        self.contact_geoms = []
        n_sides = 3
        half_width = radius * np.tan(np.pi / n_sides / 2)
        for i in range(n_sides):
            coll_name = f"{self.name}_collider_{i}"
            angle = np.pi / n_sides * i
            quat = np.zeros(4)
            euler = np.array([0, 0, angle])
            mujoco.mju_euler2Quat(quat, euler, "xyz")
            box_geom = ET.Element(
                "geom",
                name=coll_name,
                type="box",
                pos="0 0 0",
                size=f"{radius} {half_width} {half_height}",
                quat=" ".join(map(str, quat)),
                solimp="0.998 0.998 0.001",
                solref="0.001 2",
                density="100",
                friction="0.95 0.3 0.1",
            )
            self.body.append(box_geom)
            self.contact_geoms.append(coll_name)

        bottle_joint = ET.Element(
            "joint",
            name=f"{self.name}_joint",
            type="free",
            damping="0.0005",
        )
        self.body.append(bottle_joint)


class PrimitiveFixture:
    DEFAULT_RGB = [0.8, 0.8, 0.8]

    def __init__(
        self,
        name: str,
        pos: np.ndarray = np.array([0.0, 0.0, 0.8]),
        half_size: np.ndarray = np.array([0.1, 0.1, 0.001]),
        rgb: Optional[str] = None,
    ):
        """
        A simple primitive fixture as a flat box.

        Args:
            half_size: Half-sizes in [x, y, z] directions. Default creates a 20cm x 20cm x 2mm box.
        """
        self.half_size = half_size

        self.assets = [
            ET.Element(
                "texture",
                type="2d",
                name=f"{name}",
                builtin="flat",
                rgb1=" ".join(map(str, self.DEFAULT_RGB if rgb is None else rgb)),
                width="512",
                height="512",
            ),
            ET.Element(
                "material",
                name=f"{name}",
                texture=f"{name}",
                texuniform="true",
                reflectance="0.05",  # Less reflective than bottle
            ),
        ]

        self.body = ET.Element("body", name=f"{name}_body", pos=array_to_string(pos))

        # Visual geometry
        fixture_vis_geom = ET.Element(
            "geom",
            name=f"{name}_vis",
            pos="0 0 0",
            size=f"{half_size[0]} {half_size[1]} {half_size[2]}",
            type="box",
            material=f"{name}",
            group="1",
            conaffinity="0",
            contype="0",
        )
        self.body.append(fixture_vis_geom)

        # Collision geometry - just a single box since it's already a simple shape
        self.contact_geoms = []
        fixture_collider = ET.Element(
            "geom",
            name=f"{name}_collider",
            type="box",
            pos="0 0 0",
            size=f"{half_size[0]} {half_size[1]} {half_size[2]}",
            solimp="0.998 0.998 0.001",
            solref="0.001 2",
            density="100",
            friction="0.6 0.01 0.001",  # Similar to add_fixture_body friction
        )
        self.body.append(fixture_collider)
        self.contact_geoms.append("fixture_collider")


class PnPBottle(LocoManipulationEnv, DexMGConfigHelper):
    TABLE_GRADIENT: Gradient = Gradient(
        np.array([0.68, 0.34, 0.07, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    )
    DEFAULT_BOTTLE_POS: np.ndarray = np.array([0.4, 0, 0.77])
    BOTTLE_POS_RANGE_X = (-0.08, 0.04)
    BOTTLE_POS_RANGE_Y = (-0.08, 0.08)

    def __init__(self, *args, **kwargs):
        self.objects = {}
        super().__init__(*args, **kwargs)

    def _load_model(self):
        self.mujoco_objects = [self._create_table("table_body", [0.5, 0, 0], [0, 0, np.pi / 2])]

        super()._load_model()

        self.bottle = self._create_bottle()

    @staticmethod
    def _create_table(name: str, position: list[float], euler: list[float]) -> MJCFObject:
        table = MJCFObject(
            name=name,
            mjcf_path=xml_path_completion(
                "objects/omniverse/locomanip/lab_table/model.xml", root=robocasa.models.assets_root
            ),
            scale=1.0,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=10,
            friction=(1, 1, 1),
            static=True,
        )
        table.set_pos(position)
        table.set_euler(euler)
        return table

    def _create_bottle(
        self, name: str = "bottle", rgb: Optional[list[float]] = None
    ) -> PrimitiveBottle:
        bottle = PrimitiveBottle(name=name, radius=0.03, half_height=0.075, rgb=rgb)
        self.model.asset.extend(bottle.assets)
        self.model.worldbody.append(bottle.body)
        self.objects[name] = {"name": f"{name}_body"}
        return bottle

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        if not self.deterministic_reset:
            self._randomize_bottle_placement()
            self._randomize_table_texture()

    def _randomize_bottle_placement(
        self, name: str = "bottle", base_pos: Optional[np.ndarray] = None
    ):
        if not self.deterministic_reset:
            bottle_joint = f"{name}_joint"
            base_pos = self.DEFAULT_BOTTLE_POS if base_pos is None else base_pos

            random_x = self.rng.uniform(*self.BOTTLE_POS_RANGE_X)
            random_y = self.rng.uniform(*self.BOTTLE_POS_RANGE_Y)
            new_pos = base_pos + np.array([random_x, random_y, 0])

            current_qpos = self.sim.data.get_joint_qpos(bottle_joint)
            new_qpos = current_qpos.copy()
            new_qpos[:3] = new_pos

            self.sim.data.set_joint_qpos(bottle_joint, new_qpos)

    def _randomize_table_texture(self):
        table = self.mujoco_objects[0]
        randomize_materials_rgba(
            rng=self.rng, mjcf_obj=table, gradient=self.TABLE_GRADIENT, linear=True
        )

    def _setup_references(self):
        super()._setup_references()

        self.obj_body_id = {}
        for name, model in self.objects.items():
            self.obj_body_id[name] = self.sim.model.body_name2id(model["name"])

    def _check_success(self):
        check_grasp = self._check_grasp(self.robots[0].gripper["right"], self.bottle.contact_geoms)

        bottle_z = self.sim.data.body_xpos[self.obj_body_id["bottle"]][2]
        table_z = self.mujoco_objects[0].top_offset[2]
        check_bottle_in_air = bottle_z > table_z + 0.2
        # check bottle and table collision
        # check_bottle_in_air = not self.check_contact("bottle", "table")
        return check_grasp and check_bottle_in_air

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.objects["bottle"]["name"], obj_type="body"),
        )

    def get_subtask_term_signals(self):
        signals = dict()
        signals["grasp_bottle"] = int(
            self._check_grasp(self.robots[0].gripper["right"], self.bottle.contact_geoms)
        )
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()


def create_shelf(pos: list[float], euler: list[float]) -> MJCFObject:
    shelf = MJCFObject(
        name="shelf_body",
        mjcf_path=xml_path_completion(
            "objects/aigc/shelf/model.xml", root=robocasa.models.assets_root
        ),
        scale=[1.0, 1.0, 1.0],
        solimp=(0.998, 0.998, 0.001),
        solref=(0.001, 1),
        density=10,
        friction=(1, 1, 1),
        static=True,
    )
    shelf.set_pos(pos)
    shelf.set_euler(euler)
    return shelf


class PickBottleShelf(PnPBottle):
    def _load_model(self):
        # Create both the original table and the target table
        self.mujoco_objects = [create_shelf(pos=[0.8, 0.4, 0], euler=[0, 0, np.pi / 2])]

        LocoManipulationEnv._load_model(self)

        self.bottle = self._create_bottle()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        LocoManipulationEnv._reset_internal(self)

        if not self.deterministic_reset:
            # Base position on ground (z=0.075 is bottle radius)
            # Level 2 of shelf
            self._randomize_bottle_placement(base_pos=np.array([0.7, 0.4, 0.376660 + 0.075 + 0.02]))
            self._randomize_table_texture()
            RobotPoseRandomizer.set_arm(self, elbow_qpos=-0.5, shoulder_pitch_qpos=0.5)


class PnPBottleHigh(PnPBottle):
    def _load_model(self):
        self.mujoco_objects = [self._create_table("table_body", [0.5, 0, 0.1], [0, 0, np.pi / 2])]

        LocoManipulationEnv._load_model(self)

        self.bottle = self._create_bottle()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        LocoManipulationEnv._reset_internal(self)

        # Randomize bottle position within +/- 0.1 range on x and y axes
        if not self.deterministic_reset:
            # Base position of the bottle
            base_pos = np.array([0.4, 0, 0.875])

            # Add random offset within +/- 0.1 range for x and y
            random_x = np.random.uniform(-0.1, 0.1)
            random_y = np.random.uniform(-0.1, 0.1)
            # New randomized position (keep z constant)
            new_pos = base_pos + np.array([random_x, random_y, 0])

            # Set the bottle position using the free joint
            # For free joints, qpos includes [x, y, z, qw, qx, qy, qz]
            current_qpos = self.sim.data.get_joint_qpos("bottle_joint")
            new_qpos = current_qpos.copy()
            new_qpos[:3] = new_pos  # Update position (x, y, z)

            self.sim.data.set_joint_qpos("bottle_joint", new_qpos)

    def _setup_observables(self):
        observables = super()._setup_observables()

        @sensor(modality="object")
        def obj_pos(obs_cache):
            return self.sim.data.body_xpos[self.obj_body_id["bottle"]]

        @sensor(modality="object")
        def obj_quat(obs_cache):
            return self.sim.data.body_xquat[self.obj_body_id["bottle"]]

        @sensor(modality="object")
        def obj_linear_vel(obs_cache):
            return self.sim.data.get_body_xvelp("bottle_body")

        @sensor(modality="object")
        def obj_angular_vel(obs_cache):
            return self.sim.data.get_body_xvelr("bottle_body")

        sensors = [obj_pos, obj_quat, obj_linear_vel, obj_angular_vel]
        names = [s.__name__ for s in sensors]

        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def get_privileged_obs_keys(self):
        return {
            "obj_pos": (3,),
            "obj_quat": (4,),
            "obj_linear_vel": (3,),
            "obj_angular_vel": (3,),
        }


class NavPickBottle(PnPBottle):
    """
    Pick-and-Place Bottle environment with robot position randomized at reset.
    """

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.3, -0.16), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class PnPBottleRandRobotPose(NavPickBottle):
    pass


class VisualReach(LocoManipulationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super()._load_model()

        self.create_visual_only_goal_cube()

    def create_visual_only_goal_cube(self):
        cube_tex = ET.Element(
            "texture",
            type="2d",
            name="cube",
            builtin="flat",
            rgb1="1.0 0.0 0.0",
            width="512",
            height="512",
        )
        cube_mat = ET.Element(
            "material",
            name="cube",
            texture="cube",
            texuniform="true",
            reflectance="0.1",
        )
        self.model.asset.append(cube_tex)
        self.model.asset.append(cube_mat)

        self.objects = {}
        cube_body = ET.Element("body", name="cube_body", pos="0.4 0 0.875")

        cube_vis_geom = ET.Element(
            "geom",
            name="cube_vis",
            pos="0 0 0",
            size="0.0375 0.0375 0.0375",
            type="box",
            material="cube",
            group="1",
            conaffinity="0",
            contype="0",
        )

        cube_body.append(cube_vis_geom)
        self.model.worldbody.append(cube_body)
        self.objects["cube"] = {"name": "cube_body"}

    def _setup_references(self):
        super()._setup_references()

        self.obj_body_id = {}
        for name, model in self.objects.items():
            self.obj_body_id[name] = self.sim.model.body_name2id(model["name"])

    def _check_success(self):
        # check_grasp = self._check_grasp(self.robots[0].gripper["right"], self.objects["bottle"])
        # check_reach = self._check_reach(self.objects["bottle"])
        return True

    def _check_reach(self, obj_name):
        raise NotImplementedError
        # To be implemented by the subclass

    def get_object(self):
        return dict(
            cube=dict(obj_name=self.objects["cube"].root_body, obj_type="body"),
        )

    def reset_obj_pos(self):
        # reset object pos randomly around bottle_body pos="0.4 0 0.875"
        init_pos = np.array([0.4, 0, 0.875])
        random_x = np.random.uniform(-0.3, 0.15)
        random_y = np.random.uniform(-0.15, 0.15)
        random_z = np.random.uniform(-0.15, 0.30)
        self.sim.model.body_pos[self.obj_body_id["cube"]] = init_pos + np.array(
            [random_x, random_y, random_z]
        )

    def set_cameras(self):
        super().set_cameras()
        self.mujoco_arena.set_camera(
            camera_name="egoview",
            pos=[0.078, 0, 1.308],
            quat=[0.66491268, 0.24112495, -0.24112507, -0.66453637],
            camera_attribs=dict(fovy="90"),
        )

    def _setup_observables(self):
        observables = super()._setup_observables()

        @sensor(modality="object")
        def obj_pos(obs_cache):
            return self.sim.data.body_xpos[self.obj_body_id["cube"]]

        @sensor(modality="object")
        def obj_quat(obs_cache):
            return self.sim.data.body_xquat[self.obj_body_id["cube"]]

        @sensor(modality="object")
        def obj_linear_vel(obs_cache):
            return self.sim.data.get_body_xvelp("cube_body")

        @sensor(modality="object")
        def obj_angular_vel(obs_cache):
            return self.sim.data.get_body_xvelr("cube_body")

        sensors = [obj_pos, obj_quat, obj_linear_vel, obj_angular_vel]
        names = [s.__name__ for s in sensors]

        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def get_privileged_obs_keys(self):
        return {
            "obj_pos": (3,),
            "obj_quat": (4,),
            "obj_linear_vel": (3,),
            "obj_angular_vel": (3,),
        }


class PnPBottleFixtureToFixture(PnPBottle):
    """
    Task: Robot picks up bottle and places it on a fixture.

    Initialization: bottle rests on a source fixture.

    Idea: by changing the location of the target fixture, we can change the data generation task layout for
    these placement related tasks.
    """

    SEPARATION_THRESH_M: float = 0.0005  # 0.05 cm
    DISTMAX_SCAN_M: float = 0.05  # 5 cm window for distance queries
    _SRC_NAME = "start_fixture"
    _TGT_NAME = "target_fixture"
    _FIXTURE_HALF_SIZE = np.array([0.05, 0.05, 0.001])
    _BOTTLE_HALF_HEIGHT = 0.075
    _X_SRC_RANGE = (0.30, 0.55)
    _X_TGT_RANGE = (0.30, 0.55)
    _Y_SRC_RANGE = (-0.20, -0.05)
    _Y_TGT_RANGE = (0.05, 0.20)
    _SRC_FIXTURE_VISIBLE = False
    _TGT_FIXTURE_VISIBLE = True

    def _load_model(self):
        self.mujoco_objects = [self._create_table("table_body", [0.5, 0, 0], [0, 0, np.pi / 2])]
        LocoManipulationEnv._load_model(self)
        self.bottle = self._create_bottle()
        self._create_fixture(self._SRC_NAME, visible=self._SRC_FIXTURE_VISIBLE, rgb="1 0 0")
        self._create_fixture(self._TGT_NAME, visible=self._TGT_FIXTURE_VISIBLE, rgb="0 1 0")
        self._src_body = f"{self._SRC_NAME}_body"
        self._tgt_body = f"{self._TGT_NAME}_body"
        self._src_coll = f"{self._SRC_NAME}_collider"
        self._tgt_coll = f"{self._TGT_NAME}_collider"

    def _create_fixture(self, name: str, visible: bool, rgb: Optional[str] = None) -> None:
        """Create a flat box fixture; add assets + body to the compiled model."""
        fx = PrimitiveFixture(
            name=name, pos=np.array([0.0, 0.0, 0.0]), half_size=self._FIXTURE_HALF_SIZE, rgb=rgb
        )

        # Make source fixture invisible (keep collision only)
        if not visible:
            # Find the visual geom and hide it
            for child in list(fx.body):
                if child.tag == "geom" and child.get("name") == f"{name}_vis":
                    child.set("rgba", "0 0 0 0")  # invisible visual
                    break

        # Register assets + body into the scene graph
        self.model.asset.extend(fx.assets)
        self.model.worldbody.append(fx.body)

    def _setup_references(self):
        super()._setup_references()
        # Table root body is "<name>_main" (same convention as target_table above)
        self.table_body_id = self.sim.model.body_name2id("table_body_main")
        self.src_fixture_id = self.sim.model.body_name2id(self._src_body)
        self.tgt_fixture_id = self.sim.model.body_name2id(self._tgt_body)

    def _check_success(self) -> bool:
        """Bottle touches target fixture collider and is upright."""
        bottle_on_target = self.check_contact(self.bottle.contact_geoms, [self._tgt_coll])
        bottle_upright = check_obj_upright(self, "bottle", threshold=0.8, symmetric=True)
        return bottle_on_target and bottle_upright

    # --- runtime table height ---
    def _table_top_z(self) -> float:
        base_z = float(self.sim.data.body_xpos[self.table_body_id][2])
        top_offset_z = float(self.mujoco_objects[0].top_offset[2])
        return base_z + top_offset_z

    def _reset_internal(self):
        LocoManipulationEnv._reset_internal(self)

        if not self.deterministic_reset:
            # Sample fixture XY, compute Z from current table pose
            x_src = self.rng.uniform(*self._X_SRC_RANGE)
            y_src = self.rng.uniform(*self._Y_SRC_RANGE)

            x_tgt = self.rng.uniform(*self._X_TGT_RANGE)
            y_tgt = self.rng.uniform(*self._Y_TGT_RANGE)
            # y_tgt = self.rng.uniform(*self._Y_TGT_RANGE)

            z_top = self._table_top_z()  # dynamic table top
            src_pos = np.array([x_src, y_src, z_top])
            tgt_pos = np.array([x_tgt, y_tgt, z_top])

            # Reset fixture body poses (static bodies): write to model; MuJoCo will use it after forward()
            self.sim.model.body_pos[self.src_fixture_id] = src_pos
            self.sim.model.body_pos[self.tgt_fixture_id] = tgt_pos

            # Place bottle on source fixture: top of fixture + bottle half-height + tiny clearance
            bottle_z = z_top + self._FIXTURE_HALF_SIZE[2] + self._BOTTLE_HALF_HEIGHT + 0.002
            qpos = self.sim.data.get_joint_qpos("bottle_joint").copy()
            qpos[:3] = np.array([x_src, y_src, bottle_z])
            qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # upright
            self.sim.data.set_joint_qpos("bottle_joint", qpos)

            self._randomize_table_texture()
            RobotPoseRandomizer.set_pose(self, (-0.3, -0.16), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))

    # --- distance via MuJoCo ---
    def _min_signed_distance_mj(self, geoms_a: list[str], geoms_b: list[str]) -> float:
        model, data = self.sim.model, self.sim.data
        dmin = np.inf
        fromto = np.empty(6, dtype=np.float64)
        a_ids = [model.geom_name2id(n) for n in geoms_a]
        b_ids = [model.geom_name2id(n) for n in geoms_b]
        for ga in a_ids:
            for gb in b_ids:
                dist = mujoco.mj_geomDistance(
                    model._model, data._data, ga, gb, self.DISTMAX_SCAN_M + 0.01, fromto
                )
                dmin = min(dmin, float(dist))
        return dmin

    def get_subtask_term_signals(self) -> dict[str, int]:
        """
        1 iff (no contact between bottle and source fixture) AND
             (min signed distance > DISTMAX_SCAN_M).
        """
        in_contact = self.check_contact(self.bottle.contact_geoms, [self._src_coll])
        min_dist = self._min_signed_distance_mj(self.bottle.contact_geoms, [self._src_coll])
        return {
            "obj_off_source_fixture": int((not in_contact) and (min_dist > self.DISTMAX_SCAN_M))
        }

    def get_object(self) -> dict:
        return dict(
            bottle=dict(obj_name=self.objects["bottle"]["name"], obj_type="body"),
            source_fixture=dict(obj_name=self._src_body, obj_type="body"),
            target_fixture=dict(obj_name=self._tgt_body, obj_type="body"),
        )

    @staticmethod
    def task_config() -> dict:
        task = DexMGConfigHelper.AttrDict()
        # Subtask 1: pick (leave source fixture)
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal="obj_off_source_fixture",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Subtask 2: place on target fixture
        task.task_spec_0.subtask_2 = dict(
            object_ref="target_fixture",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Default filler for task_spec_1, mirroring other tasks
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()


class PnPBottleFixtureToFixtureSourceDemo(PnPBottleFixtureToFixture):
    """
    Environment for collecting source demo for PnPBottleFixtureToFixture tasks.
    """

    _X_SRC_RANGE = (0.375, 0.375)
    _X_TGT_RANGE = (0.375, 0.375)
    _Y_SRC_RANGE = (-0.15, -0.15)
    _Y_TGT_RANGE = (0.1, 0.1)
    _SRC_FIXTURE_VISIBLE = False
    _TGT_FIXTURE_VISIBLE = True


class PnPBottleShelfToTable(PnPBottleFixtureToFixture):
    """
    Task: Robot picks up bottle from a fixture on a shelf and places it on a fixture on a table.

    Initialization: bottle rests on a source fixture on the shelf.
    Target: place bottle on target fixture on the table.
    """

    # Adjust ranges for shelf-to-table layout
    _X_SRC_RANGE = (-0.05, 0.05)  # Shelf position range
    _X_TGT_RANGE = (-0.05 - 0.2, 0.05 - 0.2)  # Table position range
    # TODO: could be better to have some 'center' specified here
    _Y_SRC_RANGE = (-0.05, 0.05)  # Shelf position range
    _Y_TGT_RANGE = (-0.05, 0.05)  # Table position range
    _SRC_FIXTURE_VISIBLE = True
    _TGT_FIXTURE_VISIBLE = True
    _FIXTURE_HALF_SIZE = np.array([0.05, 0.05, 0.001])

    # Shelf height constants (from PnPBottleShelf)
    # _SHELF_HEIGHT = 0.386660  # Level 2 of shelf from the original shelf environment
    _SHELF_HEIGHT = 0.753321 + 0.015  # Level 3 of shelf from the original shelf environment

    def _load_model(self):
        # Create both shelf and table
        self.mujoco_objects = [
            self._create_table("table_body", [0.5, 0.6, 0], [0, 0, np.pi / 2]),
            create_shelf(pos=[0.8, -0.4, 0], euler=[0, 0, np.pi / 2]),
        ]

        LocoManipulationEnv._load_model(self)

        self.bottle = self._create_bottle()
        self._create_fixture(self._SRC_NAME, visible=self._SRC_FIXTURE_VISIBLE, rgb="1 0 0")
        self._create_fixture(self._TGT_NAME, visible=self._TGT_FIXTURE_VISIBLE, rgb="0 1 0")
        self._src_body = f"{self._SRC_NAME}_body"
        self._tgt_body = f"{self._TGT_NAME}_body"
        self._src_coll = f"{self._SRC_NAME}_collider"
        self._tgt_coll = f"{self._TGT_NAME}_collider"

    def _setup_references(self):
        super()._setup_references()
        # Add reference to shelf
        self.shelf_body_id = self.sim.model.body_name2id("shelf_body_main")

    def _shelf_top_z(self) -> float:
        """Get the Z coordinate of the shelf top surface"""
        # Use the same shelf height as in PnPBottleShelf
        return self._SHELF_HEIGHT

    def _shelf_xy(self) -> tuple[float, float]:
        """Get the XY coordinates of the shelf"""
        return self.sim.data.body_xpos[self.shelf_body_id][:2]

    def _table_xy(self) -> tuple[float, float]:
        """Get the XY coordinates of the table"""
        return self.sim.data.body_xpos[self.table_body_id][:2]

    def _reset_internal(self):
        LocoManipulationEnv._reset_internal(self)

        if not self.deterministic_reset:
            # Sample fixture XY positions
            x_src = self.rng.uniform(*self._X_SRC_RANGE)
            y_src = self.rng.uniform(*self._Y_SRC_RANGE)

            x_tgt = self.rng.uniform(*self._X_TGT_RANGE)
            y_tgt = self.rng.uniform(*self._Y_TGT_RANGE)

            # Source fixture on shelf
            z_shelf = self._shelf_top_z()
            x_shelf, y_shelf = self._shelf_xy()
            src_pos = np.array([x_src, y_src, z_shelf])
            src_pos += np.array([x_shelf, y_shelf, 0])

            # table pos
            # Target fixture on table
            z_table = self._table_top_z()
            x_table, y_table = self._table_xy()
            tgt_pos = np.array([x_tgt, y_tgt, z_table])
            tgt_pos += np.array([x_table, y_table, 0])

            # Reset fixture body poses
            self.sim.model.body_pos[self.src_fixture_id] = src_pos
            self.sim.model.body_pos[self.tgt_fixture_id] = tgt_pos

            # Place bottle on source fixture (shelf): top of fixture + bottle half-height + clearance
            bottle_z = z_shelf + self._FIXTURE_HALF_SIZE[2] + self._BOTTLE_HALF_HEIGHT + 0.002
            qpos = self.sim.data.get_joint_qpos("bottle_joint").copy()
            qpos[:3] = np.array([src_pos[0], src_pos[1], bottle_z])
            qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # upright
            self.sim.data.set_joint_qpos("bottle_joint", qpos)

            self._randomize_table_texture()
            RobotPoseRandomizer.set_pose(self, (-0.3, -0.16), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))

    def _randomize_table_texture(self):
        """Randomize textures for the table (shelf texture is static)"""
        # Only randomize the table texture (index 1), not the shelf
        table = self.mujoco_objects[1]
        randomize_materials_rgba(
            rng=self.rng, mjcf_obj=table, gradient=self.TABLE_GRADIENT, linear=True
        )


class PnPBottleTableToTable(PnPBottle):
    def _load_model(self):
        # Create both the original table and the target table
        self.mujoco_objects = [
            self._create_table("table_body", [0.5, 0, 0], [0, 0, np.pi / 2]),
            self._create_table("target_table_body", [0.5, 1.2, 0], [0, 0, np.pi / 2]),
        ]

        LocoManipulationEnv._load_model(self)

        self.bottle = self._create_bottle()

    def _setup_references(self):
        super()._setup_references()

        # Add reference to target table - note the _main suffix
        self.target_table_body_id = self.sim.model.body_name2id("target_table_body_main")

    def _check_success(self):
        """Check if bottle is successfully placed on the target table"""
        bottle_on_table = self.check_contact(self.bottle.contact_geoms, self.mujoco_objects[1])
        bottle_is_upright = check_obj_upright(self, "bottle", threshold=0.8, symmetric=True)
        return bottle_on_table and bottle_is_upright

    def _randomize_table_texture(self):
        """Randomize textures for both tables"""
        # Randomize original table
        original_table = self.mujoco_objects[0]
        randomize_materials_rgba(
            rng=self.rng, mjcf_obj=original_table, gradient=self.TABLE_GRADIENT, linear=True
        )

        # Randomize target table
        target_table = self.mujoco_objects[1]
        randomize_materials_rgba(
            rng=self.rng, mjcf_obj=target_table, gradient=self.TABLE_GRADIENT, linear=True
        )

    def get_object(self):
        return dict(
            bottle=dict(obj_name=self.objects["bottle"]["name"], obj_type="body"),
            target_table=dict(obj_name="target_table_body_main", obj_type="body"),
        )

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="bottle",
            subtask_term_signal="obj_off_table",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Second subtask for placing on target table
        task.task_spec_0.subtask_2 = dict(
            object_ref="target_table",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()

    def get_subtask_term_signals(self):
        """
        Retrieve signals used to define subtask termination conditions.

        Returns:
            dict: Dictionary mapping signal names to their current values
        """
        signals = dict()

        obj_z = self.sim.data.body_xpos[self.obj_body_id["bottle"]][2]
        target_table_pos = self.sim.data.body_xpos[self.target_table_body_id]
        target_table_z = target_table_pos[2] + self.mujoco_objects[1].top_offset[2]

        th = 0.15
        signals["obj_off_table"] = int(obj_z - target_table_z > th)

        return signals


class PickBottleGround(PnPBottle):
    """
    Pick-and-Place Bottle environment with bottle initialized on the ground.
    """

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        LocoManipulationEnv._reset_internal(self)

        if not self.deterministic_reset:
            # Base position on ground (z=0.075 is bottle radius)
            self._randomize_bottle_placement(base_pos=np.ndarray([0.4, 0, 0.075]))
            self._randomize_table_texture()

    def _randomize_table_texture(self):
        pass

    def _check_success(self):
        check_grasp = self._check_grasp(self.robots[0].gripper["right"], "bottle")

        bottle_z = self.sim.data.body_xpos[self.obj_body_id["bottle"]][2]
        ground_z = 0
        check_bottle_in_air = bottle_z > ground_z + 0.2
        # check bottle and table collision
        # check_bottle_in_air = not self.check_contact("bottle", "table")
        return check_grasp and check_bottle_in_air

    def _load_model(self):
        self.mujoco_objects = []

        super(PnPBottle, self)._load_model()
        self._create_bottle()


class PickBottles(PnPBottle):
    BOTTLE_POS_RANGE_X = (-0.08, 0.04)
    BOTTLE_POS_RANGE_Y = (-0.04, 0.04)

    COLOURS: list[list[float]] = [[0.3, 0.7, 0.8], [0.8, 0.4, 0.3]]
    BOTTLES_COUNT = 2
    Y_OFFSET_STEP = 0.1

    @staticmethod
    def _get_bottle_names() -> list[str]:
        return [f"bottle_{i}" for i in range(PickBottles.BOTTLES_COUNT)]

    def _load_model(self):
        self.mujoco_objects = [self._create_table("table_body", [0.5, 0, 0], [0, 0, np.pi / 2])]

        LocoManipulationEnv._load_model(self)

        self.bottles = self._create_bottles()

    def _create_bottles(self) -> list[PrimitiveBottle]:
        bottles = []
        for i, name in enumerate(self._get_bottle_names()):
            rgb = self.COLOURS[i % len(self.COLOURS)]
            bottles.append(self._create_bottle(name=name, rgb=rgb))
        return bottles

    def _reset_internal(self):
        LocoManipulationEnv._reset_internal(self)

        n = len(self.bottles)
        offsets = np.arange(n) - (n - 1) / 2.0
        for i, bottle in enumerate(self.bottles):
            self._randomize_bottle_placement(
                name=bottle.name,
                base_pos=self.DEFAULT_BOTTLE_POS
                + np.array([0, self.Y_OFFSET_STEP * offsets[i], 0]),
            )
            self._randomize_table_texture()

    def _check_success(self):
        for bottle in self.bottles:
            check_grasp = self._check_grasp(
                self.robots[0].gripper["right"], bottle.contact_geoms
            ) or self._check_grasp(self.robots[0].gripper["left"], bottle.contact_geoms)
            bottle_z = self.sim.data.body_xpos[self.obj_body_id[bottle.name]][2]
            table_z = self.mujoco_objects[0].top_offset[2]
            check_bottle_in_air = bottle_z > table_z + 0.2
            if check_grasp and check_bottle_in_air:
                continue
            return False
        return True

    def get_object(self):
        result = {}
        for bottle in self.bottles:
            result[bottle.name] = dict(obj_name=self.objects[bottle.name]["name"], obj_type="body")
        return result

    def get_subtask_term_signals(self):
        signals = dict()
        for bottle in self.bottles:
            signals[f"grasp_{bottle.name}"] = int(
                self._check_grasp(self.robots[0].gripper["right"], bottle.contact_geoms)
                or self._check_grasp(self.robots[0].gripper["left"], bottle.contact_geoms)
            )
        return signals

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()
        bottle_names = PickBottles._get_bottle_names()
        assert len(bottle_names) == 2
        for i, name in enumerate(bottle_names):
            subtask = dict(
                object_ref=name,
                subtask_term_signal=f"grasp_{name}",
                subtask_term_offset_range=None,
                selection_strategy="random",
                selection_strategy_kwargs=None,
                action_noise=0.05,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
            spec_attr = f"task_spec_{i}"
            setattr(getattr(task, spec_attr), "subtask_1", subtask)
        return task.to_dict()


class NavPickBottles(PickBottles):
    """
    PickBottles environment with robot position randomized further from table at reset.
    """

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            RobotPoseRandomizer.set_pose(self, (-0.3, -0.16), (-0.2, 0.2), (-np.pi / 6, np.pi / 6))


class PnPBottlesTableToTable(PickBottles):
    def _load_model(self):
        self.mujoco_objects = [
            self._create_table("table_body", [0.5, 0, 0], [0, 0, np.pi / 2]),
            self._create_table("target_table_body", [0.5, 1.2, 0], [0, 0, np.pi / 2]),
        ]

        LocoManipulationEnv._load_model(self)

        self.bottles = self._create_bottles()

    def _check_success(self):
        """Check if bottles are successfully placed on the target table"""
        for bottle in self.bottles:
            bottle_on_table = self.check_contact(bottle.contact_geoms, self.mujoco_objects[1])
            bottle_is_upright = check_obj_upright(self, bottle.name, threshold=0.8, symmetric=True)
            if bottle_on_table and bottle_is_upright:
                continue
            return False
        return True

    def _setup_references(self):
        super()._setup_references()

        # Add reference to target table - note the _main suffix
        self.target_table_body_id = self.sim.model.body_name2id("target_table_body_main")

    def get_object(self):
        result = super().get_object()
        result["target_table"] = dict(obj_name="target_table_body_main", obj_type="body")
        return result

    @staticmethod
    def task_config():
        task = DexMGConfigHelper.AttrDict()

        bottle_names = PickBottles._get_bottle_names()
        assert len(bottle_names) == 2
        for i, name in enumerate(bottle_names):

            # pick subtask per arm
            subtask = dict(
                object_ref=name,
                subtask_term_signal=f"{name}_off_table",
                subtask_term_offset_range=None,
                selection_strategy="random",
                selection_strategy_kwargs=None,
                action_noise=0.05,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
            spec_attr = f"task_spec_{i}"
            setattr(getattr(task, spec_attr), "subtask_1", subtask)

            # place subtask per arm
            subtask = dict(
                object_ref="target_table",
                subtask_term_signal=None,
                subtask_term_offset_range=None,
                selection_strategy="random",
                selection_strategy_kwargs=None,
                action_noise=0.05,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
            spec_attr = f"task_spec_{i}"
            setattr(getattr(task, spec_attr), "subtask_2", subtask)

        return task.to_dict()

    def get_subtask_term_signals(self):
        signals = dict()
        for bottle in self.bottles:
            obj_z = self.sim.data.body_xpos[self.obj_body_id[bottle.name]][2]
            target_table_pos = self.sim.data.body_xpos[self.target_table_body_id]
            target_table_z = target_table_pos[2] + self.mujoco_objects[1].top_offset[2]
            th = 0.15
            signals[f"{bottle.name}_off_table"] = int(obj_z - target_table_z > th)
        return signals
