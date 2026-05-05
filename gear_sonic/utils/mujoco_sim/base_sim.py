"""MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

import json
import os
import pathlib
from pathlib import Path
import pickle
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ZMQ_PACKED_HEADER_SIZE = 1280
INSPIRE_HAND_COMMAND_OPEN = np.array([1000, 1000, 1000, 1000, 800, 200], dtype=np.float64)
INSPIRE_HAND_COMMAND_CLOSE = np.array([0, 0, 0, 0, 200, 800], dtype=np.float64)
INSPIRE_HAND_SIDE_PREFIX = {
    "left": "left_inspire_L_",
    "right": "right_inspire_R_",
}
INSPIRE_HAND_FINGER_SUFFIXES = {
    "little": ("pinky_proximal_joint", "pinky_intermediate_joint"),
    "ring": ("ring_proximal_joint", "ring_intermediate_joint"),
    "middle": ("middle_proximal_joint", "middle_intermediate_joint"),
    "index": ("index_proximal_joint", "index_intermediate_joint"),
}
INSPIRE_HAND_COMMAND_INDEX = {
    "little": 0,
    "ring": 1,
    "middle": 2,
    "index": 3,
    "thumb_bend": 4,
    "thumb_rot": 5,
}
INSPIRE_HAND_THUMB_BEND_SUFFIXES = (
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
)
INSPIRE_HAND_THUMB_ROT_SUFFIX = "thumb_proximal_yaw_joint"


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = None
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST, dtype=np.float64)
        self.camera_configs = camera_configs
        self.inspire_hand_socket = None
        self.inspire_hand_targets = {"left": {}, "right": {}}
        self.inspire_hand_actuator_index = {}

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen

        self.init_scene()
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = str(pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand()
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body("pelvis").id

        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        self.body_joint_names = []
        self.left_hand_joint_names = []
        self.right_hand_joint_names = []
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
                self.body_joint_names.append(name)
            elif "left_hand" in name:
                self.left_hand_index.append(i)
                self.left_hand_joint_names.append(name)
            elif "right_hand" in name:
                self.right_hand_index.append(i)
                self.right_hand_joint_names.append(name)

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS
        assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
        assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(self.left_hand_index)
        self.right_hand_index = np.array(self.right_hand_index)
        self._init_actuator_indices()
        self._init_inspire_hand_control()

    def _actuator_ids_for_joint_indices(self, joint_indices: np.ndarray, label: str) -> np.ndarray:
        actuator_ids = []
        for joint_id in joint_indices:
            matches = np.where(self.mj_model.actuator_trnid[:, 0] == int(joint_id))[0]
            if len(matches) == 0:
                joint_name = self.mj_model.joint(int(joint_id)).name
                raise ValueError(f"No actuator found for {label} joint '{joint_name}'")
            actuator_ids.append(int(matches[0]))
        return np.array(actuator_ids, dtype=np.int64)

    def _init_actuator_indices(self):
        self.body_actuator_index = self._actuator_ids_for_joint_indices(
            self.body_joint_index, "body"
        )
        self.left_hand_actuator_index = self._actuator_ids_for_joint_indices(
            self.left_hand_index, "left hand"
        )
        self.right_hand_actuator_index = self._actuator_ids_for_joint_indices(
            self.right_hand_index, "right hand"
        )

        self.torques = np.zeros(self.mj_model.nu, dtype=np.float64)
        self.ctrl_limit = np.full(self.mj_model.nu, np.inf, dtype=np.float64)
        limit_count = min(len(self.torque_limit), self.mj_model.nu)
        self.ctrl_limit[:limit_count] = self.torque_limit[:limit_count]

    def _joint_target_from_unit(self, joint_name: str, unit_value: float) -> float:
        joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Inspire hand joint '{joint_name}' is missing from the MuJoCo model")
        lower, upper = self.mj_model.jnt_range[joint_id]
        unit = float(np.clip(unit_value, 0.0, 1.0))
        return float(lower + unit * (upper - lower))

    def _inspire_command_to_joint_targets(self, side: str, command: np.ndarray) -> Dict[str, float]:
        command = np.asarray(command, dtype=np.float64).reshape(6)
        open_values = np.asarray(
            self.config.get("INSPIRE_HAND_COMMAND_OPEN", INSPIRE_HAND_COMMAND_OPEN),
            dtype=np.float64,
        )
        close_values = np.asarray(
            self.config.get("INSPIRE_HAND_COMMAND_CLOSE", INSPIRE_HAND_COMMAND_CLOSE),
            dtype=np.float64,
        )
        denom = close_values - open_values
        units = np.zeros(6, dtype=np.float64)
        valid = np.abs(denom) > 1e-8
        units[valid] = (command[valid] - open_values[valid]) / denom[valid]
        units = np.clip(units, 0.0, 1.0)

        prefix = INSPIRE_HAND_SIDE_PREFIX[side]
        targets = {}
        for channel, suffixes in INSPIRE_HAND_FINGER_SUFFIXES.items():
            unit = units[INSPIRE_HAND_COMMAND_INDEX[channel]]
            for suffix in suffixes:
                joint_name = prefix + suffix
                targets[joint_name] = self._joint_target_from_unit(joint_name, unit)

        thumb_bend = units[INSPIRE_HAND_COMMAND_INDEX["thumb_bend"]]
        for suffix in INSPIRE_HAND_THUMB_BEND_SUFFIXES:
            joint_name = prefix + suffix
            targets[joint_name] = self._joint_target_from_unit(joint_name, thumb_bend)

        thumb_rot = units[INSPIRE_HAND_COMMAND_INDEX["thumb_rot"]]
        joint_name = prefix + INSPIRE_HAND_THUMB_ROT_SUFFIX
        targets[joint_name] = self._joint_target_from_unit(joint_name, thumb_rot)
        return targets

    def _init_inspire_hand_control(self):
        self.inspire_hand_actuator_index = {}
        for prefix in INSPIRE_HAND_SIDE_PREFIX.values():
            suffixes = [INSPIRE_HAND_THUMB_ROT_SUFFIX, *INSPIRE_HAND_THUMB_BEND_SUFFIXES]
            for finger_suffixes in INSPIRE_HAND_FINGER_SUFFIXES.values():
                suffixes.extend(finger_suffixes)
            for suffix in suffixes:
                joint_name = prefix + suffix
                actuator_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
                )
                if actuator_id >= 0:
                    self.inspire_hand_actuator_index[joint_name] = actuator_id

        if self.inspire_hand_actuator_index:
            self.inspire_hand_targets = {
                "left": self._inspire_command_to_joint_targets("left", INSPIRE_HAND_COMMAND_OPEN),
                "right": self._inspire_command_to_joint_targets("right", INSPIRE_HAND_COMMAND_OPEN),
            }

        if not self.config.get("ENABLE_INSPIRE_HAND_ZMQ", False):
            return

        if not self.inspire_hand_actuator_index:
            print("Warning: ENABLE_INSPIRE_HAND_ZMQ is set, but no Inspire hand actuators exist.")
            return

        try:
            import zmq
        except ImportError:
            print("Warning: pyzmq is not installed; Inspire hand ZMQ control is disabled.")
            return

        host = self.config.get("INSPIRE_HAND_ZMQ_HOST", "localhost")
        port = int(self.config.get("INSPIRE_HAND_ZMQ_PORT", 5556))
        self.inspire_hand_topic = self.config.get("INSPIRE_HAND_ZMQ_TOPIC", "inspire_hand")
        self.inspire_hand_header_size = int(
            self.config.get("INSPIRE_HAND_ZMQ_HEADER_SIZE", ZMQ_PACKED_HEADER_SIZE)
        )
        self._inspire_zmq = zmq
        self.inspire_hand_context = zmq.Context.instance()
        self.inspire_hand_socket = self.inspire_hand_context.socket(zmq.SUB)
        self.inspire_hand_socket.setsockopt_string(zmq.SUBSCRIBE, self.inspire_hand_topic)
        self.inspire_hand_socket.setsockopt(zmq.CONFLATE, 1)
        self.inspire_hand_socket.connect(f"tcp://{host}:{port}")
        print(
            f"[InspireHandSim] Listening on tcp://{host}:{port} topic='{self.inspire_hand_topic}'"
        )

    def _decode_zmq_fields(self, payload: bytes, fields: list[dict]) -> Dict[str, np.ndarray]:
        dtypes = {
            "f32": np.float32,
            "f64": np.float64,
            "i32": np.int32,
            "u8": np.uint8,
        }
        decoded = {}
        offset = 0
        for field in fields:
            dtype = dtypes[field["dtype"]]
            shape = tuple(field.get("shape", [1]))
            count = int(np.prod(shape))
            nbytes = np.dtype(dtype).itemsize * count
            decoded[field["name"]] = (
                np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
                .astype(np.float64)
                .reshape(shape)
            )
            offset += nbytes
        return decoded

    def _poll_inspire_hand_zmq(self):
        if self.inspire_hand_socket is None:
            return

        topic = self.inspire_hand_topic.encode("utf-8")
        for _ in range(8):
            try:
                message = self.inspire_hand_socket.recv(flags=self._inspire_zmq.NOBLOCK)
            except self._inspire_zmq.Again:
                return

            if not message.startswith(topic):
                continue
            header_start = len(topic)
            header_end = header_start + self.inspire_hand_header_size
            header_raw = message[header_start:header_end].split(b"\0", 1)[0]
            if not header_raw:
                continue
            header = json.loads(header_raw.decode("utf-8"))
            values = self._decode_zmq_fields(message[header_end:], header.get("fields", []))
            for side in ("left", "right"):
                if side in values:
                    self.inspire_hand_targets[side] = self._inspire_command_to_joint_targets(
                        side, values[side]
                    )

    def _write_inspire_hand_targets(self, ctrl: np.ndarray):
        for targets in self.inspire_hand_targets.values():
            for joint_name, target in targets.items():
                actuator_id = self.inspire_hand_actuator_index.get(joint_name)
                if actuator_id is not None:
                    ctrl[actuator_id] = target

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body("torso_link").xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_actuator_index]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_actuator_index]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[
                self.right_hand_actuator_index
            ]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.elastic_band:
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        ctrl = np.zeros(self.mj_model.nu, dtype=np.float64)
        ctrl[self.body_actuator_index] = body_torques
        if self.num_hand_dof > 0:
            ctrl[self.left_hand_actuator_index] = hand_torques[: self.num_hand_dof]
            ctrl[self.right_hand_actuator_index] = hand_torques[self.num_hand_dof :]

        self._poll_inspire_hand_zmq()
        self._write_inspire_hand_targets(ctrl)
        ctrl = np.clip(ctrl, -self.ctrl_limit, self.ctrl_limit)

        self.mj_data.ctrl[:] = ctrl
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.check_fall()

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            elif "mjcf_name" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["mjcf_name"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def check_fall(self):
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Only 'default' is supported in this minimal build."
            )

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                self.sim_env.sim_step()
                now = time.time()
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            if getattr(self.sim_env, "inspire_hand_socket", None) is not None:
                self.sim_env.inspire_hand_socket.close(0)
                self.sim_env.inspire_hand_socket = None
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)
