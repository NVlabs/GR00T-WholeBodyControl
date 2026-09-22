"""Sonic VLA data exporter for G1 equipped with BrainCo Revo2 hands.

This is intentionally separate from ``run_data_exporter.py``.  Body state is
read from the regular ``g1_debug`` ZMQ stream, while the six BrainCo motors are
read directly from the DDS topics exposed by ``brainco_hand_service``:

  - observation: ``rt/brainco/{left,right}/state`` (MotorStates_)
  - action:      ``rt/brainco/{left,right}/cmd``   (MotorCmds_)

The resulting state/action schema has 41 DOF: 29 G1 body joints plus six
BrainCo motors per hand.  BrainCo values remain in their native normalized
[0, 1] coordinates and motor order:

    thumb, thumb_aux, index, middle, ring, pinky

Run from the repository root, for example:

    python gear_sonic/scripts/run_data_exporter_brainco.py \
        --task-prompt "pick up the cup" --brainco-network-interface eth0
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
    get_wrist_camera_features,
    get_wrist_camera_modality_config,
)
from gear_sonic.scripts.run_data_exporter import (
    GrootDataCollector,
    SonicDataExporterConfig,
    unpack_pose_message,
)
from gear_sonic.utils.data_collection.text_to_speech import TextToSpeech
from gear_sonic.utils.data_collection.zmq_state_subscriber import poll_robot_config_zmq

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
except ImportError as exc:  # pragma: no cover - depends on the robot environment
    ChannelFactoryInitialize = None
    ChannelSubscriber = None
    MotorCmds_ = None
    MotorStates_ = None
    _BRAINCO_DDS_IMPORT_ERROR = exc
else:
    _BRAINCO_DDS_IMPORT_ERROR = None


BRAINCO_MOTOR_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
BRAINCO_NUM_MOTORS = len(BRAINCO_MOTOR_NAMES)
BRAINCO_MAX_AGE_SEC = 0.25


@dataclass
class BraincoDataExporterConfig(SonicDataExporterConfig):
    """CLI configuration for native 6-DOF BrainCo dataset collection."""

    brainco_dds_domain_id: int = 0
    """DDS domain used by brainco_hand_service."""

    brainco_network_interface: str | None = "wlp128s20f3"
    """DDS network interface, for example eth0. None enables auto-selection."""

    brainco_message_timeout: float = 10.0
    """Seconds to wait for both hand state messages (0 = forever)."""


class BraincoHandDDSSubscriber:
    """Thread-safe latest-value subscriber for BrainCo command and feedback."""

    def __init__(self, domain_id: int = 0, network_interface: str | None = None):
        if _BRAINCO_DDS_IMPORT_ERROR is not None:
            raise ImportError(
                "unitree_sdk2py with MotorCmds_/MotorStates_ is required for BrainCo recording"
            ) from _BRAINCO_DDS_IMPORT_ERROR

        ChannelFactoryInitialize(domain_id, network_interface)
        self._lock = threading.Lock()
        self._values: dict[str, np.ndarray | float | None] = {
            "left_state": None,
            "right_state": None,
            "left_command": None,
            "right_command": None,
            "left_state_time": None,
            "right_state_time": None,
            "left_command_time": None,
            "right_command_time": None,
        }

        self._subscribers = [
            ChannelSubscriber("rt/brainco/left/state", MotorStates_),
            ChannelSubscriber("rt/brainco/right/state", MotorStates_),
            ChannelSubscriber("rt/brainco/left/cmd", MotorCmds_),
            ChannelSubscriber("rt/brainco/right/cmd", MotorCmds_),
        ]
        callbacks = [
            lambda msg: self._set("left_state", self._state_q(msg)),
            lambda msg: self._set("right_state", self._state_q(msg)),
            lambda msg: self._set("left_command", self._command_q(msg)),
            lambda msg: self._set("right_command", self._command_q(msg)),
        ]
        for subscriber, callback in zip(self._subscribers, callbacks, strict=True):
            subscriber.Init(callback, 10)

        print("[BrainCo] DDS subscriptions initialized for left/right cmd and state")

    @staticmethod
    def _validate(values: np.ndarray, source: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size != BRAINCO_NUM_MOTORS:
            raise ValueError(
                f"{source} contains {values.size} motors; expected {BRAINCO_NUM_MOTORS}"
            )
        return np.clip(values, 0.0, 1.0)

    @classmethod
    def _state_q(cls, msg) -> np.ndarray:
        return cls._validate([motor.q for motor in msg.states], "MotorStates_")

    @classmethod
    def _command_q(cls, msg) -> np.ndarray:
        return cls._validate([motor.q for motor in msg.cmds], "MotorCmds_")

    def _set(self, key: str, value: np.ndarray) -> None:
        with self._lock:
            self._values[key] = value
            self._values[f"{key}_time"] = time.monotonic()

    def snapshot(self) -> dict[str, np.ndarray | float | None]:
        with self._lock:
            return {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._values.items()
            }

    def wait_until_ready(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec if timeout_sec > 0 else None
        # Command topics are event-driven: an idle hand may not publish a command
        # at all.  Only feedback is therefore required before recording starts.
        required = ("left_state", "right_state")
        print("[BrainCo] Waiting for left/right DDS state messages ...")
        while True:
            snapshot = self.snapshot()
  
            missing = [key for key in required if snapshot[key] is None]
            if not missing:
                print("[BrainCo] Receiving both required DDS state streams")
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for BrainCo DDS streams: {missing}")
            time.sleep(0.05)

    def close(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.Close()
            except Exception:
                pass


class BraincoSchema:
    """Build 41-DOF arrays while retaining the standard G1 body ordering."""

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.left_indices = sorted(robot_model.get_joint_group_indices("left_hand"))
        self.right_indices = sorted(robot_model.get_joint_group_indices("right_hand"))
        self.hand_indices = set(self.left_indices + self.right_indices)
        self.left_insert_at = self.left_indices[0]
        self.right_insert_at = self.right_indices[0]

        self.joint_names: list[str] = []
        self.old_to_new: dict[int, int] = {}
        self.left_slice: slice | None = None
        self.right_slice: slice | None = None
        output_index = 0
        for index, name in enumerate(robot_model.joint_names):
            if index == self.left_insert_at:
                start = output_index
                self.joint_names.extend(f"left_brainco_{name}" for name in BRAINCO_MOTOR_NAMES)
                output_index += BRAINCO_NUM_MOTORS
                self.left_slice = slice(start, output_index)
            if index == self.right_insert_at:
                start = output_index
                self.joint_names.extend(f"right_brainco_{name}" for name in BRAINCO_MOTOR_NAMES)
                output_index += BRAINCO_NUM_MOTORS
                self.right_slice = slice(start, output_index)
            if index not in self.hand_indices:
                self.old_to_new[index] = output_index
                self.joint_names.append(name)
                output_index += 1

        if self.left_slice is None or self.right_slice is None:
            raise RuntimeError("Could not locate hand groups in the G1 robot model")
        if len(self.joint_names) != 41:
            raise RuntimeError(f"Expected a 41-DOF BrainCo schema, got {len(self.joint_names)}")

    def assemble(
        self, body_values: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray
    ) -> np.ndarray:
        full = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=np.asarray(body_values),
        )
        result: list[float] = []
        for index, value in enumerate(full):
            if index == self.left_insert_at:
                result.extend(np.asarray(left_hand, dtype=np.float64).reshape(6))
            if index == self.right_insert_at:
                result.extend(np.asarray(right_hand, dtype=np.float64).reshape(6))
            if index not in self.hand_indices:
                result.append(float(value))
        return np.asarray(result, dtype=np.float64)

    def body_configuration_for_fk(self, body_values: np.ndarray) -> np.ndarray:
        return self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=np.asarray(body_values)
        )


def get_brainco_features(robot_model, schema: BraincoSchema) -> dict:
    features = deepcopy(get_features_sonic_vla(robot_model))
    for key in ("observation.state", "action.wbc"):
        features[key]["shape"] = (len(schema.joint_names),)
        features[key]["names"] = schema.joint_names
    for key in ("teleop.left_hand_joints", "teleop.right_hand_joints"):
        features[key]["shape"] = (BRAINCO_NUM_MOTORS,)
        features[key]["names"] = list(BRAINCO_MOTOR_NAMES)
    return features


def get_brainco_modality_config(robot_model, schema: BraincoSchema) -> dict:
    config = deepcopy(get_modality_config_sonic_vla(robot_model))
    for group in ("left_leg", "right_leg", "waist", "left_arm", "right_arm"):
        mapped = [
            schema.old_to_new[index]
            for index in robot_model.get_joint_group_indices(group)
        ]
        config["state"][group] = {"start": min(mapped), "end": max(mapped) + 1}
    config["state"]["left_hand"] = {
        "start": schema.left_slice.start,
        "end": schema.left_slice.stop,
    }
    config["state"]["right_hand"] = {
        "start": schema.right_slice.start,
        "end": schema.right_slice.stop,
    }
    config["action"]["left_hand_joints"]["end"] = BRAINCO_NUM_MOTORS
    config["action"]["right_hand_joints"]["end"] = BRAINCO_NUM_MOTORS
    return config


class BraincoGrootDataCollector(GrootDataCollector):
    def __init__(self, *args, brainco_subscriber, brainco_schema, **kwargs):
        self._brainco_subscriber = brainco_subscriber
        self._brainco_schema = brainco_schema
        super().__init__(*args, **kwargs)

    @staticmethod
    def _fresh(snapshot: dict, key: str) -> np.ndarray | None:
        value = snapshot.get(key)
        timestamp = snapshot.get(f"{key}_time")
        if value is None or timestamp is None:
            return None
        if time.monotonic() - float(timestamp) > BRAINCO_MAX_AGE_SEC:
            return None
        return np.asarray(value, dtype=np.float32)

    def _handle_pose_message(self, raw: bytes) -> None:
        super()._handle_pose_message(raw)
        if self.latest_sonic_msg is None:
            return
        try:
            pose_data = unpack_pose_message(raw, topic="pose")
            for side in ("left", "right"):
                value = pose_data.get(f"{side}_brainco_q")
                if value is not None:
                    self.latest_sonic_msg[f"{side}_brainco_q"] = np.asarray(
                        value, dtype=np.float32
                    ).reshape(BRAINCO_NUM_MOTORS)
        except Exception as exc:
            print(f"[BrainCo] Warning: could not read pose command fields: {exc}")

    def _add_data_frame_sonic(self, t_start: float) -> bool:
        assert self.latest_proprio_msg is not None
        proprio = self.latest_proprio_msg
        snapshot = self._brainco_subscriber.snapshot()

        left_state = self._fresh(snapshot, "left_state")
        right_state = self._fresh(snapshot, "right_state")
        if any(value is None for value in (left_state, right_state)):
            self._print_and_say(
                "Waiting for fresh BrainCo left/right state messages", say=False
            )
            return False

        # Commands are sparse events, not periodic state.  Hold the latest target
        # indefinitely; before the first command for a hand, use its measured
        # position so an idle hand neither blocks collection nor gets a bogus
        # zero-valued action.
        left_action = snapshot.get("left_command")
        right_action = snapshot.get("right_command")
        if left_action is None:
            left_action = left_state
        if right_action is None:
            right_action = right_state
        left_action = np.asarray(left_action, dtype=np.float32)
        right_action = np.asarray(right_action, dtype=np.float32)

        whole_q = self._brainco_schema.assemble(proprio["body_q"], left_state, right_state)
        whole_action_wbc = self._brainco_schema.assemble(
            proprio["last_action"], left_action, right_action
        )

        fk_q = self._brainco_schema.body_configuration_for_fk(proprio["body_q"])
        self.robot_model.cache_forward_kinematics(fk_q)
        eef_parts = []
        for side in ("left", "right"):
            placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            quat = R.from_matrix(placement.rotation).as_quat(scalar_first=True)
            eef_parts.append(np.concatenate([placement.translation[:3], quat]))

        frame_data = {
            "observation.state": whole_q,
            "observation.eef_state": np.concatenate(eef_parts),
            "action.wbc": whole_action_wbc,
        }
        self._add_cpp_state_features(frame_data, proprio)
        sonic_latency_ms = self._add_sonic_pose_features(frame_data)

        # Teleop hand targets are the actual DDS commands sent to the service.
        frame_data["teleop.left_hand_joints"] = left_action.copy()
        frame_data["teleop.right_hand_joints"] = right_action.copy()

        self._add_images_to_frame_data(frame_data)
        self._log_latency_periodic(sonic_latency_ms)
        self.data_exporter.add_frame(frame_data)
        return self._finalize_frame(t_start)

    def save_and_cleanup(self):
        try:
            super().save_and_cleanup()
        finally:
            self._brainco_subscriber.close()


def main(config: BraincoDataExporterConfig) -> None:
    robot_model = get_g1_robot_model()
    schema = BraincoSchema(robot_model)
    features = get_brainco_features(robot_model, schema)
    modality_config = get_brainco_modality_config(robot_model, schema)

    if config.record_wrist_cameras:
        features.update(get_wrist_camera_features())
        wrist_modality = get_wrist_camera_modality_config()
        for key, value in wrist_modality.items():
            modality_config.setdefault(key, {}).update(value)

    print("config.brainco_network_interface",config.brainco_network_interface)
    brainco_subscriber = BraincoHandDDSSubscriber(
        domain_id=config.brainco_dds_domain_id,
        network_interface=config.brainco_network_interface,
    )
    try:
        brainco_subscriber.wait_until_ready(config.brainco_message_timeout)
    except Exception:
        brainco_subscriber.close()
        raise

    robot_config = poll_robot_config_zmq(
        config.state_zmq_host, config.state_zmq_port, config.robot_config_timeout
    )
    data_exporter = Gr00tDataExporter.create(
        save_root=f"{config.root_output_dir}/{config.dataset_name}",
        fps=config.data_collection_frequency,
        features=features,
        modality_config=modality_config,
        task=config.task_prompt,
        script_config={
            **robot_config,
            "record_wrist_cameras": config.record_wrist_cameras,
            "hand_type": "brainco_revo2",
            "hand_dof": BRAINCO_NUM_MOTORS,
            "hand_coordinates": "normalized_0_to_1",
            "hand_motor_order": list(BRAINCO_MOTOR_NAMES),
            "brainco_state_source": "dds_feedback",
            "brainco_action_source": "dds_command",
        },
    )

    text_to_speech = TextToSpeech() if config.text_to_speech else None
    collector = BraincoGrootDataCollector(
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        robot_model=robot_model,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        text_to_speech=text_to_speech,
        sonic_data_zmq_host=config.sonic_zmq_host,
        sonic_data_zmq_port=config.sonic_zmq_port,
        state_zmq_host=config.state_zmq_host,
        state_zmq_port=config.state_zmq_port,
        episode_status_zmq_port=config.episode_status_zmq_port,
        episode_status_publish_hz=config.episode_status_publish_hz,
        brainco_subscriber=brainco_subscriber,
        brainco_schema=schema,
    )
    collector.run()


if __name__ == "__main__":
    cli_config = tyro.cli(BraincoDataExporterConfig)
    if cli_config.dataset_name is None:
        cli_config.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    main(cli_config)
