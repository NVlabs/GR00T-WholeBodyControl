"""Local clean-observation SONIC true23 teleoperation in CPU MuJoCo.

This module has no DDS, Unitree, hardware, or network-actuation path.  It can
replay the pinned DadDance reference or consume robot-independent PICO causal
reference packets from the existing localhost ZMQ publisher.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import onnxruntime as ort

from gear_sonic.utils.g1_23dof_contract import (
    CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES,
    HARDWARE_23_ACTION_SCALE,
    HARDWARE_23_JOINT_NAMES,
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_TO_CANONICAL_IL29,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_23dof_xr24_soma_stream import (
    CAUSAL_HISTORY_CONTROL_DERIVATIVE_CONTRACT,
    CAUSAL_HISTORY_ENCODER_TERMS_SCHEMA_VERSION,
    CAUSAL_HISTORY_PROFILE,
    CAUSAL_HISTORY_REFERENCE_TERMS_KIND,
    causal_history_stream_contract,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import (
    _body_velocity,
    _projected_gravity,
    _quaternion_conjugate,
    _quaternion_matrix,
    _quaternion_multiply,
    _rotation_6d,
    prepare_true23_model,
    term_major_history,
    world_angular_velocity_to_body,
)

ENCODER_SHA256 = "733353148bef1eb8dd83a96416b7a89f0b5c3530ceb9e0cec9c25fdb04f56ff2"
DECODER_SHA256 = "dc4b6cf4681eafaff6bb6d70d0aad136e9a3a184337490dc32a511e45b31ec9a"
RELEASED_RETAINED_KP = np.asarray(
    (
        99.098427782326,
        99.098427782326,
        40.17923847367,
        99.098427782326,
        28.501246197378,
        28.501246197378,
        99.098427782326,
        99.098427782326,
        40.17923847367,
        99.098427782326,
        28.501246197378,
        28.501246197378,
        40.17923847367,
        *([14.250623098689] * 10),
    ),
    dtype=np.float64,
)
RELEASED_RETAINED_KD = np.asarray(
    (
        6.308801853677,
        6.308801853677,
        2.557889765101,
        6.308801853677,
        0.907222843318,
        0.907222843318,
        6.308801853677,
        6.308801853677,
        2.557889765101,
        6.308801853677,
        0.907222843318,
        0.907222843318,
        2.557889765101,
        *([0.907222843318] * 10),
    ),
    dtype=np.float64,
)
MOTION_SHA256 = "feced9269f9b26e31b16c64b006e4c9e5233692bf38097fb0d160cd7417d3923"
CONTROL_PERIOD_NS = 20_000_000
STRICT_RAW_ABS_MAX = 10.0
MINIMUM_BASE_HEIGHT_M = 0.45
MAXIMUM_BASE_TILT_RAD = 1.0
VELOCITY_FALLBACK_SHA256 = "2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28"
LINEAR_VELOCITY_JUMP_TRIGGER_MPS = 0.25
VERTICAL_VELOCITY_JUMP_TRIGGER_MPS = 0.15
ANGULAR_VELOCITY_JUMP_TRIGGER_RADPS = 0.35
FALLBACK_TILT_TRIGGER_RAD = 0.25
BALANCED_UPPER_BODY_ARM_BLEND = 0.40
BALANCED_UPPER_BODY_FIRST_HARDWARE_INDEX = 15
NATIVE_TO_IL29 = np.asarray(NATIVE_IL23_TO_CANONICAL_IL29, dtype=np.int64)
MJ_TO_NATIVE = np.asarray(MUJOCO_TO_ISAACLAB_DOF, dtype=np.int64)
NATIVE_TO_MJ = np.asarray(ISAACLAB_TO_MUJOCO_DOF, dtype=np.int64)
DEFAULT_NATIVE = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float64)[MJ_TO_NATIVE]
LOWER_BODY_IL29 = np.asarray(CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES, dtype=np.int64)

REFERENCE_TERM_KEYS = {
    "schema_version",
    "kind",
    "reference_profile",
    "reference_contract_sha256",
    "pico_anchor_source_frame_index",
    "pico_anchor_monotonic_ns",
    "control_source_frame_index",
    "control_monotonic_ns",
    "causal_history_lower_body",
    "vr_3point_local_target",
    "vr_3point_local_orn_target",
    "reference_anchor_quaternion_xyzw",
    "anchor_joint_pos_il29",
    "proof_joint_pos_il29",
    "q_ref23_native",
    "qd_ref23_native",
    "control_derivative_contract",
    "sdk_derivatives_consumed",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: Any, width: int, context: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{context} must be a {width}-value list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{context} must contain only numeric values")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{context} contains NaN or Inf")
    return result


def validate_reference_terms(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on exact robot-independent PICO transport schema."""

    if not isinstance(packet, Mapping) or set(packet) != REFERENCE_TERM_KEYS:
        raise ValueError("causal reference transport keys mismatch")
    contract = causal_history_stream_contract()
    if (
        packet["schema_version"] != CAUSAL_HISTORY_ENCODER_TERMS_SCHEMA_VERSION
        or packet["kind"] != CAUSAL_HISTORY_REFERENCE_TERMS_KIND
        or packet["reference_profile"] != CAUSAL_HISTORY_PROFILE
        or packet["reference_contract_sha256"] != contract["contract_sha256"]
        or packet["control_derivative_contract"] != CAUSAL_HISTORY_CONTROL_DERIVATIVE_CONTRACT
        or packet["sdk_derivatives_consumed"] is not False
    ):
        raise ValueError("causal reference transport contract mismatch")
    integer_names = (
        "pico_anchor_source_frame_index",
        "pico_anchor_monotonic_ns",
        "control_source_frame_index",
        "control_monotonic_ns",
    )
    if any(type(packet[name]) is not int for name in integer_names):
        raise ValueError("causal reference transport indices/times must be exact integers")
    if (
        packet["control_source_frame_index"] != packet["pico_anchor_source_frame_index"] + 1
        or packet["control_monotonic_ns"] != packet["pico_anchor_monotonic_ns"] + CONTROL_PERIOD_NS
    ):
        raise ValueError("causal reference transport is not exact q9/q10")
    for name, width in (
        ("causal_history_lower_body", 240),
        ("vr_3point_local_target", 9),
        ("vr_3point_local_orn_target", 12),
        ("reference_anchor_quaternion_xyzw", 4),
        ("anchor_joint_pos_il29", 29),
        ("proof_joint_pos_il29", 29),
        ("q_ref23_native", 23),
        ("qd_ref23_native", 23),
    ):
        _finite_vector(packet[name], width, name)
    anchor = _finite_vector(packet["q_ref23_native"], 23, "q_ref23_native")
    velocity = _finite_vector(packet["qd_ref23_native"], 23, "qd_ref23_native")
    proof = _finite_vector(packet["proof_joint_pos_il29"], 29, "proof_joint_pos_il29")
    expected_proof = anchor + velocity * 0.02
    if not np.allclose(proof[NATIVE_TO_IL29], expected_proof, rtol=0.0, atol=1.0e-9):
        raise ValueError("causal reference native q9/q10 proof mismatch")
    return {
        "anchor_index": int(packet["pico_anchor_source_frame_index"]),
        "control_index": int(packet["control_source_frame_index"]),
        "anchor_monotonic_ns": int(packet["pico_anchor_monotonic_ns"]),
        "control_monotonic_ns": int(packet["control_monotonic_ns"]),
        "contract_sha256": contract["contract_sha256"],
    }


def pico_reference_policy_history(packet: Mapping[str, Any]) -> list[np.ndarray]:
    """Rebuild exact q0..q9 native23 policy history carried by one Pico packet."""

    validate_reference_terms(packet)
    anchor = _finite_vector(packet["anchor_joint_pos_il29"], 29, "anchor IL29")
    causal = _finite_vector(packet["causal_history_lower_body"], 240, "causal lower-body history")
    positions = causal[:120].reshape(10, 12)
    velocities = causal[120:].reshape(10, 12)
    q_il29 = np.repeat(anchor.reshape(1, 29), 10, axis=0)
    dq_il29 = np.zeros((10, 29), dtype=np.float64)
    q_il29[:, LOWER_BODY_IL29] = positions
    dq_il29[:, LOWER_BODY_IL29] = velocities
    dq_il29[9, NATIVE_TO_IL29] = _finite_vector(packet["qd_ref23_native"], 23, "q9 native velocity")
    scale = np.asarray(HARDWARE_23_ACTION_SCALE, dtype=np.float64)
    default = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float64)
    result: list[np.ndarray] = []
    for index in range(10):
        q_native = q_il29[index, NATIVE_TO_IL29]
        q_hardware = q_native[NATIVE_TO_MJ]
        safe_hardware = (q_hardware - default) / scale
        relative29 = np.zeros(29, dtype=np.float64)
        relative29[NATIVE_TO_IL29] = q_native - DEFAULT_NATIVE
        action29 = np.zeros(29, dtype=np.float64)
        action29[NATIVE_TO_IL29] = safe_hardware[MJ_TO_NATIVE]
        frame = np.concatenate(
            (
                np.zeros(3, dtype=np.float64),
                relative29,
                dq_il29[index],
                action29,
                np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
            )
        ).astype(np.float32)
        if frame.shape != (93,) or not np.isfinite(frame).all():
            raise ValueError("Pico policy startup history is invalid")
        result.append(frame)
    return result


def encoder267_from_reference(packet: Mapping[str, Any], buffered_robot_pelvis_q9_wxyz: np.ndarray) -> np.ndarray:
    validate_reference_terms(packet)
    robot = np.asarray(buffered_robot_pelvis_q9_wxyz, dtype=np.float64)
    if robot.shape != (4,) or not np.isfinite(robot).all() or np.linalg.norm(robot) < 0.5:
        raise ValueError("buffered robot pelvis quaternion is invalid")
    robot /= np.linalg.norm(robot)
    reference_xyzw = _finite_vector(packet["reference_anchor_quaternion_xyzw"], 4, "reference anchor quaternion")
    reference_wxyz = reference_xyzw[[3, 0, 1, 2]]
    reference_wxyz /= np.linalg.norm(reference_wxyz)
    relative = _quaternion_multiply(_quaternion_conjugate(robot), reference_wxyz)
    result = np.concatenate(
        (
            _finite_vector(packet["causal_history_lower_body"], 240, "lower body"),
            _finite_vector(packet["vr_3point_local_target"], 9, "VR position"),
            _finite_vector(packet["vr_3point_local_orn_target"], 12, "VR orientation"),
            _rotation_6d(relative),
        )
    ).astype(np.float32)
    if result.shape != (267,) or not np.isfinite(result).all():
        raise ValueError("completed encoder input is not finite 267-D")
    return result


class CleanSonicPolicy:
    def __init__(
        self,
        encoder_path: Path,
        decoder_path: Path,
        *,
        expected_decoder_sha256: str = DECODER_SHA256,
    ):
        if sha256_file(encoder_path) != ENCODER_SHA256:
            raise ValueError("causal encoder hash changed")
        if (
            not isinstance(expected_decoder_sha256, str)
            or len(expected_decoder_sha256) != 64
            or sha256_file(decoder_path) != expected_decoder_sha256
        ):
            raise ValueError("clean student decoder hash changed")
        self.encoder = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
        self.encoder_input = self.encoder.get_inputs()[0].name
        self.decoder_input = self.decoder.get_inputs()[0].name

    def infer(self, encoder267: np.ndarray, policy930: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        token64 = np.asarray(
            self.encoder.run(None, {self.encoder_input: encoder267.reshape(1, 267)})[0],
            dtype=np.float32,
        ).reshape(64)
        decoder994 = np.concatenate((token64, policy930)).astype(np.float32)
        raw = np.asarray(
            self.decoder.run(None, {self.decoder_input: decoder994.reshape(1, 994)})[0],
            dtype=np.float32,
        ).reshape(23)
        if not np.isfinite(raw).all() or float(np.max(np.abs(raw))) >= STRICT_RAW_ABS_MAX:
            raise RuntimeError("student emitted nonfinite or clipped raw action")
        return raw, decoder994


class UnitreeZeroVelocityFallbackPolicy:
    """Hash-bound Unitree locomotion actor commanded to stand still."""

    def __init__(self, policy_path: Path):
        if sha256_file(policy_path) != VELOCITY_FALLBACK_SHA256:
            raise ValueError("Unitree velocity fallback policy hash changed")
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].name != "obs"
            or inputs[0].shape != [1, 98]
            or len(outputs) != 1
            or outputs[0].name != "actions"
            or outputs[0].shape != [1, 29]
        ):
            raise ValueError("Unitree velocity fallback ONNX ABI mismatch")
        metadata = self.session.get_modelmeta().custom_metadata_map
        joint_names = tuple(metadata.get("joint_names", "").split(","))
        if len(joint_names) != 29 or len(set(joint_names)) != 29:
            raise ValueError("Unitree velocity fallback joint metadata mismatch")
        self.hardware_indices = np.asarray(
            [joint_names.index(name) for name in HARDWARE_23_JOINT_NAMES], dtype=np.int64
        )

        def vector(name: str) -> np.ndarray:
            result = np.asarray(
                [float(item) for item in metadata.get(name, "").split(",")],
                dtype=np.float64,
            )
            if result.shape != (29,) or not np.isfinite(result).all():
                raise ValueError(f"Unitree velocity fallback {name} metadata mismatch")
            return result

        self.default_joint_pos = vector("default_joint_pos")
        self.action_scale = vector("action_scale")
        self.stiffness = vector("joint_stiffness")
        self.damping = vector("joint_damping")
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.query_count = 0

    def reset(self) -> None:
        self.previous_action.fill(0.0)
        self.query_count = 0

    def activate(self, joint_position_hardware: np.ndarray) -> None:
        q = np.asarray(joint_position_hardware, dtype=np.float64)
        if q.shape != (23,) or not np.isfinite(q).all():
            raise ValueError("fallback activation joint position invalid")
        q29 = self.default_joint_pos.copy()
        q29[self.hardware_indices] = q
        self.previous_action = ((q29 - self.default_joint_pos) / self.action_scale).astype(np.float32)

    def infer(
        self,
        *,
        joint_position_hardware: np.ndarray,
        joint_velocity_hardware: np.ndarray,
        base_angular_velocity_body: np.ndarray,
        projected_gravity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q29 = self.default_joint_pos.copy()
        dq29 = np.zeros(29, dtype=np.float64)
        q29[self.hardware_indices] = joint_position_hardware
        dq29[self.hardware_indices] = joint_velocity_hardware
        observation = np.concatenate(
            (
                base_angular_velocity_body,
                projected_gravity,
                np.zeros(3, dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                q29 - self.default_joint_pos,
                dq29,
                self.previous_action,
            )
        ).astype(np.float32)
        if observation.shape != (98,) or not np.isfinite(observation).all():
            raise RuntimeError("Unitree velocity fallback observation invalid")
        action = np.asarray(
            self.session.run(None, {"obs": observation.reshape(1, 98)})[0][0],
            dtype=np.float32,
        )
        if action.shape != (29,) or not np.isfinite(action).all():
            raise RuntimeError("Unitree velocity fallback action invalid")
        self.previous_action = action.copy()
        self.query_count += 1
        target29 = self.default_joint_pos + self.action_scale * action.astype(np.float64)
        return (
            target29[self.hardware_indices],
            self.stiffness[self.hardware_indices],
            self.damping[self.hardware_indices],
        )


class CleanTrue23MujocoController:
    """One exact 50-Hz clean SONIC actor driving local CPU MuJoCo."""

    def __init__(
        self,
        *,
        model_path: Path,
        physics_path: Path,
        policy: CleanSonicPolicy,
        minimum_base_height_m: float = MINIMUM_BASE_HEIGHT_M,
        maximum_base_tilt_rad: float = MAXIMUM_BASE_TILT_RAD,
    ):
        self.module, self.model, self.physics = prepare_true23_model(model_path, physics_path)
        if self.physics.decimation != 10 or not math.isclose(
            self.physics.timestep_s, 0.002, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("MuJoCo cadence differs from 50-Hz policy")
        self.data = self.module.MjData(self.model)
        self.reference_probe = self.module.MjData(self.model)
        self.policy = policy
        if (
            type(minimum_base_height_m) is not float
            or not 0.12 <= minimum_base_height_m <= MINIMUM_BASE_HEIGHT_M
            or type(maximum_base_tilt_rad) is not float
            or not MAXIMUM_BASE_TILT_RAD <= maximum_base_tilt_rad <= 2.2
        ):
            raise ValueError("native23 physical gate configuration invalid")
        self.minimum_base_height_m = minimum_base_height_m
        self.maximum_base_tilt_rad = maximum_base_tilt_rad
        self.previous_safe_native = np.zeros(23, dtype=np.float32)
        self.history: list[np.ndarray] = []
        self.buffered_robot_pelvis_q9 = np.asarray((1.0, 0.0, 0.0, 0.0))
        self.completed = 0
        # Keep failure reporting uniform when the optional supervisor is disabled.
        self.fallback_active = False
        self.fallback_trigger: str | None = None
        self.fallback_transition: int | None = None

    def use_released_retained_gains(self) -> None:
        """Select the exact gains used by the passing native23 physical replays."""

        np.copyto(self.physics.kp, RELEASED_RETAINED_KP)
        np.copyto(self.physics.kd, RELEASED_RETAINED_KD)

    def reference_root_height(self, joint_position_hardware: np.ndarray) -> float:
        """Return pelvis height that places the lower ankle body at 0.06 m."""

        q = np.asarray(joint_position_hardware, dtype=np.float64)
        if q.shape != (23,) or not np.isfinite(q).all():
            raise ValueError("reference joint position must be finite native23 hardware order")
        probe = self.module.MjData(self.model)
        probe.qpos[:3] = (0.0, 0.0, 0.0)
        probe.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        probe.qpos[7:] = q
        self.module.mj_forward(self.model, probe)
        ankles = (
            self.module.mj_name2id(self.model, self.module.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"),
            self.module.mj_name2id(self.model, self.module.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"),
        )
        if min(ankles) < 1:
            raise ValueError("native23 ankle body missing")
        height = 0.06 - min(float(probe.xpos[index, 2]) for index in ankles)
        if not 0.20 <= height <= 0.95:
            raise ValueError("reference root height outside native23 envelope")
        return height

    def retarget_pico_reference_packet(self, packet: Mapping[str, Any]) -> dict[str, Any]:
        """Replace raw XR pose targets with pinned native23 FK targets."""

        validate_reference_terms(packet)
        q9_native = _finite_vector(packet["q_ref23_native"], 23, "q9 native")
        q9_hardware = q9_native[NATIVE_TO_MJ]
        probe = self.reference_probe
        probe.qpos[:3] = (0.0, 0.0, self.reference_root_height(q9_hardware))
        probe.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        probe.qpos[7:] = q9_hardware
        probe.qvel[:] = 0.0
        self.module.mj_forward(self.model, probe)
        body_pos = np.asarray(probe.xpos[1:], dtype=np.float64)
        body_quat = np.asarray(probe.xquat[1:], dtype=np.float64)
        anchor_pos = body_pos[0]
        anchor_quat = body_quat[0]
        inverse = _quaternion_matrix(anchor_quat).T
        vr_pos: list[float] = []
        vr_quat: list[float] = []
        for body_index, offset in (
            (18, (0.18, -0.025, 0.0)),
            (23, (0.18, 0.025, 0.0)),
            (13, (0.0, 0.0, 0.35)),
        ):
            quaternion = body_quat[body_index]
            point = body_pos[body_index] + _quaternion_matrix(quaternion) @ np.asarray(offset)
            vr_pos.extend((inverse @ (point - anchor_pos)).tolist())
            vr_quat.extend(_quaternion_multiply(_quaternion_conjugate(anchor_quat), quaternion).tolist())
        result = copy.deepcopy(dict(packet))
        result["vr_3point_local_target"] = vr_pos
        result["vr_3point_local_orn_target"] = vr_quat
        result["reference_anchor_quaternion_xyzw"] = anchor_quat[[1, 2, 3, 0]].tolist()
        validate_reference_terms(result)
        return result

    def reset(
        self,
        *,
        base_position: np.ndarray,
        base_quaternion_wxyz: np.ndarray,
        joint_position_hardware: np.ndarray,
        root_velocity: np.ndarray | None = None,
        joint_velocity_hardware: np.ndarray | None = None,
        buffered_robot_pelvis_q9: np.ndarray | None = None,
    ) -> None:
        self.data.qpos[:3] = base_position
        self.data.qpos[3:7] = base_quaternion_wxyz
        self.data.qpos[7:] = joint_position_hardware
        self.data.qvel[:] = 0.0
        if root_velocity is not None:
            self.data.qvel[:6] = root_velocity
        if joint_velocity_hardware is not None:
            self.data.qvel[6:] = joint_velocity_hardware
        self.module.mj_forward(self.model, self.data)
        self.previous_safe_native.fill(0.0)
        self.history = []
        self.buffered_robot_pelvis_q9 = (
            np.asarray(base_quaternion_wxyz, dtype=np.float64).copy()
            if buffered_robot_pelvis_q9 is None
            else np.asarray(buffered_robot_pelvis_q9, dtype=np.float64).copy()
        )
        self.completed = 0
        self.fallback_active = False
        self.fallback_trigger = None
        self.fallback_transition = None

    def _policy_frame(self) -> np.ndarray:
        q_hardware = np.asarray(self.data.qpos[7:], dtype=np.float64)
        dq_hardware = np.asarray(self.data.qvel[6:], dtype=np.float64)
        q_native = q_hardware[MJ_TO_NATIVE]
        dq_native = dq_hardware[MJ_TO_NATIVE]
        q29 = np.zeros(29, dtype=np.float64)
        dq29 = np.zeros(29, dtype=np.float64)
        action29 = np.zeros(29, dtype=np.float64)
        q29[NATIVE_TO_IL29] = q_native - DEFAULT_NATIVE
        dq29[NATIVE_TO_IL29] = dq_native
        action29[NATIVE_TO_IL29] = self.previous_safe_native
        _, angular_world = _body_velocity(self.module, self.model, self.data, "pelvis", (0.0, 0.0, 0.0))
        angular = world_angular_velocity_to_body(self.data.qpos[3:7], angular_world)
        gravity = _projected_gravity(self.data.qpos[3:7])
        result = np.concatenate((angular, q29, dq29, action29, gravity)).astype(np.float32)
        if result.shape != (93,) or not np.isfinite(result).all():
            raise RuntimeError("measured policy frame is invalid")
        return result

    def step(self, encoder267: np.ndarray) -> dict[str, float]:
        current_pelvis = np.asarray(self.data.qpos[3:7], dtype=np.float64).copy()
        frame = self._policy_frame()
        self.history = [frame.copy() for _ in range(10)] if not self.history else [*self.history[1:], frame]
        raw, _ = self.policy.infer(encoder267, term_major_history(self.history))
        safe_native, target_hardware = safe_target_transform_numpy(raw)
        maximum_torque = 0.0
        for _ in range(self.physics.decimation):
            q = np.asarray(self.data.qpos[7:], dtype=np.float64)
            dq = np.asarray(self.data.qvel[6:], dtype=np.float64)
            torque = np.clip(
                self.physics.kp * (target_hardware.astype(np.float64) - q) - self.physics.kd * dq,
                -self.physics.effort,
                self.physics.effort,
            )
            maximum_torque = max(maximum_torque, float(np.max(np.abs(torque))))
            self.data.ctrl[:] = torque
            self.module.mj_step(self.model, self.data)
        self.buffered_robot_pelvis_q9 = current_pelvis
        self.previous_safe_native = safe_native.astype(np.float32, copy=True)
        self.completed += 1
        gravity = _projected_gravity(self.data.qpos[3:7])
        tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
        height = float(self.data.qpos[2])
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("MuJoCo state became nonfinite")
        if height < self.minimum_base_height_m or tilt > self.maximum_base_tilt_rad:
            raise RuntimeError(f"physical gate failed: height={height:.6f}, tilt={tilt:.6f}")
        return {
            "base_height_m": height,
            "base_tilt_rad": tilt,
            "max_abs_raw_action": float(np.max(np.abs(raw))),
            "max_abs_torque_nm": maximum_torque,
        }


class SupervisedCleanTrue23MujocoController(CleanTrue23MujocoController):
    """SONIC teleop with an automatic, latched zero-velocity balance fallback."""

    def __init__(
        self,
        *,
        model_path: Path,
        physics_path: Path,
        policy: CleanSonicPolicy,
        fallback_policy: UnitreeZeroVelocityFallbackPolicy,
    ) -> None:
        super().__init__(model_path=model_path, physics_path=physics_path, policy=policy)
        self.fallback_policy = fallback_policy
        self.fallback_active = False
        self.fallback_trigger: str | None = None
        self.fallback_transition: int | None = None
        self._previous_linear_velocity_world = np.zeros(3, dtype=np.float64)
        self._previous_angular_velocity_world = np.zeros(3, dtype=np.float64)

    def reset(self, **kwargs: Any) -> None:
        super().reset(**kwargs)
        linear, angular = _body_velocity(self.module, self.model, self.data, "pelvis", (0.0, 0.0, 0.0))
        self._previous_linear_velocity_world = linear
        self._previous_angular_velocity_world = angular
        self.fallback_policy.reset()
        self.fallback_active = False
        self.fallback_trigger = None
        self.fallback_transition = None

    def _activate_if_needed(
        self,
        linear_world: np.ndarray,
        angular_world: np.ndarray,
        tilt: float,
    ) -> None:
        if self.fallback_active:
            return
        linear_jump = float(np.linalg.norm((linear_world - self._previous_linear_velocity_world)[:2]))
        vertical_jump = float(abs(linear_world[2] - self._previous_linear_velocity_world[2]))
        angular_jump = float(np.linalg.norm(angular_world - self._previous_angular_velocity_world))
        if linear_jump >= LINEAR_VELOCITY_JUMP_TRIGGER_MPS:
            trigger = "linear_velocity_jump"
        elif vertical_jump >= VERTICAL_VELOCITY_JUMP_TRIGGER_MPS:
            trigger = "vertical_velocity_jump"
        elif angular_jump >= ANGULAR_VELOCITY_JUMP_TRIGGER_RADPS:
            trigger = "angular_velocity_jump"
        elif tilt >= FALLBACK_TILT_TRIGGER_RAD:
            trigger = "base_tilt"
        else:
            return
        self.fallback_policy.activate(np.asarray(self.data.qpos[7:], dtype=np.float64))
        self.fallback_active = True
        self.fallback_trigger = trigger
        self.fallback_transition = self.completed

    def step(self, encoder267: np.ndarray) -> dict[str, float | str | bool | None]:
        current_pelvis = np.asarray(self.data.qpos[3:7], dtype=np.float64).copy()
        linear_world, angular_world = _body_velocity(self.module, self.model, self.data, "pelvis", (0.0, 0.0, 0.0))
        gravity = _projected_gravity(self.data.qpos[3:7])
        tilt_before = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
        self._activate_if_needed(linear_world, angular_world, tilt_before)
        if not self.fallback_active:
            evidence = dict(super().step(encoder267))
            mode = "sonic_teleop"
        else:
            angular_body = world_angular_velocity_to_body(self.data.qpos[3:7], angular_world)
            target, kp, kd = self.fallback_policy.infer(
                joint_position_hardware=np.asarray(self.data.qpos[7:], dtype=np.float64),
                joint_velocity_hardware=np.asarray(self.data.qvel[6:], dtype=np.float64),
                base_angular_velocity_body=angular_body,
                projected_gravity=gravity,
            )
            maximum_torque = 0.0
            for _ in range(self.physics.decimation):
                q = np.asarray(self.data.qpos[7:], dtype=np.float64)
                dq = np.asarray(self.data.qvel[6:], dtype=np.float64)
                torque = np.clip(
                    kp * (target - q) - kd * dq,
                    -self.physics.effort,
                    self.physics.effort,
                )
                maximum_torque = max(maximum_torque, float(np.max(np.abs(torque))))
                self.data.ctrl[:] = torque
                self.module.mj_step(self.model, self.data)
            self.buffered_robot_pelvis_q9 = current_pelvis
            self.completed += 1
            gravity = _projected_gravity(self.data.qpos[3:7])
            tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
            height = float(self.data.qpos[2])
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("fallback MuJoCo state became nonfinite")
            if height < MINIMUM_BASE_HEIGHT_M or tilt > MAXIMUM_BASE_TILT_RAD:
                raise RuntimeError(f"fallback physical gate failed: height={height:.6f}, tilt={tilt:.6f}")
            evidence = {
                "base_height_m": height,
                "base_tilt_rad": tilt,
                "max_abs_raw_action": 0.0,
                "max_abs_torque_nm": maximum_torque,
            }
            mode = "unitree_zero_velocity_fallback"
        linear_after, angular_after = _body_velocity(self.module, self.model, self.data, "pelvis", (0.0, 0.0, 0.0))
        self._previous_linear_velocity_world = linear_after
        self._previous_angular_velocity_world = angular_after
        evidence.update(
            {
                "controller_mode": mode,
                "fallback_active": self.fallback_active,
                "fallback_trigger": self.fallback_trigger,
            }
        )
        return evidence


class BalancedUpperBodyTrue23MujocoController:
    """Zero-velocity balance with a bounded eight-arm PICO reference blend."""

    def __init__(
        self,
        *,
        model_path: Path,
        physics_path: Path,
        balance_policy: UnitreeZeroVelocityFallbackPolicy,
        arm_blend: float = BALANCED_UPPER_BODY_ARM_BLEND,
    ) -> None:
        if not math.isclose(arm_blend, BALANCED_UPPER_BODY_ARM_BLEND, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("balanced upper-body arm blend is not the reviewed value")
        self.module, self.model, self.physics = prepare_true23_model(model_path, physics_path)
        if self.physics.decimation != 10 or not math.isclose(
            self.physics.timestep_s, 0.002, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("MuJoCo cadence differs from 50-Hz policy")
        self.data = self.module.MjData(self.model)
        self.balance_policy = balance_policy
        self.arm_blend = arm_blend
        self.completed = 0

    def reset(self) -> None:
        q0 = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float64)
        self.data.qpos[:3] = (0.0, 0.0, 0.76)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[7:] = q0
        self.data.qvel[:] = 0.0
        self.module.mj_forward(self.model, self.data)
        self.balance_policy.reset()
        self.balance_policy.activate(q0)
        self.completed = 0

    def step_reference(self, packet: Mapping[str, Any]) -> dict[str, float]:
        q_reference, _ = reference_initial_state(packet)
        _, angular_world = _body_velocity(self.module, self.model, self.data, "pelvis", (0.0, 0.0, 0.0))
        target_hardware, kp, kd = self.balance_policy.infer(
            joint_position_hardware=np.asarray(self.data.qpos[7:], dtype=np.float64),
            joint_velocity_hardware=np.asarray(self.data.qvel[6:], dtype=np.float64),
            base_angular_velocity_body=world_angular_velocity_to_body(self.data.qpos[3:7], angular_world),
            projected_gravity=_projected_gravity(self.data.qpos[3:7]),
        )
        target_hardware = np.asarray(target_hardware, dtype=np.float64).copy()
        arm_slice = slice(BALANCED_UPPER_BODY_FIRST_HARDWARE_INDEX, 23)
        target_hardware[arm_slice] += self.arm_blend * (q_reference[arm_slice] - target_hardware[arm_slice])
        maximum_torque = 0.0
        for _ in range(self.physics.decimation):
            q = np.asarray(self.data.qpos[7:], dtype=np.float64)
            dq = np.asarray(self.data.qvel[6:], dtype=np.float64)
            torque = np.clip(
                kp * (target_hardware - q) - kd * dq,
                -self.physics.effort,
                self.physics.effort,
            )
            maximum_torque = max(maximum_torque, float(np.max(np.abs(torque))))
            self.data.ctrl[:] = torque
            self.module.mj_step(self.model, self.data)
        self.completed += 1
        gravity = _projected_gravity(self.data.qpos[3:7])
        tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
        height = float(self.data.qpos[2])
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("balanced upper-body MuJoCo state became nonfinite")
        if height < MINIMUM_BASE_HEIGHT_M or tilt > FALLBACK_TILT_TRIGGER_RAD:
            raise RuntimeError(f"balanced upper-body physical gate failed: height={height:.6f}, tilt={tilt:.6f}")
        arm_error = np.asarray(self.data.qpos[7:], dtype=np.float64)[arm_slice] - q_reference[arm_slice]
        return {
            "base_height_m": height,
            "base_tilt_rad": tilt,
            "arm_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(arm_error)))),
            "max_abs_torque_nm": maximum_torque,
        }


def motion_reference_terms(motion: Mapping[str, np.ndarray], q9: int) -> dict[str, Any]:
    indices = np.arange(q9 - 9, q9 + 1)
    proof = indices + 1
    positions = motion["joint_pos"][indices, :12]
    velocities = (motion["joint_pos"][proof, :12] - positions) / np.float32(0.02)
    anchor_pos = motion["body_pos_w"][q9, 0]
    anchor_quat = motion["body_quat_w"][q9, 0]
    inverse = _quaternion_matrix(anchor_quat).T
    vr_pos: list[float] = []
    vr_quat: list[float] = []
    for body_index, offset in (
        (18, (0.18, -0.025, 0.0)),
        (23, (0.18, 0.025, 0.0)),
        (13, (0.0, 0.0, 0.35)),
    ):
        position = motion["body_pos_w"][q9, body_index]
        quaternion = motion["body_quat_w"][q9, body_index]
        point = position + _quaternion_matrix(quaternion) @ np.asarray(offset)
        vr_pos.extend((inverse @ (point - anchor_pos)).tolist())
        vr_quat.extend(_quaternion_multiply(_quaternion_conjugate(anchor_quat), quaternion).tolist())
    q9_hardware = motion["joint_pos"][q9].astype(np.float64)
    q10_hardware = motion["joint_pos"][q9 + 1].astype(np.float64)
    q9_native = q9_hardware[MJ_TO_NATIVE]
    q10_native = q10_hardware[MJ_TO_NATIVE]
    anchor_il29 = np.zeros(29, dtype=np.float64)
    proof_il29 = np.zeros(29, dtype=np.float64)
    anchor_il29[NATIVE_TO_IL29] = q9_native
    proof_il29[NATIVE_TO_IL29] = q10_native
    base_ns = 1_000_000_000 + q9 * CONTROL_PERIOD_NS
    return {
        "schema_version": CAUSAL_HISTORY_ENCODER_TERMS_SCHEMA_VERSION,
        "kind": CAUSAL_HISTORY_REFERENCE_TERMS_KIND,
        "reference_profile": CAUSAL_HISTORY_PROFILE,
        "reference_contract_sha256": causal_history_stream_contract()["contract_sha256"],
        "pico_anchor_source_frame_index": q9,
        "pico_anchor_monotonic_ns": base_ns,
        "control_source_frame_index": q9 + 1,
        "control_monotonic_ns": base_ns + CONTROL_PERIOD_NS,
        "causal_history_lower_body": np.concatenate((positions.reshape(-1), velocities.reshape(-1))).tolist(),
        "vr_3point_local_target": vr_pos,
        "vr_3point_local_orn_target": vr_quat,
        "reference_anchor_quaternion_xyzw": anchor_quat[[1, 2, 3, 0]].tolist(),
        "anchor_joint_pos_il29": anchor_il29.tolist(),
        "proof_joint_pos_il29": proof_il29.tolist(),
        "q_ref23_native": q9_native.tolist(),
        "qd_ref23_native": ((q10_native - q9_native) * 50.0).tolist(),
        "control_derivative_contract": CAUSAL_HISTORY_CONTROL_DERIVATIVE_CONTRACT,
        "sdk_derivatives_consumed": False,
    }


def run_motion_replay(
    *,
    repository_root: Path,
    steps: int = 510,
    viewer: bool = False,
) -> dict[str, Any]:
    root = repository_root.resolve()
    motion_path = root / (
        "artifacts/g1_native124_multimotion/scaling_all61/retimed_v2/npz/J_Dance0_StepTouch_slow2.npz"
    )
    if sha256_file(motion_path) != MOTION_SHA256:
        raise ValueError("DadDance motion hash changed")
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    policy = CleanSonicPolicy(
        root / "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx",
        root / "artifacts/g1_true23/steptouch_balanced_teacher_lowrank_preserve_alpha010_v1.decoder.onnx",
    )
    controller = CleanTrue23MujocoController(
        model_path=root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        physics_path=root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        policy=policy,
    )
    q10 = 10
    root_velocity = np.concatenate(
        (
            motion["body_lin_vel_w"][q10, 0],
            _quaternion_matrix(motion["body_quat_w"][q10, 0]).T @ motion["body_ang_vel_w"][q10, 0],
        )
    )
    reference_q9 = motion["body_quat_w"][9, 0].astype(np.float64)
    reference_q10 = motion["body_quat_w"][10, 0].astype(np.float64)
    current_q10 = reference_q10.copy()
    delta = _quaternion_multiply(current_q10, _quaternion_conjugate(reference_q10))
    w, x, y, z = delta / np.linalg.norm(delta)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    alignment = np.asarray((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))
    virtual_q9 = _quaternion_multiply(alignment, reference_q9)
    controller.reset(
        base_position=motion["body_pos_w"][q10, 0],
        base_quaternion_wxyz=current_q10,
        joint_position_hardware=motion["joint_pos"][q10],
        root_velocity=root_velocity,
        joint_velocity_hardware=motion["joint_vel"][q10],
        buffered_robot_pelvis_q9=virtual_q9,
    )
    min_height = float("inf")
    max_tilt = 0.0
    max_raw = 0.0
    max_torque = 0.0
    passive = None
    if viewer:
        import mujoco.viewer

        passive = mujoco.viewer.launch_passive(controller.model, controller.data)
    try:
        next_step = time.monotonic()
        for transition in range(steps):
            packet = motion_reference_terms(motion, 9 + transition)
            encoder267 = encoder267_from_reference(packet, controller.buffered_robot_pelvis_q9)
            evidence = controller.step(encoder267)
            min_height = min(min_height, evidence["base_height_m"])
            max_tilt = max(max_tilt, evidence["base_tilt_rad"])
            max_raw = max(max_raw, evidence["max_abs_raw_action"])
            max_torque = max(max_torque, evidence["max_abs_torque_nm"])
            if passive is not None:
                passive.sync()
                next_step += 0.02
                time.sleep(max(0.0, next_step - time.monotonic()))
    finally:
        if passive is not None:
            passive.close()
    return {
        "schema_version": 1,
        "kind": "g1_true23_clean_mujoco_teleop_replay",
        "mode": "motion_replay",
        "completed_transitions": controller.completed,
        "first_q9": 9,
        "last_q9": 8 + controller.completed,
        "minimum_base_height_m": min_height,
        "maximum_base_tilt_rad": max_tilt,
        "maximum_absolute_raw_action": max_raw,
        "maximum_absolute_torque_nm": max_torque,
        "passed": controller.completed == steps,
        "authorization": {
            "simulator_only": True,
            "dds_opened": False,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }


SUPERVISOR_SCREEN_CASES = (
    ("nominal", None, 0.0, None),
    ("linear_x_positive_early", 0, 0.5, 50),
    ("linear_x_negative_mid", 0, -0.5, 200),
    ("linear_y_positive_late", 1, 0.5, 350),
    ("linear_y_negative_early", 1, -0.5, 50),
    ("linear_z_positive_mid", 2, 0.2, 200),
    ("linear_z_negative_late", 2, -0.2, 350),
    ("angular_x_positive_early", 3, 0.52, 50),
    ("angular_x_negative_mid", 3, -0.52, 200),
    ("angular_y_positive_late", 4, 0.52, 350),
    ("angular_y_negative_early", 4, -0.52, 50),
    ("angular_z_positive_mid", 5, 0.78, 200),
    ("angular_z_negative_late", 5, -0.78, 350),
)


def run_supervisor_disturbance_qualification(*, repository_root: Path, steps: int = 510) -> dict[str, Any]:
    """Run the fixed nominal plus six-axis teleop fallback matrix."""

    if steps != 510:
        raise ValueError("supervisor qualification requires exactly 510 transitions")
    root = repository_root.resolve()
    motion_path = root / (
        "artifacts/g1_native124_multimotion/scaling_all61/retimed_v2/npz/J_Dance0_StepTouch_slow2.npz"
    )
    if sha256_file(motion_path) != MOTION_SHA256:
        raise ValueError("StepTouch supervisor motion hash changed")
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    encoder_path = root / "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx"
    decoder_path = (
        root / "artifacts/g1_true23/steptouch_balanced_teacher_lowrank_preserve_alpha010_v1.decoder.onnx"
    )
    fallback_path = root / (
        "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
    )
    controller = SupervisedCleanTrue23MujocoController(
        model_path=root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        physics_path=root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        policy=CleanSonicPolicy(encoder_path, decoder_path),
        fallback_policy=UnitreeZeroVelocityFallbackPolicy(fallback_path),
    )
    q10 = 10
    root_velocity = np.concatenate(
        (
            motion["body_lin_vel_w"][q10, 0],
            _quaternion_matrix(motion["body_quat_w"][q10, 0]).T @ motion["body_ang_vel_w"][q10, 0],
        )
    )
    records: list[dict[str, Any]] = []
    for name, qvel_axis, impulse, apply_transition in SUPERVISOR_SCREEN_CASES:
        controller.reset(
            base_position=motion["body_pos_w"][q10, 0],
            base_quaternion_wxyz=motion["body_quat_w"][q10, 0],
            joint_position_hardware=motion["joint_pos"][q10],
            root_velocity=root_velocity,
            joint_velocity_hardware=motion["joint_vel"][q10],
            buffered_robot_pelvis_q9=motion["body_quat_w"][9, 0],
        )
        minimum_height = float("inf")
        maximum_tilt = 0.0
        maximum_torque = 0.0
        failure: dict[str, Any] | None = None
        try:
            for transition in range(steps):
                if apply_transition == transition:
                    if qvel_axis is None:
                        raise RuntimeError("disturbance case lacks root qvel axis")
                    controller.data.qvel[qvel_axis] += impulse
                    controller.module.mj_forward(controller.model, controller.data)
                packet = motion_reference_terms(motion, 9 + transition)
                evidence = controller.step(encoder267_from_reference(packet, controller.buffered_robot_pelvis_q9))
                minimum_height = min(minimum_height, float(evidence["base_height_m"]))
                maximum_tilt = max(maximum_tilt, float(evidence["base_tilt_rad"]))
                maximum_torque = max(maximum_torque, float(evidence["max_abs_torque_nm"]))
        except Exception as error:
            failure = {
                "transition": controller.completed,
                "type": type(error).__name__,
                "message": str(error),
            }
        expected_trigger = None
        if qvel_axis in (0, 1):
            expected_trigger = "linear_velocity_jump"
        elif qvel_axis == 2:
            expected_trigger = "vertical_velocity_jump"
        elif qvel_axis in (3, 4, 5):
            expected_trigger = "angular_velocity_jump"
        passed = bool(
            failure is None
            and controller.completed == steps
            and minimum_height >= MINIMUM_BASE_HEIGHT_M
            and maximum_tilt <= MAXIMUM_BASE_TILT_RAD
            and (
                (
                    apply_transition is None
                    and controller.fallback_active is False
                    and controller.fallback_policy.query_count == 0
                )
                or (
                    apply_transition is not None
                    and controller.fallback_active is True
                    and controller.fallback_transition == apply_transition
                    and controller.fallback_trigger == expected_trigger
                    and controller.fallback_policy.query_count == steps - apply_transition
                )
            )
        )
        records.append(
            {
                "name": name,
                "root_qvel_axis": qvel_axis,
                "impulse": impulse,
                "apply_transition": apply_transition,
                "completed_transitions": controller.completed,
                "fallback_active": controller.fallback_active,
                "fallback_trigger": controller.fallback_trigger,
                "fallback_first_transition": controller.fallback_transition,
                "fallback_query_count": controller.fallback_policy.query_count,
                "minimum_base_height_m": minimum_height,
                "maximum_base_tilt_rad": maximum_tilt,
                "maximum_absolute_torque_nm": maximum_torque,
                "failure": failure,
                "passed": passed,
            }
        )
    return {
        "schema_version": 1,
        "kind": "g1_true23_supervised_teleop_disturbance_qualification_v1",
        "case_count": len(records),
        "passed_case_count": sum(record["passed"] is True for record in records),
        "motion_sha256": MOTION_SHA256,
        "encoder_sha256": ENCODER_SHA256,
        "sonic_decoder_sha256": DECODER_SHA256,
        "velocity_fallback_sha256": VELOCITY_FALLBACK_SHA256,
        "triggers": {
            "linear_velocity_jump_mps": LINEAR_VELOCITY_JUMP_TRIGGER_MPS,
            "vertical_velocity_jump_mps": VERTICAL_VELOCITY_JUMP_TRIGGER_MPS,
            "angular_velocity_jump_radps": ANGULAR_VELOCITY_JUMP_TRIGGER_RADPS,
            "base_tilt_rad": FALLBACK_TILT_TRIGGER_RAD,
            "fallback_latched_until_reset": True,
        },
        "records": records,
        "passed": all(record["passed"] is True for record in records),
        "authorization": {
            "simulator_only": True,
            "dds_opened": False,
            "network_actuation_used": False,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }


def run_reference_sequence(
    *,
    controller: CleanTrue23MujocoController,
    packets: Iterable[Mapping[str, Any]],
    steps: int,
    maximum_age_ns: int | None = None,
    step_callback: Callable[[], None] | None = None,
    retarget_native23_fk: bool = False,
) -> dict[str, Any]:
    previous: dict[str, Any] | None = None
    min_height = float("inf")
    max_tilt = 0.0
    max_age = 0
    fallback_transition_count = 0
    for packet in packets:
        summary = validate_reference_terms(packet)
        if previous is not None and (
            summary["control_index"] != previous["control_index"] + 1
            or summary["control_monotonic_ns"] != previous["control_monotonic_ns"] + CONTROL_PERIOD_NS
        ):
            raise RuntimeError("PICO causal transport lost contiguous 50-Hz frame")
        if maximum_age_ns is not None:
            age = time.monotonic_ns() - summary["control_monotonic_ns"]
            if age < 0 or age > maximum_age_ns:
                raise RuntimeError(f"PICO causal transport stale: {age} ns")
            max_age = max(max_age, age)
        policy_packet = controller.retarget_pico_reference_packet(packet) if retarget_native23_fk else packet
        encoder267 = encoder267_from_reference(policy_packet, controller.buffered_robot_pelvis_q9)
        evidence = controller.step(encoder267)
        fallback_transition_count += int(evidence.get("controller_mode") == "unitree_zero_velocity_fallback")
        if step_callback is not None:
            step_callback()
        min_height = min(min_height, evidence["base_height_m"])
        max_tilt = max(max_tilt, evidence["base_tilt_rad"])
        previous = summary
        if controller.completed == steps:
            break
    if controller.completed != steps:
        raise RuntimeError(f"reference stream ended at {controller.completed}/{steps}")
    return {
        "schema_version": 1,
        "kind": "g1_true23_clean_mujoco_teleop_session",
        "mode": "pico_causal_zmq",
        "completed_transitions": controller.completed,
        "first_control_source_frame_index": None if previous is None else previous["control_index"] - steps + 1,
        "last_control_source_frame_index": None if previous is None else previous["control_index"],
        "minimum_base_height_m": min_height,
        "maximum_base_tilt_rad": max_tilt,
        "maximum_reference_age_ns": max_age,
        "fallback_transition_count": fallback_transition_count,
        "fallback_active": bool(getattr(controller, "fallback_active", False)),
        "fallback_trigger": getattr(controller, "fallback_trigger", None),
        "fallback_first_transition": getattr(controller, "fallback_transition", None),
        "pico_reference_retargeting": (
            "pinned_native23_forward_kinematics" if retarget_native23_fk else "raw_pico_targets"
        ),
        "passed": True,
        "authorization": {
            "simulator_only": True,
            "dds_opened": False,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }


def run_balanced_upper_body_reference_sequence(
    *,
    controller: BalancedUpperBodyTrue23MujocoController,
    packets: Iterable[Mapping[str, Any]],
    steps: int,
    maximum_age_ns: int | None = None,
    step_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    previous: dict[str, Any] | None = None
    min_height = float("inf")
    max_tilt = 0.0
    max_arm_rmse = 0.0
    max_torque = 0.0
    max_age = 0
    for packet in packets:
        summary = validate_reference_terms(packet)
        if previous is not None and (
            summary["control_index"] != previous["control_index"] + 1
            or summary["control_monotonic_ns"] != previous["control_monotonic_ns"] + CONTROL_PERIOD_NS
        ):
            raise RuntimeError("PICO causal transport lost contiguous 50-Hz frame")
        if maximum_age_ns is not None:
            age = time.monotonic_ns() - summary["control_monotonic_ns"]
            if age < 0 or age > maximum_age_ns:
                raise RuntimeError(f"PICO causal transport stale: {age} ns")
            max_age = max(max_age, age)
        evidence = controller.step_reference(packet)
        if step_callback is not None:
            step_callback()
        min_height = min(min_height, evidence["base_height_m"])
        max_tilt = max(max_tilt, evidence["base_tilt_rad"])
        max_arm_rmse = max(max_arm_rmse, evidence["arm_tracking_rmse_rad"])
        max_torque = max(max_torque, evidence["max_abs_torque_nm"])
        previous = summary
        if controller.completed == steps:
            break
    if controller.completed != steps:
        raise RuntimeError(f"reference stream ended at {controller.completed}/{steps}")
    return {
        "schema_version": 1,
        "kind": "g1_true23_balanced_upper_body_mujoco_teleop_session",
        "mode": "pico_saved_balanced_upper_body_replay",
        "completed_transitions": controller.completed,
        "first_control_source_frame_index": (None if previous is None else previous["control_index"] - steps + 1),
        "last_control_source_frame_index": None if previous is None else previous["control_index"],
        "minimum_base_height_m": min_height,
        "maximum_base_tilt_rad": max_tilt,
        "maximum_arm_tracking_rmse_rad": max_arm_rmse,
        "maximum_absolute_torque_nm": max_torque,
        "maximum_reference_age_ns": max_age,
        "arm_reference_blend": controller.arm_blend,
        "balance_owned_hardware_joint_count": BALANCED_UPPER_BODY_FIRST_HARDWARE_INDEX,
        "pico_arm_hardware_joint_count": 23 - BALANCED_UPPER_BODY_FIRST_HARDWARE_INDEX,
        "full_sonic_policy_used": False,
        "passed": True,
        "authorization": {
            "simulator_only": True,
            "dds_opened": False,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }


def reference_initial_state(packet: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    validate_reference_terms(packet)
    q9_native = _finite_vector(packet["q_ref23_native"], 23, "q9 native")
    qd_native = _finite_vector(packet["qd_ref23_native"], 23, "qd native")
    q10_native = q9_native + qd_native * 0.02
    return q10_native[NATIVE_TO_MJ], qd_native[NATIVE_TO_MJ]


__all__ = [
    "BalancedUpperBodyTrue23MujocoController",
    "CleanSonicPolicy",
    "CleanTrue23MujocoController",
    "SupervisedCleanTrue23MujocoController",
    "UnitreeZeroVelocityFallbackPolicy",
    "DECODER_SHA256",
    "ENCODER_SHA256",
    "VELOCITY_FALLBACK_SHA256",
    "encoder267_from_reference",
    "pico_reference_policy_history",
    "motion_reference_terms",
    "reference_initial_state",
    "run_motion_replay",
    "run_balanced_upper_body_reference_sequence",
    "run_reference_sequence",
    "run_supervisor_disturbance_qualification",
    "validate_reference_terms",
]
