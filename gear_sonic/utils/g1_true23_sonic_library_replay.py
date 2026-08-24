"""Genuine true23 causal SONIC replay for retargeted library motions.

This is simulator-only.  Crawl motions use reference-relative physical gates;
the upright teleoperation controller and its stricter height gate stay unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnxruntime as ort

from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_TO_CANONICAL_IL29,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    ENCODER_SHA256,
    STRICT_RAW_ABS_MAX,
    CleanTrue23MujocoController,
    encoder267_from_reference,
    motion_reference_terms,
    sha256_file,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import (
    _projected_gravity,
    _quaternion_matrix,
    term_major_history,
    world_angular_velocity_to_body,
)

FPS = 50.0
TRACKED_BODY_INDICES = (0, 6, 12, 18, 23)
TRACKED_BODY_NAMES = ("pelvis", "left_ankle", "right_ankle", "left_wrist", "right_wrist")
MINIMUM_ABSOLUTE_HEIGHT_M = 0.12
MAXIMUM_ABSOLUTE_TILT_RAD = 2.20
MAXIMUM_PELVIS_ORIENTATION_ERROR_RAD = 1.50
MAXIMUM_RELATIVE_TRACKED_BODY_POSITION_ERROR_M = 1.00
MAXIMUM_JOINT_TRACKING_RMSE_RAD = 0.75
ORIGINAL_TRUE23_DECODER_SHA256 = "f18139aa5b98619a5d0e84a9de8378a5081c556d427b20fdc55e4fe917549740"
NATIVE_TO_IL29 = np.asarray(NATIVE_IL23_TO_CANONICAL_IL29, dtype=np.int64)
MJ_TO_NATIVE = np.asarray(MUJOCO_TO_ISAACLAB_DOF, dtype=np.int64)
DEFAULT_NATIVE = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float64)[MJ_TO_NATIVE]
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


def validate_library_motion(motion: Mapping[str, np.ndarray]) -> int:
    required = {
        "fps": (1,),
        "joint_pos": (None, 23),
        "joint_vel": (None, 23),
        "body_pos_w": (None, 24, 3),
        "body_quat_w": (None, 24, 4),
        "body_lin_vel_w": (None, 24, 3),
        "body_ang_vel_w": (None, 24, 3),
    }
    if set(motion) != set(required):
        raise ValueError("true23 library motion keys mismatch")
    frame_count = int(np.asarray(motion["joint_pos"]).shape[0])
    if frame_count < 12:
        raise ValueError("true23 library motion is too short")
    for name, shape in required.items():
        value = np.asarray(motion[name])
        expected = tuple(frame_count if item is None else item for item in shape)
        if value.shape != expected:
            raise ValueError(f"{name} shape mismatch: {value.shape} != {expected}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if not math.isclose(float(np.asarray(motion["fps"])[0]), FPS, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("true23 library motion must be exactly 50 Hz")
    norms = np.linalg.norm(np.asarray(motion["body_quat_w"], dtype=np.float64), axis=2)
    if float(np.max(np.abs(norms - 1.0))) > 2.0e-5:
        raise ValueError("true23 library motion has non-unit body quaternion")
    return frame_count


def _quaternion_error_rad(left_wxyz: np.ndarray, right_wxyz: np.ndarray) -> float:
    left = np.asarray(left_wxyz, dtype=np.float64)
    right = np.asarray(right_wxyz, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))))


class ExactHashSonicPolicy:
    """Same true23 ABI as CleanSonicPolicy, with an explicit decoder identity."""

    def __init__(
        self,
        encoder_path: Path,
        decoder_path: Path,
        *,
        expected_decoder_sha256: str,
    ) -> None:
        if sha256_file(encoder_path) != ENCODER_SHA256:
            raise ValueError("causal encoder hash changed")
        if sha256_file(decoder_path) != expected_decoder_sha256:
            raise ValueError("requested true23 decoder hash changed")
        self.encoder = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
        encoder_inputs = self.encoder.get_inputs()
        encoder_outputs = self.encoder.get_outputs()
        decoder_inputs = self.decoder.get_inputs()
        decoder_outputs = self.decoder.get_outputs()
        if (
            len(encoder_inputs) != 1
            or encoder_inputs[0].shape != [1, 267]
            or len(encoder_outputs) != 1
            or encoder_outputs[0].shape != [1, 64]
            or len(decoder_inputs) != 1
            or decoder_inputs[0].shape != [1, 994]
            or len(decoder_outputs) != 1
            or decoder_outputs[0].shape != [1, 23]
        ):
            raise ValueError("requested true23 SONIC ONNX ABI mismatch")
        self.encoder_input = encoder_inputs[0].name
        self.decoder_input = decoder_inputs[0].name

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


class ReferenceActionPolicy:
    """Physical-feasibility witness; never classified as SONIC inference."""

    def __init__(self) -> None:
        self.raw = np.zeros(23, dtype=np.float32)
        self.raw_clip_coordinate_count = 0
        self.target_projection_max_abs_rad = 0.0

    def set_target(self, target_hardware: np.ndarray, *, project_to_raw_clip: bool = False) -> None:
        target = np.asarray(target_hardware, dtype=np.float64)
        default = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float64)
        scale = np.asarray(HARDWARE_23_ACTION_SCALE, dtype=np.float64)
        positive = np.asarray(SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE, dtype=np.float64)
        negative = np.asarray(SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE, dtype=np.float64)
        safe_delta = target - default
        capacity = np.where(safe_delta >= 0.0, positive, negative)
        ratio = safe_delta / capacity
        if np.any(np.abs(ratio) >= 1.0) and not project_to_raw_clip:
            raise ValueError("reference target is outside safe transform domain")
        if project_to_raw_clip:
            ratio = np.clip(ratio, -1.0 + 1.0e-7, 1.0 - 1.0e-7)
        raw_hardware = capacity * np.arctanh(ratio) / scale
        self.raw = raw_hardware[MJ_TO_NATIVE].astype(np.float32)
        raw_clip_mask = np.abs(self.raw) >= np.float32(10.0)
        if np.any(raw_clip_mask) and not project_to_raw_clip:
            raise ValueError("reference target requires true23 raw clipping")
        if project_to_raw_clip:
            self.raw_clip_coordinate_count = int(np.count_nonzero(raw_clip_mask))
            self.raw = np.clip(self.raw, np.float32(-10.0), np.float32(10.0))
        else:
            self.raw_clip_coordinate_count = 0
        _, recovered = safe_target_transform_numpy(self.raw)
        self.target_projection_max_abs_rad = float(np.max(np.abs(recovered.astype(np.float64) - target)))
        if not project_to_raw_clip and self.target_projection_max_abs_rad > 2.0e-5:
            raise ValueError("reference target safe-transform round trip failed")

    def infer(self, encoder267: np.ndarray, policy930: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del encoder267, policy930
        return self.raw.copy(), np.zeros(994, dtype=np.float32)


def _reference_policy_frame(motion: Mapping[str, np.ndarray], index: int) -> np.ndarray:
    q_hardware = np.asarray(motion["joint_pos"][index], dtype=np.float64)
    dq_hardware = np.asarray(motion["joint_vel"][index], dtype=np.float64)
    q_native = q_hardware[MJ_TO_NATIVE]
    dq_native = dq_hardware[MJ_TO_NATIVE]
    q29 = np.zeros(29, dtype=np.float64)
    dq29 = np.zeros(29, dtype=np.float64)
    action29 = np.zeros(29, dtype=np.float64)
    q29[NATIVE_TO_IL29] = q_native - DEFAULT_NATIVE
    dq29[NATIVE_TO_IL29] = dq_native
    safe_hardware = (q_hardware - np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float64)) / np.asarray(
        HARDWARE_23_ACTION_SCALE, dtype=np.float64
    )
    action29[NATIVE_TO_IL29] = safe_hardware[MJ_TO_NATIVE]
    angular = world_angular_velocity_to_body(
        motion["body_quat_w"][index, 0],
        motion["body_ang_vel_w"][index, 0],
    )
    gravity = _projected_gravity(motion["body_quat_w"][index, 0])
    result = np.concatenate((angular, q29, dq29, action29, gravity)).astype(np.float32)
    if result.shape != (93,) or not np.isfinite(result).all():
        raise ValueError("reference policy history frame is invalid")
    return result


class LibraryMotionTrue23Controller(CleanTrue23MujocoController):
    """Same causal actor/action transform, with crawl-aware tracking evidence."""

    def step_library(
        self,
        encoder267: np.ndarray,
        *,
        reference_body_pos_w: np.ndarray,
        reference_body_quat_w: np.ndarray,
        reference_joint_pos: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray]:
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

        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("MuJoCo state became nonfinite")
        gravity = _projected_gravity(self.data.qpos[3:7])
        tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
        height = float(self.data.qpos[2])
        actual_body_pos = np.asarray(self.data.xpos[1:], dtype=np.float64)
        reference_pos = np.asarray(reference_body_pos_w, dtype=np.float64)
        position_errors = np.linalg.norm(
            actual_body_pos[np.asarray(TRACKED_BODY_INDICES)] - reference_pos[np.asarray(TRACKED_BODY_INDICES)],
            axis=1,
        )
        relative_position_errors = np.linalg.norm(
            (actual_body_pos[np.asarray(TRACKED_BODY_INDICES)] - actual_body_pos[TRACKED_BODY_INDICES[0]])
            - (reference_pos[np.asarray(TRACKED_BODY_INDICES)] - reference_pos[TRACKED_BODY_INDICES[0]]),
            axis=1,
        )
        pelvis_orientation_error = _quaternion_error_rad(
            np.asarray(self.data.qpos[3:7], dtype=np.float64),
            np.asarray(reference_body_quat_w, dtype=np.float64)[0],
        )
        joint_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.asarray(self.data.qpos[7:], dtype=np.float64)
                        - np.asarray(reference_joint_pos, dtype=np.float64)
                    )
                )
            )
        )
        evidence: dict[str, Any] = {
            "base_height_m": height,
            "base_tilt_rad": tilt,
            "pelvis_position_error_m": float(position_errors[0]),
            "pelvis_orientation_error_rad": pelvis_orientation_error,
            "maximum_tracked_body_position_error_m": float(np.max(position_errors)),
            "maximum_relative_tracked_body_position_error_m": float(np.max(relative_position_errors)),
            "tracked_body_position_errors_m": {
                name: float(value) for name, value in zip(TRACKED_BODY_NAMES, position_errors, strict=True)
            },
            "joint_tracking_rmse_rad": joint_rmse,
            "max_abs_raw_action": float(np.max(np.abs(raw))),
            "max_abs_torque_nm": maximum_torque,
        }
        failures: list[str] = []
        if height < MINIMUM_ABSOLUTE_HEIGHT_M:
            failures.append("absolute_height")
        if tilt > MAXIMUM_ABSOLUTE_TILT_RAD:
            failures.append("absolute_tilt")
        if pelvis_orientation_error > MAXIMUM_PELVIS_ORIENTATION_ERROR_RAD:
            failures.append("pelvis_orientation_tracking")
        if (
            evidence["maximum_relative_tracked_body_position_error_m"]
            > MAXIMUM_RELATIVE_TRACKED_BODY_POSITION_ERROR_M
        ):
            failures.append("relative_tracked_body_position")
        if joint_rmse > MAXIMUM_JOINT_TRACKING_RMSE_RAD:
            failures.append("joint_tracking")
        evidence["gate_failures"] = failures
        return evidence, np.asarray(self.data.qpos, dtype=np.float64).copy()


def run_library_motion_replay(
    *,
    repository_root: Path,
    motion_path: Path,
    maximum_steps: int | None = None,
    decoder_path: Path | None = None,
    expected_decoder_sha256: str | None = None,
    controller_mode: str = "sonic",
    initial_state_motion_path: Path | None = None,
    gain_profile: str = "true23_native",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = repository_root.resolve()
    resolved_motion = motion_path.resolve()
    with np.load(resolved_motion, allow_pickle=False) as archive:
        motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    frame_count = validate_library_motion(motion)
    state_motion = motion
    if initial_state_motion_path is not None:
        with np.load(initial_state_motion_path.resolve(), allow_pickle=False) as archive:
            state_motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        if validate_library_motion(state_motion) != frame_count:
            raise ValueError("initial state motion frame count mismatch")
    available_steps = frame_count - 11
    steps = available_steps if maximum_steps is None else min(int(maximum_steps), available_steps)
    if steps <= 0:
        raise ValueError("maximum_steps must be positive")
    if controller_mode not in {"sonic", "reference_pd"}:
        raise ValueError("controller_mode must be sonic or reference_pd")
    if gain_profile not in {"true23_native", "released_retained"}:
        raise ValueError("gain_profile must be true23_native or released_retained")
    encoder_path = root / "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx"
    resolved_decoder: Path | None = None
    expected_decoder: str | None = None
    if controller_mode == "sonic":
        resolved_decoder = (
            root / "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.decoder.onnx"
            if decoder_path is None
            else decoder_path.resolve()
        )
        if expected_decoder_sha256 is None:
            expected_decoder = ORIGINAL_TRUE23_DECODER_SHA256
        else:
            expected_decoder = expected_decoder_sha256
        if len(expected_decoder) != 64:
            raise ValueError("expected decoder SHA256 must contain 64 hex characters")
        policy: Any = ExactHashSonicPolicy(
            encoder_path,
            resolved_decoder,
            expected_decoder_sha256=expected_decoder,
        )
    else:
        policy = ReferenceActionPolicy()
    controller = LibraryMotionTrue23Controller(
        model_path=root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        physics_path=root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        policy=policy,
    )
    if gain_profile == "released_retained":
        np.copyto(controller.physics.kp, RELEASED_RETAINED_KP)
        np.copyto(controller.physics.kd, RELEASED_RETAINED_KD)
    root_velocity = np.concatenate(
        (
            state_motion["body_lin_vel_w"][10, 0],
            _quaternion_matrix(state_motion["body_quat_w"][10, 0]).T @ state_motion["body_ang_vel_w"][10, 0],
        )
    )
    controller.reset(
        base_position=state_motion["body_pos_w"][10, 0],
        base_quaternion_wxyz=state_motion["body_quat_w"][10, 0],
        joint_position_hardware=state_motion["joint_pos"][10],
        root_velocity=root_velocity,
        joint_velocity_hardware=state_motion["joint_vel"][10],
        buffered_robot_pelvis_q9=state_motion["body_quat_w"][9, 0],
    )
    controller.history = [_reference_policy_frame(state_motion, index) for index in range(10)]
    qpos = [np.asarray(controller.data.qpos, dtype=np.float64).copy()]
    scalar_names = (
        "base_height_m",
        "base_tilt_rad",
        "pelvis_position_error_m",
        "pelvis_orientation_error_rad",
        "maximum_tracked_body_position_error_m",
        "maximum_relative_tracked_body_position_error_m",
        "joint_tracking_rmse_rad",
        "max_abs_raw_action",
        "max_abs_torque_nm",
    )
    series: dict[str, list[float]] = {name: [] for name in scalar_names}
    failure: dict[str, Any] | None = None
    try:
        for transition in range(steps):
            q9 = 9 + transition
            reference_index = q9 + 2
            if isinstance(policy, ReferenceActionPolicy):
                policy.set_target(motion["joint_pos"][reference_index])
            packet = motion_reference_terms(motion, q9)
            evidence, state = controller.step_library(
                encoder267_from_reference(packet, controller.buffered_robot_pelvis_q9),
                reference_body_pos_w=motion["body_pos_w"][reference_index],
                reference_body_quat_w=motion["body_quat_w"][reference_index],
                reference_joint_pos=motion["joint_pos"][reference_index],
            )
            qpos.append(state)
            for name in scalar_names:
                series[name].append(float(evidence[name]))
            if evidence["gate_failures"]:
                failure = {
                    "transition": transition,
                    "q9": q9,
                    "type": "CrawlPhysicalGateFailure",
                    "message": "crawl physical gate failed: " + ",".join(evidence["gate_failures"]),
                    "evidence": evidence,
                }
                break
    except Exception as error:
        qpos.append(np.asarray(controller.data.qpos, dtype=np.float64).copy())
        failure = {
            "transition": controller.completed - 1,
            "q9": 8 + controller.completed,
            "type": type(error).__name__,
            "message": str(error),
        }
    completed = controller.completed
    report = {
        "schema_version": 1,
        "kind": "g1_true23_genuine_sonic_library_motion_mujoco_replay",
        "physical_model": "g1_23dof_rev_1_0",
        "physical_dof": 23,
        "actuator_count": int(controller.model.nu),
        "decoder_output_dof": 23 if controller_mode == "sonic" else None,
        "source_29dof_physics_used": False,
        "motion_path": str(resolved_motion),
        "motion_sha256": sha256_file(resolved_motion),
        "controller_mode": controller_mode,
        "gain_profile": gain_profile,
        "decoder_path": None if resolved_decoder is None else str(resolved_decoder),
        "decoder_sha256": expected_decoder,
        "history_initialization": "causal_reference_q0_through_q9_then_measured_q10",
        "initial_state_motion_path": (
            None if initial_state_motion_path is None else str(initial_state_motion_path.resolve())
        ),
        "frame_count": frame_count,
        "requested_transitions": steps,
        "completed_transitions": completed,
        "first_q9": 9,
        "last_q9": 8 + completed,
        "failure": failure,
        "gates": {
            "minimum_absolute_height_m": MINIMUM_ABSOLUTE_HEIGHT_M,
            "maximum_absolute_tilt_rad": MAXIMUM_ABSOLUTE_TILT_RAD,
            "maximum_pelvis_orientation_error_rad": MAXIMUM_PELVIS_ORIENTATION_ERROR_RAD,
            "maximum_relative_tracked_body_position_error_m": MAXIMUM_RELATIVE_TRACKED_BODY_POSITION_ERROR_M,
            "maximum_joint_tracking_rmse_rad": MAXIMUM_JOINT_TRACKING_RMSE_RAD,
            "world_position_tracking_diagnostic_only": True,
        },
        "metrics": {
            "minimum_base_height_m": min(series["base_height_m"], default=None),
            "maximum_base_tilt_rad": max(series["base_tilt_rad"], default=None),
            "maximum_pelvis_position_error_m": max(series["pelvis_position_error_m"], default=None),
            "maximum_pelvis_orientation_error_rad": max(series["pelvis_orientation_error_rad"], default=None),
            "maximum_tracked_body_position_error_m": max(
                series["maximum_tracked_body_position_error_m"], default=None
            ),
            "maximum_relative_tracked_body_position_error_m": max(
                series["maximum_relative_tracked_body_position_error_m"], default=None
            ),
            "maximum_joint_tracking_rmse_rad": max(series["joint_tracking_rmse_rad"], default=None),
            "maximum_absolute_raw_action": max(series["max_abs_raw_action"], default=None),
            "maximum_absolute_torque_nm": max(series["max_abs_torque_nm"], default=None),
            "horizontal_displacement_m": float(np.linalg.norm(np.asarray(qpos[-1])[:2] - np.asarray(qpos[0])[:2])),
        },
        "passed": failure is None and completed == steps,
        "authorization": {
            "simulator_only": True,
            "dds_opened": False,
            "network_used": False,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }
    arrays = {
        "qpos": np.ascontiguousarray(qpos, dtype=np.float32),
        "completed_q9": np.arange(9, 9 + completed, dtype=np.int64),
        **{name: np.asarray(values, dtype=np.float32) for name, values in series.items()},
    }
    return report, arrays


__all__ = [
    "LibraryMotionTrue23Controller",
    "run_library_motion_replay",
    "validate_library_motion",
]
