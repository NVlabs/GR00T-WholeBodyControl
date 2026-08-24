"""Deterministic MJLab diagnostic for the frozen 21204+SONIC-V2 composite.

The selected native124 policy never drives the simulator directly.  Its
hardware target is first expressed in plain SONIC native coordinates.  That
pre-transform action reaches MJLab's exact V11 action term, which applies the
V2 transform once and exposes both the safe native action and hardware target.

This module has no transport or hardware APIs.  A successful nominal window is
diagnostic evidence only; it does not admit teacher labels or authorize use on
a robot.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np

from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTOR_STATE_SHA256,
    CHECKPOINT_SHA256,
    EXPORT_REPORT_SHA256,
    MANIFEST_SHA256,
    ONNX_SHA256,
    SELECTION_SHA256,
    Native124Checkpoint21204Policy,
    load_checkpoint21204_binding,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_CONSTANTS_SHA256,
    SAFE_TARGET_FORMULA_SHA256,
)
from gear_sonic.utils.g1_true23_teacher_support import (
    SUPPORT_CONFIG_SHA256,
    TEACHER_COMPOSITE_CONSTANTS_SHA256,
    TEACHER_COMPOSITE_FORMULA_SHA256,
    Checkpoint21204TeacherComposite,
    compose_checkpoint21204_teacher_action,
    load_teacher_support_contract,
)

CONTRACT_RELATIVE_PATH = Path("gear_sonic/config/sim_validation/g1_true23_native124_21204_composite_mjlab_v1.json")
CONTRACT_SHA256 = "841eb09a6210fc0753cfdc795570a758811baa3eeb458f0bc5e30e624667b2ff"
CONTRACT_KIND = "g1_true23_native124_21204_composite_mjlab_contract_v1"
REPORT_KIND = "g1_true23_native124_21204_composite_mjlab_report_v1"
WINDOW_STEPS = 500
WARMUP_STEPS = 2
INITIAL_Q9 = 9
FINAL_Q9 = INITIAL_Q9 + WARMUP_STEPS + WINDOW_STEPS
ACTION_DIM = 23


class CompositeWindowQuarantine(RuntimeError):
    """Carry partial simulator evidence out of a fail-closed window."""

    def __init__(self, message: str, payload: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.payload = dict(payload)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _finite_float32_vector(value: object, context: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.float32 or result.shape != (ACTION_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"{context} must be finite float32 [23]")
    return np.ascontiguousarray(result).copy()


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != 1
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("role") != "offline_mjlab_nominal_composite_diagnostic_only"
    ):
        raise ValueError("composite MJLab contract identity mismatch")
    identity = _mapping(contract.get("artifact_identity"), "artifact_identity")
    expected_identity = {
        "iteration": 21204,
        "tracker_manifest_sha256": MANIFEST_SHA256,
        "support_config_sha256": SUPPORT_CONFIG_SHA256,
        "selection_sha256": SELECTION_SHA256,
        "export_report_sha256": EXPORT_REPORT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "actor_state_sha256": ACTOR_STATE_SHA256,
        "onnx_sha256": ONNX_SHA256,
    }
    if dict(identity) != expected_identity:
        raise ValueError("composite MJLab artifact identity mismatch")
    motions = contract.get("motions")
    if motions != [
        {
            "name": "neutral_smoke",
            "basename": "g1_true23_causal_neutral_acquisition_v3.npz",
            "sha256": "228ebc200191339da2b8cff6f9f2d5be6342c0603dbde05fc7eddfe5d044fe1d",
            "fps": 50.0,
            "frame_count": 600,
        },
        {
            "name": "dad_dance_primary",
            "basename": "B_DadDance.npz",
            "sha256": "a4962f1e4df45ca70ada473a962b52527b17ac667dfb17ef3cd37d4ed21c3bfb",
            "fps": 50.0,
            "frame_count": 2090,
        },
    ]:
        raise ValueError("composite MJLab motion identities mismatch")
    array_contract = _mapping(
        contract.get("motion_array_contract"),
        "motion_array_contract",
    )
    if array_contract.get("required_keys") != [
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]:
        raise ValueError("composite MJLab motion-array contract mismatch")
    window = _mapping(contract.get("causal_window"), "causal_window")
    if dict(window) != {
        "num_envs": 1,
        "sampling_mode": "start",
        "initial_q9_index": INITIAL_Q9,
        "synchronization_warmup_transitions": WARMUP_STEPS,
        "teacher_control_transitions": WINDOW_STEPS,
        "expected_final_q9_index": FINAL_Q9,
        "control_hz": 50.0,
        "play_mode": True,
        "wrapper_action_clip": None,
    }:
        raise ValueError("composite MJLab causal-window contract mismatch")
    chain = _mapping(contract.get("action_chain"), "action_chain")
    if (
        chain.get("teacher_composite_formula_sha256") != TEACHER_COMPOSITE_FORMULA_SHA256
        or chain.get("teacher_composite_constants_sha256") != TEACHER_COMPOSITE_CONSTANTS_SHA256
        or chain.get("safe_target_formula_sha256") != SAFE_TARGET_FORMULA_SHA256
        or chain.get("safe_target_constants_sha256") != SAFE_TARGET_CONSTANTS_SHA256
        or chain.get("plain_action_term_required") is not False
        or chain.get("safe_target_v11_action_term_required") is not True
        or chain.get("environment_input_semantics") != "plain_sonic_raw_native_pre_safe_transform"
        or chain.get("external_or_wrapper_clipping_permitted") is not False
        or chain.get("plain_sonic_raw_abs_strict_max") != 10.0
        or chain.get("actual_action_atol") != 0.0
        or chain.get("actual_target_atol") != 1.0e-5
    ):
        raise ValueError("composite MJLab action-chain contract mismatch")
    gate = _mapping(contract.get("nominal_gate"), "nominal_gate")
    velocity = np.asarray(gate.get("velocity_limit_hardware_radps"))
    if (
        gate.get("minimum_base_height_m") != 0.45
        or gate.get("maximum_base_tilt_rad") != 1.0
        or gate.get("maximum_joint_velocity_ratio") != 1.0
        or gate.get("maximum_tracking_rmse_rad") != 0.75
        or gate.get("inference_duration_ms_strict_max") != 20.0
        or velocity.shape != (ACTION_DIM,)
        or not np.isfinite(velocity).all()
        or np.any(velocity <= 0.0)
    ):
        raise ValueError("composite MJLab nominal gate mismatch")
    boundaries = _mapping(contract.get("boundaries"), "boundaries")
    required_true = {"simulator_only", "nominal_diagnostic_only"}
    required_false = {
        "robot_or_network_commands_permitted",
        "actuation_permitted",
        "teacher_labels_admitted",
        "support_qualified",
        "deployment_ready",
        "promotion_eligible",
        "hardware_authorized",
    }
    if any(boundaries.get(name) is not True for name in required_true) or any(
        boundaries.get(name) is not False for name in required_false
    ):
        raise ValueError("composite MJLab safety boundary mismatch")


def load_composite_mjlab_contract(
    repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Load the exact diagnostic contract and verify all frozen identities."""

    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[2]
    path = (root / CONTRACT_RELATIVE_PATH).resolve()
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError("composite MJLab contract must be a regular repository file")
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != CONTRACT_SHA256:
        raise ValueError(f"composite MJLab contract SHA256 mismatch: expected {CONTRACT_SHA256}, got {actual}")
    try:
        contract = _mapping(json.loads(payload.decode("utf-8")), "contract")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("composite MJLab contract must be UTF-8 JSON") from error
    _validate_contract(contract)
    load_teacher_support_contract(root)
    return contract


def validate_diagnostic_motion(
    motion_path: str | Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash and shape-check the sole admitted deterministic motion input."""

    path = Path(motion_path).expanduser().resolve()
    motions = contract.get("motions")
    if not isinstance(motions, list):
        raise ValueError("contract motions must be a list")
    matching = [item for item in motions if isinstance(item, Mapping) and item.get("basename") == path.name]
    if len(matching) != 1:
        raise ValueError("motion is not one exact admitted diagnostic input")
    motion_contract = matching[0]
    array_contract = _mapping(
        contract.get("motion_array_contract"),
        "motion_array_contract",
    )
    if path.is_symlink() or not path.is_file():
        raise ValueError("neutral motion must be a regular non-symlink file")
    actual_sha = _sha256_file(path)
    if actual_sha != motion_contract["sha256"]:
        raise ValueError("diagnostic motion SHA256 mismatch")
    required_keys = tuple(array_contract["required_keys"])
    frames = int(motion_contract["frame_count"])
    with np.load(path, allow_pickle=False) as data:
        if tuple(data.files) != required_keys:
            raise ValueError("diagnostic motion NPZ key order/contents mismatch")
        fps = np.asarray(data["fps"])
        expected_shapes = {
            "joint_pos": (frames, ACTION_DIM),
            "joint_vel": (frames, ACTION_DIM),
            "body_pos_w": (frames, 24, 3),
            "body_quat_w": (frames, 24, 4),
            "body_lin_vel_w": (frames, 24, 3),
            "body_ang_vel_w": (frames, 24, 3),
        }
        if fps.shape != (1,) or float(fps[0]) != float(motion_contract["fps"]):
            raise ValueError("diagnostic motion FPS mismatch")
        for name, shape in expected_shapes.items():
            value = np.asarray(data[name])
            if value.dtype != np.float32 or value.shape != shape:
                raise ValueError(f"diagnostic motion {name} must be float32 {list(shape)}")
            if not np.isfinite(value).all():
                raise ValueError(f"diagnostic motion {name} contains NaN or Inf")
        quaternions = np.asarray(data["body_quat_w"], dtype=np.float64)
        norms = np.linalg.norm(quaternions, axis=-1)
        if np.any(np.abs(norms - 1.0) > 1.0e-3):
            raise ValueError("diagnostic motion body quaternions are not unit WXYZ")
    return {
        "path": str(path),
        "name": str(motion_contract["name"]),
        "basename": path.name,
        "sha256": actual_sha,
        "fps": float(motion_contract["fps"]),
        "frame_count": frames,
        "joint_count": ACTION_DIM,
        "body_count": 24,
    }


def composite_action_diagnostic_record(
    composite: Checkpoint21204TeacherComposite,
) -> dict[str, Any]:
    """Expose every linked action space without admitting a training label."""

    raw = _finite_float32_vector(
        composite.teacher_raw_action_hardware,
        "selected teacher raw action",
    )
    candidate = _finite_float32_vector(
        composite.teacher_candidate_target_hardware,
        "selected teacher candidate target",
    )
    plain = _finite_float32_vector(
        composite.teacher_action_native,
        "plain SONIC raw native diagnostic",
    )
    safe = _finite_float32_vector(
        composite.teacher_applied_safe_action_native,
        "applied safe native action",
    )
    target = _finite_float32_vector(
        composite.teacher_target_hardware,
        "composite teacher target",
    )
    if float(np.max(np.abs(plain))) >= 10.0:
        raise ValueError("composite action requires forbidden raw clipping")
    projection = (target - candidate).astype(np.float32, copy=False)
    return {
        "selected_teacher_raw_action_hardware": raw.tolist(),
        "teacher_candidate_target_hardware": candidate.tolist(),
        "plain_sonic_raw_native_diagnostic": plain.tolist(),
        "applied_safe_native_action": safe.tolist(),
        "teacher_composite_target_hardware": target.tolist(),
        "projection_delta_hardware": projection.tolist(),
        "max_abs_selected_teacher_raw_action": float(np.max(np.abs(raw))),
        "max_abs_plain_sonic_raw_native": float(np.max(np.abs(plain))),
        "max_abs_applied_safe_native_action": float(np.max(np.abs(safe))),
        "projection_linf_rad": float(np.max(np.abs(projection))),
        "projection_l2_rad": float(np.linalg.norm(projection.astype(np.float64))),
        "diagnostic_only": True,
        "teacher_label_admitted": False,
    }


def _numpy_23(value: Any, context: str) -> np.ndarray:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime-only dependency
        raise RuntimeError("PyTorch is required for MJLab diagnostics") from error
    if not isinstance(value, torch.Tensor) or value.shape != (1, ACTION_DIM):
        raise ValueError(f"{context} must be a tensor [1,23]")
    if value.dtype != torch.float32 or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{context} must be finite float32")
    return value.detach().to(device="cpu").contiguous().numpy()[0].copy()


def _scalar_tensor(value: Any, context: str) -> float:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime-only dependency
        raise RuntimeError("PyTorch is required for MJLab diagnostics") from error
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"{context} must be a scalar tensor")
    result = float(value.detach().to(device="cpu").item())
    if not math.isfinite(result):
        raise ValueError(f"{context} is nonfinite")
    return result


def _q9(command: Any) -> int:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime-only dependency
        raise RuntimeError("PyTorch is required for MJLab diagnostics") from error
    value = getattr(command, "time_steps", None)
    if not isinstance(value, torch.Tensor) or value.shape != (1,):
        raise ValueError("MJLab motion time_steps must be [1]")
    return int(value.detach().to(device="cpu").item())


def _json_safe(value: Any) -> Any:
    """Convert small MJLab failure metadata without accepting nonfinite values."""

    try:
        import torch
    except ImportError:  # pragma: no cover - runtime always has torch
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        return _json_safe(value.detach().to(device="cpu").numpy())
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("MJLab failure metadata contains NaN or Inf")
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("MJLab failure metadata contains NaN or Inf")
        return value
    return repr(value)


def _single_record(batch: Mapping[str, Any], *, require_candidate: bool) -> Mapping[str, Any]:
    if batch.get("record_count") != 1:
        raise ValueError("shadow collector must return one environment record")
    records = batch.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("shadow collector record payload mismatch")
    record = _mapping(records[0], "shadow record")
    expected = "shadow_candidate" if require_candidate else "quarantine"
    if record.get("verdict") != expected:
        reasons = record.get("quarantine_reasons")
        raise RuntimeError(f"shadow transition expected {expected}, got {record.get('verdict')}: {reasons}")
    return record


def _step_metrics(
    *,
    raw_env: Any,
    composite: Checkpoint21204TeacherComposite,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    robot = raw_env.scene["robot"]
    action_term = raw_env.action_manager.get_term("joint_pos")
    actual_raw = _numpy_23(action_term.raw_action, "plain action raw action")
    actual_target = _numpy_23(
        action_term.processed_action,
        "plain action processed target",
    )
    expected_action = _finite_float32_vector(
        composite.teacher_action_native,
        "expected pre-transform native action",
    )
    expected_target = _finite_float32_vector(
        composite.teacher_target_hardware,
        "expected composite target",
    )
    action_atol = float(contract["action_chain"]["actual_action_atol"])
    target_atol = float(contract["action_chain"]["actual_target_atol"])
    action_match = bool(np.allclose(actual_raw, expected_action, rtol=0.0, atol=action_atol))
    target_match = bool(np.allclose(actual_target, expected_target, rtol=0.0, atol=target_atol))
    if not action_match or not target_match:
        raise RuntimeError("V11 MJLab action semantics differ from frozen composite")
    actual_safe = _numpy_23(
        action_term.safe_native_action,
        "V11 applied safe native action",
    )
    expected_safe = _finite_float32_vector(
        composite.teacher_applied_safe_action_native,
        "expected applied safe native action",
    )
    if not np.allclose(actual_safe, expected_safe, rtol=0.0, atol=target_atol):
        raise RuntimeError("V11 MJLab applied-safe action differs from composite")

    measured = _numpy_23(robot.data.joint_pos, "measured joint position")
    velocity = _numpy_23(robot.data.joint_vel, "measured joint velocity")
    soft_limits = getattr(robot.data, "soft_joint_pos_limits", None)
    if (
        not isinstance(soft_limits, torch.Tensor)
        or soft_limits.dtype != torch.float32
        or soft_limits.shape != (1, ACTION_DIM, 2)
        or not bool(torch.isfinite(soft_limits).all().item())
    ):
        raise ValueError("MJLab soft joint limits must be finite float32 [1,23,2]")
    limits = soft_limits.detach().to(device="cpu").numpy()[0]
    target_violation = (expected_target < limits[:, 0]) | (expected_target > limits[:, 1])
    measured_violation = (measured < limits[:, 0]) | (measured > limits[:, 1])

    encoder_bias = _numpy_23(robot.data.encoder_bias, "encoder bias")
    actuator_target = actual_target - encoder_bias
    actuator_target_violation = (actuator_target < limits[:, 0]) | (actuator_target > limits[:, 1])
    velocity_limit = np.asarray(
        contract["nominal_gate"]["velocity_limit_hardware_radps"],
        dtype=np.float32,
    )
    velocity_ratio = np.abs(velocity) / velocity_limit

    root_position = getattr(robot.data, "root_link_pos_w", None)
    if not isinstance(root_position, torch.Tensor) or root_position.shape != (1, 3):
        raise ValueError("MJLab root_link_pos_w must be [1,3]")
    base_height = _scalar_tensor(root_position[:, 2], "base height")
    gravity = getattr(robot.data, "projected_gravity_b", None)
    if not isinstance(gravity, torch.Tensor) or gravity.shape != (1, 3):
        raise ValueError("MJLab projected_gravity_b must be [1,3]")
    gravity_z = float(gravity[0, 2].detach().to(device="cpu").item())
    base_tilt = math.acos(max(-1.0, min(1.0, -gravity_z)))
    tracking_rmse = float(np.sqrt(np.mean(np.square(expected_target.astype(np.float64) - measured))))
    return {
        "actual_raw_native_action": actual_raw.tolist(),
        "actual_applied_safe_native_action": actual_safe.tolist(),
        "actual_unbiased_target_hardware": actual_target.tolist(),
        "actual_actuator_target_after_encoder_bias_hardware": actuator_target.tolist(),
        "measured_joint_position_hardware": measured.tolist(),
        "measured_joint_velocity_hardware": velocity.tolist(),
        "action_semantics_match": action_match and target_match,
        "target_soft_limit_violation_count": int(np.count_nonzero(target_violation)),
        "actuator_target_soft_limit_violation_count": int(np.count_nonzero(actuator_target_violation)),
        "measured_soft_limit_violation_count": int(np.count_nonzero(measured_violation)),
        "joint_velocity_limit_violation_count": int(np.count_nonzero(velocity_ratio > 1.0)),
        "maximum_joint_velocity_ratio": float(np.max(velocity_ratio)),
        "base_height_m": base_height,
        "base_tilt_rad": base_tilt,
        "target_tracking_rmse_rad": tracking_rmse,
    }


def _summary(records: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    if len(records) != WINDOW_STEPS:
        raise RuntimeError(f"composite window has {len(records)} records, expected {WINDOW_STEPS}")
    metrics = [record["metrics"] for record in records]
    actions = [record["composite_action"] for record in records]
    latencies = np.asarray(
        [record["teacher_join_inference_duration_ms"] for record in records],
        dtype=np.float64,
    )
    q9_values = [int(record["q9_before"]) for record in records]
    expected_q9 = list(range(INITIAL_Q9 + WARMUP_STEPS, FINAL_Q9))
    if q9_values != expected_q9:
        raise RuntimeError("composite diagnostic q9 sequence is not contiguous")
    counts = {
        "termination_count": 0,
        "q9_discontinuity_count": 0,
        "nonfinite_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": sum(not bool(metric["action_semantics_match"]) for metric in metrics),
        "target_soft_limit_violation_count": sum(
            int(metric["target_soft_limit_violation_count"]) for metric in metrics
        ),
        "actuator_target_soft_limit_violation_count": sum(
            int(metric["actuator_target_soft_limit_violation_count"]) for metric in metrics
        ),
        "measured_soft_limit_violation_count": sum(
            int(metric["measured_soft_limit_violation_count"]) for metric in metrics
        ),
        "joint_velocity_limit_violation_count": sum(
            int(metric["joint_velocity_limit_violation_count"]) for metric in metrics
        ),
    }
    summary = {
        "record_count": len(records),
        "q9_first": q9_values[0],
        "q9_last_before": q9_values[-1],
        "q9_final_after": int(records[-1]["q9_after"]),
        **counts,
        "minimum_base_height_m": min(float(metric["base_height_m"]) for metric in metrics),
        "maximum_base_tilt_rad": max(float(metric["base_tilt_rad"]) for metric in metrics),
        "maximum_joint_velocity_ratio": max(float(metric["maximum_joint_velocity_ratio"]) for metric in metrics),
        "maximum_tracking_rmse_rad": max(float(metric["target_tracking_rmse_rad"]) for metric in metrics),
        "mean_tracking_rmse_rad": float(np.mean([metric["target_tracking_rmse_rad"] for metric in metrics])),
        "maximum_selected_teacher_raw_abs": max(
            float(action["max_abs_selected_teacher_raw_action"]) for action in actions
        ),
        "maximum_plain_sonic_raw_native_abs": max(
            float(action["max_abs_plain_sonic_raw_native"]) for action in actions
        ),
        "mean_projection_linf_rad": float(np.mean([action["projection_linf_rad"] for action in actions])),
        "maximum_projection_linf_rad": max(float(action["projection_linf_rad"]) for action in actions),
        "mean_projection_l2_rad": float(np.mean([action["projection_l2_rad"] for action in actions])),
        "teacher_join_inference_duration_ms_p50": float(np.percentile(latencies, 50)),
        "teacher_join_inference_duration_ms_p99": float(np.percentile(latencies, 99)),
        "teacher_join_inference_duration_ms_max": float(np.max(latencies)),
    }
    gate = contract["nominal_gate"]
    required_zero = tuple(gate["required_zero_counts"])
    zero_pass = all(int(summary[name]) == 0 for name in required_zero)
    threshold_pass = (
        summary["minimum_base_height_m"] >= float(gate["minimum_base_height_m"])
        and summary["maximum_base_tilt_rad"] <= float(gate["maximum_base_tilt_rad"])
        and summary["maximum_joint_velocity_ratio"] <= float(gate["maximum_joint_velocity_ratio"])
        and summary["maximum_tracking_rmse_rad"] <= float(gate["maximum_tracking_rmse_rad"])
        and summary["teacher_join_inference_duration_ms_max"] < float(gate["inference_duration_ms_strict_max"])
        and summary["maximum_plain_sonic_raw_native_abs"] < 10.0
        and summary["q9_final_after"] == FINAL_Q9
    )
    summary["nominal_gate_pass"] = bool(zero_pass and threshold_pass)
    return summary


def _partial_summary(
    records: Sequence[Mapping[str, Any]],
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize an incomplete window without converting it into a pass."""

    result: dict[str, Any] = {
        "record_count": len(records),
        "required_record_count": WINDOW_STEPS,
        "nominal_gate_pass": False,
        "whole_window_quarantined": True,
        "failure_stage": failure.get("stage"),
        "failure_step": failure.get("step"),
    }
    if records:
        result.update(
            {
                "q9_first": int(records[0]["q9_before"]),
                "q9_last_before": int(records[-1]["q9_before"]),
                "q9_last_after": int(records[-1]["q9_after"]),
                "minimum_base_height_m": min(float(record["metrics"]["base_height_m"]) for record in records),
                "maximum_base_tilt_rad": max(float(record["metrics"]["base_tilt_rad"]) for record in records),
                "maximum_tracking_rmse_rad": max(
                    float(record["metrics"]["target_tracking_rmse_rad"]) for record in records
                ),
                "maximum_projection_linf_rad": max(
                    float(record["composite_action"]["projection_linf_rad"]) for record in records
                ),
            }
        )
    return result


def _run_mjlab_window(
    *,
    repository_root: Path,
    motion_path: Path,
    seed: int,
    device: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    import torch

    from gear_sonic.envs.mjlab.sonic_true23 import (
        prime_sonic_true23_training_environment,
    )
    from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
        make_causal_history_recovery_env_cfg,
    )
    from gear_sonic.envs.mjlab.sonic_true23_causal_history_safe_target_v11 import (
        SafeTargetNativeIl23JointPositionAction,
        SafeTargetNativeIl23JointPositionActionCfg,
    )
    from gear_sonic.utils.g1_23dof_safe_target_transform import (
        safe_target_transform_contract,
    )
    from gear_sonic.utils.g1_true23_native124_21204_mjlab_shadow import (
        Native124Checkpoint21204MjlabShadowCollector,
    )

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if device != "cuda:0":
        raise ValueError("frozen composite MJLab diagnostic requires device cuda:0")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    configure_torch_backends()
    torch.cuda.set_device(0)

    cfg = make_causal_history_recovery_env_cfg(
        motion_file=str(motion_path),
        num_envs=1,
        play=True,
    )
    cfg.actions["joint_pos"] = SafeTargetNativeIl23JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
    )
    cfg.seed = seed
    command_cfg = cfg.commands["motion"]
    command_cfg.sampling_mode = "start"
    action_cfg = cfg.actions["joint_pos"]
    if type(action_cfg) is not SafeTargetNativeIl23JointPositionActionCfg:
        raise ValueError("diagnostic environment must use exact V11 safe-target action cfg")
    if cfg.scene.num_envs != 1 or command_cfg.sampling_mode != "start":
        raise ValueError("diagnostic environment batch/start sampling drift")
    if cfg.observations["tokenizer"].enable_corruption is not False:
        raise ValueError("diagnostic tokenizer corruption must be disabled")
    if cfg.observations["policy"].enable_corruption is not False:
        raise ValueError("diagnostic policy corruption must be disabled")
    environment_audit = {
        "schema": "g1_true23_native124_21204_composite_nominal_env_v1",
        "base_environment": "causal_history_recovery_play",
        "action_cfg": (f"{type(action_cfg).__module__}:{type(action_cfg).__qualname__}"),
        "target_transform": safe_target_transform_contract(),
        "training_reward_profile_claimed": False,
        "v11_action_semantics_only": True,
        "play_mode": True,
        "sampling_mode": "start",
        "wrapper_action_clip": None,
    }

    binding = load_checkpoint21204_binding(repository_root)
    teacher = Native124Checkpoint21204Policy(binding)
    collector = Native124Checkpoint21204MjlabShadowCollector(teacher)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    records: list[dict[str, Any]] = []
    warmup: list[dict[str, Any]] = []
    prime: Mapping[str, Any] | None = None
    environment_record: dict[str, Any] = {
        "device": device,
        "seed": seed,
        "num_envs": 1,
        "play_mode": True,
        "sampling_mode": "start",
        "wrapper_action_clip": None,
        "action_cfg": f"{type(action_cfg).__module__}:{type(action_cfg).__qualname__}",
        "action_term": None,
        "safe_target_action_term_used": True,
        "safe_target_transform_application_count": 1,
        "environment_audit": environment_audit,
        "robot_or_network_commands_performed": False,
    }
    failure_context: dict[str, Any] = {"stage": "wrapper_construction"}
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        if wrapped.clip_actions is not None:
            raise ValueError("diagnostic wrapper clipping must be disabled")
        if type(env.action_manager.get_term("joint_pos")) is not SafeTargetNativeIl23JointPositionAction:
            raise ValueError("diagnostic environment executed a non-V11 action term")
        action_term = env.action_manager.get_term("joint_pos")
        environment_record["action_term"] = f"{type(action_term).__module__}:{type(action_term).__qualname__}"
        failure_context = {"stage": "environment_prime"}
        prime = prime_sonic_true23_training_environment(wrapped)
        observations = wrapped.get_observations()
        command = env.command_manager.get_term("motion")
        if _q9(command) != INITIAL_Q9:
            raise RuntimeError("start-sampled diagnostic did not begin at q9=9")

        zero = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=device)
        candidate: Mapping[str, Any] | None = None
        candidate_latency_ms: float | None = None
        for warmup_index in range(WARMUP_STEPS):
            q9_before = _q9(command)
            failure_context = {
                "stage": "synchronization_warmup",
                "step": warmup_index,
                "q9_before": q9_before,
            }
            before = collector.before_step(wrapped, observations, zero)
            step_result = wrapped.step(zero)
            started = time.perf_counter_ns()
            batch = collector.after_step(wrapped, step_result, before)
            duration_ms = (time.perf_counter_ns() - started) / 1.0e6
            done = int(step_result[2][0].detach().to(device="cpu").item())
            q9_after = _q9(command)
            failure_context.update(
                {
                    "q9_after": q9_after,
                    "done": done,
                    "shadow_batch": _json_safe(batch),
                    "extras_log": _json_safe(step_result[3].get("log")),
                }
            )
            if done != 0:
                raise RuntimeError("diagnostic warmup terminated")
            if q9_after != q9_before + 1:
                raise RuntimeError("diagnostic warmup q9 discontinuity")
            require_candidate = warmup_index == WARMUP_STEPS - 1
            record = _single_record(batch, require_candidate=require_candidate)
            if not require_candidate and record.get("quarantine_reasons") != [
                "first_reset_or_resynchronization_frame"
            ]:
                raise RuntimeError("unexpected first warmup quarantine reason")
            warmup.append(
                {
                    "index": warmup_index,
                    "q9_before": q9_before,
                    "q9_after": q9_after,
                    "shadow_verdict": record["verdict"],
                    "quarantine_reasons": record["quarantine_reasons"],
                    "teacher_join_inference_duration_ms": duration_ms,
                }
            )
            observations = step_result[0]
            if require_candidate:
                candidate = record
                candidate_latency_ms = duration_ms

        if candidate is None or candidate_latency_ms is None:
            raise RuntimeError("diagnostic synchronization produced no teacher candidate")

        for step_index in range(WINDOW_STEPS):
            q9_before = _q9(command)
            failure_context = {
                "stage": "teacher_control",
                "step": step_index,
                "q9_before": q9_before,
                "candidate_shadow_record": _json_safe(candidate),
            }
            if int(candidate["q9_reference_index_after"]) != q9_before:
                raise RuntimeError("teacher candidate q9 does not match current command")
            raw = np.asarray(candidate["raw_tracker_action_hardware"], dtype=np.float32)
            composite = compose_checkpoint21204_teacher_action(
                raw,
                repository_root=repository_root,
            )
            action_record = composite_action_diagnostic_record(composite)
            failure_context["composite_action"] = action_record
            action = torch.as_tensor(
                composite.teacher_action_native,
                dtype=torch.float32,
                device=device,
            ).reshape(1, ACTION_DIM)
            before = collector.before_step(wrapped, observations, action)
            step_result = wrapped.step(action)
            started = time.perf_counter_ns()
            batch = collector.after_step(wrapped, step_result, before)
            next_latency_ms = (time.perf_counter_ns() - started) / 1.0e6
            done = int(step_result[2][0].detach().to(device="cpu").item())
            q9_after = _q9(command)
            failure_context.update(
                {
                    "q9_after": q9_after,
                    "done": done,
                    "shadow_batch": _json_safe(batch),
                    "extras_log": _json_safe(step_result[3].get("log")),
                }
            )
            if done != 0:
                raise RuntimeError(f"teacher-controlled transition {step_index} terminated")
            if q9_after != q9_before + 1:
                raise RuntimeError(f"teacher-controlled transition {step_index} has q9 discontinuity")
            next_candidate = _single_record(batch, require_candidate=True)
            metrics = _step_metrics(
                raw_env=env,
                composite=composite,
                contract=contract,
            )
            records.append(
                {
                    "step": step_index,
                    "q9_before": q9_before,
                    "q10_proof_before": q9_before + 1,
                    "q9_after": q9_after,
                    "q10_proof_after": q9_after + 1,
                    "teacher_join_inference_duration_ms": candidate_latency_ms,
                    "composite_action": action_record,
                    "metrics": metrics,
                    "diagnostic_only": True,
                    "teacher_label_admitted": False,
                    "support_qualified": False,
                }
            )
            observations = step_result[0]
            candidate = next_candidate
            candidate_latency_ms = next_latency_ms
        summary = _summary(records, contract)
        return {
            "prime": prime,
            "environment": environment_record,
            "warmup": warmup,
            "records": records,
            "summary": summary,
            "failure": None,
        }
    except Exception as error:
        failure_context["error"] = f"{type(error).__name__}: {error}"
        payload = {
            "prime": prime,
            "environment": environment_record,
            "warmup": warmup,
            "records": records,
            "summary": _partial_summary(records, failure_context),
            "failure": failure_context,
        }
        raise CompositeWindowQuarantine(str(failure_context["error"]), payload) from error
    finally:
        env.close()


def _report_base(seed: int, device: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "diagnostic_only": True,
        "simulator_only": True,
        "robot_or_network_commands_performed": False,
        "actuation_permitted": False,
        "teacher_labels_admitted": False,
        "support_qualified": False,
        "deployment_ready": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "computed_pass": False,
        "verdict": "quarantine",
        "error": None,
        "contract": {
            "path": str(CONTRACT_RELATIVE_PATH).replace("\\", "/"),
            "sha256": CONTRACT_SHA256,
        },
        "runtime_request": {
            "seed": seed,
            "device": device,
            "teacher_control_transitions": WINDOW_STEPS,
        },
        "motion": None,
        "prime": None,
        "environment": None,
        "warmup": [],
        "records": [],
        "summary": None,
        "failure": None,
    }


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_report_new(output_path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write one explicit report path with exclusive-create/no-overwrite semantics."""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report)
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    return output


def run_composite_mjlab_diagnostic(
    *,
    repository_root: str | Path,
    motion_path: str | Path,
    output_path: str | Path,
    seed: int = 20260805,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run and persist one exact 500-step nominal window, or quarantine it."""

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic report: {output}")
    report = _report_base(seed, device)
    try:
        root = Path(repository_root).expanduser().resolve()
        if root.is_symlink() or not root.is_dir():
            raise ValueError("repository_root must be a regular directory")
        contract = load_composite_mjlab_contract(root)
        motion = validate_diagnostic_motion(motion_path, contract)
        report["motion"] = motion
        result = _run_mjlab_window(
            repository_root=root,
            motion_path=Path(motion["path"]),
            seed=seed,
            device=device,
            contract=contract,
        )
        report.update(result)
        computed_pass = bool(result["summary"]["nominal_gate_pass"])
        report["computed_pass"] = computed_pass
        report["verdict"] = "nominal_diagnostic_pass" if computed_pass else "quarantine"
    except CompositeWindowQuarantine as error:
        report.update(error.payload)
        report["error"] = str(error)
        report["computed_pass"] = False
        report["verdict"] = "quarantine"
    except Exception as error:  # fail-closed report is the diagnostic product
        report["error"] = f"{type(error).__name__}: {error}"
        report["computed_pass"] = False
        report["verdict"] = "quarantine"
    write_report_new(output, report)
    return report


assert FINAL_Q9 == 511
