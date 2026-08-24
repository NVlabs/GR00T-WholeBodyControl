"""No-update parity gate for the causal native124 selected-V2 seam.

The gate is simulator-only.  It runs the frozen iteration-21204 CPU ONNX
policy through the causal PPO wrapper and compares stable trajectory semantics
against the durable DadDance composite report.  It never trains, deploys, or
opens a hardware/network transport.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from gear_sonic.envs.mjlab.native124_selected_v2_ankle_adaptation import (
    SELECTED_21204_ACTION_SCALE_HARDWARE,
    SELECTED_21204_HOME_Q_HARDWARE,
)
from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
    DAD_DANCE_FRAME_COUNT,
    DAD_DANCE_RELATIVE_PATH,
    DAD_DANCE_SHA256,
)
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ONNX_SHA256,
    Native124Checkpoint21204Policy,
    load_checkpoint21204_binding,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
)
from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import (
    composite_action_diagnostic_record,
    load_composite_mjlab_contract,
)
from gear_sonic.utils.g1_true23_teacher_support import (
    Checkpoint21204TeacherComposite,
    compose_checkpoint21204_teacher_action,
)

PARITY_KIND = "g1_true23_native124_selected_v2_causal_no_update_parity_v1"
BASELINE_RELATIVE_PATH = Path("artifacts/g1_true23/model21204_sonic_v2_daddance_seed20260805_nominal500_v3.json")
BASELINE_SHA256 = "aff51f0dc92446025c09196a000aec291b9fcf44576fdffab8e016da8adab4fa"
DEFAULT_OUTPUT_RELATIVE_PATH = Path(
    "artifacts/g1_true23/model21204_selected_v2_causal_wrapper_daddance_seed20260805_parity_v1.json"
)
SEED = 20260805
DEVICE = "cuda:0"
WARMUP_STEPS = 2
MAX_CONTROL_TRANSITIONS = 500
EXPECTED_COMPLETED_TRANSITIONS = 47
EXPECTED_DONE_CONTROL_TRANSITION = 47
EXPECTED_DONE_Q9 = 58
EXPECTED_RESET_Q9 = 9

ACTION_VECTOR_ATOL = 3.25e-4
MEASURED_POSITION_ATOL = 5.0e-5
MEASURED_VELOCITY_ATOL = 1.25e-3
TERMINAL_ACTOR_ATOL = 2.0e-4
METRIC_SCALAR_ATOL = 3.0e-5
WARMUP_TARGET_ATOL = 1.0e-6
NUMERICAL_PARITY_CHARACTERIZATION = "bounded CUDA/Warp numerical parity; not bitwise determinism"
NUMERICAL_TOLERANCE_CALIBRATION = {
    "fresh_exact_implementation_replays": 2,
    "maximum_observed_absolute_delta": {
        "action_vector_or_projection": 1.6021728515625e-4,
        "measured_joint_position_hardware": 1.5079975128173828e-5,
        "measured_joint_velocity_hardware": 6.113387644290924e-4,
        "terminal_actor_observation_124": 8.311867713928223e-5,
        "metric_scalar": 1.4439225196838379e-5,
    },
    "basis": "smallest rounded ceilings at or above approximately 2x observed cross-process CUDA/Warp drift",
}

_ACTION_VECTOR_FIELDS = (
    "selected_teacher_raw_action_hardware",
    "teacher_candidate_target_hardware",
    "plain_sonic_raw_native_diagnostic",
    "applied_safe_native_action",
    "teacher_composite_target_hardware",
    "projection_delta_hardware",
)
_ACTION_SCALAR_FIELDS = (
    "max_abs_selected_teacher_raw_action",
    "max_abs_plain_sonic_raw_native",
    "max_abs_applied_safe_native_action",
    "projection_linf_rad",
    "projection_l2_rad",
)
_METRIC_ACTION_VECTOR_FIELDS = (
    "actual_raw_native_action",
    "actual_applied_safe_native_action",
    "actual_unbiased_target_hardware",
    "actual_actuator_target_after_encoder_bias_hardware",
)
_METRIC_STATE_VECTOR_ATOLS = {
    "measured_joint_position_hardware": MEASURED_POSITION_ATOL,
    "measured_joint_velocity_hardware": MEASURED_VELOCITY_ATOL,
}
_METRIC_EXACT_FIELDS = (
    "action_semantics_match",
    "target_soft_limit_violation_count",
    "actuator_target_soft_limit_violation_count",
    "measured_soft_limit_violation_count",
    "joint_velocity_limit_violation_count",
)
_METRIC_SCALAR_FIELDS = (
    "maximum_joint_velocity_ratio",
    "base_height_m",
    "base_tilt_rad",
    "target_tracking_rmse_rad",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _finite_vector(value: object, context: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (23,) or not np.isfinite(result).all():
        raise ValueError(f"{context} must be finite float32 [23]")
    return result


def _finite_actor_vector(value: object, context: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (124,) or not np.isfinite(result).all():
        raise ValueError(f"{context} must be finite float32 [124]")
    return result


def _tensor_vector(value: Any, context: str) -> np.ndarray:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (1, 23)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(f"{context} must be finite float32 [1,23]")
    return value.detach().to(device="cpu").contiguous().numpy()[0].copy()


def _scalar(value: Any, context: str) -> float:
    import torch

    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"{context} must be a scalar tensor")
    result = float(value.detach().to(device="cpu").item())
    if not math.isfinite(result):
        raise ValueError(f"{context} is nonfinite")
    return result


def _q9(command: Any) -> int:
    import torch

    value = getattr(command, "time_steps", None)
    if not isinstance(value, torch.Tensor) or value.shape != (1,) or value.dtype != torch.long:
        raise ValueError("motion command q9 must be int64 [1]")
    return int(value.detach().to(device="cpu").item())


def _json_safe(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().to(device="cpu").numpy())
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("report value contains NaN or Inf")
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
            raise ValueError("report value contains NaN or Inf")
        return value
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def load_durable_baseline(repository_root: str | Path) -> dict[str, Any]:
    """Load and validate the exact durable failed DadDance report."""

    root = Path(repository_root).resolve()
    path = (root / BASELINE_RELATIVE_PATH).resolve(strict=True)
    if not path.is_file() or _sha256(path) != BASELINE_SHA256:
        raise ValueError("durable DadDance baseline SHA256 mismatch")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("durable DadDance baseline is not UTF-8 JSON") from error
    if not isinstance(report, dict):
        raise ValueError("durable DadDance baseline must be an object")
    summary = _mapping(report.get("summary"), "baseline.summary")
    failure = _mapping(report.get("failure"), "baseline.failure")
    records = report.get("records")
    if (
        report.get("kind") != "g1_true23_native124_21204_composite_mjlab_report_v1"
        or report.get("verdict") != "quarantine"
        or report.get("computed_pass") is not False
        or not isinstance(records, list)
        or len(records) != EXPECTED_COMPLETED_TRANSITIONS
        or summary.get("record_count") != EXPECTED_COMPLETED_TRANSITIONS
        or summary.get("q9_first") != 11
        or summary.get("q9_last_after") != EXPECTED_DONE_Q9
        or failure.get("stage") != "teacher_control"
        or failure.get("step") != EXPECTED_DONE_CONTROL_TRANSITION
        or failure.get("q9_before") != EXPECTED_DONE_Q9
        or failure.get("q9_after") != EXPECTED_RESET_Q9
        or failure.get("done") != 1
    ):
        raise ValueError("durable DadDance baseline semantic identity mismatch")
    return report


def derive_selected_warmup_raw_hardware() -> np.ndarray:
    """Selected raw whose affine candidate is the SONIC V2 default target."""

    default = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32)
    home = np.asarray(SELECTED_21204_HOME_Q_HARDWARE, dtype=np.float32)
    scale = np.asarray(SELECTED_21204_ACTION_SCALE_HARDWARE, dtype=np.float32)
    result = ((default - home) / scale).astype(np.float32)
    if result.shape != (23,) or not np.isfinite(result).all():
        raise RuntimeError("selected warmup inverse affine is invalid")
    return result


def prove_warmup_action_equivalence(
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Prove selected-space warmup equals the old plain-native zero warmup."""

    import torch

    from gear_sonic.envs.mjlab.native124_selected_v2_ankle_adaptation import (
        selected21204_raw_hardware_to_sonic_v2_torch,
    )
    from gear_sonic.utils.g1_23dof_safe_target_transform import (
        safe_target_transform_torch,
    )

    raw = torch.as_tensor(
        derive_selected_warmup_raw_hardware(),
        dtype=torch.float32,
        device=device,
    ).reshape(1, 23)
    selected = selected21204_raw_hardware_to_sonic_v2_torch(raw)
    old_safe, old_target = safe_target_transform_torch(torch.zeros_like(raw))
    default = torch.as_tensor(
        SAFE_TARGET_DEFAULT_Q_HARDWARE,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 23)
    candidate_delta = torch.max(torch.abs(selected.candidate_target_hardware - default)).item()
    plain_delta = torch.max(torch.abs(selected.plain_sonic_raw_action_native)).item()
    safe_delta = torch.max(torch.abs(selected.safe_native_action - old_safe)).item()
    target_delta = torch.max(torch.abs(selected.final_target_hardware - old_target)).item()
    passed = max(candidate_delta, plain_delta, safe_delta, target_delta) <= WARMUP_TARGET_ATOL
    if not passed:
        raise RuntimeError("selected warmup does not reproduce old plain-native zero warmup")
    return {
        "selected_raw_action_hardware": raw[0].detach().cpu().tolist(),
        "candidate_target_hardware": selected.candidate_target_hardware[0].detach().cpu().tolist(),
        "old_plain_native_action": torch.zeros_like(raw)[0].detach().cpu().tolist(),
        "old_safe_native_action": old_safe[0].detach().cpu().tolist(),
        "old_final_target_hardware": old_target[0].detach().cpu().tolist(),
        "candidate_default_linf_rad": float(candidate_delta),
        "plain_zero_linf": float(plain_delta),
        "safe_action_linf": float(safe_delta),
        "final_target_linf_rad": float(target_delta),
        "atol": WARMUP_TARGET_ATOL,
        "passed": True,
    }


def _action_record_and_metrics(
    *,
    raw_env: Any,
    composite: Checkpoint21204TeacherComposite,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record baseline-compatible stable action spaces and simulator metrics."""

    import torch

    action_record = composite_action_diagnostic_record(composite)
    action_term = raw_env.action_manager.get_term("joint_pos")
    actual_selected = _tensor_vector(action_term.raw_action, "selected action raw")
    actual_candidate = _tensor_vector(action_term.candidate_target_hardware, "selected candidate target")
    actual_plain = _tensor_vector(action_term.plain_sonic_raw_action_native, "plain SONIC raw")
    actual_safe = _tensor_vector(action_term.safe_native_action, "safe native action")
    actual_target = _tensor_vector(action_term.processed_action, "final hardware target")
    expected = {
        "selected": _finite_vector(composite.teacher_raw_action_hardware, "expected selected raw"),
        "candidate": _finite_vector(composite.teacher_candidate_target_hardware, "expected candidate"),
        "plain": _finite_vector(composite.teacher_action_native, "expected plain SONIC raw"),
        "safe": _finite_vector(composite.teacher_applied_safe_action_native, "expected safe native"),
        "target": _finite_vector(composite.teacher_target_hardware, "expected final target"),
    }
    actual = {
        "selected": actual_selected,
        "candidate": actual_candidate,
        "plain": actual_plain,
        "safe": actual_safe,
        "target": actual_target,
    }
    action_atol = float(contract["action_chain"]["actual_action_atol"])
    target_atol = float(contract["action_chain"]["actual_target_atol"])
    for name in ("selected", "candidate", "plain"):
        if not np.allclose(actual[name], expected[name], rtol=0.0, atol=action_atol):
            raise RuntimeError(f"selected-V2 action-chain {name} mismatch")
    for name in ("safe", "target"):
        if not np.allclose(actual[name], expected[name], rtol=0.0, atol=target_atol):
            raise RuntimeError(f"selected-V2 action-chain {name} mismatch")

    robot = raw_env.scene["robot"]
    measured = _tensor_vector(robot.data.joint_pos, "measured joint position")
    velocity = _tensor_vector(robot.data.joint_vel, "measured joint velocity")
    soft_limits = robot.data.soft_joint_pos_limits
    if (
        not isinstance(soft_limits, torch.Tensor)
        or soft_limits.shape != (1, 23, 2)
        or soft_limits.dtype != torch.float32
        or not bool(torch.isfinite(soft_limits).all().item())
    ):
        raise ValueError("soft limits must be finite float32 [1,23,2]")
    limits = soft_limits.detach().cpu().numpy()[0]
    encoder_bias = _tensor_vector(robot.data.encoder_bias, "encoder bias")
    actuator_target = actual_target - encoder_bias
    target_violation = (expected["target"] < limits[:, 0]) | (expected["target"] > limits[:, 1])
    actuator_violation = (actuator_target < limits[:, 0]) | (actuator_target > limits[:, 1])
    measured_violation = (measured < limits[:, 0]) | (measured > limits[:, 1])
    velocity_limits = np.asarray(
        contract["nominal_gate"]["velocity_limit_hardware_radps"],
        dtype=np.float32,
    )
    if velocity_limits.shape != (23,) or np.any(velocity_limits <= 0.0):
        raise ValueError("velocity limit contract mismatch")
    velocity_ratio = np.abs(velocity) / velocity_limits
    base_height = _scalar(robot.data.root_link_pos_w[:, 2], "base height")
    gravity_z = _scalar(robot.data.projected_gravity_b[:, 2], "projected gravity z")
    base_tilt = math.acos(max(-1.0, min(1.0, -gravity_z)))
    tracking_rmse = float(np.sqrt(np.mean(np.square(expected["target"].astype(np.float64) - measured))))
    metrics = {
        "actual_raw_native_action": actual_plain.tolist(),
        "actual_applied_safe_native_action": actual_safe.tolist(),
        "actual_unbiased_target_hardware": actual_target.tolist(),
        "actual_actuator_target_after_encoder_bias_hardware": actuator_target.tolist(),
        "measured_joint_position_hardware": measured.tolist(),
        "measured_joint_velocity_hardware": velocity.tolist(),
        "action_semantics_match": True,
        "target_soft_limit_violation_count": int(np.count_nonzero(target_violation)),
        "actuator_target_soft_limit_violation_count": int(np.count_nonzero(actuator_violation)),
        "measured_soft_limit_violation_count": int(np.count_nonzero(measured_violation)),
        "joint_velocity_limit_violation_count": int(np.count_nonzero(velocity_ratio > 1.0)),
        "maximum_joint_velocity_ratio": float(np.max(velocity_ratio)),
        "base_height_m": base_height,
        "base_tilt_rad": base_tilt,
        "target_tracking_rmse_rad": tracking_rmse,
    }
    return action_record, metrics


def _termination_names(extras: Mapping[str, Any]) -> list[str]:
    log = extras.get("log")
    if not isinstance(log, Mapping):
        return []
    prefix = "Episode_Termination/"
    result = []
    for key, value in log.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        numeric = float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
        if numeric != 0.0:
            result.append(key[len(prefix) :])
    return sorted(result)


def _append_mismatch(
    mismatches: list[dict[str, Any]],
    *,
    path: str,
    expected: Any,
    actual: Any,
    delta: float | None = None,
    atol: float | None = None,
) -> None:
    mismatch: dict[str, Any] = {"path": path, "expected": expected, "actual": actual}
    if delta is not None:
        mismatch["absolute_delta"] = delta
    if atol is not None:
        mismatch["atol"] = atol
    mismatches.append(_json_safe(mismatch))


def _compare_exact(
    mismatches: list[dict[str, Any]],
    path: str,
    expected: Any,
    actual: Any,
) -> None:
    if actual != expected:
        _append_mismatch(mismatches, path=path, expected=expected, actual=actual)


def _compare_scalar(
    mismatches: list[dict[str, Any]],
    maxima: dict[str, float],
    path: str,
    expected: object,
    actual: object,
    atol: float,
) -> None:
    expected_value = float(expected)
    actual_value = float(actual)
    delta = abs(actual_value - expected_value)
    maxima[path.rsplit(".", 1)[-1]] = max(maxima.get(path.rsplit(".", 1)[-1], 0.0), delta)
    if not math.isfinite(delta) or delta > atol:
        _append_mismatch(
            mismatches,
            path=path,
            expected=expected_value,
            actual=actual_value,
            delta=delta,
            atol=atol,
        )


def _compare_vector(
    mismatches: list[dict[str, Any]],
    maxima: dict[str, float],
    path: str,
    expected: object,
    actual: object,
    atol: float,
) -> None:
    expected_value = _finite_vector(expected, f"baseline {path}")
    actual_value = _finite_vector(actual, f"replay {path}")
    delta = float(np.max(np.abs(actual_value - expected_value)))
    maxima[path.rsplit(".", 1)[-1]] = max(maxima.get(path.rsplit(".", 1)[-1], 0.0), delta)
    if delta > atol:
        _append_mismatch(
            mismatches,
            path=path,
            expected=expected_value.tolist(),
            actual=actual_value.tolist(),
            delta=delta,
            atol=atol,
        )


def _compare_actor_vector(
    mismatches: list[dict[str, Any]],
    maxima: dict[str, float],
    path: str,
    expected: object,
    actual: object,
    atol: float,
) -> None:
    expected_value = _finite_actor_vector(expected, f"baseline {path}")
    actual_value = _finite_actor_vector(actual, f"replay {path}")
    delta = float(np.max(np.abs(actual_value - expected_value)))
    maxima[path.rsplit(".", 1)[-1]] = max(maxima.get(path.rsplit(".", 1)[-1], 0.0), delta)
    if delta > atol:
        _append_mismatch(
            mismatches,
            path=path,
            expected=expected_value.tolist(),
            actual=actual_value.tolist(),
            delta=delta,
            atol=atol,
        )


def compare_replay_to_baseline(
    *,
    replay_records: Sequence[Mapping[str, Any]],
    termination: Mapping[str, Any] | None,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare stable action, state, metric, and termination semantics."""

    baseline_records = baseline.get("records")
    if not isinstance(baseline_records, list):
        raise ValueError("baseline records are unavailable")
    mismatches: list[dict[str, Any]] = []
    maxima: dict[str, float] = {}
    _compare_exact(
        mismatches,
        "records.count",
        len(baseline_records),
        len(replay_records),
    )
    for index, (expected_record, actual_record) in enumerate(zip(baseline_records, replay_records, strict=False)):
        expected_record = _mapping(expected_record, f"baseline record {index}")
        actual_record = _mapping(actual_record, f"replay record {index}")
        for field in ("step", "q9_before", "q10_proof_before", "q9_after", "q10_proof_after"):
            _compare_exact(
                mismatches,
                f"records[{index}].{field}",
                expected_record[field],
                actual_record[field],
            )
        expected_action = _mapping(expected_record["composite_action"], "baseline action")
        actual_action = _mapping(actual_record["composite_action"], "replay action")
        for field in _ACTION_VECTOR_FIELDS:
            _compare_vector(
                mismatches,
                maxima,
                f"records[{index}].composite_action.{field}",
                expected_action[field],
                actual_action[field],
                ACTION_VECTOR_ATOL,
            )
        for field in _ACTION_SCALAR_FIELDS:
            _compare_scalar(
                mismatches,
                maxima,
                f"records[{index}].composite_action.{field}",
                expected_action[field],
                actual_action[field],
                ACTION_VECTOR_ATOL,
            )
        expected_metrics = _mapping(expected_record["metrics"], "baseline metrics")
        actual_metrics = _mapping(actual_record["metrics"], "replay metrics")
        for field in _METRIC_ACTION_VECTOR_FIELDS:
            _compare_vector(
                mismatches,
                maxima,
                f"records[{index}].metrics.{field}",
                expected_metrics[field],
                actual_metrics[field],
                ACTION_VECTOR_ATOL,
            )
        for field, atol in _METRIC_STATE_VECTOR_ATOLS.items():
            _compare_vector(
                mismatches,
                maxima,
                f"records[{index}].metrics.{field}",
                expected_metrics[field],
                actual_metrics[field],
                atol,
            )
        for field in _METRIC_EXACT_FIELDS:
            _compare_exact(
                mismatches,
                f"records[{index}].metrics.{field}",
                expected_metrics[field],
                actual_metrics[field],
            )
        for field in _METRIC_SCALAR_FIELDS:
            _compare_scalar(
                mismatches,
                maxima,
                f"records[{index}].metrics.{field}",
                expected_metrics[field],
                actual_metrics[field],
                METRIC_SCALAR_ATOL,
            )

    baseline_failure = _mapping(baseline.get("failure"), "baseline failure")
    if termination is None:
        _append_mismatch(
            mismatches,
            path="termination",
            expected="ee_body_pos at control transition 47/q9=58",
            actual=None,
        )
    else:
        for field, expected in (
            ("control_transition", EXPECTED_DONE_CONTROL_TRANSITION),
            ("q9_before", EXPECTED_DONE_Q9),
            ("q9_after", EXPECTED_RESET_Q9),
            ("done", 1),
        ):
            _compare_exact(mismatches, f"termination.{field}", expected, termination.get(field))
        _compare_exact(
            mismatches,
            "termination.names",
            ["ee_body_pos"],
            termination.get("names"),
        )
        expected_failure_action = _mapping(
            baseline_failure.get("composite_action"),
            "baseline failure composite action",
        )
        actual_failure_action = _mapping(
            termination.get("composite_action"),
            "replay termination composite action",
        )
        for field in _ACTION_VECTOR_FIELDS:
            _compare_vector(
                mismatches,
                maxima,
                f"termination.composite_action.{field}",
                expected_failure_action[field],
                actual_failure_action[field],
                ACTION_VECTOR_ATOL,
            )
        for field in _ACTION_SCALAR_FIELDS:
            _compare_scalar(
                mismatches,
                maxima,
                f"termination.composite_action.{field}",
                expected_failure_action[field],
                actual_failure_action[field],
                ACTION_VECTOR_ATOL,
            )
        baseline_candidate = _mapping(
            baseline_failure.get("candidate_shadow_record"),
            "baseline failure candidate shadow record",
        )
        _compare_actor_vector(
            mismatches,
            maxima,
            "termination.actor_observation_124",
            baseline_candidate.get("selected_observation_124"),
            termination.get("actor_observation_124"),
            TERMINAL_ACTOR_ATOL,
        )

    semantic_boundary_actual = {
        "completed_transitions": len(replay_records),
        "done_control_transition": None if termination is None else termination.get("control_transition"),
        "done_q9": None if termination is None else termination.get("q9_before"),
        "reset_q9": None if termination is None else termination.get("q9_after"),
        "done": None if termination is None else termination.get("done"),
        "termination_names": None if termination is None else termination.get("names"),
    }
    semantic_boundary_expected = {
        "completed_transitions": EXPECTED_COMPLETED_TRANSITIONS,
        "done_control_transition": EXPECTED_DONE_CONTROL_TRANSITION,
        "done_q9": EXPECTED_DONE_Q9,
        "reset_q9": EXPECTED_RESET_Q9,
        "done": 1,
        "termination_names": ["ee_body_pos"],
    }

    return {
        "passed": not mismatches,
        "parity_characterization": NUMERICAL_PARITY_CHARACTERIZATION,
        "semantic_boundary": {
            "exact_match": semantic_boundary_actual == semantic_boundary_expected,
            "expected": semantic_boundary_expected,
            "actual": semantic_boundary_actual,
        },
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "maximum_absolute_deltas_by_field": maxima,
        "tolerances": {
            "action_vector_and_projection": ACTION_VECTOR_ATOL,
            "measured_joint_position": MEASURED_POSITION_ATOL,
            "measured_joint_velocity": MEASURED_VELOCITY_ATOL,
            "terminal_actor_observation": TERMINAL_ACTOR_ATOL,
            "metric_scalar": METRIC_SCALAR_ATOL,
            "rtol": 0.0,
        },
        "tolerance_calibration": NUMERICAL_TOLERANCE_CALIBRATION,
        "excluded_unstable_fields": [
            "teacher_join_inference_duration_ms",
            "timestamps",
            "serialized_json_bytes",
        ],
    }


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"record_count": 0}
    metrics = [record["metrics"] for record in records]
    actions = [record["composite_action"] for record in records]
    return {
        "record_count": len(records),
        "q9_first": int(records[0]["q9_before"]),
        "q9_last_before": int(records[-1]["q9_before"]),
        "q9_last_after": int(records[-1]["q9_after"]),
        "minimum_base_height_m": min(float(metric["base_height_m"]) for metric in metrics),
        "maximum_base_tilt_rad": max(float(metric["base_tilt_rad"]) for metric in metrics),
        "maximum_tracking_rmse_rad": max(float(metric["target_tracking_rmse_rad"]) for metric in metrics),
        "maximum_projection_linf_rad": max(float(action["projection_linf_rad"]) for action in actions),
    }


def run_no_update_causal_parity(
    *,
    repository_root: str | Path,
    seed: int = SEED,
    device: str = DEVICE,
) -> dict[str, Any]:
    """Run exact DadDance replay until first done or 500 control actions."""

    if seed != SEED or device != DEVICE:
        raise ValueError("frozen parity requires seed=20260805 and device=cuda:0")
    root = Path(repository_root).resolve()
    baseline = load_durable_baseline(root)
    contract = load_composite_mjlab_contract(root)
    motion_path = (root / DAD_DANCE_RELATIVE_PATH).resolve(strict=True)
    if _sha256(motion_path) != DAD_DANCE_SHA256:
        raise ValueError("DadDance motion hash drift")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    import torch

    from gear_sonic.envs.mjlab.native124_selected_v2_ankle_adaptation import (
        Selected21204HardwareToSonicV2JointPositionAction,
    )
    from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
        Native124SelectedV2CausalAdaptationWrapper,
        Native124SelectedV2CausalMotionCommand,
        make_native124_selected_v2_causal_adaptation_env_cfg,
        prime_native124_selected_v2_causal_adaptation_environment,
    )

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    configure_torch_backends()
    torch.cuda.set_device(0)

    cfg = make_native124_selected_v2_causal_adaptation_env_cfg(
        motion_file=motion_path,
        num_envs=1,
        play=True,
    )
    cfg.seed = seed
    binding = load_checkpoint21204_binding(root)
    policy = Native124Checkpoint21204Policy(binding)
    warmup_proof = prove_warmup_action_equivalence(device=device)
    warmup_action = torch.as_tensor(
        warmup_proof["selected_raw_action_hardware"],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 23)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    records: list[dict[str, Any]] = []
    warmup_records: list[dict[str, Any]] = []
    termination: dict[str, Any] | None = None
    try:
        wrapped = Native124SelectedV2CausalAdaptationWrapper(env, clip_actions=None)
        prime = prime_native124_selected_v2_causal_adaptation_environment(wrapped)
        command = env.command_manager.get_term("motion")
        action_term = env.action_manager.get_term("joint_pos")
        if type(command) is not Native124SelectedV2CausalMotionCommand:
            raise RuntimeError("parity replay command type drift")
        if type(action_term) is not Selected21204HardwareToSonicV2JointPositionAction:
            raise RuntimeError("parity replay action term type drift")
        if _q9(command) != 9:
            raise RuntimeError("parity replay did not prime at q9=9")
        observations = wrapped.get_observations()

        for warmup_index in range(WARMUP_STEPS):
            q9_before = _q9(command)
            observations, _, dones, _ = wrapped.step(warmup_action)
            q9_after = _q9(command)
            if int(dones[0].detach().cpu().item()) != 0 or q9_after != q9_before + 1:
                raise RuntimeError("parity warmup terminated or changed q9 discontinuously")
            actual_target = _tensor_vector(action_term.processed_action, "actual warmup target")
            old_target = _finite_vector(warmup_proof["old_final_target_hardware"], "old warmup target")
            target_delta = float(np.max(np.abs(actual_target - old_target)))
            if target_delta > WARMUP_TARGET_ATOL:
                raise RuntimeError("actual selected warmup target differs from old warmup")
            warmup_records.append(
                {
                    "index": warmup_index,
                    "q9_before": q9_before,
                    "q9_after": q9_after,
                    "actual_final_target_hardware": actual_target.tolist(),
                    "old_final_target_linf_rad": target_delta,
                }
            )

        if _q9(command) != 11:
            raise RuntimeError("two parity warmups did not end at q9=11")
        for control_transition in range(MAX_CONTROL_TRANSITIONS):
            q9_before = _q9(command)
            actor = observations["actor"]
            if actor.shape != (1, 124) or actor.dtype != torch.float32:
                raise ValueError("causal wrapper actor must be float32 [1,124]")
            actor_numpy = actor.detach().to(device="cpu").contiguous().numpy().copy()
            selected_raw = policy.run(actor_numpy)
            composite = compose_checkpoint21204_teacher_action(
                selected_raw,
                repository_root=root,
            )
            action_record = composite_action_diagnostic_record(composite)
            action = torch.as_tensor(selected_raw, dtype=torch.float32, device=device).reshape(1, 23)
            observations, _, dones, extras = wrapped.step(action)
            done = int(dones[0].detach().cpu().item())
            q9_after = _q9(command)
            if done:
                termination = {
                    "control_transition": control_transition,
                    "q9_before": q9_before,
                    "q9_after": q9_after,
                    "done": done,
                    "names": _termination_names(extras),
                    "composite_action": action_record,
                    "actor_observation_124": actor_numpy[0].tolist(),
                    "actor_observation_sha256": hashlib.sha256(actor_numpy.tobytes()).hexdigest(),
                    "extras_log": _json_safe(extras.get("log")),
                }
                break
            if q9_after != q9_before + 1:
                raise RuntimeError("parity control q9 discontinuity")
            actual_action_record, metrics = _action_record_and_metrics(
                raw_env=env,
                composite=composite,
                contract=contract,
            )
            records.append(
                {
                    "step": control_transition,
                    "q9_before": q9_before,
                    "q10_proof_before": q9_before + 1,
                    "q9_after": q9_after,
                    "q10_proof_after": q9_after + 1,
                    "actor_observation_sha256": hashlib.sha256(actor_numpy.tobytes()).hexdigest(),
                    "composite_action": actual_action_record,
                    "metrics": metrics,
                }
            )
        comparison = compare_replay_to_baseline(
            replay_records=records,
            termination=termination,
            baseline=baseline,
        )
        summary = _summary(records)
        baseline_summary = _mapping(baseline["summary"], "baseline summary")
        summary_deltas = {}
        for field in (
            "minimum_base_height_m",
            "maximum_base_tilt_rad",
            "maximum_tracking_rmse_rad",
            "maximum_projection_linf_rad",
        ):
            summary_deltas[field] = abs(float(summary[field]) - float(baseline_summary[field]))
        passed = bool(comparison["passed"])
        return {
            "schema_version": 1,
            "kind": PARITY_KIND,
            "passed": passed,
            "verdict": "bounded_cuda_warp_numerical_parity" if passed else "causal_seam_divergence",
            "parity_characterization": NUMERICAL_PARITY_CHARACTERIZATION,
            "baseline": {
                "relative_path": BASELINE_RELATIVE_PATH.as_posix(),
                "sha256": BASELINE_SHA256,
                "stable_fields_only": True,
            },
            "policy": {
                "kind": "frozen_iteration_21204_cpu_onnx",
                "relative_path": binding.onnx_path.relative_to(root).as_posix(),
                "sha256": ONNX_SHA256,
                "provider": "CPUExecutionProvider",
                "updates_performed": 0,
            },
            "motion": {
                "relative_path": DAD_DANCE_RELATIVE_PATH.as_posix(),
                "sha256": DAD_DANCE_SHA256,
                "frame_count": DAD_DANCE_FRAME_COUNT,
            },
            "environment": {
                "seed": seed,
                "device": device,
                "num_envs": 1,
                "play": True,
                "sampling_mode": "start",
                "wrapper_action_clip": None,
                "safe_target_v2_application_count": 1,
            },
            "prime": prime,
            "warmup_equivalence": warmup_proof,
            "warmup": warmup_records,
            "records": records,
            "termination": termination,
            "summary": summary,
            "baseline_summary_absolute_deltas": summary_deltas,
            "comparison": comparison,
            "training_performed": False,
            "hardware_or_network_commands_performed": False,
            "deployment_authorized": False,
        }
    finally:
        env.close()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_report_new(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> Path:
    """Persist once under artifacts/g1_true23; never overwrite evidence."""

    root = Path(repository_root).resolve()
    evidence_root = (root / "artifacts/g1_true23").resolve(strict=True)
    output = Path(path)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    try:
        output.relative_to(evidence_root)
    except ValueError as error:
        raise ValueError("parity report output must stay under artifacts/g1_true23") from error
    if not output.parent.is_dir():
        raise ValueError("parity report parent directory must already exist")
    with output.open("xb") as stream:
        stream.write(canonical_json_bytes(report))
    return output


def failure_report(error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": PARITY_KIND,
        "passed": False,
        "verdict": "causal_seam_runtime_error",
        "error": f"{type(error).__name__}: {error}",
        "training_performed": False,
        "hardware_or_network_commands_performed": False,
        "deployment_authorized": False,
    }
