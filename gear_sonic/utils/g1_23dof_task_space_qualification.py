"""Aggregate held-out task-space retargeting qualification results.

This module deliberately does not run retargeting.  It consumes fresh per-clip
``RetargetResult.summary()``-style records plus orchestration metadata and
produces the one report that may authorize fixed-horizon expert collection
(Step 1B).  It never authorizes training, deployment, or an expert gate.

Status records use this explicit shape::

    {"completed": True, "fresh": True, "ok": True, "skipped": False}

Metadata records must contain ``category``.  ``independence_key`` is preferred;
``source_path`` and then the clip ID are conservative fallbacks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import Any

QUALIFICATION_SCHEMA = "g1_true23_task_space_heldout_qualification_v1"
QUALIFICATION_CATEGORIES = (
    "idle",
    "walk",
    "turn",
    "crouch",
    "reach_lift",
    "fast",
)
CATEGORY_MIN_COST_IMPROVEMENT = {
    "idle": 0.0,
    "walk": 0.05,
    "turn": 0.05,
    "crouch": 0.05,
    "reach_lift": 0.10,
    "fast": 0.10,
}

MIN_CLIP_FRAMES = 100
MIN_CATEGORY_CLIPS = 2
MIN_CATEGORY_FRAMES = 500
MIN_VALID_FRAME_FRACTION = 0.95
MAX_FOOT_ERROR_M = 0.005
MAX_COM_REGRESSION_M = 0.001
MAX_HAND_REGRESSION_M = 0.01
MIN_AGGREGATE_COST_IMPROVEMENT = 0.10
MIN_BILATERAL_HAND_IMPROVEMENT = 0.20
MAX_BILATERAL_HAND_ERROR_M = 0.08
MAX_QUALIFICATION_TIME_SCALE = 1.25

_EPSILON = 1.0e-12


def _require_mapping(
    value: object,
    path: str,
    failures: list[str],
) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        failures.append(f"missing or invalid {path}: expected object")
        return None
    return value


def _require_number(
    mapping: Mapping[str, object] | None,
    key: str,
    path: str,
    failures: list[str],
) -> float | None:
    if mapping is None or key not in mapping:
        failures.append(f"missing {path}.{key}")
        return None
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, Real):
        failures.append(f"invalid {path}.{key}: expected finite number")
        return None
    result = float(value)
    if not math.isfinite(result):
        failures.append(f"invalid {path}.{key}: expected finite number")
        return None
    return result


def _require_nonnegative_integer(
    mapping: Mapping[str, object] | None,
    key: str,
    path: str,
    failures: list[str],
) -> int | None:
    if mapping is None or key not in mapping:
        failures.append(f"missing {path}.{key}")
        return None
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        failures.append(f"invalid {path}.{key}: expected nonnegative integer")
        return None
    return int(value)


def _require_bool(
    mapping: Mapping[str, object] | None,
    key: str,
    path: str,
    failures: list[str],
) -> bool | None:
    if mapping is None or key not in mapping:
        failures.append(f"missing {path}.{key}")
        return None
    value = mapping[key]
    if not isinstance(value, bool):
        failures.append(f"invalid {path}.{key}: expected boolean")
        return None
    return value


def _task_metric(
    per_task: Mapping[str, object] | None,
    task_name: str,
    metric_name: str,
    failures: list[str],
) -> float | None:
    task_path = f"summary.per_task_error.{task_name}"
    task = _require_mapping(
        per_task.get(task_name) if per_task is not None else None,
        task_path,
        failures,
    )
    return _require_number(task, metric_name, task_path, failures)


def _status_result(status: object) -> tuple[bool, list[str]]:
    failures: list[str] = []
    record = _require_mapping(status, "status", failures)
    completed = _require_bool(record, "completed", "status", failures)
    fresh = _require_bool(record, "fresh", "status", failures)
    ok = _require_bool(record, "ok", "status", failures)
    skipped = _require_bool(record, "skipped", "status", failures)
    if completed is False:
        failures.append("status.completed must be true")
    if fresh is False:
        failures.append("status.fresh must be true")
    if ok is False:
        failures.append("status.ok must be true")
    if skipped is True:
        failures.append("status.skipped must be false for first qualification")
    return not failures, failures


def _safe_improvement(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0.0:
        return None
    return (before - after) / before


def _evaluate_clip(summary: object) -> dict[str, Any]:
    failures: list[str] = []
    record = _require_mapping(summary, "summary", failures)

    frame_count = _require_nonnegative_integer(
        record, "frame_count", "summary", failures
    )
    gate_passed = _require_bool(
        record, "kinematic_gate_passed", "summary", failures
    )
    valid_fraction = _require_number(
        record, "expert_valid_frame_fraction", "summary", failures
    )
    cost_before = _require_number(
        record, "weighted_task_error_before_mean", "summary", failures
    )
    cost_after = _require_number(
        record, "weighted_task_error_after_mean", "summary", failures
    )
    safe_envelope_hits = _require_nonnegative_integer(
        record, "position_limit_hit_count", "summary", failures
    )
    hard_limit_violations = _require_nonnegative_integer(
        record, "hard_position_limit_violation_count", "summary", failures
    )
    relaxation_count = _require_nonnegative_integer(
        record,
        "trajectory_constraint_relaxation_count",
        "summary",
        failures,
    )
    action_gt_ten = _require_number(
        record, "native_action_abs_gt_ten_fraction", "summary", failures
    )
    time_scale: float | None = 1.0
    if record is not None and "retiming" in record:
        retiming = _require_mapping(
            record.get("retiming"), "summary.retiming", failures
        )
        time_scale = _require_number(
            retiming, "time_scale", "summary.retiming", failures
        )

    constraints = _require_mapping(
        record.get("constraints") if record is not None else None,
        "summary.constraints",
        failures,
    )
    velocity_limit = _require_number(
        constraints, "max_velocity_rad_s", "summary.constraints", failures
    )
    acceleration_limit = _require_number(
        constraints, "max_acceleration_rad_s2", "summary.constraints", failures
    )
    velocity_measured = _require_number(
        constraints,
        "measured_velocity_abs_max_rad_s",
        "summary.constraints",
        failures,
    )
    acceleration_measured = _require_number(
        constraints,
        "measured_acceleration_abs_max_rad_s2",
        "summary.constraints",
        failures,
    )

    per_task = _require_mapping(
        record.get("per_task_error") if record is not None else None,
        "summary.per_task_error",
        failures,
    )
    left_foot_max = _task_metric(
        per_task,
        "left_foot",
        "position_error_after_max_m",
        failures,
    )
    right_foot_max = _task_metric(
        per_task,
        "right_foot",
        "position_error_after_max_m",
        failures,
    )
    com_before = _task_metric(
        per_task,
        "whole_robot_com",
        "position_error_before_mean_m",
        failures,
    )
    com_after = _task_metric(
        per_task,
        "whole_robot_com",
        "position_error_after_mean_m",
        failures,
    )
    left_hand_before = _task_metric(
        per_task,
        "left_hand",
        "position_error_before_mean_m",
        failures,
    )
    left_hand_after = _task_metric(
        per_task,
        "left_hand",
        "position_error_after_mean_m",
        failures,
    )
    right_hand_before = _task_metric(
        per_task,
        "right_hand",
        "position_error_before_mean_m",
        failures,
    )
    right_hand_after = _task_metric(
        per_task,
        "right_hand",
        "position_error_after_mean_m",
        failures,
    )

    if frame_count is not None and frame_count < MIN_CLIP_FRAMES:
        failures.append(f"frame_count {frame_count} is below {MIN_CLIP_FRAMES}")
    if gate_passed is False:
        failures.append("kinematic_gate_passed is not true")
    if valid_fraction is not None and valid_fraction < MIN_VALID_FRAME_FRACTION:
        failures.append(
            f"expert_valid_frame_fraction {valid_fraction:.6f} is below "
            f"{MIN_VALID_FRAME_FRACTION:.6f}"
        )
    if cost_before is not None and cost_after is not None and cost_after >= cost_before:
        failures.append("weighted task error did not improve")
    if hard_limit_violations is not None and hard_limit_violations != 0:
        failures.append(
            "hard_position_limit_violation_count is "
            f"{hard_limit_violations}, expected zero"
        )
    if relaxation_count is not None and relaxation_count != 0:
        failures.append(
            "trajectory_constraint_relaxation_count is "
            f"{relaxation_count}, expected zero"
        )
    if action_gt_ten is not None and action_gt_ten != 0.0:
        failures.append(
            f"native_action_abs_gt_ten_fraction is {action_gt_ten}, expected zero"
        )
    if time_scale is not None:
        if time_scale < 1.0:
            failures.append("retiming time_scale must be at least one")
        elif time_scale > MAX_QUALIFICATION_TIME_SCALE:
            failures.append(
                f"retiming time_scale {time_scale:.6g} exceeds qualification "
                f"limit {MAX_QUALIFICATION_TIME_SCALE:.6g}"
            )
    if (
        velocity_measured is not None
        and velocity_limit is not None
        and velocity_measured > velocity_limit + _EPSILON
    ):
        failures.append(
            f"measured velocity {velocity_measured:.9g} exceeds configured "
            f"{velocity_limit:.9g} rad/s"
        )
    if (
        acceleration_measured is not None
        and acceleration_limit is not None
        and acceleration_measured > acceleration_limit + _EPSILON
    ):
        failures.append(
            f"measured acceleration {acceleration_measured:.9g} exceeds configured "
            f"{acceleration_limit:.9g} rad/s^2"
        )
    for side, foot_max in (("left", left_foot_max), ("right", right_foot_max)):
        if foot_max is not None and foot_max > MAX_FOOT_ERROR_M + _EPSILON:
            failures.append(
                f"{side} foot max error {foot_max:.9g} m exceeds "
                f"{MAX_FOOT_ERROR_M:.9g} m"
            )
    if (
        com_before is not None
        and com_after is not None
        and com_after > com_before + MAX_COM_REGRESSION_M + _EPSILON
    ):
        failures.append(
            f"COM regressed by {com_after - com_before:.9g} m; "
            f"limit is {MAX_COM_REGRESSION_M:.9g} m"
        )
    for side, before, after in (
        ("left", left_hand_before, left_hand_after),
        ("right", right_hand_before, right_hand_after),
    ):
        if (
            before is not None
            and after is not None
            and after > before + MAX_HAND_REGRESSION_M + _EPSILON
        ):
            failures.append(
                f"{side} hand regressed by {after - before:.9g} m; "
                f"limit is {MAX_HAND_REGRESSION_M:.9g} m"
            )

    bilateral_before = (
        None
        if left_hand_before is None or right_hand_before is None
        else (left_hand_before + right_hand_before) / 2.0
    )
    bilateral_after = (
        None
        if left_hand_after is None or right_hand_after is None
        else (left_hand_after + right_hand_after) / 2.0
    )
    foot_max_values = [
        value for value in (left_foot_max, right_foot_max) if value is not None
    ]

    return {
        "hard_gate_passed": not failures,
        "hard_gate_failures": failures,
        "frame_count": frame_count,
        "metrics": {
            "expert_valid_frame_fraction": valid_fraction,
            "weighted_task_error_before_mean": cost_before,
            "weighted_task_error_after_mean": cost_after,
            "cost_improvement": _safe_improvement(cost_before, cost_after),
            "bilateral_hand_error_before_mean_m": bilateral_before,
            "bilateral_hand_error_after_mean_m": bilateral_after,
            "bilateral_hand_improvement": _safe_improvement(
                bilateral_before, bilateral_after
            ),
            "whole_robot_com_error_before_mean_m": com_before,
            "whole_robot_com_error_after_mean_m": com_after,
            "stance_foot_error_after_max_m": (
                max(foot_max_values) if len(foot_max_values) == 2 else None
            ),
            "safe_envelope_hit_count": safe_envelope_hits,
            "hard_position_limit_violation_count": hard_limit_violations,
            "trajectory_constraint_relaxation_count": relaxation_count,
            "native_action_abs_gt_ten_fraction": action_gt_ten,
            "retiming_time_scale": time_scale,
            "configured_velocity_max_rad_s": velocity_limit,
            "measured_velocity_abs_max_rad_s": velocity_measured,
            "configured_acceleration_max_rad_s2": acceleration_limit,
            "measured_acceleration_abs_max_rad_s2": acceleration_measured,
        },
    }


def _weighted_mean(clips: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for clip in clips:
        frame_count = clip.get("frame_count")
        metrics = clip.get("metrics")
        value = metrics.get(metric) if isinstance(metrics, Mapping) else None
        if not isinstance(frame_count, int) or not isinstance(value, float):
            return None
        numerator += frame_count * value
        denominator += frame_count
    if denominator == 0:
        return None
    return numerator / denominator


def _aggregate_metrics(clips: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    frame_count = sum(
        value
        for clip in clips
        if isinstance((value := clip.get("frame_count")), int)
    )
    valid_fraction = _weighted_mean(clips, "expert_valid_frame_fraction")
    cost_before = _weighted_mean(clips, "weighted_task_error_before_mean")
    cost_after = _weighted_mean(clips, "weighted_task_error_after_mean")
    hand_before = _weighted_mean(clips, "bilateral_hand_error_before_mean_m")
    hand_after = _weighted_mean(clips, "bilateral_hand_error_after_mean_m")
    com_before = _weighted_mean(clips, "whole_robot_com_error_before_mean_m")
    com_after = _weighted_mean(clips, "whole_robot_com_error_after_mean_m")
    foot_values = []
    for clip in clips:
        metrics = clip.get("metrics")
        value = (
            metrics.get("stance_foot_error_after_max_m")
            if isinstance(metrics, Mapping)
            else None
        )
        if isinstance(value, float):
            foot_values.append(value)
    return {
        "clip_count": len(clips),
        "frame_count": frame_count,
        "expert_valid_frame_fraction": valid_fraction,
        "weighted_task_error_before_mean": cost_before,
        "weighted_task_error_after_mean": cost_after,
        "cost_improvement": _safe_improvement(cost_before, cost_after),
        "bilateral_hand_error_before_mean_m": hand_before,
        "bilateral_hand_error_after_mean_m": hand_after,
        "bilateral_hand_improvement": _safe_improvement(hand_before, hand_after),
        "whole_robot_com_error_before_mean_m": com_before,
        "whole_robot_com_error_after_mean_m": com_after,
        "stance_foot_error_after_max_m": (
            max(foot_values) if len(foot_values) == len(clips) and clips else None
        ),
    }


def _category_report(
    category: str,
    clips: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    metrics = _aggregate_metrics(clips)
    failures: list[str] = []
    independence_count = len(
        {
            clip["independence_key"]
            for clip in clips
            if isinstance(clip.get("independence_key"), str)
        }
    )
    frame_count = metrics["frame_count"]
    valid_fraction = metrics["expert_valid_frame_fraction"]
    cost_improvement = metrics["cost_improvement"]
    hand_before = metrics["bilateral_hand_error_before_mean_m"]
    hand_after = metrics["bilateral_hand_error_after_mean_m"]
    hand_improvement = metrics["bilateral_hand_improvement"]
    com_before = metrics["whole_robot_com_error_before_mean_m"]
    com_after = metrics["whole_robot_com_error_after_mean_m"]

    if len(clips) < MIN_CATEGORY_CLIPS:
        failures.append(
            f"clip_count {len(clips)} is below {MIN_CATEGORY_CLIPS}"
        )
    if independence_count < MIN_CATEGORY_CLIPS:
        failures.append(
            f"independent_clip_count {independence_count} is below "
            f"{MIN_CATEGORY_CLIPS}"
        )
    if not isinstance(frame_count, int) or frame_count < MIN_CATEGORY_FRAMES:
        failures.append(
            f"frame_count {frame_count} is below {MIN_CATEGORY_FRAMES}"
        )
    hard_failure_count = sum(not bool(clip["hard_gate_passed"]) for clip in clips)
    if hard_failure_count:
        failures.append(f"{hard_failure_count} clip(s) failed hard gates")
    if valid_fraction is None:
        failures.append("expert_valid_frame_fraction unavailable")
    elif valid_fraction < MIN_VALID_FRAME_FRACTION:
        failures.append(
            f"expert_valid_frame_fraction {valid_fraction:.6f} is below "
            f"{MIN_VALID_FRAME_FRACTION:.6f}"
        )
    threshold = CATEGORY_MIN_COST_IMPROVEMENT[category]
    if cost_improvement is None:
        failures.append("cost_improvement unavailable")
    elif cost_improvement + _EPSILON < threshold:
        failures.append(
            f"cost_improvement {cost_improvement:.6f} is below {threshold:.6f}"
        )

    if category == "reach_lift":
        if hand_improvement is None:
            failures.append("bilateral_hand_improvement unavailable")
        elif hand_improvement + _EPSILON < MIN_BILATERAL_HAND_IMPROVEMENT:
            failures.append(
                f"bilateral_hand_improvement {hand_improvement:.6f} is below "
                f"{MIN_BILATERAL_HAND_IMPROVEMENT:.6f}"
            )
        if hand_after is None:
            failures.append("bilateral_hand_error_after_mean_m unavailable")
        elif hand_after > MAX_BILATERAL_HAND_ERROR_M + _EPSILON:
            failures.append(
                f"bilateral_hand_error_after_mean_m {hand_after:.6f} exceeds "
                f"{MAX_BILATERAL_HAND_ERROR_M:.6f}"
            )
    else:
        if hand_before is None or hand_after is None:
            failures.append("pooled bilateral hand metrics unavailable")
        elif hand_after > hand_before + _EPSILON:
            failures.append("pooled bilateral hand error regressed")
        if com_before is None or com_after is None:
            failures.append("pooled COM metrics unavailable")
        elif com_after > com_before + _EPSILON:
            failures.append("pooled COM error regressed")

    return {
        "gate_passed": not failures,
        "gate_failures": failures,
        "independent_clip_count": independence_count,
        "thresholds": {
            "min_clip_count": MIN_CATEGORY_CLIPS,
            "min_frame_count": MIN_CATEGORY_FRAMES,
            "min_valid_frame_fraction": MIN_VALID_FRAME_FRACTION,
            "min_cost_improvement": threshold,
            "min_bilateral_hand_improvement": (
                MIN_BILATERAL_HAND_IMPROVEMENT
                if category == "reach_lift"
                else None
            ),
            "max_bilateral_hand_error_m": (
                MAX_BILATERAL_HAND_ERROR_M
                if category == "reach_lift"
                else None
            ),
            "require_pooled_hand_and_com_non_regression": category != "reach_lift",
        },
        "metrics": metrics,
    }


def build_task_space_qualification_report(
    summaries_by_clip: Mapping[str, Mapping[str, object]],
    metadata_by_clip: Mapping[str, Mapping[str, object]],
    statuses_by_clip: Mapping[str, Mapping[str, object]],
    requested_clip_ids: Sequence[str],
    declared_categories: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build deterministic, JSON-serializable held-out qualification report.

    All aggregation uses frame-weighted means.  Only unique requested clip IDs
    participate.  Duplicate IDs are reported as qualification failures rather
    than silently changing their statistical weight.
    """

    if declared_categories is None:
        selected_categories = QUALIFICATION_CATEGORIES
    else:
        if isinstance(declared_categories, str):
            raise ValueError("declared_categories must be a nonempty sequence")
        selected_categories = tuple(declared_categories)
        if not selected_categories:
            raise ValueError("declared_categories must be a nonempty sequence")
        if any(not isinstance(category, str) for category in selected_categories):
            raise ValueError("declared_categories must contain category strings")
        if len(set(selected_categories)) != len(selected_categories):
            raise ValueError("declared_categories must not contain duplicates")
        unknown_categories = [
            category
            for category in selected_categories
            if category not in QUALIFICATION_CATEGORIES
        ]
        if unknown_categories:
            raise ValueError(
                "declared_categories contains unknown categories: "
                + ", ".join(repr(category) for category in unknown_categories)
            )

    requested = list(requested_clip_ids)
    unique_requested = list(dict.fromkeys(requested))
    qualification_failures: list[str] = []
    if len(unique_requested) != len(requested):
        qualification_failures.append("requested_clip_ids contains duplicates")
    if not unique_requested:
        qualification_failures.append("requested_clip_ids is empty")

    clip_reports: dict[str, dict[str, object]] = {}
    clips_by_category: dict[str, list[Mapping[str, Any]]] = {
        category: [] for category in selected_categories
    }
    all_clip_records: list[Mapping[str, Any]] = []
    requested_state_passed = True

    for clip_id in unique_requested:
        state_passed, state_failures = _status_result(statuses_by_clip.get(clip_id))
        requested_state_passed = requested_state_passed and state_passed

        metadata_failures: list[str] = []
        metadata = _require_mapping(
            metadata_by_clip.get(clip_id), "metadata", metadata_failures
        )
        category_value = metadata.get("category") if metadata is not None else None
        category = category_value if isinstance(category_value, str) else None
        if category is None:
            metadata_failures.append("missing or invalid metadata.category")
        elif category not in QUALIFICATION_CATEGORIES:
            metadata_failures.append(f"unknown metadata.category {category!r}")
        elif category not in selected_categories:
            metadata_failures.append(
                f"metadata.category {category!r} is outside declared_categories"
            )
            qualification_failures.append(
                f"requested clip {clip_id!r} has category outside declared_categories"
            )

        independence_value: object = clip_id
        independence_basis = "clip_id"
        if metadata is not None and "independence_key" in metadata:
            independence_value = metadata["independence_key"]
            independence_basis = "independence_key"
        elif metadata is not None and "source_path" in metadata:
            independence_value = metadata["source_path"]
            independence_basis = "source_path"
        if not isinstance(independence_value, str) or not independence_value.strip():
            metadata_failures.append(
                f"invalid metadata.{independence_basis}: expected nonempty string"
            )
            independence_key = None
        else:
            independence_key = independence_value

        clip = _evaluate_clip(summaries_by_clip.get(clip_id))
        clip["clip_id"] = clip_id
        clip["category"] = category
        clip["independence_key"] = independence_key
        all_hard_failures = [*metadata_failures, *clip["hard_gate_failures"]]
        clip["hard_gate_failures"] = all_hard_failures
        clip["hard_gate_passed"] = not all_hard_failures
        clip["status_gate_passed"] = state_passed
        clip["status_gate_failures"] = state_failures
        clip["independence_basis"] = independence_basis
        clip_reports[clip_id] = clip
        if category in clips_by_category:
            all_clip_records.append(clip)
        if category in clips_by_category:
            clips_by_category[category].append(clip)

    categories = {
        category: _category_report(category, clips_by_category[category])
        for category in selected_categories
    }
    aggregate_metrics = _aggregate_metrics(all_clip_records)
    aggregate_failures: list[str] = []
    if not requested_state_passed:
        aggregate_failures.append(
            "not all requested clips completed fresh, successful, and unskipped"
        )
    failed_categories = [
        category
        for category, report in categories.items()
        if not report["gate_passed"]
    ]
    if failed_categories:
        aggregate_failures.append(
            "category gates failed: " + ", ".join(failed_categories)
        )
    hard_violation_count = sum(
        len(clip["hard_gate_failures"]) for clip in all_clip_records
    )
    if hard_violation_count:
        aggregate_failures.append(
            f"per-clip hard violation count is {hard_violation_count}, expected zero"
        )

    valid_fraction = aggregate_metrics["expert_valid_frame_fraction"]
    if valid_fraction is None:
        aggregate_failures.append("aggregate expert_valid_frame_fraction unavailable")
    elif valid_fraction < MIN_VALID_FRAME_FRACTION:
        aggregate_failures.append(
            f"aggregate expert_valid_frame_fraction {valid_fraction:.6f} is below "
            f"{MIN_VALID_FRAME_FRACTION:.6f}"
        )
    cost_improvement = aggregate_metrics["cost_improvement"]
    if cost_improvement is None:
        aggregate_failures.append("aggregate cost_improvement unavailable")
    elif cost_improvement + _EPSILON < MIN_AGGREGATE_COST_IMPROVEMENT:
        aggregate_failures.append(
            f"aggregate cost_improvement {cost_improvement:.6f} is below "
            f"{MIN_AGGREGATE_COST_IMPROVEMENT:.6f}"
        )
    hand_improvement = aggregate_metrics["bilateral_hand_improvement"]
    if hand_improvement is None:
        aggregate_failures.append("aggregate bilateral_hand_improvement unavailable")
    elif hand_improvement + _EPSILON < MIN_BILATERAL_HAND_IMPROVEMENT:
        aggregate_failures.append(
            f"aggregate bilateral_hand_improvement {hand_improvement:.6f} is below "
            f"{MIN_BILATERAL_HAND_IMPROVEMENT:.6f}"
        )
    hand_after = aggregate_metrics["bilateral_hand_error_after_mean_m"]
    if hand_after is None:
        aggregate_failures.append(
            "aggregate bilateral_hand_error_after_mean_m unavailable"
        )
    elif hand_after > MAX_BILATERAL_HAND_ERROR_M + _EPSILON:
        aggregate_failures.append(
            f"aggregate bilateral_hand_error_after_mean_m {hand_after:.6f} exceeds "
            f"{MAX_BILATERAL_HAND_ERROR_M:.6f}"
        )
    aggregate_failures.extend(qualification_failures)

    qualification_gate_passed = not aggregate_failures
    return {
        "schema": QUALIFICATION_SCHEMA,
        "schema_version": 1,
        "deployment_ready": False,
        "expert_gate_passed": False,
        "authorization": (
            "step_1b_fixed_horizon_expert_collection_only"
            if qualification_gate_passed
            else "none"
        ),
        "step_1b_authorized": qualification_gate_passed,
        "qualification_gate_passed": qualification_gate_passed,
        "qualification_gate_failures": aggregate_failures,
        "requested_clip_count": len(requested),
        "unique_requested_clip_count": len(unique_requested),
        "declared_categories": list(selected_categories),
        "requested_state_passed": requested_state_passed,
        "hard_violation_count": hard_violation_count,
        "thresholds": {
            "min_clip_frames": MIN_CLIP_FRAMES,
            "min_category_clips": MIN_CATEGORY_CLIPS,
            "min_category_frames": MIN_CATEGORY_FRAMES,
            "min_valid_frame_fraction": MIN_VALID_FRAME_FRACTION,
            "max_foot_error_m": MAX_FOOT_ERROR_M,
            "max_com_regression_m": MAX_COM_REGRESSION_M,
            "max_hand_regression_m": MAX_HAND_REGRESSION_M,
            "min_aggregate_cost_improvement": MIN_AGGREGATE_COST_IMPROVEMENT,
            "min_bilateral_hand_improvement": MIN_BILATERAL_HAND_IMPROVEMENT,
            "max_bilateral_hand_error_m": MAX_BILATERAL_HAND_ERROR_M,
        },
        "metrics": aggregate_metrics,
        "categories": categories,
        "clips": clip_reports,
    }
