"""Audit frame-level regressions in true-23 task-space retarget artifacts.

This consumes only the persisted ``experts`` and ``reports`` directories from
``retarget_g1_29dof_to_23dof_task_space``.  It deliberately does not authorize
an expert for training or deployment.  Its purpose is to identify every frame
that fails the retargeter's expert-valid rule and attach diagnostic correlates
that can explain clusters: temporal constraint ceilings, contact transitions,
fast desired hand motion, safe-envelope hits, and exhausted solver iterations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np

AUDIT_SCHEMA = "g1_true23_task_space_regression_audit_v2"
AUDIT_SCHEMA_VERSION = 2
CLIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_CONTACT_RADIUS_FRAMES = 2
DEFAULT_CEILING_FRACTION = 0.99
DEFAULT_HIGH_HAND_SPEED_M_S = 1.0
DEFAULT_LONG_RUN_MIN_FRAMES = 10
DEFAULT_WORST_FRAME_COUNT = 50

CONTEXT_CAUSES = (
    "feasible_seed_already_worse",
    "contact_transition_exact",
    "near_contact_transition",
    "velocity_ceiling",
    "acceleration_ceiling",
    "high_desired_hand_speed",
    "safe_envelope_hit",
    "solver_iteration_ceiling",
)


@dataclass(frozen=True)
class AuditThresholds:
    """Thresholds for diagnostic correlates; validity still follows report config."""

    contact_radius_frames: int = DEFAULT_CONTACT_RADIUS_FRAMES
    ceiling_fraction: float = DEFAULT_CEILING_FRACTION
    high_hand_speed_m_s: float = DEFAULT_HIGH_HAND_SPEED_M_S
    long_run_min_frames: int = DEFAULT_LONG_RUN_MIN_FRAMES
    worst_frame_count: int = DEFAULT_WORST_FRAME_COUNT

    def __post_init__(self) -> None:
        if self.contact_radius_frames < 0:
            raise ValueError("contact_radius_frames must be non-negative")
        if not 0.0 < self.ceiling_fraction <= 1.0:
            raise ValueError("ceiling_fraction must be in (0, 1]")
        if self.high_hand_speed_m_s <= 0.0:
            raise ValueError("high_hand_speed_m_s must be positive")
        if self.long_run_min_frames < 1:
            raise ValueError("long_run_min_frames must be positive")
        if self.worst_frame_count < 1:
            raise ValueError("worst_frame_count must be positive")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "contact_radius_frames": self.contact_radius_frames,
            "ceiling_fraction": self.ceiling_fraction,
            "high_hand_speed_m_s": self.high_hand_speed_m_s,
            "long_run_min_frames": self.long_run_min_frames,
            "worst_frame_count": self.worst_frame_count,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: object, *, name: str) -> int:
    numeric = _finite_number(value, name=name)
    result = int(numeric)
    if numeric != result or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _frame_array(
    artifact: Mapping[str, np.ndarray],
    name: str,
    frame_count: int,
    *,
    ndim: int = 1,
) -> np.ndarray:
    if name not in artifact:
        raise ValueError(f"expert artifact missing {name!r}")
    values = np.asarray(artifact[name])
    if values.ndim != ndim or values.shape[0] != frame_count:
        raise ValueError(
            f"{name} must have rank {ndim} and {frame_count} frames, got {values.shape}"
        )
    if values.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be numeric or boolean")
    if values.dtype.kind in "iuf" and not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf")
    return values


def _read_task_names(artifact: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    if "task_names" not in artifact:
        raise ValueError("expert artifact missing 'task_names'")
    values = np.asarray(artifact["task_names"])
    if values.ndim != 1 or values.dtype.kind not in "SU":
        raise ValueError("task_names must be a one-dimensional string array")
    names = tuple(str(value) for value in values.tolist())
    if not names or len(set(names)) != len(names):
        raise ValueError("task_names must be non-empty and unique")
    return names


def _retarget_schema_version(
    artifact: Mapping[str, np.ndarray], report: Mapping[str, Any], *, clip_id: str
) -> int:
    if "schema_version" not in artifact:
        raise ValueError(f"expert artifact {clip_id} missing schema_version")
    values = np.asarray(artifact["schema_version"])
    if values.shape != (1,) or values.dtype.kind not in "iu":
        raise ValueError("schema_version must be one integer")
    artifact_version = int(values[0])
    report_version = _positive_int(
        report.get("schema_version"), name=f"{clip_id}.schema_version"
    )
    if artifact_version != report_version:
        raise ValueError(
            f"{clip_id} artifact schema_version {artifact_version} "
            f"does not match report {report_version}"
        )
    if artifact_version not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"unsupported task-space retarget schema_version {artifact_version}")
    return artifact_version


def _gate_regression(before: np.ndarray, after: np.ndarray, relative_tolerance: float) -> np.ndarray:
    tolerance = relative_tolerance * np.maximum(1.0, np.abs(before))
    return after > before + tolerance


def _gate_excess(before: np.ndarray, after: np.ndarray, relative_tolerance: float) -> np.ndarray:
    scale = np.maximum(1.0, np.abs(before))
    return (after - before - relative_tolerance * scale) / scale


def _contiguous_windows(frames: np.ndarray) -> list[tuple[int, int]]:
    if len(frames) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(frames) > 1) + 1
    chunks = np.split(frames, boundaries)
    return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks]


def _contact_transitions(contact_flags: np.ndarray) -> tuple[np.ndarray, dict[int, list[str]]]:
    changed = contact_flags[1:] != contact_flags[:-1]
    frames = np.flatnonzero(np.any(changed, axis=1)) + 1
    sides: dict[int, list[str]] = {}
    side_names = ("left", "right")
    for frame in frames:
        sides[int(frame)] = [
            side_names[index] for index in np.flatnonzero(changed[frame - 1])
        ]
    return frames.astype(np.int64), sides


def _nearest_distances(frame_count: int, event_frames: np.ndarray) -> np.ndarray:
    if len(event_frames) == 0:
        return np.full(frame_count, -1, dtype=np.int64)
    frames = np.arange(frame_count, dtype=np.int64)[:, None]
    return np.min(np.abs(frames - event_frames[None, :]), axis=1)


def _hand_speeds(
    desired_task_pos_w: np.ndarray,
    task_names: tuple[str, ...],
    fps: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    hand_names = [name for name in ("left_hand", "right_hand") if name in task_names]
    if not hand_names:
        raise ValueError("task_names contains no left_hand or right_hand task")
    by_hand: dict[str, np.ndarray] = {}
    for name in hand_names:
        positions = desired_task_pos_w[:, task_names.index(name), :]
        velocity = np.gradient(positions, 1.0 / fps, axis=0, edge_order=1)
        by_hand[name] = np.linalg.norm(velocity, axis=1)
    maximum = np.max(np.stack(list(by_hand.values()), axis=1), axis=1)
    return maximum, by_hand


def _ratio_or_none(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _percentiles_or_none(values: Iterable[int | float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _association(mask: np.ndarray, invalid: np.ndarray) -> dict[str, int | float | None]:
    present_count = int(np.sum(mask))
    absent_count = int(len(mask) - present_count)
    invalid_present = int(np.sum(mask & invalid))
    invalid_absent = int(np.sum(~mask & invalid))
    present_rate = _ratio_or_none(invalid_present, present_count)
    absent_rate = _ratio_or_none(invalid_absent, absent_count)
    lift = (
        None
        if present_rate is None or absent_rate in (None, 0.0)
        else float(present_rate / absent_rate)
    )
    return {
        "context_frame_count": present_count,
        "invalid_context_frame_count": invalid_present,
        "invalid_rate_when_present": present_rate,
        "invalid_rate_when_absent": absent_rate,
        "invalid_rate_lift": lift,
    }


def _task_deltas(
    artifact: Mapping[str, np.ndarray],
    task_names: tuple[str, ...],
    frame_count: int,
    frame: int,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for task_name in task_names:
        position_before = _frame_array(
            artifact,
            f"task_{task_name}_position_error_before_m",
            frame_count,
        )
        position_after = _frame_array(
            artifact,
            f"task_{task_name}_position_error_after_m",
            frame_count,
        )
        orientation_before = _frame_array(
            artifact,
            f"task_{task_name}_orientation_error_before_rad",
            frame_count,
        )
        orientation_after = _frame_array(
            artifact,
            f"task_{task_name}_orientation_error_after_rad",
            frame_count,
        )
        result[task_name] = {
            "position_before_m": float(position_before[frame]),
            "position_after_m": float(position_after[frame]),
            "position_delta_m": float(position_after[frame] - position_before[frame]),
            "orientation_before_rad": float(orientation_before[frame]),
            "orientation_after_rad": float(orientation_after[frame]),
            "orientation_delta_rad": float(
                orientation_after[frame] - orientation_before[frame]
            ),
        }
    return result


def _clip_id_from_report(report: Mapping[str, Any], report_path: Path) -> str:
    """Read current provenance, with filename fallback for pre-manifest reports."""

    filename_suffix = ".retarget.json"
    filename_clip_id = (
        report_path.name[: -len(filename_suffix)]
        if report_path.name.endswith(filename_suffix)
        else ""
    )
    value = report.get("clip_id", filename_clip_id)
    if not isinstance(value, str) or not CLIP_ID_PATTERN.fullmatch(value):
        raise ValueError(f"invalid clip_id in {report_path}: {value!r}")
    if "clip_id" in report and value != filename_clip_id:
        raise ValueError(
            f"report clip_id {value!r} does not match filename {filename_clip_id!r}"
        )
    return value


def _load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"report must contain a JSON object: {path}")
    return payload


def _cause_counts(cause_masks: Mapping[str, np.ndarray], invalid: np.ndarray) -> dict[str, int]:
    return {name: int(np.sum(mask & invalid)) for name, mask in cause_masks.items()}


def _frame_diagnostic(
    *,
    clip_id: str,
    frame: int,
    fps: float,
    before: np.ndarray,
    after: np.ndarray,
    validity_excesses: Mapping[str, np.ndarray],
    priority_before: list[np.ndarray],
    priority_after: list[np.ndarray],
    feasible_seed: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    iterations: np.ndarray,
    position_hits: np.ndarray,
    nearest_contact: np.ndarray,
    transition_sides: Mapping[int, list[str]],
    max_hand_speed: np.ndarray,
    hand_speeds: Mapping[str, np.ndarray],
    validity_failure_masks: Mapping[str, np.ndarray],
    cause_masks: Mapping[str, np.ndarray],
    artifact: Mapping[str, np.ndarray],
    task_names: tuple[str, ...],
    frame_count: int,
) -> dict[str, Any]:
    max_excess = max(float(values[frame]) for values in validity_excesses.values())
    return {
        "clip_id": clip_id,
        "frame": frame,
        "time_s": float(frame / fps),
        "severity_max_normalized_gate_excess": max_excess,
        "validity_gate_excesses": {
            name: float(values[frame]) for name, values in validity_excesses.items()
        },
        "validity_failures": [
            name for name, mask in validity_failure_masks.items() if bool(mask[frame])
        ],
        "causes": [name for name, mask in cause_masks.items() if bool(mask[frame])],
        "weighted_task_error_before": float(before[frame]),
        "weighted_task_error_after": float(after[frame]),
        "weighted_task_error_delta": float(after[frame] - before[frame]),
        "weighted_task_error_feasible_seed": float(feasible_seed[frame]),
        "protected_priorities": [
            {
                "priority": priority,
                "before": float(priority_before[priority][frame]),
                "after": float(priority_after[priority][frame]),
                "delta": float(
                    priority_after[priority][frame] - priority_before[priority][frame]
                ),
            }
            for priority in range(len(priority_before))
        ],
        "nearest_contact_transition_distance_frames": (
            None if nearest_contact[frame] < 0 else int(nearest_contact[frame])
        ),
        "contact_transition_sides": transition_sides.get(frame, []),
        "trajectory_velocity_abs_max_rad_s": float(velocity[frame]),
        "trajectory_acceleration_abs_max_rad_s2": float(acceleration[frame]),
        "solver_iterations": int(round(float(iterations[frame]))),
        "safe_envelope_hit_count": int(round(float(position_hits[frame]))),
        "desired_hand_speed_max_m_s": float(max_hand_speed[frame]),
        "desired_hand_speed_m_s": {
            name: float(values[frame]) for name, values in hand_speeds.items()
        },
        "task_error_deltas": _task_deltas(
            artifact,
            task_names,
            frame_count,
            frame,
        ),
    }


def analyze_clip(
    report_path: Path,
    expert_path: Path,
    thresholds: AuditThresholds = AuditThresholds(),
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Analyze one persisted report/expert pair.

    Second return value contains frame masks used for frame-weighted aggregation.
    """

    report = _load_report(report_path)
    clip_id = _clip_id_from_report(report, report_path)
    config = report.get("retarget_config")
    if not isinstance(config, dict):
        raise ValueError(f"report {clip_id} missing retarget_config")
    relative_tolerance = _finite_number(
        config.get("priority_relative_tolerance"),
        name=f"{clip_id}.priority_relative_tolerance",
    )
    if relative_tolerance < 0.0:
        raise ValueError("priority_relative_tolerance must be non-negative")
    protected_tiers = _positive_int(
        config.get("protected_priority_tiers"),
        name=f"{clip_id}.protected_priority_tiers",
    )
    max_velocity = _finite_number(
        config.get("max_velocity_rad_s"), name=f"{clip_id}.max_velocity_rad_s"
    )
    max_acceleration = _finite_number(
        config.get("max_acceleration_rad_s2"),
        name=f"{clip_id}.max_acceleration_rad_s2",
    )
    max_iterations = _positive_int(
        config.get("max_iterations"), name=f"{clip_id}.max_iterations"
    )
    fps = _finite_number(report.get("fps"), name=f"{clip_id}.fps")
    if fps <= 0.0 or max_velocity <= 0.0 or max_acceleration <= 0.0:
        raise ValueError(f"{clip_id} fps and trajectory ceilings must be positive")

    try:
        with np.load(expert_path, allow_pickle=False) as archive:
            artifact = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load expert artifact {expert_path}: {exc}") from exc
    schema_version = _retarget_schema_version(artifact, report, clip_id=clip_id)

    frame_count = _positive_int(
        report.get("frame_count"), name=f"{clip_id}.frame_count"
    )
    stored_valid: np.ndarray | None = None
    if "expert_valid" in artifact:
        stored_valid = np.asarray(artifact["expert_valid"])
        if (
            stored_valid.ndim != 1
            or stored_valid.dtype.kind != "b"
            or len(stored_valid) != frame_count
        ):
            raise ValueError(
                "expert_valid must be a one-dimensional boolean array matching frame_count"
            )
    if frame_count < 2:
        raise ValueError("expert artifact must contain at least two frames")

    before = _frame_array(artifact, "weighted_task_error_before", frame_count)
    after = _frame_array(artifact, "weighted_task_error_after", frame_count)
    feasible_seed = _frame_array(
        artifact, "weighted_task_error_feasible_seed", frame_count
    )
    weighted_regression = _gate_regression(before, after, relative_tolerance)
    weighted_excess = _gate_excess(before, after, relative_tolerance)
    feasible_seed_regression = _gate_regression(
        before, feasible_seed, relative_tolerance
    )

    priority_before: list[np.ndarray] = []
    priority_after: list[np.ndarray] = []
    priority_excess: list[np.ndarray] = []
    protected_regressions: list[np.ndarray] = []
    feasible_seed_protected: list[np.ndarray] = []
    for priority in range(protected_tiers):
        tier_before = _frame_array(
            artifact, f"priority_{priority}_error_before", frame_count
        )
        tier_after = _frame_array(
            artifact, f"priority_{priority}_error_after", frame_count
        )
        tier_seed = _frame_array(
            artifact, f"priority_{priority}_error_feasible_seed", frame_count
        )
        priority_before.append(tier_before)
        priority_after.append(tier_after)
        priority_excess.append(
            _gate_excess(tier_before, tier_after, relative_tolerance)
        )
        protected_regressions.append(
            _gate_regression(tier_before, tier_after, relative_tolerance)
        )
        feasible_seed_protected.append(
            _gate_regression(tier_before, tier_seed, relative_tolerance)
        )

    constraint_relaxation = _frame_array(
        artifact, "constraint_relaxation_count", frame_count
    )
    validity_failure_masks: dict[str, np.ndarray] = {
        "weighted_regression": weighted_regression,
    }
    validity_excesses: dict[str, np.ndarray] = {
        "weighted_regression": weighted_excess,
    }
    validity_rule: dict[str, Any] = {
        "schema_version": schema_version,
        "priority_relative_tolerance": relative_tolerance,
    }
    if schema_version <= 2:
        validity_rule.update(
            {
                "name": "legacy_weighted_and_protected_priority_non_regression",
                "protected_priority_tiers": protected_tiers,
            }
        )
        for priority, regression in enumerate(protected_regressions):
            name = f"protected_priority_{priority}_regression"
            validity_failure_masks[name] = regression
            validity_excesses[name] = priority_excess[priority]
    else:
        max_foot_position_error = _finite_number(
            config.get("valid_max_foot_position_error_m"),
            name=f"{clip_id}.valid_max_foot_position_error_m",
        )
        max_foot_orientation_regression = _finite_number(
            config.get("valid_max_foot_orientation_regression_rad"),
            name=f"{clip_id}.valid_max_foot_orientation_regression_rad",
        )
        max_com_regression = _finite_number(
            config.get("valid_max_com_regression_m"),
            name=f"{clip_id}.valid_max_com_regression_m",
        )
        native_action_clip = _finite_number(
            config.get("native_action_clip"), name=f"{clip_id}.native_action_clip"
        )
        if (
            max_foot_position_error < 0.0
            or max_foot_orientation_regression < 0.0
            or max_com_regression < 0.0
            or native_action_clip <= 0.0
        ):
            raise ValueError(f"{clip_id} physical validity limits are invalid")
        validity_rule.update(
            {
                "name": "physical_task_bounds_v3",
                "valid_max_foot_position_error_m": max_foot_position_error,
                "valid_max_foot_orientation_regression_rad": (
                    max_foot_orientation_regression
                ),
                "valid_max_com_regression_m": max_com_regression,
                "native_action_clip": native_action_clip,
            }
        )
        for foot_name in ("left_foot", "right_foot"):
            position_after = _frame_array(
                artifact,
                f"task_{foot_name}_position_error_after_m",
                frame_count,
            )
            orientation_before = _frame_array(
                artifact,
                f"task_{foot_name}_orientation_error_before_rad",
                frame_count,
            )
            orientation_after = _frame_array(
                artifact,
                f"task_{foot_name}_orientation_error_after_rad",
                frame_count,
            )
            position_name = f"{foot_name}_position_threshold"
            orientation_name = f"{foot_name}_orientation_regression"
            validity_failure_masks[position_name] = (
                position_after > max_foot_position_error
            )
            validity_failure_masks[orientation_name] = (
                orientation_after
                > orientation_before + max_foot_orientation_regression
            )
            validity_excesses[position_name] = (
                position_after - max_foot_position_error
            ) / max(max_foot_position_error, 1.0e-12)
            validity_excesses[orientation_name] = (
                orientation_after
                - orientation_before
                - max_foot_orientation_regression
            ) / max(max_foot_orientation_regression, 1.0e-12)
        com_before = _frame_array(
            artifact,
            "task_whole_robot_com_position_error_before_m",
            frame_count,
        )
        com_after = _frame_array(
            artifact,
            "task_whole_robot_com_position_error_after_m",
            frame_count,
        )
        action_target = _frame_array(
            artifact, "action_target_native", frame_count, ndim=2
        )
        if action_target.shape[1] < 1:
            raise ValueError("action_target_native must contain at least one action")
        action_abs_max = np.max(np.abs(action_target), axis=1)
        validity_failure_masks["com_regression_threshold"] = (
            com_after > com_before + max_com_regression
        )
        validity_failure_masks["constraint_relaxation"] = constraint_relaxation != 0
        validity_failure_masks["native_action_clip_exceeded"] = (
            action_abs_max > native_action_clip
        )
        validity_excesses["com_regression_threshold"] = (
            com_after - com_before - max_com_regression
        ) / max(max_com_regression, 1.0e-12)
        validity_excesses["constraint_relaxation"] = constraint_relaxation
        validity_excesses["native_action_clip_exceeded"] = (
            action_abs_max - native_action_clip
        ) / native_action_clip

    recomputed_valid = np.ones(frame_count, dtype=bool)
    for failure in validity_failure_masks.values():
        recomputed_valid &= ~failure
    invalid = ~recomputed_valid
    mask_mismatches = (
        np.asarray([], dtype=np.int64)
        if stored_valid is None
        else np.flatnonzero(recomputed_valid != stored_valid).astype(np.int64)
    )

    contact_flags = _frame_array(
        artifact, "contact_flags", frame_count, ndim=2
    ).astype(bool)
    if contact_flags.shape[1] != 2:
        raise ValueError("contact_flags must have shape [frames, 2]")
    transition_frames, transition_sides = _contact_transitions(contact_flags)
    nearest_contact = _nearest_distances(frame_count, transition_frames)
    exact_transition = nearest_contact == 0
    near_transition = (nearest_contact >= 0) & (
        nearest_contact <= thresholds.contact_radius_frames
    )

    desired_task_pos = _frame_array(
        artifact, "desired_task_pos_w", frame_count, ndim=3
    )
    if desired_task_pos.shape[2] != 3:
        raise ValueError("desired_task_pos_w must have shape [frames, tasks, 3]")
    task_names = _read_task_names(artifact)
    if desired_task_pos.shape[1] != len(task_names):
        raise ValueError("desired_task_pos_w task axis does not match task_names")
    max_hand_speed, hand_speeds = _hand_speeds(desired_task_pos, task_names, fps)

    velocity = _frame_array(
        artifact, "trajectory_velocity_abs_max", frame_count
    )
    acceleration = _frame_array(
        artifact, "trajectory_acceleration_abs_max", frame_count
    )
    iterations = _frame_array(artifact, "solver_iterations", frame_count)
    position_hits = _frame_array(
        artifact, "position_limit_hit_count", frame_count
    )
    feasible_seed_any = feasible_seed_regression.copy()
    for regression in feasible_seed_protected:
        feasible_seed_any |= regression

    cause_masks: dict[str, np.ndarray] = {
        **validity_failure_masks,
        **{
            f"protected_priority_{priority}_regression": regression
            for priority, regression in enumerate(protected_regressions)
            if f"protected_priority_{priority}_regression"
            not in validity_failure_masks
        },
        "feasible_seed_already_worse": feasible_seed_any,
        "contact_transition_exact": exact_transition,
        "near_contact_transition": near_transition,
        "velocity_ceiling": velocity >= max_velocity * thresholds.ceiling_fraction,
        "acceleration_ceiling": (
            acceleration >= max_acceleration * thresholds.ceiling_fraction
        ),
        "high_desired_hand_speed": (
            max_hand_speed >= thresholds.high_hand_speed_m_s
        ),
        "safe_envelope_hit": position_hits > 0.0,
        "solver_iteration_ceiling": iterations >= max_iterations,
    }

    invalid_frames = np.flatnonzero(invalid).astype(np.int64)
    all_invalid_diagnostics = [
        _frame_diagnostic(
            clip_id=clip_id,
            frame=int(frame),
            fps=fps,
            before=before,
            after=after,
            validity_excesses=validity_excesses,
            priority_before=priority_before,
            priority_after=priority_after,
            feasible_seed=feasible_seed,
            velocity=velocity,
            acceleration=acceleration,
            iterations=iterations,
            position_hits=position_hits,
            nearest_contact=nearest_contact,
            transition_sides=transition_sides,
            max_hand_speed=max_hand_speed,
            hand_speeds=hand_speeds,
            validity_failure_masks=validity_failure_masks,
            cause_masks=cause_masks,
            artifact=artifact,
            task_names=task_names,
            frame_count=frame_count,
        )
        for frame in invalid_frames
    ]
    ranked = sorted(
        all_invalid_diagnostics,
        key=lambda item: (
            -item["severity_max_normalized_gate_excess"],
            item["frame"],
        ),
    )

    windows: list[dict[str, Any]] = []
    for start, end in _contiguous_windows(invalid_frames):
        window_mask = np.zeros(frame_count, dtype=bool)
        window_mask[start : end + 1] = True
        distances = nearest_contact[start : end + 1]
        valid_distances = distances[distances >= 0]
        windows.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_time_s": float(start / fps),
                "end_time_s": float(end / fps),
                "length_frames": end - start + 1,
                "long_run": end - start + 1 >= thresholds.long_run_min_frames,
                "cause_counts": _cause_counts(cause_masks, window_mask),
                "nearest_contact_transition_distance_frames": (
                    None
                    if len(valid_distances) == 0
                    else int(np.min(valid_distances))
                ),
            }
        )

    invalid_distances = nearest_contact[invalid]
    invalid_distances = invalid_distances[invalid_distances >= 0]
    context_associations = {
        name: _association(cause_masks[name], invalid) for name in CONTEXT_CAUSES
    }
    expected_hash = report.get("expert_output_sha256")
    hash_matches = (
        None
        if not isinstance(expected_hash, str)
        else _sha256(expert_path) == expected_hash
    )
    long_windows = [window for window in windows if window["long_run"]]
    clip_report: dict[str, Any] = {
        "clip_id": clip_id,
        "category": report.get("category", "uncategorized"),
        "deployment_ready": False,
        "expert_authorized": False,
        "frame_count": frame_count,
        "fps": fps,
        "invalid_frame_count": int(len(invalid_frames)),
        "invalid_frame_fraction": float(np.mean(invalid)),
        "invalid_frames": invalid_frames.tolist(),
        "invalid_windows": windows,
        "invalid_window_count": len(windows),
        "long_invalid_window_count": len(long_windows),
        "long_invalid_frame_count": int(
            sum(window["length_frames"] for window in long_windows)
        ),
        "expert_valid_stored_present": stored_valid is not None,
        "expert_valid_recomputed_matches_stored": (
            None if stored_valid is None else len(mask_mismatches) == 0
        ),
        "expert_valid_mismatch_frames": mask_mismatches.tolist(),
        "expert_artifact_hash_matches_report": hash_matches,
        "integrity_passed": (
            stored_valid is not None
            and len(mask_mismatches) == 0
            and hash_matches is not False
        ),
        "validity_rule": validity_rule,
        "validity_failure_counts": _cause_counts(validity_failure_masks, invalid),
        "cause_counts_on_invalid_frames": _cause_counts(cause_masks, invalid),
        "context_associations": context_associations,
        "contact_transitions": {
            "transition_count": int(len(transition_frames)),
            "transition_frames": transition_frames.tolist(),
            "changed_sides_by_frame": {
                str(frame): sides for frame, sides in sorted(transition_sides.items())
            },
            "invalid_nearest_distance_frames": _percentiles_or_none(
                invalid_distances
            ),
        },
        "invalid_frame_diagnostics": all_invalid_diagnostics,
        "worst_frames": ranked[: thresholds.worst_frame_count],
    }
    aggregate_masks = {"invalid": invalid, **cause_masks}
    return clip_report, aggregate_masks


def _find_clip_pairs(input_root: Path) -> list[tuple[Path, Path]]:
    reports_root = input_root / "reports"
    experts_root = input_root / "experts"
    if not reports_root.is_dir() or not experts_root.is_dir():
        raise ValueError("input root must contain reports/ and experts/ directories")
    report_paths = sorted(reports_root.glob("*.retarget.json"), key=lambda path: path.name)
    if not report_paths:
        raise ValueError(f"no *.retarget.json files found in {reports_root}")
    pairs: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for report_path in report_paths:
        report = _load_report(report_path)
        clip_id = _clip_id_from_report(report, report_path)
        folded = clip_id.casefold()
        if folded in seen:
            raise ValueError(f"duplicate clip_id in reports: {clip_id}")
        seen.add(folded)
        expert_path = experts_root / f"{clip_id}.task_space.npz"
        if not expert_path.is_file():
            raise ValueError(f"missing expert artifact for {clip_id}: {expert_path}")
        pairs.append((report_path, expert_path))
    return pairs


def _combine_associations(
    clips: list[dict[str, Any]],
) -> dict[str, dict[str, int | float | None]]:
    combined: dict[str, dict[str, int | float | None]] = {}
    for cause in CONTEXT_CAUSES:
        present = sum(
            int(clip["context_associations"][cause]["context_frame_count"])
            for clip in clips
        )
        invalid_present = sum(
            int(clip["context_associations"][cause]["invalid_context_frame_count"])
            for clip in clips
        )
        frame_count = sum(int(clip["frame_count"]) for clip in clips)
        invalid_count = sum(int(clip["invalid_frame_count"]) for clip in clips)
        absent = frame_count - present
        invalid_absent = invalid_count - invalid_present
        present_rate = _ratio_or_none(invalid_present, present)
        absent_rate = _ratio_or_none(invalid_absent, absent)
        combined[cause] = {
            "context_frame_count": present,
            "invalid_context_frame_count": invalid_present,
            "invalid_rate_when_present": present_rate,
            "invalid_rate_when_absent": absent_rate,
            "invalid_rate_lift": (
                None
                if present_rate is None or absent_rate in (None, 0.0)
                else float(present_rate / absent_rate)
            ),
        }
    return combined


def _assert_json_finite(value: object, *, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_json_finite(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_json_finite(child, location=f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {location}")
    elif isinstance(value, np.generic):
        raise TypeError(f"NumPy scalar escaped JSON conversion at {location}")


def build_regression_audit(
    input_root: Path,
    thresholds: AuditThresholds = AuditThresholds(),
) -> dict[str, Any]:
    """Build deterministic, frame-weighted audit for all completed clips."""

    root = input_root.resolve()
    clips: list[dict[str, Any]] = []
    aggregate_masks: list[dict[str, np.ndarray]] = []
    for report_path, expert_path in _find_clip_pairs(root):
        clip, masks = analyze_clip(report_path, expert_path, thresholds)
        clips.append(clip)
        aggregate_masks.append(masks)
    clips.sort(key=lambda item: item["clip_id"])

    frame_count = sum(int(clip["frame_count"]) for clip in clips)
    invalid_count = sum(int(clip["invalid_frame_count"]) for clip in clips)
    cause_names = sorted(
        {
            name
            for masks in aggregate_masks
            for name in masks
            if name != "invalid"
        }
    )
    aggregate_causes = {
        name: sum(
            int(np.sum(masks[name] & masks["invalid"]))
            for masks in aggregate_masks
            if name in masks
        )
        for name in cause_names
    }
    worst = sorted(
        (
            frame
            for clip in clips
            for frame in clip["invalid_frame_diagnostics"]
        ),
        key=lambda item: (
            -item["severity_max_normalized_gate_excess"],
            item["clip_id"],
            item["frame"],
        ),
    )[: thresholds.worst_frame_count]
    invalid_distances = [
        frame["nearest_contact_transition_distance_frames"]
        for clip in clips
        for frame in clip["invalid_frame_diagnostics"]
        if frame["nearest_contact_transition_distance_frames"] is not None
    ]
    report: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "deployment_ready": False,
        "expert_authorized": False,
        "training_expert_authorized": False,
        "authorization": "diagnostic_only_not_training_or_deployment_authorization",
        "input_root": str(root),
        "thresholds": thresholds.as_dict(),
        "clip_count": len(clips),
        "frame_count": frame_count,
        "invalid_frame_count": invalid_count,
        "invalid_frame_fraction": float(invalid_count / frame_count),
        "invalid_window_count": sum(int(clip["invalid_window_count"]) for clip in clips),
        "long_invalid_window_count": sum(
            int(clip["long_invalid_window_count"]) for clip in clips
        ),
        "long_invalid_frame_count": sum(
            int(clip["long_invalid_frame_count"]) for clip in clips
        ),
        "cause_counts_on_invalid_frames": aggregate_causes,
        "context_associations": _combine_associations(clips),
        "contact_transition_invalid_distance_frames": _percentiles_or_none(
            invalid_distances
        ),
        "integrity_passed": all(bool(clip["integrity_passed"]) for clip in clips),
        "expert_valid_mismatch_count": sum(
            len(clip["expert_valid_mismatch_frames"]) for clip in clips
        ),
        "expert_valid_missing_clip_count": sum(
            not bool(clip["expert_valid_stored_present"]) for clip in clips
        ),
        "clips": clips,
        "worst_frames": worst,
    }
    _assert_json_finite(report)
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        temporary.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output", type=Path, help="optional JSON output; stdout otherwise")
    parser.add_argument(
        "--contact-radius-frames", type=int, default=DEFAULT_CONTACT_RADIUS_FRAMES
    )
    parser.add_argument("--ceiling-fraction", type=float, default=DEFAULT_CEILING_FRACTION)
    parser.add_argument(
        "--high-hand-speed-m-s", type=float, default=DEFAULT_HIGH_HAND_SPEED_M_S
    )
    parser.add_argument(
        "--long-run-min-frames", type=int, default=DEFAULT_LONG_RUN_MIN_FRAMES
    )
    parser.add_argument(
        "--worst-frame-count", type=int, default=DEFAULT_WORST_FRAME_COUNT
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        thresholds = AuditThresholds(
            contact_radius_frames=args.contact_radius_frames,
            ceiling_fraction=args.ceiling_fraction,
            high_hand_speed_m_s=args.high_hand_speed_m_s,
            long_run_min_frames=args.long_run_min_frames,
            worst_frame_count=args.worst_frame_count,
        )
        report = build_regression_audit(args.input_root, thresholds)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.output is None:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))  # noqa: T201
    else:
        _write_json_atomic(args.output, report)
        print(f"regression audit: {args.output.resolve()}")  # noqa: T201
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
