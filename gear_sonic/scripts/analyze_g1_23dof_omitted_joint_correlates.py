"""Offline, non-causal audit of six G1 joints omitted by true23 retargeting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

AUDIT_SCHEMA = "g1_true23_omitted_joint_correlate_audit_v1"
OMITTED_JOINTS = (
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
INTERPOLATION = "pchip_root_cubic_clipped_joints_rotation_spline_quaternion"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def _stat(values: np.ndarray) -> dict[str, float | None]:
    values = _finite(values, "stat values")
    if not len(values):
        return {key: None for key in ("count", "mean", "median", "p05", "p95", "max")}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _macro(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "iqr": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
    }


def _auc(valid: np.ndarray, score: np.ndarray) -> float | None:
    pos, neg = score[valid], score[~valid]
    if not len(pos) or not len(neg):
        return None
    ranks = np.empty(len(score), dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1]
    ends = np.r_[starts[1:], len(score)]
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
    return float((np.sum(ranks[valid]) - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _cliff(invalid: np.ndarray, valid: np.ndarray) -> float | None:
    if not len(invalid) or not len(valid):
        return None
    auc = _auc(np.r_[np.zeros(len(invalid), dtype=bool), np.ones(len(valid), dtype=bool)], np.r_[invalid, valid])
    return None if auc is None else float(1.0 - 2.0 * auc)


def _derivatives(values: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact core trajectory_derivatives free-initial convention."""
    interval = np.diff(values, axis=0) * fps
    velocity = np.vstack((interval[:1], interval))
    acceleration = np.vstack((np.zeros_like(velocity[:1]), np.diff(velocity, axis=0) * fps))
    return velocity, acceleration


def _features(values: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    velocity, acceleration = _derivatives(values, fps)
    groups = {
        "waist_l2": slice(0, 2),
        "left_wrist_l2": slice(2, 4),
        "right_wrist_l2": slice(4, 6),
        "all_six_l2": slice(0, 6),
    }
    features = {f"{name}_angle_abs_rad": np.abs(values[:, i]) for i, name in enumerate(OMITTED_JOINTS)}
    features.update(
        {f"{name}_velocity_abs_rad_s": np.abs(velocity[:, i]) for i, name in enumerate(OMITTED_JOINTS)}
    )
    features.update(
        {f"{name}_acceleration_abs_rad_s2": np.abs(acceleration[:, i]) for i, name in enumerate(OMITTED_JOINTS)}
    )
    for name, part in groups.items():
        features[f"{name}_angle"] = np.linalg.norm(values[:, part], axis=1)
        features[f"{name}_velocity"] = np.linalg.norm(velocity[:, part], axis=1)
        features[f"{name}_acceleration"] = np.linalg.norm(acceleration[:, part], axis=1)
    features["all_six_max_abs_angle_rad"] = np.max(np.abs(values), axis=1)
    features["all_six_max_abs_velocity_rad_s"] = np.max(np.abs(velocity), axis=1)
    features["all_six_max_abs_acceleration_rad_s2"] = np.max(np.abs(acceleration), axis=1)
    return features


def _ranks(values: np.ndarray) -> np.ndarray:
    """Tie-aware normalized percentile ranks in [0, 1]."""
    if len(values) < 2:
        return np.zeros(len(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends, strict=True):
        result[order[start:end]] = ((start + end - 1) / 2.0) / (len(values) - 1)
    return result


def _lift(valid: np.ndarray, score: np.ndarray) -> float | None:
    if not np.any(valid) or not np.any(~valid):
        return None
    threshold = np.percentile(score[valid], 90)
    selected = score >= threshold
    if not np.any(selected):
        return None
    return float(np.mean((~valid)[selected]) / np.mean(~valid))


def _windows(
    mask: np.ndarray,
    source_time_s: np.ndarray,
    fps: float,
    features: dict[str, np.ndarray],
    base_count: int,
    raw_count: int,
) -> list[dict[str, Any]]:
    frames = np.flatnonzero(mask)
    if not len(frames):
        return []
    cuts = np.flatnonzero(np.diff(frames) != 1) + 1
    return [
        {
            "start_frame": int(x[0]),
            "end_frame": int(x[-1]),
            "frame_count": int(len(x)),
            "start_time_s": float(x[0] / fps),
            "end_time_s": float(x[-1] / fps),
            "source_start_time_s": float(source_time_s[x[0]]),
            "source_end_time_s": float(source_time_s[x[-1]]),
            "base_phase_start": float(x[0] * (base_count - 1) / (len(mask) - 1)),
            "base_phase_end": float(x[-1] * (base_count - 1) / (len(mask) - 1)),
            "raw_phase_start": float(x[0] * (raw_count - 1) / (len(mask) - 1)),
            "raw_phase_end": float(x[-1] * (raw_count - 1) / (len(mask) - 1)),
            "nearest_source_row_start": int(round(x[0] * (raw_count - 1) / (len(mask) - 1))),
            "nearest_source_row_end": int(round(x[-1] * (raw_count - 1) / (len(mask) - 1))),
            "feature_summary": {name: _stat(value[x]) for name, value in features.items()},
        }
        for x in np.split(frames, cuts)
    ]


def _reconstruct(csv_path: Path, report: dict[str, Any], frames: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("source CSV has no rows")
    missing = [name for name in OMITTED_JOINTS if f"{name}_dof" not in rows[0]]
    if missing:
        raise ValueError(f"CSV lacks omitted joints: {missing}")
    try:
        raw = np.deg2rad(np.asarray([[float(row[f"{name}_dof"]) for name in OMITTED_JOINTS] for row in rows]))
    except (TypeError, ValueError) as exc:
        raise ValueError("source CSV omitted-joint values must be numeric") from exc
    raw = _finite(raw, "CSV joints")
    n = len(raw)
    fps_source = float(report["fps_source"])
    fps_target = float(report["fps_target"])
    if fps_target > fps_source:
        raise ValueError("fps_target must not exceed fps_source")
    base_count = int(math.floor(n / fps_source * fps_target))
    retiming = report.get("retiming")
    if not isinstance(retiming, dict) or retiming.get("interpolation") != INTERPOLATION:
        raise ValueError("retiming interpolation label mismatch")
    if (
        int(retiming.get("base_frame_count", -1)) != base_count
        or int(retiming.get("retimed_frame_count", -1)) != frames
    ):
        raise ValueError("retiming frame counts mismatch")
    if n < 3 or base_count < 3 or frames < 3:
        raise ValueError("CubicSpline requires at least three frames")
    base_phase = np.linspace(0.0, n - 1.0, base_count)
    base = CubicSpline(np.arange(n, dtype=np.float64), raw, axis=0)(base_phase)
    base = np.clip(base, raw.min(axis=0), raw.max(axis=0))
    retime_phase = np.linspace(0.0, base_count - 1.0, frames)
    result = CubicSpline(np.arange(base_count, dtype=np.float64), base, axis=0)(retime_phase)
    result = np.clip(result, base.min(axis=0), base.max(axis=0))
    raw_phase = retime_phase * (n - 1.0) / (base_count - 1.0)
    return result, raw_phase / fps_source, base_count, n


def _clip(root: Path, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("report must be object")
    version = report.get("schema_version")
    if version not in (4, 5, 6):
        raise ValueError("report schema_version must be 4, 5, or 6")
    clip_id = report.get("clip_id")
    if not isinstance(clip_id, str) or not clip_id:
        raise ValueError("invalid clip_id")
    expert_path = root / "experts" / f"{clip_id}.task_space.npz"
    if not expert_path.is_file():
        raise ValueError(f"missing expert {clip_id}")
    if report.get("expert_output_sha256") != _sha256(expert_path):
        raise ValueError("expert lineage hash mismatch")
    csv = Path(str(report.get("source_csv", "")))
    if not csv.is_file() or report.get("source_csv_sha256") != _sha256(csv):
        raise ValueError("source CSV lineage hash mismatch")
    with np.load(expert_path, allow_pickle=False) as z:
        schema = np.asarray(z["schema_version"])
        valid = np.asarray(z["expert_valid"])
    if schema.shape != (1,) or int(schema[0]) != version:
        raise ValueError("expert/report schema mismatch")
    if valid.ndim != 1 or valid.dtype != np.bool_:
        raise ValueError("expert_valid must be bool vector")
    frames = len(valid)
    if int(report.get("frame_count", -1)) != frames:
        raise ValueError("report/expert frame count mismatch")
    values, source_time, base_count, raw_count = _reconstruct(csv, report, frames)
    features = _features(values, float(report["fps_target"]))
    per_joint = {}
    for name, score in features.items():
        invalid = score[~valid]
        good = score[valid]
        ranks = _ranks(score)
        per_joint[name] = {
            "invalid": _stat(invalid),
            "valid": _stat(good),
            "within_clip_invalid_percentiles": _stat(invalid),
            "within_clip_valid_percentiles": _stat(good),
            "within_clip_invalid_rank_percentiles": _stat(ranks[~valid]),
            "within_clip_valid_rank_percentiles": _stat(ranks[valid]),
            "auc_valid_higher": _auc(valid, score),
            "cliffs_delta_invalid_vs_valid": _cliff(invalid, good),
            "top_valid_decile_invalid_lift": _lift(valid, score),
        }
    return {
        "clip_id": clip_id,
        "frame_count": frames,
        "invalid_frame_count": int(np.sum(~valid)),
        "source_csv": str(csv),
        "omitted_joint_values_rad": values,
        "feature_arrays": features,
        "valid": valid,
        "source_time_s": source_time,
        "feature_stats": per_joint,
        "invalid_windows": _windows(
            ~valid, source_time, float(report["fps_target"]), features, base_count, raw_count
        ),
    }


def build_omitted_joint_audit(output_root: Path) -> dict[str, Any]:
    reports = sorted((output_root / "reports").glob("*.retarget.json"))
    if not reports:
        raise ValueError("no retarget reports")
    clips = [_clip(output_root, path) for path in reports]
    valid = np.concatenate([x.pop("valid") for x in clips])
    feature_arrays = [x.pop("feature_arrays") for x in clips]
    [x.pop("omitted_joint_values_rad") for x in clips]
    [x.pop("source_time_s") for x in clips]
    features = {name: np.concatenate([item[name] for item in feature_arrays]) for name in feature_arrays[0]}
    pooled = {}
    macro = {}
    for name, score in features.items():
        pooled[name] = {
            "invalid": _stat(score[~valid]),
            "valid": _stat(score[valid]),
            "auc_valid_higher": _auc(valid, score),
            "cliffs_delta_invalid_vs_valid": _cliff(score[~valid], score[valid]),
            "top_valid_decile_invalid_lift": _lift(valid, score),
        }
        macro[name] = {
            metric: _macro(
                [
                    clip["feature_stats"][name][metric]
                    for clip in clips
                    if clip["feature_stats"][name][metric] is not None
                ]
            )
            for metric in ("auc_valid_higher", "cliffs_delta_invalid_vs_valid", "top_valid_decile_invalid_lift")
        }
    return {
        "schema": AUDIT_SCHEMA,
        "schema_version": 1,
        "interpretation": "Descriptive correlations only; no causal inference or authorization.",
        "interpolation": INTERPOLATION,
        "frame_count": int(len(valid)),
        "invalid_frame_count": int(np.sum(~valid)),
        "clips": clips,
        "pooled_feature_stats": pooled,
        "macro_per_clip_feature_stats": macro,
        "expert_authorized": False,
        "training_expert_authorized": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    payload = build_omitted_joint_audit(args.output_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
