"""Build constrained true23 task-space experts from BONES-SEED G1 clips.

Each selected CSV produces three offline-only artifacts:

* ``motions/<clip_id>.npz``: standard MJLab 23-DoF tracking motion.
* ``experts/<clip_id>.task_space.npz``: direct subset, IK expert, task poses,
  contacts, action targets, and per-frame diagnostics.
* ``reports/<clip_id>.retarget.json``: provenance and aggregate quality evidence.

Input may be one CSV, a recursively scanned directory, or a JSON manifest. A
manifest has this shape::

    {
      "schema": "g1_true23_task_space_retarget_manifest_v1",
      "source_root": "optional/path/relative/to/manifest",
      "fps_source": 120.0,
      "clips": [
        {
          "clip_id": "walk_001",
          "source_csv": "session/motion.csv",
          "category": "walk",
          "fps_source": 120.0
        }
      ]
    }

The tool never edits a policy checkpoint and never grants deployment status.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback
from typing import Any
import uuid

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.spatial.transform import Rotation, RotationSpline

from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_RAW_ACTION_CLIP,
)
from gear_sonic.utils.g1_23dof_task_space_qualification import (
    QUALIFICATION_CATEGORIES,
    build_task_space_qualification_report,
)
from gear_sonic.utils.g1_23dof_task_space_retarget import (
    DEFAULT_SOURCE_MODEL,
    DEFAULT_TARGET_MODEL,
    TASK_SPACE_RETARGET_SCHEMA_VERSION,
    RetargetConfig,
    build_mjlab_motion_arrays,
    load_models,
    retarget_trajectory,
    safe_target_joint_bounds,
    trajectory_derivatives,
)

MANIFEST_SCHEMA = "g1_true23_task_space_retarget_manifest_v1"
BATCH_SCHEMA = "g1_true23_task_space_retarget_batch_v6"
ARTIFACT_PROVENANCE_SCHEMA = "g1_true23_task_space_retarget_artifact_v6"
ARTIFACT_PROVENANCE_SCHEMA_VERSION = 6
RETARGET_RESULT_SCHEMA = "g1_29dof_to_true23_task_space_retarget_v6"
SERIALIZATION_CERTIFICATE_BASIS = "serialized_float32_position_arrays"
AUTO_RETIME_MAX_CONVERGENCE_ITERATIONS = 12
AUTO_RETIME_DERIVATIVE_RELATIVE_TOLERANCE = 1.0e-6
_SAFE_MANIFEST_CLIP_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class ClipSpec:
    """One collision-free conversion request."""

    clip_id: str
    csv_path: Path
    category: str
    fps_source: float
    qualification_categories: tuple[str, ...] = QUALIFICATION_CATEGORIES


@dataclass(frozen=True)
class OutputTriplet:
    """Files constituting one resumable conversion artifact."""

    motion: Path
    expert: Path
    report: Path

    def paths(self) -> tuple[Path, Path, Path]:
        return (self.motion, self.expert, self.report)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        _write_json(temporary, payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_csv_trajectory(
    csv_path: Path,
    *,
    source_joint_names: tuple[str, ...],
    fps_source: float,
    fps_target: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from gear_sonic.data_process.convert_bones_seed_to_mjlab_npz import (
        read_bones_csv,
        resample_indices,
    )

    root_pos, root_quat, joints = read_bones_csv(csv_path)
    missing = [name for name in source_joint_names if name not in joints]
    if missing:
        raise ValueError(f"CSV lacks source joints: {missing}")
    frame_count = len(resample_indices(len(root_pos), fps_source, fps_target))
    source_joints = np.stack(
        [joints[name] for name in source_joint_names], axis=1
    )
    query_phase = np.linspace(0.0, float(len(source_joints) - 1), frame_count)
    return _interpolate_trajectory_at_phase(
        root_pos,
        root_quat,
        source_joints,
        query_phase,
    )


def _trajectory_derivative_maxima(
    joint_pos: np.ndarray,
    fps: float,
) -> tuple[float, float]:
    velocity = np.diff(joint_pos, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    velocity_max = float(np.max(np.abs(velocity))) if velocity.size else 0.0
    acceleration_max = (
        float(np.max(np.abs(acceleration))) if acceleration.size else 0.0
    )
    return velocity_max, acceleration_max


def _interpolate_trajectory_at_phase(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joints: np.ndarray,
    query_phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_phase = np.arange(len(joints), dtype=np.float64)
    retimed_root = PchipInterpolator(source_phase, root_pos, axis=0)(query_phase)
    retimed_joints = CubicSpline(source_phase, joints, axis=0)(query_phase)
    retimed_joints = np.clip(
        retimed_joints,
        np.min(joints, axis=0),
        np.max(joints, axis=0),
    )
    quaternion_xyzw = np.asarray(root_quat_wxyz)[:, [1, 2, 3, 0]].copy()
    for index in range(1, len(quaternion_xyzw)):
        if float(np.dot(quaternion_xyzw[index - 1], quaternion_xyzw[index])) < 0.0:
            quaternion_xyzw[index] *= -1.0
    rotations = Rotation.from_quat(quaternion_xyzw)
    retimed_xyzw = RotationSpline(source_phase, rotations)(query_phase).as_quat()
    retimed_quat = retimed_xyzw[:, [3, 0, 1, 2]]
    return retimed_root, retimed_quat, retimed_joints


def _interpolate_time_scaled_trajectory(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joints: np.ndarray,
    time_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if time_scale < 1.0:
        raise ValueError("time scale must be at least one")
    if len(joints) < 3 or time_scale == 1.0:
        return root_pos, root_quat_wxyz, joints, 1.0
    frame_count = int(math.ceil((len(joints) - 1) * time_scale)) + 1
    query_phase = np.linspace(0.0, float(len(joints) - 1), frame_count)
    actual_scale = (frame_count - 1) / (len(joints) - 1)
    retimed_root, retimed_quat, retimed_joints = _interpolate_trajectory_at_phase(
        root_pos,
        root_quat_wxyz,
        joints,
        query_phase,
    )
    return retimed_root, retimed_quat, retimed_joints, actual_scale


def _auto_retime_trajectory(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    source_joints: np.ndarray,
    *,
    source_joint_names: tuple[str, ...],
    target_joint_names: tuple[str, ...],
    fps: float,
    max_velocity_rad_s: float,
    max_acceleration_rad_s2: float,
    retained_lower_bounds: np.ndarray,
    retained_upper_bounds: np.ndarray,
    velocity_fraction: float,
    acceleration_fraction: float,
    max_time_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    retained_indices = np.asarray(
        [source_index[name] for name in target_joint_names], dtype=np.int64
    )
    retained = source_joints[:, retained_indices]
    retained_lower = np.asarray(retained_lower_bounds, dtype=np.float64)
    retained_upper = np.asarray(retained_upper_bounds, dtype=np.float64)
    if retained_lower.shape != (len(retained_indices),) or retained_upper.shape != (
        len(retained_indices),
    ):
        raise ValueError("retained safe bounds must match target joint count")
    if np.any(retained_lower >= retained_upper):
        raise ValueError("retained safe bounds are empty")
    retained = np.clip(retained, retained_lower, retained_upper)
    source_velocity_max, source_acceleration_max = _trajectory_derivative_maxima(
        retained, fps
    )
    velocity_budget = max_velocity_rad_s * velocity_fraction
    acceleration_budget = max_acceleration_rad_s2 * acceleration_fraction
    requested_scale = max(
        1.0,
        source_velocity_max / velocity_budget,
        math.sqrt(source_acceleration_max / acceleration_budget),
    )
    if requested_scale > max_time_scale:
        raise ValueError(
            f"required time scale {requested_scale:.6g} exceeds "
            f"configured maximum {max_time_scale:.6g}"
        )

    actual_scale = requested_scale
    retimed_root = root_pos
    retimed_quat = root_quat_wxyz
    retimed_joints = source_joints
    for _ in range(AUTO_RETIME_MAX_CONVERGENCE_ITERATIONS):
        (
            retimed_root,
            retimed_quat,
            retimed_joints,
            actual_scale,
        ) = _interpolate_time_scaled_trajectory(
            root_pos,
            root_quat_wxyz,
            source_joints,
            actual_scale,
        )
        retimed_retained = np.clip(
            retimed_joints[:, retained_indices],
            retained_lower,
            retained_upper,
        )
        velocity_max, acceleration_max = _trajectory_derivative_maxima(
            retimed_retained, fps
        )
        correction = max(
            1.0,
            velocity_max / velocity_budget,
            math.sqrt(acceleration_max / acceleration_budget),
        )
        if correction <= 1.0 + AUTO_RETIME_DERIVATIVE_RELATIVE_TOLERANCE:
            break
        requested_scale = actual_scale * correction * 1.01
        if requested_scale > max_time_scale:
            raise ValueError(
                f"required time scale {requested_scale:.6g} exceeds "
                f"configured maximum {max_time_scale:.6g}"
            )
        actual_scale = requested_scale
    else:
        raise RuntimeError(
            "automatic retiming did not converge after "
            f"{AUTO_RETIME_MAX_CONVERGENCE_ITERATIONS} iterations: "
            f"time scale {actual_scale:.6g}, velocity {velocity_max:.6g}/"
            f"{velocity_budget:.6g} rad/s, acceleration "
            f"{acceleration_max:.6g}/{acceleration_budget:.6g} rad/s^2"
        )

    # Recompute from the exact arrays returned to the caller. Keep this audit
    # outside the convergence controller so a future change to its stopping
    # condition cannot silently certify an over-budget trajectory.
    audited_retained = np.clip(
        retimed_joints[:, retained_indices],
        retained_lower,
        retained_upper,
    )
    audited_velocity = np.diff(audited_retained, axis=0) * fps
    audited_acceleration = np.diff(audited_velocity, axis=0) * fps
    velocity_max = (
        float(np.max(np.abs(audited_velocity)))
        if audited_velocity.size
        else 0.0
    )
    acceleration_max = (
        float(np.max(np.abs(audited_acceleration)))
        if audited_acceleration.size
        else 0.0
    )
    audit_limit = 1.0 + AUTO_RETIME_DERIVATIVE_RELATIVE_TOLERANCE
    audit_passed = (
        math.isfinite(velocity_max)
        and math.isfinite(acceleration_max)
        and velocity_max <= velocity_budget * audit_limit
        and acceleration_max <= acceleration_budget * audit_limit
    )
    if not audit_passed:
        raise RuntimeError(
            "automatic retiming failed final derivative audit: "
            f"velocity {velocity_max:.6g}/{velocity_budget:.6g} rad/s, "
            f"acceleration {acceleration_max:.6g}/{acceleration_budget:.6g} "
            "rad/s^2"
        )

    return retimed_root, retimed_quat, retimed_joints, {
        "enabled": True,
        "interpolation": (
            "pchip_root_cubic_clipped_joints_rotation_spline_quaternion"
        ),
        "base_frame_count": int(len(source_joints)),
        "retimed_frame_count": int(len(retimed_joints)),
        "time_scale": float(actual_scale),
        "source_velocity_abs_max_rad_s": source_velocity_max,
        "source_acceleration_abs_max_rad_s2": source_acceleration_max,
        "retimed_velocity_abs_max_rad_s": velocity_max,
        "retimed_acceleration_abs_max_rad_s2": acceleration_max,
        "velocity_budget_rad_s": velocity_budget,
        "acceleration_budget_rad_s2": acceleration_budget,
        "constraint_signal": "safe_action_reachable_clipped_retained23",
    }


def _model_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(1, model.njnt)
    )


def _validate_manifest_clip_id(value: object, *, index: int) -> str:
    if not isinstance(value, str) or not _SAFE_MANIFEST_CLIP_ID.fullmatch(value):
        raise ValueError(
            f"manifest clips[{index}].clip_id must match "
            "[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    return value


def _validate_derived_clip_id(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"unsafe CSV stem cannot be used as clip_id: {value!r}")
    return value


def _positive_fps(value: object, *, location: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be a finite positive number")
    try:
        fps = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be a finite positive number") from exc
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{location} must be a finite positive number")
    return fps


def _source_identity(path: Path) -> str:
    """Cross-platform conservative identity used to reject repeated sources."""

    return str(path.resolve()).replace("\\", "/").casefold()


def _validate_unique_specs(specs: list[ClipSpec], *, source: str) -> None:
    clip_ids: dict[str, ClipSpec] = {}
    csv_sources: dict[str, ClipSpec] = {}
    for spec in specs:
        clip_key = spec.clip_id.casefold()
        if clip_key in clip_ids:
            other = clip_ids[clip_key]
            raise ValueError(
                f"{source} has case-insensitive duplicate clip_id values: "
                f"{other.clip_id!r}, {spec.clip_id!r}"
            )
        clip_ids[clip_key] = spec

        source_key = _source_identity(spec.csv_path)
        if source_key in csv_sources:
            other = csv_sources[source_key]
            raise ValueError(
                f"{source} selects the same source CSV more than once: "
                f"{other.csv_path}, {spec.csv_path}"
            )
        csv_sources[source_key] = spec


def _manifest_qualification_categories(payload: dict[str, Any]) -> tuple[str, ...]:
    value = payload.get("qualification_categories")
    if value is None:
        return QUALIFICATION_CATEGORIES
    if not isinstance(value, list) or not value:
        raise ValueError(
            "manifest qualification_categories must be a non-empty JSON array"
        )
    if any(not isinstance(category, str) for category in value):
        raise ValueError("manifest qualification_categories must contain strings")
    if any(not category.strip() for category in value):
        raise ValueError(
            "manifest qualification_categories must contain non-empty strings"
        )
    if len(set(value)) != len(value):
        raise ValueError(
            "manifest qualification_categories must not contain duplicates"
        )
    unknown = [
        category for category in value if category not in QUALIFICATION_CATEGORIES
    ]
    if unknown:
        raise ValueError(
            "manifest qualification_categories contains unknown categories: "
            + ", ".join(repr(category) for category in unknown)
        )
    return tuple(value)


def _load_manifest_specs(
    manifest_path: Path,
    *,
    cli_source_root: Path | None,
    default_fps_source: float,
) -> tuple[list[ClipSpec], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    schema = payload.get("schema", MANIFEST_SCHEMA)
    if schema != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported manifest schema: {schema!r}")
    qualification_categories_declared = "qualification_categories" in payload
    qualification_categories = _manifest_qualification_categories(payload)

    if cli_source_root is not None:
        source_root = cli_source_root.resolve()
    else:
        manifest_source_root = payload.get("source_root")
        if manifest_source_root is None:
            source_root = manifest_path.parent
        elif isinstance(manifest_source_root, str) and manifest_source_root:
            candidate = Path(manifest_source_root)
            source_root = (
                candidate if candidate.is_absolute() else manifest_path.parent / candidate
            ).resolve()
        else:
            raise ValueError("manifest source_root must be a non-empty path string")

    manifest_fps_source = _positive_fps(
        payload.get("fps_source", default_fps_source),
        location="manifest fps_source",
    )
    entries = payload.get("clips")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest clips must be a non-empty JSON array")

    specs: list[ClipSpec] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest clips[{index}] must be a JSON object")
        clip_id = _validate_manifest_clip_id(entry.get("clip_id"), index=index)
        source_csv = entry.get("source_csv")
        if not isinstance(source_csv, str) or not source_csv:
            raise ValueError(
                f"manifest clips[{index}].source_csv must be a non-empty path string"
            )
        source_path = Path(source_csv)
        csv_path = (
            source_path if source_path.is_absolute() else source_root / source_path
        ).resolve()
        category_value = entry.get("category", "uncategorized")
        if not isinstance(category_value, str) or not category_value.strip():
            raise ValueError(
                f"manifest clips[{index}].category must be a non-empty string"
            )
        category = category_value.strip()
        if (
            qualification_categories_declared
            and category not in qualification_categories
        ):
            raise ValueError(
                f"manifest clips[{index}].category {category!r} is outside "
                "qualification_categories"
            )
        fps_source = _positive_fps(
            entry.get("fps_source", manifest_fps_source),
            location=f"manifest clips[{index}].fps_source",
        )
        specs.append(
            ClipSpec(
                clip_id=clip_id,
                csv_path=csv_path,
                category=category,
                fps_source=fps_source,
                qualification_categories=qualification_categories,
            )
        )
    _validate_unique_specs(specs, source="manifest")
    provenance = {
        "mode": "manifest",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_root": str(source_root),
        "manifest_schema": MANIFEST_SCHEMA,
        "qualification_categories": list(qualification_categories),
    }
    return specs, provenance


def _load_input_specs(input_path: Path, *, fps_source: float) -> tuple[list[ClipSpec], dict[str, Any]]:
    input_path = input_path.resolve()
    if input_path.is_file():
        paths = [input_path]
        mode = "file"
    elif input_path.is_dir():
        paths = sorted(input_path.rglob("*.csv"))
        mode = "directory"
    else:
        raise ValueError(f"input does not exist: {input_path}")
    if not paths:
        raise ValueError(f"no CSV files found under {input_path}")

    specs = [
        ClipSpec(
            clip_id=_validate_derived_clip_id(path.stem),
            csv_path=path,
            category="uncategorized",
            fps_source=fps_source,
        )
        for path in paths
    ]
    try:
        _validate_unique_specs(specs, source=f"{mode} input")
    except ValueError as exc:
        if mode == "directory":
            raise ValueError(f"{exc}; use --manifest with unique clip_id values") from exc
        raise
    return specs, {"mode": mode, "input": str(input_path)}


def _output_triplet(output_root: Path, clip_id: str) -> OutputTriplet:
    return OutputTriplet(
        motion=output_root / "motions" / f"{clip_id}.npz",
        expert=output_root / "experts" / f"{clip_id}.task_space.npz",
        report=output_root / "reports" / f"{clip_id}.retarget.json",
    )


def _cache_expectations(
    *,
    spec: ClipSpec,
    outputs: OutputTriplet,
    source_hash: str,
    source_model_path: Path,
    source_model_hash: str,
    target_model_path: Path,
    target_model_hash: str,
    fps_target: float,
    retarget_config: dict[str, Any],
    retiming_config: dict[str, Any],
) -> dict[str, Any]:
    expectations = {
        "schema": RETARGET_RESULT_SCHEMA,
        "schema_version": TASK_SPACE_RETARGET_SCHEMA_VERSION,
        "artifact_provenance_schema": ARTIFACT_PROVENANCE_SCHEMA,
        "artifact_provenance_schema_version": ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "clip_id": spec.clip_id,
        "category": spec.category,
        "fps_source": spec.fps_source,
        "fps_target": fps_target,
        "source_csv": str(spec.csv_path.resolve()),
        "source_csv_sha256": source_hash,
        "source_model": str(source_model_path.resolve()),
        "source_model_sha256": source_model_hash,
        "target_model": str(target_model_path.resolve()),
        "target_model_sha256": target_model_hash,
        "motion_output": str(outputs.motion.resolve()),
        "expert_output": str(outputs.expert.resolve()),
        "report_output": str(outputs.report.resolve()),
        "retarget_config": retarget_config,
        "retiming_config": retiming_config,
    }
    if spec.qualification_categories != QUALIFICATION_CATEGORIES:
        expectations["qualification_categories"] = list(
            spec.qualification_categories
        )
    return expectations


def _verified_cached_summary(
    outputs: OutputTriplet,
    expected: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    existing_count = sum(path.is_file() for path in outputs.paths())
    if existing_count != len(outputs.paths()):
        return None, "missing" if existing_count == 0 else "partial"
    try:
        payload = json.loads(outputs.report.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None, "invalid report payload"
        mismatches = [key for key, value in expected.items() if payload.get(key) != value]
        if mismatches:
            return None, f"provenance mismatch: {', '.join(mismatches)}"
        if payload.get("serialization_constraint_audit_passed") is not True:
            return None, "missing serialization constraint certificate"
        constraints = payload.get("constraints")
        if not isinstance(constraints, dict) or constraints.get(
            "certificate_basis"
        ) != SERIALIZATION_CERTIFICATE_BASIS:
            return None, "invalid serialization certificate basis"
        if payload.get("motion_output_sha256") != _sha256(outputs.motion):
            return None, "motion output hash mismatch"
        if payload.get("expert_output_sha256") != _sha256(outputs.expert):
            return None, "expert output hash mismatch"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"cache verification error: {type(exc).__name__}: {exc}"
    return payload, "verified"


def _publish_triplet(
    outputs: OutputTriplet,
    *,
    motion_arrays: dict[str, np.ndarray],
    expert_arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> None:
    """Stage all artifacts before replacing any final path; publish report last."""

    transaction = uuid.uuid4().hex
    staged = OutputTriplet(
        motion=outputs.motion.with_name(f".{outputs.motion.name}.{transaction}.partial"),
        expert=outputs.expert.with_name(f".{outputs.expert.name}.{transaction}.partial"),
        report=outputs.report.with_name(f".{outputs.report.name}.{transaction}.partial"),
    )
    try:
        _write_npz(staged.motion, motion_arrays)
        _write_npz(staged.expert, expert_arrays)
        summary["motion_output_sha256"] = _sha256(staged.motion)
        summary["expert_output_sha256"] = _sha256(staged.expert)
        _write_json(staged.report, summary)
        staged.motion.replace(outputs.motion)
        staged.expert.replace(outputs.expert)
        staged.report.replace(outputs.report)
    finally:
        for path in staged.paths():
            path.unlink(missing_ok=True)


def _convert_one(
    spec: ClipSpec,
    output_root: Path,
    source_model_path: Path,
    target_model_path: Path,
    source_model_hash: str,
    target_model_hash: str,
    fps_target: float,
    min_frames: int,
    config_values: dict[str, Any],
    retiming_values: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any] | None]:
    outputs = _output_triplet(output_root, spec.clip_id)
    try:
        if spec.csv_path.stat().st_size < 1024:
            return spec.clip_id, "pending: input file is empty/incomplete", None

        config = RetargetConfig(**config_values)
        config_provenance = asdict(config)
        retiming_config = dict(retiming_values or {"enabled": False})
        source_hash = _sha256(spec.csv_path)
        expected = _cache_expectations(
            spec=spec,
            outputs=outputs,
            source_hash=source_hash,
            source_model_path=source_model_path,
            source_model_hash=source_model_hash,
            target_model_path=target_model_path,
            target_model_hash=target_model_hash,
            fps_target=fps_target,
            retarget_config=config_provenance,
            retiming_config=retiming_config,
        )
        cached_summary, cache_state = _verified_cached_summary(outputs, expected)
        if cached_summary is not None:
            return spec.clip_id, "skipped: verified cache", cached_summary

        source_model, target_model = load_models(source_model_path, target_model_path)
        source_joint_names = _model_joint_names(source_model)
        target_joint_names = _model_joint_names(target_model)
        retained_lower, retained_upper = safe_target_joint_bounds(
            target_model,
            safe_limit_guard_rad=config.safe_limit_guard_rad,
            native_action_clip=config.native_action_clip,
        )
        root_pos, root_quat, source_joints = _load_csv_trajectory(
            spec.csv_path,
            source_joint_names=source_joint_names,
            fps_source=spec.fps_source,
            fps_target=fps_target,
        )
        if retiming_config.get("enabled"):
            root_pos, root_quat, source_joints, retiming = _auto_retime_trajectory(
                root_pos,
                root_quat,
                source_joints,
                source_joint_names=source_joint_names,
                target_joint_names=target_joint_names,
                fps=fps_target,
                max_velocity_rad_s=config.max_velocity_rad_s,
                max_acceleration_rad_s2=config.max_acceleration_rad_s2,
                retained_lower_bounds=retained_lower,
                retained_upper_bounds=retained_upper,
                velocity_fraction=float(retiming_config["velocity_fraction"]),
                acceleration_fraction=float(
                    retiming_config["acceleration_fraction"]
                ),
                max_time_scale=float(retiming_config["max_time_scale"]),
            )
        else:
            source_index = {
                name: index for index, name in enumerate(source_joint_names)
            }
            retained = source_joints[
                :,
                np.asarray(
                    [source_index[name] for name in target_joint_names],
                    dtype=np.int64,
                ),
            ]
            retained = np.clip(retained, retained_lower, retained_upper)
            velocity_max, acceleration_max = _trajectory_derivative_maxima(
                retained, fps_target
            )
            retiming = {
                "enabled": False,
            "interpolation": (
                "continuous_pchip_root_cubic_clipped_joints_rotation_spline_quaternion"
            ),
                "base_frame_count": int(len(source_joints)),
                "retimed_frame_count": int(len(source_joints)),
                "time_scale": 1.0,
                "source_velocity_abs_max_rad_s": velocity_max,
                "source_acceleration_abs_max_rad_s2": acceleration_max,
                "retimed_velocity_abs_max_rad_s": velocity_max,
                "retimed_acceleration_abs_max_rad_s2": acceleration_max,
                "constraint_signal": (
                    "safe_action_reachable_clipped_retained23"
                ),
            }
        if len(source_joints) < min_frames:
            return (
                spec.clip_id,
                f"rejected: only {len(source_joints)} frames after resampling",
                None,
            )
        result = retarget_trajectory(
            source_model=source_model,
            target_model=target_model,
            root_pos_w=root_pos,
            root_quat_wxyz=root_quat,
            source_joint_pos_hardware=source_joints,
            fps=fps_target,
            config=config,
        )
        motion_arrays = build_mjlab_motion_arrays(target_model, result)
        expert_arrays = result.adaptation_arrays()
        for arrays in (motion_arrays, expert_arrays):
            invalid = [
                name
                for name, value in arrays.items()
                if np.asarray(value).dtype.kind in "biufc"
                and not np.all(np.isfinite(value))
            ]
            if invalid:
                raise ValueError(f"non-finite output arrays: {invalid}")
            unsupported = [
                name
                for name, value in arrays.items()
                if np.asarray(value).dtype.kind not in "biufcUS"
            ]
            if unsupported:
                raise ValueError(f"unsupported output array dtypes: {unsupported}")
        expert_arrays["task_names"] = np.asarray(result.task_names, dtype=np.str_)

        dt = 1.0 / fps_target
        serialized_joint_velocity, serialized_joint_acceleration = (
            trajectory_derivatives(
                np.asarray(motion_arrays["joint_pos"], dtype=np.float64),
                dt,
            )
        )
        serialized_root_velocity, serialized_root_acceleration = (
            trajectory_derivatives(
                np.asarray(expert_arrays["root_offset_w"], dtype=np.float64),
                dt,
            )
        )
        serialized_joint_velocity_max = float(
            np.max(np.abs(serialized_joint_velocity))
        )
        serialized_joint_acceleration_max = float(
            np.max(np.abs(serialized_joint_acceleration))
        )
        serialized_root_velocity_max = float(
            np.max(np.abs(serialized_root_velocity))
        )
        serialized_root_acceleration_max = float(
            np.max(np.abs(serialized_root_acceleration))
        )
        if serialized_joint_velocity_max > config.max_velocity_rad_s + 1.0e-8:
            raise ValueError("serialized joint path violates velocity limit")
        if (
            serialized_joint_acceleration_max
            > config.max_acceleration_rad_s2 + 1.0e-6
        ):
            raise ValueError("serialized joint path violates acceleration limit")
        if (
            serialized_root_velocity_max
            > config.max_root_offset_velocity_m_s + 1.0e-8
        ):
            raise ValueError("serialized root-offset path violates velocity limit")
        if (
            serialized_root_acceleration_max
            > config.max_root_offset_acceleration_m_s2 + 1.0e-6
        ):
            raise ValueError(
                "serialized root-offset path violates acceleration limit"
            )

        summary = result.summary()
        summary["serialization_constraint_audit_passed"] = True
        summary["constraints"].update(
            {
                "measured_velocity_abs_max_rad_s": (
                    serialized_joint_velocity_max
                ),
                "measured_acceleration_abs_max_rad_s2": (
                    serialized_joint_acceleration_max
                ),
                "measured_root_offset_velocity_abs_max_m_s": (
                    serialized_root_velocity_max
                ),
                "measured_root_offset_acceleration_abs_max_m_s2": (
                    serialized_root_acceleration_max
                ),
                "certificate_basis": SERIALIZATION_CERTIFICATE_BASIS,
            }
        )
        summary.update(expected)
        summary.update(
            {
                "retiming": retiming,
                "cache_disposition": (
                    "fresh" if cache_state == "missing" else f"recomputed: {cache_state}"
                ),
                "expert_gate_passed": False,
                "expert_gate_reason": (
                    "offline kinematic conversion only; fixed-horizon closed-loop "
                    "expert qualification has not run"
                ),
            }
        )
        _publish_triplet(
            outputs,
            motion_arrays=motion_arrays,
            expert_arrays=expert_arrays,
            summary=summary,
        )
        return spec.clip_id, f"ok: {len(source_joints)} frames", summary
    except Exception as exc:  # noqa: BLE001
        return spec.clip_id, f"failed: {type(exc).__name__}: {exc}", None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="CSV file or directory")
    source.add_argument("--manifest", type=Path, help="selected-clip JSON manifest")
    parser.add_argument("--source-root", type=Path, help="override manifest source_root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, default=Path(DEFAULT_SOURCE_MODEL))
    parser.add_argument("--target-model", type=Path, default=Path(DEFAULT_TARGET_MODEL))
    parser.add_argument("--fps-source", type=float, default=120.0)
    parser.add_argument("--fps-target", type=float, default=50.0)
    parser.add_argument("--min-frames", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("--max-velocity", type=float, default=8.0)
    parser.add_argument("--max-acceleration", type=float, default=80.0)
    parser.add_argument(
        "--native-action-clip", type=float, default=SAFE_TARGET_RAW_ACTION_CLIP
    )
    parser.add_argument("--auto-retime", action="store_true")
    parser.add_argument("--retime-velocity-fraction", type=float, default=0.75)
    parser.add_argument("--retime-acceleration-fraction", type=float, default=0.75)
    parser.add_argument("--max-time-scale", type=float, default=8.0)
    parser.add_argument("--report", type=Path, help="optional aggregate JSON report")
    parser.add_argument(
        "--qualification-report",
        type=Path,
        help=(
            "write and enforce held-out Step 1A qualification; requires --manifest"
        ),
    )
    return parser


def _status_kind(status: str) -> str:
    return status.partition(":")[0]


def _aggregate_report(
    *,
    specs: list[ClipSpec],
    selection: dict[str, Any],
    results: dict[str, str],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = {
        kind: sum(_status_kind(status) == kind for status in results.values())
        for kind in ("ok", "skipped", "pending", "failed", "rejected")
    }
    completed_count = counts["ok"] + counts["skipped"]
    return {
        "schema": BATCH_SCHEMA,
        "deployment_ready": False,
        "selection": selection,
        "input_count": len(specs),
        "processed_count": len(results),
        "completed_count": completed_count,
        "ok_count": counts["ok"],
        "skipped_count": counts["skipped"],
        "pending_count": counts["pending"],
        "failed_count": counts["failed"],
        "rejected_count": counts["rejected"],
        "all_complete": completed_count == len(specs),
        "clip_provenance": {
            spec.clip_id: {
                "source_csv": str(spec.csv_path.resolve()),
                "category": spec.category,
                "fps_source": spec.fps_source,
                "qualification_categories": list(spec.qualification_categories),
            }
            for spec in specs
        },
        "results": results,
        "successful_summaries": summaries,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.source_root is not None and args.manifest is None:
        raise SystemExit("--source-root requires --manifest")
    if args.qualification_report is not None and args.manifest is None:
        raise SystemExit("--qualification-report requires --manifest")
    if not args.source_model.is_file():
        raise SystemExit(f"source model not found: {args.source_model}")
    if not args.target_model.is_file():
        raise SystemExit(f"target model not found: {args.target_model}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least one")
    if args.min_frames < 1:
        raise SystemExit("--min-frames must be at least one")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least one")
    for name, value in (
        ("--retime-velocity-fraction", args.retime_velocity_fraction),
        ("--retime-acceleration-fraction", args.retime_acceleration_fraction),
    ):
        if not 0.0 < value <= 1.0:
            raise SystemExit(f"{name} must be in (0, 1]")
    if args.max_time_scale < 1.0:
        raise SystemExit("--max-time-scale must be at least one")
    try:
        fps_source = _positive_fps(args.fps_source, location="--fps-source")
        fps_target = _positive_fps(args.fps_target, location="--fps-target")
        if args.manifest is not None:
            specs, selection = _load_manifest_specs(
                args.manifest,
                cli_source_root=args.source_root,
                default_fps_source=fps_source,
            )
        else:
            specs, selection = _load_input_specs(args.input, fps_source=fps_source)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.limit is not None:
        specs = specs[: args.limit]

    for directory in ("motions", "experts", "reports"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    config_values = {
        "max_iterations": args.max_iterations,
        "max_velocity_rad_s": args.max_velocity,
        "max_acceleration_rad_s2": args.max_acceleration,
        "native_action_clip": args.native_action_clip,
    }
    retiming_values = {
        "enabled": bool(args.auto_retime),
        "velocity_fraction": args.retime_velocity_fraction,
        "acceleration_fraction": args.retime_acceleration_fraction,
        "max_time_scale": args.max_time_scale,
    }
    try:
        RetargetConfig(**config_values)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    source_model_hash = _sha256(args.source_model)
    target_model_hash = _sha256(args.target_model)

    results: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _convert_one,
                spec,
                args.output,
                args.source_model,
                args.target_model,
                source_model_hash,
                target_model_hash,
                fps_target,
                args.min_frames,
                config_values,
                retiming_values,
            ): spec
            for spec in specs
        }
        for processed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            spec = futures[future]
            try:
                clip_id, status, summary = future.result()
                if clip_id != spec.clip_id:
                    raise RuntimeError(
                        f"worker returned clip_id {clip_id!r} for {spec.clip_id!r}"
                    )
            except Exception as exc:  # noqa: BLE001
                clip_id = spec.clip_id
                status = f"failed: worker exception: {type(exc).__name__}: {exc}"
                summary = None
            results[clip_id] = status
            if summary is not None:
                summaries[clip_id] = summary
            print(f"[{processed}/{len(specs)}] {clip_id}: {status}", flush=True)

    aggregate = _aggregate_report(
        specs=specs,
        selection=selection,
        results=results,
        summaries=summaries,
    )
    qualification = None
    if args.qualification_report is not None:
        status_records = {}
        for spec in specs:
            status_kind = _status_kind(results.get(spec.clip_id, "failed: missing"))
            summary = summaries.get(spec.clip_id)
            artifact_was_fresh = (
                isinstance(summary, dict)
                and isinstance(summary.get("cache_disposition"), str)
                and (
                    summary["cache_disposition"] == "fresh"
                    or summary["cache_disposition"].startswith("recomputed:")
                )
            )
            status_records[spec.clip_id] = {
                "completed": status_kind in {"ok", "skipped"},
                "fresh": status_kind == "ok" or artifact_was_fresh,
                "ok": status_kind in {"ok", "skipped"},
                "skipped": status_kind == "skipped" and not artifact_was_fresh,
            }
        metadata = {
            spec.clip_id: {
                "category": spec.category,
                "source_path": str(spec.csv_path.resolve()),
                "independence_key": _source_identity(spec.csv_path),
            }
            for spec in specs
        }
        qualification = build_task_space_qualification_report(
            summaries,
            metadata,
            status_records,
            [spec.clip_id for spec in specs],
            declared_categories=selection["qualification_categories"],
        )
        aggregate["heldout_qualification"] = qualification
        args.qualification_report.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.qualification_report, qualification)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.report, aggregate)
    qualification_passed = (
        qualification is None or qualification["qualification_gate_passed"]
    )
    return 0 if aggregate["all_complete"] and qualification_passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(130) from None
