"""Fail-closed audits for revision-pinned Unitree G1 Inspire episodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import numbers
from pathlib import Path

import numpy as np

from gear_sonic.data.unitree_conversion.contracts import (
    INSPIRE_GATE_REASONS,
    ColumnStatistics,
    DiagnosticReport,
    ScalarExtrema,
    SourceSpec,
    TimestampAudit,
)
from gear_sonic.data.unitree_conversion.lerobot_v3_source import (
    V3DataSchema,
    default_lerobot_cache_base,
    load_pinned_v3_episode,
)
from gear_sonic.data.unitree_conversion.quaternion import validate_wxyz

_CURRENT_KEY = "observation.state.robot_q_current"
_DESIRED_KEY = "action.robot_q_desired"
_HAND_STATE_KEY = "observation.state.hand_state"
_HAND_CMD_KEY = "action.hand_cmd"
_PRIMARY_CAMERA = "observation.images.cam_0"
_SECONDARY_CAMERA = "observation.images.cam_1"
_SUPPORTED_REPO_ID = "unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly"
_SUPPORTED_DATASET_PATH = "G1_WB_Dex5_Pickup_Pillow"

_SOURCE_SCHEMA = V3DataSchema(
    float_vector_columns=(_CURRENT_KEY, _DESIRED_KEY, _HAND_STATE_KEY, _HAND_CMD_KEY),
    float_scalar_columns=("timestamp",),
    integer_scalar_columns=("task_index", "frame_index", "episode_index"),
)


def _names(prefix: str, size: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{index}" for index in range(size))


_EXPECTED_FEATURES: dict[str, tuple[str, tuple[int, ...], tuple[str, ...] | None]] = {
    _PRIMARY_CAMERA: ("video", (480, 640, 3), ("height", "width", "channel")),
    _SECONDARY_CAMERA: ("video", (480, 640, 3), ("height", "width", "channel")),
    "observation.state.ee_state": ("float32", (12,), _names("ee_state", 12)),
    _HAND_STATE_KEY: ("float32", (12,), _names("hand_state", 12)),
    _CURRENT_KEY: ("float32", (36,), _names("robot_q_current", 36)),
    "action.ee_action": ("float32", (12,), _names("ee_action", 12)),
    _HAND_CMD_KEY: ("float32", (12,), _names("hand_cmd", 12)),
    _DESIRED_KEY: ("float32", (36,), _names("robot_q_desired", 36)),
    "timestamp": ("float32", (1,), None),
    "frame_index": ("int64", (1,), None),
    "episode_index": ("int64", (1,), None),
    "index": ("int64", (1,), None),
    "task_index": ("int64", (1,), None),
}

_STATISTIC_KEYS = (
    "robot_q_current",
    "robot_q_desired",
    "hand_state",
    "hand_cmd",
)


def _numeric_array(value: object, *, field_name: str) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a numeric array") from error
    if source.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must be a numeric array")
    array = np.array(source, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def _motion_array(value: object, *, field_name: str, width: int) -> np.ndarray:
    array = _numeric_array(value, field_name=field_name)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{field_name} must have shape [N,{width}]; got {array.shape}")
    return array


def _column_statistics(array: np.ndarray) -> tuple[ColumnStatistics, ...]:
    return tuple(
        ColumnStatistics(
            minimum=float(array[:, column].min()),
            maximum=float(array[:, column].max()),
            mean=float(array[:, column].mean(dtype=np.float64)),
            std=float(array[:, column].std(dtype=np.float64, ddof=0)),
        )
        for column in range(array.shape[1])
    )


def _extrema(values: np.ndarray) -> ScalarExtrema:
    return ScalarExtrema(minimum=float(values.min()), maximum=float(values.max()))


def _root_quaternions_are_valid(arrays: Mapping[str, np.ndarray]) -> bool:
    valid = True
    for field_name in ("robot_q_current", "robot_q_desired"):
        for quaternion in arrays[field_name][:, 3:7]:
            try:
                validate_wxyz(quaternion)
            except ValueError:
                valid = False
    return valid


def _diagnose_arrays(
    *,
    current: object,
    desired: object,
    hand_state: object,
    hand_cmd: object,
    timestamps: object,
    source_repo_id: str | None,
    source_revision: str | None,
    source_episode_id: int | None,
    source_task_indices: tuple[int, ...] | None,
    primary_camera: str | None,
    camera_map: Mapping[str, str],
) -> DiagnosticReport:
    arrays = {
        "robot_q_current": _motion_array(current, field_name="robot_q_current", width=36),
        "robot_q_desired": _motion_array(desired, field_name="robot_q_desired", width=36),
        "hand_state": _motion_array(hand_state, field_name="hand_state", width=12),
        "hand_cmd": _motion_array(hand_cmd, field_name="hand_cmd", width=12),
    }
    row_counts = {array.shape[0] for array in arrays.values()}
    if len(row_counts) != 1:
        raise ValueError("all Inspire motion arrays must have a common row count")
    frame_count = row_counts.pop()
    if frame_count < 2:
        raise ValueError("Inspire diagnostics require at least two rows")

    timeline = _numeric_array(timestamps, field_name="timestamps")
    if timeline.shape != (frame_count,):
        raise ValueError(f"timestamps must have shape [{frame_count}]; got {timeline.shape}")
    steps = np.diff(timeline)
    if not np.all(steps > 0.0):
        raise ValueError("timestamps must be strictly increasing")

    current_norms = np.linalg.norm(arrays["robot_q_current"][:, 3:7], axis=1)
    desired_norms = np.linalg.norm(arrays["robot_q_desired"][:, 3:7], axis=1)
    status = "blocked_unverified" if _root_quaternions_are_valid(arrays) else "source_schema_error"
    return DiagnosticReport(
        status=status,
        gate_reasons=INSPIRE_GATE_REASONS,
        encoder_invoked=False,
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        source_episode_id=source_episode_id,
        frame_count=frame_count,
        timestamp_audit=TimestampAudit(
            finite_count=frame_count,
            strictly_increasing=True,
            start=float(timeline[0]),
            end=float(timeline[-1]),
            minimum_step=float(steps.min()),
            maximum_step=float(steps.max()),
        ),
        finite_counts={
            **{key: int(arrays[key].size) for key in _STATISTIC_KEYS},
            "timestamps": int(timeline.size),
        },
        current_root_quaternion_norm=_extrema(current_norms),
        desired_root_quaternion_norm=_extrema(desired_norms),
        column_statistics={key: _column_statistics(arrays[key]) for key in _STATISTIC_KEYS},
        source_task_indices=source_task_indices,
        primary_camera=primary_camera,
        camera_map=camera_map,
    )


def diagnose_inspire_arrays(
    *,
    current: object,
    desired: object,
    hand_state: object,
    hand_cmd: object,
    timestamps: object,
) -> DiagnosticReport:
    """Audit documented Inspire arrays while keeping all missing semantic gates closed."""
    return _diagnose_arrays(
        current=current,
        desired=desired,
        hand_state=hand_state,
        hand_cmd=hand_cmd,
        timestamps=timestamps,
        source_repo_id=None,
        source_revision=None,
        source_episode_id=None,
        source_task_indices=None,
        primary_camera=None,
        camera_map={},
    )


def _validate_source_spec(source_spec: object, episode_id: object) -> SourceSpec:
    if not isinstance(source_spec, SourceSpec) or not source_spec.approved:
        raise ValueError("source_spec must be an approved SourceSpec")
    if (
        source_spec.repo_id != _SUPPORTED_REPO_ID
        or source_spec.dataset_path != _SUPPORTED_DATASET_PATH
        or source_spec.label != "inspire"
    ):
        raise ValueError("source_spec must match the supported Inspire adapter identity")
    if source_spec.episode_count is None:
        raise ValueError("source_spec must have a pinned episode_count")
    if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id < 0:
        raise ValueError("episode_id must be a nonnegative integer")
    if episode_id >= source_spec.episode_count:
        raise ValueError("episode_id must be smaller than pinned episode_count")
    if episode_id not in source_spec.episodes:
        raise ValueError("episode_id must be selected in source_spec.episodes")
    if source_spec.primary_camera != _PRIMARY_CAMERA or dict(source_spec.camera_map) != {
        _PRIMARY_CAMERA: "diagnostic.primary_camera"
    }:
        raise ValueError("source_spec primary camera mapping must match pinned Inspire metadata")
    return source_spec


def _validate_metadata(source_spec: SourceSpec, metadata: object) -> Mapping[str, object]:
    total_episodes = getattr(metadata, "total_episodes", None)
    if (
        isinstance(total_episodes, bool)
        or not isinstance(total_episodes, numbers.Integral)
        or total_episodes != source_spec.episode_count
    ):
        raise ValueError("dataset metadata total_episodes must exactly match pinned episode_count")
    fps = getattr(metadata, "fps", None)
    if isinstance(fps, bool) or not isinstance(fps, numbers.Real) or not math.isfinite(float(fps)) or fps != 30:
        raise ValueError("dataset metadata fps must be numeric 30")
    features = getattr(metadata, "features", None)
    if not isinstance(features, Mapping):
        raise ValueError("dataset metadata features must be a mapping")
    if set(features) != set(_EXPECTED_FEATURES):
        raise ValueError("dataset metadata feature keys must exactly match the pinned Inspire source")
    for key, (dtype, shape, names) in _EXPECTED_FEATURES.items():
        feature = features[key]
        if not isinstance(feature, Mapping):
            raise ValueError(f"metadata feature {key} must be a mapping")
        try:
            actual_shape = tuple(feature.get("shape"))
        except TypeError as error:
            raise ValueError(f"metadata feature {key} has invalid shape") from error
        actual_names_value = feature.get("names")
        if actual_names_value is None:
            actual_names = None
        elif isinstance(actual_names_value, (str, bytes)):
            actual_names = (actual_names_value,)
        else:
            try:
                actual_names = tuple(actual_names_value)
            except TypeError as error:
                raise ValueError(f"metadata feature {key} has invalid names") from error
        if feature.get("dtype") != dtype or actual_shape != shape or actual_names != names:
            raise ValueError(f"metadata feature {key} must match pinned dtype, shape, and names")
    return features


def _integer(value: object, *, field_name: str, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{field_name} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return result


def _timestamp(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def diagnose_inspire_episode(
    source_spec: SourceSpec,
    episode_id: int,
    root: str | Path | None = None,
    *,
    snapshot_downloader: Callable[..., object] | None = None,
) -> DiagnosticReport:
    """Load and audit one selected exact-revision Inspire episode without creating output."""
    source_spec = _validate_source_spec(source_spec, episode_id)
    dataset = load_pinned_v3_episode(
        source_spec,
        episode_id,
        cache_base=default_lerobot_cache_base() if root is None else root,
        snapshot_downloader=snapshot_downloader,
        schema=_SOURCE_SCHEMA,
        download_videos=False,
    )
    if dataset.revision != source_spec.revision or dataset.meta.revision != source_spec.revision:
        raise ValueError("loaded dataset revision must equal pinned source revision")
    _validate_metadata(source_spec, dataset.meta)

    current: list[object] = []
    desired: list[object] = []
    hand_state: list[object] = []
    hand_cmd: list[object] = []
    timestamps: list[float] = []
    task_indices: list[int] = []
    for row_number, row in enumerate(dataset.hf_dataset):
        source_episode_id = _integer(row["episode_index"], field_name=f"row {row_number} episode_index")
        if source_episode_id != episode_id:
            raise ValueError(f"row {row_number} episode_index does not match requested episode")
        frame_index = _integer(row["frame_index"], field_name=f"row {row_number} frame_index")
        if frame_index != row_number:
            raise ValueError(f"expected frame_index {row_number}, got {frame_index}")
        task_index = _integer(
            row["task_index"],
            field_name=f"row {row_number} task_index",
            nonnegative=True,
        )
        current.append(row[_CURRENT_KEY])
        desired.append(row[_DESIRED_KEY])
        hand_state.append(row[_HAND_STATE_KEY])
        hand_cmd.append(row[_HAND_CMD_KEY])
        timestamps.append(_timestamp(row["timestamp"], field_name=f"row {row_number} timestamp"))
        task_indices.append(task_index)

    return _diagnose_arrays(
        current=current,
        desired=desired,
        hand_state=hand_state,
        hand_cmd=hand_cmd,
        timestamps=timestamps,
        source_repo_id=source_spec.repo_id,
        source_revision=source_spec.revision,
        source_episode_id=episode_id,
        source_task_indices=tuple(task_indices),
        primary_camera=source_spec.primary_camera,
        camera_map=source_spec.camera_map,
    )
