"""Deployment-equivalent 30-to-50 Hz motion and image resampling."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import numpy as np

from gear_sonic.data.unitree_conversion.quaternion import (
    _quat_slerp_deployment_normalized,
    validate_wxyz,
)

_SOURCE_FPS = 30
_TARGET_FPS = 50
_TIMESTAMP_WARNING_SECONDS = 0.001
_TIMESTAMP_REJECTION_SECONDS = 1.0 / 60.0


@dataclass(frozen=True)
class TimestampAuditReport:
    """Deterministic diagnostic comparison against the nominal 30 Hz grid."""

    frame_count: int
    max_grid_error_seconds: float
    warning: bool
    rejected: bool

    def __post_init__(self) -> None:
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int) or self.frame_count < 2:
            raise ValueError("frame_count must be a non-boolean integer of at least two")
        if (
            isinstance(self.max_grid_error_seconds, bool)
            or not isinstance(self.max_grid_error_seconds, numbers.Real)
            or not math.isfinite(float(self.max_grid_error_seconds))
            or self.max_grid_error_seconds < 0.0
        ):
            raise ValueError("max_grid_error_seconds must be a finite nonnegative real number")
        if type(self.warning) is not bool or type(self.rejected) is not bool:
            raise ValueError("warning and rejected flags must be built-in bool values")

        max_error = float(self.max_grid_error_seconds)
        expected_rejected = max_error > _TIMESTAMP_REJECTION_SECONDS
        expected_warning = max_error > _TIMESTAMP_WARNING_SECONDS or expected_rejected
        if self.warning is not expected_warning or self.rejected is not expected_rejected:
            raise ValueError("warning and rejected flags must agree with the strict timestamp thresholds")
        object.__setattr__(self, "max_grid_error_seconds", max_error)


def _real_numeric_array(value: object, *, field_name: str) -> tuple[np.ndarray, np.dtype]:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a real numeric array") from error
    if source.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must be a real numeric array")
    result = np.array(source, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return result, source.dtype


def _motion_array(value: object, *, field_name: str) -> tuple[np.ndarray, np.dtype]:
    result, source_dtype = _real_numeric_array(value, field_name=field_name)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 1:
        raise ValueError(f"{field_name} must have shape [N,D] with N>=2 and D>=1; got {result.shape}")
    return result, source_dtype


def _target_frame_count(source_frame_count: int) -> int:
    # Preserve the deployed C++ binary64 evaluation order. It differs from
    # integer floor(N * 50 / 30) at values such as N=69.
    motion_seconds = float(source_frame_count) / float(_SOURCE_FPS)
    return math.floor(motion_seconds * float(_TARGET_FPS))


def _sampling_coordinates(source_frame_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_frame_count = _target_frame_count(source_frame_count)
    target_indices = np.arange(target_frame_count, dtype=np.int64)
    # The division-before-multiplication order is also intentional: at j=205,
    # deployment evaluates f30 immediately below 123 and therefore uses f0=122.
    source_coordinates = (target_indices.astype(np.float64) / float(_TARGET_FPS)) * float(_SOURCE_FPS)
    source_index_0 = np.floor(source_coordinates).astype(np.int64)
    source_index_1 = np.minimum(source_index_0 + 1, source_frame_count - 1)
    alphas = source_coordinates - source_index_0
    return source_index_0, source_index_1, alphas


def resample_positions(positions: object) -> tuple[np.ndarray, np.ndarray]:
    """Resample one finite 30 Hz position episode and derive stored 50 Hz velocity."""
    source, _ = _motion_array(positions, field_name="positions")
    source_index_0, source_index_1, alphas = _sampling_coordinates(source.shape[0])
    weight_0 = 1.0 - alphas
    weight_1 = 1.0 - weight_0
    resampled = np.array(
        weight_0[:, None] * source[source_index_0] + weight_1[:, None] * source[source_index_1],
        dtype=np.float64,
        order="C",
        copy=True,
    )

    velocities = np.empty_like(resampled, order="C")
    velocities[:-1] = 50.0 * (resampled[1:] - resampled[:-1])
    velocities[-1] = velocities[-2]
    return resampled, velocities


def resample_quaternions(quaternions: object) -> np.ndarray:
    """Resample one 30 Hz WXYZ root-orientation episode with deployment SLERP."""
    source, _ = _real_numeric_array(quaternions, field_name="quaternions")
    if source.ndim != 2 or source.shape[0] < 2 or source.shape[1] != 4:
        raise ValueError(f"quaternions must have shape [N,4] with N>=2; got {source.shape}")

    normalized = np.empty(source.shape, dtype=np.float64, order="C")
    for index, quaternion in enumerate(source):
        normalized[index], _ = validate_wxyz(quaternion)

    source_index_0, source_index_1, alphas = _sampling_coordinates(source.shape[0])
    result = np.empty((alphas.shape[0], 4), dtype=np.float64, order="C")
    for target_index, (index_0, index_1, alpha) in enumerate(
        zip(source_index_0, source_index_1, alphas, strict=True)
    ):
        result[target_index] = _quat_slerp_deployment_normalized(
            normalized[index_0],
            normalized[index_1],
            alpha,
        )
    return np.array(result, dtype=np.float64, order="C", copy=True)


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{field_name} must be a positive non-boolean integer")
    return int(value)


def clamped_future_indices(frame_count: object, width: object = 10) -> np.ndarray:
    """Return episode-local future indices, clamped at the final frame."""
    count = _positive_integer(frame_count, field_name="frame_count")
    window_width = _positive_integer(width, field_name="width")
    rows = np.arange(count, dtype=np.int64)[:, None]
    offsets = np.arange(window_width, dtype=np.int64)[None, :]
    return np.array(np.minimum(rows + offsets, count - 1), dtype=np.int64, order="C", copy=True)


def future_windows(
    q50: object,
    dq50: object,
    width: object = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather owned pose and velocity windows without crossing an episode boundary."""
    positions, positions_dtype = _real_numeric_array(q50, field_name="q50")
    velocities, velocities_dtype = _real_numeric_array(dq50, field_name="dq50")
    if positions_dtype != velocities_dtype:
        raise ValueError("q50 and dq50 must have the same source dtype")
    if positions.shape != velocities.shape:
        raise ValueError("q50 and dq50 must have exactly the same shape")
    if positions.ndim != 2 or positions.shape[0] < 1 or positions.shape[1] < 1:
        raise ValueError("q50 and dq50 must have shape [T,D] with T>=1 and D>=1")

    indices = clamped_future_indices(positions.shape[0], width=width)
    position_windows = np.array(positions[indices], dtype=np.float64, order="C", copy=True)
    velocity_windows = np.array(velocities[indices], dtype=np.float64, order="C", copy=True)
    return position_windows, velocity_windows


def nearest_image_indices(source_frame_count: object) -> np.ndarray:
    """Map each 50 Hz frame to a 30 Hz image using round-half-up and clamp."""
    count = _positive_integer(source_frame_count, field_name="source frame_count")
    if count < 2:
        raise ValueError("source frame_count must be at least two")
    target_indices = np.arange(_target_frame_count(count), dtype=np.int64)
    rounded = (target_indices * (2 * _SOURCE_FPS) + _TARGET_FPS) // (2 * _TARGET_FPS)
    return np.array(np.minimum(rounded, count - 1), dtype=np.int64, order="C", copy=True)


def audit_source_timestamps(timestamps: object) -> TimestampAuditReport:
    """Audit timestamps against 30 Hz while leaving resampling on the nominal frame grid."""
    timeline, _ = _real_numeric_array(timestamps, field_name="timestamps")
    if timeline.ndim != 1 or timeline.shape[0] < 2:
        raise ValueError(f"timestamps must have shape [N] with N>=2; got {timeline.shape}")
    if not np.all(np.diff(timeline) > 0.0):
        raise ValueError("timestamps must be strictly increasing")

    relative = timeline - timeline[0]
    nominal = np.arange(timeline.shape[0], dtype=np.float64) / float(_SOURCE_FPS)
    max_error = float(np.max(np.abs(relative - nominal)))
    rejected = max_error > _TIMESTAMP_REJECTION_SECONDS
    warning = max_error > _TIMESTAMP_WARNING_SECONDS or rejected
    return TimestampAuditReport(
        frame_count=timeline.shape[0],
        max_grid_error_seconds=max_error,
        warning=warning,
        rejected=rejected,
    )
