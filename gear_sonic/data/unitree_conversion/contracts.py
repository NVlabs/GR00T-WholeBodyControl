"""Validated contracts for Unitree conversion provenance and canonical data."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

_REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_VALID_SCOPES = frozenset({"smoke", "full"})
_VALID_DIAGNOSTIC_STATUSES = frozenset({"blocked_unverified", "source_schema_error"})
INSPIRE_GATE_REASONS = (
    "missing_authoritative_29_joint_order",
    "missing_robot_q_desired_semantics",
    "missing_inspire_to_dex3_calibration",
)
_DIAGNOSTIC_ARRAY_WIDTHS = {
    "robot_q_current": 36,
    "robot_q_desired": 36,
    "hand_state": 12,
    "hand_cmd": 12,
}


def validate_revision(value: object, *, field_name: str = "revision") -> str:
    """Return an immutable Git revision or raise a contract error."""
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a 40-character immutable revision containing only hexadecimal characters"
        )
    return value


def validate_sha256(value: object, *, field_name: str = "sha256") -> str:
    """Return a SHA-256 digest or raise a contract error."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character SHA-256 containing only hexadecimal characters")
    return value


def _nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _canonical_dataset_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("dataset_path must be '.' or a safe canonical POSIX-relative path")
    if value == ".":
        return value
    path = PurePosixPath(value)
    if (
        value.strip() != value
        or "\\" in value
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
        or any(character in "*?[]" for character in value)
        or path.is_absolute()
        or not path.parts
        or any(part in {".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("dataset_path must be '.' or a safe canonical POSIX-relative path")
    return value


def _finite_float64_array(value: object, *, field_name: str) -> np.ndarray:
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


def _task_index_array(value: object) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("task_indices must contain nonnegative int64-compatible integers") from error
    if source.dtype.kind not in "iu":
        raise ValueError("task_indices must contain nonnegative int64-compatible integers")
    if source.size:
        if np.any(source < 0) or np.any(source > np.iinfo(np.int64).max):
            raise ValueError("task_indices must contain nonnegative int64-compatible integers")
    return np.array(source, dtype=np.int64, order="C", copy=True)


@dataclass(frozen=True)
class SourceVideoSegment:
    """One exact episode interval inside a materialized LeRobot v3 video shard."""

    source_key: str
    path: Path
    from_timestamp: float
    to_timestamp: float
    start_frame: int
    end_frame: int
    frame_count: int

    def __post_init__(self) -> None:
        _nonempty_string(self.source_key, field_name="video segment source_key")
        try:
            candidate = Path(self.path).expanduser()
            if candidate.is_symlink():
                raise ValueError("video segment path must be a materialized regular file")
            path = candidate.resolve(strict=True)
        except (OSError, TypeError, ValueError) as error:
            raise ValueError("video segment path must be a materialized regular file") from error
        if not path.is_file():
            raise ValueError("video segment path must be a materialized regular file")

        timestamps: list[float] = []
        for field_name in ("from_timestamp", "to_timestamp"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"video segment {field_name} must be finite nonnegative")
            timestamp = float(value)
            if not math.isfinite(timestamp) or timestamp < 0.0:
                raise ValueError(f"video segment {field_name} must be finite nonnegative")
            timestamps.append(timestamp)
        from_timestamp, to_timestamp = timestamps
        if to_timestamp <= from_timestamp:
            raise ValueError("video segment to_timestamp must be strictly larger than from_timestamp")

        indices: list[int] = []
        for field_name in ("start_frame", "end_frame", "frame_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"video segment {field_name} must be a nonnegative integer")
            indices.append(value)
        start_frame, end_frame, frame_count = indices
        if end_frame <= start_frame:
            raise ValueError("video segment end_frame must be strictly larger than start_frame")
        if frame_count <= 0 or end_frame - start_frame != frame_count:
            raise ValueError("video segment frame_count must equal end_frame - start_frame")

        rounded_start = round(from_timestamp * 30.0)
        rounded_end = round(to_timestamp * 30.0)
        if abs(from_timestamp * 30.0 - rounded_start) > 1e-6:
            raise ValueError("video segment from_timestamp must map to an exact 30 Hz frame index")
        if abs(to_timestamp * 30.0 - rounded_end) > 1e-6:
            raise ValueError("video segment to_timestamp must map to an exact 30 Hz frame index")
        if start_frame != rounded_start:
            raise ValueError("video segment start_frame must match from_timestamp at 30 Hz")
        if end_frame != rounded_end:
            raise ValueError("video segment end_frame must match to_timestamp at 30 Hz")

        object.__setattr__(self, "path", path)
        object.__setattr__(self, "from_timestamp", from_timestamp)
        object.__setattr__(self, "to_timestamp", to_timestamp)


@dataclass
class CanonicalEpisode:
    """Mutable, array-owning canonical representation of one source episode."""

    source_repo_id: str
    source_revision: str
    source_episode_id: int
    source_fps: int
    timestamps: np.ndarray
    task_indices: np.ndarray
    observed_root_wxyz: np.ndarray
    reference_root_wxyz: np.ndarray
    observed_body_q: np.ndarray
    desired_body_q: np.ndarray
    observed_left_hand: np.ndarray
    observed_right_hand: np.ndarray
    desired_left_hand: np.ndarray
    desired_right_hand: np.ndarray
    video_segments: Mapping[str, SourceVideoSegment] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_string(self.source_repo_id, field_name="source_repo_id")
        validate_revision(self.source_revision, field_name="source_revision")
        if (
            isinstance(self.source_episode_id, bool)
            or not isinstance(self.source_episode_id, int)
            or self.source_episode_id < 0
        ):
            raise ValueError("source_episode_id must be a nonnegative integer")
        if isinstance(self.source_fps, bool) or not isinstance(self.source_fps, int) or self.source_fps != 30:
            raise ValueError("source_fps must be exactly integer 30")

        timestamps = _finite_float64_array(self.timestamps, field_name="timestamps")
        if timestamps.ndim != 1:
            raise ValueError(f"timestamps must have shape (N,); got {timestamps.shape}")
        row_count = timestamps.shape[0]
        if row_count < 2:
            raise ValueError("canonical episodes require at least two rows")
        if not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("timestamps must be strictly increasing")

        task_indices = _task_index_array(self.task_indices)
        if task_indices.shape != (row_count,):
            raise ValueError(f"task_indices must have shape ({row_count},); got {task_indices.shape}")

        expected_shapes = {
            "observed_root_wxyz": (row_count, 4),
            "reference_root_wxyz": (row_count, 4),
            "observed_body_q": (row_count, 29),
            "desired_body_q": (row_count, 29),
            "observed_left_hand": (row_count, 7),
            "observed_right_hand": (row_count, 7),
            "desired_left_hand": (row_count, 7),
            "desired_right_hand": (row_count, 7),
        }
        copied_arrays: dict[str, np.ndarray] = {}
        for field_name, expected_shape in expected_shapes.items():
            array = _finite_float64_array(getattr(self, field_name), field_name=field_name)
            if array.shape != expected_shape:
                raise ValueError(f"{field_name} must have shape {expected_shape}; got {array.shape}")
            copied_arrays[field_name] = array

        if not isinstance(self.video_segments, Mapping):
            raise ValueError("video_segments must be a mapping")
        video_segments: dict[str, SourceVideoSegment] = {}
        for source_key, segment in self.video_segments.items():
            _nonempty_string(source_key, field_name="video segment key")
            if not isinstance(segment, SourceVideoSegment):
                raise ValueError("every video segment must be a SourceVideoSegment")
            if source_key != segment.source_key:
                raise ValueError("video segment key must equal segment.source_key")
            if segment.frame_count != row_count:
                raise ValueError("video segment frame_count must equal canonical episode row count")
            video_segments[source_key] = segment

        self.timestamps = timestamps
        self.task_indices = task_indices
        self.video_segments = MappingProxyType(dict(sorted(video_segments.items())))
        for field_name, array in copied_arrays.items():
            setattr(self, field_name, array)


@dataclass
class ResampledEpisode:
    """Owned 50 Hz representation of one episode, before token generation."""

    source_repo_id: str
    source_revision: str
    source_episode_id: int
    body_joint_names: tuple[str, ...]
    task_indices: np.ndarray
    task_texts: tuple[str, ...]
    observed_root_wxyz: np.ndarray
    reference_root_wxyz: np.ndarray
    observed_body_q: np.ndarray
    desired_body_q: np.ndarray
    desired_body_velocity: np.ndarray
    observed_left_hand: np.ndarray
    observed_right_hand: np.ndarray
    desired_left_hand: np.ndarray
    desired_right_hand: np.ndarray

    def __post_init__(self) -> None:
        from gear_sonic.data.unitree_conversion.quaternion import validate_wxyz

        _nonempty_string(self.source_repo_id, field_name="source_repo_id")
        validate_revision(self.source_revision, field_name="source_revision")
        if (
            isinstance(self.source_episode_id, bool)
            or not isinstance(self.source_episode_id, int)
            or self.source_episode_id < 0
        ):
            raise ValueError("source_episode_id must be a nonnegative integer")

        if isinstance(self.body_joint_names, (str, bytes)):
            raise ValueError("body_joint_names must contain exactly 29 unique semantic names")
        try:
            body_joint_names = tuple(self.body_joint_names)
        except TypeError as error:
            raise ValueError("body_joint_names must contain exactly 29 unique semantic names") from error
        if len(body_joint_names) != 29 or any(
            not isinstance(name, str) or not name.strip() for name in body_joint_names
        ):
            raise ValueError("body_joint_names must contain exactly 29 unique semantic names")
        if len(set(body_joint_names)) != 29:
            raise ValueError("body_joint_names must contain exactly 29 unique semantic names")

        observed_roots = _finite_float64_array(
            self.observed_root_wxyz,
            field_name="observed_root_wxyz",
        )
        if observed_roots.ndim != 2 or observed_roots.shape[1] != 4:
            raise ValueError(f"observed_root_wxyz must have shape (T50, 4); got {observed_roots.shape}")
        frame_count = observed_roots.shape[0]
        if frame_count < 2:
            raise ValueError("resampled episodes require at least two 50 Hz frames")

        expected_shapes = {
            "reference_root_wxyz": (frame_count, 4),
            "observed_body_q": (frame_count, 29),
            "desired_body_q": (frame_count, 29),
            "desired_body_velocity": (frame_count, 29),
            "observed_left_hand": (frame_count, 7),
            "observed_right_hand": (frame_count, 7),
            "desired_left_hand": (frame_count, 7),
            "desired_right_hand": (frame_count, 7),
        }
        arrays = {"observed_root_wxyz": observed_roots}
        for field_name, expected_shape in expected_shapes.items():
            array = _finite_float64_array(getattr(self, field_name), field_name=field_name)
            if array.shape != expected_shape:
                raise ValueError(f"{field_name} must have shape {expected_shape}; got {array.shape}")
            arrays[field_name] = array

        for field_name in ("observed_root_wxyz", "reference_root_wxyz"):
            roots = arrays[field_name]
            for frame_index in range(frame_count):
                roots[frame_index], _ = validate_wxyz(roots[frame_index])

        task_indices = _task_index_array(self.task_indices)
        if task_indices.shape != (frame_count,):
            raise ValueError(f"task_indices must have shape ({frame_count},); got {task_indices.shape}")
        if isinstance(self.task_texts, (str, bytes)):
            raise ValueError(f"task_texts must contain exactly {frame_count} strings")
        try:
            task_texts = tuple(self.task_texts)
        except TypeError as error:
            raise ValueError(f"task_texts must contain exactly {frame_count} strings") from error
        if len(task_texts) != frame_count:
            raise ValueError(f"task_texts must contain exactly {frame_count} strings")
        for text in task_texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("task_texts must contain nonempty UTF-8 strings")
            try:
                text.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("task_texts must contain nonempty UTF-8 strings") from error

        self.body_joint_names = body_joint_names
        self.task_indices = task_indices
        self.task_texts = task_texts
        for field_name, array in arrays.items():
            setattr(self, field_name, array)

    @property
    def frame_count(self) -> int:
        """Number of target frames in this isolated episode."""
        return int(self.observed_root_wxyz.shape[0])

    @property
    def target_fps(self) -> int:
        """The fixed deployment timeline frequency."""
        return 50


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


@dataclass(frozen=True)
class ColumnStatistics:
    """Deterministic float64 population statistics for one source column."""

    minimum: float
    maximum: float
    mean: float
    std: float

    def __post_init__(self) -> None:
        minimum = _finite_number(self.minimum, field_name="column minimum")
        maximum = _finite_number(self.maximum, field_name="column maximum")
        mean = _finite_number(self.mean, field_name="column mean")
        std = _finite_number(self.std, field_name="column std")
        if minimum > maximum or not minimum <= mean <= maximum:
            raise ValueError("column statistics must satisfy minimum <= mean <= maximum")
        if std < 0.0:
            raise ValueError("column std must be nonnegative")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)


@dataclass(frozen=True)
class ScalarExtrema:
    """Finite minimum and maximum of one audited scalar series."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        minimum = _finite_number(self.minimum, field_name="extrema minimum")
        maximum = _finite_number(self.maximum, field_name="extrema maximum")
        if minimum > maximum:
            raise ValueError("extrema minimum must not exceed maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)


@dataclass(frozen=True)
class TimestampAudit:
    """Audit of an accepted finite, strictly increasing source timeline."""

    finite_count: int
    strictly_increasing: bool
    start: float
    end: float
    minimum_step: float
    maximum_step: float

    def __post_init__(self) -> None:
        if isinstance(self.finite_count, bool) or not isinstance(self.finite_count, int) or self.finite_count < 2:
            raise ValueError("timestamp finite_count must be an integer of at least two")
        if self.strictly_increasing is not True:
            raise ValueError("accepted timestamp audit must be strictly increasing")
        start = _finite_number(self.start, field_name="timestamp start")
        end = _finite_number(self.end, field_name="timestamp end")
        minimum_step = _finite_number(self.minimum_step, field_name="timestamp minimum_step")
        maximum_step = _finite_number(self.maximum_step, field_name="timestamp maximum_step")
        if end <= start or minimum_step <= 0.0 or maximum_step < minimum_step:
            raise ValueError("timestamp audit values are inconsistent with a strictly increasing timeline")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "minimum_step", minimum_step)
        object.__setattr__(self, "maximum_step", maximum_step)


@dataclass(frozen=True)
class DiagnosticReport:
    """Immutable, fail-closed audit of one unverified Inspire episode."""

    status: str
    gate_reasons: tuple[str, ...]
    encoder_invoked: bool
    source_repo_id: str | None
    source_revision: str | None
    source_episode_id: int | None
    frame_count: int
    timestamp_audit: TimestampAudit
    finite_counts: Mapping[str, int]
    current_root_quaternion_norm: ScalarExtrema
    desired_root_quaternion_norm: ScalarExtrema
    column_statistics: Mapping[str, tuple[ColumnStatistics, ...]]
    source_task_indices: tuple[int, ...] | None = None
    primary_camera: str | None = None
    camera_map: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_DIAGNOSTIC_STATUSES:
            raise ValueError(f"Inspire diagnostic status must be one of {sorted(_VALID_DIAGNOSTIC_STATUSES)}")
        if tuple(self.gate_reasons) != INSPIRE_GATE_REASONS:
            raise ValueError("Inspire diagnostic gate_reasons must equal the exact semantic gate reasons")
        object.__setattr__(self, "gate_reasons", INSPIRE_GATE_REASONS)
        if self.encoder_invoked is not False:
            raise ValueError("Inspire diagnostics require encoder_invoked to be false")
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int) or self.frame_count < 2:
            raise ValueError("diagnostic frame_count must be an integer of at least two")
        if not isinstance(self.timestamp_audit, TimestampAudit):
            raise ValueError("timestamp_audit must be a TimestampAudit")
        if self.timestamp_audit.finite_count != self.frame_count:
            raise ValueError("timestamp audit count must equal frame_count")
        if not isinstance(self.current_root_quaternion_norm, ScalarExtrema) or not isinstance(
            self.desired_root_quaternion_norm, ScalarExtrema
        ):
            raise ValueError("root quaternion audits must be ScalarExtrema values")

        identities = (self.source_repo_id, self.source_revision, self.source_episode_id)
        if any(value is None for value in identities) and not all(value is None for value in identities):
            raise ValueError("source identity fields must be either all present or all absent")
        if self.source_repo_id is not None:
            _nonempty_string(self.source_repo_id, field_name="source_repo_id")
            validate_revision(self.source_revision, field_name="source_revision")
            if (
                isinstance(self.source_episode_id, bool)
                or not isinstance(self.source_episode_id, int)
                or self.source_episode_id < 0
            ):
                raise ValueError("source_episode_id must be a nonnegative integer")
            if self.source_task_indices is None:
                raise ValueError("source_task_indices are required with source identity")
        elif self.source_task_indices is not None:
            raise ValueError("source_task_indices require source identity")

        if self.source_task_indices is not None:
            if isinstance(self.source_task_indices, (str, bytes)):
                raise ValueError("source_task_indices must be an integer tuple")
            try:
                task_indices = tuple(self.source_task_indices)
            except TypeError as error:
                raise ValueError("source_task_indices must be an integer tuple") from error
            if len(task_indices) != self.frame_count or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in task_indices
            ):
                raise ValueError("source_task_indices must contain one nonnegative integer per frame")
            object.__setattr__(self, "source_task_indices", task_indices)

        if not isinstance(self.finite_counts, Mapping):
            raise ValueError("finite_counts must be a mapping")
        finite_counts = dict(self.finite_counts)
        expected_count_keys = (*_DIAGNOSTIC_ARRAY_WIDTHS, "timestamps")
        if tuple(finite_counts) != expected_count_keys:
            raise ValueError("finite_counts must use the exact diagnostic field order")
        for key, width in _DIAGNOSTIC_ARRAY_WIDTHS.items():
            if finite_counts[key] != self.frame_count * width:
                raise ValueError(f"finite count for {key} does not match frame_count")
        if finite_counts["timestamps"] != self.frame_count:
            raise ValueError("finite count for timestamps does not match frame_count")
        object.__setattr__(self, "finite_counts", MappingProxyType(finite_counts))

        if not isinstance(self.column_statistics, Mapping):
            raise ValueError("column_statistics must be a mapping")
        statistics = {key: tuple(value) for key, value in self.column_statistics.items()}
        if tuple(statistics) != tuple(_DIAGNOSTIC_ARRAY_WIDTHS):
            raise ValueError("column_statistics must use the exact diagnostic field order")
        for key, width in _DIAGNOSTIC_ARRAY_WIDTHS.items():
            if len(statistics[key]) != width or not all(
                isinstance(item, ColumnStatistics) for item in statistics[key]
            ):
                raise ValueError(f"column statistics for {key} must contain exactly {width} entries")
        object.__setattr__(self, "column_statistics", MappingProxyType(statistics))

        if self.primary_camera is None:
            if self.camera_map:
                raise ValueError("camera_map must be empty when primary_camera is absent")
            object.__setattr__(self, "camera_map", MappingProxyType({}))
        else:
            _nonempty_string(self.primary_camera, field_name="primary_camera")
            if not isinstance(self.camera_map, Mapping):
                raise ValueError("camera_map must be a mapping")
            camera_map = dict(self.camera_map)
            if self.primary_camera not in camera_map:
                raise ValueError("primary_camera must be present in camera_map")
            for source_key, target_key in camera_map.items():
                _nonempty_string(source_key, field_name="camera_map source key")
                _nonempty_string(target_key, field_name="camera_map target key")
            object.__setattr__(self, "camera_map", MappingProxyType(camera_map))

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-safe diagnostic payload."""
        return {
            "status": self.status,
            "gate_reasons": list(self.gate_reasons),
            "encoder_invoked": self.encoder_invoked,
            "source_repo_id": self.source_repo_id,
            "source_revision": self.source_revision,
            "source_episode_id": self.source_episode_id,
            "source_task_indices": (None if self.source_task_indices is None else list(self.source_task_indices)),
            "frame_count": self.frame_count,
            "timestamp_audit": {
                "finite_count": self.timestamp_audit.finite_count,
                "strictly_increasing": self.timestamp_audit.strictly_increasing,
                "start": self.timestamp_audit.start,
                "end": self.timestamp_audit.end,
                "minimum_step": self.timestamp_audit.minimum_step,
                "maximum_step": self.timestamp_audit.maximum_step,
            },
            "finite_counts": dict(self.finite_counts),
            "current_root_quaternion_norm": {
                "minimum": self.current_root_quaternion_norm.minimum,
                "maximum": self.current_root_quaternion_norm.maximum,
            },
            "desired_root_quaternion_norm": {
                "minimum": self.desired_root_quaternion_norm.minimum,
                "maximum": self.desired_root_quaternion_norm.maximum,
            },
            "column_statistics": {
                key: [
                    {
                        "minimum": item.minimum,
                        "maximum": item.maximum,
                        "mean": item.mean,
                        "std": item.std,
                    }
                    for item in values
                ]
                for key, values in self.column_statistics.items()
            },
            "primary_camera": self.primary_camera,
            "camera_map": dict(self.camera_map),
        }


@dataclass(frozen=True)
class ArtifactSpec:
    """Byte-exact identity of an artifact in an immutable repository revision."""

    repo_id: str
    revision: str
    filename: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _nonempty_string(self.repo_id, field_name="repo_id")
        validate_revision(self.revision)
        _nonempty_string(self.filename, field_name="filename")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("artifact size must be a positive integer")
        validate_sha256(self.sha256)


@dataclass(frozen=True)
class SourceSpec:
    """Immutable dataset identity plus operator-reviewed conversion selection."""

    approved: bool
    repo_id: str
    revision: str
    dataset_path: str = "."
    episode_count: int | None = None
    episodes: tuple[int, ...] = ()
    primary_camera: str | None = None
    camera_map: Mapping[str, str] = field(default_factory=dict)
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise ValueError("approved must be a boolean")
        _nonempty_string(self.repo_id, field_name="repo_id")
        validate_revision(self.revision)
        object.__setattr__(self, "dataset_path", _canonical_dataset_path(self.dataset_path))
        if self.label is not None:
            _nonempty_string(self.label, field_name="source label")

        episode_count = self.episode_count
        if episode_count is not None and (
            isinstance(episode_count, bool) or not isinstance(episode_count, int) or episode_count <= 0
        ):
            raise ValueError("episode_count must be a positive integer when provided")

        try:
            episodes = tuple(self.episodes)
        except TypeError as error:
            raise ValueError("episodes must be an iterable of integer episode IDs") from error
        if any(isinstance(item, bool) or not isinstance(item, int) for item in episodes):
            raise ValueError("every episode ID must be an integer")
        if any(item < 0 for item in episodes):
            raise ValueError("every episode ID must be nonnegative")
        if tuple(sorted(set(episodes))) != episodes:
            raise ValueError("episode IDs must be unique and strictly increasing")
        if episode_count is None and episodes:
            raise ValueError("episode_count is required when episode IDs are provided")
        if episode_count is not None and any(item >= episode_count for item in episodes):
            raise ValueError("every episode ID must be smaller than episode_count")
        object.__setattr__(self, "episodes", episodes)

        if self.primary_camera is not None:
            _nonempty_string(self.primary_camera, field_name="primary_camera")
        if not isinstance(self.camera_map, Mapping):
            raise ValueError("camera_map must be a mapping of source keys to target keys")
        camera_map = dict(self.camera_map)
        for source_key, target_key in camera_map.items():
            _nonempty_string(source_key, field_name="camera_map source key")
            _nonempty_string(target_key, field_name="camera_map target key")
        object.__setattr__(self, "camera_map", MappingProxyType(camera_map))

        if self.approved:
            if episode_count is None or not episodes:
                raise ValueError("approved sources require episode_count and at least one episode ID")
            if self.primary_camera is None:
                raise ValueError("approved sources require a primary_camera")
            if self.primary_camera not in camera_map:
                raise ValueError("primary_camera must be present in camera_map")


@dataclass(frozen=True)
class CollectionMembership:
    """Ordered dataset membership observed in one Hugging Face collection."""

    slug: str
    repo_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.slug, field_name="collection slug")
        if isinstance(self.repo_ids, (str, bytes)):
            raise ValueError("collection repo_ids must be a non-string iterable")
        try:
            repo_ids = tuple(self.repo_ids)
        except TypeError as error:
            raise ValueError("collection repo_ids must be an iterable") from error
        for repo_id in repo_ids:
            _nonempty_string(repo_id, field_name="collection repository ID")
        if len(set(repo_ids)) != len(repo_ids):
            raise ValueError("collection membership contains duplicate repositories")
        object.__setattr__(self, "repo_ids", repo_ids)


@dataclass(frozen=True)
class SourceLock:
    """Revision-locked artifact and ordered dataset provenance."""

    version: int
    scope: str
    semantic_repo_commit: str
    encoder: ArtifactSpec
    observation_config: ArtifactSpec
    sources: tuple[SourceSpec, ...]
    collections: tuple[CollectionMembership, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ValueError("source lock version must be an integer")
        if self.version != 1:
            raise ValueError(f"unsupported source lock version: {self.version!r}")
        if not isinstance(self.scope, str) or self.scope not in _VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_VALID_SCOPES)}, got {self.scope!r}")
        validate_revision(
            self.semantic_repo_commit,
            field_name="semantic_repo_commit",
        )
        if not isinstance(self.encoder, ArtifactSpec):
            raise ValueError("encoder must be an ArtifactSpec")
        if not isinstance(self.observation_config, ArtifactSpec):
            raise ValueError("observation_config must be an ArtifactSpec")
        for field_name in ("repo_id", "revision"):
            if getattr(self.encoder, field_name) != getattr(self.observation_config, field_name):
                raise ValueError(f"encoder and observation_config must share the same {field_name}")

        try:
            sources = tuple(self.sources)
            collections = tuple(self.collections)
        except TypeError as error:
            raise ValueError("sources and collections must be iterable") from error
        if not all(isinstance(source, SourceSpec) for source in sources):
            raise ValueError("sources must contain only SourceSpec entries")
        if not all(isinstance(membership, CollectionMembership) for membership in collections):
            raise ValueError("collections must contain only CollectionMembership entries")
        repo_ids = tuple(source.repo_id for source in sources)
        if len(set(repo_ids)) != len(repo_ids):
            raise ValueError("source lock contains duplicate dataset repositories")
        labels = tuple(source.label for source in sources if source.label is not None)
        if len(set(labels)) != len(labels):
            raise ValueError("source lock contains duplicate source labels")

        if self.scope == "full":
            if not sources:
                raise ValueError("full source locks require at least one source")
            if not collections:
                raise ValueError("full source locks require at least one collection membership")

        if collections:
            collection_slugs = tuple(membership.slug for membership in collections)
            if len(set(collection_slugs)) != len(collection_slugs):
                raise ValueError("source lock contains duplicate collection slugs")
            recorded_repo_ids = tuple(repo_id for membership in collections for repo_id in membership.repo_ids)
            if recorded_repo_ids != repo_ids:
                raise ValueError("ordered collection membership must exactly match ordered sources")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "collections", collections)

        if self.scope == "smoke":
            for label in ("dex3", "inspire"):
                if label not in labels:
                    raise ValueError(f"smoke source lock is missing {label!r}")

    def source(self, label: str) -> SourceSpec:
        """Return a named smoke source."""
        for source in self.sources:
            if source.label == label:
                return source
        raise AttributeError(f"source lock has no source named {label!r}")

    @property
    def dex3(self) -> SourceSpec:
        return self.source("dex3")

    @property
    def inspire(self) -> SourceSpec:
        return self.source("inspire")
