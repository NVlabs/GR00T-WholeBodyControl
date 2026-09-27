"""Strict two-pass video preflight and 30-to-50 Hz RGB streaming."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import math
import numbers
import os
from pathlib import Path
import stat
import string
from types import MappingProxyType
from typing import BinaryIO

import av
import numpy as np

from gear_sonic.data.unitree_conversion.contracts import SourceVideoSegment
from gear_sonic.data.unitree_conversion.resampling import nearest_image_indices

_SOURCE_FPS = 30
_TARGET_FPS = 50
_VALID_CAMERA_STATUSES = frozenset({"exact", "missing", "invalid"})


def _integer_at_least(value: object, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        rendered_minimum = "two" if minimum == 2 else str(minimum)
        raise ValueError(f"{field_name} must be a non-boolean integer of at least {rendered_minimum}")
    return int(value)


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative non-boolean integer")
    return int(value)


def _nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _rgb_size(value: object, *, field_name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be exactly two positive integer values")
    try:
        size = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field_name} must be exactly two positive integer values") from error
    if len(size) != 2 or any(
        isinstance(component, bool) or not isinstance(component, numbers.Integral) or component <= 0
        for component in size
    ):
        raise ValueError(f"{field_name} must be exactly two positive integer values")
    return int(size[0]), int(size[1])


def _sha256_handle(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    try:
        stream.seek(0)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        stream.seek(0)
    except (OSError, ValueError) as error:
        raise ValueError("cannot hash the owned video file handle") from error
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        metadata = path.stat()
    except OSError as error:
        raise ValueError("video path must be an existing regular file") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("video path must be an existing regular file")
    return metadata.st_size, metadata.st_mtime_ns, metadata.st_dev, metadata.st_ino


def _handle_identity(stream: BinaryIO) -> tuple[int, int, int, int]:
    try:
        metadata = os.fstat(stream.fileno())
    except (OSError, ValueError) as error:
        raise ValueError("owned video file handle is not readable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("owned video file handle must reference a regular file")
    return metadata.st_size, metadata.st_mtime_ns, metadata.st_dev, metadata.st_ino


def _verify_bound_resource(
    stream: BinaryIO,
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int],
    expected_sha256: str,
    error_message: str,
) -> None:
    try:
        handle_identity = _handle_identity(stream)
        digest = _sha256_handle(stream)
        path_identity = _file_identity(path)
    except ValueError as error:
        raise ValueError(error_message) from error
    if handle_identity != expected_identity or path_identity != expected_identity or digest != expected_sha256:
        raise ValueError(error_message)


@dataclass(frozen=True)
class VideoTimeline:
    """Immutable result of a complete source-video inspection pass."""

    path: Path
    expected_frames: int
    decoded_frames: int
    source_fps: int
    rgb_size: tuple[int, int]
    sha256: str
    file_size: int
    mtime_ns: int
    device: int
    inode: int
    source_key: str | None
    from_timestamp: float
    to_timestamp: float
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("path must be an absolute pathlib.Path")
        expected = _integer_at_least(self.expected_frames, field_name="expected_frames", minimum=2)
        decoded = _integer_at_least(self.decoded_frames, field_name="decoded_frames", minimum=2)
        if decoded != expected:
            raise ValueError("decoded_frames must equal expected_frames")
        if type(self.source_fps) is not int or self.source_fps != _SOURCE_FPS:
            raise ValueError("source_fps must be integer 30")
        size = _rgb_size(self.rgb_size, field_name="rgb_size")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in string.hexdigits for character in self.sha256)
            or self.sha256 != self.sha256.lower()
        ):
            raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest")
        file_size = _integer_at_least(self.file_size, field_name="file_size", minimum=1)
        mtime_ns = _nonnegative_integer(self.mtime_ns, field_name="mtime_ns")
        device = _nonnegative_integer(self.device, field_name="device")
        inode = _nonnegative_integer(self.inode, field_name="inode")
        if self.source_key is not None:
            _nonempty_string(self.source_key, field_name="source_key")
        timestamps: list[float] = []
        for field_name in ("from_timestamp", "to_timestamp"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"{field_name} must be finite nonnegative")
            timestamps.append(float(value))
        from_timestamp, to_timestamp = timestamps
        if to_timestamp <= from_timestamp:
            raise ValueError("to_timestamp must be strictly larger than from_timestamp")
        start_frame = _nonnegative_integer(self.start_frame, field_name="start_frame")
        end_frame = _nonnegative_integer(self.end_frame, field_name="end_frame")
        if end_frame - start_frame != decoded:
            raise ValueError("video interval end_frame - start_frame must equal decoded_frames")
        if abs(from_timestamp * _SOURCE_FPS - start_frame) > 1e-6:
            raise ValueError("from_timestamp must match start_frame on the 30 Hz grid")
        if abs(to_timestamp * _SOURCE_FPS - end_frame) > 1e-6:
            raise ValueError("to_timestamp must match end_frame on the 30 Hz grid")
        object.__setattr__(self, "expected_frames", expected)
        object.__setattr__(self, "decoded_frames", decoded)
        object.__setattr__(self, "rgb_size", size)
        object.__setattr__(self, "file_size", file_size)
        object.__setattr__(self, "mtime_ns", mtime_ns)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "inode", inode)
        object.__setattr__(self, "from_timestamp", from_timestamp)
        object.__setattr__(self, "to_timestamp", to_timestamp)
        object.__setattr__(self, "start_frame", start_frame)
        object.__setattr__(self, "end_frame", end_frame)


@dataclass(frozen=True)
class ResampledVideoFrame:
    """One target-indexed, owned, deeply read-only RGB frame."""

    target_index: int
    source_index: int
    rgb: np.ndarray

    def __post_init__(self) -> None:
        target_index = _nonnegative_integer(self.target_index, field_name="target_index")
        source_index = _nonnegative_integer(self.source_index, field_name="source_index")
        try:
            rgb = np.asarray(self.rgb)
        except (TypeError, ValueError) as error:
            raise ValueError("rgb must be an HWC RGB uint8 array") from error
        if rgb.dtype != np.dtype(np.uint8) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be an HWC RGB uint8 array")
        if rgb.shape[0] < 1 or rgb.shape[1] < 1:
            raise ValueError("rgb must be an HWC RGB uint8 array")

        shape = rgb.shape
        immutable_bytes = bytes(np.ascontiguousarray(rgb).tobytes(order="C"))
        owned_rgb = np.frombuffer(immutable_bytes, dtype=np.uint8).reshape(shape)
        object.__setattr__(self, "target_index", target_index)
        object.__setattr__(self, "source_index", source_index)
        object.__setattr__(self, "rgb", owned_rgb)


@dataclass(frozen=True)
class CameraStreamReport:
    """Result of preflighting one source camera in one episode."""

    status: str
    timeline: VideoTimeline | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, str):
            raise ValueError("camera report status must be a string")
        if self.status not in _VALID_CAMERA_STATUSES:
            raise ValueError(f"camera report status must be one of {sorted(_VALID_CAMERA_STATUSES)}")
        if self.status == "exact":
            if not isinstance(self.timeline, VideoTimeline):
                raise ValueError("exact camera report requires a timeline")
            if self.reason is not None:
                raise ValueError("exact camera report must not contain a reason")
            return
        if self.timeline is not None:
            raise ValueError("nonexact camera report must not contain a timeline")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("nonexact camera report requires a reason")


@dataclass(frozen=True)
class EpisodeCameraReport:
    """Immutable camera preflight results for one selected source episode."""

    source_repo_id: str
    source_episode_id: int
    source_frame_count: int
    primary_camera: str
    streams: Mapping[str, CameraStreamReport] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_string(self.source_repo_id, field_name="source_repo_id")
        episode_id = _nonnegative_integer(self.source_episode_id, field_name="source_episode_id")
        frame_count = _integer_at_least(
            self.source_frame_count,
            field_name="source_frame_count",
            minimum=2,
        )
        _nonempty_string(self.primary_camera, field_name="primary_camera")
        if not isinstance(self.streams, Mapping):
            raise ValueError("streams must be a mapping")
        streams: dict[str, CameraStreamReport] = {}
        for camera, report in self.streams.items():
            _nonempty_string(camera, field_name="stream camera key")
            if not isinstance(report, CameraStreamReport):
                raise ValueError("every stream value must be a CameraStreamReport")
            if report.status == "exact" and report.timeline.decoded_frames != frame_count:
                raise ValueError("exact timeline decoded_frames must equal source_frame_count")
            streams[camera] = report
        streams = dict(sorted(streams.items()))
        object.__setattr__(self, "source_episode_id", episode_id)
        object.__setattr__(self, "source_frame_count", frame_count)
        object.__setattr__(self, "streams", MappingProxyType(streams))


@dataclass(frozen=True)
class TargetCameraSchema:
    """One immutable, cohort-wide source-to-target camera schema."""

    primary_source: str
    primary_target: str
    source_to_target: Mapping[str, str]
    omission_reasons: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_string(self.primary_source, field_name="primary_source")
        _nonempty_string(self.primary_target, field_name="primary_target")
        if not isinstance(self.source_to_target, Mapping) or not self.source_to_target:
            raise ValueError("source_to_target must be a nonempty mapping")
        source_to_target = dict(self.source_to_target)
        for source, target in source_to_target.items():
            _nonempty_string(source, field_name="source camera key")
            _nonempty_string(target, field_name="target camera key")
        if source_to_target.get(self.primary_source) != self.primary_target:
            raise ValueError("source_to_target must contain the exact primary mapping")
        included = {
            self.primary_source: source_to_target[self.primary_source],
            **{
                source: source_to_target[source]
                for source in sorted(source_to_target)
                if source != self.primary_source
            },
        }
        if len(set(included.values())) != len(included):
            raise ValueError("target camera keys must be unique")
        if not isinstance(self.omission_reasons, Mapping):
            raise ValueError("omission_reasons must be a mapping")
        omissions: dict[str, tuple[str, ...]] = {}
        for source, reasons_value in self.omission_reasons.items():
            _nonempty_string(source, field_name="omitted source camera key")
            if source in included or isinstance(reasons_value, (str, bytes)):
                raise ValueError("omitted cameras must be disjoint and contain reason tuples")
            try:
                reasons = tuple(reasons_value)
            except TypeError as error:
                raise ValueError("omitted cameras must contain reason tuples") from error
            if not reasons or any(not isinstance(reason, str) or not reason for reason in reasons):
                raise ValueError("omitted cameras must contain nonempty reason tuples")
            omissions[source] = reasons
        omissions = dict(sorted(omissions.items()))
        object.__setattr__(self, "source_to_target", MappingProxyType(included))
        object.__setattr__(self, "omission_reasons", MappingProxyType(omissions))

    @property
    def target_features(self) -> tuple[str, ...]:
        return tuple(self.source_to_target.values())


def _video_stream(container: av.container.InputContainer) -> av.video.stream.VideoStream:
    streams = tuple(container.streams.video)
    if len(streams) != 1:
        raise ValueError(f"video file must contain exactly one video stream; got {len(streams)}")
    stream = streams[0]
    rate = stream.average_rate
    if rate is None or Fraction(rate) != _SOURCE_FPS:
        rendered_rate = "unknown" if rate is None else str(rate)
        raise ValueError(f"nominal video fps {rendered_rate} != required 30")
    return stream


def _decode_rgb(frame: av.VideoFrame, *, expected_size: tuple[int, int], frame_index: int) -> np.ndarray:
    try:
        rgb = frame.to_ndarray(format="rgb24")
    except (av.error.FFmpegError, TypeError, ValueError) as error:
        raise ValueError(f"video frame {frame_index} is not RGB-convertible") from error
    width, height = expected_size
    expected_shape = (height, width, 3)
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.dtype(np.uint8):
        raise ValueError(f"video frame {frame_index} is not RGB uint8")
    if rgb.shape != expected_shape:
        raise ValueError(f"video frame {frame_index} RGB shape {rgb.shape} != expected {expected_shape}")
    return rgb


def _frame_global_index(frame: av.VideoFrame) -> int:
    if frame.pts is None or frame.time_base is None:
        raise ValueError("video frame PTS and time_base are required")
    try:
        frame_position = Fraction(frame.pts) * Fraction(frame.time_base) * _SOURCE_FPS
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("video frame PTS must map to the nominal 30 Hz grid") from error
    nearest = round(frame_position)
    if abs(float(frame_position - nearest)) > 1e-6:
        raise ValueError("video frame PTS must map to the nominal 30 Hz grid")
    return int(nearest)


def _segment_rgb_frames(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    *,
    start_frame: int,
    end_frame: int,
    expected_size: tuple[int, int],
) -> Iterator[np.ndarray]:
    if stream.time_base is None:
        raise ValueError("video stream time_base is required")
    seek_position = Fraction(start_frame, _SOURCE_FPS) / Fraction(stream.time_base)
    container.seek(math.floor(seek_position), backward=True, any_frame=False, stream=stream)

    previous_global_index: int | None = None
    expected_global_index = start_frame
    decoded_frames = 0
    for frame in container.decode(stream):
        global_index = _frame_global_index(frame)
        if previous_global_index is not None:
            if global_index <= previous_global_index:
                raise ValueError("video frame PTS must be strictly increasing without duplicates")
            if global_index != previous_global_index + 1:
                raise ValueError("video frame PTS must be consecutive on the nominal 30 Hz grid")
        previous_global_index = global_index
        if global_index < start_frame:
            continue
        if global_index >= end_frame:
            break
        if global_index != expected_global_index:
            raise ValueError(f"video interval frame offset {global_index} != expected {expected_global_index}")
        rgb = _decode_rgb(
            frame,
            expected_size=expected_size,
            frame_index=decoded_frames,
        )
        decoded_frames += 1
        expected_global_index += 1
        yield rgb

    expected_frames = end_frame - start_frame
    if decoded_frames != expected_frames:
        raise ValueError(f"decoded frame count {decoded_frames} != data frame count {expected_frames}")


def inspect_video(
    path: str | Path,
    *,
    expected_frames: object,
    expected_size: object,
    segment: SourceVideoSegment | None = None,
) -> VideoTimeline:
    """Decode an entire pinned 30 Hz source video and return immutable metadata."""
    frame_count = _integer_at_least(expected_frames, field_name="expected_frames", minimum=2)
    size = _rgb_size(expected_size, field_name="expected_size")
    try:
        resolved_path = Path(path).expanduser().resolve(strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("video path must be an existing regular file") from error
    if segment is not None:
        if not isinstance(segment, SourceVideoSegment):
            raise ValueError("segment must be a SourceVideoSegment")
        if segment.path != resolved_path:
            raise ValueError("segment path must equal video path")
        if segment.frame_count != frame_count:
            raise ValueError("segment frame_count must equal expected_frames")
        source_key = segment.source_key
        from_timestamp = segment.from_timestamp
        to_timestamp = segment.to_timestamp
        start_frame = segment.start_frame
        end_frame = segment.end_frame
    else:
        source_key = None
        from_timestamp = 0.0
        to_timestamp = frame_count / _SOURCE_FPS
        start_frame = 0
        end_frame = frame_count
    decoded_frames = 0
    try:
        with resolved_path.open("rb") as source_handle:
            identity = _handle_identity(source_handle)
            digest = _sha256_handle(source_handle)
            if _file_identity(resolved_path) != identity:
                raise ValueError("video file changed during inspection")
            with av.open(source_handle, mode="r") as container:
                stream = _video_stream(container)
                for _ in _segment_rgb_frames(
                    container,
                    stream,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    expected_size=size,
                ):
                    decoded_frames += 1
            _verify_bound_resource(
                source_handle,
                resolved_path,
                expected_identity=identity,
                expected_sha256=digest,
                error_message="video file changed during inspection",
            )
    except ValueError:
        raise
    except (av.error.FFmpegError, OSError) as error:
        raise ValueError(f"video decode failed for {resolved_path}") from error

    if decoded_frames != frame_count:
        raise ValueError(f"decoded frame count {decoded_frames} != data frame count {frame_count}")

    return VideoTimeline(
        path=resolved_path,
        expected_frames=frame_count,
        decoded_frames=decoded_frames,
        source_fps=_SOURCE_FPS,
        rgb_size=size,
        sha256=digest,
        file_size=identity[0],
        mtime_ns=identity[1],
        device=identity[2],
        inode=identity[3],
        source_key=source_key,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def iter_resampled_video(
    timeline: VideoTimeline,
    *,
    target_fps: object = _TARGET_FPS,
) -> Iterator[ResampledVideoFrame]:
    """Stream exact nearest-index RGB; callers must exhaust this integrity-checked iterator."""
    if not isinstance(timeline, VideoTimeline):
        raise ValueError("timeline must be a VideoTimeline")
    if type(target_fps) is not int or target_fps != _TARGET_FPS:
        raise ValueError("target_fps must be integer 50")
    identity = (
        timeline.file_size,
        timeline.mtime_ns,
        timeline.device,
        timeline.inode,
    )
    selected_indices = nearest_image_indices(timeline.decoded_frames)
    current_source_index = -1
    current_rgb: np.ndarray | None = None
    pending_frame: ResampledVideoFrame | None = None

    try:
        with timeline.path.open("rb") as source_handle:
            _verify_bound_resource(
                source_handle,
                timeline.path,
                expected_identity=identity,
                expected_sha256=timeline.sha256,
                error_message="video file changed after inspection",
            )
            with av.open(source_handle, mode="r") as container:
                stream = _video_stream(container)
                decoder = iter(
                    _segment_rgb_frames(
                        container,
                        stream,
                        start_frame=timeline.start_frame,
                        end_frame=timeline.end_frame,
                        expected_size=timeline.rgb_size,
                    )
                )
                for target_index, requested_source_index in enumerate(selected_indices):
                    while current_source_index < int(requested_source_index):
                        try:
                            current_rgb = next(decoder)
                        except StopIteration as error:
                            raise ValueError(
                                f"second-pass decoded frame count {current_source_index + 1} "
                                f"!= inspected frame count {timeline.decoded_frames}"
                            ) from error
                        current_source_index += 1
                    if current_rgb is None:
                        raise ValueError("second-pass decoder produced no selected frame")
                    next_output = ResampledVideoFrame(
                        target_index=target_index,
                        source_index=int(requested_source_index),
                        rgb=current_rgb,
                    )
                    if pending_frame is not None:
                        yield pending_frame
                    pending_frame = next_output

                try:
                    next(decoder)
                except StopIteration:
                    pass
                else:
                    raise ValueError(
                        f"second-pass decoded more than inspected frame count {timeline.decoded_frames}"
                    )
            _verify_bound_resource(
                source_handle,
                timeline.path,
                expected_identity=identity,
                expected_sha256=timeline.sha256,
                error_message="video file changed after inspection",
            )
    except ValueError:
        raise
    except (av.error.FFmpegError, OSError) as error:
        raise ValueError(f"second-pass video decode failed for {timeline.path}") from error
    if pending_frame is None:
        raise ValueError("second-pass decoder produced no target frames")
    yield pending_frame


def _camera_map(value: object, *, primary_camera: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("camera_map must be a mapping")
    camera_map = dict(value)
    if not camera_map:
        raise ValueError("camera_map must not be empty")
    if any(
        not isinstance(source, str) or not source.strip() or not isinstance(target, str) or not target.strip()
        for source, target in camera_map.items()
    ):
        raise ValueError("camera_map keys and values must be nonempty strings")
    if len(set(camera_map.values())) != len(camera_map):
        raise ValueError("target camera keys must be unique")
    if primary_camera not in camera_map:
        raise ValueError("primary_camera must be present in camera_map")
    return camera_map


def _episode_failure(report: EpisodeCameraReport, stream: CameraStreamReport | None) -> str:
    if stream is None:
        status = "missing"
        reason = "source stream absent"
    else:
        status = stream.status
        reason = stream.reason
    return f"{report.source_repo_id} episode {report.source_episode_id}: {status}: {reason}"


def choose_target_camera_schema(
    episode_reports: Sequence[EpisodeCameraReport],
    camera_map: Mapping[str, str],
) -> TargetCameraSchema:
    """Choose one immutable camera feature set for an entire selected cohort."""
    if isinstance(episode_reports, (str, bytes)):
        raise ValueError("episode_reports must be a non-string sequence")
    try:
        reports = tuple(episode_reports)
    except TypeError as error:
        raise ValueError("episode_reports must be a sequence") from error
    if not reports:
        raise ValueError("episode_reports must not be empty")
    if not all(isinstance(report, EpisodeCameraReport) for report in reports):
        raise ValueError("every episode report must be an EpisodeCameraReport")
    identities = tuple((report.source_repo_id, report.source_episode_id) for report in reports)
    if len(set(identities)) != len(identities):
        raise ValueError("episode report identities must be unique")
    primary_camera = reports[0].primary_camera
    if any(report.primary_camera != primary_camera for report in reports[1:]):
        raise ValueError("every episode report must use the same primary_camera")
    configured_map = _camera_map(camera_map, primary_camera=primary_camera)
    ordered_reports = tuple(sorted(reports, key=lambda report: (report.source_repo_id, report.source_episode_id)))

    for report in ordered_reports:
        primary_stream = report.streams.get(primary_camera)
        if primary_stream is None or primary_stream.status != "exact":
            status = "missing" if primary_stream is None else primary_stream.status
            reason = "source stream absent" if primary_stream is None else primary_stream.reason
            raise ValueError(
                f"required primary camera {primary_camera} for {report.source_repo_id} "
                f"episode {report.source_episode_id} failed: {status}: {reason}"
            )

    included: dict[str, str] = {primary_camera: configured_map[primary_camera]}
    omissions: dict[str, tuple[str, ...]] = {}
    for source_camera in sorted(source for source in configured_map if source != primary_camera):
        failures = tuple(
            _episode_failure(report, report.streams.get(source_camera))
            for report in ordered_reports
            if report.streams.get(source_camera) is None or report.streams[source_camera].status != "exact"
        )
        if failures:
            omissions[source_camera] = failures
        else:
            included[source_camera] = configured_map[source_camera]

    return TargetCameraSchema(
        primary_source=primary_camera,
        primary_target=configured_map[primary_camera],
        source_to_target=included,
        omission_reasons=omissions,
    )
