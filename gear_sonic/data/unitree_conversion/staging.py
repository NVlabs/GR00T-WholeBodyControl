"""Immutable per-episode staging and deterministic Unitree dataset merge."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import tempfile
from types import MappingProxyType
from typing import BinaryIO, Iterator, Protocol
from urllib.parse import quote

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gear_sonic.data.unitree_conversion.contracts import validate_revision, validate_sha256
from gear_sonic.data.unitree_conversion.validation import (
    TARGET_FRAME_SCHEMA,
    TARGET_JOINT_NAMES,
    TARGET_VIDEO_KEYS,
    ValidationReport,
    probe_video_50fps_stream,
    validate_target_rows,
    validation_report_from_inspections,
)

_STAGE_SCHEMA_VERSION = "unitree-sonic-stage-v1"
_MERGE_SCHEMA_VERSION = "unitree-sonic-merge-v1"
_VIDEO_KEY_PATTERN = re.compile(r"^observation\.images\.[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REPO_PART_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CANONICAL_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_CANONICAL_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_CONVERSION_SCRIPT_CONFIG = {"unitree_conversion_schema": _MERGE_SCHEMA_VERSION}
_VIDEO_FEATURE_INFO = {
    "video.height": 480,
    "video.width": 640,
    "video.codec": "h264",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "video.fps": 50,
    "video.channels": 3,
    "has_audio": False,
}


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_source_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("source file paths must be safe canonical POSIX-relative paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError("source file paths must be safe canonical POSIX-relative paths")
    return value


def _validated_sha(value: object, *, field_name: str) -> str:
    digest = validate_sha256(value, field_name=field_name)
    if digest != digest.lower():
        raise ValueError(f"{field_name} must use lowercase hexadecimal")
    return digest


@dataclass(frozen=True)
class StageIdentity:
    """All immutable inputs that give one episode stage its identity."""

    source_repo_id: str
    source_revision: str
    source_episode_id: int
    source_file_sha256: Mapping[str, str]
    conversion_config_sha256: str
    converter_version: str
    encoder_sha256: str
    encoder_config_sha256: str
    source_lock_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_repo_id, str) or not self.source_repo_id.strip():
            raise ValueError("source_repo_id must be a nonempty repository identifier")
        parts = self.source_repo_id.split("/")
        if len(parts) < 2 or any(_REPO_PART_PATTERN.fullmatch(part) is None for part in parts):
            raise ValueError("source_repo_id must be a safe slash-separated repository identifier")
        revision = validate_revision(self.source_revision, field_name="source_revision")
        if revision != revision.lower():
            raise ValueError("source_revision must use lowercase hexadecimal")
        if type(self.source_episode_id) is not int or self.source_episode_id < 0:
            raise ValueError("source_episode_id must be a nonnegative integer")
        if not isinstance(self.source_file_sha256, Mapping) or not self.source_file_sha256:
            raise ValueError("source_file_sha256 must contain exact source file hashes")
        files: dict[str, str] = {}
        for source_path, digest in self.source_file_sha256.items():
            path = _safe_source_path(source_path)
            files[path] = _validated_sha(digest, field_name=f"source file {path} SHA-256")
        if not isinstance(self.converter_version, str) or not self.converter_version.strip():
            raise ValueError("converter_version must be a nonempty string")
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in self.converter_version):
            raise ValueError("converter_version must not contain control characters")
        object.__setattr__(self, "source_file_sha256", MappingProxyType(dict(sorted(files.items()))))
        for field_name in (
            "conversion_config_sha256",
            "encoder_sha256",
            "encoder_config_sha256",
            "source_lock_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _validated_sha(getattr(self, field_name), field_name=field_name),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_repo_id": self.source_repo_id,
            "source_revision": self.source_revision,
            "source_episode_id": self.source_episode_id,
            "source_file_sha256": dict(self.source_file_sha256),
            "conversion_config_sha256": self.conversion_config_sha256,
            "converter_version": self.converter_version,
            "encoder_sha256": self.encoder_sha256,
            "encoder_config_sha256": self.encoder_config_sha256,
            "source_lock_sha256": self.source_lock_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> StageIdentity:
        if not isinstance(value, dict):
            raise ValueError("stage identity must be an object")
        expected = {
            "source_repo_id",
            "source_revision",
            "source_episode_id",
            "source_file_sha256",
            "conversion_config_sha256",
            "converter_version",
            "encoder_sha256",
            "encoder_config_sha256",
            "source_lock_sha256",
        }
        if set(value) != expected:
            raise ValueError("stage identity contains missing or unexpected fields")
        return cls(**value)

    @property
    def conversion_key(self) -> tuple[object, ...]:
        return (
            self.source_repo_id,
            self.source_revision,
            self.conversion_config_sha256,
            self.converter_version,
            self.encoder_sha256,
            self.encoder_config_sha256,
            self.source_lock_sha256,
        )


def _immutable_array(value: np.ndarray) -> np.ndarray:
    data = bytes(np.ascontiguousarray(value).tobytes(order="C"))
    return np.frombuffer(data, dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class StagePayload:
    """Detached typed target rows and explicit target-camera video sources."""

    rows: Sequence[Mapping[str, object]]
    videos: Mapping[str, Path | bytes]

    def __post_init__(self) -> None:
        if isinstance(self.rows, (str, bytes)):
            raise ValueError("stage rows must be a sequence of target mappings")
        try:
            rows = tuple(self.rows)
        except TypeError as error:
            raise ValueError("stage rows must be a sequence of target mappings") from error
        validate_target_rows(rows)
        owned_rows: list[Mapping[str, object]] = []
        for row in rows:
            owned: dict[str, object] = {}
            for key in TARGET_FRAME_SCHEMA:
                value = row[key]
                owned[key] = _immutable_array(value) if isinstance(value, np.ndarray) else value
            owned_rows.append(MappingProxyType(owned))

        if not isinstance(self.videos, Mapping) or not self.videos:
            raise ValueError("stage videos must be a nonempty mapping")
        videos: dict[str, Path | bytes] = {}
        for key, source in self.videos.items():
            if not isinstance(key, str) or _VIDEO_KEY_PATTERN.fullmatch(key) is None:
                raise ValueError("video target keys must be safe observation.images.* feature names")
            if key not in TARGET_VIDEO_KEYS:
                raise ValueError(f"video target key {key!r} is unsupported")
            if isinstance(source, bytes):
                if not source:
                    raise ValueError(f"video target {key} bytes must be nonempty")
                videos[key] = bytes(source)
            elif isinstance(source, Path):
                if source.is_symlink():
                    raise ValueError(f"video target {key} must be a regular source file, not a symlink")
                try:
                    path = source.absolute()
                    metadata = path.stat()
                except OSError as error:
                    raise ValueError(f"video target {key} must be a readable regular file") from error
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(f"video target {key} must be a readable regular file")
                videos[key] = path
            else:
                raise ValueError(f"video target {key} must be supplied as pathlib.Path or bytes")
        if "observation.images.ego_view" not in videos:
            raise ValueError("stage videos require observation.images.ego_view")
        object.__setattr__(self, "rows", tuple(owned_rows))
        object.__setattr__(self, "videos", MappingProxyType(dict(sorted(videos.items()))))


@dataclass(frozen=True)
class StageResult:
    path: Path
    identity: StageIdentity
    manifest_digest: str
    reused: bool = False


@dataclass(frozen=True)
class StagedParquetArtifact:
    """A checksum-bound frame parquet inside a revalidated immutable stage."""

    path: Path
    sha256: str
    size: int
    row_count: int


@dataclass(frozen=True)
class StagedVideoArtifact:
    """A checksum-bound target video inside a revalidated immutable stage."""

    path: Path
    sha256: str
    size: int


@dataclass(frozen=True)
class MergeEpisode:
    path: Path
    identity: StageIdentity
    frame_data: StagedParquetArtifact
    videos: Mapping[str, StagedVideoArtifact]
    stage_manifest_digest: str
    stage_checksums_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "videos", MappingProxyType(dict(sorted(self.videos.items()))))

    @property
    def row_count(self) -> int:
        return self.frame_data.row_count


@dataclass(frozen=True)
class MergeVerification:
    """Observed final-dataset facts returned by a merge backend."""

    source_episode_ids: tuple[int, ...]
    episode_lengths: tuple[int, ...]
    total_frames: int
    task_to_index: Mapping[str, int]
    frame_content_sha256: str
    video_frame_counts: Mapping[int, Mapping[str, int]]
    stats_source_episode_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_to_index", MappingProxyType(dict(self.task_to_index)))
        _validated_sha(self.frame_content_sha256, field_name="frame_content_sha256")
        object.__setattr__(
            self,
            "video_frame_counts",
            MappingProxyType(
                {
                    key: MappingProxyType(dict(sorted(value.items())))
                    for key, value in sorted(self.video_frame_counts.items())
                }
            ),
        )


class MergeBackend(Protocol):
    def __call__(
        self,
        output: Path,
        episodes: tuple[MergeEpisode, ...],
        task_catalog: tuple[str, ...],
    ) -> None: ...


class FinalValidator(Protocol):
    def __call__(
        self,
        output: Path,
        episodes: tuple[MergeEpisode, ...],
        task_catalog: tuple[str, ...],
    ) -> MergeVerification: ...


def _canonical_path(value: str | Path, *, field_name: str, strict: bool) -> Path:
    raw = Path(value)
    if ".." in raw.parts:
        raise ValueError(f"{field_name} must be canonical and must not contain '..' aliases")
    absolute = raw.absolute()
    try:
        resolved = absolute.resolve(strict=strict)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{field_name} must resolve to a canonical path") from error
    if resolved != absolute:
        raise ValueError(f"{field_name} must be canonical and must not contain path aliases")
    return resolved


def stage_path(output_root: str | Path, identity: StageIdentity) -> Path:
    """Return the deterministic immutable path for an episode identity."""
    if not isinstance(identity, StageIdentity):
        raise TypeError("identity must be a StageIdentity")
    repo_safe = quote(identity.source_repo_id, safe="-._")
    return (
        _canonical_path(output_root, field_name="stage output root", strict=False)
        / ".staging"
        / repo_safe
        / identity.source_revision
        / f"episode-{identity.source_episode_id}"
        / identity.conversion_config_sha256
    )


@contextmanager
def _open_bound_file(path: Path) -> Iterator[BinaryIO]:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact is not a regular file: {path.name}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise ValueError(f"artifact changed while opening: {path.name}")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                yield stream
        finally:
            os.close(descriptor)
        final = path.lstat()
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        ):
            raise ValueError(f"artifact changed while reading: {path.name}")
    except OSError as error:
        raise ValueError(f"cannot securely read artifact: {path}") from error


def _hash_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def _secure_read(path: Path) -> bytes:
    with _open_bound_file(path) as stream:
        return stream.read()


def _hash_secure_file(path: Path) -> str:
    with _open_bound_file(path) as stream:
        return _hash_stream(stream)


def _copy_bound_file(source: Path, destination: Path) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    output_descriptor = os.open(destination, flags, 0o644)
    digest = hashlib.sha256()
    size = 0
    try:
        with _open_bound_file(source) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
                view = memoryview(block)
                while view:
                    written = os.write(output_descriptor, view)
                    view = view[written:]
        os.fsync(output_descriptor)
    finally:
        os.close(output_descriptor)
    return digest.hexdigest(), size


def _replace_with_bound_file(source: StagedVideoArtifact, destination: Path) -> None:
    """Atomically replace an exporter video with its checksum-bound stage artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with _open_bound_file(source.path) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
                view = memoryview(block)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if digest.hexdigest() != source.sha256 or size != source.size:
            raise ValueError("staged video changed during immutable publication copy")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(temporary):
            temporary.unlink()


def _inspect_bound_video(path: Path, *, expected_sha256: str, field_name: str) -> tuple[int, tuple[int, int]]:
    with _open_bound_file(path) as stream:
        if _hash_stream(stream) != expected_sha256:
            raise ValueError(f"{field_name} checksum differs before decode")
        inspection = probe_video_50fps_stream(stream, field_name=field_name)
        if _hash_stream(stream) != expected_sha256:
            raise ValueError(f"{field_name} checksum differs after decode")
        return inspection


def _inspect_bound_video_media_info(path: Path, *, expected_sha256: str) -> dict[str, object]:
    """Read the LeRobot video-info fields from one checksum-bound video inode."""
    try:
        with _open_bound_file(path) as stream:
            if _hash_stream(stream) != expected_sha256:
                raise ValueError("exported video checksum differs before media-info inspection")
            with av.open(stream, mode="r") as container:
                video_streams = tuple(container.streams.video)
                if len(video_streams) != 1:
                    raise ValueError("exported video media info requires exactly one video stream")
                video = video_streams[0]
                info = {
                    "video.height": int(video.height),
                    "video.width": int(video.width),
                    "video.codec": video.codec.canonical_name,
                    "video.pix_fmt": video.pix_fmt,
                    "video.is_depth_map": False,
                    "video.fps": int(video.base_rate),
                    "video.channels": 3 if video.pix_fmt == "yuv420p" else None,
                    "has_audio": bool(container.streams.audio),
                }
            if _hash_stream(stream) != expected_sha256:
                raise ValueError("exported video checksum differs after media-info inspection")
            return info
    except av.error.FFmpegError as error:
        raise ValueError("exported video media info is unreadable") from error


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(path: Path) -> None:
    directories = [path]
    for root, directory_names, file_names in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in file_names:
            candidate = root_path / name
            metadata = candidate.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("publication tree must contain only regular files and directories")
            descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for name in directory_names:
            candidate = root_path / name
            if not stat.S_ISDIR(candidate.lstat().st_mode):
                raise ValueError("publication tree must contain only regular files and directories")
            directories.append(candidate)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _ensure_safe_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while not os.path.lexists(cursor):
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if os.path.lexists(cursor) and not stat.S_ISDIR(cursor.lstat().st_mode):
        raise ValueError(f"path component is not a real directory: {cursor}")
    for component in reversed(missing):
        component.mkdir()
        _fsync_directory(component)
        _fsync_directory(component.parent)
    cursor = path
    while True:
        if not stat.S_ISDIR(cursor.lstat().st_mode):
            raise ValueError(f"path component is not a real directory: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Invoke Linux renameat2(RENAME_NOREPLACE), failing closed if unavailable."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameat2(RENAME_NOREPLACE) is unavailable") from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"immutable publication target appeared concurrently: {destination}")
    if error_number in {errno.ENOSYS, errno.EINVAL}:
        raise OSError(error_number, "renameat2(RENAME_NOREPLACE) is unavailable", destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _atomic_publish(source: Path, destination: Path) -> None:
    """Atomically rename a sibling directory without replacing any existing target."""
    _rename_noreplace(source, destination)


def _arrow_type(spec) -> pa.DataType:
    scalar = pa.from_numpy_dtype(spec.dtype)
    return pa.list_(scalar, list_size=spec.shape[0])


def _expected_stage_arrow_schema() -> pa.Schema:
    fields: list[pa.Field] = []
    for key, spec in TARGET_FRAME_SCHEMA.items():
        if key == "task":
            field_type = pa.string()
        elif key == "timestamp":
            field_type = pa.float32()
        else:
            field_type = _arrow_type(spec)
        fields.append(pa.field(key, field_type, nullable=False))
    return pa.schema(fields)


def _rows_to_parquet(rows: Sequence[Mapping[str, object]]) -> bytes:
    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    for key, spec in TARGET_FRAME_SCHEMA.items():
        if key == "task":
            array = pa.array([row[key] for row in rows], type=pa.string())
            field_type = pa.string()
        elif key == "timestamp":
            array = pa.array([row[key] for row in rows], type=pa.float32())
            field_type = pa.float32()
        else:
            assert spec is not None
            stacked = np.stack([row[key] for row in rows])
            values = pa.array(stacked.reshape(-1), type=pa.from_numpy_dtype(spec.dtype))
            array = pa.FixedSizeListArray.from_arrays(values, spec.shape[0])
            field_type = _arrow_type(spec)
        arrays.append(array)
        fields.append(pa.field(key, field_type, nullable=False))
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="NONE", use_dictionary=False, write_statistics=True)
    return sink.getvalue().to_pybytes()


def _arrow_batch_to_rows(batch: pa.RecordBatch) -> tuple[Mapping[str, object], ...]:
    table = pa.Table.from_batches([batch])
    columns: dict[str, object] = {}
    for key, spec in TARGET_FRAME_SCHEMA.items():
        column = table.column(key).combine_chunks()
        if key == "task":
            columns[key] = column.to_pylist()
        elif key == "timestamp":
            columns[key] = column.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        else:
            assert spec is not None
            values = column.values.to_numpy(zero_copy_only=False).reshape(table.num_rows, *spec.shape)
            columns[key] = values.astype(spec.dtype, copy=False)
    rows: list[Mapping[str, object]] = []
    for index in range(table.num_rows):
        row: dict[str, object] = {}
        for key in TARGET_FRAME_SCHEMA:
            values = columns[key]
            if key == "task":
                row[key] = values[index]
            elif key == "timestamp":
                row[key] = np.float32(values[index])
            else:
                row[key] = np.array(values[index], copy=True)
        rows.append(MappingProxyType(row))
    return tuple(rows)


def iter_stage_rows(
    episode: MergeEpisode | StagedParquetArtifact,
    *,
    batch_size: int = 128,
) -> Iterator[Mapping[str, object]]:
    """Stream exact validated target rows from a checksum-bound parquet artifact."""
    artifact = episode.frame_data if isinstance(episode, MergeEpisode) else episode
    if not isinstance(artifact, StagedParquetArtifact):
        raise TypeError("episode must be a MergeEpisode or StagedParquetArtifact")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    with _open_bound_file(artifact.path) as stream:
        if os.fstat(stream.fileno()).st_size != artifact.size or _hash_stream(stream) != artifact.sha256:
            raise ValueError("frame-data.parquet changed before streaming")
        parquet = pq.ParquetFile(stream)
        if parquet.schema_arrow != _expected_stage_arrow_schema():
            raise ValueError("frame-data.parquet schema must match deterministic target order and types")
        if parquet.metadata.num_rows != artifact.row_count or artifact.row_count < 1:
            raise ValueError("frame-data.parquet row count differs from its immutable stage manifest")
        offset = 0
        initial_root_fields: tuple[np.ndarray, np.ndarray] | None = None
        for batch in parquet.iter_batches(batch_size=batch_size):
            rows = _arrow_batch_to_rows(batch)
            initial_root_fields = validate_target_rows(
                rows,
                start_index=offset,
                initial_root_fields=initial_root_fields,
            )
            yield from rows
            offset += len(rows)
        if offset != artifact.row_count:
            raise ValueError("frame-data.parquet streamed row count differs from its manifest")
        if _hash_stream(stream) != artifact.sha256:
            raise ValueError("frame-data.parquet changed after streaming")


def _identity_manifest(identity: StageIdentity) -> dict[str, object]:
    return identity.to_dict()


def _artifact_paths(root: Path) -> tuple[str, ...]:
    artifacts: list[str] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory_name in directory_names:
            if not stat.S_ISDIR((current_path / directory_name).lstat().st_mode):
                raise ValueError("stage contains a symlink or non-directory artifact")
        for file_name in file_names:
            candidate = current_path / file_name
            if not stat.S_ISREG(candidate.lstat().st_mode):
                raise ValueError("stage contains a symlink or non-file artifact")
            artifacts.append(candidate.relative_to(root).as_posix())
    return tuple(sorted(artifacts))


def _reject_symlink_components(path: Path) -> None:
    cursor = path.absolute()
    while True:
        try:
            metadata = cursor.lstat()
        except OSError as error:
            raise ValueError(f"publication path component is missing: {cursor}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"publication path must not contain symlink components: {cursor}")
        if cursor.parent == cursor:
            return
        cursor = cursor.parent


def _checksums_bytes(root: Path, artifacts: Iterable[str]) -> bytes:
    lines = [f"{_hash_secure_file(root / artifact)}  {artifact}\n" for artifact in sorted(artifacts)]
    return "".join(lines).encode("utf-8")


def _parse_json_canonical(data: bytes, *, artifact: str) -> dict[str, object]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{artifact} must contain canonical JSON") from error
    if not isinstance(value, dict) or _canonical_json(value) != data:
        raise ValueError(f"{artifact} must contain canonical JSON")
    return value


@dataclass(frozen=True)
class _VerifiedStage:
    manifest: Mapping[str, object]
    identity: StageIdentity
    frame_data: StagedParquetArtifact
    videos: Mapping[str, StagedVideoArtifact]
    report: ValidationReport
    manifest_digest: str
    checksums_digest: str


def _verify_checksum_tree(root: Path, *, checksum_name: str) -> None:
    _reject_symlink_components(root)
    if not root.exists() or not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("publication root must be a real directory")
    actual_artifacts = _artifact_paths(root)
    if checksum_name not in actual_artifacts:
        raise ValueError(f"{checksum_name} is missing")
    checksum_data = _secure_read(root / checksum_name)
    try:
        text = checksum_data.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{checksum_name} must be ASCII") from error
    entries: list[tuple[str, str]] = []
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n") or len(line) < 67 or line[64:66] != "  ":
            raise ValueError(f"{checksum_name} is malformed")
        digest, relative = line[:64], line[66:-1]
        _validated_sha(digest, field_name=f"{checksum_name} digest")
        _safe_source_path(relative)
        entries.append((relative, digest))
    expected_artifacts = tuple(path for path in actual_artifacts if path != checksum_name)
    if tuple(path for path, _ in entries) != expected_artifacts:
        raise ValueError(f"{checksum_name} must list every artifact exactly once in sorted order")
    if len(set(path for path, _ in entries)) != len(entries):
        raise ValueError(f"{checksum_name} contains duplicate artifacts")
    if any(_hash_secure_file(root / path) != digest for path, digest in entries):
        raise ValueError(f"{checksum_name} does not match artifact bytes")
    if _checksums_bytes(root, expected_artifacts) != checksum_data:
        raise ValueError(f"{checksum_name} is not canonical")


def _verify_stage(path: Path, expected_identity: StageIdentity | None = None) -> _VerifiedStage:
    _verify_checksum_tree(path, checksum_name="checksums.sha256")
    manifest_data = _secure_read(path / "manifest.json")
    manifest = _parse_json_canonical(manifest_data, artifact="manifest.json")
    if set(manifest) != {"schema_version", "identity", "row_count", "columns", "frame_data", "videos"}:
        raise ValueError("manifest.json contains missing or unexpected fields")
    if manifest["schema_version"] != _STAGE_SCHEMA_VERSION:
        raise ValueError("manifest.json schema version is unsupported")
    identity = StageIdentity.from_dict(manifest["identity"])
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("stage identity does not match")
    expected_columns = [
        {
            "name": key,
            "dtype": "string" if key == "task" else "float32" if key == "timestamp" else str(spec.dtype),
            "shape": [] if spec is None else list(spec.shape),
        }
        for key, spec in TARGET_FRAME_SCHEMA.items()
    ]
    if manifest["columns"] != expected_columns or manifest["frame_data"] != "frame-data.parquet":
        raise ValueError("manifest.json does not describe the exact staged frame schema")
    row_count = manifest["row_count"]
    if type(row_count) is not int or row_count < 1:
        raise ValueError("manifest row_count must be a positive integer")
    parquet_path = path / "frame-data.parquet"
    frame_data = StagedParquetArtifact(
        path=parquet_path,
        sha256=_hash_secure_file(parquet_path),
        size=parquet_path.lstat().st_size,
        row_count=row_count,
    )
    if sum(1 for _ in iter_stage_rows(frame_data)) != row_count:
        raise ValueError("manifest row_count differs from frame-data.parquet")
    video_manifest = manifest["videos"]
    if not isinstance(video_manifest, dict):
        raise ValueError("manifest videos must be an object")
    video_artifacts: dict[str, StagedVideoArtifact] = {}
    video_inspections: dict[str, tuple[int, tuple[int, int]]] = {}
    for key, metadata in sorted(video_manifest.items()):
        if key not in TARGET_VIDEO_KEYS or not isinstance(metadata, dict):
            raise ValueError("manifest contains an unsupported video target")
        expected_artifact = f"videos/{key}.mp4"
        if set(metadata) != {"artifact", "sha256", "size", "frame_count", "fps", "width", "height"}:
            raise ValueError("manifest video metadata contains missing or unexpected fields")
        if metadata["artifact"] != expected_artifact or metadata["fps"] != 50:
            raise ValueError("manifest video artifact or fps is invalid")
        artifact_path = path / expected_artifact
        if (
            metadata["sha256"] != _hash_secure_file(artifact_path)
            or metadata["size"] != artifact_path.stat().st_size
        ):
            raise ValueError("manifest video digest or size differs from artifact")
        video_inspections[key] = _inspect_bound_video(
            artifact_path,
            expected_sha256=metadata["sha256"],
            field_name=f"video {key}",
        )
        video_artifacts[key] = StagedVideoArtifact(
            path=artifact_path,
            sha256=metadata["sha256"],
            size=metadata["size"],
        )
    report = ValidationReport(
        success=True,
        row_count=row_count,
        field_names=tuple(TARGET_FRAME_SCHEMA),
        video_frame_counts={key: inspection[0] for key, inspection in video_inspections.items()},
        video_dimensions={key: inspection[1] for key, inspection in video_inspections.items()},
    )
    for key in report.video_frame_counts:
        metadata = video_manifest[key]
        width, height = report.video_dimensions[key]
        if metadata["frame_count"] != report.video_frame_counts[key] or (
            metadata["width"],
            metadata["height"],
        ) != (width, height):
            raise ValueError("manifest video structure differs from decoded artifact")
    validation_data = _secure_read(path / "validation.json")
    if _parse_json_canonical(validation_data, artifact="validation.json") != report.to_dict():
        raise ValueError("validation.json differs from complete structural validation")
    expected_artifacts = {
        "checksums.sha256",
        "manifest.json",
        "validation.json",
        "frame-data.parquet",
        *(f"videos/{key}.mp4" for key in video_artifacts),
    }
    if set(_artifact_paths(path)) != expected_artifacts:
        raise ValueError("stage contains missing or unexpected artifacts")
    return _VerifiedStage(
        manifest=MappingProxyType(manifest),
        identity=identity,
        frame_data=frame_data,
        videos=MappingProxyType(video_artifacts),
        report=report,
        manifest_digest=_sha256(manifest_data),
        checksums_digest=_sha256(_secure_read(path / "checksums.sha256")),
    )


def can_resume(path: str | Path, identity: StageIdentity) -> bool:
    """Return true only for a complete, exact, non-symlink immutable stage."""
    if not isinstance(identity, StageIdentity):
        return False
    try:
        _verify_stage(Path(path).absolute(), identity)
    except (OSError, TypeError, ValueError):
        return False
    return True


def write_stage(output_root: str | Path, identity: StageIdentity, payload: StagePayload) -> StageResult:
    """Validate and atomically publish one immutable episode stage."""
    if not isinstance(identity, StageIdentity):
        raise TypeError("identity must be a StageIdentity")
    if not isinstance(payload, StagePayload):
        raise TypeError("payload must be a StagePayload")
    final = stage_path(output_root, identity)
    _ensure_safe_directory(final.parent)
    if os.path.lexists(final):
        if can_resume(final, identity):
            verified = _verify_stage(final, identity)
            return StageResult(final, identity, verified.manifest_digest, reused=True)
        raise FileExistsError(f"immutable stage already exists but is incomplete or mismatched: {final}")

    temp = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        validate_target_rows(payload.rows)
        parquet_data = _rows_to_parquet(payload.rows)
        _write_file(temp / "frame-data.parquet", parquet_data)
        video_manifest: dict[str, dict[str, object]] = {}
        video_inspections: dict[str, tuple[int, tuple[int, int]]] = {}
        for key, source in payload.videos.items():
            artifact = f"videos/{key}.mp4"
            artifact_path = temp / artifact
            if isinstance(source, bytes):
                _write_file(artifact_path, source)
                digest = _sha256(source)
                size = len(source)
            else:
                digest, size = _copy_bound_file(source, artifact_path)
            video_inspections[key] = _inspect_bound_video(
                artifact_path,
                expected_sha256=digest,
                field_name=f"video {key}",
            )
        report = validation_report_from_inspections(payload.rows, video_inspections)
        for key in payload.videos:
            artifact = f"videos/{key}.mp4"
            artifact_path = temp / artifact
            digest = _hash_secure_file(artifact_path)
            size = artifact_path.stat().st_size
            width, height = report.video_dimensions[key]
            video_manifest[key] = {
                "artifact": artifact,
                "sha256": digest,
                "size": size,
                "frame_count": report.video_frame_counts[key],
                "fps": 50,
                "width": width,
                "height": height,
            }
        manifest = {
            "schema_version": _STAGE_SCHEMA_VERSION,
            "identity": _identity_manifest(identity),
            "row_count": len(payload.rows),
            "columns": [
                {
                    "name": key,
                    "dtype": "string" if key == "task" else "float32" if key == "timestamp" else str(spec.dtype),
                    "shape": [] if spec is None else list(spec.shape),
                }
                for key, spec in TARGET_FRAME_SCHEMA.items()
            ],
            "frame_data": "frame-data.parquet",
            "videos": video_manifest,
        }
        _write_file(temp / "manifest.json", _canonical_json(manifest))
        _write_file(temp / "validation.json", _canonical_json(report.to_dict()))
        artifacts = _artifact_paths(temp)
        _write_file(temp / "checksums.sha256", _checksums_bytes(temp, artifacts))
        _fsync_tree(temp)
        verified = _verify_stage(temp, identity)
        _atomic_publish(temp, final)
        _fsync_directory(final.parent)
        return StageResult(final, identity, verified.manifest_digest, reused=False)
    except BaseException:
        if os.path.lexists(temp):
            shutil.rmtree(temp)
        raise


def load_stage_rows(path: str | Path) -> tuple[Mapping[str, object], ...]:
    """Return fully revalidated, detached rows from one immutable stage."""
    return tuple(iter_stage_rows(_verify_stage(Path(path).absolute()).frame_data))


def _stage_input(value: StageResult | str | Path) -> MergeEpisode:
    if isinstance(value, StageResult):
        path = _canonical_path(value.path, field_name="stage input path", strict=True)
        expected_identity = value.identity
    elif isinstance(value, (str, Path)):
        path = _canonical_path(value, field_name="stage input path", strict=True)
        expected_identity = None
    else:
        raise TypeError("merge stage inputs must be StageResult or paths")
    try:
        verified = _verify_stage(path, expected_identity)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"stage is not complete and exact: {path}") from error
    if isinstance(value, StageResult) and value.manifest_digest != verified.manifest_digest:
        raise ValueError(f"stage is not complete and exact: {path}")
    return MergeEpisode(
        path=path,
        identity=verified.identity,
        frame_data=verified.frame_data,
        videos=verified.videos,
        stage_manifest_digest=verified.manifest_digest,
        stage_checksums_digest=verified.checksums_digest,
    )


def _merge_manifest(episodes: tuple[MergeEpisode, ...], task_catalog: tuple[str, ...]) -> dict[str, object]:
    first = episodes[0].identity
    return {
        "schema_version": _MERGE_SCHEMA_VERSION,
        "source_repo_id": first.source_repo_id,
        "source_revision": first.source_revision,
        "conversion_identity": {
            "conversion_config_sha256": first.conversion_config_sha256,
            "converter_version": first.converter_version,
            "encoder_sha256": first.encoder_sha256,
            "encoder_config_sha256": first.encoder_config_sha256,
            "source_lock_sha256": first.source_lock_sha256,
        },
        "source_episode_ids": [episode.identity.source_episode_id for episode in episodes],
        "episode_lengths": [episode.row_count for episode in episodes],
        "total_frames": sum(episode.row_count for episode in episodes),
        "video_keys": list(episodes[0].videos),
        "task_catalog": list(task_catalog),
        "stages": [
            {
                "source_episode_id": episode.identity.source_episode_id,
                "source_file_sha256": dict(episode.identity.source_file_sha256),
                "manifest_sha256": episode.stage_manifest_digest,
                "checksums_sha256": episode.stage_checksums_digest,
            }
            for episode in episodes
        ],
    }


def _expected_merge_verification(
    episodes: tuple[MergeEpisode, ...], task_catalog: tuple[str, ...]
) -> MergeVerification:
    task_to_index = {task: index for index, task in enumerate(task_catalog)}
    content = hashlib.sha256()
    video_counts: dict[int, dict[str, int]] = {}
    cursor = 0
    for target_episode_index, episode in enumerate(episodes):
        for local_index, row in enumerate(iter_stage_rows(episode)):
            _update_frame_content_digest(
                content,
                row=row,
                source_episode_id=episode.identity.source_episode_id,
                target_episode_index=target_episode_index,
                local_index=local_index,
                global_index=cursor,
                task_index=task_to_index[row["task"]],
            )
            cursor += 1
        video_counts[episode.identity.source_episode_id] = {key: episode.row_count for key in episode.videos}
    return MergeVerification(
        source_episode_ids=tuple(episode.identity.source_episode_id for episode in episodes),
        episode_lengths=tuple(episode.row_count for episode in episodes),
        total_frames=cursor,
        task_to_index=task_to_index,
        frame_content_sha256=content.hexdigest(),
        video_frame_counts=video_counts,
        stats_source_episode_ids=tuple(episode.identity.source_episode_id for episode in episodes),
    )


def _digest_part(digest: object, data: bytes) -> None:
    digest.update(struct.pack(">Q", len(data)))
    digest.update(data)


def _update_frame_content_digest(
    digest: object,
    *,
    row: Mapping[str, object],
    source_episode_id: int,
    target_episode_index: int,
    local_index: int,
    global_index: int,
    task_index: int,
) -> None:
    digest.update(
        struct.pack(
            ">qqqqq",
            source_episode_id,
            target_episode_index,
            local_index,
            global_index,
            task_index,
        )
    )
    for key in TARGET_FRAME_SCHEMA:
        _digest_part(digest, key.encode("utf-8"))
        value = row[key]
        if isinstance(value, np.ndarray):
            _digest_part(digest, value.dtype.str.encode("ascii"))
            _digest_part(digest, repr(value.shape).encode("ascii"))
            _digest_part(digest, np.ascontiguousarray(value).tobytes())
        elif isinstance(value, np.float32):
            _digest_part(digest, b"<f4")
            _digest_part(digest, value.tobytes())
        elif isinstance(value, str):
            _digest_part(digest, value.encode("utf-8"))
        else:
            raise ValueError(f"cannot digest staged frame field {key}")


def _verification_to_dict(report: MergeVerification) -> dict[str, object]:
    return {
        "success": True,
        "source_episode_ids": list(report.source_episode_ids),
        "episode_lengths": list(report.episode_lengths),
        "total_frames": report.total_frames,
        "task_to_index": dict(report.task_to_index),
        "frame_content_sha256": report.frame_content_sha256,
        "video_frame_counts": {str(key): dict(value) for key, value in report.video_frame_counts.items()},
        "stats_source_episode_ids": list(report.stats_source_episode_ids),
    }


def _verify_merged_output(
    path: Path,
    expected_manifest: dict[str, object],
    expected_validation: dict[str, object] | None = None,
) -> None:
    _verify_checksum_tree(path, checksum_name="dataset-checksums.sha256")
    manifest = _parse_json_canonical(_secure_read(path / "source-manifest.json"), artifact="source-manifest.json")
    if manifest != expected_manifest:
        raise ValueError("source-manifest.json does not match requested stage inputs")
    validation = _parse_json_canonical(
        _secure_read(path / "merge-validation.json"), artifact="merge-validation.json"
    )
    if validation.get("success") is not True:
        raise ValueError("merge-validation.json does not record success")
    if expected_validation is not None and validation != expected_validation:
        raise ValueError("merge-validation.json does not match requested stage inputs")


def _validate_merge_inputs(
    stage_values: Iterable[StageResult | str | Path],
) -> tuple[MergeEpisode, ...]:
    values = tuple(stage_values)
    if not values:
        raise ValueError("merge requires at least one stage")
    episodes = tuple(_stage_input(value) for value in values)
    repositories = {episode.identity.source_repo_id for episode in episodes}
    if len(repositories) != 1:
        raise ValueError("merge requires exactly one source repository")
    conversion_keys = {episode.identity.conversion_key for episode in episodes}
    if len(conversion_keys) != 1:
        raise ValueError("merge stages must have the same conversion identity and artifact locks")
    episode_ids = [episode.identity.source_episode_id for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("merge contains a duplicate source episode ID")
    episodes = tuple(sorted(episodes, key=lambda episode: episode.identity.source_episode_id))
    video_keys = {tuple(episode.videos) for episode in episodes}
    if len(video_keys) != 1:
        raise ValueError("merge stages must have one cohort-wide target video schema")
    return episodes


def _reject_output_input_overlap(final: Path, episodes: tuple[MergeEpisode, ...]) -> None:
    resolved_final = final.resolve(strict=False)
    for episode in episodes:
        stage = episode.path.resolve(strict=True)
        if resolved_final == stage or stage in resolved_final.parents or resolved_final in stage.parents:
            raise ValueError("merge output overlaps input staging ancestry")
        staging_roots = [parent for parent in (stage, *stage.parents) if parent.name == ".staging"]
        if any(resolved_final == root or root in resolved_final.parents for root in staging_roots):
            raise ValueError("merge output overlaps input staging ancestry")


def _canonical_output_path(output: str | Path) -> Path:
    absolute = Path(output).absolute()
    if os.path.lexists(absolute) and stat.S_ISLNK(absolute.lstat().st_mode):
        raise FileExistsError(f"existing output is immutable, incomplete, or mismatched: {absolute}")
    return _canonical_path(output, field_name="merge output", strict=False)


def merge_stages(
    stages: Iterable[StageResult | str | Path],
    output: str | Path,
    *,
    merge_backend: MergeBackend | None = None,
    final_validator: FinalValidator | None = None,
    resume_existing: bool = False,
) -> Path:
    """Fully revalidate, deterministically merge, and atomically publish stages."""
    episodes = _validate_merge_inputs(stages)
    tasks = tuple(sorted({row["task"] for episode in episodes for row in iter_stage_rows(episode)}))
    expected_manifest = _merge_manifest(episodes, tasks)
    expected_report = _expected_merge_verification(episodes, tasks)
    expected_validation = _verification_to_dict(expected_report)
    final = _canonical_output_path(output)
    _reject_output_input_overlap(final, episodes)
    _ensure_safe_directory(final.parent)
    if os.path.lexists(final):
        if resume_existing:
            try:
                resumed_report = _validate_gr00t_output(final, episodes, tasks)
                if _verification_to_dict(resumed_report) != expected_validation:
                    raise ValueError("resumed output validation differs from exact requested dataset")
                _verify_merged_output(final, expected_manifest, expected_validation)
            except (OSError, TypeError, ValueError):
                pass
            else:
                return final
        raise FileExistsError(f"existing output is immutable, incomplete, or mismatched: {final}")

    temp = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    temp.rmdir()
    try:
        backend = _default_merge_backend if merge_backend is None else merge_backend
        backend(temp, episodes, tasks)
        if not temp.exists() or not stat.S_ISDIR(temp.lstat().st_mode):
            raise ValueError("merge backend did not create a real output directory")
        _write_file(temp / "source-manifest.json", _canonical_json(expected_manifest))
        _write_file(temp / "merge-validation.json", _canonical_json(expected_validation))
        artifacts = _artifact_paths(temp)
        _write_file(temp / "dataset-checksums.sha256", _checksums_bytes(temp, artifacts))
        _fsync_tree(temp)
        if final_validator is not None:
            hook_report = final_validator(temp, episodes, tasks)
            if not isinstance(hook_report, MergeVerification):
                raise ValueError("additional final validation hook must return a MergeVerification")
            if _verification_to_dict(hook_report) != expected_validation:
                raise ValueError("additional final validation hook differs from exact requested dataset")
        report = _validate_gr00t_output(temp, episodes, tasks)
        if not isinstance(report, MergeVerification):
            raise ValueError("built-in final validator must return a MergeVerification")
        if _verification_to_dict(report) != expected_validation:
            raise ValueError("merge backend verification differs from exact requested dataset")
        _verify_merged_output(temp, expected_manifest, expected_validation)
        _atomic_publish(temp, final)
        _fsync_directory(final.parent)
        return final
    except BaseException:
        if os.path.lexists(temp):
            shutil.rmtree(temp)
        raise


def dataset_manifest_digest(path: str | Path) -> str:
    """Hash the canonical, path-independent logical merge manifest."""
    root = Path(path).absolute()
    _verify_checksum_tree(root, checksum_name="dataset-checksums.sha256")
    data = _secure_read(root / "source-manifest.json")
    _parse_json_canonical(data, artifact="source-manifest.json")
    return _sha256(data)


class _AssetFreeSchemaRobot:
    joint_names = list(TARGET_JOINT_NAMES)
    num_joints = 43
    _groups = {
        "left_leg": tuple(range(0, 6)),
        "right_leg": tuple(range(6, 12)),
        "waist": tuple(range(12, 15)),
        "left_arm": tuple(range(15, 22)),
        "left_hand": tuple(range(22, 29)),
        "right_arm": tuple(range(29, 36)),
        "right_hand": tuple(range(36, 43)),
    }

    def get_joint_group_indices(self, group: str) -> tuple[int, ...]:
        return self._groups[group]


def _target_dataset_contract(
    video_keys: tuple[str, ...],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    from gear_sonic.data.features_sonic_vla import (
        get_features_sonic_vla,
        get_modality_config_sonic_vla,
        get_wrist_camera_features,
        get_wrist_camera_modality_config,
    )

    robot_model = _AssetFreeSchemaRobot()
    features = get_features_sonic_vla(robot_model)
    wrist_features = get_wrist_camera_features()
    for key in video_keys:
        if key in wrist_features:
            features[key] = wrist_features[key]
        if key not in features or features[key]["dtype"] != "video":
            raise ValueError(f"target video key {key!r} is not supported by the SONIC VLA schema")
    for key in tuple(features):
        if features[key]["dtype"] == "video" and key not in video_keys:
            del features[key]

    modality = get_modality_config_sonic_vla(robot_model)
    wrist_modality = get_wrist_camera_modality_config()["video"]
    target_to_modality = {value["original_key"]: name for name, value in wrist_modality.items()}
    for key in video_keys:
        if key in target_to_modality:
            name = target_to_modality[key]
            modality["video"][name] = wrist_modality[name]
    modality["video"] = {
        name: value for name, value in modality["video"].items() if value["original_key"] in video_keys
    }
    return features, modality


def _create_gr00t_exporter(
    output: Path,
    video_keys: tuple[str, ...],
    task_catalog: tuple[str, ...],
):
    """Lazily construct the repository exporter with the exact 50 Hz schema."""
    from gear_sonic.data.exporter import Gr00tDataExporter

    features, modality = _target_dataset_contract(video_keys)

    exporter = Gr00tDataExporter.create(
        save_root=output,
        fps=50,
        features=features,
        modality_config=modality,
        task=task_catalog[0],
        script_config=dict(_CONVERSION_SCRIPT_CONFIG),
        robot_type="g1",
        overwrite_existing=False,
        pre_encoded_videos=True,
    )
    for expected_index, task in enumerate(task_catalog):
        exporter.meta.add_task(task)
        if exporter.meta.get_task_index(task) != expected_index:
            raise ValueError("Gr00tDataExporter did not preregister lexicographic task IDs")
    return exporter


def _default_merge_backend(
    output: Path,
    episodes: tuple[MergeEpisode, ...],
    task_catalog: tuple[str, ...],
) -> None:
    """Stream staged rows and register immutable pre-encoded videos."""
    video_keys = tuple(episodes[0].videos)
    exporter = _create_gr00t_exporter(output, video_keys, task_catalog)
    for target_episode_index, episode in enumerate(episodes):
        for key, artifact in episode.videos.items():
            destination = output / exporter.meta.get_video_file_path(target_episode_index, key)
            _replace_with_bound_file(artifact, destination)
            exporter.register_pre_encoded_video(key, destination)
        for row in iter_stage_rows(episode):
            frame = dict(row)
            timestamp = frame["timestamp"]
            assert isinstance(timestamp, np.float32)
            frame["timestamp"] = np.array([timestamp], dtype=np.float32)
            exporter.add_frame(frame)
        exporter.save_episode()


def _normalize_feature(feature: Mapping[str, object]) -> dict[str, object]:
    return {
        "dtype": feature["dtype"],
        "shape": tuple(feature["shape"]),
        "names": feature.get("names"),
    }


def _expected_output_arrow_schema(features: Mapping[str, Mapping[str, object]]) -> pa.Schema:
    fields: list[pa.Field] = []
    for key, feature in features.items():
        dtype = feature["dtype"]
        if dtype == "video":
            continue
        shape = tuple(feature["shape"])
        scalar_type = pa.from_numpy_dtype(np.dtype(dtype))
        field_type = scalar_type if shape == (1,) else pa.list_(scalar_type, list_size=shape[0])
        fields.append(pa.field(key, field_type, nullable=True))
    return pa.schema(fields)


def _output_batch_rows(batch: pa.RecordBatch) -> tuple[dict[str, object], ...]:
    table = pa.Table.from_batches([batch])
    rows = [dict() for _ in range(table.num_rows)]
    for key, spec in TARGET_FRAME_SCHEMA.items():
        if key == "task":
            continue
        column = table.column(key).combine_chunks()
        if key == "timestamp":
            values = column.to_numpy(zero_copy_only=False)
            if values.dtype != np.dtype(np.float32):
                raise ValueError("exported timestamp Arrow values must be exact float32 scalars")
            for index, value in enumerate(values):
                rows[index][key] = np.float32(value)
            continue
        assert spec is not None
        if spec.shape == (1,):
            values = column.to_numpy(zero_copy_only=False)
            if values.dtype != spec.dtype:
                raise ValueError(f"exported {key} Arrow scalar dtype differs from {spec.dtype}")
            for index, value in enumerate(values):
                rows[index][key] = np.array([value], dtype=spec.dtype)
        else:
            values = column.values.to_numpy(zero_copy_only=False)
            if values.dtype != spec.dtype:
                raise ValueError(f"exported {key} Arrow list dtype differs from {spec.dtype}")
            values = values.reshape(table.num_rows, *spec.shape)
            for index in range(table.num_rows):
                rows[index][key] = np.array(values[index], copy=True)
    for key, dtype in (
        ("frame_index", np.int64),
        ("episode_index", np.int64),
        ("index", np.int64),
        ("task_index", np.int64),
    ):
        values = table.column(key).combine_chunks().to_numpy(zero_copy_only=False)
        if values.dtype != np.dtype(dtype):
            raise ValueError(f"exported {key} Arrow values must have exact dtype {np.dtype(dtype)}")
        for index, value in enumerate(values):
            rows[index][key] = dtype(value)
    return tuple(rows)


class _StreamingStats:
    def __init__(
        self,
        features: Mapping[str, Mapping[str, object]],
        *,
        expected_count: int,
    ) -> None:
        if type(expected_count) is not int or expected_count < 1:
            raise ValueError("expected_count must be a positive integer")
        self.features = features
        self.expected_count = expected_count
        self.count = 0
        self._temporary = tempfile.TemporaryDirectory(prefix="unitree-sonic-stats-")
        self.temporary_root = Path(self._temporary.name)
        self.values: dict[str, np.memmap] = {}
        for index, (key, feature) in enumerate(features.items()):
            shape = (expected_count, *tuple(feature["shape"]))
            self.values[key] = np.memmap(
                self.temporary_root / f"feature-{index:04d}.bin",
                dtype=np.dtype(feature["dtype"]),
                mode="w+",
                shape=shape,
            )

    def __enter__(self) -> _StreamingStats:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        for values in self.values.values():
            values.flush()
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.values.clear()
        self._temporary.cleanup()

    def update(self, values: Mapping[str, object]) -> None:
        if self.count >= self.expected_count:
            raise ValueError("exported episode statistics received more frames than expected")
        for key, feature in self.features.items():
            dtype = np.dtype(feature["dtype"])
            value = np.asarray(values[key], dtype=dtype).reshape(tuple(feature["shape"]))
            self.values[key][self.count] = value
        self.count += 1

    def _expected(self) -> Mapping[str, object]:
        from lerobot.common.datasets.lerobot_dataset import compute_episode_stats

        if self.count != self.expected_count:
            raise ValueError("exported episode statistics frame count differs")
        arrays: dict[str, np.ndarray] = {}
        for key, feature in self.features.items():
            values = self.values[key][: self.count]
            if tuple(feature["shape"]) == (1,) and values.ndim == 2:
                values = values[:, 0]
            arrays[key] = values
        return compute_episode_stats(arrays, dict(self.features))

    def verify(self, actual: Mapping[str, object]) -> None:
        if set(actual) != set(self.features) or self.count != self.expected_count:
            raise ValueError("exported episode statistics feature keys differ")
        expected_by_feature = self._expected()
        for key, feature in self.features.items():
            statistics = actual[key]
            if not isinstance(statistics, Mapping) or set(statistics) != {
                "min",
                "max",
                "mean",
                "std",
                "count",
            }:
                raise ValueError(f"exported episode statistics structure differs for {key}")
            for statistic, expected_value in expected_by_feature[key].items():
                actual_value = np.asarray(statistics[statistic])
                if (
                    actual_value.shape != np.asarray(expected_value).shape
                    or not np.isfinite(actual_value).all()
                    or not np.array_equal(actual_value, expected_value)
                ):
                    raise ValueError(f"exported episode statistics differ for {key}.{statistic}")


def _json_equivalent(value: object) -> object:
    """Normalize tuples and NumPy-free metadata through its JSON representation."""
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _validate_exact_gr00t_metadata(
    info: Mapping[str, object],
    modality_config: Mapping[str, object],
    episodes: tuple[MergeEpisode, ...],
    task_catalog: tuple[str, ...],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
    from lerobot.common.datasets.utils import DEFAULT_CHUNK_SIZE, DEFAULT_FEATURES

    video_keys = tuple(episodes[0].videos)
    target_features, target_modality = _target_dataset_contract(video_keys)
    expected_features = {**target_features, **DEFAULT_FEATURES}
    for key in video_keys:
        expected_features[key] = {**expected_features[key], "info": dict(_VIDEO_FEATURE_INFO)}
    expected_total_frames = sum(episode.row_count for episode in episodes)
    expected_info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "g1",
        "total_episodes": len(episodes),
        "total_frames": expected_total_frames,
        "total_tasks": len(task_catalog),
        "total_videos": len(episodes) * len(video_keys),
        "total_chunks": (len(episodes) + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": 50,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": _CANONICAL_DATA_PATH,
        "video_path": _CANONICAL_VIDEO_PATH,
        "features": expected_features,
        "script_config": dict(_CONVERSION_SCRIPT_CONFIG),
        "discarded_episode_indices": [],
    }
    if _json_equivalent(info) != _json_equivalent(expected_info):
        raise ValueError("exported metadata differs from the exact G1 SONIC dataset contract")
    if _json_equivalent(modality_config) != _json_equivalent(target_modality):
        raise ValueError("exported modality metadata differs from the exact G1 SONIC contract")
    return expected_features, target_modality


def _contained_output_artifact(output: Path, relative: str | Path, *, field_name: str) -> Path:
    """Resolve one canonical metadata-derived artifact and fail closed on escape."""
    relative_text = Path(relative).as_posix()
    posix = PurePosixPath(relative_text)
    if (
        not relative_text
        or posix.is_absolute()
        or posix.as_posix() != relative_text
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"{field_name} must be a canonical relative output path")
    candidate = output.joinpath(*posix.parts)
    try:
        root = output.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{field_name} must resolve to an existing output artifact") from error
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"{field_name} resolved path must remain contained under output")
    return candidate


def _expected_final_artifacts(meta: object, episodes: tuple[MergeEpisode, ...]) -> set[str]:
    artifacts = {
        "meta/info.json",
        "meta/modality.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "source-manifest.json",
        "merge-validation.json",
        "dataset-checksums.sha256",
    }
    for target_index, episode in enumerate(episodes):
        artifacts.add(Path(meta.get_data_file_path(target_index)).as_posix())
        for key in episode.videos:
            artifacts.add(Path(meta.get_video_file_path(target_index, key)).as_posix())
    return artifacts


def _read_exact_indexed_jsonl(
    output: Path,
    relative: str,
    *,
    expected_count: int,
    expected_fields: frozenset[str],
    index_field: str,
    label: str,
) -> tuple[Mapping[str, object], ...]:
    path = _contained_output_artifact(output, relative, field_name=relative)
    _reject_symlink_components(path)
    try:
        lines = _secure_read(path).decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must contain exact UTF-8 JSON lines") from error
    if len(lines) != expected_count:
        raise ValueError(f"{label} must contain the exact record count {expected_count}")
    records: list[Mapping[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {line_number} must contain exact valid JSON") from error
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError(f"{label} records must contain the exact fields {sorted(expected_fields)}")
        index = record[index_field]
        if type(index) is not int:
            raise ValueError(f"{label} {index_field} values must be exact integers")
        records.append(record)
    indices = tuple(record[index_field] for record in records)
    expected_indices = tuple(range(expected_count))
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} contains a duplicate {index_field}")
    if indices != expected_indices:
        raise ValueError(f"{label} {index_field} values must equal the exact contiguous range")
    return tuple(records)


def _expected_episode_records(episodes: tuple[MergeEpisode, ...]) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for episode_index, episode in enumerate(episodes):
        tasks: list[str] = []
        seen: set[str] = set()
        for row in iter_stage_rows(episode):
            task = row["task"]
            if task not in seen:
                seen.add(task)
                tasks.append(task)
        records.append(
            {
                "episode_index": episode_index,
                "tasks": tasks,
                "length": episode.row_count,
            }
        )
    return tuple(records)


def _validate_raw_episode_stats_schema(
    records: tuple[Mapping[str, object], ...],
    nonvideo_features: Mapping[str, object],
) -> None:
    expected_features = set(nonvideo_features)
    expected_statistics = {"min", "max", "mean", "std", "count"}
    for record in records:
        stats = record["stats"]
        if not isinstance(stats, Mapping) or set(stats) != expected_features:
            raise ValueError("episodes_stats.jsonl records must contain the exact feature fields")
        if any(
            not isinstance(feature_stats, Mapping) or set(feature_stats) != expected_statistics
            for feature_stats in stats.values()
        ):
            raise ValueError("episodes_stats.jsonl feature records must contain the exact statistic fields")


def _validate_gr00t_output(
    output: Path,
    episodes: tuple[MergeEpisode, ...],
    task_catalog: tuple[str, ...],
) -> MergeVerification:
    """Independently reopen and validate every final LeRobot dataset artifact."""
    from gear_sonic.data.exporter import Gr00tDatasetMetadata

    metadata: dict[str, Mapping[str, object]] = {}
    for relative in ("meta/info.json", "meta/modality.json"):
        metadata_path = _contained_output_artifact(output, relative, field_name=relative)
        try:
            value = json.loads(_secure_read(metadata_path))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"{relative} must contain valid JSON metadata") from error
        if not isinstance(value, dict):
            raise ValueError(f"{relative} must contain a JSON metadata object")
        metadata[relative] = value
    expected_features, _ = _validate_exact_gr00t_metadata(
        metadata["meta/info.json"],
        metadata["meta/modality.json"],
        episodes,
        task_catalog,
    )
    raw_task_records = _read_exact_indexed_jsonl(
        output,
        "meta/tasks.jsonl",
        expected_count=len(task_catalog),
        expected_fields=frozenset({"task_index", "task"}),
        index_field="task_index",
        label="global task catalog",
    )
    expected_task_records = tuple({"task_index": index, "task": task} for index, task in enumerate(task_catalog))
    if raw_task_records != expected_task_records:
        raise ValueError("global task catalog must contain the exact lexicographically sorted task values")
    raw_episode_records = _read_exact_indexed_jsonl(
        output,
        "meta/episodes.jsonl",
        expected_count=len(episodes),
        expected_fields=frozenset({"episode_index", "tasks", "length"}),
        index_field="episode_index",
        label="episode metadata",
    )
    if raw_episode_records != _expected_episode_records(episodes):
        raise ValueError("episodes.jsonl records differ from the exact staged episode metadata")
    raw_episode_stats = _read_exact_indexed_jsonl(
        output,
        "meta/episodes_stats.jsonl",
        expected_count=len(episodes),
        expected_fields=frozenset({"episode_index", "stats"}),
        index_field="episode_index",
        label="episode statistics",
    )
    nonvideo_features = {key: feature for key, feature in expected_features.items() if feature["dtype"] != "video"}
    _validate_raw_episode_stats_schema(raw_episode_stats, nonvideo_features)
    meta = Gr00tDatasetMetadata(repo_id="tmp/tmp_dataset", root=output)
    if _json_equivalent(meta.info) != _json_equivalent(metadata["meta/info.json"]) or _json_equivalent(
        meta.modality_config
    ) != _json_equivalent(metadata["meta/modality.json"]):
        raise ValueError("exported metadata changed during independent validation")
    if meta.total_episodes != len(episodes):
        raise ValueError("exported episode count differs from requested merge")
    expected_total = sum(episode.row_count for episode in episodes)
    if meta.total_frames != expected_total:
        raise ValueError("exported global frame count differs from requested merge")
    expected_episode_indices = set(range(len(episodes)))
    if set(meta.episodes) != expected_episode_indices:
        raise ValueError("exported episode metadata keys must equal the exact requested episode set")
    if set(meta.episodes_stats) != expected_episode_indices:
        raise ValueError("exported episode statistics keys must equal the exact requested episode set")
    if set(_artifact_paths(output)) != _expected_final_artifacts(meta, episodes):
        raise ValueError("exported dataset contains an unexpected artifact or differs from the canonical tree")
    task_to_index = {task: meta.get_task_index(task) for task in task_catalog}
    expected_task_to_index = {task: index for index, task in enumerate(task_catalog)}
    expected_index_to_task = {index: task for index, task in enumerate(task_catalog)}
    if (
        task_to_index != expected_task_to_index
        or meta.total_tasks != len(task_catalog)
        or dict(meta.tasks) != expected_index_to_task
    ):
        raise ValueError("exported global task catalog is not lexicographic")
    if set(meta.features) != set(expected_features) or any(
        _normalize_feature(meta.features[key]) != _normalize_feature(expected_features[key])
        for key in expected_features
    ):
        raise ValueError("exported feature metadata differs from the exact SONIC VLA schema")
    expected_arrow_schema = _expected_output_arrow_schema(expected_features)

    episode_lengths: list[int] = []
    video_counts: dict[int, dict[str, int]] = {}
    stats_ids: list[int] = []
    content = hashlib.sha256()
    global_index = 0
    for target_index, episode in enumerate(episodes):
        parquet_path = _contained_output_artifact(
            output,
            meta.get_data_file_path(target_index),
            field_name="exported data path",
        )
        staged_rows = iter(iter_stage_rows(episode))
        local_index = 0
        episode_tasks: list[str] = []
        seen_tasks: set[str] = set()
        with _StreamingStats(nonvideo_features, expected_count=episode.row_count) as stats:
            with _open_bound_file(parquet_path) as stream:
                before_digest = _hash_stream(stream)
                parquet = pq.ParquetFile(stream)
                if parquet.schema_arrow.remove_metadata() != expected_arrow_schema:
                    raise ValueError("exported parquet Arrow schema, types, order, or nullability differ")
                if parquet.metadata.num_rows != episode.row_count:
                    raise ValueError("exported parquet row count differs from staged episode")
                for batch in parquet.iter_batches(batch_size=128):
                    if any(batch.column(index).null_count for index in range(batch.num_columns)):
                        raise ValueError("exported parquet must not contain null values")
                    for actual in _output_batch_rows(batch):
                        try:
                            expected = next(staged_rows)
                        except StopIteration as error:
                            raise ValueError("exported parquet has more rows than staged episode") from error
                        for key in TARGET_FRAME_SCHEMA:
                            if key == "task":
                                continue
                            expected_value = expected[key]
                            actual_value = actual[key]
                            if isinstance(expected_value, np.ndarray):
                                if not isinstance(actual_value, np.ndarray) or not np.array_equal(
                                    actual_value, expected_value
                                ):
                                    raise ValueError(f"exported nonvideo value differs for {key}")
                            elif not isinstance(actual_value, np.float32) or (
                                actual_value.tobytes() != expected_value.tobytes()
                            ):
                                raise ValueError("exported timestamp differs bitwise from staged scalar")
                        task = expected["task"]
                        task_index = task_to_index[task]
                        if (
                            actual["frame_index"] != local_index
                            or actual["episode_index"] != target_index
                            or actual["index"] != global_index
                            or actual["task_index"] != task_index
                        ):
                            raise ValueError("exported frame, episode, global, or task index differs")
                        if task not in seen_tasks:
                            seen_tasks.add(task)
                            episode_tasks.append(task)
                        _update_frame_content_digest(
                            content,
                            row=expected,
                            source_episode_id=episode.identity.source_episode_id,
                            target_episode_index=target_index,
                            local_index=local_index,
                            global_index=global_index,
                            task_index=task_index,
                        )
                        stats.update(actual)
                        local_index += 1
                        global_index += 1
                try:
                    next(staged_rows)
                except StopIteration:
                    pass
                else:
                    raise ValueError("exported parquet has fewer rows than staged episode")
                if _hash_stream(stream) != before_digest:
                    raise ValueError("exported parquet changed during independent validation")
            stats.verify(raw_episode_stats[target_index]["stats"])
        episode_lengths.append(local_index)
        counts: dict[str, int] = {}
        for key in episode.videos:
            video_path = _contained_output_artifact(
                output,
                meta.get_video_file_path(target_index, key),
                field_name=f"exported video path {key}",
            )
            video_digest = _hash_secure_file(video_path)
            if video_digest != episode.videos[key].sha256:
                raise ValueError(f"exported video content differs from staged video for {key}")
            if _inspect_bound_video_media_info(video_path, expected_sha256=video_digest) != _VIDEO_FEATURE_INFO:
                raise ValueError(f"exported video media info differs from metadata for {key}")
            count, _ = _inspect_bound_video(
                video_path,
                expected_sha256=video_digest,
                field_name=f"exported video {key}",
            )
            counts[key] = count
        video_counts[episode.identity.source_episode_id] = counts
        if target_index not in meta.episodes or target_index not in meta.episodes_stats:
            raise ValueError("exported episode metadata or statistics are missing")
        episode_metadata = meta.episodes[target_index]
        if (
            episode_metadata != raw_episode_records[target_index]
            or episode_metadata.get("length") != local_index
            or episode_metadata.get("tasks") != episode_tasks
        ):
            raise ValueError("exported episode length or first-occurrence task order differs")
        stats_ids.append(episode.identity.source_episode_id)
    return MergeVerification(
        source_episode_ids=tuple(episode.identity.source_episode_id for episode in episodes),
        episode_lengths=tuple(episode_lengths),
        total_frames=global_index,
        task_to_index=task_to_index,
        frame_content_sha256=content.hexdigest(),
        video_frame_counts=video_counts,
        stats_source_episode_ids=tuple(stats_ids),
    )
