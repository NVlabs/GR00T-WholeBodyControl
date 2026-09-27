"""Exact-revision materialization and direct reads for pinned LeRobot v3 sources."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import string
from types import MappingProxyType

from gear_sonic.data.unitree_conversion.contracts import SourceSpec, SourceVideoSegment


@dataclass(frozen=True)
class V3DataSchema:
    """Typed Parquet columns required by one exact v3 source adapter."""

    float_vector_columns: tuple[str, ...]
    float_scalar_columns: tuple[str, ...]
    integer_scalar_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        groups: list[tuple[str, ...]] = []
        for field_name in (
            "float_vector_columns",
            "float_scalar_columns",
            "integer_scalar_columns",
        ):
            value = getattr(self, field_name)
            if isinstance(value, (str, bytes)):
                raise ValueError(f"{field_name} must be a non-string iterable")
            try:
                columns = tuple(value)
            except TypeError as error:
                raise ValueError(f"{field_name} must be an iterable") from error
            if any(not isinstance(column, str) or not column for column in columns):
                raise ValueError("schema column names must be nonempty strings")
            object.__setattr__(self, field_name, columns)
            groups.append(columns)
        columns = tuple(column for group in groups for column in group)
        if len(columns) != len(set(columns)):
            raise ValueError("schema column names must be unique")
        if "episode_index" not in self.integer_scalar_columns:
            raise ValueError("integer_scalar_columns must include episode_index")

    @property
    def columns(self) -> tuple[str, ...]:
        return self.float_vector_columns + self.float_scalar_columns + self.integer_scalar_columns


DEX3_DATA_SCHEMA = V3DataSchema(
    float_vector_columns=("observation.state", "action"),
    float_scalar_columns=("timestamp",),
    integer_scalar_columns=("task_index", "frame_index", "episode_index"),
)


@dataclass(frozen=True)
class V3SourceMeta:
    """Metadata surface consumed by a strict source-adapter validation."""

    revision: str
    total_episodes: object
    fps: object
    features: Mapping[str, object]


@dataclass(frozen=True)
class V3SourceDataset:
    """Minimal dataset-compatible wrapper around raw selected v3 Parquet rows."""

    root: Path
    revision: str
    meta: V3SourceMeta
    hf_dataset: tuple[Mapping[str, object], ...]
    video_segments: Mapping[str, SourceVideoSegment]

    def __post_init__(self) -> None:
        if not isinstance(self.video_segments, Mapping):
            raise ValueError("video_segments must be a mapping")
        segments: dict[str, SourceVideoSegment] = {}
        for source_key, segment in self.video_segments.items():
            if not isinstance(source_key, str) or not source_key:
                raise ValueError("video segment keys must be nonempty strings")
            if not isinstance(segment, SourceVideoSegment) or segment.source_key != source_key:
                raise ValueError("video segment values must match their source keys")
            segments[source_key] = segment
        object.__setattr__(self, "video_segments", MappingProxyType(dict(sorted(segments.items()))))


@dataclass(frozen=True)
class _V3Info:
    total_episodes: object
    fps: object
    data_path: str
    video_path: str
    features: dict[str, object]
    video_keys: tuple[str, ...]


def default_lerobot_cache_base() -> Path:
    """Mirror LeRobot's Hugging Face cache-base convention without importing LeRobot."""
    lerobot_home = os.getenv("HF_LEROBOT_HOME")
    if lerobot_home is not None:
        return Path(lerobot_home).expanduser()

    default_home = Path.home() / ".cache"
    xdg_cache_home = os.getenv("XDG_CACHE_HOME", str(default_home))
    hf_home = os.getenv("HF_HOME", str(Path(xdg_cache_home) / "huggingface"))
    return Path(os.path.expandvars(hf_home)).expanduser() / "lerobot"


def revision_scoped_root(cache_base: str | Path, repo_id: str, revision: str) -> Path:
    """Return a cache-contained path unique to one repository identity and revision."""
    base = Path(cache_base).expanduser().resolve()
    repository_identity = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()
    return base / f"repo-{repository_identity}" / f"revision-{revision}"


def _default_snapshot_downloader() -> Callable[..., str]:
    from huggingface_hub import snapshot_download

    return snapshot_download


def _verify_snapshot_location(downloaded: object, scoped_root: Path) -> None:
    try:
        downloaded_root = Path(downloaded).expanduser().resolve()
    except TypeError as error:
        raise ValueError("snapshot downloader must return the revision-scoped root") from error
    if downloaded_root != scoped_root.resolve():
        raise ValueError("snapshot downloader must return the revision-scoped root")


def _reject_duplicate_metadata_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate metadata key: {key}")
        result[key] = value
    return result


def _regular_file_under_root(scoped_root: Path, relative_path: PurePosixPath) -> Path:
    candidate = scoped_root.joinpath(*relative_path.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(scoped_root.resolve())
    except (FileNotFoundError, OSError, ValueError) as error:
        message = f"materialized file is missing or outside revision-scoped root: {relative_path}"
        raise ValueError(message) from error
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"materialized file is not a regular file: {relative_path}")
    return candidate


def _dataset_root(scoped_root: Path, dataset_path: str) -> Path:
    candidate = scoped_root if dataset_path == "." else scoped_root.joinpath(*PurePosixPath(dataset_path).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(scoped_root.resolve())
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError("dataset root is missing or outside revision-scoped root") from error
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("dataset root must be a safe regular directory")
    return candidate


def _repo_relative_path(dataset_path: str, path: PurePosixPath) -> PurePosixPath:
    if dataset_path == ".":
        return path
    return PurePosixPath(dataset_path) / path


def _read_info(scoped_root: Path) -> _V3Info:
    info_path = _regular_file_under_root(scoped_root, PurePosixPath("meta/info.json"))
    try:
        with info_path.open(encoding="utf-8") as stream:
            info = json.load(stream, object_pairs_hook=_reject_duplicate_metadata_keys)
    except json.JSONDecodeError as error:
        raise ValueError("metadata info.json must contain valid JSON") from error
    if not isinstance(info, dict):
        raise ValueError("metadata info.json must contain a JSON object")

    required_fields = (
        "codebase_version",
        "chunks_size",
        "fps",
        "total_episodes",
        "data_path",
        "video_path",
        "features",
    )
    for field_name in required_fields:
        if field_name not in info:
            raise ValueError(f"metadata is missing required field {field_name}")

    if info["codebase_version"] != "v3.0":
        raise ValueError("metadata codebase_version must be exactly 'v3.0'")
    chunks_size = info["chunks_size"]
    if isinstance(chunks_size, bool) or not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError("metadata chunks_size must be a positive integer")
    data_path = info["data_path"]
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("metadata data_path must be a nonempty string")
    video_path = info["video_path"]
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("metadata video_path must be a nonempty string")

    features = info["features"]
    if not isinstance(features, dict) or not features:
        raise ValueError("metadata features must be a nonempty mapping")
    video_keys: list[str] = []
    for feature_key, feature in features.items():
        if not isinstance(feature_key, str) or not feature_key:
            raise ValueError("metadata features keys must be nonempty strings")
        if not isinstance(feature, dict):
            raise ValueError(f"metadata features entry {feature_key!r} must be a mapping")
        dtype = feature.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"metadata features entry {feature_key!r} dtype must be a nonempty string")
        if dtype == "video":
            video_keys.append(feature_key)

    return _V3Info(
        total_episodes=info["total_episodes"],
        fps=info["fps"],
        data_path=data_path,
        video_path=video_path,
        features=features,
        video_keys=tuple(video_keys),
    )


def _episode_metadata_files(scoped_root: Path) -> tuple[Path, ...]:
    episodes_root = scoped_root / "meta/episodes"
    try:
        resolved_root = episodes_root.resolve(strict=True)
        resolved_root.relative_to(scoped_root.resolve())
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError("episode metadata must contain at least one parquet file") from error
    if episodes_root.is_symlink() or not episodes_root.is_dir():
        raise ValueError("episode metadata directory must be a safe regular directory")

    try:
        candidates = sorted(episodes_root.rglob("*.parquet"))
    except OSError as error:
        raise ValueError("episode metadata parquet files must be readable") from error
    if not candidates:
        raise ValueError("episode metadata must contain at least one parquet file")

    files: list[Path] = []
    for candidate in candidates:
        relative_path = PurePosixPath(candidate.relative_to(scoped_root).as_posix())
        files.append(_regular_file_under_root(scoped_root, relative_path))
    return tuple(files)


def _metadata_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"episode metadata {field_name} must be a nonnegative integer")
    return value


def _metadata_timestamp(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"episode metadata {field_name} must be finite nonnegative")
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError(f"episode metadata {field_name} must be finite nonnegative")
    return timestamp


def _selected_episode_row(
    scoped_root: Path,
    *,
    episode_id: int,
    video_keys: tuple[str, ...],
) -> dict[str, int | float]:
    import pyarrow.parquet as pq

    integer_columns = (
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "length",
        *(f"videos/{key}/{part}" for key in video_keys for part in ("chunk_index", "file_index")),
    )
    timestamp_columns = tuple(
        f"videos/{key}/{part}" for key in video_keys for part in ("from_timestamp", "to_timestamp")
    )
    required_columns = integer_columns + timestamp_columns
    seen_episode_ids: set[int] = set()
    selected: dict[str, int | float] | None = None
    for metadata_path in _episode_metadata_files(scoped_root):
        try:
            table = pq.read_table(metadata_path)
        except Exception as error:
            raise ValueError(f"episode metadata parquet must be readable: {metadata_path}") from error
        missing = tuple(column for column in required_columns if table.column_names.count(column) != 1)
        if missing:
            raise ValueError(f"episode metadata parquet is missing required columns: {missing}")

        for row_index in range(table.num_rows):
            row: dict[str, int | float] = {
                column: _metadata_integer(table[column][row_index].as_py(), field_name=column)
                for column in integer_columns
            }
            row.update(
                {
                    column: _metadata_timestamp(table[column][row_index].as_py(), field_name=column)
                    for column in timestamp_columns
                }
            )
            source_episode_id = row["episode_index"]
            if source_episode_id in seen_episode_ids:
                raise ValueError(f"duplicate episode_index in episode metadata: {source_episode_id}")
            seen_episode_ids.add(source_episode_id)
            if source_episode_id == episode_id:
                selected = row

    if selected is None:
        raise ValueError(f"no episode metadata row for selected episode {episode_id}")
    return selected


def _render_path(
    template: str,
    *,
    field_name: str,
    required_fields: frozenset[str],
    values: Mapping[str, object],
) -> PurePosixPath:
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"metadata {field_name} template is malformed") from error

    fields: list[str] = []
    for _, template_field, format_spec, conversion in parsed:
        if template_field is None:
            continue
        if template_field not in required_fields or conversion is not None or "{" in format_spec:
            raise ValueError(f"metadata {field_name} template contains unsupported fields")
        fields.append(template_field)
    if len(fields) != len(set(fields)) or set(fields) != required_fields:
        raise ValueError(f"metadata {field_name} template must contain each required field exactly once")

    try:
        rendered = template.format(**values)
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError(f"metadata {field_name} template cannot be rendered") from error
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in rendered) or any(
        character in "*?[]" for character in rendered
    ):
        raise ValueError(f"metadata {field_name} template must render a safe literal relative path")
    relative_path = PurePosixPath(rendered)
    if (
        not rendered
        or "\\" in rendered
        or relative_path.is_absolute()
        or not relative_path.parts
        or ".." in relative_path.parts
    ):
        raise ValueError(f"metadata {field_name} template must render a safe relative path")
    return relative_path


def _selected_paths(
    episode_row: Mapping[str, int | float],
    info: _V3Info,
) -> tuple[PurePosixPath, ...]:
    data_path = _render_path(
        info.data_path,
        field_name="data_path",
        required_fields=frozenset({"chunk_index", "file_index"}),
        values={
            "chunk_index": episode_row["data/chunk_index"],
            "file_index": episode_row["data/file_index"],
        },
    )
    if data_path.suffix != ".parquet":
        raise ValueError("metadata data_path template must render a parquet path")

    paths = [data_path]
    for video_key in info.video_keys:
        paths.append(
            _render_path(
                info.video_path,
                field_name="video_path",
                required_fields=frozenset({"chunk_index", "file_index", "video_key"}),
                values={
                    "chunk_index": episode_row[f"videos/{video_key}/chunk_index"],
                    "file_index": episode_row[f"videos/{video_key}/file_index"],
                    "video_key": video_key,
                },
            )
        )
    if len(paths) != len(set(paths)):
        raise ValueError("metadata templates render duplicate selected episode paths")
    return tuple(paths)


def _validate_data_schema(schema: object, contract: V3DataSchema) -> None:
    import pyarrow as pa

    schema_names = tuple(schema.names)
    invalid_columns = tuple(column for column in contract.columns if schema_names.count(column) != 1)
    if invalid_columns:
        raise ValueError(f"selected v3 data parquet is missing required columns: {invalid_columns}")

    for column in contract.float_vector_columns:
        data_type = schema.field(column).type
        is_list = (
            pa.types.is_list(data_type)
            or pa.types.is_large_list(data_type)
            or pa.types.is_fixed_size_list(data_type)
        )
        if not is_list or not pa.types.is_floating(data_type.value_type):
            raise ValueError(f"selected v3 data column {column} has incompatible type")
    for column in contract.float_scalar_columns:
        if not pa.types.is_floating(schema.field(column).type):
            raise ValueError(f"selected v3 data column {column} has incompatible type")
    for column in contract.integer_scalar_columns:
        if not pa.types.is_integer(schema.field(column).type):
            raise ValueError(f"selected v3 data column {column} has incompatible type")


def _read_selected_data(
    data_path: Path,
    episode_id: int,
    schema_contract: V3DataSchema,
) -> tuple[Mapping[str, object], ...]:
    import pyarrow.parquet as pq

    try:
        schema = pq.read_schema(data_path)
    except Exception as error:
        raise ValueError(f"selected v3 data parquet must be readable: {data_path}") from error
    _validate_data_schema(schema, schema_contract)
    try:
        table = pq.read_table(
            data_path,
            columns=list(schema_contract.columns),
            filters=[("episode_index", "=", episode_id)],
        )
    except Exception as error:
        raise ValueError(f"selected v3 data parquet must be readable: {data_path}") from error
    if table.num_rows == 0:
        raise ValueError(f"selected v3 data contains no rows for episode {episode_id}")

    rows: list[Mapping[str, object]] = []
    for row_index in range(table.num_rows):
        row = {column: table[column][row_index].as_py() for column in schema_contract.columns}
        source_episode_id = row["episode_index"]
        if isinstance(source_episode_id, bool) or source_episode_id != episode_id:
            raise ValueError(f"selected v3 data contains wrong episode row: {source_episode_id!r}")
        rows.append(row)
    return tuple(rows)


def load_pinned_v3_episode(
    source_spec: SourceSpec,
    episode_id: int,
    *,
    cache_base: str | Path,
    snapshot_downloader: Callable[..., object] | None = None,
    schema: V3DataSchema = DEX3_DATA_SCHEMA,
    download_videos: bool = True,
) -> V3SourceDataset:
    """Materialize and directly read one exact-revision LeRobot v3 episode."""
    if not isinstance(source_spec, SourceSpec):
        raise ValueError("source_spec must be a SourceSpec")
    if not isinstance(schema, V3DataSchema):
        raise ValueError("schema must be a V3DataSchema")
    if not isinstance(download_videos, bool):
        raise ValueError("download_videos must be a boolean")
    scoped_root = revision_scoped_root(cache_base, source_spec.repo_id, source_spec.revision)
    downloader = _default_snapshot_downloader() if snapshot_downloader is None else snapshot_downloader
    common_kwargs = {
        "repo_id": source_spec.repo_id,
        "repo_type": "dataset",
        "revision": source_spec.revision,
        "local_dir": scoped_root,
    }
    metadata_pattern = _repo_relative_path(source_spec.dataset_path, PurePosixPath("meta/**"))
    downloaded = downloader(**common_kwargs, allow_patterns=[str(metadata_pattern)])
    _verify_snapshot_location(downloaded, scoped_root)

    dataset_root = _dataset_root(scoped_root, source_spec.dataset_path)
    info = _read_info(dataset_root)
    episode_row = _selected_episode_row(
        dataset_root,
        episode_id=episode_id,
        video_keys=info.video_keys,
    )
    selected_paths = _selected_paths(episode_row, info)
    materialized_selection = selected_paths if download_videos else selected_paths[:1]
    repository_paths = tuple(
        _repo_relative_path(source_spec.dataset_path, path) for path in materialized_selection
    )
    downloaded = downloader(**common_kwargs, allow_patterns=[str(path) for path in repository_paths])
    _verify_snapshot_location(downloaded, scoped_root)
    materialized_paths = tuple(_regular_file_under_root(dataset_root, path) for path in materialized_selection)
    rows = _read_selected_data(materialized_paths[0], episode_id, schema)
    episode_length = episode_row["length"]
    if not isinstance(episode_length, int) or episode_length != len(rows):
        raise ValueError("episode metadata length must equal selected data row count")

    video_segments: dict[str, SourceVideoSegment] = {}
    if download_videos:
        for source_key, video_path in zip(info.video_keys, materialized_paths[1:], strict=True):
            from_timestamp = episode_row[f"videos/{source_key}/from_timestamp"]
            to_timestamp = episode_row[f"videos/{source_key}/to_timestamp"]
            if not isinstance(from_timestamp, float) or not isinstance(to_timestamp, float):
                raise ValueError("episode video timestamps must be floating-point metadata")
            video_segments[source_key] = SourceVideoSegment(
                source_key=source_key,
                path=video_path,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                start_frame=round(from_timestamp * 30.0),
                end_frame=round(to_timestamp * 30.0),
                frame_count=episode_length,
            )
    return V3SourceDataset(
        root=dataset_root,
        revision=source_spec.revision,
        meta=V3SourceMeta(
            revision=source_spec.revision,
            total_episodes=info.total_episodes,
            fps=info.fps,
            features=info.features,
        ),
        hf_dataset=rows,
        video_segments=video_segments,
    )
