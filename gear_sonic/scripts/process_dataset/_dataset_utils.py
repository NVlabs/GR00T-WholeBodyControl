"""Shared, loss-conscious helpers for the custom LeRobot dataset tools."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Iterable

import pandas as pd


EPISODE_RE = re.compile(r"episode_(\d+)")
CHUNK_RE = re.compile(r"chunk-\d+")
GENERATED_META_FILES = {
    "info.json",
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "interaction_metadata.jsonl",
    "tasks.jsonl",
    "source_datasets.jsonl",
    "episode_provenance.jsonl",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=4, ensure_ascii=False)
        stream.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def discover_datasets(root: Path) -> list[Path]:
    """Find all LeRobot leaves below root, including root itself."""
    root = root.resolve()
    if (root / "meta" / "info.json").is_file():
        return [root]
    datasets = sorted(p.parent.parent for p in root.rglob("meta/info.json"))
    if not datasets:
        raise ValueError(f"No LeRobot datasets found below: {root}")
    return datasets


def load_dataset(dataset: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info = read_json(dataset / "meta" / "info.json")
    episodes = read_jsonl(dataset / "meta" / "episodes.jsonl")
    indices = [int(record["episode_index"]) for record in episodes]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate episode indices in {dataset}")
    if info.get("total_episodes") != len(episodes):
        raise ValueError(
            f"{dataset}: info.json says {info.get('total_episodes')} episodes, "
            f"but episodes.jsonl contains {len(episodes)}"
        )
    return info, sorted(episodes, key=lambda record: int(record["episode_index"]))


def episode_index_from_path(relative_path: Path) -> int | None:
    found = {int(value) for value in EPISODE_RE.findall(relative_path.as_posix())}
    if len(found) > 1:
        raise ValueError(f"Path contains conflicting episode indices: {relative_path}")
    return next(iter(found)) if found else None


def rewrite_episode_tokens(value: Any, old_index: int, new_index: int, chunks_size: int) -> Any:
    """Recursively rewrite episode/chunk references inside metadata."""
    if isinstance(value, str):
        value = re.sub(
            rf"episode_{old_index:06d}(?!\d)",
            f"episode_{new_index:06d}",
            value,
        )
        return CHUNK_RE.sub(f"chunk-{new_index // chunks_size:03d}", value)
    if isinstance(value, list):
        return [rewrite_episode_tokens(item, old_index, new_index, chunks_size) for item in value]
    if isinstance(value, dict):
        return {
            key: rewrite_episode_tokens(item, old_index, new_index, chunks_size)
            for key, item in value.items()
        }
    return value


def remap_episode_record(
    record: dict[str, Any], old_index: int, new_index: int, chunks_size: int
) -> dict[str, Any]:
    result = rewrite_episode_tokens(deepcopy(record), old_index, new_index, chunks_size)
    result["episode_index"] = new_index
    return result


def remap_asset_path(relative_path: Path, old_index: int, new_index: int, chunks_size: int) -> Path:
    text = rewrite_episode_tokens(relative_path.as_posix(), old_index, new_index, chunks_size)
    return Path(text)


def iter_assets(dataset: Path) -> Iterable[tuple[Path, int | None]]:
    """Yield files and their path-encoded episode index, if any."""
    for source in sorted(path for path in dataset.rglob("*") if path.is_file()):
        relative = source.relative_to(dataset)
        if relative.parent == Path("meta") and relative.name in GENERATED_META_FILES:
            continue
        yield relative, episode_index_from_path(relative)


def validate_asset_indices(dataset: Path, known_indices: set[int]) -> None:
    unknown = sorted(
        (relative, index)
        for relative, index in iter_assets(dataset)
        if index is not None and index not in known_indices
    )
    if unknown:
        preview = ", ".join(str(path) for path, _ in unknown[:5])
        raise ValueError(
            f"{dataset}: found episode assets not listed in episodes.jsonl: {preview}"
        )


def copy_json_asset_with_rewritten_paths(
    source: Path,
    destination: Path,
    old_index: int,
    new_index: int,
    chunks_size: int,
) -> None:
    if source.suffix == ".jsonl":
        records = read_jsonl(source)
        write_jsonl(
            destination,
            (rewrite_episode_tokens(record, old_index, new_index, chunks_size) for record in records),
        )
    else:
        write_json(
            destination,
            rewrite_episode_tokens(read_json(source), old_index, new_index, chunks_size),
        )


def copy_asset(
    source_dataset: Path,
    destination_dataset: Path,
    relative_path: Path,
    old_index: int,
    new_index: int,
    chunks_size: int,
    parquet_transform: Callable[[Path, Path], int],
    video_transform: Callable[[Path, Path, Path], None] | None = None,
) -> int | None:
    """Copy one episode asset and return its parquet row count, if applicable."""
    destination_relative = remap_asset_path(relative_path, old_index, new_index, chunks_size)
    destination = destination_dataset / destination_relative
    if destination.exists():
        raise ValueError(f"Output collision while copying {relative_path}: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source_dataset / relative_path

    if source.suffix == ".parquet":
        return parquet_transform(source, destination)
    if source.suffix == ".mp4" and video_transform is not None:
        video_transform(source, destination, relative_path)
        return None
    if source.suffix in {".json", ".jsonl"}:
        copy_json_asset_with_rewritten_paths(
            source, destination, old_index, new_index, chunks_size
        )
        return None
    shutil.copy2(source, destination)
    return None


def copy_non_episode_assets(source_dataset: Path, destination_dataset: Path) -> None:
    """Preserve dataset-level custom files that are not regenerated metadata."""
    for relative, episode_index in iter_assets(source_dataset):
        if episode_index is not None:
            continue
        destination = destination_dataset / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dataset / relative, destination)


def indexed_records(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        index = int(record["episode_index"])
        if index in result:
            raise ValueError(f"Duplicate episode index {index} in {path}")
        result[index] = record
    return result


def get_video_features(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: value
        for key, value in info.get("features", {}).items()
        if value.get("dtype") in {"video", "image"}
    }


def choose_canonical_features(infos: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate non-video schema and select the modal definition per video key."""
    reference = infos[0].get("features", {})
    reference_keys = set(reference)
    for number, info in enumerate(infos[1:], 2):
        if set(info.get("features", {})) != reference_keys:
            raise ValueError(f"Dataset #{number} has a different feature set")
        for key in reference_keys:
            if reference[key].get("dtype") in {"video", "image"}:
                continue
            if info["features"][key] != reference[key]:
                raise ValueError(f"Dataset #{number} has incompatible feature {key!r}")

    result = deepcopy(reference)
    for key in get_video_features(infos[0]):
        serialized = [json.dumps(info["features"][key], sort_keys=True) for info in infos]
        winner, _ = Counter(serialized).most_common(1)[0]
        result[key] = json.loads(winner)
    return result


def video_dimensions(feature: dict[str, Any]) -> tuple[int, int]:
    shape = feature.get("shape", [])
    if len(shape) < 2:
        raise ValueError(f"Video feature has no HxW shape: {feature}")
    return int(shape[1]), int(shape[0])


def video_key_for_path(relative_path: Path, video_keys: Iterable[str]) -> str | None:
    parts = set(relative_path.parts)
    matches = [key for key in video_keys if key in parts]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous video key for {relative_path}: {matches}")
    return matches[0] if matches else None


def transcode_video(source: Path, destination: Path, width: int, height: int, fps: int) -> None:
    """Resize without dropping decoded frames; used only for schema mismatches."""
    import av

    input_container = av.open(str(source))
    input_stream = input_container.streams.video[0]
    output_container = av.open(str(destination), mode="w")
    output_stream = output_container.add_stream("h264", rate=fps)
    output_stream.width = width
    output_stream.height = height
    output_stream.pix_fmt = "yuv420p"
    decoded = 0
    try:
        for frame in input_container.decode(input_stream):
            decoded += 1
            resized = frame.reformat(width=width, height=height, format="yuv420p")
            for packet in output_stream.encode(resized):
                output_container.mux(packet)
        for packet in output_stream.encode():
            output_container.mux(packet)
    finally:
        input_container.close()
        output_container.close()
    if decoded == 0:
        raise ValueError(f"Video contains no decodable frames: {source}")
    verification_container = av.open(str(destination))
    verification_stream = verification_container.streams.video[0]
    output_frames = sum(1 for _ in verification_container.decode(verification_stream))
    verification_container.close()
    if output_frames != decoded:
        raise ValueError(
            f"Video frame count changed while resizing {source}: "
            f"{decoded} -> {output_frames}"
        )


def make_staging_directory(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))


def commit_staging(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    staging.replace(output)


def remove_staging(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)


def ensure_safe_paths(input_root: Path, output: Path) -> None:
    input_root = input_root.resolve()
    output = output.resolve()
    if input_root == output:
        raise ValueError("Input and output must be different (these tools are non-destructive)")
    if input_root in output.parents:
        raise ValueError("Output must not be placed inside the input tree")


def update_info_counts(
    info: dict[str, Any], output: Path, episode_count: int, frame_count: int, task_count: int
) -> dict[str, Any]:
    result = deepcopy(info)
    chunks_size = int(result.get("chunks_size", 1000))
    result["total_episodes"] = episode_count
    result["total_frames"] = frame_count
    result["total_tasks"] = task_count
    result["total_videos"] = sum(1 for _ in output.rglob("*.mp4"))
    result["total_chunks"] = (episode_count + chunks_size - 1) // chunks_size
    result["splits"] = {"train": f"0:{episode_count}"}
    result.pop("discarded_episode_indices", None)
    return result


def rewrite_parquet_indices(
    source: Path,
    destination: Path,
    episode_index: int,
    global_start: int,
    task_index_map: dict[int, int] | None = None,
) -> int:
    frame = pd.read_parquet(source)
    length = len(frame)
    frame["episode_index"] = episode_index
    frame["frame_index"] = range(length)
    frame["index"] = range(global_start, global_start + length)
    if task_index_map is not None and "task_index" in frame.columns:
        unknown = set(int(value) for value in frame["task_index"].unique()) - set(task_index_map)
        if unknown:
            raise ValueError(f"{source}: unknown task indices {sorted(unknown)}")
        frame["task_index"] = frame["task_index"].map(task_index_map).astype(frame["task_index"].dtype)
    frame.to_parquet(destination, index=False)
    return length
