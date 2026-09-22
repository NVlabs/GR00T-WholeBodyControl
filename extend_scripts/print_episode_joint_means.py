#!/usr/bin/env python3
"""Print the mean value of every joint for each exported Parquet episode.

The script is read-only. Joint names are loaded from the dataset's
``meta/info.json`` when available.

Example:

    python gear_sonic_deploy/scripts/print_episode_joint_means.py \
        outputs/handshake_first_iter/data/chunk-000
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


EPISODE_PATTERN = re.compile(r"episode_(\d+)\.parquet$")
BLOCK_WIDTH = 100


def _import_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit(
            "This script requires numpy and pyarrow. Install them with: "
            "pip install numpy pyarrow"
        ) from error
    return np, parquet


def _episode_index(path: Path) -> int:
    match = EPISODE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid episode filename: {path.name}")
    return int(match.group(1))


def _find_dataset_root(chunk_directory: Path) -> Path | None:
    """Find a parent containing meta/info.json, starting from the chunk path."""
    for candidate in (chunk_directory, *chunk_directory.parents):
        if (candidate / "meta" / "info.json").is_file():
            return candidate
    return None


def _load_joint_names(
    chunk_directory: Path,
    field: str,
) -> tuple[list[str] | None, Path | None]:
    dataset_root = _find_dataset_root(chunk_directory)
    if dataset_root is None:
        return None, None

    info_path = dataset_root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read metadata {info_path}: {error}") from error

    feature = info.get("features", {}).get(field, {})
    names = feature.get("names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        return None, info_path
    return names, info_path


def _resolve_episode_files(
    chunk_directory: Path,
    selected_indices: list[int] | None,
) -> list[Path]:
    files = sorted(
        (
            path
            for path in chunk_directory.glob("episode_*.parquet")
            if EPISODE_PATTERN.fullmatch(path.name)
        ),
        key=_episode_index,
    )
    if not files:
        raise SystemExit(f"No episode_*.parquet files found in {chunk_directory}")

    if selected_indices is None:
        return files

    by_index = {_episode_index(path): path for path in files}
    missing = [index for index in selected_indices if index not in by_index]
    if missing:
        raise SystemExit(
            f"Requested episode indices are absent from {chunk_directory}: {missing}"
        )
    return [by_index[index] for index in selected_indices]


def _read_episode(
    path: Path,
    field: str,
    np: Any,
    parquet: Any,
) -> tuple[int, Any, float | None]:
    available_columns = parquet.ParquetFile(path).schema_arrow.names
    if field not in available_columns:
        raise SystemExit(
            f"Field {field!r} is missing from {path.name}. Available fields: "
            + ", ".join(available_columns)
        )

    columns = [field]
    if "timestamp" in available_columns:
        columns.append("timestamp")
    if "episode_index" in available_columns:
        columns.append("episode_index")
    table = parquet.read_table(path, columns=columns)
    if table.num_rows == 0:
        raise SystemExit(f"Episode is empty: {path}")

    try:
        values = np.asarray(table[field].to_pylist(), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Field {field!r} is not a regular numeric vector: {error}") from error
    if values.ndim != 2 or values.shape[1] == 0:
        raise SystemExit(
            f"Field {field!r} in {path.name} must have shape [frames, joints], "
            f"got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise SystemExit(f"Non-finite values found in {field!r} of {path.name}")

    filename_index = _episode_index(path)
    if "episode_index" in table.column_names:
        stored_indices = set(int(value) for value in table["episode_index"].to_pylist())
        if stored_indices != {filename_index}:
            raise SystemExit(
                f"Episode index mismatch in {path.name}: stored values {sorted(stored_indices)}"
            )

    duration = None
    if "timestamp" in table.column_names:
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        if timestamps.size > 1 and np.isfinite(timestamps).all():
            duration = float(timestamps[-1] - timestamps[0])
    return table.num_rows, values.mean(axis=0), duration


def _display_names(metadata_names: list[str] | None, joint_count: int) -> list[str]:
    if metadata_names is None:
        return [f"joint_{index}" for index in range(joint_count)]
    if len(metadata_names) != joint_count:
        raise SystemExit(
            f"Metadata contains {len(metadata_names)} joint names, but the selected "
            f"field contains {joint_count} values per frame"
        )
    return metadata_names


def _print_episode_block(
    path: Path,
    field: str,
    frame_count: int,
    duration: float | None,
    means: Any,
    names: list[str],
    joint_indices: list[int],
    precision: int,
) -> None:
    episode_index = _episode_index(path)
    duration_text = f" | duration: {duration:.3f} s" if duration is not None else ""
    print("\n" + "=" * BLOCK_WIDTH)
    print(f"EPISODE {episode_index:06d}")
    print("=" * BLOCK_WIDTH)
    print(f"file:     {path}")
    print(f"field:    {field}")
    print(f"frames:   {frame_count}{duration_text}")
    print("-" * BLOCK_WIDTH)
    print(f"{'index':>5}  {'joint':<48} {'mean':>18}")
    print(f"{'-' * 5}  {'-' * 48} {'-' * 18}")
    for index in joint_indices:
        print(f"{index:>5}  {names[index]:<48} {means[index]:>18.{precision}f}")
    print("=" * BLOCK_WIDTH, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print per-joint mean values separately for every Parquet episode."
    )
    parser.add_argument(
        "chunk_directory",
        type=Path,
        help="directory containing episode_*.parquet files, e.g. data/chunk-000",
    )
    parser.add_argument(
        "--field",
        default="observation.state",
        help="vector field to average (default: observation.state)",
    )
    parser.add_argument(
        "--episode-indices",
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help="only process selected episode indices",
    )
    parser.add_argument(
        "--joint-indices",
        "--joints",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help="only print selected vector/joint indices",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="digits after the decimal point (default: 6)",
    )
    args = parser.parse_args()

    if not args.chunk_directory.is_dir():
        parser.error(f"directory does not exist: {args.chunk_directory}")
    if args.precision < 0 or args.precision > 15:
        parser.error("--precision must be between 0 and 15")
    for option_name in ("episode_indices", "joint_indices"):
        values = getattr(args, option_name)
        if values is None:
            continue
        if any(value < 0 for value in values):
            parser.error(f"--{option_name.replace('_', '-')} must be non-negative")
        if len(values) != len(set(values)):
            parser.error(f"--{option_name.replace('_', '-')} must not contain duplicates")
    return args


def main() -> int:
    args = parse_args()
    np, parquet = _import_dependencies()
    episode_files = _resolve_episode_files(
        args.chunk_directory.resolve(), args.episode_indices
    )
    metadata_names, info_path = _load_joint_names(args.chunk_directory.resolve(), args.field)

    print("=" * BLOCK_WIDTH)
    print("PER-EPISODE JOINT MEAN CHECK")
    print("=" * BLOCK_WIDTH)
    print(f"chunk:    {args.chunk_directory.resolve()}")
    print(f"episodes: {len(episode_files)}")
    print(f"field:    {args.field}")
    print(f"metadata: {info_path if info_path is not None else 'not found'}")

    expected_joint_count: int | None = None
    total_frames = 0
    for path in episode_files:
        frame_count, means, duration = _read_episode(path, args.field, np, parquet)
        joint_count = int(means.size)
        if expected_joint_count is None:
            expected_joint_count = joint_count
        elif joint_count != expected_joint_count:
            raise SystemExit(
                f"Joint count changed in {path.name}: {joint_count}, "
                f"expected {expected_joint_count}"
            )

        names = _display_names(metadata_names, joint_count)
        joint_indices = (
            list(range(joint_count))
            if args.joint_indices is None
            else args.joint_indices
        )
        invalid = [index for index in joint_indices if index >= joint_count]
        if invalid:
            raise SystemExit(
                f"Invalid joint indices {invalid}; {args.field!r} has "
                f"{joint_count} values (valid range: 0..{joint_count - 1})"
            )
        _print_episode_block(
            path=path,
            field=args.field,
            frame_count=frame_count,
            duration=duration,
            means=means,
            names=names,
            joint_indices=joint_indices,
            precision=args.precision,
        )
        total_frames += frame_count

    print("\n" + "#" * BLOCK_WIDTH)
    print(
        f"DONE: {len(episode_files)} episodes, {total_frames} frames, "
        f"field={args.field!r}"
    )
    print("#" * BLOCK_WIDTH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
