#!/usr/bin/env python3
"""Create a LeRobot dataset copy in which every episode has one task.

The task is represented in three pieces of LeRobot metadata:

* ``meta/tasks.jsonl`` maps an integer ``task_index`` to task text;
* the ``task_index`` column in every parquet file assigns that task per frame;
* ``meta/episodes.jsonl`` contains the human-readable ``tasks`` list per episode.

Run this from the data-collection environment, for example::

    .venv_data_collection/bin/python gear_sonic/scripts/process_dataset/set_dataset_task.py \
        outputs/sep-10-2026-cleaned-merged \
        --output-path outputs/sep-10-2026-hug-only --task hug

Use ``--dry-run`` first when working with a valuable dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - depends on the caller's environment
    raise SystemExit(
        "pyarrow is required. Run this with .venv_data_collection/bin/python."
    ) from exc


TASK_INDEX = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path, help="Source LeRobot dataset root (never modified)")
    parser.add_argument(
        "--output-path", required=True, type=Path,
        help="New dataset directory; it must not already exist",
    )
    parser.add_argument("--task", required=True, help="The sole task text, e.g. 'hug'")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report changes without writing files"
    )
    parser.add_argument(
        "--hardlink-unchanged-files", action="store_true",
        help=(
            "Use hard links for the initial copy to save disk space. Parquet and metadata "
            "are safely replaced afterwards, but do not edit linked videos in either dataset."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            records.append(record)
    return records


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write a replacement in the same directory, then atomically rename it."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = Path(tmp.name)
    try:
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def jsonl_bytes(records: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )


def dataset_parquet_paths(root: Path) -> list[Path]:
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise ValueError(f"No parquet files found under {root / 'data'}")
    return paths


def validate_dataset(root: Path) -> tuple[Path, Path, Path, list[Path], list[dict]]:
    meta = root / "meta"
    info_path = meta / "info.json"
    tasks_path = meta / "tasks.jsonl"
    episodes_path = meta / "episodes.jsonl"
    missing = [str(path) for path in (info_path, tasks_path, episodes_path) if not path.is_file()]
    if missing:
        raise ValueError("Not a complete LeRobot dataset; missing: " + ", ".join(missing))

    episodes = load_jsonl(episodes_path)
    if not episodes:
        raise ValueError(f"No episode records found in {episodes_path}")
    episode_indices = [record.get("episode_index") for record in episodes]
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in episode_indices):
        raise ValueError(f"Every record in {episodes_path} must have a non-negative integer episode_index")
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError(f"Duplicate episode_index values in {episodes_path}")

    parquet_paths = dataset_parquet_paths(root)
    for parquet_path in parquet_paths:
        schema = pq.read_schema(parquet_path)
        if "task_index" not in schema.names:
            raise ValueError(f"Missing task_index column: {parquet_path}")
    return info_path, tasks_path, episodes_path, parquet_paths, episodes


def rewrite_parquet_task_index(path: Path) -> int:
    table = pq.read_table(path)
    column_index = table.schema.get_field_index("task_index")
    old_column = table.column(column_index)
    # Keep the original Arrow type (normally int64) so LeRobot metadata stays valid.
    replacement = pa.array([TASK_INDEX] * table.num_rows, type=old_column.type)
    table = table.set_column(column_index, "task_index", replacement)

    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        pq.write_table(table, temp_path, compression="zstd")
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return table.num_rows


def main() -> int:
    args = parse_args()
    task = args.task.strip()
    if not task:
        raise SystemExit("--task must not be empty or whitespace")

    source_root = args.dataset_path.expanduser().resolve()
    output_root = args.output_path.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Source dataset directory does not exist: {source_root}")
    if output_root.exists():
        raise SystemExit(f"Output path already exists and will not be overwritten: {output_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise SystemExit("--output-path must not be the source dataset or a directory inside it")

    try:
        info_path, tasks_path, episodes_path, parquet_paths, episodes = validate_dataset(source_root)
        with info_path.open(encoding="utf-8") as stream:
            info = json.load(stream)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Validation failed: {exc}") from exc

    frame_count = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_paths)
    print(f"Source dataset: {source_root}")
    print(f"Output dataset: {output_root}")
    print(f"Episodes: {len(episodes)}; parquet files: {len(parquet_paths)}; frames: {frame_count}")
    print(f"New sole task: task_index={TASK_INDEX}, task={task!r}")
    if args.dry_run:
        print("Dry run: no files were changed.")
        return 0

    copy_function = os.link if args.hardlink_unchanged_files else shutil.copy2
    try:
        shutil.copytree(source_root, output_root, copy_function=copy_function)
    except OSError as exc:
        raise SystemExit(
            f"Could not create output dataset at {output_root}: {exc}. "
            "The source dataset was not modified."
        ) from exc

    # From this point onward all writes use the independent output directory.
    info_path = output_root / "meta/info.json"
    tasks_path = output_root / "meta/tasks.jsonl"
    episodes_path = output_root / "meta/episodes.jsonl"
    parquet_paths = dataset_parquet_paths(output_root)

    # Rewrite metadata only after all parquet schemas have passed validation.
    rewritten_episodes = [{**record, "tasks": [task]} for record in episodes]
    info["total_tasks"] = 1

    # The parquet files are individually atomic. Do not interrupt this process once
    # it starts; a Ctrl+C can otherwise leave the *output* dataset partially updated.
    for index, parquet_path in enumerate(parquet_paths, start=1):
        rewrite_parquet_task_index(parquet_path)
        if index % 50 == 0 or index == len(parquet_paths):
            print(f"Rewrote task_index in {index}/{len(parquet_paths)} parquet files")

    atomic_write_bytes(tasks_path, jsonl_bytes([{"task_index": TASK_INDEX, "task": task}]))
    atomic_write_bytes(episodes_path, jsonl_bytes(rewritten_episodes))
    atomic_write_bytes(
        info_path,
        (json.dumps(info, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"),
    )
    print("Done. All frames and episode metadata now use the requested single task.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
