#!/usr/bin/env python3
"""Merge all recursively discovered custom LeRobot datasets into one dataset.

No episode is filtered by default, including episodes still marked discarded.
Use remove_discarded.py first (recommended), or explicitly pass
--exclude-discarded. Custom interaction metadata, raw telemetry, depth data,
and unknown path-indexed episode sidecars are retained and reindexed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from typing import Any

from _dataset_utils import (
    choose_canonical_features,
    commit_staging,
    copy_asset,
    discover_datasets,
    ensure_safe_paths,
    get_video_features,
    indexed_records,
    iter_assets,
    load_dataset,
    make_staging_directory,
    read_jsonl,
    remap_episode_record,
    remove_staging,
    rewrite_parquet_indices,
    transcode_video,
    update_info_counts,
    validate_asset_indices,
    video_dimensions,
    video_key_for_path,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively merge custom LeRobot datasets.")
    parser.add_argument("input_root", type=Path, help="Folder containing nested LeRobot datasets")
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Merged dataset folder (default: <input>-merged next to input)",
    )
    parser.add_argument(
        "--exclude-discarded",
        action="store_true",
        help="Explicitly omit discarded episodes (default keeps absolutely every episode)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan only")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output folder")
    return parser.parse_args()


def task_maps(dataset: Path) -> tuple[dict[int, str], list[dict[str, Any]]]:
    records = read_jsonl(dataset / "meta" / "tasks.jsonl")
    mapping = {int(record["task_index"]): str(record["task"]) for record in records}
    if len(mapping) != len(records):
        raise ValueError(f"Duplicate task indices in {dataset / 'meta/tasks.jsonl'}")
    return mapping, records


def common_script_config(infos: list[dict[str, Any]], canonical_features: dict[str, Any]) -> dict[str, Any] | None:
    """Retain shared acquisition config while removing per-session identity."""
    configs = [deepcopy(info.get("script_config")) for info in infos]
    if not configs or configs[0] is None:
        return None
    for config in configs:
        if config is None:
            return None
        config.pop("session_profile", None)
    base = configs[0]
    for key in list(base):
        if any(config.get(key) != base[key] for config in configs[1:]):
            base.pop(key)
    webcam = canonical_features.get("observation.images.webcam")
    if webcam:
        width, height = video_dimensions(webcam)
        base["webcam_resolution"] = [width, height]
    base["merged_sessions"] = True
    return base


def person_name(input_root: Path, dataset: Path) -> str:
    """Return the subject directory name for a nested session dataset."""
    relative = dataset.relative_to(input_root)
    if len(relative.parts) >= 2:
        return relative.parts[0]
    # Also produce a useful value if this script is run on one dataset directly.
    return dataset.parent.name


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output = (
        args.output_path.resolve()
        if args.output_path
        else input_root.with_name(f"{input_root.name}-merged")
    )
    ensure_safe_paths(input_root, output)
    datasets = discover_datasets(input_root)

    loaded = [load_dataset(dataset) for dataset in datasets]
    infos = [item[0] for item in loaded]
    fps_values = {int(info.get("fps", 0)) for info in infos}
    if len(fps_values) != 1:
        raise ValueError(f"Datasets have different FPS values: {sorted(fps_values)}")
    fps = next(iter(fps_values))
    canonical_features = choose_canonical_features(infos)
    canonical_video_features = get_video_features({"features": canonical_features})

    global_tasks: list[str] = []
    source_task_maps: list[dict[int, str]] = []
    for dataset in datasets:
        mapping, _ = task_maps(dataset)
        source_task_maps.append(mapping)
        for task in mapping.values():
            if task not in global_tasks:
                global_tasks.append(task)
    task_to_index = {task: index for index, task in enumerate(global_tasks)}

    selected: list[tuple[int, dict[str, Any]]] = []
    discarded_count = 0
    for dataset_number, ((info, episodes), dataset) in enumerate(zip(loaded, datasets)):
        known = {int(record["episode_index"]) for record in episodes}
        validate_asset_indices(dataset, known)
        discarded = {int(value) for value in info.get("discarded_episode_indices", [])}
        for episode in episodes:
            old_index = int(episode["episode_index"])
            if args.exclude_discarded and old_index in discarded:
                discarded_count += 1
                continue
            selected.append((dataset_number, episode))

    if not selected:
        raise ValueError("No episodes selected for merge")
    print(f"Found {len(datasets)} datasets and {len(selected)} selected episodes.")
    if args.exclude_discarded:
        print(f"Explicitly excluding {discarded_count} discarded episodes.")
    else:
        print("Discarded markers are ignored: every episode will be retained.")
    print(f"Tasks: {global_tasks}")
    for key, feature in canonical_video_features.items():
        width, height = video_dimensions(feature)
        variants = {video_dimensions(info["features"][key]) for info in infos}
        if len(variants) > 1:
            print(f"Video {key}: normalize {sorted(variants)} -> {(width, height)} without frame filtering")
    print(f"Output: {output}")
    if args.dry_run:
        print("Dry run complete; nothing was written.")
        return 0
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite")

    staging = make_staging_directory(output)
    try:
        reference_info = deepcopy(infos[0])
        chunks_size = int(reference_info.get("chunks_size", 1000))
        assets = []
        stats_maps = []
        interaction_maps = []
        for dataset, (_, episodes) in zip(datasets, loaded):
            by_episode = {int(record["episode_index"]): [] for record in episodes}
            for relative, index in iter_assets(dataset):
                if index is not None:
                    by_episode[index].append(relative)
            assets.append(by_episode)
            stats_maps.append(indexed_records(dataset / "meta" / "episodes_stats.jsonl"))
            interaction_maps.append(indexed_records(dataset / "meta" / "interaction_metadata.jsonl"))

        output_episodes = []
        output_stats = []
        output_interactions = []
        episode_provenance = []
        global_frame_index = 0

        for new_index, (dataset_number, episode) in enumerate(selected):
            dataset = datasets[dataset_number]
            info = infos[dataset_number]
            old_index = int(episode["episode_index"])
            source_tasks = source_task_maps[dataset_number]
            parquet_task_map = {old: task_to_index[name] for old, name in source_tasks.items()}
            parquet_rows: list[int] = []

            def transform_video(src: Path, dst: Path, relative: Path) -> None:
                key = video_key_for_path(relative, canonical_video_features)
                if key is None:
                    shutil.copy2(src, dst)
                    return
                source_feature = info["features"][key]
                target_feature = canonical_video_features[key]
                if video_dimensions(source_feature) == video_dimensions(target_feature):
                    shutil.copy2(src, dst)
                else:
                    width, height = video_dimensions(target_feature)
                    transcode_video(src, dst, width, height, fps)

            for relative in assets[dataset_number][old_index]:
                rows = copy_asset(
                    dataset,
                    staging,
                    relative,
                    old_index,
                    new_index,
                    chunks_size,
                    parquet_transform=lambda src, dst, ni=new_index, gi=global_frame_index, tm=parquet_task_map: rewrite_parquet_indices(
                        src, dst, ni, gi, tm
                    ),
                    video_transform=transform_video,
                )
                if rows is not None:
                    parquet_rows.append(rows)
            if len(parquet_rows) != 1:
                raise ValueError(
                    f"{dataset}: episode {old_index} has {len(parquet_rows)} parquet files; expected 1"
                )
            length = parquet_rows[0]
            if length != int(episode.get("length", length)):
                raise ValueError(f"{dataset}: episode {old_index} parquet/metadata length mismatch")
            remapped = remap_episode_record(episode, old_index, new_index, chunks_size)
            remapped["length"] = length
            output_episodes.append(remapped)
            if old_index in stats_maps[dataset_number]:
                output_stats.append(
                    remap_episode_record(
                        stats_maps[dataset_number][old_index], old_index, new_index, chunks_size
                    )
                )
            if old_index in interaction_maps[dataset_number]:
                interaction = remap_episode_record(
                    interaction_maps[dataset_number][old_index], old_index, new_index, chunks_size
                )
            else:
                # Keep the merged sidecar complete even for an older source
                # dataset that did not yet write interaction_metadata.jsonl.
                interaction = {"episode_index": new_index}
            interaction["person"] = person_name(input_root, dataset)
            output_interactions.append(interaction)
            episode_provenance.append(
                {
                    "episode_index": new_index,
                    "source_dataset": str(dataset.relative_to(input_root)),
                    "source_episode_index": old_index,
                    "source_was_discarded": old_index
                    in {int(value) for value in info.get("discarded_episode_indices", [])},
                }
            )
            global_frame_index += length
            if (new_index + 1) % 25 == 0 or new_index + 1 == len(selected):
                print(f"Merged {new_index + 1}/{len(selected)} episodes")

        meta = staging / "meta"
        write_jsonl(meta / "episodes.jsonl", output_episodes)
        if output_stats:
            write_jsonl(meta / "episodes_stats.jsonl", output_stats)
        if output_interactions:
            write_jsonl(meta / "interaction_metadata.jsonl", output_interactions)
        write_jsonl(
            meta / "tasks.jsonl",
            ({"task_index": index, "task": task} for index, task in enumerate(global_tasks)),
        )
        write_jsonl(meta / "episode_provenance.jsonl", episode_provenance)
        write_jsonl(
            meta / "source_datasets.jsonl",
            (
                {
                    "source_dataset_index": index,
                    "source_dataset": str(dataset.relative_to(input_root)),
                    "source_info": info,
                }
                for index, (dataset, info) in enumerate(zip(datasets, infos))
            ),
        )

        modality = datasets[0] / "meta" / "modality.json"
        if modality.exists():
            shutil.copy2(modality, meta / "modality.json")
        reference_info["features"] = canonical_features
        reference_info["script_config"] = common_script_config(infos, canonical_features)
        reference_info["merged_dataset"] = {
            "source_dataset_count": len(datasets),
            "source_datasets_path": "meta/source_datasets.jsonl",
            "episode_provenance_path": "meta/episode_provenance.jsonl",
        }
        write_json(
            meta / "info.json",
            update_info_counts(
                reference_info, staging, len(selected), global_frame_index, len(global_tasks)
            ),
        )
        commit_staging(staging, output, args.overwrite)
    except Exception:
        remove_staging(staging)
        raise

    print(f"Done: {output} ({len(selected)} episodes, {global_frame_index} frames)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
