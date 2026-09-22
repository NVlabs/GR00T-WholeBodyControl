#!/usr/bin/env python3
"""Remove only explicitly discarded episodes from every nested dataset.

The output mirrors the input directory layout. Surviving parquet rows are not
filtered; only LeRobot bookkeeping indices are rewritten after renumbering.
Videos, raw telemetry, depth, and unknown episode sidecars are copied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _dataset_utils import (
    commit_staging,
    copy_asset,
    copy_non_episode_assets,
    discover_datasets,
    ensure_safe_paths,
    indexed_records,
    iter_assets,
    load_dataset,
    make_staging_directory,
    read_jsonl,
    remap_episode_record,
    remove_staging,
    rewrite_parquet_indices,
    update_info_counts,
    validate_asset_indices,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively remove only episodes listed in discarded_episode_indices."
    )
    parser.add_argument("input_root", type=Path, help="Folder containing nested LeRobot datasets")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="New mirrored folder (default: <input>-cleaned next to input)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts only")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output folder")
    return parser.parse_args()


def clean_dataset(source: Path, destination: Path) -> tuple[int, int, int]:
    info, episodes = load_dataset(source)
    known_indices = {int(record["episode_index"]) for record in episodes}
    validate_asset_indices(source, known_indices)
    discarded = {int(index) for index in info.get("discarded_episode_indices", [])}
    unknown_discarded = discarded - known_indices
    if unknown_discarded:
        raise ValueError(
            f"{source}: discarded indices are absent from episodes.jsonl: "
            f"{sorted(unknown_discarded)}"
        )
    kept = [record for record in episodes if int(record["episode_index"]) not in discarded]
    mapping = {int(record["episode_index"]): new for new, record in enumerate(kept)}
    chunks_size = int(info.get("chunks_size", 1000))
    stats = indexed_records(source / "meta" / "episodes_stats.jsonl")
    interactions = indexed_records(source / "meta" / "interaction_metadata.jsonl")
    tasks = read_jsonl(source / "meta" / "tasks.jsonl")

    destination.mkdir(parents=True, exist_ok=True)
    copy_non_episode_assets(source, destination)
    global_index = 0
    output_episodes = []
    output_stats = []
    output_interactions = []

    assets_by_episode: dict[int, list[Path]] = {index: [] for index in known_indices}
    for relative, index in iter_assets(source):
        if index is not None:
            assets_by_episode[index].append(relative)

    for episode in kept:
        old_index = int(episode["episode_index"])
        new_index = mapping[old_index]
        parquet_rows: list[int] = []
        for relative in assets_by_episode[old_index]:
            rows = copy_asset(
                source,
                destination,
                relative,
                old_index,
                new_index,
                chunks_size,
                parquet_transform=lambda src, dst, ni=new_index, gi=global_index: rewrite_parquet_indices(
                    src, dst, ni, gi
                ),
            )
            if rows is not None:
                parquet_rows.append(rows)
        if len(parquet_rows) != 1:
            raise ValueError(
                f"{source}: episode {old_index} has {len(parquet_rows)} parquet files; expected 1"
            )
        length = parquet_rows[0]
        expected_length = int(episode.get("length", length))
        if length != expected_length:
            raise ValueError(
                f"{source}: episode {old_index} metadata length {expected_length} != parquet {length}"
            )
        remapped = remap_episode_record(episode, old_index, new_index, chunks_size)
        remapped["length"] = length
        output_episodes.append(remapped)
        if old_index in stats:
            output_stats.append(remap_episode_record(stats[old_index], old_index, new_index, chunks_size))
        if old_index in interactions:
            output_interactions.append(
                remap_episode_record(interactions[old_index], old_index, new_index, chunks_size)
            )
        global_index += length

    meta = destination / "meta"
    write_jsonl(meta / "episodes.jsonl", output_episodes)
    if stats:
        write_jsonl(meta / "episodes_stats.jsonl", output_stats)
    if interactions:
        write_jsonl(meta / "interaction_metadata.jsonl", output_interactions)
    if tasks:
        write_jsonl(meta / "tasks.jsonl", tasks)
    write_json(
        meta / "info.json",
        update_info_counts(info, destination, len(kept), global_index, len(tasks)),
    )
    return len(episodes), len(kept), global_index


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else input_root.with_name(f"{input_root.name}-cleaned")
    )
    ensure_safe_paths(input_root, output_root)
    datasets = discover_datasets(input_root)

    plans = []
    total_before = total_after = 0
    for dataset in datasets:
        info, episodes = load_dataset(dataset)
        known = {int(record["episode_index"]) for record in episodes}
        validate_asset_indices(dataset, known)
        discarded = {int(index) for index in info.get("discarded_episode_indices", [])}
        unknown_discarded = discarded - known
        if unknown_discarded:
            raise ValueError(
                f"{dataset}: discarded indices are absent from episodes.jsonl: "
                f"{sorted(unknown_discarded)}"
            )
        total_before += len(episodes)
        total_after += len(episodes) - len(discarded)
        plans.append((dataset, discarded))
        print(f"{dataset.relative_to(input_root)}: {len(episodes)} -> {len(episodes)-len(discarded)} episodes; discard {sorted(discarded)}")

    print(
        f"\nFound {len(datasets)} datasets: {total_before} episodes, "
        f"{total_before-total_after} discarded, {total_after} will be kept."
    )
    print(f"Output: {output_root}")
    if args.dry_run:
        print("Dry run complete; nothing was written.")
        return 0
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_root}; use --overwrite")

    staging = make_staging_directory(output_root)
    try:
        for number, (dataset, _) in enumerate(plans, 1):
            relative = dataset.relative_to(input_root) if dataset != input_root else Path(".")
            destination = staging / relative
            before, after, frames = clean_dataset(dataset, destination)
            print(f"[{number}/{len(plans)}] {relative}: kept {after}/{before} episodes, {frames} frames")
        commit_staging(staging, output_root, args.overwrite)
    except Exception:
        remove_staging(staging)
        raise
    print(f"Done: {output_root} ({total_after}/{total_before} episodes kept)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
