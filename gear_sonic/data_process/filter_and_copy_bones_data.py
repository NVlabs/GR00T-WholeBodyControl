#!/usr/bin/env python3
"""
Script to filter and copy bones data from bones_gmr to single_pkls directory.

This script copies motion files from the bones_gmr directory structure while
filtering out unwanted sequences based on keywords in filenames.
It preserves the bones_xxx directory structure in the destination.
"""

import argparse
from functools import partial
import glob
from multiprocessing import Pool, cpu_count
import os.path as osp
from pathlib import Path
import shutil

from tqdm import tqdm


def should_filter_out(filename, filter_keywords, include_keywords=None):
    """
    Check if a filename contains any of the filter keywords.

    Args:
        filename (str): The filename to check
        filter_keywords (list): List of keywords to filter out

    Returns:
        bool: True if file should be filtered out, False otherwise
    """
    filename_lower = filename.lower()
    if include_keywords is None:
        return any(keyword.lower() in filename_lower for keyword in filter_keywords)
    else:
        return any(keyword.lower() in filename_lower for keyword in filter_keywords) or (
            not any(keyword.lower() in filename_lower for keyword in include_keywords)
        )


def process_bones_directory(
    bones_dir, dest_path, filter_keywords, dry_run, verbose, include_keywords=None
):
    """
    Process a single bones directory - worker function for multiprocessing.

    Args:
        bones_dir (Path): Source bones directory to process
        dest_path (Path): Destination base directory
        filter_keywords (list): Keywords to filter out
        dry_run (bool): If True, don't actually copy files
        verbose (bool): If True, show detailed output

    Returns:
        tuple: (total_files, copied_files, filtered_files)
    """
    dest_bones_dir = dest_path / bones_dir.name

    if not dry_run:
        dest_bones_dir.mkdir(parents=True, exist_ok=True)

    # Find all pkl files in this bones directory

    pkl_files = list(glob.glob(osp.join(bones_dir, "**", "*.pkl"), recursive=True))
    total_files = len(pkl_files)
    copied_files = 0
    filtered_files = 0

    if verbose:
        print(f"Processing {bones_dir.name}: {total_files} files")

    for pkl_file in pkl_files:
        base = osp.basename(pkl_file)
        parent = osp.basename(osp.dirname(pkl_file))
        name_to_check = f"{parent}/{base}"

        if (
            should_filter_out(name_to_check, filter_keywords, include_keywords)
            and not base == "metadata.pkl"
        ):
            filtered_files += 1
            if verbose:
                print(f"  FILTERED: {osp.basename(pkl_file)}")
        else:
            copied_files += 1
            dest_file = osp.join(dest_bones_dir, osp.basename(pkl_file))

            if not dry_run:
                shutil.copy2(pkl_file, dest_file)

    return (total_files, copied_files, filtered_files)


def copy_filtered_bones_data(
    source_dir,
    dest_dir,
    filter_keywords,
    dry_run=False,
    verbose=False,
    workers=None,
    filter_file=None,
):
    """
    Copy bones data while filtering out unwanted sequences.

    Args:
        source_dir (Path): Source directory containing bones_xxx subdirs
        dest_dir (Path): Destination directory
        filter_keywords (list): Keywords to filter out
        dry_run (bool): If True, only show what would be copied
        verbose (bool): If True, show detailed output
        workers (int): Number of worker processes. If None, uses all CPU cores.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    include_keywords = None
    if filter_file is not None:
        with open(filter_file) as f:
            include_keywords = f.read().splitlines()

    if not source_path.exists():
        print(f"Error: Source directory {source_path} does not exist!")
        return False

    # Find all bones_xxx directories
    bones_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    if not bones_dirs:
        print(f"No bones_xxx directories found in {source_path}")
        return False

    # Determine number of workers
    if workers is None:
        workers = cpu_count()
    workers = max(1, min(workers, len(bones_dirs)))  # Don't use more workers than directories

    print(f"Found {len(bones_dirs)} bones directories to process")
    print(f"Using {workers} worker processes")

    if verbose:
        for d in bones_dirs:
            print(f"  {d.name}")

    # Process directories in parallel
    worker_func = partial(
        process_bones_directory,
        dest_path=dest_path,
        filter_keywords=filter_keywords,
        dry_run=dry_run,
        verbose=verbose,
        include_keywords=include_keywords,
    )

    total_files = 0
    copied_files = 0
    filtered_files = 0
    if workers == 1:
        # Single-threaded execution for easier debugging
        results = []
        for bones_dir in tqdm(bones_dirs, desc="Processing bones directories"):
            results.append(worker_func(bones_dir))
    else:
        # Multi-process execution
        with Pool(processes=workers) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_func, bones_dirs),
                    total=len(bones_dirs),
                    desc="Processing bones directories",
                )
            )

    # Aggregate results
    for result in results:
        total_files += result[0]
        copied_files += result[1]
        filtered_files += result[2]

    # Summary
    print(f"\n{'='*60}")
    print("FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Source directory: {source_path}")
    print(f"Destination directory: {dest_path}")
    print(f"Total files found: {total_files}")
    print(f"Files copied: {copied_files}")
    print(f"Files filtered out: {filtered_files}")
    print(f"Filter keywords: {', '.join(filter_keywords)}")

    if dry_run:
        print("\nDRY RUN - No files were actually copied")
    else:
        print(f"\nFiles successfully copied to: {dest_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Filter and copy bones data")
    parser.add_argument(
        "--source",
        default="data/bones_gmr/0903_all/",
        help="Source directory containing bones_xxx subdirectories",
    )
    parser.add_argument(
        "--dest", default="data/single_pkls/", help="Destination directory for filtered bones data"
    )
    parser.add_argument(
        "--filter-keywords",
        default=[
            "bed",
            "bike",
            "chair",
            "climb",
            "com_up_50cm",
            "sitting",
            "step_on",
            "seat",
            "table",
            "_sit_",
            "sit_", "ladder",
            "crutch",
            "_bed_",
            "_ride_",
            "scooter",
            "stepdown",
            "acrobatics_",
            "box_HSPU",
            "cartwheel",
            "50cm_box_",
            "on_box", "fall_from",
            "handstand_ff_",
            "on_1m",
            "form_box",
            "off_1m",
            "230m",
            "jump_over_obstacle_",
            "lift_crate_come_up_",
            "jump_to_shoulder_roll",
            "kozak_dance",
            "stair",
            "handstand",
            "box_jump",
            "monkey_jump",
            "safety_roll",
            "box_dips",
            "walking_on_edge",
            "push_obstacle",
        ],
        nargs="+",
        help="Keywords to filter out from filenames",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be copied without actually copying"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--add-keywords", nargs="+", help="Additional keywords to add to the default filter list"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: use all CPU cores)",
    )

    parser.add_argument("--filter_file", default=None, help="Filter file to use")

    args = parser.parse_args()

    # Combine default and additional keywords
    filter_keywords = args.filter_keywords
    if args.add_keywords:
        filter_keywords.extend(args.add_keywords)

    print(f"Filtering out files containing: {', '.join(filter_keywords)}")

    success = copy_filtered_bones_data(
        source_dir=args.source,
        dest_dir=args.dest,
        filter_keywords=filter_keywords,
        dry_run=args.dry_run,
        verbose=args.verbose,
        workers=args.workers,
        filter_file=args.filter_file,
    )

    return 0 if success else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
