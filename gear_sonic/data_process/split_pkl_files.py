#!/usr/bin/env python3
"""
Script to break large pickle files into individual motion sequence files.

This script reads large pickle files and breaks each motion sequence
within them into individual pickle files in subdirectories.
This enables motion_lib_base.py to use directory mode for efficient loading.
Supports any pkl file format, not just bone motion files.
"""

import argparse
from pathlib import Path
import shutil
import sys
import time

import joblib
from tqdm import tqdm


def create_output_structure(output_dir, clean=False):
    """Create the output directory structure."""
    output_path = Path(output_dir)

    if clean and output_path.exists():
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_path}")
    return output_path


def extract_motion_metadata(motion_data):
    """Extract metadata (length, fps) from motion data."""
    metadata = {}

    # Check common fields that might indicate fps
    fps = None
    if hasattr(motion_data, "get"):
        fps = motion_data.get("fps", motion_data.get("frame_rate", motion_data.get("framerate")))

    # If no fps found, try to infer from common values or set default
    if fps is None:
        fps = 30.0  # Default fps

    # Get length - check for common motion data structures
    length = 0
    length = motion_data["root_trans_offset"].shape[0]

    # If still no length found, try to get it from the data structure
    if length == 0 and hasattr(motion_data, "__len__"):
        length = len(motion_data)

    return {"length": length, "fps": fps, "duration": length / fps if fps > 0 else 0.0}


def process_motion_file(input_file, output_dir, verbose=False):
    """Process a single large motion file and break it into individual files."""
    input_path = Path(input_file)

    # Create subdirectory named after the input file
    file_subdir = input_path.stem
    output_path = Path(output_dir) / file_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Loading {input_path.name}...")
        motion_data = joblib.load(input_path)
        print(f"Loaded {len(motion_data)} motion sequences -> {output_path}")

        # Collect metadata for all motion sequences
        metadata = {}

        # Process each motion sequence
        for motion_key, motion_sequence_data in tqdm(
            motion_data.items(), desc=f"Processing {input_path.name}"
        ):
            individual_filepath = output_path / f"{motion_key}.pkl"
            individual_dict = {motion_key: motion_sequence_data}
            joblib.dump(individual_dict, individual_filepath)

            # Extract metadata for this motion sequence
            motion_metadata = extract_motion_metadata(motion_sequence_data)
            metadata[motion_key] = motion_metadata

        # Save metadata file
        metadata_filepath = output_path / "metadata.pkl"
        joblib.dump(metadata, metadata_filepath)

        if verbose:
            print(
                f"Successfully processed {input_path.name} -> {len(motion_data)} individual files + metadata"
            )
        return True

    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Break pickle files into individual sequences")
    parser.add_argument("input", help="Input directory containing pkl files or single pkl file")
    parser.add_argument(
        "--output",
        default="data/processed_pkl/",
        help="Output directory for individual motion files",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--file-pattern",
        default="*.pkl",
        help="Pattern to match input files (only used for directories)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove output directory if it already exists"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist!")
        return 1

    # Create output directory
    output_path = create_output_structure(args.output, args.clean)

    # Determine input files
    if input_path.is_file():
        # Single file input
        if not input_path.suffix == ".pkl":
            print(f"Error: Input file must be a .pkl file, got {input_path.suffix}")
            return 1
        input_files = [input_path]
        print(f"Processing single file: {input_path}")
    else:
        # Directory input
        input_files = sorted(input_path.glob(args.file_pattern))
        if not input_files:
            print(f"No files found matching pattern {args.file_pattern} in {input_path}")
            return 1
        print(f"Processing directory: {input_path}")

    print(f"Found {len(input_files)} files to process:")
    for f in input_files[:5]:
        print(f"  {f.name}")
    if len(input_files) > 5:
        print(f"  ... and {len(input_files) - 5} more files")

    # Process all files
    successful = failed = total_individual_files = 0
    start_time = time.time()

    for input_file in input_files:
        print(f"\n{'='*60}")
        if process_motion_file(input_file, output_path, args.verbose):
            successful += 1
            # Count files created (excluding metadata)
            subdir_path = output_path / input_file.stem
            individual_files = [
                f for f in subdir_path.glob("*.pkl") if not f.name.endswith("metadata.pkl")
            ]
            total_individual_files += len(individual_files)
            print(
                f"Created {len(individual_files)} individual files + metadata from {input_file.name}"
            )
        else:
            failed += 1

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Files processed successfully: {successful}")
    print(f"Files failed: {failed}")
    print(f"Total individual motion files created: {total_individual_files}")
    print(f"Processing time: {elapsed:.2f} seconds")

    if successful > 0:
        print(f"\nSuccess! Individual motion files are available in: {output_path}")
        print(f"To use with motion_lib_base.py: motion_file = '{output_path}'")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
