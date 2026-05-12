#!/usr/bin/env python3
"""
Download GEAR-SONIC model checkpoints and training data from Hugging Face Hub.

Repository: https://huggingface.co/nvidia/GEAR-SONIC

Usage:
    python download_from_hf.py                    # ONNX models for deployment
    python download_from_hf.py --training          # PyTorch checkpoint + SMPL data
    python download_from_hf.py --sample            # Sample data only (quick start)
    python download_from_hf.py --output-dir /path  # custom output directory
    python download_from_hf.py --no-planner        # skip planner model
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = "nvidia/GEAR-SONIC"

# (filename in HF repo, local destination relative to output_dir)
POLICY_FILES = [
    ("model_encoder.onnx", "policy/release/model_encoder.onnx"),
    ("model_decoder.onnx", "policy/release/model_decoder.onnx"),
    ("observation_config.yaml", "policy/release/observation_config.yaml"),
]

PLANNER_FILE = ("planner_sonic.onnx", "planner/target_vel/V2/planner_sonic.onnx")

TRAINING_FILES = [
    ("sonic_release/last.pt", "sonic_release/last.pt"),
    ("sonic_release/config.yaml", "sonic_release/config.yaml"),
]

SMPL_TAR_PARTS_PREFIX = "bones_seed_smpl/bones_seed_smpl.tar.part_"
SMPL_TAR_PARTS = [f"{SMPL_TAR_PARTS_PREFIX}a{c}" for c in "abcdefg"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download GEAR-SONIC checkpoints from Hugging Face Hub"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save files. "
            "Defaults to gear_sonic_deploy/ (deploy) or repo root (training)."
        ),
    )
    parser.add_argument(
        "--no-planner",
        action="store_true",
        help="Skip downloading the kinematic planner ONNX model",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Download training checkpoint + SMPL motion data (~30 GB)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download sample motion data only (1 walking sequence, ~4 MB)",
    )
    parser.add_argument(
        "--no-smpl",
        action="store_true",
        help="With --training, skip SMPL data download (checkpoint only)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var / run `hf auth login`)",
    )
    return parser.parse_args()


def _ensure_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        return hf_hub_download, snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Install it with:")
        print("  pip install huggingface_hub")
        sys.exit(1)


def download_file(hf_hub_download, repo_id, hf_filename, local_dest, token=None):
    """Download hf_filename from the Hub and place it at local_dest."""
    print(f"  Downloading {hf_filename} ...", flush=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=hf_filename,
        token=token,
    )
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, local_dest)
    print(f"  -> {local_dest}")


def download_and_extract_smpl(hf_hub_download, repo_id, output_dir, token=None):
    """Download split tar parts and extract SMPL data."""
    parts_dir = output_dir / "bones_seed_smpl"
    parts_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {len(SMPL_TAR_PARTS)} parts (~30 GB total) ...", flush=True)
    part_paths = []
    for hf_filename in SMPL_TAR_PARTS:
        local_name = Path(hf_filename).name
        local_dest = parts_dir / local_name
        if local_dest.exists():
            print(f"  (cached) {local_name}")
            part_paths.append(local_dest)
            continue
        cached = hf_hub_download(repo_id=repo_id, filename=hf_filename, token=token)
        shutil.copy2(cached, local_dest)
        part_paths.append(local_dest)
        print(f"  Downloaded {local_name}")

    # Reassemble and extract
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting to {data_dir}/smpl_filtered/ ...", flush=True)

    # cat parts | tar xf - -C data/
    cat_cmd = f"cat {parts_dir}/bones_seed_smpl.tar.part_*"
    tar_cmd = f"tar xf - -C {data_dir}"
    result = subprocess.run(
        f"{cat_cmd} | {tar_cmd}",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: Extraction failed: {result.stderr}")
        sys.exit(1)

    # Count extracted files
    smpl_dir = data_dir / "smpl_filtered"
    if smpl_dir.exists():
        n_files = sum(1 for f in smpl_dir.iterdir() if f.suffix == ".pkl")
        print(f"  -> {smpl_dir} ({n_files} PKL files)")
    else:
        print(f"  WARNING: Expected {smpl_dir} but directory not found")

    # Clean up tar parts
    print("  Cleaning up tar parts ...")
    shutil.rmtree(parts_dir)


def download_sample_data(snapshot_download, repo_id, output_dir, token=None):
    """Download sample motion data (1 walking sequence)."""
    print("  Downloading sample data ...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns="sample_data/*",
        local_dir=str(output_dir),
        token=token,
    )
    sample_dir = output_dir / "sample_data"
    if sample_dir.exists():
        n_files = sum(1 for _ in sample_dir.rglob("*.pkl"))
        print(f"  -> {sample_dir} ({n_files} PKL files)")


def main():
    args = parse_args()
    hf_hub_download, snapshot_download = _ensure_huggingface_hub()

    repo_root = Path(__file__).resolve().parent

    if args.training or args.sample:
        output_dir = args.output_dir if args.output_dir else repo_root
    else:
        output_dir = args.output_dir if args.output_dir else repo_root / "gear_sonic_deploy"

    print("=" * 60)
    print("  GEAR-SONIC — Hugging Face Model Downloader")
    print(f"  Repository : {REPO_ID}")
    print(f"  Output dir : {output_dir}")
    if args.training:
        print(f"  Mode       : training (checkpoint + SMPL data)")
    elif args.sample:
        print(f"  Mode       : sample data (quick start)")
    else:
        print(f"  Mode       : deployment (ONNX models)")
    print("=" * 60)

    if args.sample:
        print("\n[Sample Data]")
        download_sample_data(snapshot_download, REPO_ID, output_dir, token=args.token)

    elif args.training:
        print("\n[Checkpoint]")
        for hf_filename, local_rel in TRAINING_FILES:
            download_file(
                hf_hub_download, REPO_ID, hf_filename,
                output_dir / local_rel, token=args.token,
            )

        if not args.no_smpl:
            print("\n[SMPL Motion Data]")
            download_and_extract_smpl(hf_hub_download, REPO_ID, output_dir, token=args.token)
        else:
            print("\n[SMPL Motion Data] Skipped (--no-smpl)")

    else:
        print("\n[Policy]")
        for hf_filename, local_rel in POLICY_FILES:
            download_file(
                hf_hub_download, REPO_ID, hf_filename,
                output_dir / local_rel, token=args.token,
            )

        if not args.no_planner:
            print("\n[Planner]")
            hf_filename, local_rel = PLANNER_FILE
            download_file(
                hf_hub_download, REPO_ID, hf_filename,
                output_dir / local_rel, token=args.token,
            )

    print("\n" + "=" * 60)
    print("  Done! Files saved under:")
    print(f"  {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
