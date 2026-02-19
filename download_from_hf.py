#!/usr/bin/env python3
"""
Download GEAR-SONIC model checkpoints from Hugging Face Hub.

Repository: https://huggingface.co/nvidia/GEAR-SONIC

Usage:
    python download_from_hf.py
    python download_from_hf.py --output-dir /path/to/output
    python download_from_hf.py --no-planner
"""

import argparse
import shutil
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download GEAR-SONIC checkpoints from Hugging Face Hub"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save files. Defaults to gear_sonic_deploy/ "
            "next to this script."
        ),
    )
    parser.add_argument(
        "--no-planner",
        action="store_true",
        help="Skip downloading the kinematic planner ONNX model",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var / run huggingface-cli login)",
    )
    return parser.parse_args()


def _ensure_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download
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


def main():
    args = parse_args()
    hf_hub_download = _ensure_huggingface_hub()

    repo_root = Path(__file__).resolve().parent
    output_dir = args.output_dir if args.output_dir else repo_root / "gear_sonic_deploy"

    print("=" * 56)
    print("  GEAR-SONIC — Hugging Face Model Downloader")
    print(f"  Repository : {REPO_ID}")
    print(f"  Output dir : {output_dir}")
    print("=" * 56)

    print("\n[Policy]")
    for hf_filename, local_rel in POLICY_FILES:
        download_file(hf_hub_download, REPO_ID, hf_filename, output_dir / local_rel, token=args.token)

    if not args.no_planner:
        print("\n[Planner]")
        hf_filename, local_rel = PLANNER_FILE
        download_file(hf_hub_download, REPO_ID, hf_filename, output_dir / local_rel, token=args.token)

    print("\n" + "=" * 56)
    print("  Done! Files saved under:")
    print(f"  {output_dir}")
    print("=" * 56)


if __name__ == "__main__":
    main()
