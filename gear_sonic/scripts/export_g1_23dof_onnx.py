#!/usr/bin/env python3
"""Export a trained true23 checkpoint only after explicit simulation evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from gear_sonic.utils.g1_23dof_artifact import export_validated_true23_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a static float32 teleop encoder [1,267] -> [1,64] and "
            "true23 decoder [1,994] -> [1,23] only after validating "
            "checkpoint-bound nominal/disturbance simulation evidence."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Trained weights-only *.promotion.pt checkpoint; full trainer resume files are rejected.",
    )
    parser.add_argument("simulation_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Defaults to <output-prefix>.metadata.json.",
    )
    args = parser.parse_args()

    encoder_path, decoder_path, metadata_path, metadata = export_validated_true23_artifact(
        args.checkpoint,
        args.simulation_report,
        args.output,
        metadata_path=args.metadata_output,
    )
    print(f"validated teleop encoder ONNX: {encoder_path}")  # noqa: T201
    print(f"validated true23 decoder ONNX: {decoder_path}")  # noqa: T201
    print(f"metadata sidecar: {metadata_path}")  # noqa: T201
    print(  # noqa: T201
        f"encoder SHA-256: {metadata['hashes']['encoder_onnx_sha256']}"
    )
    print(  # noqa: T201
        f"decoder SHA-256: {metadata['hashes']['decoder_onnx_sha256']}"
    )


if __name__ == "__main__":
    main()
