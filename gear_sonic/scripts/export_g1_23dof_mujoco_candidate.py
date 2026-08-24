#!/usr/bin/env python3
"""Export non-deployable true23 ONNX bytes for MuJoCo Sim2Sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from gear_sonic.utils.g1_23dof_mujoco_promotion import (
    export_true23_mujoco_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export immutable candidate ONNX from a genuinely trained true23 "
            "*.promotion.pt checkpoint. Candidate output is never deployment "
            "authorization; MuJoCo evidence and a promotion sidecar are required."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("encoder_output", type=Path)
    parser.add_argument("decoder_output", type=Path)
    parser.add_argument("metadata_output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder, decoder, metadata, result = export_true23_mujoco_candidate(
        args.checkpoint,
        args.encoder_output,
        args.decoder_output,
        args.metadata_output,
    )
    print(f"non-deployable candidate encoder: {encoder}")  # noqa: T201
    print(f"non-deployable candidate decoder: {decoder}")  # noqa: T201
    print(f"candidate metadata: {metadata}")  # noqa: T201
    print(  # noqa: T201
        "candidate encoder SHA-256: "
        f"{result['hashes']['encoder_onnx_sha256']}"
    )
    print(  # noqa: T201
        "candidate decoder SHA-256: "
        f"{result['hashes']['decoder_onnx_sha256']}"
    )
    print("deployment authorized: false")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
