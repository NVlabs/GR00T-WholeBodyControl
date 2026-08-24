#!/usr/bin/env python3
"""Promote or re-verify an exact true23 MuJoCo candidate package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_23dof_mujoco_promotion import (
    promote_true23_mujoco_candidate,
    verify_true23_mujoco_promotion,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the exact ONNX pair in the approved MuJoCo model and bind "
            "passing raw traces into a self-hashed promotion sidecar. This "
            "authorizes immutable deployment bytes, never active motor control."
        )
    )
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-onnx", type=Path, required=True)
    parser.add_argument("--decoder-onnx", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Require an existing sidecar and fully recompute candidate, raw "
            "trace, ONNX, and MuJoCo evidence without writing files."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    kwargs = {
        "checkpoint_path": args.checkpoint,
        "encoder_path": args.encoder_onnx,
        "decoder_path": args.decoder_onnx,
        "metadata_path": args.metadata,
        "report_path": args.report,
    }
    if args.verify_only:
        result = verify_true23_mujoco_promotion(args.sidecar, **kwargs)
        action = "verified"
    else:
        result = promote_true23_mujoco_candidate(args.sidecar, **kwargs)
        action = "created"
    print(f"MuJoCo promotion sidecar {action}: {args.sidecar.resolve()}")  # noqa: T201
    print(  # noqa: T201
        "promotion payload SHA-256: "
        f"{result['promotion_payload_sha256']}"
    )
    print("deployment bytes authorized: true")  # noqa: T201
    print("active motor control authorized: false")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
