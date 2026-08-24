#!/usr/bin/env python3
"""Create or verify a causal true23 MuJoCo promotion JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_causal_promotion import (
    causal_promotion_body,
    verify_causal_promotion,
    write_new_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-onnx", type=Path, required=True)
    parser.add_argument("--decoder-onnx", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = {
        "checkpoint_path": args.checkpoint,
        "encoder_path": args.encoder_onnx,
        "decoder_path": args.decoder_onnx,
        "metadata_path": args.metadata,
        "full_report_path": args.full_report,
    }
    if args.verify_only:
        verify_causal_promotion(args.output, **inputs)
        print(f"causal promotion verified: {args.output.resolve()}")  # noqa: T201
        return 0
    output = write_new_json(args.output, causal_promotion_body(**inputs))
    print(f"causal promotion created: {output}")  # noqa: T201
    print("active motor control authorized: false")  # noqa: T201
    print("gantry sidecar still required: true")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
