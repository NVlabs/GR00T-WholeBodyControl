"""Export a native Unitree G1 124-to-23 RSL actor without constructing MJLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_23dof_native124_actor_export import (
    export_native124_actor,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely reconstruct the exact native 124-to-23 RSL-RL actor, "
            "export a static ONNX, and write a hash-bound lineage report. "
            "Existing outputs are never overwritten."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--reference-onnx",
        type=Path,
        help="Optional immutable actor used for pre-export and post-export parity.",
    )
    parser.add_argument(
        "--expected-reference-sha256",
        help="Required SHA-256 when --reference-onnx is supplied.",
    )
    parser.add_argument("--parity-seed", type=int, default=11500)
    parser.add_argument("--json", action="store_true", help="Print the complete report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = export_native124_actor(
        args.checkpoint,
        args.output,
        args.report,
        reference_onnx_path=args.reference_onnx,
        expected_reference_sha256=args.expected_reference_sha256,
        parity_seed=args.parity_seed,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"exported actor: {report['export']['output_path']}")
        print(f"actor SHA256: {report['export']['output_sha256']}")
        print(f"lineage report: {args.report.resolve()}")
        print("MJLab environment constructed: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
