"""Export exact MJLab resume weights for simulation/shadow diagnostics only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_PROFILE,
)
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes
from gear_sonic.utils.g1_23dof_mjlab_diagnostic_onnx import (
    export_mjlab_diagnostic_onnx,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export static true23 encoder/decoder ONNX from one weights-only "
            "MJLab resume. Outputs are diagnostic-only and never authorize "
            "deployment, promotion, or active motor control."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument(
        "--expected-reference-profile",
        choices=(CAUSAL_HISTORY_PROFILE,),
        default=None,
        help="Required for causal_model_N.pt checkpoints.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    encoder, decoder, metadata, report = export_mjlab_diagnostic_onnx(
        args.checkpoint,
        args.output_prefix,
        expected_lineage_sha256=args.expected_lineage_sha256,
        expected_reference_profile=args.expected_reference_profile,
    )
    if args.json:
        print(canonical_json_bytes(report).decode("utf-8"), end="")  # noqa: T201
    else:
        print(f"diagnostic encoder: {encoder}")  # noqa: T201
        print(f"diagnostic decoder: {decoder}")  # noqa: T201
        print(f"diagnostic metadata: {metadata}")  # noqa: T201
        print("deployment ready: false")  # noqa: T201
        print("active motor control authorized: false")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
