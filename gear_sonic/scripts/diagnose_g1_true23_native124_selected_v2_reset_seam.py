#!/usr/bin/env python3
"""Run non-qualifying deterministic Stage-1 training-reset-seam diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_native124_selected_v2_reset_seam_diagnostic import (
    TrainingResetSeamDiagnosticRequest,
    failure_report,
    run_training_reset_seam_diagnostic,
    write_training_reset_seam_diagnostic_new,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--warm-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = TrainingResetSeamDiagnosticRequest(
        repository_root=args.repository_root,
        warm_checkpoint=args.warm_checkpoint,
        expected_warm_sha256=args.expected_sha256,
        output=args.output,
    )
    try:
        report = run_training_reset_seam_diagnostic(request)
    except Exception as error:  # Persist exact non-qualifying failure evidence.
        report = failure_report(error, request)
    output = write_training_reset_seam_diagnostic_new(request, report)
    print(output)  # noqa: T201
    return 0 if report.get("completed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
