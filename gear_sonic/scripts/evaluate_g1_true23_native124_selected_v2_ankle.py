#!/usr/bin/env python3
"""Evaluate one hash-bound Stage-1 ankle warm restart on DadDance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_native124_selected_v2_ankle_evaluation import (
    WarmEvaluationRequest,
    failure_report,
    run_warm_daddance_evaluation,
    write_warm_evaluation_report_new,
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
    request = WarmEvaluationRequest(
        repository_root=args.repository_root,
        warm_checkpoint=args.warm_checkpoint,
        expected_warm_sha256=args.expected_sha256,
        output=args.output,
    )
    try:
        report = run_warm_daddance_evaluation(request)
    except Exception as error:  # Persist fail-closed evidence, then fail CLI.
        report = failure_report(error, request)
    output = write_warm_evaluation_report_new(request, report)
    print(output)  # noqa: T201
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
