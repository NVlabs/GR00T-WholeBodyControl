#!/usr/bin/env python3
"""Qualify one hash-bound SONIC decoder in a pinned closed-loop window."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_sonic_student_closed_loop_qualification import (
    MODES,
    StudentQualificationRequest,
    failure_report,
    run_student_qualification,
    write_student_qualification_new,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-candidate-manifest-sha256",
        required=True,
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = StudentQualificationRequest(
        repository_root=args.repository_root,
        candidate_manifest=args.candidate_manifest,
        expected_candidate_manifest_sha256=(args.expected_candidate_manifest_sha256),
        output=args.output,
        mode=args.mode,
    )
    try:
        report = run_student_qualification(request)
    except Exception as error:  # Persist fail-closed pre-rollout errors.
        report = failure_report(error, request)
    output = write_student_qualification_new(request, report)
    print(output)  # noqa: T201
    return 0 if report.get("qualified_requested_mode") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
