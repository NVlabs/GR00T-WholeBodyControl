#!/usr/bin/env python3
"""Run one hash-bound SONIC student-to-teacher recovery diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_sonic_student_teacher_recovery import (
    CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH,
    CURRENT_CANDIDATE_MANIFEST_SHA256,
    MODES,
    RecoveryRequest,
    failure_report,
    run_recovery_diagnostic,
    write_recovery_report_new,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH,
    )
    parser.add_argument(
        "--expected-candidate-manifest-sha256",
        default=CURRENT_CANDIDATE_MANIFEST_SHA256,
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = RecoveryRequest(
        repository_root=args.repository_root,
        candidate_manifest=args.candidate_manifest,
        expected_candidate_manifest_sha256=args.expected_candidate_manifest_sha256,
        output=args.output,
        mode=args.mode,
    )
    try:
        report = run_recovery_diagnostic(request)
    except Exception as error:  # Persist fail-closed pre-rollout/runtime errors.
        report = failure_report(error, request)
    output = write_recovery_report_new(request, report)
    print(output)  # noqa: T201
    return 0 if report.get("recovered_to_original_q9_518_boundary") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
