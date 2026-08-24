#!/usr/bin/env python3
"""Run pinned continuous or phase selected-source DadDance qualification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_native124_selected_source_full_clip_qualification import (
    PHASE_WINDOW_IDS,
    FullClipQualificationRequest,
    failure_report,
    run_full_clip_qualification,
    write_full_clip_qualification_new,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--continuous", action="store_true")
    mode.add_argument("--phase-window", choices=PHASE_WINDOW_IDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = FullClipQualificationRequest(
        repository_root=args.repository_root,
        output=args.output,
        mode="continuous" if args.continuous else "phase",
        phase_window_id=args.phase_window,
    )
    try:
        report = run_full_clip_qualification(request)
    except Exception as error:  # Persist exact fail-closed evidence.
        report = failure_report(error, request)
    output = write_full_clip_qualification_new(request, report)
    print(output)  # noqa: T201
    return 0 if report.get("qualified_requested_mode") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
