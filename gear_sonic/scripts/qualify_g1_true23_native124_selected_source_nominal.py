#!/usr/bin/env python3
"""Qualify exact selected 21204 source on fixed-start DadDance nominal slice."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_native124_selected_source_nominal_qualification import (
    SelectedSourceNominalQualificationRequest,
    failure_report,
    run_selected_source_nominal_qualification,
    write_selected_source_nominal_qualification_new,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = SelectedSourceNominalQualificationRequest(
        repository_root=args.repository_root,
        output=args.output,
    )
    try:
        report = run_selected_source_nominal_qualification(request)
    except Exception as error:  # Persist exact fail-closed qualification evidence.
        report = failure_report(error, request)
    output = write_selected_source_nominal_qualification_new(request, report)
    print(output)  # noqa: T201
    return 0 if report.get("qualified_nominal_slice") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
