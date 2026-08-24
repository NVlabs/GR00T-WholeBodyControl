#!/usr/bin/env python3
"""Run the frozen no-update causal-wrapper DadDance parity gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_native124_selected_v2_causal_parity import (
    DEFAULT_OUTPUT_RELATIVE_PATH,
    DEVICE,
    SEED,
    failure_report,
    run_no_update_causal_parity,
    write_report_new,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_RELATIVE_PATH,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    try:
        report = run_no_update_causal_parity(
            repository_root=root,
            seed=args.seed,
            device=args.device,
        )
    except Exception as error:  # Persist fail-closed evidence, then fail CLI.
        report = failure_report(error)
    output = write_report_new(
        args.output,
        report,
        repository_root=root,
    )
    print(output)
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
