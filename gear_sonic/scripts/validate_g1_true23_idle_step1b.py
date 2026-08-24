"""Validate a supplied idle-only Step 1B campaign without launching simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_step1b_qualification import (
    DEFAULT_CONTRACT_PATH,
    qualify_step1b_report_fail_closed,
    write_json_atomic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Step1B evidence report JSON")
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="versioned Step1B contract JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional new qualification JSON; existing files are never overwritten",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = qualify_step1b_report_fail_closed(args.report, contract_path=args.contract)
    if args.output is not None:
        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if not result["evidence_valid"]:
        return 2
    return 0 if result["qualification_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
