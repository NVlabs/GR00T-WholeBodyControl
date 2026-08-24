#!/usr/bin/env python3
"""Qualify student-prefix plus exact-teacher cutoff50 controller for one sealed seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_sonic_hybrid_cutoff50_qualification import (
    ALLOWED_RUNTIME_SEEDS,
    preflight,
    run_qualification,
    write_failure,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        item = sub.add_parser(command)
        item.add_argument("--repository-root", type=Path, default=Path.cwd())
        item.add_argument("--runtime-seed", type=int, choices=ALLOWED_RUNTIME_SEEDS, required=True)
        item.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        print(
            json.dumps(
                preflight(
                    repository_root=args.repository_root,
                    output=args.output,
                    runtime_seed=args.runtime_seed,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    try:
        report = run_qualification(
            repository_root=args.repository_root,
            output=args.output,
            runtime_seed=args.runtime_seed,
        )
    except Exception as error:
        failure = write_failure(
            repository_root=args.repository_root,
            output=args.output,
            runtime_seed=args.runtime_seed,
            error=error,
        )
        print(json.dumps({"passed": False, "failure": str(failure), "error_type": type(error).__name__}))
        return 1
    passed = report.get("verdict") == "mixed_controller_recovery_passed"
    print(
        json.dumps(
            {
                "passed": passed,
                "verdict": report.get("verdict"),
                "runtime_seed": args.runtime_seed,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
