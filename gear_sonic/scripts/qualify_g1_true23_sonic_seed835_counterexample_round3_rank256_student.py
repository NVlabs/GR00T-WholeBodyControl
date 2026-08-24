"""Qualify rank256 counterexample SONIC candidate for one exact 510-step episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_sonic_seed835_counterexample_round3_rank256_student_qualification import (
    ALLOWED_RUNTIME_SEEDS,
    run_qualification,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-seed", type=int, choices=ALLOWED_RUNTIME_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_qualification(
        repository_root=args.repository_root,
        output=args.output,
        runtime_seed=args.runtime_seed,
    )
    print(
        json.dumps(
            {
                "qualified": report.get("qualified_requested_mode"),
                "verdict": report.get("verdict"),
                "runtime_seed": args.runtime_seed,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.get("qualified_requested_mode") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
