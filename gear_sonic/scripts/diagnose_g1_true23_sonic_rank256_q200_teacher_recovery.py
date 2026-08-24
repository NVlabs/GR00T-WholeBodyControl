"""Run exact q200 selected-teacher recovery for rank256 SONIC student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_sonic_rank256_q200_teacher_recovery import preflight, run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        item = sub.add_parser(command)
        item.add_argument("--repository-root", type=Path, default=Path.cwd())
        item.add_argument("--output", type=Path, required=True)
    sub.choices["run"].add_argument("--execute-cuda-rollout", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run" and args.execute_cuda_rollout is not True:
        parser.error("run requires --execute-cuda-rollout")
    if args.command == "preflight":
        body = preflight(repository_root=args.repository_root, output=args.output)
    else:
        body = run(repository_root=args.repository_root, output=args.output)
    print(
        json.dumps(
            {
                "ready": body.get("ready"),
                "verdict": body.get("verdict"),
                "attempted_transitions": body.get("attempted_transitions"),
                "controller_counts": body.get("controller_counts"),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
