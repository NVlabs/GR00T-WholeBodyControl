"""Inspect one safe exact-policy true23 MJLab training resume checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes
from gear_sonic.utils.g1_23dof_mjlab_training import (
    checkpoint_training_summary,
    load_mjlab_training_checkpoint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a weights-only-safe true23 MJLab training resume. "
            "This command never creates promotion or deployment artifacts."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-lineage-sha256")
    parser.add_argument("--minimum-update-count", type=int)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print canonical JSON summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoint = load_mjlab_training_checkpoint(
        args.checkpoint,
        expected_lineage_sha256=args.expected_lineage_sha256,
        minimum_update_count=args.minimum_update_count,
    )
    summary = checkpoint_training_summary(checkpoint)
    if args.json:
        print(canonical_json_bytes(summary).decode("utf-8"), end="")  # noqa: T201
    else:
        print(f"checkpoint role: {summary['checkpoint_role']}")  # noqa: T201
        print(f"update count: {summary['update_count']}")  # noqa: T201
        print(  # noqa: T201
            "simulation candidate review allowed: "
            f"{str(summary['training_gate']['simulation_candidate_review_allowed']).lower()}"
        )
        print("deployment ready: false")  # noqa: T201
        print("promotion eligible: false")  # noqa: T201
        print(  # noqa: T201
            "lineage: "
            + json.dumps(
                {
                    "lineage_sha256": summary["lineage_sha256"],
                    "policy_state_sha256": summary["policy_state_sha256"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
