"""Aggregate exact deterministic full-profile MJLab diagnostic seed shards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_23dof_mjlab_diagnostic_mujoco import (
    aggregate_mjlab_diagnostic_shards,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and aggregate three incomplete offline diagnostic seed shards."
    )
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = aggregate_mjlab_diagnostic_shards(
        shard_paths=args.shard,
        output_path=args.output,
    )
    print(f"true23 MJLab full diagnostic aggregate PASS: {args.output.resolve()}")  # noqa: T201
    print("deployment_ready=false active_motor_control_authorized=false")  # noqa: T201
    return 0 if report["computed_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
