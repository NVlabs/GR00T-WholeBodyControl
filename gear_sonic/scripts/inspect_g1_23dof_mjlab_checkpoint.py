#!/usr/bin/env python3
"""Safely audit MJLab/RSL-RL checkpoint compatibility with SONIC true23."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_23dof_contract import DEFAULT_REFERENCE_PROFILE
from gear_sonic.utils.g1_23dof_mjlab_bridge import (
    audit_mjlab_rsl_rl_checkpoint,
    write_mjlab_audit_report,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tensor-safely inspect an MJLab/RSL-RL checkpoint. This command "
            "never converts or promotes stock RSL-RL weights."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--reference-profile",
        default=DEFAULT_REFERENCE_PROFILE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new diagnostic JSON path; overwrite is refused.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_mjlab_rsl_rl_checkpoint(
        args.checkpoint,
        reference_profile=args.reference_profile,
    )
    if args.output is not None:
        output = write_mjlab_audit_report(report, args.output)
        print(f"MJLab checkpoint audit: {output}")  # noqa: T201
    print(  # noqa: T201
        "architecture compatible: "
        f"{str(report['compatibility']['architecture_compatible']).lower()}"
    )
    print("promotion eligible: false")  # noqa: T201
    print("promotion checkpoint written: false")  # noqa: T201
    print(  # noqa: T201
        "truthful role: "
        f"{report['truthful_bridge_boundary']['stock_checkpoint_role']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
