#!/usr/bin/env python3
"""Run the frozen model21204 + SONIC-V2 composite in offline MJLab."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import (
    run_composite_mjlab_diagnostic,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    default_motion = (
        repository_root / "artifacts/g1_native124_multimotion/scaling_all61/feasible_v1/npz/B_DadDance.npz"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Run one deterministic 500-transition offline MJLab diagnostic of "
            "the hash-locked iteration-21204 tracker composed with the exact "
            "SONIC V2 safe-target transform. No hardware or network interfaces "
            "are opened, and the report cannot authorize deployment."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--motion",
        type=Path,
        default=default_motion,
        help=("Exact hash-locked B_DadDance input (default), or the admitted 600-frame neutral smoke motion."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON report path; existing files are never overwritten.",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_composite_mjlab_diagnostic(
        repository_root=args.repository_root,
        motion_path=args.motion,
        output_path=args.output,
        seed=args.seed,
        device="cuda:0",
    )
    status = "PASS" if report["computed_pass"] else "QUARANTINE"
    print(  # noqa: T201
        f"model21204 + SONIC-V2 MJLab nominal {status}: {args.output.expanduser().resolve()}"
    )
    if report["error"]:
        print(f"error: {report['error']}")  # noqa: T201
    print(  # noqa: T201
        "teacher_labels_admitted=false support_qualified=false deployment_ready=false hardware_authorized=false"
    )
    return 0 if report["computed_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
