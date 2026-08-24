"""Run one scale-2 SONIC to selected-teacher recovery diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path

from gear_sonic.scripts import train_g1_true23_sonic_task_space_ppo_full_support as publication
from gear_sonic.utils.g1_true23_sonic_scale2_teacher_recovery import (
    MODES,
    Scale2RecoveryRequest,
    preflight,
    run,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    pre.add_argument("--mode", choices=MODES, default="q250")
    pre.add_argument("--runtime-seed", type=int)
    pre.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/g1_true23/g1_true23_sonic_scale2_teacher_recovery_q250_v1.json"),
    )
    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    diagnose.add_argument("--mode", choices=MODES, required=True)
    diagnose.add_argument("--runtime-seed", type=int)
    diagnose.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = Scale2RecoveryRequest(args.repository_root, args.output, args.mode, args.runtime_seed)
    if args.command == "preflight":
        report = preflight(request)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    try:
        report = run(request)
    except Exception as error:
        report = {
            "schema_version": 1,
            "kind": "g1_true23_sonic_scale2_teacher_recovery_failure_v1",
            "mode": args.mode,
            "error": {"type": type(error).__name__, "message": str(error)},
            "teacher_labels_admitted": 0,
            "training_arrays_present": False,
            "training_performed": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
    output = request.output_path
    if os.path.lexists(output):
        raise FileExistsError(f"scale2 recovery output exists: {output}")
    publication._write_json_exclusive(output, report)  # noqa: SLF001
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if report.get("qualification", {}).get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
