"""Run clean-observation reference-versus-failed-seed scale-2 probe."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path

from gear_sonic.utils.g1_true23_sonic_scale2_clean_cross_seed_probe import (
    ProbeRequest,
    preflight,
    run,
    write_json_exclusive,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "probe"):
        command = sub.add_parser(name)
        command.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
        command.add_argument(
            "--output",
            type=Path,
            default=Path("artifacts/g1_true23/g1_true23_sonic_scale2_clean_cross_seed_probe_v1.json"),
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = ProbeRequest(args.repository_root, args.output)
    if args.command == "preflight":
        report = preflight(request)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    if os.path.lexists(request.output_path):
        raise FileExistsError(f"cross-seed probe output exists: {request.output_path}")
    report = run(request)
    write_json_exclusive(request.output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if report.get("seed_sensitivity_proven") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
