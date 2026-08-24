"""Write exact current-state authentic PICO true23 teleop readiness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gear_sonic.utils.g1_true23_pico_teleop_readiness import audit_pico_teleop_readiness


def _exclusive_json(path: Path, report: dict) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--health-report", type=Path, required=True)
    parser.add_argument("--live-consumer-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_pico_teleop_readiness(
        repository_root=args.repository_root,
        health_report_path=args.health_report,
        live_consumer_report_path=args.live_consumer_report,
    )
    _exclusive_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["simulator_qualification_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
