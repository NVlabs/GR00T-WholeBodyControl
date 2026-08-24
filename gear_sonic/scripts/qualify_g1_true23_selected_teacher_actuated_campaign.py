"""Run selected-teacher full-episode simulator qualification campaign."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path

from gear_sonic.utils.g1_true23_selected_teacher_actuated_campaign import (
    CampaignRequest,
    preflight,
    run_campaign,
    write_json_exclusive,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("artifacts/g1_true23/g1_true23_selected_teacher_actuated_campaign_v1.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "qualify"):
        command = sub.add_parser(name)
        command.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _progress(record: Mapping[str, object]) -> None:
    print(  # noqa: T201
        json.dumps(
            {
                "run_id": record.get("run_id"),
                "passed": record.get("passed"),
                "completed_transitions": record.get("completed_transitions"),
                "recovered": record.get("recovered"),
                "first_issue": record.get("first_issue"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _failure_report(error: Exception) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "g1_true23_selected_teacher_actuated_campaign_failure_v1",
        "campaign_qualified": False,
        "error": {
            "type": type(error).__name__,
            "detail_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
        },
        "published_teacher_label_count": 0,
        "published_training_row_count": 0,
        "training_performed": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
    }


def _write_failure_exclusive(path: Path, report: Mapping[str, object]) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = CampaignRequest(args.repository_root, args.output)
    if args.command == "preflight":
        report = preflight(request)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    if os.path.lexists(request.output_path):
        raise FileExistsError(f"teacher campaign output exists: {request.output_path}")
    try:
        report = run_campaign(request, progress=_progress)
    except Exception as error:
        report = _failure_report(error)
        _write_failure_exclusive(request.output_path, report)
    else:
        write_json_exclusive(request.output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if report.get("campaign_qualified") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
