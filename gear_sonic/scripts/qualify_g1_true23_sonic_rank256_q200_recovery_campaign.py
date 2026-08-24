#!/usr/bin/env python3
"""Run exact rank256 q200 10+10 recovery qualification campaign."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gear_sonic.utils import g1_true23_sonic_rank256_q200_recovery_qualification_campaign as q200


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _progress(record: Mapping[str, Any]) -> None:
    recovery_time = record.get("recovery_time_s")
    recovery_text = "not_performed" if recovery_time is None else f"{float(recovery_time):.3f}s"
    print(  # noqa: T201
        " ".join(
            (
                f"scenario={record['scenario']}",
                f"index={int(record['rollout_index']):02d}",
                f"seed={int(record['seed'])}",
                f"pass={str(record.get('passed') is True).lower()}",
                f"steps={int(record.get('attempted_transitions') or 0)}",
                f"recovery={recovery_text}",
            )
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = q200.run(
            repository_root=args.repository_root,
            output=args.output,
            progress=_progress,
        )
    except Exception as error:
        report = q200.failure_report(
            error,
            repository_root=args.repository_root,
            output=args.output,
        )
    output = q200.write(
        report,
        repository_root=args.repository_root,
        output=args.output,
    )
    print(f"report={output}", flush=True)  # noqa: T201
    return 0 if report.get("campaign_qualified") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
