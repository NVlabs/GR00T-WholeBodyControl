"""Read-only probe for exact PICO XR24 + ankle -> G1 IL29 retargeting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gear_sonic.utils.g1_23dof_contract import REFERENCE_PROFILES
from gear_sonic.utils.g1_23dof_pico_retargeted_producer import (
    probe_exact_retargeter,
    validate_raw_capture,
    validate_retarget_trace_contract,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe pinned SOMA retargeter and validate immutable raw/trace "
            "contracts. Opens no DDS, ZMQ, ADB, XR service, or robot channel."
        )
    )
    parser.add_argument(
        "--soma-source-root",
        type=Path,
        help="Pinned NVIDIA/soma-retargeter checkout to hash and inspect.",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        help="Optional immutable XR24+ankle raw-capture JSON.",
    )
    parser.add_argument(
        "--atomic-snapshot",
        type=Path,
        help=(
            "Optional saved output from hardened xrt.get_body_snapshot(). "
            "Raw MotionTracking fallback/pairing is not accepted."
        ),
    )
    parser.add_argument(
        "--trace",
        type=Path,
        help="Optional blocked G1 IL29 retarget-trace JSON.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(REFERENCE_PROFILES),
        default=next(iter(REFERENCE_PROFILES)),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.capture is None) != (args.trace is None):
        raise ValueError("--capture and --trace must be supplied together")

    atomic_snapshot = (
        _load_json(args.atomic_snapshot)
        if args.atomic_snapshot is not None
        else None
    )
    report = probe_exact_retargeter(
        soma_source_root=args.soma_source_root,
        bodytracking_snapshot=atomic_snapshot,
    )
    if args.capture is not None:
        capture = _load_json(args.capture)
        trace = _load_json(args.trace)
        report["capture"] = validate_raw_capture(capture)
        report["trace"] = validate_retarget_trace_contract(
            capture,
            trace,
            profile=args.profile,
        )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
