"""Validate the opening neutral-standing hold in a saved PICO XR24 capture.

This command is deliberately offline and read-only.  It opens no XR, Unitree,
DDS, ZMQ, or policy channel and writes no files.  A passing report only admits
the capture to the pinned SOMA replay; it never authorizes robot actuation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import statistics
from typing import Any

from gear_sonic.utils.g1_23dof_pico_retargeted_producer import (
    validate_raw_capture,
)
from gear_sonic.utils.g1_23dof_xr24_soma_adapter import (
    XR24_NEUTRAL_STANDING_HOLD_FRAMES,
    assess_xr24_neutral_standing,
    resample_raw_capture_50hz,
)

_AUTHORIZATION = {
    "read_only": True,
    "dds_opened": False,
    "robot_channel_opened": False,
    "actuation_authorized": False,
    "robot_commands_published": False,
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def assess_standing_capture(
    capture: Mapping[str, Any],
    *,
    hold_frames: int = XR24_NEUTRAL_STANDING_HOLD_FRAMES,
) -> dict[str, Any]:
    """Assess consecutive opening 50 Hz frames with the shared adapter gate."""

    if (
        isinstance(hold_frames, bool)
        or not isinstance(hold_frames, int)
        or hold_frames < 1
    ):
        raise ValueError("hold_frames must be a positive integer")
    capture_summary = validate_raw_capture(capture)
    samples = resample_raw_capture_50hz(capture)
    if len(samples) < hold_frames:
        raise ValueError(
            "XR24 capture is too short for neutral-standing hold: "
            f"need {hold_frames} resampled frames, got {len(samples)}"
        )

    frame_reports = [
        assess_xr24_neutral_standing(samples[index]["body_poses"])
        for index in range(hold_frames)
    ]
    check_names = tuple(frame_reports[0]["checks"])
    metric_names = tuple(frame_reports[0]["metrics"])
    checks = {
        name: all(report["checks"][name] for report in frame_reports)
        for name in check_names
    }
    metric_ranges = {}
    for name in metric_names:
        values = [float(report["metrics"][name]) for report in frame_reports]
        metric_ranges[name] = {
            "min": min(values),
            "mean": statistics.fmean(values),
            "max": max(values),
        }
    failed_frames = [
        {
            "hold_index": index,
            "source_frame_index": int(samples[index]["source_frame_index"]),
            "failed_checks": [
                name
                for name, passed in report["checks"].items()
                if not passed
            ],
        }
        for index, report in enumerate(frame_reports)
        if not report["pass"]
    ]
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "kind": "g1_true23_xr24_standing_capture_validation",
        "status": (
            "neutral_standing_acquisition_pass"
            if passed
            else "neutral_standing_acquisition_reject"
        ),
        "pass": passed,
        "capture_session_id": capture["session_id"],
        "capture_sha256": capture_summary["sha256"],
        "capture_frame_count": capture_summary["frame_count"],
        "resampled_frame_count": len(samples),
        "hold_frame_count": hold_frames,
        "hold_reference_monotonic_ns": [
            int(samples[0]["reference_monotonic_ns"]),
            int(samples[hold_frames - 1]["reference_monotonic_ns"]),
        ],
        "checks": checks,
        "metric_ranges": metric_ranges,
        "failed_frames": failed_frames,
        "next_step": (
            "admit_capture_to_pinned_soma_replay"
            if passed
            else "stand_upright_recalibrate_and_recapture"
        ),
        "authorization": dict(_AUTHORIZATION),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only validation of the opening neutral-standing hold in a "
            "saved hardened PICO XR24 capture."
        )
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=XR24_NEUTRAL_STANDING_HOLD_FRAMES,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = assess_standing_capture(
        _load_json(args.capture.resolve()),
        hold_frames=args.hold_frames,
    )
    print(_canonical_json(report))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
