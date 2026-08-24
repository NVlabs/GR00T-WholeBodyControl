"""Emit one read-only true23 semantic lower-body reference window as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from gear_sonic.utils.g1_23dof_semantic_reference import (
    PROFILE_RELEASED_LOW_LATENCY_STEP1,
    PROFILE_TRUE23_STEP5,
    build_recorded_reference_window,
    validate_semantic_reference_window,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build positions10x12+velocities10x12 from a complete recorded "
            "G1 reference. Opens no DDS, ZMQ, or robot channel."
        )
    )
    parser.add_argument("motion_dir", type=Path)
    parser.add_argument(
        "--profile",
        choices=(
            PROFILE_TRUE23_STEP5,
            PROFILE_RELEASED_LOW_LATENCY_STEP1,
        ),
        default=PROFILE_TRUE23_STEP5,
    )
    parser.add_argument("--playback-frame", type=int, required=True)
    parser.add_argument(
        "--playback-epoch-monotonic-ns",
        type=int,
        required=True,
        help="Monotonic timestamp assigned to reference frame zero.",
    )
    parser.add_argument(
        "--emitted-monotonic-ns",
        type=int,
        help="Defaults to current time.monotonic_ns().",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    emitted_ns = (
        time.monotonic_ns()
        if args.emitted_monotonic_ns is None
        else args.emitted_monotonic_ns
    )
    window = build_recorded_reference_window(
        args.motion_dir,
        profile=args.profile,
        playback_frame_index=args.playback_frame,
        playback_epoch_monotonic_ns=args.playback_epoch_monotonic_ns,
        emitted_monotonic_ns=emitted_ns,
    )
    validate_semantic_reference_window(window, motion_dir=args.motion_dir)
    print(  # noqa: T201
        json.dumps(
            window,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
