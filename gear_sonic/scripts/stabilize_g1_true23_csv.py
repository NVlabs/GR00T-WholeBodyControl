"""Smoothly project risky G1 leg phases toward a stable true-23 stance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HOME_LEGS = np.asarray(
    (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0),
    dtype=np.float64,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--taper", type=int, default=200)
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--stabilize-root", action="store_true")
    args = parser.parse_args()
    motion = np.loadtxt(args.input, delimiter=",", dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != 30:
        raise ValueError("expected a 30-column true-23 CSV")
    if not (0 <= args.start < args.stop <= len(motion)) or args.taper <= 0:
        raise ValueError("invalid stabilization interval or taper")
    if not np.isfinite(args.strength) or not 0.0 <= args.strength <= 1.0:
        raise ValueError("strength must be within [0, 1]")

    mask = np.zeros(len(motion), dtype=np.float64)
    mask[args.start : args.stop] = 1.0
    left_start = max(0, args.start - args.taper)
    left_count = args.start - left_start
    if left_count:
        phase = np.linspace(0.0, 1.0, left_count, endpoint=False)
        mask[left_start : args.start] = 0.5 - 0.5 * np.cos(np.pi * phase)
    right_stop = min(len(motion), args.stop + args.taper)
    right_count = right_stop - args.stop
    if right_count:
        phase = np.linspace(0.0, 1.0, right_count, endpoint=False)
        mask[args.stop : right_stop] = 0.5 + 0.5 * np.cos(np.pi * phase)
    blend = (args.strength * mask)[:, None]
    motion[:, 7:19] = motion[:, 7:19] * (1.0 - blend) + HOME_LEGS * blend
    if args.stabilize_root:
        root_rpy = Rotation.from_quat(motion[:, 3:7]).as_euler("xyz")
        root_rpy[:, :2] *= 1.0 - blend
        motion[:, 3:7] = Rotation.from_euler("xyz", root_rpy).as_quat()
        motion[:, 2] = motion[:, 2] * (1.0 - blend[:, 0]) + 0.8 * blend[:, 0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, motion, delimiter=",", fmt="%.9f")
    root_note = "+root" if args.stabilize_root else ""
    print(
        f"stabilized legs{root_note} [{args.start}:{args.stop}], "
        f"strength={args.strength:.3f}: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
