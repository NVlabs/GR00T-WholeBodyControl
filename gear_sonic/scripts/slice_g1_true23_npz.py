"""Extract a contiguous native true-23 motion window for hard-phase replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        frames = int(source["joint_pos"].shape[0])
        if args.start < 0 or args.stop > frames or args.stop - args.start < 500:
            raise ValueError("slice must be in bounds and contain at least 500 frames")
        payload = {
            name: np.asarray(source[name][args.start : args.stop], dtype=np.float32)
            for name in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            )
        }
        fps = np.asarray(source["fps"], dtype=np.float64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.savez_compressed(stream, fps=fps, **payload)
    print(f"[{args.start}:{args.stop}] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
