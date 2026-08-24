"""Time-stretch a 30-column G1 true-23 motion CSV without dropping poses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def retime(
    source: Path,
    destination: Path,
    *,
    slowdown: float | None,
    max_joint_speed_norm: float | None,
    input_fps: float,
) -> tuple[int, int, str]:
    if (slowdown is None) == (max_joint_speed_norm is None):
        raise ValueError("choose exactly one of slowdown or max-joint-speed-norm")
    if slowdown is not None and (not np.isfinite(slowdown) or slowdown < 1.0):
        raise ValueError("slowdown must be finite and at least 1.0")
    if max_joint_speed_norm is not None and (
        not np.isfinite(max_joint_speed_norm) or max_joint_speed_norm <= 0.0
    ):
        raise ValueError("max-joint-speed-norm must be finite and positive")
    if not np.isfinite(input_fps) or input_fps <= 0.0:
        raise ValueError("input-fps must be finite and positive")

    motion = np.loadtxt(source, delimiter=",", dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != 30 or motion.shape[0] < 2:
        raise ValueError("expected at least two rows of 30-column true-23 motion")
    source_dt = 1.0 / input_fps
    if slowdown is not None:
        old_t = np.arange(motion.shape[0], dtype=np.float64) * source_dt * slowdown
        mode = f"{slowdown:.3f}x slower"
    else:
        assert max_joint_speed_norm is not None
        joint_distance = np.linalg.norm(np.diff(motion[:, 7:30], axis=0), axis=1)
        intervals = np.maximum(source_dt, joint_distance / max_joint_speed_norm)
        old_t = np.concatenate(([0.0], np.cumsum(intervals)))
        mode = f"adaptive <= {max_joint_speed_norm:.3f} rad/s joint norm"
    new_count = int(np.ceil(old_t[-1] * input_fps)) + 1
    new_t = np.linspace(0.0, old_t[-1], new_count)

    result = np.empty((new_count, 30), dtype=np.float64)
    for column in (*range(3), *range(7, 30)):
        result[:, column] = np.interp(new_t, old_t, motion[:, column])
    quaternions = motion[:, 3:7].copy()  # scipy and source CSV both use xyzw.
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    result[:, 3:7] = Slerp(old_t, Rotation.from_quat(quaternions))(new_t).as_quat()

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, result, delimiter=",", fmt="%.9f")
    return motion.shape[0], new_count, mode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slowdown", type=float)
    parser.add_argument("--max-joint-speed-norm", type=float)
    parser.add_argument("--input-fps", type=float, default=60.0)
    args = parser.parse_args()
    sources = sorted(args.input.glob("*.csv")) if args.input.is_dir() else [args.input]
    if not sources:
        raise FileNotFoundError("no input CSV files found")
    for source in sources:
        destination = args.output / source.name if args.input.is_dir() else args.output
        old_count, new_count, mode = retime(
            source,
            destination,
            slowdown=args.slowdown,
            max_joint_speed_norm=args.max_joint_speed_norm,
            input_fps=args.input_fps,
        )
        print(f"{source.name}: {old_count} -> {new_count} frames ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
