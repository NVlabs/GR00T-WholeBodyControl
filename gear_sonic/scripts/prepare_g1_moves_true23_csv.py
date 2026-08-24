"""Convert headerless G1 Moves 29-DoF CSV files to the native true-23 layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

RETAINED_29_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26)


def convert(source: Path, destination: Path) -> None:
    values = np.loadtxt(source, delimiter=",", dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 36 or values.shape[0] < 2:
        raise ValueError(f"{source} must contain headerless [frames,36] G1 Moves data")
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains NaN or Inf")
    quaternion_norm = np.linalg.norm(values[:, 3:7], axis=1)
    if np.any(np.abs(quaternion_norm - 1.0) > 1.0e-3):
        raise ValueError(f"{source} contains non-unit root quaternions")
    output = np.concatenate((values[:, :7], values[:, 7 + np.asarray(RETAINED_29_INDICES)]), axis=1)
    if output.shape != (values.shape[0], 30):
        raise RuntimeError("true23 CSV conversion contract drifted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, output, delimiter=",", fmt="%.9f")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(args.input.glob("*.csv")) if args.input.is_dir() else [args.input]
    if not sources:
        raise FileNotFoundError("no input CSV files found")
    for source in sources:
        destination = args.output / source.name if args.input.is_dir() else args.output
        convert(source, destination)
        print(f"{source.name} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
