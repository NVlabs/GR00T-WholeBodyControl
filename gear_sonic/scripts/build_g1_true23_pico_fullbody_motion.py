"""Build one full-body native-23 MJLab motion from saved PICO causal packets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_pico_fullbody_motion import (
    DEFAULT_MINIMUM_FRAMES,
    load_and_build_pico_fullbody_motion,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--minimum-frames", type=int, default=DEFAULT_MINIMUM_FRAMES)
    parser.add_argument(
        "--reachable-raw-abs",
        type=float,
        help="Project only unreachable packet joints into native23 safe-action image.",
    )
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    packets = args.packets if args.packets.is_absolute() else root / args.packets
    output_npz = args.output_npz if args.output_npz.is_absolute() else root / args.output_npz
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    if os.path.lexists(output_npz) or os.path.lexists(output_json):
        raise FileExistsError("PICO full-body motion output exists")

    arrays, report = load_and_build_pico_fullbody_motion(
        repository_root=root,
        packet_path=packets,
        minimum_frames=args.minimum_frames,
        reachable_raw_abs=args.reachable_raw_abs,
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with output_npz.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report["output_npz"] = str(output_npz.resolve())
    report["output_sha256"] = sha256_file(output_npz.resolve(strict=True))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
