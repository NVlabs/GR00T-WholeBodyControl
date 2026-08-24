"""Project a true23 PICO motion into SONIC V2 reachability and rebuild FK."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_pico_library_projection import project_library_motion_to_safe_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reachable-raw-abs", type=float, default=9.0)
    parser.add_argument("--minimum-frames", type=int)
    parser.add_argument(
        "--root-mode",
        choices=("preserve", "canonical_upright_ankle"),
        default="preserve",
    )
    args = parser.parse_args()
    root = args.repository_root.resolve()
    source = args.input if args.input.is_absolute() else root / args.input
    output_npz = args.output_npz if args.output_npz.is_absolute() else root / args.output_npz
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    if os.path.lexists(output_npz) or os.path.lexists(output_json):
        raise FileExistsError("output path already exists")
    with np.load(source.resolve(strict=True), allow_pickle=False) as archive:
        motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    arrays, report = project_library_motion_to_safe_image(
        repository_root=root,
        motion=motion,
        reachable_raw_abs=float(args.reachable_raw_abs),
        root_mode=args.root_mode,
        minimum_frames=args.minimum_frames,
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with output_npz.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report.update(
        {
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source.resolve()),
            "output_npz": str(output_npz.resolve()),
            "output_npz_sha256": sha256_file(output_npz.resolve()),
        }
    )
    with output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
