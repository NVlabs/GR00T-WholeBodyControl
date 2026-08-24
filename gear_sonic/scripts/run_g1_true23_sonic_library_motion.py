"""Run one retargeted SONIC library motion through genuine true23 physics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_sonic_library_replay import run_library_motion_replay


def _exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--decoder", type=Path)
    parser.add_argument("--expected-decoder-sha256")
    parser.add_argument("--controller-mode", choices=("sonic", "reference_pd"), default="sonic")
    parser.add_argument(
        "--gain-profile",
        choices=("true23_native", "released_retained"),
        default="true23_native",
    )
    parser.add_argument("--initial-state-motion", type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    motion = args.motion if args.motion.is_absolute() else root / args.motion
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_npz = args.output_npz if args.output_npz.is_absolute() else root / args.output_npz
    if os.path.lexists(output_json) or os.path.lexists(output_npz):
        raise FileExistsError("output path already exists")
    decoder = args.decoder
    if decoder is not None and not decoder.is_absolute():
        decoder = root / decoder
    initial_state_motion = args.initial_state_motion
    if initial_state_motion is not None and not initial_state_motion.is_absolute():
        initial_state_motion = root / initial_state_motion
    report, arrays = run_library_motion_replay(
        repository_root=root,
        motion_path=motion,
        maximum_steps=args.maximum_steps,
        decoder_path=decoder,
        expected_decoder_sha256=args.expected_decoder_sha256,
        controller_mode=args.controller_mode,
        initial_state_motion_path=initial_state_motion,
        gain_profile=args.gain_profile,
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with output_npz.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report["trajectory_npz"] = str(output_npz)
    _exclusive_json(output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
