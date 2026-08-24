"""Build the exact diagnostic-only Step1A idle corpus for stock MJLab loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_supported_idle_corpus import (
    build_supported_idle_corpus,
    default_sidecar_path,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STEP1B_INPUT_ROOT = (
    REPOSITORY_ROOT / "artifacts" / "g1_true23_step1b_idle_nominal_20260806_v1" / "inputs" / "schema6"
)
DEFAULT_CHANGE_MOTION = STEP1B_INPUT_ROOT / "idle__220713__change_idle_left_a021" / "motion.npz"
DEFAULT_HANDS_MOTION = STEP1B_INPUT_ROOT / "idle__220721__hands_on_back_loop_a036m" / "motion.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--change-motion",
        type=Path,
        default=DEFAULT_CHANGE_MOTION,
        help="Pinned 362-frame change-idle schema-v6 motion NPZ.",
    )
    parser.add_argument(
        "--hands-motion",
        type=Path,
        default=DEFAULT_HANDS_MOTION,
        help="Pinned 547-frame hands-on-back schema-v6 motion NPZ.",
    )
    parser.add_argument("--output", type=Path, required=True, help="New corpus NPZ path.")
    parser.add_argument(
        "--sidecar",
        type=Path,
        help="New sidecar path; defaults to <output-stem>.spans.json beside output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sidecar = args.sidecar or default_sidecar_path(args.output)
    report = build_supported_idle_corpus(
        change_motion=args.change_motion,
        hands_motion=args.hands_motion,
        corpus_path=args.output,
        sidecar_path=sidecar,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
