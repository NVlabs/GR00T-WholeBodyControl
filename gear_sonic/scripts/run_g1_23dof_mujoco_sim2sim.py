"""Run offline, native-23 MuJoCo Sim2Sim validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_23dof_mujoco_sim2sim import (
    run_sim2sim_validation,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact G1 rev-1.0 true23 policy at 50 Hz in MuJoCo. "
            "No DDS, network, PICO, or robot command interfaces are opened."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mjcf", type=Path)
    parser.add_argument("--encoder-onnx", type=Path)
    parser.add_argument("--decoder-onnx", type=Path)
    parser.add_argument("--metadata", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_sim2sim_validation(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config,
        mjcf_path=args.mjcf,
        encoder_onnx_path=args.encoder_onnx,
        decoder_onnx_path=args.decoder_onnx,
        metadata_path=args.metadata,
    )
    status = "PASS" if report["computed_pass"] else "FAIL"
    eligibility = (
        "promotion-eligible"
        if report["promotion_eligible"]
        else "diagnostic-only"
    )
    print(  # noqa: T201
        f"true23 MuJoCo Sim2Sim {status} ({eligibility}): "
        f"{args.output.resolve()}"
    )
    return 0 if report["computed_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
