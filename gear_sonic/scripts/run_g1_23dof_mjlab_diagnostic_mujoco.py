"""Run strict diagnostic-only true23 MJLab ONNX validation in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_PROFILE,
)
from gear_sonic.utils.g1_23dof_mjlab_diagnostic_mujoco import (
    diagnostic_bundle_paths,
    run_mjlab_diagnostic_mujoco,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline true23 diagnostic ONNX at 50 Hz in MuJoCo. "
            "Never opens DDS, network, PICO, or robot command interfaces and "
            "never authorizes active motor control."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-prefix",
        type=Path,
        required=True,
        help="Extensionless prefix for .encoder.onnx/.decoder.onnx/.diagnostic.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help=(
            "Run only this deterministic profile seed (repeatable). "
            "A subset is always emitted as an incomplete diagnostic shard."
        ),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mjcf", type=Path)
    parser.add_argument(
        "--expected-reference-profile",
        choices=(CAUSAL_HISTORY_PROFILE,),
        default=None,
        help="Required for causal_model_N.pt checkpoints.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    encoder, decoder, metadata = diagnostic_bundle_paths(
        args.diagnostic_prefix
    )
    report = run_mjlab_diagnostic_mujoco(
        checkpoint_path=args.checkpoint,
        encoder_path=encoder,
        decoder_path=decoder,
        metadata_path=metadata,
        output_path=args.output,
        profile=args.profile,
        config_path=args.config,
        mjcf_path=args.mjcf,
        expected_reference_profile=args.expected_reference_profile,
        seed_subset=args.seeds,
    )
    status = "PASS" if report["computed_pass"] else "FAIL"
    print(  # noqa: T201
        f"true23 MJLab diagnostic MuJoCo {args.profile} {status}: "
        f"{args.output.resolve()}"
    )
    if report["error"]:
        print(f"error: {report['error']}")  # noqa: T201
    print(  # noqa: T201
        "deployment_ready=false active_motor_control_authorized=false"
    )
    return 0 if report["computed_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
