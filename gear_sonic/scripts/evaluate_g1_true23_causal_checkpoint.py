#!/usr/bin/env python3
"""Export and evaluate one exact causal-history true23 MJLab checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_PROFILE,
    causal_history_profile_contract,
)
from gear_sonic.utils.g1_23dof_mjlab_diagnostic_mujoco import (
    run_mjlab_diagnostic_mujoco,
)
from gear_sonic.utils.g1_23dof_mjlab_diagnostic_onnx import (
    export_mjlab_diagnostic_onnx,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a causal_model_N.pt checkpoint with CPU ONNX parity, then "
            "run strict nominal and disturbance MuJoCo diagnostics. This path "
            "accepts only true23_causal_step1_history_0p02s_v1 semantics and "
            "never creates robot-control authorization."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="New extensionless prefix for encoder/decoder/metadata outputs.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="New *.json path containing 'diagnostic' in its filename.",
    )
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mjcf", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    encoder, decoder, metadata, export_report = export_mjlab_diagnostic_onnx(
        args.checkpoint,
        args.output_prefix,
        expected_lineage_sha256=args.expected_lineage_sha256,
        expected_reference_profile=CAUSAL_HISTORY_PROFILE,
    )
    report = run_mjlab_diagnostic_mujoco(
        checkpoint_path=args.checkpoint,
        encoder_path=encoder,
        decoder_path=decoder,
        metadata_path=metadata,
        output_path=args.report,
        profile=args.profile,
        config_path=args.config,
        mjcf_path=args.mjcf,
        expected_reference_profile=CAUSAL_HISTORY_PROFILE,
    )
    semantic = causal_history_profile_contract()
    status = "PASS" if report["computed_pass"] else "FAIL"
    print(  # noqa: T201
        f"causal true23 ONNX parity PASS: {encoder} {decoder}"
    )
    print(  # noqa: T201
        f"causal profile: {export_report['source']['reference_profile']} "
        f"contract_sha256={semantic['contract_sha256']}"
    )
    print(  # noqa: T201
        f"MuJoCo {args.profile} {status}: {args.report.resolve()}"
    )
    print(  # noqa: T201
        "deployment_ready=false active_motor_control_authorized=false"
    )
    if report["error"]:
        print(f"error: {report['error']}")  # noqa: T201
    return 0 if report["computed_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
