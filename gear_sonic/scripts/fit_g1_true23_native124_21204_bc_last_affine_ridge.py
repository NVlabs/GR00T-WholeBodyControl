"""CLI for deterministic CPU-only SONIC true23 final-affine BC fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_native124_21204_bc_last_affine_ridge import (
    OfflineBCRequest,
    failure_outcome,
    preflight_offline_bc,
    publish_offline_bc_outcome_new,
    run_offline_bc_fit,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hash-locked CPU-only offline BC fit. Produces a simulator candidate only; "
            "no support, DAgger, deployment, hardware, or network authorization."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "fit"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--repository-root",
            type=Path,
            default=REPOSITORY_ROOT,
        )
    subparsers.choices["fit"].add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help=(
            "suffix-free direct child of artifacts/g1_true23; creates an exclusive manifest "
            "and creates .decoder.onnx only when all offline gates pass"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "preflight":
        try:
            report = preflight_offline_bc(args.repository_root)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            parser.exit(
                1,
                f"offline BC preflight failed without outputs: {type(error).__name__}: {error}\n",
            )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0

    request = OfflineBCRequest(
        repository_root=args.repository_root,
        output_prefix=args.output_prefix,
    )
    preflight = None
    try:
        preflight = preflight_offline_bc(args.repository_root)
        outcome = run_offline_bc_fit(args.repository_root)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        outcome = failure_outcome(error, preflight)
    try:
        decoder, manifest, body = publish_offline_bc_outcome_new(request, outcome)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        parser.exit(
            1,
            f"offline BC result could not be published: {type(error).__name__}: {error}\n",
        )
    print(
        json.dumps(
            {
                "decoder": None if decoder is None else str(decoder),
                "manifest": str(manifest),
                "classification": body["classification"],
                "gate_issues": body["gate_issues"],
                "eligible_for_closed_loop_simulator_experiment": body[
                    "eligible_for_closed_loop_simulator_experiment"
                ],
                "boundaries": body["boundaries"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if outcome.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
