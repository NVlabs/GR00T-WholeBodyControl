"""Fit the nominal multi-seed final-affine SONIC student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge import (
    FitRequest,
    preflight,
    publish,
    run_fit,
)

DEFAULT_OUTPUT = Path("artifacts/g1_true23/sonic_nominal_multiseed_bc_last_affine_ridge_v1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "fit"):
        item = sub.add_parser(command)
        item.add_argument("--repository-root", type=Path, default=Path.cwd())
        item.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = FitRequest(args.repository_root, args.output_prefix)
    if args.command == "preflight":
        print(json.dumps(preflight(request), indent=2, sort_keys=True))
        return 0
    outcome = run_fit(request)
    decoder, manifest, body = publish(request, outcome)
    print(
        json.dumps(
            {
                "passed": outcome.passed,
                "decoder": None if decoder is None else str(decoder),
                "manifest": str(manifest),
                "candidate_decoder_sha256": body["export"]["candidate_decoder_sha256"],
                "gate_issues": body["gate_issues"],
                "selected_lambda": body["fit"]["selected_lambda"],
                "oof_ratio_to_base": body["fit"]["leave_one_run_out"]["ratio_to_base"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
