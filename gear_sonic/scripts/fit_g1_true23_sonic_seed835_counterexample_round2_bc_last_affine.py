"""Fit second final-affine SONIC student on seed835 counterexample rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge import FitRequest, publish
from gear_sonic.utils.g1_true23_sonic_seed835_counterexample_round2_bc_last_affine import preflight, run_fit

DEFAULT_OUTPUT = Path("artifacts/g1_true23/sonic_seed835_counterexample_round2_bc_last_affine_v1")


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
                "round2_base_rmse": body["metrics"]["round2_counterexample"]["base"]["rmse"],
                "round2_candidate_rmse": body["metrics"]["round2_counterexample"]["candidate"]["rmse"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
