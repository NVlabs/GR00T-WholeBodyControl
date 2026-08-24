#!/usr/bin/env python3
"""Collect seed835 student failure-prefix counterexample DAgger rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils import g1_true23_sonic_nominal_dagger_cutoff50_collection as base
from gear_sonic.utils.g1_true23_sonic_seed835_failure_prefix_dagger import collect, preflight, publish


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "collect"):
        item = sub.add_parser(command)
        item.add_argument("--repository-root", type=Path, default=Path.cwd())
        item.add_argument("--output-prefix", type=Path, required=True)
    sub.choices["collect"].add_argument("--execute-cuda-rollout", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "collect" and args.execute_cuda_rollout is not True:
        parser.error("collect requires --execute-cuda-rollout")
    request = base.CollectionRequest(args.repository_root, args.output_prefix)
    if args.command == "preflight":
        print(json.dumps(preflight(request), indent=2, sort_keys=True))
        return 0
    arrays, materials = collect(request)
    npz, manifest, body = publish(request, arrays, materials)
    print(
        json.dumps(
            {
                "npz": str(npz),
                "manifest": str(manifest),
                "rows": body["rows"],
                "classification": body["boundaries"]["classification"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
