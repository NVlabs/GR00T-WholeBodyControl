#!/usr/bin/env python3
"""Collect exact selected-teacher bootstrap rows for one sealed heldout seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from gear_sonic.utils import g1_true23_native124_21204_bootstrap_mjlab as base
from gear_sonic.utils.g1_true23_native124_21204_multiseed_bootstrap import (
    ALLOWED_SEEDS,
    collect_and_publish,
    preflight,
    write_failure,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "collect"):
        item = sub.add_parser(command)
        item.add_argument("--repository-root", type=Path, default=Path.cwd())
        item.add_argument("--runtime-seed", type=int, choices=ALLOWED_SEEDS, required=True)
        item.add_argument("--output-prefix", type=Path, required=True)
    sub.choices["collect"].add_argument("--execute-cuda-rollout", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "collect" and args.execute_cuda_rollout is not True:
        parser.error("collect requires --execute-cuda-rollout")
    request = base.BootstrapCollectionRequest(args.repository_root, args.output_prefix)
    receipt: Mapping[str, Any] | None = None
    try:
        receipt = preflight(request, seed=args.runtime_seed)
        if args.command == "preflight":
            print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
            return 0
        npz, manifest, body = collect_and_publish(request, seed=args.runtime_seed)
        print(
            json.dumps(
                {
                    "npz": str(npz),
                    "manifest": str(manifest),
                    "runtime_seed": args.runtime_seed,
                    "qualification": body["qualification"],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0 if body["qualification"]["whole_run_quarantined"] is False else 2
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if args.command == "preflight":
            parser.exit(1, f"multiseed bootstrap preflight failed: {type(error).__name__}: {error}\n")
        failure = write_failure(
            request,
            error,
            seed=args.runtime_seed,
            preflight_receipt=receipt,
        )
        parser.exit(1, f"multiseed bootstrap failed; quarantine: {failure}\n")


if __name__ == "__main__":
    raise SystemExit(main())
