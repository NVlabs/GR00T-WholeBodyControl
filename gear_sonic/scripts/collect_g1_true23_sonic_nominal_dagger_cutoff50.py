"""Collect one exact cutoff50 nominal intervention-BC episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_sonic_nominal_dagger_cutoff50_collection import (
    CollectionRequest,
    collect,
    preflight,
    publish,
    write_failure,
)

DEFAULT_OUTPUT = Path("artifacts/g1_true23/sonic_nominal_dagger_cutoff50_seed20260805_v1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "collect"):
        item = sub.add_parser(command)
        item.add_argument("--repository-root", type=Path, default=Path.cwd())
        item.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = CollectionRequest(args.repository_root, args.output_prefix)
    if args.command == "preflight":
        print(json.dumps(preflight(request), indent=2, sort_keys=True))
        return 0
    try:
        arrays, materials = collect(request)
        npz, manifest, body = publish(request, arrays, materials)
    except Exception as error:
        failure = write_failure(request, error)
        print(json.dumps({"passed": False, "failure": str(failure), "error_type": type(error).__name__}))
        return 1
    print(
        json.dumps(
            {
                "passed": True,
                "npz": str(npz),
                "manifest": str(manifest),
                "npz_sha256": body["artifact"]["npz_sha256"],
                "row_count": body["rows"]["total"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
