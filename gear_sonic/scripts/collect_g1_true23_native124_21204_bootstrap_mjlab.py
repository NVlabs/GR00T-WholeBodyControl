"""CLI for the fixed 510-row selected-21204 teacher bootstrap collector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gear_sonic.utils.g1_true23_native124_21204_bootstrap_mjlab import (
    BootstrapCollectionRequest,
    preflight_bootstrap_collection,
    publish_bootstrap_evidence_new,
    run_teacher_bootstrap_collection,
    write_bootstrap_failure_manifest_new,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hash-locked simulator-only teacher bootstrap collection. "
            "No support, DAgger, promotion, deployment, or hardware claim."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "collect"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--repository-root",
            type=Path,
            default=REPOSITORY_ROOT,
        )
        command.add_argument(
            "--output-prefix",
            type=Path,
            required=True,
            help=("suffix-free path under artifacts/g1_true23; creates .npz and .manifest.json with no overwrite"),
        )
    collect = subparsers.choices["collect"]
    collect.add_argument(
        "--execute-cuda-rollout",
        action="store_true",
        help="required acknowledgement for the 510-step CUDA/MJLab simulator run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    request = BootstrapCollectionRequest(
        repository_root=args.repository_root,
        output_prefix=args.output_prefix,
    )
    if args.command == "collect" and args.execute_cuda_rollout is not True:
        parser.error("collect requires --execute-cuda-rollout")
    preflight = None
    try:
        preflight = preflight_bootstrap_collection(request)
        if args.command == "preflight":
            print(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False))
            return 0
        arrays, materials = run_teacher_bootstrap_collection(
            request,
            preflight=preflight,
        )
        npz, manifest, body = publish_bootstrap_evidence_new(
            arrays,
            npz_path=request.npz_path,
            manifest_path=request.manifest_path,
            materials=materials,
            repository_root=request.root,
        )
        print(
            json.dumps(
                {
                    "npz": str(npz),
                    "manifest": str(manifest),
                    "qualification": body["qualification"],
                    "boundaries": body["boundaries"],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0 if body["qualification"]["whole_run_quarantined"] is False else 2
    except BaseException as error:
        # KeyboardInterrupt/SystemExit before a runtime attempt should retain
        # normal CLI behavior.  All ordinary preflight/runtime failures create
        # one quarantine-only manifest and never a partial NPZ.
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if args.command == "preflight":
            parser.exit(
                1,
                f"bootstrap preflight failed without writing outputs: {type(error).__name__}: {error}\n",
            )
        try:
            failure = write_bootstrap_failure_manifest_new(
                request,
                error,
                preflight=preflight,
            )
        except BaseException as publication_error:
            parser.exit(
                1,
                f"bootstrap failed: {type(error).__name__}: {error}\n"
                f"failure manifest not published: {type(publication_error).__name__}: "
                f"{publication_error}\n",
            )
        parser.exit(
            1,
            f"bootstrap failed and was quarantined in {failure}: {type(error).__name__}: {error}\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
