"""Collect nominal multi-seed selected-teacher behavior-cloning dataset."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path

from gear_sonic.utils.g1_true23_selected_teacher_nominal_multiseed_collection import (
    CollectionRequest,
    collect,
    load_training_candidate,
    preflight,
    publish_new,
    write_failure_new,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PREFIX = Path("artifacts/g1_true23/selected_teacher_nominal_multiseed_train8_heldout2_v1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "collect", "validate"):
        command = sub.add_parser(name)
        command.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
        command.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser


def _progress(record: Mapping[str, object]) -> None:
    print(  # noqa: T201
        json.dumps(
            {
                "run_index": record.get("run_index"),
                "seed": record.get("seed"),
                "passed": record.get("passed"),
                "heldout": record.get("heldout"),
                "published_training_rows": record.get("published_training_rows"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = CollectionRequest(args.repository_root, args.output_prefix)
    if args.command == "preflight":
        report = preflight(request)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    if args.command == "validate":
        arrays, manifest = load_training_candidate(request)
        report = {
            "valid": True,
            "row_count": int(arrays["decoder994"].shape[0]),
            "npz_sha256": manifest["artifact"]["npz_sha256"],
            "teacher_label_rows": manifest["rows"]["teacher_label_rows"],
            "support_qualified": manifest["boundaries"]["support_qualified"],
            "hardware_authorized": manifest["boundaries"]["hardware_authorized"],
        }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0
    for path in (request.npz_path, request.manifest_path, request.failure_path):
        if os.path.lexists(path):
            raise FileExistsError(f"nominal collection output exists: {path}")
    try:
        arrays, materials = collect(request, progress=_progress)
        npz, manifest, body = publish_new(request, arrays, materials)
        loaded, _ = load_training_candidate(request)
        report = {
            "published": True,
            "npz_path": str(npz),
            "manifest_path": str(manifest),
            "npz_sha256": body["artifact"]["npz_sha256"],
            "training_rows": int(loaded["decoder994"].shape[0]),
            "teacher_labels_admitted": body["rows"]["teacher_label_rows"],
            "support_qualified": False,
            "training_performed": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        if not os.path.lexists(request.npz_path) and not os.path.lexists(request.manifest_path):
            write_failure_new(request, error)
        raise
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
