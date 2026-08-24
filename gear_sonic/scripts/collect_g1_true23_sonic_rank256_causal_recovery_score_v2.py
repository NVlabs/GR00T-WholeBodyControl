"""Retry causal recovery-score collection with observed push-support gate."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from typing import Any

from gear_sonic.scripts import collect_g1_true23_sonic_rank256_causal_recovery_score as v1
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_causal_recovery_score_v2.json"
)
CONTRACT_SHA256 = "7673737a9b8995499b0be8baa530dfd58ddcab9610174e357c0d48f14b8c431b"
MIN_VALID_BY_GROUP = {"nominal": 100, "exact_impulse": 30}
RESULT_FILENAME = "rank256_causal_recovery_score_collection_result_v2.json"
FAILURE_FILENAME = "rank256_causal_recovery_score_collection_failure_v2.json"
DIRECTION_FILENAME = "rank256_causal_recovery_score_direction_v2.pt"


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("causal recovery-score v2 contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    parents = body.get("parents", {})
    objective = body.get("objective", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_causal_recovery_score_contract_v2"
        or objective.get("minimum_valid_transitions") != MIN_VALID_BY_GROUP
        or objective.get("exact_impulse_threshold_derived_from_v1_observed_post_push_horizon") is not True
        or body.get("gradient", {}).get("scales") != [0.0]
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("causal recovery-score v2 contract semantic mismatch")
    checks = (
        (root / parents["v1_contract_relative_path"], parents["v1_contract_sha256"]),
        (Path(parents["v1_failed_run_path"]) / "preflight.json", parents["v1_failed_preflight_sha256"]),
        (
            Path(parents["v1_failed_run_path"]) / "material_manifest.json",
            parents["v1_failed_material_manifest_sha256"],
        ),
        (Path(parents["causal_trace_path"]), parents["causal_trace_sha256"]),
        (Path(parents["iterative_screen_path"]), parents["iterative_screen_sha256"]),
    )
    for raw, expected in checks:
        resolved = raw.resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file() or sha256_file(resolved) != expected:
            raise ValueError("causal recovery-score v2 parent mismatch")
    return body


@contextmanager
def _scope(root: Path) -> Iterator[None]:
    contract = _load_contract(root)
    with v1._scope(root):  # noqa: SLF001
        parent = v1.grouped.parent
        parent_names = (
            "CONTRACT_RELATIVE_PATH",
            "CONTRACT_SHA256",
            "KIND",
            "RESULT_FILENAME",
            "FAILURE_FILENAME",
            "DIRECTION_FILENAME",
            "EXTRA_SOURCE_RELATIVE_PATHS",
            "_load_contract",
        )
        saved_parent = {name: getattr(parent, name) for name in parent_names}
        saved_thresholds = v1.MIN_VALID_BY_GROUP
        merged = copy.deepcopy(saved_parent["_load_contract"](root))
        for section in ("collection", "gradient", "evaluation", "boundaries"):
            merged[section] = {**dict(merged[section]), **dict(contract[section])}
        merged["kind"] = contract["kind"]
        merged["role"] = contract["role"]
        merged["parents"] = copy.deepcopy(contract["parents"])
        merged["objective"] = copy.deepcopy(contract["objective"])

        def load_current(_root: Path) -> Mapping[str, Any]:
            return copy.deepcopy(merged)

        try:
            v1.MIN_VALID_BY_GROUP = dict(MIN_VALID_BY_GROUP)
            parent.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
            parent.CONTRACT_SHA256 = CONTRACT_SHA256
            parent.KIND = "g1_true23_sonic_rank256_causal_recovery_score_collection_result_v2"
            parent.RESULT_FILENAME = RESULT_FILENAME
            parent.FAILURE_FILENAME = FAILURE_FILENAME
            parent.DIRECTION_FILENAME = DIRECTION_FILENAME
            parent.EXTRA_SOURCE_RELATIVE_PATHS = (
                *saved_parent["EXTRA_SOURCE_RELATIVE_PATHS"],
                Path("gear_sonic/scripts/collect_g1_true23_sonic_rank256_causal_recovery_score_v2.py"),
            )
            parent._load_contract = load_current
            yield
        finally:
            v1.MIN_VALID_BY_GROUP = saved_thresholds
            for name, value in saved_parent.items():
                setattr(parent, name, value)


def preflight(repository_root: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        report = dict(v1.grouped.parent.preflight(root))
    report["causal_recovery_objective_v2"] = {
        "minimum_valid_transitions": dict(MIN_VALID_BY_GROUP),
        "v1_failed_before_gradient": True,
        "direction_only": True,
    }
    return report


def run(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        return v1.grouped.parent.run(root, run_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=ROOT)
    collect = sub.add_parser("collect")
    collect.add_argument("--repository-root", type=Path, default=ROOT)
    collect.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight(args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    try:
        result = run(args.repository_root, args.run_dir)
    except Exception as error:
        print(f"causal recovery score v2 failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result["assessment"], indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
