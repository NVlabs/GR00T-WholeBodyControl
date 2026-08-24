"""Adapt exact teacher bootstrap collection to sealed heldout observation seeds."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from gear_sonic.utils import g1_true23_native124_21204_bootstrap_mjlab as base

CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_native124_21204_multiseed_bootstrap_v1.json"
)
CONTRACT_SHA256 = "031b4445b72247d069fa60d39a09da253550a14314b43efa4d38b0603e7ea57c"
CONTRACT_KIND = "g1_true23_native124_21204_multiseed_teacher_bootstrap_contract_v1"
ALLOWED_SEEDS = (611723381, 835868017, 921108064)
SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_native124_21204_multiseed_bootstrap.py"),
    Path("gear_sonic/scripts/collect_g1_true23_native124_21204_multiseed_bootstrap.py"),
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if base.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("multiseed bootstrap contract hash mismatch")
    body = _mapping(json.loads(path.read_text(encoding="utf-8")), "contract")
    runtime = _mapping(body.get("runtime"), "runtime")
    boundaries = _mapping(body.get("boundaries"), "boundaries")
    if (
        body.get("schema_version") != 1
        or body.get("kind") != CONTRACT_KIND
        or runtime.get("allowed_seeds") != list(ALLOWED_SEEDS)
        or runtime.get("rows_per_seed") != base.TOTAL_ROWS
        or runtime.get("teacher_controls_every_transition") is not True
        or runtime.get("student_controls_zero_transitions") is not True
        or runtime.get("whole_run_quarantine") is not True
        or boundaries.get("behavior_cloning_input_candidate") is not True
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("multiseed bootstrap contract semantic drift")
    inputs = _mapping(body.get("inputs"), "inputs")
    for name, raw_entry in inputs.items():
        entry = _mapping(raw_entry, name)
        candidate = (root / str(entry["path"])).resolve(strict=True)
        if candidate.is_symlink() or not candidate.is_file() or base.sha256_file(candidate) != entry["sha256"]:
            raise ValueError(f"multiseed bootstrap input mismatch: {name}")
    teacher_report = _mapping(
        json.loads((root / str(inputs["seed835_teacher_nominal_pass"]["path"])).read_text(encoding="utf-8")),
        "seed835 teacher report",
    )
    hybrid_failure = _mapping(
        json.loads((root / str(inputs["seed835_cutoff50_failure"]["path"])).read_text(encoding="utf-8")),
        "seed835 hybrid failure",
    )
    if (
        teacher_report.get("qualified_nominal_slice") is not True
        or teacher_report.get("heldout_seed_diagnostic", {}).get("runtime_seed") != 835868017
        or hybrid_failure.get("verdict") != "mixed_controller_recovery_failed"
        or hybrid_failure.get("controller_counts") != {"student": 50, "teacher": 46}
    ):
        raise ValueError("multiseed bootstrap decision evidence drift")
    return body


@contextmanager
def collection_scope(root: Path, seed: int) -> Iterator[None]:
    load_contract(root)
    if seed not in ALLOWED_SEEDS:
        raise ValueError("multiseed bootstrap seed is not sealed")
    previous_seed = base.FIXED_SEED
    previous_sources = base.EXECUTED_SOURCE_FILE_RELATIVE_PATHS
    try:
        base.FIXED_SEED = seed
        base.EXECUTED_SOURCE_FILE_RELATIVE_PATHS = (*previous_sources, *SOURCE_PATHS)
        yield
    finally:
        base.FIXED_SEED = previous_seed
        base.EXECUTED_SOURCE_FILE_RELATIVE_PATHS = previous_sources


def preflight(request: base.BootstrapCollectionRequest, *, seed: int) -> Mapping[str, Any]:
    with collection_scope(request.root, seed):
        receipt = dict(base.preflight_bootstrap_collection(request))
    receipt["multiseed_contract_sha256"] = CONTRACT_SHA256
    receipt["runtime_seed"] = seed
    receipt["simulator_constructed"] = False
    receipt["hardware_authorized"] = False
    return receipt


def collect_and_publish(
    request: base.BootstrapCollectionRequest, *, seed: int
) -> tuple[Path, Path, Mapping[str, Any]]:
    with collection_scope(request.root, seed):
        receipt = dict(base.preflight_bootstrap_collection(request))
        arrays, raw_materials = base.run_teacher_bootstrap_collection(request, preflight=receipt)
        materials = dict(raw_materials)
        materials["multiseed_adapter"] = {
            "contract": {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": CONTRACT_SHA256},
            "runtime_seed": seed,
            "teacher_actuated_rows": base.TOTAL_ROWS,
            "student_actuated_rows": 0,
            "hardware_authorized": False,
        }
        return base.publish_bootstrap_evidence_new(
            arrays,
            npz_path=request.npz_path,
            manifest_path=request.manifest_path,
            materials=materials,
            repository_root=request.root,
        )


def write_failure(
    request: base.BootstrapCollectionRequest,
    error: BaseException,
    *,
    seed: int,
    preflight_receipt: Mapping[str, Any] | None,
) -> Path:
    with collection_scope(request.root, seed):
        return base.write_bootstrap_failure_manifest_new(request, error, preflight=preflight_receipt)


__all__ = ["ALLOWED_SEEDS", "collect_and_publish", "collection_scope", "load_contract", "preflight"]
