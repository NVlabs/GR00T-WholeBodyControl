"""Qualify the known working student-prefix plus exact-teacher cutoff140 controller."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_cutoff140_collection as cutoff140,
    g1_true23_sonic_nominal_dagger_student_qualification as adapter,
    g1_true23_sonic_student_teacher_recovery as recovery,
)

CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_hybrid_cutoff140_qualification_v1.json"
)
CONTRACT_SHA256 = "c5572049848ef88c45e05f9a5a3d3750aeda4f6bdfb024257cd423bccdc1b608"
CONTRACT_KIND = "g1_true23_sonic_hybrid_cutoff140_qualification_contract_v1"
ALLOWED_RUNTIME_SEEDS = (20260805, 611723381, 835868017, 921108064)
SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_sonic_hybrid_cutoff140_qualification.py"),
    Path("gear_sonic/scripts/qualify_g1_true23_sonic_hybrid_cutoff140.py"),
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if recovery.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("hybrid qualification contract hash mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    runtime = _mapping(body.get("runtime"), "runtime")
    boundaries = _mapping(body.get("boundaries"), "boundaries")
    if (
        body.get("schema_version") != 1
        or body.get("kind") != CONTRACT_KIND
        or runtime.get("mode") != "cutoff140"
        or runtime.get("total_transitions") != 510
        or runtime.get("student_transitions") != 140
        or runtime.get("teacher_transitions") != 370
        or runtime.get("allowed_seeds") != list(ALLOWED_RUNTIME_SEEDS)
        or boundaries.get("mixed_controller_teleop_candidate") is not True
        or boundaries.get("pure_student_qualified") is not False
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("hybrid qualification contract semantic drift")
    inputs = _mapping(body.get("inputs"), "inputs")
    for name, entry in inputs.items():
        value = _mapping(entry, name)
        candidate = (root / str(value["path"])).resolve(strict=True)
        if candidate.is_symlink() or not candidate.is_file() or recovery.sha256_file(candidate) != value["sha256"]:
            raise ValueError(f"hybrid qualification input mismatch: {name}")
    prior = json.loads((root / str(inputs["prior_passing_manifest"]["path"])).read_text(encoding="utf-8"))
    prior_report = _mapping(
        _mapping(prior.get("materials"), "prior materials").get("recovery_report"), "prior report"
    )
    if (
        prior.get("kind") != cutoff140.MANIFEST_KIND
        or prior_report.get("verdict") != "mixed_controller_recovery_passed"
        or prior_report.get("recovered_to_original_q9_518_boundary") is not True
        or prior_report.get("attempted_transitions") != 510
        or prior_report.get("controller_counts") != {"student": 140, "teacher": 370}
    ):
        raise ValueError("prior cutoff140 pass evidence drift")
    return body


@contextmanager
def _source_scope() -> Iterator[None]:
    before = recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS
    try:
        recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS = (*before, *SOURCE_PATHS)
        yield
    finally:
        recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS = before


@contextmanager
def _qualification_scope(root: Path, runtime_seed: int) -> Iterator[None]:
    if runtime_seed not in ALLOWED_RUNTIME_SEEDS:
        raise ValueError("hybrid runtime seed is not sealed")
    with (
        adapter._adapter_scope(root, runtime_seed),  # noqa: SLF001
        cutoff140._current_recovery_scope(root),  # noqa: SLF001
        _source_scope(),
    ):
        yield


def _request(root: Path, output: Path) -> recovery.RecoveryRequest:
    return recovery.RecoveryRequest(
        repository_root=root,
        candidate_manifest=root / adapter.CANDIDATE_MANIFEST_RELATIVE_PATH,
        expected_candidate_manifest_sha256=adapter.CANDIDATE_MANIFEST_SHA256,
        output=output,
        mode="cutoff140",
    )


def preflight(*, repository_root: Path, output: Path, runtime_seed: int) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    load_contract(root)
    request = _request(root, output)
    if os.path.lexists(request.output_path):
        raise FileExistsError("hybrid qualification output exists")
    with _qualification_scope(root, runtime_seed):
        receipt = dict(recovery.preflight_recovery(request))
    receipt["hybrid_contract_sha256"] = CONTRACT_SHA256
    receipt["runtime_seed"] = runtime_seed
    receipt["simulator_constructed"] = False
    receipt["hardware_authorized"] = False
    return receipt


def run_qualification(*, repository_root: Path, output: Path, runtime_seed: int) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    contract = load_contract(root)
    request = _request(root, output)
    if os.path.lexists(request.output_path):
        raise FileExistsError("hybrid qualification output exists")
    with _qualification_scope(root, runtime_seed):
        report = dict(recovery.run_recovery_diagnostic(request))
        report["hybrid_qualification"] = {
            "contract": {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": CONTRACT_SHA256},
            "runtime_seed": runtime_seed,
            "student_candidate_manifest_sha256": adapter.CANDIDATE_MANIFEST_SHA256,
            "student_candidate_decoder_sha256": adapter.CANDIDATE_DECODER_SHA256,
            "controller": "student_q9_9_through_148_then_exact_selected_teacher_q9_149_through_518",
            "mixed_controller_teleop_candidate": True,
            "pure_student_qualified": False,
            "teacher_labels_admitted": False,
            "hardware_authorized": False,
        }
        report["hybrid_boundaries"] = dict(contract["boundaries"])
        recovery.write_recovery_report_new(request, report)
    return report


def write_failure(*, repository_root: Path, output: Path, runtime_seed: int, error: BaseException) -> Path:
    root = repository_root.expanduser().resolve(strict=True)
    request = _request(root, output)
    report = recovery.failure_report(error, request)
    report["hybrid_qualification"] = {
        "contract": {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        "runtime_seed": runtime_seed,
        "mixed_controller_teleop_candidate": False,
        "pure_student_qualified": False,
        "teacher_labels_admitted": False,
        "hardware_authorized": False,
    }
    return recovery.write_recovery_report_new(request, report)


__all__ = ["ALLOWED_RUNTIME_SEEDS", "load_contract", "preflight", "run_qualification", "write_failure"]
