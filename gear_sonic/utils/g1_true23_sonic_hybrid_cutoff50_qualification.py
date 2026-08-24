"""Qualify a q9=59 teacher handoff after cutoff100 failed heldout835."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from gear_sonic.utils import (
    g1_true23_sonic_hybrid_cutoff100_qualification as cutoff100,
    g1_true23_sonic_nominal_dagger_student_qualification as adapter,
    g1_true23_sonic_student_teacher_recovery as recovery,
)

CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_hybrid_cutoff50_qualification_v1.json"
)
CONTRACT_SHA256 = "f97ba5aa9e97c3edd7c7f883e10e68c8afa255c0c055570bfac42d02b834ca6a"
CONTRACT_KIND = "g1_true23_sonic_hybrid_cutoff50_qualification_contract_v1"
ALLOWED_RUNTIME_SEEDS = cutoff100.ALLOWED_RUNTIME_SEEDS
SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_sonic_hybrid_cutoff50_qualification.py"),
    Path("gear_sonic/scripts/qualify_g1_true23_sonic_hybrid_cutoff50.py"),
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def _read_bound_json(root: Path, entry: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    path = (root / str(entry["path"])).resolve(strict=True)
    if path.is_symlink() or not path.is_file() or recovery.sha256_file(path) != entry["sha256"]:
        raise ValueError(f"cutoff50 qualification input mismatch: {context}")
    return _mapping(json.loads(path.read_text(encoding="utf-8")), context)


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if recovery.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("cutoff50 qualification contract hash mismatch")
    body = _mapping(json.loads(path.read_text(encoding="utf-8")), "contract")
    runtime = _mapping(body.get("runtime"), "runtime")
    boundaries = _mapping(body.get("boundaries"), "boundaries")
    decision = _mapping(body.get("decision"), "decision")
    if (
        body.get("schema_version") != 1
        or body.get("kind") != CONTRACT_KIND
        or runtime.get("mode") != "cutoff50"
        or runtime.get("total_transitions") != 510
        or runtime.get("student_transitions") != 50
        or runtime.get("teacher_transitions") != 460
        or runtime.get("student_last_q9") != 58
        or runtime.get("teacher_first_q9") != 59
        or runtime.get("allowed_seeds") != list(ALLOWED_RUNTIME_SEEDS)
        or decision.get("stop_if_cutoff50_fails_heldout835") is not True
        or boundaries.get("mixed_controller_teleop_candidate") is not True
        or boundaries.get("pure_student_qualified") is not False
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("cutoff50 qualification contract semantic drift")

    inputs = _mapping(body.get("inputs"), "inputs")
    for name, raw_entry in inputs.items():
        entry = _mapping(raw_entry, name)
        candidate = (root / str(entry["path"])).resolve(strict=True)
        if candidate.is_symlink() or not candidate.is_file() or recovery.sha256_file(candidate) != entry["sha256"]:
            raise ValueError(f"cutoff50 qualification input mismatch: {name}")

    seed611 = _read_bound_json(root, _mapping(inputs["seed611_cutoff100_pass"], "seed611"), "seed611")
    known = _read_bound_json(root, _mapping(inputs["known_seed_cutoff100_pass"], "known"), "known")
    heldout = _read_bound_json(root, _mapping(inputs["heldout835_cutoff100_failure"], "heldout835"), "heldout835")
    heldout_done = _mapping(heldout.get("first_done"), "heldout835 first_done")
    heldout_ee = _mapping(heldout_done.get("ee_body_position_scalar_evidence"), "heldout835 terminal ee")
    if any(
        report.get("verdict") != "mixed_controller_recovery_passed"
        or report.get("recovered_to_original_q9_518_boundary") is not True
        or report.get("controller_counts") != {"student": 100, "teacher": 410}
        for report in (seed611, known)
    ):
        raise ValueError("cutoff50 prior pass evidence drift")
    if (
        heldout.get("verdict") != "mixed_controller_recovery_failed"
        or heldout.get("recovered_to_original_q9_518_boundary") is not False
        or heldout.get("controller_counts") != {"student": 89, "teacher": 0}
        or heldout_done.get("q9_before") != 97
        or heldout_done.get("termination_names") != ["ee_body_pos"]
        or heldout_ee.get("dominant_absolute_z_error_body") != "left_wrist_roll_rubber_hand"
    ):
        raise ValueError("cutoff50 decision evidence drift")
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
    with cutoff100._qualification_scope(root, runtime_seed), _source_scope():  # noqa: SLF001
        yield


def _request(root: Path, output: Path) -> recovery.RecoveryRequest:
    return recovery.RecoveryRequest(
        repository_root=root,
        candidate_manifest=root / adapter.CANDIDATE_MANIFEST_RELATIVE_PATH,
        expected_candidate_manifest_sha256=adapter.CANDIDATE_MANIFEST_SHA256,
        output=output,
        mode="cutoff50",
    )


def preflight(*, repository_root: Path, output: Path, runtime_seed: int) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    load_contract(root)
    request = _request(root, output)
    if os.path.lexists(request.output_path):
        raise FileExistsError("cutoff50 qualification output exists")
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
        raise FileExistsError("cutoff50 qualification output exists")
    with _qualification_scope(root, runtime_seed):
        report = dict(recovery.run_recovery_diagnostic(request))
        report["hybrid_qualification"] = {
            "contract": {"path": CONTRACT_RELATIVE_PATH.as_posix(), "sha256": CONTRACT_SHA256},
            "runtime_seed": runtime_seed,
            "student_candidate_manifest_sha256": adapter.CANDIDATE_MANIFEST_SHA256,
            "student_candidate_decoder_sha256": adapter.CANDIDATE_DECODER_SHA256,
            "controller": "student_q9_9_through_58_then_exact_selected_teacher_q9_59_through_518",
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
