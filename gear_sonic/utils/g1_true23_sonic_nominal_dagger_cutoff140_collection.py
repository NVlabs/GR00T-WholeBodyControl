"""Capture cutoff140 intervention rows for the improved SONIC student."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_cutoff50_collection as base,
    g1_true23_sonic_nominal_dagger_student_qualification as adapter,
    g1_true23_sonic_student_teacher_recovery as recovery,
)

CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_nominal_dagger_cutoff140_collection_v2.json"
)
CONTRACT_SHA256 = "291a788ad513d7c2be75bc3dd56170a6ab5c43926b28c04cf8854d2b049f7e2f"
MANIFEST_KIND = "g1_true23_sonic_nominal_dagger_cutoff140_collection_manifest_v2"
STUDENT_ROWS = 140
TEACHER_ROWS = 370


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if base.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("cutoff140 contract hash mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    if (
        body.get("schema_version") != 2
        or body.get("kind") != "g1_true23_sonic_nominal_dagger_cutoff140_collection_contract_v2"
        or body.get("runtime", {}).get("mode") != "cutoff140"
        or body.get("runtime", {}).get("student_controlled_transitions") != STUDENT_ROWS
        or body.get("runtime", {}).get("teacher_controlled_transitions") != TEACHER_ROWS
        or body.get("rows", {}).get("published_total") != 510
        or body.get("boundaries", {}).get("support_qualified") is not False
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("cutoff140 contract semantic drift")
    return body


def preflight(request: base.CollectionRequest) -> Mapping[str, Any]:
    root = request.root
    contract = load_contract(root)
    paths = {
        name: base._regular_file(root, base._mapping(entry, name), name)  # noqa: SLF001
        for name, entry in contract["inputs"].items()
    }
    failed = json.loads(paths["failed_student_report"].read_text(encoding="utf-8"))
    if (
        failed.get("verdict") != "student_qualification_failed"
        or failed.get("attempted_transitions") != 227
        or failed.get("first_done", {}).get("transition") != 226
        or failed.get("first_done", {}).get("termination_names") != ["ee_body_pos"]
    ):
        raise ValueError("cutoff140 parent failure drift")
    return {
        "ready": True,
        "contract_sha256": CONTRACT_SHA256,
        "candidate_manifest_sha256": adapter.CANDIDATE_MANIFEST_SHA256,
        "candidate_decoder_sha256": adapter.CANDIDATE_DECODER_SHA256,
        "student_rows": STUDENT_ROWS,
        "teacher_rows": TEACHER_ROWS,
        "simulator_constructed": False,
        "teacher_labels_admitted": 0,
        "hardware_authorized": False,
    }


@contextmanager
def _current_recovery_scope(root: Path) -> Iterator[None]:
    original = {
        "CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH": recovery.CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH,
        "CURRENT_CANDIDATE_MANIFEST_SHA256": recovery.CURRENT_CANDIDATE_MANIFEST_SHA256,
        "CURRENT_CANDIDATE_DECODER_SHA256": recovery.CURRENT_CANDIDATE_DECODER_SHA256,
        "RECOVERY_CONTRACT_SHA256": recovery.RECOVERY_CONTRACT_SHA256,
        "load_recovery_contract": recovery.load_recovery_contract,
        "RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS": recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS,
    }
    contract = json.loads(
        (root / "gear_sonic/config/sim_validation/g1_true23_sonic_student_teacher_recovery_v4.json").read_text(
            encoding="utf-8"
        )
    )
    contract["candidate"] = {
        "manifest_relative_path": adapter.CANDIDATE_MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_sha256": adapter.CANDIDATE_MANIFEST_SHA256,
        "decoder_sha256": adapter.CANDIDATE_DECODER_SHA256,
        "encoder_sha256": recovery.student.CAUSAL_ENCODER_SHA256,
        "bc_contract_sha256": recovery.student.BC_CONTRACT_SHA256,
    }

    def load_current(_root: Path) -> Mapping[str, Any]:
        recovery._validate_recovery_contract(contract)  # noqa: SLF001
        return copy.deepcopy(contract)

    try:
        recovery.CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH = adapter.CANDIDATE_MANIFEST_RELATIVE_PATH
        recovery.CURRENT_CANDIDATE_MANIFEST_SHA256 = adapter.CANDIDATE_MANIFEST_SHA256
        recovery.CURRENT_CANDIDATE_DECODER_SHA256 = adapter.CANDIDATE_DECODER_SHA256
        recovery.RECOVERY_CONTRACT_SHA256 = recovery.sha256_bytes(recovery.canonical_json_bytes(contract))
        recovery.load_recovery_contract = load_current
        recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS = (
            *original["RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS"],
            CONTRACT_RELATIVE_PATH,
            Path("gear_sonic/utils/g1_true23_sonic_nominal_dagger_cutoff140_collection.py"),
            Path("gear_sonic/scripts/collect_g1_true23_sonic_nominal_dagger_cutoff140.py"),
        )
        yield
    finally:
        for name, value in original.items():
            setattr(recovery, name, value)


def _request(root: Path) -> recovery.RecoveryRequest:
    return recovery.RecoveryRequest(
        root,
        root / adapter.CANDIDATE_MANIFEST_RELATIVE_PATH,
        adapter.CANDIDATE_MANIFEST_SHA256,
        Path("artifacts/g1_true23/.cutoff140-collector-unused.json"),
        "cutoff140",
    )


@contextmanager
def _base_patch(root: Path) -> Iterator[None]:
    saved = {
        "STUDENT_ROWS": base.STUDENT_ROWS,
        "TEACHER_ROWS": base.TEACHER_ROWS,
        "CONTRACT_RELATIVE_PATH": base.CONTRACT_RELATIVE_PATH,
        "MANIFEST_KIND": base.MANIFEST_KIND,
        "CANDIDATE_DECODER_SHA256": base.CANDIDATE_DECODER_SHA256,
        "adapter": base.adapter,
        "load_contract": base.load_contract,
        "preflight": base.preflight,
        "_recovery_scope": base._recovery_scope,  # noqa: SLF001
        "_recovery_request": base._recovery_request,  # noqa: SLF001
    }
    try:
        base.STUDENT_ROWS = STUDENT_ROWS
        base.TEACHER_ROWS = TEACHER_ROWS
        base.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
        base.MANIFEST_KIND = MANIFEST_KIND
        base.CANDIDATE_DECODER_SHA256 = adapter.CANDIDATE_DECODER_SHA256
        base.adapter = adapter
        base.load_contract = load_contract
        base.preflight = preflight
        base._recovery_scope = _current_recovery_scope  # noqa: SLF001
        base._recovery_request = _request  # noqa: SLF001
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def collect_and_publish(
    request: base.CollectionRequest,
) -> tuple[Path, Path, Mapping[str, Any]]:
    preflight(request)
    with _base_patch(request.root):
        arrays, materials = base.collect(request)
        return base.publish(request, arrays, materials)


def write_failure(request: base.CollectionRequest, error: BaseException) -> Path:
    with _base_patch(request.root):
        return base.write_failure(request, error)


__all__ = ["collect_and_publish", "load_contract", "preflight", "write_failure"]
