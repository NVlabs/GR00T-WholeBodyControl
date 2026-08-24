"""Exact q200 selected-teacher recovery diagnostic for rank256 SONIC student."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from gear_sonic.utils import (
    g1_true23_sonic_seed835_counterexample_round3_rank256_student_qualification as adapter,
    g1_true23_sonic_student_teacher_recovery as recovery,
)

CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_q200_teacher_recovery_v1.json"
)
CONTRACT_SHA256 = "c7693102814b2d992a134b88c89f146de656e1cf34dc14b755c2c1e0d32e6683"
CONTRACT_KIND = "g1_true23_sonic_rank256_q200_teacher_recovery_contract_v1"
RUNTIME_SEED = 835868017
MODE = "cutoff191"
SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_sonic_rank256_q200_teacher_recovery.py"),
    Path("gear_sonic/scripts/diagnose_g1_true23_sonic_rank256_q200_teacher_recovery.py"),
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if recovery.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("q200 recovery contract hash mismatch")
    body = _mapping(json.loads(path.read_text(encoding="utf-8")), "contract")
    window = _mapping(body.get("window"), "window")
    gates = _mapping(body.get("gates"), "gates")
    if (
        body.get("schema_version") != 1
        or body.get("kind") != CONTRACT_KIND
        or window.get("student_transition_count") != 191
        or window.get("student_last_q9") != 199
        or window.get("teacher_transition_count") != 319
        or window.get("teacher_first_q9") != 200
        or gates.get("expected_controller_counts") != {"student": 191, "teacher": 319}
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("q200 recovery contract semantic drift")
    for name, raw_entry in _mapping(body.get("inputs"), "inputs").items():
        entry = _mapping(raw_entry, name)
        candidate = (root / str(entry["path"])).resolve(strict=True)
        if candidate.is_symlink() or not candidate.is_file() or recovery.sha256_file(candidate) != entry["sha256"]:
            raise ValueError(f"q200 recovery input mismatch: {name}")
    return body


@contextmanager
def _scope(root: Path) -> Iterator[None]:
    with adapter._adapter_scope(root, RUNTIME_SEED):  # noqa: SLF001
        saved = {
            "MODE_STUDENT_TRANSITIONS": recovery.MODE_STUDENT_TRANSITIONS,
            "MODES": recovery.MODES,
            "CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH": recovery.CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH,
            "CURRENT_CANDIDATE_MANIFEST_SHA256": recovery.CURRENT_CANDIDATE_MANIFEST_SHA256,
            "CURRENT_CANDIDATE_DECODER_SHA256": recovery.CURRENT_CANDIDATE_DECODER_SHA256,
            "RECOVERY_CONTRACT_SHA256": recovery.RECOVERY_CONTRACT_SHA256,
            "load_recovery_contract": recovery.load_recovery_contract,
            "RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS": recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS,
        }
        mode_map = {**recovery.MODE_STUDENT_TRANSITIONS, MODE: 191}
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
        contract["modes"][MODE] = {
            "student_transition_count": 191,
            "student_first_q9": 9,
            "student_last_q9": 199,
            "teacher_transition_count": 319,
            "teacher_first_q9": 200,
            "teacher_last_q9": 518,
        }

        def load_current(_root: Path) -> Mapping[str, Any]:
            recovery._validate_recovery_contract(contract)  # noqa: SLF001
            return copy.deepcopy(contract)

        try:
            recovery.MODE_STUDENT_TRANSITIONS = mode_map
            recovery.MODES = tuple(mode_map)
            recovery.CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH = adapter.CANDIDATE_MANIFEST_RELATIVE_PATH
            recovery.CURRENT_CANDIDATE_MANIFEST_SHA256 = adapter.CANDIDATE_MANIFEST_SHA256
            recovery.CURRENT_CANDIDATE_DECODER_SHA256 = adapter.CANDIDATE_DECODER_SHA256
            recovery.RECOVERY_CONTRACT_SHA256 = recovery.sha256_bytes(recovery.canonical_json_bytes(contract))
            recovery.load_recovery_contract = load_current
            recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS = (
                *saved["RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS"],
                *SOURCE_PATHS,
            )
            yield
        finally:
            for name, value in saved.items():
                setattr(recovery, name, value)


def _request(root: Path, output: Path) -> recovery.RecoveryRequest:
    return recovery.RecoveryRequest(
        repository_root=root,
        candidate_manifest=root / adapter.CANDIDATE_MANIFEST_RELATIVE_PATH,
        expected_candidate_manifest_sha256=adapter.CANDIDATE_MANIFEST_SHA256,
        output=output,
        mode=MODE,
    )


def preflight(*, repository_root: Path, output: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    load_contract(root)
    with _scope(root):
        inner = recovery._preflight_internal(_request(root, output))  # noqa: SLF001
    return {
        "ready": inner.get("ready") is True,
        "issues": list(inner.get("issues", ())),
        "contract_sha256": CONTRACT_SHA256,
        "runtime_seed": RUNTIME_SEED,
        "mode": MODE,
        "student_last_q9": 199,
        "teacher_first_q9": 200,
        "simulator_constructed": False,
        "simulator_steps": 0,
        "hardware_authorized": False,
    }


def run(*, repository_root: Path, output: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    output_path = output if output.is_absolute() else root / output
    if os.path.lexists(output_path):
        raise FileExistsError("q200 recovery output exists")
    pre = preflight(repository_root=root, output=output)
    if pre.get("ready") is not True:
        raise RuntimeError("q200 recovery preflight not ready")
    with _scope(root):
        request = _request(root, output)
        report = dict(recovery.run_recovery_diagnostic(request))
        report["q200_recovery_contract"] = {
            "path": CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": CONTRACT_SHA256,
            "student_last_q9": 199,
            "teacher_first_q9": 200,
            "teacher_labels_admitted": False,
            "hardware_authorized": False,
        }
        recovery.write_recovery_report_new(request, report)
    if (
        report.get("verdict") != "mixed_controller_recovery_passed"
        or report.get("attempted_transitions") != 510
        or report.get("controller_counts") != {"student": 191, "teacher": 319}
        or report.get("recovered_to_original_q9_518_boundary") is not True
    ):
        raise RuntimeError("q200 selected-teacher recovery failed")
    return report


__all__ = ["CONTRACT_SHA256", "load_contract", "preflight", "run"]
