"""Fail-closed teacher-actuated full-episode qualification campaign."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from gear_sonic.utils import (
    g1_true23_sonic_recovery_qualification_campaign as base_campaign,
    g1_true23_sonic_scale2_recovery_qualification_campaign as mixed_campaign,
    g1_true23_sonic_scale2_teacher_recovery as recovery,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_selected_teacher_actuated_campaign_contract_v1"
REPORT_KIND = "g1_true23_selected_teacher_actuated_campaign_report_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_selected_teacher_actuated_campaign_v1.json"
)
CONTRACT_SHA256 = "fe042ba8c199a7e0a3fd614f5f822be1468ac2f5a5ec87d6302bdbf9ca72a112"
EXECUTED_SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_selected_teacher_actuated_campaign.py"),
    Path("gear_sonic/scripts/qualify_g1_true23_selected_teacher_actuated_campaign.py"),
    Path("gear_sonic/utils/g1_true23_sonic_scale2_teacher_recovery.py"),
    Path("gear_sonic/utils/g1_true23_sonic_scale2_recovery_qualification_campaign.py"),
    Path("gear_sonic/utils/g1_true23_sonic_recovery_qualification_campaign.py"),
)

TOTAL_TRANSITIONS = 510
STUDENT_TRANSITIONS = 0
TEACHER_TRANSITIONS = 510
GLOBAL_IMPULSE_TRANSITION = 50
IMPULSE_Q9 = 59
FIRST_DISTURBED_Q9 = 60
BASELINE_TRANSITIONS = tuple(range(40, 50))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be mapping")
    return value


@dataclass(frozen=True)
class CampaignRequest:
    repository_root: Path
    output: Path

    @property
    def root(self) -> Path:
        value = self.repository_root.expanduser().resolve(strict=True)
        if value.is_symlink() or not value.is_dir():
            raise ValueError("teacher campaign repository root invalid")
        return value

    @property
    def output_path(self) -> Path:
        candidate = self.output.expanduser()
        candidate = candidate if candidate.is_absolute() else self.root / candidate
        value = candidate.resolve(strict=False)
        if not value.is_relative_to(self.root) or value.suffix.lower() != ".json":
            raise ValueError("teacher campaign output must be repository-contained JSON")
        if candidate.is_symlink() or value.is_symlink():
            raise ValueError("teacher campaign output cannot be symlink")
        parent = value.parent.resolve(strict=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("teacher campaign output parent invalid")
        return value


def _source_binding(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in EXECUTED_SOURCE_RELATIVE_PATHS:
        path = (root / relative).resolve(strict=True)
        if path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"teacher campaign source invalid: {relative.as_posix()}")
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"files": files, "binding_sha256": _sha256_bytes(_canonical_bytes(files))}


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("teacher campaign contract mismatch")
    contract = mixed_campaign._strict_json(path, "teacher campaign contract")  # noqa: SLF001
    window = _mapping(contract.get("window"), "teacher campaign window")
    cohort = _mapping(contract.get("cohort"), "teacher campaign cohort")
    disturbance = _mapping(contract.get("disturbance"), "teacher campaign disturbance")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("role") != "offline_mjlab_teacher_actuated_full_episode_qualification_only"
        or window.get("mode") != "q9"
        or window.get("total_transitions") != TOTAL_TRANSITIONS
        or window.get("student_transition_count") != STUDENT_TRANSITIONS
        or window.get("teacher_transition_count") != TEACHER_TRANSITIONS
        or window.get("teacher_first_q9") != 9
        or window.get("last_action_q9") != 518
        or cohort.get("total_rollouts") != 20
        or cohort.get("rollouts_per_scenario") != 10
        or cohort.get("fail_fast") is not True
        or disturbance.get("global_apply_transition") != GLOBAL_IMPULSE_TRANSITION
        or disturbance.get("apply_q9") != IMPULSE_Q9
        or disturbance.get("first_disturbed_observation_q9") != FIRST_DISTURBED_Q9
        or disturbance.get("baseline_global_transitions") != list(BASELINE_TRANSITIONS)
        or disturbance.get("stable_recovery_steps") != mixed_campaign.RECOVERY_STABLE_STEPS
        or disturbance.get("maximum_recovery_time_s") != 2.0
    ):
        raise ValueError("teacher campaign contract semantics mismatch")
    return contract


def _verify_prerequisites(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    prerequisite = _mapping(contract["prerequisites"], "teacher campaign prerequisites")
    base_path = (root / str(prerequisite["base_campaign_contract_relative_path"])).resolve(strict=True)
    q9_path = (root / str(prerequisite["teacher_q9_pass_report_relative_path"])).resolve(strict=True)
    if (
        base_path.is_symlink()
        or q9_path.is_symlink()
        or sha256_file(base_path) != prerequisite["base_campaign_contract_sha256"]
        or sha256_file(q9_path) != prerequisite["teacher_q9_pass_report_sha256"]
    ):
        raise ValueError("teacher campaign prerequisite bytes mismatch")
    base_contract = base_campaign.load_campaign_contract(root)
    q9 = mixed_campaign._strict_json(q9_path, "teacher q9 prerequisite")  # noqa: SLF001
    qualification = _mapping(q9.get("qualification"), "teacher q9 qualification")
    if (
        q9.get("kind") != "g1_true23_sonic_scale2_teacher_recovery_diagnostic_v1"
        or q9.get("mode") != "q9"
        or qualification.get("passed") is not True
        or q9.get("completed_transitions") != TOTAL_TRANSITIONS
        or q9.get("controller_counts") != {"student": STUDENT_TRANSITIONS, "teacher": TEACHER_TRANSITIONS}
        or q9.get("first_done", {}).get("termination_names") != ["time_out"]
        or q9.get("teacher_labels_admitted") != 0
        or q9.get("training_performed") is not False
    ):
        raise ValueError("teacher q9 prerequisite is not exact safe pass")
    return {"base_contract": base_contract, "q9": q9}


def _base_request(root: Path, seed: int | None = None) -> recovery.Scale2RecoveryRequest:
    return recovery.Scale2RecoveryRequest(
        root,
        root / "artifacts/g1_true23/.teacher-campaign-inner-unused.json",
        "q9",
        runtime_seed=seed,
    )


@contextmanager
def _q9_campaign_constants():
    names = {
        "STUDENT_TRANSITIONS": STUDENT_TRANSITIONS,
        "TEACHER_TRANSITIONS": TEACHER_TRANSITIONS,
        "GLOBAL_IMPULSE_TRANSITION": GLOBAL_IMPULSE_TRANSITION,
        "IMPULSE_Q9": IMPULSE_Q9,
        "FIRST_DISTURBED_Q9": FIRST_DISTURBED_Q9,
        "BASELINE_TRANSITIONS": BASELINE_TRANSITIONS,
    }
    original = {name: getattr(mixed_campaign, name) for name in names}
    for name, value in names.items():
        setattr(mixed_campaign, name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(mixed_campaign, name, value)


def _preflight_internal(request: CampaignRequest) -> dict[str, Any]:
    root = request.root
    contract = load_contract(root)
    prerequisites = _verify_prerequisites(root, contract)
    sources = _source_binding(root)
    recovery_preflight = recovery._preflight_internal(_base_request(root))  # noqa: SLF001
    if recovery_preflight.get("ready") is not True:
        raise RuntimeError("teacher campaign recovery preflight not ready")
    specs = base_campaign.campaign_run_specs(root)
    if len(specs) != 20:
        raise RuntimeError("teacher campaign spec count mismatch")
    return {
        "ready": True,
        "root": root,
        "contract": contract,
        "prerequisites": prerequisites,
        "sources": sources,
        "recovery_preflight": recovery_preflight,
        "specs": specs,
    }


def preflight(request: CampaignRequest) -> dict[str, Any]:
    try:
        value = _preflight_internal(request)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "g1_true23_selected_teacher_actuated_campaign_preflight_v1",
            "ready": True,
            "planned_run_count": 20,
            "window": recovery.resolve_window("q9").to_dict(),
            "contract_sha256": CONTRACT_SHA256,
            "teacher_q9_prerequisite_report_sha256": value["contract"]["prerequisites"][
                "teacher_q9_pass_report_sha256"
            ],
            "executed_source_binding_sha256": value["sources"]["binding_sha256"],
            "simulator_constructed": False,
            "simulator_steps": 0,
            "published_teacher_label_count": 0,
            "published_training_row_count": 0,
            "training_performed": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "g1_true23_selected_teacher_actuated_campaign_preflight_v1",
            "ready": False,
            "error": {"type": type(error).__name__, "detail_sha256": _sha256_bytes(str(error).encode())},
            "simulator_constructed": False,
            "simulator_steps": 0,
            "published_teacher_label_count": 0,
            "published_training_row_count": 0,
            "training_performed": False,
            "hardware_authorized": False,
        }


def run_campaign(
    request: CampaignRequest, progress: Callable[[Mapping[str, Any]], None] | None = None
) -> dict[str, Any]:
    preflight_value = _preflight_internal(request)
    root = request.root
    contract = preflight_value["contract"]
    base_contract = preflight_value["prerequisites"]["base_contract"]
    cached = preflight_value["recovery_preflight"]
    specs: Sequence[Any] = preflight_value["specs"]
    records: list[dict[str, Any]] = []
    with _q9_campaign_constants():
        for spec in specs:
            probe = mixed_campaign._StepProbe(spec, contract)  # noqa: SLF001
            try:
                random.seed(spec.seed)
                np.random.seed(spec.seed % (2**32))
                torch.manual_seed(spec.seed)
                torch.cuda.manual_seed_all(spec.seed)
                run_request = _base_request(root, spec.seed)
                with mixed_campaign._instrumented_run(spec, probe, cached):  # noqa: SLF001
                    full_report = recovery.run(run_request)
                probe_report = probe.report()
                record = mixed_campaign._assess_run(  # noqa: SLF001
                    spec, full_report, probe_report, base_contract
                )
            except Exception as error:
                record = mixed_campaign._failure_record(spec, error, probe.report())  # noqa: SLF001
            records.append(record)
            if progress is not None:
                progress(record)
            if record["passed"] is not True:
                break
    sources_after = _source_binding(root)
    passed_count = sum(record["passed"] is True for record in records)
    nominal_passed = sum(record["passed"] is True and record["scenario"] == "nominal" for record in records)
    disturbance_passed = sum(
        record["passed"] is True and record["scenario"] == "disturbance" for record in records
    )
    recovered_count = sum(
        record["passed"] is True and record["scenario"] == "disturbance" and record["recovered"] is True
        for record in records
    )
    seeds = [record["effective_seed"] for record in records]
    qualified = bool(
        len(records) == 20
        and passed_count == nominal_passed + disturbance_passed == 20
        and nominal_passed == disturbance_passed == recovered_count == 10
        and len(set(seeds)) == 20
        and sources_after == preflight_value["sources"]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "campaign_qualified": qualified,
        "verdict": "selected_teacher_actuated_campaign_passed"
        if qualified
        else "selected_teacher_actuated_campaign_failed",
        "campaign_contract_sha256": CONTRACT_SHA256,
        "teacher_q9_prerequisite_report_sha256": contract["prerequisites"]["teacher_q9_pass_report_sha256"],
        "planned_run_count": 20,
        "completed_run_count": len(records),
        "passed_run_count": passed_count,
        "nominal_passed_count": nominal_passed,
        "disturbance_passed_count": disturbance_passed,
        "disturbance_recovered_count": recovered_count,
        "first_failed_run_id": next((record["run_id"] for record in records if not record["passed"]), None),
        "executed_source_binding_before_sha256": preflight_value["sources"]["binding_sha256"],
        "executed_source_binding_after_sha256": sources_after["binding_sha256"],
        "sources_unchanged": sources_after == preflight_value["sources"],
        "runs": records,
        "fresh_disjoint_teacher_collection_authorized": qualified,
        "teacher_support_qualified": qualified,
        "published_teacher_label_count": 0,
        "published_training_row_count": 0,
        "dagger_data": False,
        "training_performed": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
    }
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("kind") != REPORT_KIND:
        raise ValueError("teacher campaign report identity mismatch")
    records = report.get("runs")
    if not isinstance(records, list) or len(records) != report.get("completed_run_count"):
        raise ValueError("teacher campaign run list mismatch")
    qualified = report.get("campaign_qualified") is True
    recomputed = bool(
        len(records) == 20
        and all(isinstance(record, Mapping) and record.get("passed") is True for record in records)
        and sum(record.get("scenario") == "nominal" for record in records) == 10
        and sum(record.get("scenario") == "disturbance" and record.get("recovered") is True for record in records)
        == 10
        and report.get("sources_unchanged") is True
    )
    if qualified != recomputed:
        raise ValueError("teacher campaign qualification recomputation mismatch")
    if (
        report.get("fresh_disjoint_teacher_collection_authorized") is not qualified
        or report.get("teacher_support_qualified") is not qualified
        or report.get("published_teacher_label_count") != 0
        or report.get("published_training_row_count") != 0
        or report.get("training_performed") is not False
        or report.get("hardware_authorized") is not False
    ):
        raise ValueError("teacher campaign claim boundary mismatch")


def write_json_exclusive(path: Path, report: Mapping[str, Any]) -> None:
    validate_report(report)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


__all__ = [
    "CampaignRequest",
    "load_contract",
    "preflight",
    "run_campaign",
    "validate_report",
    "write_json_exclusive",
]
