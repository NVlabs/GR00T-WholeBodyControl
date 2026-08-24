"""Compare two clean-observation seeds for the exact scale-2 student."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from gear_sonic.utils import g1_true23_sonic_scale2_observation_corruption_probe as core
from gear_sonic.utils.g1_23dof_artifact import sha256_file

SCHEMA_VERSION = 1
REPORT_KIND = "g1_true23_sonic_scale2_clean_cross_seed_probe_report_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_scale2_clean_cross_seed_probe_v1.json"
)
CONTRACT_SHA256 = "643d0dda173ac3785bdd88194246dbda0142312d1fa3b7ee969f36deccb1ced2"
REFERENCE_SEED = 20260805
FAILED_SEED = 611723381
EXECUTED_SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_sonic_scale2_clean_cross_seed_probe.py"),
    Path("gear_sonic/scripts/probe_g1_true23_sonic_scale2_clean_cross_seed.py"),
    Path("gear_sonic/utils/g1_true23_sonic_scale2_observation_corruption_probe.py"),
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ProbeRequest:
    repository_root: Path
    output: Path

    @property
    def root(self) -> Path:
        value = self.repository_root.expanduser().resolve(strict=True)
        if value.is_symlink() or not value.is_dir():
            raise ValueError("cross-seed probe repository root invalid")
        return value

    @property
    def output_path(self) -> Path:
        candidate = self.output.expanduser()
        candidate = candidate if candidate.is_absolute() else self.root / candidate
        value = candidate.resolve(strict=False)
        if not value.is_relative_to(self.root) or value.suffix.lower() != ".json":
            raise ValueError("cross-seed probe output must be repository-contained JSON")
        if candidate.is_symlink() or value.is_symlink() or value.parent.resolve(strict=True).is_symlink():
            raise ValueError("cross-seed probe output path invalid")
        return value


def _source_binding(root: Path) -> dict[str, Any]:
    files = []
    for relative in EXECUTED_SOURCE_RELATIVE_PATHS:
        path = (root / relative).resolve(strict=True)
        if path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"cross-seed source invalid: {relative.as_posix()}")
        files.append({"path": relative.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"files": files, "binding_sha256": _sha256_bytes(_canonical_bytes(files))}


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("cross-seed contract mismatch")
    contract = core._strict_json(path, "cross-seed contract")  # noqa: SLF001
    runs = contract.get("runs")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != "g1_true23_sonic_scale2_clean_cross_seed_probe_contract_v1"
        or not isinstance(runs, list)
        or [run.get("seed") for run in runs if isinstance(run, Mapping)] != [REFERENCE_SEED, FAILED_SEED]
        or any(run.get("tokenizer_corruption_enabled") is not False for run in runs)
        or runs[0].get("required_minimum_completed_transitions") != 241
        or runs[1].get("required_completed_transitions") != 236
        or runs[1].get("required_terminal_q9") != 244
    ):
        raise ValueError("cross-seed contract semantics mismatch")
    return contract


def _verify_prerequisite(root: Path, contract: Mapping[str, Any]) -> None:
    prerequisite = contract["prerequisites"]
    path = (root / prerequisite["observation_probe_relative_path"]).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != prerequisite["observation_probe_sha256"]:
        raise ValueError("cross-seed prerequisite bytes mismatch")
    report = core._strict_json(path, "cross-seed prerequisite")  # noqa: SLF001
    clean = report.get("clean")
    if (
        report.get("verdict") != prerequisite["required_verdict"]
        or report.get("noisy_reproduced_campaign_failure") is not True
        or not isinstance(clean, Mapping)
        or clean.get("completed_transitions") != prerequisite["failed_clean_completed_transitions"]
        or clean.get("terminal_q9") != prerequisite["failed_clean_terminal_q9"]
        or clean.get("termination_names") != prerequisite["failed_clean_termination_names"]
    ):
        raise ValueError("cross-seed prerequisite semantics mismatch")


def _preflight_internal(request: ProbeRequest) -> dict[str, Any]:
    root = request.root
    contract = load_contract(root)
    _verify_prerequisite(root, contract)
    sources = _source_binding(root)
    core_request = core.ProbeRequest(root, root / "artifacts/g1_true23/.cross-seed-inner-unused.json")
    core_internal = core._preflight_internal(core_request)  # noqa: SLF001
    return {"root": root, "contract": contract, "sources": sources, "core": core_internal}


def preflight(request: ProbeRequest) -> dict[str, Any]:
    try:
        value = _preflight_internal(request)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "g1_true23_sonic_scale2_clean_cross_seed_probe_preflight_v1",
            "ready": True,
            "contract_sha256": CONTRACT_SHA256,
            "seeds": [REFERENCE_SEED, FAILED_SEED],
            "executed_source_binding_sha256": value["sources"]["binding_sha256"],
            "simulator_constructed": False,
            "simulator_steps": 0,
            "training_performed": False,
            "teacher_queries": 0,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "g1_true23_sonic_scale2_clean_cross_seed_probe_preflight_v1",
            "ready": False,
            "error": {"type": type(error).__name__, "detail_sha256": _sha256_bytes(str(error).encode())},
            "simulator_constructed": False,
            "simulator_steps": 0,
            "training_performed": False,
            "teacher_queries": 0,
            "hardware_authorized": False,
        }


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    actions = result.pop("_actions")
    result.pop("_ee")
    result["first_raw_action_sha256"] = (
        None if not actions else _sha256_bytes(np.asarray(actions[0], dtype=np.float32).tobytes(order="C"))
    )
    return result


def run(request: ProbeRequest) -> dict[str, Any]:
    internal = _preflight_internal(request)
    reference = core._run_one(  # noqa: SLF001
        internal=internal["core"], corruption_enabled=False, runtime_seed=REFERENCE_SEED
    )
    failed = core._run_one(  # noqa: SLF001
        internal=internal["core"], corruption_enabled=False, runtime_seed=FAILED_SEED
    )
    comparison = core._compare(reference, failed, core.load_contract(internal["root"]))  # noqa: SLF001
    reference_survives = reference["completed_transitions"] > 236
    failed_reproduced = bool(
        failed["completed_transitions"] == 236
        and failed["terminal_q9"] == 244
        and failed["termination_names"] == ["anchor_pos"]
        and all(
            failed[name] == 0
            for name in (
                "nonfinite_count",
                "q9_discontinuity_count",
                "raw_clip_required_count",
                "action_semantics_mismatch_count",
            )
        )
    )
    initial_observations_equal = bool(
        reference["initial_encoder267_sha256"] == failed["initial_encoder267_sha256"]
        and reference["initial_policy930_sha256"] == failed["initial_policy930_sha256"]
    )
    initial_actions_equal = bool(np.array_equal(reference["_actions"][0], failed["_actions"][0]))
    sources_after = _source_binding(internal["root"])
    sources_unchanged = sources_after == internal["sources"]
    seed_sensitivity_proven = failed_reproduced and reference_survives and sources_unchanged
    if seed_sensitivity_proven and not initial_observations_equal:
        verdict = "clean_observation_or_reset_seed_leak_proven"
    elif seed_sensitivity_proven:
        verdict = "clean_initial_state_equal_but_seed_or_runtime_divergence_proven"
    else:
        verdict = "clean_cross_seed_sensitivity_not_proven"
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "contract_sha256": CONTRACT_SHA256,
        "seed_sensitivity_proven": seed_sensitivity_proven,
        "failed_seed_clean_reproduced": failed_reproduced,
        "reference_clean_survived_beyond_failed_boundary": reference_survives,
        "initial_clean_observation_hashes_equal": initial_observations_equal,
        "initial_clean_raw_actions_equal": initial_actions_equal,
        "verdict": verdict,
        "reference_clean": _compact(reference),
        "failed_seed_clean": _compact(failed),
        "comparison": comparison,
        "executed_source_binding_before_sha256": internal["sources"]["binding_sha256"],
        "executed_source_binding_after_sha256": sources_after["binding_sha256"],
        "sources_unchanged": sources_unchanged,
        "training_performed": False,
        "optimizer_steps": 0,
        "teacher_queries": 0,
        "teacher_labels": 0,
        "support_qualified": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
        "network_or_external_actuation": False,
    }
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("kind") != REPORT_KIND:
        raise ValueError("cross-seed report identity mismatch")
    reference = report.get("reference_clean")
    failed = report.get("failed_seed_clean")
    if not isinstance(reference, Mapping) or not isinstance(failed, Mapping):
        raise ValueError("cross-seed report mode results missing")
    failed_reproduced = bool(
        failed.get("completed_transitions") == 236
        and failed.get("terminal_q9") == 244
        and failed.get("termination_names") == ["anchor_pos"]
        and all(
            failed.get(name) == 0
            for name in (
                "nonfinite_count",
                "q9_discontinuity_count",
                "raw_clip_required_count",
                "action_semantics_mismatch_count",
            )
        )
    )
    reference_survives = isinstance(reference.get("completed_transitions"), int) and (
        reference["completed_transitions"] > 236
    )
    proven = failed_reproduced and reference_survives and report.get("sources_unchanged") is True
    if (
        report.get("failed_seed_clean_reproduced") is not failed_reproduced
        or report.get("reference_clean_survived_beyond_failed_boundary") is not reference_survives
        or report.get("seed_sensitivity_proven") is not proven
    ):
        raise ValueError("cross-seed report recomputation mismatch")
    if any(
        report.get(key) != value
        for key, value in {
            "training_performed": False,
            "optimizer_steps": 0,
            "teacher_queries": 0,
            "teacher_labels": 0,
            "support_qualified": False,
            "promotion_or_deployment": False,
            "hardware_authorized": False,
            "network_or_external_actuation": False,
        }.items()
    ):
        raise ValueError("cross-seed boundary mismatch")


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
        path.unlink(missing_ok=True)
        raise
