"""Exact 10+10 rank256 q200 selected-teacher recovery campaign."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
from typing import Any

from gear_sonic.utils import (
    g1_true23_sonic_rank256_q200_teacher_recovery as q200,
    g1_true23_sonic_recovery_qualification_campaign as campaign,
)

OVERLAY_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/"
    "g1_true23_sonic_rank256_q200_recovery_qualification_campaign_v1.json"
)
OVERLAY_SHA256 = "6dfd070423705996b6debed1b01055f492aa2f0681de4b48a641336f9e8f3005"
OVERLAY_KIND = "g1_true23_sonic_rank256_q200_recovery_qualification_campaign_overlay_v1"
CAMPAIGN_CONTRACT_KIND = "g1_true23_sonic_rank256_q200_recovery_qualification_campaign_contract_v1"
REPORT_KIND = "g1_true23_sonic_rank256_q200_recovery_qualification_campaign_report_v1"
ROLE = "offline_mjlab_rank256_q200_mixed_controller_qualification_only"
MODE = "cutoff191"

STUDENT_TRANSITIONS = 191
TEACHER_TRANSITIONS = 319
GLOBAL_IMPULSE_TRANSITION = 241
IMPULSE_Q9 = 250
FIRST_DISTURBED_OBSERVATION_TRANSITION = 242
FIRST_DISTURBED_OBSERVATION_Q9 = 251
BASELINE_TRANSITIONS = tuple(range(231, 241))

PREREQUISITE_REPORT_NAME = "q200_pass_report"
PREREQUISITE_REPORT_PATH_KEY = "q200_pass_report_relative_path"
PREREQUISITE_REPORT_SHA_KEY = "q200_pass_report_sha256"
PREREQUISITE_REPORT_OUTPUT_KEY = "q200_prerequisite_report_sha256"

SOURCE_PATHS = (
    OVERLAY_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_sonic_rank256_q200_recovery_qualification_campaign.py"),
    Path("gear_sonic/scripts/qualify_g1_true23_sonic_rank256_q200_recovery_campaign.py"),
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def _strict_json(path: Path, context: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, context)


def _bound_file(root: Path, entry: Mapping[str, Any], context: str) -> Path:
    path = (root / str(entry.get("path"))).resolve(strict=True)
    if (
        path.is_symlink()
        or not path.is_file()
        or campaign._sha256_file(path) != entry.get("sha256")  # noqa: SLF001
    ):
        raise ValueError(f"q200 campaign bound file mismatch: {context}")
    return path


def load_overlay(root: Path) -> Mapping[str, Any]:
    path = (root / OVERLAY_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or campaign._sha256_file(path) != OVERLAY_SHA256:  # noqa: SLF001
        raise ValueError("q200 campaign overlay hash mismatch")
    overlay = _strict_json(path, "q200 campaign overlay")
    if (
        overlay.get("schema_version") != 1
        or overlay.get("kind") != OVERLAY_KIND
        or overlay.get("role") != ROLE
        or _mapping(overlay.get("window"), "window").get("mode") != MODE
        or _mapping(overlay.get("window"), "window").get("student_transition_count")
        != STUDENT_TRANSITIONS
        or _mapping(overlay.get("window"), "window").get("teacher_transition_count")
        != TEACHER_TRANSITIONS
        or _mapping(overlay.get("boundaries"), "boundaries").get("hardware_authorized")
        is not False
    ):
        raise ValueError("q200 campaign overlay semantic mismatch")
    for name in ("base_campaign_contract", "q200_recovery_contract", "q200_pass_report"):
        _bound_file(root, _mapping(overlay.get(name), name), name)
    candidate = _mapping(overlay.get("candidate"), "candidate")
    _bound_file(
        root,
        {"path": candidate.get("manifest_path"), "sha256": candidate.get("manifest_sha256")},
        "candidate manifest",
    )
    if candidate.get("decoder_sha256") != q200.adapter.CANDIDATE_DECODER_SHA256:
        raise ValueError("q200 campaign candidate decoder mismatch")
    timing = _mapping(overlay.get("disturbance_timing"), "disturbance timing")
    expected_timing = {
        "teacher_local_apply_transition": 50,
        "global_apply_transition": GLOBAL_IMPULSE_TRANSITION,
        "apply_q9": IMPULSE_Q9,
        "first_disturbed_observation_global_transition": FIRST_DISTURBED_OBSERVATION_TRANSITION,
        "first_disturbed_observation_q9": FIRST_DISTURBED_OBSERVATION_Q9,
        "baseline_global_transitions": list(BASELINE_TRANSITIONS),
        "scan_first_global_transition": GLOBAL_IMPULSE_TRANSITION,
    }
    if dict(timing) != expected_timing:
        raise ValueError("q200 campaign disturbance timing mismatch")
    return overlay


def _synthesized_contract(root: Path, overlay: Mapping[str, Any]) -> Mapping[str, Any]:
    base_entry = _mapping(overlay["base_campaign_contract"], "base contract")
    base = copy.deepcopy(_strict_json(_bound_file(root, base_entry, "base contract"), "base contract"))
    candidate = _mapping(overlay["candidate"], "candidate")
    pass_report = _mapping(overlay["q200_pass_report"], "q200 pass report")
    recovery_contract = _mapping(overlay["q200_recovery_contract"], "q200 recovery contract")

    base["kind"] = CAMPAIGN_CONTRACT_KIND
    base["role"] = ROLE
    prerequisites = dict(_mapping(base["prerequisites"], "base prerequisites"))
    for key in (
        "cutoff50_pass_report_relative_path",
        "cutoff50_pass_report_sha256",
    ):
        prerequisites.pop(key, None)
    prerequisites.update(
        {
            "recovery_contract_relative_path": recovery_contract["path"],
            "recovery_contract_sha256": recovery_contract["sha256"],
            PREREQUISITE_REPORT_PATH_KEY: pass_report["path"],
            PREREQUISITE_REPORT_SHA_KEY: pass_report["sha256"],
            "candidate_manifest_relative_path": candidate["manifest_path"],
            "candidate_manifest_sha256": candidate["manifest_sha256"],
            "candidate_decoder_sha256": candidate["decoder_sha256"],
        }
    )
    base["prerequisites"] = prerequisites
    base["window"] = {
        **dict(_mapping(overlay["window"], "overlay window")),
        "control_hz": 50.0,
        "final_timeout_only": True,
    }
    timing = _mapping(overlay["disturbance_timing"], "disturbance timing")
    disturbance = dict(_mapping(base["disturbance"], "base disturbance"))
    disturbance.update(
        {
            "global_apply_transition": timing["global_apply_transition"],
            "apply_q9": timing["apply_q9"],
            "first_disturbed_observation_global_transition": timing[
                "first_disturbed_observation_global_transition"
            ],
            "first_disturbed_observation_q9": timing["first_disturbed_observation_q9"],
            "baseline_global_transitions": timing["baseline_global_transitions"],
        }
    )
    metric = dict(_mapping(disturbance["recovery_metric"], "recovery metric"))
    metric.update(
        {
            "threshold_formula": "mean(metric[global231:global241])+0.1",
            "scan_first_global_transition": GLOBAL_IMPULSE_TRANSITION,
            "recovery_time_formula": "(stable_streak_end_global_transition-241+1)*0.02",
        }
    )
    disturbance["recovery_metric"] = metric
    base["disturbance"] = disturbance
    base["next_stage_boundary"] = {
        "fresh_disjoint_collection_required_after_campaign_pass": True,
        "collector_controller_scope": "teacher_actuated_suffix_only",
        "collector_teacher_first_q9": 200,
        "collector_teacher_last_q9": 518,
        "collector_rows_per_successful_run": TEACHER_TRANSITIONS,
        "campaign_rows_reusable_for_training": 0,
    }
    return base


@contextmanager
def _scope(root: Path) -> Iterator[None]:
    overlay = load_overlay(root)
    synthesized = _synthesized_contract(root, overlay)
    names = (
        "CONTRACT_KIND",
        "REPORT_KIND",
        "CONTRACT_RELATIVE_PATH",
        "CONTRACT_SHA256",
        "RECOVERY_ROLE",
        "RECOVERY_MODE",
        "STUDENT_TRANSITIONS",
        "TEACHER_TRANSITIONS",
        "STUDENT_FIRST_Q9",
        "STUDENT_LAST_Q9",
        "TEACHER_FIRST_Q9",
        "TEACHER_LAST_Q9",
        "GLOBAL_IMPULSE_TRANSITION",
        "IMPULSE_Q9",
        "FIRST_DISTURBED_OBSERVATION_TRANSITION",
        "FIRST_DISTURBED_OBSERVATION_Q9",
        "BASELINE_TRANSITIONS",
        "RECOVERY_SCAN_FIRST",
        "PREREQUISITE_REPORT_NAME",
        "PREREQUISITE_REPORT_PATH_KEY",
        "PREREQUISITE_REPORT_SHA_KEY",
        "PREREQUISITE_REPORT_OUTPUT_KEY",
        "EXECUTED_SOURCE_RELATIVE_PATHS",
        "TOP_LEVEL_KEYS",
        "load_campaign_contract",
    )
    saved = {name: getattr(campaign, name) for name in names}

    def load_current(_root: str | Path | None = None) -> Mapping[str, Any]:
        campaign._validate_contract(synthesized)  # noqa: SLF001
        return copy.deepcopy(synthesized)

    with q200._scope(root):  # noqa: SLF001
        try:
            campaign.CONTRACT_KIND = CAMPAIGN_CONTRACT_KIND
            campaign.REPORT_KIND = REPORT_KIND
            campaign.CONTRACT_RELATIVE_PATH = OVERLAY_RELATIVE_PATH
            campaign.CONTRACT_SHA256 = OVERLAY_SHA256
            campaign.RECOVERY_ROLE = ROLE
            campaign.RECOVERY_MODE = MODE
            campaign.STUDENT_TRANSITIONS = STUDENT_TRANSITIONS
            campaign.TEACHER_TRANSITIONS = TEACHER_TRANSITIONS
            campaign.STUDENT_FIRST_Q9 = 9
            campaign.STUDENT_LAST_Q9 = 199
            campaign.TEACHER_FIRST_Q9 = 200
            campaign.TEACHER_LAST_Q9 = 518
            campaign.GLOBAL_IMPULSE_TRANSITION = GLOBAL_IMPULSE_TRANSITION
            campaign.IMPULSE_Q9 = IMPULSE_Q9
            campaign.FIRST_DISTURBED_OBSERVATION_TRANSITION = FIRST_DISTURBED_OBSERVATION_TRANSITION
            campaign.FIRST_DISTURBED_OBSERVATION_Q9 = FIRST_DISTURBED_OBSERVATION_Q9
            campaign.BASELINE_TRANSITIONS = BASELINE_TRANSITIONS
            campaign.RECOVERY_SCAN_FIRST = GLOBAL_IMPULSE_TRANSITION
            campaign.PREREQUISITE_REPORT_NAME = PREREQUISITE_REPORT_NAME
            campaign.PREREQUISITE_REPORT_PATH_KEY = PREREQUISITE_REPORT_PATH_KEY
            campaign.PREREQUISITE_REPORT_SHA_KEY = PREREQUISITE_REPORT_SHA_KEY
            campaign.PREREQUISITE_REPORT_OUTPUT_KEY = PREREQUISITE_REPORT_OUTPUT_KEY
            campaign.EXECUTED_SOURCE_RELATIVE_PATHS = (
                *saved["EXECUTED_SOURCE_RELATIVE_PATHS"],
                *SOURCE_PATHS,
            )
            campaign.TOP_LEVEL_KEYS = frozenset(
                (set(saved["TOP_LEVEL_KEYS"]) - {"cutoff50_prerequisite_report_sha256"})
                | {PREREQUISITE_REPORT_OUTPUT_KEY}
            )
            campaign.load_campaign_contract = load_current
            yield
        finally:
            for name, value in saved.items():
                setattr(campaign, name, value)


def make_request(*, repository_root: Path, output: Path) -> campaign.CampaignRequest:
    return campaign.CampaignRequest(repository_root=repository_root, output=output)


def preflight(*, repository_root: Path, output: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        return campaign.preflight_campaign(make_request(repository_root=root, output=output))


def run(
    *,
    repository_root: Path,
    output: Path,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    if os.path.lexists(output if output.is_absolute() else root / output):
        raise FileExistsError("q200 campaign output exists")
    with _scope(root):
        return campaign.run_campaign(
            make_request(repository_root=root, output=output),
            progress=progress,
        )


def failure_report(
    error: BaseException,
    *,
    repository_root: Path,
    output: Path,
) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        return campaign.failure_report(
            error,
            make_request(repository_root=root, output=output),
        )


def write(
    report: Mapping[str, Any],
    *,
    repository_root: Path,
    output: Path,
) -> Path:
    root = repository_root.expanduser().resolve(strict=True)
    with _scope(root):
        return campaign.write_campaign_report_new(
            make_request(repository_root=root, output=output),
            report,
        )


__all__ = ["OVERLAY_SHA256", "failure_report", "load_overlay", "preflight", "run", "write"]
