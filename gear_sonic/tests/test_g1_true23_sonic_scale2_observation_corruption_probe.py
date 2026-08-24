from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils import g1_true23_sonic_scale2_observation_corruption_probe as probe

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_and_failed_campaign_prerequisite_are_exact() -> None:
    contract = probe.load_contract(REPOSITORY_ROOT)
    probe._verify_prerequisites(REPOSITORY_ROOT, contract)  # noqa: SLF001
    assert contract["seed"] == 611723381
    assert contract["modes"][0]["required_completed_transitions"] == 206
    assert contract["modes"][0]["required_terminal_q9"] == 214
    assert contract["modes"][1]["tokenizer_corruption_enabled"] is False


def test_request_output_is_repository_contained() -> None:
    request = probe.ProbeRequest(
        REPOSITORY_ROOT,
        Path("artifacts/g1_true23/scale2_corruption_probe_test.json"),
    )
    assert request.output_path == (REPOSITORY_ROOT / request.output).resolve()


def test_comparison_reports_first_q9_crossings() -> None:
    contract = probe.load_contract(REPOSITORY_ROOT)
    zero = np.zeros(23, dtype=np.float32)
    noisy = {
        "_actions": [zero, zero, zero],
        "_ee": [np.zeros(4, dtype=np.float32) for _ in range(3)],
    }
    clean = {
        "_actions": [zero, np.full(23, 0.02, dtype=np.float32), np.full(23, 0.2, dtype=np.float32)],
        "_ee": [
            np.zeros(4, dtype=np.float32),
            np.full(4, 0.002, dtype=np.float32),
            np.full(4, 0.03, dtype=np.float32),
        ],
    }
    result = probe._compare(noisy, clean, contract)  # noqa: SLF001
    assert result["common_last_q9"] == 11
    assert result["first_raw_action_linf_threshold_q9"]["0.01"] == 10
    assert result["first_raw_action_linf_threshold_q9"]["0.1"] == 11
    assert result["first_ee_z_linf_threshold_q9"]["0.001"] == 10
    assert result["first_ee_z_linf_threshold_q9"]["0.025"] == 11


def _mode(*, completed: int, terminal_q9: int, termination_names: list[str]) -> dict[str, object]:
    return {
        "completed_transitions": completed,
        "terminal_q9": terminal_q9,
        "termination_names": termination_names,
        "nonfinite_count": 0,
        "q9_discontinuity_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": 0,
    }


def test_report_validator_accepts_decisive_clean_pass_and_rejects_claim_drift() -> None:
    report = {
        "schema_version": 1,
        "kind": probe.REPORT_KIND,
        "noisy_reproduced_campaign_failure": True,
        "clean_completed_full_episode": True,
        "verdict": "observation_corruption_is_decisive_failed_seed_cause",
        "noisy": _mode(completed=206, terminal_q9=214, termination_names=["ee_body_pos"]),
        "clean": _mode(completed=510, terminal_q9=518, termination_names=["time_out"]),
        "sources_unchanged": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "teacher_queries": 0,
        "teacher_labels": 0,
        "support_qualified": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
        "network_or_external_actuation": False,
    }
    probe.validate_report(report)
    report["teacher_queries"] = 1
    with pytest.raises(ValueError, match="boundary"):
        probe.validate_report(report)


def test_report_validator_forbids_clean_run_without_noisy_reproduction() -> None:
    report = {
        "schema_version": 1,
        "kind": probe.REPORT_KIND,
        "noisy_reproduced_campaign_failure": False,
        "clean_completed_full_episode": True,
        "verdict": "observation_corruption_not_sufficiently_proven",
        "noisy": _mode(completed=205, terminal_q9=213, termination_names=["ee_body_pos"]),
        "clean": _mode(completed=510, terminal_q9=518, termination_names=["time_out"]),
        "sources_unchanged": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "teacher_queries": 0,
        "teacher_labels": 0,
        "support_qualified": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
        "network_or_external_actuation": False,
    }
    with pytest.raises(ValueError, match="without noisy reproduction"):
        probe.validate_report(report)
