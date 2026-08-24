from __future__ import annotations

from pathlib import Path

import pytest

from gear_sonic.utils import g1_true23_sonic_scale2_clean_cross_seed_probe as probe

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_and_observation_probe_prerequisite_are_exact() -> None:
    contract = probe.load_contract(REPOSITORY_ROOT)
    probe._verify_prerequisite(REPOSITORY_ROOT, contract)  # noqa: SLF001
    assert [run["seed"] for run in contract["runs"]] == [20260805, 611723381]
    assert all(run["tokenizer_corruption_enabled"] is False for run in contract["runs"])


def test_request_output_is_repository_contained() -> None:
    request = probe.ProbeRequest(
        REPOSITORY_ROOT,
        Path("artifacts/g1_true23/scale2_clean_cross_seed_test.json"),
    )
    assert request.output_path == (REPOSITORY_ROOT / request.output).resolve()


def _mode(*, completed: int, terminal_q9: int) -> dict[str, object]:
    return {
        "completed_transitions": completed,
        "terminal_q9": terminal_q9,
        "termination_names": ["anchor_pos"],
        "nonfinite_count": 0,
        "q9_discontinuity_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": 0,
    }


def test_report_validation_accepts_clean_seed_sensitivity() -> None:
    report = {
        "schema_version": 1,
        "kind": probe.REPORT_KIND,
        "seed_sensitivity_proven": True,
        "failed_seed_clean_reproduced": True,
        "reference_clean_survived_beyond_failed_boundary": True,
        "reference_clean": _mode(completed=357, terminal_q9=365),
        "failed_seed_clean": _mode(completed=236, terminal_q9=244),
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
    report["seed_sensitivity_proven"] = False
    with pytest.raises(ValueError, match="recomputation"):
        probe.validate_report(report)


def test_report_validation_rejects_training_claim() -> None:
    report = {
        "schema_version": 1,
        "kind": probe.REPORT_KIND,
        "seed_sensitivity_proven": True,
        "failed_seed_clean_reproduced": True,
        "reference_clean_survived_beyond_failed_boundary": True,
        "reference_clean": _mode(completed=357, terminal_q9=365),
        "failed_seed_clean": _mode(completed=236, terminal_q9=244),
        "sources_unchanged": True,
        "training_performed": True,
        "optimizer_steps": 0,
        "teacher_queries": 0,
        "teacher_labels": 0,
        "support_qualified": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
        "network_or_external_actuation": False,
    }
    with pytest.raises(ValueError, match="boundary"):
        probe.validate_report(report)
