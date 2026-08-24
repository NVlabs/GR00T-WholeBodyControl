from __future__ import annotations

import copy
from pathlib import Path

import pytest

from gear_sonic.utils import g1_true23_selected_teacher_actuated_campaign as campaign

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_reuses_exact_twenty_seed_vector_cohort() -> None:
    contract = campaign.load_contract(REPOSITORY_ROOT)
    prerequisites = campaign._verify_prerequisites(REPOSITORY_ROOT, contract)  # noqa: SLF001
    specs = campaign.base_campaign.campaign_run_specs(REPOSITORY_ROOT)
    assert len(specs) == 20
    assert len({spec.seed for spec in specs}) == 20
    assert [spec.scenario for spec in specs[:4]] == ["nominal", "disturbance", "nominal", "disturbance"]
    assert contract["disturbance"]["global_apply_transition"] == 50
    assert contract["disturbance"]["apply_q9"] == 59
    assert prerequisites["q9"]["qualification"]["passed"] is True


def test_request_output_is_repository_contained() -> None:
    request = campaign.CampaignRequest(
        REPOSITORY_ROOT,
        Path("artifacts/g1_true23/teacher_campaign_test.json"),
    )
    assert request.output_path == (REPOSITORY_ROOT / request.output).resolve()


def _passing_probe(spec: object) -> dict[str, object]:
    scenario = getattr(spec, "scenario")
    return {
        "effective_seed": getattr(spec, "seed"),
        "step_calls_started": 510,
        "transitions_completed": 510,
        "metric_sample_count": 510,
        "metric_trajectory_sha256": "1" * 64,
        "impulse_count": 0 if scenario == "nominal" else 1,
        "impulse_apply_transition": None if scenario == "nominal" else 50,
        "impulse_apply_q9": None if scenario == "nominal" else 59,
        "impulse_qvel_write_exact": None if scenario == "nominal" else True,
        "impulse_readback_max_absolute_error": None if scenario == "nominal" else 0.0,
        "target_root_qvel_sha256": None if scenario == "nominal" else "2" * 64,
        "realized_root_qvel_sha256": None if scenario == "nominal" else "3" * 64,
        "baseline_mean": None if scenario == "nominal" else 0.2,
        "recovery_threshold": None if scenario == "nominal" else 0.3,
        "stable_streak_end_transition": None if scenario == "nominal" else 54,
        "recovery_time_s": None if scenario == "nominal" else 0.1,
        "recovered": None if scenario == "nominal" else True,
    }


def test_q9_pass_report_satisfies_nominal_run_assessor() -> None:
    contract = campaign.load_contract(REPOSITORY_ROOT)
    prerequisites = campaign._verify_prerequisites(REPOSITORY_ROOT, contract)  # noqa: SLF001
    spec = campaign.base_campaign.campaign_run_specs(REPOSITORY_ROOT)[0]
    report = copy.deepcopy(dict(prerequisites["q9"]))
    report["effective_seed"] = spec.seed
    with campaign._q9_campaign_constants():  # noqa: SLF001
        record = campaign.mixed_campaign._assess_run(  # noqa: SLF001
            spec,
            report,
            _passing_probe(spec),
            prerequisites["base_contract"],
        )
    assert record["passed"] is True
    assert record["student_controller_count"] == 0
    assert record["teacher_controller_count"] == 510


def test_q9_campaign_constants_are_scoped_and_recovery_uses_five_steps() -> None:
    contract = campaign.load_contract(REPOSITORY_ROOT)
    spec = campaign.base_campaign.campaign_run_specs(REPOSITORY_ROOT)[1]
    original = campaign.mixed_campaign.GLOBAL_IMPULSE_TRANSITION
    with campaign._q9_campaign_constants():  # noqa: SLF001
        assert campaign.mixed_campaign.GLOBAL_IMPULSE_TRANSITION == 50
        probe = campaign.mixed_campaign._StepProbe(spec, contract)  # noqa: SLF001
        probe.effective_seed = spec.seed
        probe.step_calls_started = 510
        probe.transitions_completed = 510
        probe.impulse_count = 1
        probe.impulse_apply_transition = 50
        probe.impulse_apply_q9 = 59
        probe.qvel_write_exact = True
        probe.readback_error = 0.0
        probe.metric_values = [0.2] * 510
        result = probe.report()
    assert campaign.mixed_campaign.GLOBAL_IMPULSE_TRANSITION == original
    assert result["recovered"] is True
    assert result["stable_streak_end_transition"] == 54
    assert result["recovery_time_s"] == pytest.approx(0.1)


def test_report_validation_recomputes_qualification() -> None:
    records = []
    for index in range(20):
        scenario = "nominal" if index % 2 == 0 else "disturbance"
        records.append(
            {"passed": True, "scenario": scenario, "recovered": True if scenario == "disturbance" else None}
        )
    report = {
        "schema_version": 1,
        "kind": campaign.REPORT_KIND,
        "campaign_qualified": True,
        "completed_run_count": 20,
        "runs": records,
        "sources_unchanged": True,
        "fresh_disjoint_teacher_collection_authorized": True,
        "teacher_support_qualified": True,
        "published_teacher_label_count": 0,
        "published_training_row_count": 0,
        "training_performed": False,
        "hardware_authorized": False,
    }
    campaign.validate_report(report)
    report["runs"][0]["passed"] = False
    with pytest.raises(ValueError, match="recomputation"):
        campaign.validate_report(report)
