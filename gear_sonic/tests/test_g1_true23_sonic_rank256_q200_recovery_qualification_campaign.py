from __future__ import annotations

from pathlib import Path

from gear_sonic.utils import (
    g1_true23_sonic_rank256_q200_recovery_qualification_campaign as q200,
)

ROOT = Path(__file__).resolve().parents[2]


def test_overlay_and_synthesized_contract_are_exact() -> None:
    overlay = q200.load_overlay(ROOT)
    contract = q200._synthesized_contract(ROOT, overlay)  # noqa: SLF001
    assert contract["window"] == {
        "mode": "cutoff191",
        "anchor_q9": 9,
        "last_action_q9": 518,
        "total_transitions": 510,
        "student_transition_count": 191,
        "student_first_q9": 9,
        "student_last_q9": 199,
        "teacher_transition_count": 319,
        "teacher_first_q9": 200,
        "teacher_last_q9": 518,
        "control_hz": 50.0,
        "final_timeout_only": True,
    }
    assert contract["disturbance"]["global_apply_transition"] == 241
    assert contract["disturbance"]["apply_q9"] == 250
    assert contract["disturbance"]["baseline_global_transitions"] == list(range(231, 241))
    assert contract["next_stage_boundary"]["collector_rows_per_successful_run"] == 319


def test_scope_reuses_exact_twenty_specs_and_q200_mode() -> None:
    with q200._scope(ROOT):  # noqa: SLF001
        specs = q200.campaign.campaign_run_specs(ROOT)
        contract = q200.campaign.load_campaign_contract(ROOT)
        assert len(specs) == 20
        assert len({spec.seed for spec in specs}) == 20
        assert q200.campaign.RECOVERY_MODE == "cutoff191"
        assert q200.campaign.STUDENT_TRANSITIONS == 191
        assert q200.campaign.TEACHER_TRANSITIONS == 319
        assert contract["prerequisites"]["candidate_decoder_sha256"].startswith("d1a20a09")


def test_preflight_is_no_simulator(monkeypatch, tmp_path: Path) -> None:
    sentinel = {
        "ready": True,
        "issues": [],
        "simulator_constructed": False,
        "published_teacher_label_count": 0,
        "published_training_row_count": 0,
    }
    monkeypatch.setattr(q200.campaign, "preflight_campaign", lambda _request: sentinel)
    observed = q200.preflight(repository_root=ROOT, output=tmp_path / "unused.json")
    assert observed is sentinel
    assert observed["simulator_constructed"] is False
