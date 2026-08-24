from __future__ import annotations

import copy
from pathlib import Path

import pytest

from gear_sonic.trl.mjlab import sonic_task_space_ppo_quarter_step_runner as quarter

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _evaluation(*, update: int, completed: int, q9: int, reward: float) -> dict[str, object]:
    return {
        "update_count": update,
        "completed_transitions": completed,
        "terminal_q9": q9,
        "termination_names": ["ee_body_pos"],
        "episode_return": reward,
        "nonfinite_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": 0,
        "q9_discontinuity_count": 0,
    }


def test_quarter_step_contract_is_hash_locked_and_exact() -> None:
    contract = quarter.load_quarter_step_contract(REPOSITORY_ROOT)
    assert contract["kind"] == quarter.CONTRACT_KIND
    assert contract["update"]["learning_rate"] == quarter.QUARTER_LEARNING_RATE
    assert contract["line_search_evidence"]["sha256"] == quarter.LINE_SEARCH_REPORT_SHA256
    assert contract["line_search_evidence"]["selected_probe"] == "alpha_plus_0_25"


def test_quarter_step_contract_rejects_semantic_drift() -> None:
    contract = copy.deepcopy(dict(quarter.load_quarter_step_contract(REPOSITORY_ROOT)))
    contract["update"]["learning_rate"] = 1.0e-5
    with pytest.raises(ValueError, match="semantic mismatch"):
        quarter.validate_quarter_step_contract(contract)


def test_quarter_step_candidate_requires_q164_and_clean_counts() -> None:
    baseline = _evaluation(update=0, completed=155, q9=163, reward=-123.0)
    unchanged = _evaluation(update=1, completed=155, q9=163, reward=-122.0)
    passed = _evaluation(update=1, completed=156, q9=164, reward=-122.0)
    assert quarter.assess_quarter_step_evaluations([baseline, unchanged])["candidate_selected"] is False
    result = quarter.assess_quarter_step_evaluations([baseline, passed])
    assert result["candidate_selected"] is True
    tampered = copy.deepcopy(passed)
    tampered["raw_clip_required_count"] = 1
    assert quarter.assess_quarter_step_evaluations([baseline, tampered])["candidate_selected"] is False


def test_quarter_step_candidate_enforces_paired_reward_floor() -> None:
    baseline = _evaluation(update=0, completed=155, q9=163, reward=-100.0)
    catastrophic = _evaluation(update=1, completed=156, q9=164, reward=-151.0)
    result = quarter.assess_quarter_step_evaluations([baseline, catastrophic])
    assert result["candidate_reward_floor"] == -150.0
    assert result["candidate_selected"] is False


def test_quarter_step_boundary_claims_remain_false() -> None:
    result = quarter.assess_quarter_step_evaluations(
        [
            _evaluation(update=0, completed=155, q9=163, reward=-100.0),
            _evaluation(update=1, completed=156, q9=164, reward=-99.0),
        ]
    )
    assert result["diagnostic_candidate_only"] is True
    for name in ("support_qualified", "promotion_eligible", "deployment_ready", "hardware_authorized"):
        assert result[name] is False
