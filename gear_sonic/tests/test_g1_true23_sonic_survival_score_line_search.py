from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs
from gear_sonic.utils import g1_true23_sonic_survival_score_line_search as survival

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _record(scale: float, completed: int, q9: int, reward: float) -> dict[str, object]:
    return {
        "scale": scale,
        "completed_transitions": completed,
        "terminal_q9": q9,
        "termination_names": ["ee_body_pos"],
        "episode_return": reward,
        "policy_state_sha256": f"{int((scale + 2) * 10):064x}",
        "nonfinite_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": 0,
        "q9_discontinuity_count": 0,
    }


def _records() -> list[dict[str, object]]:
    return [_record(scale, 155, 163, -120.0) for scale in survival.SCALES]


def test_survival_contract_is_hash_locked_and_exact() -> None:
    contract = survival.load_survival_contract(REPOSITORY_ROOT)
    assert contract["kind"] == survival.CONTRACT_KIND
    assert contract["survival_score"]["direction_l2_target"] == survival.TARGET_DIRECTION_L2
    assert contract["survival_score"]["scales"] == list(survival.SCALES)


def test_survival_contract_rejects_gradient_scope_drift() -> None:
    contract = copy.deepcopy(dict(survival.load_survival_contract(REPOSITORY_ROOT)))
    contract["survival_score"]["gradient_batch_size"] = 256
    with pytest.raises(ValueError, match="semantic mismatch"):
        survival.validate_survival_contract(contract)


def test_first_episode_mask_excludes_every_autoreset_row() -> None:
    dones = torch.zeros((fs.NUM_STEPS_PER_ENV, fs.NUM_ENVS, 1), dtype=torch.uint8)
    dones[10, 0, 0] = 1
    dones[159, 1, 0] = 1
    active, success = survival.first_episode_mask_and_success(dones)
    assert bool(active[:11, 0].all()) is True
    assert bool(active[11:, 0].any()) is False
    assert bool(success[0]) is False
    assert bool(success[1]) is False
    assert int(success.sum()) == fs.NUM_ENVS - 2


def test_negative_gradient_direction_has_exact_predeclared_norm() -> None:
    gradients = {
        name: torch.full((index + 1,), float(index + 1), dtype=torch.float32)
        for index, name in enumerate(survival.STATE_PARAMETER_NAMES)
    }
    direction, evidence = survival.normalized_negative_gradient_direction(gradients)
    assert evidence["normalized_direction_l2"] == pytest.approx(survival.TARGET_DIRECTION_L2, abs=1e-10)
    for name in gradients:
        assert torch.all(torch.sign(direction[name]) == -torch.sign(gradients[name]))


def test_survival_assessment_selects_longest_clean_candidate() -> None:
    records = _records()
    baseline_index = list(survival.SCALES).index(0.0)
    records[baseline_index] = _record(0.0, 155, 163, -120.0)
    records[list(survival.SCALES).index(0.5)] = _record(0.5, 170, 178, -119.0)
    records[list(survival.SCALES).index(1.0)] = _record(1.0, 160, 168, -110.0)
    result = survival.assess_survival_evaluations(records)
    assert result["candidate_selected"] is True
    assert result["selected_scale"] == 0.5
    assert result["selected_completed_transitions"] == 170
    assert result["support_qualified"] is False


def test_survival_assessment_rejects_catastrophic_or_dirty_candidate() -> None:
    records = _records()
    baseline_index = list(survival.SCALES).index(0.0)
    records[baseline_index] = _record(0.0, 155, 163, -100.0)
    candidate_index = list(survival.SCALES).index(0.25)
    records[candidate_index] = _record(0.25, 170, 178, -151.0)
    assert survival.assess_survival_evaluations(records)["candidate_selected"] is False
    records[candidate_index] = _record(0.25, 170, 178, -99.0)
    records[candidate_index]["raw_clip_required_count"] = 1
    assert survival.assess_survival_evaluations(records)["candidate_selected"] is False
