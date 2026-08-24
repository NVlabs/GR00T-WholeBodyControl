from __future__ import annotations

import torch

from gear_sonic.scripts import (
    train_g1_true23_sonic_rank256_post_impulse_episodic_survival_score as episodic,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.linspace(260, 510, 128)
    active = torch.arange(510).unsqueeze(1) < lengths.to(torch.int64).unsqueeze(0)
    return lengths, active


def test_episodic_weights_exclude_pre_impulse_and_dead_robots() -> None:
    lengths, active = _fixture()
    weights, evidence = episodic._episodic_post_impulse_weights(lengths, active)
    first = episodic.parent.FIRST_RECOVERY_TRANSITION

    assert torch.count_nonzero(weights[:first]) == 0
    assert torch.count_nonzero(weights[first:]) > 0
    assert torch.count_nonzero(weights[~active]) == 0
    assert evidence[0]["pre_impulse_nonzero_weight_count"] == 0
    assert evidence[0]["first_recovery_q9"] == 251


def test_episodic_survivor_outcomes_are_centered() -> None:
    lengths, active = _fixture()
    weights, evidence = episodic._episodic_post_impulse_weights(lengths, active)
    first = episodic.parent.FIRST_RECOVERY_TRANSITION
    first_values = weights[first, active[first]]

    assert abs(float(first_values.mean())) < 1e-6
    assert evidence[0]["survivor_outcome_population_std"] >= episodic.parent.MIN_STD


def test_episodic_survivor_gate() -> None:
    lengths = torch.full((128,), 100.0)
    active = torch.arange(510).unsqueeze(1) < lengths.to(torch.int64).unsqueeze(0)
    try:
        episodic._episodic_post_impulse_weights(lengths, active)
    except RuntimeError as error:
        assert "survivor gate" in str(error)
    else:
        raise AssertionError("insufficient first-recovery population accepted")
