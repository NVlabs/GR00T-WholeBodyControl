from __future__ import annotations

import torch

from gear_sonic.scripts import train_g1_true23_sonic_rank256_post_impulse_survival_score as post


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.linspace(260, 510, 128)
    active = torch.arange(510).unsqueeze(1) < lengths.to(torch.int64).unsqueeze(0)
    return lengths, active


def test_weights_exclude_every_pre_impulse_action() -> None:
    lengths, active = _fixture()
    weights, evidence = post._post_impulse_weights(lengths, active)

    assert weights.shape == (510, 128)
    assert torch.count_nonzero(weights[: post.FIRST_RECOVERY_TRANSITION]) == 0
    assert torch.count_nonzero(weights[post.FIRST_RECOVERY_TRANSITION :]) > 0
    assert torch.count_nonzero(weights[~active]) == 0
    assert evidence[0]["pre_impulse_nonzero_weight_count"] == 0
    assert evidence[0]["first_recovery_q9"] == 251


def test_weights_center_each_valid_transition() -> None:
    lengths, active = _fixture()
    weights, evidence = post._post_impulse_weights(lengths, active)
    for transition in range(evidence[0]["first_valid_transition"], evidence[0]["last_valid_transition"] + 1):
        values = weights[transition, active[transition]]
        if torch.count_nonzero(values):
            assert abs(float(values.mean())) < 1e-6


def test_first_recovery_survivor_gate() -> None:
    lengths = torch.full((128,), 100.0)
    active = torch.arange(510).unsqueeze(1) < lengths.to(torch.int64).unsqueeze(0)
    try:
        post._post_impulse_weights(lengths, active)
    except RuntimeError as error:
        assert "survivor gate" in str(error)
    else:
        raise AssertionError("insufficient recovery survivors accepted")
