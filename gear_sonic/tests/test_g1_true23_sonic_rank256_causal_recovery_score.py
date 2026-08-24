from __future__ import annotations

import torch

from gear_sonic.scripts import collect_g1_true23_sonic_rank256_causal_recovery_score as causal


def test_causal_score_rewards_small_reference_relative_error() -> None:
    zero = torch.zeros(2)
    score, terms = causal._causal_score(
        anchor_pos_error=torch.tensor([0.0, 0.25]),
        anchor_ori_error=zero,
        anchor_lin_vel_error=zero,
        anchor_ang_vel_error=zero,
        non_timeout_terminal=torch.tensor([False, True]),
    )
    assert score.tolist() == [0.0, -52.0]
    assert terms["anchor_position"].tolist() == [0.0, 2.0]


def test_return_weights_credit_only_observable_response() -> None:
    steps, envs = causal.grouped.parent.COLLECTION_STEPS, causal.grouped.parent.NUM_ENVS
    scores = torch.zeros((steps, envs))
    active = torch.ones((steps, envs), dtype=torch.bool)
    generator = torch.Generator().manual_seed(17)
    scores[causal.FIRST_CREDITED_TRANSITION :] = torch.randn(
        (steps - causal.FIRST_CREDITED_TRANSITION, envs), generator=generator
    )
    weights, proof = causal._return_weights(scores, active)
    assert torch.count_nonzero(weights[: causal.FIRST_CREDITED_TRANSITION]) == 0
    assert torch.count_nonzero(weights[causal.FIRST_CREDITED_TRANSITION :]) > 0
    assert proof["first_credited_q9"] == 251
    assert [entry["name"] for entry in proof["groups"]] == ["nominal", "exact_impulse"]


def test_return_weights_require_group_variance() -> None:
    steps, envs = causal.grouped.parent.COLLECTION_STEPS, causal.grouped.parent.NUM_ENVS
    scores = torch.zeros((steps, envs))
    active = torch.ones((steps, envs), dtype=torch.bool)
    try:
        causal._return_weights(scores, active)
    except RuntimeError as error:
        assert "valid transition gate" in str(error)
    else:
        raise AssertionError("zero-variance causal returns accepted")


def test_group_specific_valid_threshold_is_reported(monkeypatch) -> None:
    steps, envs = causal.grouped.parent.COLLECTION_STEPS, causal.grouped.parent.NUM_ENVS
    generator = torch.Generator().manual_seed(23)
    scores = torch.randn((steps, envs), generator=generator)
    active = torch.ones((steps, envs), dtype=torch.bool)
    monkeypatch.setattr(causal, "MIN_VALID_BY_GROUP", {"nominal": 100, "exact_impulse": 30})
    _weights, proof = causal._return_weights(scores, active)
    assert [entry["required_valid_transition_count"] for entry in proof["groups"]] == [100, 30]
