from __future__ import annotations

import torch

from gear_sonic.scripts import train_g1_true23_sonic_rank256_post_impulse_reward_to_go_score as rtg


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.linspace(260, 510, 128).to(torch.int64)
    active = torch.arange(510).unsqueeze(1) < lengths.unsqueeze(0)
    rewards = torch.zeros((510, 128), dtype=torch.float32)
    for env in range(128):
        rewards[:, env] = torch.linspace(0.0, float(env) / 128.0, 510)
        rewards[lengths[env] - 1, env] -= 100.0
    return rewards, active


def test_reward_to_go_excludes_pre_impulse_and_dead_rows() -> None:
    rewards, active = _fixture()
    weights, proof = rtg._reward_to_go_weights(rewards, active)

    assert torch.count_nonzero(weights[: rtg.FIRST_CREDITED_TRANSITION]) == 0
    assert torch.count_nonzero(weights[rtg.FIRST_CREDITED_TRANSITION :]) > 0
    assert torch.count_nonzero(weights[~active]) == 0
    assert proof["pre_impulse_nonzero_weight_count"] == 0
    assert proof["first_credited_q9"] == 251


def test_reward_to_go_centers_each_valid_q9_population() -> None:
    rewards, active = _fixture()
    weights, proof = rtg._reward_to_go_weights(rewards, active)
    for transition in range(proof["first_valid"]["transition"], proof["last_valid"]["transition"] + 1):
        values = weights[transition, active[transition]]
        if torch.count_nonzero(values):
            assert abs(float(values.mean())) < 1e-6


def test_reward_to_go_survivor_gate() -> None:
    rewards = torch.zeros((510, 128))
    active = torch.arange(510).unsqueeze(1) < torch.full((1, 128), 100)
    try:
        rtg._reward_to_go_weights(rewards, active)
    except RuntimeError as error:
        assert "survivor gate" in str(error)
    else:
        raise AssertionError("insufficient q251 population accepted")
