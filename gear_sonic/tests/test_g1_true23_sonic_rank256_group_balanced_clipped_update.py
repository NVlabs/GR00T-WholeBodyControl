from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gear_sonic.scripts import train_g1_true23_sonic_rank256_group_balanced_clipped_update as update
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def test_contract_is_hash_locked_and_fail_closed() -> None:
    path = ROOT / update.CONTRACT_RELATIVE_PATH
    body = json.loads(path.read_text(encoding="utf-8"))
    assert sha256_file(path) == update.CONTRACT_SHA256
    assert body["collection"]["groups"] == [
        {"name": "nominal", "start": 0, "stop": 64, "impulse": False},
        {"name": "exact_impulse", "start": 64, "stop": 128, "impulse": True},
    ]
    assert body["optimizer"]["optimizer_steps"] == 4
    assert body["optimizer"]["critic_used"] is False
    assert body["optimizer"]["updated_parameters"] == list(update.fs.TRAINABLE_ACTOR_PARAMETERS)
    assert body["boundaries"]["hardware_authorized"] is False
    assert body["boundaries"]["robot_or_network_commands_permitted"] is False


def test_reward_to_go_stops_at_terminal_and_excludes_autoreset() -> None:
    rewards = torch.tensor([[1.0, 10.0], [2.0, 20.0], [100.0, 30.0]])
    dones = torch.tensor([[False, False], [True, False], [False, True]])
    active = torch.tensor([[True, True], [True, True], [False, True]])
    observed = update._reward_to_go(rewards, dones, active)  # noqa: SLF001
    expected = torch.tensor(
        [
            [1.0 + update.GAMMA * 2.0, 10.0 + update.GAMMA * (20.0 + update.GAMMA * 30.0)],
            [2.0, 20.0 + update.GAMMA * 30.0],
            [0.0, 30.0],
        ]
    )
    assert torch.allclose(observed, expected)


def test_group_advantages_are_separately_standardized() -> None:
    returns = torch.arange(256, dtype=torch.float32).reshape(2, 128)
    active = torch.ones((2, 128), dtype=torch.bool)
    advantages, evidence = update._standardized_group_advantages(returns, active)  # noqa: SLF001
    for start, stop in ((0, 64), (64, 128)):
        values = advantages[:, start:stop].reshape(-1)
        assert float(values.mean()) == pytest.approx(0.0, abs=1.0e-6)
        assert float(values.std(unbiased=False)) == pytest.approx(1.0, abs=1.0e-6)
    assert [row["name"] for row in evidence] == ["nominal", "exact_impulse"]
    assert [row["active_transition_count"] for row in evidence] == [128, 128]


def test_mixed_impulse_vectors_leave_nominal_half_zero() -> None:
    vectors = update._mixed_impulse_vectors(torch.device("cpu"))  # noqa: SLF001
    assert vectors.shape == (128, 6)
    assert torch.count_nonzero(vectors[:64]) == 0
    expected = torch.from_numpy(update.disturbance.FAILED_VECTOR)
    assert torch.equal(vectors[64:], expected.expand(64, -1))


def test_survivor_count_uses_q9_index_and_group_slice() -> None:
    active = torch.zeros((510, 128), dtype=torch.bool)
    active[241, 64:112] = True
    active[286, 64:80] = True
    assert update._survivor_count(active, 250, 64, 128) == 48  # noqa: SLF001
    assert update._survivor_count(active, 295, 64, 128) == 16  # noqa: SLF001
    with pytest.raises(ValueError, match="outside collection"):
        update._survivor_count(active, 519, 64, 128)  # noqa: SLF001


def test_preflight_is_read_only_when_bound_linux_inputs_exist() -> None:
    if not Path("/root/g1_true23_runs/sonic_rank256_constrained_recovery_mix_v1").is_dir():
        pytest.skip("bound WSL artifacts unavailable")
    report = update.preflight(ROOT)
    assert report["ready"] is True
    assert report["simulator_constructed"] is False
    assert report["training_transitions"] == 0
    assert report["optimizer_steps"] == 0
    assert report["hardware_authorized"] is False
