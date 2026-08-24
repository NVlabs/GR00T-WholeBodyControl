from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_PROFILE,
    causal_history_lower_body,
    causal_history_profile_contract,
)


class _CommandManager:
    def __init__(self, command: object) -> None:
        self.command = command

    def get_term(self, name: str) -> object:
        assert name == "motion"
        return self.command


def _fake_env(anchor: int = 9) -> SimpleNamespace:
    frames = torch.arange(30, dtype=torch.float32)[:, None, None]
    joints = torch.arange(23, dtype=torch.float32)[None, None, :]
    joint_pos = frames * 100.0 + joints
    command = SimpleNamespace(
        time_steps=torch.tensor([anchor], dtype=torch.long),
        motion=SimpleNamespace(
            joint_pos=joint_pos[:, 0],
            time_step_total=joint_pos.shape[0],
        ),
    )
    return SimpleNamespace(
        num_envs=1,
        command_manager=_CommandManager(command),
    )


def test_causal_profile_is_distinct_and_hash_bound() -> None:
    contract = causal_history_profile_contract()
    assert contract["profile"] == CAUSAL_HISTORY_PROFILE
    assert contract["position_offsets_from_anchor_s"] == pytest.approx(
        [-0.18, -0.16, -0.14, -0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0]
    )
    assert contract["anchor_frame"] == "q9"
    assert contract["proof_frame"] == "q10"
    assert contract["control_and_proprioception_frame"] == "q10_current"
    assert contract["future_samples_relative_to_emission"] is False
    assert contract["released_profile_relabel_permitted"] is False
    assert contract["contract_sha256"] == (
        "e25aa962368c6dc8022d7574716f95c77f632fd255a7d010824ee5edc762669c"
    )


def test_causal_lower_body_uses_q0_through_q10_proof() -> None:
    result = causal_history_lower_body(_fake_env(anchor=9))
    assert result.shape == (1, 240)
    position = result[0, :120].reshape(10, 12)
    velocity = result[0, 120:].reshape(10, 12)
    expected_position = torch.stack(
        [torch.arange(12, dtype=torch.float32) + 100.0 * frame for frame in range(10)]
    )
    torch.testing.assert_close(position, expected_position)
    torch.testing.assert_close(velocity, torch.full((10, 12), 5000.0))


def test_causal_lower_body_rejects_missing_oldest_or_proof() -> None:
    with pytest.raises(ValueError, match="q0 oldest"):
        causal_history_lower_body(_fake_env(anchor=8))
    with pytest.raises(ValueError, match="q10 forward-difference proof"):
        causal_history_lower_body(_fake_env(anchor=29))
