from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gear_sonic.trl.mjlab.sonic_recovery_blend_policy import (
    BLEND_DURATION_STEPS,
    BLEND_START_Q9,
    SonicRecoveryBlendPolicy,
    smooth_blend_alpha,
)


def test_smooth_blend_alpha_exact_boundaries() -> None:
    q9 = torch.tensor(
        [BLEND_START_Q9 - 1, BLEND_START_Q9, BLEND_START_Q9 + 5, BLEND_START_Q9 + 10, 999],
        dtype=torch.long,
    )
    result = smooth_blend_alpha(q9)
    assert result.shape == (5, 1)
    assert torch.equal(result[:, 0], torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


@pytest.mark.parametrize(
    ("q9", "start", "duration"),
    [
        (torch.zeros((1, 1), dtype=torch.long), BLEND_START_Q9, BLEND_DURATION_STEPS),
        (torch.zeros(1, dtype=torch.float32), BLEND_START_Q9, BLEND_DURATION_STEPS),
        (torch.zeros(1, dtype=torch.long), True, BLEND_DURATION_STEPS),
        (torch.zeros(1, dtype=torch.long), BLEND_START_Q9, 0),
    ],
)
def test_smooth_blend_alpha_rejects_bad_schema(
    q9: torch.Tensor,
    start: int,
    duration: int,
) -> None:
    with pytest.raises(ValueError):
        smooth_blend_alpha(q9, start_q9=start, duration_steps=duration)


class _ConstantActor(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, observations: object, stochastic_output: bool = False) -> torch.Tensor:
        del observations
        assert stochastic_output is False
        return torch.full((3, 23), self.value, dtype=torch.float32)


def test_recovery_blend_policy_uses_one_causal_smooth_crossfade() -> None:
    command = SimpleNamespace(time_steps=torch.tensor([359, 365, 370], dtype=torch.long))
    raw_env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    policy = SonicRecoveryBlendPolicy(_ConstantActor(2.0), _ConstantActor(6.0), raw_env)
    output = policy({}, stochastic_output=False)
    assert output.shape == (3, 23)
    assert torch.equal(output[:, 0], torch.tensor([2.0, 4.0, 6.0]))
    with pytest.raises(ValueError, match="deterministic"):
        policy({}, stochastic_output=True)
    with pytest.raises(RuntimeError, match="distillation"):
        policy.export_true23_policy_state()


def test_recovery_blend_policy_rejects_nonfinite_action() -> None:
    command = SimpleNamespace(time_steps=torch.tensor([360, 360, 360], dtype=torch.long))
    raw_env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    policy = SonicRecoveryBlendPolicy(_ConstantActor(float("nan")), _ConstantActor(0.0), raw_env)
    with pytest.raises(RuntimeError, match="non-finite"):
        policy({})
