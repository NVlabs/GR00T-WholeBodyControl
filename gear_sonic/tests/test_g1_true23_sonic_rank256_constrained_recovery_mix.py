from __future__ import annotations

import torch

from gear_sonic.scripts import screen_g1_true23_sonic_rank256_constrained_recovery_mix as mix


def test_mix_matrix_is_bounded_and_unique() -> None:
    names = [
        mix._name(grouped, causal, scenario)
        for grouped in mix.GROUPED_COEFFICIENTS
        for causal in mix.CAUSAL_COEFFICIENTS
        for scenario in mix.SCENARIOS
    ]
    assert len(names) == 18
    assert len(set(names)) == 18


def test_state_applies_all_three_directions() -> None:
    baseline = {"frozen": torch.tensor([9.0])}
    directions: dict[str, dict[str, torch.Tensor]] = {
        "reward_direction": {},
        "grouped_direction": {},
        "causal_direction": {},
    }
    for name in mix.DIRECTION_KEYS:
        baseline[name] = torch.tensor([1.0])
        directions["reward_direction"][name] = torch.tensor([1.0])
        directions["grouped_direction"][name] = torch.tensor([2.0])
        directions["causal_direction"][name] = torch.tensor([3.0])
    state = mix._state(baseline, directions, 0.5, 4.0)
    assert state["frozen"].item() == 9.0
    assert all(state[name].item() == 15.0 for name in mix.DIRECTION_KEYS)
