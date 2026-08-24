from __future__ import annotations

import torch

from gear_sonic.scripts import screen_g1_true23_sonic_rank256_iterative_grouped as screen


def test_names_are_unique_and_stable() -> None:
    names = [screen._name(scale, scenario) for scale in screen.SCALES for scenario in screen.SCENARIOS]
    assert len(names) == 20
    assert len(set(names)) == 20
    assert screen._name(-0.5, "disturbance") == "scale_m0p5_disturbance.json"


def test_state_applies_one_screen_direction() -> None:
    baseline = {"frozen": torch.tensor([7.0])}
    direction: dict[str, torch.Tensor] = {}
    for name in screen.iterative.DIRECTION_KEYS:
        baseline[name] = torch.tensor([1.0])
        direction[name] = torch.tensor([0.5])
    state = screen._state(baseline, direction, -2.0)
    assert state["frozen"].item() == 7.0
    assert all(state[name].item() == 0.0 for name in screen.iterative.DIRECTION_KEYS)
