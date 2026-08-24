from __future__ import annotations

import torch

from gear_sonic.scripts import screen_g1_true23_sonic_rank256_causal_recovery as screen


def test_screen_names_are_unique() -> None:
    names = [screen._name(scale, scenario) for scale in screen.SCALES for scenario in screen.SCENARIOS]
    assert len(names) == 22
    assert len(set(names)) == 22


def test_state_changes_only_direction_tensors() -> None:
    baseline = {"frozen": torch.tensor([5.0])}
    direction: dict[str, torch.Tensor] = {}
    for name in screen.DIRECTION_KEYS:
        baseline[name] = torch.tensor([1.0])
        direction[name] = torch.tensor([0.25])
    state = screen._state(baseline, direction, 4.0)
    assert state["frozen"].item() == 5.0
    assert all(state[name].item() == 2.0 for name in screen.DIRECTION_KEYS)


def _record(scale: float, scenario: str, steps: int) -> dict[str, object]:
    record: dict[str, object] = {
        "scale": scale,
        "scenario": scenario,
        "completed_transitions": steps,
        "policy_state_sha256": f"{scale}-{scenario}",
    }
    for name in screen.collector.v1.grouped.parent.ZERO_COUNTS:
        record[name] = 0
    return record


def test_assessment_requires_absolute_joint_gate() -> None:
    records = []
    for scale in screen.SCALES:
        records.extend((_record(scale, "nominal", 485), _record(scale, "disturbance", 288)))
    target = screen.SCALES.index(0.5) * 2
    records[target]["completed_transitions"] = 486
    result = screen._assess(records)
    assert result["candidate_selected"] is True
    assert result["selected_scale"] == 0.5
