from __future__ import annotations

import torch

from gear_sonic.scripts import train_g1_true23_sonic_rank256_iterative_grouped_survival_score as iterative


def test_shift_changes_only_declared_tensors() -> None:
    baseline = {"frozen": torch.tensor([3.0])}
    reward: dict[str, torch.Tensor] = {}
    for index, name in enumerate(iterative.DIRECTION_KEYS):
        baseline[name] = torch.tensor([float(index)])
        reward[name] = torch.tensor([0.25])
    shifted = iterative._shifted_state(baseline, reward)
    assert shifted["frozen"].item() == 3.0
    for name in iterative.DIRECTION_KEYS:
        assert torch.equal(shifted[name], baseline[name] + 0.25)


def _record(scale: float, scenario: str, steps: int) -> dict[str, object]:
    record: dict[str, object] = {
        "scale": scale,
        "scenario": scenario,
        "completed_transitions": steps,
        "policy_state_sha256": f"{scale}-{scenario}",
    }
    for name in iterative.grouped.parent.ZERO_COUNTS:
        record[name] = 0
    return record


def test_assessment_uses_original_absolute_gate() -> None:
    records = []
    for scale in iterative.SCALES:
        records.extend((_record(scale, "nominal", 483), _record(scale, "disturbance", 288)))
    records[2 * iterative.SCALES.index(0.25)]["completed_transitions"] = 486
    result = iterative._assess(records)
    assert result["candidate_selected"] is True
    assert result["selected_scale"] == 0.25
    assert result["selected_nominal_completed_transitions"] == 486
    assert result["selected_disturbance_completed_transitions"] == 288


def test_assessment_rejects_shifted_only_improvement() -> None:
    records = []
    for scale in iterative.SCALES:
        records.extend((_record(scale, "nominal", 485), _record(scale, "disturbance", 288)))
    assert iterative._assess(records)["candidate_selected"] is False
