from __future__ import annotations

import torch

from gear_sonic.scripts import (
    train_g1_true23_sonic_rank256_grouped_multiobjective_survival_score as grouped,
)


def test_group_weights_are_centered_and_equal_contribution() -> None:
    lengths = torch.cat(
        (
            torch.linspace(300, 500, 64),
            torch.linspace(100, 300, 64),
        )
    )
    steps = torch.arange(grouped.parent.COLLECTION_STEPS).unsqueeze(1)
    active = steps < lengths.to(torch.int64).unsqueeze(0)
    weights, evidence = grouped._group_standardized_weights(lengths, active)

    assert weights.shape == (510, 128)
    assert [entry["name"] for entry in evidence] == ["nominal", "exact_impulse"]
    contributions = [float(weights[:, start:stop].abs().sum()) for _name, start, stop in grouped.GROUPS]
    assert abs(contributions[0] - contributions[1]) / max(contributions) < 0.2
    assert torch.count_nonzero(weights[~active]) == 0


def test_group_weights_require_variance() -> None:
    lengths = torch.full((128,), 200.0)
    active = torch.arange(510).unsqueeze(1) < lengths.to(torch.int64).unsqueeze(0)
    try:
        grouped._group_standardized_weights(lengths, active)
    except RuntimeError as error:
        assert "variance insufficient" in str(error)
    else:
        raise AssertionError("constant outcome group accepted")
