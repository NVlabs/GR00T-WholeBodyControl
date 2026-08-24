from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.scripts.fit_g1_true23_sonic_multimotion_head import (
    ALPHAS,
    VALIDATION_TAIL_ROWS,
    _alpha_name,
    select_ridge,
)


def test_alpha_names_are_stable_and_native23_candidates_bounded() -> None:
    assert tuple(_alpha_name(value) for value in ALPHAS) == (
        "alpha010",
        "alpha025",
        "alpha050",
        "alpha075",
        "alpha100",
    )
    with pytest.raises(ValueError, match="unsupported"):
        _alpha_name(0.33)


def test_select_ridge_uses_tail_from_every_motion() -> None:
    random = np.random.default_rng(7)
    lengths = (VALIDATION_TAIL_ROWS + 20,) * 3
    hidden = random.normal(size=(sum(lengths), 8)).astype(np.float32)
    coefficients = random.normal(size=(23, 8)).astype(np.float32)
    residual = hidden @ coefficients.T
    ridge, grid = select_ridge(hidden, residual, lengths)
    assert ridge in {item["ridge"] for item in grid}
    assert len(grid) == 7
    assert min(item["validation_rmse"] for item in grid) < 0.01


def test_select_ridge_rejects_short_or_inconsistent_groups() -> None:
    hidden = np.zeros((303, 4), dtype=np.float32)
    residual = np.zeros((303, 23), dtype=np.float32)
    with pytest.raises(ValueError, match="group lengths"):
        select_ridge(hidden, residual, (101, 101))
    with pytest.raises(ValueError, match="group lengths"):
        select_ridge(hidden, residual, (100, 101, 102))
