from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.scripts.fit_g1_true23_sonic_nominal_dagger_lowrank_last_block_bc import _parser
from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_lowrank_last_block_bc as fit,
    g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge as nominal_fit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_contract_split_and_weights_are_exact() -> None:
    contract = fit.load_contract(REPO_ROOT)
    optimization = fit.optimization_indices()
    heldout = fit.heldout_indices()
    assert optimization.shape == (4080,)
    assert heldout.shape == (1020,)
    assert np.intersect1d(optimization, heldout).size == 0
    assert np.array_equal(np.sort(np.concatenate((optimization, heldout))), np.arange(5100))
    weights = fit.optimization_weights(contract)
    assert float(weights.mean()) == pytest.approx(1.0)
    assert weights[0] / weights[10] == pytest.approx(4.0)
    assert weights[3060] / weights[3110] == pytest.approx(4.0)
    assert weights[3570] / weights[3710] == pytest.approx(4.0)


def test_formula_prediction_has_exact_shape_and_is_finite() -> None:
    rows = 3
    prediction = fit._formula_prediction(  # noqa: SLF001
        np.zeros((rows, 512), dtype=np.float32),
        np.zeros((rows, 8), dtype=np.float32),
        np.zeros((512, 8), dtype=np.float32),
        np.zeros(512, dtype=np.float32),
        np.zeros((23, 512), dtype=np.float32),
        np.arange(23, dtype=np.float32),
    )
    assert prediction.shape == (rows, 23)
    assert np.array_equal(prediction[0], np.arange(23, dtype=np.float32))


def test_fixed_rank_projection_is_exact_and_sign_canonical() -> None:
    rng = np.random.default_rng(7)
    hidden = rng.normal(size=(64, 12))
    projection = fit._fixed_rank_projection(hidden, 8)  # noqa: SLF001
    assert projection.rank == 8
    assert projection.basis.shape == (8, 12)
    for row in projection.basis:
        assert row[int(np.argmax(np.abs(row)))] >= 0.0


def test_real_preflight_binds_failure_without_simulator() -> None:
    request = nominal_fit.FitRequest(REPO_ROOT, Path("artifacts/g1_true23/unused_lowrank_bc_test"))
    receipt = fit.preflight(request)
    assert receipt["ready"] is True
    assert receipt["optimization_rows"] == 4080
    assert receipt["heldout_rows"] == 1020
    assert receipt["penultimate_delta_rank"] == 8
    assert receipt["simulator_constructed"] is False
    assert receipt["training_updates"] == 0


def test_cli_has_no_optimizer_or_device_overrides() -> None:
    help_text = _parser().format_help()
    for forbidden in ("--steps", "--batch", "--lr", "--rank", "--device", "--seed"):
        assert forbidden not in help_text
