from __future__ import annotations

from pathlib import Path

import numpy as np

from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_multiseed_round3_bc_last_affine_ridge as fit,
)
from gear_sonic.utils.g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge import FitRequest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_and_weights_are_exact() -> None:
    contract = fit.load_contract(REPOSITORY_ROOT)
    weights = fit._weights(contract)  # noqa: SLF001
    assert weights.shape == (5610,)
    assert np.isclose(weights.mean(), 1.0, atol=1e-14)
    assert weights[4590] > weights[4690]
    assert weights[5100] > weights[5200]


def test_preflight_loads_all_admitted_rows_without_simulator(tmp_path: Path) -> None:
    request = FitRequest(REPOSITORY_ROOT, tmp_path / "round3")
    receipt = fit.preflight(request)
    assert receipt["ready"] is True
    assert receipt["total_rows"] == 5610
    assert receipt["simulator_constructed"] is False
    assert receipt["training_updates"] == 0
