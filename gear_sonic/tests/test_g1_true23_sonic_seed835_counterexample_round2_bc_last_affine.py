from __future__ import annotations

from pathlib import Path

import numpy as np

from gear_sonic.utils import g1_true23_sonic_seed835_counterexample_round2_bc_last_affine as fit
from gear_sonic.utils.g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge import FitRequest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_weights_round2_tail_highest() -> None:
    contract = fit.load_contract(REPOSITORY_ROOT)
    weights = fit._weights(contract)  # noqa: SLF001
    assert weights.shape == (5357,)
    assert np.isclose(weights.mean(), 1.0, atol=1e-14)
    assert weights[-1] > weights[-51]
    assert weights[-1] > weights[0]


def test_preflight_loads_both_counterexamples_without_sim(tmp_path: Path) -> None:
    receipt = fit.preflight(FitRequest(REPOSITORY_ROOT, tmp_path / "candidate"))
    assert receipt["ready"] is True
    assert receipt["total_rows"] == 5357
    assert receipt["round1_counterexample_rows"] == 89
    assert receipt["round2_counterexample_rows"] == 168
    assert receipt["simulator_constructed"] is False
