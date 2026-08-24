from __future__ import annotations

from pathlib import Path

import numpy as np

from gear_sonic.utils import g1_true23_sonic_seed835_counterexample_round3_bc_last_affine as fit
from gear_sonic.utils.g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge import FitRequest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_weights_round3_tail_highest() -> None:
    contract = fit.load_contract(REPOSITORY_ROOT)
    weights = fit._weights(contract)  # noqa: SLF001
    assert weights.shape == (5823,)
    assert np.isclose(weights.mean(), 1.0, atol=1e-14)
    assert weights[-1] > weights[-51]
    assert weights[-1] > weights[0]


def test_preflight_loads_round3_counterexample_without_sim(tmp_path: Path) -> None:
    receipt = fit.preflight(FitRequest(REPOSITORY_ROOT, tmp_path / "candidate"))
    assert receipt["ready"] is True
    assert receipt["total_rows"] == 5823
    assert receipt["round3_counterexample_rows"] == 466
    assert receipt["simulator_constructed"] is False
