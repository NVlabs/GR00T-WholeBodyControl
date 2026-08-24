from __future__ import annotations

from pathlib import Path

import numpy as np

from gear_sonic.utils import g1_true23_sonic_seed835_counterexample_round3_rank256_bc_last_affine as fit
from gear_sonic.utils.g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge import FitRequest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_pins_exact_rank256() -> None:
    contract = fit.load_contract(REPOSITORY_ROOT)
    assert contract["fit"]["pca_exact_rank"] == 256
    weights = fit._weights(contract)  # noqa: SLF001
    assert weights.shape == (5823,)
    assert np.isclose(weights.mean(), 1.0, atol=1e-14)


def test_preflight_binds_rejected_rank42_without_sim(tmp_path: Path) -> None:
    receipt = fit.preflight(FitRequest(REPOSITORY_ROOT, tmp_path / "candidate"))
    assert receipt["ready"] is True
    assert receipt["pca_exact_rank"] == 256
    assert receipt["simulator_constructed"] is False
