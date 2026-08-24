from __future__ import annotations

from pathlib import Path

import pytest

from gear_sonic.scripts.fit_g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge import (
    _parser,
)
from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge as fit,
    g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge as nominal_fit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_contract_weights_and_lineage_are_exact() -> None:
    contract = fit.load_contract(REPO_ROOT)
    weights = fit._weights(contract)  # noqa: SLF001
    assert weights.shape == (5100,)
    assert float(weights.mean()) == pytest.approx(1.0)
    assert weights[0] / weights[10] == pytest.approx(4.0)
    assert weights[4080] / weights[4130] == pytest.approx(4.0)
    assert weights[4590] / weights[4730] == pytest.approx(4.0)
    assert contract["inputs"]["cutoff140_npz"]["sha256"] == (
        "1479bb7b17bd3137ba7dd811b447938b0100aae93ced1339983e1e5162933a3d"
    )
    assert nominal_fit.sha256_file(REPO_ROOT / fit.CONTRACT_RELATIVE_PATH) == fit.CONTRACT_SHA256


def test_real_preflight_validates_both_interventions_without_simulator() -> None:
    request = nominal_fit.FitRequest(
        REPO_ROOT,
        Path("artifacts/g1_true23/unused_round2_bc_test"),
    )
    receipt = fit.preflight(request)
    assert receipt["ready"] is True
    assert receipt["nominal_rows"] == 4080
    assert receipt["cutoff50_rows"] == 510
    assert receipt["cutoff140_rows"] == 510
    assert receipt["student_on_policy_shadow_rows"] == 190
    assert receipt["teacher_actuated_recovery_rows"] == 830
    assert receipt["simulator_constructed"] is False
    assert receipt["training_updates"] == 0


def test_cli_has_no_fit_or_device_overrides() -> None:
    help_text = _parser().format_help()
    for forbidden in ("--lambda", "--weight", "--rank", "--device", "--seed"):
        assert forbidden not in help_text
