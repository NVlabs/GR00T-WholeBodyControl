from __future__ import annotations

from pathlib import Path

from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge as base,
    g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge_v2 as fit,
    g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge as nominal_fit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v2_contract_binds_v1_rejection_and_five_percent_gate() -> None:
    contract = fit.load_contract(REPO_ROOT)
    assert contract["kind"] == fit.CONTRACT_KIND
    assert contract["gates"]["maximum_cutoff140_intervention_rmse_ratio_to_base"] == 0.95
    assert contract["inputs"]["rejected_v1_manifest"]["sha256"] == (
        "e29b565cdc8152223edeb763b76856855cd205fa63893a9604d685253e228f87"
    )
    assert nominal_fit.sha256_file(REPO_ROOT / fit.CONTRACT_RELATIVE_PATH) == fit.CONTRACT_SHA256


def test_version_scope_restores_v1_globals() -> None:
    before = (base.CONTRACT_KIND, base.MANIFEST_KIND, base.CONTRACT_RELATIVE_PATH, base.CONTRACT_SHA256)
    with fit._version_scope():  # noqa: SLF001
        assert base.CONTRACT_KIND == fit.CONTRACT_KIND
        assert base.MANIFEST_KIND == fit.MANIFEST_KIND
        assert base.CONTRACT_RELATIVE_PATH == fit.CONTRACT_RELATIVE_PATH
        assert base.CONTRACT_SHA256 == fit.CONTRACT_SHA256
    assert (base.CONTRACT_KIND, base.MANIFEST_KIND, base.CONTRACT_RELATIVE_PATH, base.CONTRACT_SHA256) == before


def test_real_v2_preflight_is_no_simulator() -> None:
    request = nominal_fit.FitRequest(REPO_ROOT, Path("artifacts/g1_true23/unused_round2_v2_test"))
    receipt = fit.preflight(request)
    assert receipt["ready"] is True
    assert receipt["contract_sha256"] == fit.CONTRACT_SHA256
    assert receipt["simulator_constructed"] is False
    assert receipt["training_updates"] == 0
