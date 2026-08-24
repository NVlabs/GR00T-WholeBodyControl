from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.scripts.fit_g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge import (
    _parser,
)
from gear_sonic.utils import g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge as fit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_contract_folds_and_weights_are_exact() -> None:
    contract = fit.load_contract(REPO_ROOT)
    assert fit.contract_sha256(REPO_ROOT) == fit.sha256_file(REPO_ROOT / fit.CONTRACT_RELATIVE_PATH)
    folds = fit.leave_one_run_out_folds()
    assert len(folds) == fit.RUN_COUNT
    all_validation = np.concatenate([validation for _, validation in folds])
    assert np.array_equal(all_validation, np.arange(fit.TOTAL_ROWS))
    for run, (training, validation) in enumerate(folds):
        assert validation[0] == run * fit.ROWS_PER_RUN
        assert validation[-1] == (run + 1) * fit.ROWS_PER_RUN - 1
        assert np.intersect1d(training, validation).size == 0
    weights = fit.row_weights(contract)
    assert weights.shape == (fit.TOTAL_ROWS,)
    assert float(weights.mean()) == pytest.approx(1.0)
    assert weights[0] / weights[fit.RESET_PREFIX] == pytest.approx(4.0)


def test_weighted_ridge_is_finite_and_bias_is_regularized() -> None:
    rows = 32
    scores = np.stack((np.linspace(-1.0, 1.0, rows), np.cos(np.arange(rows))), axis=1)
    residual = np.zeros((rows, fit.ACTION_DIM), dtype=np.float64)
    residual[:, 0] = 2.0 * scores[:, 0] - 0.4 * scores[:, 1] + 0.7
    coefficient, intercept, condition = fit._weighted_ridge(  # noqa: SLF001
        scores, residual, np.linspace(0.5, 1.5, rows), 1e-6
    )
    assert coefficient.shape == (fit.ACTION_DIM, 2)
    assert intercept.shape == (fit.ACTION_DIM,)
    assert np.isfinite(condition)
    assert coefficient[0, 0] == pytest.approx(2.0, abs=1e-4)
    assert intercept[0] == pytest.approx(0.7, abs=1e-4)


def test_real_preflight_admits_exact_dataset_without_simulator() -> None:
    request = fit.FitRequest(
        REPO_ROOT,
        Path("artifacts/g1_true23/unused_nominal_multiseed_bc_test"),
    )
    report = fit.preflight(request)
    assert report["ready"] is True
    assert report["dataset_rows"] == 4080
    assert report["bootstrap_rows"] == 510
    assert report["heldout_labels_used"] is False
    assert report["simulator_constructed"] is False
    assert report["training_updates"] == 0


def test_publication_is_exclusive_and_payload_hashed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    evidence = root / "artifacts" / "g1_true23"
    evidence.mkdir(parents=True)
    request = fit.FitRequest(root, Path("artifacts/g1_true23/candidate"))
    outcome = fit.FitOutcome(
        passed=True,
        decoder_bytes=b"candidate",
        report={"export": {"candidate_decoder_sha256": fit._sha256_bytes(b"candidate")}},  # noqa: SLF001
    )
    decoder, manifest, body = fit.publish(request, outcome)
    assert decoder is not None and decoder.read_bytes() == b"candidate"
    written = json.loads(manifest.read_text(encoding="utf-8"))
    unhashed = dict(written)
    claimed = unhashed.pop("manifest_payload_sha256")
    assert claimed == fit._sha256_bytes(fit._canonical_bytes(unhashed))  # noqa: SLF001
    assert body["artifact"]["overwrite_permitted"] is False
    with pytest.raises(FileExistsError):
        fit.publish(request, outcome)


def test_cli_has_no_hyperparameter_or_device_overrides() -> None:
    parser = _parser()
    help_text = parser.format_help()
    for forbidden in ("--lambda", "--weight", "--rank", "--device", "--seed"):
        assert forbidden not in help_text
