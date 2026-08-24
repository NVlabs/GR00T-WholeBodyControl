from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils import (
    g1_true23_native124_21204_bootstrap_mjlab as bootstrap,
    g1_true23_selected_teacher_nominal_multiseed_collection as collection,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_NPZ = (
    REPOSITORY_ROOT / "artifacts/g1_true23/native124_selected_21204_teacher_bootstrap_seed20260805_v1.npz"
)
BOOTSTRAP_MANIFEST = (
    REPOSITORY_ROOT
    / "artifacts/g1_true23/native124_selected_21204_teacher_bootstrap_seed20260805_v1.manifest.json"
)


def test_contract_and_prerequisites_pin_narrow_nominal_cohort() -> None:
    contract = collection.load_contract(REPOSITORY_ROOT)
    prerequisites = collection._verify_prerequisites(REPOSITORY_ROOT, contract)  # noqa: SLF001
    assert contract["cohort"]["nominal_seeds"] == list(collection.NOMINAL_SEEDS)
    assert contract["cohort"]["training_run_indices"] == list(collection.TRAINING_RUN_INDICES)
    assert contract["cohort"]["heldout_run_indices"] == list(collection.HELDOUT_RUN_INDICES)
    assert contract["admission"]["support_qualified"] is False
    assert len(prerequisites["nominal_specs"]) == 10


def test_request_stays_under_artifact_root() -> None:
    request = collection.CollectionRequest(
        REPOSITORY_ROOT,
        Path("artifacts/g1_true23/nominal_multiseed_test"),
    )
    assert request.npz_path == (REPOSITORY_ROOT / "artifacts/g1_true23/nominal_multiseed_test.npz").resolve()
    with pytest.raises(ValueError, match="stay under"):
        _ = collection.CollectionRequest(REPOSITORY_ROOT, Path("outside")).prefix


def test_runtime_seed_scope_restores_bootstrap_constant() -> None:
    original = bootstrap.FIXED_SEED
    with collection._runtime_seed_scope(611723381):  # noqa: SLF001
        assert bootstrap.FIXED_SEED == 611723381
    assert bootstrap.FIXED_SEED == original


def _valid_multiseed_arrays() -> dict[str, np.ndarray]:
    source, _ = bootstrap.load_bootstrap_training_candidate(
        BOOTSTRAP_NPZ,
        BOOTSTRAP_MANIFEST,
        repository_root=REPOSITORY_ROOT,
    )
    arrays = {
        name: np.concatenate([source[name].copy() for _ in collection.TRAINING_RUN_INDICES], axis=0)
        for name in bootstrap.ARRAY_SPECS
    }
    arrays["collection_run_index"] = np.repeat(
        np.arange(len(collection.TRAINING_RUN_INDICES), dtype=np.int64), bootstrap.TOTAL_ROWS
    )
    arrays["runtime_seed"] = np.concatenate(
        [
            np.full(bootstrap.TOTAL_ROWS, collection.NOMINAL_SEEDS[index], dtype=np.int64)
            for index in collection.TRAINING_RUN_INDICES
        ]
    )
    return arrays


def test_dataset_validator_reassesses_each_run_and_pins_lineage() -> None:
    arrays = _valid_multiseed_arrays()
    validated = collection._validate_dataset_arrays(arrays, REPOSITORY_ROOT)  # noqa: SLF001
    assert validated["decoder994"].shape == (collection.TRAINING_ROWS, 994)
    assert validated["teacher_label_raw_native23"].shape == (collection.TRAINING_ROWS, 23)
    assert validated["runtime_seed"][0] == collection.NOMINAL_SEEDS[0]
    arrays["runtime_seed"][0] += 1
    with pytest.raises(ValueError, match="lineage"):
        collection._validate_dataset_arrays(arrays, REPOSITORY_ROOT)  # noqa: SLF001


def test_json_safe_rejects_nonfinite_or_arrays() -> None:
    with pytest.raises(TypeError, match="unsupported"):
        collection._json_safe(np.zeros(1, dtype=np.float32))  # noqa: SLF001
    with pytest.raises(TypeError, match="unsupported"):
        collection._json_safe(float("nan"))  # noqa: SLF001
