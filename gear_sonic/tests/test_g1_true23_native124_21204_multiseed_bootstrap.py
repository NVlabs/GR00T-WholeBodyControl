from __future__ import annotations

from pathlib import Path

import pytest

from gear_sonic.utils import (
    g1_true23_native124_21204_bootstrap_mjlab as base,
    g1_true23_native124_21204_multiseed_bootstrap as multiseed,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_binds_three_heldout_teacher_seeds() -> None:
    contract = multiseed.load_contract(REPOSITORY_ROOT)
    assert contract["runtime"]["allowed_seeds"] == list(multiseed.ALLOWED_SEEDS)
    assert contract["runtime"]["rows_per_seed"] == 510
    assert contract["runtime"]["teacher_controls_every_transition"] is True


def test_collection_scope_sets_seed_sources_and_restores() -> None:
    seed_before = base.FIXED_SEED
    sources_before = base.EXECUTED_SOURCE_FILE_RELATIVE_PATHS
    with multiseed.collection_scope(REPOSITORY_ROOT, 835868017):
        assert base.FIXED_SEED == 835868017
        assert set(multiseed.SOURCE_PATHS).issubset(base.EXECUTED_SOURCE_FILE_RELATIVE_PATHS)
    assert base.FIXED_SEED == seed_before
    assert base.EXECUTED_SOURCE_FILE_RELATIVE_PATHS == sources_before


def test_collection_scope_rejects_unsealed_seed() -> None:
    with pytest.raises(ValueError, match="seed is not sealed"):
        with multiseed.collection_scope(REPOSITORY_ROOT, 123):
            pass
