from __future__ import annotations

from pathlib import Path

import pytest

from gear_sonic.utils import (
    g1_true23_native124_selected_source_nominal_qualification as nominal,
    g1_true23_selected_teacher_seed835_nominal_diagnostic as diagnostic,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_seed_scope_is_exact_and_restored() -> None:
    before = nominal.FIXED_SEED
    with diagnostic._seed_scope():  # noqa: SLF001
        assert nominal.FIXED_SEED == diagnostic.RUNTIME_SEED
    assert nominal.FIXED_SEED == before


def test_parent_failure_is_hash_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostic, "PARENT_FAILURE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="failure evidence drift"):
        diagnostic.run(repository_root=REPOSITORY_ROOT, output=Path("unused.json"))
