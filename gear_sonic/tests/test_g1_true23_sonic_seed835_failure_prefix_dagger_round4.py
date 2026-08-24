from __future__ import annotations

from pathlib import Path

import numpy as np

from gear_sonic.utils import g1_true23_sonic_seed835_failure_prefix_dagger_round4 as collector

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_binds_round4_failure_boundary() -> None:
    contract = collector.load_contract(REPOSITORY_ROOT)
    assert contract["rows"]["total"] == 486
    assert contract["rows"]["q9_last"] == 494


def test_scope_patches_generic_collector_and_restores() -> None:
    before = (collector.base.ROWS, collector.base.adapter)
    with collector._scope():  # noqa: SLF001
        assert collector.base.ROWS == 486
        assert collector.base.adapter is collector.adapter
    assert (collector.base.ROWS, collector.base.adapter) == before


def test_array_validator_exact_q9() -> None:
    widths = {
        "encoder267": 267,
        "token64": 64,
        "policy930": 930,
        "decoder994": 994,
        "student_raw_native23": 23,
        "selected_observation124": 124,
        "teacher_raw_hardware23": 23,
        "teacher_label_raw_native23": 23,
        "executed_raw_native23": 23,
        "executed_safe_native23": 23,
        "executed_final_target_hardware23": 23,
    }
    arrays = {name: np.zeros((486, width), dtype=np.float32) for name, width in widths.items()}
    arrays["q9"] = np.arange(9, 495, dtype=np.int64)
    collector._validate_arrays(arrays)  # noqa: SLF001
