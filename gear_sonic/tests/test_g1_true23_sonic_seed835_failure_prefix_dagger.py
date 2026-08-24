from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils import g1_true23_sonic_seed835_failure_prefix_dagger as collector

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_binds_exact_failure_prefix() -> None:
    contract = collector.load_contract(REPOSITORY_ROOT)
    assert contract["rows"]["total"] == 89
    assert contract["rows"]["q9_last"] == 97
    assert contract["admission"]["expected_controller_counts"] == {"student": 89, "teacher": 0}


def test_array_validator_accepts_only_exact_student_prefix() -> None:
    arrays = {
        "encoder267": np.zeros((89, 267), dtype=np.float32),
        "token64": np.zeros((89, 64), dtype=np.float32),
        "policy930": np.zeros((89, 930), dtype=np.float32),
        "decoder994": np.zeros((89, 994), dtype=np.float32),
        "student_raw_native23": np.zeros((89, 23), dtype=np.float32),
        "selected_observation124": np.zeros((89, 124), dtype=np.float32),
        "teacher_raw_hardware23": np.zeros((89, 23), dtype=np.float32),
        "teacher_label_raw_native23": np.zeros((89, 23), dtype=np.float32),
        "executed_raw_native23": np.zeros((89, 23), dtype=np.float32),
        "executed_safe_native23": np.zeros((89, 23), dtype=np.float32),
        "executed_final_target_hardware23": np.zeros((89, 23), dtype=np.float32),
        "q9": np.arange(9, 98, dtype=np.int64),
    }
    collector._validate_arrays(arrays)  # noqa: SLF001
    arrays["q9"][-1] = 96
    with pytest.raises(ValueError, match="q9 mismatch"):
        collector._validate_arrays(arrays)  # noqa: SLF001
