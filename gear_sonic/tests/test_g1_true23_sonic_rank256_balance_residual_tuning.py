from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gear_sonic.scripts import screen_g1_true23_sonic_rank256_balance_residual_tuning as tuning


def test_tuning_contract_hash_and_proven_sign() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / tuning.CONTRACT_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == tuning.CONTRACT_SHA256
    body = json.loads(path.read_text())
    assert body["controller"]["signal"]["deadband"] == 0.3
    assert body["controller"]["maximum_applied_residual_abs"] == 1.0
    assert body["controller"]["variants"] == [list(value) for value in tuning.VARIANTS]
    assert all(ankle <= 0.0 and hip == 0.0 for _name, ankle, hip in tuning.VARIANTS)
    assert body["evaluation"]["meaningful_improvement_minimum_steps"] == 20
