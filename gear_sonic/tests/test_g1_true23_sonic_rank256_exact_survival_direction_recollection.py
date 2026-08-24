from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gear_sonic.scripts import recollect_g1_true23_sonic_rank256_exact_survival_direction as recollect


def test_recollection_contract_hash_and_boundaries() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / recollect.CONTRACT_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == recollect.CONTRACT_SHA256
    body = json.loads(path.read_text())
    assert body["boundaries"]["evaluation_runs"] == 0
    assert body["boundaries"]["optimizer_steps"] == 0
    assert body["boundaries"]["hardware_authorized"] is False
    assert len(body["expected_evidence"]["scaled_direction_state_sha256"]) == 64
