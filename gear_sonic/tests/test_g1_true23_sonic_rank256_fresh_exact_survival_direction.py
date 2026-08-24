from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gear_sonic.scripts import collect_g1_true23_sonic_rank256_fresh_exact_survival_direction as fresh


def test_fresh_direction_contract_and_boundaries() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / fresh.CONTRACT_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fresh.CONTRACT_SHA256
    body = json.loads(path.read_text())
    assert body["collection"]["historical_byte_reproduction_required"] is False
    assert body["collection"]["fresh_direction_identity_required"] is True
    assert body["boundaries"]["evaluation_runs"] == 0
    assert body["boundaries"]["hardware_authorized"] is False
