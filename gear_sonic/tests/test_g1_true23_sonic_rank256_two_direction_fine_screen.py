from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gear_sonic.scripts import screen_g1_true23_sonic_rank256_two_directions_fine as fine


def test_fine_contract_hash_and_bracket() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / fine.CONTRACT_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fine.CONTRACT_SHA256
    body = json.loads(path.read_text())
    assert body["screen"]["coefficients"] == [list(value) for value in fine.COEFFICIENTS]
    assert (0.0, 1.0) in fine.COEFFICIENTS
    assert (0.5, 1.0) in fine.COEFFICIENTS
    assert (0.75, 1.0) in fine.COEFFICIENTS
    assert body["boundaries"]["training_transitions"] == 0
    assert body["boundaries"]["hardware_authorized"] is False
