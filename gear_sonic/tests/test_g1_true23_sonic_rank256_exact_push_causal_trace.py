from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gear_sonic.scripts import trace_g1_true23_sonic_rank256_exact_push as causal


def test_contract_hash_and_boundaries() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / causal.CONTRACT_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == causal.CONTRACT_SHA256
    body = json.loads(path.read_text())
    assert body["scenario"]["impulse_transition"] == 241
    assert body["scenario"]["impulse_q9"] == 250
    assert body["boundaries"]["training_transitions"] == 0
    assert body["boundaries"]["hardware_authorized"] is False


def test_first_q9() -> None:
    assert causal._first_q9([9, 10, 11], [0.0, 0.2, 0.5], 0.2) == 10  # noqa: SLF001
    assert causal._first_q9([9], [0.1], 0.2) is None  # noqa: SLF001


def test_native_joint_names_are_exact_23() -> None:
    assert len(causal.NATIVE_JOINT_NAMES) == 23
    assert len(set(causal.NATIVE_JOINT_NAMES)) == 23
