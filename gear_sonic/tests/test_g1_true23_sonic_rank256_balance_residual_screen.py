from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gear_sonic.scripts import screen_g1_true23_sonic_rank256_balance_residual as residual


def test_contract_hash_and_variants() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / residual.CONTRACT_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == residual.CONTRACT_SHA256
    body = json.loads(path.read_text())
    assert body["controller"]["variants"] == [list(value) for value in residual.VARIANTS]
    assert body["boundaries"]["training_transitions"] == 0
    assert body["boundaries"]["hardware_authorized"] is False


def test_pitch_indices_match_native_contract() -> None:
    assert residual.HIP_PITCH_INDICES == (0, 1)
    assert residual.ANKLE_PITCH_INDICES == (15, 16)


def test_evaluation_uses_changed_policy_slot_and_restores_seed() -> None:
    source = Path(residual.__file__).read_text()
    assert "original_seed = fs.FIXED_SEED" in source
    assert "update_count=1" in source
    assert "fs.FIXED_SEED = original_seed" in source


def test_impulse_proxy_delegates_wrapper_abi() -> None:
    class Wrapped:
        clip_actions = None
        max_episode_length = 510
        device = "cuda:0"

    proxy = residual.rank256._ImpulseProxy(Wrapped(), object())  # noqa: SLF001
    assert proxy.clip_actions is None
    assert proxy.max_episode_length == 510
    assert proxy.device == "cuda:0"
