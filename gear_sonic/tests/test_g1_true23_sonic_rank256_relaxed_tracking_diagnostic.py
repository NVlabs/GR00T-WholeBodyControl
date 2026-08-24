from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gear_sonic.scripts import diagnose_g1_true23_sonic_rank256_relaxed_tracking as diagnostic


def test_contract_hash_and_boundaries() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / diagnostic.CONTRACT_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == diagnostic.CONTRACT_SHA256
    body = json.loads(path.read_text())
    assert body["tracking_abort_override"]["disabled_only"] == list(diagnostic.DISABLED_TERMINATIONS)
    assert body["tracking_abort_override"]["preserved"] == list(diagnostic.PRESERVED_TERMINATIONS)
    assert body["boundaries"]["policy_mutations"] == 0
    assert body["boundaries"]["hardware_authorized"] is False


def test_exclusive_writer(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    diagnostic._write_json_exclusive(target, {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True}
    try:
        diagnostic._write_json_exclusive(target, {"ok": False})
    except FileExistsError:
        pass
    else:
        raise AssertionError("relaxed diagnostic overwrote evidence")
