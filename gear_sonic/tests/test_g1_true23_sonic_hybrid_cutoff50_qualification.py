from __future__ import annotations

import json
from pathlib import Path

import pytest

from gear_sonic.utils import (
    g1_true23_sonic_hybrid_cutoff50_qualification as qualification,
    g1_true23_sonic_student_teacher_recovery as recovery,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_binds_cutoff100_decision_and_cutoff50_partition() -> None:
    contract = qualification.load_contract(REPOSITORY_ROOT)
    assert contract["runtime"]["mode"] == "cutoff50"
    assert contract["runtime"]["student_transitions"] == 50
    assert contract["runtime"]["teacher_transitions"] == 460
    assert contract["decision"]["heldout835_cutoff100_terminal_q9"] == 97
    assert contract["decision"]["stop_if_cutoff50_fails_heldout835"] is True


def test_request_is_exact_cutoff50() -> None:
    request = qualification._request(REPOSITORY_ROOT, Path("unused.json"))  # noqa: SLF001
    assert request.mode == "cutoff50"
    assert request.window == recovery.resolve_recovery_window("cutoff50")
    assert request.window.student_transitions == 50
    assert request.window.teacher_transitions == 460


def test_contract_rejects_heldout_failure_semantic_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = json.loads((REPOSITORY_ROOT / qualification.CONTRACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    failure_entry = contract["inputs"]["heldout835_cutoff100_failure"]
    failure = json.loads((REPOSITORY_ROOT / failure_entry["path"]).read_text(encoding="utf-8"))
    failure["controller_counts"]["student"] = 88
    bad = tmp_path / "bad_failure.json"
    bad.write_text(json.dumps(failure), encoding="utf-8")
    failure_entry["path"] = str(bad)
    failure_entry["sha256"] = recovery.sha256_file(bad)
    config = tmp_path / "contract.json"
    config.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(qualification, "CONTRACT_RELATIVE_PATH", config)
    monkeypatch.setattr(qualification, "CONTRACT_SHA256", recovery.sha256_file(config))
    with pytest.raises(ValueError, match="decision evidence drift"):
        qualification.load_contract(REPOSITORY_ROOT)
