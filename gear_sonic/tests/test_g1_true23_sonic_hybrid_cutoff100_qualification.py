from __future__ import annotations

import json
from pathlib import Path

import pytest

from gear_sonic.utils import (
    g1_true23_sonic_hybrid_cutoff100_qualification as qualification,
    g1_true23_sonic_student_teacher_recovery as recovery,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_binds_cutoff140_decision_and_cutoff100_partition() -> None:
    contract = qualification.load_contract(REPOSITORY_ROOT)
    assert contract["runtime"]["mode"] == "cutoff100"
    assert contract["runtime"]["student_transitions"] == 100
    assert contract["runtime"]["teacher_transitions"] == 410
    assert contract["decision"]["seed611_cutoff140_terminal_q9"] == 287
    assert contract["decision"]["stop_if_cutoff100_fails_seed611"] is True


def test_request_is_exact_cutoff100() -> None:
    request = qualification._request(REPOSITORY_ROOT, Path("unused.json"))  # noqa: SLF001
    assert request.mode == "cutoff100"
    assert request.window == recovery.resolve_recovery_window("cutoff100")
    assert request.window.student_transitions == 100
    assert request.window.teacher_transitions == 410


def test_contract_rejects_prior_failure_semantic_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = json.loads((REPOSITORY_ROOT / qualification.CONTRACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    failure_entry = contract["inputs"]["seed611_cutoff140_failure"]
    failure = json.loads((REPOSITORY_ROOT / failure_entry["path"]).read_text(encoding="utf-8"))
    failure["first_done"]["q9_before"] = 286
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
