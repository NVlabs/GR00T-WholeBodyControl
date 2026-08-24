from __future__ import annotations

from pathlib import Path

from gear_sonic.utils import g1_true23_sonic_rank256_q200_teacher_recovery as diagnostic

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_contract_pins_q200_handoff() -> None:
    contract = diagnostic.load_contract(REPOSITORY_ROOT)
    assert contract["window"]["student_last_q9"] == 199
    assert contract["window"]["teacher_first_q9"] == 200
    assert contract["window"]["teacher_transition_count"] == 319


def test_scope_adds_exact_mode_and_restores() -> None:
    before = diagnostic.recovery.MODE_STUDENT_TRANSITIONS
    with diagnostic._scope(REPOSITORY_ROOT):  # noqa: SLF001
        window = diagnostic.recovery.resolve_recovery_window("cutoff191")
        assert window.student_last_q9 == 199
        assert window.teacher_first_q9 == 200
    assert diagnostic.recovery.MODE_STUDENT_TRANSITIONS == before


def test_request_constructs_only_inside_dynamic_scope(tmp_path: Path) -> None:
    with diagnostic._scope(REPOSITORY_ROOT):  # noqa: SLF001
        request = diagnostic._request(REPOSITORY_ROOT, tmp_path / "report.json")  # noqa: SLF001
        assert request.mode == "cutoff191"
