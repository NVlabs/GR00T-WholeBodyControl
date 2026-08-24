from __future__ import annotations

from pathlib import Path

from gear_sonic.utils import g1_true23_sonic_scale2_teacher_recovery as recovery

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _support() -> dict[str, object]:
    return {
        "minimum_base_height_m": 0.7,
        "maximum_base_tilt_rad": 0.2,
        "maximum_joint_velocity_ratio": 0.3,
        "maximum_tracking_rmse_rad": 0.2,
        "maximum_plain_sonic_raw_native_abs": 3.0,
        "nonfinite_count": 0,
        "q9_discontinuity_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": 0,
        "hard_safety_violation_count": 0,
        "soft_safety_warning_count": 0,
    }


def _done() -> dict[str, object]:
    return {
        "transition": 509,
        "q9_before": 518,
        "q9_after_autoreset": 9,
        "termination_names": ["time_out"],
        "is_timeout": True,
        "is_terminated": False,
    }


def test_scale2_recovery_contract_and_windows_are_exact() -> None:
    contract = recovery.load_contract(REPOSITORY_ROOT)
    assert contract["stop_policy"]["stop_teacher_route_after_q225_failure"] is True
    q250 = recovery.resolve_window("q250")
    assert q250.student_transitions == 241
    assert q250.teacher_transitions == 269
    assert q250.student_last_q9 == 249
    assert q250.teacher_first_q9 == 250
    q225 = recovery.resolve_window("q225")
    assert q225.student_transitions == 216
    assert q225.teacher_transitions == 294
    q200 = recovery.resolve_window("q200")
    assert q200.student_transitions == 191
    assert q200.student_last_q9 == 199
    assert q200.teacher_transitions == 319
    assert q200.teacher_first_q9 == 200
    v2 = recovery.load_contract(REPOSITORY_ROOT, "q200")
    assert v2["evidence_basis"]["q200_margin_before_earliest_failure_frames"] == 14

    q175 = recovery.resolve_window("q175")
    assert q175.student_transitions == 166
    assert q175.student_last_q9 == 174
    assert q175.teacher_transitions == 344
    assert q175.teacher_first_q9 == 175
    v3 = recovery.load_contract(REPOSITORY_ROOT, "q175")
    assert v3["evidence_basis"]["q200_failure_q9"] == 214
    assert v3["evidence_basis"]["q175_margin_before_earliest_failure_frames"] == 39

    q9 = recovery.resolve_window("q9")
    assert q9.student_transitions == 0
    assert q9.student_last_q9 == 8
    assert q9.teacher_transitions == 510
    assert q9.teacher_first_q9 == 9
    v4 = recovery.load_contract(REPOSITORY_ROOT, "q9")
    assert v4["evidence_basis"]["q175_failure_q9"] == 281
    assert v4["evidence_basis"]["selected_teacher_nominal500_qualified"] is True


def test_scale2_recovery_request_accepts_platform_path_subclasses() -> None:
    request = recovery.Scale2RecoveryRequest(
        REPOSITORY_ROOT,
        Path("artifacts/g1_true23/scale2_recovery_test.json"),
        "q250",
    )
    assert request.window.teacher_first_q9 == 250
    assert request.seed == 20260805
    assert request.output_path == (REPOSITORY_ROOT / "artifacts/g1_true23/scale2_recovery_test.json").resolve()

    overridden = recovery.Scale2RecoveryRequest(
        REPOSITORY_ROOT,
        Path("artifacts/g1_true23/scale2_recovery_seed_test.json"),
        "q250",
        runtime_seed=17,
    )
    assert overridden.seed == 17


def test_scale2_recovery_assessment_passes_only_exact_final_timeout() -> None:
    window = recovery.resolve_window("q250")
    result = recovery.assess(
        window=window,
        started=510,
        completed=510,
        student_queries=510,
        teacher_queries=510,
        controller_counts={"student": 241, "teacher": 269},
        first_done=_done(),
        history_checks=510,
        history_shifts=509,
        support=_support(),
        teacher_parity={"violation_count": 0},
        teacher_composite={"mismatch_count": 0},
        selected_state={"mismatch_count": 0, "handoff_observed": True},
        bindings_unchanged=True,
        partial_failure=None,
    )
    assert result["passed"] is True
    assert result["teacher_labels_admitted"] == 0


def test_scale2_recovery_assessment_rejects_early_done() -> None:
    window = recovery.resolve_window("q225")
    done = _done()
    done["transition"] = 400
    done["q9_before"] = 409
    result = recovery.assess(
        window=window,
        started=401,
        completed=401,
        student_queries=401,
        teacher_queries=401,
        controller_counts={"student": 216, "teacher": 185},
        first_done=done,
        history_checks=401,
        history_shifts=400,
        support=_support(),
        teacher_parity={"violation_count": 0},
        teacher_composite={"mismatch_count": 0},
        selected_state={"mismatch_count": 0, "handoff_observed": True},
        bindings_unchanged=True,
        partial_failure=None,
    )
    assert result["passed"] is False
    assert result["teacher_labels_admitted"] == 0


def test_rollout_safety_counts_are_propagated_into_support() -> None:
    support = recovery._with_rollout_safety_counts(  # noqa: SLF001
        {"minimum_base_height_m": 0.7},
        {"hard_safety_violation_count": 0, "soft_safety_warning_count": 0},
    )
    assert support == {
        "minimum_base_height_m": 0.7,
        "hard_safety_violation_count": 0,
        "soft_safety_warning_count": 0,
    }
