from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23 import native_actions_to_hardware_targets
from gear_sonic.utils import g1_23dof_native124_21204_adapter as adapter, g1_true23_teacher_support as support
from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ZERO_23 = (0.0,) * 23
ZERO_64 = (0.0,) * 64
ZERO_267 = (0.0,) * 267
ZERO_930 = (0.0,) * 930
ZERO_994 = (0.0,) * 994
SPLIT_SHA = "a" * 64
STUDENT_SHA = "b" * 64
SUPPORT_WITNESS = (
    REPOSITORY_ROOT / "artifacts/g1_true23/selected_source_daddance_full_clip_continuous_seed20260805_v2.json"
)
ZERO_TEACHER = support.compose_checkpoint21204_teacher_action(
    ZERO_23,
    repository_root=REPOSITORY_ROOT,
)


def _witness_action_chain() -> dict[str, list[float]]:
    evidence = json.loads(SUPPORT_WITNESS.read_text(encoding="utf-8"))
    return dict(evidence["partial_action_contract_failure"]["action_chain"])


def _contract() -> dict[str, object]:
    return dict(support.load_teacher_support_contract(REPOSITORY_ROOT))


def _global_evidence() -> dict[str, object]:
    contract = _contract()
    identity = dict(contract["artifact_identity"])
    gate = dict(contract["global_gate"])
    return {
        "schema_version": 1,
        "kind": support.GLOBAL_EVIDENCE_KIND,
        "artifact_identity": identity,
        "onnx_input_count": 1,
        "onnx_input_name": "obs",
        "onnx_input_shape": [1, 124],
        "onnx_output_name": "actions",
        "onnx_output_shape": [1, 23],
        "abi_passed": True,
        "parity_passed": True,
        "parity_case_count": 82,
        "parity_max_absolute_error": 9.5367431640625e-06,
        "clip_count": 61,
        "rollouts_per_clip": 10,
        "steps_per_rollout": 500,
        "mean_completion_score": 71.31147540983606,
        "mean_survival_score": 81.22950819672134,
        "nonperfect_clip_count": 38,
        "perfect_clip_count": 23,
        "original_six_completed_rollouts": 60,
        "original_six_total_rollouts": 60,
        "zero_completion_clips": list(gate["zero_completion_clips"]),
        "new_failing_named_clips": [],
    }


def _identity(
    *,
    window_id: str = "window-0",
    session_id: str = "session-0",
    family: str = "reach",
    split: str = "train",
) -> dict[str, object]:
    return {
        "window_id": window_id,
        "source_session_id": session_id,
        "family": family,
        "split": split,
        "split_manifest_sha256": SPLIT_SHA,
        "student_checkpoint_sha256": STUDENT_SHA,
        "learner_iteration": 25,
        "collection_seed": 101,
        "exploration_seed": 202,
    }


def _source_samples() -> list[dict[str, int]]:
    return [
        {
            "source_frame_index": 700 + index,
            "reference_monotonic_ns": 1_000_000_000 + index * 20_000_000,
        }
        for index in range(510)
    ]


def _row(
    index: int,
    identity: dict[str, object],
    teacher_identity: dict[str, object],
) -> dict[str, object]:
    anchor_index = index + 9
    control_index = index + 10
    candidate_target = tuple(float(value) for value in ZERO_TEACHER.teacher_candidate_target_hardware)
    target = tuple(float(value) for value in ZERO_TEACHER.teacher_target_hardware)
    return {
        "row_index": index,
        **identity,
        "teacher_identity": teacher_identity,
        "anchor_source_frame_index": 700 + anchor_index,
        "anchor_reference_monotonic_ns": 1_000_000_000 + anchor_index * 20_000_000,
        "control_source_frame_index": 700 + control_index,
        "control_monotonic_ns": 1_000_000_000 + control_index * 20_000_000,
        "encoder267": ZERO_267,
        "token64": ZERO_64,
        "proprio930": ZERO_930,
        "decoder994": ZERO_994,
        "student_mean_action_native": ZERO_23,
        "student_raw_action_native": ZERO_23,
        "student_applied_action_native": ZERO_23,
        "teacher_raw_action_hardware": ZERO_23,
        "teacher_candidate_target_hardware": candidate_target,
        "teacher_target_hardware": target,
        "next_state": {
            "q_hardware": target,
            "qd_hardware": ZERO_23,
            "base_angular_velocity": (0.0, 0.0, 0.0),
            "torso_quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
        },
        "pico_age_ms": 10.0,
        "lowstate_age_ms": 5.0,
        "inference_duration_ms": 2.0,
        "base_height_m": 0.8,
        "base_tilt_rad": 0.1,
        "joint_velocity_ratio": 0.2,
        "tracking_error_rad": 0.1,
        "measured_joint_limit_failure_count": 0,
        "teacher_target_soft_limit_failure_count": 0,
        "target_clamp_failure_count": 0,
        "nonfinite_count": 0,
        "reset_occurred": False,
        "reference_resampled": False,
        "clip_boundary": False,
        "session_boundary": False,
        "terminated": False,
        "timed_out": False,
        "done": False,
        "anchor_failure": False,
        "ee_failure": False,
        "reported_nonfinite": False,
        "sim_advanced": True,
    }


def _window(
    *,
    window_id: str = "window-0",
    session_id: str = "session-0",
    family: str = "reach",
    split: str = "train",
) -> dict[str, object]:
    identity = _identity(
        window_id=window_id,
        session_id=session_id,
        family=family,
        split=split,
    )
    teacher_identity = dict(_contract()["artifact_identity"])
    return {
        "schema_version": 1,
        "kind": support.WINDOW_KIND,
        **identity,
        "teacher_identity": teacher_identity,
        "source_samples": _source_samples(),
        "rows": [_row(index, identity, teacher_identity) for index in range(500)],
    }


def _qualification_runs(window: dict[str, object]) -> list[dict[str, object]]:
    contract = _contract()
    qualification = dict(contract["qualification"])
    identity = {
        name: window[name]
        for name in (
            "window_id",
            "source_session_id",
            "family",
            "split",
            "split_manifest_sha256",
            "student_checkpoint_sha256",
            "learner_iteration",
            "collection_seed",
            "exploration_seed",
        )
    }
    runs: list[dict[str, object]] = []
    for mode in ("nominal", "disturbance"):
        for rollout_index, seed in enumerate(qualification[f"{mode}_seeds"]):
            run: dict[str, object] = {
                "schema_version": 1,
                "kind": support.QUALIFICATION_KIND,
                **identity,
                "teacher_identity": window["teacher_identity"],
                "mode": mode,
                "rollout_index": rollout_index,
                "seed": seed,
                "completed": True,
                "steps_requested": 500,
                "steps_executed": 500,
                "anchor_failure_count": 0,
                "ee_failure_count": 0,
                "measured_joint_limit_failure_count": 0,
                "teacher_target_unrepresentable_count": 0,
                "student_raw_clip_required_count": 0,
                "target_clamp_failure_count": 0,
                "nonfinite_count": 0,
                "reset_count": 0,
                "missing_step_count": 0,
                "duplicate_step_count": 0,
                "minimum_base_height_m": 0.7,
                "maximum_base_tilt_rad": 0.2,
                "maximum_joint_velocity_ratio": 0.3,
                "tracking_rmse_rad": 0.2,
                "maximum_abs_student_raw_action": 1.0,
                "maximum_pico_age_ms": 20.0,
                "maximum_lowstate_age_ms": 10.0,
                "maximum_inference_duration_ms": 3.0,
                "disturbance_profile": "none",
                "disturbance_applied": False,
            }
            if mode == "disturbance":
                disturbance = dict(qualification["disturbance"])
                run.update(
                    {
                        "disturbance_profile": disturbance["profile"],
                        "disturbance_applied": True,
                        "apply_step": disturbance["apply_step"],
                        "baseline_steps": disturbance["baseline_steps"],
                        "stable_recovery_steps": disturbance["stable_recovery_steps"],
                        "recovery_margin": disturbance["recovery_margin"],
                        "recovery_fraction": 1.0,
                        "maximum_recovery_time_s": 1.0,
                        "disturbance_linear_velocity_delta_m_s": (0.1, 0.0, 0.0),
                        "disturbance_angular_velocity_delta_rad_s": (0.0, 0.1, 0.0),
                    }
                )
            runs.append(run)
    return runs


def _assess(window: dict[str, object]) -> support.TeacherSupportVerdict:
    return support.assess_teacher_support_window(
        window=window,
        qualification_runs=_qualification_runs(window),
        global_evidence=_global_evidence(),
        repository_root=REPOSITORY_ROOT,
    )


def test_frozen_config_hash_and_module_has_no_runtime_or_writing_api() -> None:
    config_path = REPOSITORY_ROOT / support.SUPPORT_CONFIG_RELATIVE_PATH
    assert support._sha256_bytes(config_path.read_bytes()) == support.SUPPORT_CONFIG_SHA256
    support.load_teacher_support_contract(REPOSITORY_ROOT)
    assert support.TRACKER_MANIFEST_SHA256 == adapter.MANIFEST_SHA256
    composite = _contract()["teacher_composite_contract"]
    np.testing.assert_array_equal(
        np.asarray(composite["checkpoint21204_home_q_hardware"], dtype=np.float32),
        adapter.HOME_Q_HARDWARE,
    )
    np.testing.assert_array_equal(
        np.asarray(composite["checkpoint21204_action_scale_hardware"], dtype=np.float32),
        adapter.ACTION_SCALE_HARDWARE,
    )

    source_path = Path(support.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        fragment in name
        for name in imports
        for fragment in ("onnx", "mujoco", "mjlab", "transport", "unitree_sdk", "socket")
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called_attributes & {"write_bytes", "write_text", "open", "unlink", "replace"}


def test_teacher_target_inverse_matches_final_v11_safe_target_contract() -> None:
    native = np.linspace(-0.25, 0.25, 23, dtype=np.float32)
    _, expected_target = safe_target_transform_numpy(native)
    label = support.teacher_hardware_target_to_sonic_native_label(
        expected_target,
        repository_root=REPOSITORY_ROOT,
    )
    np.testing.assert_allclose(label, native, rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(
        support.sonic_native_label_to_hardware_target(label),
        expected_target,
        rtol=0.0,
        atol=1.0e-5,
    )
    assert support.FINAL_TEACHER_ACTION_TRANSFORM is support.TeacherActionTransform.SONIC_V11_SAFE_TARGET_V2

    diagnostic_native = np.linspace(-1.5, 1.5, 23, dtype=np.float32)
    linear_target = native_actions_to_hardware_targets(torch.from_numpy(diagnostic_native)[None, :]).numpy()[0]
    diagnostic_label = support.teacher_hardware_target_to_native_action(
        linear_target,
        transform=support.TeacherActionTransform.LINEAR_DIAGNOSTIC_V1,
        repository_root=REPOSITORY_ROOT,
    )
    np.testing.assert_allclose(diagnostic_label, diagnostic_native, rtol=0.0, atol=2.0e-7)


def test_checkpoint21204_teacher_composite_matches_every_frozen_link() -> None:
    raw_hardware = np.linspace(-0.2, 0.2, 23, dtype=np.float32)
    composite = support.compose_checkpoint21204_teacher_action(
        raw_hardware,
        repository_root=REPOSITORY_ROOT,
    )
    candidate = adapter.checkpoint21204_raw_action_to_hardware_targets(raw_hardware)
    plain_hardware = (candidate - support.SONIC_HARDWARE_DEFAULT_Q) / support.SONIC_HARDWARE_ACTION_SCALE
    expected_label = plain_hardware[np.asarray(MUJOCO_TO_ISAACLAB_DOF)]
    expected_safe, expected_target = safe_target_transform_numpy(expected_label.astype(np.float32, copy=False))
    np.testing.assert_array_equal(composite.teacher_raw_action_hardware, raw_hardware)
    np.testing.assert_array_equal(
        composite.teacher_candidate_target_hardware,
        candidate,
    )
    np.testing.assert_allclose(
        composite.teacher_action_native,
        expected_label,
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        composite.teacher_applied_safe_action_native,
        expected_safe,
    )
    np.testing.assert_array_equal(composite.teacher_target_hardware, expected_target)


def test_teacher_composite_rejects_nonfinite_raw_action() -> None:
    raw = np.full(23, np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="teacher_raw_action_hardware"):
        support.compose_checkpoint21204_teacher_action(raw, repository_root=REPOSITORY_ROOT)


def test_teacher_composite_rejects_strict_raw_clip_domain() -> None:
    raw = np.full(23, 10.0, dtype=np.float32)
    with pytest.raises(ValueError, match="plain SONIC raw clipping"):
        support.compose_checkpoint21204_teacher_action(raw, repository_root=REPOSITORY_ROOT)


def test_teacher_composite_rejects_forward_roundtrip_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    original_inverse = support._invert_teacher_hardware_target
    raw = np.zeros(23, dtype=np.float32)

    def poisoned_inverse(target: np.ndarray, action_contract: dict[str, object]) -> np.ndarray:
        inverse = original_inverse(target, action_contract).copy()
        inverse[0] += np.float32(1.0e-3)
        return inverse

    monkeypatch.setattr(support, "_invert_teacher_hardware_target", poisoned_inverse)
    with pytest.raises(ValueError, match="safe native action does not recover V2 forward transform"):
        support.compose_checkpoint21204_teacher_action(raw, repository_root=REPOSITORY_ROOT)


def test_teacher_composite_rejects_tanh_raw_saturation_forward_equivalent_witness() -> None:
    chain = _witness_action_chain()
    selected_raw = np.asarray(chain["selected_actor_raw_action_hardware"], dtype=np.float32)
    final_target = np.asarray(chain["final_target_hardware"], dtype=np.float32)

    composite = support.compose_checkpoint21204_teacher_action(
        selected_raw,
        repository_root=REPOSITORY_ROOT,
    )
    np.testing.assert_allclose(composite.teacher_target_hardware, final_target, rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(
        composite.teacher_action_native,
        np.asarray(chain["plain_sonic_raw_action_native"], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-7,
    )
    inverse_raw = support.teacher_hardware_target_to_sonic_native_label(
        composite.teacher_target_hardware,
        repository_root=REPOSITORY_ROOT,
    )
    max_inverse_error = float(
        np.max(np.abs(composite.teacher_action_native.astype(np.float64) - inverse_raw.astype(np.float64)))
    )
    assert max_inverse_error > 1.0e-5


def test_exact_ten_plus_ten_qualifies_and_exports_all_500_train_rows() -> None:
    window = _window()
    runs = _qualification_runs(window)
    verdict = support.assess_teacher_support_window(
        window=window,
        qualification_runs=runs,
        global_evidence=_global_evidence(),
        repository_root=REPOSITORY_ROOT,
    )
    assert verdict.admitted
    assert verdict.training_exportable
    assert verdict.admitted_row_count == 500
    assert verdict.nominal_rollout_count == verdict.disturbance_rollout_count == 10
    assert verdict.to_record()["teacher_labels_included"] is False

    export = support.build_support_admitted_training_export(
        window=window,
        qualification_runs=runs,
        global_evidence=_global_evidence(),
        repository_root=REPOSITORY_ROOT,
    )
    assert export["kind"] == support.TRAINING_EXPORT_KIND
    assert export["row_count"] == 500
    assert len(export["rows"]) == 500
    np.testing.assert_array_equal(
        export["rows"][0]["teacher_action_native"],
        ZERO_TEACHER.teacher_action_native,
    )
    np.testing.assert_array_equal(
        export["rows"][0]["teacher_candidate_target_hardware"],
        ZERO_TEACHER.teacher_candidate_target_hardware,
    )
    assert export["student_applied_action_transform"] == support.SAFE_TARGET_TRANSFORM_KIND
    assert export["student_applied_link_atol"] == 1.0e-6
    assert "student_raw_action_native" in export["rows"][0]
    assert "student_applied_action_native" in export["rows"][0]


def test_one_bad_frame_quarantines_whole_window_and_exposes_no_label() -> None:
    window = _window()
    rows = list(window["rows"])
    rows[137] = dict(rows[137], anchor_failure=True)
    window["rows"] = rows
    verdict = _assess(window)
    assert not verdict.admitted
    assert verdict.admitted_row_count == 0
    record = verdict.to_record()
    assert record["kind"] == support.QUARANTINE_KIND
    assert record["teacher_labels_included"] is False
    assert any("rows[137].anchor_failure" in reason for reason in verdict.quarantine_reasons)
    with pytest.raises(ValueError, match="quarantined"):
        support.build_support_admitted_training_export(
            window=window,
            qualification_runs=_qualification_runs(window),
            global_evidence=_global_evidence(),
            repository_root=REPOSITORY_ROOT,
        )


@pytest.mark.parametrize("fault", ["short_window", "missing_run", "duplicate_run", "wrong_seed"])
def test_incomplete_or_duplicate_window_qualification_fails_closed(fault: str) -> None:
    window = _window()
    runs = _qualification_runs(window)
    if fault == "short_window":
        window["rows"] = list(window["rows"])[:-1]
    elif fault == "missing_run":
        runs.pop()
    elif fault == "duplicate_run":
        runs[-1] = dict(runs[-2])
    else:
        runs[0] = dict(runs[0], seed=int(runs[0]["seed"]) + 1)
    verdict = support.assess_teacher_support_window(
        window=window,
        qualification_runs=runs,
        global_evidence=_global_evidence(),
        repository_root=REPOSITORY_ROOT,
    )
    assert not verdict.admitted
    assert verdict.admitted_row_count == 0


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("nonfinite", "encoder267"),
        ("timestamp_gap", "timestamp gap"),
        ("freshness", "pico_age_ms"),
        ("concat", "not exact token64/proprio930 concat"),
    ],
)
def test_row_contract_faults_quarantine_whole_window(fault: str, reason: str) -> None:
    window = _window()
    rows = list(window["rows"])
    row = dict(rows[4])
    if fault == "nonfinite":
        row["encoder267"] = (float("nan"),) + ZERO_267[1:]
    elif fault == "freshness":
        row["pico_age_ms"] = 60.01
    elif fault == "concat":
        row["decoder994"] = (1.0,) + ZERO_994[1:]
    else:
        samples = list(window["source_samples"])
        samples[20] = dict(samples[20], reference_monotonic_ns=samples[20]["reference_monotonic_ns"] + 1)
        window["source_samples"] = samples
    rows[4] = row
    window["rows"] = rows
    verdict = _assess(window)
    assert not verdict.admitted
    assert any(reason in item for item in verdict.quarantine_reasons)


def test_unrepresentable_target_and_partial_pre_admission_label_are_rejected() -> None:
    window = _window()
    rows = list(window["rows"])
    row = dict(rows[0])
    row["teacher_raw_action_hardware"] = (20.0,) * 23
    row["teacher_action_native"] = ZERO_23
    rows[0] = row
    window["rows"] = rows
    verdict = _assess(window)
    assert not verdict.admitted
    assert any(
        "teacher composite requires plain SONIC raw clipping" in item for item in verdict.quarantine_reasons
    )
    assert any("pre-admission label is forbidden" in item for item in verdict.quarantine_reasons)


@pytest.mark.parametrize("fault", ["raw", "candidate", "composite"])
def test_teacher_raw_candidate_and_composite_links_are_all_bound(fault: str) -> None:
    window = _window()
    rows = list(window["rows"])
    row = dict(rows[19])
    if fault == "raw":
        row["teacher_raw_action_hardware"] = (0.1,) + ZERO_23[1:]
    elif fault == "candidate":
        candidate = list(row["teacher_candidate_target_hardware"])
        candidate[3] += 1.0e-3
        row["teacher_candidate_target_hardware"] = candidate
    else:
        composite = list(row["teacher_target_hardware"])
        composite[7] += 1.0e-3
        row["teacher_target_hardware"] = composite
    rows[19] = row
    window["rows"] = rows
    verdict = _assess(window)
    assert not verdict.admitted
    assert verdict.admitted_row_count == 0
    expected = {
        "raw": "teacher raw/HOME/SCALE link",
        "candidate": "teacher raw/HOME/SCALE link",
        "composite": "checkpoint21204-to-V2 composite",
    }[fault]
    assert any(expected in item for item in verdict.quarantine_reasons)


def test_safe_target_raw_clip_boundary_is_strict_and_never_silently_clipped() -> None:
    boundary_raw = np.full(23, 10.0, dtype=np.float32)
    _, boundary_target = safe_target_transform_numpy(boundary_raw)
    with pytest.raises(ValueError, match="strict V11 safe reachable domain"):
        support.teacher_hardware_target_to_sonic_native_label(
            boundary_target,
            repository_root=REPOSITORY_ROOT,
        )

    window = _window()
    rows = list(window["rows"])
    rows[8] = dict(rows[8], student_raw_action_native=(10.0,) + ZERO_23[1:])
    window["rows"] = rows
    verdict = _assess(window)
    assert not verdict.admitted
    assert any("requires raw clipping" in item for item in verdict.quarantine_reasons)


def test_student_applied_action_must_match_exact_v2_transform_of_raw_action() -> None:
    window = _window()
    rows = list(window["rows"])
    raw = np.linspace(-0.2, 0.2, 23, dtype=np.float32)
    expected_applied, _ = safe_target_transform_numpy(raw)
    bad_applied = expected_applied.copy()
    bad_applied[11] += 2.0e-5
    rows[31] = dict(
        rows[31],
        student_raw_action_native=raw.tolist(),
        student_applied_action_native=bad_applied.tolist(),
    )
    window["rows"] = rows
    verdict = _assess(window)
    assert not verdict.admitted
    assert verdict.admitted_row_count == 0
    assert any(
        "does not match exact V2 transform of student raw action" in item for item in verdict.quarantine_reasons
    )


def test_heldout_window_can_be_admitted_but_never_training_exported() -> None:
    window = _window(split="heldout")
    runs = _qualification_runs(window)
    verdict = support.assess_teacher_support_window(
        window=window,
        qualification_runs=runs,
        global_evidence=_global_evidence(),
        repository_root=REPOSITORY_ROOT,
    )
    assert verdict.admitted
    assert not verdict.training_exportable
    with pytest.raises(ValueError, match="heldout"):
        support.build_support_admitted_training_export(
            window=window,
            qualification_runs=runs,
            global_evidence=_global_evidence(),
            repository_root=REPOSITORY_ROOT,
        )


def _verdict(
    family: str,
    split: str,
    session_index: int,
    *,
    admitted: bool = True,
) -> support.TeacherSupportVerdict:
    return support.TeacherSupportVerdict(
        window_id=f"{family}-{split}-{session_index}-window".replace("/", "_"),
        source_session_id=f"{family}-{split}-{session_index}".replace("/", "_"),
        family=family,
        split=split,
        split_manifest_sha256=SPLIT_SHA,
        student_checkpoint_sha256=STUDENT_SHA,
        learner_iteration=25,
        admitted=admitted,
        training_exportable=admitted and split == "train",
        admitted_row_count=500 if admitted else 0,
        nominal_rollout_count=10,
        disturbance_rollout_count=10,
        quarantine_reasons=() if admitted else ("test quarantine",),
    )


def test_six_family_whole_session_tranche_and_split_leakage() -> None:
    families = _contract()["split_tranche"]["families"]
    verdicts = [
        _verdict(family, split, index)
        for family in families
        for split, index in (("train", 0), ("train", 1), ("heldout", 2))
    ]
    tranche = support.assess_teacher_support_tranche(verdicts, repository_root=REPOSITORY_ROOT)
    assert tranche.qualified
    expected_counts = {"total": 3, "train": 2, "heldout": 1, "admitted_train": 2}
    assert all(counts == expected_counts for counts in tranche.family_session_counts.values())

    leaked = replace(
        verdicts[2],
        source_session_id=verdicts[0].source_session_id,
    )
    bad = support.assess_teacher_support_tranche(
        [*verdicts[:2], leaked, *verdicts[3:]],
        repository_root=REPOSITORY_ROOT,
    )
    assert not bad.qualified
    assert any("leaks across train/heldout" in reason for reason in bad.reasons)


def test_global_regression_floor_failure_quarantines_window() -> None:
    window = _window()
    evidence = _global_evidence()
    evidence["mean_survival_score"] = 81.0
    verdict = support.assess_teacher_support_window(
        window=window,
        qualification_runs=_qualification_runs(window),
        global_evidence=evidence,
        repository_root=REPOSITORY_ROOT,
    )
    assert not verdict.admitted
    assert any("mean_survival_score" in reason for reason in verdict.quarantine_reasons)
