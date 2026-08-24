from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation as causal
import gear_sonic.envs.mjlab.sonic_true23_student_qualification as student_env
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTION_SCALE_HARDWARE,
    HOME_Q_HARDWARE,
    checkpoint21204_raw_action_to_hardware_targets,
    load_checkpoint21204_binding,
)
import gear_sonic.utils.g1_true23_sonic_student_teacher_recovery as recovery

ROOT = Path(__file__).resolve().parents[2]
DAD_DANCE = ROOT / causal.DAD_DANCE_RELATIVE_PATH


def test_recovery_contract_and_modes_are_exact() -> None:
    contract = recovery.load_recovery_contract(ROOT)
    assert contract["kind"] == "g1_true23_sonic_student_teacher_recovery_contract_v3"
    assert contract["bindings"]["previous_recovery_contract_sha256"] == (
        recovery.PREVIOUS_RECOVERY_CONTRACT_SHA256
    )
    assert contract["bindings"]["base_recovery_contract_sha256"] == (recovery.BASE_RECOVERY_CONTRACT_SHA256)
    assert contract["report"]["terminal_ee_position_arrays_permitted"] is False
    assert contract["report"]["teacher_raw_or_action_arrays_permitted"] is False
    assert contract["report"]["training_arrays_permitted"] is False

    assert recovery.MODES == ("cutoff50", "cutoff75", "cutoff100", "cutoff140")
    cutoff50 = recovery.resolve_recovery_window("cutoff50")
    assert cutoff50.to_dict() == {
        "mode": "cutoff50",
        "anchor_q9": 9,
        "last_q9": 518,
        "transitions": 510,
        "student_transition_count": 50,
        "student_first_q9": 9,
        "student_last_q9": 58,
        "teacher_transition_count": 460,
        "teacher_first_q9": 59,
        "teacher_last_q9": 518,
    }
    cutoff75 = recovery.resolve_recovery_window("cutoff75")
    assert cutoff75.to_dict() == {
        "mode": "cutoff75",
        "anchor_q9": 9,
        "last_q9": 518,
        "transitions": 510,
        "student_transition_count": 75,
        "student_first_q9": 9,
        "student_last_q9": 83,
        "teacher_transition_count": 435,
        "teacher_first_q9": 84,
        "teacher_last_q9": 518,
    }
    cutoff100 = recovery.resolve_recovery_window("cutoff100")
    assert cutoff100.to_dict() == {
        "mode": "cutoff100",
        "anchor_q9": 9,
        "last_q9": 518,
        "transitions": 510,
        "student_transition_count": 100,
        "student_first_q9": 9,
        "student_last_q9": 108,
        "teacher_transition_count": 410,
        "teacher_first_q9": 109,
        "teacher_last_q9": 518,
    }
    cutoff140 = recovery.resolve_recovery_window("cutoff140")
    assert cutoff140.student_transitions == 140
    assert cutoff140.teacher_transitions == 370
    assert cutoff140.student_last_q9 == 148
    assert cutoff140.teacher_first_q9 == 149
    assert cutoff140.last_q9 == 518
    with pytest.raises(ValueError, match="unsupported recovery mode"):
        recovery.resolve_recovery_window("cutoff500")


def test_controller_switch_is_unshifted() -> None:
    window = recovery.resolve_recovery_window("cutoff50")
    assert window.controller(49) == "student"
    assert window.anchor_q9 + 49 == 58
    assert window.controller(50) == "teacher"
    assert window.anchor_q9 + 50 == 59
    with pytest.raises(ValueError, match="outside window"):
        window.controller(510)


def test_executed_source_binding_contains_sealed_contract_chain() -> None:
    binding = recovery.executed_recovery_source_binding(ROOT)
    paths = {record["path"] for record in binding["recovery_files"]}
    assert recovery.BASE_RECOVERY_CONTRACT_RELATIVE_PATH.as_posix() in paths
    assert recovery.PREVIOUS_RECOVERY_CONTRACT_RELATIVE_PATH.as_posix() in paths
    assert recovery.RECOVERY_CONTRACT_RELATIVE_PATH.as_posix() in paths
    assert binding["recovery_source_file_count"] == 6


def test_selected_previous_action_uses_actual_final_hardware_target() -> None:
    raw = np.linspace(-2.0, 2.0, 23, dtype=np.float32)
    target = checkpoint21204_raw_action_to_hardware_targets(raw)
    recovered = recovery.selected_previous_raw_from_final_target(target)
    independent = ((target - HOME_Q_HARDWARE) / ACTION_SCALE_HARDWARE).astype(np.float32)
    np.testing.assert_array_equal(recovered, independent)
    np.testing.assert_allclose(recovered, raw, rtol=0.0, atol=recovery.ACTION_LINK_ATOL)
    zero_target = HOME_Q_HARDWARE.copy()
    np.testing.assert_array_equal(
        recovery.selected_previous_raw_from_final_target(zero_target),
        np.zeros(23, dtype=np.float32),
    )


def test_scope_and_failure_report_never_admit_labels_or_training() -> None:
    scope = recovery.recovery_scope("cutoff140")
    assert scope["teacher_labels_admitted"] is False
    assert scope["training_arrays_present"] is False
    assert scope["dagger_data"] is False
    request = recovery.RecoveryRequest(
        repository_root=ROOT,
        candidate_manifest=recovery.CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH,
        expected_candidate_manifest_sha256=recovery.CURRENT_CANDIDATE_MANIFEST_SHA256,
        output=Path("artifacts/g1_true23/unused_recovery_test.json"),
        mode="cutoff140",
    )
    report = recovery.failure_report(RuntimeError("synthetic failure"), request)
    encoded = json.dumps(report, sort_keys=True)
    assert '"published_teacher_label_count": 0' in encoded
    assert '"teacher_labels_admitted": false' in encoded
    assert '"training_performed": false' in encoded
    assert "teacher_raw_action_hardware" not in encoded
    assert "pre_action_arrays" not in encoded


def test_publication_boundary_rejects_teacher_and_array_values() -> None:
    recovery._assert_publication_boundary(  # noqa: SLF001
        {"digest": {"shape": [23], "sha256": "0" * 64}, "training_performed": False}
    )
    with pytest.raises(RuntimeError, match="forbidden field"):
        recovery._assert_publication_boundary(  # noqa: SLF001
            {"teacher_raw_action_hardware": [0.0] * 23}
        )
    with pytest.raises(RuntimeError, match="array object"):
        recovery._assert_publication_boundary({"value": np.zeros(23, dtype=np.float32)})  # noqa: SLF001
    with pytest.raises(RuntimeError, match="forbidden field"):
        recovery._assert_publication_boundary(  # noqa: SLF001
            {"ee_body_position_errors": []}
        )


def test_terminal_ee_position_evidence_is_scalar_and_diagnostic() -> None:
    raw = [
        {
            "name": "left_wrist_yaw_link",
            "command_body_index": 17,
            "reference_position_w_m": [0.0, 0.0, 0.0],
            "measured_position_w_m": [0.1, -0.2, 0.26],
            "error_measured_minus_reference_m": [0.1, -0.2, 0.26],
            "error_norm_m": float(np.linalg.norm([0.1, -0.2, 0.26])),
            "absolute_z_error_m": 0.26,
            "termination_threshold_m": 0.25,
            "z_termination_breached": True,
        },
        {
            "name": "right_wrist_yaw_link",
            "command_body_index": 25,
            "reference_position_w_m": [0.0, 0.0, 0.0],
            "measured_position_w_m": [-0.03, 0.02, -0.24],
            "error_measured_minus_reference_m": [-0.03, 0.02, -0.24],
            "error_norm_m": float(np.linalg.norm([-0.03, 0.02, -0.24])),
            "absolute_z_error_m": 0.24,
            "termination_threshold_m": 0.25,
            "z_termination_breached": False,
        },
    ]
    compact = recovery._compact_terminal_ee_position_evidence(raw)  # noqa: SLF001
    assert compact["configured_body_count"] == 2
    assert compact["breached_body_count"] == 1
    assert compact["dominant_absolute_z_error_body"] == "left_wrist_yaw_link"
    assert compact["dominant_signed_z_error_m"] == 0.26
    assert compact["termination_threshold_m"] == 0.25
    assert compact["bodies"][1]["error_x_m"] == -0.03
    assert compact["position_or_error_vectors_published"] is False
    encoded = json.dumps(compact, sort_keys=True)
    assert "error_measured_minus_reference_m" not in encoded
    assert "reference_position_w_m" not in encoded
    assert "measured_position_w_m" not in encoded
    recovery._assert_publication_boundary(  # noqa: SLF001
        {"ee_body_position_scalar_evidence": compact}
    )


def test_digest_publishes_only_metadata_and_hashes() -> None:
    digest = recovery._ArrayDigestAccumulator("test_digest")  # noqa: SLF001
    values = np.arange(23, dtype=np.float32)
    digest.add(
        transition=0,
        q9=9,
        controller="student",
        arrays={"selected_teacher_onnx_raw_hardware23": values},
    )
    report = digest.report()
    assert report["contains_array_values"] is False
    assert report["teacher_action_arrays_published"] is False
    assert report["first"]["arrays"][0]["shape"] == [23]
    assert report["first"]["arrays"][0]["sha256"]
    assert values.tolist() not in report["first"]["arrays"]


def test_recovery_terminal_recorder_guarantees_original_reset_on_capture_error() -> None:
    class FakeEnv:
        common_step_counter = 1

        def __init__(self) -> None:
            self.reset_calls = 0

        def _reset_idx(self, _env_ids: object = None) -> None:
            self.reset_calls += 1

    env = FakeEnv()
    recorder = recovery._RecoveryTerminalRecorder(  # noqa: SLF001
        env,
        np.ones(23, dtype=np.float32),
    )
    recorder.arm(
        transition=509,
        q9_before=518,
        controller="teacher",
        behavior_raw=np.zeros(23, dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="expected environment zero"):
        env._reset_idx(None)
    assert env.reset_calls == 1
    recorder.restore()


def test_runtime_failure_reports_started_and_completed_simulator_activity() -> None:
    request = recovery.RecoveryRequest(
        repository_root=ROOT,
        candidate_manifest=recovery.CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH,
        expected_candidate_manifest_sha256=recovery.CURRENT_CANDIDATE_MANIFEST_SHA256,
        output=Path("artifacts/g1_true23/unused_runtime_failure_test.json"),
        mode="cutoff100",
    )
    preflight = {
        "ready": True,
        "issues": [],
        "student": {
            "candidate": {
                "manifest_sha256": recovery.CURRENT_CANDIDATE_MANIFEST_SHA256,
                "decoder_sha256": recovery.CURRENT_CANDIDATE_DECODER_SHA256,
            }
        },
    }
    report = recovery._runtime_failure_report(  # noqa: SLF001
        error=RuntimeError("post-step capture failed"),
        request=request,
        preflight=preflight,
        stage="rollout_transition",
        transition=0,
        q9=9,
        step_calls_started=1,
        attempted=0,
        student_inferences=1,
        teacher_queries=1,
        controller_counts={"student": 0, "teacher": 0},
        cleanup_errors=[],
    )
    assert report["simulator_step_calls_started"] == 1
    assert report["attempted_transitions"] == 0
    assert report["safety"]["simulator_steps"] == 1
    assert report["safety"]["simulator_step_calls_started"] == 1
    assert report["safety"]["simulator_transitions_completed"] == 0
    assert report["safety"]["student_inferences"] == 1
    assert report["safety"]["teacher_queries"] == 1


def _passing_assessment_inputs(mode: str = "cutoff100") -> dict[str, object]:
    window = recovery.resolve_recovery_window(mode)
    return {
        "window": window,
        "step_calls_started": 510,
        "attempted": 510,
        "student_inference_count": 510,
        "teacher_parity": {"query_count": 510, "passed": True},
        "controller_counts": {
            "student": window.student_transitions,
            "teacher": window.teacher_transitions,
        },
        "first_done": {
            "transition": 509,
            "q9_before": 518,
            "q9_after_autoreset": 9,
            "episode_length_pre_reset": 510,
            "termination_names": ["time_out"],
            "is_timeout": True,
            "is_terminated": False,
            "capture_errors": [],
            "autoreset_history": {
                "local_transition": 0,
                "actual_history_depth": 1,
                "reset_padding_count": 9,
                "term_major_policy930_exact": True,
                "previous_action_slice_zero": True,
            },
            "selected124_autoreset": {"previous_selected_raw_is_exact_zero": True},
        },
        "history_check_count": 510,
        "history_shift_count": 509,
        "action_semantics": {"passed": True},
        "teacher_composite": {"passed": True},
        "selected_state": {
            "build_count": 510,
            "nonterminal_update_count": 509,
            "mismatch_count": 0,
            "handoff_observed": True,
            "autoreset_synchronization_count": 1,
            "autoreset_previous_selected_raw_is_exact_zero": True,
        },
        "support_summary": {
            "termination_count": 0,
            "q9_discontinuity_count": 0,
            "nonfinite_count": 0,
            "raw_clip_required_count": 0,
            "action_semantics_mismatch_count": 0,
            "target_soft_limit_violation_count": 0,
            "actuator_target_soft_limit_violation_count": 0,
            "measured_soft_limit_violation_count": 0,
            "joint_velocity_limit_violation_count": 0,
            "teacher_parity_violation_count": 0,
            "teacher_composite_mismatch_count": 0,
            "selected_state_mismatch_count": 0,
            "minimum_base_height_m": 0.6,
            "maximum_base_tilt_rad": 0.5,
            "maximum_joint_velocity_ratio": 0.5,
            "maximum_tracking_rmse_rad": 0.2,
            "maximum_plain_sonic_raw_native_abs": 2.0,
        },
        "frozen_models": {"all_preflight_bound_inputs_unchanged": True},
        "bindings_unchanged": True,
        "partial_failure": None,
    }


def test_assessment_requires_whole_recovery_and_exact_seam() -> None:
    values = _passing_assessment_inputs()
    passed = recovery.assess_recovery(**values)
    assert passed["recovered_to_original_q9_518_boundary"] is True
    assert passed["claims"]["student_alone_qualified"] is False
    assert passed["claims"]["teacher_labels_admitted"] is False

    early = copy.deepcopy(values)
    early["attempted"] = 109
    early["first_done"]["transition"] = 108  # type: ignore[index]
    assert recovery.assess_recovery(**early)["recovered_to_original_q9_518_boundary"] is False

    shifted = copy.deepcopy(values)
    shifted["selected_state"]["handoff_observed"] = False  # type: ignore[index]
    assert recovery.assess_recovery(**shifted)["recovered_to_original_q9_518_boundary"] is False


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    (
        ("first_done", "q9_after_autoreset", 10),
        ("first_done", "episode_length_pre_reset", 509),
        ("first_done", "capture_errors", ["capture failed"]),
        ("autoreset_history", "local_transition", 1),
        ("autoreset_history", "actual_history_depth", 2),
        ("autoreset_history", "reset_padding_count", 8),
        ("autoreset_history", "term_major_policy930_exact", False),
        ("autoreset_history", "previous_action_slice_zero", False),
        ("selected124_autoreset", "previous_selected_raw_is_exact_zero", False),
    ),
)
def test_assessment_rejects_each_terminal_autoreset_field(
    section: str,
    field: str,
    bad_value: object,
) -> None:
    values = _passing_assessment_inputs()
    first_done = values["first_done"]
    assert isinstance(first_done, dict)
    if section == "first_done":
        first_done[field] = bad_value
    else:
        nested = first_done[section]
        assert isinstance(nested, dict)
        nested[field] = bad_value
    result = recovery.assess_recovery(**values)
    assert result["recovered_to_original_q9_518_boundary"] is False
    assert result["gates"]["only_final_timeout"] is False


def test_writer_is_exclusive_atomic_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "recovery.json"
    monkeypatch.setattr(recovery.evaluation, "_evaluation_output_path", lambda _root, _output: output)
    request = recovery.RecoveryRequest(
        repository_root=ROOT,
        candidate_manifest=recovery.CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH,
        expected_candidate_manifest_sha256=recovery.CURRENT_CANDIDATE_MANIFEST_SHA256,
        output=Path("ignored.json"),
        mode="cutoff100",
    )
    report = recovery.failure_report(RuntimeError("synthetic"), request)
    assert recovery.write_recovery_report_new(request, report) == output
    assert output.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        recovery.write_recovery_report_new(request, report)


def test_exact_teacher_pt_cpu_onnx_constructs_and_matches_zero_probe() -> None:
    try:
        binding = load_checkpoint21204_binding(ROOT)
        pair = recovery._ExactTeacherPair(binding, "cpu")  # noqa: SLF001
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"exact teacher runtime unavailable: {error}")
    observation = torch.zeros((1, 124), dtype=torch.float32)
    output, error = pair.infer(observation)
    assert output.shape == (23,)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert error <= recovery.TEACHER_PARITY_ATOL
    report = pair.report()
    assert report["actor_state_unchanged"] is True
    assert report["violation_count"] == 0


def test_real_mjlab_recovery_config_constructs_without_rollout() -> None:
    if causal._MJLAB_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab runtime unavailable")
    cfg = student_env.make_sonic_true23_student_qualification_env_cfg(
        motion_file=str(DAD_DANCE),
        num_envs=1,
        anchor_q9=9,
        transitions=510,
    )
    audit = student_env.audit_sonic_true23_student_qualification_env_cfg(
        cfg,
        expected_anchor_q9=9,
        expected_transitions=510,
    )
    assert tuple(cfg.observations) == ("tokenizer", "policy", "critic")
    assert "actor" not in cfg.observations
    assert audit["safe_target_transform_application_count"] == 1
    assert audit["last_action_q9"] == 518
    assert audit["resolved_max_episode_length"] == 510


def test_wsl_cuda_recovery_reset_state_constructs_without_steps() -> None:
    if causal._MJLAB_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab runtime unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime unavailable")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    cfg = student_env.make_sonic_true23_student_qualification_env_cfg(
        motion_file=str(DAD_DANCE),
        num_envs=1,
        anchor_q9=9,
        transitions=510,
    )
    cfg.seed = recovery.student.FIXED_SEED
    env = ManagerBasedRlEnv(cfg=cfg, device=recovery.student.DEVICE)
    wrapped = None
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        recovery.prime_sonic_true23_training_environment(wrapped)
        observations = wrapped.get_observations()
        selected = recovery._Selected124State(env)  # noqa: SLF001
        proof = recovery._reset_seam(env, observations, selected)  # noqa: SLF001
        actor124, pre_step_torso = selected.build()
        assert proof["prime_q9"] == 9
        assert proof["selected124"]["previous_selected_raw_is_exact_zero"] is True
        assert actor124.shape == (1, 124)
        assert actor124.dtype == torch.float32
        assert bool(torch.isfinite(actor124).all())
        assert pre_step_torso.shape == (1, 4)
        assert selected.build_count == 1
        assert selected.update_count == 0
        assert int(env.common_step_counter) == 0
        assert int(env._sim_step_counter) == 0  # noqa: SLF001
        assert int(env.episode_length_buf[0].item()) == 0
    finally:
        (wrapped or env).close()
