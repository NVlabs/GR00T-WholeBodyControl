from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab import (
    native124_selected_v2_causal_adaptation as causal,
    sonic_true23_student_qualification as student_env,
)
from gear_sonic.utils import g1_true23_sonic_student_closed_loop_qualification as qualification
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    ISAACLAB_TO_MUJOCO_DOF,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DAD_DANCE = REPO_ROOT / causal.DAD_DANCE_RELATIVE_PATH


def _candidate_manifest(contract: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": qualification.BC_SCHEMA_VERSION,
        "kind": qualification.BC_MANIFEST_KIND,
        "classification": "offline_bc_eligible_for_closed_loop_simulator_experiment",
        "eligible_for_closed_loop_simulator_experiment": True,
        "gate_issues": [],
        "contract": {
            "path": qualification.BC_CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": qualification.BC_CONTRACT_SHA256,
        },
        "artifact": {
            "decoder_filename": "candidate.decoder.onnx",
            "decoder_sha256": "1" * 64,
            "decoder_size_bytes": 123,
        },
        "lineage": {
            "contract_sha256": qualification.BC_CONTRACT_SHA256,
            "bootstrap_npz_sha256": qualification.BOOTSTRAP_NPZ_SHA256,
            "bootstrap_manifest_sha256": qualification.BOOTSTRAP_MANIFEST_SHA256,
            "bootstrap_manifest_payload_sha256": (qualification.BOOTSTRAP_MANIFEST_PAYLOAD_SHA256),
            "bootstrap_contract_sha256": qualification.BOOTSTRAP_CONTRACT_SHA256,
            "causal_encoder_sha256": qualification.CAUSAL_ENCODER_SHA256,
            "source_decoder_sha256": qualification.CAUSAL_DECODER_SHA256,
            "source_checkpoint_present": False,
            "optimizer_state": None,
            "resume_capable": False,
            "bootstrap_manifest_kind": qualification.BOOTSTRAP_MANIFEST_KIND,
        },
        "boundaries": {
            "offline_behavior_cloning_only": True,
            "teacher_controlled_training_data": True,
            "reset_prefix_support_admitted": False,
            "support_qualification_performed": False,
            "support_admitted": False,
            "on_policy_data": False,
            "dagger_data": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
            "robot_or_network_commands_permitted": False,
        },
        "claims": {
            "blocked_diagnostic_is_independent_heldout_evidence": False,
            "all_510_fit_is_generalization_evidence": False,
            "closed_loop_simulator_qualified": False,
            "full_clip_qualified": False,
        },
        "runtime": {
            "provider": "CPUExecutionProvider",
            "row_count": qualification.TOTAL_ROWS,
            "no_simulator_or_gpu_used": True,
            "no_robot_hardware_or_network_commands_performed": True,
        },
        "fit": {
            "gate_issues": [],
            "offline_fit_gates_passed": True,
        },
        "export": {
            "attempted": True,
            "passed": True,
            "only_final_affine_changed": True,
            "encoder_unchanged": True,
            "decoder_trunk_unchanged": True,
            "changed_initializer_names": [
                qualification.FINAL_WEIGHT_NAME,
                qualification.FINAL_BIAS_NAME,
            ],
            "provider": "CPUExecutionProvider",
            "reference_max_abs_error": 1.0e-6,
            "candidate_decoder_sha256": "1" * 64,
            "candidate_decoder_size_bytes": 123,
            "onnx_embedded_metadata_updated": False,
            "onnx_embedded_metadata_unchanged": True,
            "adapted_lineage_record_location": "external_hash_bound_manifest_only",
            "same_runtime_repeat_fit_head_byte_identical": True,
            "same_runtime_repeat_fit_onnx_byte_identical": True,
            "cross_runtime_byte_determinism_claimed": False,
            "abi": {
                "input_name": "obs_dict",
                "input_shape": [1, qualification.DECODER_DIM],
                "input_dtype": "float32",
                "output_name": "action",
                "output_shape": [1, qualification.ACTION_DIM],
                "output_dtype": "float32",
                "dynamic_axes": False,
                "opset": 13,
            },
            "action_semantics": {
                "output": "pre_safe_transform_plain_sonic_raw_native23",
                "action_order": "native_physx_il23_bfs_v1",
                "v2_transform_application_count": 1,
                "wrapper_action_clip": None,
            },
            "actual_float32_ort_resubstitution_gates": {
                "gate_issues": [],
                "passed": True,
            },
        },
        "_contract_used_by_fixture": contract["kind"],
    }


def _gate() -> dict[str, object]:
    return {
        "required_zero_counts": [
            "termination_count",
            "q9_discontinuity_count",
            "nonfinite_count",
            "raw_clip_required_count",
            "action_semantics_mismatch_count",
            "target_soft_limit_violation_count",
            "actuator_target_soft_limit_violation_count",
            "measured_soft_limit_violation_count",
            "joint_velocity_limit_violation_count",
        ],
        "minimum_base_height_m": 0.45,
        "maximum_base_tilt_rad": 1.0,
        "maximum_joint_velocity_ratio": 1.0,
        "maximum_tracking_rmse_rad": 0.75,
        "plain_sonic_raw_abs_strict_max": 10.0,
    }


def _passing_frozen_input_evidence() -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for index, name in enumerate(qualification.FROZEN_INPUT_NAMES, start=1):
        expected = f"{index:064x}"
        files[name] = {
            "path": f"/frozen/{name}",
            "expected_sha256": expected,
            "sha256_before": expected,
            "sha256_after": expected,
            "before_matches_expected": True,
            "after_matches_expected": True,
            "unchanged": True,
            "before_error": None,
            "after_error": None,
        }
    return {
        "file_count": len(files),
        "all_preflight_bound_inputs_unchanged": True,
        "files": files,
        "training_updates": 0,
    }


def _passing_assessment(window: qualification.StudentQualificationWindow) -> dict[str, object]:
    gate = _gate()
    support = {name: 0 for name in gate["required_zero_counts"]}
    support.update(
        {
            "minimum_base_height_m": 0.5,
            "maximum_base_tilt_rad": 0.5,
            "maximum_joint_velocity_ratio": 0.8,
            "maximum_tracking_rmse_rad": 0.2,
            "maximum_plain_sonic_raw_native_abs": 2.0,
        }
    )
    first_history = {
        "actual_history_depth": 1,
        "reset_padding_count": 9,
        "previous_action_slice_zero": True,
    }
    last_history = {
        "actual_history_depth": 10,
        "reset_padding_count": 0,
        "previous_action_slice_zero": True,
    }
    return {
        "window": window,
        "gate": gate,
        "preflight_ready": True,
        "reset_seam": {
            "prime_q9": window.anchor_q9,
            "first_student_action_q9": window.anchor_q9,
            "fixed_warmup_or_action_substitution_steps": 0,
            "action_substitution": False,
            "first_policy_history": first_history,
            "causal_q9_buffer_finite": True,
        },
        "first_done": {
            "transition": window.transitions - 1,
            "q9_before": window.last_q9,
            "q9_after_autoreset": window.anchor_q9,
            "episode_length_pre_reset": window.transitions,
            "termination_names": ["time_out"],
            "is_timeout": True,
            "is_terminated": False,
            "terminal_capture_errors": [],
            "autoreset_history": first_history,
        },
        "attempted": window.transitions,
        "partition_counts": {
            "burn_in_transition_count": window.burn_in_transitions,
            "scored_transition_count": window.transitions - window.burn_in_transitions,
            "unexpected_done_before_final_count": 0,
        },
        "history": {
            "check_count": window.transitions,
            "single_append_shift_check_count": window.transitions - 1,
            "first": first_history,
            "last": last_history,
            "autoreset": first_history,
        },
        "rollout_summary": {
            "transition_count": window.transitions,
            "hard_safety_violation_count": 0,
            "soft_safety_warning_count": 0,
        },
        "support_summary": support,
        "action_semantics": {
            "passed": True,
            "check_count": window.transitions,
            "mismatch_count": 0,
            "raw_clip_coordinate_count": 0,
        },
        "tuple_boundary": {
            "tuple_count": window.transitions,
            "behavior_controller_is_student": True,
            "teacher_action_present": False,
            "terminal_autoreset_observation_is_next_state": False,
            "last_tuple": {
                "done": True,
                "next_policy_observation_valid": False,
            },
        },
        "frozen_models": _passing_frozen_input_evidence(),
        "partial_failure": None,
    }


def test_pinned_windows_are_exact() -> None:
    expected = {
        "initial510": (9, 510, 0, 518),
        "continuous": (9, 2080, 0, 2088),
        "w0": (9, 500, 0, 508),
        "w1": (409, 500, 100, 908),
        "w2": (809, 500, 100, 1308),
        "w3": (1209, 500, 100, 1708),
        "w4": (1609, 480, 100, 2088),
    }
    assert tuple(expected) == qualification.MODES
    for mode, values in expected.items():
        window = qualification.resolve_student_qualification_window(mode)
        assert (
            window.anchor_q9,
            window.transitions,
            window.burn_in_transitions,
            window.last_q9,
        ) == values


@pytest.mark.parametrize("transitions", [480, 500, 510, 2080])
def test_exact_episode_duration_resolves_requested_ceil(transitions: int) -> None:
    control_dt = student_env.CONTROL_DT_S
    duration = student_env._exact_episode_length_s(control_dt, transitions)  # noqa: SLF001
    assert duration == math.nextafter(transitions * control_dt, -math.inf)
    assert math.ceil(duration / control_dt) == transitions


def test_no_arbitrary_window_is_accepted() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        qualification.resolve_student_qualification_window("q9_123")
    with pytest.raises(ValueError, match="loses final q10"):
        qualification.StudentQualificationWindow("continuous", 9, 2081)


@pytest.mark.parametrize("mode", qualification.MODES)
def test_scope_never_preclaims_tuple_support_or_deploy(mode: str) -> None:
    scope = qualification.qualification_scope(mode)
    assert scope["on_policy_student_tuple_boundary_proven"] is False
    assert scope["on_policy_student_tuple_boundary_status"] == "not_yet_proven"
    assert scope["teacher_queried"] is False
    assert scope["teacher_labels_admitted"] is False
    assert scope["promotion_authorized"] is False
    assert scope["deployment_authorized"] is False
    assert scope["hardware_authorized"] is False


def test_request_rejects_bad_hash_and_manifest_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64 lowercase"):
        qualification.StudentQualificationRequest(
            REPO_ROOT,
            Path("artifacts/g1_true23/x.json"),
            "BAD",
            Path("artifacts/g1_true23/out.json"),
            "initial510",
        )
    request = qualification.StudentQualificationRequest(
        REPO_ROOT,
        tmp_path / "outside.json",
        "0" * 64,
        Path("artifacts/g1_true23/out.json"),
        "initial510",
    )
    with pytest.raises(ValueError, match="stay under"):
        _ = request.manifest_path


def test_contract_binds_final_bc_and_full_schedule_hashes() -> None:
    contract = qualification.qualification_contract(REPO_ROOT, "continuous")
    assert len(qualification.FROZEN_INPUT_NAMES) == 17
    assert "bootstrap_contract" in qualification.FROZEN_INPUT_NAMES
    assert "teacher_checkpoint" in qualification.FROZEN_INPUT_NAMES
    assert qualification.BC_CONTRACT_SHA256 == ("d41d5d05f90c8fef0d88ba89bd4795c8deacafac57893dc8118e70b66db1087f")
    assert contract["model"]["offline_bc_contract_sha256"] == qualification.BC_CONTRACT_SHA256
    assert contract["schedule"]["continuous_sha256"] == qualification.CONTINUOUS_CONTRACT_SHA256
    assert contract["schedule"]["phase_sha256"] == qualification.PHASE_SCHEDULE_SHA256
    assert contract["action"]["application_count"] == 1
    assert contract["action"]["wrapper_clip_actions"] is None
    bootstrap_sources = contract["bootstrap_executed_sources"]
    external_roots = {entry["relative_root"] for entry in bootstrap_sources["trees"]}
    assert "external_dependencies/mjlab/src/mjlab" in external_roots
    assert "external_dependencies/unitree_rl_mjlab/src" in external_roots
    physical_assets = contract["source_materials"]["physical_model_assets"]
    assert physical_assets["file_count"] == 43
    assert physical_assets["extension_counts"] == {".stl": 38, ".xml": 5}
    assert len(physical_assets["files"]) == 43


def test_executed_source_binding_is_sorted_complete_and_tamper_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(path.as_posix() for path in qualification.EXECUTED_SOURCE_RELATIVE_PATHS)
    assert len(paths) == 21
    assert paths == tuple(sorted(paths))
    assert "gear_sonic/utils/g1_23dof_contract.py" in paths
    assert "gear_sonic/utils/g1_23dof_native124_21204_adapter.py" in paths
    assert "gear_sonic/utils/g1_true23_teacher_support.py" in paths
    before = qualification.executed_source_binding(REPO_ROOT)
    assert before["file_count"] == 21

    original_sha256_file = qualification.sha256_file
    tampered = (REPO_ROOT / "gear_sonic/utils/g1_23dof_contract.py").resolve()

    def _tampered_sha256(path: Path) -> str:
        if Path(path).resolve() == tampered:
            return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(qualification, "sha256_file", _tampered_sha256)
    after = qualification.executed_source_binding(REPO_ROOT)
    assert after["file_count"] == 21
    assert after["binding_sha256"] != before["binding_sha256"]
    assert after != before


def test_physical_model_asset_binding_is_complete_and_tamper_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = qualification.physical_model_asset_binding(REPO_ROOT)
    assert before["file_count"] == qualification.PHYSICAL_MODEL_ASSET_FILE_COUNT == 43
    paths = [entry["path"] for entry in before["files"]]
    assert paths == sorted(paths)
    assert before["extension_counts"] == {".stl": 38, ".xml": 5}

    original_sha256_file = qualification.sha256_file
    tampered = (REPO_ROOT / qualification.PHYSICAL_MODEL_ASSET_TREE_RELATIVE_PATH / paths[0]).resolve()

    def _tampered_sha256(path: Path) -> str:
        if Path(path).resolve() == tampered:
            return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(qualification, "sha256_file", _tampered_sha256)
    after = qualification.physical_model_asset_binding(REPO_ROOT)
    assert after["manifest_sha256"] != before["manifest_sha256"]
    assert after != before


def test_preflight_missing_candidate_is_nonready_and_does_not_construct_simulator() -> None:
    request = qualification.StudentQualificationRequest(
        REPO_ROOT,
        Path("artifacts/g1_true23/missing_student_candidate.manifest.json"),
        "0" * 64,
        Path("artifacts/g1_true23/missing_student_candidate_preflight_unused.json"),
        "initial510",
    )
    report = qualification.preflight_student_qualification(request)
    assert report["ready"] is False
    assert report["issues"] == ["candidate_manifest_missing"]
    assert report["safety"]["simulator_constructed"] is False
    assert report["scope"]["on_policy_student_tuple_boundary_proven"] is False


def test_candidate_manifest_parser_accepts_only_closed_loop_eligible_boundary() -> None:
    contract = dict(qualification.load_offline_bc_contract(REPO_ROOT))
    manifest = _candidate_manifest(contract)
    qualification._validate_candidate_manifest_fields(manifest, contract)  # noqa: SLF001

    overclaim = copy.deepcopy(manifest)
    overclaim["boundaries"]["dagger_data"] = True
    with pytest.raises(ValueError, match="overclaim"):
        qualification._validate_candidate_manifest_fields(overclaim, contract)  # noqa: SLF001

    failed_fit = copy.deepcopy(manifest)
    failed_fit["fit"]["offline_fit_gates_passed"] = False
    with pytest.raises(ValueError, match="fit gates"):
        qualification._validate_candidate_manifest_fields(failed_fit, contract)  # noqa: SLF001


def test_observation_arrays_accepts_real_tensordict_keys_without_iteration() -> None:
    tensordict = pytest.importorskip("tensordict")
    tokenizer = torch.zeros((1, qualification.SONIC_TRUE23_TOKENIZER_DIM), dtype=torch.float32)
    tokenizer[0, 0] = 1.0
    tokenizer[0, 1:] = torch.arange(qualification.ENCODER_DIM, dtype=torch.float32)
    policy = torch.arange(qualification.PROPRIO_DIM, dtype=torch.float32).reshape(1, -1)
    critic = torch.zeros((1, student_env.SONIC_TRUE23_CRITIC_DIM), dtype=torch.float32)
    observations = tensordict.TensorDict(
        {"tokenizer": tokenizer, "policy": policy, "critic": critic},
        batch_size=[1],
    )
    encoder267, policy930 = qualification._observation_arrays(observations)  # noqa: SLF001
    np.testing.assert_array_equal(encoder267, tokenizer[0, 1:].numpy())
    np.testing.assert_array_equal(policy930, policy[0].numpy())

    reordered = tensordict.TensorDict(
        {"policy": policy, "tokenizer": tokenizer, "critic": critic},
        batch_size=[1],
    )
    assert tuple(reordered.keys()) != ("tokenizer", "policy", "critic")
    with pytest.raises(ValueError, match="groups drift"):
        qualification._observation_arrays(reordered)  # noqa: SLF001


def test_action_semantics_proves_raw_to_v2_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = np.linspace(-1.0, 1.0, qualification.ACTION_DIM, dtype=np.float32)
    safe, final = safe_target_transform_numpy(raw)
    candidate = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32) + (
        raw[np.asarray(ISAACLAB_TO_MUJOCO_DOF, dtype=np.int64)]
        * np.asarray(HARDWARE_23_ACTION_SCALE, dtype=np.float32)
    )
    chain = {
        "raw_native": torch.from_numpy(raw.reshape(1, -1)),
        "candidate_target_hardware": torch.from_numpy(candidate.reshape(1, -1)),
        "safe_native": torch.from_numpy(safe.reshape(1, -1)),
        "final_target_hardware": torch.from_numpy(final.reshape(1, -1)),
        "raw_clip_mask_native": torch.zeros((1, 23), dtype=torch.bool),
    }
    monkeypatch.setattr(qualification, "capture_student_action_chain", lambda _env: chain)
    report = qualification._action_semantics(object(), raw)  # noqa: SLF001
    assert report["passed"] is True
    assert report["raw_clip_coordinate_count"] == 0
    assert max(report["maximum_absolute_error_by_link"].values()) == 0.0


def test_action_semantics_rejects_double_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = np.full(23, 0.8, dtype=np.float32)
    safe, final = safe_target_transform_numpy(raw)
    _, double_final = safe_target_transform_numpy(safe)
    candidate = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32) + (
        raw[np.asarray(ISAACLAB_TO_MUJOCO_DOF, dtype=np.int64)]
        * np.asarray(HARDWARE_23_ACTION_SCALE, dtype=np.float32)
    )
    chain = {
        "raw_native": torch.from_numpy(raw.reshape(1, -1)),
        "candidate_target_hardware": torch.from_numpy(candidate.reshape(1, -1)),
        "safe_native": torch.from_numpy(safe.reshape(1, -1)),
        "final_target_hardware": torch.from_numpy(double_final.reshape(1, -1)),
        "raw_clip_mask_native": torch.zeros((1, 23), dtype=torch.bool),
    }
    monkeypatch.setattr(qualification, "capture_student_action_chain", lambda _env: chain)
    report = qualification._action_semantics(object(), raw)  # noqa: SLF001
    assert report["passed"] is False
    assert report["maximum_absolute_error_by_link"]["final_target_hardware"] > 0.0
    assert not np.array_equal(final, double_final)


@pytest.mark.parametrize("mode", qualification.MODES)
def test_pure_assessment_passes_only_exact_complete_window(mode: str) -> None:
    window = qualification.resolve_student_qualification_window(mode)
    report = qualification.assess_student_qualification(**_passing_assessment(window))
    assert report["qualified_requested_mode"] is True
    assert report["student_on_policy_tuple_boundary_gate_passed"] is True
    assert report["scope"]["on_policy_student_tuple_boundary_proven"] is True


def test_partial_failure_keeps_tuple_claim_false() -> None:
    window = qualification.resolve_student_qualification_window("initial510")
    inputs = _passing_assessment(window)
    inputs["partial_failure"] = {"qualification_must_fail": True}
    report = qualification.assess_student_qualification(**inputs)
    assert report["qualified_requested_mode"] is False
    assert report["structured_partial_failure_absent"] is False
    assert report["scope"]["on_policy_student_tuple_boundary_proven"] is False


@pytest.mark.parametrize(
    "tampered_name",
    ["support_contract", "bootstrap_contract", "teacher_checkpoint"],
)
def test_parent_artifact_tamper_fails_frozen_input_gate(
    tmp_path: Path,
    tampered_name: str,
) -> None:
    specs: dict[str, dict[str, str]] = {}
    for name in qualification.FROZEN_INPUT_NAMES:
        path = tmp_path / name
        path.write_bytes(f"frozen:{name}".encode())
        specs[name] = {
            "path": str(path),
            "expected_sha256": qualification.sha256_file(path),
        }
    before = qualification._snapshot_preflight_bound_files(specs)  # noqa: SLF001
    (tmp_path / tampered_name).write_bytes(b"tampered during rollout")
    after = qualification._snapshot_preflight_bound_files(specs)  # noqa: SLF001
    frozen = qualification._frozen_input_evidence(before, after)  # noqa: SLF001
    assert before["all_match_expected"] is True
    assert after["all_match_expected"] is False
    assert frozen["all_preflight_bound_inputs_unchanged"] is False
    assert frozen["files"][tampered_name]["unchanged"] is False

    window = qualification.resolve_student_qualification_window("initial510")
    inputs = _passing_assessment(window)
    inputs["frozen_models"] = frozen
    report = qualification.assess_student_qualification(**inputs)
    assert report["qualified_requested_mode"] is False
    assert report["all_preflight_bound_input_files_unchanged_gate_passed"] is False


def test_tuple_digest_marks_terminal_autoreset_as_new_episode() -> None:
    accumulator = qualification._TupleDigestAccumulator()  # noqa: SLF001
    zeros = {
        "encoder267": np.zeros(267, dtype=np.float32),
        "token64": np.zeros(64, dtype=np.float32),
        "policy930": np.zeros(930, dtype=np.float32),
        "decoder994": np.zeros(994, dtype=np.float32),
        "student_raw": np.zeros(23, dtype=np.float32),
    }
    semantics = {
        "chain": {
            "safe_native": [0.0] * 23,
            "final_target_hardware": list(SAFE_TARGET_DEFAULT_Q_HARDWARE),
        }
    }
    physical = {
        "joint_position_hardware23": [0.0] * 23,
        "joint_velocity_hardware23": [0.0] * 23,
        "base_angular_velocity3": [0.0] * 3,
        "torso_quaternion_wxyz4": [1.0, 0.0, 0.0, 0.0],
    }
    accumulator.add(
        local_transition=0,
        q9=9,
        action_semantics=semantics,
        post_physical=physical,
        reward=0.0,
        done=True,
        **zeros,
    )
    report = accumulator.report()
    assert report["tuple_count"] == 1
    assert report["last_tuple"]["next_policy_observation_valid"] is False
    assert report["last_tuple"]["autoreset_observation_is_new_episode"] is True


def test_atomic_writer_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    evidence = root / "artifacts" / "g1_true23"
    evidence.mkdir(parents=True)
    request = qualification.StudentQualificationRequest(
        root,
        Path("artifacts/g1_true23/missing.manifest.json"),
        "0" * 64,
        Path("artifacts/g1_true23/report.json"),
        "initial510",
    )
    output = qualification.write_student_qualification_new(
        request,
        {"qualified_requested_mode": False, "finite": 1.0},
    )
    assert output.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        qualification.write_student_qualification_new(request, {"second": True})


def test_real_mjlab_student_config_constructs_without_rollout() -> None:
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
    assert cfg.observations["tokenizer"].enable_corruption is True
    assert cfg.observations["policy"].enable_corruption is True
    assert audit["safe_target_transform_application_count"] == 1
    assert audit["last_action_q9"] == 518
    assert audit["resolved_max_episode_length"] == 510
    assert audit["domain_randomization"] is False


def test_cuda_wrapper_initial510_constructs_exact_horizon_without_steps() -> None:
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
    cfg.seed = qualification.FIXED_SEED
    env = ManagerBasedRlEnv(cfg=cfg, device=qualification.DEVICE)
    wrapped = None
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        observations = wrapped.get_observations()
        encoder267, policy930 = qualification._observation_arrays(observations)  # noqa: SLF001
        assert tuple(observations.keys()) == ("tokenizer", "policy", "critic")
        assert encoder267.shape == (qualification.ENCODER_DIM,)
        assert encoder267.dtype == np.float32
        assert policy930.shape == (qualification.PROPRIO_DIM,)
        assert policy930.dtype == np.float32
        assert np.isfinite(encoder267).all()
        assert np.isfinite(policy930).all()
        assert float(observations["tokenizer"][0, 0].item()) == 1.0
        assert wrapped.clip_actions is None
        assert env.max_episode_length == 510
        assert wrapped.max_episode_length == 510
        assert int(env.episode_length_buf[0].item()) == 0
        assert int(env.common_step_counter) == 0
        assert int(env._sim_step_counter) == 0  # noqa: SLF001
    finally:
        (wrapped or env).close()
