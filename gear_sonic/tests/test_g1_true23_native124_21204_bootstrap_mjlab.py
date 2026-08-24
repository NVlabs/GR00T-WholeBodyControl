from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.scripts.collect_g1_true23_native124_21204_bootstrap_mjlab import (
    _parser,
)
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    safe_target_transform_numpy,
)
import gear_sonic.utils.g1_true23_native124_21204_bootstrap_mjlab as bootstrap
from gear_sonic.utils.g1_true23_teacher_support import (
    load_teacher_support_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _valid_arrays() -> dict[str, np.ndarray]:
    arrays = {name: np.zeros(shape, dtype=dtype) for name, (dtype, shape) in bootstrap.ARRAY_SPECS.items()}
    row = np.arange(bootstrap.TOTAL_ROWS, dtype=np.int64)
    prefix = row < bootstrap.RESET_PREFIX_ROWS
    arrays["row_index"][:] = row
    arrays["reset_prefix"][:] = prefix
    arrays["steady_history"][:] = ~prefix
    arrays["history_depth"][:] = np.minimum(row + 1, bootstrap.RESET_PREFIX_ROWS)
    arrays["reset_padding_count"][:] = np.maximum(
        bootstrap.RESET_PREFIX_ROWS - 1 - row,
        0,
    )
    arrays["q9_reference_index"][:] = bootstrap.INITIAL_Q9 + row
    arrays["control_state_index"][:] = bootstrap.INITIAL_Q9 + 1 + row
    arrays["next_control_state_index"][:] = bootstrap.INITIAL_Q9 + 2 + row

    rng = np.random.default_rng(20260809)
    arrays["encoder267"][:] = rng.normal(
        0.0,
        0.1,
        arrays["encoder267"].shape,
    ).astype(np.float32)
    arrays["token64"][:] = rng.normal(
        0.0,
        0.1,
        arrays["token64"].shape,
    ).astype(np.float32)
    arrays["proprio930"][:] = rng.normal(
        0.0,
        0.1,
        arrays["proprio930"].shape,
    ).astype(np.float32)
    arrays["decoder994"][:] = np.concatenate(
        (arrays["token64"], arrays["proprio930"]),
        axis=1,
    )
    arrays["selected_observation124"][:] = rng.normal(
        0.0,
        0.1,
        arrays["selected_observation124"].shape,
    ).astype(np.float32)

    token, raw = bootstrap.replay_bootstrap_model_outputs(
        arrays["encoder267"],
        arrays["selected_observation124"],
        repository_root=REPO_ROOT,
    )
    arrays["token64"][:] = token
    arrays["decoder994"][:] = np.concatenate(
        (arrays["token64"], arrays["proprio930"]),
        axis=1,
    )
    arrays["teacher_action_pt_hardware23"][:] = raw
    arrays["teacher_action_onnx_hardware23"][:] = raw
    arrays["teacher_parity_max_abs"][:] = 0.0
    arrays["teacher_action_applied_hardware23"][:] = raw

    support = load_teacher_support_contract(REPO_ROOT)
    composite = support["teacher_composite_contract"]
    home = np.asarray(composite["checkpoint21204_home_q_hardware"], dtype=np.float32)
    scale = np.asarray(
        composite["checkpoint21204_action_scale_hardware"],
        dtype=np.float32,
    )
    candidate = (home + scale * raw).astype(np.float32, copy=False)
    arrays["teacher_candidate_target_hardware23"][:] = candidate
    action = support["student_action_contract"]
    default = np.asarray(action["hardware_default_q"], dtype=np.float32)
    sonic_scale = np.asarray(action["hardware_action_scale"], dtype=np.float32)
    permutation = np.asarray(action["mujoco_to_isaaclab_dof"], dtype=np.int64)
    plain = (((candidate - default) / sonic_scale)[:, permutation]).astype(
        np.float32,
        copy=False,
    )
    safe, target = safe_target_transform_numpy(plain)
    arrays["teacher_label_raw_native23"][:] = plain
    arrays["teacher_applied_safe_native23"][:] = safe
    arrays["teacher_final_target_hardware23"][:] = target
    arrays["raw_clip_mask_native23"][:] = np.abs(plain) >= 10.0
    arrays["max_abs_plain_sonic_raw_native"][:] = np.max(np.abs(plain), axis=1)

    arrays["next_joint_position_hardware23"][:] = target
    arrays["next_joint_velocity_hardware23"][:] = 0.0
    arrays["next_base_angular_velocity3"][:] = 0.0
    arrays["next_torso_quaternion_wxyz4"][:, 0] = 1.0
    arrays["simulator_advanced"][:] = True
    arrays["done"][:] = False
    arrays["terminated"][:] = False
    arrays["timed_out"][:] = False
    arrays["command_resampled"][:] = False
    arrays["base_height_m"][:] = 0.8
    arrays["base_tilt_rad"][:] = 0.1
    arrays["maximum_joint_velocity_ratio"][:] = 0.2
    arrays["maximum_actuator_force_ratio"][:] = 0.3
    arrays["target_tracking_rmse_rad"][:] = 0.2
    return arrays


def _issue_codes(assessment: bootstrap.BootstrapAssessment) -> set[str]:
    return {issue.code for issue in assessment.issues}


def _request(tmp_path: Path) -> bootstrap.BootstrapCollectionRequest:
    evidence = tmp_path / "artifacts" / "g1_true23"
    evidence.mkdir(parents=True)
    return bootstrap.BootstrapCollectionRequest(
        repository_root=tmp_path,
        output_prefix=Path("artifacts/g1_true23/bootstrap_seed20260805"),
    )


def _fake_policy_history(transition: int) -> tuple[SimpleNamespace, dict[str, torch.Tensor]]:
    depth = min(transition + 1, bootstrap.RESET_PREFIX_ROWS)
    padding = bootstrap.RESET_PREFIX_ROWS - depth
    buffers = {}
    flattened = []
    for term_index, (name, width) in enumerate(bootstrap.POLICY_TERM_WIDTHS.items()):
        history = torch.zeros((1, bootstrap.RESET_PREFIX_ROWS, width), dtype=torch.float32)
        if not (transition == 0 and name == "previous_action"):
            history[:] = float(term_index + 1)
            for index in range(padding + 1, bootstrap.RESET_PREFIX_ROWS):
                history[:, index] += float(index - padding)
        buffers[name] = SimpleNamespace(
            max_length=bootstrap.RESET_PREFIX_ROWS,
            current_length=torch.tensor([depth], dtype=torch.long),
            buffer=history,
        )
        flattened.append(history.reshape(1, -1))
    manager = SimpleNamespace(
        _group_obs_term_names={"policy": list(bootstrap.POLICY_TERM_WIDTHS)},
        _group_obs_term_history_buffer={"policy": buffers},
    )
    raw_env = SimpleNamespace(observation_manager=manager)
    return raw_env, {"policy": torch.cat(flattened, dim=-1)}


def test_contract_binds_exact_510_reset_prefix_and_permanent_boundaries() -> None:
    contract = bootstrap.load_bootstrap_contract(REPO_ROOT)
    rows = contract["causal_rows"]
    assert rows["total_rows"] == 510
    assert rows["reset_prefix_rows"] == 10
    assert rows["real_h10_rows"] == 500
    assert rows["initial_q9_reference_index"] == 9
    assert rows["last_action_q9_reference_index"] == 518
    assert rows["last_control_state_index"] == 519
    assert rows["last_next_control_state_index"] == 520
    assert rows["post_collection_q9_reference_index"] == 519
    assert rows["episode_timeout_steps"] == 511
    assert rows["reset_prefix_history_depth"] == list(range(1, 11))
    assert rows["reset_prefix_padding_count"] == list(range(9, -1, -1))
    assert contract["artifact_identity"]["causal_encoder"]["source_checkpoint_required"] is False
    observation = contract["observation_and_label_contract"]
    assert observation["teacher_support_contract_sha256"] == bootstrap.SUPPORT_CONFIG_SHA256
    assert observation["tokenizer_corruption_enabled"] is True
    assert observation["policy_corruption_enabled"] is True
    assert observation["actor_corruption_enabled"] is False
    assert observation["policy_reset_previous_action"] == "zero_native23_until_first_teacher_action"
    assert contract["runtime_binding"]["source_bytes_rehashed_after_collection"] is True
    assert (
        contract["artifact_identity"]["teacher"]["composite_contract_sha256"]
        == bootstrap.COMPOSITE_CONTRACT_SHA256
    )
    boundaries = contract["boundaries"]
    for name in (
        "support_admitted",
        "on_policy_data",
        "dagger_data",
        "student_policy_present",
        "promotion_eligible",
        "deployment_ready",
        "hardware_authorized",
    ):
        assert boundaries[name] is False


def test_clean_510_rows_are_bootstrap_bc_only_and_all_eligible() -> None:
    arrays = _valid_arrays()
    result = bootstrap.assess_bootstrap_arrays(arrays, repository_root=REPO_ROOT)
    assert result.quarantined is False
    assert result.bootstrap_bc_eligible_rows == 510
    report = result.to_dict()
    assert report["reset_prefix_bootstrap_rows"] == 10
    assert report["real_h10_bootstrap_rows"] == 500
    assert report["support_admitted_rows"] == 0
    assert report["on_policy_rows"] == 0
    assert report["dagger_rows"] == 0


def test_actual_policy_history_depth_backfill_and_flattening_are_proved() -> None:
    raw_env, observations = _fake_policy_history(3)
    proof = bootstrap._policy_history_runtime_proof(raw_env, observations, 3)  # noqa: SLF001
    assert proof["actual_history_depth"] == 4
    assert proof["reset_padding_count"] == 6
    assert proof["term_major_policy930_exact"] is True

    raw_env.observation_manager._group_obs_term_history_buffer["policy"]["joint_vel"].current_length[:] = 5
    with pytest.raises(RuntimeError, match="actual history depth mismatch"):
        bootstrap._policy_history_runtime_proof(raw_env, observations, 3)  # noqa: SLF001


def test_actual_policy_history_must_shift_by_exactly_one_frame() -> None:
    previous_env, _ = _fake_policy_history(2)
    current_env, _ = _fake_policy_history(3)
    previous = bootstrap._policy_history_runtime_snapshot(previous_env)  # noqa: SLF001
    current = bootstrap._policy_history_runtime_snapshot(current_env)  # noqa: SLF001
    bootstrap._assert_policy_history_shift(previous, current, 3)  # noqa: SLF001
    current["base_ang_vel"][0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="more or less than once"):
        bootstrap._assert_policy_history_shift(previous, current, 3)  # noqa: SLF001


def test_reset_policy_previous_action_must_be_true_zero() -> None:
    raw_env, observations = _fake_policy_history(0)
    previous = raw_env.observation_manager._group_obs_term_history_buffer["policy"]["previous_action"].buffer
    previous[0, 0, 0] = 0.25
    observations["policy"] = torch.cat(
        [
            raw_env.observation_manager._group_obs_term_history_buffer["policy"][name].buffer.reshape(1, -1)
            for name in bootstrap.POLICY_TERM_WIDTHS
        ],
        dim=-1,
    )
    with pytest.raises(RuntimeError, match="backfill mismatch|previous-action history"):
        bootstrap._policy_history_runtime_proof(raw_env, observations, 0)  # noqa: SLF001


def test_teacher_policy_previous_action_is_zero_only_at_reset() -> None:
    safe = torch.arange(23, dtype=torch.float32).reshape(1, 23) / 10.0
    action_manager = SimpleNamespace(get_term=lambda _name: SimpleNamespace(safe_native_action=safe))
    env = SimpleNamespace(
        action_manager=action_manager,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        num_envs=1,
        device=torch.device("cpu"),
    )
    reset = bootstrap.bootstrap_teacher_policy_previous_action(env)
    assert reset.shape == (1, 29)
    assert not bool(torch.count_nonzero(reset))
    env.episode_length_buf[:] = 1
    after = bootstrap.bootstrap_teacher_policy_previous_action(env)
    assert bool(torch.count_nonzero(after))


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    (
        ("prefix", "reset_prefix_flag_mismatch"),
        ("history", "history_depth_mismatch"),
        ("q9", "q9_reference_index_mismatch"),
        ("decoder", "decoder994_concat_mismatch"),
        ("token", "model250_encoder_token_replay_mismatch"),
        ("onnx_replay", "selected_onnx_action_replay_mismatch"),
        ("parity", "teacher_pt_onnx_parity_exceeded"),
        ("applied", "teacher_applied_action_mismatch"),
        ("target", "teacher_final_target_mismatch"),
        ("done", "done_or_autoreset"),
        ("resampled", "command_resampled"),
        ("soft", "nonzero_target_inner_margin_violation_count"),
        ("height", "base_height_below_gate"),
        ("quaternion", "next_torso_quaternion_not_unit"),
    ),
)
def test_any_row_failure_quarantines_all_510_rows(
    mutation: str,
    expected_issue: str,
) -> None:
    arrays = _valid_arrays()
    row = 17
    if mutation == "prefix":
        arrays["reset_prefix"][row] = ~arrays["reset_prefix"][row]
    elif mutation == "history":
        arrays["history_depth"][row] = 9
    elif mutation == "q9":
        arrays["q9_reference_index"][row] += 1
    elif mutation == "decoder":
        arrays["decoder994"][row, 71] += np.float32(0.01)
    elif mutation == "token":
        arrays["token64"][row, 7] += np.float32(0.01)
        arrays["decoder994"][row, 7] = arrays["token64"][row, 7]
    elif mutation == "onnx_replay":
        arrays["teacher_action_onnx_hardware23"][row, 3] += np.float32(2.0e-5)
        arrays["teacher_action_pt_hardware23"][row, 3] += np.float32(2.0e-5)
        arrays["teacher_action_applied_hardware23"][row, 3] += np.float32(2.0e-5)
    elif mutation == "parity":
        arrays["teacher_action_onnx_hardware23"][row, 3] += np.float32(2.0e-5)
        arrays["teacher_parity_max_abs"][row] = np.float32(2.0e-5)
    elif mutation == "applied":
        arrays["teacher_action_applied_hardware23"][row, 1] += np.float32(0.01)
    elif mutation == "target":
        arrays["teacher_final_target_hardware23"][row, 5] += np.float32(2.0e-5)
    elif mutation == "done":
        arrays["done"][row] = True
    elif mutation == "resampled":
        arrays["command_resampled"][row] = True
    elif mutation == "soft":
        arrays["target_inner_margin_violation_count"][row] = 1
    elif mutation == "height":
        arrays["base_height_m"][row] = np.float32(0.449)
    elif mutation == "quaternion":
        arrays["next_torso_quaternion_wxyz4"][row, 0] = np.float32(0.5)
    else:  # pragma: no cover
        raise AssertionError(mutation)
    result = bootstrap.assess_bootstrap_arrays(arrays, repository_root=REPO_ROOT)
    assert result.quarantined is True
    assert result.bootstrap_bc_eligible_rows == 0
    assert expected_issue in _issue_codes(result)


def test_malformed_or_student_augmented_npz_schema_fails_closed() -> None:
    arrays = _valid_arrays()
    arrays.pop("encoder267")
    with pytest.raises(ValueError, match="array keys mismatch"):
        bootstrap.assess_bootstrap_arrays(arrays, repository_root=REPO_ROOT)

    arrays = _valid_arrays()
    arrays["student_action23"] = np.zeros((510, 23), dtype=np.float32)
    with pytest.raises(ValueError, match="array keys mismatch"):
        bootstrap.assess_bootstrap_arrays(arrays, repository_root=REPO_ROOT)

    arrays = _valid_arrays()
    arrays["token64"] = arrays["token64"].astype(np.float64)
    with pytest.raises(ValueError, match="token64 must be float32"):
        bootstrap.assess_bootstrap_arrays(arrays, repository_root=REPO_ROOT)


def test_no_overwrite_npz_and_compact_manifest_round_trip(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    npz = tmp_path / "bootstrap.npz"
    manifest = tmp_path / "bootstrap.manifest.json"
    published_npz, published_manifest, body = bootstrap.publish_bootstrap_evidence_new(
        arrays,
        npz_path=npz,
        manifest_path=manifest,
        materials={
            "source_checkpoint_present": False,
            "source_checkpoint_verified": False,
            "verification_limitation": "PT source absent; direct ONNX metadata binding only",
        },
        repository_root=REPO_ROOT,
    )
    assert published_npz == npz.resolve()
    assert published_manifest == manifest.resolve()
    assert body["artifact"]["npz_sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest()
    assert body["artifact"]["array_count"] == len(bootstrap.ARRAY_SPECS)
    assert body["qualification"]["bootstrap_bc_eligible_rows"] == 510
    assert body["boundaries"]["support_admitted"] is False
    assert "rows" in body and "records" not in body and "trajectory" not in body

    written = json.loads(manifest.read_text(encoding="utf-8"))
    claimed = written.pop("manifest_payload_sha256")
    assert hashlib.sha256(canonical_json_bytes(written)).hexdigest() == claimed
    with np.load(npz, allow_pickle=False) as archive:
        assert set(archive.files) == set(bootstrap.ARRAY_SPECS)
        assert np.array_equal(archive["decoder994"], arrays["decoder994"])
        assert np.array_equal(archive["reset_prefix"][:10], np.ones(10, dtype=bool))
        assert not np.any(archive["reset_prefix"][10:])
    admitted, admitted_manifest = bootstrap.load_bootstrap_training_candidate(
        npz,
        manifest,
        repository_root=REPO_ROOT,
    )
    assert np.array_equal(admitted["decoder994"], arrays["decoder994"])
    assert admitted_manifest["artifact"]["strict_loader_admissible"] is True

    with pytest.raises(FileExistsError, match="overwrite"):
        bootstrap.publish_bootstrap_evidence_new(
            arrays,
            npz_path=npz,
            manifest_path=manifest,
            materials={},
            repository_root=REPO_ROOT,
        )


def test_quarantined_full_npz_never_admits_partial_rows(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    arrays["done"][509] = True
    _, manifest, body = bootstrap.publish_bootstrap_evidence_new(
        arrays,
        npz_path=tmp_path / "quarantine.npz",
        manifest_path=tmp_path / "quarantine.manifest.json",
        materials={},
        repository_root=REPO_ROOT,
    )
    assert manifest.is_file()
    qualification = body["qualification"]
    assert qualification["whole_run_quarantined"] is True
    assert qualification["bootstrap_bc_eligible_rows"] == 0
    assert qualification["reset_prefix_bootstrap_rows"] == 0
    assert qualification["real_h10_bootstrap_rows"] == 0
    assert body["artifact"]["classification"] == "quarantined_diagnostic_only"
    with pytest.raises(ValueError, match="rejects quarantined"):
        bootstrap.load_bootstrap_training_candidate(
            tmp_path / "quarantine.npz",
            manifest,
            repository_root=REPO_ROOT,
        )


def test_runtime_failure_manifest_has_no_npz_and_no_support_claim(tmp_path: Path) -> None:
    request = _request(tmp_path)
    failure = bootstrap.write_bootstrap_failure_manifest_new(
        request,
        RuntimeError("sim stopped"),
    )
    assert failure.is_file()
    assert not Path(f"{request.prefix}.npz").exists()
    report = json.loads(failure.read_text(encoding="utf-8"))
    assert report["whole_run_quarantined"] is True
    assert report["bootstrap_bc_eligible_rows"] == 0
    assert report["partial_npz_published"] is False
    assert report["boundaries"]["support_admitted"] is False
    assert report["boundaries"]["dagger_data"] is False
    assert not list(failure.parent.glob(".teacher-bootstrap-*.tmp"))


def test_executed_source_binding_is_compact_and_deterministic() -> None:
    first = bootstrap.executed_bootstrap_source_binding(REPO_ROOT)
    second = bootstrap.executed_bootstrap_source_binding(REPO_ROOT)
    assert first == second
    assert len(first["binding_sha256"]) == 64
    assert first["files"]["file_count"] == len(bootstrap.EXECUTED_SOURCE_FILE_RELATIVE_PATHS)
    assert [tree["relative_root"] for tree in first["trees"]] == [
        path.as_posix() for path in bootstrap.EXECUTED_SOURCE_TREE_RELATIVE_PATHS
    ]


def test_request_scope_and_cli_have_no_seed_device_or_threshold_knobs(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    assert request.npz_path.name.endswith(".npz")
    assert request.manifest_path.name.endswith(".manifest.json")
    outside = bootstrap.BootstrapCollectionRequest(
        repository_root=tmp_path,
        output_prefix=Path("outside/bootstrap"),
    )
    with pytest.raises(ValueError, match="must stay under"):
        _ = outside.prefix

    parser = _parser()
    preflight = parser.parse_args(["preflight", "--output-prefix", "artifacts/g1_true23/example"])
    assert preflight.command == "preflight"
    collect = parser.parse_args(
        [
            "collect",
            "--output-prefix",
            "artifacts/g1_true23/example",
            "--execute-cuda-rollout",
        ]
    )
    assert collect.execute_cuda_rollout is True
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--seed" not in option_strings
    assert "--device" not in option_strings
    assert "--threshold" not in option_strings


def test_source_checkpoint_limitation_is_explicit_and_currently_real() -> None:
    contract = bootstrap.load_bootstrap_contract(REPO_ROOT)
    encoder = contract["artifact_identity"]["causal_encoder"]
    source = REPO_ROOT / encoder["bundle_directory"] / encoder["source_checkpoint_filename"]
    assert encoder["source_checkpoint_required"] is False
    assert encoder["source_checkpoint_sha256"] == bootstrap.CAUSAL_SOURCE_CHECKPOINT_SHA256
    assert not source.exists()
