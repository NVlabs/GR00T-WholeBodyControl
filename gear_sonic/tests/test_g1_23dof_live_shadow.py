import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from gear_sonic.utils import (
    g1_23dof_live_shadow as live,
    g1_23dof_semantic_reference as semantic,
)
from gear_sonic.utils.g1_23dof_contract import (
    DEFAULT_REFERENCE_PROFILE,
    reference_profile_contract,
)


def _terms(
    source_ns: int,
    frame_index: int = 0,
    *,
    command: list[float] | None = None,
    profile: str = DEFAULT_REFERENCE_PROFILE,
) -> dict:
    reference_contract = reference_profile_contract(profile)
    return {
        "schema_version": live.ENCODER_TERMS_SCHEMA_VERSION,
        "kind": live.ENCODER_TERMS_KIND,
        "pico_source_frame_index": frame_index,
        "pico_source_monotonic_ns": source_ns,
        "future_frame_offsets_s": list(
            reference_contract["future_frame_offsets_s"]
        ),
        "command_multi_future_lower_body": (
            [0.0] * 240 if command is None else list(command)
        ),
        "vr_3point_local_target": [0.0] * 9,
        "vr_3point_local_orn_target": [0.0] * 12,
        "motion_anchor_ori_b": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    }


def _term_major_history(frames: list[list[float]]) -> list[float]:
    return [
        *(value for frame in frames for value in frame[0:3]),
        *(value for frame in frames for value in frame[3:32]),
        *(value for frame in frames for value in frame[32:61]),
        *(value for frame in frames for value in frame[61:90]),
        *(value for frame in frames for value in frame[90:93]),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_zero_onnx(
    path: Path,
    *,
    input_name: str,
    input_dim: int,
    output_name: str,
    output_dim: int,
) -> None:
    import onnx
    from onnx import TensorProto, helper

    value = helper.make_tensor(
        name=f"{output_name}_constant",
        data_type=TensorProto.FLOAT,
        dims=[1, output_dim],
        vals=[0.0] * output_dim,
    )
    graph = helper.make_graph(
        [helper.make_node("Constant", [], [output_name], value=value)],
        "zero_model",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, input_dim])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, output_dim])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)


def test_encoder_observation_rejects_past_window() -> None:
    reference_contract = reference_profile_contract(DEFAULT_REFERENCE_PROFILE)
    terms = _terms(1)
    observation = live.build_encoder_observation(
        terms,
        expected_reference_contract=reference_contract,
    )
    assert len(observation) == 267
    assert observation[240:249] == [0.0] * 9
    terms["future_frame_offsets_s"] = [-(index / 50.0) for index in range(10)]
    with pytest.raises(ValueError, match="artifact reference profile"):
        live.build_encoder_observation(
            terms,
            expected_reference_contract=reference_contract,
        )


def test_term_major_history_missing_slots() -> None:
    first = live.build_proprio_frame(
        hardware_q=live.HARDWARE_DEFAULT_Q,
        hardware_dq=[0.0] * 23,
        imu_gyroscope=[1.0, 2.0, 3.0],
        imu_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        previous_action_native=[0.0] * 23,
    )
    last = first.copy()
    last[0:3] = [4.0, 5.0, 6.0]
    last[3] = 0.25
    history = _term_major_history([first] * 9 + [last])
    live.validate_proprio_history(history)
    assert history[0:3] == [1.0, 2.0, 3.0]
    assert history[27:30] == [4.0, 5.0, 6.0]
    assert history[live.JOINT_POSITION_HISTORY_OFFSET + 9 * 29] == 0.25
    history[live.JOINT_VELOCITY_HISTORY_OFFSET + 4 * 29 + 25] = 1.0
    with pytest.raises(ValueError, match="fixed slot"):
        live.validate_proprio_history(history)


def test_cpp_shadow_gate_does_not_claim_external_training_proof() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = (
        repo_root
        / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/"
        "g1_true23_shadow_gate.cpp"
    ).read_text(encoding="utf-8")
    assert "Exact trained + simulation-validated" not in source
    assert "did not open the training checkpoint" in source
    assert "external Python readiness verification" in source


def test_complete_evidence_is_term_major_and_fail_closed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / live.LIVE_PRODUCER_SOURCE
    source_path.parent.mkdir(parents=True)
    source_path.write_text("// pinned source\n", encoding="utf-8")
    core_path = repo_root / live.LIVE_PRODUCER_CORE
    core_path.parent.mkdir(parents=True)
    core_path.write_text("// pinned core\n", encoding="utf-8")
    audit_path = repo_root / live.LIVE_PRODUCER_AUDIT
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text("# pinned audit\n", encoding="utf-8")
    producer = tmp_path / live.LIVE_PRODUCER_FILENAME
    producer.write_bytes(b"\x7fELF" + b"\0" * 2048)
    approval_path = tmp_path / "approved_live_shadow.json"
    approval_path.write_text(
        json.dumps(
            {
                "schema_version": live.APPROVAL_SCHEMA_VERSION,
                "kind": live.APPROVAL_KIND,
                "promotion_enabled": True,
                "evidence_schema_version": live.LIVE_EVIDENCE_SCHEMA_VERSION,
                "producer_kind": live.LIVE_PRODUCER_KIND,
                "producer_filename": live.LIVE_PRODUCER_FILENAME,
                "binary_format": "ELF",
                "source": {
                    "relpath": live.LIVE_PRODUCER_SOURCE,
                    "sha256": _sha256(source_path),
                },
                "core": {
                    "relpath": live.LIVE_PRODUCER_CORE,
                    "sha256": _sha256(core_path),
                },
                "binary_sha256": _sha256(producer),
                "audit": {
                    "relpath": live.LIVE_PRODUCER_AUDIT,
                    "sha256": _sha256(audit_path),
                },
                "review_state": "approved",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts = {}
    for key in sorted(live._ARTIFACT_HASH_KEYS):
        path = tmp_path / key
        path.write_bytes(key.encode())
        artifacts[key] = path
    reference_profile = DEFAULT_REFERENCE_PROFILE
    expected_reference_contract = reference_profile_contract(reference_profile)
    artifacts["metadata"].write_text(
        json.dumps(
            {
                "schema_version": live.ARTIFACT_SCHEMA_VERSION,
                "reference_profile": reference_profile,
                "reference_contract": expected_reference_contract,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_zero_onnx(
        artifacts["encoder_onnx"],
        input_name="teleop_obs",
        input_dim=267,
        output_name="token",
        output_dim=64,
    )
    _write_zero_onnx(
        artifacts["decoder_onnx"],
        input_name="obs_dict",
        input_dim=994,
        output_name="action",
        output_dim=23,
    )

    start_ns = 1_000_000_000
    times = [start_ns + index * 20_000_000 for index in range(live.MIN_LIVE_SAMPLES)]
    checked_in_motion = (
        Path(__file__).resolve().parents[2]
        / "gear_sonic_deploy"
        / "reference"
        / "example"
        / "squat_001__A359"
    )
    semantic_source_relpath = (
        "gear_sonic_deploy/reference/example/squat_001__A359"
    )
    replay_motion = repo_root / semantic_source_relpath
    replay_motion.mkdir(parents=True)
    for filename in ("joint_pos.csv", "joint_vel.csv"):
        (replay_motion / filename).write_bytes(
            (checked_in_motion / filename).read_bytes()
        )
    pico_samples = []
    lowstate_samples = []
    inference_samples = []
    expected_frames: list[list[float]] = []
    for index, timestamp in enumerate(times):
        pico_samples.append(
            {
                "monotonic_ns": timestamp,
                "body_source_frame_index": index,
                "body_source_ns": timestamp,
                "body_pose": [0.0] * 168,
                "tracker_ids": [1, 2],
                "tracking_state": "BT_VALID",
                "calibrated": True,
                "left_source_ns": timestamp,
                "left_pose": [0.0] * 7,
                "left_tracking_bits": 3,
                "right_source_ns": timestamp,
                "right_pose": [0.0] * 7,
                "right_tracking_bits": 3,
            }
        )
        lowstate = {
            "monotonic_ns": timestamp,
            "tick": index,
            "mode_machine": 4,
            "crc_expected": 7,
            "crc_computed": 7,
            "hardware_joint_ids": list(live.HARDWARE_JOINT_IDS),
            "q": list(live.HARDWARE_DEFAULT_Q),
            "dq": [0.0] * 23,
            "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "imu_gyroscope": [float(index), 0.0, 0.0],
        }
        lowstate_samples.append(lowstate)
        frame = live.build_proprio_frame(
            hardware_q=lowstate["q"],
            hardware_dq=lowstate["dq"],
            imu_gyroscope=lowstate["imu_gyroscope"],
            imu_quaternion_wxyz=lowstate["imu_quaternion_wxyz"],
            previous_action_native=[0.0] * 23,
        )
        if not expected_frames:
            expected_frames = [frame.copy() for _ in range(10)]
        else:
            expected_frames = [*expected_frames[1:], frame]
        history = _term_major_history(expected_frames)
        action = [0.0] * 23
        bounds = live._output_bounds(
            action,
            None if index == 0 else action,
            None if index == 0 else 0.02,
        )
        semantic_reference_window = semantic.build_recorded_reference_window(
            replay_motion,
            profile=reference_profile,
            playback_frame_index=index,
            playback_epoch_monotonic_ns=start_ns,
            emitted_monotonic_ns=timestamp,
            source_relpath=semantic_source_relpath,
        )
        encoder_terms = _terms(
            timestamp,
            index,
            command=semantic_reference_window[
                "command_multi_future_lower_body"
            ],
            profile=reference_profile,
        )
        observation = live.build_encoder_observation(
            encoder_terms,
            expected_reference_contract=expected_reference_contract,
        )
        token = [0.0] * 64
        inference_samples.append(
            {
                "monotonic_ns": timestamp,
                "pico_body_source_ns": timestamp,
                "lowstate_tick": index,
                "semantic_reference_window": semantic_reference_window,
                "encoder_terms": encoder_terms,
                "observation": observation,
                "token": token,
                "proprio_history": history,
                "decoder_input": [*token, *history],
                "previous_action_native": action.copy(),
                "native_action": action.copy(),
                "hardware_action": action.copy(),
                "output_bounds": bounds,
            }
        )

    source_contract = {
        "encoder_terms_kind": live.ENCODER_TERMS_KIND,
        "encoder_terms_schema_version": live.ENCODER_TERMS_SCHEMA_VERSION,
        "semantic_reference_window_kind": (
            semantic.SEMANTIC_REFERENCE_WINDOW_KIND
        ),
        "semantic_reference_window_schema_version": (
            semantic.SEMANTIC_REFERENCE_SCHEMA_VERSION
        ),
        "reference_profile": reference_profile,
        "reference_contract": expected_reference_contract,
        "command_semantics": (
            "artifact_profile_positions120_then_velocities120"
        ),
        "encoder_term_order": list(live.TELEOP_ENCODER_INPUT_TERM_ORDER),
        "decoder_layout": (
            "token64_then_term_major_"
            "angvel_h10_qrel_h10_dq_h10_previous_action_h10_gravity_h10_"
            "each_term_oldest_to_newest"
        ),
        "missing_canonical_il29_slots": list(live.SOURCE_IL29_EXCLUDED_INDICES),
        "control_period_s": 0.02,
    }
    evidence = {
        "schema_version": live.LIVE_EVIDENCE_SCHEMA_VERSION,
        "kind": live.LIVE_EVIDENCE_KIND,
        "captured_at_utc": "2026-07-30T00:00:00Z",
        "producer": {
            "kind": live.LIVE_PRODUCER_KIND,
            "version": live.LIVE_PRODUCER_VERSION,
            "filename": live.LIVE_PRODUCER_FILENAME,
            "sha256": _sha256(producer),
            "binary_format": "ELF",
            "source_relpath": live.LIVE_PRODUCER_SOURCE,
            "source_sha256": _sha256(source_path),
        },
        "artifact_hashes": {key: _sha256(path) for key, path in artifacts.items()},
        "source_contract": source_contract,
        "window": {
            "start_monotonic_ns": times[0],
            "end_monotonic_ns": times[-1],
        },
        "pico_samples": pico_samples,
        "lowstate_samples": lowstate_samples,
        "inference_samples": inference_samples,
        "summary": {
            "sample_count": live.MIN_LIVE_SAMPLES,
            "semantic_reference_window_count": live.MIN_LIVE_SAMPLES,
            "reference_profile": reference_profile,
            "reference_horizon_s": expected_reference_contract["horizon_s"],
            "reference_source_kind": semantic.SOURCE_RECORDED_MOTION,
            "pico_effective_hz": 50.0,
            "lowstate_effective_hz": 50.0,
            "inference_effective_hz": 50.0,
            "all_outputs_finite": True,
            "target_limit_violation_count": 0,
            "target_slew_violation_count": 0,
            "normalized_action_max_abs": 0.0,
            "target_slew_ratio_max": 0.0,
            "onnx_replay_sample_count": live.MIN_LIVE_SAMPLES,
            "onnx_replay_atol": live.ONNX_REPLAY_ATOL,
            "onnx_replay_rtol": live.ONNX_REPLAY_RTOL,
            "missing_required_fields": [],
        },
        "authorization": {
            "lowcmd_publisher_present": False,
            "command_writer_present": False,
            "motion_switcher_present": False,
            "robot_mutation_authorized": False,
        },
    }
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["promotion_enabled"] = False
    approval["review_state"] = "blocked_missing_semantic_pico_future_terms"
    approval_path.write_text(json.dumps(approval, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="promotion is disabled"):
        live.validate_live_shadow_evidence(
            evidence,
            producer_path=producer,
            artifact_paths=artifacts,
            repo_root=repo_root,
            now_utc=now,
            approval_manifest_path=approval_path,
        )
    approval["promotion_enabled"] = True
    approval["review_state"] = "approved"
    approval_path.write_text(json.dumps(approval, sort_keys=True), encoding="utf-8")

    summary = live.validate_live_shadow_evidence(
        evidence,
        producer_path=producer,
        artifact_paths=artifacts,
        repo_root=repo_root,
        now_utc=now,
        approval_manifest_path=approval_path,
    )
    assert summary["sample_count"] == live.MIN_LIVE_SAMPLES
    assert summary["target_limit_violation_count"] == 0
    assert summary["reference_profile"] == reference_profile
    assert summary["semantic_reference_window_count"] == live.MIN_LIVE_SAMPLES

    valid_metadata_bytes = artifacts["metadata"].read_bytes()
    mismatched_metadata = json.loads(valid_metadata_bytes)
    mismatched_metadata["reference_profile"] = (
        semantic.PROFILE_RELEASED_LOW_LATENCY_STEP1
    )
    artifacts["metadata"].write_text(
        json.dumps(mismatched_metadata, sort_keys=True),
        encoding="utf-8",
    )
    evidence["artifact_hashes"]["metadata"] = _sha256(artifacts["metadata"])
    with pytest.raises(ValueError, match="reference_contract does not match profile"):
        live.validate_live_shadow_evidence(
            evidence,
            producer_path=producer,
            artifact_paths=artifacts,
            repo_root=repo_root,
            now_utc=now,
            approval_manifest_path=approval_path,
        )
    artifacts["metadata"].write_bytes(valid_metadata_bytes)
    evidence["artifact_hashes"]["metadata"] = _sha256(artifacts["metadata"])

    sample = evidence["inference_samples"][3]
    pristine_recorded_window = copy.deepcopy(sample["semantic_reference_window"])
    pristine_encoder_terms = copy.deepcopy(sample["encoder_terms"])
    pristine_observation = list(sample["observation"])
    sample["semantic_reference_window"]["joint_pos_lower_body"][0] += 0.25
    sample["semantic_reference_window"]["command_multi_future_lower_body"][0] += 0.25
    sample["encoder_terms"]["command_multi_future_lower_body"][0] += 0.25
    sample["observation"][0] += 0.25
    with pytest.raises(ValueError, match="positions disagree with source CSV"):
        live.validate_live_shadow_evidence(
            evidence,
            producer_path=producer,
            artifact_paths=artifacts,
            repo_root=repo_root,
            now_utc=now,
            approval_manifest_path=approval_path,
        )
    sample["semantic_reference_window"] = pristine_recorded_window
    sample["encoder_terms"] = pristine_encoder_terms
    sample["observation"] = pristine_observation

    recorded_window = sample["semantic_reference_window"]
    fabricated_planner_window = copy.deepcopy(recorded_window)
    fabricated_planner_window["source"].update(
        {
            "kind": semantic.SOURCE_PLANNER,
            "source_relpath": None,
            "identity_sha256": "a" * 64,
            "temporal_semantics": "precomputed_future",
            "joint_pos_sha256": None,
            "joint_vel_sha256": None,
        }
    )
    sample["semantic_reference_window"] = fabricated_planner_window
    with pytest.raises(ValueError, match="approved process-bound producer"):
        live.validate_live_shadow_evidence(
            evidence,
            producer_path=producer,
            artifact_paths=artifacts,
            repo_root=repo_root,
            now_utc=now,
            approval_manifest_path=approval_path,
        )
    sample["semantic_reference_window"] = recorded_window

    low_latency_window = semantic.build_recorded_reference_window(
        replay_motion,
        profile=semantic.PROFILE_RELEASED_LOW_LATENCY_STEP1,
        playback_frame_index=3,
        playback_epoch_monotonic_ns=start_ns,
        emitted_monotonic_ns=times[3],
        source_relpath=semantic_source_relpath,
    )
    original_terms = sample["encoder_terms"]
    original_observation = sample["observation"]
    low_latency_terms = _terms(
        times[3],
        3,
        command=low_latency_window["command_multi_future_lower_body"],
        profile=semantic.PROFILE_RELEASED_LOW_LATENCY_STEP1,
    )
    sample["semantic_reference_window"] = low_latency_window
    sample["encoder_terms"] = low_latency_terms
    sample["observation"] = live.build_encoder_observation(
        low_latency_terms,
        expected_reference_contract=reference_profile_contract(
            semantic.PROFILE_RELEASED_LOW_LATENCY_STEP1
        ),
    )
    with pytest.raises(ValueError, match="disagrees with artifact metadata"):
        live.validate_live_shadow_evidence(
            evidence,
            producer_path=producer,
            artifact_paths=artifacts,
            repo_root=repo_root,
            now_utc=now,
            approval_manifest_path=approval_path,
        )
    sample["semantic_reference_window"] = recorded_window
    sample["encoder_terms"] = original_terms
    sample["observation"] = original_observation

    jumped_frame_index = 23
    jumped_epoch_ns = (
        times[3] - jumped_frame_index * semantic.SOURCE_SAMPLE_PERIOD_NS
    )
    jumped_window = semantic.build_recorded_reference_window(
        replay_motion,
        profile=reference_profile,
        playback_frame_index=jumped_frame_index,
        playback_epoch_monotonic_ns=jumped_epoch_ns,
        emitted_monotonic_ns=times[3],
        source_relpath=semantic_source_relpath,
    )
    jumped_terms = _terms(
        times[3],
        3,
        command=jumped_window["command_multi_future_lower_body"],
        profile=reference_profile,
    )
    sample["semantic_reference_window"] = jumped_window
    sample["encoder_terms"] = jumped_terms
    sample["observation"] = live.build_encoder_observation(
        jumped_terms,
        expected_reference_contract=expected_reference_contract,
    )
    with pytest.raises(ValueError, match="playback epoch changed"):
        live.validate_live_shadow_evidence(
            evidence,
            producer_path=producer,
            artifact_paths=artifacts,
            repo_root=repo_root,
            now_utc=now,
            approval_manifest_path=approval_path,
        )
    sample["semantic_reference_window"] = recorded_window
    sample["encoder_terms"] = original_terms
    sample["observation"] = original_observation

    sample["encoder_terms"]["pico_source_frame_index"] += 1
    with pytest.raises(ValueError, match="PICO capture binding mismatch"):
        live.validate_live_shadow_evidence(
            evidence,
            producer_path=producer,
            artifact_paths=artifacts,
            repo_root=repo_root,
            now_utc=now,
            approval_manifest_path=approval_path,
        )
    sample["encoder_terms"]["pico_source_frame_index"] -= 1

    evidence["inference_samples"][3]["token"][0] = 0.25
    with pytest.raises(ValueError, match="CPU ONNX replay"):
        live.validate_live_shadow_evidence(
            evidence,
            producer_path=producer,
            artifact_paths=artifacts,
            repo_root=repo_root,
            now_utc=now,
            approval_manifest_path=approval_path,
        )
    evidence["inference_samples"][3]["token"][0] = 0.0

    evidence["inference_samples"][3]["native_action"][0] = 0.25
    with pytest.raises(ValueError, match="CPU ONNX replay"):
        live.validate_live_shadow_evidence(
            evidence,
            producer_path=producer,
            artifact_paths=artifacts,
            repo_root=repo_root,
            now_utc=now,
            approval_manifest_path=approval_path,
        )
    evidence["inference_samples"][3]["native_action"][0] = 0.0

    older_term_indices = (
        live.ANGULAR_VELOCITY_HISTORY_OFFSET,
        live.JOINT_POSITION_HISTORY_OFFSET,
        live.JOINT_VELOCITY_HISTORY_OFFSET,
        live.GRAVITY_HISTORY_OFFSET,
    )
    for history_index in older_term_indices:
        evidence["inference_samples"][12]["proprio_history"][history_index] += 0.25
        evidence["inference_samples"][12]["decoder_input"][64 + history_index] += 0.25
        with pytest.raises(ValueError, match="sequential term-major"):
            live.validate_live_shadow_evidence(
                evidence,
                producer_path=producer,
                artifact_paths=artifacts,
                repo_root=repo_root,
                now_utc=now,
                approval_manifest_path=approval_path,
            )
        evidence["inference_samples"][12]["proprio_history"][history_index] -= 0.25
        evidence["inference_samples"][12]["decoder_input"][64 + history_index] -= 0.25

    evidence["inference_samples"][0]["proprio_history"][
        live.PREVIOUS_ACTION_HISTORY_OFFSET + 8
    ] = 1.0
    with pytest.raises(ValueError, match="fixed slot"):
        live.validate_live_shadow_evidence(
            evidence,
            producer_path=producer,
            artifact_paths=artifacts,
            repo_root=repo_root,
            now_utc=now,
            approval_manifest_path=approval_path,
        )
