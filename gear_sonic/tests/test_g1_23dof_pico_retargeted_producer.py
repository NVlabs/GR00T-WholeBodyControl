from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gear_sonic.scripts import probe_g1_23dof_pico_retargeter as probe_cli
from gear_sonic.utils import g1_23dof_pico_retargeted_producer as producer
from gear_sonic.utils.g1_23dof_contract import (
    REFERENCE_PROFILE_LOW_LATENCY,
    REFERENCE_PROFILE_NORMAL,
    SOURCE_IL29_JOINT_NAMES,
)
from gear_sonic.utils.g1_23dof_semantic_reference import (
    SOURCE_RETARGETED_DELAYED,
    SOURCE_SAMPLE_PERIOD_NS,
)


def _body_positions(frame_index: int) -> list[list[float]]:
    x = frame_index * 0.001
    positions = [
        [x, 1.00, 0.00],
        [x - 0.10, 0.95, 0.00],
        [x + 0.10, 0.95, 0.00],
        [x, 1.15, 0.00],
        [x - 0.10, 0.55, 0.00],
        [x + 0.10, 0.55, 0.00],
        [x, 1.35, 0.00],
        [x - 0.10, 0.12, 0.00],
        [x + 0.10, 0.12, 0.00],
        [x, 1.52, 0.00],
        [x - 0.10, 0.06, 0.16],
        [x + 0.10, 0.06, 0.16],
        [x, 1.67, 0.00],
        [x - 0.12, 1.54, 0.00],
        [x + 0.12, 1.54, 0.00],
        [x, 1.84, 0.00],
        [x - 0.24, 1.54, 0.00],
        [x + 0.24, 1.54, 0.00],
        [x - 0.43, 1.37, 0.00],
        [x + 0.43, 1.37, 0.00],
        [x - 0.57, 1.20, 0.00],
        [x + 0.57, 1.20, 0.00],
        [x - 0.62, 1.16, 0.00],
        [x + 0.62, 1.16, 0.00],
    ]
    assert len(positions) == 24
    return positions


def _bodytracking_snapshot(
    frame_index: int,
    *,
    source_start_ns: int = 3_000_000_000,
) -> dict:
    timestamp_ns = source_start_ns + frame_index * SOURCE_SAMPLE_PERIOD_NS
    positions = _body_positions(frame_index)
    identity = [0.0, 0.0, 0.0, 1.0]
    poses = [[*position, *identity] for position in positions]
    return {
        "contract": producer.ATOMIC_SNAPSHOT_CONTRACT,
        "derivative_layout_contract": producer.DERIVATIVE_LAYOUT_CONTRACT,
        "source_coherence_contract": producer.SOURCE_COHERENCE_CONTRACT,
        "body_timestamp_contract": (
            producer.LOCAL_POSE_BODY_TIMESTAMP_CONTRACT
        ),
        "joint_timestamp_contract": producer.POSITIVE_JOINT_TIMESTAMP_CONTRACT,
        "available": True,
        "timestamp_ns": timestamp_ns,
        "sample_sequence": frame_index + 1,
        "poses": poses,
        "velocities": [[0.05, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(24)],
        "accelerations": [[0.0] * 6 for _ in range(24)],
        "joint_timestamps_ns": [timestamp_ns] * 24,
        "health_supported": True,
        "health_available": True,
        "health_valid": True,
        "health_schema_version": 1,
        "health_sample_sequence": frame_index + 1,
        "health_timestamp_ns": timestamp_ns,
        "health_client_build": "xrobotoolkit-pico-health-v1",
        "health_calibration_result": 0,
        "health_calibrated": True,
        "health_tracking_mode": producer.BODY_TRACKING_MODE_CODE,
        "health_connect_state_result": 0,
        "health_tracker_count": 2,
        "health_unique_tracker_count": 2,
        "health_body_state_result": 0,
        "health_is_tracking": True,
        "health_tracking_state_code": producer.BODY_TRACKING_SUCCESS_CODE,
        "health_body_state_code": producer.BODY_TRACKING_VALID_STATE_CODE,
        "health_body_error_code": 0,
        "health_connected_band_count": 2,
        "health_body_data_result": 0,
        "health_body_role_count": 24,
    }


def _capture(frame_count: int = 49) -> dict:
    host_start_ns = 5_000_000_000
    frames = []
    for index in range(frame_count):
        frames.append(
            producer.raw_frame_from_bodytracking_xrt_snapshot(
                _bodytracking_snapshot(index),
                frame_index=index,
                capture_monotonic_ns=(
                    host_start_ns + index * SOURCE_SAMPLE_PERIOD_NS
                ),
            )
        )
    return {
        "schema_version": producer.RAW_CAPTURE_SCHEMA_VERSION,
        "kind": producer.RAW_CAPTURE_KIND,
        "session_id": "synthetic-kinematically-valid-xr24",
        "source": {
            "atomic_snapshot_contract": producer.ATOMIC_SNAPSHOT_CONTRACT,
            "xrobotoolkit_sdk_sha256": "a" * 64,
            "pc_service_sha256": "b" * 64,
            "pico_client_build": "xrobotoolkit-pico-health-v1",
            "pico_client_apk_sha256": "d" * 64,
            "derivative_layout_contract": producer.DERIVATIVE_LAYOUT_CONTRACT,
            "sdk_derivatives_control_usable": False,
            "control_derivative_contract": (
                producer.POSITION_DERIVED_CONTROL_DERIVATIVE_CONTRACT
            ),
            "source_coherence_contract": producer.SOURCE_COHERENCE_CONTRACT,
            "body_timestamp_contract": (
                producer.LOCAL_POSE_BODY_TIMESTAMP_CONTRACT
            ),
            "joint_timestamp_contract": (
                producer.POSITIVE_JOINT_TIMESTAMP_CONTRACT
            ),
            "ankle_role_semantics": producer.ANKLE_ROLE_SEMANTICS,
            "ankle_role_indices": dict(producer.ANKLE_ROLE_TO_BODY_INDEX),
            "body_joint_order": list(producer.XRT_BODY_JOINT_NAMES),
        },
        "frames": frames,
        "authorization": {
            "read_only": True,
            "dds_opened": False,
            "robot_channel_opened": False,
            "actuation_authorized": False,
            "robot_commands_published": False,
        },
    }


def _recorded_il29_positions(count: int) -> list[list[float]]:
    motion_dir = (
        Path(__file__).resolve().parents[2]
        / "gear_sonic_deploy"
        / "reference"
        / "example"
        / "squat_001__A359"
    )
    with (motion_dir / "joint_pos.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.reader(stream)
        assert next(reader) == [f"joint_{index}" for index in range(29)]
        rows = [[float(value) for value in row] for row in reader]
    assert len(rows) >= count
    return rows[:count]


def _mj29_from_il29(il29: list[float]) -> list[float]:
    return [
        il29[target_index]
        for target_index in producer.SOMA_MJ29_TO_CANONICAL_IL29
    ]


def _trace(capture: dict, *, sample_count: int = 48) -> dict:
    capture_summary = producer.validate_raw_capture(capture)
    il29_positions = _recorded_il29_positions(sample_count)
    samples = []
    for index, il29 in enumerate(il29_positions):
        velocity = None
        if index + 1 < sample_count:
            velocity = [
                (next_value - current_value) * 50.0
                for current_value, next_value in zip(
                    il29,
                    il29_positions[index + 1],
                    strict=True,
                )
            ]
        samples.append(
            {
                "source_frame_index": index,
                "reference_monotonic_ns": capture["frames"][index][
                    "capture_monotonic_ns"
                ],
                "capture_monotonic_ns": capture["frames"][index + 1][
                    "capture_monotonic_ns"
                ],
                "raw_bracket_indices": [index, index + 1],
                "raw_interpolation_alpha": 0.0,
                "joint_pos_mj29": _mj29_from_il29(il29),
                "joint_pos_il29": il29,
                "joint_vel_il29": velocity,
            }
        )
    return {
        "schema_version": producer.RETARGET_TRACE_SCHEMA_VERSION,
        "kind": producer.RETARGET_TRACE_KIND,
        "source_kind": SOURCE_RETARGETED_DELAYED,
        "source_session_id": capture["session_id"],
        "raw_capture_sha256": capture_summary["sha256"],
        "producer": {
            "kind": producer.RETARGET_PRODUCER_KIND,
            "adapter_sha256": "c" * 64,
            "repository": producer.PINNED_SOMA_REPOSITORY,
            "commit": producer.PINNED_SOMA_COMMIT,
            "package_version": producer.PINNED_SOMA_PACKAGE_VERSION,
            "python_requires": ">=3.12",
            "newton_version": producer.PINNED_NEWTON_VERSION,
            "warp_version": producer.PINNED_WARP_VERSION,
            "config_hash_semantics": producer.PINNED_CONFIG_HASH_SEMANTICS,
            "config_sha256": dict(producer.PINNED_CONFIG_SHA256),
            "xr24_soma_adapter_version": producer.XR24_SOMA_ADAPTER_VERSION,
            "ankle_input_semantics": producer.ANKLE_ROLE_SEMANTICS,
        },
        "sample_period_ns": SOURCE_SAMPLE_PERIOD_NS,
        "joint_order_mj29": list(producer.SOMA_MJ29_JOINT_NAMES),
        "joint_order_il29": list(SOURCE_IL29_JOINT_NAMES),
        "mj29_to_il29": list(producer.SOMA_MJ29_TO_CANONICAL_IL29),
        "samples": samples,
        "certification": {
            "exact_backend_replay_performed": False,
            "raw_capture_replayed": False,
            "full_il29_verified": False,
            "promotion_eligible": False,
            "status": producer.RETARGET_REPLAY_STATUS_BLOCKED,
        },
        "authorization": {
            "read_only": True,
            "dds_opened": False,
            "robot_channel_opened": False,
            "actuation_authorized": False,
            "robot_commands_published": False,
        },
    }


def test_synthetic_kinematically_valid_raw_capture_is_complete_and_read_only() -> None:
    capture = _capture()
    summary = producer.validate_raw_capture(capture)

    assert summary["frame_count"] == 49
    assert len(summary["sha256"]) == 64
    assert summary["read_only"] is True
    assert capture["source"]["ankle_role_indices"] == {
        "left_ankle": 7,
        "right_ankle": 8,
    }
    assert len(capture["frames"][0]["body_velocities"]) == 24
    assert capture["frames"][0]["health_tracker_count"] == 2
    assert capture["frames"][0]["health_body_role_count"] == 24


def test_only_versioned_healthy_bodytracking_snapshot_is_accepted() -> None:
    snapshot = _bodytracking_snapshot(0)

    assert producer.validate_bodytracking_xrt_snapshot(snapshot) is snapshot
    report = producer.probe_exact_retargeter(bodytracking_snapshot=snapshot)
    assert report["atomic_snapshot_valid"] is True
    assert report["ready"] is False
    assert report["promotion_eligible"] is False

    retained_motion_diagnostic = dict(snapshot)
    retained_motion_diagnostic["contract"] = (
        "xrt_xr24_plus_retained_motion_unapproved_v0"
    )
    with pytest.raises(ValueError, match="exact keys"):
        producer.validate_bodytracking_xrt_snapshot(
            {**retained_motion_diagnostic, "trackers": {}}
        )

    incompatible = dict(snapshot)
    incompatible["derivative_layout_contract"] = "legacy_mixed_va_wva"
    with pytest.raises(ValueError, match="derivative layout"):
        producer.validate_bodytracking_xrt_snapshot(incompatible)

    incoherent = dict(snapshot)
    incoherent["source_coherence_contract"] = "independent_getter_pair"
    with pytest.raises(ValueError, match="source coherence"):
        producer.validate_bodytracking_xrt_snapshot(incoherent)

    missing_coherence = dict(snapshot)
    missing_coherence.pop("source_coherence_contract")
    with pytest.raises(ValueError, match="exact keys"):
        producer.validate_bodytracking_xrt_snapshot(missing_coherence)


def test_bodytracking_snapshot_materializes_fused_ankle_roles_without_motion() -> None:
    frame = producer.raw_frame_from_bodytracking_xrt_snapshot(
        _bodytracking_snapshot(0),
        frame_index=0,
        capture_monotonic_ns=5_000_000_000,
    )
    assert frame["health_tracker_count"] == 2
    assert frame["health_body_role_count"] == 24
    assert "motion_tracker_poses" not in frame


def test_bodytracking_snapshot_accepts_explicit_unavailable_joint_clock() -> None:
    snapshot = _bodytracking_snapshot(0)
    snapshot["body_timestamp_contract"] = (
        producer.PACKET_HEALTH_BODY_TIMESTAMP_CONTRACT
    )
    snapshot["joint_timestamp_contract"] = (
        producer.UNAVAILABLE_JOINT_TIMESTAMP_CONTRACT
    )
    snapshot["joint_timestamps_ns"] = [0] * 24

    validated = producer.validate_bodytracking_xrt_snapshot(snapshot)
    assert validated["joint_timestamps_ns"] == [0] * 24
    assert validated["timestamp_ns"] == validated["health_timestamp_ns"]


def test_bodytracking_snapshot_accepts_exact_head_only_joint_clock() -> None:
    snapshot = _bodytracking_snapshot(0)
    head_timestamp_ns = snapshot["timestamp_ns"]
    snapshot["body_timestamp_contract"] = (
        producer.HEAD_LOCAL_POSE_BODY_TIMESTAMP_CONTRACT
    )
    snapshot["joint_timestamp_contract"] = (
        producer.HEAD_ONLY_JOINT_TIMESTAMP_CONTRACT
    )
    snapshot["joint_timestamps_ns"] = [0] * 24
    snapshot["joint_timestamps_ns"][15] = head_timestamp_ns

    validated = producer.validate_bodytracking_xrt_snapshot(snapshot)
    assert validated["joint_timestamps_ns"][15] == head_timestamp_ns
    assert sum(value > 0 for value in validated["joint_timestamps_ns"]) == 1


def test_head_only_joint_clock_rejects_wrong_or_extra_role_timestamp() -> None:
    snapshot = _bodytracking_snapshot(0)
    snapshot["body_timestamp_contract"] = (
        producer.HEAD_LOCAL_POSE_BODY_TIMESTAMP_CONTRACT
    )
    snapshot["joint_timestamp_contract"] = (
        producer.HEAD_ONLY_JOINT_TIMESTAMP_CONTRACT
    )
    snapshot["joint_timestamps_ns"] = [0] * 24
    snapshot["joint_timestamps_ns"][14] = snapshot["timestamp_ns"]
    with pytest.raises(ValueError, match="index 15 positive and 23 zeros"):
        producer.validate_bodytracking_xrt_snapshot(snapshot)

    snapshot["joint_timestamps_ns"][15] = snapshot["timestamp_ns"]
    with pytest.raises(ValueError, match="index 15 positive and 23 zeros"):
        producer.validate_bodytracking_xrt_snapshot(snapshot)

    snapshot["joint_timestamps_ns"][14] = 0
    snapshot["timestamp_ns"] += 1
    with pytest.raises(ValueError, match="does not match PICO HEAD"):
        producer.validate_bodytracking_xrt_snapshot(snapshot)


def test_bodytracking_snapshot_rejects_hidden_or_mixed_joint_clock() -> None:
    missing_contract = _bodytracking_snapshot(0)
    missing_contract.pop("joint_timestamp_contract")
    with pytest.raises(ValueError, match="exact keys"):
        producer.validate_bodytracking_xrt_snapshot(missing_contract)

    mixed = _bodytracking_snapshot(0)
    mixed["body_timestamp_contract"] = (
        producer.PACKET_HEALTH_BODY_TIMESTAMP_CONTRACT
    )
    mixed["joint_timestamp_contract"] = (
        producer.UNAVAILABLE_JOINT_TIMESTAMP_CONTRACT
    )
    mixed["joint_timestamps_ns"] = [0] * 23 + [mixed["timestamp_ns"]]
    with pytest.raises(ValueError, match="24 exact zeros"):
        producer.validate_bodytracking_xrt_snapshot(mixed)

    substituted = _bodytracking_snapshot(0)
    substituted["joint_timestamp_contract"] = (
        producer.UNAVAILABLE_JOINT_TIMESTAMP_CONTRACT
    )
    substituted["body_timestamp_contract"] = (
        producer.PACKET_HEALTH_BODY_TIMESTAMP_CONTRACT
    )
    with pytest.raises(ValueError, match="24 exact zeros"):
        producer.validate_bodytracking_xrt_snapshot(substituted)


def test_unavailable_joint_clock_requires_exact_same_packet_health_time() -> None:
    snapshot = _bodytracking_snapshot(0)
    snapshot["body_timestamp_contract"] = (
        producer.PACKET_HEALTH_BODY_TIMESTAMP_CONTRACT
    )
    snapshot["joint_timestamp_contract"] = (
        producer.UNAVAILABLE_JOINT_TIMESTAMP_CONTRACT
    )
    snapshot["joint_timestamps_ns"] = [0] * 24
    snapshot["timestamp_ns"] += 1
    with pytest.raises(ValueError, match="does not match hardened health"):
        producer.validate_bodytracking_xrt_snapshot(snapshot)


def test_current_atomic_snapshot_shapes_fail_closed_on_missing_raw_fields() -> None:
    body = _bodytracking_snapshot(0)
    incomplete_body_snapshot = {
        key: value
        for key, value in body.items()
        if key not in {"velocities", "accelerations", "joint_timestamps_ns"}
    }
    with pytest.raises(ValueError, match="exact keys"):
        producer.validate_bodytracking_xrt_snapshot(incomplete_body_snapshot)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("health_valid", False, "health_valid mismatch"),
        ("health_schema_version", True, "health_schema_version"),
        ("health_tracking_mode", 1, "health_tracking_mode mismatch"),
        ("health_unique_tracker_count", 1, "must prove two ankle bands"),
        ("health_body_state_code", 0, "health_body_state_code mismatch"),
        ("health_body_role_count", 23, "health_body_role_count mismatch"),
    ],
)
def test_bodytracking_snapshot_rejects_false_health_proof(
    field: str,
    value: object,
    message: str,
) -> None:
    snapshot = _bodytracking_snapshot(0)
    snapshot[field] = value
    with pytest.raises(ValueError, match=message):
        producer.validate_bodytracking_xrt_snapshot(snapshot)


def test_raw_capture_rejects_fused_ankle_health_and_kinematic_tampering() -> None:
    capture = _capture()
    capture["source"]["ankle_role_indices"]["left_ankle"] = 8
    with pytest.raises(ValueError, match="role indices mismatch"):
        producer.validate_raw_capture(capture)

    capture = _capture()
    capture["frames"][4]["health_tracker_count"] = 1
    with pytest.raises(ValueError, match="must prove two ankle bands"):
        producer.validate_raw_capture(capture)

    capture = _capture()
    capture["frames"][4]["body_poses"][7][1] += 0.2
    with pytest.raises(ValueError, match="bone length changed discontinuously"):
        producer.validate_raw_capture(capture)


def test_raw_capture_marks_sdk_derivatives_untrusted_for_control() -> None:
    capture = _capture()
    capture["frames"][0]["body_velocities"][0][0] = 4.7846e6
    capture["frames"][0]["body_accelerations"][0][2] = 5.8945e21
    summary = producer.validate_raw_capture(capture)
    assert summary["sdk_derivatives_control_usable"] is False
    assert summary["control_derivative_contract"] == (
        producer.POSITION_DERIVED_CONTROL_DERIVATIVE_CONTRACT
    )
    assert summary["promotion_eligible"] is False

    capture = _capture()
    capture["source"]["sdk_derivatives_control_usable"] = True
    with pytest.raises(ValueError, match="explicitly control-unusable"):
        producer.validate_raw_capture(capture)

    capture = _capture()
    capture["source"]["control_derivative_contract"] = "sdk_derivatives_v1"
    with pytest.raises(ValueError, match="control derivative contract"):
        producer.validate_raw_capture(capture)


def test_legacy_v2_capture_is_diagnostic_only_and_never_promotable() -> None:
    capture = _capture()
    capture["schema_version"] = producer.LEGACY_RAW_CAPTURE_SCHEMA_VERSION
    for frame in capture["frames"]:
        frame["schema_version"] = producer.LEGACY_RAW_CAPTURE_SCHEMA_VERSION
    del capture["source"]["sdk_derivatives_control_usable"]
    del capture["source"]["control_derivative_contract"]

    summary = producer.validate_raw_capture(capture)
    assert summary["legacy_v2_diagnostic_only"] is True
    assert summary["sdk_derivatives_control_usable"] is False
    assert summary["promotion_eligible"] is False


def test_checked_in_wrist_only_zero_pose_streamer_payload_is_rejected() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "pico_manager_thread_server.py"
    ).read_text(encoding="utf-8")
    assert "joint_pos = np.zeros(29)" in source
    assert '"joint_vel": np.zeros((N, 29))' in source

    joint_pos = [[0.0] * 29 for _ in range(47)]
    for frame in joint_pos:
        for wrist_index in (23, 24, 25, 26, 27, 28):
            frame[wrist_index] = 0.1
    payload = {"joint_pos": joint_pos, "joint_vel": [[0.0] * 29 for _ in range(47)]}
    with pytest.raises(ValueError, match="wrist-only with zero lower body"):
        producer.reject_legacy_pose_payload_as_semantic_reference(payload)


def test_recorded_g1_trace_contract_stays_nonpromotable_without_replay() -> None:
    capture = _capture()
    trace = _trace(capture)
    summary = producer.validate_retarget_trace_contract(
        capture,
        trace,
        profile=REFERENCE_PROFILE_NORMAL,
    )

    assert summary["raw_frame_count"] == 49
    assert summary["retarget_position_sample_count"] == 48
    assert summary["derivable_semantic_frame_count"] == 47
    assert summary["exact_backend_replay_performed"] is False
    assert summary["promotion_eligible"] is False
    assert summary["status"] == producer.RETARGET_REPLAY_STATUS_BLOCKED


def test_trace_requires_profile_proof_plus_forward_velocity_sample() -> None:
    normal_capture = _capture(48)
    normal_trace = _trace(normal_capture, sample_count=47)
    with pytest.raises(ValueError, match="at least 48 position samples"):
        producer.validate_retarget_trace_contract(
            normal_capture,
            normal_trace,
            profile=REFERENCE_PROFILE_NORMAL,
        )

    low_latency_capture = _capture(12)
    low_latency_trace = _trace(low_latency_capture, sample_count=11)
    with pytest.raises(ValueError, match="at least 12 position samples"):
        producer.validate_retarget_trace_contract(
            low_latency_capture,
            low_latency_trace,
            profile=REFERENCE_PROFILE_LOW_LATENCY,
        )

    valid_low_latency_capture = _capture(13)
    valid_low_latency_trace = _trace(
        valid_low_latency_capture,
        sample_count=12,
    )
    summary = producer.validate_retarget_trace_contract(
        valid_low_latency_capture,
        valid_low_latency_trace,
        profile=REFERENCE_PROFILE_LOW_LATENCY,
    )
    assert summary["retarget_position_sample_count"] == 12
    assert summary["derivable_semantic_frame_count"] == 11
    assert summary["required_semantic_frame_count"] == 11


def test_trace_tampering_and_fake_certification_are_rejected() -> None:
    capture = _capture()
    trace = _trace(capture)
    trace["raw_capture_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="raw-capture hash mismatch"):
        producer.validate_retarget_trace_contract(
            capture,
            trace,
            profile=REFERENCE_PROFILE_NORMAL,
        )

    trace = _trace(capture)
    trace["samples"][3]["joint_pos_il29"][0] += 0.1
    with pytest.raises(ValueError, match="not exact SOMA MJ29 reorder"):
        producer.validate_retarget_trace_contract(
            capture,
            trace,
            profile=REFERENCE_PROFILE_NORMAL,
        )

    trace = _trace(capture)
    trace["samples"][3]["joint_vel_il29"][0] += 0.1
    with pytest.raises(ValueError, match="not 50 Hz forward difference"):
        producer.validate_retarget_trace_contract(
            capture,
            trace,
            profile=REFERENCE_PROFILE_NORMAL,
        )

    trace = _trace(capture)
    trace["certification"]["promotion_eligible"] = True
    with pytest.raises(ValueError, match="must remain non-promotable"):
        producer.validate_retarget_trace_contract(
            capture,
            trace,
            profile=REFERENCE_PROFILE_NORMAL,
        )

    trace = _trace(capture)
    for sample_index, sample in enumerate(trace["samples"]):
        sample["joint_pos_mj29"] = [0.0] * 29
        sample["joint_pos_il29"] = [0.0] * 29
        sample["joint_vel_il29"] = (
            None if sample_index == len(trace["samples"]) - 1 else [0.0] * 29
        )
    with pytest.raises(ValueError, match="zero lower-body"):
        producer.validate_retarget_trace_contract(
            capture,
            trace,
            profile=REFERENCE_PROFILE_NORMAL,
        )


def test_probe_and_materialization_remain_fail_closed(tmp_path: Path, capsys) -> None:
    report = producer.probe_exact_retargeter(soma_source_root=tmp_path)
    assert report["ready"] is False
    assert report["promotion_eligible"] is False
    assert report["adapter_approved"] is False
    assert any("adapter lacks approved exact replay fixture" in item for item in report["blockers"])

    with pytest.raises(
        producer.ExactRetargeterUnavailable,
        match="ankle-fused XR24 capture",
    ):
        producer.materialize_semantic_frames()

    assert probe_cli.main([]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["authorization"]["dds_opened"] is False


def test_pinned_config_hashing_is_git_line_ending_invariant(tmp_path: Path) -> None:
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{\n  "value": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')

    assert producer._normalized_utf8_sha256(  # noqa: SLF001
        lf
    ) == producer._normalized_utf8_sha256(crlf)  # noqa: SLF001
    assert producer.PINNED_CONFIG_SHA256 == {
        "soma_to_g1_retargeter_config.json": (
            "befc09515e3b4f75f85561ec757fd795d03904abade21bb9ed9940363797953e"
        ),
        "soma_to_g1_scaler_config.json": (
            "1f9da8ae28500a27bea90d2aaf3949df4992e3db7ca0a92fcc58cf8080854efb"
        ),
        "g1_feet_stabilizer_config.json": (
            "68ab0dc2318eb91272d6c067de6c3b85c27de6cdab83a7d9d09013f9b75dd738"
        ),
    }


def test_probe_cli_consumes_saved_bodytracking_snapshot_but_stays_no_go(
    tmp_path: Path,
    capsys,
) -> None:
    snapshot = _bodytracking_snapshot(0)
    snapshot_path = tmp_path / "atomic.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    assert (
        probe_cli.main(["--atomic-snapshot", str(snapshot_path)])
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["atomic_snapshot_valid"] is True
    assert payload["ready"] is False
    assert payload["promotion_eligible"] is False


def test_checked_in_approval_is_explicit_no_go() -> None:
    approval_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "deployment"
        / "g1_true23_pico_retargeter_approval.json"
    )
    approval = json.loads(approval_path.read_text(encoding="utf-8"))

    assert approval["promotion_enabled"] is False
    assert approval["current_hardened_client_compatible"] is True
    assert approval["required_producer"]["adapter_sha256"] is None
    assert (
        approval["required_producer"]["config_hash_semantics"]
        == producer.PINNED_CONFIG_HASH_SEMANTICS
    )
    assert approval["required_producer"]["config_sha256"] == (
        producer.PINNED_CONFIG_SHA256
    )
    assert approval["required_evidence"]["exact_backend_replay_performed"] is False
    assert approval["required_evidence"]["full_il29_verified"] is False
    assert approval["authorization"]["robot_commands_published"] is False

    protocol_path = approval_path.with_name(
        "g1_true23_pico_retargeter_protocol.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw = protocol["raw_capture"]
    assert raw["binding_getter"] == "get_body_snapshot"
    assert raw["raw_motion_tracking_mode_allowed"] is False
    assert raw["raw_motion_tracker_data_required"] is False
    assert raw["bodytracking_fused_ankle_roles_required"] is True
    assert raw["ankle_role_indices"] == {"left_ankle": 7, "right_ankle": 8}
