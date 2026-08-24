from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import pytest

from gear_sonic.scripts import capture_g1_23dof_pico_raw as capture_cli
from gear_sonic.utils.g1_23dof_pico_retargeted_producer import (
    ATOMIC_SNAPSHOT_CONTRACT,
    BODY_TRACKING_MODE_CODE,
    BODY_TRACKING_SUCCESS_CODE,
    BODY_TRACKING_VALID_STATE_CODE,
    DERIVATIVE_LAYOUT_CONTRACT,
    LOCAL_POSE_BODY_TIMESTAMP_CONTRACT,
    POSITIVE_JOINT_TIMESTAMP_CONTRACT,
    SOURCE_COHERENCE_CONTRACT,
    validate_raw_capture,
)
from gear_sonic.utils.g1_23dof_semantic_reference import SOURCE_SAMPLE_PERIOD_NS


def _positions(frame_index: int) -> list[list[float]]:
    x = frame_index * 0.001
    return [
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


def _snapshot(frame_index: int) -> dict:
    timestamp_ns = 3_000_000_000 + frame_index * SOURCE_SAMPLE_PERIOD_NS
    identity = [0.0, 0.0, 0.0, 1.0]
    return {
        "contract": ATOMIC_SNAPSHOT_CONTRACT,
        "derivative_layout_contract": DERIVATIVE_LAYOUT_CONTRACT,
        "source_coherence_contract": SOURCE_COHERENCE_CONTRACT,
        "body_timestamp_contract": LOCAL_POSE_BODY_TIMESTAMP_CONTRACT,
        "joint_timestamp_contract": POSITIVE_JOINT_TIMESTAMP_CONTRACT,
        "available": True,
        "timestamp_ns": timestamp_ns,
        "sample_sequence": frame_index + 1,
        "poses": [[*position, *identity] for position in _positions(frame_index)],
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
        "health_tracking_mode": BODY_TRACKING_MODE_CODE,
        "health_connect_state_result": 0,
        "health_tracker_count": 2,
        "health_unique_tracker_count": 2,
        "health_body_state_result": 0,
        "health_is_tracking": True,
        "health_tracking_state_code": BODY_TRACKING_SUCCESS_CODE,
        "health_body_state_code": BODY_TRACKING_VALID_STATE_CODE,
        "health_body_error_code": 0,
        "health_connected_band_count": 2,
        "health_body_data_result": 0,
        "health_body_role_count": 24,
    }


class _FakeXrt:
    def __init__(self, snapshots: Sequence[dict]):
        self._snapshots = list(snapshots)
        self._index = 0
        self.init_calls = 0
        self.close_calls = 0

    def init(self):
        self.init_calls += 1

    def close(self):
        self.close_calls += 1

    def get_body_snapshot(self):
        snapshot = self._snapshots[min(self._index, len(self._snapshots) - 1)]
        self._index += 1
        return snapshot


def test_collect_raw_capture_is_advancing_valid_and_non_actuating():
    result = capture_cli.collect_raw_capture(
        _FakeXrt([_snapshot(index) for index in range(3)]),
        frame_count=3,
        frame_timeout_s=0.1,
        session_id="live-test",
        xrobotoolkit_sdk_sha256="a" * 64,
        pc_service_sha256="b" * 64,
        pico_client_apk_sha256="c" * 64,
    )
    summary = validate_raw_capture(result)
    assert summary["frame_count"] == 3
    assert result["source"]["pico_client_build"] == "xrobotoolkit-pico-health-v1"
    assert result["authorization"] == {
        "read_only": True,
        "dds_opened": False,
        "robot_channel_opened": False,
        "actuation_authorized": False,
        "robot_commands_published": False,
    }


def test_capture_rejects_stale_body_sequence():
    with pytest.raises(TimeoutError, match="did not advance"):
        capture_cli.collect_raw_capture(
            _FakeXrt([_snapshot(0)]),
            frame_count=2,
            frame_timeout_s=0.001,
            session_id="stale-test",
            xrobotoolkit_sdk_sha256="a" * 64,
            pc_service_sha256="b" * 64,
            pico_client_apk_sha256="c" * 64,
        )


def test_raw_capture_rejects_nonadvancing_health_timestamp():
    result = capture_cli.collect_raw_capture(
        _FakeXrt([_snapshot(index) for index in range(3)]),
        frame_count=3,
        frame_timeout_s=0.1,
        session_id="health-stale-test",
        xrobotoolkit_sdk_sha256="a" * 64,
        pc_service_sha256="b" * 64,
        pico_client_apk_sha256="c" * 64,
    )
    result["frames"][1]["health_timestamp_ns"] = result["frames"][0][
        "health_timestamp_ns"
    ]
    with pytest.raises(ValueError, match="health_timestamp_ns did not advance"):
        validate_raw_capture(result)


def test_raw_capture_rejects_nonadvancing_body_and_health_sequence():
    result = capture_cli.collect_raw_capture(
        _FakeXrt([_snapshot(index) for index in range(3)]),
        frame_count=3,
        frame_timeout_s=0.1,
        session_id="sequence-stale-test",
        xrobotoolkit_sdk_sha256="a" * 64,
        pc_service_sha256="b" * 64,
        pico_client_apk_sha256="c" * 64,
    )
    stale_sequence = result["frames"][0]["body_sample_sequence"]
    result["frames"][1]["body_sample_sequence"] = stale_sequence
    result["frames"][1]["health_sample_sequence"] = stale_sequence
    with pytest.raises(ValueError, match="body_sample_sequence did not advance"):
        validate_raw_capture(result)


def test_capture_output_never_overwrites(tmp_path: Path):
    output = tmp_path / "capture.json"
    capture_cli._write_exclusive(output, b"first")  # noqa: SLF001
    with pytest.raises(FileExistsError):
        capture_cli._write_exclusive(output, b"second")  # noqa: SLF001
    assert output.read_bytes() == b"first"


def test_main_closes_xrobotoolkit_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sdk_binary = tmp_path / "xrobotoolkit_sdk.so"
    service_binary = tmp_path / "RoboticsServiceProcess"
    output = tmp_path / "capture.json"
    sdk_binary.write_bytes(b"sdk")
    service_binary.write_bytes(b"service")
    xrt = _FakeXrt([_snapshot(index) for index in range(2)])
    monkeypatch.setattr(capture_cli, "_load_xrt", lambda: xrt)
    monkeypatch.setattr(capture_cli, "_sdk_binary_path", lambda _: sdk_binary)

    result = capture_cli.main(
        [
            "--output",
            str(output),
            "--pico-client-apk-sha256",
            "c" * 64,
            "--service-binary",
            str(service_binary),
            "--frames",
            "2",
            "--session-id",
            "close-success-test",
        ]
    )

    assert result == 0
    assert xrt.init_calls == 1
    assert xrt.close_calls == 1
    assert validate_raw_capture(json.loads(output.read_text()))["frame_count"] == 2


def test_main_closes_xrobotoolkit_after_capture_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sdk_binary = tmp_path / "xrobotoolkit_sdk.so"
    service_binary = tmp_path / "RoboticsServiceProcess"
    output = tmp_path / "capture.json"
    sdk_binary.write_bytes(b"sdk")
    service_binary.write_bytes(b"service")
    xrt = _FakeXrt([_snapshot(0)])
    monkeypatch.setattr(capture_cli, "_load_xrt", lambda: xrt)
    monkeypatch.setattr(capture_cli, "_sdk_binary_path", lambda _: sdk_binary)

    with pytest.raises(TimeoutError, match="did not advance"):
        capture_cli.main(
            [
                "--output",
                str(output),
                "--pico-client-apk-sha256",
                "c" * 64,
                "--service-binary",
                str(service_binary),
                "--frames",
                "2",
                "--frame-timeout-s",
                "0.001",
                "--session-id",
                "close-failure-test",
            ]
        )

    assert xrt.init_calls == 1
    assert xrt.close_calls == 1
    assert not output.exists()
