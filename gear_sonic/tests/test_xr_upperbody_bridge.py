import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE
from gear_sonic.utils.teleop.xr_upperbody_bridge import (
    BridgeConfig,
    UpperBodyFilter,
    build_manager_state_message,
    build_frame_planner_message,
    build_inspection_report,
    collect_range_warnings,
    load_frames,
    normalize_live_source_payload,
    run_local_zmq_smoke,
    split_xr_dex3_dual_hand_joints,
    synthesize_missing_velocities,
    xr_g1_29_arm_to_sonic_upper_body,
)


def _valid_position(offset: float = 0.0) -> list[float]:
    return [offset + 0.01 * i for i in range(17)]


def _decode_header(message: bytes, topic: str = "planner") -> dict:
    start = len(topic)
    raw = message[start : start + HEADER_SIZE].rstrip(b"\x00")
    return json.loads(raw.decode("utf-8"))


def test_load_jsonl_frames_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "upperbody.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": 0.0,
                        "upper_body_position": _valid_position(),
                        "left_hand_joints": [0.1] * 7,
                    }
                ),
                json.dumps(
                    {
                        "timestamp": 0.02,
                        "upper_body_position": _valid_position(0.1),
                        "upper_body_velocity": [0.2] * 17,
                        "right_hand_joints": [0.3] * 7,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    frames = load_frames(path)

    assert len(frames) == 2
    assert frames[0].timestamp == 0.0
    assert frames[0].upper_body_position.shape == (17,)
    assert np.allclose(frames[0].upper_body_velocity, np.zeros(17))
    assert np.allclose(frames[0].left_hand_joints, np.full(7, 0.1))
    assert frames[0].right_hand_joints is None
    assert np.allclose(frames[1].upper_body_velocity, np.full(17, 0.2))


def test_load_npz_frames(tmp_path: Path) -> None:
    path = tmp_path / "upperbody.npz"
    np.savez(
        path,
        timestamp=np.array([0.0, 0.02], dtype=np.float64),
        upper_body_position=np.stack([_valid_position(), _valid_position(0.1)]).astype(np.float32),
        left_hand_joints=np.ones((2, 7), dtype=np.float32),
    )

    frames = load_frames(path)

    assert len(frames) == 2
    assert frames[1].timestamp == 0.02
    assert np.allclose(frames[1].left_hand_joints, np.ones(7))
    assert frames[1].right_hand_joints is None


def test_load_jsonl_dual_arm_position_maps_to_sonic_upper_body(tmp_path: Path) -> None:
    path = tmp_path / "arms_only.jsonl"
    arm = [float(i) for i in range(14)]
    path.write_text(
        json.dumps(
            {
                "timestamp": 0.0,
                "dual_arm_position": arm,
                "left_hand_joints": [0.1] * 7,
                "right_hand_joints": [0.2] * 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    frames = load_frames(path)

    assert np.allclose(
        frames[0].upper_body_position,
        [
            0.0,
            0.0,
            0.0,
            0.0,
            7.0,
            1.0,
            8.0,
            2.0,
            9.0,
            3.0,
            10.0,
            4.0,
            11.0,
            5.0,
            12.0,
            6.0,
            13.0,
        ],
    )


def test_load_jsonl_dual_hand_joints_splits_left_and_right(tmp_path: Path) -> None:
    path = tmp_path / "hands.jsonl"
    path.write_text(
        json.dumps(
            {
                "dual_arm_position": [0.0] * 14,
                "dual_hand_joints": [float(i) for i in range(14)],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    frames = load_frames(path)

    assert np.allclose(frames[0].left_hand_joints, np.arange(7, dtype=np.float32))
    assert np.allclose(frames[0].right_hand_joints, np.arange(7, 14, dtype=np.float32))


def test_live_payload_control_flags_are_source_lifecycle_metadata() -> None:
    frame = normalize_live_source_payload(
        {
            "dual_arm_position": [0.0] * 14,
            "dual_hand_joints": [0.0] * 14,
            "mode": 4,
            "movement": [0.0, 0.0, 0.0],
            "facing": [1.0, 0.0, 0.0],
            "speed": 0.0,
            "height": 0.5,
            "toggle_data_collection": True,
            "toggle_data_abort": True,
            "stop": True,
        }
    )

    assert frame.mode == 4
    assert frame.height == 0.5
    assert frame.toggle_data_collection is True
    assert frame.toggle_data_abort is True
    assert frame.stop is True

    planner_message = build_frame_planner_message(frame, BridgeConfig())
    planner_header = _decode_header(planner_message)
    planner_fields = {field["name"] for field in planner_header["fields"]}
    manager_message = build_manager_state_message(
        stream_mode=5,
        toggle_data_collection=frame.toggle_data_collection,
        toggle_data_abort=frame.toggle_data_abort,
    )
    manager_header = _decode_header(manager_message, topic="manager_state")
    manager_fields = {field["name"] for field in manager_header["fields"]}

    assert "stop" not in planner_fields
    assert "stop" not in manager_fields


def test_rejects_bad_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"upper_body_position": [0.0] * 16}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="upper_body_position"):
        load_frames(path)


def test_xr_g1_29_arm_to_sonic_upper_body_with_waist() -> None:
    arm = np.arange(14, dtype=np.float32)
    upper = xr_g1_29_arm_to_sonic_upper_body(arm, waist=[0.1, 0.2, 0.3])

    assert np.allclose(upper[:3], [0.1, 0.2, 0.3])
    assert np.allclose(upper[3:], [0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13])


def test_split_xr_dex3_dual_hand_joints() -> None:
    left, right = split_xr_dex3_dual_hand_joints(np.arange(14, dtype=np.float32))

    assert np.allclose(left, [0, 1, 2, 3, 4, 5, 6])
    assert np.allclose(right, [7, 8, 9, 10, 11, 12, 13])


def test_synthesize_missing_velocities_from_timestamps() -> None:
    frames = load_frames(
        [
            {"timestamp": 0.0, "dual_arm_position": [0.0] * 14},
            {"timestamp": 0.5, "dual_arm_position": [0.5] * 14},
        ]
    )

    synthesized = synthesize_missing_velocities(frames, default_hz=50.0)

    assert np.allclose(synthesized[0].upper_body_velocity, np.zeros(17))
    assert np.allclose(synthesized[1].upper_body_velocity[:3], np.zeros(3))
    assert np.allclose(synthesized[1].upper_body_velocity[3:], np.full(14, 1.0))


def test_range_warning_detects_degrees_like_values() -> None:
    frames = load_frames(
        [
            {
                "dual_arm_position": [0.0] * 13 + [180.0],
                "dual_hand_joints": [0.0] * 14,
            }
        ]
    )

    warnings = collect_range_warnings(frames, arm_warn_rad=6.5, hand_warn_rad=3.2)

    assert any("upper_body_position" in warning for warning in warnings)


def test_inspection_report_includes_mapping_and_warnings() -> None:
    frames = load_frames(
        [
            {
                "timestamp": 0.0,
                "dual_arm_position": [0.1] * 14,
                "dual_hand_joints": [0.2] * 14,
            }
        ]
    )

    report = build_inspection_report(frames, source="sample.jsonl")

    assert "sample.jsonl" in report
    assert "frames: 1" in report
    assert "first upper_body_position[17]" in report
    assert "left_hand_joints range" in report


def test_local_zmq_smoke_receives_planner_and_manager_state() -> None:
    frames = load_frames([{"dual_arm_position": [0.0] * 14, "dual_hand_joints": [0.0] * 14}])

    result = run_local_zmq_smoke(frames, BridgeConfig(), port=0, timeout_s=2.0)

    assert int(result["planner"]["mode"][0]) == 0
    assert result["planner"]["upper_body_position"].shape == (17,)
    assert int(result["manager_state"]["stream_mode"][0]) == 5


def test_normalize_live_source_payload_from_genonai_style_json() -> None:
    frame = normalize_live_source_payload(
        {
            "timestamp": 1.25,
            "dual_arm_position": [0.1] * 14,
            "dual_hand_joints": [0.2] * 14,
        }
    )

    assert frame.timestamp == 1.25
    assert frame.upper_body_position.shape == (17,)
    assert np.allclose(frame.upper_body_position[:3], np.zeros(3))
    assert np.allclose(frame.left_hand_joints, np.full(7, 0.2))
    assert np.allclose(frame.right_hand_joints, np.full(7, 0.2))


def test_filter_clips_and_limits_position_steps() -> None:
    config = BridgeConfig(max_abs_joint=1.0, max_joint_step=0.25)
    filt = UpperBodyFilter(config)
    first = filt.apply(np.full(17, 0.9, dtype=np.float32))
    second = filt.apply(np.full(17, 2.0, dtype=np.float32))

    assert np.allclose(first, np.full(17, 0.9))
    assert np.allclose(second, np.full(17, 1.0))


def test_build_planner_message_fields() -> None:
    frames = load_frames(
        [
            {
                "upper_body_position": _valid_position(),
                "upper_body_velocity": [0.2] * 17,
                "left_hand_joints": [0.3] * 7,
                "right_hand_joints": [0.4] * 7,
            }
        ]
    )
    message = build_frame_planner_message(frames[0], BridgeConfig())
    header = _decode_header(message)
    fields = {field["name"]: field for field in header["fields"]}

    assert message.startswith(b"planner")
    assert fields["mode"]["shape"] == [1]
    assert fields["movement"]["shape"] == [3]
    assert fields["facing"]["shape"] == [3]
    assert fields["upper_body_position"]["shape"] == [17]
    assert fields["upper_body_velocity"]["shape"] == [17]
    assert fields["left_hand_joints"]["shape"] == [7]
    assert fields["right_hand_joints"]["shape"] == [7]


def test_build_manager_state_message_for_data_exporter() -> None:
    message = build_manager_state_message(stream_mode=5)
    header = _decode_header(message, topic="manager_state")
    fields = {field["name"]: field for field in header["fields"]}

    assert message.startswith(b"manager_state")
    assert fields["stream_mode"]["dtype"] == "i32"
    assert fields["toggle_data_collection"]["dtype"] == "bool"
    assert fields["toggle_data_abort"]["dtype"] == "bool"


def test_cli_dry_run_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from gear_sonic.scripts.xr_upperbody_bridge import main

    path = tmp_path / "upperbody.jsonl"
    path.write_text(
        json.dumps({"upper_body_position": _valid_position()}) + "\n",
        encoding="utf-8",
    )

    rc = main(["--input", str(path), "--dry-run", "--once", "--hz", "50"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Loaded 1 frame" in captured.out
    assert "Dry run published 1 frame" in captured.out
