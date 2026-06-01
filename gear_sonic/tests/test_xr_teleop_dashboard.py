from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.utils.teleop.xr_dashboard import (
    DashboardHttpHandler,
    DashboardState,
    assert_read_only_socket_type,
    encode_color_image,
    summarize_values,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)


def sample_xr_payload(**overrides):
    payload = {
        "timestamp": 1.0,
        "dual_arm_position": [0.1 * i for i in range(14)],
        "dual_hand_joints": [0.01 * i for i in range(14)],
        "mode": 2,
        "movement": [0.25, -0.5, 0.0],
        "facing": [0.9, 0.1, 0.0],
        "speed": 0.4,
        "height": 0.2,
        "toggle_data_collection": False,
        "toggle_data_abort": False,
        "stop": False,
    }
    payload.update(overrides)
    return payload


def test_initial_edge_state_is_unknown():
    state = DashboardState()
    snapshot = state.snapshot()
    assert snapshot["edge_state"]["command"] == "not_yet_received"
    assert snapshot["collection"]["inferred"] is True
    assert snapshot["collection"]["state"] == "unknown"


def test_xr_payload_uses_bridge_normalization_and_summarizes_transforms():
    state = DashboardState()
    state.update_xr_payload(sample_xr_payload(), receive_time=10.0)
    snapshot = state.snapshot(now=10.1)

    assert snapshot["xr"]["status"] == "live"
    assert snapshot["xr"]["mode"] == 2
    assert snapshot["xr"]["movement"] == [0.25, -0.5, 0.0]
    assert snapshot["xr"]["stop"] is False
    assert snapshot["bridge_transform"]["raw_dual_arm"]["count"] == 14
    assert snapshot["bridge_transform"]["mapped_upper_body"]["count"] == 17
    assert snapshot["hands"]["left"]["summary"]["count"] == 7
    assert snapshot["hands"]["right"]["summary"]["count"] == 7


def test_collection_latch_is_inferred_from_toggle_edges():
    state = DashboardState()
    state.update_xr_payload(sample_xr_payload(toggle_data_collection=True), receive_time=20.0)
    first = state.snapshot(now=21.0)
    assert first["collection"]["state"] == "recording_inferred"
    assert first["collection"]["seconds_since_last_record_edge"] == 1.0

    state.update_xr_payload(sample_xr_payload(toggle_data_collection=True), receive_time=22.0)
    second = state.snapshot(now=22.5)
    assert second["collection"]["state"] == "idle_inferred"
    assert second["collection"]["seconds_since_last_record_edge"] == 0.5


def test_summarize_values_handles_absent_and_numeric_values():
    assert summarize_values(None)["count"] == 0
    summary = summarize_values(np.array([1.0, -2.0, 3.0], dtype=np.float32))
    assert summary == {"count": 3, "min": -2.0, "max": 3.0, "mean": 0.666667}


def test_bridge_planner_and_command_decode():
    state = DashboardState()
    state.update_bridge_message(
        build_planner_message(
            mode=4,
            movement=[1.0, 0.0, 0.0],
            facing=[0.0, 1.0, 0.0],
            height=0.3,
            upper_body_position=[0.1] * 17,
        ),
        receive_time=30.0,
    )
    state.update_bridge_message(
        build_command_message(start=True, stop=False, planner=True),
        receive_time=31.0,
    )
    snapshot = state.snapshot(now=31.1)
    assert snapshot["bridge"]["planner_mode"] == 4
    assert snapshot["bridge_transform"]["planner_upper_body"]["count"] == 17
    assert snapshot["edge_state"]["command"] == "received"
    assert snapshot["bridge"]["latest_command"]["start"] == [1.0]
    assert snapshot["bridge"]["latest_command"]["stop"] == [0.0]


def test_camera_frame_lookup_absent_returns_none():
    state = DashboardState()
    assert state.get_camera_frame("ego_view") is None
    state.update_camera_frame("ego_view", b"jpeg-bytes", "image/jpeg", receive_time=40.0)
    frame = state.get_camera_frame("ego_view")
    assert frame is not None
    assert frame.content == b"jpeg-bytes"
    assert frame.content_type == "image/jpeg"


def test_read_only_socket_type_rejects_control_socket_types():
    assert DashboardHttpHandler is not None
    assert_read_only_socket_type("xr", "SUB")
    with pytest.raises(ValueError, match="read-only"):
        assert_read_only_socket_type("bridge", "PUB")
    with pytest.raises(ValueError, match="read-only"):
        assert_read_only_socket_type("bridge", "REQ")


def test_encode_color_image_rejects_non_color():
    assert encode_color_image(np.zeros((4, 4), dtype=np.uint16)) is None


def test_encode_color_image_returns_jpeg_bytes_for_rgb():
    pytest.importorskip("cv2")
    encoded = encode_color_image(np.zeros((4, 4, 3), dtype=np.uint8))
    assert encoded is not None
    content, content_type = encoded
    assert content.startswith(b"\xff\xd8")
    assert content_type == "image/jpeg"
