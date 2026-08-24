import numpy as np
import pytest

from gear_sonic.utils.teleop.input_watchdog import (
    InputWatchdogAction,
    evaluate_body_input,
    send_stop_burst,
)


class _FakeSocket:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


class _FakeReader:
    disconnected = False
    supports_trusted_source_timestamps = True

    def __init__(self, sample):
        self.sample = sample

    def get_latest(self):
        return self.sample


def _sample(timestamp):
    body_poses = np.zeros((24, 7), dtype=np.float32)
    body_poses[:, 6] = 1.0
    controller_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return {
        "timestamp_monotonic": timestamp,
        "body_frame_displacement_m": 0.0,
        "body_frame_angular_displacement_rad": 0.0,
        "dt": 0.02,
        "left_controller_timestamp_monotonic": timestamp,
        "right_controller_timestamp_monotonic": timestamp,
        "motion_tracker_timestamp_monotonic": timestamp,
        "motion_tracker_count": 2,
        "motion_tracker_serial_numbers": ("ankle-left", "ankle-right"),
        "body_tracker_health_supported": True,
        "body_tracker_health_available": True,
        "body_tracker_health_valid": True,
        "body_tracker_health_schema_version": 1,
        "body_tracker_health_sample_sequence": 10,
        "body_tracker_health_source_timestamp_ns": 10_000,
        "body_tracker_health_timestamp_monotonic": timestamp,
        "body_tracker_health_calibration_result": 0,
        "body_tracker_health_calibrated": True,
        "body_tracker_health_tracking_mode": 0,
        "body_tracker_health_connect_state_result": 0,
        "body_tracker_health_tracker_count": 2,
        "body_tracker_health_unique_tracker_count": 2,
        "body_tracker_health_body_state_result": 0,
        "body_tracker_health_is_tracking": True,
        "body_tracker_health_tracking_state_code": 0,
        "body_tracker_health_body_state_code": 1,
        "body_tracker_health_body_error_code": 0,
        "body_tracker_health_connected_band_count": 2,
        "body_tracker_health_body_data_result": 0,
        "body_tracker_health_body_role_count": 24,
        "controller_tracking_health_supported": True,
        "controller_tracking_health_available": True,
        "controller_tracking_health_valid": True,
        "controller_tracking_health_schema_version": 1,
        "controller_tracking_health_sample_sequence": 10,
        "controller_tracking_health_source_timestamp_ns": 10_000,
        "controller_tracking_health_timestamp_monotonic": timestamp,
        "controller_tracking_health_left_device_valid": True,
        "controller_tracking_health_left_is_tracked_available": True,
        "controller_tracking_health_left_is_tracked": True,
        "controller_tracking_health_left_tracking_state_available": True,
        "controller_tracking_health_left_tracking_state": 3,
        "controller_tracking_health_left_valid": True,
        "controller_tracking_health_right_device_valid": True,
        "controller_tracking_health_right_is_tracked_available": True,
        "controller_tracking_health_right_is_tracked": True,
        "controller_tracking_health_right_tracking_state_available": True,
        "controller_tracking_health_right_tracking_state": 3,
        "controller_tracking_health_right_valid": True,
        "controller_data": {
            "left_pose": controller_pose.copy(),
            "right_pose": controller_pose.copy(),
            "left_trigger_value": 0.0,
            "right_trigger_value": 0.0,
            "left_squeeze_value": 0.0,
            "right_squeeze_value": 0.0,
            "left_thumbstick": np.zeros(2),
            "right_thumbstick": np.zeros(2),
            "left_thumbstick_click": False,
            "right_thumbstick_click": False,
            "left_primary_click": False,
            "left_secondary_click": False,
            "right_primary_click": False,
            "right_secondary_click": False,
        },
        "body_poses_np": body_poses,
    }


def test_evaluate_body_input_continues_only_with_fresh_tracking():
    action, sample, reason = evaluate_body_input(
        _FakeReader(_sample(10.0)),
        active=True,
        timeout_s=0.5,
        monotonic_now=10.2,
    )

    assert action is InputWatchdogAction.CONTINUE
    assert sample is not None
    assert reason == ""


@pytest.mark.parametrize(
    ("active", "expected"),
    [
        (False, InputWatchdogAction.WAIT),
        (True, InputWatchdogAction.STOP),
    ],
)
def test_evaluate_body_input_fails_closed_by_manager_state(active, expected):
    action, sample, reason = evaluate_body_input(
        _FakeReader(_sample(9.0)),
        active=active,
        timeout_s=0.5,
        monotonic_now=10.0,
    )

    assert action is expected
    assert sample is None
    assert "stale" in reason


def test_evaluate_body_input_stops_on_stale_controller():
    sample = _sample(10.0)
    sample["left_controller_timestamp_monotonic"] = 9.0

    action, result, reason = evaluate_body_input(
        _FakeReader(sample),
        active=True,
        timeout_s=0.5,
        monotonic_now=10.0,
    )

    assert action is InputWatchdogAction.STOP
    assert result is None
    assert "left controller is stale" in reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("motion_tracker_timestamp_monotonic", 9.0),
        ("motion_tracker_count", 0),
        ("motion_tracker_serial_numbers", ()),
    ],
)
def test_evaluate_body_input_uses_live_body_source_not_mutually_exclusive_raw_motion(
    field,
    value,
):
    sample = _sample(10.0)
    sample[field] = value

    action, result, reason = evaluate_body_input(
        _FakeReader(sample),
        active=True,
        timeout_s=0.5,
        monotonic_now=10.0,
    )

    assert action is InputWatchdogAction.CONTINUE
    assert result is sample
    assert reason == ""


@pytest.mark.parametrize(
    ("field", "value", "reason_fragment"),
    [
        ("body_tracker_health_supported", False, "does not expose hardened"),
        ("body_tracker_health_calibrated", False, "not calibrated"),
        ("body_tracker_health_tracker_count", 1, "connected tracker count"),
        ("body_tracker_health_unique_tracker_count", 1, "unique tracker count"),
        ("body_tracker_health_body_state_code", 2, "not BT_VALID"),
        ("body_tracker_health_is_tracking", False, "tracking lost"),
        ("body_tracker_health_timestamp_monotonic", 9.0, "health is stale"),
    ],
)
def test_evaluate_body_input_stops_on_bad_hardened_tracker_health(
    field,
    value,
    reason_fragment,
):
    sample = _sample(10.0)
    sample[field] = value

    action, result, reason = evaluate_body_input(
        _FakeReader(sample),
        active=True,
        timeout_s=0.5,
        monotonic_now=10.0,
    )

    assert action is InputWatchdogAction.STOP
    assert result is None
    assert reason_fragment in reason


@pytest.mark.parametrize(
    ("field", "value", "reason_fragment"),
    [
        ("controller_tracking_health_supported", False, "does not expose hardened"),
        ("controller_tracking_health_left_device_valid", False, "device is invalid"),
        ("controller_tracking_health_left_is_tracked", False, "tracking lost"),
        ("controller_tracking_health_right_tracking_state", 1, "position+rotation"),
        ("controller_tracking_health_source_timestamp_ns", 10_001, "same packet"),
        ("controller_tracking_health_timestamp_monotonic", 9.0, "health is stale"),
    ],
)
def test_evaluate_body_input_stops_on_bad_controller_tracking_health(
    field,
    value,
    reason_fragment,
):
    sample = _sample(10.0)
    sample[field] = value

    action, result, reason = evaluate_body_input(
        _FakeReader(sample),
        active=True,
        timeout_s=0.5,
        monotonic_now=10.0,
    )

    assert action is InputWatchdogAction.STOP
    assert result is None
    assert reason_fragment in reason


def test_evaluate_body_input_stops_on_nonfinite_controller_axis():
    sample = _sample(10.0)
    sample["controller_data"]["right_thumbstick"][0] = np.nan

    action, result, reason = evaluate_body_input(
        _FakeReader(sample),
        active=True,
        timeout_s=0.5,
        monotonic_now=10.0,
    )

    assert action is InputWatchdogAction.STOP
    assert result is None
    assert "right controller pose/axis contains NaN or Inf" in reason


@pytest.mark.parametrize(
    ("field", "value", "reason_fragment"),
    [
        ("body_frame_displacement_m", 0.5, "body tracking jumped"),
        ("body_frame_angular_displacement_rad", np.pi, "body tracking rotated"),
    ],
)
def test_evaluate_body_input_stops_on_body_frame_discontinuity(
    field,
    value,
    reason_fragment,
):
    sample = _sample(10.0)
    sample[field] = value

    action, result, reason = evaluate_body_input(
        _FakeReader(sample),
        active=True,
        timeout_s=0.5,
        monotonic_now=10.0,
    )

    assert action is InputWatchdogAction.STOP
    assert result is None
    assert reason_fragment in reason


def test_evaluate_body_input_stops_when_safety_button_is_missing():
    sample = _sample(10.0)
    del sample["controller_data"]["right_secondary_click"]

    action, result, reason = evaluate_body_input(
        _FakeReader(sample),
        active=True,
        timeout_s=0.5,
        monotonic_now=10.0,
    )

    assert action is InputWatchdogAction.STOP
    assert result is None
    assert "right controller is missing required secondary_click" in reason


def test_evaluate_body_input_rejects_reader_without_source_timestamps():
    reader = _FakeReader(_sample(10.0))
    reader.supports_trusted_source_timestamps = False

    action, result, reason = evaluate_body_input(
        reader,
        active=True,
        timeout_s=0.5,
        monotonic_now=10.0,
    )

    assert action is InputWatchdogAction.STOP
    assert result is None
    assert "does not expose trusted source frame timestamps" in reason


@pytest.mark.parametrize(
    ("field", "value", "reason_fragment"),
    [
        ("left_trigger_value", 1.5, "trigger/grip is outside expected range"),
        ("left_primary_click", float("inf"), "button state is outside expected range"),
    ],
)
def test_evaluate_body_input_stops_on_out_of_range_controller_input(
    field,
    value,
    reason_fragment,
):
    sample = _sample(10.0)
    sample["controller_data"][field] = value

    action, result, reason = evaluate_body_input(
        _FakeReader(sample),
        active=True,
        timeout_s=0.5,
        monotonic_now=10.0,
    )

    assert action is InputWatchdogAction.STOP
    assert result is None
    assert reason_fragment in reason


def test_send_stop_burst_repeats_message_across_window():
    socket = _FakeSocket()
    sleeps = []

    send_stop_burst(
        socket,
        b"stop",
        count=10,
        interval_s=0.02,
        sleep_fn=sleeps.append,
    )

    assert socket.messages == [b"stop"] * 10
    assert sleeps == [0.02] * 9


@pytest.mark.parametrize(
    ("count", "interval_s", "message"),
    [
        (0, 0.02, "count must be at least 1"),
        (1, -0.01, "interval_s must be a non-negative finite value"),
        (1, float("inf"), "interval_s must be a non-negative finite value"),
    ],
)
def test_send_stop_burst_rejects_invalid_configuration(count, interval_s, message):
    with pytest.raises(ValueError, match=message):
        send_stop_burst(
            _FakeSocket(),
            b"stop",
            count=count,
            interval_s=interval_s,
        )
