import numpy as np
import pytest

from gear_sonic.utils.teleop import input_readers
from gear_sonic.utils.teleop.input_readers import get_fresh_body_sample


def test_pico_reader_freshness_uses_body_timestamp(monkeypatch):
    class _FakeXrt:
        @staticmethod
        def get_body_timestamp_ns():
            return 123

        @staticmethod
        def get_time_stamp_ns():
            return 999

    monkeypatch.setattr(input_readers, "xrt", _FakeXrt())

    assert input_readers.PicoReader().get_timestamp_ns() == 123


def _native_controller_snapshot(
    timestamp_ns,
    *,
    health_sequence=None,
    health_timestamp_ns=None,
    health_valid=True,
):
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    snapshot = {
        "left_pose": pose,
        "right_pose": pose,
        "left_trigger_value": 0.0,
        "right_trigger_value": 0.0,
        "left_squeeze_value": 0.0,
        "right_squeeze_value": 0.0,
        "left_thumbstick": [0.0, 0.0],
        "right_thumbstick": [0.0, 0.0],
        "left_thumbstick_click": False,
        "right_thumbstick_click": False,
        "left_primary_click": False,
        "left_secondary_click": False,
        "right_primary_click": False,
        "right_secondary_click": False,
        "left_menu_button": False,
        "right_menu_button": False,
        "left_timestamp_ns": timestamp_ns,
        "right_timestamp_ns": timestamp_ns,
    }
    if health_sequence is not None:
        source_timestamp_ns = (
            timestamp_ns if health_timestamp_ns is None else health_timestamp_ns
        )
        snapshot.update(
            {
                "health_supported": True,
                "health_available": True,
                "health_valid": health_valid,
                "health_schema_version": 1,
                "health_sample_sequence": health_sequence,
                "health_timestamp_ns": source_timestamp_ns,
                "health_left_device_valid": health_valid,
                "health_left_is_tracked_available": health_valid,
                "health_left_is_tracked": health_valid,
                "health_left_tracking_state_available": health_valid,
                "health_left_tracking_state": 3 if health_valid else 0,
                "health_left_valid": health_valid,
                "health_right_device_valid": health_valid,
                "health_right_is_tracked_available": health_valid,
                "health_right_is_tracked": health_valid,
                "health_right_tracking_state_available": health_valid,
                "health_right_tracking_state": 3 if health_valid else 0,
                "health_right_valid": health_valid,
            }
        )
    return snapshot


def _native_body_health_snapshot(sequence):
    return {
        "health_supported": True,
        "health_available": True,
        "health_valid": True,
        "health_schema_version": 1,
        "health_sample_sequence": sequence,
        "health_timestamp_ns": sequence * 1_000_000,
        "health_calibration_result": 0,
        "health_calibrated": True,
        "health_tracking_mode": 0,
        "health_connect_state_result": 0,
        "health_tracker_count": 2,
        "health_unique_tracker_count": 2,
        "health_body_state_result": 0,
        "health_is_tracking": True,
        "health_tracking_state_code": 0,
        "health_body_state_code": 1,
        "health_body_error_code": 0,
        "health_connected_band_count": 2,
        "health_body_data_result": 0,
        "health_body_role_count": 24,
    }


def test_pico_reader_requires_second_controller_source_frame():
    reader = input_readers.PicoReader()

    reader._update_controller_freshness(10.0, _native_controller_snapshot(100))  # noqa: SLF001
    assert reader._controller_last_new_data_time == {"left": None, "right": None}  # noqa: SLF001

    reader._update_controller_freshness(10.1, _native_controller_snapshot(100))  # noqa: SLF001
    assert reader._controller_last_new_data_time == {"left": None, "right": None}  # noqa: SLF001

    reader._update_controller_freshness(10.2, _native_controller_snapshot(101))  # noqa: SLF001
    assert reader._controller_last_new_data_time == {  # noqa: SLF001
        "left": 10.2,
        "right": 10.2,
    }


def test_pico_reader_rejects_nonpositive_and_bootstrap_body_frames(monkeypatch):
    reader = input_readers.PicoReader()

    class _FakeXrt:
        def __init__(self):
            self.stamps = iter((0, 100, 200, 300))
            self.accepted_body_snapshots = 0
            self.controller_sequence = 0

        def get_body_snapshot(self):
            stamp = next(self.stamps)
            if stamp == 300:
                self.accepted_body_snapshots += 1
                reader._stop.set()  # noqa: SLF001
            return {
                "available": True,
                "timestamp_ns": stamp,
                "poses": _body_sample()["body_poses_np"],
                **_native_body_health_snapshot(
                    {0: 1, 100: 2, 200: 3, 300: 4}[stamp]
                ),
            }

        def get_controller_snapshot(self):
            self.controller_sequence += 1
            packet_timestamp_ns = (self.controller_sequence + 2) * 1_000_000
            return _native_controller_snapshot(
                packet_timestamp_ns,
                health_sequence=self.controller_sequence,
                health_timestamp_ns=packet_timestamp_ns,
            )

        @staticmethod
        def get_motion_tracker_snapshot():
            return {
                "count": 2,
                "serial_numbers": ["ankle-left", "ankle-right"],
                "timestamp_ns": 200,
            }

    fake_xrt = _FakeXrt()
    monkeypatch.setattr(input_readers, "xrt", fake_xrt)
    monkeypatch.setattr(input_readers.time, "sleep", lambda _seconds: None)

    reader._run()  # noqa: SLF001

    assert fake_xrt.accepted_body_snapshots == 1
    assert reader.get_latest()["timestamp_ns"] == 300


def test_pico_reader_preserves_complete_atomic_xr24_ankle_snapshot(monkeypatch):
    reader = input_readers.PicoReader()
    body_velocities = np.full((24, 6), 0.125)
    body_accelerations = np.full((24, 6), 0.25)
    tracker_poses = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0],
            [0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    tracker_velocities = np.full((2, 6), 0.375)
    tracker_accelerations = np.full((2, 6), 0.5)

    class _FakeXrt:
        def __init__(self):
            self.sequence = 0
            self.current_timestamp_ns = 0

        def get_xr24_ankle_snapshot(self):
            self.sequence += 1
            self.current_timestamp_ns = self.sequence * 1_000_000
            if self.sequence == 4:
                reader._stop.set()  # noqa: SLF001
            body = {
                "available": True,
                "timestamp_ns": self.current_timestamp_ns,
                "poses": _body_sample()["body_poses_np"],
                "velocities": body_velocities,
                "accelerations": body_accelerations,
                "joint_timestamps_ns": [self.current_timestamp_ns] * 24,
                "sample_sequence": self.sequence,
                **_native_body_health_snapshot(self.sequence),
            }
            trackers = {
                "available": True,
                "count": 2,
                "timestamp_ns": self.current_timestamp_ns,
                "serial_numbers": ["ankle-left", "ankle-right"],
                "poses": tracker_poses,
                "velocities": tracker_velocities,
                "accelerations": tracker_accelerations,
            }
            return {
                "contract": "xrt_xr24_plus_two_ankles_atomic_v1",
                "derivative_layout_contract": (
                    "linear_xyz_then_angular_xyz_v1"
                ),
                "source_coherence_contract": (
                    "same_packet_xr24_plus_raw_ankles_v1"
                ),
                "body": body,
                "trackers": trackers,
            }

        def get_controller_snapshot(self):
            return _native_controller_snapshot(
                self.current_timestamp_ns,
                health_sequence=self.sequence,
                health_timestamp_ns=self.current_timestamp_ns,
            )

        @staticmethod
        def get_motion_tracker_snapshot():
            raise AssertionError("atomic snapshot must supply tracker data")

    monkeypatch.setattr(input_readers, "xrt", _FakeXrt())
    monkeypatch.setattr(input_readers.time, "sleep", lambda _seconds: None)

    reader._run()  # noqa: SLF001

    sample = reader.get_latest()
    assert sample is not None
    assert (
        sample["xrt_atomic_snapshot_contract"]
        == "xrt_xr24_plus_two_ankles_atomic_v1"
    )
    np.testing.assert_array_equal(sample["body_velocities_np"], body_velocities)
    np.testing.assert_array_equal(
        sample["body_accelerations_np"],
        body_accelerations,
    )
    assert sample["body_joint_timestamps_ns"] == (
        sample["timestamp_ns"],
    ) * 24
    np.testing.assert_array_equal(
        sample["motion_tracker_poses_np"],
        tracker_poses,
    )
    np.testing.assert_array_equal(
        sample["motion_tracker_velocities_np"],
        tracker_velocities,
    )
    np.testing.assert_array_equal(
        sample["motion_tracker_accelerations_np"],
        tracker_accelerations,
    )
    assert (
        sample["motion_tracker_source_timestamp_ns"]
        == sample["timestamp_ns"]
    )


def test_pico_reader_does_not_promote_retained_cross_mode_trackers(monkeypatch):
    reader = input_readers.PicoReader()
    body_velocities = np.full((24, 6), 0.125)
    body_accelerations = np.full((24, 6), 0.25)

    class _FakeXrt:
        def __init__(self):
            self.sequence = 0
            self.current_timestamp_ns = 0

        @staticmethod
        def get_xr24_ankle_snapshot():
            return {
                "contract": "xrt_xr24_plus_retained_motion_unapproved_v0",
                "derivative_layout_contract": (
                    "linear_xyz_then_angular_xyz_v1"
                ),
                "source_coherence_contract": (
                    "mutually_exclusive_body_and_motion_retained_state_v0"
                ),
                "body": {},
                "trackers": {},
            }

        def get_body_snapshot(self):
            self.sequence += 1
            self.current_timestamp_ns = self.sequence * 1_000_000
            if self.sequence == 4:
                reader._stop.set()  # noqa: SLF001
            return {
                "contract": (
                    "xrt_xr24_body_tracker_fused_ankles_atomic_v1"
                ),
                "source_coherence_contract": (
                    "same_packet_xr24_body_tracking_v1"
                ),
                "available": True,
                "timestamp_ns": self.current_timestamp_ns,
                "poses": _body_sample()["body_poses_np"],
                "velocities": body_velocities,
                "accelerations": body_accelerations,
                "derivative_layout_contract": (
                    "linear_xyz_then_angular_xyz_v1"
                ),
                "joint_timestamps_ns": [self.current_timestamp_ns] * 24,
                "sample_sequence": self.sequence,
                **_native_body_health_snapshot(self.sequence),
            }

        def get_controller_snapshot(self):
            return _native_controller_snapshot(
                self.current_timestamp_ns,
                health_sequence=self.sequence,
                health_timestamp_ns=self.current_timestamp_ns,
            )

        def get_motion_tracker_snapshot(self):
            return {
                "available": True,
                "count": 2,
                "timestamp_ns": self.current_timestamp_ns,
                "serial_numbers": ["ankle-left", "ankle-right"],
                "poses": np.zeros((2, 7)),
                "velocities": np.zeros((2, 6)),
                "accelerations": np.zeros((2, 6)),
            }

    monkeypatch.setattr(input_readers, "xrt", _FakeXrt())
    monkeypatch.setattr(input_readers.time, "sleep", lambda _seconds: None)

    reader._run()  # noqa: SLF001

    sample = reader.get_latest()
    assert sample is not None
    assert sample["xrt_atomic_snapshot_contract"] is None
    assert (
        sample["xrt_body_derivative_layout_contract"]
        == "linear_xyz_then_angular_xyz_v1"
    )
    assert (
        sample["xrt_body_snapshot_contract"]
        == "xrt_xr24_body_tracker_fused_ankles_atomic_v1"
    )
    assert (
        sample["xrt_body_source_coherence_contract"]
        == "same_packet_xr24_body_tracking_v1"
    )
    np.testing.assert_array_equal(sample["body_velocities_np"], body_velocities)
    np.testing.assert_array_equal(
        sample["body_accelerations_np"],
        body_accelerations,
    )


def test_pico_reader_keeps_stock_body_for_noncontrol_diagnostics(monkeypatch):
    reader = input_readers.PicoReader()

    class _FakeXrt:
        def __init__(self):
            self.stamps = iter((100, 200))

        def get_body_snapshot(self):
            stamp = next(self.stamps)
            if stamp == 200:
                reader._stop.set()  # noqa: SLF001
            return {
                "available": True,
                "timestamp_ns": stamp,
                "poses": _body_sample()["body_poses_np"],
                "health_supported": False,
            }

        @staticmethod
        def get_controller_snapshot():
            return _native_controller_snapshot(200)

        @staticmethod
        def get_motion_tracker_snapshot():
            return {
                "count": 0,
                "serial_numbers": [],
                "timestamp_ns": 0,
            }

    monkeypatch.setattr(input_readers, "xrt", _FakeXrt())
    monkeypatch.setattr(input_readers.time, "sleep", lambda _seconds: None)

    reader._run()  # noqa: SLF001

    sample = reader.get_latest()
    assert sample is not None
    assert sample["body_tracker_health_supported"] is False
    accepted, reason = get_fresh_body_sample(
        reader,
        max_age_s=0.5,
        monotonic_now=sample["timestamp_monotonic"],
        require_body_tracker_health=True,
    )
    assert accepted is None
    assert "does not expose hardened" in reason


class _FakeReader:
    supports_trusted_source_timestamps = True

    def __init__(self, sample=None, disconnected=False):
        self._sample = sample
        self.disconnected = disconnected

    def get_latest(self):
        return self._sample


class _FailingReader:
    disconnected = False
    supports_trusted_source_timestamps = True

    def get_latest(self):
        raise RuntimeError("reader boom")


def _body_sample(timestamp_monotonic=10.0, body_poses=None):
    if body_poses is None:
        body_poses = np.zeros((24, 7), dtype=np.float32)
        body_poses[:, 6] = 1.0
    return {
        "timestamp_monotonic": timestamp_monotonic,
        "body_poses_np": body_poses,
        "body_frame_displacement_m": 0.0,
        "body_frame_angular_displacement_rad": 0.0,
        "dt": 0.02,
    }


def _body_sample_with_tracker_health(timestamp_monotonic=10.0, **overrides):
    sample = {
        **_body_sample(timestamp_monotonic=timestamp_monotonic),
        "body_tracker_health_supported": True,
        "body_tracker_health_available": True,
        "body_tracker_health_valid": True,
        "body_tracker_health_schema_version": 1,
        "body_tracker_health_sample_sequence": 10,
        "body_tracker_health_source_timestamp_ns": 10_000,
        "body_tracker_health_timestamp_monotonic": timestamp_monotonic,
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
    }
    sample.update(overrides)
    return sample


def _body_sample_with_all_tracking_health(timestamp_monotonic=10.0, **overrides):
    sample = {
        **_body_sample_with_tracker_health(timestamp_monotonic=timestamp_monotonic),
        "controller_tracking_health_supported": True,
        "controller_tracking_health_available": True,
        "controller_tracking_health_valid": True,
        "controller_tracking_health_schema_version": 1,
        "controller_tracking_health_sample_sequence": 10,
        "controller_tracking_health_source_timestamp_ns": 10_000,
        "controller_tracking_health_timestamp_monotonic": timestamp_monotonic,
    }
    for side in ("left", "right"):
        sample.update(
            {
                f"controller_tracking_health_{side}_device_valid": True,
                f"controller_tracking_health_{side}_is_tracked_available": True,
                f"controller_tracking_health_{side}_is_tracked": True,
                f"controller_tracking_health_{side}_tracking_state_available": True,
                f"controller_tracking_health_{side}_tracking_state": 3,
                f"controller_tracking_health_{side}_valid": True,
            }
        )
    sample.update(overrides)
    return sample


def test_get_fresh_body_sample_accepts_recent_finite_tracking():
    sample = _body_sample()

    result, reason = get_fresh_body_sample(
        _FakeReader(sample),
        max_age_s=0.5,
        monotonic_now=10.2,
    )

    assert result is sample
    assert reason == ""


def test_get_fresh_body_sample_accepts_required_hardened_tracker_health():
    sample = _body_sample_with_tracker_health()

    result, reason = get_fresh_body_sample(
        _FakeReader(sample),
        max_age_s=0.5,
        monotonic_now=10.2,
        require_body_tracker_health=True,
    )

    assert result is sample
    assert reason == ""


def test_get_fresh_body_sample_accepts_required_controller_tracking_health():
    sample = _body_sample_with_all_tracking_health()

    result, reason = get_fresh_body_sample(
        _FakeReader(sample),
        max_age_s=0.5,
        monotonic_now=10.2,
        require_controller_tracking_health=True,
    )

    assert result is sample
    assert reason == ""


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"controller_tracking_health_supported": False}, "does not expose hardened"),
        ({"controller_tracking_health_left_device_valid": False}, "device is invalid"),
        ({"controller_tracking_health_left_is_tracked": False}, "tracking lost"),
        ({"controller_tracking_health_right_tracking_state": 1}, "position+rotation"),
        ({"controller_tracking_health_source_timestamp_ns": 10_001}, "same packet"),
        ({"controller_tracking_health_timestamp_monotonic": 9.0}, "health is stale"),
    ],
)
def test_get_fresh_body_sample_rejects_bad_controller_tracking_health(
    overrides,
    reason_fragment,
):
    result, reason = get_fresh_body_sample(
        _FakeReader(_body_sample_with_all_tracking_health(**overrides)),
        max_age_s=0.5,
        monotonic_now=10.0,
        require_controller_tracking_health=True,
    )

    assert result is None
    assert reason_fragment in reason


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"body_tracker_health_supported": False}, "does not expose hardened"),
        ({"body_tracker_health_available": False}, "unavailable"),
        ({"body_tracker_health_calibrated": False}, "not calibrated"),
        ({"body_tracker_health_tracker_count": 1}, "connected tracker count"),
        ({"body_tracker_health_unique_tracker_count": 1}, "unique tracker count"),
        ({"body_tracker_health_connected_band_count": 1}, "band count"),
        ({"body_tracker_health_body_state_code": 2}, "not BT_VALID"),
        ({"body_tracker_health_body_data_result": 1}, "data query failed"),
        ({"body_tracker_health_timestamp_monotonic": 9.0}, "health is stale"),
    ],
)
def test_get_fresh_body_sample_rejects_bad_hardened_tracker_health(
    overrides,
    reason_fragment,
):
    result, reason = get_fresh_body_sample(
        _FakeReader(_body_sample_with_tracker_health(**overrides)),
        max_age_s=0.5,
        monotonic_now=10.0,
        require_body_tracker_health=True,
    )

    assert result is None
    assert reason_fragment in reason


def test_get_fresh_body_sample_rejects_reader_without_explicit_timestamp_capability():
    class _UnknownReader:
        disconnected = False

        @staticmethod
        def get_latest():
            return _body_sample()

    result, reason = get_fresh_body_sample(
        _UnknownReader(),
        max_age_s=0.5,
        monotonic_now=10.0,
    )

    assert result is None
    assert "does not expose trusted source frame timestamps" in reason


def _body_pose_with_position(*, joint: int, xyz: tuple[float, float, float]):
    body_poses = np.zeros((24, 7), dtype=np.float32)
    body_poses[:, 6] = 1.0
    body_poses[joint, :3] = xyz
    return body_poses


@pytest.mark.parametrize(
    ("reader", "reason_fragment"),
    [
        (_FakeReader(), "no body-tracking sample"),
        (_FakeReader(_body_sample(), disconnected=True), "disconnected"),
        (_FailingReader(), "reader failed"),
        (_FakeReader(_body_sample(timestamp_monotonic=9.0)), "stale"),
        (_FakeReader(_body_sample(timestamp_monotonic=10.1)), "future"),
        (
            _FakeReader(_body_sample(body_poses=np.full((24, 7), np.nan))),
            "NaN or Inf",
        ),
        (
            _FakeReader(_body_sample(body_poses=np.full((24, 7), "bad", dtype=object))),
            "not numeric",
        ),
        (
            _FakeReader(_body_sample(body_poses=np.zeros((24, 7)))),
            "invalid joint quaternion",
        ),
        (
            _FakeReader(_body_sample(body_poses=np.zeros((3, 7)))),
            "expected at least (24, 7)",
        ),
        (
            _FakeReader(_body_sample(body_poses=_body_pose_with_position(joint=5, xyz=(11.0, 0.0, 0.0)))),
            "outside safe bounds",
        ),
        (
            _FakeReader(_body_sample(body_poses=_body_pose_with_position(joint=5, xyz=(4.0, 0.0, 0.0)))),
            "anatomically implausible",
        ),
        (
            _FakeReader(
                {
                    **_body_sample(),
                    "body_frame_displacement_m": 1.01,
                }
            ),
            "body tracking jumped",
        ),
        (
            _FakeReader(
                {
                    **_body_sample(),
                    "body_frame_angular_displacement_rad": np.pi,
                }
            ),
            "body tracking rotated",
        ),
    ],
)
def test_get_fresh_body_sample_rejects_unsafe_tracking(reader, reason_fragment):
    result, reason = get_fresh_body_sample(
        reader,
        max_age_s=0.5,
        monotonic_now=10.0,
    )

    assert result is None
    assert reason_fragment in reason


def test_get_fresh_body_sample_rejects_invalid_timeout():
    with pytest.raises(ValueError, match="positive finite"):
        get_fresh_body_sample(_FakeReader(_body_sample()), max_age_s=0.0)


def test_get_fresh_body_sample_rejects_nonfinite_manager_clock():
    result, reason = get_fresh_body_sample(
        _FakeReader(_body_sample()),
        max_age_s=0.5,
        monotonic_now=float("nan"),
    )

    assert result is None
    assert "manager monotonic time is not finite" in reason


def _sample_with_motion_trackers(**overrides):
    sample = {
        **_body_sample(),
        "motion_tracker_count": 2,
        "motion_tracker_serial_numbers": ("ankle-left", "ankle-right"),
        "motion_tracker_timestamp_monotonic": 10.0,
    }
    sample.update(overrides)
    return sample


def test_get_fresh_body_sample_accepts_two_unique_fresh_motion_trackers():
    sample = _sample_with_motion_trackers()

    result, reason = get_fresh_body_sample(
        _FakeReader(sample),
        max_age_s=0.5,
        monotonic_now=10.2,
        require_motion_trackers=True,
    )

    assert result is sample
    assert reason == ""


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"motion_tracker_count": 1}, "two are required"),
        (
            {"motion_tracker_serial_numbers": ("ankle-left", "ankle-left")},
            "two unique",
        ),
        ({"motion_tracker_timestamp_monotonic": 9.0}, "motion trackers are stale"),
        ({"motion_tracker_timestamp_monotonic": None}, "no fresh tracking timestamp"),
    ],
)
def test_get_fresh_body_sample_rejects_unsafe_motion_trackers(overrides, reason_fragment):
    result, reason = get_fresh_body_sample(
        _FakeReader(_sample_with_motion_trackers(**overrides)),
        max_age_s=0.5,
        monotonic_now=10.0,
        require_motion_trackers=True,
    )

    assert result is None
    assert reason_fragment in reason


def _controller_snapshot(**overrides):
    inputs = {
        "trigger_value": 0.0,
        "squeeze_value": 0.0,
        "thumbstick_x": 0.0,
        "thumbstick_y": 0.0,
        "thumbstick_click": 0.0,
        "primary_click": 0.0,
        "secondary_click": 0.0,
    }
    inputs.update(overrides)
    return {"inputs": inputs}


def test_controller_ingestion_requires_both_complete_controllers():
    complete = _controller_snapshot()
    assert (
        input_readers._build_controller_dict(  # noqa: SLF001
            {"left_controller": complete, "right_controller": None}
        )
        is None
    )

    incomplete = _controller_snapshot()
    del incomplete["inputs"]["secondary_click"]
    assert (
        input_readers._build_controller_dict(  # noqa: SLF001
            {"left_controller": complete, "right_controller": incomplete}
        )
        is None
    )
