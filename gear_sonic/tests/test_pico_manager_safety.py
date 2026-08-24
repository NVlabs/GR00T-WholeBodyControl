import importlib
import sys
from types import ModuleType
from unittest.mock import patch

import msgpack
import numpy as np
import pytest

from gear_sonic.utils.teleop.zmq.control_session import ControlSessionError


def _load_manager():
    """Load state-machine code without requiring Torch for these CPU-only tests."""
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        torch_stub = ModuleType("torch")
        rotation_stub = ModuleType("gear_sonic.trl.utils.rotation_conversion")
        transform_stub = ModuleType("gear_sonic.trl.utils.torch_transform")

        def unused_torch_function(*_args, **_kwargs):
            raise AssertionError("Torch helper must not run in manager safety tests")

        rotation_stub.decompose_rotation_aa = unused_torch_function
        for name in (
            "angle_axis_to_quaternion",
            "compute_human_joints",
            "quat_apply",
            "quat_inv",
            "quaternion_to_angle_axis",
            "quaternion_to_rotation_matrix",
        ):
            setattr(transform_stub, name, unused_torch_function)

        with patch.dict(
            sys.modules,
            {
                "torch": torch_stub,
                rotation_stub.__name__: rotation_stub,
                transform_stub.__name__: transform_stub,
            },
        ):
            return importlib.import_module("gear_sonic.scripts.pico_manager_thread_server")

    return importlib.import_module("gear_sonic.scripts.pico_manager_thread_server")


manager = _load_manager()


def test_managed_pose_mode_exit_preserves_source_sequence():
    streamer = object.__new__(manager.PoseStreamer)
    streamer.frame_buffer = {"frame_index": [41]}
    streamer.prev_stamp_ns = 1
    streamer.prev_smpl_pose_np = object()
    streamer.prev_smpl_joints_np = object()
    streamer.prev_body_quat_np = object()
    streamer.next_target_ns = 2
    streamer.buffer_cleared = False
    streamer.step = 42
    streamer.receiver_epoch = bytes(range(1, 17))

    streamer.on_mode_exit()

    assert streamer.frame_buffer == {}
    assert streamer.step == 42


class _FakeReader:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class _FakeSocket:
    def __init__(self):
        self.bound_to = None
        self.messages = []
        self.closed = False
        self.socket_options = []

    def setsockopt(self, option, value):
        self.socket_options.append((option, value))

    def bind(self, endpoint):
        self.bound_to = endpoint

    def send(self, message):
        self.messages.append(message)

    def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self, socket):
        self._socket = socket
        self.terminated = False

    def socket(self, _socket_type):
        return self._socket

    def term(self):
        self.terminated = True


class _FakeThreePointPose:
    def __init__(self, **_kwargs):
        self.calibrations = []
        self.calibration_results = []
        self.closed = False

    def calibrate_now(self, body_poses):
        self.calibrations.append(body_poses)
        if self.calibration_results:
            return self.calibration_results.pop(0)
        return True

    def close(self):
        self.closed = True


class _FakePoseStreamer:
    def __init__(self, **_kwargs):
        self.samples = []
        self.exit_count = 0
        self.reset_count = 0
        self.session_tokens = None
        self.closed = False

    def run_once(self, sample):
        self.samples.append(sample)

    def on_mode_exit(self):
        self.exit_count += 1

    def reset_yaw(self):
        self.reset_count += 1

    def set_control_session(self, receiver_epoch, publisher_session):
        self.session_tokens = (receiver_epoch, publisher_session)

    def close(self):
        self.closed = True


class _FakePlannerStreamer:
    def __init__(self, **_kwargs):
        self.calls = []
        self.reset_count = 0
        self.frozen_target_results = []
        self.frozen_target_attempts = 0
        self.recalibration_results = []
        self.recalibration_attempts = 0
        self.session_tokens = None
        self.closed = False

    def run_once(self, stream_mode, sample):
        self.calls.append((stream_mode, sample))

    def reset_yaw(self):
        self.reset_count += 1

    def save_upper_body_position_target(self):
        self.frozen_target_attempts += 1
        if self.frozen_target_results:
            return self.frozen_target_results.pop(0)
        return True

    def recalibrate_for_vr3pt(self):
        self.recalibration_attempts += 1
        if self.recalibration_results:
            return self.recalibration_results.pop(0)
        return True

    def set_control_session(self, receiver_epoch, publisher_session):
        self.session_tokens = (receiver_epoch, publisher_session)

    def close(self):
        self.closed = True


class _WatchdogFeed:
    def __init__(self, events):
        self._events = iter(events)
        self.active_states = []

    def __call__(self, _reader, *, active, timeout_s):
        assert timeout_s == pytest.approx(0.5)
        self.active_states.append(active)
        event = next(self._events)
        if isinstance(event, BaseException):
            raise event
        return event


class _Harness:
    def __init__(self, monkeypatch, events, feedback_events=()):
        self.reader = _FakeReader()
        self.socket = _FakeSocket()
        self.context = _FakeContext(self.socket)
        self.three_point = _FakeThreePointPose()
        self.pose_streamer = _FakePoseStreamer()
        self.planner_streamer = _FakePlannerStreamer()
        self.watchdog = _WatchdogFeed(events)
        self.stop_calls = []
        self.command_calls = []
        self.feedback_checks = 0
        self.feedback_events = iter(feedback_events)
        self.control_session_closed = False
        self.setup_order = []

        monkeypatch.setattr(manager, "_init_input_source", lambda *_args: self.reader)
        monkeypatch.setattr(manager.zmq, "Context", lambda: self.context)
        monkeypatch.setattr(manager.time, "sleep", lambda _seconds: None)

        def create_three_point(**_kwargs):
            self.setup_order.append("three_point")
            return self.three_point

        def create_pose_streamer(**_kwargs):
            self.setup_order.append("pose_streamer")
            return self.pose_streamer

        def create_planner_streamer(**_kwargs):
            self.setup_order.append("planner_streamer")
            return self.planner_streamer

        monkeypatch.setattr(manager, "ThreePointPose", create_three_point)
        monkeypatch.setattr(manager, "PoseStreamer", create_pose_streamer)
        monkeypatch.setattr(manager, "PlannerStreamer", create_planner_streamer)
        monkeypatch.setattr(manager, "evaluate_body_input", self.watchdog)
        monkeypatch.setattr(
            manager,
            "_send_stop_burst",
            lambda _socket, *, planner, control_session=None: self.stop_calls.append(planner),
        )
        monkeypatch.setattr(manager, "pack_pose_message", lambda *_args, **_kwargs: b"state")

        def record_command(*, start, stop, planner):
            call = {"start": start, "stop": stop, "planner": planner}
            self.command_calls.append(call)
            return repr(call).encode()

        harness = self

        class FakeControlSession:
            receiver_epoch = bytes(range(1, 17))
            publisher_session = bytes(range(17, 33))

            def claim(self, _socket, *, timeout_s):
                assert timeout_s == pytest.approx(5.0)
                harness.setup_order.append("claim")

            def verify_feedback(self):
                harness.feedback_checks += 1
                try:
                    event = next(harness.feedback_events)
                except StopIteration:
                    return
                if isinstance(event, BaseException):
                    raise event

            def build_command(self, *, start, stop, planner, claim=False):
                assert not claim
                return record_command(start=start, stop=stop, planner=planner)

            def close(self):
                harness.control_session_closed = True

        monkeypatch.setattr(
            manager,
            "ControlSessionClient",
            lambda **_kwargs: self.setup_order.append("control_session") or FakeControlSession(),
        )

    def run(self):
        manager.run_pico_manager(input_timeout_s=0.5)

    def assert_cleaned_up(self):
        assert self.reader.stopped
        assert self.three_point.closed
        assert self.pose_streamer.closed
        assert self.planner_streamer.closed
        assert self.control_session_closed
        assert self.socket.closed
        assert self.context.terminated


def _sample(*, start_combo=False, ax_pressed=False, by_pressed=False, left_axis_click=False):
    body_poses = np.zeros((24, 7), dtype=np.float32)
    body_poses[:, 6] = 1.0
    return {
        "body_poses_np": body_poses,
        "controller_data": {
            "right_primary_click": float(start_combo or ax_pressed),
            "right_secondary_click": float(start_combo or by_pressed),
            "left_primary_click": float(start_combo or ax_pressed),
            "left_secondary_click": float(start_combo or by_pressed),
            "left_thumbstick_click": float(left_axis_click),
            "left_squeeze_value": 0.0,
        },
    }


def _continue(sample):
    return manager.InputWatchdogAction.CONTINUE, sample, ""


def test_manager_rejects_input_without_source_frame_timestamps(monkeypatch):
    monkeypatch.setattr(
        manager,
        "_init_input_source",
        lambda *_args: pytest.fail("unsafe input source must be rejected before initialization"),
    )

    with pytest.raises(ValueError, match="requires XRoboToolkit"):
        manager.run_pico_manager(input_source="isaac-teleop")


def test_manager_constructs_streamers_before_permanent_claim(monkeypatch):
    harness = _Harness(monkeypatch, [KeyboardInterrupt()])

    harness.run()

    assert harness.socket.socket_options == [(manager.zmq.LINGER, 0)]
    assert harness.setup_order == [
        "control_session",
        "three_point",
        "pose_streamer",
        "planner_streamer",
        "claim",
    ]
    expected_tokens = (bytes(range(1, 17)), bytes(range(17, 33)))
    assert harness.pose_streamer.session_tokens == expected_tokens
    assert harness.planner_streamer.session_tokens == expected_tokens
    assert harness.socket.messages == []
    harness.assert_cleaned_up()


@pytest.mark.parametrize(
    "failure_stage",
    ["socket_option", "bind", "control_session", "three_point", "pose_streamer", "planner_streamer"],
)
def test_setup_failure_closes_every_constructed_resource(monkeypatch, failure_stage):
    reader = _FakeReader()
    socket = _FakeSocket()
    context = _FakeContext(socket)
    created = {}
    claim_calls = []

    class Closable:
        def __init__(self, name):
            self.name = name
            self.closed = False
            created[name] = self

        def close(self):
            self.closed = True

        def claim(self, *_args, **_kwargs):
            claim_calls.append(self.name)

    def factory(name):
        def create(**_kwargs):
            if failure_stage == name:
                raise RuntimeError(f"{name} failed")
            return Closable(name)

        return create

    if failure_stage == "socket_option":
        socket.setsockopt = lambda *_args: (_ for _ in ()).throw(RuntimeError("socket option failed"))
    elif failure_stage == "bind":
        socket.bind = lambda _endpoint: (_ for _ in ()).throw(RuntimeError("bind failed"))

    monkeypatch.setattr(manager, "_init_input_source", lambda *_args: reader)
    monkeypatch.setattr(manager.zmq, "Context", lambda: context)
    monkeypatch.setattr(manager.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        manager,
        "ControlSessionClient",
        factory("control_session"),
    )
    monkeypatch.setattr(manager, "ThreePointPose", factory("three_point"))
    monkeypatch.setattr(manager, "PoseStreamer", factory("pose_streamer"))
    monkeypatch.setattr(manager, "PlannerStreamer", factory("planner_streamer"))

    with pytest.raises(RuntimeError, match="failed"):
        manager.run_pico_manager(input_timeout_s=0.5)

    assert reader.stopped
    assert socket.closed
    assert context.terminated
    assert all(resource.closed for resource in created.values())
    assert claim_calls == []


def test_cleanup_closes_publisher_with_zero_linger_before_context_term():
    cleanup_order = []

    class LingerAwareSocket:
        def close(self, *, linger=None):
            cleanup_order.append(("socket", linger))

    class Context:
        def term(self):
            cleanup_order.append(("context", None))

    manager._cleanup_manager_resources(socket=LingerAwareSocket(), context=Context())

    assert cleanup_order == [("socket", 0), ("context", None)]


def test_manager_requires_release_then_repress_before_engagement(monkeypatch):
    held = _sample(start_combo=True)
    released = _sample(start_combo=False)
    harness = _Harness(
        monkeypatch,
        [
            _continue(held),
            _continue(held),
            _continue(released),
            _continue(held),
            KeyboardInterrupt(),
        ],
    )

    harness.run()

    assert harness.watchdog.active_states == [False, False, False, False, True]
    assert len(harness.three_point.calibrations) == 1
    assert harness.three_point.calibrations[0] is held["body_poses_np"]
    assert len(harness.planner_streamer.calls) == 1
    stream_mode, sample = harness.planner_streamer.calls[0]
    assert stream_mode is manager.StreamMode.PLANNER
    assert sample is held
    assert harness.command_calls == [{"start": True, "stop": False, "planner": True}]
    assert harness.stop_calls == [True]
    harness.assert_cleaned_up()


def test_manager_stays_off_after_failed_calibration_until_release_and_retry(monkeypatch):
    released = _sample(start_combo=False)
    pressed = _sample(start_combo=True)
    harness = _Harness(
        monkeypatch,
        [
            _continue(released),
            _continue(pressed),
            _continue(pressed),
            _continue(released),
            _continue(pressed),
            KeyboardInterrupt(),
        ],
    )
    harness.three_point.calibration_results = [False, True]

    harness.run()

    assert harness.watchdog.active_states == [False, False, False, False, False, True]
    assert len(harness.three_point.calibrations) == 2
    assert len(harness.planner_streamer.calls) == 1
    assert harness.command_calls == [{"start": True, "stop": False, "planner": True}]
    assert harness.stop_calls == [True]
    harness.assert_cleaned_up()


def test_frozen_transition_rejects_bad_feedback_until_button_release_and_repress(monkeypatch):
    released = _sample()
    start = _sample(start_combo=True)
    ax_pressed = _sample(ax_pressed=True)
    by_pressed = _sample(by_pressed=True)
    harness = _Harness(
        monkeypatch,
        [
            _continue(released),
            _continue(start),
            _continue(released),
            _continue(ax_pressed),
            _continue(released),
            _continue(by_pressed),
            _continue(by_pressed),
            _continue(released),
            _continue(by_pressed),
            KeyboardInterrupt(),
        ],
    )
    harness.planner_streamer.frozen_target_results = [False, True]

    harness.run()

    assert harness.planner_streamer.frozen_target_attempts == 2
    assert [mode for mode, _sample_value in harness.planner_streamer.calls][-1] is (
        manager.StreamMode.PLANNER_FROZEN_UPPER_BODY
    )
    assert harness.pose_streamer.exit_count == 1
    assert harness.command_calls == [
        {"start": True, "stop": False, "planner": True},
        {"start": True, "stop": False, "planner": False},
        {"start": True, "stop": False, "planner": True},
    ]
    assert harness.stop_calls == [True]
    harness.assert_cleaned_up()


def test_vr3pt_transition_rejects_bad_feedback_until_click_release_and_repress(monkeypatch):
    released = _sample()
    start = _sample(start_combo=True)
    axis_pressed = _sample(left_axis_click=True)
    harness = _Harness(
        monkeypatch,
        [
            _continue(released),
            _continue(start),
            _continue(released),
            _continue(axis_pressed),
            _continue(axis_pressed),
            _continue(released),
            _continue(axis_pressed),
            KeyboardInterrupt(),
        ],
    )
    harness.planner_streamer.recalibration_results = [False, True]

    harness.run()

    assert harness.planner_streamer.recalibration_attempts == 2
    streamed_modes = [mode for mode, _sample_value in harness.planner_streamer.calls]
    assert streamed_modes[2:5] == [
        manager.StreamMode.PLANNER,
        manager.StreamMode.PLANNER,
        manager.StreamMode.PLANNER,
    ]
    assert streamed_modes[-1] is manager.StreamMode.PLANNER_VR_3PT
    assert harness.command_calls == [
        {"start": True, "stop": False, "planner": True},
        {"start": True, "stop": False, "planner": True},
    ]
    assert harness.stop_calls == [True]
    harness.assert_cleaned_up()


def test_failed_calibration_clears_all_partial_state(monkeypatch):
    three_point = object.__new__(manager.ThreePointPose)
    three_point.log_prefix = "test"
    three_point._calibration_pending = True
    three_point._calibration_neck_quat_inv = np.ones(4)
    three_point._calibration_lwrist_offset = np.ones(3)
    three_point._calibration_rwrist_offset = np.ones(3)
    three_point._calibration_lwrist_rot_offset = object()
    three_point._calibration_rwrist_rot_offset = object()
    three_point._override_robot_q = np.ones(29)

    def fail_after_partial_mutation(_pose):
        three_point._calibration_neck_quat_inv = np.full(4, 2.0)
        three_point._calibration_lwrist_offset = np.full(3, 2.0)
        raise RuntimeError("synthetic FK failure")

    monkeypatch.setattr(manager, "_process_3pt_pose", lambda _poses: np.zeros((3, 7)))
    three_point._capture_calibration = fail_after_partial_mutation

    assert not three_point.calibrate_now(np.zeros((24, 7)))
    assert not three_point._calibration_pending
    assert three_point._calibration_neck_quat_inv is None
    assert three_point._calibration_lwrist_offset is None
    assert three_point._calibration_rwrist_offset is None
    assert three_point._calibration_lwrist_rot_offset is None
    assert three_point._calibration_rwrist_rot_offset is None
    assert three_point._override_robot_q is None


class _FeedbackPoller:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.closed = False

    def get_data(self):
        return next(self.payloads)

    def close(self):
        self.closed = True


def _feedback_payload(**overrides):
    payload = {
        "index": 1,
        "timestamp_monotonic_ns": 10_000_000_000,
        "body_q_measured": [0.01 * index for index in range(29)],
        "left_hand_q": [0.0] * 7,
        "right_hand_q": [0.0] * 7,
        "left_hand_feedback_valid": True,
        "right_hand_feedback_valid": True,
        # Viz aliases are deliberately not consumed by FeedbackReader.
        "left_hand_q_measured": [0.0] * 7,
        "right_hand_q_measured": [0.0] * 7,
    }
    payload.update(overrides)
    return msgpack.packb(payload, use_bin_type=True)


def _feedback_payload_without(field):
    payload = msgpack.unpackb(_feedback_payload(), raw=False)
    del payload[field]
    return msgpack.packb(payload, use_bin_type=True)


def _feedback_reader(monkeypatch, payloads):
    poller = _FeedbackPoller(payloads)
    monkeypatch.setattr(manager, "ZMQPoller", lambda **_kwargs: poller)
    return manager.FeedbackReader(), poller


def test_feedback_reader_accepts_only_complete_fresh_snapshot(monkeypatch):
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(manager.time, "monotonic_ns", lambda: clock["now_ns"])
    reader, poller = _feedback_reader(monkeypatch, [_feedback_payload(), None, None])

    assert reader.poll_feedback()
    assert reader.full_body_q_measured.shape == (29,)
    assert reader.upper_body_position_target.shape == (17,)
    assert reader.left_hand_position_target.shape == (7,)
    assert reader.right_hand_position_target.shape == (7,)
    assert np.all(np.isfinite(reader.full_body_q_measured))

    clock["now_ns"] = 10_500_000_000
    assert reader.poll_feedback()
    clock["now_ns"] = 10_500_000_001
    assert not reader.poll_feedback()

    reader.close()
    assert poller.closed


@pytest.mark.parametrize(
    "invalid_payload",
    [
        _feedback_payload(body_q_measured=[0.0] * 28),
        _feedback_payload(left_hand_q=[0.0] * 6),
        _feedback_payload(right_hand_q=[0.0] * 8),
        _feedback_payload(body_q_measured=[0.0] * 28 + [float("nan")]),
        _feedback_payload(left_hand_q=[0.0] * 3 + [0.06] + [0.0] * 3),
        _feedback_payload(index=-1),
        _feedback_payload(index=True),
        _feedback_payload_without("index"),
        msgpack.packb([0.0] * 29, use_bin_type=True),
    ],
    ids=[
        "body-shape",
        "left-hand-shape",
        "right-hand-shape",
        "non-finite",
        "joint-bound",
        "negative-index",
        "boolean-index",
        "missing-index",
        "not-map",
    ],
)
def test_feedback_reader_invalid_latest_snapshot_clears_cached_targets(monkeypatch, invalid_payload):
    monkeypatch.setattr(manager.time, "monotonic_ns", lambda: 10_000_000_000)
    reader, _poller = _feedback_reader(monkeypatch, [_feedback_payload(), invalid_payload])

    assert reader.poll_feedback()
    assert not reader.poll_feedback()
    assert reader.full_body_q_measured is None
    assert reader.upper_body_position_target is None
    assert reader.left_hand_position_target is None
    assert reader.right_hand_position_target is None
    assert reader.last_valid_feedback_monotonic_ns is None


def test_feedback_reader_rejects_duplicate_and_regressed_indices(monkeypatch):
    monkeypatch.setattr(manager.time, "monotonic_ns", lambda: 10_000_000_000)
    reader, _poller = _feedback_reader(
        monkeypatch,
        [
            _feedback_payload(index=7),
            _feedback_payload(index=7),
            _feedback_payload(index=6),
            _feedback_payload(index=8),
        ],
    )

    assert reader.poll_feedback()
    assert reader.last_feedback_index == 7
    assert not reader.poll_feedback()
    assert reader.last_feedback_index == 7
    assert not reader.poll_feedback()
    assert reader.last_feedback_index == 7
    assert reader.poll_feedback()
    assert reader.last_feedback_index == 8


@pytest.mark.parametrize(
    "invalid_payload",
    [
        _feedback_payload_without("timestamp_monotonic_ns"),
        _feedback_payload(timestamp_monotonic_ns=-1),
        _feedback_payload(timestamp_monotonic_ns=True),
        _feedback_payload(timestamp_monotonic_ns=10_000_000_000.0),
    ],
    ids=["missing", "negative", "boolean", "float"],
)
def test_feedback_reader_requires_exact_monotonic_source_timestamp(monkeypatch, invalid_payload):
    monkeypatch.setattr(manager.time, "monotonic_ns", lambda: 10_000_000_000)
    reader, _poller = _feedback_reader(monkeypatch, [invalid_payload])

    assert not reader.poll_feedback()
    assert reader.full_body_q_measured is None


@pytest.mark.parametrize(
    "source_timestamp_ns",
    [9_499_999_999, 10_000_000_001],
    ids=["older-than-500ms", "future"],
)
def test_feedback_reader_rejects_stale_or_future_first_queued_sample(monkeypatch, source_timestamp_ns):
    monkeypatch.setattr(manager.time, "monotonic_ns", lambda: 10_000_000_000)
    reader, _poller = _feedback_reader(
        monkeypatch,
        [_feedback_payload(timestamp_monotonic_ns=source_timestamp_ns)],
    )

    assert not reader.poll_feedback()
    assert reader.last_feedback_index is None
    assert reader.last_valid_feedback_monotonic_ns is None


@pytest.mark.parametrize(
    "invalid_payload",
    [
        _feedback_payload_without("left_hand_feedback_valid"),
        _feedback_payload_without("right_hand_feedback_valid"),
        _feedback_payload(left_hand_feedback_valid=False),
        _feedback_payload(right_hand_feedback_valid=False),
        _feedback_payload(left_hand_feedback_valid=1),
        _feedback_payload(right_hand_feedback_valid=1),
    ],
    ids=["left-missing", "right-missing", "left-false", "right-false", "left-int", "right-int"],
)
def test_feedback_reader_requires_explicit_true_hand_feedback_flags(monkeypatch, invalid_payload):
    monkeypatch.setattr(manager.time, "monotonic_ns", lambda: 10_000_000_000)
    reader, _poller = _feedback_reader(monkeypatch, [invalid_payload])

    assert not reader.poll_feedback()
    assert reader.left_hand_position_target is None
    assert reader.right_hand_position_target is None


def test_feedback_reader_uses_canonical_measured_hands_not_viz_aliases(monkeypatch):
    monkeypatch.setattr(manager.time, "monotonic_ns", lambda: 10_000_000_000)
    measured_left = [0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1]
    measured_right = [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1]
    reader, _poller = _feedback_reader(
        monkeypatch,
        [
            _feedback_payload(
                left_hand_q=measured_left,
                right_hand_q=measured_right,
                left_hand_q_measured=[9.0] * 7,
                right_hand_q_measured=[-9.0] * 7,
            )
        ],
    )

    assert reader.poll_feedback()
    np.testing.assert_array_equal(reader.left_hand_position_target, measured_left)
    np.testing.assert_array_equal(reader.right_hand_position_target, measured_right)


@pytest.mark.parametrize(
    "invalid_payload",
    [
        _feedback_payload(body_q_measured=[0.0] * 5 + [0.4] + [0.0] * 23),
        _feedback_payload(left_hand_q=[0.0] * 3 + [0.06] + [0.0] * 3),
        _feedback_payload(right_hand_q=[0.0] * 3 + [-0.06] + [0.0] * 3),
    ],
    ids=["body-left-ankle-roll", "left-hand-joint-3", "right-hand-joint-3"],
)
def test_feedback_reader_enforces_deployed_joint_specific_limits(monkeypatch, invalid_payload):
    monkeypatch.setattr(manager.time, "monotonic_ns", lambda: 10_000_000_000)
    reader, _poller = _feedback_reader(monkeypatch, [invalid_payload])

    assert not reader.poll_feedback()


def test_feedback_reader_rejects_non_loopback_source_clock(monkeypatch):
    monkeypatch.setattr(
        manager,
        "ZMQPoller",
        lambda **_kwargs: pytest.fail("invalid host must be rejected before opening ZMQ"),
    )

    with pytest.raises(ValueError, match="must be loopback"):
        manager.FeedbackReader(zmq_feedback_host="192.168.123.42")


def test_vr3pt_recalibration_never_substitutes_zero_feedback():
    class MissingFeedback:
        full_body_q_measured = None

        @staticmethod
        def poll_feedback():
            return False

    class ThreePoint:
        @staticmethod
        def reset_with_measured_q(_body_q):
            pytest.fail("missing feedback must not schedule recalibration")

    streamer = object.__new__(manager.PlannerStreamer)
    streamer.feedback_reader = MissingFeedback()
    streamer.three_point = ThreePoint()

    assert not streamer.recalibrate_for_vr3pt()


def test_failed_recalibration_preserves_last_committed_frozen_targets():
    class Feedback:
        upper_body_position_target = np.arange(17, dtype=np.float64)
        left_hand_position_target = np.arange(7, dtype=np.float64)
        right_hand_position_target = -np.arange(7, dtype=np.float64)
        full_body_q_measured = np.arange(29, dtype=np.float64)
        polls = 0

        def poll_feedback(self):
            self.polls += 1
            if self.polls == 1:
                return True
            self.upper_body_position_target = None
            self.left_hand_position_target = None
            self.right_hand_position_target = None
            self.full_body_q_measured = None
            return False

    class ThreePoint:
        @staticmethod
        def reset_with_measured_q(_body_q):
            pytest.fail("invalid replacement feedback must not recalibrate")

    streamer = object.__new__(manager.PlannerStreamer)
    streamer.feedback_reader = Feedback()
    streamer.three_point = ThreePoint()
    streamer.frozen_upper_body_position = None
    streamer.frozen_left_hand_position = None
    streamer.frozen_right_hand_position = None

    assert streamer.save_upper_body_position_target()
    frozen_body = streamer.frozen_upper_body_position.copy()
    frozen_left = streamer.frozen_left_hand_position.copy()
    frozen_right = streamer.frozen_right_hand_position.copy()
    assert not streamer.recalibrate_for_vr3pt()
    np.testing.assert_array_equal(streamer.frozen_upper_body_position, frozen_body)
    np.testing.assert_array_equal(streamer.frozen_left_hand_position, frozen_left)
    np.testing.assert_array_equal(streamer.frozen_right_hand_position, frozen_right)


def test_tracking_loss_while_active_sends_one_planner_stop_burst(monkeypatch):
    released = _sample(start_combo=False)
    pressed = _sample(start_combo=True)
    harness = _Harness(
        monkeypatch,
        [
            _continue(released),
            _continue(pressed),
            (manager.InputWatchdogAction.STOP, None, "body tracking is stale"),
        ],
    )

    harness.run()

    assert harness.watchdog.active_states == [False, False, True]
    assert harness.stop_calls == [True]
    harness.assert_cleaned_up()


def test_manual_stop_sends_one_planner_stop_burst(monkeypatch):
    released = _sample(start_combo=False)
    pressed = _sample(start_combo=True)
    harness = _Harness(
        monkeypatch,
        [
            _continue(released),
            _continue(pressed),
            _continue(released),
            _continue(pressed),
        ],
    )

    harness.run()

    assert harness.watchdog.active_states == [False, False, True, True]
    assert harness.stop_calls == [True]
    harness.assert_cleaned_up()


def test_exception_while_active_sends_stop_and_cleans_up(monkeypatch):
    released = _sample(start_combo=False)
    pressed = _sample(start_combo=True)
    harness = _Harness(
        monkeypatch,
        [
            _continue(released),
            _continue(pressed),
            RuntimeError("watchdog exploded"),
        ],
    )

    with pytest.raises(RuntimeError, match="watchdog exploded"):
        harness.run()

    assert harness.watchdog.active_states == [False, False, True]
    assert harness.stop_calls == [True]
    harness.assert_cleaned_up()


def test_control_session_feedback_loss_while_active_sends_stop(monkeypatch):
    released = _sample(start_combo=False)
    pressed = _sample(start_combo=True)
    harness = _Harness(
        monkeypatch,
        [
            _continue(released),
            _continue(pressed),
        ],
        feedback_events=[
            None,
            None,
            ControlSessionError("native receiver epoch changed"),
        ],
    )

    with pytest.raises(ControlSessionError, match="epoch changed"):
        harness.run()

    assert harness.watchdog.active_states == [False, False]
    assert harness.feedback_checks == 3
    assert harness.stop_calls == [True]
    harness.assert_cleaned_up()
