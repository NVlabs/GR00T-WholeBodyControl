from collections import defaultdict, deque
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from gear_sonic.scripts import pico_manager_thread_server as server
from gear_sonic.scripts.pico_manager_thread_server import StreamMode, _mode_after_refusal


@pytest.mark.parametrize(
    "current, new, parent, expected",
    [
        # Driving from the body pose: leave for the mode the button would pick.
        (StreamMode.POSE, StreamMode.POSE, StreamMode.PLANNER, StreamMode.PLANNER),
        (
            StreamMode.PLANNER_VR_3PT,
            StreamMode.PLANNER_VR_3PT,
            StreamMode.PLANNER_FROZEN_UPPER_BODY,
            StreamMode.PLANNER_FROZEN_UPPER_BODY,
        ),
        # A button press into a pose-driven mode this tick does not stick.
        (StreamMode.PLANNER, StreamMode.POSE, StreamMode.PLANNER, StreamMode.PLANNER),
        (
            StreamMode.PLANNER_FROZEN_UPPER_BODY,
            StreamMode.PLANNER_VR_3PT,
            StreamMode.PLANNER,
            StreamMode.PLANNER_FROZEN_UPPER_BODY,
        ),
        # Already independent of the body pose: unchanged.
        (StreamMode.POSE, StreamMode.OFF, StreamMode.PLANNER, StreamMode.OFF),
        (StreamMode.PLANNER, StreamMode.PLANNER, StreamMode.PLANNER, StreamMode.PLANNER),
        (StreamMode.POSE, StreamMode.POSE_PAUSE, StreamMode.PLANNER, StreamMode.POSE_PAUSE),
    ],
)
def test_a_refusal_leaves_every_pose_driven_mode(current, new, parent, expected):
    assert _mode_after_refusal(current, new, parent) == expected


def test_pose_streamer_drops_its_window_when_the_reader_withdraws_its_sample():
    """Poses buffered before a refusal must not be resent or interpolated after it."""
    streamer = server.PoseStreamer.__new__(server.PoseStreamer)
    streamer.reader = SimpleNamespace(get_latest=lambda: None)
    streamer.frame_buffer = defaultdict(lambda: deque(maxlen=5))
    streamer.frame_buffer["frame_index"].extend([1, 2, 3])
    streamer.prev_stamp_ns = 123
    streamer.prev_smpl_pose_np = object()
    streamer.prev_smpl_joints_np = object()
    streamer.prev_body_quat_np = object()
    streamer.next_target_ns = 456
    streamer.prev_epoch = 0
    streamer.buffer_cleared = False

    streamer.run_once()

    assert not streamer.frame_buffer
    assert streamer.prev_stamp_ns is None
    assert streamer.prev_smpl_pose_np is None
    assert streamer.next_target_ns is None
    assert streamer.buffer_cleared


class _StopRun(Exception):
    pass


def _streamer_with_history(epoch):
    streamer = server.PoseStreamer.__new__(server.PoseStreamer)
    streamer.frame_buffer = defaultdict(lambda: deque(maxlen=5))
    streamer.frame_buffer["frame_index"].extend([1, 2, 3])
    streamer.prev_stamp_ns = 123
    streamer.prev_smpl_pose_np = object()
    streamer.prev_smpl_joints_np = object()
    streamer.prev_body_quat_np = object()
    streamer.next_target_ns = 456
    streamer.prev_epoch = epoch
    streamer.buffer_cleared = False
    streamer.skeleton_source_mode = "pico"
    streamer.parent_indices = None
    streamer.device = None
    return streamer


def test_pose_streamer_drops_its_window_when_a_refusal_happened_between_polls(monkeypatch):
    """The reader refused and recovered before this poll, so the cache was never
    seen empty; the new epoch on the recovered sample is what says so."""
    monkeypatch.setattr(server, "compute_from_body_poses", mock.Mock(side_effect=_StopRun))
    streamer = _streamer_with_history(epoch=0)
    streamer.reader = SimpleNamespace(get_latest=lambda: {"body_poses_np": None, "epoch": 1})

    with pytest.raises(_StopRun):  # stop once it moves on to the new sample
        streamer.run_once()

    assert not streamer.frame_buffer
    assert streamer.prev_stamp_ns is None
    assert streamer.prev_epoch == 1


def test_pose_streamer_keeps_its_window_while_the_epoch_holds(monkeypatch):
    monkeypatch.setattr(server, "compute_from_body_poses", mock.Mock(side_effect=_StopRun))
    streamer = _streamer_with_history(epoch=3)
    streamer.reader = SimpleNamespace(get_latest=lambda: {"body_poses_np": None, "epoch": 3})

    with pytest.raises(_StopRun):
        streamer.run_once()

    assert list(streamer.frame_buffer["frame_index"]) == [1, 2, 3]
    assert streamer.prev_stamp_ns == 123


class _ClearingReader:
    """Replays Kevin's schedule: the timestamp read always finds the cache
    cleared (0), but a valid frame lands before the sample read."""

    def __init__(self):
        self.sample = {"body_poses_np": np.zeros((24, 7)), "epoch": 0}

    def get_timestamp_ns(self):
        return 0

    def get_latest(self):
        return self.sample


def _planner_streamer(monkeypatch, reader):
    sent = []
    monkeypatch.setattr(
        server, "build_planner_message", lambda mode, movement, facing, **kw: (mode, movement, kw)
    )
    monkeypatch.setattr(server, "get_abxy_buttons", lambda reader: (False,) * 4)
    monkeypatch.setattr(server, "get_controller_axes", lambda reader: (0.0, 1.0, 0.0, 0.0))
    monkeypatch.setattr(server, "get_controller_inputs", lambda reader: (False, 0.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(
        server,
        "compute_hand_joints_from_inputs",
        lambda *a: (np.zeros(7), np.zeros(7)),
    )
    streamer = server.PlannerStreamer.__new__(server.PlannerStreamer)
    streamer.socket = SimpleNamespace(send=sent.append)
    streamer.reader = reader
    streamer.three_point = SimpleNamespace(process_smpl_pose=lambda poses: np.ones((3, 7)))
    streamer.dt = 0.0
    streamer.mode = server.LocomotionMode.WALK
    streamer.prev_ab = streamer.prev_xy = False
    streamer.yaw_accumulator = server.YawAccumulator()
    streamer.last_send = 0.0
    streamer.last_xrt_timestamp = None
    streamer.last_stream_mode = None
    streamer.left_hand_ik_solver = streamer.right_hand_ik_solver = None
    return streamer, sent


def test_leaving_vr_3pt_sends_a_packet_without_vr_even_on_a_stale_timestamp(monkeypatch):
    streamer, sent = _planner_streamer(monkeypatch, _ClearingReader())

    streamer.run_once(StreamMode.PLANNER_VR_3PT)
    assert sent[-1][2]["vr_3pt_position"] is not None  # the VR target went out

    # The refusal fallback: same (cleared) timestamp, new mode.
    streamer.run_once(StreamMode.PLANNER)
    assert len(sent) == 2, "the fallback packet was skipped as stale"
    mode, movement, fields = sent[-1]
    assert fields["vr_3pt_position"] is None
    assert fields["vr_3pt_orientation"] is None
    # Nothing fresh to act on, so it stands still rather than replaying the stick.
    assert mode == server.LocomotionMode.IDLE.value
    assert movement == [0.0, 0.0, 0.0]

    # Still stale, mode unchanged: back to sending nothing.
    streamer.run_once(StreamMode.PLANNER)
    assert len(sent) == 2


def test_legacy_loop_refuses_isaac_teleop_before_opening_anything(monkeypatch):
    """It cannot act on a refusal, so it must not start streaming at all."""
    opened = mock.Mock()
    monkeypatch.setattr(server, "_init_input_source", opened)
    with pytest.raises(SystemExit, match="--manager"):
        server.run_pico(input_source="isaac-teleop")
    opened.assert_not_called()
