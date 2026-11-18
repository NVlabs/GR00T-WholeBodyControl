from collections import defaultdict, deque
from types import SimpleNamespace

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
    streamer.buffer_cleared = False

    streamer.run_once()

    assert not streamer.frame_buffer
    assert streamer.prev_stamp_ns is None
    assert streamer.prev_smpl_pose_np is None
    assert streamer.next_target_ns is None
    assert streamer.buffer_cleared
