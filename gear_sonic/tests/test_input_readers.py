import threading
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np

from gear_sonic.utils.teleop import input_readers
from gear_sonic.utils.teleop.input_readers import _body_data_to_24x7


class MockBodyJoints:
    """Match the indexed accessor exposed by Isaac Teleop 1.4 BodyJoints."""

    def __init__(self, joints):
        self._joints = joints

    def joints(self, index):
        return self._joints[index]


def _joint(position, orientation, *, is_valid=True):
    return SimpleNamespace(
        is_valid=is_valid,
        pose=SimpleNamespace(
            position=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
            orientation=SimpleNamespace(
                x=orientation[0],
                y=orientation[1],
                z=orientation[2],
                w=orientation[3],
            ),
        ),
    )


def test_body_data_to_24x7_accepts_isaac_teleop_1_4_schema():
    joints = [
        _joint(
            (index + 0.1, index + 0.2, index + 0.3),
            (0.0, 0.0, 0.0, 1.0),
        )
        for index in range(24)
    ]
    body_pose = SimpleNamespace(joints=MockBodyJoints(joints))

    result = _body_data_to_24x7(body_pose)

    assert result is not None
    assert result.shape == (24, 7)
    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result[0],
        np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result[23],
        np.array([23.1, 23.2, 23.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


def test_body_data_to_24x7_returns_none_when_all_joints_are_invalid():
    joints = [
        _joint((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), is_valid=False)
        for _ in range(24)
    ]
    body_pose = SimpleNamespace(joints=MockBodyJoints(joints))

    assert _body_data_to_24x7(body_pose) is None


def test_body_data_to_24x7_keeps_ros2_bridge_dictionary_compatibility():
    body_pose = {
        "joint_positions": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "joint_orientations": [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    }

    result = _body_data_to_24x7(body_pose)

    assert result is not None
    assert result.shape == (24, 7)
    np.testing.assert_allclose(
        result[0],
        np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result[1],
        np.array([0.4, 0.5, 0.6, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(result[2:], np.zeros((22, 7), dtype=np.float32))


def test_clear_latest_is_not_undone_by_a_frame_already_in_flight():
    """A frame read before clear_latest() must not land in the cache after it.

    clear_latest() exists so a caller -- calibration, say -- can discard a pose
    captured against a reference frame that no longer applies. Emptying the
    cache is not enough on its own: the reader thread may already be midway
    through a frame it acquired *before* the call, and publishing that frame
    afterwards restores exactly the pose the caller discarded, with no sign
    that anything went wrong.

    The reader is parked inside its first frame read here, so the clear lands
    squarely in that window.
    """
    in_flight = threading.Event()
    stop_reading = threading.Event()
    frames_read = []

    class _BlockingClient:
        def _get_tracker_data(self):
            frames_read.append(1)
            if len(frames_read) == 1:
                in_flight.wait(5.0)  # held open across the clear
            else:
                # No later frame arrives, so anything in the cache at the end
                # came from the frame that was in flight across the clear.
                stop_reading.wait(5.0)
            return SimpleNamespace(full_body=object(), full_body_source="pico")

        def get_timestamp_ns(self):
            return time.monotonic_ns()

        def close(self):
            pass

    poses = np.tile(np.array([0.0, 0, 0, 0, 0, 0, 1.0]), (24, 1))

    with mock.patch.object(input_readers, "IsaacTeleopClient", lambda **kw: _BlockingClient()):
        reader = input_readers.IsaacTeleopReader(skeleton_profile="pico")

    # Pose conversion and the geometry guards are not what is under test here.
    with mock.patch.object(input_readers, "_body_data_to_24x7", lambda data: poses.copy()), \
         mock.patch.object(input_readers, "match_orientation_convention", lambda *a, **kw: None), \
         mock.patch.object(input_readers, "_build_controller_dict", lambda raw: {}):
        thread = threading.Thread(target=reader._run, daemon=True)
        thread.start()
        try:
            time.sleep(0.3)  # reader is now parked inside the first frame read
            reader.clear_latest()
            assert reader.get_latest() is None, "cache not empty right after the clear"

            in_flight.set()  # the pre-clear frame finishes and tries to publish
            time.sleep(0.1)

            assert reader.get_latest() is None, (
                "a frame acquired before clear_latest() was published after it, "
                "restoring the pose the caller discarded"
            )
        finally:
            reader._stop.set()
            in_flight.set()
            stop_reading.set()
            thread.join(timeout=2.0)
