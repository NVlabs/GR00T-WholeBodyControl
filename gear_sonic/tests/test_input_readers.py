import sys
import threading
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from gear_sonic.utils.teleop import input_readers
from gear_sonic.utils.teleop.input_readers import _body_data_to_24x7
from gear_sonic.utils.teleop.skeleton_guards import (
    FROZEN_WINDOW,
    ConventionMatch,
    MatchOutcome,
)


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


_IDENTITY_POSES = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (24, 1))


class _ScriptedClient:
    """Stands in for IsaacTeleopClient: replays one snapshot per poll.

    Each frame is the (24, 7) array the patched _body_data_to_24x7 returns for
    that poll, or None for a poll with no body data. The reader is stopped
    once the frames run out.
    """

    def __init__(self, reader_ref, frames, full_body_source="pico"):
        self._reader_ref = reader_ref
        self._frames = list(frames)
        self._source = full_body_source
        self.polls = 0

    def _get_tracker_data(self):
        if self.polls >= len(self._frames):
            self._reader_ref[0]._stop.set()
            return None
        frame = self._frames[self.polls]
        self.polls += 1
        return SimpleNamespace(
            left_controller=None,
            right_controller=None,
            full_body=frame,
            full_body_source=self._source,
        )

    def get_timestamp_ns(self):
        return time.monotonic_ns()

    def close(self):
        pass


def _run_reader(monkeypatch, frames, skeleton_profile="pico", full_body_source="pico"):
    """Run IsaacTeleopReader._run synchronously over ``frames``; return the reader."""
    reader_ref = [None]
    client = _ScriptedClient(reader_ref, frames, full_body_source)
    monkeypatch.setattr(input_readers, "IsaacTeleopClient", lambda **kw: client)
    monkeypatch.setattr(
        input_readers,
        "_body_data_to_24x7",
        lambda data: None if data is None else np.array(data, copy=True),
    )
    reader = input_readers.IsaacTeleopReader(poll_hz=1000.0, skeleton_profile=skeleton_profile)
    reader_ref[0] = reader
    reader._run()
    return reader


def _live_frames(n, seed=0):
    """``n`` frames with ~1 cm of natural sway, so none of them reads as frozen."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        f = _IDENTITY_POSES.copy()
        f[:, :3] += rng.normal(scale=0.01, size=(24, 3))
        frames.append(f)
    return frames


def _decided(source):
    return ConventionMatch(MatchOutcome.DECIDED, source, {"pico": 0.0, "quest": 60.0}, 60.0)


def _patch_correction(monkeypatch, correct):
    monkeypatch.setitem(
        sys.modules,
        "isaacteleop.retargeting_engine.utilities",
        SimpleNamespace(correct_body_orientations=correct),
    )


def test_reader_forwards_a_live_pico_skeleton(monkeypatch):
    monkeypatch.setattr(input_readers, "match_orientation_convention", lambda *a: _decided("pico"))
    reader = _run_reader(monkeypatch, _live_frames(40))
    assert reader.get_latest() is not None
    assert reader.refusal_count == 0


def test_a_frozen_pico_skeleton_is_refused_and_counted(monkeypatch):
    monkeypatch.setattr(input_readers, "match_orientation_convention", lambda *a: _decided("pico"))
    reader = _run_reader(monkeypatch, [*_live_frames(40), *[_IDENTITY_POSES] * 40])
    assert reader.get_latest() is None
    assert reader.refusal_count > 0


def test_a_contradicting_convention_is_refused_and_counted(monkeypatch):
    monkeypatch.setattr(input_readers, "match_orientation_convention", lambda *a: _decided("quest"))
    reader = _run_reader(monkeypatch, _live_frames(5))
    assert reader.get_latest() is None
    assert reader.refusal_count == 5


def test_clear_latest_is_not_counted_as_a_refusal(monkeypatch):
    """Calibration clears the cache; that must not throw the manager out of POSE."""
    monkeypatch.setattr(input_readers, "match_orientation_convention", lambda *a: _decided("pico"))
    reader = _run_reader(monkeypatch, _live_frames(40))
    reader.clear_latest()
    assert reader.get_latest() is None
    assert reader.refusal_count == 0


def test_a_dropped_frame_does_not_let_a_frozen_pico_skeleton_through(monkeypatch):
    """Gaps used to reset the frozen-pose window, so it never filled."""
    monkeypatch.setattr(input_readers, "match_orientation_convention", lambda *a: _decided("pico"))
    frozen_with_gaps = [None if i % 4 == 3 else _IDENTITY_POSES for i in range(80)]
    reader = _run_reader(monkeypatch, [*frozen_with_gaps, _IDENTITY_POSES])
    assert reader.get_latest() is None


def test_a_dropped_frame_does_not_stall_a_live_pico_skeleton(monkeypatch):
    """The frozen-pose window survives a gap, so live data resumes at once."""
    monkeypatch.setattr(input_readers, "match_orientation_convention", lambda *a: _decided("pico"))
    frames = [*_live_frames(40), None, *_live_frames(1, seed=1)]
    assert _run_reader(monkeypatch, frames).get_latest() is not None


def test_pico_frames_wait_for_the_frozen_window_to_fill(monkeypatch):
    """Until the window fills, frozen and live look alike, so nothing is forwarded."""
    monkeypatch.setattr(input_readers, "match_orientation_convention", lambda *a: _decided("pico"))
    waiting = _run_reader(monkeypatch, _live_frames(FROZEN_WINDOW - 1))
    assert waiting.get_latest() is None
    assert waiting.refusal_count == 0, "warm-up is not a fault"
    assert _run_reader(monkeypatch, _live_frames(FROZEN_WINDOW)).get_latest() is not None


def test_reader_refuses_a_non_finite_frame(monkeypatch):
    monkeypatch.setattr(input_readers, "match_orientation_convention", lambda *a: _decided("pico"))
    bad = _live_frames(1, seed=1)[0]
    bad[5, 3] = np.nan
    reader = _run_reader(monkeypatch, [*_live_frames(40), bad])
    assert reader.get_latest() is None
    assert reader.refusal_count == 1


def test_a_failing_correction_refuses_the_frame_and_keeps_reading(monkeypatch):
    def _correct(orientations, profile):
        raise ValueError(f"unknown profile {profile!r}")

    _patch_correction(monkeypatch, _correct)
    monkeypatch.setattr(input_readers, "match_orientation_convention", lambda *a: _decided("quest"))
    frames = _live_frames(5)
    reader = _run_reader(monkeypatch, frames, skeleton_profile="quest", full_body_source="quest")
    assert reader._client.polls == len(frames)  # the thread survived every failure
    assert reader.get_latest() is None
    assert reader.refusal_count == len(frames)


def test_the_correction_always_runs_with_its_own_profile(monkeypatch):
    """Clearing the profile mid-frame cannot hand the saved function a None profile."""
    seen = []

    def _correct(orientations, profile):
        seen.append(profile)
        return orientations

    _patch_correction(monkeypatch, _correct)
    monkeypatch.setattr(input_readers, "IsaacTeleopClient", lambda **kw: object())
    reader = input_readers.IsaacTeleopReader(skeleton_profile="quest")
    correct_fn, profile = reader._correction
    reader.set_skeleton_profile(None)  # another thread clears it after the read
    correct_fn(_IDENTITY_POSES[:, 3:], profile)
    assert seen == ["quest"]
    assert reader._correction is None


def test_an_unexpected_reader_failure_surfaces_from_get_latest(monkeypatch):
    monkeypatch.setattr(input_readers, "match_orientation_convention", lambda *a: _decided("quest"))
    _patch_correction(monkeypatch, lambda o, p: o)

    def _timestamp_fails(self):
        raise OSError("gone")

    monkeypatch.setattr(_ScriptedClient, "get_timestamp_ns", _timestamp_fails)
    reader = _run_reader(
        monkeypatch, _live_frames(2), skeleton_profile="quest", full_body_source="quest"
    )
    with pytest.raises(OSError, match="gone"):
        reader.get_latest()


def test_explicit_quest_without_the_correction_names_the_fix(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaacteleop.retargeting_engine.utilities", None)
    monkeypatch.setattr(input_readers, "IsaacTeleopClient", lambda **kw: object())
    with pytest.raises(input_readers.SkeletonCorrectionUnavailable, match="IsaacTeleop#921"):
        input_readers.IsaacTeleopReader(skeleton_profile="quest")
