import numpy as np
import pytest

from gear_sonic.utils.teleop.wuji_revo2 import (
    LatestWujiSkeleton,
    REVO2_JOINT_NAMES,
    WujiRevo2Bridge,
    WujiSkeletonFrame,
)


def _skeleton() -> np.ndarray:
    return np.arange(63, dtype=np.float32).reshape(21, 3)


class _FakeRetargeter:
    def __init__(self) -> None:
        self.calls = []

    def retarget_side(self, side, fingertips):
        self.calls.append((side, fingertips.copy()))
        return np.array([-1.0, 2.0, 0.2, 0.8, 1.4, 2.0])


def test_wuji_frame_validates_and_extracts_fingertips():
    frame = WujiSkeletonFrame.from_joints(_skeleton(), "right", sequence=3)

    assert frame.joints.shape == (21, 3)
    assert frame.fingertips.shape == (5, 3)
    np.testing.assert_array_equal(frame.fingertips, _skeleton()[[4, 8, 12, 16, 20]])
    assert frame.joints.flags.writeable is False


def test_latest_store_drains_only_newest_frame():
    store = LatestWujiSkeleton()
    store.submit(_skeleton(), "left", sequence=1, received_monotonic=10.0)
    store.submit(_skeleton() + 1.0, "left", sequence=2, received_monotonic=10.01)

    frame = store.drain_latest(now=10.02, max_age_sec=0.1)

    assert frame is not None
    assert frame.sequence == 2
    assert store.drain_latest(now=10.02, max_age_sec=0.1) is None


def test_bridge_clips_named_revo2_target():
    backend = _FakeRetargeter()
    bridge = WujiRevo2Bridge(backend)
    frame = WujiSkeletonFrame.from_joints(
        _skeleton(), "right", sequence=4, received_monotonic=20.0
    )

    target = bridge.process(frame, now=20.01)

    assert target is not None
    assert target.joint_names == tuple(f"right_{name}" for name in REVO2_JOINT_NAMES)
    assert target.positions_rad[0] == 0.0
    assert target.positions_rad[1] == 1.5184
    limits = (1.0472, 1.5184, 1.4661, 1.4661, 1.4661, 1.4661)
    assert all(0.0 <= value <= limit for value, limit in zip(target.positions_rad, limits))


def test_bridge_resets_on_stale_frame():
    backend = _FakeRetargeter()
    bridge = WujiRevo2Bridge(backend, stale_after_sec=0.1)
    frame = WujiSkeletonFrame.from_joints(
        _skeleton(), "right", sequence=1, received_monotonic=30.0
    )

    assert bridge.process(frame, now=30.2) is None


def test_invalid_skeleton_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        WujiSkeletonFrame.from_joints(np.zeros((20, 3)), "left", sequence=1)
