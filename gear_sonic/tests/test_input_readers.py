from types import SimpleNamespace

import numpy as np

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
