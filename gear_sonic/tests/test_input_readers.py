import numpy as np

from gear_sonic.utils.teleop.input_readers import _body_data_to_24x7


def test_body_data_dict_converts_to_padded_float32_array():
    body_data = {
        "joint_positions": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "joint_orientations": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
    }

    body_poses = _body_data_to_24x7(body_data)

    assert body_poses is not None
    assert body_poses.shape == (24, 7)
    assert body_poses.dtype == np.float32
    np.testing.assert_allclose(
        body_poses[:2],
        np.array(
            [
                [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                [0.4, 0.5, 0.6, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(body_poses[2:], np.zeros((22, 7), dtype=np.float32))


def test_body_data_dict_with_empty_joint_arrays_returns_none():
    body_data = {"joint_positions": [], "joint_orientations": []}

    assert _body_data_to_24x7(body_data) is None
