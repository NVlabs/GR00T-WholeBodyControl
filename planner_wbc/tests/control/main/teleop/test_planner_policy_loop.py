from pathlib import Path

import numpy as np
import pytest

from planner_wbc.control.main.teleop.run_planner_policy_loop import (
    REQUIRED_UPPER_BODY_DATASET_JOINTS,
    load_trajectory_dataset,
    load_trajectory_fps,
)


def test_load_named_trajectory_format(tmp_path: Path):
    path = tmp_path / "named.npz"
    expected_qpos = np.arange(34, dtype=np.float32).reshape(2, 17)
    np.savez(
        path,
        qpos=expected_qpos,
        joint_names=np.asarray(REQUIRED_UPPER_BODY_DATASET_JOINTS),
    )

    qpos, joint_names = load_trajectory_dataset(path)

    np.testing.assert_array_equal(qpos, expected_qpos)
    assert joint_names == REQUIRED_UPPER_BODY_DATASET_JOINTS
    assert load_trajectory_fps(path) is None


def test_load_retargeted_single_demo_format(tmp_path: Path):
    path = tmp_path / "retargeted.npz"
    expected_qpos = np.arange(34, dtype=np.float64).reshape(2, 17)
    np.savez(path, qpos=expected_qpos[np.newaxis].astype(object), fps=np.asarray([30.0]))

    qpos, joint_names = load_trajectory_dataset(path)

    np.testing.assert_array_equal(qpos, expected_qpos.astype(np.float32))
    assert qpos.shape == (2, 17)
    assert qpos.dtype == np.float32
    assert joint_names == REQUIRED_UPPER_BODY_DATASET_JOINTS
    assert load_trajectory_fps(path) == 30.0


def test_missing_joint_names_requires_known_width(tmp_path: Path):
    path = tmp_path / "unknown.npz"
    np.savez(path, qpos=np.zeros((1, 2, 16), dtype=object))

    with pytest.raises(KeyError, match="known 17-joint retargeted format"):
        load_trajectory_dataset(path)


def test_multiple_demonstrations_are_rejected(tmp_path: Path):
    path = tmp_path / "multiple.npz"
    np.savez(path, qpos=np.zeros((2, 2, 17), dtype=object))

    with pytest.raises(ValueError, match="multiple demonstrations"):
        load_trajectory_dataset(path)
