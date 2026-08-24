# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils import g1_true23_step1c_onnx_diagnostic as step1c

EXPECTED_MODES = frozenset(
    (
        "static_frame0",
        "static_frame0_zero_velocity",
        "full_terminal_hold",
        "full_terminal_hold_zero_velocity",
    )
)


def _env(path: Path, *, bad: bool = False) -> None:
    path.write_text(
        """joint_ids_map: [0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,22,23,24,25,26]\ndefault_joint_pos: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]\nactions:\n  JointPositionAction:\n    scale: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]\n    offset: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]\nobservations:\n  motion_command: {scale: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], history_length: 1}\n  motion_anchor_ori_b: {scale: [1,1,1,1,1,1], history_length: 1}\n  base_ang_vel: {scale: [1,1,1], history_length: 1}\n  joint_pos_rel: {scale: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], history_length: 1}\n  joint_vel_rel: {scale: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], history_length: 1}\n  last_action: {scale: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], history_length: 1}\n""".replace(
            "joint_ids_map", "wrong_ids" if bad else "joint_ids_map"
        ),
        encoding="utf-8",
    )


def test_env_and_observation_exact_shape(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "env.yaml"
    _env(path)
    env = step1c.load_unitree_env_params(path)
    qref = np.arange(23, dtype=np.float32)
    q_hardware = np.arange(100, 123, dtype=np.float32)
    qd_hardware = np.arange(200, 223, dtype=np.float32)
    obs = step1c.build_observation(
        qref_native=qref,
        qdref_native=np.arange(23),
        torso_relative_quaternion_wxyz=(1, 0, 0, 0),
        base_angular_velocity=(1, 2, 3),
        q_hardware=q_hardware,
        qd_hardware=qd_hardware,
        previous_raw_action_native=np.zeros(23),
        env=env,
    )
    assert obs.shape == (1, 124) and obs.dtype == np.float32
    np.testing.assert_array_equal(obs[0, :23], qref)
    np.testing.assert_array_equal(obs[0, 55:78], q_hardware)
    np.testing.assert_array_equal(obs[0, 78:101], qd_hardware)


def test_env_fails_closed_on_topology(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "env.yaml"
    _env(path, bad=True)
    with pytest.raises(ValueError, match="joint_ids_map"):
        step1c.load_unitree_env_params(path)


def _clip() -> SimpleNamespace:
    values = np.arange(3 * 23, dtype=np.float64).reshape(3, 23)
    return SimpleNamespace(
        frame_count=3,
        target_joint_pos_hardware=values + 100.0,
        target_joint_vel_hardware=values + 200.0,
    )


def test_modes_include_existing_and_zero_velocity_holds() -> None:
    assert step1c.MODES == EXPECTED_MODES


@pytest.mark.parametrize("index", (0, 2, 3, 8))
def test_static_frame0_zero_velocity_always_holds_frame_zero(index: int) -> None:
    clip = _clip()

    qref, qdref = step1c._reference(clip, index, "static_frame0_zero_velocity", None)

    np.testing.assert_array_equal(qref, clip.target_joint_pos_hardware[0].astype(np.float32))
    np.testing.assert_array_equal(qdref, np.zeros(23, dtype=np.float32))


def test_full_terminal_hold_zero_velocity_changes_after_final_source_frame() -> None:
    clip = _clip()

    final_qref, final_qdref = step1c._reference(
        clip, clip.frame_count - 1, "full_terminal_hold_zero_velocity", None
    )
    held_qref, held_qdref = step1c._reference(clip, clip.frame_count, "full_terminal_hold_zero_velocity", None)

    expected_qref = clip.target_joint_pos_hardware[-1].astype(np.float32)
    np.testing.assert_array_equal(final_qref, expected_qref)
    np.testing.assert_array_equal(final_qdref, clip.target_joint_vel_hardware[-1].astype(np.float32))
    np.testing.assert_array_equal(held_qref, expected_qref)
    np.testing.assert_array_equal(held_qdref, np.zeros(23, dtype=np.float32))


@pytest.mark.parametrize(
    ("mode", "index", "expected_frame"),
    (
        ("static_frame0", 8, 0),
        ("full_terminal_hold", 1, 1),
        ("full_terminal_hold", 3, 2),
    ),
)
def test_existing_modes_keep_reference_velocity(mode: str, index: int, expected_frame: int) -> None:
    clip = _clip()

    qref, qdref = step1c._reference(clip, index, mode, None)

    np.testing.assert_array_equal(qref, clip.target_joint_pos_hardware[expected_frame].astype(np.float32))
    np.testing.assert_array_equal(qdref, clip.target_joint_vel_hardware[expected_frame].astype(np.float32))
