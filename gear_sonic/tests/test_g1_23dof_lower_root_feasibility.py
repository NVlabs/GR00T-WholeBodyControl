from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from gear_sonic.utils.g1_23dof_contract import (  # noqa: E402
    HARDWARE_23_JOINT_NAMES,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (  # noqa: E402
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
)
from gear_sonic.utils.g1_23dof_task_space_retarget import (  # noqa: E402
    RetargetConfig,
    load_models,
    retarget_trajectory,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODEL = REPOSITORY_ROOT / "gear_sonic/data/robots/g1/g1_29dof.xml"
TARGET_MODEL = (
    REPOSITORY_ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
)


def _source_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(1, model.njnt)
    )


def _safe_envelope_knee_adversary(
    source_model: mujoco.MjModel,
    frame_count: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_names = _source_joint_names(source_model)
    retained_default = dict(
        zip(
            HARDWARE_23_JOINT_NAMES,
            SAFE_TARGET_DEFAULT_Q_HARDWARE,
            strict=True,
        )
    )
    pose = np.asarray([retained_default.get(name, 0.0) for name in source_names])
    # Legal on source body, but below exact true23 clip-10 reachable floor
    # (0.07310578 rad). Fixed-root clipping moves both feet by about 39 mm.
    pose[source_names.index("left_knee_joint")] = -0.05
    pose[source_names.index("right_knee_joint")] = -0.05
    joints = np.repeat(pose[None, :], frame_count, axis=0)
    root_pos = np.repeat(np.asarray([[0.0, 0.0, 0.8]]), frame_count, axis=0)
    root_quat = np.repeat(
        np.asarray([[1.0, 0.0, 0.0, 0.0]]), frame_count, axis=0
    )
    return root_pos, root_quat, joints


def test_staged_lower_root_solver_recovers_safe_envelope_foot_drift() -> None:
    source_model, target_model = load_models(SOURCE_MODEL, TARGET_MODEL)
    source_root, root_quat, source_joints = _safe_envelope_knee_adversary(
        source_model
    )
    config = RetargetConfig(max_iterations=8)

    result = retarget_trajectory(
        source_model=source_model,
        target_model=target_model,
        root_pos_w=source_root,
        root_quat_wxyz=root_quat,
        source_joint_pos_hardware=source_joints,
        fps=50.0,
        config=config,
        contact_flags=np.ones((len(source_joints), 2), dtype=bool),
    )

    knee_indices = np.asarray(
        [
            HARDWARE_23_JOINT_NAMES.index("left_knee_joint"),
            HARDWARE_23_JOINT_NAMES.index("right_knee_joint"),
        ]
    )
    assert np.all(result.direct_joint_pos_hardware[:, knee_indices] > -0.05)

    for side in ("left", "right"):
        before = result.diagnostics[
            f"task_{side}_foot_position_error_before_m"
        ]
        after = result.diagnostics[f"task_{side}_foot_position_error_after_m"]
        assert np.min(before) > config.valid_max_foot_position_error_m
        assert np.max(after) <= config.valid_max_foot_position_error_m

    np.testing.assert_allclose(result.source_root_pos_w, source_root)
    np.testing.assert_allclose(
        result.root_pos_w,
        result.source_root_pos_w + result.root_offset_w,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.max(np.linalg.norm(result.root_offset_w, axis=1)) > 0.005
    assert np.max(np.abs(result.root_offset_w)) <= config.max_root_offset_m

    dt = 1.0 / result.fps
    joint_velocity = np.vstack(
        (
            np.zeros((1, len(HARDWARE_23_JOINT_NAMES))),
            np.diff(result.joint_pos_hardware, axis=0) / dt,
        )
    )
    joint_acceleration = np.vstack(
        (
            np.zeros((1, len(HARDWARE_23_JOINT_NAMES))),
            np.diff(joint_velocity, axis=0) / dt,
        )
    )
    assert np.max(np.abs(joint_velocity)) <= config.max_velocity_rad_s + 1.0e-8
    assert (
        np.max(np.abs(joint_acceleration))
        <= config.max_acceleration_rad_s2 + 1.0e-6
    )
    root_velocity = np.vstack(
        (np.zeros((1, 3)), np.diff(result.root_offset_w, axis=0) / dt)
    )
    root_acceleration = np.vstack(
        (np.zeros((1, 3)), np.diff(root_velocity, axis=0) / dt)
    )
    assert (
        np.max(np.abs(root_velocity))
        <= config.max_root_offset_velocity_m_s + 1.0e-8
    )
    assert (
        np.max(np.abs(root_acceleration))
        <= config.max_root_offset_acceleration_m_s2 + 1.0e-6
    )
    assert not np.any(result.diagnostics["constraint_relaxation_count"])
