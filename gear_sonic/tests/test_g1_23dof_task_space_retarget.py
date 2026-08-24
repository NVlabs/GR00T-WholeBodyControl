from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from gear_sonic.utils import g1_23dof_task_space_retarget as retarget_module  # noqa: E402
from gear_sonic.utils.g1_23dof_contract import (  # noqa: E402
    HARDWARE_23_JOINT_NAMES,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (  # noqa: E402
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_INNER_LOWER_HARDWARE,
    SAFE_TARGET_INNER_UPPER_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_23dof_task_space_retarget import (  # noqa: E402
    DEFAULT_TASKS,
    RetargetConfig,
    TaskSpec,
    build_mjlab_motion_arrays,
    build_residual_supervision,
    infer_foot_contacts,
    load_models,
    retarget_trajectory,
    trajectory_derivatives,
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


def _synthetic_source(
    source_model: mujoco.MjModel,
    frame_count: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_names = _source_joint_names(source_model)
    target_default = dict(
        zip(HARDWARE_23_JOINT_NAMES, SAFE_TARGET_DEFAULT_Q_HARDWARE, strict=True)
    )
    pose = np.asarray([target_default.get(name, 0.0) for name in source_names])
    pose[source_names.index("waist_roll_joint")] = 0.20
    pose[source_names.index("waist_pitch_joint")] = -0.15
    pose[source_names.index("left_wrist_pitch_joint")] = 0.30
    pose[source_names.index("right_wrist_yaw_joint")] = -0.25
    joints = np.repeat(pose[None, :], frame_count, axis=0)
    root_pos = np.repeat(np.asarray([[0.0, 0.0, 0.8]]), frame_count, axis=0)
    root_quat = np.repeat(np.asarray([[1.0, 0.0, 0.0, 0.0]]), frame_count, axis=0)
    return root_pos, root_quat, joints


@pytest.fixture(scope="module")
def retargeted():
    source_model, target_model = load_models(SOURCE_MODEL, TARGET_MODEL)
    root_pos, root_quat, joints = _synthetic_source(source_model)
    config = RetargetConfig(max_iterations=12)
    result = retarget_trajectory(
        source_model=source_model,
        target_model=target_model,
        root_pos_w=root_pos,
        root_quat_wxyz=root_quat,
        source_joint_pos_hardware=joints,
        fps=50.0,
        config=config,
        contact_flags=np.ones((len(joints), 2), dtype=bool),
    )
    return source_model, target_model, result


def test_task_points_and_priority_contract_are_exact() -> None:
    by_name = {task.name: task for task in DEFAULT_TASKS}
    assert by_name["left_hand"].source_body == "left_wrist_yaw_link"
    assert by_name["left_hand"].target_body == "left_wrist_roll_rubber_hand"
    assert by_name["left_hand"].source_point == (0.18, -0.025, 0.0)
    assert by_name["left_hand"].target_point == (0.18, -0.025, 0.0)
    assert by_name["right_hand"].source_point == (0.18, 0.025, 0.0)
    assert by_name["head_proxy"].source_point == (0.0, 0.0, 0.35)
    assert by_name["head_proxy"].target_point == (0.0, 0.0, 0.35)
    assert by_name["left_foot"].priority < by_name["whole_robot_com"].priority
    assert by_name["whole_robot_com"].priority < by_name["left_hand"].priority


def test_hierarchical_retarget_improves_expert_without_breaking_feet_or_limits(
    retargeted,
) -> None:
    _source_model, _target_model, result = retargeted
    before = result.diagnostics["weighted_task_error_before"]
    after = result.diagnostics["weighted_task_error_after"]
    assert np.all(after <= before + 1.0e-7)
    assert np.any(after < before - 1.0e-5)
    for foot_name in ("left_foot", "right_foot"):
        assert np.max(
            result.diagnostics[f"task_{foot_name}_position_error_after_m"]
        ) <= result.config.valid_max_foot_position_error_m
        assert np.all(
            result.diagnostics[
                f"task_{foot_name}_orientation_error_after_rad"
            ]
            <= result.diagnostics[
                f"task_{foot_name}_orientation_error_before_rad"
            ]
            + result.config.valid_max_foot_orientation_regression_rad
        )
    assert np.mean(result.diagnostics["priority_3_error_after"]) < np.mean(
        result.diagnostics["priority_3_error_before"]
    )

    lower = np.asarray(SAFE_TARGET_INNER_LOWER_HARDWARE)
    upper = np.asarray(SAFE_TARGET_INNER_UPPER_HARDWARE)
    assert np.all(result.joint_pos_hardware > lower)
    assert np.all(result.joint_pos_hardware < upper)
    assert np.max(result.diagnostics["trajectory_velocity_abs_max"]) <= 8.0 + 1.0e-8
    assert (
        np.max(result.diagnostics["trajectory_acceleration_abs_max"])
        <= 80.0 + 1.0e-6
    )
    assert not np.any(result.diagnostics["constraint_relaxation_count"])


def test_expert_action_inverts_exact_safe_target_transform(retargeted) -> None:
    _source_model, _target_model, result = retargeted
    _, reconstructed = safe_target_transform_numpy(
        result.action_target_native.astype(np.float32)
    )
    np.testing.assert_allclose(
        reconstructed,
        result.joint_pos_hardware,
        rtol=0.0,
        atol=1.0e-6,
    )
    assert np.max(np.abs(result.action_target_native)) < result.config.native_action_clip
    assert result.summary()["hard_position_limit_violation_count"] == 0


def test_solver_envelope_is_reachable_after_native_action_clipping() -> None:
    source_model, target_model = load_models(SOURCE_MODEL, TARGET_MODEL)
    root_pos, root_quat, joints = _synthetic_source(source_model, frame_count=3)
    source_names = _source_joint_names(source_model)
    joints[:, source_names.index("left_wrist_roll_joint")] = 1.9
    joints[:, source_names.index("right_wrist_roll_joint")] = -1.9
    config = RetargetConfig(max_iterations=2, native_action_clip=10.0)
    result = retarget_trajectory(
        source_model=source_model,
        target_model=target_model,
        root_pos_w=root_pos,
        root_quat_wxyz=root_quat,
        source_joint_pos_hardware=joints,
        fps=50.0,
        config=config,
        contact_flags=np.ones((len(joints), 2), dtype=bool),
    )

    assert np.max(np.abs(result.action_target_native)) < config.native_action_clip
    assert np.max(np.abs(result.direct_action_native)) < config.native_action_clip
    assert result.summary()["native_action_abs_gt_ten_fraction"] == 0.0


def test_standard_motion_and_residual_dataset_contracts(retargeted) -> None:
    _source_model, target_model, result = retargeted
    motion = build_mjlab_motion_arrays(target_model, result)
    frame_count = len(result.joint_pos_hardware)
    assert motion["joint_pos"].shape == (frame_count, 23)
    assert motion["body_pos_w"].shape == (frame_count, 24, 3)
    assert motion["body_quat_w"].shape == (frame_count, 24, 4)
    assert all(np.all(np.isfinite(value)) for value in motion.values())
    stored_velocity, stored_acceleration = trajectory_derivatives(
        motion["joint_pos"].astype(np.float64), 1.0 / result.fps
    )
    np.testing.assert_allclose(
        motion["joint_vel"], stored_velocity, rtol=0.0, atol=1.0e-6
    )
    expert_arrays = result.adaptation_arrays()
    np.testing.assert_allclose(
        expert_arrays["joint_vel_hardware"],
        stored_velocity,
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        expert_arrays["joint_acc_hardware"],
        stored_acceleration,
        rtol=0.0,
        atol=1.0e-4,
    )
    np.testing.assert_array_equal(
        expert_arrays["trajectory_velocity_abs_max"],
        np.max(np.abs(stored_velocity), axis=1).astype(np.float32),
    )
    np.testing.assert_array_equal(
        expert_arrays["trajectory_acceleration_abs_max"],
        np.max(np.abs(stored_acceleration), axis=1).astype(np.float32),
    )
    stored_root_velocity, stored_root_acceleration = trajectory_derivatives(
        expert_arrays["root_offset_w"].astype(np.float64), 1.0 / result.fps
    )
    np.testing.assert_array_equal(
        expert_arrays["root_offset_velocity_abs_max"],
        np.max(np.abs(stored_root_velocity), axis=1).astype(np.float32),
    )
    np.testing.assert_array_equal(
        expert_arrays["root_offset_acceleration_abs_max"],
        np.max(np.abs(stored_root_acceleration), axis=1).astype(np.float32),
    )
    assert expert_arrays["trajectory_derivative_convention"].tolist() == [
        "free_initial_velocity_equal_first_interval"
    ]
    assert result.summary()["constraints"]["derivative_convention"] == (
        "free_initial_velocity_equal_first_interval"
    )

    decoder_input = np.zeros((frame_count, 994), dtype=np.float32)
    base_action = np.full((frame_count, 23), 0.125, dtype=np.float32)
    dataset = build_residual_supervision(
        result=result,
        decoder_input=decoder_input,
        base_action_native=base_action,
    )
    np.testing.assert_allclose(
        dataset["residual_action_native"],
        dataset["expert_action_native"] - base_action,
    )
    assert dataset["expert_valid"].shape == (frame_count,)
    assert dataset["expert_valid"].dtype == np.bool_
    with pytest.raises(ValueError, match="994"):
        build_residual_supervision(
            result=result,
            decoder_input=decoder_input[:, :-1],
            base_action_native=base_action,
        )


def test_contact_inference_is_deterministic_and_shape_checked() -> None:
    feet = np.zeros((5, 2, 3), dtype=np.float64)
    feet[:, :, 2] = 0.03
    first = infer_foot_contacts(feet, fps=50.0)
    assert np.array_equal(first, infer_foot_contacts(feet, fps=50.0))
    assert np.all(first)
    with pytest.raises(ValueError, match="shape"):
        infer_foot_contacts(np.zeros((5, 3)), fps=50.0)


def test_hierarchy_accepts_an_intentionally_empty_priority_tier() -> None:
    source_model, target_model = load_models(SOURCE_MODEL, TARGET_MODEL)
    root_pos, root_quat, joints = _synthetic_source(source_model, frame_count=3)
    tasks = tuple(
        replace(task, priority=2)
        if task.name in {"left_hand", "right_hand"}
        else task
        for task in DEFAULT_TASKS
    )
    result = retarget_trajectory(
        source_model=source_model,
        target_model=target_model,
        root_pos_w=root_pos,
        root_quat_wxyz=root_quat,
        source_joint_pos_hardware=joints,
        fps=50.0,
        tasks=tasks,
        config=RetargetConfig(max_iterations=2),
        contact_flags=np.ones((len(joints), 2), dtype=bool),
    )

    assert np.all(np.isfinite(result.joint_pos_hardware))


def test_explicit_solver_seed_preserves_inactive_joints_during_fallback() -> None:
    source_model, target_model = load_models(SOURCE_MODEL, TARGET_MODEL)
    root_pos, root_quat, source_joints = _synthetic_source(
        source_model, frame_count=2
    )
    source_layout = retarget_module._model_layout(source_model)
    config = RetargetConfig(
        max_iterations=1,
        max_iteration_step_rad=0.01,
        protected_priority_tiers=1,
    )
    target_layout = retarget_module._safe_target_layout(
        retarget_module._model_layout(target_model),
        config.safe_limit_guard_rad,
        config.native_action_clip,
    )
    task = TaskSpec(
        name="torso_orientation_only",
        kind="body",
        source_body="torso_link",
        target_body="torso_link",
        orientation_weight=1.0,
    )
    source_data = mujoco.MjData(source_model)
    retarget_module._set_configuration(
        source_model,
        source_data,
        source_layout,
        root_pos[0],
        root_quat[0],
        source_joints[0],
    )
    targets = retarget_module._task_targets(source_model, source_data, (task,))
    source_index = {
        name: index for index, name in enumerate(source_layout.joint_names)
    }
    direct = np.asarray(
        [source_joints[0, source_index[name]] for name in target_layout.joint_names]
    )
    direct = np.clip(direct, target_layout.lower, target_layout.upper)

    waist_index = target_layout.joint_names.index("waist_yaw_joint")
    frozen_hip_index = target_layout.joint_names.index("left_hip_pitch_joint")
    seed = direct.copy()
    seed[waist_index] = np.clip(
        direct[waist_index] + 0.5,
        target_layout.lower[waist_index],
        target_layout.upper[waist_index],
    )
    seed[frozen_hip_index] = np.clip(
        direct[frozen_hip_index] + 0.05,
        target_layout.lower[frozen_hip_index],
        target_layout.upper[frozen_hip_index],
    )
    assert seed[frozen_hip_index] != direct[frozen_hip_index]

    solved, diagnostics = retarget_module._solve_frame(
        target_model,
        mujoco.MjData(target_model),
        target_layout,
        (task,),
        targets,
        root_pos[0],
        root_quat[0],
        direct,
        None,
        (False, False),
        target_layout.lower,
        target_layout.upper,
        config,
        seed=seed,
        fallback_seed=direct,
        active_joint_indices=np.asarray([waist_index], dtype=np.int64),
    )

    assert diagnostics["solver_used_feasible_seed_fallback"] == 1.0
    assert solved[frozen_hip_index] == seed[frozen_hip_index]
    assert solved[waist_index] == direct[waist_index]
