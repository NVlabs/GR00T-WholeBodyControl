from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab import native124_selected_v2_causal_adaptation as causal
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    HOME_Q_HARDWARE,
    build_checkpoint21204_observation,
)
from gear_sonic.utils.g1_true23_native124_21204_mjlab_shadow import (
    _relative_rotation_6d,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DAD_DANCE = REPOSITORY_ROOT / causal.DAD_DANCE_RELATIVE_PATH


def _yaw_quaternion(angle: float) -> torch.Tensor:
    return torch.tensor(
        [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)],
        dtype=torch.float32,
    )


def test_dad_dance_motion_is_the_only_hash_locked_initial_motion() -> None:
    assert causal.validate_dad_dance_motion_file(DAD_DANCE) == DAD_DANCE.resolve()
    contract = causal.causal_adaptation_contract()
    assert contract["motion"]["sha256"] == causal.DAD_DANCE_SHA256
    assert contract["motion"]["sampling_mode"] == "start"
    assert contract["wrapper_action_clip"] is None
    assert contract["autoreset"]["sampled_action_substitution"] is False


def test_fixed_anchor_and_episode_window_validation_are_exact() -> None:
    assert causal.resolve_causal_reset_anchor_index(None) == 9
    assert causal.resolve_causal_reset_anchor_index(9) == 9
    assert causal.resolve_causal_reset_anchor_index(409) == 409
    assert causal.resolve_causal_reset_anchor_index(2088) == 2088
    assert (
        causal.causal_episode_last_q9(
            reset_anchor_index=9,
            episode_steps=2080,
        )
        == 2088
    )
    assert (
        causal.causal_episode_last_q9(
            reset_anchor_index=1609,
            episode_steps=480,
        )
        == 2088
    )

    for invalid in (True, 8, 2089, 9.0, "9"):
        with pytest.raises(ValueError, match="fixed_anchor_index"):
            causal.resolve_causal_reset_anchor_index(invalid)  # type: ignore[arg-type]
    for invalid_steps in (False, 0, -1, 1.0):
        with pytest.raises(ValueError, match="episode_steps"):
            causal.causal_episode_last_q9(
                reset_anchor_index=9,
                episode_steps=invalid_steps,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="overrun DadDance q10 proof"):
        causal.causal_episode_last_q9(
            reset_anchor_index=1609,
            episode_steps=481,
        )


def test_fixed_anchor_cfg_is_additive_and_caps_only_the_remaining_phase() -> None:
    if causal._MJLAB_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab configuration runtime unavailable")

    default_cfg = causal.make_native124_selected_v2_causal_adaptation_env_cfg(
        motion_file=DAD_DANCE,
        num_envs=1,
    )
    default_command = default_cfg.commands["motion"]
    assert default_command.fixed_anchor_index is None
    assert default_command.sampling_mode == "start"
    assert default_cfg.episode_length_s == 10.0

    fixed_cfg = causal.make_native124_selected_v2_causal_adaptation_env_cfg(
        motion_file=DAD_DANCE,
        num_envs=1,
        fixed_anchor_index=1609,
    )
    fixed_command = fixed_cfg.commands["motion"]
    assert fixed_command.fixed_anchor_index == 1609
    assert fixed_command.sampling_mode == "start"
    assert fixed_cfg.episode_length_s == 480 * causal.CONTROL_DT_S


def test_exact_command_fixed_sampling_precedes_start_mode_force() -> None:
    if causal._MJLAB_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab command runtime unavailable")

    command = object.__new__(causal.Native124SelectedV2CausalMotionCommand)
    command._causal_min_anchor = 9  # noqa: SLF001
    command._causal_max_anchor = 2088  # noqa: SLF001
    command._force_causal_start = True  # noqa: SLF001
    command._fixed_anchor_index = 409  # noqa: SLF001
    command.time_steps = torch.tensor([9, 9, 9], dtype=torch.long)
    command.metrics = {
        "sampling_entropy": torch.full((3,), -1.0),
        "sampling_top1_prob": torch.full((3,), -1.0),
        "sampling_top1_bin": torch.full((3,), -1.0),
    }

    command._uniform_sampling(torch.tensor([0, 2], dtype=torch.long))  # noqa: SLF001

    assert command.time_steps.tolist() == [409, 9, 409]
    assert command.reset_anchor_index == 409
    assert command.metrics["sampling_entropy"].tolist() == [0.0, 0.0, 0.0]
    assert command.metrics["sampling_top1_prob"].tolist() == [1.0, 1.0, 1.0]
    assert command.metrics["sampling_top1_bin"][0].item() == pytest.approx(400 / 2080)

    command._fixed_anchor_index = None  # noqa: SLF001
    command.time_steps.fill_(1234)
    command._uniform_sampling(torch.tensor([1], dtype=torch.long))  # noqa: SLF001
    assert command.time_steps.tolist() == [1234, 9, 1234]
    assert command.reset_anchor_index == 9


def test_forward_difference_uses_q10_proof_and_rejects_last_frame() -> None:
    motion = torch.arange(12 * 23, dtype=torch.float32).reshape(12, 23) / 100.0
    indices = torch.tensor([9, 10], dtype=torch.long)
    q9, q10, qd = causal.causal_forward_difference_at_indices(motion, indices)

    assert torch.equal(q9, motion[indices])
    assert torch.equal(q10, motion[indices + 1])
    assert torch.equal(qd, (motion[indices + 1] - motion[indices]) * 50.0)

    with pytest.raises(ValueError, match="q10 proof"):
        causal.causal_forward_difference_at_indices(
            motion,
            torch.tensor([11], dtype=torch.long),
        )


def test_torch_actor_builder_matches_frozen_numpy_adapter_order_and_rotation() -> None:
    rng = np.random.default_rng(20260809)
    batch_size = 3
    q9 = rng.normal(size=(batch_size, 23)).astype(np.float32)
    q10 = rng.normal(size=(batch_size, 23)).astype(np.float32)
    gyro = rng.normal(size=(batch_size, 3)).astype(np.float32)
    measured_q = rng.normal(size=(batch_size, 23)).astype(np.float32)
    measured_qd = rng.normal(size=(batch_size, 23)).astype(np.float32)
    previous = rng.normal(size=(batch_size, 23)).astype(np.float32)
    robot_torso = rng.normal(size=(batch_size, 4)).astype(np.float32)
    reference_torso = rng.normal(size=(batch_size, 4)).astype(np.float32)
    robot_torso /= np.linalg.norm(robot_torso, axis=1, keepdims=True)
    reference_torso /= np.linalg.norm(reference_torso, axis=1, keepdims=True)

    actual = causal.build_native124_selected_v2_causal_actor(
        reference_q9_hardware=torch.from_numpy(q9),
        reference_q10_hardware=torch.from_numpy(q10),
        robot_torso_q9_wxyz=torch.from_numpy(robot_torso),
        reference_torso_q9_wxyz=torch.from_numpy(reference_torso),
        base_angular_velocity_q10=torch.from_numpy(gyro),
        joint_pos_biased_q10_hardware=torch.from_numpy(measured_q),
        joint_velocity_q10_hardware=torch.from_numpy(measured_qd),
        previous_effective_selected_raw_hardware=torch.from_numpy(previous),
    ).numpy()

    expected = []
    for row in range(batch_size):
        rotation = _relative_rotation_6d(robot_torso[row], reference_torso[row])
        expected.append(
            build_checkpoint21204_observation(
                q_ref_hardware=q9[row],
                qd_ref_hardware=(q10[row] - q9[row]) * np.float32(50.0),
                torso_motion_anchor_ori_b=rotation,
                base_angular_velocity=gyro[row],
                q_measured_hardware=measured_q[row],
                qd_measured_hardware=measured_qd[row],
                previous_applied_raw_action_hardware=previous[row],
            )[0]
        )
    expected_array = np.stack(expected)

    np.testing.assert_allclose(actual, expected_array, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(actual[:, 55:78], measured_q - HOME_Q_HARDWARE, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(actual[:, 101:124], previous)


def test_reset_virtual_torso_is_torso_specific_not_pelvis_anchor() -> None:
    current = _yaw_quaternion(0.8).reshape(1, 4)
    torso_q9 = _yaw_quaternion(0.15).reshape(1, 4)
    torso_q10 = _yaw_quaternion(0.35).reshape(1, 4)
    pelvis_q9 = _yaw_quaternion(-0.4).reshape(1, 4)
    pelvis_q10 = _yaw_quaternion(-0.1).reshape(1, 4)

    torso_virtual = causal.virtual_q9_torso_quaternion(
        current,
        torso_q9,
        torso_q10,
    )
    pelvis_virtual = causal.virtual_q9_torso_quaternion(
        current,
        pelvis_q9,
        pelvis_q10,
    )

    assert not torch.allclose(torso_virtual, pelvis_virtual, atol=1.0e-6, rtol=0.0)
    rotation = causal.relative_torso_rotation_6d(torso_virtual, torso_q9)
    assert rotation.shape == (1, 6)
    assert torch.isfinite(rotation).all()


def test_autoreset_rows_use_virtual_torso_and_zero_history_without_action_substitution() -> None:
    done = torch.tensor([False, True, False], dtype=torch.bool)
    pre_step = torch.stack((_yaw_quaternion(0.1), _yaw_quaternion(0.2), _yaw_quaternion(0.3)))
    virtual = torch.stack((_yaw_quaternion(0.4), _yaw_quaternion(0.5), _yaw_quaternion(0.6)))
    effective = torch.arange(3 * 23, dtype=torch.float32).reshape(3, 23)
    sampled_copy = effective.clone()

    torso, previous = causal.merge_causal_history_after_step(
        done_mask=done,
        pre_step_robot_torso_wxyz=pre_step,
        reset_virtual_robot_torso_q9_wxyz=virtual,
        effective_selected_raw_action_hardware=effective,
    )

    assert torch.equal(torso[~done], pre_step[~done])
    assert torch.equal(torso[done], virtual[done])
    assert torch.equal(previous[~done], effective[~done])
    assert torch.equal(previous[done], torch.zeros((1, 23), dtype=torch.float32))
    assert torch.equal(effective, sampled_copy)


def test_real_mjlab_wrapper_reset_and_two_causal_steps() -> None:
    """WSL/CUDA proof; never runs more than two simulator transitions."""

    if causal._MJLAB_IMPORT_ERROR is not None or causal._WRAPPER_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab RSL runtime unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA MJLab runtime unavailable")

    from mjlab.envs import ManagerBasedRlEnv

    cfg = causal.make_native124_selected_v2_causal_adaptation_env_cfg(
        motion_file=DAD_DANCE,
        num_envs=1,
        play=True,
    )
    cfg.seed = 20260809
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        wrapped = causal.Native124SelectedV2CausalAdaptationWrapper(
            env,
            clip_actions=None,
        )
        reset_observations, _ = wrapped.reset()
        prime = causal.prime_native124_selected_v2_causal_adaptation_environment(wrapped)
        observations = wrapped.get_observations()
        assert reset_observations["actor"].shape == (1, 124)
        assert observations["actor"].shape == (1, 124)
        assert observations["critic"].shape == (1, causal.SONIC_TRUE23_CRITIC_DIM)
        assert observations["policy"].shape == (1, causal.SONIC_TRUE23_POLICY_DIM)
        assert prime["action_substitution"] is False

        initial = wrapped.diagnostics
        assert initial.reset_virtual_torso_mask.tolist() == [True]
        assert torch.equal(
            initial.previous_effective_selected_raw_hardware,
            torch.zeros((1, 23), dtype=torch.float32, device="cuda:0"),
        )
        command = env.command_manager.get_term("motion")
        body_names = tuple(command.cfg.body_names)
        torso_index = body_names.index(causal.TORSO_BODY_NAME)
        pelvis_index = body_names.index(causal.PELVIS_BODY_NAME)
        assert torso_index != command.motion_anchor_body_index
        torso_series = command.motion.body_quat_w[:, torso_index]
        pelvis_series = command.motion.body_quat_w[:, pelvis_index]
        same_orientation = torch.abs(torch.sum(torso_series * pelvis_series, dim=-1))
        assert torch.any(same_orientation < 1.0 - 1.0e-6)

        zero = torch.zeros((1, 23), dtype=torch.float32, device="cuda:0")
        for _ in range(2):
            q9_before = command.time_steps.clone()
            pre_step_torso = wrapped._current_robot_torso().clone()  # noqa: SLF001
            step_observations, _, dones, _ = wrapped.step(zero)
            assert dones.tolist() == [0]
            diagnostics = wrapped.diagnostics
            assert torch.equal(diagnostics.q9_indices, q9_before + 1)
            assert diagnostics.q10_proof_indices.tolist() == [int(q9_before.item()) + 2]
            assert diagnostics.reset_virtual_torso_mask.tolist() == [False]
            assert torch.equal(
                diagnostics.buffered_robot_torso_q9_wxyz,
                pre_step_torso,
            )
            action_term = env.action_manager.get_term("joint_pos")
            assert torch.equal(action_term.raw_action, zero)
            assert torch.allclose(
                diagnostics.previous_effective_selected_raw_hardware,
                action_term.effective_selected_raw_action_hardware,
                atol=0.0,
                rtol=0.0,
            )
            assert torch.equal(
                step_observations["actor"][:, 101:124],
                diagnostics.previous_effective_selected_raw_hardware,
            )
            assert torch.equal(step_observations["critic"], env.obs_buf["critic"])
            expected_qd = (
                command.motion.joint_pos[command.time_steps + 1] - command.motion.joint_pos[command.time_steps]
            ) * 50.0
            assert torch.equal(command.joint_vel, expected_qd)
            assert torch.equal(command.command[:, 23:46], expected_qd)
    finally:
        env.close()


def test_real_mjlab_fixed_phase_reset_writes_q10_before_one_causal_step() -> None:
    """Bounded CUDA proof: fixed q9=409 selects phase-specific robot q10=410."""

    if causal._MJLAB_IMPORT_ERROR is not None or causal._WRAPPER_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab RSL runtime unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA MJLab runtime unavailable")

    from mjlab.envs import ManagerBasedRlEnv

    from gear_sonic.envs.mjlab.native124_selected_v2_ankle_task import (
        make_native124_selected_v2_ankle_task_env_cfg,
    )

    cfg = make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=str(DAD_DANCE),
        num_envs=1,
        play=True,
        fixed_anchor_index=409,
    )
    cfg.seed = 20260809
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        wrapped = causal.Native124SelectedV2CausalAdaptationWrapper(
            env,
            clip_actions=None,
        )
        wrapped.reset()
        causal.prime_native124_selected_v2_causal_adaptation_environment(wrapped)

        command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        proof = 410
        assert command.reset_anchor_index == 409
        assert command.time_steps.tolist() == [409]
        assert int(env.common_step_counter) == 0
        assert int(env._sim_step_counter) == 0  # noqa: SLF001
        torch.testing.assert_close(
            robot.data.joint_pos[0],
            command.motion.joint_pos[proof],
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            robot.data.joint_vel[0],
            command.motion.joint_vel[proof],
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            robot.data.root_link_pos_w[0],
            command.motion.body_pos_w[proof, 0] + env.scene.env_origins[0],
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            robot.data.root_link_quat_w[0],
            command.motion.body_quat_w[proof, 0],
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            robot.data.root_link_lin_vel_w[0],
            command.motion.body_lin_vel_w[proof, 0],
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            robot.data.root_link_ang_vel_w[0],
            command.motion.body_ang_vel_w[proof, 0],
            rtol=0.0,
            atol=1.0e-6,
        )

        reset_diagnostics = wrapped.diagnostics
        assert reset_diagnostics.q9_indices.tolist() == [409]
        assert reset_diagnostics.q10_proof_indices.tolist() == [410]
        assert reset_diagnostics.reset_virtual_torso_mask.tolist() == [True]
        assert torch.count_nonzero(reset_diagnostics.previous_effective_selected_raw_hardware).item() == 0

        zero = torch.zeros((1, 23), dtype=torch.float32, device="cuda:0")
        _, _, dones, _ = wrapped.step(zero)
        assert dones.tolist() == [0]
        assert wrapped.diagnostics.q9_indices.tolist() == [410]
        assert wrapped.diagnostics.q10_proof_indices.tolist() == [411]
        assert wrapped.diagnostics.reset_virtual_torso_mask.tolist() == [False]
    finally:
        env.close()
