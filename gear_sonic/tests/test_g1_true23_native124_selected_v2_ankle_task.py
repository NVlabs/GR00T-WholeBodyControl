from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.envs.mjlab import (
    native124_selected_v2_ankle_task as task,
    native124_selected_v2_causal_adaptation as causal,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DAD_DANCE = REPOSITORY_ROOT / causal.DAD_DANCE_RELATIVE_PATH


class _FakeActionManager:
    def __init__(self, term: object) -> None:
        self.term = term

    def get_term(self, name: str) -> object:
        assert name == "joint_pos"
        return self.term


def test_nominal_episode_horizon_preserves_default_and_late_q10_proof() -> None:
    assert task.nominal_ankle_episode_steps_for_anchor(None) == 500
    assert task.nominal_ankle_episode_steps_for_anchor(9) == 500
    assert task.nominal_ankle_episode_steps_for_anchor(1589) == 500
    assert task.nominal_ankle_episode_steps_for_anchor(1590) == 499
    assert task.nominal_ankle_episode_steps_for_anchor(1609) == 480
    assert task.nominal_ankle_episode_steps_for_anchor(2088) == 1

    with pytest.raises(ValueError, match="fixed_anchor_index"):
        task.nominal_ankle_episode_steps_for_anchor(2089)


def test_static_contract_is_four_ankle_task_space_not_joint_imitation() -> None:
    contract = task.native124_selected_v2_ankle_task_contract()

    assert contract["trainable_output_rows"]["hardware_indices"] == [4, 5, 10, 11]
    assert contract["trainable_output_rows"]["native_indices"] == [15, 19, 16, 20]
    assert contract["causal"]["reference"] == "q9"
    assert contract["causal"]["forward_difference_proof"] == "q10"
    assert contract["causal"]["privileged_critic_dim"] == 256
    assert contract["motion"]["contact_labels_present"] is False
    assert "action_target_reference_l2" in contract["forbidden_objectives"]
    assert contract["runner_action_clip_required"] is None
    assert contract["hardware_authorized"] is False
    assert contract["deployment_ready"] is False


def test_v2_projection_penalty_uses_only_four_hardware_ankles() -> None:
    candidate = torch.zeros((2, 23), dtype=torch.float32)
    final = candidate.clone()
    candidate[0, 4] = 0.5
    candidate[0, 10] = -0.25
    candidate[1, 3] = 100.0  # frozen knee must not affect ankle objective
    term = SimpleNamespace(
        candidate_target_hardware=candidate,
        final_target_hardware=final,
        raw_clip_mask_native=torch.zeros((2, 23), dtype=torch.bool),
    )
    env = SimpleNamespace(num_envs=2, action_manager=_FakeActionManager(term))

    value = task.v2_ankle_projection_l2(env)

    assert value.shape == (2,)
    assert value[0] > 0.0
    assert value[1] == 0.0


def test_v2_raw_clip_termination_is_whole_action_fail_closed() -> None:
    mask = torch.zeros((3, 23), dtype=torch.bool)
    mask[1, 19] = True
    mask[2, 3] = True
    term = SimpleNamespace(
        candidate_target_hardware=torch.zeros((3, 23), dtype=torch.float32),
        final_target_hardware=torch.zeros((3, 23), dtype=torch.float32),
        raw_clip_mask_native=mask,
    )
    env = SimpleNamespace(num_envs=3, action_manager=_FakeActionManager(term))

    assert task.v2_raw_clip_violation(env).tolist() == [False, True, True]


def test_v2_raw_clip_termination_rejects_non_boolean_mask() -> None:
    term = SimpleNamespace(
        candidate_target_hardware=torch.zeros((1, 23), dtype=torch.float32),
        final_target_hardware=torch.zeros((1, 23), dtype=torch.float32),
        raw_clip_mask_native=torch.zeros((1, 23), dtype=torch.float32),
    )
    env = SimpleNamespace(num_envs=1, action_manager=_FakeActionManager(term))

    with pytest.raises(ValueError, match="bool"):
        task.v2_raw_clip_violation(env)


def test_feet_slip_is_actual_contact_only_and_always_on() -> None:
    found = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    velocity = torch.tensor(
        [
            [[3.0, 4.0, 9.0], [100.0, 100.0, 0.0]],
            [[1.0, 2.0, 0.0], [2.0, 3.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    robot = SimpleNamespace(data=SimpleNamespace(site_lin_vel_w=velocity))
    sensor = SimpleNamespace(data=SimpleNamespace(found=found))
    scene = {"robot": robot, task.FOOT_CONTACT_SENSOR_NAME: sensor}
    asset_cfg = SimpleNamespace(name="robot", site_ids=[0, 1])
    env = SimpleNamespace(num_envs=2, scene=scene)

    value = task.always_on_feet_slip(
        env,
        sensor_name=task.FOOT_CONTACT_SENSOR_NAME,
        asset_cfg=asset_cfg,
    )

    torch.testing.assert_close(value, torch.tensor([25.0, 18.0]))


def test_causal_proof_guard_rejects_missing_q10() -> None:
    command = SimpleNamespace(
        time_steps=torch.tensor([8, 9], dtype=torch.long),
        motion=SimpleNamespace(time_step_total=10),
    )
    manager = SimpleNamespace(get_term=lambda name: command)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        command_manager=manager,
    )

    with pytest.raises(RuntimeError, match="q10 proof"):
        task.causal_proof_frame_guard(env)


def test_source_bound_weights_match_selected_task_and_causal_safety() -> None:
    contract = task.native124_selected_v2_ankle_task_contract()["rewards"]

    assert contract["motion_ankle_pos"] == {"weight": 1.5, "std_m": 0.15}
    assert contract["motion_ankle_ori"] == {"weight": 1.0, "std_rad": 0.25}
    assert contract["motion_global_root_pos"] == {
        "weight": 0.5,
        "std_m": 0.3,
        "causal_reference": "q9_pelvis",
    }
    assert contract["motion_global_root_ori"] == {
        "weight": 0.5,
        "std_rad": 0.4,
        "causal_reference": "q9_pelvis",
    }
    assert contract["joint_torques_l2"]["weight"] == -2.0e-3
    assert contract["actuator_saturation"] == {
        "weight": -5.0,
        "threshold_ratio": 0.9,
    }
    assert contract["target_soft_limit"] == {
        "weight": -50.0,
        "inner_margin_fraction": 0.025,
    }
    assert contract["evaluator_aligned_recovery"] == {
        "weight": -25.0,
        "metric": "pelvis_tilt+pelvis_height_error+applied_target_tracking_rmse",
    }
    assert contract["non_timeout_termination"]["cost_per_event_at_50hz"] == -100.0


def test_real_mjlab_task_config_constructs_without_rollout() -> None:
    if causal._MJLAB_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab runtime unavailable")

    cfg = task.make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=str(DAD_DANCE),
        num_envs=1,
        play=False,
    )
    audit = task.audit_native124_selected_v2_ankle_task_env_cfg(cfg)

    assert audit["critic_observation_dim"] == 256
    assert "action_target_reference_l2" not in cfg.rewards
    assert len(cfg.scene.sensors) == 2


def test_real_mjlab_environment_constructs_without_simulator_step() -> None:
    if causal._MJLAB_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab runtime unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA MJLab runtime unavailable")

    from mjlab.envs import ManagerBasedRlEnv

    cfg = task.make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=str(DAD_DANCE),
        num_envs=1,
        play=False,
    )
    cfg.seed = 20260809
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        assert env.common_step_counter == 0
        assert task.FOOT_CONTACT_SENSOR_NAME in env.scene.sensors
        assert env.scene[task.FOOT_CONTACT_SENSOR_NAME].data.found.shape == (1, 2)
    finally:
        env.close()


def test_real_cuda_ankle_task_wrapper_prime_and_two_zero_action_steps() -> None:
    """WSL/CUDA smoke; never runs more than two simulator transitions."""

    if causal._MJLAB_IMPORT_ERROR is not None or causal._WRAPPER_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab RSL runtime unavailable")
    if not torch.cuda.is_available():
        pytest.skip("CUDA MJLab runtime unavailable")

    from mjlab.envs import ManagerBasedRlEnv

    cfg = task.make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=str(DAD_DANCE),
        num_envs=1,
        play=False,
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

        assert prime["action_substitution"] is False
        assert reset_observations["actor"].shape == (1, 124)
        assert reset_observations["critic"].shape == (1, causal.SONIC_TRUE23_CRITIC_DIM)
        assert torch.isfinite(reset_observations["actor"]).all()
        assert torch.isfinite(reset_observations["critic"]).all()

        command = env.command_manager.get_term("motion")
        zero = torch.zeros((1, 23), dtype=torch.float32, device="cuda:0")
        for _ in range(2):
            q9_before = command.time_steps.clone()
            observations, reward, dones, _ = wrapped.step(zero)

            assert observations["actor"].shape == (1, 124)
            assert observations["critic"].shape == (1, causal.SONIC_TRUE23_CRITIC_DIM)
            assert torch.isfinite(observations["actor"]).all()
            assert torch.isfinite(observations["critic"]).all()
            assert reward.numel() == 1
            assert torch.isfinite(reward).all()
            assert dones.tolist() == [0]

            diagnostics = wrapped.diagnostics
            assert torch.equal(diagnostics.q9_indices, q9_before + 1)
            assert torch.equal(command.time_steps, q9_before + 1)

            action_term = env.action_manager.get_term("joint_pos")
            assert action_term.raw_clip_mask_native.shape == (1, 23)
            assert not torch.any(action_term.raw_clip_mask_native)

            contact = env.scene[task.FOOT_CONTACT_SENSOR_NAME].data
            assert contact.found.shape == (1, 2)
            assert contact.force.shape == (1, 2, 3)
            assert contact.force_history.shape == (1, 2, 4, 3)
            assert contact.current_air_time.shape == (1, 2)
            assert contact.current_contact_time.shape == (1, 2)
            assert torch.isfinite(contact.found).all()
            assert torch.isfinite(contact.force).all()
            assert torch.isfinite(contact.force_history).all()
            assert torch.isfinite(contact.current_air_time).all()
            assert torch.isfinite(contact.current_contact_time).all()
            assert torch.all(contact.found >= 0)
            assert torch.equal(
                contact.found > 0,
                contact.current_contact_time > 0,
            )
    finally:
        env.close()
