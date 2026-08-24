"""Fail-closed contract tests for causal stand acquisition v4."""

from pathlib import Path

import pytest

from gear_sonic.envs.mjlab.sonic_true23_causal_history_stand_acquisition_v4 import (
    causal_stand_acquisition_v4_contract,
    make_causal_history_stand_acquisition_v4_env_cfg,
)


def test_v4_terminal_cost_survives_dt_scaling() -> None:
    contract = causal_stand_acquisition_v4_contract()
    assert contract["non_timeout_termination"] == {
        "function": "mjlab.envs.mdp.is_terminated",
        "weight": -5000.0,
        "control_dt_s": 0.02,
        "cost_per_event": -100.0,
    }
    assert contract["alive_reward_weight"] == 5.0
    assert contract["tracking_weight_multiplier"] == 2.0


def test_v4_reference_cannot_wrap_inside_one_episode() -> None:
    contract = causal_stand_acquisition_v4_contract()
    assert contract["sampling_mode"] == "start"
    assert contract["fixed_causal_start_anchor"] == 9
    assert contract["maximum_episode_steps"] == 500
    assert contract["neutral_frame_count"] == 600
    assert contract["minimum_unused_tail_frames"] == 90
    assert contract["in_episode_reference_resample_permitted"] is False
    assert contract["runtime_reference_exhaustion_guard"] is True


def test_v4_zeroes_rsi_without_weakening_safety() -> None:
    contract = causal_stand_acquisition_v4_contract()
    assert contract["reset_pose_range"] == {}
    assert contract["reset_velocity_range"] == {}
    assert contract["reset_joint_position_range_rad"] == [0.0, 0.0]
    assert contract["ee_body_pos_termination_threshold_m"] == 0.25
    assert contract["action_target_reference_penalty_preserved"] is True
    assert contract["action_target_soft_limit_barrier_preserved"] is True
    assert contract["actual_joint_soft_limit_penalty_preserved"] is True


def test_v4_constructed_mjlab_cfg_matches_contract() -> None:
    pytest.importorskip("mjlab")
    motion = Path(
        "/root/.cache/g1_true23_mjlab/recovery/"
        "g1_true23_causal_neutral_acquisition_v3.npz"
    )
    if not motion.is_file():
        pytest.skip("pinned WSL neutral acquisition motion is unavailable")
    cfg = make_causal_history_stand_acquisition_v4_env_cfg(
        motion_file=str(motion),
        num_envs=4,
        play=False,
    )
    control_dt = cfg.sim.mujoco.timestep * cfg.decimation
    command = cfg.commands["motion"]
    assert control_dt == 0.02
    assert round(cfg.episode_length_s / control_dt) == 500
    assert command.sampling_mode == "start"
    assert command.pose_range == {}
    assert command.velocity_range == {}
    assert command.joint_position_range == (0.0, 0.0)
    assert "push_robot" not in cfg.events
    assert cfg.rewards["non_timeout_termination"].weight == -5000.0
    assert cfg.rewards["alive"].weight == 5.0
    assert cfg.rewards["fixed_start_reference_exhaustion_guard"].weight == 1.0
    assert cfg.terminations["ee_body_pos"].params["threshold"] == 0.25
    assert cfg.rewards["action_target_reference_l2"].weight == -2.0
    assert cfg.rewards["action_target_soft_limit_barrier"].weight == -10.0
    assert cfg.rewards["joint_limit"].weight == -20.0
    assert cfg.rewards["action_rate_l2"].weight == -0.2
