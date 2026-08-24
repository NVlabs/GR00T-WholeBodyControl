"""Contract tests for nominal causal stand acquisition v5."""

from pathlib import Path

import pytest

from gear_sonic.envs.mjlab.sonic_true23_causal_history_stand_acquisition_v5 import (
    causal_stand_acquisition_v5_contract,
    make_causal_history_stand_acquisition_v5_env_cfg,
)


def test_v5_incentive_math_is_explicit() -> None:
    contract = causal_stand_acquisition_v5_contract()
    assert contract["alive_reward"]["reward_per_nonterminal_step"] == 0.4
    barrier = contract["target_soft_limit_barrier"]
    assert barrier["maximum_normalized_penalty"] == 0.25
    assert barrier["maximum_cost_per_step"] == -0.05
    assert barrier["physical_soft_limit_gate_weakened"] is False
    assert contract["alive_to_maximum_target_barrier_step_ratio"] == 8.0
    assert contract["non_timeout_terminal_cost_per_event"] == -100.0


def test_v5_keeps_physical_gates_and_marks_later_stages() -> None:
    contract = causal_stand_acquisition_v5_contract()
    assert contract["actual_joint_soft_limit_penalty_weight"] == -20.0
    assert contract["action_target_reference_penalty_weight"] == -2.0
    assert contract["ee_body_pos_termination_threshold_m"] == 0.25
    assert contract["sampling_mode"] == "start"
    assert contract["maximum_pre_timeout_anchor"] == 508
    assert contract["maximum_valid_anchor"] == 598
    assert contract["domain_randomization_finetune_required"] is True
    assert contract["push_disturbance_finetune_required"] is True
    assert contract["transition_and_dance_finetune_required"] is True
    assert contract["deployment_ready"] is False


def test_v5_constructed_mjlab_cfg_matches_contract() -> None:
    pytest.importorskip("mjlab")
    motion = Path(
        "/root/.cache/g1_true23_mjlab/recovery/"
        "g1_true23_causal_neutral_acquisition_v3.npz"
    )
    if not motion.is_file():
        pytest.skip("pinned WSL neutral acquisition motion is unavailable")
    cfg = make_causal_history_stand_acquisition_v5_env_cfg(
        motion_file=str(motion), num_envs=4, play=False
    )
    assert cfg.commands["motion"].sampling_mode == "start"
    assert cfg.rewards["alive"].weight == 20.0
    assert cfg.rewards["non_timeout_termination"].weight == -5000.0
    assert cfg.rewards["action_target_soft_limit_barrier"].weight == -10.0
    assert cfg.terminations["ee_body_pos"].params["threshold"] == 0.25
    assert cfg.events == {}
