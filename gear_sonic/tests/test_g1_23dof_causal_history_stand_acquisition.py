"""Source-contract tests for causal stand acquisition v3."""

from gear_sonic.envs.mjlab.sonic_true23_causal_history_stand_acquisition import (
    causal_stand_acquisition_contract,
)


def test_stand_acquisition_is_strict_and_track_first() -> None:
    contract = causal_stand_acquisition_contract()
    assert contract["motion_scope"] == "neutral_stand_only"
    assert contract["alive_reward"] == {
        "function": "mjlab.envs.mdp.is_alive",
        "weight": 5.0,
    }
    assert contract["tracking_weight_multiplier"] == 2.0
    assert contract["ee_body_pos_termination_threshold_m"] == 0.25
    assert contract["interval_pushes_enabled"] is False


def test_stand_acquisition_preserves_action_safety_rewards() -> None:
    contract = causal_stand_acquisition_contract()
    assert contract["action_target_reference_penalty_preserved"] is True
    assert contract["action_target_soft_limit_barrier_preserved"] is True
    assert contract["actual_joint_soft_limit_penalty_preserved"] is True
    assert contract["disturbance_finetune_required_before_promotion"] is True
