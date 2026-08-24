"""Contract tests for low-exploration causal stand acquisition v6."""

from gear_sonic.scripts.train_g1_23dof_mjlab_causal_history_stand_acquisition_v6 import (
    causal_stand_acquisition_v6_contract,
)


def test_v6_exploration_change_is_explicit_and_bounded() -> None:
    contract = causal_stand_acquisition_v6_contract()
    assert contract["exploration_std_before_first_rollout"] == 0.1
    assert contract["exploration_std_applies_to"] == "training_gaussian_only"
    assert contract["network_weights_changed_before_first_rollout"] is False
    assert contract["exploration_state_change_declared_in_resolved_lineage"] is True
    assert contract["entropy_coefficient"] == 0.0


def test_v6_preserves_strict_acquisition_gates() -> None:
    contract = causal_stand_acquisition_v6_contract()
    assert contract["strict_ee_threshold_m"] == 0.25
    assert contract["actual_joint_limit_penalty_weight"] == -20.0
    assert contract["action_target_reference_penalty_weight"] == -2.0
    assert contract["action_target_soft_limit_barrier_weight"] == -10.0
    assert contract["target_barrier_maximum_normalized_penalty"] == 0.25
    assert contract["fixed_start_no_wrap_guard_preserved"] is True
    assert contract["rsi_noise_disabled"] is True
    assert contract["deployment_ready"] is False
