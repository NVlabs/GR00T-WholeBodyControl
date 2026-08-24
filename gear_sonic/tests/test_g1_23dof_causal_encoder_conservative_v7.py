"""Contract tests for bounded encoder-only causal v7."""

from gear_sonic.scripts.train_g1_23dof_mjlab_causal_encoder_conservative_v7 import (
    causal_encoder_conservative_v7_contract,
)


def test_v7_is_encoder_only_and_optimizer_excludes_frozen_state() -> None:
    contract = causal_encoder_conservative_v7_contract()
    assert contract["actor_trainable_prefix"] == (
        "core.actor_module.encoders.teleop."
    )
    assert contract["decoder_frozen"] is True
    assert contract["exploration_std_frozen"] is True
    assert contract["optimizer_excludes_decoder_and_std"] is True


def test_v7_conservative_boundaries_are_fail_closed() -> None:
    contract = causal_encoder_conservative_v7_contract()
    assert contract["learning_rate"] == 1.0e-9
    assert contract["schedule"] == "fixed"
    assert contract["ppo_clip"] == 0.05
    assert contract["learning_epochs"] == 1
    assert contract["entropy_coefficient"] == 0.0
    assert contract["calibration_kl_limit"] == 2.0e-3
    assert contract["encoder_relative_l2_limit"] == 2.0e-3
    assert contract["fail_closed_boundary_frequency_updates"] == 1
    assert contract["allowed_checkpoint_updates"] == [0, 10, 25, 50, 100]
    assert contract["rejected_update_behavior"] == (
        "poison_process_keep_last_exact_checkpoint"
    )
