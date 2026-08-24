"""Contract tests for continuous final-affine causal v8."""

from gear_sonic.scripts.train_g1_23dof_mjlab_causal_final_affine_projected_v8 import (
    causal_final_affine_projected_v8_contract,
)


def test_v8_trains_only_exact_final_affine_pair() -> None:
    contract = causal_final_affine_projected_v8_contract()
    assert contract["trainable_actor_parameters"] == [
        "core.actor_module.decoders.g1_dyn.module.16.bias",
        "core.actor_module.decoders.g1_dyn.module.16.weight",
    ]
    assert contract["encoder_and_fsq_frozen"] is True
    assert contract["decoder_hidden_layers_frozen"] is True
    assert contract["exploration_std_frozen"] is True


def test_v8_uses_exact_cumulative_kl_projection() -> None:
    contract = causal_final_affine_projected_v8_contract()
    assert contract["learning_rate"] == 1.0e-7
    assert contract["schedule"] == "fixed"
    assert contract["ppo_clip"] == 0.05
    assert contract["learning_epochs"] == 1
    assert contract["entropy_coefficient"] == 0.0
    assert contract["calibration_kl_limit"] == 2.0e-3
    assert contract["projection_target_kl"] == 1.8e-3
    assert contract["projection_basis"] == (
        "cumulative_final_affine_delta_from_model0"
    )
    assert contract["fail_closed_boundary_frequency_updates"] == 1
    assert contract["allowed_checkpoint_updates"] == [0, 10, 25, 50]
