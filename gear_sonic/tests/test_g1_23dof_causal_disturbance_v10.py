"""Contract tests for 100-update causal disturbance v10."""

from gear_sonic.scripts.train_g1_23dof_mjlab_causal_disturbance_v10 import (
    causal_final_affine_projected_v10_contract,
)


def test_v10_keeps_exact_final_affine_trust_region() -> None:
    contract = causal_final_affine_projected_v10_contract()
    assert contract["trainable_actor_parameters"] == [
        "core.actor_module.decoders.g1_dyn.module.16.bias",
        "core.actor_module.decoders.g1_dyn.module.16.weight",
    ]
    assert contract["all_other_actor_parameters_and_std_frozen"] is True
    assert contract["calibration_kl_limit"] == 2.0e-3
    assert contract["projection_math_unchanged_from_v8"] is True


def test_v10_uses_stronger_bounded_hundred_update_run() -> None:
    contract = causal_final_affine_projected_v10_contract()
    assert contract["learning_rate"] == 5.0e-7
    assert contract["planned_accepted_updates"] == 100
    assert contract["allowed_checkpoint_updates"] == [0, 10, 25, 50, 100]
    assert contract["v9_executed_task_contract_unchanged"] is True
