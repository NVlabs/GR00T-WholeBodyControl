"""Contract tests for causal disturbance fine-tuning v9."""

from gear_sonic.envs.mjlab.sonic_true23_causal_history_disturbance_v9 import (
    causal_history_disturbance_v9_contract,
)


def test_v9_restores_exact_base_disturbance_set() -> None:
    contract = causal_history_disturbance_v9_contract()
    assert contract["restored_exact_base_events"] == [
        "push_robot",
        "base_com",
        "encoder_bias",
        "foot_friction",
    ]
    assert contract["physical_termination_and_reference_gates_unchanged"] is True


def test_v9_uses_uncapped_stronger_target_barrier_and_v8_boundary() -> None:
    contract = causal_history_disturbance_v9_contract()
    assert contract["target_soft_limit_barrier_weight"] == -50.0
    assert contract["target_soft_limit_barrier_uncapped"] is True
    assert contract["v8_final_affine_and_cumulative_kl_boundary_retained"] is True
    assert contract["deployment_ready"] is False
