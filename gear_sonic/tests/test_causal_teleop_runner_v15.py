from gear_sonic.trl.mjlab.causal_teleop_runner_v15 import (
    FIXED_EXPLORATION_STD,
    LEARNING_RATE,
    OFFICIAL_INIT_SHA256,
    OFFICIAL_POLICY_SHA256,
    is_v15_trainable_actor_parameter,
)


def test_v15_exact_final_affine_scope() -> None:
    assert is_v15_trainable_actor_parameter("core.decoders.g1_dyn.module.16.weight")
    assert is_v15_trainable_actor_parameter("core.decoders.g1_dyn.module.16.bias")
    assert not is_v15_trainable_actor_parameter("core.decoders.g1_dyn.module.14.weight")
    assert not is_v15_trainable_actor_parameter("distribution.std_param")


def test_v15_constants_are_conservative_and_hash_bound() -> None:
    assert LEARNING_RATE == 1.0e-6
    assert FIXED_EXPLORATION_STD == 0.10
    assert len(OFFICIAL_INIT_SHA256) == 64
    assert len(OFFICIAL_POLICY_SHA256) == 64
