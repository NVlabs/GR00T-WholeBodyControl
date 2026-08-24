from __future__ import annotations

from types import SimpleNamespace

import torch

from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import (
    causal_multimotion_v14_contract,
    corpus_clip_time_out,
)
from gear_sonic.trl.mjlab.causal_teleop_runner_v14 import (
    FIXED_EXPLORATION_STD,
    LEARNING_RATE,
    is_v14_trainable_actor_parameter,
)


def test_corpus_clip_timeout_fires_before_boundary_crossing() -> None:
    command = SimpleNamespace(
        time_steps=torch.tensor([18, 19, 20], dtype=torch.long),
        _env_clip_stop=torch.tensor([20, 20, 22], dtype=torch.long),
    )
    env = SimpleNamespace(
        num_envs=3,
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    assert torch.equal(corpus_clip_time_out(env), torch.tensor([False, True, False]))


def test_v14_trainable_scope_is_exact_final_block_and_head() -> None:
    assert is_v14_trainable_actor_parameter("core.actor_module.decoders.g1_dyn.module.14.weight")
    assert is_v14_trainable_actor_parameter("core.actor_module.decoders.g1_dyn.module.16.bias")
    assert not is_v14_trainable_actor_parameter("core.actor_module.decoders.g1_dyn.module.12.weight")
    assert not is_v14_trainable_actor_parameter("distribution.std_param")
    assert LEARNING_RATE == 5.0e-6
    assert FIXED_EXPLORATION_STD == 0.10


def test_v14_contract_is_simulator_only_and_death_proof() -> None:
    contract = causal_multimotion_v14_contract()
    assert contract["action_boundary"] == "raw_native23_to_safe_target_v11_once"
    assert contract["clip_boundary"] == "full_environment_timeout_reset_before_crossing"
    assert contract["non_timeout_termination_weight"] == -5000.0
    assert contract["deployment_ready"] is False
    assert contract["hardware_authorized"] is False
