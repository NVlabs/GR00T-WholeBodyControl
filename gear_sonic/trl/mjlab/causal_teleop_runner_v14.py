"""Conservative multi-motion PPO runner initialized from proven recovery actor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from gear_sonic.trl.mjlab.causal_history_runner import CausalHistoryMjlabOnPolicyRunner
from gear_sonic.trl.mjlab.sonic_recovery_blend_policy import (
    RECOVERY_CHECKPOINT_PATH,
    RECOVERY_CHECKPOINT_SHA256,
    RECOVERY_POLICY_SHA256,
    load_hash_bound_policy_state,
)
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state

TRAINABLE_MODULES = ("decoders.g1_dyn.module.14.", "decoders.g1_dyn.module.16.")
LEARNING_RATE = 5.0e-6
FIXED_EXPLORATION_STD = 0.10


def is_v14_trainable_actor_parameter(name: str) -> bool:
    return any(fragment in name for fragment in TRAINABLE_MODULES)


class CausalTeleopRunnerV14(CausalHistoryMjlabOnPolicyRunner):
    """Fresh PPO state; exact recovery actor; final hidden block/head only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        actor = self.alg.get_policy()
        state = load_hash_bound_policy_state(
            RECOVERY_CHECKPOINT_PATH,
            expected_checkpoint_sha256=RECOVERY_CHECKPOINT_SHA256,
            expected_policy_sha256=RECOVERY_POLICY_SHA256,
        )
        std_names = [name for name in state if name.rsplit(".", 1)[-1] in {"std", "log_std"}]
        if len(std_names) != 1:
            raise ValueError("v14 recovery checkpoint requires one std tensor")
        network = {name: value for name, value in state.items() if name != std_names[0]}
        actor.core.load_state_dict(network, strict=True)
        with torch.no_grad():
            actor.distribution.std_param.fill_(FIXED_EXPLORATION_STD)

        named = dict(actor.named_parameters())
        trainable_names: list[str] = []
        frozen_names: list[str] = []
        for name, parameter in named.items():
            trainable = is_v14_trainable_actor_parameter(name)
            parameter.requires_grad_(trainable)
            (trainable_names if trainable else frozen_names).append(name)
        if len(trainable_names) != 4:
            raise ValueError(f"v14 expected exactly four actor tensors, got {trainable_names}")
        if any("std" in name for name in trainable_names):
            raise ValueError("v14 exploration std must remain frozen")

        old_optimizer = self.alg.optimizer
        defaults = dict(old_optimizer.defaults)
        defaults["lr"] = LEARNING_RATE
        critic_parameters = [
            parameter for parameter in self.alg.critic.parameters() if parameter.requires_grad
        ]
        self.alg.optimizer = type(old_optimizer)(
            [*(named[name] for name in trainable_names), *critic_parameters],
            **defaults,
        )
        self.alg.learning_rate = LEARNING_RATE
        self._v14_frozen_initial = {
            name: named[name].detach().clone() for name in frozen_names
        }
        exported = actor.export_true23_policy_state()
        policy_sha = inspect_true23_policy_state(
            {"policy_state_dict": exported},
            reference_profile="released_low_latency_step1_0p02s",
        )
        expected_overlay_sha = inspect_true23_policy_state(
            {
                "policy_state_dict": {
                    **network,
                    "std": torch.full((23,), FIXED_EXPLORATION_STD, dtype=torch.float32),
                }
            },
            reference_profile="released_low_latency_step1_0p02s",
        )
        if policy_sha != expected_overlay_sha:
            raise RuntimeError("v14 live recovery overlay hash mismatch")

        runtime = {
            "schema": "g1_true23_causal_teleop_runtime_v14",
            "recovery_checkpoint_path": str(RECOVERY_CHECKPOINT_PATH),
            "recovery_checkpoint_sha256": RECOVERY_CHECKPOINT_SHA256,
            "recovery_policy_sha256_before_std_pin": RECOVERY_POLICY_SHA256,
            "initial_overlay_policy_sha256": policy_sha,
            "fixed_exploration_std": FIXED_EXPLORATION_STD,
            "learning_rate": LEARNING_RATE,
            "trainable_actor_parameters": sorted(trainable_names),
            "frozen_actor_parameters": sorted(frozen_names),
            "critic_fresh": True,
            "optimizer_fresh": True,
            "resume_checkpoint_loaded": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        try:
            log_dir = Path(args[2]).expanduser().resolve()
        except (IndexError, TypeError):
            log_dir = Path(getattr(self, "log_dir", ".")).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "v14_runtime_contract.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._v14_runtime = runtime

    def assert_frozen_actor_unchanged(self) -> None:
        named = dict(self.alg.get_policy().named_parameters())
        drifted = [
            name
            for name, initial in self._v14_frozen_initial.items()
            if not torch.equal(named[name].detach(), initial)
        ]
        if drifted:
            raise RuntimeError(f"v14 frozen actor tensors changed: {drifted[:5]}")


__all__ = [
    "CausalTeleopRunnerV14",
    "FIXED_EXPLORATION_STD",
    "LEARNING_RATE",
    "TRAINABLE_MODULES",
    "is_v14_trainable_actor_parameter",
]
