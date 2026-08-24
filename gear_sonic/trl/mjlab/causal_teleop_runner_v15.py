"""Conservative final-affine PPO from official multi-motion SONIC init."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from gear_sonic.trl.mjlab.causal_history_runner import CausalHistoryMjlabOnPolicyRunner
from gear_sonic.trl.mjlab.sonic_recovery_blend_policy import load_hash_bound_policy_state
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state

OFFICIAL_INIT_PATH = Path(
    "/mnt/z/codex/GR00T-WholeBodyControl/sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"
)
OFFICIAL_INIT_SHA256 = "5e5be23982f15eaf2eb1d52b2433d081b5f43c260ecb5055457edf222b77c9bb"
OFFICIAL_POLICY_SHA256 = "39049d5018608f198dee1d2ea5e0f465d212dc4572c036d21378a36f80c1fc2f"
TRAINABLE_MODULE = "decoders.g1_dyn.module.16."
LEARNING_RATE = 1.0e-6
FIXED_EXPLORATION_STD = 0.10


def is_v15_trainable_actor_parameter(name: str) -> bool:
    return TRAINABLE_MODULE in name


class CausalTeleopRunnerV15(CausalHistoryMjlabOnPolicyRunner):
    """Fresh PPO state; official actor; final affine weight/bias only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        actor = self.alg.get_policy()
        state = load_hash_bound_policy_state(
            OFFICIAL_INIT_PATH,
            expected_checkpoint_sha256=OFFICIAL_INIT_SHA256,
            expected_policy_sha256=OFFICIAL_POLICY_SHA256,
        )
        std_names = [name for name in state if name.rsplit(".", 1)[-1] in {"std", "log_std"}]
        if len(std_names) != 1:
            raise ValueError("v15 official checkpoint requires one std tensor")
        network = {name: value for name, value in state.items() if name != std_names[0]}
        actor.core.load_state_dict(network, strict=True)
        with torch.no_grad():
            actor.distribution.std_param.fill_(FIXED_EXPLORATION_STD)

        named = dict(actor.named_parameters())
        trainable_names: list[str] = []
        frozen_names: list[str] = []
        for name, parameter in named.items():
            trainable = is_v15_trainable_actor_parameter(name)
            parameter.requires_grad_(trainable)
            (trainable_names if trainable else frozen_names).append(name)
        if len(trainable_names) != 2:
            raise ValueError(f"v15 expected final affine weight/bias, got {trainable_names}")
        if any("std" in name for name in trainable_names):
            raise ValueError("v15 exploration std must remain frozen")

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
        self._v15_frozen_initial = {
            name: named[name].detach().clone() for name in frozen_names
        }
        policy_sha = inspect_true23_policy_state(
            {"policy_state_dict": actor.export_true23_policy_state()},
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
            raise RuntimeError("v15 live official overlay hash mismatch")

        runtime = {
            "schema": "g1_true23_causal_teleop_runtime_v15",
            "official_init_path": str(OFFICIAL_INIT_PATH),
            "official_init_sha256": OFFICIAL_INIT_SHA256,
            "official_policy_sha256_before_std_pin": OFFICIAL_POLICY_SHA256,
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
        log_dir = Path(args[2] if len(args) > 2 else getattr(self, "log_dir", ".")).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "v15_runtime_contract.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._v15_runtime = runtime

    def assert_frozen_actor_unchanged(self) -> None:
        named = dict(self.alg.get_policy().named_parameters())
        drifted = [
            name
            for name, initial in self._v15_frozen_initial.items()
            if not torch.equal(named[name].detach(), initial)
        ]
        if drifted:
            raise RuntimeError(f"v15 frozen actor tensors changed: {drifted[:5]}")


__all__ = [
    "CausalTeleopRunnerV15",
    "FIXED_EXPLORATION_STD",
    "LEARNING_RATE",
    "OFFICIAL_INIT_PATH",
    "OFFICIAL_INIT_SHA256",
    "OFFICIAL_POLICY_SHA256",
    "TRAINABLE_MODULE",
    "is_v15_trainable_actor_parameter",
]
