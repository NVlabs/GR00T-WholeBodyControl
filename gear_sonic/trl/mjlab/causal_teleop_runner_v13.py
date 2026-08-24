"""Teleoperation training runner: full decoder, frozen SONIC encoder.

The v8/v10 lineage trains two tensors — the final decoder affine — at 5e-7
behind a KL trust region. That is a calibration probe: it is built to barely
move the policy, and it cannot learn a new embodiment's output mapping. Runs
built on it plateaued with the policy essentially unchanged from its warm start.

This runner takes the opposite side of the tradeoff that matters here. The warm
start carries genuine SONIC low-latency weights, and the value worth preserving
is the teleoperation encoder and token backbone, which already know how to read
operator motion. What does not transfer is the decoder, whose output head was
reshaped from 29 to 23 joints. So the encoders stay frozen and the whole
``g1_dyn`` decoder trains, at a learning rate that can actually move it.

Exploration std is trainable: a fixed acquisition std is right for a calibration
probe and wrong for a run that has to explore a new action space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from gear_sonic.trl.mjlab.causal_history_runner import (
    CausalHistoryMjlabOnPolicyRunner,
)

# Parameter-name fragments. Matching on a fragment rather than an exact name
# keeps this working whether the actor is wrapped (``core.actor_module...``) or
# not (``actor_module...``).
DECODER_FRAGMENT = "decoders.g1_dyn."
ENCODER_FRAGMENT = "encoders."
STD_FRAGMENT = "std"

V13_LEARNING_RATE = 1.0e-4
V13_TRAIN_EXPLORATION_STD = True


def _is_decoder(name: str) -> bool:
    return DECODER_FRAGMENT in name


def _is_encoder(name: str) -> bool:
    return ENCODER_FRAGMENT in name


def _is_std(name: str) -> bool:
    return name.split(".")[-1] == STD_FRAGMENT or name == STD_FRAGMENT


class CausalTeleopRunnerV13(CausalHistoryMjlabOnPolicyRunner):
    """Train the 23-joint decoder while preserving the SONIC teleop encoder."""

    learning_rate_override = V13_LEARNING_RATE
    train_exploration_std = V13_TRAIN_EXPLORATION_STD
    runtime_schema = "g1_true23_causal_teleop_runtime_v13"
    runtime_filename = "v13_runtime_contract.json"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        actor = self.alg.get_policy()
        named = dict(actor.named_parameters())
        if not any(_is_decoder(name) for name in named):
            raise ValueError(
                "v13 expected a g1_dyn decoder in the actor namespace; "
                f"saw {sorted(named)[:6]}"
            )
        if not any(_is_encoder(name) for name in named):
            raise ValueError("v13 expected encoder parameters to freeze")

        trainable_names: list[str] = []
        frozen_names: list[str] = []
        for name, parameter in named.items():
            trainable = _is_decoder(name) or (
                self.train_exploration_std and _is_std(name)
            )
            parameter.requires_grad_(trainable)
            (trainable_names if trainable else frozen_names).append(name)

        trainable_actor = [named[name] for name in trainable_names]
        trainable_critic = [
            parameter
            for parameter in self.alg.critic.parameters()
            if parameter.requires_grad
        ]

        old_optimizer = self.alg.optimizer
        defaults = dict(old_optimizer.defaults)
        defaults["lr"] = float(self.learning_rate_override)
        self.alg.optimizer = type(old_optimizer)(
            [*trainable_actor, *trainable_critic], **defaults
        )
        self.alg.learning_rate = float(self.learning_rate_override)

        self._v13_frozen_initial = {
            name: named[name].detach().clone() for name in frozen_names
        }

        runtime = {
            "schema": self.runtime_schema,
            "trainable_actor_parameters": sorted(trainable_names),
            "frozen_actor_parameters": sorted(frozen_names),
            "trainable_actor_tensor_count": len(trainable_actor),
            "frozen_actor_tensor_count": len(frozen_names),
            "optimizer_critic_parameter_count": len(trainable_critic),
            "learning_rate": float(self.learning_rate_override),
            "exploration_std_trainable": bool(self.train_exploration_std),
            "encoder_frozen": True,
            "rationale": (
                "warm start carries SONIC low-latency encoder weights; the "
                "decoder head was reshaped 29->23 and must be retrained"
            ),
        }
        try:
            log_dir = Path(args[2]).expanduser().resolve()
        except (IndexError, TypeError):
            log_dir = Path(getattr(self, "log_dir", ".")).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / self.runtime_filename).write_text(
            __import__("json").dumps(runtime, indent=1, sort_keys=True)
        )
        self._v13_runtime = runtime

    def assert_encoder_unchanged(self, *, atol: float = 0.0) -> None:
        """Fail loudly if a frozen encoder tensor moved during training."""
        named = dict(self.alg.get_policy().named_parameters())
        drifted = [
            name
            for name, initial in self._v13_frozen_initial.items()
            if not torch.allclose(named[name].detach(), initial, atol=atol, rtol=0.0)
        ]
        if drifted:
            raise RuntimeError(f"frozen actor parameters changed: {drifted[:5]}")
