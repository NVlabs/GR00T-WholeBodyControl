"""Encoder-only conservative PPO boundary for causal stand acquisition."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from gear_sonic.trl.mjlab.causal_stand_acquisition_runner_v6 import (
    ACQUISITION_EXPLORATION_STD,
    CausalStandAcquisitionRunnerV6,
)

# The first source-bound probe at 1e-6 produced calibration KL 0.284836471
# after one update.  Keep a 50-update cumulative margin below the immutable
# 0.002 gate; this remains a real, non-zero encoder update in float32.
FIXED_LEARNING_RATE = 1.0e-9
PPO_CLIP = 0.05
CALIBRATION_KL_LIMIT = 2.0e-3
ENCODER_RELATIVE_L2_LIMIT = 2.0e-3
ALLOWED_CHECKPOINT_UPDATES = frozenset({0, 10, 25, 50, 100})
_ENCODER_PREFIX = "core.actor_module.encoders.teleop."
_DECODER_PREFIX = "core.actor_module.decoders.g1_dyn."
_STD_NAME = "distribution.std_param"


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class CausalEncoderConservativeRunnerV7(CausalStandAcquisitionRunnerV6):
    """Freeze balance decoding and fail closed on conservative drift."""

    def __init__(
        self,
        *args: Any,
        resolved_config: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        contract = resolved_config.get("causal_encoder_conservative_v7")
        if not isinstance(contract, Mapping):
            raise ValueError("v7 runner requires resolved conservative contract")
        super().__init__(*args, resolved_config=resolved_config, **kwargs)
        actor = self.alg.get_policy()
        named_actor = dict(actor.named_parameters())
        encoder_names = sorted(
            name for name in named_actor if name.startswith(_ENCODER_PREFIX)
        )
        decoder_names = sorted(
            name for name in named_actor if name.startswith(_DECODER_PREFIX)
        )
        if not encoder_names or not decoder_names or _STD_NAME not in named_actor:
            raise ValueError("v7 actor parameter namespace is incomplete")
        known = set(encoder_names) | set(decoder_names) | {_STD_NAME}
        if set(named_actor) != known:
            raise ValueError(
                "v7 actor has unexpected parameters: "
                f"{sorted(set(named_actor) - known)}"
            )
        for name, parameter in named_actor.items():
            parameter.requires_grad_(name in encoder_names)

        self._v7_encoder_names = tuple(encoder_names)
        self._v7_decoder_names = tuple(decoder_names)
        self._v7_encoder_initial = {
            name: named_actor[name].detach().clone() for name in encoder_names
        }
        self._v7_decoder_initial = {
            name: named_actor[name].detach().clone() for name in decoder_names
        }
        self._v7_std_initial = named_actor[_STD_NAME].detach().clone()
        if not torch.all(
            self._v7_std_initial == ACQUISITION_EXPLORATION_STD
        ):
            raise ValueError("v7 initial exploration std differs from 0.10")

        trainable_actor = [named_actor[name] for name in encoder_names]
        trainable_critic = [
            parameter for parameter in self.alg.critic.parameters() if parameter.requires_grad
        ]
        old_optimizer = self.alg.optimizer
        defaults = dict(old_optimizer.defaults)
        defaults["lr"] = FIXED_LEARNING_RATE
        self.alg.optimizer = type(old_optimizer)(
            [*trainable_actor, *trainable_critic],
            **defaults,
        )
        self.alg.learning_rate = FIXED_LEARNING_RATE
        self._v7_frozen_parameter_ids = {
            id(named_actor[name]) for name in (*decoder_names, _STD_NAME)
        }
        self._v7_trainable_actor_ids = {id(parameter) for parameter in trainable_actor}
        self._verify_optimizer_boundary()

        calibration_obs = self.env.get_observations().to(self.device)
        self._v7_calibration_obs = calibration_obs.clone()
        with torch.inference_mode():
            self._v7_calibration_mean = actor(
                self._v7_calibration_obs,
                stochastic_output=False,
            ).detach().clone()
        runtime = {
            "schema": "g1_true23_causal_encoder_conservative_runtime_v7",
            "trainable_actor_parameters": list(self._v7_encoder_names),
            "frozen_decoder_parameters": list(self._v7_decoder_names),
            "frozen_exploration_parameter": _STD_NAME,
            "optimizer_actor_parameter_count": len(trainable_actor),
            "optimizer_critic_parameter_count": len(trainable_critic),
            "fixed_learning_rate": FIXED_LEARNING_RATE,
            "ppo_clip": PPO_CLIP,
            "calibration_kl_limit": CALIBRATION_KL_LIMIT,
            "encoder_relative_l2_limit": ENCODER_RELATIVE_L2_LIMIT,
            "decoder_initial_sha256": _state_sha256(self._v7_decoder_initial),
            "std_initial_sha256": _state_sha256(
                {_STD_NAME: self._v7_std_initial}
            ),
            "encoder_initial_sha256": _state_sha256(self._v7_encoder_initial),
        }
        log_dir = Path(args[2]).expanduser().resolve()
        runtime_path = log_dir / "v7_runtime_contract.json"
        encoded = json.dumps(runtime, indent=2, sort_keys=True) + "\n"
        if runtime_path.exists():
            if runtime_path.read_text(encoding="utf-8") != encoded:
                raise ValueError("existing v7 runtime contract differs")
        else:
            runtime_path.write_text(encoded, encoding="utf-8")
        print(  # noqa: T201
            "V7_TRAINABLE_ACTOR_PARAMETERS=" + json.dumps(encoder_names)
        )

    def _verify_optimizer_boundary(self) -> None:
        optimizer_ids = {
            id(parameter)
            for group in self.alg.optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids & self._v7_frozen_parameter_ids:
            raise RuntimeError("v7 optimizer contains decoder or exploration std")
        if not self._v7_trainable_actor_ids.issubset(optimizer_ids):
            raise RuntimeError("v7 optimizer omits encoder parameters")
        for group in self.alg.optimizer.param_groups:
            if float(group["lr"]) != FIXED_LEARNING_RATE:
                raise RuntimeError("v7 optimizer learning rate drifted")
        if float(self.alg.learning_rate) != FIXED_LEARNING_RATE:
            raise RuntimeError("v7 algorithm learning rate drifted")

    def _conservative_boundary(self) -> None:
        actor = self.alg.get_policy()
        named_actor = dict(actor.named_parameters())
        self._verify_optimizer_boundary()
        for name, initial in self._v7_decoder_initial.items():
            if not torch.equal(named_actor[name].detach(), initial):
                self._training_state_poisoned = True
                raise RuntimeError("v7 decoder changed; retaining last checkpoint")
        if not torch.equal(named_actor[_STD_NAME].detach(), self._v7_std_initial):
            self._training_state_poisoned = True
            raise RuntimeError("v7 exploration std changed; retaining last checkpoint")

        delta_sq = torch.zeros((), device=self.device)
        base_sq = torch.zeros((), device=self.device)
        for name, initial in self._v7_encoder_initial.items():
            current = named_actor[name]
            delta_sq += torch.sum(torch.square(current - initial))
            base_sq += torch.sum(torch.square(initial))
        relative_l2 = torch.sqrt(delta_sq / base_sq.clamp_min(1.0e-12))
        with torch.inference_mode():
            current_mean = actor(
                self._v7_calibration_obs,
                stochastic_output=False,
            )
            calibration_kl = 0.5 * torch.mean(
                torch.sum(
                    torch.square(
                        (current_mean - self._v7_calibration_mean)
                        / self._v7_std_initial
                    ),
                    dim=-1,
                )
            )
        relative_l2_value = float(relative_l2.detach())
        calibration_kl_value = float(calibration_kl.detach())
        print(  # noqa: T201
            "V7_CONSERVATIVE_BOUNDARY="
            + json.dumps(
                {
                    "calibration_kl": calibration_kl_value,
                    "completed_update_count": self._require_counter_coherence(),
                    "encoder_relative_l2": relative_l2_value,
                },
                sort_keys=True,
            )
        )
        if relative_l2_value > ENCODER_RELATIVE_L2_LIMIT:
            self._training_state_poisoned = True
            raise RuntimeError(
                "v7 encoder drift budget exceeded "
                f"({relative_l2_value:.9g} > {ENCODER_RELATIVE_L2_LIMIT:.9g}); "
                "retaining last checkpoint"
            )
        if calibration_kl_value > CALIBRATION_KL_LIMIT:
            self._training_state_poisoned = True
            raise RuntimeError(
                "v7 calibration KL exceeded "
                f"({calibration_kl_value:.9g} > {CALIBRATION_KL_LIMIT:.9g}); "
                "retaining last checkpoint"
            )

    def _save_numbered_checkpoint(self) -> None:
        self._conservative_boundary()
        completed = self._require_counter_coherence()
        if completed in ALLOWED_CHECKPOINT_UPDATES:
            super()._save_numbered_checkpoint()
