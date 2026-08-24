"""Continuous final-affine conservative PPO boundary for true23 causal stand."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from gear_sonic.trl.mjlab.causal_stand_acquisition_runner_v6 import (
    ACQUISITION_EXPLORATION_STD,
    CausalStandAcquisitionRunnerV6,
)

FIXED_LEARNING_RATE = 1.0e-7
PPO_CLIP = 0.05
CALIBRATION_KL_LIMIT = 2.0e-3
PROJECTION_TARGET_KL = 1.8e-3
ALLOWED_CHECKPOINT_UPDATES = frozenset({0, 10, 25, 50})
_FINAL_NAMES = (
    "core.actor_module.decoders.g1_dyn.module.16.bias",
    "core.actor_module.decoders.g1_dyn.module.16.weight",
)
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


class CausalFinalAffineProjectedRunnerV8(CausalStandAcquisitionRunnerV6):
    """Train only continuous final affine layer under exact cumulative KL."""

    required_contract_key = "causal_final_affine_projected_v8"
    fixed_learning_rate = FIXED_LEARNING_RATE
    allowed_checkpoint_updates = ALLOWED_CHECKPOINT_UPDATES
    runtime_schema = "g1_true23_causal_final_affine_projected_runtime_v8"
    runtime_filename = "v8_runtime_contract.json"
    telemetry_prefix = "V8"

    def __init__(
        self,
        *args: Any,
        resolved_config: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        contract = resolved_config.get(self.required_contract_key)
        if not isinstance(contract, Mapping):
            raise ValueError(
                f"runner requires resolved projection contract {self.required_contract_key}"
            )
        self._v8_fixed_learning_rate = float(self.fixed_learning_rate)
        self._v8_allowed_checkpoint_updates = frozenset(
            self.allowed_checkpoint_updates
        )
        super().__init__(*args, resolved_config=resolved_config, **kwargs)
        actor = self.alg.get_policy()
        named_actor = dict(actor.named_parameters())
        if not set(_FINAL_NAMES).issubset(named_actor) or _STD_NAME not in named_actor:
            raise ValueError("v8 actor parameter namespace is incomplete")

        for name, parameter in named_actor.items():
            parameter.requires_grad_(name in _FINAL_NAMES)
        self._v8_final_initial = {
            name: named_actor[name].detach().clone() for name in _FINAL_NAMES
        }
        self._v8_frozen_initial = {
            name: parameter.detach().clone()
            for name, parameter in named_actor.items()
            if name not in _FINAL_NAMES
        }
        if not torch.all(
            self._v8_frozen_initial[_STD_NAME] == ACQUISITION_EXPLORATION_STD
        ):
            raise ValueError("v8 initial exploration std differs from 0.10")

        trainable_actor = [named_actor[name] for name in _FINAL_NAMES]
        trainable_critic = [
            parameter
            for parameter in self.alg.critic.parameters()
            if parameter.requires_grad
        ]
        old_optimizer = self.alg.optimizer
        defaults = dict(old_optimizer.defaults)
        defaults["lr"] = self._v8_fixed_learning_rate
        self.alg.optimizer = type(old_optimizer)(
            [*trainable_actor, *trainable_critic],
            **defaults,
        )
        self.alg.learning_rate = self._v8_fixed_learning_rate
        self._v8_expected_optimizer_ids = {
            id(parameter) for parameter in (*trainable_actor, *trainable_critic)
        }
        self._v8_frozen_parameter_ids = {
            id(parameter)
            for name, parameter in named_actor.items()
            if name not in _FINAL_NAMES
        }
        self._verify_optimizer_boundary()

        self._v8_calibration_obs = self.env.get_observations().to(self.device).clone()
        with torch.inference_mode():
            self._v8_calibration_mean = actor(
                self._v8_calibration_obs,
                stochastic_output=False,
            ).detach().clone()
        self._v8_last_projection: dict[str, float | int] | None = None

        runtime = {
            "schema": self.runtime_schema,
            "trainable_actor_parameters": list(_FINAL_NAMES),
            "frozen_actor_parameters": sorted(self._v8_frozen_initial),
            "optimizer_actor_parameter_count": len(trainable_actor),
            "optimizer_critic_parameter_count": len(trainable_critic),
            "fixed_learning_rate": self._v8_fixed_learning_rate,
            "ppo_clip": PPO_CLIP,
            "calibration_kl_limit": CALIBRATION_KL_LIMIT,
            "projection_target_kl": PROJECTION_TARGET_KL,
            "projection_basis": "cumulative_final_affine_delta_from_model0",
            "frozen_initial_sha256": _state_sha256(self._v8_frozen_initial),
            "final_initial_sha256": _state_sha256(self._v8_final_initial),
        }
        log_dir = Path(args[2]).expanduser().resolve()
        runtime_path = log_dir / self.runtime_filename
        encoded = json.dumps(runtime, indent=2, sort_keys=True) + "\n"
        if runtime_path.exists():
            if runtime_path.read_text(encoding="utf-8") != encoded:
                raise ValueError("existing projected runtime contract differs")
        else:
            runtime_path.write_text(encoded, encoding="utf-8")
        print(  # noqa: T201
            self.telemetry_prefix
            + "_TRAINABLE_ACTOR_PARAMETERS="
            + json.dumps(_FINAL_NAMES)
        )

    def _verify_optimizer_boundary(self) -> None:
        optimizer_ids = {
            id(parameter)
            for group in self.alg.optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != self._v8_expected_optimizer_ids:
            raise RuntimeError("v8 optimizer parameter boundary differs")
        if optimizer_ids & self._v8_frozen_parameter_ids:
            raise RuntimeError("v8 optimizer contains frozen actor state")
        for group in self.alg.optimizer.param_groups:
            if float(group["lr"]) != self._v8_fixed_learning_rate:
                raise RuntimeError("v8 optimizer learning rate drifted")
        if float(self.alg.learning_rate) != self._v8_fixed_learning_rate:
            raise RuntimeError("v8 algorithm learning rate drifted")

    def _assert_frozen_actor_exact(self) -> None:
        named_actor = dict(self.alg.get_policy().named_parameters())
        for name, initial in self._v8_frozen_initial.items():
            if not torch.equal(named_actor[name].detach(), initial):
                raise RuntimeError(f"v8 frozen actor parameter changed: {name}")

    def _calibration_kl(self) -> torch.Tensor:
        with torch.inference_mode():
            current_mean = self.alg.get_policy()(
                self._v8_calibration_obs,
                stochastic_output=False,
            )
            std = self._v8_frozen_initial[_STD_NAME]
            return 0.5 * torch.mean(
                torch.sum(
                    torch.square((current_mean - self._v8_calibration_mean) / std),
                    dim=-1,
                )
            )

    def _project_completed_update(
        self,
        previous: Mapping[str, torch.Tensor],
    ) -> None:
        self._verify_optimizer_boundary()
        self._assert_frozen_actor_exact()
        named_actor = dict(self.alg.get_policy().named_parameters())
        candidate = {
            name: named_actor[name].detach().clone() for name in _FINAL_NAMES
        }
        if not any(
            not torch.equal(candidate[name], previous[name]) for name in _FINAL_NAMES
        ):
            raise RuntimeError("v8 PPO update made no final-affine change")

        candidate_kl = float(self._calibration_kl())
        if not math.isfinite(candidate_kl):
            raise RuntimeError("v8 candidate calibration KL is non-finite")
        alpha = 1.0
        if candidate_kl > CALIBRATION_KL_LIMIT:
            alpha = math.sqrt(PROJECTION_TARGET_KL / candidate_kl)
            with torch.no_grad():
                for name in _FINAL_NAMES:
                    initial = self._v8_final_initial[name]
                    named_actor[name].copy_(
                        initial + alpha * (candidate[name] - initial)
                    )

        accepted_kl = float(self._calibration_kl())
        predicted_kl = candidate_kl * alpha * alpha
        if (
            not math.isfinite(accepted_kl)
            or accepted_kl > CALIBRATION_KL_LIMIT
            or abs(accepted_kl - predicted_kl) > 2.0e-6
        ):
            with torch.no_grad():
                for name in _FINAL_NAMES:
                    named_actor[name].copy_(previous[name])
            raise RuntimeError("v8 exact cumulative KL projection failed")
        if not any(
            not torch.equal(named_actor[name].detach(), previous[name])
            for name in _FINAL_NAMES
        ):
            raise RuntimeError("v8 projected update made no final-affine change")
        self._assert_frozen_actor_exact()
        record: dict[str, float | int] = {
            "accepted_kl": accepted_kl,
            "candidate_kl": candidate_kl,
            "predicted_kl": predicted_kl,
            "projection_alpha": alpha,
            "proposed_update_count": self.completed_update_count + 1,
        }
        self._v8_last_projection = record
        print(  # noqa: T201
            self.telemetry_prefix
            + "_PROJECTED_UPDATE="
            + json.dumps(record, sort_keys=True)
        )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        original_update = self.alg.update

        def projected_update() -> Any:
            named_actor = dict(self.alg.get_policy().named_parameters())
            previous = {
                name: named_actor[name].detach().clone() for name in _FINAL_NAMES
            }
            result = original_update()
            self._project_completed_update(previous)
            return result

        self.alg.update = projected_update
        try:
            super().learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        finally:
            self.alg.update = original_update

    def _conservative_boundary(self) -> None:
        self._verify_optimizer_boundary()
        self._assert_frozen_actor_exact()
        calibration_kl = float(self._calibration_kl())
        if not math.isfinite(calibration_kl) or calibration_kl > CALIBRATION_KL_LIMIT:
            self._training_state_poisoned = True
            raise RuntimeError("v8 accepted calibration KL exceeds hard limit")

    def _save_numbered_checkpoint(self) -> None:
        self._conservative_boundary()
        completed = self._require_counter_coherence()
        if completed in self._v8_allowed_checkpoint_updates:
            super()._save_numbered_checkpoint()
