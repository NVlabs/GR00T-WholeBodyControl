"""Exact SONIC true-23 actor for RSL-RL 5 / MJLab.

This is not a converter for stock MJLab policies.  It instantiates the released
SONIC teleop encoder, exact FSQ bottleneck, and native 23-output decoder before
the first optimizer update.  RSL-RL therefore optimizes the deployment
topology itself instead of training an incompatible monolithic MLP.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gear_sonic.utils.g1_23dof_artifact import (
    decoder_layer_dims_for_profile,
    inspect_true23_policy_state,
)
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    checkpoint_stage,
    load_safe_true23_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import (
    DEPLOYMENT_DECODER_INPUT_DIM,
    DEPLOYMENT_HISTORY_LENGTH,
    OBS_LAYOUT_PADDED_IL29,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
    TELEOP_FSQ_LEVEL,
    TELEOP_TOKEN_COUNT,
    TELEOP_TOKEN_WIDTH,
    TOKEN_DIM,
)

_ENCODER_DIMS = (267, 2048, 1024, 512, 512, 64)
_ENCODER_PREFIX = "actor_module.encoders.teleop.module."
_DECODER_PREFIX = "actor_module.decoders.g1_dyn.module."
_ALLOWED_STAGES = {"checkpoint_initialization", "trained"}


class _ExactSiluMlp(nn.Module):
    """Sequential SiLU MLP whose state keys match released SONIC checkpoints."""

    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(dims, dims[1:])
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index + 2 != len(dims):
                layers.append(nn.SiLU())
        self.module = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(value)


class _SonicActorModule(nn.Module):
    """State-dict-compatible encoder/decoder container."""

    def __init__(self, decoder_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({"teleop": _ExactSiluMlp(_ENCODER_DIMS)})
        self.decoders = nn.ModuleDict(
            {"g1_dyn": _ExactSiluMlp(decoder_dims)}
        )


class True23SonicCore(nn.Module):
    """Exact ``267 -> FSQ64`` plus ``[FSQ64,H10-930] -> 23`` policy."""

    def __init__(self, warm_start_path: str | Path) -> None:
        super().__init__()
        self.warm_start_path = Path(warm_start_path).expanduser().resolve()
        checkpoint = load_safe_true23_checkpoint(
            self.warm_start_path,
            map_location="cpu",
        )
        stage = checkpoint_stage(checkpoint)
        if stage not in _ALLOWED_STAGES:
            raise ValueError(
                "MJLab warm start must be an initialization or genuinely "
                f"trained true23 checkpoint, got {stage!r}"
            )
        metadata = checkpoint["g1_23dof_metadata"]
        if (
            metadata.get("history_length") != DEPLOYMENT_HISTORY_LENGTH
            or metadata.get("observation_layout") != OBS_LAYOUT_PADDED_IL29
            or metadata.get("decoder_input_dim")
            != DEPLOYMENT_DECODER_INPUT_DIM
            or metadata.get("decoder_output_dim") != TARGET_DOF
        ):
            raise ValueError("MJLab warm start does not satisfy exact H10 true23 contract")

        self.reference_profile = str(metadata["reference_profile"])
        decoder_dims = decoder_layer_dims_for_profile(self.reference_profile)
        if (
            decoder_dims[0] != DEPLOYMENT_DECODER_INPUT_DIM
            or decoder_dims[-1] != TARGET_DOF
        ):
            raise ValueError("true23 decoder topology is incompatible with MJLab actor")
        self.actor_module = _SonicActorModule(decoder_dims)

        policy_state = checkpoint["policy_state_dict"]
        inspect_true23_policy_state(
            checkpoint,
            reference_profile=self.reference_profile,
        )
        network_state = {
            key: value
            for key, value in policy_state.items()
            if key.startswith((_ENCODER_PREFIX, _DECODER_PREFIX))
        }
        missing, unexpected = self.load_state_dict(network_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "true23 warm-start network keys do not exactly match MJLab core: "
                f"missing={missing}, unexpected={unexpected}"
            )
        noise_keys = [
            key
            for key in policy_state
            if key.rsplit(".", 1)[-1] in {"std", "log_std"}
        ]
        if len(noise_keys) != 1:
            raise ValueError("true23 warm start must contain exactly one action-noise tensor")
        noise_key = noise_keys[0]
        noise = policy_state[noise_key].detach().to(dtype=torch.float32, device="cpu")
        if tuple(noise.shape) != (TARGET_DOF,):
            raise ValueError("true23 warm-start action noise must have shape [23]")
        self.initial_std = (
            noise.exp() if noise_key.rsplit(".", 1)[-1] == "log_std" else noise
        )
        if not torch.isfinite(self.initial_std).all() or (self.initial_std <= 0).any():
            raise ValueError("true23 warm-start action standard deviation is invalid")

        initialization_report = checkpoint.get("g1_23dof_initialization_report")
        self.warm_start_stage = stage
        self.warm_start_metadata = copy.deepcopy(dict(metadata))
        self.warm_start_initialization_report = (
            copy.deepcopy(dict(initialization_report))
            if isinstance(initialization_report, Mapping)
            else None
        )

    @staticmethod
    def _fsq_train(latent: torch.Tensor) -> torch.Tensor:
        """Exact FSQ forward values with a straight-through rounding gradient."""
        half_l = (TELEOP_FSQ_LEVEL - 1) * (1.0 + 1.0e-3) / 2.0
        shift = math.atanh(0.5 / half_l)
        bounded = torch.tanh(latent + shift) * half_l - 0.5
        rounded_ste = bounded + (torch.round(bounded) - bounded).detach()
        return rounded_ste / (TELEOP_FSQ_LEVEL // 2)

    def encode(self, teleop_obs: torch.Tensor) -> torch.Tensor:
        if teleop_obs.shape[-1] != TELEOP_ENCODER_INPUT_DIM:
            raise ValueError(
                "true23 MJLab teleop observation must have dimension "
                f"{TELEOP_ENCODER_INPUT_DIM}, got {teleop_obs.shape[-1]}"
            )
        latent = self.actor_module.encoders["teleop"](teleop_obs)
        latent = latent.reshape(
            *teleop_obs.shape[:-1],
            TELEOP_TOKEN_COUNT,
            TELEOP_TOKEN_WIDTH,
        )
        return self._fsq_train(latent).reshape(
            *teleop_obs.shape[:-1],
            TOKEN_DIM,
        )

    def forward(
        self,
        teleop_obs: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        if proprioception.shape[-1] != (
            DEPLOYMENT_DECODER_INPUT_DIM - TOKEN_DIM
        ):
            raise ValueError(
                "true23 MJLab proprioception must have dimension 930, got "
                f"{proprioception.shape[-1]}"
            )
        token = self.encode(teleop_obs)
        decoder_input = torch.cat((token, proprioception), dim=-1)
        action = self.actor_module.decoders["g1_dyn"](decoder_input)
        if action.shape[-1] != TARGET_DOF:
            raise RuntimeError("true23 MJLab decoder did not emit exactly 23 actions")
        return action

    def export_policy_state(self, std: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the exact weights-only SONIC policy namespace."""
        if tuple(std.shape) != (TARGET_DOF,):
            raise ValueError("exported action std must have shape [23]")
        result = {
            key: value.detach().cpu().to(torch.float32).contiguous()
            for key, value in self.state_dict().items()
        }
        result["std"] = std.detach().cpu().to(torch.float32).contiguous()
        return result


class True23SonicActorModel(nn.Module):
    """RSL-RL 5 model interface backed by :class:`True23SonicCore`."""

    is_recurrent = False

    def __init__(
        self,
        obs: Any,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        warm_start_path: str,
        tokenizer_obs_group: str = "tokenizer",
        proprioception_obs_group: str = "policy",
        distribution_cfg: dict[str, Any] | None = None,
        hidden_dims: tuple[int, ...] | list[int] = (),
        activation: str = "silu",
        obs_normalization: bool = False,
    ) -> None:
        del hidden_dims, activation
        super().__init__()
        if output_dim != TARGET_DOF:
            raise ValueError(
                f"true23 MJLab actor requires output_dim=23, got {output_dim}"
            )
        if obs_set != "actor":
            raise ValueError("True23SonicActorModel may only be used as the actor")
        if obs_normalization:
            raise ValueError("true23 exact actor forbids additional observation normalization")
        active = tuple(obs_groups[obs_set])
        required = (tokenizer_obs_group, proprioception_obs_group)
        if active != required:
            raise ValueError(
                "true23 MJLab actor observation groups must be exactly "
                f"{required}, got {active}"
            )
        tokenizer_dim = int(obs[tokenizer_obs_group].shape[-1])
        proprio_dim = int(obs[proprioception_obs_group].shape[-1])
        if tokenizer_dim not in {
            TELEOP_ENCODER_INPUT_DIM,
            TELEOP_ENCODER_INPUT_DIM + 1,
        }:
            raise ValueError(
                "MJLab tokenizer group must be teleop267 or "
                f"encoder-index+teleop268, got {tokenizer_dim}"
            )
        if proprio_dim != DEPLOYMENT_DECODER_INPUT_DIM - TOKEN_DIM:
            raise ValueError(
                f"MJLab policy group must be H10-930, got {proprio_dim}"
            )
        self.tokenizer_obs_group = tokenizer_obs_group
        self.proprioception_obs_group = proprioception_obs_group
        self.tokenizer_has_encoder_index = (
            tokenizer_dim == TELEOP_ENCODER_INPUT_DIM + 1
        )
        self.core = True23SonicCore(warm_start_path)

        if distribution_cfg is None:
            distribution_cfg = {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        distribution_cfg = dict(distribution_cfg)
        class_name = distribution_cfg.pop("class_name", "GaussianDistribution")
        if class_name not in {
            "GaussianDistribution",
            "rsl_rl.modules:GaussianDistribution",
            "rsl_rl.modules.distribution:GaussianDistribution",
        }:
            raise ValueError("true23 MJLab actor requires GaussianDistribution")
        if distribution_cfg.get("std_type", "scalar") != "scalar":
            raise ValueError("true23 MJLab actor requires direct per-action std")
        from rsl_rl.modules.distribution import GaussianDistribution

        self.distribution = GaussianDistribution(
            output_dim,
            init_std=float(distribution_cfg.get("init_std", 1.0)),
            std_type="scalar",
        )
        with torch.no_grad():
            self.distribution.std_param.copy_(self.core.initial_std)

    @property
    def reference_profile(self) -> str:
        return self.core.reference_profile

    def _split_observations(self, obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = obs[self.tokenizer_obs_group]
        if self.tokenizer_has_encoder_index:
            encoder_index = tokenizer[..., :1]
            if not torch.isfinite(encoder_index).all():
                raise ValueError("MJLab true23 encoder_index contains non-finite values")
            tokenizer = tokenizer[..., 1:]
        return tokenizer, obs[self.proprioception_obs_group]

    def forward(
        self,
        obs: Any,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories

            obs = unpad_trajectories(obs, masks)
        teleop_obs, proprioception = self._split_observations(obs)
        mean = self.core(teleop_obs, proprioception)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: Any = None,
    ) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> None:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: Any) -> None:
        del obs

    def export_true23_policy_state(self) -> dict[str, torch.Tensor]:
        return self.core.export_policy_state(self.distribution.std_param)

    def as_jit(self) -> nn.Module:
        raise RuntimeError(
            "RSL monolithic JIT export is forbidden for true23; use the "
            "hash-bound SONIC encoder/decoder export pipeline"
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        raise RuntimeError(
            "RSL monolithic ONNX export is forbidden for true23; use the "
            "hash-bound SONIC encoder/decoder export pipeline"
        )
