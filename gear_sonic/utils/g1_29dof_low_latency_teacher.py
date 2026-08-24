"""Exact deterministic runtime for released low-latency 29-DoF SONIC teacher.

Only exact hash-pinned ``low_latency/last.pt`` is accepted.  Checkpoint loading
uses existing audited legacy-release boundary; arbitrary checkpoint paths never
reach ``torch.load(..., weights_only=False)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from gear_sonic.scripts.init_g1_23dof_checkpoint import (
    _load_pinned_legacy_release,
)
from gear_sonic.utils.g1_23dof_contract import (
    LOW_LATENCY_RELEASE_HF_REVISION,
    LOW_LATENCY_RELEASE_SHA256,
    REFERENCE_PROFILE_LOW_LATENCY,
    SOURCE_IL29_JOINT_NAMES,
)

SEMANTIC_INPUT_DIM = 267
PROPRIO_INPUT_DIM = 930
TOKEN_DIM = 64
ACTION_DIM = 29
FSQ_TOKEN_SHAPE = (2, 32)
FSQ_LEVEL = 32

ENCODER_DIMS = (267, 2048, 1024, 512, 512, 64)
DECODER_DIMS = (994, 4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 29)

_ENCODER_PREFIX = "actor_module.encoders.teleop.module."
_DECODER_PREFIX = "actor_module.decoders.g1_dyn.module."

Layer = tuple[torch.Tensor, torch.Tensor]


def exact_fsq32(latent: torch.Tensor) -> torch.Tensor:
    """Return exact forward values of released two-token, 32-level FSQ."""

    if not isinstance(latent, torch.Tensor):
        raise TypeError("FSQ latent must be a torch.Tensor")
    if latent.dtype != torch.float32:
        raise ValueError("FSQ latent must use float32")
    if latent.ndim < 1 or latent.shape[-1] != TOKEN_DIM:
        raise ValueError(f"FSQ latent must end in dimension {TOKEN_DIM}")
    if not torch.isfinite(latent).all():
        raise ValueError("FSQ latent contains non-finite values")

    value = latent.reshape(*latent.shape[:-1], *FSQ_TOKEN_SHAPE)
    half_l = (FSQ_LEVEL - 1) * (1.0 + 1.0e-3) / 2.0
    shift = math.atanh(0.5 / half_l)
    bounded = torch.tanh(value + shift) * half_l - 0.5
    return torch.round(bounded).div(FSQ_LEVEL // 2).reshape(*latent.shape)


def _expected_component_keys(prefix: str, dims: Sequence[int]) -> set[str]:
    return {
        f"{prefix}{2 * layer_index}.{parameter}"
        for layer_index in range(len(dims) - 1)
        for parameter in ("weight", "bias")
    }


def _extract_component(
    policy_state: Mapping[str, Any],
    *,
    prefix: str,
    dims: Sequence[int],
    device: torch.device,
    context: str,
) -> tuple[Layer, ...]:
    """Validate one exact SiLU MLP namespace and retain only inference tensors."""

    expected_keys = _expected_component_keys(prefix, dims)
    actual_keys = {key for key in policy_state if key.startswith(prefix)}
    if actual_keys != expected_keys:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    layers: list[Layer] = []
    for layer_index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        module_index = 2 * layer_index
        weight = policy_state[f"{prefix}{module_index}.weight"]
        bias = policy_state[f"{prefix}{module_index}.bias"]
        if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
            raise ValueError(f"{context} layer {module_index} parameters must be tensors")
        if weight.dtype != torch.float32 or bias.dtype != torch.float32:
            raise ValueError(f"{context} layer {module_index} parameters must use float32")
        if tuple(weight.shape) != (output_dim, input_dim):
            raise ValueError(
                f"{context} layer {module_index} weight shape must be "
                f"{(output_dim, input_dim)}, got {tuple(weight.shape)}"
            )
        if tuple(bias.shape) != (output_dim,):
            raise ValueError(
                f"{context} layer {module_index} bias shape must be {(output_dim,)}, got {tuple(bias.shape)}"
            )
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError(f"{context} layer {module_index} contains non-finite values")
        layers.append(
            (
                weight.detach().to(device=device),
                bias.detach().to(device=device),
            )
        )
    return tuple(layers)


def _resolve_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("teacher device must be cpu or cuda[:index]") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("teacher device must be cpu or cuda[:index]")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _numpy_float32_matrix(value: Any, width: int, context: str) -> tuple[np.ndarray, bool]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{context} must be a numpy.ndarray")
    if value.dtype != np.float32:
        raise ValueError(f"{context} must use float32")
    squeezed = value.ndim == 1
    if squeezed:
        if value.shape != (width,):
            raise ValueError(f"{context} must have shape [{width}] or [batch,{width}]")
        value = value.reshape(1, width)
    elif value.ndim != 2 or value.shape[1] != width or value.shape[0] < 1:
        raise ValueError(f"{context} must have shape [{width}] or [batch,{width}]")
    if not np.isfinite(value).all():
        raise ValueError(f"{context} contains non-finite values")
    return np.ascontiguousarray(value), squeezed


class LowLatency29DoFTeacher:
    """Pinned encoder + FSQ + decoder emitting deterministic native IL29 means."""

    action_joint_names = SOURCE_IL29_JOINT_NAMES

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = _resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        checkpoint, checkpoint_sha256, release = _load_pinned_legacy_release(self.checkpoint_path)
        if checkpoint_sha256 != LOW_LATENCY_RELEASE_SHA256:
            raise ValueError("teacher checkpoint is not pinned low_latency/last.pt")
        if release.get("source_revision") != LOW_LATENCY_RELEASE_HF_REVISION:
            raise ValueError("teacher release revision differs from pinned low-latency release")
        if release.get("reference_profile") != REFERENCE_PROFILE_LOW_LATENCY:
            raise ValueError("teacher checkpoint does not use released low-latency profile")

        policy_state = checkpoint.get("policy_state_dict")
        if not isinstance(policy_state, Mapping) or not policy_state:
            raise ValueError("teacher checkpoint lacks non-empty policy_state_dict")
        self._encoder = _extract_component(
            policy_state,
            prefix=_ENCODER_PREFIX,
            dims=ENCODER_DIMS,
            device=self.device,
            context="low-latency teleop encoder",
        )
        self._decoder = _extract_component(
            policy_state,
            prefix=_DECODER_PREFIX,
            dims=DECODER_DIMS,
            device=self.device,
            context="low-latency g1_dyn decoder",
        )
        self.checkpoint_sha256 = checkpoint_sha256

        # Full trainer resume carries optimizer/critic state.  Drop it as soon as
        # exact actor tensors have been retained (CPU tensors share storage).
        del policy_state, checkpoint
        gc.collect()

    @staticmethod
    def _mlp(layers: Sequence[Layer], value: torch.Tensor) -> torch.Tensor:
        for index, (weight, bias) in enumerate(layers):
            value = F.linear(value, weight, bias)
            if index + 1 != len(layers):
                value = F.silu(value)
        return value

    def infer_batch(self, semantic267: np.ndarray, proprio930: np.ndarray) -> np.ndarray:
        """Infer native IL29 action means for float32 batches."""

        semantic, _ = _numpy_float32_matrix(semantic267, SEMANTIC_INPUT_DIM, "semantic267")
        proprio, _ = _numpy_float32_matrix(proprio930, PROPRIO_INPUT_DIM, "proprio930")
        if semantic.shape[0] != proprio.shape[0]:
            raise ValueError("semantic267 and proprio930 batch sizes differ")

        with torch.inference_mode():
            semantic_tensor = torch.from_numpy(semantic).to(device=self.device)
            proprio_tensor = torch.from_numpy(proprio).to(device=self.device)
            latent = self._mlp(self._encoder, semantic_tensor)
            token = exact_fsq32(latent)
            decoder_input = torch.cat((token, proprio_tensor), dim=-1)
            action = self._mlp(self._decoder, decoder_input)

        expected_shape = (semantic.shape[0], ACTION_DIM)
        if action.dtype != torch.float32 or tuple(action.shape) != expected_shape:
            raise RuntimeError(
                f"low-latency teacher output must be float32 {expected_shape}, "
                f"got {action.dtype} {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all():
            raise RuntimeError("low-latency teacher emitted non-finite action")
        return np.ascontiguousarray(action.detach().cpu().numpy(), dtype=np.float32)

    def infer(self, semantic267: np.ndarray, proprio930: np.ndarray) -> np.ndarray:
        """Infer one native IL29 action mean from ``[267]`` and ``[930]``."""

        semantic, semantic_squeezed = _numpy_float32_matrix(semantic267, SEMANTIC_INPUT_DIM, "semantic267")
        proprio, proprio_squeezed = _numpy_float32_matrix(proprio930, PROPRIO_INPUT_DIM, "proprio930")
        if not semantic_squeezed or not proprio_squeezed:
            raise ValueError("infer requires unbatched semantic267[267] and proprio930[930]")
        return self.infer_batch(semantic, proprio)[0]

    def descriptor(self) -> dict[str, Any]:
        """Return immutable inference semantics suitable for evidence manifests."""

        return {
            "controller_semantics": "neural_state_feedback_29dof_low_latency_policy",
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_revision": LOW_LATENCY_RELEASE_HF_REVISION,
            "reference_profile": REFERENCE_PROFILE_LOW_LATENCY,
            "semantic_input_dim": SEMANTIC_INPUT_DIM,
            "proprio_input_dim": PROPRIO_INPUT_DIM,
            "encoder_dims": list(ENCODER_DIMS),
            "quantizer": "fsq_tanh_round_even_levels_32_v1",
            "token_shape": list(FSQ_TOKEN_SHAPE),
            "decoder_dims": list(DECODER_DIMS),
            "activation": "SiLU",
            "output_space": "native_il29_action",
            "action_joint_names": list(self.action_joint_names),
            "action_distribution": "deterministic_mean",
            "dtype": "float32",
            "device": str(self.device),
            "torch_version": torch.__version__,
        }


# Compatibility name used by Step1B runner discovery work.
ExactLowLatencyTeacher = LowLatency29DoFTeacher


__all__ = [
    "ACTION_DIM",
    "DECODER_DIMS",
    "ENCODER_DIMS",
    "ExactLowLatencyTeacher",
    "LowLatency29DoFTeacher",
    "PROPRIO_INPUT_DIM",
    "SEMANTIC_INPUT_DIM",
    "exact_fsq32",
]
