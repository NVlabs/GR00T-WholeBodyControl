"""Checkpoint initialization for a true G1 23-DoF policy.

This module transfers compatible SONIC weights, then creates a real 23-row
action head.  It never masks a 29-row policy output at runtime.
"""

from __future__ import annotations

from typing import Any, Mapping

from gear_sonic.utils.g1_23dof_checkpoint_io import (
    build_safe_initialization_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import (
    DEFAULT_REFERENCE_PROFILE,
    NATIVE_IL23_TO_CANONICAL_IL29,
    OBS_LAYOUT_CHECKPOINT_INIT_29,
    SOURCE_DOF,
    TARGET_DOF,
    decoder_input_keep_indices_29_to_23,
    decoder_shape,
    make_artifact_metadata,
)

_REMOVED_POLICY_PREFIXES = (
    "actor_module.encoders.g1.",
    "actor_module.encoders.smpl.",
    "actor_module.decoders.g1_kin.",
)
_ALLOWED_TARGET_POLICY_PREFIXES = (
    "actor_module.encoders.teleop.",
    "actor_module.decoders.g1_dyn.",
)
_ALLOWED_TARGET_POLICY_EXACT_KEYS = {"std", "log_std"}


def _indices_like(tensor, indices):
    import torch

    return torch.tensor(indices, dtype=torch.long, device=tensor.device)


def _select(tensor, dimension: int, indices):
    return tensor.index_select(dimension, _indices_like(tensor, indices)).clone()


def initialize_policy_state_dict(
    source_state: Mapping[str, Any],
    *,
    history_length: int = 10,
    target_observation_layout: str = OBS_LAYOUT_CHECKPOINT_INIT_29,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create teleop-only true-23 policy initialization from SONIC state.

    Padded-29 warm-start keeps decoder input columns unchanged.  Native-23
    initialization selects columns by semantic observation block.  In both
    modes, decoder output head and action-distribution scale are true 23-D.
    """
    source_shape = decoder_shape(history_length, OBS_LAYOUT_CHECKPOINT_INIT_29)
    target_shape = decoder_shape(history_length, target_observation_layout)
    target_state = {
        key: value
        for key, value in source_state.items()
        if not key.startswith(_REMOVED_POLICY_PREFIXES)
    }
    unexpected_keys = sorted(
        key
        for key in target_state
        if key not in _ALLOWED_TARGET_POLICY_EXACT_KEYS
        and not key.startswith(_ALLOWED_TARGET_POLICY_PREFIXES)
    )
    if unexpected_keys:
        raise ValueError(
            "unexpected policy tensors outside the true23 transfer allowlist: "
            f"{unexpected_keys}"
        )

    decoder_weights = {
        key: value
        for key, value in target_state.items()
        if "actor_module.decoders.g1_dyn." in key
        and key.endswith(".weight")
        and getattr(value, "ndim", None) == 2
    }
    input_keys = [
        key
        for key, value in decoder_weights.items()
        if value.shape[1] == source_shape.input_dim
    ]
    if len(input_keys) != 1:
        raise ValueError(
            "expected one g1_dyn input weight with "
            f"{source_shape.input_dim} columns; got {input_keys}"
        )
    input_key = input_keys[0]
    input_weight = target_state[input_key]
    if target_shape.input_dim != source_shape.input_dim:
        keep = decoder_input_keep_indices_29_to_23(history_length)
        target_state[input_key] = _select(input_weight, 1, keep)

    output_weight_keys = [
        key
        for key, value in decoder_weights.items()
        if key != input_key and value.shape[0] == SOURCE_DOF
    ]
    if len(output_weight_keys) != 1:
        raise ValueError(
            f"expected one released {SOURCE_DOF}-row g1_dyn output weight; "
            f"got {output_weight_keys}"
        )
    output_weight_key = output_weight_keys[0]
    target_state[output_weight_key] = _select(
        target_state[output_weight_key],
        0,
        NATIVE_IL23_TO_CANONICAL_IL29,
    )

    output_bias_key = output_weight_key.removesuffix(".weight") + ".bias"
    if output_bias_key not in target_state:
        raise ValueError(f"missing decoder output bias: {output_bias_key}")
    if target_state[output_bias_key].shape[0] != SOURCE_DOF:
        raise ValueError(
            f"{output_bias_key} must be the released {SOURCE_DOF}-D bias"
        )
    target_state[output_bias_key] = _select(
        target_state[output_bias_key],
        0,
        NATIVE_IL23_TO_CANONICAL_IL29,
    )

    noise_keys = [
        key
        for key, value in target_state.items()
        if key.rsplit(".", 1)[-1] in {"std", "log_std"}
        and getattr(value, "ndim", None) == 1
        and value.shape[0] == SOURCE_DOF
    ]
    if len(noise_keys) != 1:
        raise ValueError(
            f"expected one released {SOURCE_DOF}-D std or log_std tensor; "
            f"got {noise_keys}"
        )
    noise_key = noise_keys[0]
    target_state[noise_key] = _select(
        target_state[noise_key],
        0,
        NATIVE_IL23_TO_CANONICAL_IL29,
    )

    report = {
        "source_decoder_input_dim": source_shape.input_dim,
        "target_decoder_input_dim": target_shape.input_dim,
        "target_decoder_output_dim": TARGET_DOF,
        "input_weight_key": input_key,
        "output_weight_key": output_weight_key,
        "noise_key": noise_key,
        "removed_policy_prefixes": list(_REMOVED_POLICY_PREFIXES),
        "optimizer_reused": False,
        "value_model_reused": False,
        "initialization_only": True,
    }
    return target_state, report


def initialize_checkpoint(
    source_checkpoint: Mapping[str, Any],
    *,
    history_length: int = 10,
    target_observation_layout: str = OBS_LAYOUT_CHECKPOINT_INIT_29,
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
    source_checkpoint_sha256: str | None = None,
    source_revision: str | None = None,
    source_family: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return checkpoint safe to load as weights-only initialization."""
    if "policy_state_dict" not in source_checkpoint:
        raise ValueError("checkpoint has no policy_state_dict")
    policy, report = initialize_policy_state_dict(
        source_checkpoint["policy_state_dict"],
        history_length=history_length,
        target_observation_layout=target_observation_layout,
    )
    # Construct a new mapping. Copying the source would preserve trainer
    # ``state``/``args`` and unknown future resume state, defeating a
    # weights-only initialization.
    report = {
        **report,
        "reference_profile": reference_profile,
    }
    if source_checkpoint_sha256 is not None:
        report["source_checkpoint_sha256"] = source_checkpoint_sha256
    if source_revision is not None:
        report["source_revision"] = source_revision
    if source_family is not None:
        report["source_family"] = source_family
    checkpoint = build_safe_initialization_checkpoint(
        policy_state_dict=policy,
        metadata=make_artifact_metadata(
            history_length=history_length,
            observation_layout=target_observation_layout,
            checkpoint_stage="checkpoint_initialization",
            reference_profile=reference_profile,
        ),
        initialization_report=report,
    )
    return checkpoint, report
