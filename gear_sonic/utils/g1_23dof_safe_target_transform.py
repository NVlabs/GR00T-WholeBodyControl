"""Exact differentiable safe-target transform for true23 deployment.

The transform is intentionally shared by MJLab training, ONNX export, and
MuJoCo/deployment validation.  Policy actions enter and leave in native
IsaacLab-23 order; joint-position target calculations use hardware order.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch

from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
    TARGET_DOF,
)

SAFE_TARGET_TRANSFORM_KIND = "asymmetric_zero_preserving_tanh_raw_clip_v2"
SAFE_TARGET_RAW_ACTION_CLIP = 10.0
SAFE_TARGET_SOFT_LIMIT_FACTOR = 0.9
SAFE_TARGET_MARGIN_RAD = 0.012
SAFE_TARGET_ENCODER_BIAS_ENVELOPE_RAD = 0.01
SAFE_TARGET_GUARANTEED_GUARD_RAD = 0.0019
SAFE_TARGET_NOMINAL_GUARD_RAD = (
    SAFE_TARGET_MARGIN_RAD - SAFE_TARGET_ENCODER_BIAS_ENVELOPE_RAD
)

# KNEES_BENT_KEYFRAME, hardware/MuJoCo order.
SAFE_TARGET_DEFAULT_Q_HARDWARE = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.2, 0.2, 0.0, 0.6, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0,
)

# URDF/MJCF hard joint ranges, hardware/MuJoCo order.
SAFE_TARGET_HARD_LOWER_HARDWARE = (
    -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.618, -3.0892, -1.5882, -2.618, -1.0472, -1.97222,
    -3.0892, -2.2515, -2.618, -1.0472, -1.97222,
)
SAFE_TARGET_HARD_UPPER_HARDWARE = (
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
    2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
    2.618, 2.6704, 2.2515, 2.618, 2.0944, 1.97222,
    2.6704, 1.5882, 2.618, 2.0944, 1.97222,
)


def _soft_limits() -> tuple[tuple[float, ...], tuple[float, ...]]:
    low: list[float] = []
    high: list[float] = []
    for hard_low, hard_high in zip(
        SAFE_TARGET_HARD_LOWER_HARDWARE,
        SAFE_TARGET_HARD_UPPER_HARDWARE,
        strict=True,
    ):
        midpoint = (hard_low + hard_high) * 0.5
        half = (hard_high - hard_low) * SAFE_TARGET_SOFT_LIMIT_FACTOR * 0.5
        low.append(midpoint - half)
        high.append(midpoint + half)
    return tuple(low), tuple(high)


SAFE_TARGET_SOFT_LOWER_HARDWARE, SAFE_TARGET_SOFT_UPPER_HARDWARE = _soft_limits()
SAFE_TARGET_INNER_LOWER_HARDWARE = tuple(
    value + SAFE_TARGET_MARGIN_RAD for value in SAFE_TARGET_SOFT_LOWER_HARDWARE
)
SAFE_TARGET_INNER_UPPER_HARDWARE = tuple(
    value - SAFE_TARGET_MARGIN_RAD for value in SAFE_TARGET_SOFT_UPPER_HARDWARE
)
SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE = tuple(
    high - default
    for high, default in zip(
        SAFE_TARGET_INNER_UPPER_HARDWARE,
        SAFE_TARGET_DEFAULT_Q_HARDWARE,
        strict=True,
    )
)
SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE = tuple(
    default - low
    for default, low in zip(
        SAFE_TARGET_DEFAULT_Q_HARDWARE,
        SAFE_TARGET_INNER_LOWER_HARDWARE,
        strict=True,
    )
)

if not all(value > 0.0 for value in SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE):
    raise AssertionError("true23 default exceeds positive safe-target capacity")
if not all(value > 0.0 for value in SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE):
    raise AssertionError("true23 default exceeds negative safe-target capacity")
if SAFE_TARGET_NOMINAL_GUARD_RAD < SAFE_TARGET_GUARANTEED_GUARD_RAD:
    raise AssertionError("safe-target guard is below its conservative guarantee")

SAFE_TARGET_FORMULA = (
    f"clipped_native=clip(raw_native,-{SAFE_TARGET_RAW_ACTION_CLIP:g},"
    f"{SAFE_TARGET_RAW_ACTION_CLIP:g});"
    "hw=clipped_native[index(ISAACLAB_TO_MUJOCO_DOF)];d=hw*scale;"
    "d_safe=where(d>=0,p*tanh(d/p),n*tanh(d/n));"
    "p=soft_hi-margin-default;n=default-(soft_lo+margin);"
    "target_unbiased=default+d_safe;safe_hw=d_safe/scale;"
    "safe_native=safe_hw[index(MUJOCO_TO_ISAACLAB_DOF)]"
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SAFE_TARGET_FORMULA_SHA256 = hashlib.sha256(SAFE_TARGET_FORMULA.encode("utf-8")).hexdigest()
SAFE_TARGET_CONSTANTS_SHA256 = _canonical_sha256(
    {
        "action_scale_hardware": HARDWARE_23_ACTION_SCALE,
        "default_q_hardware": SAFE_TARGET_DEFAULT_Q_HARDWARE,
        "encoder_bias_envelope_rad": SAFE_TARGET_ENCODER_BIAS_ENVELOPE_RAD,
        "guaranteed_guard_rad": SAFE_TARGET_GUARANTEED_GUARD_RAD,
        "nominal_guard_rad": SAFE_TARGET_NOMINAL_GUARD_RAD,
        "raw_action_clip": SAFE_TARGET_RAW_ACTION_CLIP,
        "hard_lower_hardware": SAFE_TARGET_HARD_LOWER_HARDWARE,
        "hard_upper_hardware": SAFE_TARGET_HARD_UPPER_HARDWARE,
        "isaaclab_to_mujoco": ISAACLAB_TO_MUJOCO_DOF,
        "margin_rad": SAFE_TARGET_MARGIN_RAD,
        "mujoco_to_isaaclab": MUJOCO_TO_ISAACLAB_DOF,
        "soft_limit_factor": SAFE_TARGET_SOFT_LIMIT_FACTOR,
    }
)


def safe_target_transform_torch(
    raw_native_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(safe_native_action, unbiased_hardware_target)`` in float32."""

    if raw_native_action.shape[-1] != TARGET_DOF:
        raise ValueError("safe-target input must have last dimension 23")
    if raw_native_action.dtype != torch.float32:
        raise ValueError("safe-target input must be float32")
    device = raw_native_action.device
    hw_indices = torch.as_tensor(
        ISAACLAB_TO_MUJOCO_DOF, dtype=torch.long, device=device
    )
    native_indices = torch.as_tensor(
        MUJOCO_TO_ISAACLAB_DOF, dtype=torch.long, device=device
    )
    clipped_native = torch.clamp(
        raw_native_action,
        min=-SAFE_TARGET_RAW_ACTION_CLIP,
        max=SAFE_TARGET_RAW_ACTION_CLIP,
    )
    raw_hardware = clipped_native.index_select(-1, hw_indices)
    scale = torch.as_tensor(
        HARDWARE_23_ACTION_SCALE, dtype=torch.float32, device=device
    )
    default = torch.as_tensor(
        SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=torch.float32, device=device
    )
    positive = torch.as_tensor(
        SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
        dtype=torch.float32,
        device=device,
    )
    negative = torch.as_tensor(
        SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
        dtype=torch.float32,
        device=device,
    )
    delta = raw_hardware * scale
    safe_delta = torch.where(
        delta >= 0.0,
        positive * torch.tanh(delta / positive),
        negative * torch.tanh(delta / negative),
    )
    target_unbiased = default + safe_delta
    safe_hardware = safe_delta / scale
    safe_native = safe_hardware.index_select(-1, native_indices)
    return safe_native, target_unbiased


def safe_target_transform_numpy(
    raw_native_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy float32 implementation with the identical ordered operations."""

    raw = np.asarray(raw_native_action)
    if raw.shape[-1] != TARGET_DOF:
        raise ValueError("safe-target input must have last dimension 23")
    if raw.dtype != np.float32:
        raise ValueError("safe-target input must be float32")
    clipped_native = np.clip(
        raw,
        np.float32(-SAFE_TARGET_RAW_ACTION_CLIP),
        np.float32(SAFE_TARGET_RAW_ACTION_CLIP),
    )
    raw_hardware = clipped_native[
        ..., np.asarray(ISAACLAB_TO_MUJOCO_DOF, dtype=np.int64)
    ]
    scale = np.asarray(HARDWARE_23_ACTION_SCALE, dtype=np.float32)
    default = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32)
    positive = np.asarray(SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE, dtype=np.float32)
    negative = np.asarray(SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE, dtype=np.float32)
    delta = raw_hardware * scale
    safe_delta = np.where(
        delta >= np.float32(0.0),
        positive * np.tanh(delta / positive),
        negative * np.tanh(delta / negative),
    ).astype(np.float32, copy=False)
    target_unbiased = default + safe_delta
    safe_hardware = safe_delta / scale
    safe_native = safe_hardware[..., np.asarray(MUJOCO_TO_ISAACLAB_DOF, dtype=np.int64)]
    return safe_native.astype(np.float32, copy=False), target_unbiased.astype(np.float32, copy=False)


def safe_target_transform_contract() -> dict[str, Any]:
    """Serializable contract bound into training/export evidence."""

    return {
        "schema": "g1_true23_safe_target_transform_v2",
        "kind": SAFE_TARGET_TRANSFORM_KIND,
        "float_dtype": "float32",
        "input_order": "native_isaaclab_23",
        "output_action_order": "native_isaaclab_23",
        "target_order": "hardware_mujoco_23",
        "raw_action_clip": SAFE_TARGET_RAW_ACTION_CLIP,
        "raw_action_clip_order": "native_isaaclab_23_before_permutation",
        "soft_limit_factor": SAFE_TARGET_SOFT_LIMIT_FACTOR,
        "margin_rad": SAFE_TARGET_MARGIN_RAD,
        "encoder_bias_envelope_rad": SAFE_TARGET_ENCODER_BIAS_ENVELOPE_RAD,
        "guaranteed_post_bias_guard_rad": SAFE_TARGET_GUARANTEED_GUARD_RAD,
        "nominal_post_bias_guard_rad": SAFE_TARGET_NOMINAL_GUARD_RAD,
        "default_q_hardware": list(SAFE_TARGET_DEFAULT_Q_HARDWARE),
        "action_scale_hardware": list(HARDWARE_23_ACTION_SCALE),
        "hard_lower_hardware": list(SAFE_TARGET_HARD_LOWER_HARDWARE),
        "hard_upper_hardware": list(SAFE_TARGET_HARD_UPPER_HARDWARE),
        "soft_lower_hardware": list(SAFE_TARGET_SOFT_LOWER_HARDWARE),
        "soft_upper_hardware": list(SAFE_TARGET_SOFT_UPPER_HARDWARE),
        "inner_lower_hardware": list(SAFE_TARGET_INNER_LOWER_HARDWARE),
        "inner_upper_hardware": list(SAFE_TARGET_INNER_UPPER_HARDWARE),
        "isaaclab_to_mujoco": list(ISAACLAB_TO_MUJOCO_DOF),
        "mujoco_to_isaaclab": list(MUJOCO_TO_ISAACLAB_DOF),
        "previous_action_semantics": "applied_safe_native_action",
        "constants_sha256": SAFE_TARGET_CONSTANTS_SHA256,
        "formula": SAFE_TARGET_FORMULA,
        "formula_sha256": SAFE_TARGET_FORMULA_SHA256,
    }
