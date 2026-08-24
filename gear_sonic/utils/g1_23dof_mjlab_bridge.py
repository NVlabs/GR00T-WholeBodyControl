"""Fail-closed audit boundary for MJLab/RSL-RL true23 checkpoints.

Stock MJLab checkpoints are useful teachers and training records.  They are
not SONIC promotion checkpoints: the stock RSL-RL G1 actor is a normalized
single-input MLP, while SONIC deployment requires an exact teleop encoder,
FSQ token contract, H10 proprioception history, and true23 decoder.

This module never relabels RSL-RL weights or fabricates SONIC training
evidence.  It only loads tensor-safe checkpoint content, reports structural
compatibility, and explains what a custom training-time runner must prove.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    inspect_true23_policy_state,
    sha256_file,
)
from gear_sonic.utils.g1_23dof_contract import (
    DEFAULT_REFERENCE_PROFILE,
    DEPLOYMENT_DECODER_INPUT_DIM,
    DEPLOYMENT_HISTORY_LENGTH,
    HARDWARE_JOINT_IDS,
    MINIMUM_TRAINING_UPDATES,
    OBS_LAYOUT_PADDED_IL29,
    REFERENCE_PROFILE_LOW_LATENCY,
    REFERENCE_PROFILE_NORMAL,
    REFERENCE_PROFILES,
    ROBOT_MODEL,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
    TOKEN_DIM,
    reference_profile_contract,
)

AUDIT_SCHEMA_VERSION = 1
AUDIT_KIND = "g1_true23_mjlab_rsl_rl_checkpoint_audit"
BRIDGE_DECISION = "blocked_posthoc_conversion"

_ENCODER_DIMS = (TELEOP_ENCODER_INPUT_DIM, 2048, 1024, 512, 512, TOKEN_DIM)
_DECODER_DIMS = {
    REFERENCE_PROFILE_NORMAL: (
        DEPLOYMENT_DECODER_INPUT_DIM,
        2048,
        2048,
        1024,
        1024,
        512,
        512,
        TARGET_DOF,
    ),
    REFERENCE_PROFILE_LOW_LATENCY: (
        DEPLOYMENT_DECODER_INPUT_DIM,
        4096,
        4096,
        2048,
        2048,
        1024,
        1024,
        512,
        512,
        TARGET_DOF,
    ),
}


def _safe_load(path: Path) -> Mapping[str, Any]:
    if path.is_symlink():
        raise ValueError(f"MJLab checkpoint must not be a symlink: {path}")
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"MJLab checkpoint must be a regular file: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            "MJLab checkpoint is not tensor-safe; audit refuses arbitrary "
            "pickle objects"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("MJLab checkpoint root must be a mapping")
    if any(type(key) is not str or not key for key in value):
        raise ValueError(
            "MJLab checkpoint root keys must be non-empty strings"
        )
    return value


def _tensor_mapping(value: Any, context: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{context} must be a non-empty tensor mapping")
    result: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if type(key) is not str or not key:
            raise ValueError(f"{context} keys must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{context}[{key!r}] must be a tensor")
        result[key] = tensor.detach().cpu().contiguous()
    return result


def _extract_actor_state(
    checkpoint: Mapping[str, Any],
) -> tuple[str, dict[str, torch.Tensor]]:
    has_current = "actor_state_dict" in checkpoint
    has_legacy = "model_state_dict" in checkpoint
    if has_current and has_legacy:
        raise ValueError(
            "ambiguous RSL-RL checkpoint contains current and legacy actor roots"
        )
    if has_current:
        return (
            "rsl_rl_v5_actor_state_dict",
            _tensor_mapping(
                checkpoint["actor_state_dict"],
                "actor_state_dict",
            ),
        )
    if not has_legacy:
        raise ValueError(
            "checkpoint has neither actor_state_dict nor legacy model_state_dict"
        )
    combined = _tensor_mapping(
        checkpoint["model_state_dict"],
        "model_state_dict",
    )
    actor: dict[str, torch.Tensor] = {}
    for key, tensor in combined.items():
        if key.startswith("actor."):
            actor[key.removeprefix("actor.")] = tensor
        elif key.startswith("actor_obs_normalizer."):
            actor[
                "obs_normalizer."
                + key.removeprefix("actor_obs_normalizer.")
            ] = tensor
        elif key in {"std", "log_std"}:
            actor[key] = tensor
    if not actor:
        raise ValueError("legacy model_state_dict contains no actor tensors")
    return "rsl_rl_legacy_model_state_dict", actor


def _tensor_manifest(
    tensors: Mapping[str, torch.Tensor],
) -> tuple[str, list[dict[str, Any]], int]:
    digest = hashlib.sha256()
    descriptors: list[dict[str, Any]] = []
    total_parameters = 0
    for key in sorted(tensors):
        tensor = tensors[key].detach().cpu().contiguous()
        shape = list(tensor.shape)
        count = tensor.numel()
        total_parameters += count
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(shape))
        digest.update(tensor.numpy().tobytes(order="C"))
        descriptors.append(
            {
                "key": key,
                "dtype": str(tensor.dtype),
                "shape": shape,
                "parameter_count": count,
            }
        )
    return digest.hexdigest(), descriptors, total_parameters


def _iteration(checkpoint: Mapping[str, Any]) -> int | None:
    value = checkpoint.get("iter")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _target_contract(reference_profile: str) -> dict[str, Any]:
    if reference_profile not in REFERENCE_PROFILES:
        raise ValueError(
            f"unsupported true23 reference profile: {reference_profile!r}"
        )
    return {
        "robot_model": ROBOT_MODEL,
        "history_length": DEPLOYMENT_HISTORY_LENGTH,
        "observation_layout": OBS_LAYOUT_PADDED_IL29,
        "hardware_joint_ids": list(HARDWARE_JOINT_IDS),
        "teleop_encoder": {
            "input_dim": TELEOP_ENCODER_INPUT_DIM,
            "layer_dims": list(_ENCODER_DIMS),
            "activation": "SiLU",
            "output_dim": TOKEN_DIM,
            "quantizer": "fsq_tanh_round_ste_even_levels_v1",
        },
        "decoder": {
            "input_dim": DEPLOYMENT_DECODER_INPUT_DIM,
            "layer_dims": list(_DECODER_DIMS[reference_profile]),
            "activation": "SiLU",
            "output_dim": TARGET_DOF,
        },
        "reference_profile": reference_profile,
        "reference_contract": reference_profile_contract(reference_profile),
        "onnx_contract": {
            "artifact_count": 2,
            "encoder_input_dim": TELEOP_ENCODER_INPUT_DIM,
            "decoder_input_dim": DEPLOYMENT_DECODER_INPUT_DIM,
            "decoder_output_dim": TARGET_DOF,
            "static_float32": True,
            "opset": 13,
        },
    }


def audit_mjlab_rsl_rl_checkpoint(
    checkpoint_path: str | Path,
    *,
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
) -> dict[str, Any]:
    """Inspect architecture safely; never return promotion authorization."""

    path = Path(checkpoint_path).resolve()
    checkpoint = _safe_load(path)
    checkpoint_format, actor_state = _extract_actor_state(checkpoint)
    actor_hash, tensor_descriptors, parameter_count = _tensor_manifest(
        actor_state
    )
    iteration = _iteration(checkpoint)
    architecture_error = None
    try:
        target_policy_hash = inspect_true23_policy_state(
            {"policy_state_dict": actor_state},
            reference_profile=reference_profile,
        )
        architecture_compatible = True
    except ValueError as exc:
        target_policy_hash = None
        architecture_compatible = False
        architecture_error = str(exc)

    stock_rsl_shape = any(
        key.startswith(("mlp.", "obs_normalizer.", "distribution."))
        for key in actor_state
    )
    iteration_threshold_met = (
        iteration is not None and iteration >= MINIMUM_TRAINING_UPDATES
    )
    findings = []
    if stock_rsl_shape:
        findings.append(
            "actor uses stock RSL-RL monolithic MLP/normalizer namespaces"
        )
    if not architecture_compatible:
        findings.append(
            "actor cannot reconstruct the exact SONIC encoder/decoder pair"
        )
    if not iteration_threshold_met:
        findings.append(
            f"checkpoint iter does not reach {MINIMUM_TRAINING_UPDATES}"
        )
    findings.extend(
        (
            "RSL-RL iter is not independently verified optimizer-update lineage",
            "checkpoint does not bind approved warm-start and initial policy hash",
            "checkpoint does not bind exact motion dataset material",
            "checkpoint does not bind approved resolved training config/source/assets",
            "joint and observation semantics cannot be inferred from tensor shapes",
        )
    )

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "conversion_permitted": False,
        "bridge_decision": BRIDGE_DECISION,
        "source_checkpoint": {
            "filename": path.name,
            "sha256": sha256_file(path),
            "safe_weights_only_load": True,
            "format": checkpoint_format,
            "root_keys": sorted(str(key) for key in checkpoint),
            "iteration": iteration,
        },
        "actor_state": {
            "sha256": actor_hash,
            "tensor_count": len(actor_state),
            "parameter_count": parameter_count,
            "tensors": tensor_descriptors,
            "stock_rsl_namespace_detected": stock_rsl_shape,
        },
        "target_contract": _target_contract(reference_profile),
        "compatibility": {
            "architecture_compatible": architecture_compatible,
            "architecture_error": architecture_error,
            "target_policy_state_sha256": target_policy_hash,
            "iteration_threshold_met": iteration_threshold_met,
            "training_lineage_verified": False,
            "observation_semantics_verified": False,
            "action_semantics_verified": False,
            "onnx_pair_semantics_verified": False,
        },
        "findings": findings,
        "truthful_bridge_boundary": {
            "stock_checkpoint_role": "teacher_or_diagnostic_only",
            "posthoc_relabeling_forbidden": True,
            "required_training_path": (
                "custom exact-policy MJLab/RSL-RL runner instrumented before "
                "training starts"
            ),
            "required_evidence": [
                "exact SONIC encoder/FSQ/H10/true23 decoder architecture",
                "approved source release and initial policy state hash",
                "at least 50 independently recorded optimizer updates",
                "final policy state hash different from initialization",
                "exact resolved MJLab task and RSL-RL training config",
                "pinned MJLab, RSL-RL, task, asset, and runner source hashes",
                "exact motion dataset archive and processed manifest",
                "explicit 23-joint observation and action ordering semantics",
                "new externally-trained evidence schema and checked-in approval",
                "candidate ONNX parity followed by exact MuJoCo promotion replay",
            ],
        },
        "promotion_pt_written": False,
    }


def write_mjlab_audit_report(
    report: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Atomically write one new diagnostic report; refuse overwrite."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(report)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite audit report: {output}"
            )
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output
