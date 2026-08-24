"""Fail-closed diagnostic ONNX export from exact true23 MJLab resumes.

This module deliberately creates neither a promotion artifact nor a deployment
artifact.  Its outputs are limited to MuJoCo and Pico shadow diagnostics and
cannot authorize active motor control.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import warnings

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_PROFILE,
    causal_history_profile_contract,
)
from gear_sonic.utils import g1_23dof_artifact as artifact
from gear_sonic.utils.g1_23dof_contract import (
    DEPLOYMENT_DECODER_INPUT_DIM,
    DEPLOYMENT_HISTORY_LENGTH,
    OBS_LAYOUT_PADDED_IL29,
    REFERENCE_PROFILE_LOW_LATENCY,
    REQUIRED_MODE_MACHINE,
    ROBOT_MODEL,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
    TOKEN_DIM,
)
from gear_sonic.utils.g1_23dof_mjlab_training import (
    MJLAB_CHECKPOINT_ROLE,
    load_mjlab_training_checkpoint,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    safe_target_transform_contract,
    safe_target_transform_torch,
)

DIAGNOSTIC_SCHEMA_VERSION = 1
DIAGNOSTIC_SAFE_TARGET_SCHEMA_VERSION = 2
DIAGNOSTIC_BUNDLE_KIND = "g1_true23_mjlab_diagnostic_onnx_pair"
DIAGNOSTIC_EMBEDDED_KIND = "g1_true23_mjlab_diagnostic_onnx"
DIAGNOSTIC_ALLOWED_USES = (
    "mujoco_sim2sim_diagnostic",
    "pico_shadow_diagnostic",
)
DIAGNOSTIC_FORBIDDEN_USES = (
    "active_motor_control",
    "deployment",
    "promotion",
)

_ENCODER_PREFIX = "actor_module.encoders.teleop.module."
_DECODER_PREFIX = "actor_module.decoders.g1_dyn.module."
_SAFETY_FLAGS = {
    "diagnostic_only": True,
    "deployment_ready": False,
    "promotion_eligible": False,
    "active_motor_control_authorized": False,
}
_BUNDLE_KEYS = {
    "schema_version",
    "kind",
    *_SAFETY_FLAGS,
    "checkpoint_role",
    "allowed_uses",
    "forbidden_uses",
    "no_robot_or_network_commands_performed",
    "source",
    "contract",
    "artifacts",
    "hashes",
    "validation",
    "metadata_payload_sha256",
}
_SOURCE_KEYS = {
    "checkpoint_filename",
    "checkpoint_update_count",
    "reference_profile",
    "simulation_candidate_review_allowed",
}
_CONTRACT_KEYS = {
    "robot_model",
    "required_mode_machine",
    "native_action_dof",
    "history_length",
    "observation_layout",
    "onnx_opset",
    "encoder",
    "decoder",
}
_SAFE_TARGET_CONTRACT_KEYS = _CONTRACT_KEYS | {
    "decoder_output_semantics",
    "external_safe_target_transform_allowed",
    "previous_action_semantics",
    "safe_target_transform",
}
_GRAPH_KEYS = {
    "input_name",
    "input_shape",
    "input_dtype",
    "output_name",
    "output_shape",
    "output_dtype",
    "dynamic_axes",
}
_ARTIFACT_KEYS = {
    "encoder_onnx_filename",
    "decoder_onnx_filename",
    "metadata_filename",
}
_HASH_KEYS = {
    "checkpoint_sha256",
    "lineage_sha256",
    "policy_state_sha256",
    "encoder_state_sha256",
    "decoder_state_sha256",
    "encoder_onnx_sha256",
    "decoder_onnx_sha256",
    "encoder_embedded_metadata_sha256",
    "decoder_embedded_metadata_sha256",
}
_SAFE_TARGET_HASH_KEYS = _HASH_KEYS | {"safe_target_transform_sha256"}
_VALIDATION_KEYS = {
    "weights_only_checkpoint_validated",
    "exact_policy_reconstructed",
    "simulation_candidate_review_gate_validated",
    "teleop_encoder",
    "true23_decoder",
    "paired_inference",
}
_ORT_PARITY_KEYS = {
    "onnx_checker_full_check",
    "shape_inference",
    "ort_provider",
    "parity_case_count",
    "parity_atol",
    "parity_rtol",
    "parity_max_abs_error",
    "parity_max_rel_error",
    "parity_inputs_sha256",
    "parity_outputs_sha256",
}
_PAIRED_PARITY_KEYS = {
    "performed",
    "provider",
    "dtype",
    "case_count",
    "all_outputs_finite",
    "parity_atol",
    "parity_rtol",
    "max_token_abs_error",
    "max_action_abs_error",
    "outputs_sha256",
}
_EMBEDDED_KEYS = {
    "schema_version",
    "kind",
    "artifact_role",
    *_SAFETY_FLAGS,
    "checkpoint_role",
    "robot_model",
    "required_mode_machine",
    "checkpoint_update_count",
    "reference_profile",
    "input_name",
    "input_shape",
    "output_name",
    "output_shape",
    "input_dtype",
    "output_dtype",
    "onnx_opset",
    "checkpoint_sha256",
    "lineage_sha256",
    "policy_state_sha256",
    "encoder_state_sha256",
    "decoder_state_sha256",
}
_SAFE_TARGET_EMBEDDED_KEYS = _EMBEDDED_KEYS | {
    "decoder_output_semantics",
    "external_safe_target_transform_allowed",
    "safe_target_transform_sha256",
}


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def _safe_target_transform_sha256(contract: Mapping[str, Any]) -> str:
    return artifact.sha256_bytes(artifact.canonical_json_bytes(contract))


def _resolved_safe_target_transform(
    resolved: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return exact safe transform, rejecting partial V11/V12 relabels."""

    task = resolved.get("causal_history_safe_target_v11")
    training_v11 = resolved.get("causal_safe_target_training_v11")
    training_v12 = resolved.get("causal_safe_target_training_v12")
    training_count = sum(
        value is not None for value in (training_v11, training_v12)
    )
    if task is None and training_count == 0:
        return None
    if not isinstance(task, Mapping) or training_count != 1:
        raise ValueError(
            "safe-target resolved task requires exactly one training contract"
        )
    expected = safe_target_transform_contract()
    if (
        task.get("schema") != "g1_true23_causal_history_safe_target_v11"
        or task.get("restart_from_model0") is not True
        or task.get(
            "v9_push_domain_randomization_and_physical_gates_unchanged"
        )
        is not True
        or task.get("safe_target_transform") != expected
        or task.get("target_transform_trained_in_loop") is not True
        or task.get("previous_action_is_applied_safe_native_action") is not True
        or task.get("evaluator_aligned_rapid_recovery_weight") != -25.0
        or task.get("post_hoc_clamp_relabel") is not False
        or task.get("deployment_ready") is not False
    ):
        raise ValueError("V11 safe-target task contract mismatch")
    executed = task.get("executed_environment")
    expected_recovery = {
        "function": (
            "gear_sonic.envs.mjlab."
            "sonic_true23_causal_history_safe_target_v11:"
            "evaluator_aligned_recovery_metric"
        ),
        "weight": -25.0,
        "metric": "tilt+abs(pelvis_height-reference)+target_tracking_rmse",
        "continuous_under_interval_pushes": True,
    }
    if (
        not isinstance(executed, Mapping)
        or executed.get("target_transform") != expected
        or executed.get("history_uses_applied_safe_native_action") is not True
        or executed.get("encoder_bias_applied_after_unbiased_target") is not True
        or executed.get("evaluator_aligned_recovery") != expected_recovery
    ):
        raise ValueError("V11 executed safe-target environment contract mismatch")
    if training_v11 is not None:
        if not isinstance(training_v11, Mapping) or (
            training_v11.get("schema")
            != "g1_true23_causal_safe_target_training_v11"
            or training_v11.get("restart_from_model0") is not True
            or training_v11.get(
                "v9_push_domain_randomization_and_physical_gates_unchanged"
            )
            is not True
            or training_v11.get("safe_target_transform_in_training_loop")
            is not True
            or training_v11.get("deployment_ready") is not False
        ):
            raise ValueError("V11 safe-target training contract mismatch")
    else:
        expected_parameters = [
            "core.actor_module.decoders.g1_dyn.module.16.bias",
            "core.actor_module.decoders.g1_dyn.module.16.weight",
        ]
        if not isinstance(training_v12, Mapping) or (
            training_v12.get("schema")
            != "g1_true23_causal_safe_target_training_v12"
            or training_v12.get("restart_from_model0") is not True
            or training_v12.get("trainable_actor_parameters")
            != expected_parameters
            or training_v12.get("all_other_actor_parameters_and_std_frozen")
            is not True
            or training_v12.get("learning_rate") != 5.0e-7
            or training_v12.get("schedule") != "fixed"
            or training_v12.get("ppo_clip") != 0.05
            or training_v12.get("learning_epochs") != 1
            or training_v12.get("entropy_coefficient") != 0.0
            or training_v12.get("calibration_kl_limit") != 2.0e-3
            or training_v12.get("projection_target_kl") != 1.8e-3
            or training_v12.get("planned_accepted_updates") != 100
            or training_v12.get("allowed_checkpoint_updates")
            != [0, 10, 25, 50, 100]
            or training_v12.get("v11_transform_and_recovery_task_unchanged")
            is not True
            or training_v12.get("deployment_ready") is not False
        ):
            raise ValueError("V12 safe-target training contract mismatch")
    return expected


def _wrap_safe_target_decoder(decoder: Any) -> Any:
    """Embed exact V11 transform in exported decoder; never apply it outside."""

    import torch
    from torch import nn

    class SafeTargetDecoder(nn.Module):
        def __init__(self, raw_decoder: nn.Module) -> None:
            super().__init__()
            self.raw_decoder = raw_decoder

        def forward(self, decoder_input: torch.Tensor) -> torch.Tensor:
            raw_native_action = self.raw_decoder(decoder_input)
            safe_native_action, _target = safe_target_transform_torch(
                raw_native_action
            )
            return safe_native_action

    return SafeTargetDecoder(decoder).eval()


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_strict_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"metadata contains non-finite {token}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"diagnostic metadata is invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("diagnostic metadata must be an object")
    return value


def _validate_output_name(path: Path, expected_suffix: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"diagnostic output may not be a symlink: {expanded}")
    output = expanded.resolve()
    lowered = output.name.lower()
    if ".promotion" in lowered or "deployment" in lowered:
        raise ValueError(
            "diagnostic output filename may not contain .promotion or deployment"
        )
    if output.suffix.lower() != expected_suffix:
        raise ValueError(
            f"diagnostic output must end in {expected_suffix}: {output}"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    return output


def diagnostic_output_paths(
    output_prefix: str | Path,
) -> tuple[Path, Path, Path]:
    """Derive three new-only diagnostic paths from an extensionless prefix."""

    prefix = Path(output_prefix).expanduser()
    lowered_prefix = prefix.name.lower()
    if ".promotion" in lowered_prefix or "deployment" in lowered_prefix:
        raise ValueError(
            "diagnostic output filename may not contain .promotion or deployment"
        )
    if not prefix.name or prefix.suffix:
        raise ValueError("diagnostic output prefix must be extensionless")
    encoder = _validate_output_name(
        prefix.with_name(f"{prefix.name}.encoder.onnx"),
        ".onnx",
    )
    decoder = _validate_output_name(
        prefix.with_name(f"{prefix.name}.decoder.onnx"),
        ".onnx",
    )
    metadata = _validate_output_name(
        prefix.with_name(f"{prefix.name}.diagnostic.json"),
        ".json",
    )
    if len({encoder, decoder, metadata}) != 3:
        raise ValueError("diagnostic output paths must differ")
    return encoder, decoder, metadata


def _component_state_sha256(
    policy_state: Mapping[str, Any],
    prefix: str,
) -> str:
    import torch

    selected = {
        key: value for key, value in policy_state.items() if key.startswith(prefix)
    }
    if not selected:
        raise ValueError(f"policy has no tensors under {prefix}")
    digest = hashlib.sha256()
    for key in sorted(selected):
        tensor = selected[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"policy_state_dict[{key!r}] is not a tensor")
        contiguous = tensor.detach().cpu().contiguous()
        if contiguous.dtype is not torch.float32 or not torch.isfinite(
            contiguous
        ).all():
            raise ValueError(f"policy_state_dict[{key!r}] must be finite float32")
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(artifact.canonical_json_bytes(list(contiguous.shape)))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_material(
    checkpoint_path: Path,
    *,
    expected_lineage_sha256: str,
    expected_reference_profile: str | None = None,
) -> tuple[Mapping[str, Any], Any, Any, dict[str, Any]]:
    checkpoint_path = checkpoint_path.expanduser()
    if checkpoint_path.is_symlink():
        raise ValueError("MJLab diagnostic checkpoint may not be a symlink")
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MJLab diagnostic checkpoint missing: {checkpoint_path}")
    causal_export = expected_reference_profile == CAUSAL_HISTORY_PROFILE
    if expected_reference_profile not in (None, CAUSAL_HISTORY_PROFILE):
        raise ValueError(
            "diagnostic exporter supports only default released semantics or "
            f"exact causal profile {CAUSAL_HISTORY_PROFILE!r}"
        )
    checkpoint_pattern = (
        r"causal_model_[0-9]+\.pt" if causal_export else r"model_[0-9]+\.pt"
    )
    if re.fullmatch(checkpoint_pattern, checkpoint_path.name) is None:
        raise ValueError(
            "MJLab diagnostic source filename does not match its semantic "
            f"namespace: expected {checkpoint_pattern}"
        )
    expected_lineage_sha256 = _require_sha256(
        expected_lineage_sha256,
        "expected_lineage_sha256",
    )
    checkpoint = load_mjlab_training_checkpoint(
        checkpoint_path,
        expected_lineage_sha256=expected_lineage_sha256,
        map_location="cpu",
    )
    gate = checkpoint["training_gate"]
    if gate.get("simulation_candidate_review_allowed") is not True:
        raise ValueError(
            "MJLab diagnostic export requires "
            "simulation_candidate_review_allowed=true"
        )
    if (
        gate.get("deployment_ready") is not False
        or gate.get("promotion_eligible") is not False
    ):
        raise ValueError("MJLab diagnostic checkpoint safety gate mismatch")

    lineage = checkpoint["lineage"]
    architecture_profile = lineage["warm_start"]["reference_profile"]
    reference_profile = architecture_profile
    output_transform: Mapping[str, Any] | None = None
    if causal_export:
        expected_contract = causal_history_profile_contract()
        resolved = lineage["materials"]["resolved_config"]["payload"]
        if resolved.get("reference_profile") != CAUSAL_HISTORY_PROFILE:
            raise ValueError(
                "causal diagnostic checkpoint resolved reference_profile mismatch"
            )
        if resolved.get("semantic_profile") != expected_contract:
            raise ValueError(
                "causal diagnostic checkpoint semantic contract/hash mismatch"
            )
        if architecture_profile != REFERENCE_PROFILE_LOW_LATENCY:
            raise ValueError(
                "causal diagnostic architecture must derive from exact released "
                "low-latency topology"
            )
        initialization = resolved.get("architecture_initialization")
        if not isinstance(initialization, Mapping) or (
            initialization.get("source_profile")
            != REFERENCE_PROFILE_LOW_LATENCY
            or initialization.get("source_future_semantics_inherited") is not False
            or initialization.get("checkpoint_relabelled") is not False
            or initialization.get("retraining_required") is not True
        ):
            raise ValueError(
                "causal diagnostic architecture initialization boundary mismatch"
            )
        recovery = resolved.get("recovery")
        if not isinstance(recovery, Mapping) or (
            recovery.get("released_future_profile_exporter_must_reject") is not True
            or recovery.get("checkpoint_filename_pattern")
            != "causal_model_N.pt"
        ):
            raise ValueError("causal diagnostic recovery boundary mismatch")
        output_transform = _resolved_safe_target_transform(resolved)
        reference_profile = CAUSAL_HISTORY_PROFILE
    pair_checkpoint = {
        "policy_state_dict": checkpoint["policy_state_dict"],
        # Causal training keeps exact low-latency network topology.  Semantic
        # identity is bound separately below; never relabel topology helpers.
        "g1_23dof_metadata": {"reference_profile": architecture_profile},
    }
    encoder, decoder, reconstructed_policy_hash = (
        artifact.build_true23_policy_pair(pair_checkpoint)
    )
    policy_hash = _require_sha256(
        checkpoint["policy_state_sha256"],
        "checkpoint policy_state_sha256",
    )
    if reconstructed_policy_hash != policy_hash:
        raise ValueError("reconstructed exact policy hash differs from checkpoint")
    if output_transform is not None:
        decoder = _wrap_safe_target_decoder(decoder)
    material = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": artifact.sha256_file(checkpoint_path),
        "lineage_sha256": expected_lineage_sha256,
        "policy_state_sha256": policy_hash,
        "encoder_state_sha256": _component_state_sha256(
            checkpoint["policy_state_dict"],
            _ENCODER_PREFIX,
        ),
        "decoder_state_sha256": _component_state_sha256(
            checkpoint["policy_state_dict"],
            _DECODER_PREFIX,
        ),
        "update_count": checkpoint["update_count"],
        "reference_profile": reference_profile,
        "output_transform": output_transform,
    }
    return checkpoint, encoder, decoder, material


def _embedded_metadata(
    *,
    role: str,
    input_name: str,
    input_shape: list[int],
    output_name: str,
    output_shape: list[int],
    material: Mapping[str, Any],
) -> dict[str, Any]:
    if role not in {"teleop_encoder", "true23_decoder"}:
        raise ValueError(f"unsupported diagnostic ONNX role: {role}")
    output_transform = material.get("output_transform")
    schema_version = (
        DIAGNOSTIC_SAFE_TARGET_SCHEMA_VERSION
        if output_transform is not None
        else DIAGNOSTIC_SCHEMA_VERSION
    )
    body = {
        "schema_version": schema_version,
        "kind": DIAGNOSTIC_EMBEDDED_KIND,
        "artifact_role": role,
        **_SAFETY_FLAGS,
        "checkpoint_role": MJLAB_CHECKPOINT_ROLE,
        "robot_model": ROBOT_MODEL,
        "required_mode_machine": REQUIRED_MODE_MACHINE,
        "checkpoint_update_count": material["update_count"],
        "reference_profile": material["reference_profile"],
        "input_name": input_name,
        "input_shape": input_shape,
        "output_name": output_name,
        "output_shape": output_shape,
        "input_dtype": "float32",
        "output_dtype": "float32",
        "onnx_opset": artifact.ONNX_OPSET_VERSION,
        "checkpoint_sha256": material["checkpoint_sha256"],
        "lineage_sha256": material["lineage_sha256"],
        "policy_state_sha256": material["policy_state_sha256"],
        "encoder_state_sha256": material["encoder_state_sha256"],
        "decoder_state_sha256": material["decoder_state_sha256"],
    }
    if output_transform is not None:
        body.update(
            {
                "decoder_output_semantics": "applied_safe_native_action",
                "external_safe_target_transform_allowed": False,
                "safe_target_transform_sha256": (
                    _safe_target_transform_sha256(output_transform)
                ),
            }
        )
    return body


def _set_embedded_metadata(model: Any, value: Mapping[str, Any]) -> None:
    import onnx

    onnx.helper.set_model_props(
        model,
        {
            artifact.ONNX_METADATA_KEY: artifact.canonical_json_bytes(value)
            .decode("utf-8")
            .rstrip("\n")
        },
    )


def _get_embedded_metadata(model: Any) -> Mapping[str, Any]:
    if len(model.metadata_props) != 1:
        raise ValueError(
            "diagnostic ONNX must contain exactly one true23 metadata property"
        )
    values = {
        entry.key: entry.value
        for entry in model.metadata_props
    }
    if set(values) != {artifact.ONNX_METADATA_KEY}:
        raise ValueError(
            "diagnostic ONNX must contain exactly one true23 metadata property"
        )
    try:
        value = json.loads(
            values[artifact.ONNX_METADATA_KEY],
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"embedded metadata contains non-finite {token}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError("diagnostic ONNX metadata is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("diagnostic ONNX metadata must be an object")
    schema_version = value.get("schema_version")
    expected_keys = (
        _SAFE_TARGET_EMBEDDED_KEYS
        if schema_version == DIAGNOSTIC_SAFE_TARGET_SCHEMA_VERSION
        else _EMBEDDED_KEYS
    )
    _require_exact_keys(value, expected_keys, "diagnostic ONNX metadata")
    if schema_version not in {
        DIAGNOSTIC_SCHEMA_VERSION,
        DIAGNOSTIC_SAFE_TARGET_SCHEMA_VERSION,
    }:
        raise ValueError("diagnostic ONNX metadata schema_version mismatch")
    if any(value.get(key) is not expected for key, expected in _SAFETY_FLAGS.items()):
        raise ValueError("diagnostic ONNX safety flags mismatch")
    if schema_version == DIAGNOSTIC_SAFE_TARGET_SCHEMA_VERSION and (
        value.get("decoder_output_semantics") != "applied_safe_native_action"
        or value.get("external_safe_target_transform_allowed") is not False
        or value.get("safe_target_transform_sha256")
        != _safe_target_transform_sha256(safe_target_transform_contract())
    ):
        raise ValueError("diagnostic ONNX safe-target metadata mismatch")
    return value


def _export_one(
    *,
    module: Any,
    output_path: Path,
    input_name: str,
    input_dim: int,
    output_name: str,
    output_dim: int,
    embedded: Mapping[str, Any],
    structure_validator: Any,
) -> dict[str, Any]:
    import onnx
    from onnx import shape_inference
    import torch

    example = torch.zeros(1, input_dim, dtype=torch.float32)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            module,
            example,
            output_path,
            export_params=True,
            input_names=[input_name],
            output_names=[output_name],
            opset_version=artifact.ONNX_OPSET_VERSION,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
    model = onnx.load(output_path, load_external_data=False)
    model = shape_inference.infer_shapes(
        model,
        strict_mode=True,
        data_prop=True,
    )
    _set_embedded_metadata(model, embedded)
    onnx.save_model(model, output_path, save_as_external_data=False)
    model = onnx.load(output_path, load_external_data=False)
    structure_validator(model)
    if _get_embedded_metadata(model) != embedded:
        raise ValueError("diagnostic ONNX embedded metadata changed")
    return artifact.validate_ort_parity(
        module,
        output_path,
        input_name=input_name,
        input_dim=input_dim,
        output_name=output_name,
        output_dim=output_dim,
    )


def _paired_parity(
    encoder: Any,
    decoder: Any,
    encoder_path: Path,
    decoder_path: Path,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    import torch

    encoder_session = ort.InferenceSession(
        str(encoder_path),
        providers=["CPUExecutionProvider"],
    )
    decoder_session = ort.InferenceSession(
        str(decoder_path),
        providers=["CPUExecutionProvider"],
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260803)
    teleop_vectors = (
        torch.zeros(1, TELEOP_ENCODER_INPUT_DIM, dtype=torch.float32),
        torch.linspace(
            -1.0,
            1.0,
            TELEOP_ENCODER_INPUT_DIM,
            dtype=torch.float32,
        ).reshape(1, -1),
        torch.randn(
            1,
            TELEOP_ENCODER_INPUT_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.1,
    )
    proprioception_dim = DEPLOYMENT_DECODER_INPUT_DIM - TOKEN_DIM
    proprio_vectors = (
        torch.zeros(1, proprioception_dim, dtype=torch.float32),
        torch.linspace(
            1.0,
            -1.0,
            proprioception_dim,
            dtype=torch.float32,
        ).reshape(1, -1),
        torch.randn(
            1,
            proprioception_dim,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.1,
    )
    max_token_error = 0.0
    max_action_error = 0.0
    outputs_digest = hashlib.sha256()
    for teleop, proprioception in zip(teleop_vectors, proprio_vectors):
        with torch.no_grad():
            torch_token = encoder(teleop).detach().cpu().numpy()
            torch_input = torch.cat(
                (torch.from_numpy(torch_token), proprioception),
                dim=-1,
            )
            torch_action = decoder(torch_input).detach().cpu().numpy()
        ort_token = encoder_session.run(
            [artifact.ENCODER_ONNX_OUTPUT_NAME],
            {artifact.ENCODER_ONNX_INPUT_NAME: teleop.numpy()},
        )[0]
        ort_input = np.concatenate((ort_token, proprioception.numpy()), axis=-1)
        ort_action = decoder_session.run(
            [artifact.ONNX_OUTPUT_NAME],
            {artifact.ONNX_INPUT_NAME: ort_input.astype(np.float32, copy=False)},
        )[0]
        arrays = (torch_token, ort_token, torch_action, ort_action)
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("paired diagnostic inference produced non-finite output")
        if torch_token.shape != (1, TOKEN_DIM) or ort_token.shape != (1, TOKEN_DIM):
            raise ValueError("paired diagnostic token shape is not [1,64]")
        if torch_action.shape != (1, TARGET_DOF) or ort_action.shape != (
            1,
            TARGET_DOF,
        ):
            raise ValueError("paired diagnostic action shape is not [1,23]")
        token_error = float(np.max(np.abs(torch_token - ort_token), initial=0.0))
        action_error = float(
            np.max(np.abs(torch_action - ort_action), initial=0.0)
        )
        max_token_error = max(max_token_error, token_error)
        max_action_error = max(max_action_error, action_error)
        if not np.allclose(
            torch_token,
            ort_token,
            rtol=artifact.PARITY_RTOL,
            atol=artifact.PARITY_ATOL,
        ) or not np.allclose(
            torch_action,
            ort_action,
            rtol=artifact.PARITY_RTOL,
            atol=artifact.PARITY_ATOL,
        ):
            raise ValueError(
                "paired diagnostic PyTorch/ONNX Runtime parity failed"
            )
        outputs_digest.update(ort_token.tobytes())
        outputs_digest.update(ort_action.tobytes())
    return {
        "performed": True,
        "provider": "CPUExecutionProvider",
        "dtype": "float32",
        "case_count": 3,
        "all_outputs_finite": True,
        "parity_atol": artifact.PARITY_ATOL,
        "parity_rtol": artifact.PARITY_RTOL,
        "max_token_abs_error": round(max_token_error, 12),
        "max_action_abs_error": round(max_action_error, 12),
        "outputs_sha256": outputs_digest.hexdigest(),
    }


def _contract(
    output_transform: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "robot_model": ROBOT_MODEL,
        "required_mode_machine": REQUIRED_MODE_MACHINE,
        "native_action_dof": TARGET_DOF,
        "history_length": DEPLOYMENT_HISTORY_LENGTH,
        "observation_layout": OBS_LAYOUT_PADDED_IL29,
        "onnx_opset": artifact.ONNX_OPSET_VERSION,
        "encoder": {
            "input_name": artifact.ENCODER_ONNX_INPUT_NAME,
            "input_shape": [1, TELEOP_ENCODER_INPUT_DIM],
            "input_dtype": "float32",
            "output_name": artifact.ENCODER_ONNX_OUTPUT_NAME,
            "output_shape": [1, TOKEN_DIM],
            "output_dtype": "float32",
            "dynamic_axes": False,
        },
        "decoder": {
            "input_name": artifact.ONNX_INPUT_NAME,
            "input_shape": [1, DEPLOYMENT_DECODER_INPUT_DIM],
            "input_dtype": "float32",
            "output_name": artifact.ONNX_OUTPUT_NAME,
            "output_shape": [1, TARGET_DOF],
            "output_dtype": "float32",
            "dynamic_axes": False,
        },
    }
    if output_transform is not None:
        body.update(
            {
                "decoder_output_semantics": "applied_safe_native_action",
                "external_safe_target_transform_allowed": False,
                "previous_action_semantics": "applied_safe_native_action",
                "safe_target_transform": dict(output_transform),
            }
        )
    return body


def _metadata(
    *,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
    encoder_bytes_path: Path,
    decoder_bytes_path: Path,
    material: Mapping[str, Any],
    embedded_by_role: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    output_transform = material.get("output_transform")
    schema_version = (
        DIAGNOSTIC_SAFE_TARGET_SCHEMA_VERSION
        if output_transform is not None
        else DIAGNOSTIC_SCHEMA_VERSION
    )
    body = {
        "schema_version": schema_version,
        "kind": DIAGNOSTIC_BUNDLE_KIND,
        **_SAFETY_FLAGS,
        "checkpoint_role": MJLAB_CHECKPOINT_ROLE,
        "allowed_uses": list(DIAGNOSTIC_ALLOWED_USES),
        "forbidden_uses": list(DIAGNOSTIC_FORBIDDEN_USES),
        "no_robot_or_network_commands_performed": True,
        "source": {
            "checkpoint_filename": material["checkpoint_path"].name,
            "checkpoint_update_count": material["update_count"],
            "reference_profile": material["reference_profile"],
            "simulation_candidate_review_allowed": True,
        },
        "contract": _contract(output_transform),
        "artifacts": {
            "encoder_onnx_filename": encoder_path.name,
            "decoder_onnx_filename": decoder_path.name,
            "metadata_filename": metadata_path.name,
        },
        "hashes": {
            "checkpoint_sha256": material["checkpoint_sha256"],
            "lineage_sha256": material["lineage_sha256"],
            "policy_state_sha256": material["policy_state_sha256"],
            "encoder_state_sha256": material["encoder_state_sha256"],
            "decoder_state_sha256": material["decoder_state_sha256"],
            "encoder_onnx_sha256": artifact.sha256_file(
                encoder_bytes_path
            ),
            "decoder_onnx_sha256": artifact.sha256_file(
                decoder_bytes_path
            ),
            "encoder_embedded_metadata_sha256": artifact.sha256_bytes(
                artifact.canonical_json_bytes(
                    embedded_by_role["teleop_encoder"]
                )
            ),
            "decoder_embedded_metadata_sha256": artifact.sha256_bytes(
                artifact.canonical_json_bytes(
                    embedded_by_role["true23_decoder"]
                )
            ),
        },
        "validation": dict(validation),
    }
    if output_transform is not None:
        body["hashes"]["safe_target_transform_sha256"] = (
            _safe_target_transform_sha256(output_transform)
        )
    return {
        **body,
        "metadata_payload_sha256": artifact.sha256_bytes(
            artifact.canonical_json_bytes(body)
        ),
    }


def _temporary_path(final: Path, suffix: str) -> Path:
    final.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=final.parent,
        prefix=f".{final.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    return Path(name)


def _publish_new(temporary: Path, final: Path) -> None:
    try:
        os.link(temporary, final)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite diagnostic output: {final}"
        ) from exc
    temporary.unlink()


def export_mjlab_diagnostic_onnx(
    checkpoint_path: str | Path,
    output_prefix: str | Path,
    *,
    expected_lineage_sha256: str,
    expected_reference_profile: str | None = None,
) -> tuple[Path, Path, Path, Mapping[str, Any]]:
    """Export immutable diagnostic-only encoder/decoder ONNX from one resume."""

    encoder_path, decoder_path, metadata_path = diagnostic_output_paths(
        output_prefix
    )
    _checkpoint, encoder, decoder, material = _checkpoint_material(
        Path(checkpoint_path),
        expected_lineage_sha256=expected_lineage_sha256,
        expected_reference_profile=expected_reference_profile,
    )
    embedded_by_role = {
        "teleop_encoder": _embedded_metadata(
            role="teleop_encoder",
            input_name=artifact.ENCODER_ONNX_INPUT_NAME,
            input_shape=[1, TELEOP_ENCODER_INPUT_DIM],
            output_name=artifact.ENCODER_ONNX_OUTPUT_NAME,
            output_shape=[1, TOKEN_DIM],
            material=material,
        ),
        "true23_decoder": _embedded_metadata(
            role="true23_decoder",
            input_name=artifact.ONNX_INPUT_NAME,
            input_shape=[1, DEPLOYMENT_DECODER_INPUT_DIM],
            output_name=artifact.ONNX_OUTPUT_NAME,
            output_shape=[1, TARGET_DOF],
            material=material,
        ),
    }
    temporary_paths: list[Path] = []
    published: list[Path] = []
    try:
        temporary_encoder = _temporary_path(encoder_path, ".tmp.onnx")
        temporary_paths.append(temporary_encoder)
        temporary_decoder = _temporary_path(decoder_path, ".tmp.onnx")
        temporary_paths.append(temporary_decoder)
        encoder_validation = _export_one(
            module=encoder,
            output_path=temporary_encoder,
            input_name=artifact.ENCODER_ONNX_INPUT_NAME,
            input_dim=TELEOP_ENCODER_INPUT_DIM,
            output_name=artifact.ENCODER_ONNX_OUTPUT_NAME,
            output_dim=TOKEN_DIM,
            embedded=embedded_by_role["teleop_encoder"],
            structure_validator=artifact.validate_encoder_onnx_structure,
        )
        decoder_validation = _export_one(
            module=decoder,
            output_path=temporary_decoder,
            input_name=artifact.ONNX_INPUT_NAME,
            input_dim=DEPLOYMENT_DECODER_INPUT_DIM,
            output_name=artifact.ONNX_OUTPUT_NAME,
            output_dim=TARGET_DOF,
            embedded=embedded_by_role["true23_decoder"],
            structure_validator=artifact.validate_onnx_structure,
        )
        validation = {
            "weights_only_checkpoint_validated": True,
            "exact_policy_reconstructed": True,
            "simulation_candidate_review_gate_validated": True,
            "teleop_encoder": encoder_validation,
            "true23_decoder": decoder_validation,
            "paired_inference": _paired_parity(
                encoder,
                decoder,
                temporary_encoder,
                temporary_decoder,
            ),
        }
        metadata = _metadata(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            metadata_path=metadata_path,
            encoder_bytes_path=temporary_encoder,
            decoder_bytes_path=temporary_decoder,
            material=material,
            embedded_by_role=embedded_by_role,
            validation=validation,
        )
        temporary_metadata = _temporary_path(metadata_path, ".tmp.json")
        temporary_paths.append(temporary_metadata)
        temporary_metadata.write_bytes(artifact.canonical_json_bytes(metadata))
        verify_mjlab_diagnostic_onnx(
            temporary_encoder,
            temporary_decoder,
            temporary_metadata,
            checkpoint_path=material["checkpoint_path"],
            expected_filenames=(
                encoder_path.name,
                decoder_path.name,
                metadata_path.name,
            ),
            expected_reference_profile=expected_reference_profile,
        )
        for temporary, final in (
            (temporary_encoder, encoder_path),
            (temporary_decoder, decoder_path),
            (temporary_metadata, metadata_path),
        ):
            _publish_new(temporary, final)
            published.append(final)
        return encoder_path, decoder_path, metadata_path, metadata
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def _validate_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(value, _BUNDLE_KEYS, "diagnostic bundle")
    schema_version = value.get("schema_version")
    if (
        schema_version
        not in {
            DIAGNOSTIC_SCHEMA_VERSION,
            DIAGNOSTIC_SAFE_TARGET_SCHEMA_VERSION,
        }
        or value.get("kind") != DIAGNOSTIC_BUNDLE_KIND
        or value.get("checkpoint_role") != MJLAB_CHECKPOINT_ROLE
        or value.get("allowed_uses") != list(DIAGNOSTIC_ALLOWED_USES)
        or value.get("forbidden_uses") != list(DIAGNOSTIC_FORBIDDEN_USES)
        or value.get("no_robot_or_network_commands_performed") is not True
        or any(
            value.get(key) is not expected
            for key, expected in _SAFETY_FLAGS.items()
        )
    ):
        raise ValueError("diagnostic bundle safety contract mismatch")
    safe_target_bundle = schema_version == DIAGNOSTIC_SAFE_TARGET_SCHEMA_VERSION
    for key, expected in (
        ("source", _SOURCE_KEYS),
        (
            "contract",
            _SAFE_TARGET_CONTRACT_KEYS
            if safe_target_bundle
            else _CONTRACT_KEYS,
        ),
        ("artifacts", _ARTIFACT_KEYS),
        (
            "hashes",
            _SAFE_TARGET_HASH_KEYS if safe_target_bundle else _HASH_KEYS,
        ),
        ("validation", _VALIDATION_KEYS),
    ):
        nested = value.get(key)
        if not isinstance(nested, Mapping):
            raise ValueError(f"diagnostic bundle {key} must be an object")
        _require_exact_keys(nested, expected, f"diagnostic bundle {key}")
    contract = value["contract"]
    expected_output_transform = (
        safe_target_transform_contract() if safe_target_bundle else None
    )
    if contract != _contract(expected_output_transform):
        raise ValueError("diagnostic bundle ONNX/robot contract mismatch")
    for graph_name in ("encoder", "decoder"):
        _require_exact_keys(
            contract[graph_name],
            _GRAPH_KEYS,
            f"diagnostic bundle contract.{graph_name}",
        )
    source = value["source"]
    if (
        not isinstance(source["checkpoint_filename"], str)
        or re.fullmatch(
            r"(?:causal_)?model_[0-9]+\.pt",
            source["checkpoint_filename"],
        )
        is None
        or isinstance(source["checkpoint_update_count"], bool)
        or not isinstance(source["checkpoint_update_count"], int)
        or source["checkpoint_update_count"] < 0
        or not isinstance(source["reference_profile"], str)
        or not source["reference_profile"]
        or source["simulation_candidate_review_allowed"] is not True
    ):
        raise ValueError("diagnostic bundle source is invalid")
    for key, filename in value["artifacts"].items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"diagnostic bundle artifacts.{key} is invalid")
        lowered = filename.lower()
        if ".promotion" in lowered or "deployment" in lowered:
            raise ValueError("diagnostic bundle contains forbidden output naming")
    for key, digest in value["hashes"].items():
        _require_sha256(digest, f"diagnostic bundle hashes.{key}")
    if safe_target_bundle and value["hashes"][
        "safe_target_transform_sha256"
    ] != _safe_target_transform_sha256(expected_output_transform):
        raise ValueError("diagnostic safe-target transform hash mismatch")
    validation = value["validation"]
    if any(
        validation.get(key) is not True
        for key in (
            "weights_only_checkpoint_validated",
            "exact_policy_reconstructed",
            "simulation_candidate_review_gate_validated",
        )
    ):
        raise ValueError("diagnostic bundle validation gate mismatch")
    for role in ("teleop_encoder", "true23_decoder"):
        report = validation[role]
        if not isinstance(report, Mapping):
            raise ValueError(f"diagnostic validation {role} must be an object")
        _require_exact_keys(
            report,
            _ORT_PARITY_KEYS,
            f"diagnostic validation {role}",
        )
        if (
            report["onnx_checker_full_check"] is not True
            or report["shape_inference"] is not True
            or report["ort_provider"] != "CPUExecutionProvider"
            or report["parity_case_count"] != 3
            or report["parity_atol"] != artifact.PARITY_ATOL
            or report["parity_rtol"] != artifact.PARITY_RTOL
            or not isinstance(report["parity_max_abs_error"], (int, float))
            or not math.isfinite(report["parity_max_abs_error"])
            or report["parity_max_abs_error"] < 0
            or not isinstance(report["parity_max_rel_error"], (int, float))
            or not math.isfinite(report["parity_max_rel_error"])
            or report["parity_max_rel_error"] < 0
        ):
            raise ValueError(f"diagnostic validation {role} parity mismatch")
        _require_sha256(
            report["parity_inputs_sha256"],
            f"diagnostic validation {role}.parity_inputs_sha256",
        )
        _require_sha256(
            report["parity_outputs_sha256"],
            f"diagnostic validation {role}.parity_outputs_sha256",
        )
    paired = validation["paired_inference"]
    if not isinstance(paired, Mapping):
        raise ValueError("diagnostic paired_inference must be an object")
    _require_exact_keys(
        paired,
        _PAIRED_PARITY_KEYS,
        "diagnostic paired_inference",
    )
    if (
        paired["performed"] is not True
        or paired["provider"] != "CPUExecutionProvider"
        or paired["dtype"] != "float32"
        or paired["case_count"] != 3
        or paired["all_outputs_finite"] is not True
        or paired["parity_atol"] != artifact.PARITY_ATOL
        or paired["parity_rtol"] != artifact.PARITY_RTOL
        or any(
            not isinstance(paired[key], (int, float))
            or not math.isfinite(paired[key])
            or paired[key] < 0
            for key in ("max_token_abs_error", "max_action_abs_error")
        )
    ):
        raise ValueError("diagnostic paired_inference parity mismatch")
    _require_sha256(
        paired["outputs_sha256"],
        "diagnostic paired_inference.outputs_sha256",
    )
    body = dict(value)
    payload_hash = body.pop("metadata_payload_sha256")
    _require_sha256(payload_hash, "metadata_payload_sha256")
    if artifact.sha256_bytes(artifact.canonical_json_bytes(body)) != payload_hash:
        raise ValueError("diagnostic metadata_payload_sha256 mismatch")
    return value


def verify_mjlab_diagnostic_onnx(
    encoder_path: str | Path,
    decoder_path: str | Path,
    metadata_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    expected_filenames: tuple[str, str, str] | None = None,
    expected_reference_profile: str | None = None,
) -> Mapping[str, Any]:
    """Verify safety flags, hashes, structure, lineage, and optional parity."""

    import onnx

    raw_paths = tuple(
        Path(path).expanduser()
        for path in (encoder_path, decoder_path, metadata_path)
    )
    if any(path.is_symlink() for path in raw_paths):
        raise ValueError("diagnostic artifacts may not be symlinks")
    encoder_path, decoder_path, metadata_path = (
        path.resolve() for path in raw_paths
    )
    for path in (encoder_path, decoder_path, metadata_path):
        if not path.is_file():
            raise ValueError(f"diagnostic artifact must be a regular file: {path}")
    metadata = _validate_metadata(_read_strict_json(metadata_path))
    if (
        expected_reference_profile is not None
        and metadata["source"]["reference_profile"]
        != expected_reference_profile
    ):
        raise ValueError(
            "diagnostic bundle reference_profile differs from required profile"
        )
    filenames = expected_filenames or (
        encoder_path.name,
        decoder_path.name,
        metadata_path.name,
    )
    expected_artifacts = {
        "encoder_onnx_filename": filenames[0],
        "decoder_onnx_filename": filenames[1],
        "metadata_filename": filenames[2],
    }
    if metadata["artifacts"] != expected_artifacts:
        raise ValueError("diagnostic artifact filenames differ from metadata")
    hashes = metadata["hashes"]
    if artifact.sha256_file(encoder_path) != hashes["encoder_onnx_sha256"]:
        raise ValueError("diagnostic encoder_onnx_sha256 mismatch")
    if artifact.sha256_file(decoder_path) != hashes["decoder_onnx_sha256"]:
        raise ValueError("diagnostic decoder_onnx_sha256 mismatch")
    models = {
        "teleop_encoder": onnx.load(encoder_path, load_external_data=False),
        "true23_decoder": onnx.load(decoder_path, load_external_data=False),
    }
    artifact.validate_encoder_onnx_structure(models["teleop_encoder"])
    artifact.validate_onnx_structure(models["true23_decoder"])
    embedded_by_role = {
        role: _get_embedded_metadata(model) for role, model in models.items()
    }
    embedded_material = {
        "update_count": metadata["source"]["checkpoint_update_count"],
        "reference_profile": metadata["source"]["reference_profile"],
        "output_transform": metadata["contract"].get(
            "safe_target_transform"
        ),
        **{
            key: hashes[key]
            for key in (
                "checkpoint_sha256",
                "lineage_sha256",
                "policy_state_sha256",
                "encoder_state_sha256",
                "decoder_state_sha256",
            )
        },
    }
    expected_embedded = {
        "teleop_encoder": _embedded_metadata(
            role="teleop_encoder",
            input_name=artifact.ENCODER_ONNX_INPUT_NAME,
            input_shape=[1, TELEOP_ENCODER_INPUT_DIM],
            output_name=artifact.ENCODER_ONNX_OUTPUT_NAME,
            output_shape=[1, TOKEN_DIM],
            material=embedded_material,
        ),
        "true23_decoder": _embedded_metadata(
            role="true23_decoder",
            input_name=artifact.ONNX_INPUT_NAME,
            input_shape=[1, DEPLOYMENT_DECODER_INPUT_DIM],
            output_name=artifact.ONNX_OUTPUT_NAME,
            output_shape=[1, TARGET_DOF],
            material=embedded_material,
        ),
    }
    for role, embedded in embedded_by_role.items():
        if embedded != expected_embedded[role]:
            raise ValueError(f"diagnostic {role} embedded contract mismatch")
        expected_hash = hashes[
            "encoder_embedded_metadata_sha256"
            if role == "teleop_encoder"
            else "decoder_embedded_metadata_sha256"
        ]
        if artifact.sha256_bytes(
            artifact.canonical_json_bytes(embedded)
        ) != expected_hash:
            raise ValueError(f"diagnostic {role} embedded metadata hash mismatch")
        for key in (
            "checkpoint_sha256",
            "lineage_sha256",
            "policy_state_sha256",
            "encoder_state_sha256",
            "decoder_state_sha256",
        ):
            if embedded[key] != hashes[key]:
                raise ValueError(f"diagnostic {role} {key} binding mismatch")
    if checkpoint_path is not None:
        checkpoint, encoder, decoder, material = _checkpoint_material(
            Path(checkpoint_path),
            expected_lineage_sha256=hashes["lineage_sha256"],
            expected_reference_profile=expected_reference_profile,
        )
        del checkpoint
        expected_material = {
            "checkpoint_sha256": material["checkpoint_sha256"],
            "lineage_sha256": material["lineage_sha256"],
            "policy_state_sha256": material["policy_state_sha256"],
            "encoder_state_sha256": material["encoder_state_sha256"],
            "decoder_state_sha256": material["decoder_state_sha256"],
        }
        if any(hashes[key] != value for key, value in expected_material.items()):
            raise ValueError("diagnostic bundle differs from source checkpoint")
        if (
            metadata["source"]["checkpoint_update_count"]
            != material["update_count"]
            or metadata["source"]["reference_profile"]
            != material["reference_profile"]
            or metadata["contract"].get("safe_target_transform")
            != material["output_transform"]
        ):
            raise ValueError("diagnostic source counters/profile mismatch")
        encoder_parity = artifact.validate_ort_parity(
            encoder,
            encoder_path,
            input_name=artifact.ENCODER_ONNX_INPUT_NAME,
            input_dim=TELEOP_ENCODER_INPUT_DIM,
            output_name=artifact.ENCODER_ONNX_OUTPUT_NAME,
            output_dim=TOKEN_DIM,
        )
        decoder_parity = artifact.validate_ort_parity(
            decoder,
            decoder_path,
            input_name=artifact.ONNX_INPUT_NAME,
            input_dim=DEPLOYMENT_DECODER_INPUT_DIM,
            output_name=artifact.ONNX_OUTPUT_NAME,
            output_dim=TARGET_DOF,
        )
        paired_parity = _paired_parity(
            encoder,
            decoder,
            encoder_path,
            decoder_path,
        )
        if (
            encoder_parity != metadata["validation"]["teleop_encoder"]
            or decoder_parity != metadata["validation"]["true23_decoder"]
            or paired_parity != metadata["validation"]["paired_inference"]
        ):
            raise ValueError("diagnostic ONNX parity record mismatch")
    return metadata
