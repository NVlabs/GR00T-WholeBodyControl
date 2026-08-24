"""Fail-closed checkpoint lineage for exact-policy true23 MJLab training.

This module is deliberately separate from deployment promotion.  It writes a
portable, ``torch.load(..., weights_only=True)``-safe trainer-resume checkpoint
for a custom MJLab runner that trains the exact SONIC teleop encoder and
true-23 decoder in place.  It never converts or relabels a stock RSL-RL actor,
and no checkpoint produced here is deployment-ready.  ``update_count`` means
completed outer PPO iterations (successful ``alg.update()`` calls), not
epoch/minibatch optimizer steps inside one iteration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

import torch

from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    decoder_layer_dims_for_profile,
    inspect_true23_policy_state,
    sha256_file,
)
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    INITIALIZATION_STAGE,
    checkpoint_stage,
    load_safe_true23_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import (
    APPROVED_WARM_START_RELEASES,
    CANONICAL_COMPACT_IL23_JOINT_NAMES,
    DECODER_OUTPUT_LAYOUT,
    DEPLOYMENT_DECODER_INPUT_DIM,
    DEPLOYMENT_HISTORY_LENGTH,
    EXCLUDED_HARDWARE_JOINT_IDS,
    HARDWARE_23_JOINT_NAMES,
    HARDWARE_JOINT_IDS,
    HISTORY_ORDER,
    ISAACLAB_TO_MUJOCO_DOF,
    MINIMUM_TRAINING_UPDATES,
    MISSING_OBSERVATION_FILL,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_JOINT_NAMES,
    NATIVE_IL23_TO_CANONICAL_IL29,
    OBS_LAYOUT_PADDED_IL29,
    OBSERVATION_TERM_ORDER,
    REQUIRED_MODE_MACHINE,
    ROBOT_MODEL,
    SOURCE_IL29_EXCLUDED_INDICES,
    SOURCE_IL29_JOINT_NAMES,
    SOURCE_IL29_KEEP_INDICES,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
)

MJLAB_TRAINING_CHECKPOINT_SCHEMA_VERSION = 2
MJLAB_TRAINING_LINEAGE_SCHEMA_VERSION = 2
MJLAB_MATERIAL_MANIFEST_SCHEMA_VERSION = 1
MJLAB_JOINT_SEMANTICS_SCHEMA_VERSION = 1
MJLAB_RUNTIME_PIN_SCHEMA_VERSION = 1

MJLAB_TRAINING_CHECKPOINT_KIND = "g1_true23_exact_policy_mjlab_training_resume"
MJLAB_TRAINING_LINEAGE_KIND = "g1_true23_exact_policy_mjlab_lineage"
MJLAB_CHECKPOINT_HEADER = "g1_true23_mjlab_training_checkpoint"
MJLAB_CHECKPOINT_ROLE = "training_resume_only"

UNITREE_RL_MJLAB_REPOSITORY = (
    "https://github.com/unitreerobotics/unitree_rl_mjlab.git"
)
UNITREE_RL_MJLAB_COMMIT = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
MJLAB_REPOSITORY = "https://github.com/mujocolab/mjlab.git"
MJLAB_VERSION = "1.2.0"
MJLAB_COMMIT = "5af32e378dcb93c9e881ace83cc5a3f5d373fe60"
MUJOCO_WARP_REPOSITORY = (
    "https://github.com/google-deepmind/mujoco_warp.git"
)
MUJOCO_WARP_VERSION = "3.5.0.2"
MUJOCO_WARP_COMMIT = "5a86ec28aa07741eb2e000d158f4ca4068ec146e"
MUJOCO_VERSION = "3.5.0"
PYTHON_VERSION_POLICY = ">=3.11,<3.12"
TORCH_VERSION = "2.9.0+cu128"
WARP_LANG_VERSION = "1.12.0"

_SHA256_LENGTH = 64
_FILE_MANIFEST_KINDS = frozenset(
    {"source_files", "robot_assets", "motion_dataset"}
)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={missing}, unknown={unknown}"
        )


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def _require_update_count(value: Any, context: str = "update_count") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be integer >= 0")
    return value


def _json_primitive_copy(value: Any, context: str) -> Any:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} must not contain NaN or infinity")
        return value
    if value_type in {list, tuple}:
        return [
            _json_primitive_copy(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{context} keys must be non-empty strings")
            result[key] = _json_primitive_copy(item, f"{context}.{key}")
        return result
    raise ValueError(
        f"{context} contains unsupported value type "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


def _safe_checkpoint_copy(value: Any, context: str) -> Any:
    """Copy only types accepted by PyTorch weights-only deserialization."""

    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} must not contain NaN or infinity")
        return value
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError(f"{context} tensor must use strided layout")
        tensor = value.detach().cpu().contiguous().clone()
        if (
            (tensor.is_floating_point() or tensor.is_complex())
            and not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"{context} tensor contains NaN or infinity")
        return tensor
    if value_type in {list, tuple}:
        return [
            _safe_checkpoint_copy(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str | int, Any] = {}
        for key, item in value.items():
            if type(key) not in {str, int} or (type(key) is str and not key):
                raise ValueError(
                    f"{context} keys must be non-empty strings or integers"
                )
            result[key] = _safe_checkpoint_copy(
                item,
                f"{context}[{key!r}]",
            )
        return result
    raise ValueError(
        f"{context} contains weights-only-unsafe value type "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def runtime_pin_manifest() -> dict[str, Any]:
    """Return immutable source/version pins used by true23 MJLab training."""

    components = {
        "unitree_rl_mjlab": {
            "repository": UNITREE_RL_MJLAB_REPOSITORY,
            "commit": UNITREE_RL_MJLAB_COMMIT,
        },
        "mjlab": {
            "repository": MJLAB_REPOSITORY,
            "version": MJLAB_VERSION,
            "commit": MJLAB_COMMIT,
        },
        "mujoco_warp": {
            "repository": MUJOCO_WARP_REPOSITORY,
            "package": "mujoco-warp",
            "version": MUJOCO_WARP_VERSION,
            "commit": MUJOCO_WARP_COMMIT,
            "commit_authoritative": True,
            "unitree_setup_declared_version": "3.5.0",
            "unitree_install_mode": "no_deps",
        },
        "stable_training_environment": {
            "python_version_policy": PYTHON_VERSION_POLICY,
            "torch": TORCH_VERSION,
            "mujoco": MUJOCO_VERSION,
            "warp_lang": WARP_LANG_VERSION,
            "resolution": (
                "stable_pypi_workaround_for_unavailable_official_lock_dev_build"
            ),
            "fully_frozen_official_mjlab_lock": False,
        },
    }
    return {
        "schema_version": MJLAB_RUNTIME_PIN_SCHEMA_VERSION,
        "kind": "g1_true23_mjlab_runtime_pins",
        "manifest_sha256": _canonical_sha256(components),
        "components": components,
    }


def validate_runtime_pin_manifest(value: Any) -> dict[str, Any]:
    """Reject unpinned/floating MJLab runtime claims."""

    if not isinstance(value, Mapping):
        raise ValueError("runtime pin manifest must be a mapping")
    expected = runtime_pin_manifest()
    if dict(value) != expected:
        raise ValueError("MJLab runtime pins differ from approved exact commits")
    return expected


def build_config_manifest(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    """Hash exact JSON-safe resolved task and runner configuration."""

    if not isinstance(resolved_config, Mapping) or not resolved_config:
        raise ValueError("resolved MJLab config must be a non-empty mapping")
    payload = _json_primitive_copy(
        resolved_config,
        "resolved_mjlab_config",
    )
    return {
        "schema_version": MJLAB_MATERIAL_MANIFEST_SCHEMA_VERSION,
        "kind": "resolved_mjlab_training_config",
        "payload_sha256": _canonical_sha256(payload),
        "payload": payload,
    }


def validate_config_manifest(value: Any) -> dict[str, Any]:
    """Validate config manifest structure and content hash."""

    if not isinstance(value, Mapping):
        raise ValueError("MJLab config manifest must be a mapping")
    _require_exact_keys(
        value,
        {"schema_version", "kind", "payload_sha256", "payload"},
        "MJLab config manifest",
    )
    if (
        value.get("schema_version")
        != MJLAB_MATERIAL_MANIFEST_SCHEMA_VERSION
        or value.get("kind") != "resolved_mjlab_training_config"
    ):
        raise ValueError("MJLab config manifest schema/kind mismatch")
    payload = _json_primitive_copy(
        value.get("payload"),
        "MJLab config manifest payload",
    )
    expected_sha256 = _canonical_sha256(payload)
    if value.get("payload_sha256") != expected_sha256:
        raise ValueError("MJLab config manifest payload_sha256 mismatch")
    return {
        "schema_version": MJLAB_MATERIAL_MANIFEST_SCHEMA_VERSION,
        "kind": "resolved_mjlab_training_config",
        "payload_sha256": expected_sha256,
        "payload": payload,
    }


def _logical_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context} must be non-empty normalized POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"{context} must be normalized repository-relative path")
    return value


def _reject_symlink_path(path: Path, context: str) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{context} may not traverse symlinks")
    resolved = absolute.resolve()
    if not resolved.is_file():
        raise ValueError(f"{context} must be a regular file: {resolved}")
    return resolved


def build_file_manifest(
    files: Mapping[str, str | Path],
    *,
    kind: str,
) -> dict[str, Any]:
    """Hash exact source, asset, or dataset files under stable logical names."""

    if kind not in _FILE_MANIFEST_KINDS:
        raise ValueError(f"unsupported MJLab file manifest kind: {kind!r}")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(f"{kind} manifest requires at least one file")
    if any(type(key) is not str or not key for key in files):
        raise ValueError(f"{kind} manifest keys must be non-empty strings")
    records: list[dict[str, Any]] = []
    for logical_name in sorted(files):
        normalized = _logical_path(
            logical_name,
            f"{kind} logical path",
        )
        resolved = _reject_symlink_path(
            Path(files[logical_name]),
            f"{kind} file {normalized}",
        )
        records.append(
            {
                "logical_path": normalized,
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return {
        "schema_version": MJLAB_MATERIAL_MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "manifest_sha256": _canonical_sha256(records),
        "files": records,
    }


def validate_file_manifest(
    value: Any,
    *,
    kind: str,
) -> dict[str, Any]:
    """Validate a content-addressed file manifest without trusting filenames."""

    if kind not in _FILE_MANIFEST_KINDS:
        raise ValueError(f"unsupported MJLab file manifest kind: {kind!r}")
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind} manifest must be a mapping")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "file_count",
            "total_bytes",
            "manifest_sha256",
            "files",
        },
        f"{kind} manifest",
    )
    if (
        value.get("schema_version")
        != MJLAB_MATERIAL_MANIFEST_SCHEMA_VERSION
        or value.get("kind") != kind
    ):
        raise ValueError(f"{kind} manifest schema/kind mismatch")
    raw_records = value.get("files")
    if (
        not isinstance(raw_records, Sequence)
        or isinstance(raw_records, (str, bytes))
        or not raw_records
    ):
        raise ValueError(f"{kind} manifest files must be non-empty sequence")
    records: list[dict[str, Any]] = []
    previous_path: str | None = None
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"{kind} manifest files[{index}] must be mapping")
        _require_exact_keys(
            raw_record,
            {"logical_path", "size_bytes", "sha256"},
            f"{kind} manifest files[{index}]",
        )
        logical_path = _logical_path(
            raw_record.get("logical_path"),
            f"{kind} manifest files[{index}].logical_path",
        )
        if previous_path is not None and logical_path <= previous_path:
            raise ValueError(
                f"{kind} manifest files must be strictly path-sorted and unique"
            )
        previous_path = logical_path
        size_bytes = raw_record.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ValueError(
                f"{kind} manifest files[{index}].size_bytes must be integer >= 0"
            )
        records.append(
            {
                "logical_path": logical_path,
                "size_bytes": size_bytes,
                "sha256": _require_sha256(
                    raw_record.get("sha256"),
                    f"{kind} manifest files[{index}].sha256",
                ),
            }
        )
    expected_file_count = len(records)
    expected_total_bytes = sum(record["size_bytes"] for record in records)
    expected_sha256 = _canonical_sha256(records)
    if (
        value.get("file_count") != expected_file_count
        or value.get("total_bytes") != expected_total_bytes
        or value.get("manifest_sha256") != expected_sha256
    ):
        raise ValueError(f"{kind} manifest aggregate mismatch")
    return {
        "schema_version": MJLAB_MATERIAL_MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "file_count": expected_file_count,
        "total_bytes": expected_total_bytes,
        "manifest_sha256": expected_sha256,
        "files": records,
    }


def joint_semantics_manifest() -> dict[str, Any]:
    """Bind exact 23-joint action and padded-29 observation semantics."""

    payload = {
        "robot_model": ROBOT_MODEL,
        "required_mode_machine": REQUIRED_MODE_MACHINE,
        "target_dof": TARGET_DOF,
        "hardware_joint_ids": list(HARDWARE_JOINT_IDS),
        "excluded_hardware_joint_ids": list(EXCLUDED_HARDWARE_JOINT_IDS),
        "hardware_joint_names": list(HARDWARE_23_JOINT_NAMES),
        "native_il23_joint_names": list(NATIVE_IL23_JOINT_NAMES),
        "canonical_compact_il23_joint_names": list(
            CANONICAL_COMPACT_IL23_JOINT_NAMES
        ),
        "canonical_il29_joint_names": list(SOURCE_IL29_JOINT_NAMES),
        "canonical_il29_keep_indices": list(SOURCE_IL29_KEEP_INDICES),
        "canonical_il29_missing_indices": list(
            SOURCE_IL29_EXCLUDED_INDICES
        ),
        "native_il23_to_canonical_il29": list(
            NATIVE_IL23_TO_CANONICAL_IL29
        ),
        "native_il23_to_mujoco23": list(ISAACLAB_TO_MUJOCO_DOF),
        "mujoco23_to_native_il23": list(MUJOCO_TO_ISAACLAB_DOF),
        "observation_layout": OBS_LAYOUT_PADDED_IL29,
        "observation_term_order": list(OBSERVATION_TERM_ORDER),
        "history_length": DEPLOYMENT_HISTORY_LENGTH,
        "history_order": HISTORY_ORDER,
        "missing_observation_fill": dict(MISSING_OBSERVATION_FILL),
        "teleop_encoder_input_dim": TELEOP_ENCODER_INPUT_DIM,
        "decoder_input_dim": DEPLOYMENT_DECODER_INPUT_DIM,
        "decoder_output_layout": DECODER_OUTPUT_LAYOUT,
        "runtime_output_masking": False,
    }
    return {
        "schema_version": MJLAB_JOINT_SEMANTICS_SCHEMA_VERSION,
        "kind": "g1_true23_joint_semantics",
        "payload_sha256": _canonical_sha256(payload),
        "payload": payload,
    }


def validate_joint_semantics_manifest(value: Any) -> dict[str, Any]:
    """Reject joint-order or missing-slot semantic drift."""

    if not isinstance(value, Mapping):
        raise ValueError("joint semantics manifest must be a mapping")
    expected = joint_semantics_manifest()
    if dict(value) != expected:
        raise ValueError("joint semantics differ from exact true23 contract")
    return expected


def _tensor_manifest(
    policy_state: Mapping[str, Any],
    *,
    require_finite: bool,
) -> dict[str, Any]:
    if not isinstance(policy_state, Mapping) or not policy_state:
        raise ValueError("policy_state_dict must be a non-empty mapping")
    if any(type(key) is not str or not key for key in policy_state):
        raise ValueError("tensor state keys must be non-empty strings")
    digest = hashlib.sha256()
    descriptors: list[dict[str, Any]] = []
    parameter_count = 0
    for key in sorted(policy_state):
        tensor = policy_state[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"policy_state_dict[{key!r}] must be tensor")
        if tensor.layout != torch.strided:
            raise ValueError(
                f"policy_state_dict[{key!r}] must use strided layout"
            )
        contiguous = tensor.detach().cpu().contiguous()
        if (
            require_finite
            and (contiguous.is_floating_point() or contiguous.is_complex())
            and not bool(torch.isfinite(contiguous).all())
        ):
            raise ValueError(
                f"policy_state_dict[{key!r}] contains NaN or infinity"
            )
        shape = list(contiguous.shape)
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(shape))
        # A dtype reinterpretation directly on a rank-0 tensor is rejected by
        # PyTorch when the destination element size differs (for example the
        # scalar int64 count in an RSL observation normalizer).  Flattening
        # first preserves the exact storage bytes and handles every rank,
        # including scalars and empty tensors.
        digest.update(
            contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
        descriptors.append(
            {
                "key": key,
                "dtype": str(contiguous.dtype),
                "shape": shape,
                "parameter_count": contiguous.numel(),
            }
        )
        parameter_count += contiguous.numel()
    return {
        "state_sha256": digest.hexdigest(),
        "tensor_count": len(descriptors),
        "parameter_count": parameter_count,
        "keys_sha256": _canonical_sha256(
            [descriptor["key"] for descriptor in descriptors]
        ),
        "tensors": descriptors,
    }


def _validate_tensor_manifest(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be mapping")
    _require_exact_keys(
        value,
        {
            "state_sha256",
            "tensor_count",
            "parameter_count",
            "keys_sha256",
            "tensors",
        },
        context,
    )
    _require_sha256(value.get("state_sha256"), f"{context}.state_sha256")
    _require_sha256(value.get("keys_sha256"), f"{context}.keys_sha256")
    raw_tensors = value.get("tensors")
    if (
        not isinstance(raw_tensors, Sequence)
        or isinstance(raw_tensors, (str, bytes))
        or not raw_tensors
    ):
        raise ValueError(f"{context}.tensors must be non-empty sequence")
    descriptors: list[dict[str, Any]] = []
    previous_key: str | None = None
    parameter_count = 0
    for index, item in enumerate(raw_tensors):
        if not isinstance(item, Mapping):
            raise ValueError(f"{context}.tensors[{index}] must be mapping")
        _require_exact_keys(
            item,
            {"key", "dtype", "shape", "parameter_count"},
            f"{context}.tensors[{index}]",
        )
        key = item.get("key")
        if type(key) is not str or not key:
            raise ValueError(
                f"{context}.tensors[{index}].key must be non-empty string"
            )
        if previous_key is not None and key <= previous_key:
            raise ValueError(f"{context}.tensors must be sorted and unique")
        previous_key = key
        dtype = item.get("dtype")
        if not isinstance(dtype, str) or not dtype.startswith("torch."):
            raise ValueError(f"{context}.tensors[{index}].dtype invalid")
        shape = item.get("shape")
        if (
            not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes))
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in shape
            )
        ):
            raise ValueError(f"{context}.tensors[{index}].shape invalid")
        count = item.get("parameter_count")
        expected_count = math.prod(shape)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count != expected_count
        ):
            raise ValueError(
                f"{context}.tensors[{index}].parameter_count mismatch"
            )
        descriptors.append(
            {
                "key": key,
                "dtype": dtype,
                "shape": list(shape),
                "parameter_count": count,
            }
        )
        parameter_count += count
    if (
        value.get("tensor_count") != len(descriptors)
        or value.get("parameter_count") != parameter_count
        or value.get("keys_sha256")
        != _canonical_sha256(
            [descriptor["key"] for descriptor in descriptors]
        )
    ):
        raise ValueError(f"{context} aggregate mismatch")
    return {
        "state_sha256": value["state_sha256"],
        "tensor_count": len(descriptors),
        "parameter_count": parameter_count,
        "keys_sha256": value["keys_sha256"],
        "tensors": descriptors,
    }


def _expected_policy_structure(reference_profile: str) -> list[dict[str, Any]]:
    encoder_dims = (
        TELEOP_ENCODER_INPUT_DIM,
        2048,
        1024,
        512,
        512,
        64,
    )
    decoder_dims = decoder_layer_dims_for_profile(reference_profile)
    descriptors: list[dict[str, Any]] = []

    def add_family(prefix: str, dimensions: Sequence[int]) -> None:
        for position, (input_dim, output_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True)
        ):
            layer_index = position * 2
            descriptors.extend(
                (
                    {
                        "key": f"{prefix}.{layer_index}.bias",
                        "dtype": "torch.float32",
                        "shape": [output_dim],
                        "parameter_count": output_dim,
                    },
                    {
                        "key": f"{prefix}.{layer_index}.weight",
                        "dtype": "torch.float32",
                        "shape": [output_dim, input_dim],
                        "parameter_count": output_dim * input_dim,
                    },
                )
            )

    add_family("actor_module.encoders.teleop.module", encoder_dims)
    add_family("actor_module.decoders.g1_dyn.module", decoder_dims)
    descriptors.append(
        {
            "key": "std",
            "dtype": "torch.float32",
            "shape": [TARGET_DOF],
            "parameter_count": TARGET_DOF,
        }
    )
    return sorted(descriptors, key=lambda descriptor: descriptor["key"])


def _validate_policy_against_initial_manifest(
    policy_state: Mapping[str, Any],
    initial_manifest: Mapping[str, Any],
    *,
    reference_profile: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    expected = _validate_tensor_manifest(
        initial_manifest,
        "initial policy tensor manifest",
    )
    current = _tensor_manifest(policy_state, require_finite=True)
    manifest_structure = [
        {
            "key": item["key"],
            "dtype": item["dtype"],
            "shape": item["shape"],
            "parameter_count": item["parameter_count"],
        }
        for item in expected["tensors"]
    ]
    expected_structure = _expected_policy_structure(reference_profile)
    if manifest_structure != expected_structure:
        raise ValueError(
            "initial policy tensor manifest differs from exact SONIC key set"
        )
    current_structure = [
        {
            "key": item["key"],
            "dtype": item["dtype"],
            "shape": item["shape"],
            "parameter_count": item["parameter_count"],
        }
        for item in current["tensors"]
    ]
    if current_structure != expected_structure:
        raise ValueError(
            "policy_state_dict keys/shapes/dtypes differ from exact warm-start"
        )
    inspected_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy_state},
        reference_profile=reference_profile,
    )
    if inspected_hash != current["state_sha256"]:
        raise ValueError("policy tensor hash disagrees with exact-policy audit")
    copied = {
        key: tensor.detach().cpu().contiguous().clone()
        for key, tensor in policy_state.items()
    }
    return copied, current


def build_mjlab_training_lineage(
    warm_start_checkpoint_path: str | Path,
    *,
    resolved_config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    asset_manifest: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one exact warm-start and all training material before first step."""

    warm_start_path = Path(warm_start_checkpoint_path).resolve()
    checkpoint = load_safe_true23_checkpoint(warm_start_path)
    if checkpoint_stage(checkpoint) != INITIALIZATION_STAGE:
        raise ValueError(
            "MJLab exact-policy training must start from initialization checkpoint"
        )
    initialization_report = checkpoint.get(
        "g1_23dof_initialization_report"
    )
    metadata = checkpoint.get("g1_23dof_metadata")
    if not isinstance(initialization_report, Mapping):
        raise ValueError("warm-start lacks initialization report")
    if not isinstance(metadata, Mapping):
        raise ValueError("warm-start lacks true23 metadata")
    reference_profile = initialization_report.get("reference_profile")
    if (
        not isinstance(reference_profile, str)
        or metadata.get("reference_profile") != reference_profile
    ):
        raise ValueError("warm-start reference profile mismatch")
    initial_policy_hash = inspect_true23_policy_state(
        checkpoint,
        reference_profile=reference_profile,
    )
    if (
        initialization_report.get("initialization_only") is not True
        or initialization_report.get("initial_policy_state_sha256")
        != initial_policy_hash
    ):
        raise ValueError("warm-start initial policy hash is not exact")
    source_checkpoint_sha256 = initialization_report.get(
        "source_checkpoint_sha256"
    )
    release = APPROVED_WARM_START_RELEASES.get(source_checkpoint_sha256)
    if (
        release is None
        or release.get("source_family")
        != initialization_report.get("source_family")
        or release.get("source_revision")
        != initialization_report.get("source_revision")
        or release.get("reference_profile") != reference_profile
        or release.get("initial_policy_state_sha256")
        != initial_policy_hash
    ):
        raise ValueError("warm-start is not exact approved release conversion")
    initial_manifest = _tensor_manifest(
        checkpoint["policy_state_dict"],
        require_finite=True,
    )
    if initial_manifest["state_sha256"] != initial_policy_hash:
        raise ValueError("warm-start policy manifest hash mismatch")

    body = {
        "schema_version": MJLAB_TRAINING_LINEAGE_SCHEMA_VERSION,
        "kind": MJLAB_TRAINING_LINEAGE_KIND,
        "checkpoint_role": MJLAB_CHECKPOINT_ROLE,
        "runtime_pins": runtime_pin_manifest(),
        "warm_start": {
            "checkpoint_filename": warm_start_path.name,
            "checkpoint_sha256": sha256_file(warm_start_path),
            "source_family": initialization_report.get("source_family"),
            "source_revision": initialization_report.get("source_revision"),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "reference_profile": reference_profile,
            "initial_policy_state_sha256": initial_policy_hash,
            "initial_policy_tensor_manifest": initial_manifest,
            "training_start_update_count": 0,
            "optimizer_reused": False,
            "value_model_reused": False,
        },
        "materials": {
            "resolved_config": build_config_manifest(resolved_config),
            "source_files": validate_file_manifest(
                source_manifest,
                kind="source_files",
            ),
            "robot_assets": validate_file_manifest(
                asset_manifest,
                kind="robot_assets",
            ),
            "motion_dataset": validate_file_manifest(
                dataset_manifest,
                kind="motion_dataset",
            ),
            "joint_semantics": joint_semantics_manifest(),
        },
        "minimum_policy_updates": MINIMUM_TRAINING_UPDATES,
        "policy_training_mode": "direct_exact_sonic_encoder_decoder",
        "stock_rsl_posthoc_conversion": False,
        "deployment_ready": False,
        "promotion_eligible": False,
    }
    return {
        **body,
        "lineage_sha256": _canonical_sha256(body),
    }


def validate_mjlab_training_lineage(value: Any) -> dict[str, Any]:
    """Validate immutable lineage embedded in each MJLab resume checkpoint."""

    if not isinstance(value, Mapping):
        raise ValueError("MJLab training lineage must be mapping")
    expected_keys = {
        "schema_version",
        "kind",
        "checkpoint_role",
        "runtime_pins",
        "warm_start",
        "materials",
        "minimum_policy_updates",
        "policy_training_mode",
        "stock_rsl_posthoc_conversion",
        "deployment_ready",
        "promotion_eligible",
        "lineage_sha256",
    }
    _require_exact_keys(value, expected_keys, "MJLab training lineage")
    if (
        value.get("schema_version")
        != MJLAB_TRAINING_LINEAGE_SCHEMA_VERSION
        or value.get("kind") != MJLAB_TRAINING_LINEAGE_KIND
        or value.get("checkpoint_role") != MJLAB_CHECKPOINT_ROLE
        or value.get("minimum_policy_updates")
        != MINIMUM_TRAINING_UPDATES
        or value.get("policy_training_mode")
        != "direct_exact_sonic_encoder_decoder"
        or value.get("stock_rsl_posthoc_conversion") is not False
        or value.get("deployment_ready") is not False
        or value.get("promotion_eligible") is not False
    ):
        raise ValueError("MJLab training lineage safety contract mismatch")
    runtime_pins = validate_runtime_pin_manifest(value.get("runtime_pins"))

    warm_start = value.get("warm_start")
    if not isinstance(warm_start, Mapping):
        raise ValueError("MJLab training lineage warm_start must be mapping")
    _require_exact_keys(
        warm_start,
        {
            "checkpoint_filename",
            "checkpoint_sha256",
            "source_family",
            "source_revision",
            "source_checkpoint_sha256",
            "reference_profile",
            "initial_policy_state_sha256",
            "initial_policy_tensor_manifest",
            "training_start_update_count",
            "optimizer_reused",
            "value_model_reused",
        },
        "MJLab training lineage warm_start",
    )
    checkpoint_filename = warm_start.get("checkpoint_filename")
    if (
        type(checkpoint_filename) is not str
        or not checkpoint_filename
        or Path(checkpoint_filename).name != checkpoint_filename
    ):
        raise ValueError("warm_start checkpoint_filename must be basename")
    _require_sha256(
        warm_start.get("checkpoint_sha256"),
        "warm_start.checkpoint_sha256",
    )
    source_checkpoint_sha256 = _require_sha256(
        warm_start.get("source_checkpoint_sha256"),
        "warm_start.source_checkpoint_sha256",
    )
    initial_policy_hash = _require_sha256(
        warm_start.get("initial_policy_state_sha256"),
        "warm_start.initial_policy_state_sha256",
    )
    release = APPROVED_WARM_START_RELEASES.get(source_checkpoint_sha256)
    if (
        release is None
        or warm_start.get("source_family") != release["source_family"]
        or warm_start.get("source_revision") != release["source_revision"]
        or warm_start.get("reference_profile")
        != release["reference_profile"]
        or initial_policy_hash != release["initial_policy_state_sha256"]
        or warm_start.get("training_start_update_count") != 0
        or warm_start.get("optimizer_reused") is not False
        or warm_start.get("value_model_reused") is not False
    ):
        raise ValueError("MJLab warm-start lineage is not approved")
    initial_policy_manifest = _validate_tensor_manifest(
        warm_start.get("initial_policy_tensor_manifest"),
        "warm_start.initial_policy_tensor_manifest",
    )
    if initial_policy_manifest["state_sha256"] != initial_policy_hash:
        raise ValueError("MJLab warm-start policy manifest hash mismatch")

    materials = value.get("materials")
    if not isinstance(materials, Mapping):
        raise ValueError("MJLab training lineage materials must be mapping")
    _require_exact_keys(
        materials,
        {
            "resolved_config",
            "source_files",
            "robot_assets",
            "motion_dataset",
            "joint_semantics",
        },
        "MJLab training lineage materials",
    )
    normalized_materials = {
        "resolved_config": validate_config_manifest(
            materials.get("resolved_config")
        ),
        "source_files": validate_file_manifest(
            materials.get("source_files"),
            kind="source_files",
        ),
        "robot_assets": validate_file_manifest(
            materials.get("robot_assets"),
            kind="robot_assets",
        ),
        "motion_dataset": validate_file_manifest(
            materials.get("motion_dataset"),
            kind="motion_dataset",
        ),
        "joint_semantics": validate_joint_semantics_manifest(
            materials.get("joint_semantics")
        ),
    }
    normalized_body = {
        "schema_version": MJLAB_TRAINING_LINEAGE_SCHEMA_VERSION,
        "kind": MJLAB_TRAINING_LINEAGE_KIND,
        "checkpoint_role": MJLAB_CHECKPOINT_ROLE,
        "runtime_pins": runtime_pins,
        "warm_start": {
            **dict(warm_start),
            "initial_policy_tensor_manifest": initial_policy_manifest,
        },
        "materials": normalized_materials,
        "minimum_policy_updates": MINIMUM_TRAINING_UPDATES,
        "policy_training_mode": "direct_exact_sonic_encoder_decoder",
        "stock_rsl_posthoc_conversion": False,
        "deployment_ready": False,
        "promotion_eligible": False,
    }
    expected_lineage_sha256 = _canonical_sha256(normalized_body)
    if value.get("lineage_sha256") != expected_lineage_sha256:
        raise ValueError("MJLab training lineage_sha256 mismatch")
    return {
        **normalized_body,
        "lineage_sha256": expected_lineage_sha256,
    }


def _training_gate(
    *,
    update_count: int,
    policy_state_sha256: str,
    initial_policy_state_sha256: str,
) -> dict[str, Any]:
    updates = _require_update_count(update_count)
    changed = policy_state_sha256 != initial_policy_state_sha256
    minimum_reached = updates >= MINIMUM_TRAINING_UPDATES
    return {
        "minimum_policy_updates": MINIMUM_TRAINING_UPDATES,
        "updates_since_warm_start": updates,
        "minimum_updates_reached": minimum_reached,
        "policy_changed_from_initialization": changed,
        "simulation_candidate_review_allowed": minimum_reached and changed,
        "deployment_ready": False,
        "promotion_eligible": False,
    }


def _checkpoint_header() -> dict[str, Any]:
    return {
        "schema_version": MJLAB_TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "kind": MJLAB_TRAINING_CHECKPOINT_KIND,
        "checkpoint_role": MJLAB_CHECKPOINT_ROLE,
        "weights_only_safe": True,
        "resume_state_included": True,
        "deployment_ready": False,
        "promotion_eligible": False,
        "stock_rsl_posthoc_conversion": False,
    }


def _validate_optimizer_state(value: Any) -> dict[str | int, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("optimizer_state_dict must be mapping")
    if set(value) != {"state", "param_groups"}:
        raise ValueError(
            "optimizer_state_dict must contain only state and param_groups"
        )
    copied = _safe_checkpoint_copy(value, "optimizer_state_dict")
    state = copied.get("state")
    param_groups = copied.get("param_groups")
    if not isinstance(state, Mapping):
        raise ValueError("optimizer_state_dict.state must be mapping")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError(
            "optimizer_state_dict.param_groups must be non-empty list"
        )
    for index, group in enumerate(param_groups):
        if not isinstance(group, Mapping):
            raise ValueError(
                f"optimizer_state_dict.param_groups[{index}] must be mapping"
            )
        params = group.get("params")
        if (
            not isinstance(params, list)
            or any(
                isinstance(parameter, bool)
                or not isinstance(parameter, int)
                or parameter < 0
                for parameter in params
            )
        ):
            raise ValueError(
                f"optimizer_state_dict.param_groups[{index}].params invalid"
            )
    return copied


def _validate_critic_state(
    value: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("critic_state_dict must be a non-empty mapping")
    manifest = _tensor_manifest(value, require_finite=True)
    copied = {
        key: tensor.detach().cpu().contiguous().clone()
        for key, tensor in value.items()
    }
    return copied, manifest


def _validate_trainer_state(
    value: Any,
    *,
    update_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("trainer_state must be mapping")
    if any(type(key) is not str or not key for key in value):
        raise ValueError("trainer_state keys must be non-empty strings")
    required_keys = {
        "completed_update_count",
        "current_learning_iteration",
        "env_common_step_counter",
    }
    missing = sorted(required_keys - set(value))
    if missing:
        raise ValueError(f"trainer_state missing required keys: {missing}")
    copied = _safe_checkpoint_copy(value, "trainer_state")
    completed_updates = _require_update_count(
        copied.get("completed_update_count"),
        "trainer_state.completed_update_count",
    )
    current_iteration = _require_update_count(
        copied.get("current_learning_iteration"),
        "trainer_state.current_learning_iteration",
    )
    _require_update_count(
        copied.get("env_common_step_counter"),
        "trainer_state.env_common_step_counter",
    )
    if (
        completed_updates != update_count
        or current_iteration != update_count
    ):
        raise ValueError(
            "trainer_state completed update/iteration counters must equal "
            "checkpoint update_count"
        )
    return copied


def _tensor_structure(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "key": item["key"],
            "dtype": item["dtype"],
            "shape": item["shape"],
            "parameter_count": item["parameter_count"],
        }
        for item in manifest["tensors"]
    ]


def build_mjlab_training_checkpoint(
    *,
    policy_state_dict: Mapping[str, Any],
    critic_state_dict: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any],
    update_count: int,
    trainer_state: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Build complete weights-only-safe PPO resume state; never promotion."""

    updates = _require_update_count(update_count)
    normalized_lineage = validate_mjlab_training_lineage(lineage)
    warm_start = normalized_lineage["warm_start"]
    policy_state, policy_manifest = _validate_policy_against_initial_manifest(
        policy_state_dict,
        warm_start["initial_policy_tensor_manifest"],
        reference_profile=warm_start["reference_profile"],
    )
    critic_state, critic_manifest = _validate_critic_state(
        critic_state_dict
    )
    optimizer_state = _validate_optimizer_state(optimizer_state_dict)
    normalized_trainer_state = _validate_trainer_state(
        trainer_state,
        update_count=updates,
    )
    gate = _training_gate(
        update_count=updates,
        policy_state_sha256=policy_manifest["state_sha256"],
        initial_policy_state_sha256=warm_start[
            "initial_policy_state_sha256"
        ],
    )
    return {
        MJLAB_CHECKPOINT_HEADER: _checkpoint_header(),
        "policy_state_dict": policy_state,
        "critic_state_dict": critic_state,
        "optimizer_state_dict": optimizer_state,
        "update_count": updates,
        "trainer_state": normalized_trainer_state,
        "policy_state_sha256": policy_manifest["state_sha256"],
        "critic_state_sha256": critic_manifest["state_sha256"],
        "lineage": normalized_lineage,
        "lineage_sha256": normalized_lineage["lineage_sha256"],
        "training_gate": gate,
    }


def validate_mjlab_training_checkpoint(
    checkpoint: Any,
    *,
    expected_lineage_sha256: str | None = None,
    minimum_update_count: int | None = None,
) -> Mapping[str, Any]:
    """Validate exact resume schema, policy, lineage, optimizer, and counter."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("MJLab training checkpoint root must be mapping")
    expected_root_keys = {
        MJLAB_CHECKPOINT_HEADER,
        "policy_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "update_count",
        "trainer_state",
        "policy_state_sha256",
        "critic_state_sha256",
        "lineage",
        "lineage_sha256",
        "training_gate",
    }
    _require_exact_keys(
        checkpoint,
        expected_root_keys,
        "MJLab training checkpoint",
    )
    header = checkpoint.get(MJLAB_CHECKPOINT_HEADER)
    if not isinstance(header, Mapping) or dict(header) != _checkpoint_header():
        raise ValueError("MJLab training checkpoint header mismatch")
    lineage = validate_mjlab_training_lineage(checkpoint.get("lineage"))
    lineage_sha256 = lineage["lineage_sha256"]
    if checkpoint.get("lineage_sha256") != lineage_sha256:
        raise ValueError("checkpoint lineage_sha256 mismatch")
    if expected_lineage_sha256 is not None:
        _require_sha256(
            expected_lineage_sha256,
            "expected_lineage_sha256",
        )
        if lineage_sha256 != expected_lineage_sha256:
            raise ValueError("checkpoint lineage differs from current run")
    update_count = _require_update_count(checkpoint.get("update_count"))
    if minimum_update_count is not None:
        minimum = _require_update_count(
            minimum_update_count,
            "minimum_update_count",
        )
        if update_count < minimum:
            raise ValueError(
                "checkpoint update_count is older than required resume point"
            )
    _, policy_manifest = _validate_policy_against_initial_manifest(
        checkpoint.get("policy_state_dict"),
        lineage["warm_start"]["initial_policy_tensor_manifest"],
        reference_profile=lineage["warm_start"]["reference_profile"],
    )
    if checkpoint.get("policy_state_sha256") != policy_manifest["state_sha256"]:
        raise ValueError("checkpoint policy_state_sha256 mismatch")
    _, critic_manifest = _validate_critic_state(
        checkpoint.get("critic_state_dict")
    )
    if checkpoint.get("critic_state_sha256") != critic_manifest["state_sha256"]:
        raise ValueError("checkpoint critic_state_sha256 mismatch")
    _validate_optimizer_state(checkpoint.get("optimizer_state_dict"))
    _validate_trainer_state(
        checkpoint.get("trainer_state"),
        update_count=update_count,
    )
    expected_gate = _training_gate(
        update_count=update_count,
        policy_state_sha256=policy_manifest["state_sha256"],
        initial_policy_state_sha256=lineage["warm_start"][
            "initial_policy_state_sha256"
        ],
    )
    gate = checkpoint.get("training_gate")
    if not isinstance(gate, Mapping) or dict(gate) != expected_gate:
        raise ValueError("checkpoint training_gate mismatch")
    return checkpoint


def save_mjlab_training_checkpoint(
    output_path: str | Path,
    *,
    policy_state_dict: Mapping[str, Any],
    critic_state_dict: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any],
    update_count: int,
    trainer_state: Mapping[str, Any],
    lineage: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    """Atomically save and weights-only reload-verify one resume checkpoint."""

    if type(overwrite) is not bool:
        raise TypeError("overwrite must be explicit bool")
    output = Path(output_path).resolve()
    if ".promotion." in output.name or output.name.endswith(".promotion.pt"):
        raise ValueError(
            "MJLab training resume checkpoint may not use promotion filename"
        )
    if output.is_symlink():
        raise ValueError("MJLab training checkpoint output may not be symlink")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite MJLab training checkpoint: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = build_mjlab_training_checkpoint(
        policy_state_dict=policy_state_dict,
        critic_state_dict=critic_state_dict,
        optimizer_state_dict=optimizer_state_dict,
        update_count=update_count,
        trainer_state=trainer_state,
        lineage=lineage,
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(checkpoint, temporary)
        loaded = torch.load(
            temporary,
            map_location="cpu",
            weights_only=True,
        )
        validate_mjlab_training_checkpoint(
            loaded,
            expected_lineage_sha256=lineage["lineage_sha256"],
        )
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite MJLab training checkpoint: {output}"
            )
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def load_mjlab_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_lineage_sha256: str | None = None,
    minimum_update_count: int | None = None,
    map_location: Any = "cpu",
) -> Mapping[str, Any]:
    """Load resume state without arbitrary pickle globals."""

    path = Path(checkpoint_path)
    if path.is_symlink():
        raise ValueError("MJLab training checkpoint may not be symlink")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MJLab training checkpoint missing: {path}")
    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(
            "MJLab training checkpoint is not weights-only-safe"
        ) from exc
    return validate_mjlab_training_checkpoint(
        checkpoint,
        expected_lineage_sha256=expected_lineage_sha256,
        minimum_update_count=minimum_update_count,
    )


def restore_mjlab_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    policy_module: Any,
    critic_module: Any,
    optimizer: Any,
    expected_lineage: Mapping[str, Any],
    minimum_update_count: int | None = None,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Strictly restore actor, critic, optimizer, and trainer counters."""

    lineage = validate_mjlab_training_lineage(expected_lineage)
    checkpoint = load_mjlab_training_checkpoint(
        checkpoint_path,
        expected_lineage_sha256=lineage["lineage_sha256"],
        minimum_update_count=minimum_update_count,
        map_location=map_location,
    )
    if not hasattr(policy_module, "state_dict") or not hasattr(
        policy_module,
        "load_state_dict",
    ):
        raise TypeError("policy_module must implement state_dict/load_state_dict")
    if not hasattr(optimizer, "load_state_dict"):
        raise TypeError("optimizer must implement load_state_dict")
    if not hasattr(critic_module, "state_dict") or not hasattr(
        critic_module,
        "load_state_dict",
    ):
        raise TypeError("critic_module must implement state_dict/load_state_dict")
    _validate_policy_against_initial_manifest(
        policy_module.state_dict(),
        lineage["warm_start"]["initial_policy_tensor_manifest"],
        reference_profile=lineage["warm_start"]["reference_profile"],
    )
    current_critic_manifest = _tensor_manifest(
        critic_module.state_dict(),
        require_finite=True,
    )
    checkpoint_critic_manifest = _tensor_manifest(
        checkpoint["critic_state_dict"],
        require_finite=True,
    )
    if _tensor_structure(current_critic_manifest) != _tensor_structure(
        checkpoint_critic_manifest
    ):
        raise ValueError(
            "critic_module keys/shapes/dtypes differ from checkpoint"
        )
    policy_module.load_state_dict(
        checkpoint["policy_state_dict"],
        strict=True,
    )
    critic_module.load_state_dict(
        checkpoint["critic_state_dict"],
        strict=True,
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restored_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy_module.state_dict()},
        reference_profile=lineage["warm_start"]["reference_profile"],
    )
    if restored_hash != checkpoint["policy_state_sha256"]:
        raise ValueError("restored policy hash differs from checkpoint")
    restored_critic_manifest = _tensor_manifest(
        critic_module.state_dict(),
        require_finite=True,
    )
    if (
        restored_critic_manifest["state_sha256"]
        != checkpoint["critic_state_sha256"]
    ):
        raise ValueError("restored critic hash differs from checkpoint")
    return dict(checkpoint["trainer_state"])


def checkpoint_training_summary(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Return primitive reader-facing status without deployment claims."""

    validated = validate_mjlab_training_checkpoint(checkpoint)
    return {
        "kind": MJLAB_TRAINING_CHECKPOINT_KIND,
        "checkpoint_role": MJLAB_CHECKPOINT_ROLE,
        "update_count": validated["update_count"],
        "policy_state_sha256": validated["policy_state_sha256"],
        "critic_state_sha256": validated["critic_state_sha256"],
        "lineage_sha256": validated["lineage_sha256"],
        "trainer_state": dict(validated["trainer_state"]),
        "runtime_pins": validated["lineage"]["runtime_pins"],
        "training_gate": dict(validated["training_gate"]),
        "deployment_ready": False,
        "promotion_eligible": False,
    }
