"""Exact teacher-controlled q9=9 bootstrap collection for SONIC true23.

This module creates behavior-cloning bootstrap evidence only.  The original
selected iteration-21204 PyTorch actor controls one 510-transition DadDance
rollout while its frozen ONNX export is checked on every actor observation.
The causal model-250 encoder supplies token64; its decoder is never called for
an action.  Reset-padded rows 0..9 are retained and explicitly marked, while
rows 10..509 contain a real H10 history.

Nothing here admits support, creates DAgger/on-policy data, trains a policy, or
authorizes promotion, deployment, networking, or hardware actuation.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
    CAUSAL_HISTORY_ANCHOR_INDEX,
    CONTROL_DT_S,
    DAD_DANCE_FRAME_COUNT,
    DAD_DANCE_RELATIVE_PATH,
    DAD_DANCE_SHA256,
    validate_dad_dance_motion_file,
)
from gear_sonic.envs.mjlab.sonic_true23 import (
    pad_native_il23_to_canonical_il29,
    padded_previous_action,
)
from gear_sonic.scripts.train_g1_true23_native124_selected_v2_ankle import (
    FIXED_GPU,
    FIXED_SEED,
    resolve_rsl_runtime_binding,
)
from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
    safe_tree_sha256,
    sha256_file,
    tensor_state_sha256,
)
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_mjlab_diagnostic_onnx import (
    verify_mjlab_diagnostic_onnx,
)
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTOR_STATE_SHA256,
    CHECKPOINT_SHA256,
    ONNX_SHA256,
    Native124Checkpoint21204Policy,
    load_checkpoint21204_binding,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_teacher_support import (
    SUPPORT_CONFIG_SHA256,
    load_teacher_support_contract,
)

CONTRACT_RELATIVE_PATH = Path("gear_sonic/config/sim_validation/g1_true23_native124_21204_bootstrap_v1.json")
CONTRACT_SHA256 = "38952e2459437c7a19e062384a8f587f0e3b69f2ada8eef223c63bdb0f9e2ce9"
CONTRACT_KIND = "g1_true23_native124_21204_teacher_bootstrap_contract_v1"
MANIFEST_KIND = "g1_true23_native124_21204_teacher_bootstrap_manifest_v1"
FAILURE_MANIFEST_KIND = "g1_true23_native124_21204_teacher_bootstrap_failure_v1"
SCHEMA_VERSION = 1

TOTAL_ROWS = 510
RESET_PREFIX_ROWS = 10
REAL_H10_ROWS = 500
INITIAL_Q9 = CAUSAL_HISTORY_ANCHOR_INDEX
LAST_ACTION_Q9 = INITIAL_Q9 + TOTAL_ROWS - 1
POST_COLLECTION_Q9 = INITIAL_Q9 + TOTAL_ROWS
EPISODE_TIMEOUT_STEPS = TOTAL_ROWS + 1
PARITY_ATOL = 1.0e-5
ACTION_LINK_ATOL = 1.0e-5
MODEL_REPLAY_ATOL = 1.0e-6
DEVICE = "cuda:0"

COMPOSITE_CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_native124_21204_composite_mjlab_v1.json"
)
COMPOSITE_CONTRACT_SHA256 = "841eb09a6210fc0753cfdc795570a758811baa3eeb458f0bc5e30e624667b2ff"

CAUSAL_BUNDLE_RELATIVE_PATH = Path("artifacts/g1_true23/causal_model_250_20260803")
CAUSAL_ENCODER_FILENAME = "causal_model_250.encoder.onnx"
CAUSAL_DECODER_FILENAME = "causal_model_250.decoder.onnx"
CAUSAL_METADATA_FILENAME = "causal_model_250.diagnostic.json"
CAUSAL_SOURCE_CHECKPOINT_FILENAME = "causal_model_250.pt"
CAUSAL_ENCODER_SHA256 = "733353148bef1eb8dd83a96416b7a89f0b5c3530ceb9e0cec9c25fdb04f56ff2"
CAUSAL_DECODER_SHA256 = "f18139aa5b98619a5d0e84a9de8378a5081c556d427b20fdc55e4fe917549740"
CAUSAL_METADATA_FILE_SHA256 = "b284f03dca370e48f3af590856aaed57e827bbbb5f68868b675c71c43c19f3b8"
CAUSAL_SOURCE_CHECKPOINT_SHA256 = "85bd6de646905a44190dbf32c79737082bb604ab007a90a62e4fd2fdeeee6bd9"
CAUSAL_REFERENCE_PROFILE = "true23_causal_step1_history_0p02s_v1"

ENCODER_DIM = 267
TOKEN_DIM = 64
PROPRIO_DIM = 930
DECODER_DIM = TOKEN_DIM + PROPRIO_DIM
SELECTED_OBSERVATION_DIM = 124
ACTION_DIM = 23
TOKENIZER_DIM = ENCODER_DIM + 1

TOKENIZER_CORRUPTION_ENABLED = True
POLICY_CORRUPTION_ENABLED = True
ACTOR_CORRUPTION_ENABLED = False
POLICY_TERM_WIDTHS: Mapping[str, int] = {
    "base_ang_vel": 3,
    "joint_pos_rel": 29,
    "joint_vel": 29,
    "previous_action": 29,
    "projected_gravity": 3,
}
EXECUTED_SOURCE_TREE_RELATIVE_PATHS = (
    Path("gear_sonic/envs/mjlab"),
    Path("gear_sonic/trl/mjlab"),
    Path("external_dependencies/mjlab/src/mjlab"),
    Path("external_dependencies/unitree_rl_mjlab/src"),
)
EXECUTED_SOURCE_FILE_RELATIVE_PATHS = (
    Path("gear_sonic/scripts/collect_g1_true23_native124_21204_bootstrap_mjlab.py"),
    Path("gear_sonic/scripts/train_g1_true23_native124_selected_v2_ankle.py"),
    Path("gear_sonic/utils/g1_23dof_artifact.py"),
    Path("gear_sonic/utils/g1_23dof_contract.py"),
    Path("gear_sonic/utils/g1_23dof_mjlab_diagnostic_onnx.py"),
    Path("gear_sonic/utils/g1_23dof_native124_21204_adapter.py"),
    Path("gear_sonic/utils/g1_23dof_safe_target_transform.py"),
    Path("gear_sonic/utils/g1_true23_native124_21204_bootstrap_mjlab.py"),
    Path("gear_sonic/utils/g1_true23_native124_21204_composite_mjlab.py"),
    Path("gear_sonic/utils/g1_true23_native124_selected_source_nominal_qualification.py"),
    Path("gear_sonic/utils/g1_true23_native124_selected_v2_ankle_evaluation.py"),
    Path("gear_sonic/utils/g1_true23_teacher_support.py"),
)

_COUNT_ARRAY_NAMES = (
    "raw_clip_coordinate_count",
    "target_inner_margin_violation_count",
    "target_soft_limit_violation_count",
    "actuator_target_soft_limit_violation_count",
    "measured_soft_limit_violation_count",
    "joint_velocity_limit_violation_count",
    "actuator_force_penalty_coordinate_count",
    "actuator_force_hard_limit_coordinate_count",
    "termination_name_count",
    "nonfinite_count",
)


def _spec(dtype: Any, *tail: int) -> tuple[np.dtype[Any], tuple[int, ...]]:
    return np.dtype(dtype), (TOTAL_ROWS, *tail)


ARRAY_SPECS: Mapping[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "row_index": _spec(np.int64),
    "reset_prefix": _spec(np.bool_),
    "steady_history": _spec(np.bool_),
    "history_depth": _spec(np.int64),
    "reset_padding_count": _spec(np.int64),
    "q9_reference_index": _spec(np.int64),
    "control_state_index": _spec(np.int64),
    "next_control_state_index": _spec(np.int64),
    "encoder267": _spec(np.float32, ENCODER_DIM),
    "token64": _spec(np.float32, TOKEN_DIM),
    "proprio930": _spec(np.float32, PROPRIO_DIM),
    "decoder994": _spec(np.float32, DECODER_DIM),
    "selected_observation124": _spec(np.float32, SELECTED_OBSERVATION_DIM),
    "teacher_action_pt_hardware23": _spec(np.float32, ACTION_DIM),
    "teacher_action_onnx_hardware23": _spec(np.float32, ACTION_DIM),
    "teacher_parity_max_abs": _spec(np.float32),
    "teacher_action_applied_hardware23": _spec(np.float32, ACTION_DIM),
    "teacher_candidate_target_hardware23": _spec(np.float32, ACTION_DIM),
    "teacher_label_raw_native23": _spec(np.float32, ACTION_DIM),
    "teacher_applied_safe_native23": _spec(np.float32, ACTION_DIM),
    "teacher_final_target_hardware23": _spec(np.float32, ACTION_DIM),
    "raw_clip_mask_native23": _spec(np.bool_, ACTION_DIM),
    "next_joint_position_hardware23": _spec(np.float32, ACTION_DIM),
    "next_joint_velocity_hardware23": _spec(np.float32, ACTION_DIM),
    "next_base_angular_velocity3": _spec(np.float32, 3),
    "next_torso_quaternion_wxyz4": _spec(np.float32, 4),
    "simulator_advanced": _spec(np.bool_),
    "done": _spec(np.bool_),
    "terminated": _spec(np.bool_),
    "timed_out": _spec(np.bool_),
    "command_resampled": _spec(np.bool_),
    "base_height_m": _spec(np.float32),
    "base_tilt_rad": _spec(np.float32),
    "maximum_joint_velocity_ratio": _spec(np.float32),
    "maximum_actuator_force_ratio": _spec(np.float32),
    "target_tracking_rmse_rad": _spec(np.float32),
    "max_abs_plain_sonic_raw_native": _spec(np.float32),
    **{name: _spec(np.int64) for name in _COUNT_ARRAY_NAMES},
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def _regular_file(path: Path, expected_sha256: str, context: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{context} must not be symlink: {path}")
    try:
        result = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{context} does not exist: {path}") from error
    if not result.is_file():
        raise ValueError(f"{context} must be a regular file: {result}")
    actual = sha256_file(result)
    if actual != _require_sha256(expected_sha256, f"{context} expected SHA256"):
        raise ValueError(f"{context} SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return result


def executed_bootstrap_source_binding(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Hash every Python source byte in collector, MJLab, and task trees."""

    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_candidates = [root / relative for relative in EXECUTED_SOURCE_FILE_RELATIVE_PATHS]
    if any(path.is_symlink() for path in source_candidates):
        raise ValueError("bootstrap executed source files must not be symlinks")
    source_files = [path.resolve(strict=True) for path in source_candidates]
    for relative, path in zip(EXECUTED_SOURCE_FILE_RELATIVE_PATHS, source_files, strict=True):
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"bootstrap executed source file is invalid: {relative.as_posix()}")
    file_entries = [
        {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for relative, path in zip(EXECUTED_SOURCE_FILE_RELATIVE_PATHS, source_files, strict=True)
    ]
    trees: list[dict[str, Any]] = []
    for relative in EXECUTED_SOURCE_TREE_RELATIVE_PATHS:
        candidate = root / relative
        if candidate.is_symlink():
            raise ValueError(f"bootstrap source tree must not be symlink: {relative.as_posix()}")
        tree = candidate.resolve(strict=True)
        if not tree.is_dir() or not tree.is_relative_to(root):
            raise ValueError(f"bootstrap source tree is invalid: {relative.as_posix()}")
        descendants = sorted(tree.rglob("*"), key=lambda path: path.as_posix())
        symlinks = [path for path in descendants if path.is_symlink()]
        if symlinks:
            raise ValueError(f"bootstrap source tree contains symlink: {symlinks[0]}")
        files = [path for path in descendants if path.is_file() and path.suffix == ".py"]
        if not files:
            raise ValueError(f"bootstrap source tree contains no Python files: {relative.as_posix()}")
        entries = [
            {
                "path": path.relative_to(tree).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ]
        trees.append(
            {
                "relative_root": relative.as_posix(),
                "selection": "regular_nonsymlink_**/*.py",
                "file_count": len(entries),
                "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
                "manifest_sha256": _sha256_bytes(canonical_json_bytes(entries)),
            }
        )
    body = {
        "schema": "g1_true23_native124_21204_bootstrap_executed_sources_v1",
        "files": {
            "selection": "explicit_regular_nonsymlink_files",
            "file_count": len(file_entries),
            "total_bytes": sum(int(entry["size_bytes"]) for entry in file_entries),
            "manifest_sha256": _sha256_bytes(canonical_json_bytes(file_entries)),
        },
        "trees": trees,
    }
    body["binding_sha256"] = _sha256_bytes(canonical_json_bytes(body))
    return body


def bootstrap_teacher_policy_previous_action(env: Any) -> torch.Tensor:
    """Expose SONIC-native applied action, with a true zero reset predecessor."""

    term = env.action_manager.get_term("joint_pos")
    value = getattr(term, "safe_native_action", None)
    episode = getattr(env, "episode_length_buf", None)
    if (
        type(value) is not torch.Tensor
        or value.shape != (env.num_envs, ACTION_DIM)
        or value.dtype != torch.float32
        or value.device != torch.device(env.device)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("bootstrap applied safe native action must be finite float32 [batch,23]")
    if (
        type(episode) is not torch.Tensor
        or episode.shape != (env.num_envs,)
        or episode.dtype != torch.long
        or episode.device != torch.device(env.device)
    ):
        raise ValueError("bootstrap episode counter must be device-local [batch]")
    previous = torch.where(episode[:, None] == 0, torch.zeros_like(value), value)
    return pad_native_il23_to_canonical_il29(previous)


def _configure_bootstrap_observation_semantics(env_cfg: Any) -> dict[str, Any]:
    groups = getattr(env_cfg, "observations", None)
    if not isinstance(groups, Mapping) or set(("tokenizer", "policy", "actor")) - set(groups):
        raise ValueError("bootstrap observation groups are incomplete")
    tokenizer = groups["tokenizer"]
    policy = groups["policy"]
    actor = groups["actor"]
    if (
        tokenizer.enable_corruption is not TOKENIZER_CORRUPTION_ENABLED
        or policy.enable_corruption is not POLICY_CORRUPTION_ENABLED
        or actor.enable_corruption is not ACTOR_CORRUPTION_ENABLED
    ):
        raise ValueError("bootstrap observation corruption contract drift")
    previous = policy.terms.get("previous_action")
    if previous is None or previous.func is not padded_previous_action:
        raise ValueError("bootstrap base policy previous-action term drift")
    previous.func = bootstrap_teacher_policy_previous_action
    previous.params = {}
    return {
        "tokenizer_corruption_enabled": TOKENIZER_CORRUPTION_ENABLED,
        "policy_corruption_enabled": POLICY_CORRUPTION_ENABLED,
        "actor_corruption_enabled": ACTOR_CORRUPTION_ENABLED,
        "noise_rng": "fixed_seeded_torch_cuda",
        "choice": (
            "preserve model250 training-distribution observation corruption while keeping "
            "the selected teacher actor clean"
        ),
        "policy_previous_action": "zero_native23_until_first_teacher_action_then_applied_safe_native23",
    }


def _audit_bootstrap_runtime_observation_semantics(raw_env: Any) -> None:
    groups = getattr(getattr(raw_env, "observation_manager", None), "cfg", None)
    if not isinstance(groups, Mapping):
        raise ValueError("bootstrap runtime observation config is unavailable")
    if (
        groups["tokenizer"].enable_corruption is not TOKENIZER_CORRUPTION_ENABLED
        or groups["policy"].enable_corruption is not POLICY_CORRUPTION_ENABLED
        or groups["actor"].enable_corruption is not ACTOR_CORRUPTION_ENABLED
    ):
        raise RuntimeError("bootstrap runtime observation corruption drift")
    previous = groups["policy"].terms.get("previous_action")
    if previous is None or previous.func is not bootstrap_teacher_policy_previous_action:
        raise RuntimeError("bootstrap runtime previous-action observation drift")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("role") != "teacher_controlled_reset_seam_behavior_cloning_bootstrap_only"
    ):
        raise ValueError("bootstrap contract identity mismatch")
    identity = _mapping(contract.get("artifact_identity"), "artifact_identity")
    teacher = _mapping(identity.get("teacher"), "artifact_identity.teacher")
    if (
        teacher.get("iteration") != 21204
        or teacher.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or teacher.get("actor_state_sha256") != ACTOR_STATE_SHA256
        or teacher.get("onnx_sha256") != ONNX_SHA256
        or teacher.get("composite_contract_path") != COMPOSITE_CONTRACT_RELATIVE_PATH.as_posix()
        or teacher.get("composite_contract_sha256") != COMPOSITE_CONTRACT_SHA256
    ):
        raise ValueError("bootstrap teacher identity mismatch")
    encoder = _mapping(
        identity.get("causal_encoder"),
        "artifact_identity.causal_encoder",
    )
    expected_encoder = {
        "encoder_onnx_sha256": CAUSAL_ENCODER_SHA256,
        "decoder_onnx_sha256": CAUSAL_DECODER_SHA256,
        "metadata_file_sha256": CAUSAL_METADATA_FILE_SHA256,
        "source_checkpoint_sha256": CAUSAL_SOURCE_CHECKPOINT_SHA256,
        "reference_profile": CAUSAL_REFERENCE_PROFILE,
    }
    if any(encoder.get(name) != value for name, value in expected_encoder.items()):
        raise ValueError("bootstrap causal encoder identity mismatch")
    if encoder.get("source_checkpoint_required") is not False:
        raise ValueError("bootstrap must permit direct ONNX verification when source PT is absent")
    motion = _mapping(identity.get("motion"), "artifact_identity.motion")
    if (
        motion.get("sha256") != DAD_DANCE_SHA256
        or motion.get("frame_count") != DAD_DANCE_FRAME_COUNT
        or float(motion.get("fps", 0.0)) != 50.0
    ):
        raise ValueError("bootstrap motion identity mismatch")
    rows = _mapping(contract.get("causal_rows"), "causal_rows")
    expected_rows = {
        "total_rows": TOTAL_ROWS,
        "reset_prefix_rows": RESET_PREFIX_ROWS,
        "real_h10_rows": REAL_H10_ROWS,
        "initial_q9_reference_index": INITIAL_Q9,
        "last_action_q9_reference_index": LAST_ACTION_Q9,
        "first_control_state_index": INITIAL_Q9 + 1,
        "last_control_state_index": LAST_ACTION_Q9 + 1,
        "first_next_control_state_index": INITIAL_Q9 + 2,
        "last_next_control_state_index": LAST_ACTION_Q9 + 2,
        "post_collection_q9_reference_index": POST_COLLECTION_Q9,
        "episode_timeout_steps": EPISODE_TIMEOUT_STEPS,
        "control_period_ns": 20_000_000,
    }
    if any(rows.get(name) != value for name, value in expected_rows.items()):
        raise ValueError("bootstrap causal row/index contract mismatch")
    if (
        rows.get("reset_prefix_history_depth") != list(range(1, 11))
        or rows.get("reset_prefix_padding_count") != list(range(9, -1, -1))
        or rows.get("partial_row_masking_permitted") is not False
        or rows.get("teacher_warmup_substitution_permitted") is not False
    ):
        raise ValueError("bootstrap reset-prefix contract mismatch")
    observation = _mapping(
        contract.get("observation_and_label_contract"),
        "observation_and_label_contract",
    )
    required_observation = {
        "tokenizer_dim": TOKENIZER_DIM,
        "encoder_dim": ENCODER_DIM,
        "token_dim": TOKEN_DIM,
        "proprioception_h10_dim": PROPRIO_DIM,
        "decoder_input_dim": DECODER_DIM,
        "selected_observation_dim": SELECTED_OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "parity_max_absolute_error": PARITY_ATOL,
        "model_replay_max_absolute_error": MODEL_REPLAY_ATOL,
        "teacher_support_contract_sha256": SUPPORT_CONFIG_SHA256,
        "teacher_composite_application_count": 1,
        "wrapper_action_clip": None,
        "model250_decoder_action_used": False,
        "tokenizer_corruption_enabled": TOKENIZER_CORRUPTION_ENABLED,
        "policy_corruption_enabled": POLICY_CORRUPTION_ENABLED,
        "actor_corruption_enabled": ACTOR_CORRUPTION_ENABLED,
        "policy_reset_previous_action": "zero_native23_until_first_teacher_action",
        "policy_history_runtime_proof": (
            "all_term_buffers_depth_backfill_term_major_concat_and_single_append_shift_checked_each_row"
        ),
    }
    if any(observation.get(name) != value for name, value in required_observation.items()):
        raise ValueError("bootstrap observation/label contract mismatch")
    runtime_binding = _mapping(contract.get("runtime_binding"), "runtime_binding")
    if (
        runtime_binding.get("source_bytes_hashed_before_simulator_construction") is not True
        or runtime_binding.get("source_bytes_rehashed_after_collection") is not True
        or runtime_binding.get("rsl_runtime_tree_hash_required") is not True
        or runtime_binding.get("model_artifacts_rehashed_after_collection") is not True
    ):
        raise ValueError("bootstrap runtime/source binding contract mismatch")
    npz_schema = _mapping(contract.get("npz_schema"), "npz_schema")
    if (
        npz_schema.get("format") != "numpy_npz_compressed_allow_pickle_false"
        or npz_schema.get("array_count") != len(ARRAY_SPECS)
        or npz_schema.get("row_axis_size") != TOTAL_ROWS
        or npz_schema.get("floating_dtype") != "float32"
        or npz_schema.get("integer_dtype") != "int64"
        or npz_schema.get("boolean_dtype") != "bool"
        or npz_schema.get("object_arrays_permitted") is not False
        or npz_schema.get("exact_array_names") != sorted(ARRAY_SPECS)
    ):
        raise ValueError("bootstrap exact NPZ schema mismatch")
    gate = _mapping(contract.get("nominal_integrity_gate"), "nominal_integrity_gate")
    if (
        gate.get("minimum_base_height_m") != 0.45
        or gate.get("maximum_base_tilt_rad") != 1.0
        or gate.get("maximum_joint_velocity_ratio") != 1.0
        or gate.get("maximum_tracking_rmse_rad") != 0.75
        or gate.get("plain_sonic_raw_abs_strict_max") != 10.0
        or gate.get("required_zero_count_arrays") != list(_COUNT_ARRAY_NAMES)
    ):
        raise ValueError("bootstrap nominal integrity gate mismatch")
    failure = _mapping(contract.get("failure_policy"), "failure_policy")
    if (
        failure.get("whole_run_quarantine") is not True
        or failure.get("bootstrap_bc_eligible_rows_on_any_failure") != 0
        or failure.get("partial_npz_training_export_permitted") is not False
        or failure.get("structural_or_runtime_failure_emits_manifest_only") is not True
        or failure.get("quarantine_npz_loader_admission_permitted") is not False
        or failure.get("diagnostic_quarantine_npz_permitted") is not True
        or failure.get("strict_training_loader_requires_committed_manifest") is not True
        or failure.get("failure_manifest_publication") != "temporary_fsync_then_noreplace_hardlink_commit"
    ):
        raise ValueError("bootstrap failure/quarantine policy mismatch")
    boundaries = _mapping(contract.get("boundaries"), "boundaries")
    required_true = {
        "simulator_only",
        "teacher_controlled",
        "bootstrap_behavior_cloning_only",
        "reset_prefix_is_bootstrap_only",
    }
    required_false = {
        "support_qualification_performed",
        "support_admitted",
        "on_policy_data",
        "dagger_data",
        "student_policy_present",
        "student_action_present",
        "promotion_eligible",
        "deployment_ready",
        "hardware_authorized",
        "robot_or_network_commands_permitted",
    }
    if any(boundaries.get(name) is not True for name in required_true) or any(
        boundaries.get(name) is not False for name in required_false
    ):
        raise ValueError("bootstrap permanent safety boundary mismatch")


def load_bootstrap_contract(
    repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Load and hash-verify the immutable bootstrap contract."""

    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    path = root / CONTRACT_RELATIVE_PATH
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("bootstrap contract must be a regular repository file")
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != CONTRACT_SHA256:
        raise ValueError(f"bootstrap contract SHA256 mismatch: expected {CONTRACT_SHA256}, got {actual}")
    try:
        contract = _mapping(json.loads(payload.decode("utf-8")), "bootstrap contract")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("bootstrap contract must be strict UTF-8 JSON") from error
    _validate_contract(contract)
    return contract


@dataclass(frozen=True)
class BootstrapCollectionRequest:
    """Immutable, fixed-seed simulator-only collection request."""

    repository_root: Path
    output_prefix: Path

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path) or not isinstance(
            self.output_prefix,
            Path,
        ):
            raise TypeError("bootstrap request paths must be pathlib.Path values")

    @property
    def root(self) -> Path:
        result = self.repository_root.expanduser().resolve(strict=True)
        if not result.is_dir():
            raise ValueError("repository_root must be a directory")
        return result

    @property
    def prefix(self) -> Path:
        evidence_root = (self.root / "artifacts" / "g1_true23").resolve(strict=True)
        raw = self.output_prefix.expanduser()
        if raw.suffix:
            raise ValueError("bootstrap output_prefix must not have a suffix")
        candidate = raw if raw.is_absolute() else self.root / raw
        result = candidate.resolve(strict=False)
        try:
            result.relative_to(evidence_root)
        except ValueError as error:
            raise ValueError("bootstrap output must stay under artifacts/g1_true23") from error
        if not result.parent.is_dir() or result.parent.is_symlink():
            raise ValueError("bootstrap output parent must be an existing regular directory")
        return result

    @property
    def npz_path(self) -> Path:
        result = Path(f"{self.prefix}.npz")
        if os.path.lexists(result):
            raise FileExistsError(f"refusing to overwrite bootstrap NPZ: {result}")
        return result

    @property
    def manifest_path(self) -> Path:
        result = Path(f"{self.prefix}.manifest.json")
        if os.path.lexists(result):
            raise FileExistsError(f"refusing to overwrite bootstrap manifest: {result}")
        return result


@dataclass(frozen=True)
class BootstrapIssue:
    code: str
    count: int
    first_row: int
    last_row: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "count": self.count,
            "first_row": self.first_row,
            "last_row": self.last_row,
        }


@dataclass(frozen=True)
class BootstrapAssessment:
    """Whole-run verdict; one issue reduces eligible rows from 510 to zero."""

    issues: tuple[BootstrapIssue, ...]

    @property
    def quarantined(self) -> bool:
        return bool(self.issues)

    @property
    def bootstrap_bc_eligible_rows(self) -> int:
        return 0 if self.quarantined else TOTAL_ROWS

    def to_dict(self) -> dict[str, Any]:
        return {
            "whole_run_quarantined": self.quarantined,
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
            "bootstrap_bc_training_candidate": not self.quarantined,
            "bootstrap_bc_eligible_rows": self.bootstrap_bc_eligible_rows,
            "reset_prefix_bootstrap_rows": (0 if self.quarantined else RESET_PREFIX_ROWS),
            "real_h10_bootstrap_rows": 0 if self.quarantined else REAL_H10_ROWS,
            "support_admitted_rows": 0,
            "on_policy_rows": 0,
            "dagger_rows": 0,
            "promotion_eligible": False,
            "deployment_ready": False,
        }


def _validated_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping):
        raise TypeError("bootstrap arrays must be a mapping")
    if set(arrays) != set(ARRAY_SPECS):
        raise ValueError(
            "bootstrap array keys mismatch; "
            f"missing={sorted(set(ARRAY_SPECS) - set(arrays))}, "
            f"unexpected={sorted(set(arrays) - set(ARRAY_SPECS))}"
        )
    result: dict[str, np.ndarray] = {}
    for name, (dtype, shape) in ARRAY_SPECS.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"bootstrap array {name} must be numpy.ndarray")
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(f"bootstrap array {name} must be {dtype} {shape}, got {value.dtype} {value.shape}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"bootstrap array {name} contains NaN or Inf")
        result[name] = np.ascontiguousarray(value).copy()
    return result


def _row_issue(code: str, mask: np.ndarray) -> BootstrapIssue | None:
    value = np.asarray(mask, dtype=bool)
    if value.shape != (TOTAL_ROWS,):
        raise ValueError(f"internal bootstrap issue mask drift: {code}")
    indices = np.flatnonzero(value)
    if not indices.size:
        return None
    return BootstrapIssue(
        code=code,
        count=int(indices.size),
        first_row=int(indices[0]),
        last_row=int(indices[-1]),
    )


def assess_bootstrap_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    repository_root: str | Path | None = None,
) -> BootstrapAssessment:
    """Validate exact schema and apply all whole-run bootstrap gates."""

    values = _validated_arrays(arrays)
    contract = load_bootstrap_contract(repository_root)
    gate = _mapping(contract["nominal_integrity_gate"], "nominal_integrity_gate")
    support = load_teacher_support_contract(repository_root)
    composite = _mapping(
        support["teacher_composite_contract"],
        "teacher_composite_contract",
    )
    if support.get("artifact_identity", {}).get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("bootstrap action contract teacher identity drift")
    if contract["observation_and_label_contract"].get("teacher_support_contract_sha256") != (
        SUPPORT_CONFIG_SHA256
    ):
        raise RuntimeError("bootstrap frozen action-contract SHA drift")

    issues: list[BootstrapIssue] = []

    def add(code: str, mask: np.ndarray) -> None:
        issue = _row_issue(code, mask)
        if issue is not None:
            issues.append(issue)

    row = np.arange(TOTAL_ROWS, dtype=np.int64)
    prefix = row < RESET_PREFIX_ROWS
    add("row_index_mismatch", values["row_index"] != row)
    add("reset_prefix_flag_mismatch", values["reset_prefix"] != prefix)
    add("steady_history_flag_mismatch", values["steady_history"] != ~prefix)
    add(
        "history_depth_mismatch",
        values["history_depth"] != np.minimum(row + 1, RESET_PREFIX_ROWS),
    )
    add(
        "reset_padding_count_mismatch",
        values["reset_padding_count"] != np.maximum(RESET_PREFIX_ROWS - 1 - row, 0),
    )
    add("q9_reference_index_mismatch", values["q9_reference_index"] != INITIAL_Q9 + row)
    add("control_state_index_mismatch", values["control_state_index"] != INITIAL_Q9 + 1 + row)
    add(
        "next_control_state_index_mismatch",
        values["next_control_state_index"] != INITIAL_Q9 + 2 + row,
    )

    decoder = np.concatenate((values["token64"], values["proprio930"]), axis=1)
    add("decoder994_concat_mismatch", np.any(values["decoder994"] != decoder, axis=1))

    replay_token, replay_onnx = replay_bootstrap_model_outputs(
        values["encoder267"],
        values["selected_observation124"],
        repository_root=repository_root,
    )
    token_replay_error = np.max(
        np.abs(values["token64"].astype(np.float64) - replay_token.astype(np.float64)),
        axis=1,
    )
    selected_replay_error = np.max(
        np.abs(values["teacher_action_onnx_hardware23"].astype(np.float64) - replay_onnx.astype(np.float64)),
        axis=1,
    )
    add("model250_encoder_token_replay_mismatch", token_replay_error > MODEL_REPLAY_ATOL)
    add("selected_onnx_action_replay_mismatch", selected_replay_error > MODEL_REPLAY_ATOL)

    pt = values["teacher_action_pt_hardware23"]
    onnx = values["teacher_action_onnx_hardware23"]
    parity = np.max(np.abs(pt.astype(np.float64) - onnx.astype(np.float64)), axis=1)
    add("teacher_pt_onnx_parity_exceeded", parity > PARITY_ATOL)
    add(
        "teacher_parity_record_mismatch",
        np.abs(values["teacher_parity_max_abs"].astype(np.float64) - parity) > 1.0e-7,
    )
    add(
        "teacher_applied_action_mismatch",
        np.any(values["teacher_action_applied_hardware23"] != pt, axis=1),
    )

    selected_home = np.asarray(
        composite["checkpoint21204_home_q_hardware"],
        dtype=np.float32,
    )
    selected_scale = np.asarray(
        composite["checkpoint21204_action_scale_hardware"],
        dtype=np.float32,
    )
    candidate = (selected_home + selected_scale * pt).astype(np.float32, copy=False)
    candidate_error = np.max(
        np.abs(values["teacher_candidate_target_hardware23"].astype(np.float64) - candidate.astype(np.float64)),
        axis=1,
    )
    add(
        "teacher_candidate_target_mismatch",
        candidate_error > float(composite["candidate_link_atol"]),
    )
    sonic_default = np.asarray(
        support["student_action_contract"]["hardware_default_q"],
        dtype=np.float32,
    )
    sonic_scale = np.asarray(
        support["student_action_contract"]["hardware_action_scale"],
        dtype=np.float32,
    )
    hardware_to_native = np.asarray(
        support["student_action_contract"]["mujoco_to_isaaclab_dof"],
        dtype=np.int64,
    )
    plain_hardware = ((candidate - sonic_default) / sonic_scale).astype(
        np.float32,
        copy=False,
    )
    plain_native = plain_hardware[:, hardware_to_native].astype(np.float32, copy=False)
    label_error = np.max(
        np.abs(values["teacher_label_raw_native23"].astype(np.float64) - plain_native.astype(np.float64)),
        axis=1,
    )
    add(
        "teacher_raw_native_label_mismatch",
        label_error > float(composite["composite_link_atol"]),
    )
    safe_native, final_target = safe_target_transform_numpy(plain_native)
    safe_error = np.max(
        np.abs(values["teacher_applied_safe_native23"].astype(np.float64) - safe_native.astype(np.float64)),
        axis=1,
    )
    target_error = np.max(
        np.abs(values["teacher_final_target_hardware23"].astype(np.float64) - final_target.astype(np.float64)),
        axis=1,
    )
    add("teacher_safe_action_mismatch", safe_error > ACTION_LINK_ATOL)
    add("teacher_final_target_mismatch", target_error > ACTION_LINK_ATOL)
    expected_clip = np.abs(plain_native) >= float(gate["plain_sonic_raw_abs_strict_max"])
    add(
        "raw_clip_mask_mismatch",
        np.any(values["raw_clip_mask_native23"] != expected_clip, axis=1),
    )
    add("teacher_label_requires_raw_clip", np.any(expected_clip, axis=1))
    recorded_max = np.max(np.abs(plain_native), axis=1)
    add(
        "max_abs_plain_sonic_record_mismatch",
        np.abs(values["max_abs_plain_sonic_raw_native"].astype(np.float64) - recorded_max.astype(np.float64))
        > 1.0e-6,
    )

    add("simulator_did_not_advance", ~values["simulator_advanced"])
    add("done_or_autoreset", values["done"])
    add("non_timeout_termination", values["terminated"])
    add("unexpected_timeout", values["timed_out"])
    add("command_resampled", values["command_resampled"])
    for name in _COUNT_ARRAY_NAMES:
        add(f"nonzero_{name}", values[name] != 0)
    add(
        "base_height_below_gate",
        values["base_height_m"] < float(gate["minimum_base_height_m"]),
    )
    add(
        "base_tilt_above_gate",
        values["base_tilt_rad"] > float(gate["maximum_base_tilt_rad"]),
    )
    add(
        "joint_velocity_ratio_above_gate",
        values["maximum_joint_velocity_ratio"] > float(gate["maximum_joint_velocity_ratio"]),
    )
    add(
        "tracking_rmse_above_gate",
        values["target_tracking_rmse_rad"] > float(gate["maximum_tracking_rmse_rad"]),
    )
    quaternion_norm = np.linalg.norm(
        values["next_torso_quaternion_wxyz4"].astype(np.float64),
        axis=1,
    )
    add("next_torso_quaternion_not_unit", np.abs(quaternion_norm - 1.0) > 1.0e-3)
    return BootstrapAssessment(tuple(issues))


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError("manifest material contains NaN or Inf")
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("manifest material contains NaN or Inf")
        return value
    raise TypeError(f"unsupported manifest material: {type(value).__qualname__}")


def _write_temporary_npz(parent: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=parent,
        prefix=".teacher-bootstrap-",
        suffix=".npz.tmp",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_temporary_bytes(parent: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=parent,
        prefix=".teacher-bootstrap-",
        suffix=".json.tmp",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_link_new(temporary: Path, final: Path, context: str) -> None:
    try:
        os.link(temporary, final)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {context}: {final}") from error


def publish_bootstrap_evidence_new(
    arrays: Mapping[str, np.ndarray],
    *,
    npz_path: str | Path,
    manifest_path: str | Path,
    materials: Mapping[str, Any],
    repository_root: str | Path | None = None,
) -> tuple[Path, Path, Mapping[str, Any]]:
    """Publish one immutable NPZ/manifest pair; never publish partial training data."""

    values = _validated_arrays(arrays)
    assessment = assess_bootstrap_arrays(values, repository_root=repository_root)
    npz = Path(npz_path).expanduser().resolve(strict=False)
    manifest = Path(manifest_path).expanduser().resolve(strict=False)
    if npz.suffix.lower() != ".npz" or not manifest.name.endswith(".manifest.json"):
        raise ValueError("bootstrap outputs must end in .npz and .manifest.json")
    if npz.parent != manifest.parent or not npz.parent.is_dir() or npz.parent.is_symlink():
        raise ValueError("bootstrap outputs must share one existing regular directory")
    if npz == manifest:
        raise ValueError("bootstrap NPZ and manifest paths must differ")
    for path, context in ((npz, "bootstrap NPZ"), (manifest, "bootstrap manifest")):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite {context}: {path}")

    temporary_npz: Path | None = None
    temporary_manifest: Path | None = None
    npz_published = False
    manifest_published = False
    try:
        temporary_npz = _write_temporary_npz(npz.parent, values)
        npz_sha = sha256_file(temporary_npz)
        npz_size = temporary_npz.stat().st_size
        array_schema = {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _array_sha256(value),
            }
            for name, value in sorted(values.items())
        }
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "contract": {
                "path": str(CONTRACT_RELATIVE_PATH).replace("\\", "/"),
                "sha256": CONTRACT_SHA256,
            },
            "artifact": {
                "npz_filename": npz.name,
                "npz_sha256": npz_sha,
                "npz_size_bytes": npz_size,
                "array_count": len(values),
                "array_schema": array_schema,
                "classification": (
                    "quarantined_diagnostic_only" if assessment.quarantined else "bootstrap_bc_training_candidate"
                ),
                "strict_loader_admissible": not assessment.quarantined,
            },
            "rows": {
                "total": TOTAL_ROWS,
                "reset_prefix_bootstrap_only": RESET_PREFIX_ROWS,
                "real_h10": REAL_H10_ROWS,
                "q9_reference_first": INITIAL_Q9,
                "q9_reference_last_action": LAST_ACTION_Q9,
                "control_state_first": INITIAL_Q9 + 1,
                "control_state_last": LAST_ACTION_Q9 + 1,
                "post_collection_q9": POST_COLLECTION_Q9,
                "post_collection_control_state": LAST_ACTION_Q9 + 2,
            },
            "materials": _json_safe(materials),
            "qualification": assessment.to_dict(),
            "boundaries": {
                "classification": "teacher_controlled_bootstrap_behavior_cloning_only",
                "teacher_controlled": True,
                "student_policy_present": False,
                "student_action_present": False,
                "support_qualification_performed": False,
                "support_admitted": False,
                "on_policy_data": False,
                "dagger_data": False,
                "promotion_eligible": False,
                "deployment_ready": False,
                "hardware_authorized": False,
                "robot_or_network_commands_performed": False,
            },
            "publication": {
                "protocol": "npz_hardlink_then_manifest_hardlink_commit",
                "manifest_required_for_loader_admission": True,
                "orphan_npz_admissible": False,
                "overwrite_permitted": False,
            },
        }
        body["manifest_payload_sha256"] = _sha256_bytes(canonical_json_bytes(body))
        temporary_manifest = _write_temporary_bytes(
            manifest.parent,
            canonical_json_bytes(body),
        )
        _publish_link_new(temporary_npz, npz, "bootstrap NPZ")
        npz_published = True
        _publish_link_new(temporary_manifest, manifest, "bootstrap manifest")
        manifest_published = True
        temporary_npz.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        return npz, manifest, body
    except BaseException:
        if manifest_published and os.path.lexists(manifest):
            manifest.unlink()
        if npz_published and os.path.lexists(npz):
            npz.unlink()
        raise
    finally:
        if temporary_npz is not None:
            temporary_npz.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)


def load_bootstrap_training_candidate(
    npz_path: str | Path,
    manifest_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    """Admit only a complete, non-quarantined, hash-bound bootstrap pair."""

    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    raw_npz = Path(npz_path).expanduser()
    raw_manifest = Path(manifest_path).expanduser()
    if raw_npz.is_symlink() or raw_manifest.is_symlink():
        raise ValueError("bootstrap training loader rejects symlink artifacts")
    npz = raw_npz.resolve(strict=True)
    manifest = raw_manifest.resolve(strict=True)
    if not npz.is_file() or not manifest.is_file() or npz.parent != manifest.parent:
        raise ValueError("bootstrap training pair must be regular files in one directory")
    payload = manifest.read_bytes()
    try:
        body = _mapping(json.loads(payload.decode("utf-8")), "bootstrap manifest")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("bootstrap manifest must be strict UTF-8 JSON") from error
    if payload != canonical_json_bytes(body):
        raise ValueError("bootstrap manifest must use canonical JSON bytes")
    unhashed = copy.deepcopy(dict(body))
    claimed_manifest_hash = unhashed.pop("manifest_payload_sha256", None)
    if claimed_manifest_hash != _sha256_bytes(canonical_json_bytes(unhashed)):
        raise ValueError("bootstrap manifest payload SHA256 mismatch")
    if body.get("schema_version") != SCHEMA_VERSION or body.get("kind") != MANIFEST_KIND:
        raise ValueError("bootstrap training loader rejects failure/foreign manifest")
    contract = _mapping(body.get("contract"), "bootstrap manifest contract")
    if contract.get("sha256") != CONTRACT_SHA256:
        raise ValueError("bootstrap training manifest contract drift")
    artifact = _mapping(body.get("artifact"), "bootstrap manifest artifact")
    qualification = _mapping(body.get("qualification"), "bootstrap manifest qualification")
    boundaries = _mapping(body.get("boundaries"), "bootstrap manifest boundaries")
    publication = _mapping(body.get("publication"), "bootstrap manifest publication")
    if (
        artifact.get("npz_filename") != npz.name
        or artifact.get("npz_sha256") != sha256_file(npz)
        or artifact.get("npz_size_bytes") != npz.stat().st_size
        or artifact.get("array_count") != len(ARRAY_SPECS)
    ):
        raise ValueError("bootstrap NPZ identity differs from manifest")
    if (
        artifact.get("classification") != "bootstrap_bc_training_candidate"
        or artifact.get("strict_loader_admissible") is not True
        or qualification.get("whole_run_quarantined") is not False
        or qualification.get("bootstrap_bc_training_candidate") is not True
        or qualification.get("bootstrap_bc_eligible_rows") != TOTAL_ROWS
    ):
        raise ValueError("bootstrap training loader rejects quarantined evidence")
    if (
        publication.get("protocol") != "npz_hardlink_then_manifest_hardlink_commit"
        or publication.get("manifest_required_for_loader_admission") is not True
        or publication.get("orphan_npz_admissible") is not False
        or publication.get("overwrite_permitted") is not False
    ):
        raise ValueError("bootstrap training manifest publication protocol drift")
    if any(
        boundaries.get(name) is not False
        for name in (
            "student_policy_present",
            "student_action_present",
            "support_qualification_performed",
            "support_admitted",
            "on_policy_data",
            "dagger_data",
            "promotion_eligible",
            "deployment_ready",
            "hardware_authorized",
            "robot_or_network_commands_performed",
        )
    ):
        raise ValueError("bootstrap training manifest boundary overclaim")
    if (
        boundaries.get("classification") != "teacher_controlled_bootstrap_behavior_cloning_only"
        or boundaries.get("teacher_controlled") is not True
    ):
        raise ValueError("bootstrap training manifest role drift")
    with np.load(npz, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_SPECS):
            raise ValueError("bootstrap training NPZ array keys mismatch")
        arrays = {name: archive[name].copy() for name in ARRAY_SPECS}
    values = _validated_arrays(arrays)
    schema = _mapping(artifact.get("array_schema"), "bootstrap manifest array_schema")
    if set(schema) != set(ARRAY_SPECS):
        raise ValueError("bootstrap manifest array schema keys mismatch")
    for name, value in values.items():
        entry = _mapping(schema[name], f"bootstrap manifest array_schema.{name}")
        if (
            entry.get("dtype") != str(value.dtype)
            or entry.get("shape") != list(value.shape)
            or entry.get("sha256") != _array_sha256(value)
        ):
            raise ValueError(f"bootstrap manifest array binding mismatch: {name}")
    assessment = assess_bootstrap_arrays(values, repository_root=root)
    if assessment.quarantined or assessment.bootstrap_bc_eligible_rows != TOTAL_ROWS:
        raise ValueError("bootstrap training arrays fail fresh whole-run assessment")
    return values, body


def _causal_bundle_paths(root: Path) -> dict[str, Path]:
    directory = (root / CAUSAL_BUNDLE_RELATIVE_PATH).resolve(strict=True)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("causal model-250 bundle must be a regular directory")
    return {
        "encoder": directory / CAUSAL_ENCODER_FILENAME,
        "decoder": directory / CAUSAL_DECODER_FILENAME,
        "metadata": directory / CAUSAL_METADATA_FILENAME,
        "source_checkpoint": directory / CAUSAL_SOURCE_CHECKPOINT_FILENAME,
    }


def preflight_bootstrap_collection(
    request: BootstrapCollectionRequest,
) -> dict[str, Any]:
    """Hash all fixed inputs and verify model-250 embedded ONNX metadata, no sim."""

    if type(request) is not BootstrapCollectionRequest:
        raise TypeError("request must be exact BootstrapCollectionRequest")
    root = request.root
    npz = request.npz_path
    manifest = request.manifest_path
    load_bootstrap_contract(root)
    binding = load_checkpoint21204_binding(root)
    motion = validate_dad_dance_motion_file(root / DAD_DANCE_RELATIVE_PATH)
    if sha256_file(motion) != DAD_DANCE_SHA256:
        raise RuntimeError("DadDance changed after validation")
    composite_contract_path = _regular_file(
        root / COMPOSITE_CONTRACT_RELATIVE_PATH,
        COMPOSITE_CONTRACT_SHA256,
        "selected-to-V2 composite MJLab contract",
    )

    paths = _causal_bundle_paths(root)
    encoder = _regular_file(paths["encoder"], CAUSAL_ENCODER_SHA256, "model-250 encoder ONNX")
    decoder = _regular_file(paths["decoder"], CAUSAL_DECODER_SHA256, "model-250 decoder ONNX")
    metadata_path = _regular_file(
        paths["metadata"],
        CAUSAL_METADATA_FILE_SHA256,
        "model-250 diagnostic metadata",
    )
    source_checkpoint = paths["source_checkpoint"]
    source_present = os.path.lexists(source_checkpoint)
    verified_source: Path | None = None
    if source_present:
        verified_source = _regular_file(
            source_checkpoint,
            CAUSAL_SOURCE_CHECKPOINT_SHA256,
            "model-250 source checkpoint",
        )
    metadata = verify_mjlab_diagnostic_onnx(
        encoder,
        decoder,
        metadata_path,
        checkpoint_path=verified_source,
        expected_filenames=(
            CAUSAL_ENCODER_FILENAME,
            CAUSAL_DECODER_FILENAME,
            CAUSAL_METADATA_FILENAME,
        ),
        expected_reference_profile=CAUSAL_REFERENCE_PROFILE,
    )
    hashes = _mapping(metadata.get("hashes"), "model-250 metadata.hashes")
    required_hashes = {
        "checkpoint_sha256": CAUSAL_SOURCE_CHECKPOINT_SHA256,
        "encoder_onnx_sha256": CAUSAL_ENCODER_SHA256,
        "decoder_onnx_sha256": CAUSAL_DECODER_SHA256,
        "encoder_embedded_metadata_sha256": ("bc856b32569faa65d116cf2df19ede55259deda693a203fd9faf06b12b644758"),
        "encoder_state_sha256": ("6acafb0b40d5f5fb4c0a79bcdaed784729bd14a993c656348cd1f664478c7e04"),
        "policy_state_sha256": ("c3bfcb5c42929293b62425f155b59ccb731f57c98e8852c7f1e97094525684af"),
        "lineage_sha256": ("08bbd03d0df751328e449d3624d79167587b461d6835fa1f4a8742aad9ffa82a"),
    }
    if any(hashes.get(name) != value for name, value in required_hashes.items()):
        raise ValueError("model-250 embedded/hash identity differs from bootstrap contract")
    source_binding = executed_bootstrap_source_binding(root)
    return {
        "schema": "g1_true23_native124_21204_teacher_bootstrap_preflight_v1",
        "ready": True,
        "contract_sha256": CONTRACT_SHA256,
        "teacher": {
            "checkpoint_path": str(binding.checkpoint_path),
            "checkpoint_sha256": binding.checkpoint_sha256,
            "actor_state_sha256": binding.actor_state_sha256,
            "onnx_path": str(binding.onnx_path),
            "onnx_sha256": binding.onnx_sha256,
        },
        "causal_encoder": {
            "encoder_path": str(encoder),
            "encoder_sha256": CAUSAL_ENCODER_SHA256,
            "decoder_bundle_path": str(decoder),
            "decoder_bundle_sha256": CAUSAL_DECODER_SHA256,
            "metadata_path": str(metadata_path),
            "metadata_file_sha256": CAUSAL_METADATA_FILE_SHA256,
            "source_checkpoint_path": str(source_checkpoint),
            "source_checkpoint_present": source_present,
            "source_checkpoint_verified": verified_source is not None,
            "direct_onnx_and_embedded_metadata_verified": True,
            "verification_limitation": (
                None
                if verified_source is not None
                else (
                    "causal_model_250.pt is absent; encoder/decoder ONNX, bundle metadata, "
                    "and embedded metadata are hash-verified directly, but this run cannot "
                    "reconstruct model-250 PT weights for fresh parity"
                )
            ),
            "encoder_input": {"name": "teleop_obs", "shape": [1, ENCODER_DIM]},
            "encoder_output": {"name": "token", "shape": [1, TOKEN_DIM]},
            "model250_decoder_action_used": False,
        },
        "motion": {
            "path": str(motion),
            "sha256": DAD_DANCE_SHA256,
            "frame_count": DAD_DANCE_FRAME_COUNT,
        },
        "composite_contract": {
            "path": str(composite_contract_path),
            "sha256": COMPOSITE_CONTRACT_SHA256,
        },
        "outputs": {
            "npz": str(npz),
            "manifest": str(manifest),
            "no_overwrite": True,
        },
        "fixed": {
            "seed": FIXED_SEED,
            "device": DEVICE,
            "num_envs": 1,
            "initial_q9": INITIAL_Q9,
            "transitions": TOTAL_ROWS,
            "episode_timeout_steps": EPISODE_TIMEOUT_STEPS,
            "action_substitution": False,
            "teacher_warmup_steps": 0,
            "rsl_actor": "original_selected_pt_deterministic_mean",
            "selected_onnx_provider": "CPUExecutionProvider",
            "causal_encoder_provider": "CPUExecutionProvider",
            "tokenizer_corruption_enabled": TOKENIZER_CORRUPTION_ENABLED,
            "policy_corruption_enabled": POLICY_CORRUPTION_ENABLED,
            "actor_corruption_enabled": ACTOR_CORRUPTION_ENABLED,
            "policy_reset_previous_action": "zero_native23_until_first_teacher_action",
        },
        "rsl_runtime": resolve_rsl_runtime_binding(),
        "executed_sources": source_binding,
        "boundaries": {
            "simulator_constructed": False,
            "simulator_steps": 0,
            "training_updates": 0,
            "teacher_controlled_bootstrap_only": True,
            "support_admitted": False,
            "on_policy_data": False,
            "dagger_data": False,
            "promotion_or_deployment_authorized": False,
            "hardware_or_network_authorized": False,
        },
    }


class _HashBoundCausalEncoder250:
    """Strict CPU ORT session for only the verified model-250 encoder."""

    def __init__(self, path: Path) -> None:
        self.path = _regular_file(
            path,
            CAUSAL_ENCODER_SHA256,
            "model-250 encoder before ORT load",
        )
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - environment-specific
            raise RuntimeError("onnxruntime is required for bootstrap encoder") from error
        self._session = ort.InferenceSession(
            str(self.path),
            providers=["CPUExecutionProvider"],
        )
        if self._session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("model-250 encoder provider contract mismatch")
        inputs = [(item.name, item.shape, item.type) for item in self._session.get_inputs()]
        outputs = [(item.name, item.shape, item.type) for item in self._session.get_outputs()]
        if inputs != [("teleop_obs", [1, ENCODER_DIM], "tensor(float)")]:
            raise ValueError(f"model-250 encoder input ABI mismatch: {inputs}")
        if outputs != [("token", [1, TOKEN_DIM], "tensor(float)")]:
            raise ValueError(f"model-250 encoder output ABI mismatch: {outputs}")
        self.run(np.zeros((ENCODER_DIM,), dtype=np.float32))
        _regular_file(
            self.path,
            CAUSAL_ENCODER_SHA256,
            "model-250 encoder after ORT load",
        )

    def run(self, encoder267: np.ndarray) -> np.ndarray:
        if (
            not isinstance(encoder267, np.ndarray)
            or encoder267.dtype != np.float32
            or encoder267.shape != (ENCODER_DIM,)
            or not np.isfinite(encoder267).all()
        ):
            raise ValueError("model-250 encoder input must be finite float32 [267]")
        outputs = self._session.run(
            ["token"],
            {"teleop_obs": encoder267.reshape(1, ENCODER_DIM)},
        )
        if len(outputs) != 1:
            raise RuntimeError("model-250 encoder returned unexpected output count")
        token = outputs[0]
        if (
            not isinstance(token, np.ndarray)
            or token.dtype != np.float32
            or token.shape != (1, TOKEN_DIM)
            or not np.isfinite(token).all()
        ):
            raise RuntimeError("model-250 encoder output must be finite float32 [1,64]")
        return token[0].copy()


_MODEL_SESSION_CACHE: dict[
    str,
    tuple[_HashBoundCausalEncoder250, Native124Checkpoint21204Policy],
] = {}
_MODEL_OUTPUT_CACHE: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}


def _bound_bootstrap_model_sessions(
    root: Path,
) -> tuple[_HashBoundCausalEncoder250, Native124Checkpoint21204Policy]:
    key = str(root)
    cached = _MODEL_SESSION_CACHE.get(key)
    if cached is not None:
        binding = cached[1].binding
        _regular_file(binding.checkpoint_path, CHECKPOINT_SHA256, "selected checkpoint replay")
        _regular_file(binding.onnx_path, ONNX_SHA256, "selected ONNX replay")
        _regular_file(
            cached[0].path,
            CAUSAL_ENCODER_SHA256,
            "model-250 encoder replay",
        )
        return cached
    binding = load_checkpoint21204_binding(root)
    encoder = _HashBoundCausalEncoder250(_causal_bundle_paths(root)["encoder"])
    selected = Native124Checkpoint21204Policy(binding)
    result = (encoder, selected)
    _MODEL_SESSION_CACHE[key] = result
    return result


def replay_bootstrap_model_outputs(
    encoder267: np.ndarray,
    selected_observation124: np.ndarray,
    *,
    repository_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay both hash-bound CPU ONNX models over all exact bootstrap rows."""

    if (
        not isinstance(encoder267, np.ndarray)
        or encoder267.dtype != np.float32
        or encoder267.shape != (TOTAL_ROWS, ENCODER_DIM)
        or not np.isfinite(encoder267).all()
    ):
        raise ValueError("bootstrap encoder replay input must be finite float32 [510,267]")
    if (
        not isinstance(selected_observation124, np.ndarray)
        or selected_observation124.dtype != np.float32
        or selected_observation124.shape != (TOTAL_ROWS, SELECTED_OBSERVATION_DIM)
        or not np.isfinite(selected_observation124).all()
    ):
        raise ValueError("bootstrap selected replay input must be finite float32 [510,124]")
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    key = (
        str(root),
        _array_sha256(encoder267),
        _array_sha256(selected_observation124),
    )
    cached = _MODEL_OUTPUT_CACHE.get(key)
    if cached is not None:
        _bound_bootstrap_model_sessions(root)
        return cached[0].copy(), cached[1].copy()
    encoder, selected = _bound_bootstrap_model_sessions(root)
    tokens = np.stack([encoder.run(row) for row in encoder267], axis=0).astype(
        np.float32,
        copy=False,
    )
    actions = np.stack(
        [selected.run(row.reshape(1, SELECTED_OBSERVATION_DIM)) for row in selected_observation124],
        axis=0,
    ).astype(np.float32, copy=False)
    if tokens.shape != (TOTAL_ROWS, TOKEN_DIM) or actions.shape != (TOTAL_ROWS, ACTION_DIM):
        raise RuntimeError("bootstrap model replay output shape drift")
    if len(_MODEL_OUTPUT_CACHE) >= 8:
        _MODEL_OUTPUT_CACHE.clear()
    _MODEL_OUTPUT_CACHE[key] = (tokens.copy(), actions.copy())
    _regular_file(encoder.path, CAUSAL_ENCODER_SHA256, "model-250 encoder after replay")
    _regular_file(selected.binding.onnx_path, ONNX_SHA256, "selected ONNX after replay")
    return tokens.copy(), actions.copy()


def _append(records: dict[str, list[Any]], name: str, value: Any) -> None:
    records[name].append(value)


def _finalize_records(records: Mapping[str, list[Any]]) -> dict[str, np.ndarray]:
    if set(records) != set(ARRAY_SPECS):
        raise RuntimeError("internal bootstrap record-key drift")
    arrays: dict[str, np.ndarray] = {}
    for name, (dtype, shape) in ARRAY_SPECS.items():
        if len(records[name]) != TOTAL_ROWS:
            raise RuntimeError(f"bootstrap record {name} has {len(records[name])} rows, expected {TOTAL_ROWS}")
        value = np.asarray(records[name], dtype=dtype)
        if value.shape != shape:
            raise RuntimeError(f"bootstrap record {name} finalized as {value.shape}, expected {shape}")
        arrays[name] = np.ascontiguousarray(value)
    return _validated_arrays(arrays)


def _single_numpy_tensor(
    value: Any,
    shape: tuple[int, ...],
    context: str,
    *,
    floating: bool = True,
) -> np.ndarray:
    if type(value) is not torch.Tensor or tuple(value.shape) != shape:
        raise ValueError(f"{context} must be tensor with shape {shape}")
    if floating and (not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{context} must be finite floating tensor")
    return value.detach().to(device="cpu").contiguous().numpy().copy()


def _observation_arrays(observations: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokenizer = _single_numpy_tensor(
        observations["tokenizer"],
        (1, TOKENIZER_DIM),
        "bootstrap tokenizer observation",
    )[0]
    if tokenizer[0] != np.float32(1.0):
        raise ValueError("bootstrap tokenizer encoder route must equal one")
    proprio = _single_numpy_tensor(
        observations["policy"],
        (1, PROPRIO_DIM),
        "bootstrap policy H10 observation",
    )[0]
    selected = _single_numpy_tensor(
        observations["actor"],
        (1, SELECTED_OBSERVATION_DIM),
        "bootstrap selected actor observation",
    )[0]
    return tokenizer[1:].copy(), proprio, selected


def _policy_history_runtime_proof(
    raw_env: Any,
    observations: Any,
    transition: int,
) -> dict[str, Any]:
    """Prove actual MJLab H10 depth, reset backfill, and term-major flattening."""

    if isinstance(transition, bool) or not isinstance(transition, int) or not 0 <= transition < TOTAL_ROWS:
        raise ValueError("bootstrap history transition must be in [0,509]")
    manager = getattr(raw_env, "observation_manager", None)
    names_by_group = getattr(manager, "_group_obs_term_names", None)
    buffers_by_group = getattr(manager, "_group_obs_term_history_buffer", None)
    if not isinstance(names_by_group, Mapping) or not isinstance(buffers_by_group, Mapping):
        raise ValueError("bootstrap observation manager lacks exact history internals")
    names = tuple(names_by_group.get("policy", ()))
    if names != tuple(POLICY_TERM_WIDTHS):
        raise ValueError(f"bootstrap policy history term order drift: {names}")
    buffers = buffers_by_group.get("policy")
    if not isinstance(buffers, Mapping) or set(buffers) != set(POLICY_TERM_WIDTHS):
        raise ValueError("bootstrap policy history buffer set drift")
    expected_depth = min(transition + 1, RESET_PREFIX_ROWS)
    expected_padding = max(RESET_PREFIX_ROWS - expected_depth, 0)
    flattened: list[torch.Tensor] = []
    for name in names:
        buffer = buffers[name]
        if getattr(buffer, "max_length", None) != RESET_PREFIX_ROWS:
            raise ValueError(f"bootstrap policy history length drift: {name}")
        current_length = getattr(buffer, "current_length", None)
        if (
            type(current_length) is not torch.Tensor
            or current_length.shape != (1,)
            or int(current_length.detach().cpu().item()) != expected_depth
        ):
            raise RuntimeError(f"bootstrap actual history depth mismatch at row {transition}: {name}")
        history = getattr(buffer, "buffer", None)
        width = POLICY_TERM_WIDTHS[name]
        if (
            type(history) is not torch.Tensor
            or history.shape != (1, RESET_PREFIX_ROWS, width)
            or history.dtype != torch.float32
            or not bool(torch.isfinite(history).all())
        ):
            raise ValueError(f"bootstrap policy history tensor drift: {name}")
        if expected_padding:
            repeated = history[:, : expected_padding + 1]
            first = history[:, :1].expand_as(repeated)
            if not torch.equal(repeated, first):
                raise RuntimeError(f"bootstrap reset history backfill mismatch at row {transition}: {name}")
        if transition == 0 and name == "previous_action" and bool(torch.count_nonzero(history)):
            raise RuntimeError("bootstrap reset previous-action history is not zero native23")
        flattened.append(history.reshape(1, -1))
    exact_policy = torch.cat(flattened, dim=-1)
    observed_policy = observations["policy"]
    if (
        type(observed_policy) is not torch.Tensor
        or observed_policy.shape != (1, PROPRIO_DIM)
        or not torch.equal(observed_policy, exact_policy)
    ):
        raise RuntimeError("bootstrap policy930 differs from actual term-major H10 buffers")
    return {
        "row": transition,
        "actual_history_depth": expected_depth,
        "reset_padding_count": expected_padding,
        "term_order": list(names),
        "term_major_policy930_exact": True,
        "reset_previous_action_zero_if_reset_row": True,
    }


def _policy_history_runtime_snapshot(raw_env: Any) -> dict[str, torch.Tensor]:
    manager = raw_env.observation_manager
    buffers = manager._group_obs_term_history_buffer["policy"]  # noqa: SLF001
    if not isinstance(buffers, Mapping) or set(buffers) != set(POLICY_TERM_WIDTHS):
        raise ValueError("bootstrap policy history snapshot buffer set drift")
    return {name: buffers[name].buffer.detach().clone() for name in POLICY_TERM_WIDTHS}


def _assert_policy_history_shift(
    previous: Mapping[str, torch.Tensor],
    current: Mapping[str, torch.Tensor],
    transition: int,
) -> None:
    if set(previous) != set(POLICY_TERM_WIDTHS) or set(current) != set(POLICY_TERM_WIDTHS):
        raise ValueError("bootstrap policy history shift keys drift")
    for name, width in POLICY_TERM_WIDTHS.items():
        before = previous[name]
        after = current[name]
        expected_shape = (1, RESET_PREFIX_ROWS, width)
        if before.shape != expected_shape or after.shape != expected_shape:
            raise ValueError(f"bootstrap policy history shift shape drift: {name}")
        if not torch.equal(after[:, :-1], before[:, 1:]):
            raise RuntimeError(
                f"bootstrap policy history appended more or less than once at row {transition}: {name}"
            )


def _torso_quaternion(raw_env: Any) -> np.ndarray:
    robot = raw_env.scene["robot"]
    if tuple(robot.joint_names) != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError("bootstrap robot joint order differs from compact hardware-23")
    names = tuple(robot.body_names)
    if names.count("torso_link") != 1:
        raise ValueError("bootstrap robot must contain exactly one torso_link")
    value = _single_numpy_tensor(
        robot.data.body_link_quat_w[:, names.index("torso_link"), :],
        (1, 4),
        "bootstrap torso quaternion",
    )[0]
    if abs(float(np.linalg.norm(value.astype(np.float64))) - 1.0) > 1.0e-3:
        raise ValueError("bootstrap torso quaternion is not unit WXYZ")
    return value


def _command_resampled(command: Any) -> bool:
    value = getattr(command, "_causal_resampled", None)
    if value is None:
        return False
    if type(value) is not torch.Tensor or value.shape != (1,):
        raise ValueError("bootstrap causal resample flag must be tensor [1]")
    return bool(value.detach().cpu().item())


def _actual_action_arrays(raw_env: Any) -> dict[str, np.ndarray]:
    action = raw_env.action_manager.get_term("joint_pos")

    def vector(name: str) -> np.ndarray:
        return _single_numpy_tensor(
            getattr(action, name),
            (1, ACTION_DIM),
            f"bootstrap action {name}",
        )[0]

    clip = _single_numpy_tensor(
        action.raw_clip_mask_native,
        (1, ACTION_DIM),
        "bootstrap raw clip mask",
        floating=False,
    )[0]
    if clip.dtype != np.bool_:
        raise ValueError("bootstrap raw clip mask must be bool")
    return {
        "teacher_action_applied_hardware23": vector("raw_action"),
        "teacher_candidate_target_hardware23": vector("candidate_target_hardware"),
        "teacher_label_raw_native23": vector("plain_sonic_raw_action_native"),
        "teacher_applied_safe_native23": vector("safe_native_action"),
        "teacher_final_target_hardware23": vector("processed_action"),
        "raw_clip_mask_native23": clip,
    }


def run_teacher_bootstrap_collection(
    request: BootstrapCollectionRequest,
    *,
    preflight: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run one exact 510-transition teacher-controlled simulator collection."""

    if type(request) is not BootstrapCollectionRequest:
        raise TypeError("request must be exact BootstrapCollectionRequest")
    report = (
        preflight_bootstrap_collection(request)
        if preflight is None
        else _mapping(preflight, "bootstrap preflight")
    )
    if report.get("ready") is not True or report.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("bootstrap preflight is not exact and ready")
    report_teacher = _mapping(report.get("teacher"), "bootstrap preflight teacher")
    report_encoder = _mapping(report.get("causal_encoder"), "bootstrap preflight causal_encoder")
    report_motion = _mapping(report.get("motion"), "bootstrap preflight motion")
    report_composite = _mapping(report.get("composite_contract"), "bootstrap preflight composite_contract")
    if (
        report_teacher.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or report_teacher.get("actor_state_sha256") != ACTOR_STATE_SHA256
        or report_teacher.get("onnx_sha256") != ONNX_SHA256
        or report_encoder.get("encoder_sha256") != CAUSAL_ENCODER_SHA256
        or report_encoder.get("decoder_bundle_sha256") != CAUSAL_DECODER_SHA256
        or report_encoder.get("metadata_file_sha256") != CAUSAL_METADATA_FILE_SHA256
        or report_motion.get("sha256") != DAD_DANCE_SHA256
        or report_motion.get("frame_count") != DAD_DANCE_FRAME_COUNT
        or report_composite.get("sha256") != COMPOSITE_CONTRACT_SHA256
    ):
        raise ValueError("bootstrap preflight artifact identity drift")
    root = request.root
    motion = (root / DAD_DANCE_RELATIVE_PATH).resolve(strict=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FIXED_GPU)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"

    import gear_sonic.utils.g1_true23_native124_selected_source_nominal_qualification as qualification
    import gear_sonic.utils.g1_true23_native124_selected_v2_ankle_evaluation as evaluation

    runtime_sources = evaluation._bind_evaluation_runtime_sources(root)  # noqa: SLF001
    if resolve_rsl_runtime_binding() != report["rsl_runtime"]:
        raise RuntimeError("RSL runtime changed after bootstrap preflight")
    if executed_bootstrap_source_binding(root) != report.get("executed_sources"):
        raise RuntimeError("collector/task/MJLab source bytes changed after bootstrap preflight")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.utils.torch import configure_torch_backends

    from gear_sonic.envs.mjlab.native124_selected_v2_ankle_task import (
        audit_native124_selected_v2_ankle_task_env_cfg,
        make_native124_selected_v2_ankle_task_env_cfg,
    )
    from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
        Native124SelectedV2CausalAdaptationWrapper,
        prime_native124_selected_v2_causal_adaptation_environment,
    )
    from gear_sonic.trl.mjlab.native124_selected_v2_ankle_rsl import (
        configure_selected_v2_ankle_rsl_runner,
    )
    from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import (
        CONTRACT_SHA256 as RUNTIME_COMPOSITE_CONTRACT_SHA256,
        load_composite_mjlab_contract,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("bootstrap collector requires exactly fixed visible CUDA device 0")
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED % (2**32))
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)

    composite_contract = load_composite_mjlab_contract(root)
    if RUNTIME_COMPOSITE_CONTRACT_SHA256 != COMPOSITE_CONTRACT_SHA256:
        raise RuntimeError("bootstrap composite MJLab contract SHA drift")
    velocity_limits = np.asarray(
        composite_contract["nominal_gate"]["velocity_limit_hardware_radps"],
        dtype=np.float32,
    )
    agent_cfg = evaluation._evaluation_agent_config(  # noqa: SLF001
        root,
        motion,
        request.manifest_path,
    )
    env_cfg = make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=str(motion),
        num_envs=1,
        play=False,
        fixed_anchor_index=INITIAL_Q9,
    )
    env_cfg.seed = FIXED_SEED
    observation_semantics = _configure_bootstrap_observation_semantics(env_cfg)
    task_audit = audit_native124_selected_v2_ankle_task_env_cfg(env_cfg)
    original_episode_length_s = float(env_cfg.episode_length_s)
    env_cfg.episode_length_s = EPISODE_TIMEOUT_STEPS * CONTROL_DT_S
    if abs(original_episode_length_s - 10.0) > 1.0e-12 or abs(float(env_cfg.episode_length_s) - 10.22) > 1.0e-12:
        raise RuntimeError("bootstrap 511-step timeout extension drift")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    try:
        _audit_bootstrap_runtime_observation_semantics(env)
        wrapped = Native124SelectedV2CausalAdaptationWrapper(env, clip_actions=None)
        if wrapped.max_episode_length != EPISODE_TIMEOUT_STEPS:
            raise RuntimeError("bootstrap environment does not reserve a 511th timeout step")
        prime = prime_native124_selected_v2_causal_adaptation_environment(wrapped)
        command = env.command_manager.get_term("motion")
        if (
            evaluation._q9(command) != INITIAL_Q9  # noqa: SLF001
            or int(env.common_step_counter) != 0
            or int(env._sim_step_counter) != 0
            or int(env.episode_length_buf[0].detach().cpu().item()) != 0
        ):
            raise RuntimeError("bootstrap did not prime without physics at q9=9")

        torch.manual_seed(FIXED_SEED)
        torch.cuda.manual_seed_all(FIXED_SEED)
        runner = MjlabOnPolicyRunner(
            wrapped,
            copy.deepcopy(agent_cfg),
            None,
            DEVICE,
        )
        integration = configure_selected_v2_ankle_rsl_runner(
            runner,
            repository_root=root,
            restart_path=None,
            expected_restart_sha256=None,
        )
        actor = integration.actor
        actor.eval()
        integration.assert_frozen_invariants()
        binding = load_checkpoint21204_binding(root)
        onnx_policy = Native124Checkpoint21204Policy(binding)
        causal_encoder = _HashBoundCausalEncoder250(Path(report["causal_encoder"]["encoder_path"]))

        observations = wrapped.get_observations()
        reset_proof = qualification.reset_diagnostic.prove_reset_actor_observation(
            observations,
            wrapped.diagnostics,
        )
        if (
            evaluation._q9(command) != INITIAL_Q9  # noqa: SLF001
            or int(env.common_step_counter) != 0
            or int(env._sim_step_counter) != 0
        ):
            raise RuntimeError("bootstrap runner construction changed q9 or stepped physics")

        actor_hash_before = tensor_state_sha256(actor.state_dict())
        if actor_hash_before != ACTOR_STATE_SHA256:
            raise RuntimeError("bootstrap RSL actor is not exact selected source")
        critic_hash_before = tensor_state_sha256(integration.critic.state_dict())
        optimizer_hash_before = safe_tree_sha256(
            runner.alg.optimizer.state_dict(),
            context="bootstrap optimizer before",
        )

        records: dict[str, list[Any]] = {name: [] for name in ARRAY_SPECS}
        parity_violation_count = 0
        action_semantics_mismatch_count = 0
        history_runtime_check_count = 0
        history_shift_check_count = 0
        previous_history_snapshot: Mapping[str, torch.Tensor] | None = None
        first_history_proof: Mapping[str, Any] | None = None
        last_history_proof: Mapping[str, Any] | None = None
        initial_command_counter = int(command.command_counter[0].detach().cpu().item())

        for transition in range(TOTAL_ROWS):
            q9_before = evaluation._q9(command)  # noqa: SLF001
            episode_before = int(env.episode_length_buf[0].detach().cpu().item())
            command_counter_before = int(command.command_counter[0].detach().cpu().item())
            common_before = int(env.common_step_counter)
            sim_before = int(env._sim_step_counter)
            expected_q9 = INITIAL_Q9 + transition
            if q9_before != expected_q9 or episode_before != transition:
                raise RuntimeError("bootstrap pre-step q9/episode sequence drift")

            history_proof = _policy_history_runtime_proof(env, observations, transition)
            current_history_snapshot = _policy_history_runtime_snapshot(env)
            if previous_history_snapshot is not None:
                _assert_policy_history_shift(
                    previous_history_snapshot,
                    current_history_snapshot,
                    transition,
                )
                history_shift_check_count += 1
            previous_history_snapshot = current_history_snapshot
            history_runtime_check_count += 1
            if first_history_proof is None:
                first_history_proof = history_proof
            last_history_proof = history_proof
            encoder267, proprio930, selected124 = _observation_arrays(observations)
            token64 = causal_encoder.run(encoder267)
            decoder994 = np.concatenate((token64, proprio930)).astype(
                np.float32,
                copy=False,
            )
            if decoder994.shape != (DECODER_DIM,):
                raise RuntimeError("bootstrap decoder994 construction drift")

            pt_tensor = evaluation._deterministic_actor_action(  # noqa: SLF001
                actor,
                observations,
            )
            pt_action = pt_tensor.detach().cpu().contiguous().numpy().copy()[0]
            onnx_action = onnx_policy.run(selected124.reshape(1, SELECTED_OBSERVATION_DIM))
            parity = float(np.max(np.abs(pt_action.astype(np.float64) - onnx_action.astype(np.float64))))
            parity_violation_count += int(parity > PARITY_ATOL)

            observations, _, dones, extras = wrapped.step(pt_tensor)
            done = bool(int(dones[0].detach().cpu().item()))
            if done:
                raise RuntimeError(
                    f"bootstrap terminated/autoreset at row {transition}, q9 {q9_before}; "
                    "no partial NPZ is eligible"
                )
            termination_names = evaluation._extra_termination_names(extras)  # noqa: SLF001
            q9_after = evaluation._q9(command)  # noqa: SLF001
            episode_after = int(env.episode_length_buf[0].detach().cpu().item())
            command_counter_after = int(command.command_counter[0].detach().cpu().item())
            common_after = int(env.common_step_counter)
            sim_after = int(env._sim_step_counter)
            resampled = _command_resampled(command)
            advanced = bool(
                q9_after == q9_before + 1
                and episode_after == episode_before + 1
                and common_after == common_before + 1
                and sim_after > sim_before
            )
            if not advanced or command_counter_after != command_counter_before or resampled:
                raise RuntimeError("bootstrap simulator/command continuity drift")
            if termination_names:
                raise RuntimeError("bootstrap nonterminal step exposed termination terms")

            action_arrays = _actual_action_arrays(env)
            semantics = qualification._runtime_action_semantics(  # noqa: SLF001
                env,
                pt_action,
                load_teacher_support_contract(root),
            )
            action_semantics_mismatch_count += int(semantics["match"] is not True)
            evidence = evaluation._step_evidence(env, velocity_limits)  # noqa: SLF001
            robot = env.scene["robot"]
            next_q = _single_numpy_tensor(
                robot.data.joint_pos,
                (1, ACTION_DIM),
                "bootstrap next joint position",
            )[0]
            next_qd = _single_numpy_tensor(
                robot.data.joint_vel,
                (1, ACTION_DIM),
                "bootstrap next joint velocity",
            )[0]
            next_gyro = _single_numpy_tensor(
                env.scene["robot/imu_ang_vel"].data,
                (1, 3),
                "bootstrap next base angular velocity",
            )[0]
            next_torso = _torso_quaternion(env)

            prefix = transition < RESET_PREFIX_ROWS
            _append(records, "row_index", transition)
            _append(records, "reset_prefix", prefix)
            _append(records, "steady_history", not prefix)
            _append(records, "history_depth", min(transition + 1, RESET_PREFIX_ROWS))
            _append(records, "reset_padding_count", max(RESET_PREFIX_ROWS - 1 - transition, 0))
            _append(records, "q9_reference_index", q9_before)
            _append(records, "control_state_index", q9_before + 1)
            _append(records, "next_control_state_index", q9_before + 2)
            _append(records, "encoder267", encoder267)
            _append(records, "token64", token64)
            _append(records, "proprio930", proprio930)
            _append(records, "decoder994", decoder994)
            _append(records, "selected_observation124", selected124)
            _append(records, "teacher_action_pt_hardware23", pt_action)
            _append(records, "teacher_action_onnx_hardware23", onnx_action)
            _append(records, "teacher_parity_max_abs", np.float32(parity))
            for name, value in action_arrays.items():
                _append(records, name, value)
            _append(records, "next_joint_position_hardware23", next_q)
            _append(records, "next_joint_velocity_hardware23", next_qd)
            _append(records, "next_base_angular_velocity3", next_gyro)
            _append(records, "next_torso_quaternion_wxyz4", next_torso)
            _append(records, "simulator_advanced", advanced)
            _append(records, "done", False)
            _append(records, "terminated", False)
            _append(records, "timed_out", False)
            _append(records, "command_resampled", resampled)
            _append(records, "base_height_m", evidence.scalars["base_height_m"])
            _append(records, "base_tilt_rad", evidence.scalars["base_tilt_rad"])
            _append(
                records,
                "maximum_joint_velocity_ratio",
                evidence.scalars["maximum_joint_velocity_ratio"],
            )
            _append(
                records,
                "maximum_actuator_force_ratio",
                evidence.scalars["maximum_actuator_force_ratio"],
            )
            _append(
                records,
                "target_tracking_rmse_rad",
                evidence.scalars["target_tracking_rmse_rad"],
            )
            _append(
                records,
                "max_abs_plain_sonic_raw_native",
                evidence.scalars["max_abs_plain_sonic_raw_native"],
            )
            for name in _COUNT_ARRAY_NAMES:
                if name == "termination_name_count":
                    value = len(termination_names)
                elif name == "nonfinite_count":
                    value = 0
                else:
                    value = evidence.counts[name]
                _append(records, name, value)

        if (
            evaluation._q9(command) != POST_COLLECTION_Q9  # noqa: SLF001
            or int(env.episode_length_buf[0].detach().cpu().item()) != TOTAL_ROWS
            or int(command.command_counter[0].detach().cpu().item()) != initial_command_counter
        ):
            raise RuntimeError("bootstrap final q9/episode/command counter drift")

        integration.assert_frozen_invariants()
        actor_hash_after = tensor_state_sha256(actor.state_dict())
        critic_hash_after = tensor_state_sha256(integration.critic.state_dict())
        optimizer_hash_after = safe_tree_sha256(
            runner.alg.optimizer.state_dict(),
            context="bootstrap optimizer after",
        )
        storage_step = getattr(getattr(runner.alg, "storage", None), "step", None)
        if not (
            actor_hash_after == actor_hash_before
            and critic_hash_after == critic_hash_before
            and optimizer_hash_after == optimizer_hash_before
            and integration.optimizer_steps == 0
            and runner.current_learning_iteration == 0
            and storage_step == 0
        ):
            raise RuntimeError("bootstrap training/frozen state changed during collection")
        if resolve_rsl_runtime_binding() != report["rsl_runtime"]:
            raise RuntimeError("RSL runtime changed during bootstrap collection")
        if executed_bootstrap_source_binding(root) != report.get("executed_sources"):
            raise RuntimeError("collector/task/MJLab source bytes changed during bootstrap collection")
        _regular_file(
            Path(report["causal_encoder"]["encoder_path"]),
            CAUSAL_ENCODER_SHA256,
            "model-250 encoder after collection",
        )

        arrays = _finalize_records(records)
        assessment = assess_bootstrap_arrays(arrays, repository_root=root)
        materials = {
            "preflight": {
                "contract_sha256": report["contract_sha256"],
                "rsl_runtime": report["rsl_runtime"],
                "executed_sources": report["executed_sources"],
                "teacher": report["teacher"],
                "causal_encoder": report["causal_encoder"],
                "motion": report["motion"],
                "composite_contract": report["composite_contract"],
            },
            "runtime_sources": runtime_sources,
            "task": {
                "base_task_audit": task_audit,
                "base_episode_steps": 500,
                "collector_episode_timeout_steps": EPISODE_TIMEOUT_STEPS,
                "collector_stopped_before_timeout_at_step": TOTAL_ROWS,
                "observation_semantics": observation_semantics,
                "policy_history_runtime": {
                    "check_count": history_runtime_check_count,
                    "all_rows_checked": history_runtime_check_count == TOTAL_ROWS,
                    "single_append_shift_checks": history_shift_check_count,
                    "all_inter_row_shifts_checked": history_shift_check_count == TOTAL_ROWS - 1,
                    "first_row": first_history_proof,
                    "last_row": last_history_proof,
                },
            },
            "prime": prime,
            "reset_seam": {
                "initial_q9": INITIAL_Q9,
                "fixed_teacher_warmup_steps": 0,
                "action_substitution": False,
                "reset_buffer_proof": reset_proof,
            },
            "teacher_runtime": {
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "actor_state_sha256_before": actor_hash_before,
                "actor_state_sha256_after": actor_hash_after,
                "actor_unchanged": actor_hash_after == actor_hash_before,
                "onnx_sha256": ONNX_SHA256,
                "parity_checks": TOTAL_ROWS,
                "parity_violation_count": parity_violation_count,
                "action_semantics_checks": TOTAL_ROWS,
                "action_semantics_mismatch_count": action_semantics_mismatch_count,
            },
            "training_state": {
                "critic_state_sha256_before": critic_hash_before,
                "critic_state_sha256_after": critic_hash_after,
                "optimizer_state_sha256_before": optimizer_hash_before,
                "optimizer_state_sha256_after": optimizer_hash_after,
                "optimizer_steps": integration.optimizer_steps,
                "current_learning_iteration": runner.current_learning_iteration,
                "rollout_storage_step": storage_step,
                "training_performed": False,
            },
            "final_state": {
                "q9_reference_index": POST_COLLECTION_Q9,
                "control_state_index": LAST_ACTION_Q9 + 2,
                "episode_length": TOTAL_ROWS,
            },
            "whole_run_assessment": assessment.to_dict(),
        }
        return arrays, _json_safe(materials)
    finally:
        env.close()


def write_bootstrap_failure_manifest_new(
    request: BootstrapCollectionRequest,
    error: BaseException,
    *,
    preflight: Mapping[str, Any] | None = None,
) -> Path:
    """Write one compact quarantine-only failure manifest and no NPZ."""

    if type(request) is not BootstrapCollectionRequest:
        raise TypeError("request must be exact BootstrapCollectionRequest")
    manifest = request.manifest_path
    # Any pre-existing NPZ means publication state is ambiguous; fail closed
    # rather than pairing a failure report with somebody else's bytes.
    _ = request.npz_path
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": FAILURE_MANIFEST_KIND,
        "contract_sha256": CONTRACT_SHA256,
        "whole_run_quarantined": True,
        "bootstrap_bc_training_candidate": False,
        "bootstrap_bc_eligible_rows": 0,
        "partial_npz_published": False,
        "error": f"{type(error).__name__}: {error}",
        "request": {
            "repository_root": str(request.repository_root),
            "output_prefix": str(request.output_prefix),
        },
        "preflight_completed": preflight is not None,
        "preflight_identity": (
            None
            if preflight is None
            else {
                "contract_sha256": preflight.get("contract_sha256"),
                "teacher": preflight.get("teacher"),
                "causal_encoder": preflight.get("causal_encoder"),
                "motion": preflight.get("motion"),
                "composite_contract": preflight.get("composite_contract"),
            }
        ),
        "publication": {
            "protocol": "temporary_fsync_then_noreplace_hardlink_commit",
            "overwrite_permitted": False,
            "npz_published": False,
        },
        "boundaries": {
            "teacher_controlled_bootstrap_only": True,
            "support_admitted": False,
            "on_policy_data": False,
            "dagger_data": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        },
    }
    body["manifest_payload_sha256"] = _sha256_bytes(canonical_json_bytes(body))
    encoded = canonical_json_bytes(_json_safe(body))
    temporary: Path | None = None
    try:
        temporary = _write_temporary_bytes(manifest.parent, encoded)
        _publish_link_new(temporary, manifest, "bootstrap failure manifest")
        temporary.unlink(missing_ok=True)
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "ACTION_DIM",
    "ARRAY_SPECS",
    "BootstrapAssessment",
    "BootstrapCollectionRequest",
    "BootstrapIssue",
    "CONTRACT_RELATIVE_PATH",
    "CONTRACT_SHA256",
    "EPISODE_TIMEOUT_STEPS",
    "INITIAL_Q9",
    "LAST_ACTION_Q9",
    "POST_COLLECTION_Q9",
    "REAL_H10_ROWS",
    "RESET_PREFIX_ROWS",
    "TOTAL_ROWS",
    "assess_bootstrap_arrays",
    "load_bootstrap_contract",
    "load_bootstrap_training_candidate",
    "preflight_bootstrap_collection",
    "publish_bootstrap_evidence_new",
    "replay_bootstrap_model_outputs",
    "run_teacher_bootstrap_collection",
    "executed_bootstrap_source_binding",
    "write_bootstrap_failure_manifest_new",
]
