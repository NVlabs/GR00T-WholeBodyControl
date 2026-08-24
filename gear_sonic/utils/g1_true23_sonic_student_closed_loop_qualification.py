"""Closed-loop simulator qualification for one hash-bound SONIC student.

The behavior controller is exactly model-250 encoder ONNX followed by one
offline-BC decoder ONNX.  Decoder output is plain SONIC raw native23; the
environment applies the external V2 safe-target transform exactly once.  This
module never trains, queries a teacher, sends network traffic, or authorizes
hardware/deployment use.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePath
import random
import tempfile
from types import MethodType
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
    DAD_DANCE_FRAME_COUNT,
    DAD_DANCE_RELATIVE_PATH,
    DAD_DANCE_SHA256,
    validate_dad_dance_motion_file,
)
from gear_sonic.envs.mjlab.sonic_true23 import (
    SONIC_TRUE23_TOKENIZER_DIM,
    prime_sonic_true23_training_environment,
)
from gear_sonic.envs.mjlab.sonic_true23_student_qualification import (
    DiagnosticSafeTargetNativeIl23JointPositionAction,
    audit_sonic_true23_student_qualification_env_cfg,
    capture_student_action_chain,
    make_sonic_true23_student_qualification_env_cfg,
)
from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    HARDWARE_23_JOINT_NAMES,
    ISAACLAB_TO_MUJOCO_DOF,
)
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    load_checkpoint21204_binding,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
    safe_target_transform_contract,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_native124_21204_bc_last_affine_ridge import (
    CONTRACT_RELATIVE_PATH as BC_CONTRACT_RELATIVE_PATH,
    CONTRACT_SHA256 as RUNTIME_BC_CONTRACT_SHA256,
    FINAL_BIAS_NAME,
    FINAL_WEIGHT_NAME,
    MANIFEST_KIND as BC_MANIFEST_KIND,
    SCHEMA_VERSION as BC_SCHEMA_VERSION,
    load_offline_bc_contract,
)
from gear_sonic.utils.g1_true23_native124_21204_bootstrap_mjlab import (
    ACTION_DIM,
    CAUSAL_BUNDLE_RELATIVE_PATH,
    CAUSAL_DECODER_SHA256,
    CAUSAL_ENCODER_SHA256,
    CONTRACT_RELATIVE_PATH as BOOTSTRAP_CONTRACT_RELATIVE_PATH,
    CONTRACT_SHA256 as BOOTSTRAP_CONTRACT_SHA256,
    DECODER_DIM,
    ENCODER_DIM,
    MANIFEST_KIND as BOOTSTRAP_MANIFEST_KIND,
    POLICY_TERM_WIDTHS,
    PROPRIO_DIM,
    RESET_PREFIX_ROWS,
    TOKEN_DIM,
    TOTAL_ROWS,
    executed_bootstrap_source_binding,
    load_bootstrap_training_candidate,
)
from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import (
    load_composite_mjlab_contract,
)
from gear_sonic.utils.g1_true23_native124_selected_source_full_clip_qualification import (
    PHASE_WINDOW_IDS,
)
from gear_sonic.utils.g1_true23_native124_selected_source_nominal_qualification import (
    selected_source_nominal_gate_contract,
)
import gear_sonic.utils.g1_true23_native124_selected_v2_ankle_evaluation as evaluation
from gear_sonic.utils.g1_true23_native124_selected_v2_full_clip_schedule import (
    CONTINUOUS_CONTRACT_SHA256,
    FULL_CLIP_CONTINUOUS_WINDOW,
    PHASE_SCHEDULE_SHA256,
    PHASE_WINDOWS,
    QUALIFICATION_SCHEDULE_SHA256,
    qualification_schedule_contract,
)
import gear_sonic.utils.g1_true23_native124_selected_v2_reset_seam_diagnostic as reset_diagnostic

QUALIFICATION_KIND = "g1_true23_sonic_student_closed_loop_qualification_v1"
QUALIFICATION_SCHEMA_VERSION = 1
DEVICE = "cuda:0"
FIXED_GPU = 0
FIXED_SEED = 20260805
MODES = ("initial510", "continuous", *PHASE_WINDOW_IDS)
BC_CONTRACT_SHA256 = "d41d5d05f90c8fef0d88ba89bd4795c8deacafac57893dc8118e70b66db1087f"
BOOTSTRAP_NPZ_RELATIVE_PATH = Path(
    "artifacts/g1_true23/native124_selected_21204_teacher_bootstrap_seed20260805_v1.npz"
)
BOOTSTRAP_MANIFEST_RELATIVE_PATH = Path(
    "artifacts/g1_true23/native124_selected_21204_teacher_bootstrap_seed20260805_v1.manifest.json"
)
BOOTSTRAP_NPZ_SHA256 = "136768fd1595265d9743d5a9e5f7ef38e431de9a57f9ff85246123a7d649f475"
BOOTSTRAP_MANIFEST_SHA256 = "5ab761c3b82d62c3a4524f1f195f6f187b91e56b41e854d186339ddcca86f4a1"
BOOTSTRAP_MANIFEST_PAYLOAD_SHA256 = "3f9d161da3faf1d0e73d8bf4c39058b9e2b58a6a42af6bc526b38da4c4eb9bd9"
SOURCE_DECODER_RELATIVE_PATH = CAUSAL_BUNDLE_RELATIVE_PATH / "causal_model_250.decoder.onnx"
ENCODER_RELATIVE_PATH = CAUSAL_BUNDLE_RELATIVE_PATH / "causal_model_250.encoder.onnx"
MODEL_PARITY_MAX_ABSOLUTE_ERROR = 1.0e-5
ACTION_LINK_MAX_ABSOLUTE_ERROR = 1.0e-6
PHYSICAL_MODEL_ASSET_TREE_RELATIVE_PATH = Path(
    "external_dependencies/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls"
)
PHYSICAL_MODEL_ASSET_FILE_COUNT = 43
PHYSICAL_MODEL_ASSET_EXTENSION_COUNTS = {".stl": 38, ".xml": 5}
FROZEN_INPUT_NAMES = (
    "motion_npz",
    "encoder",
    "source_decoder",
    "bootstrap_npz",
    "bootstrap_manifest",
    "bootstrap_contract",
    "bc_contract",
    "support_contract",
    "composite_contract",
    "teacher_tracker_manifest",
    "teacher_selection",
    "teacher_export_report",
    "teacher_checkpoint",
    "teacher_onnx",
    "teacher_resolved_env_evidence",
    "candidate_decoder",
    "candidate_manifest",
)
EXECUTED_SOURCE_RELATIVE_PATHS = (
    Path("gear_sonic/envs/mjlab/native124_selected_v2_ankle_adaptation.py"),
    Path("gear_sonic/envs/mjlab/native124_selected_v2_ankle_task.py"),
    Path("gear_sonic/envs/mjlab/native124_selected_v2_causal_adaptation.py"),
    Path("gear_sonic/envs/mjlab/sonic_true23.py"),
    Path("gear_sonic/envs/mjlab/sonic_true23_causal_history.py"),
    Path("gear_sonic/envs/mjlab/sonic_true23_causal_history_safe_target_v11.py"),
    Path("gear_sonic/envs/mjlab/sonic_true23_student_qualification.py"),
    Path("gear_sonic/utils/g1_23dof_artifact.py"),
    Path("gear_sonic/utils/g1_23dof_contract.py"),
    Path("gear_sonic/utils/g1_23dof_native124_21204_adapter.py"),
    Path("gear_sonic/utils/g1_23dof_safe_target_transform.py"),
    Path("gear_sonic/utils/g1_true23_native124_21204_bc_last_affine_ridge.py"),
    Path("gear_sonic/utils/g1_true23_native124_21204_bootstrap_mjlab.py"),
    Path("gear_sonic/utils/g1_true23_native124_21204_composite_mjlab.py"),
    Path("gear_sonic/utils/g1_true23_native124_selected_source_full_clip_qualification.py"),
    Path("gear_sonic/utils/g1_true23_native124_selected_source_nominal_qualification.py"),
    Path("gear_sonic/utils/g1_true23_native124_selected_v2_ankle_evaluation.py"),
    Path("gear_sonic/utils/g1_true23_native124_selected_v2_full_clip_schedule.py"),
    Path("gear_sonic/utils/g1_true23_native124_selected_v2_reset_seam_diagnostic.py"),
    Path("gear_sonic/utils/g1_true23_sonic_student_closed_loop_qualification.py"),
    Path("gear_sonic/utils/g1_true23_teacher_support.py"),
)

if RUNTIME_BC_CONTRACT_SHA256 != BC_CONTRACT_SHA256:
    raise RuntimeError("offline-BC contract SHA changed under student evaluator")


@dataclass(frozen=True)
class StudentQualificationWindow:
    mode: str
    anchor_q9: int
    transitions: int
    burn_in_transitions: int = 0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unsupported student qualification mode: {self.mode}")
        for name, value in (
            ("anchor_q9", self.anchor_q9),
            ("transitions", self.transitions),
            ("burn_in_transitions", self.burn_in_transitions),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.anchor_q9 < 9 or self.transitions <= 0:
            raise ValueError("student qualification window is outside causal domain")
        if not 0 <= self.burn_in_transitions < self.transitions:
            raise ValueError("student burn-in must be inside window")
        if self.last_q9 > DAD_DANCE_FRAME_COUNT - 2:
            raise ValueError("student window loses final q10 proof")

    @property
    def last_q9(self) -> int:
        return self.anchor_q9 + self.transitions - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "anchor_q9": self.anchor_q9,
            "transitions": self.transitions,
            "burn_in_transitions": self.burn_in_transitions,
            "first_scored_q9": self.anchor_q9 + self.burn_in_transitions,
            "last_q9": self.last_q9,
        }


def resolve_student_qualification_window(mode: str) -> StudentQualificationWindow:
    if mode == "initial510":
        return StudentQualificationWindow("initial510", 9, TOTAL_ROWS, 0)
    if mode == "continuous":
        source = FULL_CLIP_CONTINUOUS_WINDOW
    else:
        source = next((item for item in PHASE_WINDOWS if item.window_id == mode), None)
        if source is None:
            raise ValueError(f"unsupported student qualification mode: {mode}")
    return StudentQualificationWindow(
        mode,
        source.anchor_q9,
        source.transitions,
        source.burn_in_transitions,
    )


def qualification_scope(mode: str) -> dict[str, Any]:
    window = resolve_student_qualification_window(mode)
    return {
        "classification": "sonic_student_nominal_simulator_candidate_only",
        "student_behavior_controller": "hash_bound_model250_encoder_plus_bc_decoder_ort",
        "mode": mode,
        "window": window.to_dict(),
        "teacher_queried": False,
        "teacher_action_present": False,
        "teacher_support_qualification_performed": False,
        "teacher_labels_admitted": False,
        "on_policy_student_tuple_boundary_proven": False,
        "on_policy_student_tuple_boundary_status": "not_yet_proven",
        "dagger_data": False,
        "generalization_claimed": False,
        "disturbance_robustness_claimed": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }


@dataclass(frozen=True)
class StudentQualificationRequest:
    repository_root: Path
    candidate_manifest: Path
    expected_candidate_manifest_sha256: str
    output: Path
    mode: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, Path) for value in (self.repository_root, self.candidate_manifest, self.output)
        ):
            raise TypeError("student qualification paths must be pathlib.Path values")
        _require_sha256(
            self.expected_candidate_manifest_sha256,
            "expected_candidate_manifest_sha256",
        )
        resolve_student_qualification_window(self.mode)

    @property
    def root(self) -> Path:
        root = self.repository_root.expanduser().resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("repository_root must be a regular directory")
        return root

    @property
    def manifest_path(self) -> Path:
        candidate = self.candidate_manifest.expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        result = candidate.resolve(strict=False)
        evidence_root = (self.root / "artifacts" / "g1_true23").resolve(strict=True)
        try:
            result.relative_to(evidence_root)
        except ValueError as error:
            raise ValueError("candidate manifest must stay under artifacts/g1_true23") from error
        if result.suffix.lower() != ".json":
            raise ValueError("candidate manifest must end in .json")
        if candidate.is_symlink() or result.is_symlink():
            raise ValueError("candidate manifest may not be a symlink")
        return result

    @property
    def output_path(self) -> Path:
        return evaluation._evaluation_output_path(  # noqa: SLF001
            self.root,
            self.output.expanduser(),
        )

    @property
    def window(self) -> StudentQualificationWindow:
        return resolve_student_qualification_window(self.mode)


def _require_sha256(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _regular_file(path: Path, expected_sha256: str, context: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{context} may not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{context} is missing or has wrong SHA256")
    return resolved


def executed_source_binding(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relpath in EXECUTED_SOURCE_RELATIVE_PATHS:
        path = root / relpath
        if path.is_symlink():
            raise ValueError(f"student evaluator source may not be symlink: {relpath}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise ValueError(f"student evaluator source missing/outside root: {relpath}")
        records.append(
            {
                "path": relpath.as_posix(),
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    return {
        "schema": "g1_true23_sonic_student_evaluator_sources_v1",
        "file_count": len(records),
        "files": records,
        "binding_sha256": sha256_bytes(canonical_json_bytes(records)),
    }


def physical_model_asset_binding(root: Path) -> dict[str, Any]:
    """Hash every regular physical-model asset byte used by the G1 runtime."""

    candidate = root / PHYSICAL_MODEL_ASSET_TREE_RELATIVE_PATH
    if candidate.is_symlink():
        raise ValueError("student physical-model asset root may not be a symlink")
    tree = candidate.resolve(strict=True)
    if not tree.is_dir() or not tree.is_relative_to(root):
        raise ValueError("student physical-model asset tree is invalid")
    descendants = sorted(tree.rglob("*"), key=lambda path: path.relative_to(tree).as_posix())
    symlinks = [path for path in descendants if path.is_symlink()]
    if symlinks:
        raise ValueError(f"student physical-model asset tree contains symlink: {symlinks[0]}")
    special = [path for path in descendants if not path.is_file() and not path.is_dir()]
    if special:
        raise ValueError(f"student physical-model asset tree contains special entry: {special[0]}")
    files = [path for path in descendants if path.is_file()]
    extension_counts = {
        suffix: sum(path.suffix.lower() == suffix for path in files)
        for suffix in PHYSICAL_MODEL_ASSET_EXTENSION_COUNTS
    }
    if len(files) != PHYSICAL_MODEL_ASSET_FILE_COUNT or extension_counts != PHYSICAL_MODEL_ASSET_EXTENSION_COUNTS:
        raise ValueError("student physical-model asset inventory drift")
    entries = [
        {
            "path": path.relative_to(tree).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    return {
        "schema": "g1_true23_sonic_student_physical_model_assets_v1",
        "relative_root": PHYSICAL_MODEL_ASSET_TREE_RELATIVE_PATH.as_posix(),
        "selection": "all_recursive_regular_nonsymlink_files",
        "file_count": len(entries),
        "extension_counts": extension_counts,
        "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "files": entries,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(entries)),
    }


def _canonical_manifest(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    manifest = _regular_file(path, expected_sha256, "student candidate manifest")
    payload = manifest.read_bytes()
    try:
        body = _mapping(json.loads(payload.decode("utf-8")), "student candidate manifest")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("student candidate manifest must be strict UTF-8 JSON") from error
    if payload != canonical_json_bytes(body):
        raise ValueError("student candidate manifest must use canonical JSON bytes")
    unhashed = copy.deepcopy(dict(body))
    claimed = unhashed.pop("manifest_payload_sha256", None)
    if claimed != sha256_bytes(canonical_json_bytes(unhashed)):
        raise ValueError("student candidate manifest payload SHA256 mismatch")
    return body


def _candidate_decoder_path(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, str]:
    artifact = _mapping(manifest.get("artifact"), "candidate artifact")
    if (
        artifact.get("publishable_decoder") is not True
        or artifact.get("overwrite_permitted") is not False
        or artifact.get("publication_protocol") != "decoder_hardlink_then_manifest_hardlink_commit"
        or artifact.get("manifest_filename") != manifest_path.name
    ):
        raise ValueError("candidate artifact publication boundary drift")
    filename = artifact.get("decoder_filename")
    if type(filename) is not str or PurePath(filename).name != filename or not filename.endswith(".decoder.onnx"):
        raise ValueError("candidate decoder filename must be local .decoder.onnx")
    expected = _require_sha256(
        artifact.get("decoder_sha256"),
        "candidate decoder_sha256",
    )
    decoder = _regular_file(
        manifest_path.parent / filename,
        expected,
        "candidate decoder ONNX",
    )
    if artifact.get("decoder_size_bytes") != decoder.stat().st_size:
        raise ValueError("candidate decoder size differs from manifest")
    return decoder, expected


def _validate_candidate_manifest_fields(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    if (
        manifest.get("schema_version") != BC_SCHEMA_VERSION
        or manifest.get("kind") != BC_MANIFEST_KIND
        or manifest.get("classification") != "offline_bc_eligible_for_closed_loop_simulator_experiment"
        or manifest.get("eligible_for_closed_loop_simulator_experiment") is not True
        or manifest.get("gate_issues") != []
    ):
        raise ValueError("candidate manifest is failure/foreign/nonpassing")
    contract_entry = _mapping(manifest.get("contract"), "candidate contract")
    if (
        contract_entry.get("path") != BC_CONTRACT_RELATIVE_PATH.as_posix()
        or contract_entry.get("sha256") != BC_CONTRACT_SHA256
    ):
        raise ValueError("candidate manifest BC contract drift")
    lineage = _mapping(manifest.get("lineage"), "candidate lineage")
    required_lineage = {
        "contract_sha256": BC_CONTRACT_SHA256,
        "bootstrap_npz_sha256": BOOTSTRAP_NPZ_SHA256,
        "bootstrap_manifest_sha256": BOOTSTRAP_MANIFEST_SHA256,
        "bootstrap_manifest_payload_sha256": BOOTSTRAP_MANIFEST_PAYLOAD_SHA256,
        "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
        "causal_encoder_sha256": CAUSAL_ENCODER_SHA256,
        "source_decoder_sha256": CAUSAL_DECODER_SHA256,
        "source_checkpoint_present": False,
        "optimizer_state": None,
        "resume_capable": False,
        "bootstrap_manifest_kind": BOOTSTRAP_MANIFEST_KIND,
    }
    if any(lineage.get(name) != value for name, value in required_lineage.items()):
        raise ValueError("candidate manifest lineage drift")
    boundaries = _mapping(manifest.get("boundaries"), "candidate boundaries")
    if (
        boundaries.get("offline_behavior_cloning_only") is not True
        or boundaries.get("teacher_controlled_training_data") is not True
        or any(
            boundaries.get(name) is not False
            for name in (
                "reset_prefix_support_admitted",
                "support_qualification_performed",
                "support_admitted",
                "on_policy_data",
                "dagger_data",
                "promotion_eligible",
                "deployment_ready",
                "hardware_authorized",
                "robot_or_network_commands_permitted",
            )
        )
    ):
        raise ValueError("candidate manifest boundary overclaim")
    claims = _mapping(manifest.get("claims"), "candidate claims")
    if any(
        claims.get(name) is not False
        for name in (
            "blocked_diagnostic_is_independent_heldout_evidence",
            "all_510_fit_is_generalization_evidence",
            "closed_loop_simulator_qualified",
            "full_clip_qualified",
        )
    ):
        raise ValueError("candidate manifest claims qualification before evaluator")
    runtime = _mapping(manifest.get("runtime"), "candidate runtime")
    if (
        runtime.get("provider") != "CPUExecutionProvider"
        or runtime.get("row_count") != TOTAL_ROWS
        or runtime.get("no_simulator_or_gpu_used") is not True
        or runtime.get("no_robot_hardware_or_network_commands_performed") is not True
    ):
        raise ValueError("candidate offline runtime boundary drift")
    export = _mapping(manifest.get("export"), "candidate export")
    artifact = _mapping(manifest.get("artifact"), "candidate artifact")
    if (
        export.get("attempted") is not True
        or export.get("passed") is not True
        or export.get("only_final_affine_changed") is not True
        or export.get("encoder_unchanged") is not True
        or export.get("decoder_trunk_unchanged") is not True
        or export.get("changed_initializer_names") != [FINAL_WEIGHT_NAME, FINAL_BIAS_NAME]
        or export.get("provider") != "CPUExecutionProvider"
        or export.get("onnx_embedded_metadata_updated") is not False
        or export.get("onnx_embedded_metadata_unchanged") is not True
        or export.get("adapted_lineage_record_location") != "external_hash_bound_manifest_only"
        or export.get("candidate_decoder_sha256") != artifact.get("decoder_sha256")
        or export.get("candidate_decoder_size_bytes") != artifact.get("decoder_size_bytes")
        or export.get("same_runtime_repeat_fit_head_byte_identical") is not True
        or export.get("same_runtime_repeat_fit_onnx_byte_identical") is not True
        or export.get("cross_runtime_byte_determinism_claimed") is not False
        or float(export.get("reference_max_abs_error", math.inf)) > MODEL_PARITY_MAX_ABSOLUTE_ERROR
    ):
        raise ValueError("candidate export gate failed")
    abi = _mapping(export.get("abi"), "candidate export ABI")
    semantics = _mapping(export.get("action_semantics"), "candidate action semantics")
    if (
        abi
        != {
            "input_name": "obs_dict",
            "input_shape": [1, DECODER_DIM],
            "input_dtype": "float32",
            "output_name": "action",
            "output_shape": [1, ACTION_DIM],
            "output_dtype": "float32",
            "dynamic_axes": False,
            "opset": 13,
        }
        or semantics.get("output") != "pre_safe_transform_plain_sonic_raw_native23"
        or semantics.get("v2_transform_application_count") != 1
        or semantics.get("wrapper_action_clip") is not None
    ):
        raise ValueError("candidate export ABI/action semantics drift")
    fit_report = _mapping(manifest.get("fit"), "candidate fit")
    exported_gates = _mapping(
        export.get("actual_float32_ort_resubstitution_gates"),
        "candidate exported resubstitution gates",
    )
    if (
        fit_report.get("gate_issues") != []
        or fit_report.get("offline_fit_gates_passed") is not True
        or exported_gates.get("gate_issues") != []
        or exported_gates.get("passed") is not True
    ):
        raise ValueError("candidate offline fit gates failed")
    model = _mapping(contract.get("model"), "offline-BC model contract")
    if (
        model.get("decoder_input_name") != "obs_dict"
        or model.get("decoder_input_shape") != [1, DECODER_DIM]
        or model.get("decoder_output_name") != "action"
        or model.get("decoder_output_shape") != [1, ACTION_DIM]
        or model.get("final_weight_name") != FINAL_WEIGHT_NAME
        or model.get("final_bias_name") != FINAL_BIAS_NAME
        or model.get("onnx_opset") != 13
    ):
        raise ValueError("offline-BC decoder ABI contract drift")
    fit = _mapping(contract.get("fit"), "offline-BC fit contract")
    if (
        fit.get("label_semantics") != "pre_safe_transform_plain_sonic_raw_native23"
        or fit.get("v2_transform_application_count") != 1
    ):
        raise ValueError("offline-BC action semantics contract drift")


def qualification_contract(
    repository_root: str | Path,
    mode: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    gate = selected_source_nominal_gate_contract(root)
    offline_bc = load_offline_bc_contract(root)
    schedule = qualification_schedule_contract()
    window = resolve_student_qualification_window(mode)
    evaluator_sources = executed_source_binding(root)
    bootstrap_sources = executed_bootstrap_source_binding(root)
    physical_assets = physical_model_asset_binding(root)
    return {
        "schema": "g1_true23_sonic_student_closed_loop_contract_v1",
        "mode": mode,
        "window": window.to_dict(),
        "model": {
            "encoder_sha256": CAUSAL_ENCODER_SHA256,
            "source_decoder_sha256": CAUSAL_DECODER_SHA256,
            "candidate_manifest_kind": BC_MANIFEST_KIND,
            "offline_bc_contract_sha256": BC_CONTRACT_SHA256,
            "input": {"name": "obs_dict", "shape": [1, DECODER_DIM], "dtype": "float32"},
            "output": {"name": "action", "shape": [1, ACTION_DIM], "dtype": "float32"},
            "output_semantics": "plain_sonic_raw_native23_pre_v2",
            "embedded_safe_target_transform": False,
        },
        "bootstrap": {
            "npz_sha256": BOOTSTRAP_NPZ_SHA256,
            "manifest_sha256": BOOTSTRAP_MANIFEST_SHA256,
            "contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
            "reset_prefix_rows": RESET_PREFIX_ROWS,
            "real_h10_rows": TOTAL_ROWS - RESET_PREFIX_ROWS,
        },
        "action": {
            "external_transform": safe_target_transform_contract(),
            "application_count": 1,
            "raw_abs_strict_max": SAFE_TARGET_RAW_ACTION_CLIP,
            "wrapper_clip_actions": None,
            "warmup_or_substitution_steps": 0,
        },
        "gates": gate,
        "schedule": {
            "continuous_sha256": CONTINUOUS_CONTRACT_SHA256,
            "phase_sha256": PHASE_SCHEDULE_SHA256,
            "aggregate_sha256": QUALIFICATION_SCHEDULE_SHA256,
            "contract": schedule,
        },
        "offline_bc_contract": offline_bc,
        "executed_sources": evaluator_sources,
        "bootstrap_executed_sources": bootstrap_sources,
        "source_materials": {
            "physical_model_assets": physical_assets,
        },
        "student_on_policy_tuple": {
            "pre_action": ["q9", "control_q10", "encoder267", "token64", "policy930", "decoder994"],
            "behavior_action": "student_raw_native23",
            "post_action": ["v2_safe_native23", "final_target_hardware23", "physical_q11", "reward", "done"],
            "terminal_autoreset_observation_is_next_state": False,
        },
        "scope": qualification_scope(mode),
    }


def preflight_student_qualification(
    request: StudentQualificationRequest,
) -> dict[str, Any]:
    """Bind all immutable evidence; missing candidate remains non-runnable."""

    if type(request) is not StudentQualificationRequest:
        raise TypeError("request must be exact StudentQualificationRequest")
    root = request.root
    output = request.output_path
    window = request.window
    motion = validate_dad_dance_motion_file(root / DAD_DANCE_RELATIVE_PATH)
    _regular_file(motion, DAD_DANCE_SHA256, "DadDance motion")
    encoder = _regular_file(root / ENCODER_RELATIVE_PATH, CAUSAL_ENCODER_SHA256, "model250 encoder")
    source_decoder = _regular_file(
        root / SOURCE_DECODER_RELATIVE_PATH,
        CAUSAL_DECODER_SHA256,
        "model250 source decoder",
    )
    bootstrap_npz = _regular_file(
        root / BOOTSTRAP_NPZ_RELATIVE_PATH,
        BOOTSTRAP_NPZ_SHA256,
        "bootstrap NPZ",
    )
    bootstrap_manifest = _regular_file(
        root / BOOTSTRAP_MANIFEST_RELATIVE_PATH,
        BOOTSTRAP_MANIFEST_SHA256,
        "bootstrap manifest",
    )
    contract = qualification_contract(root, request.mode)
    report: dict[str, Any] = {
        "schema": "g1_true23_sonic_student_closed_loop_preflight_v1",
        "ready": False,
        "issues": [],
        "mode": request.mode,
        "window": window.to_dict(),
        "qualification_contract": contract,
        "qualification_contract_sha256": sha256_bytes(canonical_json_bytes(contract)),
        "candidate": {
            "manifest_path": str(request.manifest_path),
            "expected_manifest_sha256": request.expected_candidate_manifest_sha256,
            "present": request.manifest_path.exists(),
        },
        "fixed_inputs": {
            "encoder_path": str(encoder),
            "encoder_sha256": CAUSAL_ENCODER_SHA256,
            "source_decoder_path": str(source_decoder),
            "source_decoder_sha256": CAUSAL_DECODER_SHA256,
            "bootstrap_npz_path": str(bootstrap_npz),
            "bootstrap_npz_sha256": BOOTSTRAP_NPZ_SHA256,
            "bootstrap_manifest_path": str(bootstrap_manifest),
            "bootstrap_manifest_sha256": BOOTSTRAP_MANIFEST_SHA256,
            "motion_path": str(motion),
            "motion_sha256": DAD_DANCE_SHA256,
        },
        "output": str(output),
        "fixed_runtime": {
            "seed": FIXED_SEED,
            "device": DEVICE,
            "num_envs": 1,
            "encoder_provider": "CPUExecutionProvider",
            "decoder_provider": "CPUExecutionProvider",
            "wrapper_clip_actions": None,
            "fixed_warmup_or_action_substitution_steps": 0,
            "action_substitution": False,
        },
        "scope": qualification_scope(request.mode),
        "safety": {
            "simulator_constructed": False,
            "simulator_steps": 0,
            "training_updates": 0,
            "teacher_queries": 0,
            "network_used": False,
            "hardware_authorized": False,
        },
    }
    if not request.manifest_path.exists():
        report["issues"].append("candidate_manifest_missing")
        return report
    manifest = _canonical_manifest(
        request.manifest_path,
        request.expected_candidate_manifest_sha256,
    )
    offline_bc = load_offline_bc_contract(root)
    _validate_candidate_manifest_fields(manifest, offline_bc)
    decoder, decoder_sha256 = _candidate_decoder_path(request.manifest_path, manifest)
    if decoder_sha256 == CAUSAL_DECODER_SHA256:
        raise ValueError("candidate decoder bytes equal source decoder")
    arrays, bootstrap_body = load_bootstrap_training_candidate(
        bootstrap_npz,
        bootstrap_manifest,
        repository_root=root,
    )
    model_proof = validate_candidate_decoder(
        source_decoder=source_decoder,
        candidate_decoder=decoder,
        candidate_decoder_sha256=decoder_sha256,
        decoder994=arrays["decoder994"],
    )
    if bootstrap_body.get("manifest_payload_sha256") != BOOTSTRAP_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("bootstrap manifest payload changed")
    report["candidate"].update(
        {
            "manifest_sha256": request.expected_candidate_manifest_sha256,
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "kind": manifest["kind"],
            "decoder_path": str(decoder),
            "decoder_sha256": decoder_sha256,
            "model_proof": model_proof,
        }
    )
    report["frozen_input_files"] = _preflight_bound_file_specs(request, report)
    report["ready"] = True
    return report


def _preflight_bound_file_specs(
    request: StudentQualificationRequest,
    preflight: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Resolve every file whose bytes make the preflight proof meaningful."""

    fixed = _mapping(preflight.get("fixed_inputs"), "student fixed inputs")
    candidate = _mapping(preflight.get("candidate"), "student candidate")
    contract = _mapping(preflight.get("qualification_contract"), "student qualification contract")
    gate = _mapping(contract.get("gates"), "student qualification gates")
    threshold_sources = _mapping(gate.get("threshold_sources"), "student gate threshold sources")
    support = _mapping(threshold_sources.get("support_config"), "student support contract source")
    composite = _mapping(
        threshold_sources.get("composite_mjlab_contract"),
        "student composite contract source",
    )
    teacher = load_checkpoint21204_binding(request.root)
    specs = {
        "motion_npz": {
            "path": str(fixed["motion_path"]),
            "expected_sha256": str(fixed["motion_sha256"]),
        },
        "encoder": {
            "path": str(fixed["encoder_path"]),
            "expected_sha256": str(fixed["encoder_sha256"]),
        },
        "source_decoder": {
            "path": str(fixed["source_decoder_path"]),
            "expected_sha256": str(fixed["source_decoder_sha256"]),
        },
        "bootstrap_npz": {
            "path": str(fixed["bootstrap_npz_path"]),
            "expected_sha256": str(fixed["bootstrap_npz_sha256"]),
        },
        "bootstrap_manifest": {
            "path": str(fixed["bootstrap_manifest_path"]),
            "expected_sha256": str(fixed["bootstrap_manifest_sha256"]),
        },
        "bootstrap_contract": {
            "path": str((request.root / BOOTSTRAP_CONTRACT_RELATIVE_PATH).resolve(strict=True)),
            "expected_sha256": BOOTSTRAP_CONTRACT_SHA256,
        },
        "bc_contract": {
            "path": str((request.root / BC_CONTRACT_RELATIVE_PATH).resolve(strict=True)),
            "expected_sha256": BC_CONTRACT_SHA256,
        },
        "support_contract": {
            "path": str(support["path"]),
            "expected_sha256": str(support["sha256"]),
        },
        "composite_contract": {
            "path": str(composite["path"]),
            "expected_sha256": str(composite["sha256"]),
        },
        "teacher_tracker_manifest": {
            "path": str(teacher.manifest_path),
            "expected_sha256": teacher.manifest_sha256,
        },
        "teacher_selection": {
            "path": str(teacher.selection_path),
            "expected_sha256": teacher.selection_sha256,
        },
        "teacher_export_report": {
            "path": str(teacher.export_report_path),
            "expected_sha256": teacher.export_report_sha256,
        },
        "teacher_checkpoint": {
            "path": str(teacher.checkpoint_path),
            "expected_sha256": teacher.checkpoint_sha256,
        },
        "teacher_onnx": {
            "path": str(teacher.onnx_path),
            "expected_sha256": teacher.onnx_sha256,
        },
        "teacher_resolved_env_evidence": {
            "path": str(teacher.resolved_env_evidence_path),
            "expected_sha256": teacher.resolved_env_evidence_sha256,
        },
        "candidate_decoder": {
            "path": str(candidate["decoder_path"]),
            "expected_sha256": str(candidate["decoder_sha256"]),
        },
        "candidate_manifest": {
            "path": str(request.manifest_path),
            "expected_sha256": str(candidate["manifest_sha256"]),
        },
    }
    if tuple(specs) != FROZEN_INPUT_NAMES:
        raise RuntimeError("student frozen input inventory drift")
    for name, spec in specs.items():
        path = Path(spec["path"])
        if not path.is_absolute():
            raise ValueError(f"student frozen input {name} path must be absolute")
        _require_sha256(spec["expected_sha256"], f"student frozen input {name} SHA256")
    return specs


def _snapshot_preflight_bound_files(
    specs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Hash one complete immutable-input inventory without hiding missing files."""

    if tuple(specs) != FROZEN_INPUT_NAMES:
        raise ValueError("student frozen input snapshot inventory drift")
    files: dict[str, dict[str, Any]] = {}
    for name in FROZEN_INPUT_NAMES:
        spec = _mapping(specs[name], f"student frozen input {name}")
        path = Path(str(spec.get("path")))
        expected = _require_sha256(
            spec.get("expected_sha256"),
            f"student frozen input {name} expected SHA256",
        )
        actual: str | None = None
        error_message: str | None = None
        try:
            if path.is_symlink():
                raise ValueError("path is a symlink")
            if not path.is_file():
                raise ValueError("path is not a regular file")
            actual = sha256_file(path)
        except (OSError, ValueError) as error:
            error_message = f"{type(error).__name__}: {error}"
        files[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches_expected": actual == expected,
            "error": error_message,
        }
    return {
        "schema": "g1_true23_sonic_student_frozen_input_snapshot_v1",
        "file_count": len(files),
        "all_match_expected": all(item["matches_expected"] is True for item in files.values()),
        "files": files,
    }


def _frozen_input_evidence(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """Join before/after hashes into the fail-closed qualification gate."""

    before_files = _mapping(before.get("files"), "student frozen input before snapshot")
    after_files = _mapping(after.get("files"), "student frozen input after snapshot")
    if tuple(before_files) != FROZEN_INPUT_NAMES or tuple(after_files) != FROZEN_INPUT_NAMES:
        raise ValueError("student frozen input evidence inventory drift")
    files: dict[str, dict[str, Any]] = {}
    for name in FROZEN_INPUT_NAMES:
        before_item = _mapping(before_files[name], f"student frozen input {name} before")
        after_item = _mapping(after_files[name], f"student frozen input {name} after")
        if before_item.get("path") != after_item.get("path") or before_item.get(
            "expected_sha256"
        ) != after_item.get("expected_sha256"):
            raise ValueError(f"student frozen input {name} identity changed between snapshots")
        expected = before_item.get("expected_sha256")
        sha_before = before_item.get("actual_sha256")
        sha_after = after_item.get("actual_sha256")
        unchanged = bool(
            before_item.get("matches_expected") is True
            and after_item.get("matches_expected") is True
            and sha_before == expected
            and sha_after == expected
        )
        files[name] = {
            "path": before_item["path"],
            "expected_sha256": expected,
            "sha256_before": sha_before,
            "sha256_after": sha_after,
            "before_matches_expected": before_item.get("matches_expected") is True,
            "after_matches_expected": after_item.get("matches_expected") is True,
            "unchanged": unchanged,
            "before_error": before_item.get("error"),
            "after_error": after_item.get("error"),
        }
    all_unchanged = all(item["unchanged"] is True for item in files.values())
    return {
        "schema": "g1_true23_sonic_student_frozen_input_evidence_v1",
        "file_count": len(files),
        "all_preflight_bound_inputs_unchanged": all_unchanged,
        "files": files,
        "encoder_sha256_before": files["encoder"]["sha256_before"],
        "encoder_sha256_after": files["encoder"]["sha256_after"],
        "encoder_unchanged": files["encoder"]["unchanged"],
        "candidate_decoder_sha256_before": files["candidate_decoder"]["sha256_before"],
        "candidate_decoder_sha256_after": files["candidate_decoder"]["sha256_after"],
        "candidate_decoder_unchanged": files["candidate_decoder"]["unchanged"],
        "candidate_manifest_sha256_before": files["candidate_manifest"]["sha256_before"],
        "candidate_manifest_sha256_after": files["candidate_manifest"]["sha256_after"],
        "candidate_manifest_unchanged": files["candidate_manifest"]["unchanged"],
        "training_updates": 0,
    }


def _onnx_value_signature(value: Any) -> tuple[str, tuple[int, ...], int]:
    tensor = value.type.tensor_type
    dimensions: list[int] = []
    for dimension in tensor.shape.dim:
        if not dimension.HasField("dim_value"):
            raise ValueError("student ONNX must use static dimensions")
        dimensions.append(int(dimension.dim_value))
    return value.name, tuple(dimensions), int(tensor.elem_type)


def _initializer_arrays(model: Any) -> dict[str, np.ndarray]:
    from onnx import numpy_helper

    result = {
        item.name: np.ascontiguousarray(numpy_helper.to_array(item)).copy() for item in model.graph.initializer
    }
    if len(result) != len(model.graph.initializer):
        raise ValueError("student decoder has duplicate initializer names")
    return result


def _initializer_binding(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
        for name, value in sorted(arrays.items())
    }


def _torch_decoder_output(
    decoder994: np.ndarray,
    arrays: Mapping[str, np.ndarray],
) -> np.ndarray:
    import torch.nn.functional as functional

    if (
        decoder994.dtype != np.float32
        or decoder994.shape != (TOTAL_ROWS, DECODER_DIM)
        or not np.isfinite(decoder994).all()
    ):
        raise ValueError("decoder parity input must be finite float32 [510,994]")
    value = torch.from_numpy(np.ascontiguousarray(decoder994))
    with torch.no_grad():
        for index in range(8):
            weight = torch.from_numpy(arrays[f"layers.{index}.weight"])
            bias = torch.from_numpy(arrays[f"layers.{index}.bias"])
            value = functional.silu(functional.linear(value, weight, bias))
        value = functional.linear(
            value,
            torch.from_numpy(arrays[FINAL_WEIGHT_NAME]),
            torch.from_numpy(arrays[FINAL_BIAS_NAME]),
        )
    result = value.detach().contiguous().numpy()
    if result.dtype != np.float32 or result.shape != (TOTAL_ROWS, ACTION_DIM):
        raise RuntimeError("reconstructed candidate decoder output ABI drift")
    return result


def validate_candidate_decoder(
    *,
    source_decoder: Path,
    candidate_decoder: Path,
    candidate_decoder_sha256: str,
    decoder994: np.ndarray,
) -> dict[str, Any]:
    """Prove only final affine changed and all bootstrap rows match Torch."""

    _require_sha256(candidate_decoder_sha256, "candidate_decoder_sha256")
    source = _regular_file(source_decoder, CAUSAL_DECODER_SHA256, "source decoder proof")
    candidate = _regular_file(
        candidate_decoder,
        candidate_decoder_sha256,
        "candidate decoder proof",
    )
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - runtime-specific
        raise RuntimeError("onnx and onnxruntime are required for student preflight") from error
    source_model = onnx.load(source, load_external_data=False)
    candidate_model = onnx.load(candidate, load_external_data=False)
    onnx.checker.check_model(source_model, full_check=True)
    onnx.checker.check_model(candidate_model, full_check=True)
    expected_input = ("obs_dict", (1, DECODER_DIM), 1)
    expected_output = ("action", (1, ACTION_DIM), 1)
    for label, model in (("source", source_model), ("candidate", candidate_model)):
        if (
            len(model.graph.input) != 1
            or _onnx_value_signature(model.graph.input[0]) != expected_input
            or len(model.graph.output) != 1
            or _onnx_value_signature(model.graph.output[0]) != expected_output
        ):
            raise ValueError(f"{label} decoder static float32 ABI drift")
        opsets = [(entry.domain, int(entry.version)) for entry in model.opset_import]
        if opsets != [("", 13)]:
            raise ValueError(f"{label} decoder opset drift: {opsets}")
    if [node.SerializeToString() for node in source_model.graph.node] != [
        node.SerializeToString() for node in candidate_model.graph.node
    ]:
        raise ValueError("candidate decoder graph nodes differ from source")
    source_arrays = _initializer_arrays(source_model)
    candidate_arrays = _initializer_arrays(candidate_model)
    if set(source_arrays) != set(candidate_arrays):
        raise ValueError("candidate decoder initializer names differ from source")
    changed: list[str] = []
    for name, source_value in source_arrays.items():
        candidate_value = candidate_arrays[name]
        if source_value.shape != candidate_value.shape or source_value.dtype != candidate_value.dtype:
            raise ValueError(f"candidate decoder initializer ABI drift: {name}")
        if source_value.tobytes(order="C") != candidate_value.tobytes(order="C"):
            changed.append(name)
    if set(changed) != {FINAL_WEIGHT_NAME, FINAL_BIAS_NAME}:
        raise ValueError(f"candidate changed non-final or incomplete affine set: {sorted(changed)}")
    if candidate_arrays[FINAL_WEIGHT_NAME].shape != (ACTION_DIM, 512) or candidate_arrays[
        FINAL_BIAS_NAME
    ].shape != (ACTION_DIM,):
        raise ValueError("candidate final affine shape drift")

    session = ort.InferenceSession(
        str(candidate),
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("candidate decoder provider contract mismatch")
    ort_output = np.concatenate(
        [
            session.run(
                ["action"],
                {"obs_dict": row.reshape(1, DECODER_DIM)},
            )[0]
            for row in decoder994
        ],
        axis=0,
    )
    if (
        ort_output.dtype != np.float32
        or ort_output.shape != (TOTAL_ROWS, ACTION_DIM)
        or not np.isfinite(ort_output).all()
    ):
        raise ValueError("candidate ORT output must be finite float32 [510,23]")
    torch_output = _torch_decoder_output(decoder994, candidate_arrays)
    error = np.abs(ort_output.astype(np.float64) - torch_output.astype(np.float64))
    maximum = float(np.max(error))
    violation_count = int(np.count_nonzero(error > MODEL_PARITY_MAX_ABSOLUTE_ERROR))
    if violation_count or maximum > MODEL_PARITY_MAX_ABSOLUTE_ERROR:
        raise ValueError("candidate decoder Torch/ORT all-510 parity failed")
    source_binding = _initializer_binding(source_arrays)
    candidate_binding = _initializer_binding(candidate_arrays)
    frozen_names = sorted(set(source_arrays) - {FINAL_WEIGHT_NAME, FINAL_BIAS_NAME})
    frozen_descriptor = {name: source_binding[name] for name in frozen_names}
    return {
        "source_decoder_sha256": CAUSAL_DECODER_SHA256,
        "candidate_decoder_sha256": candidate_decoder_sha256,
        "static_float32_abi": True,
        "opset": 13,
        "changed_initializer_names": sorted(changed),
        "only_final_affine_changed": True,
        "frozen_initializer_count": len(frozen_names),
        "frozen_trunk_binding_sha256": sha256_bytes(canonical_json_bytes(frozen_descriptor)),
        "candidate_initializer_binding_sha256": sha256_bytes(canonical_json_bytes(candidate_binding)),
        "torch_ort_parity": {
            "check_rows": TOTAL_ROWS,
            "check_coordinates": int(error.size),
            "maximum_absolute_error": maximum,
            "p99_absolute_error": float(np.quantile(error, 0.99)),
            "violation_count": violation_count,
            "threshold": MODEL_PARITY_MAX_ABSOLUTE_ERROR,
            "passed": True,
        },
        "output_semantics": "plain_sonic_raw_native23_pre_v2",
        "safe_target_transform_embedded": False,
    }


class _HashBoundEncoder:
    def __init__(self, path: Path) -> None:
        self.path = _regular_file(path, CAUSAL_ENCODER_SHA256, "runtime encoder")
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - runtime-specific
            raise RuntimeError("onnxruntime is required for student evaluator") from error
        self.session = ort.InferenceSession(
            str(self.path),
            providers=["CPUExecutionProvider"],
        )
        inputs = [(item.name, item.shape, item.type) for item in self.session.get_inputs()]
        outputs = [(item.name, item.shape, item.type) for item in self.session.get_outputs()]
        if (
            self.session.get_providers() != ["CPUExecutionProvider"]
            or inputs != [("teleop_obs", [1, ENCODER_DIM], "tensor(float)")]
            or outputs != [("token", [1, TOKEN_DIM], "tensor(float)")]
        ):
            raise ValueError("runtime model250 encoder ABI/provider drift")

    def run(self, encoder267: np.ndarray) -> np.ndarray:
        _require_array(encoder267, (ENCODER_DIM,), "encoder267")
        output = self.session.run(
            ["token"],
            {"teleop_obs": encoder267.reshape(1, ENCODER_DIM)},
        )[0]
        _require_array(output, (1, TOKEN_DIM), "token64 batch")
        return output[0].copy()


class _HashBoundDecoder:
    def __init__(self, path: Path, expected_sha256: str) -> None:
        self.expected_sha256 = _require_sha256(expected_sha256, "runtime decoder SHA256")
        self.path = _regular_file(path, self.expected_sha256, "runtime candidate decoder")
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - runtime-specific
            raise RuntimeError("onnxruntime is required for student evaluator") from error
        self.session = ort.InferenceSession(
            str(self.path),
            providers=["CPUExecutionProvider"],
        )
        inputs = [(item.name, item.shape, item.type) for item in self.session.get_inputs()]
        outputs = [(item.name, item.shape, item.type) for item in self.session.get_outputs()]
        if (
            self.session.get_providers() != ["CPUExecutionProvider"]
            or inputs != [("obs_dict", [1, DECODER_DIM], "tensor(float)")]
            or outputs != [("action", [1, ACTION_DIM], "tensor(float)")]
        ):
            raise ValueError("runtime candidate decoder ABI/provider drift")

    def run(self, decoder994: np.ndarray) -> np.ndarray:
        _require_array(decoder994, (DECODER_DIM,), "decoder994")
        output = self.session.run(
            ["action"],
            {"obs_dict": decoder994.reshape(1, DECODER_DIM)},
        )[0]
        _require_array(output, (1, ACTION_DIM), "student raw action batch")
        return output[0].copy()


def _require_array(value: Any, shape: tuple[int, ...], context: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != shape
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{context} must be finite float32 {shape}")
    return value


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


def _observation_arrays(observations: Any) -> tuple[np.ndarray, np.ndarray]:
    keys = getattr(observations, "keys", None)
    if not callable(keys) or tuple(keys()) != ("tokenizer", "policy", "critic"):
        raise ValueError("student observation groups drift")
    tokenizer = _single_numpy_tensor(
        observations["tokenizer"],
        (1, SONIC_TRUE23_TOKENIZER_DIM),
        "student tokenizer observation",
    )[0]
    if tokenizer[0] != np.float32(1.0):
        raise ValueError("student tokenizer route bit must equal one")
    policy = _single_numpy_tensor(
        observations["policy"],
        (1, PROPRIO_DIM),
        "student policy H10 observation",
    )[0]
    return tokenizer[1:].copy(), policy


def _policy_history_snapshot(raw_env: Any) -> dict[str, torch.Tensor]:
    manager = raw_env.observation_manager
    buffers = manager._group_obs_term_history_buffer["policy"]  # noqa: SLF001
    if not isinstance(buffers, Mapping) or set(buffers) != set(POLICY_TERM_WIDTHS):
        raise ValueError("student policy history buffer set drift")
    return {name: buffers[name].buffer.detach().clone() for name in POLICY_TERM_WIDTHS}


def _policy_history_proof(
    raw_env: Any,
    observations: Mapping[str, Any],
    local_transition: int,
) -> dict[str, Any]:
    if isinstance(local_transition, bool) or not isinstance(local_transition, int) or local_transition < 0:
        raise ValueError("student local transition must be nonnegative integer")
    manager = raw_env.observation_manager
    names_by_group = manager._group_obs_term_names  # noqa: SLF001
    buffers_by_group = manager._group_obs_term_history_buffer  # noqa: SLF001
    names = tuple(names_by_group.get("policy", ()))
    if names != tuple(POLICY_TERM_WIDTHS):
        raise ValueError(f"student policy term order drift: {names}")
    buffers = buffers_by_group.get("policy")
    if not isinstance(buffers, Mapping) or set(buffers) != set(POLICY_TERM_WIDTHS):
        raise ValueError("student policy history buffers drift")
    expected_depth = min(local_transition + 1, RESET_PREFIX_ROWS)
    expected_padding = RESET_PREFIX_ROWS - expected_depth
    flattened: list[torch.Tensor] = []
    previous_action_zero = True
    for name in names:
        buffer = buffers[name]
        current_length = buffer.current_length
        history = buffer.buffer
        width = POLICY_TERM_WIDTHS[name]
        if (
            buffer.max_length != RESET_PREFIX_ROWS
            or type(current_length) is not torch.Tensor
            or current_length.shape != (1,)
            or int(current_length.detach().cpu().item()) != expected_depth
            or type(history) is not torch.Tensor
            or history.shape != (1, RESET_PREFIX_ROWS, width)
            or history.dtype != torch.float32
            or not bool(torch.isfinite(history).all())
        ):
            raise ValueError(f"student policy history tensor drift: {name}")
        if expected_padding:
            repeated = history[:, : expected_padding + 1]
            if not torch.equal(repeated, history[:, :1].expand_as(repeated)):
                raise RuntimeError(f"student reset history backfill mismatch: {name}")
        if local_transition == 0 and name == "previous_action":
            previous_action_zero = not bool(torch.count_nonzero(history))
            if not previous_action_zero:
                raise RuntimeError("student first actor observation previous-action slice is not zero")
        flattened.append(history.reshape(1, -1))
    exact = torch.cat(flattened, dim=-1)
    if not torch.equal(observations["policy"], exact):
        raise RuntimeError("student policy930 differs from term-major H10 buffers")
    return {
        "local_transition": local_transition,
        "actual_history_depth": expected_depth,
        "reset_padding_count": expected_padding,
        "term_order": list(names),
        "term_major_policy930_exact": True,
        "previous_action_slice_zero": previous_action_zero,
    }


def _assert_history_shift(
    previous: Mapping[str, torch.Tensor],
    current: Mapping[str, torch.Tensor],
    local_transition: int,
) -> None:
    if set(previous) != set(POLICY_TERM_WIDTHS) or set(current) != set(POLICY_TERM_WIDTHS):
        raise ValueError("student history shift keys drift")
    for name, width in POLICY_TERM_WIDTHS.items():
        before = previous[name]
        after = current[name]
        if before.shape != (1, RESET_PREFIX_ROWS, width) or after.shape != before.shape:
            raise ValueError(f"student history shift shape drift: {name}")
        if not torch.equal(after[:, :-1], before[:, 1:]):
            raise RuntimeError(f"student history appended more or less than once at {local_transition}: {name}")


def _command_resampled(command: Any) -> bool:
    value = getattr(command, "_causal_resampled", None)
    if value is None:
        return False
    if type(value) is not torch.Tensor or value.shape != (1,):
        raise ValueError("student causal resample flag must be tensor [1]")
    return bool(value.detach().cpu().item())


def _action_semantics(raw_env: Any, student_raw: np.ndarray) -> dict[str, Any]:
    _require_array(student_raw, (ACTION_DIM,), "student raw action")
    captured = capture_student_action_chain(raw_env)
    chain = {
        name: _single_numpy_tensor(value, (1, ACTION_DIM), f"student action {name}")[0]
        for name, value in captured.items()
        if name != "raw_clip_mask_native"
    }
    clip = _single_numpy_tensor(
        captured["raw_clip_mask_native"],
        (1, ACTION_DIM),
        "student action raw clip mask",
        floating=False,
    )[0]
    if clip.dtype != np.bool_:
        raise ValueError("student raw clip mask must be bool")
    expected_safe, expected_final = safe_target_transform_numpy(student_raw)
    raw_hardware = student_raw[np.asarray(ISAACLAB_TO_MUJOCO_DOF, dtype=np.int64)]
    expected_candidate = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32) + (
        raw_hardware * np.asarray(HARDWARE_23_ACTION_SCALE, dtype=np.float32)
    )
    expected_clip = np.abs(student_raw) >= np.float32(SAFE_TARGET_RAW_ACTION_CLIP)
    errors = {
        "raw_native": float(np.max(np.abs(chain["raw_native"].astype(np.float64) - student_raw))),
        "candidate_target_hardware": float(
            np.max(
                np.abs(
                    chain["candidate_target_hardware"].astype(np.float64) - expected_candidate.astype(np.float64)
                )
            )
        ),
        "safe_native": float(
            np.max(np.abs(chain["safe_native"].astype(np.float64) - expected_safe.astype(np.float64)))
        ),
        "final_target_hardware": float(
            np.max(np.abs(chain["final_target_hardware"].astype(np.float64) - expected_final.astype(np.float64)))
        ),
    }
    mask_match = bool(np.array_equal(clip, expected_clip))
    passed = bool(mask_match and all(value <= ACTION_LINK_MAX_ABSOLUTE_ERROR for value in errors.values()))
    return {
        "passed": passed,
        "maximum_absolute_error_by_link": errors,
        "raw_clip_mask_exact": mask_match,
        "raw_clip_coordinate_count": int(np.count_nonzero(clip)),
        "plain_sonic_raw_native_abs_max": float(np.max(np.abs(student_raw))),
        "chain": {
            **{name: value.tolist() for name, value in chain.items()},
            "raw_clip_mask_native": clip.tolist(),
        },
    }


class _ActionSemanticsAccumulator:
    def __init__(self, expected_count: int) -> None:
        self.expected_count = expected_count
        self.count = 0
        self.mismatch_count = 0
        self.raw_clip_coordinate_count = 0
        self.maximum_raw_abs = 0.0
        self.maxima = {
            "raw_native": 0.0,
            "candidate_target_hardware": 0.0,
            "safe_native": 0.0,
            "final_target_hardware": 0.0,
        }

    def add(self, value: Mapping[str, Any]) -> None:
        if self.count >= self.expected_count:
            raise RuntimeError("student action semantics received too many rows")
        errors = _mapping(
            value.get("maximum_absolute_error_by_link"),
            "student action link errors",
        )
        if set(errors) != set(self.maxima):
            raise ValueError("student action link error schema drift")
        for name, raw in errors.items():
            number = float(raw)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError("student action link error must be finite/nonnegative")
            self.maxima[name] = max(self.maxima[name], number)
        self.count += 1
        self.mismatch_count += int(value.get("passed") is not True)
        self.raw_clip_coordinate_count += int(value["raw_clip_coordinate_count"])
        self.maximum_raw_abs = max(
            self.maximum_raw_abs,
            float(value["plain_sonic_raw_native_abs_max"]),
        )

    def report(self) -> dict[str, Any]:
        return {
            "required_check_count": self.expected_count,
            "check_count": self.count,
            "mismatch_count": self.mismatch_count,
            "raw_clip_coordinate_count": self.raw_clip_coordinate_count,
            "maximum_plain_sonic_raw_native_abs": self.maximum_raw_abs,
            "maximum_absolute_error_by_link": dict(self.maxima),
            "threshold": ACTION_LINK_MAX_ABSOLUTE_ERROR,
            "passed": bool(
                self.count == self.expected_count
                and self.mismatch_count == 0
                and self.raw_clip_coordinate_count == 0
                and self.maximum_raw_abs < SAFE_TARGET_RAW_ACTION_CLIP
            ),
        }


class _TupleDigestAccumulator:
    """Hash exact student-controlled tuples without publishing full trajectory."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.count = 0
        self.first: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None

    @staticmethod
    def _array_record(name: str, value: np.ndarray) -> dict[str, Any]:
        contiguous = np.ascontiguousarray(value)
        return {
            "name": name,
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        }

    def add(
        self,
        *,
        local_transition: int,
        q9: int,
        encoder267: np.ndarray,
        token64: np.ndarray,
        policy930: np.ndarray,
        decoder994: np.ndarray,
        student_raw: np.ndarray,
        action_semantics: Mapping[str, Any],
        post_physical: Mapping[str, Any],
        reward: float,
        done: bool,
    ) -> None:
        arrays = {
            "encoder267": encoder267,
            "token64": token64,
            "policy930": policy930,
            "decoder994": decoder994,
            "student_raw_native23": student_raw,
            "safe_native23": np.asarray(
                action_semantics["chain"]["safe_native"],
                dtype=np.float32,
            ),
            "final_target_hardware23": np.asarray(
                action_semantics["chain"]["final_target_hardware"],
                dtype=np.float32,
            ),
            "post_joint_position_hardware23": np.asarray(
                post_physical["joint_position_hardware23"],
                dtype=np.float32,
            ),
            "post_joint_velocity_hardware23": np.asarray(
                post_physical["joint_velocity_hardware23"],
                dtype=np.float32,
            ),
            "post_base_angular_velocity3": np.asarray(
                post_physical["base_angular_velocity3"],
                dtype=np.float32,
            ),
            "post_torso_quaternion_wxyz4": np.asarray(
                post_physical["torso_quaternion_wxyz4"],
                dtype=np.float32,
            ),
        }
        records = [self._array_record(name, value) for name, value in arrays.items()]
        header = {
            "local_transition": local_transition,
            "q9_reference": q9,
            "control_state_q10": q9 + 1,
            "post_control_state_q11": q9 + 2,
            "reward": reward,
            "done": done,
            "arrays": records,
        }
        self._digest.update(canonical_json_bytes(header))
        for value in arrays.values():
            self._digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        excerpt = {
            **header,
            "behavior_controller": "student_encoder_decoder_ort",
            "teacher_action_present": False,
            "next_policy_observation_valid": not done,
            "autoreset_observation_is_new_episode": done,
        }
        if self.first is None:
            self.first = excerpt
        self.last = excerpt
        self.count += 1

    def report(self) -> dict[str, Any]:
        return {
            "schema": "g1_true23_student_on_policy_tuple_digest_v1",
            "tuple_count": self.count,
            "tuple_sha256": self._digest.hexdigest(),
            "first_tuple": self.first,
            "last_tuple": self.last,
            "behavior_controller_is_student": True,
            "teacher_action_present": False,
            "terminal_autoreset_observation_is_next_state": False,
        }


def _physical_state(raw_env: Any) -> dict[str, Any]:
    robot = raw_env.scene["robot"]
    if tuple(robot.joint_names) != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError("student runtime robot joint order drift")
    body_names = tuple(robot.body_names)
    if body_names.count("torso_link") != 1:
        raise ValueError("student runtime must contain one torso_link")
    torso_index = body_names.index("torso_link")
    return {
        "joint_position_hardware23": _single_numpy_tensor(
            robot.data.joint_pos,
            (1, ACTION_DIM),
            "student physical joint position",
        )[0].tolist(),
        "joint_velocity_hardware23": _single_numpy_tensor(
            robot.data.joint_vel,
            (1, ACTION_DIM),
            "student physical joint velocity",
        )[0].tolist(),
        "actuator_force_hardware23": _single_numpy_tensor(
            robot.data.actuator_force,
            (1, ACTION_DIM),
            "student physical actuator force",
        )[0].tolist(),
        "base_angular_velocity3": _single_numpy_tensor(
            raw_env.scene["robot/imu_ang_vel"].data,
            (1, 3),
            "student physical base angular velocity",
        )[0].tolist(),
        "root_position3": _single_numpy_tensor(
            robot.data.root_link_pos_w,
            (1, 3),
            "student physical root position",
        )[0].tolist(),
        "torso_quaternion_wxyz4": _single_numpy_tensor(
            robot.data.body_link_quat_w[:, torso_index, :],
            (1, 4),
            "student physical torso quaternion",
        )[0].tolist(),
    }


class _TerminalPreResetRecorder:
    """Intercept one autoreset and preserve terminal tuple/state exactly."""

    def __init__(self, raw_env: Any, velocity_limits: np.ndarray) -> None:
        self.raw_env = raw_env
        self.velocity_limits = velocity_limits
        self._original_reset = raw_env._reset_idx
        self._armed: dict[str, Any] | None = None
        self.captured: dict[str, Any] | None = None

        def observed_reset(_env: Any, env_ids: torch.Tensor | None = None) -> None:
            if self._armed is not None and int(_env.common_step_counter) > 0:
                if self.captured is not None:
                    raise RuntimeError("student terminal recorder captured more than once")
                if env_ids is None or int(env_ids.numel()) != 1 or int(env_ids.detach().cpu().item()) != 0:
                    raise RuntimeError("student terminal recorder expected environment zero")
                capture_errors: list[str] = []
                try:
                    action_semantics = _action_semantics(
                        _env,
                        self._armed["student_raw"],
                    )
                except Exception as error:  # preserve reset, fail later
                    action_semantics = None
                    capture_errors.append(f"action_semantics: {type(error).__name__}: {error}")
                try:
                    step_evidence = evaluation._step_evidence(  # noqa: SLF001
                        _env,
                        self.velocity_limits,
                    )
                except Exception as error:  # preserve reset, fail later
                    step_evidence = None
                    capture_errors.append(f"step_evidence: {type(error).__name__}: {error}")
                try:
                    physical = _physical_state(_env)
                except Exception as error:  # preserve reset, fail later
                    physical = None
                    capture_errors.append(f"physical_state: {type(error).__name__}: {error}")
                try:
                    ee_errors = reset_diagnostic.capture_exact_ee_position_errors(_env)
                except Exception as error:  # preserve reset, fail later
                    ee_errors = None
                    capture_errors.append(f"ee_errors: {type(error).__name__}: {error}")
                self.captured = {
                    **self._armed,
                    "episode_length_pre_reset": int(_env.episode_length_buf[0].detach().cpu().item()),
                    "termination_names": evaluation._termination_names(_env),  # noqa: SLF001
                    "is_timeout": bool(_env.termination_manager.time_outs[0].detach().cpu().item()),
                    "is_terminated": bool(_env.termination_manager.terminated[0].detach().cpu().item()),
                    "action_semantics": action_semantics,
                    "step_evidence": step_evidence,
                    "post_physical": physical,
                    "ee_body_position_errors": ee_errors,
                    "capture_errors": capture_errors,
                }
            self._original_reset(env_ids)

        raw_env._reset_idx = MethodType(observed_reset, raw_env)

    def arm(self, **record: Any) -> None:
        if self._armed is not None:
            raise RuntimeError("student terminal recorder already armed")
        self._armed = copy.deepcopy(record)
        self.captured = None

    def finish(self, *, done: bool) -> dict[str, Any] | None:
        if self._armed is None:
            raise RuntimeError("student terminal recorder was not armed")
        captured = self.captured
        self._armed = None
        self.captured = None
        if done != (captured is not None):
            raise RuntimeError("student terminal capture disagrees with done")
        return captured

    def restore(self) -> None:
        self.raw_env._reset_idx = self._original_reset


def _support_summary(
    rollout_summary: Mapping[str, Any],
    *,
    expected_final_timeout: bool,
    q9_discontinuity_count: int,
    nonfinite_count: int,
    action_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    scalars = rollout_summary["safety_scalars"]
    counts = rollout_summary["safety_count_totals"]
    return {
        "termination_count": int(not expected_final_timeout),
        "q9_discontinuity_count": q9_discontinuity_count,
        "nonfinite_count": nonfinite_count,
        "raw_clip_required_count": int(counts["raw_clip_coordinate_count"]),
        "action_semantics_mismatch_count": int(action_semantics["mismatch_count"]),
        "target_soft_limit_violation_count": int(counts["target_soft_limit_violation_count"]),
        "actuator_target_soft_limit_violation_count": int(counts["actuator_target_soft_limit_violation_count"]),
        "measured_soft_limit_violation_count": int(counts["measured_soft_limit_violation_count"]),
        "joint_velocity_limit_violation_count": int(counts["joint_velocity_limit_violation_count"]),
        "minimum_base_height_m": float(scalars["base_height_m"]["minimum"]),
        "maximum_base_tilt_rad": float(scalars["base_tilt_rad"]["maximum"]),
        "maximum_joint_velocity_ratio": float(scalars["maximum_joint_velocity_ratio"]["maximum"]),
        "maximum_tracking_rmse_rad": float(scalars["target_tracking_rmse_rad"]["maximum"]),
        "maximum_plain_sonic_raw_native_abs": float(scalars["max_abs_plain_sonic_raw_native"]["maximum"]),
        "maximum_actuator_force_ratio_evidence_only": float(scalars["maximum_actuator_force_ratio"]["maximum"]),
        "maximum_projection_linf_rad_evidence_only": float(scalars["full_v2_projection_linf_rad"]["maximum"]),
    }


def _claims(mode: str, qualified: bool) -> dict[str, Any]:
    return {
        "initial_510_nominal_slice": {
            "performed": mode == "initial510",
            "qualified": qualified if mode == "initial510" else None,
        },
        "continuous_full_clip_reachability": {
            "performed": mode == "continuous",
            "qualified": qualified if mode == "continuous" else None,
        },
        "phase_restartability": {
            "performed": mode in PHASE_WINDOW_IDS,
            "window_id": mode if mode in PHASE_WINDOW_IDS else None,
            "qualified": qualified if mode in PHASE_WINDOW_IDS else None,
        },
        "student_on_policy_tuple_boundary_proven": qualified,
        "cross_mode_inference_permitted": False,
        "teacher_queried": False,
        "teacher_support_or_label_admission": False,
        "dagger_data": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
    }


def assess_student_qualification(
    *,
    window: StudentQualificationWindow,
    gate: Mapping[str, Any],
    preflight_ready: bool,
    reset_seam: Mapping[str, Any],
    first_done: Mapping[str, Any] | None,
    attempted: int,
    partition_counts: Mapping[str, Any],
    history: Mapping[str, Any],
    rollout_summary: Mapping[str, Any],
    support_summary: Mapping[str, Any],
    action_semantics: Mapping[str, Any],
    tuple_boundary: Mapping[str, Any],
    frozen_models: Mapping[str, Any],
    partial_failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Pure fail-closed decision for one exact student-controlled window."""

    if resolve_student_qualification_window(window.mode) != window:
        raise ValueError("student assessment window is not pinned")
    reset_pass = bool(
        reset_seam.get("prime_q9") == window.anchor_q9
        and reset_seam.get("first_student_action_q9") == window.anchor_q9
        and reset_seam.get("fixed_warmup_or_action_substitution_steps") == 0
        and reset_seam.get("action_substitution") is False
        and reset_seam.get("first_policy_history", {}).get("previous_action_slice_zero") is True
        and reset_seam.get("causal_q9_buffer_finite") is True
    )
    expected_timeout = bool(
        first_done is not None
        and first_done.get("transition") == window.transitions - 1
        and first_done.get("q9_before") == window.last_q9
        and first_done.get("q9_after_autoreset") == window.anchor_q9
        and first_done.get("episode_length_pre_reset") == window.transitions
        and first_done.get("termination_names") == ["time_out"]
        and first_done.get("is_timeout") is True
        and first_done.get("is_terminated") is False
        and first_done.get("terminal_capture_errors") == []
        and first_done.get("autoreset_history", {}).get("previous_action_slice_zero") is True
        and attempted == window.transitions
        and rollout_summary.get("transition_count") == window.transitions
    )
    partition_pass = bool(
        partition_counts.get("burn_in_transition_count") == window.burn_in_transitions
        and partition_counts.get("scored_transition_count") == window.transitions - window.burn_in_transitions
        and partition_counts.get("unexpected_done_before_final_count") == 0
    )
    history_pass = bool(
        history.get("check_count") == window.transitions
        and history.get("single_append_shift_check_count") == window.transitions - 1
        and history.get("first", {}).get("actual_history_depth") == 1
        and history.get("first", {}).get("reset_padding_count") == RESET_PREFIX_ROWS - 1
        and history.get("first", {}).get("previous_action_slice_zero") is True
        and history.get("last", {}).get("actual_history_depth") == RESET_PREFIX_ROWS
        and history.get("autoreset", {}).get("actual_history_depth") == 1
        and history.get("autoreset", {}).get("previous_action_slice_zero") is True
    )
    required_zero = all(support_summary.get(name) == 0 for name in gate["required_zero_counts"])
    support_thresholds = bool(
        float(support_summary["minimum_base_height_m"]) >= float(gate["minimum_base_height_m"])
        and float(support_summary["maximum_base_tilt_rad"]) <= float(gate["maximum_base_tilt_rad"])
        and float(support_summary["maximum_joint_velocity_ratio"]) <= float(gate["maximum_joint_velocity_ratio"])
        and float(support_summary["maximum_tracking_rmse_rad"]) <= float(gate["maximum_tracking_rmse_rad"])
        and float(support_summary["maximum_plain_sonic_raw_native_abs"])
        < float(gate["plain_sonic_raw_abs_strict_max"])
    )
    safety_pass = bool(
        rollout_summary.get("hard_safety_violation_count") == 0
        and rollout_summary.get("soft_safety_warning_count") == 0
    )
    action_pass = bool(
        action_semantics.get("passed") is True
        and action_semantics.get("check_count") == window.transitions
        and action_semantics.get("mismatch_count") == 0
        and action_semantics.get("raw_clip_coordinate_count") == 0
    )
    tuple_pass = bool(
        tuple_boundary.get("tuple_count") == window.transitions
        and tuple_boundary.get("behavior_controller_is_student") is True
        and tuple_boundary.get("teacher_action_present") is False
        and tuple_boundary.get("terminal_autoreset_observation_is_next_state") is False
        and tuple_boundary.get("last_tuple", {}).get("done") is True
        and tuple_boundary.get("last_tuple", {}).get("next_policy_observation_valid") is False
    )
    frozen_files = frozen_models.get("files")
    frozen_pass = bool(
        isinstance(frozen_files, Mapping)
        and tuple(frozen_files) == FROZEN_INPUT_NAMES
        and frozen_models.get("file_count") == len(FROZEN_INPUT_NAMES)
        and frozen_models.get("all_preflight_bound_inputs_unchanged") is True
        and frozen_models.get("training_updates") == 0
        and all(
            isinstance(frozen_files[name], Mapping)
            and type(frozen_files[name].get("expected_sha256")) is str
            and len(frozen_files[name]["expected_sha256"]) == 64
            and frozen_files[name].get("sha256_before") == frozen_files[name]["expected_sha256"]
            and frozen_files[name].get("sha256_after") == frozen_files[name]["expected_sha256"]
            and frozen_files[name].get("before_matches_expected") is True
            and frozen_files[name].get("after_matches_expected") is True
            and frozen_files[name].get("unchanged") is True
            and frozen_files[name].get("before_error") is None
            and frozen_files[name].get("after_error") is None
            for name in FROZEN_INPUT_NAMES
        )
    )
    qualified = all(
        (
            preflight_ready,
            reset_pass,
            expected_timeout,
            partition_pass,
            history_pass,
            required_zero,
            support_thresholds,
            safety_pass,
            action_pass,
            tuple_pass,
            frozen_pass,
            partial_failure is None,
        )
    )
    scope = qualification_scope(window.mode)
    scope["on_policy_student_tuple_boundary_proven"] = qualified
    scope["on_policy_student_tuple_boundary_status"] = "proven_for_requested_window" if qualified else "not_proven"
    return {
        "qualified_requested_mode": qualified,
        "mode": window.mode,
        "window": window.to_dict(),
        "preflight_ready_gate_passed": preflight_ready,
        "reset_seam_gate_passed": reset_pass,
        "exact_terminal_timeout_gate_passed": expected_timeout,
        "burn_in_and_score_partition_gate_passed": partition_pass,
        "history_reset_and_single_append_gate_passed": history_pass,
        "support_required_zero_counts_gate_passed": required_zero,
        "support_nominal_threshold_gate_passed": support_thresholds,
        "hard_and_soft_safety_gate_passed": safety_pass,
        "v2_exactly_once_action_semantics_gate_passed": action_pass,
        "student_on_policy_tuple_boundary_gate_passed": tuple_pass,
        "all_preflight_bound_input_files_unchanged_gate_passed": frozen_pass,
        "frozen_model_files_gate_passed": frozen_pass,
        "structured_partial_failure_absent": partial_failure is None,
        "claims": _claims(window.mode, qualified),
        "scope": scope,
    }


def _structured_partial_failure(
    *,
    stage: str,
    transition: int,
    q9: int,
    error: BaseException,
    encoder267: np.ndarray | None = None,
    token64: np.ndarray | None = None,
    policy930: np.ndarray | None = None,
    decoder994: np.ndarray | None = None,
    student_raw: np.ndarray | None = None,
    action_semantics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    arrays = {
        "encoder267": encoder267,
        "token64": token64,
        "policy930": policy930,
        "decoder994": decoder994,
        "student_raw_native23": student_raw,
    }
    return {
        "schema": "g1_true23_sonic_student_structured_partial_failure_v1",
        "stage": stage,
        "transition": transition,
        "q9_before": q9,
        "control_state_q10": q9 + 1,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "pre_action_arrays": {name: (None if value is None else value.tolist()) for name, value in arrays.items()},
        "action_semantics": (None if action_semantics is None else dict(action_semantics)),
        "qualification_must_fail": True,
        "gate_weakened": False,
        "teacher_action_present": False,
    }


def _reset_seam_proof(
    raw_env: Any,
    observations: Mapping[str, Any],
    window: StudentQualificationWindow,
) -> dict[str, Any]:
    command = raw_env.command_manager.get_term("motion")
    first_history = _policy_history_proof(raw_env, observations, 0)
    action = raw_env.action_manager.get_term("joint_pos")
    if type(action) is not DiagnosticSafeTargetNativeIl23JointPositionAction:
        raise TypeError("student reset lacks exact diagnostic V11 action")
    if bool(torch.count_nonzero(action.raw_action)) or bool(torch.count_nonzero(action.safe_native_action)):
        raise RuntimeError("student reset action state is not zero")
    causal_pos = command.causal_robot_anchor_pos_w
    causal_quat = command.causal_robot_anchor_quat_w
    causal_finite = bool(
        type(causal_pos) is torch.Tensor
        and causal_pos.shape == (1, 3)
        and type(causal_quat) is torch.Tensor
        and causal_quat.shape == (1, 4)
        and torch.isfinite(causal_pos).all()
        and torch.isfinite(causal_quat).all()
    )
    if not causal_finite:
        raise RuntimeError("student causal q9 reset buffer is invalid")
    return {
        "prime_q9": evaluation._q9(command),  # noqa: SLF001
        "first_student_action_q9": window.anchor_q9,
        "fixed_warmup_or_action_substitution_steps": 0,
        "action_substitution": False,
        "first_action": "candidate_decoder_cpu_ort_raw_native23",
        "first_policy_history": first_history,
        "causal_q9_buffer_finite": causal_finite,
        "action_raw_and_safe_state_zero": True,
    }


def run_student_qualification(
    request: StudentQualificationRequest,
) -> dict[str, Any]:
    """Run one exact student-controlled window, or return not-ready evidence."""

    preflight = preflight_student_qualification(request)
    if preflight.get("ready") is not True:
        return {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "kind": QUALIFICATION_KIND,
            "qualified_requested_mode": False,
            "verdict": "student_qualification_not_ready",
            "preflight": preflight,
            "partial_failure": None,
            "claims": _claims(request.mode, False),
            "scope": qualification_scope(request.mode),
            "safety": {
                "simulator_constructed": False,
                "simulator_steps": 0,
                "training_updates": 0,
                "teacher_queries": 0,
                "hardware_or_network_commands_performed": False,
            },
        }

    root = request.root
    window = request.window
    gate = preflight["qualification_contract"]["gates"]
    motion = Path(preflight["fixed_inputs"]["motion_path"])
    encoder_path = Path(preflight["fixed_inputs"]["encoder_path"])
    candidate_path = Path(preflight["candidate"]["decoder_path"])
    candidate_sha256 = preflight["candidate"]["decoder_sha256"]
    frozen_input_specs = _mapping(
        preflight.get("frozen_input_files"),
        "student frozen input file specs",
    )
    frozen_inputs_before = _snapshot_preflight_bound_files(frozen_input_specs)
    if frozen_inputs_before.get("all_match_expected") is not True:
        raise RuntimeError("student preflight-bound input changed before rollout")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FIXED_GPU)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    runtime_sources = evaluation._bind_evaluation_runtime_sources(root)  # noqa: SLF001

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("student qualification requires fixed visible CUDA device 0")
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED % (2**32))
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)

    composite = load_composite_mjlab_contract(root)
    velocity_limits = np.asarray(
        composite["nominal_gate"]["velocity_limit_hardware_radps"],
        dtype=np.float32,
    )
    cfg = make_sonic_true23_student_qualification_env_cfg(
        motion_file=str(motion),
        num_envs=1,
        anchor_q9=window.anchor_q9,
        transitions=window.transitions,
    )
    cfg.seed = FIXED_SEED
    task_audit = audit_sonic_true23_student_qualification_env_cfg(
        cfg,
        expected_anchor_q9=window.anchor_q9,
        expected_transitions=window.transitions,
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
    recorder: _TerminalPreResetRecorder | None = None
    try:
        groups = env.observation_manager.cfg
        if (
            tuple(groups) != ("tokenizer", "policy", "critic")
            or groups["tokenizer"].enable_corruption is not True
            or groups["policy"].enable_corruption is not True
        ):
            raise RuntimeError("student runtime observation corruption/group drift")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        if wrapped.clip_actions is not None or wrapped.max_episode_length != window.transitions:
            raise RuntimeError("student wrapper clip/horizon drift")
        prime = prime_sonic_true23_training_environment(wrapped)
        observations = wrapped.get_observations()
        command = env.command_manager.get_term("motion")
        if (
            evaluation._q9(command) != window.anchor_q9  # noqa: SLF001
            or int(env.common_step_counter) != 0
            or int(env._sim_step_counter) != 0
            or int(env.episode_length_buf[0].detach().cpu().item()) != 0
        ):
            raise RuntimeError("student prime changed q9/counters")
        reset_seam = _reset_seam_proof(env, observations, window)
        encoder = _HashBoundEncoder(encoder_path)
        decoder = _HashBoundDecoder(candidate_path, candidate_sha256)
        recorder = _TerminalPreResetRecorder(env, velocity_limits)
        full_accumulator = evaluation.RolloutEvidenceAccumulator()
        burn_in_accumulator = evaluation.RolloutEvidenceAccumulator()
        scored_accumulator = evaluation.RolloutEvidenceAccumulator()
        action_accumulator = _ActionSemanticsAccumulator(window.transitions)
        tuple_accumulator = _TupleDigestAccumulator()
        history_count = 0
        history_shift_count = 0
        first_history: dict[str, Any] | None = None
        last_history: dict[str, Any] | None = None
        autoreset_history: dict[str, Any] | None = None
        previous_history: Mapping[str, torch.Tensor] | None = None
        attempted = 0
        q9_discontinuity_count = 0
        nonfinite_count = 0
        first_done: dict[str, Any] | None = None
        partial_failure: dict[str, Any] | None = None

        for transition in range(window.transitions):
            q9 = evaluation._q9(command)  # noqa: SLF001
            expected_q9 = window.anchor_q9 + transition
            if q9 != expected_q9:
                q9_discontinuity_count += 1
                partial_failure = _structured_partial_failure(
                    stage="pre_action_q9_continuity",
                    transition=transition,
                    q9=q9,
                    error=RuntimeError(f"expected q9 {expected_q9}, observed {q9}"),
                )
                break
            episode_before = int(env.episode_length_buf[0].detach().cpu().item())
            common_before = int(env.common_step_counter)
            sim_before = int(env._sim_step_counter)
            command_counter_before = int(command.command_counter[0].detach().cpu().item())
            encoder267: np.ndarray | None = None
            token64: np.ndarray | None = None
            policy930: np.ndarray | None = None
            decoder994: np.ndarray | None = None
            student_raw: np.ndarray | None = None
            try:
                proof = _policy_history_proof(env, observations, transition)
                snapshot = _policy_history_snapshot(env)
                if previous_history is not None:
                    _assert_history_shift(previous_history, snapshot, transition)
                    history_shift_count += 1
                previous_history = snapshot
                history_count += 1
                if first_history is None:
                    first_history = proof
                last_history = proof
                encoder267, policy930 = _observation_arrays(observations)
                token64 = encoder.run(encoder267)
                decoder994 = np.concatenate((token64, policy930)).astype(
                    np.float32,
                    copy=False,
                )
                _require_array(decoder994, (DECODER_DIM,), "student decoder994")
                student_raw = decoder.run(decoder994)
            except Exception as error:
                nonfinite_count += int("finite" in str(error).lower())
                partial_failure = _structured_partial_failure(
                    stage="student_pre_action_inference",
                    transition=transition,
                    q9=q9,
                    error=error,
                    encoder267=encoder267,
                    token64=token64,
                    policy930=policy930,
                    decoder994=decoder994,
                    student_raw=student_raw,
                )
                break

            action_tensor = torch.from_numpy(student_raw.reshape(1, ACTION_DIM)).to(
                device=DEVICE,
                dtype=torch.float32,
            )
            recorder.arm(
                transition=transition,
                q9_before=q9,
                encoder267=encoder267,
                token64=token64,
                policy930=policy930,
                decoder994=decoder994,
                student_raw=student_raw,
            )
            try:
                observations, _, dones, extras = wrapped.step(action_tensor)
            except Exception as error:
                partial_failure = _structured_partial_failure(
                    stage="environment_step",
                    transition=transition,
                    q9=q9,
                    error=error,
                    encoder267=encoder267,
                    token64=token64,
                    policy930=policy930,
                    decoder994=decoder994,
                    student_raw=student_raw,
                )
                break
            attempted += 1
            done = bool(int(dones[0].detach().cpu().item()))
            terminal = recorder.finish(done=done)
            if done:
                if terminal is None:
                    raise RuntimeError("student terminal lacks pre-reset evidence")
                action_semantics = terminal["action_semantics"]
                evidence = terminal["step_evidence"]
                post_physical = terminal["post_physical"]
                capture_errors = terminal["capture_errors"]
                if (
                    action_semantics is None
                    or type(evidence) is not evaluation.StepEvidence
                    or post_physical is None
                    or capture_errors
                ):
                    partial_failure = _structured_partial_failure(
                        stage="terminal_pre_autoreset_capture",
                        transition=transition,
                        q9=q9,
                        error=RuntimeError("; ".join(capture_errors) or "terminal capture incomplete"),
                        encoder267=encoder267,
                        token64=token64,
                        policy930=policy930,
                        decoder994=decoder994,
                        student_raw=student_raw,
                        action_semantics=action_semantics,
                    )
                    break
                action_accumulator.add(action_semantics)
                full_accumulator.add(evidence)
                if transition < window.burn_in_transitions:
                    burn_in_accumulator.add(evidence)
                else:
                    scored_accumulator.add(evidence)
                tuple_accumulator.add(
                    local_transition=transition,
                    q9=q9,
                    encoder267=encoder267,
                    token64=token64,
                    policy930=policy930,
                    decoder994=decoder994,
                    student_raw=student_raw,
                    action_semantics=action_semantics,
                    post_physical=post_physical,
                    reward=evidence.reward,
                    done=True,
                )
                autoreset_history = _policy_history_proof(env, observations, 0)
                termination_names = evaluation._extra_termination_names(extras)  # noqa: SLF001
                if termination_names != terminal["termination_names"]:
                    raise RuntimeError("student terminal extras/pre-reset terms disagree")
                first_done = {
                    "transition": transition,
                    "q9_before": q9,
                    "control_state_q10": q9 + 1,
                    "post_control_state_q11": q9 + 2,
                    "q9_after_autoreset": evaluation._q9(command),  # noqa: SLF001
                    "episode_length_pre_reset": terminal["episode_length_pre_reset"],
                    "termination_names": terminal["termination_names"],
                    "is_timeout": terminal["is_timeout"],
                    "is_terminated": terminal["is_terminated"],
                    "terminal_capture_errors": capture_errors,
                    "ee_body_position_errors": terminal["ee_body_position_errors"],
                    "pre_action": {
                        "encoder267": encoder267.tolist(),
                        "token64": token64.tolist(),
                        "policy930": policy930.tolist(),
                        "decoder994": decoder994.tolist(),
                        "student_raw_native23": student_raw.tolist(),
                    },
                    "action_semantics": action_semantics,
                    "post_physical_pre_autoreset": post_physical,
                    "step_evidence": evidence.to_dict(),
                    "autoreset_history": autoreset_history,
                    "next_policy_observation_valid": False,
                    "autoreset_observation_is_new_episode": True,
                }
                break

            if terminal is not None or evaluation._extra_termination_names(extras):  # noqa: SLF001
                raise RuntimeError("student nonterminal contains termination evidence")
            q9_after = evaluation._q9(command)  # noqa: SLF001
            episode_after = int(env.episode_length_buf[0].detach().cpu().item())
            common_after = int(env.common_step_counter)
            sim_after = int(env._sim_step_counter)
            command_counter_after = int(command.command_counter[0].detach().cpu().item())
            if (
                q9_after != q9 + 1
                or episode_after != episode_before + 1
                or common_after != common_before + 1
                or sim_after <= sim_before
                or command_counter_after != command_counter_before
                or _command_resampled(command)
            ):
                q9_discontinuity_count += 1
                partial_failure = _structured_partial_failure(
                    stage="post_action_continuity",
                    transition=transition,
                    q9=q9,
                    error=RuntimeError("student simulator/command continuity drift"),
                    encoder267=encoder267,
                    token64=token64,
                    policy930=policy930,
                    decoder994=decoder994,
                    student_raw=student_raw,
                )
                break
            try:
                action_semantics = _action_semantics(env, student_raw)
                evidence = evaluation._step_evidence(env, velocity_limits)  # noqa: SLF001
                post_physical = _physical_state(env)
            except Exception as error:
                nonfinite_count += int("finite" in str(error).lower())
                partial_failure = _structured_partial_failure(
                    stage="post_action_evidence",
                    transition=transition,
                    q9=q9,
                    error=error,
                    encoder267=encoder267,
                    token64=token64,
                    policy930=policy930,
                    decoder994=decoder994,
                    student_raw=student_raw,
                )
                break
            action_accumulator.add(action_semantics)
            full_accumulator.add(evidence)
            if transition < window.burn_in_transitions:
                burn_in_accumulator.add(evidence)
            else:
                scored_accumulator.add(evidence)
            tuple_accumulator.add(
                local_transition=transition,
                q9=q9,
                encoder267=encoder267,
                token64=token64,
                policy930=policy930,
                decoder994=decoder994,
                student_raw=student_raw,
                action_semantics=action_semantics,
                post_physical=post_physical,
                reward=evidence.reward,
                done=False,
            )

        rollout_summary = full_accumulator.report()
        burn_in_summary = burn_in_accumulator.report()
        scored_summary = scored_accumulator.report()
        action_report = action_accumulator.report()
        tuple_report = tuple_accumulator.report()
        expected_final_timeout = bool(
            first_done is not None
            and first_done.get("transition") == window.transitions - 1
            and first_done.get("q9_before") == window.last_q9
            and first_done.get("termination_names") == ["time_out"]
            and first_done.get("is_timeout") is True
            and first_done.get("is_terminated") is False
        )
        support_summary = (
            _support_summary(
                rollout_summary,
                expected_final_timeout=expected_final_timeout,
                q9_discontinuity_count=q9_discontinuity_count,
                nonfinite_count=nonfinite_count,
                action_semantics=action_report,
            )
            if rollout_summary["transition_count"]
            else {
                **{name: 1 for name in gate["required_zero_counts"]},
                "minimum_base_height_m": -1.0e30,
                "maximum_base_tilt_rad": 1.0e30,
                "maximum_joint_velocity_ratio": 1.0e30,
                "maximum_tracking_rmse_rad": 1.0e30,
                "maximum_plain_sonic_raw_native_abs": 1.0e30,
                "maximum_actuator_force_ratio_evidence_only": 1.0e30,
                "maximum_projection_linf_rad_evidence_only": 1.0e30,
            }
        )
        frozen_inputs_after = _snapshot_preflight_bound_files(frozen_input_specs)
        frozen_models = _frozen_input_evidence(frozen_inputs_before, frozen_inputs_after)
        history_report = {
            "check_count": history_count,
            "single_append_shift_check_count": history_shift_count,
            "first": first_history,
            "last": last_history,
            "autoreset": autoreset_history,
        }
        partition_counts = {
            "burn_in_transition_count": burn_in_summary["transition_count"],
            "scored_transition_count": scored_summary["transition_count"],
            "unexpected_done_before_final_count": int(
                first_done is not None and first_done.get("transition") != window.transitions - 1
            ),
        }
        qualification = assess_student_qualification(
            window=window,
            gate=gate,
            preflight_ready=True,
            reset_seam=reset_seam,
            first_done=first_done,
            attempted=attempted,
            partition_counts=partition_counts,
            history=history_report,
            rollout_summary=rollout_summary,
            support_summary=support_summary,
            action_semantics=action_report,
            tuple_boundary=tuple_report,
            frozen_models=frozen_models,
            partial_failure=partial_failure,
        )
        if executed_source_binding(root) != preflight["qualification_contract"]["executed_sources"]:
            raise RuntimeError("student evaluator source bytes changed during rollout")
        if (
            executed_bootstrap_source_binding(root)
            != preflight["qualification_contract"]["bootstrap_executed_sources"]
        ):
            raise RuntimeError("student bootstrap/runtime source bytes changed during rollout")
        if (
            physical_model_asset_binding(root)
            != preflight["qualification_contract"]["source_materials"]["physical_model_assets"]
        ):
            raise RuntimeError("student physical-model asset bytes changed during rollout")
        report = {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "kind": QUALIFICATION_KIND,
            "qualified_requested_mode": qualification["qualified_requested_mode"],
            "verdict": (
                "student_qualification_passed"
                if qualification["qualified_requested_mode"]
                else "student_qualification_failed"
            ),
            "preflight": preflight,
            "runtime_sources": runtime_sources,
            "task_audit": task_audit,
            "prime": prime,
            "reset_seam": reset_seam,
            "attempted_transitions": attempted,
            "first_done": first_done,
            "partition_counts": partition_counts,
            "history": history_report,
            "rollout": rollout_summary,
            "burn_in_rollout": burn_in_summary,
            "scored_rollout": scored_summary,
            "support_summary": support_summary,
            "action_semantics": action_report,
            "student_on_policy_tuple_boundary": tuple_report,
            "frozen_models": frozen_models,
            "partial_failure": partial_failure,
            "qualification": qualification,
            "claims": qualification["claims"],
            "scope": qualification["scope"],
            "initial510_horizon_note": (
                "horizon510 changes only post-final autoreset versus bootstrap horizon511; "
                "all q9=9..518 pre-action states/actions remain the scored boundary"
                if window.mode == "initial510"
                else None
            ),
            "safety": {
                "simulator_only": True,
                "simulator_steps": attempted,
                "training_performed": False,
                "training_updates": 0,
                "teacher_queries": 0,
                "teacher_labels_admitted": False,
                "hardware_or_network_commands_performed": False,
                "promotion_authorized": False,
                "deployment_authorized": False,
            },
        }
        return evaluation._json_safe(report)  # noqa: SLF001
    finally:
        if recorder is not None:
            recorder.restore()
        env.close()


def failure_report(
    error: BaseException,
    request: StudentQualificationRequest,
) -> dict[str, Any]:
    """Fail-closed report for errors before structured transition capture."""

    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "kind": QUALIFICATION_KIND,
        "qualified_requested_mode": False,
        "verdict": "student_qualification_runtime_error",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "request": {
            "repository_root": str(request.repository_root),
            "candidate_manifest": str(request.candidate_manifest),
            "expected_candidate_manifest_sha256": (request.expected_candidate_manifest_sha256),
            "output": str(request.output),
            "mode": request.mode,
        },
        "partial_failure": None,
        "claims": _claims(request.mode, False),
        "scope": qualification_scope(request.mode),
        "safety": {
            "training_performed": False,
            "training_updates": 0,
            "teacher_queries": 0,
            "hardware_or_network_commands_performed": False,
            "teacher_labels_admitted": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
        },
    }


def _temporary_json(parent: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=parent,
        prefix=".sonic-student-qualification-",
        suffix=".tmp",
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def write_student_qualification_new(
    request: StudentQualificationRequest,
    report: Mapping[str, Any],
) -> Path:
    """Atomically hard-link one complete JSON; never overwrite output."""

    if type(request) is not StudentQualificationRequest:
        raise TypeError("request must be exact StudentQualificationRequest")
    output = request.output_path
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("student qualification output parent must be regular directory")
    body = evaluation._json_safe(report)  # noqa: SLF001
    payload = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = _temporary_json(output.parent, payload)
    try:
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite student qualification report: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "BC_CONTRACT_SHA256",
    "MODES",
    "QUALIFICATION_KIND",
    "StudentQualificationRequest",
    "StudentQualificationWindow",
    "assess_student_qualification",
    "executed_source_binding",
    "failure_report",
    "preflight_student_qualification",
    "qualification_contract",
    "qualification_scope",
    "resolve_student_qualification_window",
    "run_student_qualification",
    "validate_candidate_decoder",
    "write_student_qualification_new",
]
