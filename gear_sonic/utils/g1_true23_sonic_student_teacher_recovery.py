"""Mixed student/teacher SONIC recoverability diagnostic.

This is deliberately not a collector.  Both policies are evaluated at every
transition, but only compact hashes and aggregate errors leave the process.
No teacher action, training tuple, support verdict, or deployment permission is
published.  The current student owns the prefix; the exact selected-21204
teacher owns the suffix through the existing SONIC V11 action term exactly once.
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
import time
from types import MethodType
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
    TORSO_BODY_NAME,
    build_native124_selected_v2_causal_actor,
    causal_forward_difference_at_indices,
    virtual_q9_torso_quaternion,
)
from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
from gear_sonic.envs.mjlab.sonic_true23_student_qualification import (
    DiagnosticSafeTargetNativeIl23JointPositionAction,
    audit_sonic_true23_student_qualification_env_cfg,
    make_sonic_true23_student_qualification_env_cfg,
)
from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
    SELECTED_ACTOR_STATE_SHA256,
    SelectedNative124Actor,
    _load_selected_source,
    tensor_state_sha256,
)
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes, sha256_bytes, sha256_file
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTION_SCALE_HARDWARE,
    ACTOR_STATE_SHA256,
    CHECKPOINT_SHA256,
    HOME_Q_HARDWARE,
    ONNX_SHA256,
    Native124Checkpoint21204Binding,
    Native124Checkpoint21204Policy,
    checkpoint21204_raw_action_to_hardware_targets,
    hardware_targets_to_checkpoint21204_raw_action,
    load_checkpoint21204_binding,
)
from gear_sonic.utils.g1_true23_native124_21204_bootstrap_mjlab import (
    ACTION_DIM,
    CONTRACT_SHA256 as BOOTSTRAP_CONTRACT_SHA256,
    DECODER_DIM,
)
from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import (
    CONTRACT_SHA256 as COMPOSITE_CONTRACT_SHA256,
    load_composite_mjlab_contract,
)
import gear_sonic.utils.g1_true23_native124_selected_v2_ankle_evaluation as evaluation
import gear_sonic.utils.g1_true23_native124_selected_v2_reset_seam_diagnostic as reset_diagnostic
import gear_sonic.utils.g1_true23_sonic_student_closed_loop_qualification as student
from gear_sonic.utils.g1_true23_teacher_support import (
    SUPPORT_CONFIG_SHA256,
    Checkpoint21204TeacherComposite,
    compose_checkpoint21204_teacher_action,
)

RECOVERY_KIND = "g1_true23_sonic_student_teacher_recovery_diagnostic_v3"
RECOVERY_SCHEMA_VERSION = 3
RECOVERY_CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_student_teacher_recovery_v3.json"
)
BASE_RECOVERY_CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_student_teacher_recovery_v1.json"
)
BASE_RECOVERY_CONTRACT_SHA256 = "59a414ef938d3d4144049950cab1ac01033e5df7d9217a06951f956751fc6cba"
PREVIOUS_RECOVERY_CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_student_teacher_recovery_v2.json"
)
PREVIOUS_RECOVERY_CONTRACT_SHA256 = "d2db339bbbd478e1ad5476743540dd58b49b9e14f9347a3c44866f0c5a634bbf"
RECOVERY_CONTRACT_SHA256 = "080a20679da78134a85a958e812651015a5f6bb3216fa5dad15c52dd0cfa325d"
MODE_STUDENT_TRANSITIONS = {"cutoff50": 50, "cutoff75": 75, "cutoff100": 100, "cutoff140": 140}
MODES = tuple(MODE_STUDENT_TRANSITIONS)
ANCHOR_Q9 = 9
LAST_Q9 = 518
TOTAL_TRANSITIONS = 510
CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH = Path(
    "artifacts/g1_true23/sonic_native124_21204_bc_last_affine_ridge_seed20260805_v2.manifest.json"
)
CURRENT_CANDIDATE_MANIFEST_SHA256 = "e43fb5e531fdccda46d2e28ce7a987c8d1d064e64ae58325e6caea4d758240db"
CURRENT_CANDIDATE_DECODER_SHA256 = "011740f86483323fc0f1c39ab25b784cf9411b401e56fee8b7a716664e921ee1"
TEACHER_PARITY_ATOL = 1.0e-5
TEACHER_ITERATION = 21204
ACTION_LINK_ATOL = 1.0e-6
STRICT_RAW_ABS_MAX = 10.0

RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS = (
    BASE_RECOVERY_CONTRACT_RELATIVE_PATH,
    PREVIOUS_RECOVERY_CONTRACT_RELATIVE_PATH,
    RECOVERY_CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/trl/mjlab/native124_selected_v2_ankle_runner.py"),
    Path("gear_sonic/utils/g1_true23_sonic_student_teacher_recovery.py"),
    Path("gear_sonic/scripts/diagnose_g1_true23_sonic_student_teacher_recovery.py"),
)
EXTERNAL_RUNTIME_SOURCE_ROOTS = (
    Path("external_dependencies/mjlab/src/mjlab"),
    Path("external_dependencies/unitree_rl_mjlab/src"),
)


@dataclass(frozen=True)
class RecoveryWindow:
    mode: str
    student_transitions: int
    teacher_transitions: int
    anchor_q9: int = ANCHOR_Q9
    transitions: int = TOTAL_TRANSITIONS

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unsupported recovery mode: {self.mode}")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.student_transitions, self.teacher_transitions, self.anchor_q9, self.transitions)
        ):
            raise TypeError("recovery window counters must be integers")
        if self.anchor_q9 != ANCHOR_Q9 or self.transitions != TOTAL_TRANSITIONS:
            raise ValueError("recovery window must retain the original initial510 boundary")
        if self.student_transitions + self.teacher_transitions != self.transitions:
            raise ValueError("recovery controller partition does not cover 510 transitions")
        expected_student = MODE_STUDENT_TRANSITIONS[self.mode]
        if self.student_transitions != expected_student:
            raise ValueError("recovery cutoff does not match mode")

    @property
    def teacher_first_q9(self) -> int:
        return self.anchor_q9 + self.student_transitions

    @property
    def student_last_q9(self) -> int:
        return self.teacher_first_q9 - 1

    @property
    def last_q9(self) -> int:
        return self.anchor_q9 + self.transitions - 1

    def controller(self, transition: int) -> str:
        if (
            isinstance(transition, bool)
            or not isinstance(transition, int)
            or not 0 <= transition < self.transitions
        ):
            raise ValueError("recovery transition is outside window")
        return "student" if transition < self.student_transitions else "teacher"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "anchor_q9": self.anchor_q9,
            "last_q9": self.last_q9,
            "transitions": self.transitions,
            "student_transition_count": self.student_transitions,
            "student_first_q9": self.anchor_q9,
            "student_last_q9": self.student_last_q9,
            "teacher_transition_count": self.teacher_transitions,
            "teacher_first_q9": self.teacher_first_q9,
            "teacher_last_q9": self.last_q9,
        }


def resolve_recovery_window(mode: str) -> RecoveryWindow:
    try:
        student_transitions = MODE_STUDENT_TRANSITIONS[mode]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unsupported recovery mode: {mode}") from error
    return RecoveryWindow(mode, student_transitions, TOTAL_TRANSITIONS - student_transitions)


def recovery_scope(mode: str) -> dict[str, Any]:
    window = resolve_recovery_window(mode)
    return {
        "classification": "mixed_controller_nominal_simulator_recoverability_diagnostic_only",
        "window": window.to_dict(),
        "student_inference_required_every_transition": True,
        "selected_teacher_pt_and_cpu_onnx_inference_required_every_transition": True,
        "behavior_controller_before_cutoff": "current_hash_bound_sonic_student",
        "behavior_controller_from_cutoff": "exact_selected21204_teacher_through_sonic_v11_once",
        "recoverability_diagnostic_requested": True,
        "teacher_labels_admitted": False,
        "training_arrays_present": False,
        "support_qualification_performed": False,
        "dagger_data": False,
        "training_performed": False,
        "generalization_claimed": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }


def _claims(mode: str, recovered: bool) -> dict[str, Any]:
    return {
        "mode": mode,
        "mixed_controller_recoverability_observed": recovered,
        "student_alone_qualified": False,
        "teacher_support_qualified": False,
        "teacher_labels_admitted": False,
        "dagger_data": False,
        "training_or_model_update": False,
        "promotion_or_deployment": False,
        "hardware_authorized": False,
    }


@dataclass(frozen=True)
class RecoveryRequest:
    repository_root: Path
    candidate_manifest: Path
    expected_candidate_manifest_sha256: str
    output: Path
    mode: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, Path) for value in (self.repository_root, self.candidate_manifest, self.output)
        ):
            raise TypeError("recovery paths must be pathlib.Path values")
        _require_sha256(self.expected_candidate_manifest_sha256, "expected candidate manifest SHA256")
        resolve_recovery_window(self.mode)

    @property
    def root(self) -> Path:
        value = self.repository_root.expanduser().resolve(strict=True)
        if value.is_symlink() or not value.is_dir():
            raise ValueError("repository root must be a regular directory")
        return value

    @property
    def manifest_path(self) -> Path:
        candidate = self.candidate_manifest.expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        result = candidate.resolve(strict=False)
        expected = (self.root / CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH).resolve(strict=False)
        if result != expected or candidate.is_symlink() or result.is_symlink():
            raise ValueError("recovery diagnostic requires the current pinned v2 candidate manifest")
        return result

    @property
    def output_path(self) -> Path:
        return evaluation._evaluation_output_path(self.root, self.output.expanduser())  # noqa: SLF001

    @property
    def window(self) -> RecoveryWindow:
        return resolve_recovery_window(self.mode)

    def student_request(self) -> student.StudentQualificationRequest:
        return student.StudentQualificationRequest(
            repository_root=self.repository_root,
            candidate_manifest=self.candidate_manifest,
            expected_candidate_manifest_sha256=self.expected_candidate_manifest_sha256,
            output=self.output,
            mode="initial510",
        )


def _require_sha256(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be lowercase SHA256")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def load_recovery_contract(repository_root: str | Path | None = None) -> Mapping[str, Any]:
    root = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    path = (root / RECOVERY_CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
        raise ValueError("recovery contract must be a regular repository file")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != RECOVERY_CONTRACT_SHA256:
        raise ValueError(f"recovery contract SHA256 mismatch: expected {RECOVERY_CONTRACT_SHA256}, got {actual}")
    try:
        contract = _mapping(json.loads(payload.decode("utf-8")), "recovery contract")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("recovery contract must be UTF-8 JSON") from error
    base_path = (root / BASE_RECOVERY_CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if base_path.is_symlink() or not base_path.is_file() or not base_path.is_relative_to(root):
        raise ValueError("base recovery contract must be a regular repository file")
    base_actual = sha256_file(base_path)
    if base_actual != BASE_RECOVERY_CONTRACT_SHA256:
        raise ValueError(
            f"base recovery contract SHA256 mismatch: expected {BASE_RECOVERY_CONTRACT_SHA256}, got {base_actual}"
        )
    previous_path = (root / PREVIOUS_RECOVERY_CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if previous_path.is_symlink() or not previous_path.is_file() or not previous_path.is_relative_to(root):
        raise ValueError("previous recovery contract must be a regular repository file")
    previous_actual = sha256_file(previous_path)
    if previous_actual != PREVIOUS_RECOVERY_CONTRACT_SHA256:
        raise ValueError(
            "previous recovery contract SHA256 mismatch: "
            f"expected {PREVIOUS_RECOVERY_CONTRACT_SHA256}, got {previous_actual}"
        )
    _validate_recovery_contract(contract)
    return contract


def _validate_recovery_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != 3
        or contract.get("kind") != "g1_true23_sonic_student_teacher_recovery_contract_v3"
        or contract.get("role") != "offline_mjlab_mixed_controller_recoverability_diagnostic_only"
    ):
        raise ValueError("recovery contract identity mismatch")
    candidate = _mapping(contract.get("candidate"), "recovery candidate")
    if candidate != {
        "manifest_relative_path": CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH.as_posix(),
        "manifest_sha256": CURRENT_CANDIDATE_MANIFEST_SHA256,
        "decoder_sha256": CURRENT_CANDIDATE_DECODER_SHA256,
        "encoder_sha256": student.CAUSAL_ENCODER_SHA256,
        "bc_contract_sha256": student.BC_CONTRACT_SHA256,
    }:
        raise ValueError("recovery candidate binding mismatch")
    teacher = _mapping(contract.get("teacher"), "recovery teacher")
    if (
        teacher.get("iteration") != TEACHER_ITERATION
        or teacher.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or teacher.get("actor_state_sha256") != ACTOR_STATE_SHA256
        or teacher.get("onnx_sha256") != ONNX_SHA256
        or teacher.get("pytorch_device") != student.DEVICE
        or teacher.get("onnx_provider") != "CPUExecutionProvider"
        or float(teacher.get("maximum_pt_onnx_absolute_error", math.inf)) != TEACHER_PARITY_ATOL
        or teacher.get("query_every_transition") is not True
    ):
        raise ValueError("recovery teacher binding mismatch")
    motion = _mapping(contract.get("motion"), "recovery motion")
    if (
        motion.get("relative_path") != student.DAD_DANCE_RELATIVE_PATH.as_posix()
        or motion.get("sha256") != student.DAD_DANCE_SHA256
        or motion.get("anchor_q9") != ANCHOR_Q9
        or motion.get("last_action_q9") != LAST_Q9
        or motion.get("total_transitions") != TOTAL_TRANSITIONS
        or float(motion.get("control_hz", -1.0)) != 50.0
    ):
        raise ValueError("recovery motion/window mismatch")
    modes = _mapping(contract.get("modes"), "recovery modes")
    if set(modes) != set(MODES):
        raise ValueError("recovery mode contract membership mismatch")
    for mode in MODES:
        window = resolve_recovery_window(mode)
        if modes.get(mode) != {
            "student_transition_count": window.student_transitions,
            "student_first_q9": window.anchor_q9,
            "student_last_q9": window.student_last_q9,
            "teacher_transition_count": window.teacher_transitions,
            "teacher_first_q9": window.teacher_first_q9,
            "teacher_last_q9": window.last_q9,
        }:
            raise ValueError(f"recovery mode contract mismatch: {mode}")
    bindings = _mapping(contract.get("bindings"), "recovery bindings")
    if bindings != {
        "previous_recovery_contract_sha256": PREVIOUS_RECOVERY_CONTRACT_SHA256,
        "base_recovery_contract_sha256": BASE_RECOVERY_CONTRACT_SHA256,
        "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
        "support_contract_sha256": SUPPORT_CONFIG_SHA256,
        "composite_contract_sha256": COMPOSITE_CONTRACT_SHA256,
    }:
        raise ValueError("recovery upstream contract binding mismatch")
    action = _mapping(contract.get("action_chain"), "recovery action chain")
    if (
        action.get("wrapper_clip_actions") is not None
        or float(action.get("plain_raw_native_strict_abs_max", -1.0)) != STRICT_RAW_ABS_MAX
        or action.get("v2_transform_application_count") != 1
        or float(action.get("maximum_actual_chain_absolute_error", math.inf)) != ACTION_LINK_ATOL
    ):
        raise ValueError("recovery action-chain mismatch")
    selected_state = _mapping(contract.get("selected124_state"), "recovery selected124 state")
    if (
        selected_state.get("layout") != "exact_checkpoint21204_causal_selected124"
        or selected_state.get("reset_torso")
        != "virtual_q9_torso_quaternion(current_robot_torso,reference_torso_q9,reference_torso_q10)"
        or selected_state.get("normal_torso") != "robot_torso_snapshotted_before_q9_to_q10_step"
        or selected_state.get("reference") != "joint_q9_and_q10_plus_torso_q9"
        or selected_state.get("robot_state") != "biased_joint_position_q10_joint_velocity_q10_gyro_q10"
        or selected_state.get("reset_previous_selected_raw") != "zeros23"
        or selected_state.get("normal_previous_selected_raw")
        != "(actual_final_unbiased_hardware_target-selected_home)/selected_scale"
        or selected_state.get("student_teacher_handoff_resets_state") is not False
    ):
        raise ValueError("recovery selected124 state contract mismatch")
    gates = _mapping(contract.get("gates"), "recovery gates")
    required_zero = [
        "termination_count",
        "q9_discontinuity_count",
        "nonfinite_count",
        "raw_clip_required_count",
        "action_semantics_mismatch_count",
        "target_soft_limit_violation_count",
        "actuator_target_soft_limit_violation_count",
        "measured_soft_limit_violation_count",
        "joint_velocity_limit_violation_count",
        "teacher_parity_violation_count",
        "teacher_composite_mismatch_count",
        "selected_state_mismatch_count",
    ]
    if (
        float(gates.get("minimum_base_height_m", math.inf)) != 0.45
        or float(gates.get("maximum_base_tilt_rad", -math.inf)) != 1.0
        or float(gates.get("maximum_joint_velocity_ratio", -math.inf)) != 1.0
        or float(gates.get("maximum_tracking_rmse_rad", -math.inf)) != 0.75
        or gates.get("required_zero_counts") != required_zero
        or gates.get("only_final_timeout_permitted") is not True
    ):
        raise ValueError("recovery nominal gate mismatch")
    report = _mapping(contract.get("report"), "recovery report")
    if report != {
        "exclusive_atomic_json": True,
        "compact_aggregates_and_digests_only": True,
        "terminal_ee_position_evidence": ("configured_body_scalar_xyz_norm_z_threshold_and_dominant_z_body"),
        "terminal_ee_position_arrays_permitted": False,
        "teacher_raw_or_action_arrays_permitted": False,
        "training_arrays_permitted": False,
    }:
        raise ValueError("recovery report boundary mismatch")
    boundaries = _mapping(contract.get("boundaries"), "recovery boundaries")
    required_true = {"simulator_only", "diagnostic_only", "recoverability_claim_only"}
    required_false = {
        "teacher_labels_admitted",
        "training_performed",
        "support_qualified",
        "dagger_data",
        "promotion_eligible",
        "deployment_ready",
        "hardware_authorized",
        "robot_or_network_commands_permitted",
    }
    if any(boundaries.get(name) is not True for name in required_true) or any(
        boundaries.get(name) is not False for name in required_false
    ):
        raise ValueError("recovery safety boundary mismatch")


def _compact_binding(value: Mapping[str, Any], digest_key: str) -> dict[str, Any]:
    return {
        "schema": value.get("schema"),
        "file_count": int(value.get("file_count", 0)),
        digest_key: value.get(digest_key),
    }


def executed_recovery_source_binding(root: Path) -> dict[str, Any]:
    base = student.executed_source_binding(root)
    records: list[dict[str, Any]] = []
    for relative in RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS:
        candidate = root / relative
        if candidate.is_symlink():
            raise ValueError(f"recovery source may not be symlink: {relative}")
        path = candidate.resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"recovery source missing/outside repository: {relative}")
        records.append(
            {"path": relative.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    descriptor = {"student_binding_sha256": base["binding_sha256"], "recovery_files": records}
    return {
        "schema": "g1_true23_sonic_student_teacher_recovery_sources_v1",
        "student_source_file_count": base["file_count"],
        "recovery_source_file_count": len(records),
        "file_count": int(base["file_count"]) + len(records),
        "student_binding_sha256": base["binding_sha256"],
        "recovery_files": records,
        "binding_sha256": sha256_bytes(canonical_json_bytes(descriptor)),
    }


def external_runtime_source_binding(root: Path) -> dict[str, Any]:
    """Hash every Python source under the two imported external runtime trees."""

    records: list[dict[str, Any]] = []
    for relative_root in EXTERNAL_RUNTIME_SOURCE_ROOTS:
        candidate = root / relative_root
        if candidate.is_symlink():
            raise ValueError(f"external runtime source root may not be symlink: {relative_root}")
        source_root = candidate.resolve(strict=True)
        if not source_root.is_dir() or not source_root.is_relative_to(root):
            raise ValueError(f"external runtime source root missing/outside repository: {relative_root}")
        paths = sorted(source_root.rglob("*.py"), key=lambda value: value.relative_to(root).as_posix())
        if not paths:
            raise ValueError(f"external runtime source root contains no Python sources: {relative_root}")
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"external runtime Python source is not regular: {path}")
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "schema": "g1_true23_recovery_external_runtime_python_sources_v1",
        "selection": "all_recursive_regular_nonsymlink_py_files",
        "roots": [value.as_posix() for value in EXTERNAL_RUNTIME_SOURCE_ROOTS],
        "file_count": len(records),
        "total_bytes": sum(int(record["size_bytes"]) for record in records),
        "files": records,
        "binding_sha256": sha256_bytes(canonical_json_bytes(records)),
    }


def _preflight_internal(request: RecoveryRequest) -> dict[str, Any]:
    if type(request) is not RecoveryRequest:
        raise TypeError("request must be exact RecoveryRequest")
    root = request.root
    contract = load_recovery_contract(root)
    if request.expected_candidate_manifest_sha256 != CURRENT_CANDIDATE_MANIFEST_SHA256:
        raise ValueError("recovery expected manifest SHA is not current pinned v2")
    if request.manifest_path != (root / CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH).resolve(strict=False):
        raise ValueError("recovery candidate path drift")
    base = student.preflight_student_qualification(request.student_request())
    if base.get("ready") is not True:
        return {
            "ready": False,
            "issues": list(base.get("issues", ())),
            "contract": contract,
            "student": base,
        }
    if base["candidate"]["decoder_sha256"] != CURRENT_CANDIDATE_DECODER_SHA256:
        raise ValueError("recovery candidate decoder is not current pinned v2")
    teacher = load_checkpoint21204_binding(root)
    if (
        teacher.checkpoint_sha256 != CHECKPOINT_SHA256
        or teacher.actor_state_sha256 != ACTOR_STATE_SHA256
        or teacher.onnx_sha256 != ONNX_SHA256
    ):
        raise ValueError("recovery teacher artifact binding drift")
    sources = executed_recovery_source_binding(root)
    external_sources = external_runtime_source_binding(root)
    assets = student.physical_model_asset_binding(root)
    frozen_specs = _mapping(base.get("frozen_input_files"), "student frozen specs")
    frozen_before = student._snapshot_preflight_bound_files(frozen_specs)  # noqa: SLF001
    if frozen_before.get("all_match_expected") is not True:
        raise RuntimeError("recovery frozen input mismatch before construction")
    return {
        "ready": True,
        "issues": [],
        "contract": contract,
        "contract_sha256": RECOVERY_CONTRACT_SHA256,
        "student": base,
        "teacher": teacher,
        "sources": sources,
        "external_sources": external_sources,
        "assets": assets,
        "frozen_specs": frozen_specs,
        "frozen_before": frozen_before,
    }


def _compact_preflight(preflight: Mapping[str, Any], request: RecoveryRequest) -> dict[str, Any]:
    base = _mapping(preflight.get("student"), "student preflight")
    candidate = _mapping(base.get("candidate"), "student candidate preflight")
    sources = preflight.get("sources")
    assets = preflight.get("assets")
    external_sources = preflight.get("external_sources")
    result = {
        "ready": preflight.get("ready") is True,
        "issues": list(preflight.get("issues", ())),
        "mode": request.mode,
        "window": request.window.to_dict(),
        "recovery_contract_sha256": RECOVERY_CONTRACT_SHA256,
        "candidate": {
            "manifest_relative_path": CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH.as_posix(),
            "manifest_sha256": candidate.get("manifest_sha256"),
            "decoder_sha256": candidate.get("decoder_sha256"),
            "encoder_sha256": student.CAUSAL_ENCODER_SHA256,
        },
        "teacher": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "actor_state_sha256": ACTOR_STATE_SHA256,
            "onnx_sha256": ONNX_SHA256,
        },
        "scope": recovery_scope(request.mode),
    }
    if isinstance(sources, Mapping):
        result["executed_sources"] = _compact_binding(sources, "binding_sha256")
    if isinstance(assets, Mapping):
        result["physical_model_assets"] = {
            "file_count": assets.get("file_count"),
            "total_bytes": assets.get("total_bytes"),
            "manifest_sha256": assets.get("manifest_sha256"),
        }
    if isinstance(external_sources, Mapping):
        result["external_runtime_sources"] = {
            "file_count": external_sources.get("file_count"),
            "total_bytes": external_sources.get("total_bytes"),
            "binding_sha256": external_sources.get("binding_sha256"),
        }
    frozen = preflight.get("frozen_before")
    if isinstance(frozen, Mapping):
        files = _mapping(frozen.get("files"), "recovery frozen files")
        descriptor = [{"name": name, "sha256": value.get("actual_sha256")} for name, value in files.items()]
        result["frozen_inputs"] = {
            "file_count": frozen.get("file_count"),
            "all_match_expected": frozen.get("all_match_expected"),
            "binding_sha256": sha256_bytes(canonical_json_bytes(descriptor)),
        }
    return result


def preflight_recovery(request: RecoveryRequest) -> dict[str, Any]:
    """Read-only compact preflight; does not construct a simulator or query a policy."""

    return _compact_preflight(_preflight_internal(request), request)


def selected_previous_raw_from_final_target(final_target_hardware: np.ndarray) -> np.ndarray:
    target = student._require_array(  # noqa: SLF001
        np.asarray(final_target_hardware, dtype=np.float32),
        (ACTION_DIM,),
        "final unbiased hardware target",
    )
    result = hardware_targets_to_checkpoint21204_raw_action(target)
    independent = ((target - HOME_Q_HARDWARE) / ACTION_SCALE_HARDWARE).astype(np.float32, copy=False)
    if not np.array_equal(result, independent):
        raise RuntimeError("selected previous-action inverse formula drift")
    round_trip = checkpoint21204_raw_action_to_hardware_targets(result)
    if float(np.max(np.abs(round_trip.astype(np.float64) - target.astype(np.float64)))) > ACTION_LINK_ATOL:
        raise RuntimeError("selected previous-action target round trip failed")
    return result.copy()


class _Selected124State:
    """External exact causal state needed by the selected actor in the student env."""

    def __init__(self, raw_env: Any) -> None:
        self.raw_env = raw_env
        self.command = raw_env.command_manager.get_term("motion")
        body_names = tuple(self.command.cfg.body_names)
        if body_names.count(TORSO_BODY_NAME) != 1:
            raise ValueError("selected124 state requires one motion torso_link")
        self.torso_index = body_names.index(TORSO_BODY_NAME)
        q9 = self._q9_tensor()
        reference_q9, reference_q10 = self._reference_torso(q9)
        current = self._current_robot_torso()
        self.buffered_torso = virtual_q9_torso_quaternion(current, reference_q9, reference_q10)
        self.previous_selected_raw = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=raw_env.device)
        self.build_count = 0
        self.update_count = 0
        self.autoreset_synchronization_count = 0
        self.reset_previous_zero = not bool(torch.count_nonzero(self.previous_selected_raw))
        self.reset_virtual_torso_sha256 = _tensor_sha256(self.buffered_torso)

    def _q9_tensor(self) -> torch.Tensor:
        value = self.command.time_steps
        if type(value) is not torch.Tensor or value.shape != (1,) or value.dtype != torch.long:
            raise ValueError("selected124 q9 must be int64 [1]")
        return value.detach().clone()

    def _reference_torso(self, q9: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self.command.motion.body_quat_w
        q9_value = reference[q9, self.torso_index]
        q10_value = reference[q9 + 1, self.torso_index]
        for name, value in (("q9", q9_value), ("q10", q10_value)):
            if value.shape != (1, 4) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError(f"selected reference torso {name} drift")
        return q9_value, q10_value

    def _current_robot_torso(self) -> torch.Tensor:
        value = self.command.robot_body_quat_w[:, self.torso_index].clone()
        if value.shape != (1, 4) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError("selected current robot torso drift")
        return value

    def build(self) -> tuple[torch.Tensor, torch.Tensor]:
        q9 = self._q9_tensor()
        reference_q9, reference_q10, _ = causal_forward_difference_at_indices(self.command.motion.joint_pos, q9)
        reference_torso_q9, _ = self._reference_torso(q9)
        robot = self.raw_env.scene["robot"]
        biased = getattr(robot.data, "joint_pos_biased", None)
        if type(biased) is not torch.Tensor:
            raise ValueError("selected124 requires biased joint positions")
        gyro = self.raw_env.scene["robot/imu_ang_vel"].data
        actor = build_native124_selected_v2_causal_actor(
            reference_q9_hardware=reference_q9,
            reference_q10_hardware=reference_q10,
            robot_torso_q9_wxyz=self.buffered_torso,
            reference_torso_q9_wxyz=reference_torso_q9,
            base_angular_velocity_q10=gyro,
            joint_pos_biased_q10_hardware=biased,
            joint_velocity_q10_hardware=robot.data.joint_vel,
            previous_effective_selected_raw_hardware=self.previous_selected_raw,
        )
        pre_step_torso = self._current_robot_torso()
        self.build_count += 1
        return actor, pre_step_torso

    def update_nonterminal(
        self,
        pre_step_torso: torch.Tensor,
        final_target_hardware: np.ndarray,
    ) -> dict[str, Any]:
        if pre_step_torso.shape != (1, 4) or pre_step_torso.dtype != torch.float32:
            raise ValueError("selected pre-step torso buffer drift")
        previous = selected_previous_raw_from_final_target(final_target_hardware)
        self.buffered_torso = pre_step_torso.detach().clone()
        self.previous_selected_raw = torch.from_numpy(previous.reshape(1, ACTION_DIM)).to(
            device=self.raw_env.device, dtype=torch.float32
        )
        recovered = checkpoint21204_raw_action_to_hardware_targets(previous)
        error = float(
            np.max(
                np.abs(
                    recovered.astype(np.float64)
                    - np.asarray(final_target_hardware, dtype=np.float32).astype(np.float64)
                )
            )
        )
        self.update_count += 1
        return {
            "maximum_target_round_trip_absolute_error": error,
            "passed": error <= ACTION_LINK_ATOL,
        }

    def synchronize_autoreset(self) -> dict[str, Any]:
        if evaluation._q9(self.command) != ANCHOR_Q9:  # noqa: SLF001
            raise RuntimeError("selected124 autoreset did not return to q9=9")
        q9 = self._q9_tensor()
        reference_q9, reference_q10 = self._reference_torso(q9)
        self.buffered_torso = virtual_q9_torso_quaternion(self._current_robot_torso(), reference_q9, reference_q10)
        self.previous_selected_raw = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=self.raw_env.device)
        previous_zero = not bool(torch.count_nonzero(self.previous_selected_raw))
        self.autoreset_synchronization_count += 1
        return {
            "q9": ANCHOR_Q9,
            "previous_selected_raw_is_exact_zero": previous_zero,
            "virtual_q9_torso_sha256": _tensor_sha256(self.buffered_torso),
            "synchronization_count": self.autoreset_synchronization_count,
        }

    def reset_proof(self) -> dict[str, Any]:
        return {
            "q9": evaluation._q9(self.command),  # noqa: SLF001
            "previous_selected_raw_is_exact_zero": self.reset_previous_zero,
            "virtual_q9_torso_sha256": self.reset_virtual_torso_sha256,
            "student_teacher_handoff_resets_selected_state": False,
        }


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


class _ExactTeacherPair:
    def __init__(self, binding: Native124Checkpoint21204Binding, device: str) -> None:
        actor_state, _, lineage = _load_selected_source(binding.checkpoint_path)
        if (
            lineage.actor_state_sha256 != SELECTED_ACTOR_STATE_SHA256
            or SELECTED_ACTOR_STATE_SHA256 != ACTOR_STATE_SHA256
        ):
            raise ValueError("selected PT actor-state identity mismatch")
        self.device = torch.device(device)
        self.actor = SelectedNative124Actor()
        self.actor.load_state_dict(actor_state, strict=True)
        self.actor.to(self.device)
        self.actor.requires_grad_(False)
        self.actor.eval()
        self.onnx = Native124Checkpoint21204Policy(binding)
        self.state_sha256_before = tensor_state_sha256(self.actor.state_dict())
        if self.state_sha256_before != ACTOR_STATE_SHA256:
            raise ValueError("selected PT actor content mismatch after construction")
        self.query_count = 0
        self.violation_count = 0
        self.maximum_absolute_error = 0.0
        self.total_absolute_error = 0.0
        self.maximum_pt_duration_ms = 0.0
        self.maximum_onnx_duration_ms = 0.0

    def infer(self, selected124: torch.Tensor) -> tuple[np.ndarray, float]:
        if selected124.shape != (1, 124) or selected124.dtype != torch.float32:
            raise ValueError("selected teacher observation must be float32 [1,124]")
        observation = selected124.detach().to(device="cpu").contiguous().numpy().copy()
        start = time.perf_counter_ns()
        with torch.inference_mode():
            pt_tensor = self.actor(selected124.to(device=self.device, dtype=torch.float32))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        pt_ms = (time.perf_counter_ns() - start) / 1.0e6
        pt = pt_tensor.detach().to(device="cpu").contiguous().numpy()[0].copy()
        student._require_array(pt, (ACTION_DIM,), "selected PT action")  # noqa: SLF001
        start = time.perf_counter_ns()
        onnx = self.onnx.run(observation)
        onnx_ms = (time.perf_counter_ns() - start) / 1.0e6
        error = float(np.max(np.abs(pt.astype(np.float64) - onnx.astype(np.float64))))
        self.query_count += 1
        self.violation_count += int(error > TEACHER_PARITY_ATOL)
        self.maximum_absolute_error = max(self.maximum_absolute_error, error)
        self.total_absolute_error += error
        self.maximum_pt_duration_ms = max(self.maximum_pt_duration_ms, pt_ms)
        self.maximum_onnx_duration_ms = max(self.maximum_onnx_duration_ms, onnx_ms)
        return onnx, error

    def report(self) -> dict[str, Any]:
        state_after = tensor_state_sha256(self.actor.state_dict())
        return {
            "query_count": self.query_count,
            "pt_device": str(self.device),
            "onnx_provider": "CPUExecutionProvider",
            "maximum_absolute_error": self.maximum_absolute_error,
            "mean_transition_maximum_absolute_error": (
                None if self.query_count == 0 else self.total_absolute_error / self.query_count
            ),
            "violation_count": self.violation_count,
            "threshold": TEACHER_PARITY_ATOL,
            "maximum_pt_inference_duration_ms_evidence_only": self.maximum_pt_duration_ms,
            "maximum_onnx_inference_duration_ms_evidence_only": self.maximum_onnx_duration_ms,
            "actor_state_sha256_before": self.state_sha256_before,
            "actor_state_sha256_after": state_after,
            "actor_state_unchanged": state_after == self.state_sha256_before,
            "passed": bool(
                self.query_count == TOTAL_TRANSITIONS
                and self.violation_count == 0
                and self.maximum_absolute_error <= TEACHER_PARITY_ATOL
                and state_after == self.state_sha256_before
            ),
        }


class _ArrayDigestAccumulator:
    """Trajectory digest with array metadata/hashes only; never retains values."""

    def __init__(self, schema: str) -> None:
        self.schema = schema
        self.digest = hashlib.sha256()
        self.count = 0
        self.first: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None

    @staticmethod
    def _record(name: str, value: np.ndarray) -> dict[str, Any]:
        array = np.ascontiguousarray(value)
        return {
            "name": name,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }

    def add(self, *, transition: int, q9: int, controller: str, arrays: Mapping[str, np.ndarray]) -> None:
        records = [self._record(name, value) for name, value in arrays.items()]
        excerpt = {"transition": transition, "q9": q9, "controller": controller, "arrays": records}
        self.digest.update(canonical_json_bytes(excerpt))
        for value in arrays.values():
            self.digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        if self.first is None:
            self.first = excerpt
        self.last = excerpt
        self.count += 1

    def report(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row_count": self.count,
            "trajectory_sha256": self.digest.hexdigest(),
            "first": self.first,
            "last": self.last,
            "contains_array_values": False,
            "teacher_action_arrays_published": False,
            "training_arrays_published": False,
        }


class _DisagreementAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total_mae = 0.0
        self.maximum = 0.0
        self.total_rmse = 0.0

    def add(self, student_raw: np.ndarray, teacher_plain_raw: np.ndarray) -> None:
        error = np.abs(student_raw.astype(np.float64) - teacher_plain_raw.astype(np.float64))
        self.count += 1
        self.total_mae += float(np.mean(error))
        self.maximum = max(self.maximum, float(np.max(error)))
        self.total_rmse += float(np.sqrt(np.mean(np.square(error))))

    def report(self) -> dict[str, Any]:
        return {
            "transition_count": self.count,
            "mean_per_transition_mae": None if not self.count else self.total_mae / self.count,
            "mean_per_transition_rmse": None if not self.count else self.total_rmse / self.count,
            "maximum_absolute_difference": self.maximum,
            "teacher_action_arrays_published": False,
        }


class _CompositeAccumulator:
    LINKS = ("raw_native", "candidate_target_hardware", "safe_native", "final_target_hardware")

    def __init__(self, expected_count: int) -> None:
        self.expected_count = expected_count
        self.count = 0
        self.mismatch_count = 0
        self.maxima = {name: 0.0 for name in self.LINKS}

    def add(self, semantics: Mapping[str, Any], expected: Checkpoint21204TeacherComposite) -> None:
        chain = _mapping(semantics.get("chain"), "actual behavior action chain")
        actual = {
            "raw_native": np.asarray(chain["raw_native"], dtype=np.float32),
            "candidate_target_hardware": np.asarray(chain["candidate_target_hardware"], dtype=np.float32),
            "safe_native": np.asarray(chain["safe_native"], dtype=np.float32),
            "final_target_hardware": np.asarray(chain["final_target_hardware"], dtype=np.float32),
        }
        wanted = {
            "raw_native": expected.teacher_action_native,
            "candidate_target_hardware": expected.teacher_candidate_target_hardware,
            "safe_native": expected.teacher_applied_safe_action_native,
            "final_target_hardware": expected.teacher_target_hardware,
        }
        errors = {
            name: float(np.max(np.abs(actual[name].astype(np.float64) - wanted[name].astype(np.float64))))
            for name in self.LINKS
        }
        self.count += 1
        mismatch = any(value > ACTION_LINK_ATOL for value in errors.values())
        self.mismatch_count += int(mismatch)
        for name, value in errors.items():
            self.maxima[name] = max(self.maxima[name], value)

    def report(self) -> dict[str, Any]:
        return {
            "required_check_count": self.expected_count,
            "check_count": self.count,
            "mismatch_count": self.mismatch_count,
            "maximum_absolute_error_by_link": dict(self.maxima),
            "threshold": ACTION_LINK_ATOL,
            "teacher_action_arrays_published": False,
            "passed": bool(
                self.count == self.expected_count
                and self.mismatch_count == 0
                and all(value <= ACTION_LINK_ATOL for value in self.maxima.values())
            ),
        }


def _compact_partial_failure(
    *,
    stage: str,
    transition: int | None,
    q9: int | None,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema": "g1_true23_sonic_student_teacher_recovery_partial_failure_v1",
        "stage": stage,
        "transition": transition,
        "q9_before": q9,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "qualification_must_fail": True,
        "arrays_published": False,
        "teacher_labels_admitted": False,
    }


class _RecoveryTerminalRecorder:
    """Capture terminal evidence while guaranteeing the environment reset runs."""

    def __init__(self, raw_env: Any, velocity_limits: np.ndarray) -> None:
        self.raw_env = raw_env
        self.velocity_limits = velocity_limits
        self._original_reset = raw_env._reset_idx
        self._armed: dict[str, Any] | None = None
        self.captured: dict[str, Any] | None = None

        def observed_reset(_env: Any, env_ids: torch.Tensor | None = None) -> None:
            try:
                if self._armed is not None and int(_env.common_step_counter) > 0:
                    if self.captured is not None:
                        raise RuntimeError("recovery terminal recorder captured more than once")
                    if env_ids is None or int(env_ids.numel()) != 1 or int(env_ids.detach().cpu().item()) != 0:
                        raise RuntimeError("recovery terminal recorder expected environment zero")
                    capture_errors: list[str] = []
                    try:
                        action_semantics = student._action_semantics(  # noqa: SLF001
                            _env,
                            self._armed["behavior_raw"],
                        )
                    except Exception as error:  # preserve reset; fail whole run later
                        action_semantics = None
                        capture_errors.append(f"action_semantics: {type(error).__name__}: {error}")
                    try:
                        step_evidence = evaluation._step_evidence(  # noqa: SLF001
                            _env,
                            self.velocity_limits,
                        )
                    except Exception as error:  # preserve reset; fail whole run later
                        step_evidence = None
                        capture_errors.append(f"step_evidence: {type(error).__name__}: {error}")
                    try:
                        physical = student._physical_state(_env)  # noqa: SLF001
                    except Exception as error:  # preserve reset; fail whole run later
                        physical = None
                        capture_errors.append(f"physical_state: {type(error).__name__}: {error}")
                    try:
                        ee_errors = reset_diagnostic.capture_exact_ee_position_errors(_env)
                    except Exception as error:  # preserve reset; fail whole run later
                        ee_errors = None
                        capture_errors.append(f"ee_errors: {type(error).__name__}: {error}")
                    try:
                        episode_length = int(_env.episode_length_buf[0].detach().cpu().item())
                        termination_names = evaluation._termination_names(_env)  # noqa: SLF001
                        is_timeout = bool(_env.termination_manager.time_outs[0].detach().cpu().item())
                        is_terminated = bool(_env.termination_manager.terminated[0].detach().cpu().item())
                    except Exception as error:  # preserve reset; fail whole run later
                        episode_length = None
                        termination_names = None
                        is_timeout = None
                        is_terminated = None
                        capture_errors.append(f"termination_metadata: {type(error).__name__}: {error}")
                    self.captured = {
                        **self._armed,
                        "episode_length_pre_reset": episode_length,
                        "termination_names": termination_names,
                        "is_timeout": is_timeout,
                        "is_terminated": is_terminated,
                        "action_semantics": action_semantics,
                        "step_evidence": step_evidence,
                        "post_physical": physical,
                        "ee_body_position_errors": ee_errors,
                        "capture_errors": capture_errors,
                    }
            finally:
                self._original_reset(env_ids)

        raw_env._reset_idx = MethodType(observed_reset, raw_env)

    def arm(self, **record: Any) -> None:
        if self._armed is not None:
            raise RuntimeError("recovery terminal recorder already armed")
        behavior = record.get("behavior_raw")
        student._require_array(behavior, (ACTION_DIM,), "terminal behavior raw")  # noqa: SLF001
        if record.get("controller") not in {"student", "teacher"}:
            raise ValueError("terminal controller identity drift")
        self._armed = copy.deepcopy(record)
        self.captured = None

    def finish(self, *, done: bool) -> dict[str, Any] | None:
        if self._armed is None:
            raise RuntimeError("recovery terminal recorder was not armed")
        captured = self.captured
        self._armed = None
        self.captured = None
        if done != (captured is not None):
            raise RuntimeError("recovery terminal capture disagrees with done")
        return captured

    def restore(self) -> None:
        self.raw_env._reset_idx = self._original_reset


def _compact_terminal_ee_position_evidence(value: Any) -> dict[str, Any]:
    """Publish exact terminal EE error scalars without position vectors."""

    if type(value) is not list or not value:
        raise ValueError("terminal EE position evidence must be a nonempty list")

    def finite_scalar(item: Any, context: str) -> float:
        if isinstance(item, bool) or not isinstance(item, (float, int, np.floating, np.integer)):
            raise TypeError(f"{context} must be numeric scalar")
        result = float(item)
        if not math.isfinite(result):
            raise ValueError(f"{context} must be finite")
        return result

    compact_bodies: list[dict[str, Any]] = []
    body_names: set[str] = set()
    body_indices: set[int] = set()
    common_threshold: float | None = None
    for body_offset, raw_record in enumerate(value):
        record = _mapping(raw_record, f"terminal EE body {body_offset}")
        name = record.get("name")
        body_index = record.get("command_body_index")
        if type(name) is not str or not name or name in body_names:
            raise ValueError("terminal EE body names must be unique nonempty strings")
        if type(body_index) is not int or body_index < 0 or body_index in body_indices:
            raise ValueError("terminal EE command body indices must be unique nonnegative integers")
        body_names.add(name)
        body_indices.add(body_index)

        vector = record.get("error_measured_minus_reference_m")
        if type(vector) is not list or len(vector) != 3:
            raise ValueError("terminal EE error vector must contain exactly three scalars")
        error_x, error_y, error_z = (
            finite_scalar(item, f"terminal EE {name} error axis {axis}")
            for axis, item in zip("xyz", vector, strict=True)
        )
        error_norm = finite_scalar(record.get("error_norm_m"), f"terminal EE {name} norm")
        absolute_z = finite_scalar(
            record.get("absolute_z_error_m"),
            f"terminal EE {name} absolute z error",
        )
        threshold = finite_scalar(
            record.get("termination_threshold_m"),
            f"terminal EE {name} threshold",
        )
        breached = record.get("z_termination_breached")
        if threshold <= 0.0:
            raise ValueError("terminal EE threshold must be positive")
        if breached is not (absolute_z > threshold):
            raise ValueError("terminal EE breach flag disagrees with z error and threshold")
        expected_norm = math.sqrt(error_x**2 + error_y**2 + error_z**2)
        if not math.isclose(error_norm, expected_norm, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("terminal EE norm disagrees with xyz error scalars")
        if not math.isclose(absolute_z, abs(error_z), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("terminal EE absolute z error disagrees with signed z error")
        if common_threshold is None:
            common_threshold = threshold
        elif threshold != common_threshold:
            raise ValueError("terminal EE configured bodies must share one z threshold")

        compact_bodies.append(
            {
                "name": name,
                "command_body_index": body_index,
                "error_x_m": error_x,
                "error_y_m": error_y,
                "error_z_m": error_z,
                "error_norm_m": error_norm,
                "absolute_z_error_m": absolute_z,
                "termination_threshold_m": threshold,
                "z_threshold_margin_m": absolute_z - threshold,
                "z_termination_breached": breached,
            }
        )

    dominant = max(compact_bodies, key=lambda item: float(item["absolute_z_error_m"]))
    return {
        "schema": "g1_true23_terminal_ee_position_scalar_evidence_v1",
        "configured_body_count": len(compact_bodies),
        "termination_axis": "z",
        "termination_threshold_m": common_threshold,
        "breached_body_count": sum(bool(item["z_termination_breached"]) for item in compact_bodies),
        "dominant_absolute_z_error_body": dominant["name"],
        "dominant_signed_z_error_m": dominant["error_z_m"],
        "dominant_absolute_z_error_m": dominant["absolute_z_error_m"],
        "dominant_z_threshold_margin_m": dominant["z_threshold_margin_m"],
        "dominant_z_termination_breached": dominant["z_termination_breached"],
        "bodies": compact_bodies,
        "position_or_error_vectors_published": False,
    }


def _terminal_compact(
    terminal: Mapping[str, Any],
    q9_after: int,
    autoreset_history: Mapping[str, Any],
    selected_autoreset: Mapping[str, Any],
) -> dict[str, Any]:
    action = _mapping(terminal.get("action_semantics"), "terminal action semantics")
    errors = _mapping(action.get("maximum_absolute_error_by_link"), "terminal action errors")
    physical = terminal.get("post_physical")
    ee = terminal.get("ee_body_position_errors")
    return {
        "transition": terminal.get("transition"),
        "q9_before": terminal.get("q9_before"),
        "q9_after_autoreset": q9_after,
        "episode_length_pre_reset": terminal.get("episode_length_pre_reset"),
        "termination_names": terminal.get("termination_names"),
        "is_timeout": terminal.get("is_timeout"),
        "is_terminated": terminal.get("is_terminated"),
        "capture_errors": terminal.get("capture_errors"),
        "controller": terminal.get("controller"),
        "action_semantics": {
            "passed": action.get("passed"),
            "maximum_absolute_error_by_link": dict(errors),
            "raw_clip_coordinate_count": action.get("raw_clip_coordinate_count"),
            "plain_sonic_raw_native_abs_max": action.get("plain_sonic_raw_native_abs_max"),
            "chain_values_published": False,
        },
        "post_physical_sha256": sha256_bytes(canonical_json_bytes(physical)),
        "ee_body_position_errors_sha256": sha256_bytes(canonical_json_bytes(ee)),
        "ee_body_position_scalar_evidence": _compact_terminal_ee_position_evidence(ee),
        "autoreset_history": dict(autoreset_history),
        "selected124_autoreset": dict(selected_autoreset),
        "terminal_autoreset_observation_is_next_state": False,
    }


def _reset_seam(
    raw_env: Any,
    observations: Mapping[str, Any],
    selected_state: _Selected124State,
) -> dict[str, Any]:
    command = raw_env.command_manager.get_term("motion")
    action = raw_env.action_manager.get_term("joint_pos")
    if type(action) is not DiagnosticSafeTargetNativeIl23JointPositionAction:
        raise TypeError("recovery reset lacks exact diagnostic V11 action")
    action_zero = not bool(torch.count_nonzero(action.raw_action)) and not bool(
        torch.count_nonzero(action.safe_native_action)
    )
    if not action_zero:
        raise RuntimeError("recovery reset action state is not zero")
    history = student._policy_history_proof(raw_env, observations, 0)  # noqa: SLF001
    proof = selected_state.reset_proof()
    if evaluation._q9(command) != ANCHOR_Q9 or proof["previous_selected_raw_is_exact_zero"] is not True:  # noqa: SLF001
        raise RuntimeError("recovery reset selected state drift")
    return {
        "prime_q9": ANCHOR_Q9,
        "first_behavior_action_q9": ANCHOR_Q9,
        "fixed_warmup_or_action_substitution_steps": 0,
        "action_substitution": False,
        "student_history": history,
        "student_action_state_zero": action_zero,
        "selected124": proof,
    }


def _support_fallback(gate: Mapping[str, Any]) -> dict[str, Any]:
    result = {name: 1 for name in gate["required_zero_counts"]}
    result.update(
        {
            "minimum_base_height_m": -1.0e30,
            "maximum_base_tilt_rad": 1.0e30,
            "maximum_joint_velocity_ratio": 1.0e30,
            "maximum_tracking_rmse_rad": 1.0e30,
            "maximum_plain_sonic_raw_native_abs": 1.0e30,
        }
    )
    return result


def assess_recovery(
    *,
    window: RecoveryWindow,
    step_calls_started: int,
    attempted: int,
    student_inference_count: int,
    teacher_parity: Mapping[str, Any],
    controller_counts: Mapping[str, Any],
    first_done: Mapping[str, Any] | None,
    history_check_count: int,
    history_shift_count: int,
    action_semantics: Mapping[str, Any],
    teacher_composite: Mapping[str, Any],
    selected_state: Mapping[str, Any],
    support_summary: Mapping[str, Any],
    frozen_models: Mapping[str, Any],
    bindings_unchanged: bool,
    partial_failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if resolve_recovery_window(window.mode) != window:
        raise ValueError("recovery assessment window is not pinned")
    expected_done = bool(
        first_done is not None
        and first_done.get("transition") == window.transitions - 1
        and first_done.get("q9_before") == window.last_q9
        and first_done.get("q9_after_autoreset") == window.anchor_q9
        and first_done.get("episode_length_pre_reset") == window.transitions
        and first_done.get("termination_names") == ["time_out"]
        and first_done.get("is_timeout") is True
        and first_done.get("is_terminated") is False
        and first_done.get("capture_errors") == []
        and isinstance(first_done.get("autoreset_history"), Mapping)
        and first_done["autoreset_history"].get("local_transition") == 0
        and first_done["autoreset_history"].get("actual_history_depth") == 1
        and first_done["autoreset_history"].get("reset_padding_count") == 9
        and first_done["autoreset_history"].get("term_major_policy930_exact") is True
        and first_done["autoreset_history"].get("previous_action_slice_zero") is True
        and isinstance(first_done.get("selected124_autoreset"), Mapping)
        and first_done["selected124_autoreset"].get("previous_selected_raw_is_exact_zero") is True
    )
    gates = {
        "complete_window": step_calls_started == window.transitions and attempted == window.transitions,
        "student_inferred_every_transition": student_inference_count == window.transitions,
        "teacher_inferred_every_transition": teacher_parity.get("query_count") == window.transitions,
        "teacher_pt_onnx_parity": teacher_parity.get("passed") is True,
        "controller_partition_exact": controller_counts
        == {
            "student": window.student_transitions,
            "teacher": window.teacher_transitions,
        },
        "only_final_timeout": expected_done,
        "history_exact": (
            history_check_count == window.transitions and history_shift_count == window.transitions - 1
        ),
        "actual_v11_action_chain_exact": action_semantics.get("passed") is True,
        "teacher_composite_chain_exact": teacher_composite.get("passed") is True,
        "selected_external_state_exact": (
            selected_state.get("build_count") == window.transitions
            and selected_state.get("nonterminal_update_count") == window.transitions - 1
            and selected_state.get("mismatch_count") == 0
            and selected_state.get("handoff_observed") is True
            and selected_state.get("autoreset_synchronization_count") == 1
            and selected_state.get("autoreset_previous_selected_raw_is_exact_zero") is True
        ),
        "nominal_safety_gate": all(
            int(support_summary.get(name, 1)) == 0
            for name in (
                "termination_count",
                "q9_discontinuity_count",
                "nonfinite_count",
                "raw_clip_required_count",
                "action_semantics_mismatch_count",
                "target_soft_limit_violation_count",
                "actuator_target_soft_limit_violation_count",
                "measured_soft_limit_violation_count",
                "joint_velocity_limit_violation_count",
                "teacher_parity_violation_count",
                "teacher_composite_mismatch_count",
                "selected_state_mismatch_count",
            )
        )
        and float(support_summary.get("minimum_base_height_m", -math.inf)) >= 0.45
        and float(support_summary.get("maximum_base_tilt_rad", math.inf)) <= 1.0
        and float(support_summary.get("maximum_joint_velocity_ratio", math.inf)) <= 1.0
        and float(support_summary.get("maximum_tracking_rmse_rad", math.inf)) <= 0.75
        and float(support_summary.get("maximum_plain_sonic_raw_native_abs", math.inf)) < STRICT_RAW_ABS_MAX,
        "frozen_inputs_unchanged": frozen_models.get("all_preflight_bound_inputs_unchanged") is True,
        "sources_assets_contracts_unchanged": bindings_unchanged,
        "no_partial_failure": partial_failure is None,
    }
    recovered = all(gates.values())
    return {
        "recovered_to_original_q9_518_boundary": recovered,
        "verdict": "mixed_controller_recovery_passed" if recovered else "mixed_controller_recovery_failed",
        "gates": gates,
        "claims": _claims(window.mode, recovered),
        "scope": recovery_scope(window.mode),
    }


def run_recovery_diagnostic(request: RecoveryRequest) -> dict[str, Any]:
    """Run one mixed-controller recoverability window; never emit labels."""

    preflight = _preflight_internal(request)
    if preflight.get("ready") is not True:
        return {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "kind": RECOVERY_KIND,
            "recovered_to_original_q9_518_boundary": False,
            "verdict": "mixed_controller_recovery_not_ready",
            "preflight": _compact_preflight(preflight, request),
            "claims": _claims(request.mode, False),
            "scope": recovery_scope(request.mode),
            "safety": _safety_record(0, 0, 0, 0),
        }

    root = request.root
    window = request.window
    base = preflight["student"]
    contract = preflight["contract"]
    gate = contract["gates"]
    motion = Path(base["fixed_inputs"]["motion_path"])
    encoder_path = Path(base["fixed_inputs"]["encoder_path"])
    candidate_path = Path(base["candidate"]["decoder_path"])
    candidate_sha256 = str(base["candidate"]["decoder_sha256"])
    frozen_specs = preflight["frozen_specs"]
    frozen_before = preflight["frozen_before"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(student.FIXED_GPU)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    runtime_sources = evaluation._bind_evaluation_runtime_sources(root)  # noqa: SLF001

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("recovery diagnostic requires fixed visible CUDA device 0")
    random.seed(student.FIXED_SEED)
    np.random.seed(student.FIXED_SEED % (2**32))
    torch.manual_seed(student.FIXED_SEED)
    torch.cuda.manual_seed_all(student.FIXED_SEED)
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)

    composite_contract = load_composite_mjlab_contract(root)
    velocity_limits = np.asarray(
        composite_contract["nominal_gate"]["velocity_limit_hardware_radps"], dtype=np.float32
    )
    cfg = make_sonic_true23_student_qualification_env_cfg(
        motion_file=str(motion), num_envs=1, anchor_q9=window.anchor_q9, transitions=window.transitions
    )
    cfg.seed = student.FIXED_SEED
    task_audit = audit_sonic_true23_student_qualification_env_cfg(
        cfg, expected_anchor_q9=window.anchor_q9, expected_transitions=window.transitions
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=student.DEVICE)
    recorder: _RecoveryTerminalRecorder | None = None
    teacher: _ExactTeacherPair | None = None
    step_calls_started = 0
    attempted = 0
    student_inference_count = 0
    controller_counts = {"student": 0, "teacher": 0}
    current_transition: int | None = None
    current_q9: int | None = None
    runtime_stage = "post_environment_construction"
    runtime_result: dict[str, Any] | None = None
    runtime_error: BaseException | None = None
    cleanup_errors: list[str] = []
    try:
        runtime_stage = "environment_static_validation"
        groups = env.observation_manager.cfg
        if (
            tuple(groups) != ("tokenizer", "policy", "critic")
            or groups["tokenizer"].enable_corruption is not True
            or groups["policy"].enable_corruption is not True
        ):
            raise RuntimeError("recovery observation corruption/group drift")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        if wrapped.clip_actions is not None or wrapped.max_episode_length != window.transitions:
            raise RuntimeError("recovery wrapper clip/horizon drift")
        prime = prime_sonic_true23_training_environment(wrapped)
        observations = wrapped.get_observations()
        command = env.command_manager.get_term("motion")
        if (
            evaluation._q9(command) != window.anchor_q9  # noqa: SLF001
            or int(env.common_step_counter) != 0
            or int(env._sim_step_counter) != 0
            or int(env.episode_length_buf[0].detach().cpu().item()) != 0
        ):
            raise RuntimeError("recovery prime changed q9/counters")
        selected_state = _Selected124State(env)
        reset_seam = _reset_seam(env, observations, selected_state)
        encoder = student._HashBoundEncoder(encoder_path)  # noqa: SLF001
        decoder = student._HashBoundDecoder(candidate_path, candidate_sha256)  # noqa: SLF001
        teacher = _ExactTeacherPair(preflight["teacher"], student.DEVICE)
        recorder = _RecoveryTerminalRecorder(env, velocity_limits)
        rollout = evaluation.RolloutEvidenceAccumulator()
        action_accumulator = student._ActionSemanticsAccumulator(window.transitions)  # noqa: SLF001
        composite_accumulator = _CompositeAccumulator(window.teacher_transitions)
        inference_digest = _ArrayDigestAccumulator("g1_true23_recovery_inference_digest_v1")
        action_digest = _ArrayDigestAccumulator("g1_true23_recovery_executed_action_digest_v1")
        prefix_disagreement = _DisagreementAccumulator()
        suffix_disagreement = _DisagreementAccumulator()
        previous_history: Mapping[str, torch.Tensor] | None = None
        history_count = 0
        history_shift_count = 0
        q9_discontinuity_count = 0
        nonfinite_count = 0
        selected_state_mismatch_count = 0
        selected_round_trip_max = 0.0
        handoff_observed = False
        first_done: dict[str, Any] | None = None
        partial_failure: dict[str, Any] | None = None
        autoreset_history: dict[str, Any] | None = None

        for transition in range(window.transitions):
            current_transition = transition
            runtime_stage = "transition_q9_read"
            q9 = evaluation._q9(command)  # noqa: SLF001
            current_q9 = q9
            runtime_stage = "rollout_transition"
            expected_q9 = window.anchor_q9 + transition
            if q9 != expected_q9:
                q9_discontinuity_count += 1
                partial_failure = _compact_partial_failure(
                    stage="pre_action_q9_continuity",
                    transition=transition,
                    q9=q9,
                    error=RuntimeError(f"expected q9 {expected_q9}, observed {q9}"),
                )
                break
            controller = window.controller(transition)
            episode_before = int(env.episode_length_buf[0].detach().cpu().item())
            common_before = int(env.common_step_counter)
            sim_before = int(env._sim_step_counter)
            command_counter_before = int(command.command_counter[0].detach().cpu().item())
            try:
                student._policy_history_proof(env, observations, transition)  # noqa: SLF001
                snapshot = student._policy_history_snapshot(env)  # noqa: SLF001
                if previous_history is not None:
                    student._assert_history_shift(previous_history, snapshot, transition)  # noqa: SLF001
                    history_shift_count += 1
                previous_history = snapshot
                history_count += 1
                encoder267, policy930 = student._observation_arrays(observations)  # noqa: SLF001
                token64 = encoder.run(encoder267)
                decoder994 = np.concatenate((token64, policy930)).astype(np.float32, copy=False)
                student._require_array(decoder994, (DECODER_DIM,), "recovery decoder994")  # noqa: SLF001
                student_raw = decoder.run(decoder994)
                student_inference_count += 1
                selected124, pre_step_torso = selected_state.build()
                teacher_raw, _ = teacher.infer(selected124)
                teacher_composite = compose_checkpoint21204_teacher_action(teacher_raw, repository_root=root)
                teacher_plain = teacher_composite.teacher_action_native
                if float(np.max(np.abs(teacher_plain))) >= STRICT_RAW_ABS_MAX:
                    raise RuntimeError("teacher composite requires forbidden raw clipping")
                (prefix_disagreement if controller == "student" else suffix_disagreement).add(
                    student_raw, teacher_plain
                )
                behavior_raw = student_raw if controller == "student" else teacher_plain
                if float(np.max(np.abs(behavior_raw))) >= STRICT_RAW_ABS_MAX:
                    raise RuntimeError("behavior action requires forbidden raw clipping")
                inference_digest.add(
                    transition=transition,
                    q9=q9,
                    controller=controller,
                    arrays={
                        "student_encoder267": encoder267,
                        "student_token64": token64,
                        "student_policy930": policy930,
                        "student_decoder994": decoder994,
                        "student_raw_native23": student_raw,
                        "selected_teacher_observation124": selected124.detach().cpu().numpy()[0].copy(),
                        "selected_teacher_onnx_raw_hardware23": teacher_raw,
                    },
                )
            except Exception as error:
                nonfinite_count += int("finite" in str(error).lower())
                partial_failure = _compact_partial_failure(
                    stage="dual_policy_pre_action_inference",
                    transition=transition,
                    q9=q9,
                    error=error,
                )
                break

            action_tensor = torch.from_numpy(behavior_raw.reshape(1, ACTION_DIM)).to(
                device=student.DEVICE, dtype=torch.float32
            )
            recorder.arm(
                transition=transition,
                q9_before=q9,
                controller=controller,
                behavior_raw=behavior_raw,
            )
            step_calls_started += 1
            try:
                observations, _, dones, extras = wrapped.step(action_tensor)
            except Exception as error:
                partial_failure = _compact_partial_failure(
                    stage="environment_step", transition=transition, q9=q9, error=error
                )
                break
            attempted += 1
            controller_counts[controller] += 1
            done = bool(int(dones[0].detach().cpu().item()))
            terminal = recorder.finish(done=done)
            if done:
                if terminal is None:
                    raise RuntimeError("recovery terminal lacks pre-autoreset evidence")
                semantics = terminal.get("action_semantics")
                evidence = terminal.get("step_evidence")
                if (
                    not isinstance(semantics, Mapping)
                    or type(evidence) is not evaluation.StepEvidence
                    or terminal.get("post_physical") is None
                    or terminal.get("capture_errors")
                ):
                    partial_failure = _compact_partial_failure(
                        stage="terminal_pre_autoreset_capture",
                        transition=transition,
                        q9=q9,
                        error=RuntimeError("terminal capture incomplete"),
                    )
                    break
            else:
                if terminal is not None or evaluation._extra_termination_names(extras):  # noqa: SLF001
                    raise RuntimeError("recovery nonterminal contains termination evidence")
                try:
                    semantics = student._action_semantics(env, behavior_raw)  # noqa: SLF001
                    evidence = evaluation._step_evidence(env, velocity_limits)  # noqa: SLF001
                except Exception as error:
                    partial_failure = _compact_partial_failure(
                        stage="post_action_evidence", transition=transition, q9=q9, error=error
                    )
                    break

            action_accumulator.add(semantics)
            rollout.add(evidence)
            chain = _mapping(semantics.get("chain"), "recovery action chain")
            final_target = np.asarray(chain["final_target_hardware"], dtype=np.float32)
            safe_native = np.asarray(chain["safe_native"], dtype=np.float32)
            action_digest.add(
                transition=transition,
                q9=q9,
                controller=controller,
                arrays={
                    "executed_plain_raw_native23": behavior_raw,
                    "executed_safe_native23": safe_native,
                    "executed_final_target_hardware23": final_target,
                },
            )
            if controller == "teacher":
                composite_accumulator.add(semantics, teacher_composite)
            if transition == window.student_transitions:
                handoff_observed = bool(
                    controller == "teacher"
                    and selected_state.update_count == window.student_transitions
                    and q9 == window.teacher_first_q9
                )

            if done:
                autoreset_history = student._policy_history_proof(env, observations, 0)  # noqa: SLF001
                selected_autoreset = selected_state.synchronize_autoreset()
                names = evaluation._extra_termination_names(extras)  # noqa: SLF001
                if names != terminal["termination_names"]:
                    raise RuntimeError("recovery terminal extras/pre-reset terms disagree")
                first_done = _terminal_compact(
                    terminal,
                    evaluation._q9(command),  # noqa: SLF001
                    autoreset_history,
                    selected_autoreset,
                )
                break

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
                or student._command_resampled(command)  # noqa: SLF001
            ):
                q9_discontinuity_count += 1
                partial_failure = _compact_partial_failure(
                    stage="post_action_continuity",
                    transition=transition,
                    q9=q9,
                    error=RuntimeError("recovery simulator/command continuity drift"),
                )
                break
            state_proof = selected_state.update_nonterminal(pre_step_torso, final_target)
            selected_round_trip_max = max(
                selected_round_trip_max,
                float(state_proof["maximum_target_round_trip_absolute_error"]),
            )
            selected_state_mismatch_count += int(state_proof["passed"] is not True)

        runtime_stage = "post_rollout_aggregation"
        rollout_summary = rollout.report()
        action_report = action_accumulator.report()
        teacher_parity = teacher.report()
        teacher_composite_report = composite_accumulator.report()
        expected_final_timeout = bool(
            first_done is not None
            and first_done.get("transition") == window.transitions - 1
            and first_done.get("q9_before") == window.last_q9
            and first_done.get("termination_names") == ["time_out"]
            and first_done.get("is_timeout") is True
            and first_done.get("is_terminated") is False
        )
        support = (
            student._support_summary(  # noqa: SLF001
                rollout_summary,
                expected_final_timeout=expected_final_timeout,
                q9_discontinuity_count=q9_discontinuity_count,
                nonfinite_count=nonfinite_count,
                action_semantics=action_report,
            )
            if rollout_summary["transition_count"]
            else _support_fallback(gate)
        )
        support.update(
            {
                "teacher_parity_violation_count": int(teacher_parity["violation_count"]),
                "teacher_composite_mismatch_count": int(teacher_composite_report["mismatch_count"]),
                "selected_state_mismatch_count": selected_state_mismatch_count,
            }
        )
        frozen_after = student._snapshot_preflight_bound_files(frozen_specs)  # noqa: SLF001
        frozen_models = student._frozen_input_evidence(frozen_before, frozen_after)  # noqa: SLF001
        runtime_stage = "post_rollout_binding"
        sources_after = executed_recovery_source_binding(root)
        external_sources_after = external_runtime_source_binding(root)
        assets_after = student.physical_model_asset_binding(root)
        contract_after = load_recovery_contract(root)
        bindings_unchanged = bool(
            sources_after == preflight["sources"]
            and external_sources_after == preflight["external_sources"]
            and assets_after == preflight["assets"]
            and contract_after == preflight["contract"]
        )
        selected_report = {
            "build_count": selected_state.build_count,
            "nonterminal_update_count": selected_state.update_count,
            "mismatch_count": selected_state_mismatch_count,
            "maximum_target_round_trip_absolute_error": selected_round_trip_max,
            "reset_previous_selected_raw_is_exact_zero": selected_state.reset_previous_zero,
            "handoff_observed": handoff_observed,
            "autoreset_synchronization_count": selected_state.autoreset_synchronization_count,
            "autoreset_previous_selected_raw_is_exact_zero": bool(
                first_done is not None
                and _mapping(
                    first_done.get("selected124_autoreset"),
                    "selected124 terminal autoreset",
                ).get("previous_selected_raw_is_exact_zero")
                is True
            ),
            "student_teacher_handoff_resets_state": False,
            "previous_action_formula": "(actual_final_unbiased_hardware_target-selected_home)/selected_scale",
        }
        qualification = assess_recovery(
            window=window,
            step_calls_started=step_calls_started,
            attempted=attempted,
            student_inference_count=student_inference_count,
            teacher_parity=teacher_parity,
            controller_counts=controller_counts,
            first_done=first_done,
            history_check_count=history_count,
            history_shift_count=history_shift_count,
            action_semantics=action_report,
            teacher_composite=teacher_composite_report,
            selected_state=selected_report,
            support_summary=support,
            frozen_models=frozen_models,
            bindings_unchanged=bindings_unchanged,
            partial_failure=partial_failure,
        )
        report = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "kind": RECOVERY_KIND,
            "recovered_to_original_q9_518_boundary": qualification["recovered_to_original_q9_518_boundary"],
            "verdict": qualification["verdict"],
            "preflight": _compact_preflight(preflight, request),
            "runtime_sources": runtime_sources,
            "task_audit": task_audit,
            "prime": prime,
            "reset_seam": reset_seam,
            "simulator_step_calls_started": step_calls_started,
            "attempted_transitions": attempted,
            "controller_counts": controller_counts,
            "student_inference_count": student_inference_count,
            "teacher_parity": teacher_parity,
            "first_done": first_done,
            "history": {
                "check_count": history_count,
                "single_append_shift_check_count": history_shift_count,
                "autoreset": autoreset_history,
            },
            "rollout": rollout_summary,
            "support_summary": support,
            "action_semantics": action_report,
            "teacher_composite_action_chain": teacher_composite_report,
            "selected124_external_state": selected_report,
            "student_teacher_disagreement": {
                "student_controlled_prefix": prefix_disagreement.report(),
                "teacher_controlled_suffix": suffix_disagreement.report(),
            },
            "inference_digest": inference_digest.report(),
            "executed_action_digest": action_digest.report(),
            "frozen_inputs": {
                "all_preflight_bound_inputs_unchanged": frozen_models.get("all_preflight_bound_inputs_unchanged"),
                "file_count": frozen_models.get("file_count"),
            },
            "bindings_unchanged": bindings_unchanged,
            "partial_failure": partial_failure,
            "qualification": qualification,
            "claims": qualification["claims"],
            "scope": qualification["scope"],
            "safety": _safety_record(
                step_calls_started,
                attempted,
                student_inference_count,
                teacher.query_count,
            ),
        }
        _assert_publication_boundary(report)
        runtime_result = evaluation._json_safe(report)  # noqa: SLF001
    except Exception as error:
        runtime_error = error
    finally:
        if recorder is not None:
            try:
                recorder.restore()
            except Exception as error:
                cleanup_errors.append(f"recorder_restore:{type(error).__name__}:{error}")
        try:
            env.close()
        except Exception as error:
            cleanup_errors.append(f"environment_close:{type(error).__name__}:{error}")
    if cleanup_errors:
        cleanup_error = RuntimeError(";".join(cleanup_errors))
        if runtime_error is None:
            runtime_error = cleanup_error
            runtime_stage = "runtime_cleanup"
    if runtime_error is not None:
        return _runtime_failure_report(
            error=runtime_error,
            request=request,
            preflight=preflight,
            stage=runtime_stage,
            transition=current_transition,
            q9=current_q9,
            step_calls_started=step_calls_started,
            attempted=attempted,
            student_inferences=student_inference_count,
            teacher_queries=0 if teacher is None else teacher.query_count,
            controller_counts=controller_counts,
            cleanup_errors=cleanup_errors,
        )
    if runtime_result is None:
        raise RuntimeError("recovery runtime produced neither report nor error")
    return runtime_result


def _runtime_failure_report(
    *,
    error: BaseException,
    request: RecoveryRequest,
    preflight: Mapping[str, Any],
    stage: str,
    transition: int | None,
    q9: int | None,
    step_calls_started: int,
    attempted: int,
    student_inferences: int,
    teacher_queries: int,
    controller_counts: Mapping[str, int],
    cleanup_errors: list[str],
) -> dict[str, Any]:
    partial = _compact_partial_failure(
        stage=stage,
        transition=transition,
        q9=q9,
        error=error,
    )
    report = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "kind": RECOVERY_KIND,
        "recovered_to_original_q9_518_boundary": False,
        "verdict": "mixed_controller_recovery_runtime_error_after_construction",
        "preflight": _compact_preflight(preflight, request),
        "simulator_step_calls_started": step_calls_started,
        "attempted_transitions": attempted,
        "student_inference_count": student_inferences,
        "teacher_query_count": teacher_queries,
        "controller_counts": dict(controller_counts),
        "partial_failure": partial,
        "cleanup_errors": list(cleanup_errors),
        "claims": _claims(request.mode, False),
        "scope": recovery_scope(request.mode),
        "safety": _safety_record(
            step_calls_started,
            attempted,
            student_inferences,
            teacher_queries,
        ),
    }
    _assert_publication_boundary(report)
    return report


def _safety_record(
    simulator_step_calls_started: int,
    simulator_transitions_completed: int,
    student_inferences: int,
    teacher_queries: int,
) -> dict[str, Any]:
    return {
        "simulator_only": True,
        "simulator_steps": simulator_step_calls_started,
        "simulator_step_calls_started": simulator_step_calls_started,
        "simulator_transitions_completed": simulator_transitions_completed,
        "student_inferences": student_inferences,
        "teacher_queries": teacher_queries,
        "teacher_labels_admitted": False,
        "published_teacher_label_count": 0,
        "training_performed": False,
        "training_updates": 0,
        "support_qualification_performed": False,
        "dagger_data": False,
        "hardware_or_network_commands_performed": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
    }


def _assert_publication_boundary(report: Mapping[str, Any]) -> None:
    """Reject known label/training array keys anywhere in the public report."""

    forbidden = {
        "teacher_raw_action_hardware",
        "selected_teacher_raw_action_hardware",
        "teacher_action_native",
        "plain_sonic_raw_native_diagnostic",
        "teacher_candidate_target_hardware",
        "teacher_target_hardware",
        "teacher_composite_target_hardware",
        "teacher_applied_safe_action_native",
        "teacher_label",
        "teacher_labels",
        "training_arrays",
        "pre_action_arrays",
        "ee_body_position_errors",
        "reference_position_w_m",
        "measured_position_w_m",
        "error_measured_minus_reference_m",
        "rows",
    }

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in forbidden:
                    raise RuntimeError(f"recovery publication contains forbidden field: {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            raise RuntimeError(f"recovery publication contains an array object: {path}")

    visit(report, "report")


def failure_report(error: BaseException, request: RecoveryRequest) -> dict[str, Any]:
    report = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "kind": RECOVERY_KIND,
        "recovered_to_original_q9_518_boundary": False,
        "verdict": "mixed_controller_recovery_runtime_error",
        "error": {"type": type(error).__name__, "message": str(error)},
        "request": {
            "repository_root": str(request.repository_root),
            "candidate_manifest": str(request.candidate_manifest),
            "expected_candidate_manifest_sha256": request.expected_candidate_manifest_sha256,
            "output": str(request.output),
            "mode": request.mode,
        },
        "partial_failure": None,
        "claims": _claims(request.mode, False),
        "scope": recovery_scope(request.mode),
        "safety": _safety_record(0, 0, 0, 0),
    }
    _assert_publication_boundary(report)
    return report


def _temporary_json(parent: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(dir=parent, prefix=".sonic-student-teacher-recovery-", suffix=".tmp")
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


def write_recovery_report_new(request: RecoveryRequest, report: Mapping[str, Any]) -> Path:
    """Atomically hard-link one report and refuse every overwrite."""

    if type(request) is not RecoveryRequest:
        raise TypeError("request must be exact RecoveryRequest")
    _assert_publication_boundary(report)
    output = request.output_path
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("recovery output parent must be a regular directory")
    body = evaluation._json_safe(report)  # noqa: SLF001
    payload = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = _temporary_json(output.parent, payload)
    try:
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite recovery report: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "ACTION_LINK_ATOL",
    "ANCHOR_Q9",
    "BASE_RECOVERY_CONTRACT_SHA256",
    "CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH",
    "CURRENT_CANDIDATE_MANIFEST_SHA256",
    "LAST_Q9",
    "MODES",
    "PREVIOUS_RECOVERY_CONTRACT_SHA256",
    "RECOVERY_CONTRACT_SHA256",
    "RECOVERY_KIND",
    "RecoveryRequest",
    "RecoveryWindow",
    "assess_recovery",
    "external_runtime_source_binding",
    "executed_recovery_source_binding",
    "failure_report",
    "load_recovery_contract",
    "preflight_recovery",
    "recovery_scope",
    "resolve_recovery_window",
    "run_recovery_diagnostic",
    "selected_previous_raw_from_final_target",
    "write_recovery_report_new",
]
