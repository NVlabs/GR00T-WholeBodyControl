"""Pure, fail-closed teacher-support gate for causal SONIC true23 tuples.

This module never runs a policy, simulator, transport, or actuator and never
writes a file.  It validates already captured evidence.  One bad row or one bad
qualification run quarantines the complete 500-row logical window.  Teacher
labels exist only in a successfully rebuilt training export for a ``train``
session; heldout and quarantine records never contain them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
    TARGET_DOF,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_CONSTANTS_SHA256,
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_FORMULA_SHA256,
    SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
    SAFE_TARGET_TRANSFORM_KIND,
    safe_target_transform_numpy,
)

SUPPORT_CONFIG_RELATIVE_PATH = Path("gear_sonic/config/sim_validation/g1_true23_native124_21204_support_v1.json")
SUPPORT_CONFIG_SHA256 = "28d29c4204a3a0c4952723a9bfb40327ea2fa4aa50ca39bb510ea6121c6119e9"
SUPPORT_CONFIG_KIND = "g1_true23_native124_21204_support_contract_v1"
TRACKER_MANIFEST_SHA256 = "03435a9c86aff7f53ca12082dbfcf44e8579a9f16cbb7e44ddde21bf8f189826"
WINDOW_KIND = "g1_true23_causal_teacher_window_v1"
QUALIFICATION_KIND = "g1_true23_causal_teacher_qualification_run_v1"
GLOBAL_EVIDENCE_KIND = "g1_true23_native124_21204_global_gate_evidence_v1"
TRAINING_EXPORT_KIND = "g1_true23_support_admitted_training_window_v1"
QUARANTINE_KIND = "g1_true23_teacher_window_quarantine_v1"
SUPPORT_VERDICT_KIND = "g1_true23_teacher_window_support_verdict_v1"
TRANCHE_VERDICT_KIND = "g1_true23_teacher_support_tranche_verdict_v1"

# Exact default shared by linear SONIC diagnostics and V11/V12 safe-target path.
# Keep this module independent of simulator modules; imported pure transform
# constants and hashes bind final supervision semantics.
SONIC_HARDWARE_DEFAULT_Q = np.asarray(
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    dtype=np.float32,
)
SONIC_HARDWARE_ACTION_SCALE = np.asarray(HARDWARE_23_ACTION_SCALE, dtype=np.float32)
_NATIVE_TO_HARDWARE = np.asarray(ISAACLAB_TO_MUJOCO_DOF, dtype=np.int64)
_HARDWARE_TO_NATIVE = np.asarray(MUJOCO_TO_ISAACLAB_DOF, dtype=np.int64)
_POSITIVE_CAPACITY = np.asarray(SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE, dtype=np.float32)
_NEGATIVE_CAPACITY = np.asarray(SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE, dtype=np.float32)


class TeacherActionTransform(str, Enum):
    """Explicit label transform; only safe-target v2 is final-supervision eligible."""

    SONIC_V11_SAFE_TARGET_V2 = "sonic_v11_safe_target_v2"
    LINEAR_DIAGNOSTIC_V1 = "sonic_true23_linear_native_v1_diagnostic_only"


FINAL_TEACHER_ACTION_TRANSFORM = TeacherActionTransform.SONIC_V11_SAFE_TARGET_V2
SAFE_TARGET_INVERSE_FORMULA = (
    "raw_hw=where(delta>=0,p*atanh(delta/p),n*atanh(delta/n))/scale;"
    "raw_native=raw_hw[index(MUJOCO_TO_ISAACLAB_DOF)]"
)
TEACHER_COMPOSITE_NAME = "checkpoint21204_target_to_sonic_v11_safe_target_v2"
TEACHER_COMPOSITE_FORMULA = (
    "candidate_hw=checkpoint21204_home+checkpoint21204_scale*teacher_raw_hw;"
    "plain_raw_hw=(candidate_hw-sonic_default)/sonic_scale;"
    "plain_raw_native=plain_raw_hw[index(MUJOCO_TO_ISAACLAB_DOF)];"
    "(_,composite_hw)=safe_target_transform_v2(plain_raw_native);label=plain_raw_native"
)
TEACHER_COMPOSITE_FORMULA_SHA256 = "07b0d87205ed0bef60c8e1b30c92923774eaabf6f4b91b22b4950bc3e47db154"
TEACHER_COMPOSITE_CONSTANTS_SHA256 = "d9ea0d5d1c556b73ff584b087008c242ba9a77a4d2405b201c22d7597eee7db9"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_exact_config(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != 1
        or contract.get("kind") != SUPPORT_CONFIG_KIND
        or contract.get("role") != "offline_simulator_whole_window_teacher_support_gate_only"
    ):
        raise ValueError("teacher-support config identity mismatch")
    identity = _mapping(contract.get("artifact_identity"), "artifact_identity")
    if identity.get("tracker_manifest_sha256") != TRACKER_MANIFEST_SHA256:
        raise ValueError("teacher-support tracker manifest binding mismatch")
    action = _mapping(contract.get("student_action_contract"), "student_action_contract")
    if (
        action.get("name") != FINAL_TEACHER_ACTION_TRANSFORM.value
        or action.get("safe_target_kind") != SAFE_TARGET_TRANSFORM_KIND
        or action.get("safe_target_constants_sha256") != SAFE_TARGET_CONSTANTS_SHA256
        or action.get("safe_target_formula_sha256") != SAFE_TARGET_FORMULA_SHA256
        or action.get("student_joint_order") != "native_physx_il23_bfs_v1"
        or action.get("teacher_target_joint_order") != "unitree_mjlab_hardware_compact_23"
        or not np.array_equal(
            np.asarray(action.get("hardware_default_q"), dtype=np.float32),
            SONIC_HARDWARE_DEFAULT_Q,
        )
        or not np.array_equal(
            np.asarray(action.get("hardware_action_scale"), dtype=np.float32),
            SONIC_HARDWARE_ACTION_SCALE,
        )
        or action.get("isaaclab_to_mujoco_dof") != list(ISAACLAB_TO_MUJOCO_DOF)
        or action.get("mujoco_to_isaaclab_dof") != list(MUJOCO_TO_ISAACLAB_DOF)
        or action.get("student_raw_action_abs_strict_max") != SAFE_TARGET_RAW_ACTION_CLIP
        or action.get("reachable_domain_boundary") != "strict_exclusive_at_raw_clip"
        or action.get("inverse_formula") != SAFE_TARGET_INVERSE_FORMULA
        or action.get("forward_round_trip_atol") != 1.0e-5
        or action.get("student_applied_link_atol") != 1.0e-6
        or action.get("external_clamp_permitted") is not False
        or action.get("linear_diagnostic_helper_available") is not True
        or action.get("linear_final_label_permitted") is not False
    ):
        raise ValueError("teacher-support student action contract mismatch")
    composite = _mapping(contract.get("teacher_composite_contract"), "teacher_composite_contract")
    candidate_home = np.asarray(composite.get("checkpoint21204_home_q_hardware"), dtype=np.float32)
    candidate_scale = np.asarray(composite.get("checkpoint21204_action_scale_hardware"), dtype=np.float32)
    composite_constants = {
        "adapter_manifest_sha256": identity.get("tracker_manifest_sha256"),
        "checkpoint21204_home_q_hardware": candidate_home.tolist(),
        "checkpoint21204_action_scale_hardware": candidate_scale.tolist(),
        "sonic_default_q_hardware": list(SAFE_TARGET_DEFAULT_Q_HARDWARE),
        "sonic_action_scale_hardware": list(HARDWARE_23_ACTION_SCALE),
        "mujoco_to_isaaclab_dof": list(MUJOCO_TO_ISAACLAB_DOF),
        "safe_target_constants_sha256": SAFE_TARGET_CONSTANTS_SHA256,
        "safe_target_formula_sha256": SAFE_TARGET_FORMULA_SHA256,
        "raw_action_clip": SAFE_TARGET_RAW_ACTION_CLIP,
    }
    if (
        composite.get("name") != TEACHER_COMPOSITE_NAME
        or composite.get("formula") != TEACHER_COMPOSITE_FORMULA
        or composite.get("formula_sha256") != TEACHER_COMPOSITE_FORMULA_SHA256
        or _sha256_bytes(TEACHER_COMPOSITE_FORMULA.encode("utf-8")) != TEACHER_COMPOSITE_FORMULA_SHA256
        or composite.get("constants_sha256") != TEACHER_COMPOSITE_CONSTANTS_SHA256
        or _canonical_sha256(composite_constants) != TEACHER_COMPOSITE_CONSTANTS_SHA256
        or candidate_home.shape != (TARGET_DOF,)
        or candidate_scale.shape != (TARGET_DOF,)
        or not np.isfinite(candidate_home).all()
        or not np.isfinite(candidate_scale).all()
        or np.any(candidate_scale <= 0.0)
        or composite.get("teacher_raw_joint_order") != "unitree_mjlab_hardware_compact_23"
        or composite.get("candidate_target_joint_order") != "unitree_mjlab_hardware_compact_23"
        or composite.get("composite_target_joint_order") != "unitree_mjlab_hardware_compact_23"
        or composite.get("label_joint_order") != "native_physx_il23_bfs_v1"
        or composite.get("label_semantics") != "pre_safe_transform_plain_sonic_raw_native"
        or composite.get("candidate_link_atol") != 1.0e-6
        or composite.get("composite_link_atol") != 1.0e-5
        or composite.get("forward_target_round_trip_atol") != 1.0e-5
        or composite.get("raw_label_strict_abs_max") != SAFE_TARGET_RAW_ACTION_CLIP
        or composite.get("clipping_permitted") is not False
    ):
        raise ValueError("teacher-support composite action contract mismatch")
    window = _mapping(contract.get("causal_window"), "causal_window")
    if window != {
        "emitted_control_rows": 500,
        "source_samples": 510,
        "history_warmup_samples": 10,
        "control_period_ns": 20_000_000,
        "encoder_dim": 267,
        "token_dim": 64,
        "proprio_dim": 930,
        "decoder_dim": 994,
        "action_dim": 23,
        "pico_age_ms_max": 60.0,
        "lowstate_age_ms_max": 40.0,
        "inference_duration_ms_strict_max": 20.0,
        "exact_decoder_concat": True,
        "partial_row_masking_permitted": False,
        "reset_padding_permitted": False,
    }:
        raise ValueError("teacher-support causal window contract mismatch")
    qualification = _mapping(contract.get("qualification"), "qualification")
    if (
        qualification.get("nominal_rollouts") != 10
        or qualification.get("disturbance_rollouts") != 10
        or qualification.get("steps_per_rollout") != 500
        or len(qualification.get("nominal_seeds", ())) != 10
        or len(qualification.get("disturbance_seeds", ())) != 10
        or len(set(qualification.get("nominal_seeds", ()))) != 10
        or len(set(qualification.get("disturbance_seeds", ()))) != 10
        or set(qualification.get("nominal_seeds", ())) & set(qualification.get("disturbance_seeds", ()))
        or qualification.get("seed_scope") != "deterministic_reproducibility_only_not_statistical_sufficiency"
    ):
        raise ValueError("teacher-support qualification contract mismatch")
    tranche = _mapping(contract.get("split_tranche"), "split_tranche")
    if (
        tranche.get("families") != ["idle/neutral", "reach", "crouch", "turn", "walk", "fast/transitions"]
        or tranche.get("session_splits") != ["train", "heldout"]
        or tranche.get("minimum_independent_sessions_per_family") != 3
        or tranche.get("minimum_train_sessions_per_family") != 2
        or tranche.get("minimum_heldout_sessions_per_family") != 1
        or tranche.get("minimum_admitted_windows_per_train_session") != 1
        or tranche.get("whole_session_split_required") is not True
        or tranche.get("whole_session_family_required") is not True
        or tranche.get("heldout_training_export_permitted") is not False
    ):
        raise ValueError("teacher-support split/tranche contract mismatch")
    boundaries = _mapping(contract.get("boundaries"), "boundaries")
    if (
        boundaries.get("pure_validation_only") is not True
        or boundaries.get("offline_or_simulator_only") is not True
        or any(
            boundaries.get(name) is not False
            for name in (
                "actuation_permitted",
                "deployment_ready",
                "hardware_authorized",
                "teacher_labels_admitted_without_gate",
                "promotion_eligible",
            )
        )
    ):
        raise ValueError("teacher-support safety boundary mismatch")


def _validate_tracker_manifest_binding(
    root: Path,
    contract: Mapping[str, Any],
) -> None:
    identity = _mapping(contract["artifact_identity"], "artifact_identity")
    relative = identity.get("tracker_manifest_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("tracker manifest path must be repository-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError("tracker manifest must be a regular repository file")
    payload = path.read_bytes()
    if _sha256_bytes(payload) != identity.get("tracker_manifest_sha256"):
        raise ValueError("tracker manifest SHA256 binding mismatch")
    try:
        manifest = _mapping(json.loads(payload.decode("utf-8")), "tracker manifest")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("tracker manifest must be valid UTF-8 JSON") from error
    manifest_identity = _mapping(manifest.get("identity"), "tracker manifest identity")
    observation = _mapping(
        manifest.get("observation_contract"),
        "tracker manifest observation_contract",
    )
    composite = _mapping(contract["teacher_composite_contract"], "teacher_composite_contract")
    expected_refs = {
        "selection": "selection_sha256",
        "export_report": "export_report_sha256",
        "checkpoint": "checkpoint_sha256",
        "onnx": "onnx_sha256",
    }
    refs_match = all(
        _mapping(manifest_identity.get(name), f"tracker manifest {name}").get("sha256")
        == identity.get(identity_key)
        for name, identity_key in expected_refs.items()
    )
    checkpoint_ref = _mapping(manifest_identity.get("checkpoint"), "tracker manifest checkpoint")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "g1_true23_native124_21204_tracker_manifest_v1"
        or manifest.get("role") != "offline_simulator_causal_tracker_teacher_candidate_only"
        or manifest_identity.get("iteration") != 21204
        or not refs_match
        or checkpoint_ref.get("actor_state_sha256") != identity.get("actor_state_sha256")
        or not np.array_equal(
            np.asarray(observation.get("home_q_hardware"), dtype=np.float32),
            np.asarray(composite["checkpoint21204_home_q_hardware"], dtype=np.float32),
        )
        or not np.array_equal(
            np.asarray(observation.get("action_scale_hardware"), dtype=np.float32),
            np.asarray(
                composite["checkpoint21204_action_scale_hardware"],
                dtype=np.float32,
            ),
        )
    ):
        raise ValueError("tracker manifest/composite constants mismatch")


def load_teacher_support_contract(
    repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Read and hash-check exact frozen support contract; never mutate it."""

    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[2]
    path = (root / SUPPORT_CONFIG_RELATIVE_PATH).resolve()
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError("teacher-support config must be a regular repository file")
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != SUPPORT_CONFIG_SHA256:
        raise ValueError(f"teacher-support config SHA256 mismatch: expected {SUPPORT_CONFIG_SHA256}, got {actual}")
    try:
        contract = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("teacher-support config must be valid UTF-8 JSON") from error
    result = _mapping(contract, "teacher-support config")
    _require_exact_config(result)
    _validate_tracker_manifest_binding(root, result)
    return result


def _finite_vector(value: object, size: int, context: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be finite shape ({size},)") from error
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{context} must be finite shape ({size},)")
    return result


@dataclass(frozen=True)
class Checkpoint21204TeacherComposite:
    """All four linked action spaces for one frozen teacher query."""

    teacher_raw_action_hardware: np.ndarray
    teacher_candidate_target_hardware: np.ndarray
    teacher_action_native: np.ndarray
    teacher_applied_safe_action_native: np.ndarray
    teacher_target_hardware: np.ndarray

    def to_record(self) -> dict[str, list[float]]:
        return {
            "teacher_raw_action_hardware": self.teacher_raw_action_hardware.tolist(),
            "teacher_candidate_target_hardware": self.teacher_candidate_target_hardware.tolist(),
            "teacher_action_native": self.teacher_action_native.tolist(),
            "teacher_applied_safe_action_native": self.teacher_applied_safe_action_native.tolist(),
            "teacher_target_hardware": self.teacher_target_hardware.tolist(),
        }


def _compose_checkpoint21204_teacher_action(
    raw_action_hardware: object,
    contract: Mapping[str, Any],
) -> Checkpoint21204TeacherComposite:
    raw = _finite_vector(
        raw_action_hardware,
        TARGET_DOF,
        "teacher_raw_action_hardware",
    )
    composite = _mapping(contract["teacher_composite_contract"], "teacher_composite_contract")
    candidate_home = np.asarray(composite["checkpoint21204_home_q_hardware"], dtype=np.float32)
    candidate_scale = np.asarray(composite["checkpoint21204_action_scale_hardware"], dtype=np.float32)
    candidate_target = (candidate_home + candidate_scale * raw).astype(np.float32, copy=False)
    plain_raw_hardware = ((candidate_target - SONIC_HARDWARE_DEFAULT_Q) / SONIC_HARDWARE_ACTION_SCALE).astype(
        np.float32, copy=False
    )
    plain_raw_native = plain_raw_hardware[_HARDWARE_TO_NATIVE].astype(np.float32, copy=False)
    strict_limit = float(composite["raw_label_strict_abs_max"])
    if float(np.max(np.abs(plain_raw_native))) >= strict_limit:
        raise ValueError("teacher composite requires plain SONIC raw clipping")
    safe_native, composite_target = safe_target_transform_numpy(plain_raw_native)
    action_contract = _mapping(contract["student_action_contract"], "student_action_contract")
    inverse_label = _invert_teacher_hardware_target(composite_target, action_contract)
    if not np.isfinite(inverse_label).all():
        raise ValueError("teacher composite safe inverse does not recover plain SONIC raw label")
    if float(np.max(np.abs(inverse_label))) >= strict_limit:
        raise ValueError("teacher composite safe inverse does not recover plain SONIC raw label")
    forward_safe, forward_target = safe_target_transform_numpy(inverse_label)
    if not np.allclose(
        forward_safe,
        safe_native,
        rtol=0.0,
        atol=float(action_contract["student_applied_link_atol"]),
    ):
        raise ValueError("teacher composite safe native action does not recover V2 forward transform")
    if not np.allclose(
        forward_target,
        composite_target,
        rtol=0.0,
        atol=float(composite["forward_target_round_trip_atol"]),
    ):
        raise ValueError("teacher composite final hardware target does not recover V2 forward transform")
    return Checkpoint21204TeacherComposite(
        teacher_raw_action_hardware=raw.copy(),
        teacher_candidate_target_hardware=candidate_target.copy(),
        teacher_action_native=plain_raw_native.copy(),
        teacher_applied_safe_action_native=safe_native.copy(),
        teacher_target_hardware=composite_target.copy(),
    )


def compose_checkpoint21204_teacher_action(
    raw_action_hardware: Sequence[float] | np.ndarray,
    *,
    repository_root: str | Path | None = None,
) -> Checkpoint21204TeacherComposite:
    """Build exact selected-target → SONIC-label → V2-target composite."""

    contract = load_teacher_support_contract(repository_root)
    return _compose_checkpoint21204_teacher_action(raw_action_hardware, contract)


def sonic_linear_native_action_to_hardware_target(
    native_action: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply legacy linear mapping for diagnostics; never final teacher labels."""

    native = _finite_vector(native_action, TARGET_DOF, "native_action")
    hardware = native[_NATIVE_TO_HARDWARE]
    return (SONIC_HARDWARE_DEFAULT_Q + hardware * SONIC_HARDWARE_ACTION_SCALE).astype(
        np.float32,
        copy=False,
    )


def teacher_hardware_target_to_linear_native_diagnostic(
    teacher_target_hardware: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Invert legacy linear map for comparison only; not a training label."""

    target = _finite_vector(
        teacher_target_hardware,
        TARGET_DOF,
        "teacher_target_hardware",
    )
    raw_hardware = (target - SONIC_HARDWARE_DEFAULT_Q) / SONIC_HARDWARE_ACTION_SCALE
    native = raw_hardware[_HARDWARE_TO_NATIVE].astype(np.float32, copy=False)
    reconstructed = sonic_linear_native_action_to_hardware_target(native)
    if not np.allclose(reconstructed, target, rtol=0.0, atol=1.0e-5):
        raise ValueError("teacher target fails diagnostic linear forward round trip")
    return native.copy()


def sonic_v11_native_action_to_hardware_target(
    native_action: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply exact V11/V12 safe-target v2 map for final teacher supervision."""

    native = _finite_vector(native_action, TARGET_DOF, "native_action")
    if float(np.max(np.abs(native))) >= SAFE_TARGET_RAW_ACTION_CLIP:
        raise ValueError("final teacher label must remain strictly inside raw clip domain")
    _, target = safe_target_transform_numpy(native)
    return target.copy()


def sonic_native_label_to_hardware_target(
    native_action: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Forward final V11/V12 safe-target teacher-label contract."""

    return sonic_v11_native_action_to_hardware_target(native_action)


def teacher_hardware_target_to_sonic_native_label(
    teacher_target_hardware: Sequence[float] | np.ndarray,
    *,
    repository_root: str | Path | None = None,
) -> np.ndarray:
    """Invert final V11/V12 safe-target map or reject unrepresentable target."""

    contract = load_teacher_support_contract(repository_root)
    action = _mapping(contract["student_action_contract"], "student_action_contract")
    return _invert_teacher_hardware_target(teacher_target_hardware, action)


def teacher_hardware_target_to_native_action(
    teacher_target_hardware: Sequence[float] | np.ndarray,
    *,
    transform: TeacherActionTransform = FINAL_TEACHER_ACTION_TRANSFORM,
    repository_root: str | Path | None = None,
) -> np.ndarray:
    """Explicit transform router; linear result stays diagnostic-only."""

    if transform is TeacherActionTransform.SONIC_V11_SAFE_TARGET_V2:
        return teacher_hardware_target_to_sonic_native_label(
            teacher_target_hardware,
            repository_root=repository_root,
        )
    if transform is TeacherActionTransform.LINEAR_DIAGNOSTIC_V1:
        return teacher_hardware_target_to_linear_native_diagnostic(teacher_target_hardware)
    raise ValueError(f"unsupported teacher action transform: {transform!r}")


def _invert_teacher_hardware_target(
    teacher_target_hardware: object,
    action_contract: Mapping[str, Any],
) -> np.ndarray:
    target = _finite_vector(
        teacher_target_hardware,
        TARGET_DOF,
        "teacher_target_hardware",
    )
    lower_raw = np.full(TARGET_DOF, -SAFE_TARGET_RAW_ACTION_CLIP, dtype=np.float32)
    upper_raw = np.full(TARGET_DOF, SAFE_TARGET_RAW_ACTION_CLIP, dtype=np.float32)
    _, reachable_lower = safe_target_transform_numpy(lower_raw)
    _, reachable_upper = safe_target_transform_numpy(upper_raw)
    if np.any(target <= reachable_lower) or np.any(target >= reachable_upper):
        raise ValueError("teacher target lies outside strict V11 safe reachable domain")
    delta = target - SONIC_HARDWARE_DEFAULT_Q
    capacity = np.where(delta >= np.float32(0.0), _POSITIVE_CAPACITY, _NEGATIVE_CAPACITY)
    ratio = delta / capacity
    if np.any(np.abs(ratio) >= np.float32(1.0)):
        raise ValueError("teacher target cannot be inverted through safe-target tanh")
    raw_hardware_delta = capacity * np.arctanh(ratio)
    raw_hardware = raw_hardware_delta / SONIC_HARDWARE_ACTION_SCALE
    native = raw_hardware[_HARDWARE_TO_NATIVE].astype(np.float32, copy=False)
    raw_limit = float(action_contract["student_raw_action_abs_strict_max"])
    if float(np.max(np.abs(native))) >= raw_limit:
        raise ValueError("teacher target requires student raw clipping")
    reconstructed = sonic_native_label_to_hardware_target(native)
    if not np.allclose(
        reconstructed,
        target,
        rtol=0.0,
        atol=float(action_contract["forward_round_trip_atol"]),
    ):
        raise ValueError("teacher target fails exact SONIC forward round trip")
    return native.copy()


class _Issues:
    def __init__(self) -> None:
        self.values: list[str] = []

    def add(self, value: str) -> None:
        if value not in self.values:
            self.values.append(value)

    def exact(self, actual: object, expected: object, context: str) -> None:
        if actual != expected:
            self.add(f"{context}: expected {expected!r}, got {actual!r}")

    def false(self, actual: object, context: str) -> None:
        if actual is not False:
            self.add(f"{context}: expected false")

    def zero_count(self, actual: object, context: str) -> None:
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != 0:
            self.add(f"{context}: expected integer zero")


def _float_metric(
    value: object,
    context: str,
    issues: _Issues,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.add(f"{context}: expected finite number")
        return None
    result = float(value)
    if not math.isfinite(result):
        issues.add(f"{context}: expected finite number")
        return None
    return result


def _array_metric(
    value: object,
    size: int,
    context: str,
    issues: _Issues,
) -> np.ndarray | None:
    try:
        return _finite_vector(value, size, context)
    except ValueError as error:
        issues.add(str(error))
        return None


def _nonempty_string(value: object, context: str, issues: _Issues) -> str:
    if not isinstance(value, str) or not value:
        issues.add(f"{context}: expected non-empty string")
        return ""
    return value


def _nonnegative_int(value: object, context: str, issues: _Issues) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        issues.add(f"{context}: expected non-negative integer")
        return -1
    return value


def _validate_global_evidence(
    evidence: object,
    contract: Mapping[str, Any],
    issues: _Issues,
) -> None:
    if not isinstance(evidence, Mapping):
        issues.add("global_evidence: expected mapping")
        return
    gate = _mapping(contract["global_gate"], "global_gate")
    identity = _mapping(contract["artifact_identity"], "artifact_identity")
    issues.exact(evidence.get("schema_version"), 1, "global_evidence.schema_version")
    issues.exact(evidence.get("kind"), GLOBAL_EVIDENCE_KIND, "global_evidence.kind")
    issues.exact(
        evidence.get("artifact_identity"),
        dict(identity),
        "global_evidence.artifact_identity",
    )
    for key in (
        "onnx_input_count",
        "onnx_input_name",
        "onnx_input_shape",
        "onnx_output_name",
        "onnx_output_shape",
        "clip_count",
        "rollouts_per_clip",
        "steps_per_rollout",
        "original_six_total_rollouts",
        "zero_completion_clips",
    ):
        issues.exact(evidence.get(key), gate[key], f"global_evidence.{key}")
    issues.exact(evidence.get("abi_passed"), True, "global_evidence.abi_passed")
    issues.exact(evidence.get("parity_passed"), True, "global_evidence.parity_passed")
    minimums = (
        ("parity_case_count", "parity_case_count_min"),
        ("mean_completion_score", "mean_completion_score_min"),
        ("mean_survival_score", "mean_survival_score_min"),
        ("perfect_clip_count", "perfect_clip_count_min"),
        ("original_six_completed_rollouts", "original_six_completed_rollouts_min"),
    )
    for actual_key, expected_key in minimums:
        value = _float_metric(evidence.get(actual_key), f"global_evidence.{actual_key}", issues)
        if value is not None and value < float(gate[expected_key]):
            issues.add(f"global_evidence.{actual_key}: below frozen floor")
    maximums = (
        ("parity_max_absolute_error", "parity_max_absolute_error_max"),
        ("nonperfect_clip_count", "nonperfect_clip_count_max"),
    )
    for actual_key, expected_key in maximums:
        value = _float_metric(evidence.get(actual_key), f"global_evidence.{actual_key}", issues)
        if value is not None and value > float(gate[expected_key]):
            issues.add(f"global_evidence.{actual_key}: above frozen ceiling")
    issues.exact(
        evidence.get("new_failing_named_clips"),
        gate["new_failing_named_clips_allowed"],
        "global_evidence.new_failing_named_clips",
    )


def _identity_fields(window: Mapping[str, Any], issues: _Issues) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("window_id", "source_session_id", "split_manifest_sha256"):
        result[name] = _nonempty_string(window.get(name), f"window.{name}", issues)
    if result["split_manifest_sha256"] and not _is_sha256(result["split_manifest_sha256"]):
        issues.add("window.split_manifest_sha256: expected lowercase SHA-256")
    result["family"] = _nonempty_string(window.get("family"), "window.family", issues)
    result["split"] = _nonempty_string(window.get("split"), "window.split", issues)
    result["student_checkpoint_sha256"] = _nonempty_string(
        window.get("student_checkpoint_sha256"),
        "window.student_checkpoint_sha256",
        issues,
    )
    if result["student_checkpoint_sha256"] and not _is_sha256(result["student_checkpoint_sha256"]):
        issues.add("window.student_checkpoint_sha256: expected lowercase SHA-256")
    result["learner_iteration"] = _nonnegative_int(
        window.get("learner_iteration"), "window.learner_iteration", issues
    )
    result["collection_seed"] = _nonnegative_int(window.get("collection_seed"), "window.collection_seed", issues)
    result["exploration_seed"] = _nonnegative_int(
        window.get("exploration_seed"), "window.exploration_seed", issues
    )
    return result


def _validate_source_samples(
    samples: object,
    contract: Mapping[str, Any],
    issues: _Issues,
) -> list[tuple[int, int]]:
    window_cfg = _mapping(contract["causal_window"], "causal_window")
    expected_count = int(window_cfg["source_samples"])
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        issues.add("window.source_samples: expected sequence")
        return []
    if len(samples) != expected_count:
        issues.add(f"window.source_samples: expected {expected_count}, got {len(samples)}")
        return []
    result: list[tuple[int, int]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            issues.add(f"source_samples[{index}]: expected mapping")
            return []
        frame = _nonnegative_int(
            sample.get("source_frame_index"),
            f"source_samples[{index}].source_frame_index",
            issues,
        )
        monotonic_ns = _nonnegative_int(
            sample.get("reference_monotonic_ns"),
            f"source_samples[{index}].reference_monotonic_ns",
            issues,
        )
        result.append((frame, monotonic_ns))
    period_ns = int(window_cfg["control_period_ns"])
    for index in range(1, len(result)):
        previous_frame, previous_ns = result[index - 1]
        frame, monotonic_ns = result[index]
        if frame != previous_frame + 1:
            issues.add(f"source_samples[{index}]: noncontiguous source frame")
        if monotonic_ns != previous_ns + period_ns:
            issues.add(f"source_samples[{index}]: timestamp gap is not exactly {period_ns} ns")
    return result


def _validate_row_arrays(
    row: Mapping[str, Any],
    index: int,
    contract: Mapping[str, Any],
    issues: _Issues,
) -> None:
    _array_metric(row.get("encoder267"), 267, f"rows[{index}].encoder267", issues)
    token = _array_metric(row.get("token64"), 64, f"rows[{index}].token64", issues)
    proprio = _array_metric(row.get("proprio930"), 930, f"rows[{index}].proprio930", issues)
    decoder = _array_metric(row.get("decoder994"), 994, f"rows[{index}].decoder994", issues)
    action_values: dict[str, np.ndarray] = {}
    for name in (
        "student_mean_action_native",
        "student_raw_action_native",
        "student_applied_action_native",
        "teacher_raw_action_hardware",
        "teacher_candidate_target_hardware",
        "teacher_target_hardware",
    ):
        value = _array_metric(row.get(name), TARGET_DOF, f"rows[{index}].{name}", issues)
        if value is not None:
            action_values[name] = value
    raw_student = action_values.get("student_raw_action_native")
    applied_student = action_values.get("student_applied_action_native")
    if raw_student is not None and float(np.max(np.abs(raw_student))) >= SAFE_TARGET_RAW_ACTION_CLIP:
        issues.add(f"rows[{index}].student_raw_action_native: requires raw clipping")
    elif raw_student is not None and applied_student is not None:
        expected_applied, _ = safe_target_transform_numpy(raw_student)
        action_contract = _mapping(
            contract["student_action_contract"],
            "student_action_contract",
        )
        if not np.allclose(
            applied_student,
            expected_applied,
            rtol=0.0,
            atol=float(action_contract["student_applied_link_atol"]),
        ):
            issues.add(
                f"rows[{index}].student_applied_action_native: "
                "does not match exact V2 transform of student raw action"
            )
    if token is not None and proprio is not None and decoder is not None:
        expected = np.concatenate((token, proprio)).astype(np.float32, copy=False)
        if not np.array_equal(decoder, expected):
            issues.add(f"rows[{index}].decoder994: not exact token64/proprio930 concat")
    raw_teacher = action_values.get("teacher_raw_action_hardware")
    candidate = action_values.get("teacher_candidate_target_hardware")
    target = action_values.get("teacher_target_hardware")
    if raw_teacher is not None:
        try:
            expected = _compose_checkpoint21204_teacher_action(raw_teacher, contract)
        except ValueError as error:
            issues.add(f"rows[{index}].teacher composite: {error}")
        else:
            composite = _mapping(
                contract["teacher_composite_contract"],
                "teacher_composite_contract",
            )
            if candidate is not None and not np.allclose(
                candidate,
                expected.teacher_candidate_target_hardware,
                rtol=0.0,
                atol=float(composite["candidate_link_atol"]),
            ):
                issues.add(
                    f"rows[{index}].teacher_candidate_target_hardware: does not match teacher raw/HOME/SCALE link"
                )
            if target is not None and not np.allclose(
                target,
                expected.teacher_target_hardware,
                rtol=0.0,
                atol=float(composite["composite_link_atol"]),
            ):
                issues.add(
                    f"rows[{index}].teacher_target_hardware: does not match checkpoint21204-to-V2 composite"
                )


def _validate_next_state(row: Mapping[str, Any], index: int, issues: _Issues) -> None:
    state = row.get("next_state")
    if not isinstance(state, Mapping):
        issues.add(f"rows[{index}].next_state: expected mapping")
        return
    for name, size in (
        ("q_hardware", 23),
        ("qd_hardware", 23),
        ("base_angular_velocity", 3),
        ("torso_quaternion_wxyz", 4),
    ):
        _array_metric(state.get(name), size, f"rows[{index}].next_state.{name}", issues)


def _validate_row_metrics(
    row: Mapping[str, Any],
    index: int,
    contract: Mapping[str, Any],
    issues: _Issues,
) -> None:
    cfg = _mapping(contract["causal_window"], "causal_window")
    qualification = _mapping(contract["qualification"], "qualification")
    comparisons = (
        ("pico_age_ms", float(cfg["pico_age_ms_max"]), False),
        ("lowstate_age_ms", float(cfg["lowstate_age_ms_max"]), False),
        ("inference_duration_ms", float(cfg["inference_duration_ms_strict_max"]), True),
        ("base_tilt_rad", float(qualification["maximum_base_tilt_rad"]), False),
        ("joint_velocity_ratio", float(qualification["maximum_joint_velocity_ratio"]), False),
    )
    for name, maximum, strict in comparisons:
        value = _float_metric(row.get(name), f"rows[{index}].{name}", issues)
        if value is not None and ((value >= maximum) if strict else (value > maximum)):
            qualifier = "strict ceiling" if strict else "ceiling"
            issues.add(f"rows[{index}].{name}: above {qualifier}")
        if value is not None and value < 0.0:
            issues.add(f"rows[{index}].{name}: negative")
    base_height = _float_metric(row.get("base_height_m"), f"rows[{index}].base_height_m", issues)
    if base_height is not None and base_height < float(qualification["minimum_base_height_m"]):
        issues.add(f"rows[{index}].base_height_m: below floor")
    tracking_error = _float_metric(row.get("tracking_error_rad"), f"rows[{index}].tracking_error_rad", issues)
    if tracking_error is not None and tracking_error < 0.0:
        issues.add(f"rows[{index}].tracking_error_rad: negative")
    for name in (
        "measured_joint_limit_failure_count",
        "teacher_target_soft_limit_failure_count",
        "target_clamp_failure_count",
        "nonfinite_count",
    ):
        issues.zero_count(row.get(name), f"rows[{index}].{name}")
    for name in (
        "reset_occurred",
        "reference_resampled",
        "clip_boundary",
        "session_boundary",
        "terminated",
        "anchor_failure",
        "ee_failure",
        "reported_nonfinite",
    ):
        issues.false(row.get(name), f"rows[{index}].{name}")
    if row.get("sim_advanced") is not True:
        issues.add(f"rows[{index}].sim_advanced: expected true")
    timed_out = row.get("timed_out")
    done = row.get("done")
    if index < int(cfg["emitted_control_rows"]) - 1:
        issues.false(timed_out, f"rows[{index}].timed_out")
        issues.false(done, f"rows[{index}].done")
    elif timed_out not in (False, True) or done is not timed_out:
        issues.add(f"rows[{index}]: final done must exactly equal boolean timed_out")


def _validate_rows(
    rows: object,
    source_samples: list[tuple[int, int]],
    identity: Mapping[str, object],
    teacher_identity: Mapping[str, Any],
    contract: Mapping[str, Any],
    issues: _Issues,
) -> None:
    cfg = _mapping(contract["causal_window"], "causal_window")
    expected_count = int(cfg["emitted_control_rows"])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        issues.add("window.rows: expected sequence")
        return
    if len(rows) != expected_count:
        issues.add(f"window.rows: expected {expected_count}, got {len(rows)}")
        return
    warmup = int(cfg["history_warmup_samples"])
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            issues.add(f"rows[{index}]: expected mapping")
            continue
        if "teacher_action_native" in row:
            issues.add(f"rows[{index}].teacher_action_native: pre-admission label is forbidden")
        issues.exact(row.get("row_index"), index, f"rows[{index}].row_index")
        for name, expected in identity.items():
            issues.exact(row.get(name), expected, f"rows[{index}].{name}")
        issues.exact(
            row.get("teacher_identity"),
            dict(teacher_identity),
            f"rows[{index}].teacher_identity",
        )
        if source_samples:
            anchor_frame, anchor_ns = source_samples[index + warmup - 1]
            control_frame, control_ns = source_samples[index + warmup]
            for name, expected in (
                ("anchor_source_frame_index", anchor_frame),
                ("anchor_reference_monotonic_ns", anchor_ns),
                ("control_source_frame_index", control_frame),
                ("control_monotonic_ns", control_ns),
            ):
                issues.exact(row.get(name), expected, f"rows[{index}].{name}")
        _validate_row_arrays(row, index, contract, issues)
        _validate_next_state(row, index, issues)
        _validate_row_metrics(row, index, contract, issues)


def _validate_run_metrics(
    run: Mapping[str, Any],
    context: str,
    mode: str,
    contract: Mapping[str, Any],
    issues: _Issues,
) -> None:
    qualification = _mapping(contract["qualification"], "qualification")
    window_cfg = _mapping(contract["causal_window"], "causal_window")
    for name in _mapping(contract["failure_policy"], "failure_policy")["required_zero_counts"]:
        issues.zero_count(run.get(name), f"{context}.{name}")
    issues.exact(run.get("completed"), True, f"{context}.completed")
    issues.exact(
        run.get("steps_requested"),
        qualification["steps_per_rollout"],
        f"{context}.steps_requested",
    )
    issues.exact(
        run.get("steps_executed"),
        qualification["steps_per_rollout"],
        f"{context}.steps_executed",
    )
    minimum = _float_metric(run.get("minimum_base_height_m"), f"{context}.minimum_base_height_m", issues)
    if minimum is not None and minimum < float(qualification["minimum_base_height_m"]):
        issues.add(f"{context}.minimum_base_height_m: below floor")
    ceilings = (
        ("maximum_base_tilt_rad", "maximum_base_tilt_rad", False),
        ("maximum_joint_velocity_ratio", "maximum_joint_velocity_ratio", False),
        ("tracking_rmse_rad", "tracking_rmse_rad_max", False),
        ("maximum_abs_student_raw_action", "native_action_abs_max", True),
        ("maximum_pico_age_ms", "pico_age_ms_max", False),
        ("maximum_lowstate_age_ms", "lowstate_age_ms_max", False),
        ("maximum_inference_duration_ms", "inference_duration_ms_strict_max", True),
    )
    for name, config_name, strict in ceilings:
        source = window_cfg if config_name in window_cfg else qualification
        value = _float_metric(run.get(name), f"{context}.{name}", issues)
        maximum = float(source[config_name])
        if value is not None and ((value >= maximum) if strict else (value > maximum)):
            qualifier = "strict ceiling" if strict else "ceiling"
            issues.add(f"{context}.{name}: above {qualifier}")
    disturbance = _mapping(qualification["disturbance"], "qualification.disturbance")
    if mode == "nominal":
        issues.exact(run.get("disturbance_profile"), "none", f"{context}.disturbance_profile")
        issues.false(run.get("disturbance_applied"), f"{context}.disturbance_applied")
        return
    issues.exact(
        run.get("disturbance_profile"),
        disturbance["profile"],
        f"{context}.disturbance_profile",
    )
    issues.exact(run.get("disturbance_applied"), True, f"{context}.disturbance_applied")
    for name in ("apply_step", "baseline_steps", "stable_recovery_steps", "recovery_margin"):
        issues.exact(run.get(name), disturbance[name], f"{context}.{name}")
    recovery = _float_metric(run.get("recovery_fraction"), f"{context}.recovery_fraction", issues)
    if recovery is not None and recovery != float(qualification["recovery_fraction_required"]):
        issues.add(f"{context}.recovery_fraction: expected exact full recovery")
    recovery_time = _float_metric(
        run.get("maximum_recovery_time_s"),
        f"{context}.maximum_recovery_time_s",
        issues,
    )
    if recovery_time is not None and recovery_time > float(qualification["maximum_recovery_time_s"]):
        issues.add(f"{context}.maximum_recovery_time_s: above ceiling")
    linear = _array_metric(
        run.get("disturbance_linear_velocity_delta_m_s"),
        3,
        f"{context}.disturbance_linear_velocity_delta_m_s",
        issues,
    )
    angular = _array_metric(
        run.get("disturbance_angular_velocity_delta_rad_s"),
        3,
        f"{context}.disturbance_angular_velocity_delta_rad_s",
        issues,
    )
    if linear is not None and np.any(
        np.abs(linear) > np.asarray(disturbance["linear_velocity_abs_max_m_s"], dtype=np.float32)
    ):
        issues.add(f"{context}.disturbance_linear_velocity_delta_m_s: outside envelope")
    if angular is not None and np.any(
        np.abs(angular) > np.asarray(disturbance["angular_velocity_abs_max_rad_s"], dtype=np.float32)
    ):
        issues.add(f"{context}.disturbance_angular_velocity_delta_rad_s: outside envelope")
    if linear is not None and angular is not None and not np.any(linear) and not np.any(angular):
        issues.add(f"{context}: disturbance impulse is zero")


def _validate_qualification_runs(
    runs: object,
    identity: Mapping[str, object],
    teacher_identity: Mapping[str, Any],
    contract: Mapping[str, Any],
    issues: _Issues,
) -> tuple[int, int]:
    qualification = _mapping(contract["qualification"], "qualification")
    expected_total = int(qualification["nominal_rollouts"]) + int(qualification["disturbance_rollouts"])
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        issues.add("qualification_runs: expected sequence")
        return 0, 0
    if len(runs) != expected_total:
        issues.add(f"qualification_runs: expected {expected_total}, got {len(runs)}")
    seen: set[tuple[str, int]] = set()
    mode_counts = {"nominal": 0, "disturbance": 0}
    for index, run in enumerate(runs):
        context = f"qualification_runs[{index}]"
        if not isinstance(run, Mapping):
            issues.add(f"{context}: expected mapping")
            continue
        issues.exact(run.get("schema_version"), 1, f"{context}.schema_version")
        issues.exact(run.get("kind"), QUALIFICATION_KIND, f"{context}.kind")
        for name, expected in identity.items():
            issues.exact(run.get(name), expected, f"{context}.{name}")
        issues.exact(
            run.get("teacher_identity"),
            dict(teacher_identity),
            f"{context}.teacher_identity",
        )
        mode = run.get("mode")
        if mode not in mode_counts:
            issues.add(f"{context}.mode: expected nominal or disturbance")
            continue
        rollout_index = run.get("rollout_index")
        if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
            issues.add(f"{context}.rollout_index: expected integer")
            continue
        expected_count = int(qualification[f"{mode}_rollouts"])
        if not 0 <= rollout_index < expected_count:
            issues.add(f"{context}.rollout_index: outside frozen range")
            continue
        key = (mode, rollout_index)
        if key in seen:
            issues.add(f"{context}: duplicate {mode} rollout index {rollout_index}")
        seen.add(key)
        mode_counts[mode] += 1
        expected_seed = qualification[f"{mode}_seeds"][rollout_index]
        issues.exact(run.get("seed"), expected_seed, f"{context}.seed")
        _validate_run_metrics(run, context, mode, contract, issues)
    for mode, count in mode_counts.items():
        expected = int(qualification[f"{mode}_rollouts"])
        if count != expected:
            issues.add(f"qualification_runs: expected {expected} unique {mode} runs, got {count}")
    return mode_counts["nominal"], mode_counts["disturbance"]


@dataclass(frozen=True)
class TeacherSupportVerdict:
    window_id: str
    source_session_id: str
    family: str
    split: str
    split_manifest_sha256: str
    student_checkpoint_sha256: str
    learner_iteration: int
    admitted: bool
    training_exportable: bool
    admitted_row_count: int
    nominal_rollout_count: int
    disturbance_rollout_count: int
    quarantine_reasons: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        quarantined = not self.admitted
        return {
            "schema_version": 1,
            "kind": QUARANTINE_KIND if quarantined else SUPPORT_VERDICT_KIND,
            "window_id": self.window_id,
            "source_session_id": self.source_session_id,
            "family": self.family,
            "split": self.split,
            "split_manifest_sha256": self.split_manifest_sha256,
            "student_checkpoint_sha256": self.student_checkpoint_sha256,
            "learner_iteration": self.learner_iteration,
            "admitted": self.admitted,
            "quarantined": quarantined,
            "whole_window_quarantine": quarantined,
            "training_exportable": self.training_exportable,
            "admitted_row_count": self.admitted_row_count,
            "teacher_labels_included": False,
            "teacher_action_transform": FINAL_TEACHER_ACTION_TRANSFORM.value,
            "teacher_composite_name": TEACHER_COMPOSITE_NAME,
            "teacher_composite_formula_sha256": TEACHER_COMPOSITE_FORMULA_SHA256,
            "teacher_composite_constants_sha256": TEACHER_COMPOSITE_CONSTANTS_SHA256,
            "nominal_rollout_count": self.nominal_rollout_count,
            "disturbance_rollout_count": self.disturbance_rollout_count,
            "quarantine_reasons": list(self.quarantine_reasons),
            "actuation_permitted": False,
            "deployment_ready": False,
            "promotion_eligible": False,
        }


def assess_teacher_support_window(
    *,
    window: Mapping[str, Any],
    qualification_runs: Sequence[Mapping[str, Any]],
    global_evidence: Mapping[str, Any],
    repository_root: str | Path | None = None,
) -> TeacherSupportVerdict:
    """Return whole-window verdict; any defect admits zero of 500 rows."""

    contract = load_teacher_support_contract(repository_root)
    issues = _Issues()
    if not isinstance(window, Mapping):
        raise TypeError("window must be a mapping")
    issues.exact(window.get("schema_version"), 1, "window.schema_version")
    issues.exact(window.get("kind"), WINDOW_KIND, "window.kind")
    identity = _identity_fields(window, issues)
    tranche = _mapping(contract["split_tranche"], "split_tranche")
    if identity["family"] not in tranche["families"]:
        issues.add("window.family: unsupported family")
    if identity["split"] not in tranche["session_splits"]:
        issues.add("window.split: unsupported split")
    teacher_identity = _mapping(contract["artifact_identity"], "artifact_identity")
    issues.exact(
        window.get("teacher_identity"),
        dict(teacher_identity),
        "window.teacher_identity",
    )
    _validate_global_evidence(global_evidence, contract, issues)
    samples = _validate_source_samples(window.get("source_samples"), contract, issues)
    _validate_rows(
        window.get("rows"),
        samples,
        identity,
        teacher_identity,
        contract,
        issues,
    )
    nominal_count, disturbance_count = _validate_qualification_runs(
        qualification_runs,
        identity,
        teacher_identity,
        contract,
        issues,
    )
    admitted = not issues.values
    row_count = int(_mapping(contract["causal_window"], "causal_window")["emitted_control_rows"])
    return TeacherSupportVerdict(
        window_id=str(identity["window_id"]),
        source_session_id=str(identity["source_session_id"]),
        family=str(identity["family"]),
        split=str(identity["split"]),
        split_manifest_sha256=str(identity["split_manifest_sha256"]),
        student_checkpoint_sha256=str(identity["student_checkpoint_sha256"]),
        learner_iteration=int(identity["learner_iteration"]),
        admitted=admitted,
        training_exportable=admitted and identity["split"] == "train",
        admitted_row_count=row_count if admitted else 0,
        nominal_rollout_count=nominal_count,
        disturbance_rollout_count=disturbance_count,
        quarantine_reasons=tuple(issues.values),
    )


def build_support_admitted_training_export(
    *,
    window: Mapping[str, Any],
    qualification_runs: Sequence[Mapping[str, Any]],
    global_evidence: Mapping[str, Any],
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Revalidate and emit labels only for one fully admitted train window."""

    verdict = assess_teacher_support_window(
        window=window,
        qualification_runs=qualification_runs,
        global_evidence=global_evidence,
        repository_root=repository_root,
    )
    if not verdict.admitted:
        raise ValueError("quarantined teacher window is not training-exportable")
    if verdict.split != "train" or not verdict.training_exportable:
        raise ValueError("heldout teacher window is never training-exportable")
    output_rows: list[dict[str, Any]] = []
    contract = load_teacher_support_contract(repository_root)
    for row in window["rows"]:
        teacher = _compose_checkpoint21204_teacher_action(
            row["teacher_raw_action_hardware"],
            contract,
        )
        output_rows.append(
            {
                "row_index": row["row_index"],
                "anchor_source_frame_index": row["anchor_source_frame_index"],
                "control_source_frame_index": row["control_source_frame_index"],
                "anchor_reference_monotonic_ns": row["anchor_reference_monotonic_ns"],
                "control_monotonic_ns": row["control_monotonic_ns"],
                "encoder267": list(row["encoder267"]),
                "token64": list(row["token64"]),
                "proprio930": list(row["proprio930"]),
                "decoder994": list(row["decoder994"]),
                "teacher_raw_action_hardware": list(row["teacher_raw_action_hardware"]),
                "teacher_candidate_target_hardware": list(row["teacher_candidate_target_hardware"]),
                "teacher_action_native": teacher.teacher_action_native.tolist(),
                "teacher_applied_safe_action_native": (teacher.teacher_applied_safe_action_native.tolist()),
                "teacher_target_hardware": list(row["teacher_target_hardware"]),
                "student_mean_action_native": list(row["student_mean_action_native"]),
                "student_raw_action_native": list(row["student_raw_action_native"]),
                "student_applied_action_native": list(row["student_applied_action_native"]),
                "next_state": dict(row["next_state"]),
            }
        )
    return {
        "schema_version": 1,
        "kind": TRAINING_EXPORT_KIND,
        "support_config_sha256": SUPPORT_CONFIG_SHA256,
        "teacher_action_transform": FINAL_TEACHER_ACTION_TRANSFORM.value,
        "teacher_composite_name": TEACHER_COMPOSITE_NAME,
        "teacher_composite_formula_sha256": TEACHER_COMPOSITE_FORMULA_SHA256,
        "teacher_composite_constants_sha256": TEACHER_COMPOSITE_CONSTANTS_SHA256,
        "safe_target_constants_sha256": SAFE_TARGET_CONSTANTS_SHA256,
        "safe_target_formula_sha256": SAFE_TARGET_FORMULA_SHA256,
        "student_applied_action_transform": SAFE_TARGET_TRANSFORM_KIND,
        "student_applied_action_semantics": "applied_safe_native_action",
        "student_raw_action_strict_abs_max": SAFE_TARGET_RAW_ACTION_CLIP,
        "student_applied_link_atol": 1.0e-6,
        "window_id": verdict.window_id,
        "source_session_id": verdict.source_session_id,
        "family": verdict.family,
        "split": verdict.split,
        "split_manifest_sha256": verdict.split_manifest_sha256,
        "student_checkpoint_sha256": verdict.student_checkpoint_sha256,
        "learner_iteration": verdict.learner_iteration,
        "row_count": len(output_rows),
        "teacher_labels_included": True,
        "support_admitted": True,
        "actuation_permitted": False,
        "deployment_ready": False,
        "promotion_eligible": False,
        "rows": output_rows,
    }


@dataclass(frozen=True)
class TeacherSupportTrancheVerdict:
    qualified: bool
    split_manifest_sha256: str
    family_session_counts: Mapping[str, Mapping[str, int]]
    reasons: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": TRANCHE_VERDICT_KIND,
            "qualified": self.qualified,
            "split_manifest_sha256": self.split_manifest_sha256,
            "family_session_counts": {
                family: dict(counts) for family, counts in self.family_session_counts.items()
            },
            "reasons": list(self.reasons),
            "training_authorized": False,
            "actuation_permitted": False,
            "deployment_ready": False,
        }


def assess_teacher_support_tranche(
    verdicts: Sequence[TeacherSupportVerdict],
    *,
    repository_root: str | Path | None = None,
) -> TeacherSupportTrancheVerdict:
    """Check six-family coverage plus immutable whole-session split discipline."""

    contract = load_teacher_support_contract(repository_root)
    tranche = _mapping(contract["split_tranche"], "split_tranche")
    issues = _Issues()
    if not verdicts:
        issues.add("tranche: no window verdicts")
    if any(type(verdict) is not TeacherSupportVerdict for verdict in verdicts):
        raise TypeError("tranche accepts only TeacherSupportVerdict values")
    window_ids: set[str] = set()
    manifest_hashes: set[str] = set()
    sessions: dict[str, dict[str, Any]] = {}
    for index, verdict in enumerate(verdicts):
        if verdict.window_id in window_ids:
            issues.add(f"verdicts[{index}]: duplicate window_id")
        window_ids.add(verdict.window_id)
        manifest_hashes.add(verdict.split_manifest_sha256)
        if verdict.split == "heldout" and verdict.training_exportable:
            issues.add(f"verdicts[{index}]: heldout window marked training-exportable")
        if verdict.training_exportable != (verdict.admitted and verdict.split == "train"):
            issues.add(f"verdicts[{index}]: training-exportable invariant mismatch")
        if verdict.admitted_row_count != (500 if verdict.admitted else 0):
            issues.add(f"verdicts[{index}]: whole-window row-count invariant mismatch")
        session = sessions.setdefault(
            verdict.source_session_id,
            {
                "family": verdict.family,
                "split": verdict.split,
                "windows": 0,
                "admitted": 0,
            },
        )
        if session["family"] != verdict.family:
            issues.add(f"session {verdict.source_session_id!r}: crosses motion families")
        if session["split"] != verdict.split:
            issues.add(f"session {verdict.source_session_id!r}: leaks across train/heldout")
        session["windows"] += 1
        session["admitted"] += int(verdict.admitted)
    if len(manifest_hashes) != 1:
        issues.add("tranche: split_manifest_sha256 is not unique")
    split_manifest_sha256 = next(iter(manifest_hashes), "")
    if split_manifest_sha256 and not _is_sha256(split_manifest_sha256):
        issues.add("tranche: split_manifest_sha256 is invalid")
    counts: dict[str, dict[str, int]] = {}
    for family in tranche["families"]:
        family_sessions = {
            session_id: value for session_id, value in sessions.items() if value["family"] == family
        }
        train = {name: value for name, value in family_sessions.items() if value["split"] == "train"}
        heldout = {name: value for name, value in family_sessions.items() if value["split"] == "heldout"}
        counts[family] = {
            "total": len(family_sessions),
            "train": len(train),
            "heldout": len(heldout),
            "admitted_train": sum(value["admitted"] > 0 for value in train.values()),
        }
        if len(family_sessions) < int(tranche["minimum_independent_sessions_per_family"]):
            issues.add(f"family {family!r}: fewer than 3 independent sessions")
        if len(train) < int(tranche["minimum_train_sessions_per_family"]):
            issues.add(f"family {family!r}: fewer than 2 train sessions")
        if len(heldout) < int(tranche["minimum_heldout_sessions_per_family"]):
            issues.add(f"family {family!r}: no heldout session")
        for session_id, value in train.items():
            if value["admitted"] < int(tranche["minimum_admitted_windows_per_train_session"]):
                issues.add(f"train session {session_id!r}: no admitted 500-row window")
    unknown_families = sorted({str(value["family"]) for value in sessions.values()} - set(tranche["families"]))
    if unknown_families:
        issues.add(f"tranche: unsupported families {unknown_families!r}")
    return TeacherSupportTrancheVerdict(
        qualified=not issues.values,
        split_manifest_sha256=split_manifest_sha256,
        family_session_counts=counts,
        reasons=tuple(issues.values),
    )


assert SONIC_HARDWARE_DEFAULT_Q.shape == SONIC_HARDWARE_ACTION_SCALE.shape == (TARGET_DOF,)
assert sorted(ISAACLAB_TO_MUJOCO_DOF) == sorted(MUJOCO_TO_ISAACLAB_DOF) == list(range(TARGET_DOF))
