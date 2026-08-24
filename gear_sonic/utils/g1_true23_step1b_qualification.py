"""Fail-closed qualification for paired idle-only Step 1B simulation evidence.

This module does not launch a simulator. It validates immutable campaign inputs,
re-opens every paired trace, and recomputes all gates. A passing result authorizes
only continued idle expert evaluation; it never authorizes DAgger, training, or
deployment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

CONTRACT_SCHEMA = "g1_true23_idle_step1b_qualification_contract_v1"
CAMPAIGN_SCHEMA = "g1_true23_idle_step1b_campaign_manifest_v1"
REPORT_SCHEMA = "g1_true23_idle_step1b_evidence_report_v1"
TRACE_SCHEMA = "g1_true23_idle_step1b_trace_v1"
RAW_AUDIT_SCHEMA = "g1_true23_idle_step1b_raw_trace_v1"
INITIAL_STATE_SCHEMA = "g1_true23_idle_step1b_initial_state_v1"
DISTURBANCE_SCHEMA = "g1_true23_idle_step1b_disturbance_schedule_v1"
REFERENCE_SCHEDULE_SCHEMA = "g1_true23_idle_step1b_reference_schedule_v1"
RUNTIME_CONFIG_SCHEMA = "g1_true23_idle_step1b_runtime_config_v1"
INPUT_ACTION_LAW_SCHEMA = "g1_true23_idle_step1b_input_action_law_v1"
QUALIFICATION_SCHEMA = "g1_true23_idle_step1b_qualification_v1"

PASS_AUTHORIZATION = "step_1b_idle_expert_only_not_training"
STEP1A_AUTHORIZATION = "step_1b_fixed_horizon_expert_collection_only"
FROZEN_CLIP_IDS = (
    "idle__220721__hands_on_back_loop_a036m",
    "idle__220713__change_idle_left_a021",
)
FROZEN_SOURCE_CSVS = (
    "220721/idle_hands_on_back_loop_001__A036_M.csv",
    "220713/change_idle_to_idle_left_003__A021.csv",
)
FROZEN_FRAME_COUNTS = (547, 362)
REFERENCE_CONTINUATION = "terminal_hold_last"
REQUIRED_HORIZON_STEPS = 500
REQUIRED_CONTROL_HZ = 50
REQUIRED_EPISODES_PER_CLIP = 1
MIN_TIMEOUT_FRACTION_FLOOR = 0.95
SHIPPED_EE_TERMINATION_THRESHOLD_M = 0.25
POST_TERMINATION_POLICY = "physics_no_reset_v1"
TEACHER_CONTROLLER_SEMANTICS = "neural_state_feedback_29dof_low_latency_policy"
TRUE23_CONTROLLER_SEMANTICS = "schema_v6_action_target_native_with_live_joint_pd_v1"
FORBIDDEN_CONTROLLER_SEMANTICS = {"source_trajectory_pd_replay"}
TRUE23_CONTROLLER_DESCRIPTOR = {
    "action_law": TRUE23_CONTROLLER_SEMANTICS,
    "state_inputs": ["joint_position", "joint_velocity"],
    "reference_input": "schema_v6_action_target_native",
    "feedback_rate": "every_physics_substep",
    "torque_law": "clip(kp*(q_target-q)-kd*dq,effort_limits)",
    "output_space": "joint_torque",
    "online_task_space_expert": False,
}
FIXED_REFERENCE_START_FRAME = 0
FIXED_SEEDS_BY_CLIP = {
    FROZEN_CLIP_IDS[0]: (1729,),
    FROZEN_CLIP_IDS[1]: (2729,),
}
FIXED_DISTURBANCE_DELTA = (0.0,) * 6
MODEL_29_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
MODEL_29_JOINT_LOWER = (
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.5307,
    -2.9671,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.618,
    -0.52,
    -0.52,
    -3.0892,
    -1.5882,
    -2.618,
    -1.0472,
    -1.97222,
    -1.61443,
    -1.61443,
    -3.0892,
    -2.2515,
    -2.618,
    -1.0472,
    -1.97222,
    -1.61443,
    -1.61443,
)
MODEL_29_JOINT_UPPER = (
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.8798,
    0.5236,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.618,
    0.52,
    0.52,
    2.6704,
    2.2515,
    2.618,
    2.0944,
    1.97222,
    1.61443,
    1.61443,
    2.6704,
    1.5882,
    2.618,
    2.0944,
    1.97222,
    1.61443,
    1.61443,
)
_TRUE23_OMITTED_JOINTS = {
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}
MODEL_23_JOINT_NAMES = tuple(name for name in MODEL_29_JOINT_NAMES if name not in _TRUE23_OMITTED_JOINTS)
MODEL_23_JOINT_LOWER = tuple(
    limit
    for name, limit in zip(MODEL_29_JOINT_NAMES, MODEL_29_JOINT_LOWER, strict=True)
    if name not in _TRUE23_OMITTED_JOINTS
)
MODEL_23_JOINT_UPPER = tuple(
    limit
    for name, limit in zip(MODEL_29_JOINT_NAMES, MODEL_29_JOINT_UPPER, strict=True)
    if name not in _TRUE23_OMITTED_JOINTS
)
MODEL_JOINT_SPECS = {
    "teacher_reference": (MODEL_29_JOINT_NAMES, MODEL_29_JOINT_LOWER, MODEL_29_JOINT_UPPER),
    "true23_expert": (MODEL_23_JOINT_NAMES, MODEL_23_JOINT_LOWER, MODEL_23_JOINT_UPPER),
}
INITIAL_STATE_SCOPES = {
    "common": "common_root_reference_projection",
    "teacher_reference": "teacher_reference_full_state",
    "true23_expert": "true23_expert_full_state",
}
STANCE_METRIC_DEFINITION = {
    "position_frame": "world",
    "orientation_metric": "quaternion_geodesic_abs_dot",
    "stance_mask": "teacher_reference_contact_true",
    "contact_mismatch": "foot_contact_state_xor_all_steps",
}
REQUIRED_STANCE_THRESHOLDS = {
    "position_error_max_m": 0.005,
    "orientation_error_max_rad": 0.005,
    "contact_mismatch_fraction_max": 0.0,
}
SELECTED_TERMINATION_GATE = {
    "stack": "unitree_rl_mjlab_g1_23dof_tracking",
    "config_relpaths": [
        "artifacts/external/unitree_rl_mjlab/src/tasks/tracking/tracking_env_cfg.py",
        "artifacts/external/unitree_rl_mjlab/src/tasks/tracking/mdp/terminations.py",
        "artifacts/external/unitree_rl_mjlab/src/tasks/tracking/config/g1_23dof/env_cfgs.py",
    ],
    "config_sha256": {
        "artifacts/external/unitree_rl_mjlab/src/tasks/tracking/tracking_env_cfg.py": (
            "33f058321a7f8a0e66dbdae3d3a9cd3ea87abd7bf8eecee54daa3c6bf541a0cb"
        ),
        "artifacts/external/unitree_rl_mjlab/src/tasks/tracking/mdp/terminations.py": (
            "fbb26adf35e8cb3af9e465615f110fc09f297d5859865603279695f16026f443"
        ),
        "artifacts/external/unitree_rl_mjlab/src/tasks/tracking/config/g1_23dof/env_cfgs.py": (
            "58ba2ee91857467913be13d40469845294b6cb3260f5c13d4d6124a6808aefc3"
        ),
    },
    "terms": {
        "time_out": {"function": "mdp.time_out", "time_out": True},
        "anchor_pos": {
            "function": "mdp.bad_anchor_pos_z_only",
            "threshold": 0.25,
        },
        "anchor_ori": {
            "function": "mdp.bad_anchor_ori",
            "projected_gravity_z_threshold": 0.8,
        },
        "ee_body_pos": {
            "function": "mdp.bad_motion_body_pos_z_only",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_roll_rubber_hand",
                "right_wrist_roll_rubber_hand",
            ],
        },
    },
}
REQUIRED_CONTACT_LINKS = ("left_foot", "right_foot")
REQUIRED_SEMANTIC_POINTS = (
    "left_foot",
    "right_foot",
    "left_hand",
    "right_hand",
    "head_proxy",
)
REQUIRED_SEMANTIC_ORIENTATIONS = ("left_foot", "right_foot")
REQUIRED_LINK_VELOCITIES = (
    "pelvis",
    "left_foot",
    "right_foot",
    "left_hand",
    "right_hand",
    "head_proxy",
)
REQUIRED_SEMANTIC_BINDINGS = {
    "left_foot": {
        "source_body": "left_ankle_roll_link",
        "target_body": "left_ankle_roll_link",
        "local_offset_m": [0.0, 0.0, 0.0],
    },
    "right_foot": {
        "source_body": "right_ankle_roll_link",
        "target_body": "right_ankle_roll_link",
        "local_offset_m": [0.0, 0.0, 0.0],
    },
    "left_hand": {
        "source_body": "left_wrist_yaw_link",
        "target_body": "left_wrist_roll_rubber_hand",
        "local_offset_m": [0.18, -0.025, 0.0],
    },
    "right_hand": {
        "source_body": "right_wrist_yaw_link",
        "target_body": "right_wrist_roll_rubber_hand",
        "local_offset_m": [0.18, 0.025, 0.0],
    },
    "head_proxy": {
        "source_body": "torso_link",
        "target_body": "torso_link",
        "local_offset_m": [0.0, 0.0, 0.35],
    },
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "sim_validation"
    / "g1_true23_idle_step1b_qualification_v1.json"
)
# Updated only when reviewed checked-in contract semantics intentionally change.
PINNED_CONTRACT_PAYLOAD_SHA256 = "a419bdf14959f290b1c6d6a354c5e80caad941e986427cc0a404451813caadce"
_METRIC_EPSILON = 1.0e-12


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes, rejecting non-finite values."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> Mapping[str, Any]:
    """Load an object-root JSON file while rejecting duplicate/non-finite values."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON evidence file {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON evidence root must be an object: {path}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _array(value: object, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be an array")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be boolean")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be integer >= {minimum}")
    return value


def _number(
    value: object,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite numeric")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{context} must be <= {maximum}")
    return result


def _optional_threshold(value: object, context: str) -> float | None:
    if value is None:
        return None
    return _number(value, context, minimum=0.0)


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def _optional_sha256(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, context)


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context)


def _vector(value: object, length: int, context: str) -> tuple[float, ...]:
    values = _array(value, context)
    if len(values) != length:
        raise ValueError(f"{context} must contain exactly {length} values")
    return tuple(_number(item, f"{context}[{index}]") for index, item in enumerate(values))


def _quaternion(value: object, context: str) -> tuple[float, float, float, float]:
    result = _vector(value, 4, context)
    norm = math.sqrt(sum(component * component for component in result))
    if abs(norm - 1.0) > 1.0e-5:
        raise ValueError(f"{context} must be a normalized xyzw quaternion")
    return result  # type: ignore[return-value]


def _name_array(value: object, context: str) -> tuple[str, ...]:
    values = tuple(_string(item, f"{context}[{index}]") for index, item in enumerate(_array(value, context)))
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{context} must contain unique non-empty names")
    return values


def _normalized_relative_path(value: object, context: str) -> PurePosixPath:
    text = _string(value, context)
    windows_path = PureWindowsPath(text)
    if "\\" in text or ":" in text or windows_path.drive or windows_path.root:
        raise ValueError(f"{context} must be a normalized POSIX relative path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path == PurePosixPath(".")
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise ValueError(f"{context} must be a normalized POSIX relative path")
    return path


def _contained_file(root: Path, value: object, context: str) -> Path:
    if root.is_symlink():
        raise ValueError(f"{context} evidence root may not be a symlink")
    root = root.resolve()
    relpath = _normalized_relative_path(value, context)
    candidate = root
    for part in relpath.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"{context} may not traverse symlinks")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{context} escapes evidence root") from exc
    if not resolved.is_file():
        raise ValueError(f"{context} file is missing")
    return resolved


def _validate_binary_ref(value: object, root: Path, context: str) -> dict[str, Any]:
    record = _mapping(value, context)
    _exact_keys(record, {"file", "sha256", "size_bytes"}, context)
    path = _contained_file(root, record["file"], f"{context}.file")
    expected_size = _integer(record["size_bytes"], f"{context}.size_bytes", minimum=1)
    if path.stat().st_size != expected_size:
        raise ValueError(f"{context} size mismatch")
    expected_hash = _sha256(record["sha256"], f"{context}.sha256")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{context} SHA-256 mismatch")
    return {"file": record["file"], "sha256": expected_hash, "size_bytes": expected_size}


def _validate_json_ref(
    value: object,
    root: Path,
    context: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    record = _mapping(value, context)
    _exact_keys(record, {"file", "sha256", "payload_sha256"}, context)
    path = _contained_file(root, record["file"], f"{context}.file")
    raw_hash = _sha256(record["sha256"], f"{context}.sha256")
    if sha256_file(path) != raw_hash:
        raise ValueError(f"{context} SHA-256 mismatch")
    payload = load_strict_json(path)
    payload_hash = _sha256(record["payload_sha256"], f"{context}.payload_sha256")
    if sha256_bytes(canonical_json_bytes(payload)) != payload_hash:
        raise ValueError(f"{context} payload SHA-256 mismatch")
    normalized = {
        "file": record["file"],
        "sha256": raw_hash,
        "payload_sha256": payload_hash,
    }
    return normalized, payload


def _validate_repository_binding(value: object, context: str) -> dict[str, Any]:
    record = _mapping(value, context)
    _exact_keys(record, {"relpath", "sha256", "size_bytes"}, context)
    relpath = _normalized_relative_path(record["relpath"], f"{context}.relpath")
    path = _contained_file(_REPOSITORY_ROOT, relpath.as_posix(), f"{context}.relpath")
    size = _integer(record["size_bytes"], f"{context}.size_bytes", minimum=1)
    digest = _sha256(record["sha256"], f"{context}.sha256")
    if path.stat().st_size != size:
        raise ValueError(f"{context} repository file size mismatch")
    if sha256_file(path) != digest:
        raise ValueError(f"{context} repository file SHA-256 mismatch")
    return {"relpath": relpath.as_posix(), "sha256": digest, "size_bytes": size}


def _validate_trusted_identity_bindings(value: object) -> dict[str, Any]:
    context = "Step1B qualification contract.trusted_identity_bindings"
    bindings = _mapping(value, context)
    _exact_keys(
        bindings,
        {"step1a", "teacher_reference", "true23_expert", "runtime"},
        context,
    )
    step1a = _mapping(bindings["step1a"], f"{context}.step1a")
    step1a_keys = {
        "support_manifest_sha256",
        "support_manifest_payload_sha256",
        "qualification_sha256",
        "qualification_payload_sha256",
        "batch_sha256",
        "batch_payload_sha256",
    }
    _exact_keys(step1a, step1a_keys, f"{context}.step1a")
    normalized_step1a = {
        key: _optional_sha256(step1a[key], f"{context}.step1a.{key}") for key in sorted(step1a_keys)
    }

    teacher = _mapping(bindings["teacher_reference"], f"{context}.teacher_reference")
    teacher_keys = {
        "model_sha256",
        "checkpoint_sha256",
        "controller_config_payload_sha256",
        "input_action_law_payload_sha256",
        "semantic_reference_payload_sha256",
    }
    _exact_keys(teacher, teacher_keys, f"{context}.teacher_reference")
    normalized_teacher = {
        key: _optional_sha256(teacher[key], f"{context}.teacher_reference.{key}") for key in sorted(teacher_keys)
    }

    expert = _mapping(bindings["true23_expert"], f"{context}.true23_expert")
    _exact_keys(
        expert,
        {
            "target_model_sha256",
            "controller_sha256",
            "controller_config_payload_sha256",
            "input_action_law_payload_sha256",
            "action_law",
            "clips",
        },
        f"{context}.true23_expert",
    )
    clip_bindings = _mapping(expert["clips"], f"{context}.true23_expert.clips")
    if set(clip_bindings) != set(FROZEN_CLIP_IDS):
        raise ValueError("trusted true23 clip identity set differs from frozen support")
    normalized_clip_bindings: dict[str, Any] = {}
    for clip_id in FROZEN_CLIP_IDS:
        clip = _mapping(clip_bindings[clip_id], f"{context}.true23_expert.clips.{clip_id}")
        keys = {"motion_sha256", "expert_sha256", "report_payload_sha256"}
        _exact_keys(clip, keys, f"{context}.true23_expert.clips.{clip_id}")
        normalized_clip_bindings[clip_id] = {
            key: _optional_sha256(clip[key], f"{context}.true23_expert.clips.{clip_id}.{key}")
            for key in sorted(keys)
        }
    normalized_expert = {
        "target_model_sha256": _optional_sha256(
            expert["target_model_sha256"], f"{context}.true23_expert.target_model_sha256"
        ),
        "controller_sha256": _optional_sha256(
            expert["controller_sha256"], f"{context}.true23_expert.controller_sha256"
        ),
        "controller_config_payload_sha256": _optional_sha256(
            expert["controller_config_payload_sha256"],
            f"{context}.true23_expert.controller_config_payload_sha256",
        ),
        "input_action_law_payload_sha256": _optional_sha256(
            expert["input_action_law_payload_sha256"],
            f"{context}.true23_expert.input_action_law_payload_sha256",
        ),
        "action_law": _optional_string(expert["action_law"], f"{context}.true23_expert.action_law"),
        "clips": normalized_clip_bindings,
    }
    if normalized_expert["action_law"] != TRUE23_CONTROLLER_SEMANTICS:
        raise ValueError("trusted true23 action law differs from the approved live-feedback controller")

    runtime = _mapping(bindings["runtime"], f"{context}.runtime")
    _exact_keys(
        runtime,
        {
            "simulator_name",
            "simulator_version",
            "runner_sha256",
            "runtime_config_payload_sha256",
            "robot_assets",
        },
        f"{context}.runtime",
    )
    robot_assets = _mapping(runtime["robot_assets"], f"{context}.runtime.robot_assets")
    if not robot_assets:
        raise ValueError("trusted runtime robot_assets must be non-empty")
    normalized_robot_assets = {
        _string(role, f"{context}.runtime.robot_assets role"): _optional_sha256(
            digest, f"{context}.runtime.robot_assets.{role}"
        )
        for role, digest in robot_assets.items()
    }
    return {
        "step1a": normalized_step1a,
        "teacher_reference": normalized_teacher,
        "true23_expert": normalized_expert,
        "runtime": {
            "simulator_name": _optional_string(runtime["simulator_name"], f"{context}.runtime.simulator_name"),
            "simulator_version": _optional_string(
                runtime["simulator_version"], f"{context}.runtime.simulator_version"
            ),
            "runner_sha256": _optional_sha256(runtime["runner_sha256"], f"{context}.runtime.runner_sha256"),
            "runtime_config_payload_sha256": _optional_sha256(
                runtime["runtime_config_payload_sha256"],
                f"{context}.runtime.runtime_config_payload_sha256",
            ),
            "robot_assets": normalized_robot_assets,
        },
    }


def _missing_trusted_bindings(bindings: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []

    def visit(value: object, path: str) -> None:
        if value is None:
            missing.append(path)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}")

    visit(bindings, "trusted_identity_bindings")
    return missing


def _validate_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    context = "Step1B qualification contract"
    payload_sha256 = sha256_bytes(canonical_json_bytes(config))
    if payload_sha256 != PINNED_CONTRACT_PAYLOAD_SHA256:
        raise ValueError("Step1B qualification contract payload is not the pinned contract")
    _exact_keys(
        config,
        {
            "schema",
            "schema_version",
            "declared_categories",
            "frozen_clips",
            "control_hz",
            "horizon_steps",
            "episodes_per_clip",
            "robustness_coverage",
            "min_timeout_fraction",
            "post_termination_policy",
            "controller_semantics",
            "selected_termination_gate",
            "campaign_schedule",
            "stance_foot_thresholds",
            "stance_foot_metric_definition",
            "semantic_point_bindings",
            "required_contact_links",
            "required_semantic_points",
            "required_semantic_orientations",
            "required_link_velocities",
            "diagnostic_thresholds",
            "trusted_identity_bindings",
        },
        context,
    )
    if config["schema"] != CONTRACT_SCHEMA or config["schema_version"] != 1:
        raise ValueError("unsupported Step1B qualification contract")
    if list(_array(config["declared_categories"], f"{context}.declared_categories")) != ["idle"]:
        raise ValueError("Step1B contract must declare only idle")

    clips = _array(config["frozen_clips"], f"{context}.frozen_clips")
    if len(clips) != len(FROZEN_CLIP_IDS):
        raise ValueError("Step1B contract must contain exactly two frozen idle clips")
    normalized_clips: list[dict[str, Any]] = []
    for index, value in enumerate(clips):
        clip = _mapping(value, f"{context}.frozen_clips[{index}]")
        _exact_keys(
            clip,
            {
                "clip_id",
                "source_csv",
                "source_frame_count",
                "reference_continuation",
            },
            f"{context}.frozen_clips[{index}]",
        )
        normalized_clips.append(
            {
                "clip_id": _string(clip["clip_id"], f"{context}.frozen_clips[{index}].clip_id"),
                "source_csv": _normalized_relative_path(
                    clip["source_csv"], f"{context}.frozen_clips[{index}].source_csv"
                ).as_posix(),
                "source_frame_count": _integer(
                    clip["source_frame_count"],
                    f"{context}.frozen_clips[{index}].source_frame_count",
                    minimum=1,
                ),
                "reference_continuation": _string(
                    clip["reference_continuation"],
                    f"{context}.frozen_clips[{index}].reference_continuation",
                ),
            }
        )
    if tuple(clip["clip_id"] for clip in normalized_clips) != FROZEN_CLIP_IDS:
        raise ValueError("Step1B contract frozen clip IDs/order differ from idle Step1A manifest")
    if tuple(clip["source_csv"] for clip in normalized_clips) != FROZEN_SOURCE_CSVS:
        raise ValueError("Step1B contract frozen source paths differ from idle Step1A manifest")
    if tuple(clip["source_frame_count"] for clip in normalized_clips) != FROZEN_FRAME_COUNTS:
        raise ValueError("Step1B contract frozen frame counts differ from idle Step1A evidence")
    if any(clip["reference_continuation"] != REFERENCE_CONTINUATION for clip in normalized_clips):
        raise ValueError("Step1B references must use terminal_hold_last; loop seams are unauthorized")

    control_hz = _integer(config["control_hz"], f"{context}.control_hz", minimum=1)
    if control_hz != REQUIRED_CONTROL_HZ:
        raise ValueError(f"Step1B control_hz must be exactly {REQUIRED_CONTROL_HZ}")
    horizon = _integer(config["horizon_steps"], f"{context}.horizon_steps", minimum=1)
    if horizon != REQUIRED_HORIZON_STEPS:
        raise ValueError(f"Step1B horizon must be exactly {REQUIRED_HORIZON_STEPS} steps")
    episodes_per_clip = _integer(config["episodes_per_clip"], f"{context}.episodes_per_clip", minimum=1)
    if episodes_per_clip != REQUIRED_EPISODES_PER_CLIP:
        raise ValueError(f"Step1B episodes_per_clip must be exactly {REQUIRED_EPISODES_PER_CLIP}")
    robustness_coverage = _string(config["robustness_coverage"], f"{context}.robustness_coverage")
    if robustness_coverage != "nominal_only_one_episode_per_clip_robustness_untested":
        raise ValueError("Step1B campaign must disclose that robustness is untested")
    timeout_min = _number(
        config["min_timeout_fraction"],
        f"{context}.min_timeout_fraction",
        minimum=MIN_TIMEOUT_FRACTION_FLOOR,
        maximum=1.0,
    )
    if timeout_min != MIN_TIMEOUT_FRACTION_FLOOR:
        raise ValueError("Step1B min_timeout_fraction must be exactly 0.95")
    if config["post_termination_policy"] != POST_TERMINATION_POLICY:
        raise ValueError(f"Step1B post_termination_policy must be {POST_TERMINATION_POLICY!r}")
    controller_semantics = _mapping(config["controller_semantics"], f"{context}.controller_semantics")
    _exact_keys(
        controller_semantics,
        {"teacher_reference", "true23_expert"},
        f"{context}.controller_semantics",
    )
    if controller_semantics["teacher_reference"] != TEACHER_CONTROLLER_SEMANTICS:
        raise ValueError("Step1B teacher must be exact neural state-feedback 29-DoF low-latency policy")
    true23_semantics = controller_semantics["true23_expert"]
    if true23_semantics != TRUE23_CONTROLLER_SEMANTICS:
        raise ValueError(f"Step1B true23 controller must be exactly {TRUE23_CONTROLLER_SEMANTICS!r}")

    selected_gate = _mapping(config["selected_termination_gate"], f"{context}.selected_termination_gate")
    if dict(selected_gate) != SELECTED_TERMINATION_GATE:
        raise ValueError("Step1B selected termination gate differs from exact true23 MJLab tracking semantics")

    campaign_schedule = _mapping(config["campaign_schedule"], f"{context}.campaign_schedule")
    _exact_keys(
        campaign_schedule,
        {"reference_start_frame", "disturbance_delta", "seeds_by_clip"},
        f"{context}.campaign_schedule",
    )
    if campaign_schedule["reference_start_frame"] != FIXED_REFERENCE_START_FRAME:
        raise ValueError("Step1B reference start frame must be frozen at zero")
    disturbance_delta = _vector(
        campaign_schedule["disturbance_delta"],
        6,
        f"{context}.campaign_schedule.disturbance_delta",
    )
    if disturbance_delta != FIXED_DISTURBANCE_DELTA:
        raise ValueError("Step1B disturbance schedule differs from frozen nominal campaign")
    raw_seeds = _mapping(campaign_schedule["seeds_by_clip"], f"{context}.campaign_schedule.seeds_by_clip")
    if set(raw_seeds) != set(FROZEN_CLIP_IDS):
        raise ValueError("Step1B seed schedule clip set differs from frozen support")
    seeds_by_clip: dict[str, list[int]] = {}
    for clip_id in FROZEN_CLIP_IDS:
        seeds = [
            _integer(seed, f"{context}.campaign_schedule.seeds_by_clip.{clip_id}[{index}]")
            for index, seed in enumerate(
                _array(raw_seeds[clip_id], f"{context}.campaign_schedule.seeds_by_clip.{clip_id}")
            )
        ]
        expected_seeds = list(FIXED_SEEDS_BY_CLIP[clip_id][:episodes_per_clip])
        if seeds != expected_seeds:
            raise ValueError("Step1B exact seed schedule differs from frozen campaign")
        seeds_by_clip[clip_id] = seeds

    stance = _mapping(config["stance_foot_thresholds"], f"{context}.stance_foot_thresholds")
    _exact_keys(
        stance,
        {
            "position_error_max_m",
            "orientation_error_max_rad",
            "contact_mismatch_fraction_max",
        },
        f"{context}.stance_foot_thresholds",
    )
    stance_thresholds = {
        "position_error_max_m": _number(
            stance["position_error_max_m"],
            f"{context}.stance_foot_thresholds.position_error_max_m",
            minimum=0.0,
        ),
        "orientation_error_max_rad": _number(
            stance["orientation_error_max_rad"],
            f"{context}.stance_foot_thresholds.orientation_error_max_rad",
            minimum=0.0,
        ),
        "contact_mismatch_fraction_max": _number(
            stance["contact_mismatch_fraction_max"],
            f"{context}.stance_foot_thresholds.contact_mismatch_fraction_max",
            minimum=0.0,
            maximum=1.0,
        ),
    }
    if stance_thresholds != REQUIRED_STANCE_THRESHOLDS:
        raise ValueError("Step1B stance-foot thresholds differ from fixed contract")
    metric_definition = _mapping(
        config["stance_foot_metric_definition"],
        f"{context}.stance_foot_metric_definition",
    )
    _exact_keys(
        metric_definition,
        set(STANCE_METRIC_DEFINITION),
        f"{context}.stance_foot_metric_definition",
    )
    if dict(metric_definition) != STANCE_METRIC_DEFINITION:
        raise ValueError("Step1B stance-foot metric definition differs from approved contract")

    semantic_bindings = _mapping(config["semantic_point_bindings"], f"{context}.semantic_point_bindings")
    expected_semantic_names = {
        "left_foot",
        "right_foot",
        "left_hand",
        "right_hand",
        "head_proxy",
    }
    _exact_keys(
        semantic_bindings,
        expected_semantic_names,
        f"{context}.semantic_point_bindings",
    )
    normalized_bindings: dict[str, Any] = {}
    for name in sorted(expected_semantic_names):
        binding = _mapping(semantic_bindings[name], f"{context}.semantic_point_bindings.{name}")
        _exact_keys(
            binding,
            {"source_body", "target_body", "local_offset_m"},
            f"{context}.semantic_point_bindings.{name}",
        )
        normalized_bindings[name] = {
            "source_body": _string(
                binding["source_body"],
                f"{context}.semantic_point_bindings.{name}.source_body",
            ),
            "target_body": _string(
                binding["target_body"],
                f"{context}.semantic_point_bindings.{name}.target_body",
            ),
            "local_offset_m": list(
                _vector(
                    binding["local_offset_m"],
                    3,
                    f"{context}.semantic_point_bindings.{name}.local_offset_m",
                )
            ),
        }
    if normalized_bindings != REQUIRED_SEMANTIC_BINDINGS:
        raise ValueError("Step1B semantic point bindings differ from schema-v6 task semantics")

    contact_links = _name_array(config["required_contact_links"], f"{context}.required_contact_links")
    semantic_points = _name_array(config["required_semantic_points"], f"{context}.required_semantic_points")
    semantic_orientations = _name_array(
        config["required_semantic_orientations"], f"{context}.required_semantic_orientations"
    )
    link_velocities = _name_array(config["required_link_velocities"], f"{context}.required_link_velocities")
    if contact_links != REQUIRED_CONTACT_LINKS:
        raise ValueError("Step1B required_contact_links differ from fixed contract")
    if semantic_points != REQUIRED_SEMANTIC_POINTS:
        raise ValueError("Step1B required_semantic_points differ from fixed contract")
    if semantic_orientations != REQUIRED_SEMANTIC_ORIENTATIONS:
        raise ValueError("Step1B required_semantic_orientations differ from fixed contract")
    if link_velocities != REQUIRED_LINK_VELOCITIES:
        raise ValueError("Step1B required_link_velocities differ from fixed contract")
    for required in ("left_foot", "right_foot"):
        if (
            required not in contact_links
            or required not in semantic_points
            or required not in semantic_orientations
        ):
            raise ValueError(f"Step1B contract lacks required stance foot {required!r}")
    for required in ("left_hand", "right_hand", "head_proxy"):
        if required not in semantic_points:
            raise ValueError(f"Step1B contract lacks diagnostic semantic point {required!r}")
    if "pelvis" not in link_velocities:
        raise ValueError("Step1B contract lacks pelvis link velocity")

    diagnostics = _mapping(config["diagnostic_thresholds"], f"{context}.diagnostic_thresholds")
    _exact_keys(
        diagnostics,
        {
            "pelvis_position_error_max_m",
            "pelvis_orientation_error_max_rad",
            "com_position_error_max_m",
            "semantic_position_error_max_m",
            "link_linear_velocity_error_max_mps",
            "link_angular_velocity_error_max_radps",
        },
        f"{context}.diagnostic_thresholds",
    )
    semantic_diagnostics = _mapping(
        diagnostics["semantic_position_error_max_m"],
        f"{context}.diagnostic_thresholds.semantic_position_error_max_m",
    )
    _exact_keys(
        semantic_diagnostics,
        {"left_hand", "right_hand", "head_proxy"},
        f"{context}.diagnostic_thresholds.semantic_position_error_max_m",
    )
    diagnostic_thresholds = {
        "pelvis_position_error_max_m": _optional_threshold(
            diagnostics["pelvis_position_error_max_m"],
            f"{context}.diagnostic_thresholds.pelvis_position_error_max_m",
        ),
        "pelvis_orientation_error_max_rad": _optional_threshold(
            diagnostics["pelvis_orientation_error_max_rad"],
            f"{context}.diagnostic_thresholds.pelvis_orientation_error_max_rad",
        ),
        "com_position_error_max_m": _optional_threshold(
            diagnostics["com_position_error_max_m"],
            f"{context}.diagnostic_thresholds.com_position_error_max_m",
        ),
        "semantic_position_error_max_m": {
            name: _optional_threshold(
                semantic_diagnostics[name],
                f"{context}.diagnostic_thresholds.semantic_position_error_max_m.{name}",
            )
            for name in ("left_hand", "right_hand", "head_proxy")
        },
        "link_linear_velocity_error_max_mps": _optional_threshold(
            diagnostics["link_linear_velocity_error_max_mps"],
            f"{context}.diagnostic_thresholds.link_linear_velocity_error_max_mps",
        ),
        "link_angular_velocity_error_max_radps": _optional_threshold(
            diagnostics["link_angular_velocity_error_max_radps"],
            f"{context}.diagnostic_thresholds.link_angular_velocity_error_max_radps",
        ),
    }
    trusted_bindings = _validate_trusted_identity_bindings(config["trusted_identity_bindings"])
    return {
        "declared_categories": ["idle"],
        "frozen_clips": normalized_clips,
        "control_hz": control_hz,
        "horizon_steps": horizon,
        "episodes_per_clip": episodes_per_clip,
        "robustness_coverage": robustness_coverage,
        "min_timeout_fraction": timeout_min,
        "post_termination_policy": POST_TERMINATION_POLICY,
        "controller_semantics": {
            "teacher_reference": TEACHER_CONTROLLER_SEMANTICS,
            "true23_expert": TRUE23_CONTROLLER_SEMANTICS,
        },
        "selected_termination_gate": dict(SELECTED_TERMINATION_GATE),
        "campaign_schedule": {
            "reference_start_frame": FIXED_REFERENCE_START_FRAME,
            "disturbance_delta": list(FIXED_DISTURBANCE_DELTA),
            "seeds_by_clip": seeds_by_clip,
        },
        "stance_foot_thresholds": stance_thresholds,
        "stance_foot_metric_definition": dict(STANCE_METRIC_DEFINITION),
        "semantic_point_bindings": normalized_bindings,
        "required_contact_links": contact_links,
        "required_semantic_points": semantic_points,
        "required_semantic_orientations": semantic_orientations,
        "required_link_velocities": link_velocities,
        "diagnostic_thresholds": diagnostic_thresholds,
        "trusted_identity_bindings": trusted_bindings,
        "missing_trusted_identity_bindings": _missing_trusted_bindings(trusted_bindings),
    }


def _validate_step1a(
    value: object,
    root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    record = _mapping(value, "campaign.step1a")
    _exact_keys(record, {"support_manifest", "qualification", "batch"}, "campaign.step1a")
    support_ref, support = _validate_json_ref(record["support_manifest"], root, "campaign.step1a.support_manifest")
    qualification_ref, qualification = _validate_json_ref(
        record["qualification"], root, "campaign.step1a.qualification"
    )
    batch_ref, batch = _validate_json_ref(record["batch"], root, "campaign.step1a.batch")

    trusted = contract["trusted_identity_bindings"]["step1a"]
    expected_refs = {
        "support_manifest": (
            trusted["support_manifest_sha256"],
            trusted["support_manifest_payload_sha256"],
        ),
        "qualification": (
            trusted["qualification_sha256"],
            trusted["qualification_payload_sha256"],
        ),
        "batch": (trusted["batch_sha256"], trusted["batch_payload_sha256"]),
    }
    actual_refs = {
        "support_manifest": support_ref,
        "qualification": qualification_ref,
        "batch": batch_ref,
    }
    for name, (raw_hash, payload_hash) in expected_refs.items():
        if actual_refs[name]["sha256"] != raw_hash or actual_refs[name]["payload_sha256"] != payload_hash:
            raise ValueError(f"Step1A {name} differs from trusted pinned identity")

    if support.get("qualification_categories") != ["idle"]:
        raise ValueError("Step1A support manifest must declare only idle")
    support_clips = _array(support.get("clips"), "Step1A support manifest.clips")
    expected_clips = contract["frozen_clips"]
    if len(support_clips) != len(expected_clips):
        raise ValueError("Step1A support manifest must contain exactly two clips")
    for index, expected in enumerate(expected_clips):
        clip = _mapping(support_clips[index], f"Step1A support manifest.clips[{index}]")
        if clip.get("clip_id") != expected["clip_id"] or clip.get("category") != "idle":
            raise ValueError("Step1A support manifest clip identity/category mismatch")
        if clip.get("source_csv") != expected["source_csv"]:
            raise ValueError("Step1A support manifest source path mismatch")

    required_qualification = {
        "deployment_ready": False,
        "expert_gate_passed": False,
        "authorization": STEP1A_AUTHORIZATION,
        "step_1b_authorized": True,
        "qualification_gate_passed": True,
        "declared_categories": ["idle"],
        "unique_requested_clip_count": len(FROZEN_CLIP_IDS),
        "hard_violation_count": 0,
        "requested_state_passed": True,
    }
    for key, expected in required_qualification.items():
        if qualification.get(key) != expected:
            raise ValueError(f"Step1A qualification.{key} does not authorize idle Step1B")
    qualification_clips = _mapping(qualification.get("clips"), "Step1A qualification.clips")
    if set(qualification_clips) != set(FROZEN_CLIP_IDS):
        raise ValueError("Step1A qualification clip set differs from frozen idle support")
    categories = _mapping(qualification.get("categories"), "Step1A qualification.categories")
    idle_category = _mapping(categories.get("idle"), "Step1A qualification.categories.idle")
    if idle_category.get("gate_passed") is not True:
        raise ValueError("Step1A idle category gate did not pass")
    for clip_id in FROZEN_CLIP_IDS:
        clip_result = _mapping(qualification_clips[clip_id], f"Step1A qualification.clips.{clip_id}")
        for key in ("hard_gate_passed", "status_gate_passed"):
            if clip_result.get(key) is not True:
                raise ValueError(f"Step1A qualification clip {clip_id!r} {key} is not true")
        for key in ("hard_gate_failures", "status_gate_failures"):
            if clip_result.get(key) != []:
                raise ValueError(f"Step1A qualification clip {clip_id!r} {key} is not empty")

    if batch.get("deployment_ready") is not False or batch.get("all_complete") is not True:
        raise ValueError("Step1A batch is not complete and non-deployable")
    if batch.get("input_count") != len(FROZEN_CLIP_IDS) or batch.get("completed_count") != len(FROZEN_CLIP_IDS):
        raise ValueError("Step1A batch clip coverage mismatch")
    for key in ("pending_count", "failed_count", "rejected_count"):
        if batch.get(key) != 0:
            raise ValueError(f"Step1A batch.{key} must be zero")
    if batch.get("ok_count") != len(FROZEN_CLIP_IDS) or batch.get("skipped_count") != 0:
        raise ValueError("Step1A batch must contain two fresh ok clips and zero skipped clips")
    provenance = _mapping(batch.get("clip_provenance"), "Step1A batch.clip_provenance")
    if set(provenance) != set(FROZEN_CLIP_IDS):
        raise ValueError("Step1A batch clip set differs from frozen idle support")
    for expected in expected_clips:
        clip = _mapping(provenance[expected["clip_id"]], f"Step1A batch clip {expected['clip_id']}")
        if clip.get("category") != "idle" or clip.get("qualification_categories") != ["idle"]:
            raise ValueError("Step1A batch clip category differs from idle support")
        source_csv = clip.get("source_csv")
        if not isinstance(source_csv, str) or not source_csv.replace("\\", "/").endswith(expected["source_csv"]):
            raise ValueError("Step1A batch clip source path mismatch")
    if "heldout_qualification" in batch and batch["heldout_qualification"] != qualification:
        raise ValueError("Step1A batch embedded qualification differs from qualification file")
    if batch.get("heldout_qualification") != qualification:
        raise ValueError("Step1A batch must embed the exact qualification payload")
    results = _mapping(batch.get("results"), "Step1A batch.results")
    summaries = _mapping(batch.get("successful_summaries"), "Step1A batch.successful_summaries")
    if set(results) != set(FROZEN_CLIP_IDS) or set(summaries) != set(FROZEN_CLIP_IDS):
        raise ValueError("Step1A batch result/summary clip sets differ from frozen support")
    expected_results = {
        clip_id: f"ok: {frame_count} frames"
        for clip_id, frame_count in zip(FROZEN_CLIP_IDS, FROZEN_FRAME_COUNTS, strict=True)
    }
    if dict(results) != expected_results:
        raise ValueError("Step1A batch results differ from frozen successful outputs")

    return {
        "support_manifest": support_ref,
        "qualification": qualification_ref,
        "batch": batch_ref,
    }


def _validate_initial_state(
    value: object,
    root: Path,
    context: str,
    *,
    expected_scope: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    reference, payload = _validate_json_ref(value, root, context)
    _exact_keys(payload, {"schema_version", "kind", "scope", "state"}, f"{context} payload")
    if payload["schema_version"] != 1 or payload["kind"] != INITIAL_STATE_SCHEMA:
        raise ValueError(f"{context} payload schema is unsupported")
    if payload["scope"] != expected_scope:
        raise ValueError(f"{context} payload scope differs from paired campaign contract")
    state = _mapping(payload["state"], f"{context} payload.state")
    root_keys = {
        "root_position_m",
        "root_orientation_xyzw",
        "root_linear_velocity_mps",
        "root_angular_velocity_radps",
        "reference_frame_index",
    }
    if expected_scope == INITIAL_STATE_SCOPES["common"]:
        _exact_keys(state, root_keys, f"{context} payload.state")
        joint_names: tuple[str, ...] | None = None
    else:
        _exact_keys(
            state,
            root_keys | {"joint_names", "joint_positions_rad", "joint_velocities_radps"},
            f"{context} payload.state",
        )
        arm = (
            "teacher_reference" if expected_scope == INITIAL_STATE_SCOPES["teacher_reference"] else "true23_expert"
        )
        expected_names, _, _ = MODEL_JOINT_SPECS[arm]
        joint_names = tuple(_name_array(state["joint_names"], f"{context} payload.state.joint_names"))
        if joint_names != expected_names:
            raise ValueError(f"{context} payload.state joint names/order differ from pinned model")
    normalized_state: dict[str, Any] = {
        "root_position_m": _vector(state["root_position_m"], 3, f"{context} payload.state.root_position_m"),
        "root_orientation_xyzw": _quaternion(
            state["root_orientation_xyzw"], f"{context} payload.state.root_orientation_xyzw"
        ),
        "root_linear_velocity_mps": _vector(
            state["root_linear_velocity_mps"], 3, f"{context} payload.state.root_linear_velocity_mps"
        ),
        "root_angular_velocity_radps": _vector(
            state["root_angular_velocity_radps"],
            3,
            f"{context} payload.state.root_angular_velocity_radps",
        ),
        "reference_frame_index": _integer(
            state["reference_frame_index"], f"{context} payload.state.reference_frame_index"
        ),
    }
    if normalized_state["reference_frame_index"] != FIXED_REFERENCE_START_FRAME:
        raise ValueError(f"{context} payload.state must start at reference frame zero")
    if joint_names is not None:
        normalized_state.update(
            {
                "joint_names": joint_names,
                "joint_positions_rad": _vector(
                    state["joint_positions_rad"],
                    len(joint_names),
                    f"{context} payload.state.joint_positions_rad",
                ),
                "joint_velocities_radps": _vector(
                    state["joint_velocities_radps"],
                    len(joint_names),
                    f"{context} payload.state.joint_velocities_radps",
                ),
            }
        )
    canonical_json_bytes(payload)
    return reference, {**payload, "state": normalized_state}


def _validate_disturbance(
    value: object,
    root: Path,
    contract: Mapping[str, Any],
    context: str,
) -> tuple[dict[str, Any], tuple[tuple[float, ...], ...]]:
    reference, payload = _validate_json_ref(value, root, context)
    _exact_keys(
        payload,
        {"schema_version", "kind", "control_hz", "deltas"},
        f"{context} payload",
    )
    if payload["schema_version"] != 1 or payload["kind"] != DISTURBANCE_SCHEMA:
        raise ValueError(f"{context} payload schema is unsupported")
    if payload["control_hz"] != contract["control_hz"]:
        raise ValueError(f"{context} payload control_hz differs from contract")
    deltas_raw = _array(payload["deltas"], f"{context} payload.deltas")
    if len(deltas_raw) != contract["horizon_steps"]:
        raise ValueError(f"{context} payload must contain exactly 500 disturbance steps")
    deltas = tuple(
        _vector(delta, 6, f"{context} payload.deltas[{index}]") for index, delta in enumerate(deltas_raw)
    )
    expected_delta = tuple(contract["campaign_schedule"]["disturbance_delta"])
    if any(delta != expected_delta for delta in deltas):
        raise ValueError(f"{context} payload differs from frozen disturbance campaign")
    return reference, deltas


def _validate_reference_schedule(
    value: object,
    root: Path,
    clip: Mapping[str, Any],
    context: str,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    reference, payload = _validate_json_ref(value, root, context)
    _exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "clip_id",
            "source_frame_count",
            "start_frame",
            "continuation",
            "frame_indices",
        },
        f"{context} payload",
    )
    if payload["schema_version"] != 1 or payload["kind"] != REFERENCE_SCHEDULE_SCHEMA:
        raise ValueError(f"{context} payload schema is unsupported")
    if payload["clip_id"] != clip["clip_id"]:
        raise ValueError(f"{context} payload clip_id differs from frozen clip")
    if payload["source_frame_count"] != clip["source_frame_count"]:
        raise ValueError(f"{context} payload source_frame_count differs from frozen evidence")
    if payload["continuation"] != REFERENCE_CONTINUATION:
        raise ValueError(f"{context} payload may not loop an uncertified idle seam")
    start = _integer(payload["start_frame"], f"{context} payload.start_frame")
    if start != FIXED_REFERENCE_START_FRAME:
        raise ValueError(f"{context} payload.start_frame must be exactly zero")
    if start >= clip["source_frame_count"]:
        raise ValueError(f"{context} payload.start_frame lies outside source clip")
    values = _array(payload["frame_indices"], f"{context} payload.frame_indices")
    if len(values) != REQUIRED_HORIZON_STEPS:
        raise ValueError(f"{context} payload must contain exactly 500 frame indices")
    frame_indices = tuple(
        _integer(value, f"{context} payload.frame_indices[{index}]") for index, value in enumerate(values)
    )
    expected = tuple(min(start + index, clip["source_frame_count"] - 1) for index in range(REQUIRED_HORIZON_STEPS))
    if frame_indices != expected:
        raise ValueError(f"{context} payload does not implement terminal_hold_last")
    return reference, frame_indices


def _validate_runtime_config_ref(
    value: object,
    root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    reference, payload = _validate_json_ref(value, root, "campaign.runtime.runtime_config")
    _exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "control_hz",
            "horizon_steps",
            "post_termination_policy",
            "shipped_end_effector_termination_threshold_m",
            "teacher_controller_semantics",
            "true23_controller_semantics",
            "resolved_config",
            "resolved_config_sha256",
        },
        "campaign.runtime.runtime_config payload",
    )
    if payload["schema_version"] != 1 or payload["kind"] != RUNTIME_CONFIG_SCHEMA:
        raise ValueError("campaign runtime config payload schema is unsupported")
    expected = {
        "control_hz": contract["control_hz"],
        "horizon_steps": REQUIRED_HORIZON_STEPS,
        "post_termination_policy": POST_TERMINATION_POLICY,
        "shipped_end_effector_termination_threshold_m": (SHIPPED_EE_TERMINATION_THRESHOLD_M),
        "teacher_controller_semantics": contract["controller_semantics"]["teacher_reference"],
        "true23_controller_semantics": contract["controller_semantics"]["true23_expert"],
    }
    for key, expected_value in expected.items():
        if payload[key] != expected_value:
            raise ValueError(f"campaign runtime config payload.{key} differs from contract")
    resolved = _mapping(payload["resolved_config"], "campaign.runtime.runtime_config payload.resolved_config")
    if not resolved:
        raise ValueError("campaign runtime resolved_config must be non-empty")
    descriptor = _mapping(
        resolved.get("controller_descriptor"),
        "campaign.runtime.runtime_config payload.resolved_config.controller_descriptor",
    )
    if dict(descriptor) != TRUE23_CONTROLLER_DESCRIPTOR:
        raise ValueError("campaign runtime controller descriptor differs from approved live-substep PD law")
    expected_hash = sha256_bytes(canonical_json_bytes(resolved))
    if (
        _sha256(
            payload["resolved_config_sha256"],
            "campaign.runtime.runtime_config payload.resolved_config_sha256",
        )
        != expected_hash
    ):
        raise ValueError("campaign runtime resolved_config SHA-256 mismatch")
    return reference


def _validate_input_action_law_ref(
    value: object,
    root: Path,
    *,
    arm: str,
    action_law: str,
    context: str,
) -> dict[str, Any]:
    reference, payload = _validate_json_ref(value, root, context)
    _exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "arm",
            "action_law",
            "state_dependence",
            "input_fields",
            "output_space",
        },
        f"{context} payload",
    )
    if payload["schema_version"] != 1 or payload["kind"] != INPUT_ACTION_LAW_SCHEMA:
        raise ValueError(f"{context} payload schema is unsupported")
    if payload["arm"] != arm or payload["action_law"] != action_law:
        raise ValueError(f"{context} payload action law identity mismatch")
    if payload["state_dependence"] != "live_robot_state_feedback":
        raise ValueError(f"{context} must consume live robot state feedback")
    input_fields = _name_array(payload["input_fields"], f"{context} payload.input_fields")
    if input_fields != ("robot_state", "reference"):
        raise ValueError(f"{context} must bind robot_state and reference inputs")
    output_space = _string(payload["output_space"], f"{context} payload.output_space")
    if arm == "true23_expert":
        if action_law != TRUE23_CONTROLLER_SEMANTICS or output_space != "joint_torque":
            raise ValueError("true23 input/action law is not the approved live-substep PD law")
    return reference


def _validate_true23_controller_config_ref(
    value: object,
    root: Path,
) -> tuple[dict[str, Any], dict[str, tuple[float, ...]]]:
    reference, payload = _validate_json_ref(value, root, "campaign.true23_expert.controller_config")
    if payload.get("schema_version") != 1 or payload.get("kind") != "state_feedback_controller_config":
        raise ValueError("campaign true23 controller config schema is unsupported")
    resolved = _mapping(payload.get("resolved_config"), "campaign true23 controller config.resolved_config")
    descriptor = _mapping(
        resolved.get("controller_descriptor"),
        "campaign true23 controller config.resolved_config.controller_descriptor",
    )
    if dict(descriptor) != TRUE23_CONTROLLER_DESCRIPTOR:
        raise ValueError("campaign true23 controller descriptor differs from approved live-substep PD law")
    gains = _mapping(
        resolved.get("physics_gains"),
        "campaign true23 controller config.resolved_config.physics_gains",
    )
    _exact_keys(gains, {"kp", "kd", "effort_limits_nm"}, "campaign true23 controller physics_gains")
    normalized = {
        "kp": _vector(gains["kp"], 23, "campaign true23 controller physics_gains.kp"),
        "kd": _vector(gains["kd"], 23, "campaign true23 controller physics_gains.kd"),
        "effort_limits_nm": _vector(
            gains["effort_limits_nm"],
            23,
            "campaign true23 controller physics_gains.effort_limits_nm",
        ),
    }
    if any(value <= 0.0 for values in normalized.values() for value in values):
        raise ValueError("campaign true23 controller gains and effort limits must be positive")
    return reference, normalized


def _validate_campaign(
    campaign: Mapping[str, Any],
    root: Path,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _exact_keys(
        campaign,
        {
            "schema_version",
            "kind",
            "declared_categories",
            "frozen_clip_ids",
            "clip_substitutions",
            "row_masks",
            "step1a",
            "teacher_reference",
            "true23_expert",
            "runtime",
            "episode_schedule",
        },
        "campaign manifest",
    )
    if campaign["schema_version"] != 1 or campaign["kind"] != CAMPAIGN_SCHEMA:
        raise ValueError("unsupported Step1B campaign manifest")
    if campaign["declared_categories"] != ["idle"]:
        raise ValueError("campaign must declare only idle")
    if tuple(_array(campaign["frozen_clip_ids"], "campaign.frozen_clip_ids")) != FROZEN_CLIP_IDS:
        raise ValueError("campaign clip IDs/order differ from frozen idle support")
    if campaign["clip_substitutions"] is not False:
        raise ValueError("campaign clip substitutions must be false")
    if campaign["row_masks"] is not False:
        raise ValueError("campaign row masks must be false")

    step1a = _validate_step1a(campaign["step1a"], root, contract)

    teacher = _mapping(campaign["teacher_reference"], "campaign.teacher_reference")
    _exact_keys(
        teacher,
        {
            "kind",
            "controller_semantics",
            "model",
            "checkpoint",
            "controller_config",
            "input_action_law",
            "semantic_reference",
        },
        "campaign.teacher_reference",
    )
    if teacher["kind"] != "exact_29dof_teacher_reference":
        raise ValueError("campaign teacher_reference kind is unsupported")
    if teacher["controller_semantics"] != TEACHER_CONTROLLER_SEMANTICS:
        raise ValueError("campaign teacher must use exact neural state-feedback 29-DoF low-latency policy")
    trusted_teacher = contract["trusted_identity_bindings"]["teacher_reference"]
    teacher_model = _validate_binary_ref(teacher["model"], root, "campaign.teacher_reference.model")
    teacher_checkpoint = _validate_binary_ref(teacher["checkpoint"], root, "campaign.teacher_reference.checkpoint")
    teacher_config, _teacher_config_payload = _validate_json_ref(
        teacher["controller_config"], root, "campaign.teacher_reference.controller_config"
    )
    teacher_action_law = _validate_input_action_law_ref(
        teacher["input_action_law"],
        root,
        arm="teacher_reference",
        action_law=TEACHER_CONTROLLER_SEMANTICS,
        context="campaign.teacher_reference.input_action_law",
    )
    teacher_semantic, _teacher_semantic_payload = _validate_json_ref(
        teacher["semantic_reference"], root, "campaign.teacher_reference.semantic_reference"
    )
    if teacher_model["sha256"] != trusted_teacher["model_sha256"]:
        raise ValueError("campaign teacher model differs from trusted identity")
    if teacher_checkpoint["sha256"] != trusted_teacher["checkpoint_sha256"]:
        raise ValueError("campaign teacher checkpoint differs from trusted identity")
    if teacher_config["payload_sha256"] != trusted_teacher["controller_config_payload_sha256"]:
        raise ValueError("campaign teacher controller config differs from trusted identity")
    if teacher_action_law["payload_sha256"] != trusted_teacher["input_action_law_payload_sha256"]:
        raise ValueError("campaign teacher input/action law differs from trusted identity")
    if teacher_semantic["payload_sha256"] != trusted_teacher["semantic_reference_payload_sha256"]:
        raise ValueError("campaign teacher semantic reference differs from trusted identity")
    teacher_normalized = {
        "kind": teacher["kind"],
        "controller_semantics": teacher["controller_semantics"],
        "model": teacher_model,
        "checkpoint": teacher_checkpoint,
        "controller_config": teacher_config,
        "input_action_law": teacher_action_law,
        "semantic_reference": teacher_semantic,
    }

    expert = _mapping(campaign["true23_expert"], "campaign.true23_expert")
    _exact_keys(
        expert,
        {
            "kind",
            "controller_semantics",
            "controller",
            "controller_config",
            "input_action_law",
            "target_model",
            "clips",
        },
        "campaign.true23_expert",
    )
    if expert["kind"] != "schema_v6_true23_expert":
        raise ValueError("campaign true23_expert kind must be schema_v6_true23_expert")
    approved_true23_semantics = contract["controller_semantics"]["true23_expert"]
    if expert["controller_semantics"] != approved_true23_semantics:
        raise ValueError("campaign true23 controller semantics differ from trusted approved contract")
    if expert["controller_semantics"] in FORBIDDEN_CONTROLLER_SEMANTICS:
        raise ValueError("source_trajectory_pd_replay is diagnostic-only")
    trusted_expert = contract["trusted_identity_bindings"]["true23_expert"]
    if expert["controller_semantics"] != trusted_expert["action_law"]:
        raise ValueError("campaign true23 controller semantics differ from trusted action law")
    expert_controller = _validate_binary_ref(expert["controller"], root, "campaign.true23_expert.controller")
    expert_controller_config, expert_physics_gains = _validate_true23_controller_config_ref(
        expert["controller_config"], root
    )
    expert_action_law = _validate_input_action_law_ref(
        expert["input_action_law"],
        root,
        arm="true23_expert",
        action_law=expert["controller_semantics"],
        context="campaign.true23_expert.input_action_law",
    )
    if expert_controller["sha256"] != trusted_expert["controller_sha256"]:
        raise ValueError("campaign true23 controller differs from trusted identity")
    if expert_controller_config["payload_sha256"] != trusted_expert["controller_config_payload_sha256"]:
        raise ValueError("campaign true23 controller config differs from trusted identity")
    if expert_action_law["payload_sha256"] != trusted_expert["input_action_law_payload_sha256"]:
        raise ValueError("campaign true23 input/action law differs from trusted identity")
    expert_clips = _array(expert["clips"], "campaign.true23_expert.clips")
    if len(expert_clips) != len(FROZEN_CLIP_IDS):
        raise ValueError("campaign true23_expert must bind exactly two clips")
    normalized_expert_clips: list[dict[str, Any]] = []
    for index, expected_id in enumerate(FROZEN_CLIP_IDS):
        clip = _mapping(expert_clips[index], f"campaign.true23_expert.clips[{index}]")
        _exact_keys(
            clip,
            {"clip_id", "motion", "expert", "report"},
            f"campaign.true23_expert.clips[{index}]",
        )
        if clip["clip_id"] != expected_id:
            raise ValueError("campaign true23 expert clip IDs/order differ from frozen support")
        report_ref, report_payload = _validate_json_ref(
            clip["report"], root, f"campaign.true23_expert.clips[{index}].report"
        )
        motion_ref = _validate_binary_ref(clip["motion"], root, f"campaign.true23_expert.clips[{index}].motion")
        expert_ref = _validate_binary_ref(clip["expert"], root, f"campaign.true23_expert.clips[{index}].expert")
        if report_payload.get("schema_version") != 6:
            raise ValueError("true23 expert report must be schema version 6")
        required_report_values = {
            "clip_id": expected_id,
            "category": "idle",
            "artifact_provenance_schema_version": 6,
            "source_model_sha256": teacher_model["sha256"],
            "target_model_sha256": trusted_expert["target_model_sha256"],
            "motion_output_sha256": motion_ref["sha256"],
            "expert_output_sha256": expert_ref["sha256"],
            "serialization_constraint_audit_passed": True,
            "expert_gate_passed": False,
        }
        for key, expected_value in required_report_values.items():
            if report_payload.get(key) != expected_value:
                raise ValueError(f"true23 expert report.{key} binding mismatch")
        constraints = _mapping(
            report_payload.get("constraints"), f"true23 expert report {expected_id}.constraints"
        )
        if constraints.get("certificate_basis") != "serialized_float32_position_arrays":
            raise ValueError("true23 expert report lacks float32 serialization certificate")
        trusted_clip = trusted_expert["clips"][expected_id]
        if (
            motion_ref["sha256"] != trusted_clip["motion_sha256"]
            or expert_ref["sha256"] != trusted_clip["expert_sha256"]
            or report_ref["payload_sha256"] != trusted_clip["report_payload_sha256"]
        ):
            raise ValueError("true23 schema-v6 triplet differs from trusted pinned identity")
        normalized_expert_clips.append(
            {
                "clip_id": expected_id,
                "motion": motion_ref,
                "expert": expert_ref,
                "report": report_ref,
            }
        )
    expert_normalized = {
        "kind": expert["kind"],
        "controller_semantics": expert["controller_semantics"],
        "controller": expert_controller,
        "controller_config": expert_controller_config,
        "physics_gains": expert_physics_gains,
        "input_action_law": expert_action_law,
        "target_model": _validate_binary_ref(expert["target_model"], root, "campaign.true23_expert.target_model"),
        "clips": normalized_expert_clips,
    }
    if expert_normalized["target_model"]["sha256"] != trusted_expert["target_model_sha256"]:
        raise ValueError("campaign true23 target model differs from trusted identity")

    runtime = _mapping(campaign["runtime"], "campaign.runtime")
    _exact_keys(
        runtime,
        {
            "simulator_name",
            "simulator_version",
            "runner",
            "runtime_config",
            "robot_assets",
            "termination_configs",
        },
        "campaign.runtime",
    )
    simulator_name = _string(runtime["simulator_name"], "campaign.runtime.simulator_name")
    simulator_version = _string(runtime["simulator_version"], "campaign.runtime.simulator_version")
    trusted_runtime = contract["trusted_identity_bindings"]["runtime"]
    if simulator_name != trusted_runtime["simulator_name"]:
        raise ValueError("campaign simulator_name differs from trusted identity")
    if simulator_version != trusted_runtime["simulator_version"]:
        raise ValueError("campaign simulator_version differs from trusted identity")
    robot_assets_raw = _array(runtime["robot_assets"], "campaign.runtime.robot_assets")
    if not robot_assets_raw:
        raise ValueError("campaign.runtime.robot_assets must be non-empty")
    robot_assets: list[dict[str, Any]] = []
    seen_asset_roles: set[str] = set()
    for index, value in enumerate(robot_assets_raw):
        asset = _mapping(value, f"campaign.runtime.robot_assets[{index}]")
        _exact_keys(asset, {"role", "artifact"}, f"campaign.runtime.robot_assets[{index}]")
        role = _string(asset["role"], f"campaign.runtime.robot_assets[{index}].role")
        if role in seen_asset_roles:
            raise ValueError("campaign.runtime.robot_assets has duplicate role")
        seen_asset_roles.add(role)
        robot_assets.append(
            {
                "role": role,
                "artifact": _validate_binary_ref(
                    asset["artifact"], root, f"campaign.runtime.robot_assets[{index}].artifact"
                ),
            }
        )
    if seen_asset_roles != set(trusted_runtime["robot_assets"]):
        raise ValueError("campaign runtime robot asset roles differ from trusted contract")
    for asset in robot_assets:
        if asset["artifact"]["sha256"] != trusted_runtime["robot_assets"][asset["role"]]:
            raise ValueError(f"campaign robot asset {asset['role']!r} differs from trusted identity")
    termination_raw = _array(runtime["termination_configs"], "campaign.runtime.termination_configs")
    expected_termination_paths = contract["selected_termination_gate"]["config_relpaths"]
    if len(termination_raw) != len(expected_termination_paths):
        raise ValueError("campaign must bind every shipped termination config")
    termination_configs = [
        _validate_repository_binding(value, f"campaign.runtime.termination_configs[{index}]")
        for index, value in enumerate(termination_raw)
    ]
    if [item["relpath"] for item in termination_configs] != expected_termination_paths:
        raise ValueError("campaign termination config paths/order differ from contract")
    for item in termination_configs:
        expected_digest = contract["selected_termination_gate"]["config_sha256"][item["relpath"]]
        if item["sha256"] != expected_digest:
            raise ValueError("campaign shipped termination config differs from pinned semantics")
    runner_ref = _validate_binary_ref(runtime["runner"], root, "campaign.runtime.runner")
    runtime_config_ref = _validate_runtime_config_ref(runtime["runtime_config"], root, contract)
    if runner_ref["sha256"] != trusted_runtime["runner_sha256"]:
        raise ValueError("campaign runner differs from trusted identity")
    if runtime_config_ref["payload_sha256"] != trusted_runtime["runtime_config_payload_sha256"]:
        raise ValueError("campaign runtime config differs from trusted identity")
    runtime_normalized = {
        "simulator_name": simulator_name,
        "simulator_version": simulator_version,
        "runner": runner_ref,
        "runtime_config": runtime_config_ref,
        "robot_assets": robot_assets,
        "termination_configs": termination_configs,
        "selected_termination_gate": contract["selected_termination_gate"],
    }

    schedule_raw = _array(campaign["episode_schedule"], "campaign.episode_schedule")
    expected_total = len(FROZEN_CLIP_IDS) * contract["episodes_per_clip"]
    if len(schedule_raw) != expected_total:
        raise ValueError(f"campaign episode schedule must contain exactly {expected_total} pairs")
    expected_clip_order = tuple(
        clip_id for clip_id in FROZEN_CLIP_IDS for _ in range(contract["episodes_per_clip"])
    )
    schedule: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    seen_identities: set[tuple[str, int, str]] = set()
    clip_contracts = {clip["clip_id"]: clip for clip in contract["frozen_clips"]}
    for index, value in enumerate(schedule_raw):
        item = _mapping(value, f"campaign.episode_schedule[{index}]")
        _exact_keys(
            item,
            {
                "pair_id",
                "clip_id",
                "seed",
                "reference_schedule",
                "common_initial_state",
                "teacher_initial_state",
                "true23_initial_state",
                "disturbance_schedule",
            },
            f"campaign.episode_schedule[{index}]",
        )
        pair_id = _string(item["pair_id"], f"campaign.episode_schedule[{index}].pair_id")
        clip_id = _string(item["clip_id"], f"campaign.episode_schedule[{index}].clip_id")
        if clip_id != expected_clip_order[index]:
            raise ValueError("campaign episode schedule clip order/count differs from contract")
        if pair_id in seen_pair_ids:
            raise ValueError("campaign episode schedule has duplicate pair_id")
        seen_pair_ids.add(pair_id)
        seed = _integer(item["seed"], f"campaign.episode_schedule[{index}].seed")
        episode_index = index % contract["episodes_per_clip"]
        expected_seed = contract["campaign_schedule"]["seeds_by_clip"][clip_id][episode_index]
        if seed != expected_seed:
            raise ValueError("campaign episode seed differs from frozen schedule")
        reference_ref, frame_indices = _validate_reference_schedule(
            item["reference_schedule"],
            root,
            clip_contracts[clip_id],
            f"campaign.episode_schedule[{index}].reference_schedule",
        )
        identity = (clip_id, seed, reference_ref["payload_sha256"])
        if identity in seen_identities:
            raise ValueError("campaign episode schedule has duplicate clip/seed/reference identity")
        seen_identities.add(identity)
        common_initial_ref, common_initial_payload = _validate_initial_state(
            item["common_initial_state"],
            root,
            f"campaign.episode_schedule[{index}].common_initial_state",
            expected_scope=INITIAL_STATE_SCOPES["common"],
        )
        teacher_initial_ref, teacher_initial_payload = _validate_initial_state(
            item["teacher_initial_state"],
            root,
            f"campaign.episode_schedule[{index}].teacher_initial_state",
            expected_scope=INITIAL_STATE_SCOPES["teacher_reference"],
        )
        true23_initial_ref, true23_initial_payload = _validate_initial_state(
            item["true23_initial_state"],
            root,
            f"campaign.episode_schedule[{index}].true23_initial_state",
            expected_scope=INITIAL_STATE_SCOPES["true23_expert"],
        )
        common_root = common_initial_payload["state"]
        for arm_name, arm_payload in (
            ("teacher", teacher_initial_payload),
            ("true23", true23_initial_payload),
        ):
            arm_state = arm_payload["state"]
            for key in (
                "root_position_m",
                "root_orientation_xyzw",
                "root_linear_velocity_mps",
                "root_angular_velocity_radps",
                "reference_frame_index",
            ):
                if arm_state[key] != common_root[key]:
                    raise ValueError(f"campaign {arm_name} initial state differs from common root projection")
        disturbance_ref, deltas = _validate_disturbance(
            item["disturbance_schedule"],
            root,
            contract,
            f"campaign.episode_schedule[{index}].disturbance_schedule",
        )
        schedule.append(
            {
                "pair_id": pair_id,
                "clip_id": clip_id,
                "seed": seed,
                "reference_schedule": reference_ref,
                "reference_frame_indices": frame_indices,
                "common_initial_state": common_initial_ref,
                "common_initial_state_payload": common_initial_payload,
                "teacher_initial_state": teacher_initial_ref,
                "teacher_initial_state_payload": teacher_initial_payload,
                "true23_initial_state": true23_initial_ref,
                "true23_initial_state_payload": true23_initial_payload,
                "disturbance_schedule": disturbance_ref,
                "disturbance_deltas": deltas,
            }
        )

    normalized = {
        "declared_categories": ["idle"],
        "frozen_clip_ids": list(FROZEN_CLIP_IDS),
        "clip_substitutions": False,
        "row_masks": False,
        "step1a": step1a,
        "teacher_reference": teacher_normalized,
        "true23_expert": expert_normalized,
        "runtime": runtime_normalized,
        "episode_count": len(schedule),
    }
    return normalized, schedule


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _quaternion_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(left, right, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _validate_trace_step(
    value: object,
    *,
    index: int,
    contract: Mapping[str, Any],
    expected_disturbance: Sequence[float],
    expected_reference_frame: int,
    episode_instance_id: str,
    joint_lower: Sequence[float],
    joint_upper: Sequence[float],
    context: str,
) -> dict[str, Any]:
    step = _mapping(value, context)
    _exact_keys(
        step,
        {
            "step_index",
            "terminated",
            "timed_out",
            "post_termination",
            "episode_instance_id",
            "reset_generation",
            "sim_advanced",
            "reported_nonfinite",
            "reference_frame_index",
            "joint_positions_rad",
            "disturbance_delta",
            "contacts",
            "pelvis",
            "com_position_m",
            "semantic_points_m",
            "semantic_orientations_xyzw",
            "link_velocities",
        },
        context,
    )
    if _integer(step["step_index"], f"{context}.step_index") != index:
        raise ValueError(f"{context}.step_index is noncontiguous")
    terminated = _boolean(step["terminated"], f"{context}.terminated")
    timed_out = _boolean(step["timed_out"], f"{context}.timed_out")
    if terminated and timed_out:
        raise ValueError(f"{context} cannot be both terminated and timed_out")
    if step["episode_instance_id"] != episode_instance_id:
        raise ValueError(f"{context}.episode_instance_id changed during fixed horizon")
    if _integer(step["reset_generation"], f"{context}.reset_generation") != 0:
        raise ValueError(f"{context}.reset_generation proves an unauthorized reset")
    if _boolean(step["sim_advanced"], f"{context}.sim_advanced") is not True:
        raise ValueError(f"{context}.sim_advanced must remain true after termination")
    if _integer(step["reference_frame_index"], f"{context}.reference_frame_index") != expected_reference_frame:
        raise ValueError(f"{context}.reference_frame_index differs from immutable schedule")
    joint_positions = _vector(step["joint_positions_rad"], len(joint_lower), f"{context}.joint_positions_rad")
    hard_limit_violation = any(
        position < lower - _METRIC_EPSILON or position > upper + _METRIC_EPSILON
        for position, lower, upper in zip(joint_positions, joint_lower, joint_upper, strict=True)
    )
    disturbance = _vector(step["disturbance_delta"], 6, f"{context}.disturbance_delta")
    if disturbance != tuple(expected_disturbance):
        raise ValueError(f"{context}.disturbance_delta differs from immutable schedule")

    contacts = _mapping(step["contacts"], f"{context}.contacts")
    if set(contacts) != set(contract["required_contact_links"]):
        raise ValueError(f"{context}.contacts keys differ from contract")
    normalized_contacts = {
        name: _boolean(contacts[name], f"{context}.contacts.{name}") for name in contract["required_contact_links"]
    }

    pelvis = _mapping(step["pelvis"], f"{context}.pelvis")
    _exact_keys(pelvis, {"position_m", "orientation_xyzw"}, f"{context}.pelvis")
    semantic_points = _mapping(step["semantic_points_m"], f"{context}.semantic_points_m")
    if set(semantic_points) != set(contract["required_semantic_points"]):
        raise ValueError(f"{context}.semantic_points_m keys differ from contract")
    orientations = _mapping(step["semantic_orientations_xyzw"], f"{context}.semantic_orientations_xyzw")
    if set(orientations) != set(contract["required_semantic_orientations"]):
        raise ValueError(f"{context}.semantic_orientations_xyzw keys differ from contract")
    velocities = _mapping(step["link_velocities"], f"{context}.link_velocities")
    if set(velocities) != set(contract["required_link_velocities"]):
        raise ValueError(f"{context}.link_velocities keys differ from contract")
    normalized_velocities: dict[str, Any] = {}
    for name in contract["required_link_velocities"]:
        velocity = _mapping(velocities[name], f"{context}.link_velocities.{name}")
        _exact_keys(
            velocity,
            {"linear_mps", "angular_radps"},
            f"{context}.link_velocities.{name}",
        )
        normalized_velocities[name] = {
            "linear_mps": _vector(velocity["linear_mps"], 3, f"{context}.link_velocities.{name}.linear_mps"),
            "angular_radps": _vector(
                velocity["angular_radps"], 3, f"{context}.link_velocities.{name}.angular_radps"
            ),
        }
    return {
        "step_index": index,
        "terminated": terminated,
        "timed_out": timed_out,
        "post_termination": _boolean(step["post_termination"], f"{context}.post_termination"),
        "hard_joint_limit_violation": hard_limit_violation,
        "joint_positions_rad": joint_positions,
        "reported_nonfinite": _boolean(step["reported_nonfinite"], f"{context}.reported_nonfinite"),
        "contacts": normalized_contacts,
        "pelvis": {
            "position_m": _vector(pelvis["position_m"], 3, f"{context}.pelvis.position_m"),
            "orientation_xyzw": _quaternion(pelvis["orientation_xyzw"], f"{context}.pelvis.orientation_xyzw"),
        },
        "com_position_m": _vector(step["com_position_m"], 3, f"{context}.com_position_m"),
        "semantic_points_m": {
            name: _vector(semantic_points[name], 3, f"{context}.semantic_points_m.{name}")
            for name in contract["required_semantic_points"]
        },
        "semantic_orientations_xyzw": {
            name: _quaternion(orientations[name], f"{context}.semantic_orientations_xyzw.{name}")
            for name in contract["required_semantic_orientations"]
        },
        "link_velocities": normalized_velocities,
    }


def _validate_trace(
    value: object,
    root: Path,
    schedule: Mapping[str, Any],
    arm: str,
    contract: Mapping[str, Any],
    context: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    reference, payload = _validate_json_ref(value, root, context)
    _exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "pair_id",
            "arm",
            "clip_id",
            "seed",
            "controller_semantics",
            "post_termination_policy",
            "episode_instance_id",
            "reset_generation",
            "reference_schedule_sha256",
            "reference_schedule_payload_sha256",
            "common_initial_state_sha256",
            "common_initial_state_payload_sha256",
            "arm_initial_state_sha256",
            "arm_initial_state_payload_sha256",
            "disturbance_schedule_sha256",
            "disturbance_schedule_payload_sha256",
            "control_hz",
            "horizon_steps",
            "joint_names",
            "hard_joint_position_lower_rad",
            "hard_joint_position_upper_rad",
            "record_count",
            "steps",
        },
        f"{context} payload",
    )
    if payload["schema_version"] != 1 or payload["kind"] != TRACE_SCHEMA:
        raise ValueError(f"{context} payload schema is unsupported")
    arm_initial_key = "teacher_initial_state" if arm == "teacher_reference" else "true23_initial_state"
    arm_initial = schedule[arm_initial_key]
    expected_header = {
        "pair_id": schedule["pair_id"],
        "arm": arm,
        "clip_id": schedule["clip_id"],
        "seed": schedule["seed"],
        "controller_semantics": contract["controller_semantics"][arm],
        "post_termination_policy": POST_TERMINATION_POLICY,
        "episode_instance_id": schedule["pair_id"],
        "reset_generation": 0,
        "reference_schedule_sha256": schedule["reference_schedule"]["sha256"],
        "reference_schedule_payload_sha256": schedule["reference_schedule"]["payload_sha256"],
        "common_initial_state_sha256": schedule["common_initial_state"]["sha256"],
        "common_initial_state_payload_sha256": schedule["common_initial_state"]["payload_sha256"],
        "arm_initial_state_sha256": arm_initial["sha256"],
        "arm_initial_state_payload_sha256": arm_initial["payload_sha256"],
        "disturbance_schedule_sha256": schedule["disturbance_schedule"]["sha256"],
        "disturbance_schedule_payload_sha256": schedule["disturbance_schedule"]["payload_sha256"],
        "control_hz": contract["control_hz"],
        "horizon_steps": contract["horizon_steps"],
        "record_count": REQUIRED_HORIZON_STEPS,
    }
    for key, expected in expected_header.items():
        if payload[key] != expected:
            raise ValueError(f"{context} payload.{key} differs from paired campaign schedule")

    joint_names = _name_array(payload["joint_names"], f"{context} payload.joint_names")
    expected_names, expected_lower, expected_upper = MODEL_JOINT_SPECS[arm]
    expected_joint_count = len(expected_names)
    if tuple(joint_names) != expected_names:
        raise ValueError(f"{context} joint names/order differ from pinned model")
    joint_lower = _vector(
        payload["hard_joint_position_lower_rad"],
        expected_joint_count,
        f"{context} payload.hard_joint_position_lower_rad",
    )
    joint_upper = _vector(
        payload["hard_joint_position_upper_rad"],
        expected_joint_count,
        f"{context} payload.hard_joint_position_upper_rad",
    )
    if any(lower >= upper for lower, upper in zip(joint_lower, joint_upper, strict=True)):
        raise ValueError(f"{context} hard joint position bounds are invalid")
    if joint_lower != expected_lower or joint_upper != expected_upper:
        raise ValueError(f"{context} hard joint position bounds differ from pinned model")
    raw_steps = _array(payload["steps"], f"{context} payload.steps")
    if len(raw_steps) != REQUIRED_HORIZON_STEPS:
        raise ValueError(f"{context} must contain exactly 500 recorded steps")
    steps = [
        _validate_trace_step(
            value,
            index=index,
            contract=contract,
            expected_disturbance=schedule["disturbance_deltas"][index],
            expected_reference_frame=schedule["reference_frame_indices"][index],
            episode_instance_id=schedule["pair_id"],
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            context=f"{context} payload.steps[{index}]",
        )
        for index, value in enumerate(raw_steps)
    ]

    terminal_seen = False
    timeout_success = False
    for index, step in enumerate(steps):
        if step["post_termination"] is not terminal_seen:
            raise ValueError(f"{context} payload.steps[{index}].post_termination is not latched")
        if step["terminated"] or step["timed_out"]:
            if terminal_seen:
                raise ValueError(f"{context} has more than one terminal event")
            terminal_seen = True
            if step["timed_out"]:
                if index != REQUIRED_HORIZON_STEPS - 1:
                    raise ValueError(f"{context} timed_out before fixed horizon")
                timeout_success = True
    return reference, steps, timeout_success


def _projected_gravity_z_wxyz(quaternion: Sequence[float]) -> float:
    _, x, y, _ = quaternion
    return -(1.0 - 2.0 * (x * x + y * y))


def _validate_raw_trace(
    value: object,
    root: Path,
    schedule: Mapping[str, Any],
    arm: str,
    contract: Mapping[str, Any],
    trace_reference: Mapping[str, Any],
    trace_steps: Sequence[Mapping[str, Any]],
    physics_gains: Mapping[str, Sequence[float]] | None,
    context: str,
) -> dict[str, Any]:
    reference, payload = _validate_json_ref(value, root, context)
    _exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "pair_id",
            "arm",
            "clip_id",
            "seed",
            "controller_semantics",
            "trace_sha256",
            "trace_payload_sha256",
            "model",
            "control_hz",
            "physics_hz",
            "horizon_steps",
            "episode_instance_id",
            "reset_generation",
            "joint_names",
            "hard_joint_position_lower_rad",
            "hard_joint_position_upper_rad",
            "actuator_names",
            "floor_geom_id",
            "geom_identities",
            "record_count",
            "steps",
        },
        f"{context} payload",
    )
    if payload["schema_version"] != 1 or payload["kind"] != RAW_AUDIT_SCHEMA:
        raise ValueError(f"{context} payload schema is unsupported")
    expected_header = {
        "pair_id": schedule["pair_id"],
        "arm": arm,
        "clip_id": schedule["clip_id"],
        "seed": schedule["seed"],
        "controller_semantics": contract["controller_semantics"][arm],
        "trace_sha256": trace_reference["sha256"],
        "trace_payload_sha256": trace_reference["payload_sha256"],
        "control_hz": contract["control_hz"],
        "horizon_steps": contract["horizon_steps"],
        "episode_instance_id": schedule["pair_id"],
        "reset_generation": 0,
        "record_count": REQUIRED_HORIZON_STEPS,
    }
    for key, expected in expected_header.items():
        if payload[key] != expected:
            raise ValueError(f"{context} payload.{key} differs from paired trace/campaign")

    expected_names, expected_lower, expected_upper = MODEL_JOINT_SPECS[arm]
    joint_count = len(expected_names)
    if tuple(_name_array(payload["joint_names"], f"{context} payload.joint_names")) != expected_names:
        raise ValueError(f"{context} joint names/order differ from pinned model")
    if (
        _vector(payload["hard_joint_position_lower_rad"], joint_count, f"{context} payload hard lower")
        != expected_lower
        or _vector(payload["hard_joint_position_upper_rad"], joint_count, f"{context} payload hard upper")
        != expected_upper
    ):
        raise ValueError(f"{context} joint bounds differ from pinned model")
    if tuple(_name_array(payload["actuator_names"], f"{context} payload.actuator_names")) != expected_names:
        raise ValueError(f"{context} actuator names/order differ from pinned model")

    model_specs = {
        "teacher_reference": {
            "repository_relpath": "gear_sonic/data/robots/g1/g1_29dof.xml",
            "sha256": "386b1bb9ea5b69ccd6fd0283a73ffea1ee052df95564e23a780125fbcbe2c645",
            "nq": 36,
            "nv": 35,
            "nu": 29,
        },
        "true23_expert": {
            "repository_relpath": "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
            "sha256": "16e304c970bfc68783ea69d01d05192e6bd9d83d62f6ee4aac0ac72ff18db612",
            "nq": 30,
            "nv": 29,
            "nu": 23,
        },
    }
    model = _mapping(payload["model"], f"{context} payload.model")
    _exact_keys(model, {"repository_relpath", "sha256", "size_bytes", "nq", "nv", "nu"}, f"{context} model")
    expected_model = model_specs[arm]
    for key in ("repository_relpath", "sha256", "nq", "nv", "nu"):
        if model[key] != expected_model[key]:
            raise ValueError(f"{context} model.{key} differs from pinned canonical model")
    _validate_repository_binding(
        {
            "relpath": model["repository_relpath"],
            "sha256": model["sha256"],
            "size_bytes": model["size_bytes"],
        },
        f"{context} model",
    )
    physics_hz = _integer(payload["physics_hz"], f"{context} payload.physics_hz", minimum=contract["control_hz"])
    if physics_hz % contract["control_hz"]:
        raise ValueError(f"{context} physics_hz must be an integer multiple of control_hz")

    floor_geom_id = _integer(payload["floor_geom_id"], f"{context} payload.floor_geom_id", minimum=0)
    geom_identities: dict[int, Mapping[str, Any]] = {}
    for index, raw_identity in enumerate(_array(payload["geom_identities"], f"{context} payload.geom_identities")):
        identity = _mapping(raw_identity, f"{context} payload.geom_identities[{index}]")
        _exact_keys(identity, {"geom_id", "geom_name", "body_id", "body_name"}, f"{context} geom identity")
        geom_id = _integer(identity["geom_id"], f"{context} geom_id", minimum=0)
        if geom_id in geom_identities:
            raise ValueError(f"{context} has duplicate geom identity")
        _string(identity["geom_name"], f"{context} geom_name")
        _integer(identity["body_id"], f"{context} body_id", minimum=0)
        _string(identity["body_name"], f"{context} body_name")
        geom_identities[geom_id] = identity
    if floor_geom_id not in geom_identities:
        raise ValueError(f"{context} floor geom lacks an identity")

    ee_names = tuple(SELECTED_TERMINATION_GATE["terms"]["ee_body_pos"]["body_names"])
    raw_steps = _array(payload["steps"], f"{context} payload.steps")
    if len(raw_steps) != REQUIRED_HORIZON_STEPS:
        raise ValueError(f"{context} must contain exactly 500 raw steps")
    terminal_seen = False
    for index, raw_step in enumerate(raw_steps):
        step_context = f"{context} payload.steps[{index}]"
        step = _mapping(raw_step, step_context)
        _exact_keys(
            step,
            {
                "step_index",
                "reference_frame_index",
                "reference_terminal_held",
                "sim_time_before_s",
                "sim_time_after_s",
                "reset_generation",
                "sim_advanced",
                "qpos_before",
                "qvel_before",
                "qpos_after",
                "qvel_after",
                "raw_action_native",
                "safe_action_native",
                "q_target_hardware_rad",
                "disturbance_delta_world",
                "pd_substeps_sha256",
                "applied_torque_first_hardware_nm",
                "applied_torque_last_hardware_nm",
                "applied_torque_peak_abs_hardware_nm",
                "raw_contact_geom_pairs",
                "torso_reference",
                "ee_reference",
                "termination_terms",
                "reported_nonfinite",
            },
            step_context,
        )
        if _integer(step["step_index"], f"{step_context}.step_index") != index:
            raise ValueError(f"{step_context}.step_index is noncontiguous")
        if step["reference_frame_index"] != schedule["reference_frame_indices"][index]:
            raise ValueError(f"{step_context} reference frame differs from frozen schedule")
        expected_held = (
            schedule["reference_frame_indices"][index]
            == FROZEN_FRAME_COUNTS[FROZEN_CLIP_IDS.index(schedule["clip_id"])] - 1
        )
        if _boolean(step["reference_terminal_held"], f"{step_context}.reference_terminal_held") != expected_held:
            raise ValueError(f"{step_context} terminal-hold flag differs from frozen schedule")
        before = _number(step["sim_time_before_s"], f"{step_context}.sim_time_before_s", minimum=0.0)
        after = _number(step["sim_time_after_s"], f"{step_context}.sim_time_after_s", minimum=0.0)
        if after <= before:
            raise ValueError(f"{step_context} simulation time did not advance")
        if step["reset_generation"] != 0 or step["sim_advanced"] is not True:
            raise ValueError(f"{step_context} proves reset or halted simulation")
        nq = expected_model["nq"]
        nv = expected_model["nv"]
        qpos_before = _vector(step["qpos_before"], nq, f"{step_context}.qpos_before")
        qvel_before = _vector(step["qvel_before"], nv, f"{step_context}.qvel_before")
        qpos_after = _vector(step["qpos_after"], nq, f"{step_context}.qpos_after")
        _vector(step["qvel_after"], nv, f"{step_context}.qvel_after")
        if index == 0:
            initial_payload_key = (
                "teacher_initial_state_payload" if arm == "teacher_reference" else "true23_initial_state_payload"
            )
            initial_state = schedule[initial_payload_key]["state"]
            common_state = schedule["common_initial_state_payload"]["state"]
            root_xyzw = common_state["root_orientation_xyzw"]
            root_wxyz = (root_xyzw[3], root_xyzw[0], root_xyzw[1], root_xyzw[2])
            if (
                qpos_before[:3] != common_state["root_position_m"]
                or qpos_before[3:7] != root_wxyz
                or qvel_before[:3] != common_state["root_linear_velocity_mps"]
                or qvel_before[3:6] != common_state["root_angular_velocity_radps"]
                or qpos_before[7:] != initial_state["joint_positions_rad"]
                or qvel_before[6:] != initial_state["joint_velocities_radps"]
            ):
                raise ValueError(f"{step_context} qpos_before/qvel_before differ from pinned initial state")
        _vector(step["raw_action_native"], joint_count, f"{step_context}.raw_action_native")
        _vector(step["safe_action_native"], joint_count, f"{step_context}.safe_action_native")
        q_target = _vector(step["q_target_hardware_rad"], joint_count, f"{step_context}.q_target_hardware_rad")
        if _vector(step["disturbance_delta_world"], 6, f"{step_context}.disturbance_delta_world") != tuple(
            schedule["disturbance_deltas"][index]
        ):
            raise ValueError(f"{step_context} disturbance differs from frozen schedule")
        _sha256(step["pd_substeps_sha256"], f"{step_context}.pd_substeps_sha256")
        applied_torque_first = _vector(
            step["applied_torque_first_hardware_nm"], joint_count, f"{step_context}.applied_torque_first"
        )
        _vector(step["applied_torque_last_hardware_nm"], joint_count, f"{step_context}.applied_torque_last")
        torque_peak = _vector(
            step["applied_torque_peak_abs_hardware_nm"], joint_count, f"{step_context}.torque_peak"
        )
        if any(value < 0.0 for value in torque_peak):
            raise ValueError(f"{step_context} torque peaks must be non-negative")
        if arm == "true23_expert":
            assert physics_gains is not None
            q = qpos_before[7:]
            dq = qvel_before[6:]
            for joint_index, (position, velocity, target, torque) in enumerate(
                zip(q, dq, q_target, applied_torque_first, strict=True)
            ):
                unclipped = (
                    physics_gains["kp"][joint_index] * (target - position)
                    - physics_gains["kd"][joint_index] * velocity
                )
                effort = physics_gains["effort_limits_nm"][joint_index]
                expected_torque = min(effort, max(-effort, unclipped))
                if abs(torque - expected_torque) > 1.0e-9:
                    raise ValueError(f"{step_context} applied torque differs from live-state PD law")
        if tuple(qpos_after[7:]) != trace_steps[index]["joint_positions_rad"]:
            raise ValueError(f"{step_context} qpos_after differs from derived trace")

        contact_feet = {"left_foot": False, "right_foot": False}
        for pair_index, raw_pair in enumerate(
            _array(step["raw_contact_geom_pairs"], f"{step_context}.raw_contact_geom_pairs")
        ):
            pair = _array(raw_pair, f"{step_context}.raw_contact_geom_pairs[{pair_index}]")
            if len(pair) != 2:
                raise ValueError(f"{step_context} raw contact pair must contain two geom IDs")
            geom_a = _integer(pair[0], f"{step_context} contact geom A", minimum=0)
            geom_b = _integer(pair[1], f"{step_context} contact geom B", minimum=0)
            if geom_a not in geom_identities or geom_b not in geom_identities:
                raise ValueError(f"{step_context} raw contact references unknown geom")
            if floor_geom_id not in (geom_a, geom_b):
                continue
            other = geom_b if geom_a == floor_geom_id else geom_a
            body_name = geom_identities[other]["body_name"]
            if body_name == "left_ankle_roll_link":
                contact_feet["left_foot"] = True
            elif body_name == "right_ankle_roll_link":
                contact_feet["right_foot"] = True
        if contact_feet != trace_steps[index]["contacts"]:
            raise ValueError(f"{step_context} derived contact flags differ from raw geom contacts")

        torso = _mapping(step["torso_reference"], f"{step_context}.torso_reference")
        _exact_keys(
            torso,
            {
                "measured_position_m",
                "measured_quaternion_wxyz",
                "reference_position_m",
                "reference_quaternion_wxyz",
            },
            f"{step_context}.torso_reference",
        )
        measured_pos = _vector(torso["measured_position_m"], 3, f"{step_context} torso measured position")
        reference_pos = _vector(torso["reference_position_m"], 3, f"{step_context} torso reference position")
        measured_quat = _quaternion(torso["measured_quaternion_wxyz"], f"{step_context} torso measured quat")
        reference_quat = _quaternion(torso["reference_quaternion_wxyz"], f"{step_context} torso reference quat")
        anchor_pos_error = abs(reference_pos[2] - measured_pos[2])
        anchor_ori_error = abs(
            _projected_gravity_z_wxyz(reference_quat) - _projected_gravity_z_wxyz(measured_quat)
        )
        ee = _mapping(step["ee_reference"], f"{step_context}.ee_reference")
        if set(ee) != set(ee_names):
            raise ValueError(f"{step_context} end-effector reference set differs from shipped gate")
        ee_errors: dict[str, float] = {}
        for name in ee_names:
            body = _mapping(ee[name], f"{step_context}.ee_reference.{name}")
            _exact_keys(
                body,
                {
                    "measured_position_m",
                    "measured_quaternion_wxyz",
                    "reference_position_m",
                    "reference_quaternion_wxyz",
                },
                f"{step_context}.ee_reference.{name}",
            )
            measured = _vector(body["measured_position_m"], 3, f"{step_context} {name} measured position")
            reference_body = _vector(body["reference_position_m"], 3, f"{step_context} {name} reference position")
            _quaternion(body["measured_quaternion_wxyz"], f"{step_context} {name} measured quaternion")
            _quaternion(body["reference_quaternion_wxyz"], f"{step_context} {name} reference quaternion")
            ee_errors[name] = abs(reference_body[2] - measured[2])

        terms = _mapping(step["termination_terms"], f"{step_context}.termination_terms")
        _exact_keys(
            terms,
            {
                "time_out",
                "anchor_pos",
                "anchor_pos_error_z_m",
                "anchor_pos_threshold_m",
                "anchor_ori",
                "anchor_ori_projected_gravity_z_error",
                "anchor_ori_threshold",
                "ee_body_pos",
                "ee_body_pos_error_z_m",
                "ee_body_pos_threshold_m",
                "terminated",
                "timed_out",
            },
            f"{step_context}.termination_terms",
        )
        expected_anchor_pos = anchor_pos_error > 0.25
        expected_anchor_ori = anchor_ori_error > 0.8
        expected_ee = any(error > 0.25 for error in ee_errors.values())
        expected_timeout = index == REQUIRED_HORIZON_STEPS - 1 and not terminal_seen
        expected_terminated = (expected_anchor_pos or expected_anchor_ori or expected_ee) and not terminal_seen
        expected_timed_out = expected_timeout and not expected_terminated
        term_expectations = {
            "time_out": expected_timeout,
            "anchor_pos": expected_anchor_pos,
            "anchor_pos_error_z_m": anchor_pos_error,
            "anchor_pos_threshold_m": 0.25,
            "anchor_ori": expected_anchor_ori,
            "anchor_ori_projected_gravity_z_error": anchor_ori_error,
            "anchor_ori_threshold": 0.8,
            "ee_body_pos": expected_ee,
            "ee_body_pos_error_z_m": ee_errors,
            "ee_body_pos_threshold_m": 0.25,
            "terminated": expected_terminated,
            "timed_out": expected_timed_out,
        }
        if dict(terms) != term_expectations:
            raise ValueError(f"{step_context} termination terms differ from raw-observable recomputation")
        if (
            trace_steps[index]["terminated"] != expected_terminated
            or trace_steps[index]["timed_out"] != expected_timed_out
        ):
            raise ValueError(f"{step_context} derived trace terminal flags differ from recomputed shipped gate")
        terminal_seen = terminal_seen or expected_terminated or expected_timed_out
        if (
            _boolean(step["reported_nonfinite"], f"{step_context}.reported_nonfinite")
            != trace_steps[index]["reported_nonfinite"]
        ):
            raise ValueError(f"{step_context} nonfinite flag differs from derived trace")
    return reference


def _metric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "max": max(values) if values else None,
    }


def _result_metric(
    values: Sequence[float],
    *,
    threshold: float | None,
    unit: str,
) -> dict[str, Any]:
    summary = _metric_summary(values)
    role = "diagnostic_non_gating" if threshold is None else "gating"
    return {**summary, "unit": unit, "gate_role": role, "threshold_max": threshold}


def _empty_fail_closed_result(error: str) -> dict[str, Any]:
    return {
        "schema": QUALIFICATION_SCHEMA,
        "schema_version": 1,
        "evidence_valid": False,
        "qualification_gate_passed": False,
        "qualification_gate_failures": [error],
        "authorization": "none",
        "dagger_authorized": False,
        "training_authorized": False,
        "deployment_ready": False,
    }


def validate_step1b_report(
    report_path: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Strictly validate evidence and return gates recomputed from raw traces.

    Structural, path, schema, or hash faults raise ``ValueError``. Gate failures
    return a valid qualification result with ``authorization == "none"``.
    """

    if report_path.is_symlink():
        raise ValueError("Step1B report path may not be a symlink")
    if contract_path.is_symlink():
        raise ValueError("Step1B contract path may not be a symlink")
    report_path = report_path.resolve()
    contract_path = contract_path.resolve()
    if not report_path.is_file():
        raise ValueError("Step1B report path must name an evidence file")
    if not contract_path.is_file():
        raise ValueError("Step1B qualification contract is missing")
    report = load_strict_json(report_path)
    config = load_strict_json(contract_path)
    contract = _validate_contract(config)
    if contract["missing_trusted_identity_bindings"]:
        raise ValueError(
            "Step1B trusted identities are not fully pinned: "
            + ", ".join(contract["missing_trusted_identity_bindings"])
        )
    if contract["controller_semantics"]["true23_expert"] is None:
        raise ValueError("Step1B true23 state-feedback controller semantics are not approved")
    contract_raw_sha256 = sha256_file(contract_path)
    contract_payload_sha256 = sha256_bytes(canonical_json_bytes(config))

    _exact_keys(
        report,
        {
            "schema_version",
            "kind",
            "contract_sha256",
            "contract_payload_sha256",
            "campaign_manifest",
            "pairs",
        },
        "Step1B evidence report",
    )
    if report["schema_version"] != 1 or report["kind"] != REPORT_SCHEMA:
        raise ValueError("unsupported Step1B evidence report")
    if _sha256(report["contract_sha256"], "report.contract_sha256") != contract_raw_sha256:
        raise ValueError("Step1B report contract SHA-256 mismatch")
    if _sha256(report["contract_payload_sha256"], "report.contract_payload_sha256") != contract_payload_sha256:
        raise ValueError("Step1B report contract payload SHA-256 mismatch")

    evidence_root = report_path.parent
    campaign_ref, campaign_payload = _validate_json_ref(
        report["campaign_manifest"], evidence_root, "report.campaign_manifest"
    )
    campaign, schedule = _validate_campaign(campaign_payload, evidence_root, contract)
    pairs_raw = _array(report["pairs"], "report.pairs")
    if len(pairs_raw) != len(schedule):
        raise ValueError("report pair count differs from immutable campaign schedule")

    failures: list[str] = []
    timeout_by_clip: dict[str, list[bool]] = {clip_id: [] for clip_id in FROZEN_CLIP_IDS}
    hard_limit_count = 0
    nonfinite_count = 0
    stance_position_errors: list[float] = []
    stance_orientation_errors: list[float] = []
    stance_contact_mismatch_count = 0
    stance_contact_sample_count = 0
    stance_by_clip_foot = {
        clip_id: {
            foot: {"position": [], "orientation": [], "contact_mismatches": 0, "contact_samples": 0}
            for foot in ("left_foot", "right_foot")
        }
        for clip_id in FROZEN_CLIP_IDS
    }
    diagnostic_values: dict[str, list[float]] = {
        "pelvis_position_error_max_m": [],
        "pelvis_orientation_error_max_rad": [],
        "com_position_error_max_m": [],
        "left_hand": [],
        "right_hand": [],
        "head_proxy": [],
        "link_linear_velocity_error_max_mps": [],
        "link_angular_velocity_error_max_radps": [],
    }
    trace_records: list[dict[str, Any]] = []

    for index, expected in enumerate(schedule):
        pair = _mapping(pairs_raw[index], f"report.pairs[{index}]")
        _exact_keys(
            pair,
            {
                "pair_id",
                "clip_id",
                "teacher_trace",
                "true23_trace",
                "teacher_raw_trace",
                "true23_raw_trace",
            },
            f"report.pairs[{index}]",
        )
        if pair["pair_id"] != expected["pair_id"] or pair["clip_id"] != expected["clip_id"]:
            raise ValueError("report pair identity/order differs from immutable campaign schedule")
        teacher_ref, teacher_steps, teacher_timeout = _validate_trace(
            pair["teacher_trace"],
            evidence_root,
            expected,
            "teacher_reference",
            contract,
            f"report.pairs[{index}].teacher_trace",
        )
        true23_ref, true23_steps, true23_timeout = _validate_trace(
            pair["true23_trace"],
            evidence_root,
            expected,
            "true23_expert",
            contract,
            f"report.pairs[{index}].true23_trace",
        )
        teacher_raw_ref = _validate_raw_trace(
            pair["teacher_raw_trace"],
            evidence_root,
            expected,
            "teacher_reference",
            contract,
            teacher_ref,
            teacher_steps,
            None,
            f"report.pairs[{index}].teacher_raw_trace",
        )
        true23_raw_ref = _validate_raw_trace(
            pair["true23_raw_trace"],
            evidence_root,
            expected,
            "true23_expert",
            contract,
            true23_ref,
            true23_steps,
            campaign["true23_expert"]["physics_gains"],
            f"report.pairs[{index}].true23_raw_trace",
        )
        timeout_by_clip[expected["clip_id"]].append(teacher_timeout and true23_timeout)
        trace_records.append(
            {
                "pair_id": expected["pair_id"],
                "clip_id": expected["clip_id"],
                "teacher_trace": teacher_ref,
                "true23_trace": true23_ref,
                "teacher_raw_trace": teacher_raw_ref,
                "true23_raw_trace": true23_raw_ref,
                "paired_timeout": teacher_timeout and true23_timeout,
            }
        )
        for teacher_step, true23_step in zip(teacher_steps, true23_steps, strict=True):
            hard_limit_count += int(teacher_step["hard_joint_limit_violation"])
            hard_limit_count += int(true23_step["hard_joint_limit_violation"])
            nonfinite_count += int(teacher_step["reported_nonfinite"])
            nonfinite_count += int(true23_step["reported_nonfinite"])
            for foot in ("left_foot", "right_foot"):
                foot_metrics = stance_by_clip_foot[expected["clip_id"]][foot]
                mismatch = teacher_step["contacts"][foot] != true23_step["contacts"][foot]
                stance_contact_sample_count += 1
                stance_contact_mismatch_count += int(mismatch)
                foot_metrics["contact_samples"] += 1
                foot_metrics["contact_mismatches"] += int(mismatch)
                if teacher_step["contacts"][foot]:
                    position_error = _distance(
                        teacher_step["semantic_points_m"][foot],
                        true23_step["semantic_points_m"][foot],
                    )
                    orientation_error = _quaternion_distance(
                        teacher_step["semantic_orientations_xyzw"][foot],
                        true23_step["semantic_orientations_xyzw"][foot],
                    )
                    stance_position_errors.append(position_error)
                    stance_orientation_errors.append(orientation_error)
                    foot_metrics["position"].append(position_error)
                    foot_metrics["orientation"].append(orientation_error)
            diagnostic_values["pelvis_position_error_max_m"].append(
                _distance(teacher_step["pelvis"]["position_m"], true23_step["pelvis"]["position_m"])
            )
            diagnostic_values["pelvis_orientation_error_max_rad"].append(
                _quaternion_distance(
                    teacher_step["pelvis"]["orientation_xyzw"],
                    true23_step["pelvis"]["orientation_xyzw"],
                )
            )
            diagnostic_values["com_position_error_max_m"].append(
                _distance(teacher_step["com_position_m"], true23_step["com_position_m"])
            )
            for name in ("left_hand", "right_hand", "head_proxy"):
                diagnostic_values[name].append(
                    _distance(
                        teacher_step["semantic_points_m"][name],
                        true23_step["semantic_points_m"][name],
                    )
                )
            for name in contract["required_link_velocities"]:
                diagnostic_values["link_linear_velocity_error_max_mps"].append(
                    _distance(
                        teacher_step["link_velocities"][name]["linear_mps"],
                        true23_step["link_velocities"][name]["linear_mps"],
                    )
                )
                diagnostic_values["link_angular_velocity_error_max_radps"].append(
                    _distance(
                        teacher_step["link_velocities"][name]["angular_radps"],
                        true23_step["link_velocities"][name]["angular_radps"],
                    )
                )

    total_timeout = sum(sum(values) for values in timeout_by_clip.values())
    total_episodes = sum(len(values) for values in timeout_by_clip.values())
    aggregate_timeout_fraction = total_timeout / total_episodes
    timeout_metrics: dict[str, Any] = {}
    for clip_id in FROZEN_CLIP_IDS:
        successes = sum(timeout_by_clip[clip_id])
        episode_count = len(timeout_by_clip[clip_id])
        fraction = successes / episode_count
        timeout_metrics[clip_id] = {
            "episode_count": episode_count,
            "paired_timeout_count": successes,
            "paired_timeout_fraction": fraction,
        }
        if fraction + _METRIC_EPSILON < contract["min_timeout_fraction"]:
            failures.append(
                f"clip {clip_id!r} paired timeout fraction {fraction:.6f} is below "
                f"{contract['min_timeout_fraction']:.6f}"
            )
    if aggregate_timeout_fraction + _METRIC_EPSILON < contract["min_timeout_fraction"]:
        failures.append(
            f"aggregate paired timeout fraction {aggregate_timeout_fraction:.6f} is below "
            f"{contract['min_timeout_fraction']:.6f}"
        )
    if hard_limit_count:
        failures.append(f"hard joint-limit violation count is {hard_limit_count}, expected zero")
    if nonfinite_count:
        failures.append(f"nonfinite trace sample count is {nonfinite_count}, expected zero")
    if not stance_position_errors or not stance_orientation_errors or not stance_contact_sample_count:
        failures.append("stance-foot evidence is unavailable")
    else:
        stance_thresholds = contract["stance_foot_thresholds"]
        if max(stance_position_errors) > stance_thresholds["position_error_max_m"] + _METRIC_EPSILON:
            failures.append(
                f"stance-foot position error max {max(stance_position_errors):.9g} m exceeds "
                f"{stance_thresholds['position_error_max_m']:.9g} m"
            )
        if max(stance_orientation_errors) > stance_thresholds["orientation_error_max_rad"] + _METRIC_EPSILON:
            failures.append(
                f"stance-foot orientation error max {max(stance_orientation_errors):.9g} rad exceeds "
                f"{stance_thresholds['orientation_error_max_rad']:.9g} rad"
            )
        contact_fraction = stance_contact_mismatch_count / stance_contact_sample_count
        if contact_fraction > stance_thresholds["contact_mismatch_fraction_max"] + _METRIC_EPSILON:
            failures.append(
                f"stance-foot contact mismatch fraction {contact_fraction:.9g} exceeds "
                f"{stance_thresholds['contact_mismatch_fraction_max']:.9g}"
            )
    per_clip_foot_metrics: dict[str, Any] = {}
    stance_thresholds = contract["stance_foot_thresholds"]
    for clip_id, feet in stance_by_clip_foot.items():
        per_clip_foot_metrics[clip_id] = {}
        for foot, values in feet.items():
            positions = values["position"]
            orientations = values["orientation"]
            contact_samples = values["contact_samples"]
            contact_mismatches = values["contact_mismatches"]
            contact_fraction_for_foot = contact_mismatches / contact_samples
            per_clip_foot_metrics[clip_id][foot] = {
                "position_error_m": _metric_summary(positions),
                "orientation_error_rad": _metric_summary(orientations),
                "contact_sample_count": contact_samples,
                "contact_mismatch_count": contact_mismatches,
                "contact_mismatch_fraction": contact_fraction_for_foot,
            }
            if not positions or not orientations:
                failures.append(f"clip {clip_id!r} {foot} has no teacher stance samples")
                continue
            if max(positions) > stance_thresholds["position_error_max_m"] + _METRIC_EPSILON:
                failures.append(f"clip {clip_id!r} {foot} stance position gate failed")
            if max(orientations) > stance_thresholds["orientation_error_max_rad"] + _METRIC_EPSILON:
                failures.append(f"clip {clip_id!r} {foot} stance orientation gate failed")
            if contact_fraction_for_foot > stance_thresholds["contact_mismatch_fraction_max"] + _METRIC_EPSILON:
                failures.append(f"clip {clip_id!r} {foot} contact mismatch gate failed")

    diagnostic_thresholds = contract["diagnostic_thresholds"]
    diagnostic_metrics = {
        "pelvis_position_error_m": _result_metric(
            diagnostic_values["pelvis_position_error_max_m"],
            threshold=diagnostic_thresholds["pelvis_position_error_max_m"],
            unit="m",
        ),
        "pelvis_orientation_error_rad": _result_metric(
            diagnostic_values["pelvis_orientation_error_max_rad"],
            threshold=diagnostic_thresholds["pelvis_orientation_error_max_rad"],
            unit="rad",
        ),
        "com_position_error_m": _result_metric(
            diagnostic_values["com_position_error_max_m"],
            threshold=diagnostic_thresholds["com_position_error_max_m"],
            unit="m",
        ),
        "semantic_position_error_m": {
            name: _result_metric(
                diagnostic_values[name],
                threshold=diagnostic_thresholds["semantic_position_error_max_m"][name],
                unit="m",
            )
            for name in ("left_hand", "right_hand", "head_proxy")
        },
        "link_linear_velocity_error_mps": _result_metric(
            diagnostic_values["link_linear_velocity_error_max_mps"],
            threshold=diagnostic_thresholds["link_linear_velocity_error_max_mps"],
            unit="m/s",
        ),
        "link_angular_velocity_error_radps": _result_metric(
            diagnostic_values["link_angular_velocity_error_max_radps"],
            threshold=diagnostic_thresholds["link_angular_velocity_error_max_radps"],
            unit="rad/s",
        ),
    }

    def apply_optional_gate(path: str, metric: Mapping[str, Any]) -> None:
        threshold = metric["threshold_max"]
        maximum = metric["max"]
        if threshold is not None and maximum is not None and maximum > threshold + _METRIC_EPSILON:
            failures.append(f"{path} max {maximum:.9g} exceeds configured threshold {threshold:.9g}")

    for name in ("pelvis_position_error_m", "pelvis_orientation_error_rad", "com_position_error_m"):
        apply_optional_gate(name, diagnostic_metrics[name])
    for name, metric in diagnostic_metrics["semantic_position_error_m"].items():
        apply_optional_gate(f"semantic_position_error_m.{name}", metric)
    apply_optional_gate("link_linear_velocity_error_mps", diagnostic_metrics["link_linear_velocity_error_mps"])
    apply_optional_gate(
        "link_angular_velocity_error_radps", diagnostic_metrics["link_angular_velocity_error_radps"]
    )

    failures.append(
        "authorization blocked: independent MuJoCo FK/COM/semantic-point recomputation "
        "from raw qpos is not implemented"
    )

    passed = not failures
    contact_fraction = (
        stance_contact_mismatch_count / stance_contact_sample_count if stance_contact_sample_count else None
    )
    return {
        "schema": QUALIFICATION_SCHEMA,
        "schema_version": 1,
        "evidence_valid": True,
        "qualification_gate_passed": passed,
        "qualification_gate_failures": failures,
        "authorization": PASS_AUTHORIZATION if passed else "none",
        "dagger_authorized": False,
        "training_authorized": False,
        "deployment_ready": False,
        "declared_categories": ["idle"],
        "frozen_clip_ids": list(FROZEN_CLIP_IDS),
        "coverage": {
            "robustness_coverage": contract["robustness_coverage"],
            "horizon_steps": REQUIRED_HORIZON_STEPS,
            "episode_count": total_episodes,
            "paired_timeout_count": total_timeout,
            "paired_timeout_fraction": aggregate_timeout_fraction,
            "per_clip": timeout_metrics,
        },
        "hard_gates": {
            "hard_joint_limit_violation_count": hard_limit_count,
            "nonfinite_trace_sample_count": nonfinite_count,
            "stance_foot_position_error_m": _metric_summary(stance_position_errors),
            "stance_foot_orientation_error_rad": _metric_summary(stance_orientation_errors),
            "stance_foot_contact_sample_count": stance_contact_sample_count,
            "stance_foot_contact_mismatch_count": stance_contact_mismatch_count,
            "stance_foot_contact_mismatch_fraction": contact_fraction,
            "per_clip_foot": per_clip_foot_metrics,
        },
        "thresholds": {
            "min_timeout_fraction_aggregate_and_per_clip": contract["min_timeout_fraction"],
            "stance_foot": contract["stance_foot_thresholds"],
            "selected_termination_gate": contract["selected_termination_gate"],
        },
        "diagnostics": diagnostic_metrics,
        "provenance": {
            "contract": {
                "sha256": contract_raw_sha256,
                "payload_sha256": contract_payload_sha256,
            },
            "report": {
                "sha256": sha256_file(report_path),
                "payload_sha256": sha256_bytes(canonical_json_bytes(report)),
            },
            "campaign_manifest": campaign_ref,
            "campaign": campaign,
            "traces": trace_records,
        },
    }


def qualify_step1b_report_fail_closed(
    report_path: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Return an authorization-none result for every invalid evidence error."""

    try:
        return validate_step1b_report(report_path, contract_path=contract_path)
    except (OSError, TypeError, ValueError) as exc:
        return _empty_fail_closed_result(f"invalid evidence: {exc}")


validate_g1_true23_idle_step1b_report = validate_step1b_report


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


__all__ = [
    "CAMPAIGN_SCHEMA",
    "CONTRACT_SCHEMA",
    "DEFAULT_CONTRACT_PATH",
    "DISTURBANCE_SCHEMA",
    "FROZEN_CLIP_IDS",
    "INITIAL_STATE_SCHEMA",
    "PASS_AUTHORIZATION",
    "QUALIFICATION_SCHEMA",
    "RAW_AUDIT_SCHEMA",
    "REPORT_SCHEMA",
    "TRACE_SCHEMA",
    "TRUE23_CONTROLLER_DESCRIPTOR",
    "TRUE23_CONTROLLER_SEMANTICS",
    "canonical_json_bytes",
    "load_strict_json",
    "qualify_step1b_report_fail_closed",
    "sha256_bytes",
    "sha256_file",
    "validate_step1b_report",
    "validate_g1_true23_idle_step1b_report",
    "write_json_atomic",
]
