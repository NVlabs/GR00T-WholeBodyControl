"""Produce honest paired MuJoCo evidence for the idle-only true23 Step 1B gate.

The two arms deliberately use different controllers:

* ``teacher_reference`` runs the exact pinned low-latency 29-DoF neural
  teacher (teleop encoder, FSQ-32, and state-feedback decoder).
* ``true23_expert`` replays the immutable schema-v6 ``action_target_native``
  reference through the exact safe-target transform and live joint PD at every
  physics substep.  It never runs task-space IK online.

Episodes always advance the full fixed horizon.  A shipped MJLab termination
is latched for reporting only; MuJoCo is neither reset nor short-circuited.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np

from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    LOW_LATENCY_RELEASE_SHA256,
    SOURCE_IL29_JOINT_NAMES,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    safe_target_transform_contract,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_29dof_low_latency_teacher import ExactLowLatencyTeacher
from gear_sonic.utils.g1_true23_step1b_qualification import (
    CAMPAIGN_SCHEMA,
    DISTURBANCE_SCHEMA,
    FIXED_SEEDS_BY_CLIP,
    FROZEN_CLIP_IDS,
    INITIAL_STATE_SCHEMA,
    INITIAL_STATE_SCOPES,
    INPUT_ACTION_LAW_SCHEMA,
    POST_TERMINATION_POLICY,
    RAW_AUDIT_SCHEMA,
    REFERENCE_SCHEDULE_SCHEMA,
    REPORT_SCHEMA,
    RUNTIME_CONFIG_SCHEMA,
    SHIPPED_EE_TERMINATION_THRESHOLD_M,
    TEACHER_CONTROLLER_SEMANTICS,
    TRACE_SCHEMA,
    TRUE23_CONTROLLER_DESCRIPTOR,
    TRUE23_CONTROLLER_SEMANTICS,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_RELPATH = "gear_sonic/utils/g1_true23_step1b_mujoco.py"
SOURCE_MODEL_RELPATH = "gear_sonic/data/robots/g1/g1_29dof.xml"
TARGET_MODEL_RELPATH = "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
SIM_CONFIG_RELPATH = "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
STEP1A_SUPPORT_RELPATH = "gear_sonic/config/sim_validation/g1_23dof_task_space_supported_idle_v1.json"
LOW_LATENCY_CONFIG_RELPATH = "low_latency/config.yaml"
LOW_LATENCY_MODEL_CONFIG_RELPATH = "low_latency/model_config.yaml"

CONTROL_HZ = 50
HORIZON_STEPS = 500
EPISODES_PER_CLIP = 1
REFERENCE_CONTINUATION = "terminal_hold_last"
TEACHER_PHYSICS_HZ = 200
TRUE23_PHYSICS_HZ = 500
TEACHER_DECIMATION = 4
TRUE23_DECIMATION = 10
TELEOP_INPUT_DIM = 267
TOKEN_DIM = 64
PROPRIO_DIM = 930
HISTORY_LENGTH = 10
TEACHER_ACTION_CLIP = 20.0
RAW_TRACE_KIND = RAW_AUDIT_SCHEMA
DIAGNOSTIC_BUNDLE_SCHEMA = "g1_true23_idle_step1b_mujoco_diagnostic_bundle_v1"
REFERENCE_INPUT_KIND = "schema_v6_action_target_native"

TRACE_HEADER_KEYS = frozenset(
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
    }
)
TRACE_STEP_KEYS = frozenset(
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
    }
)
RAW_STEP_KEYS = frozenset(
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
        "applied_torque_last_hardware_nm",
        "applied_torque_first_hardware_nm",
        "applied_torque_peak_abs_hardware_nm",
        "raw_contact_geom_pairs",
        "torso_reference",
        "ee_reference",
        "termination_terms",
        "reported_nonfinite",
    }
)
RAW_HEADER_KEYS = frozenset(
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
    }
)

SOURCE_MODEL_SHA256 = "386b1bb9ea5b69ccd6fd0283a73ffea1ee052df95564e23a780125fbcbe2c645"
TARGET_MODEL_SHA256 = "16e304c970bfc68783ea69d01d05192e6bd9d83d62f6ee4aac0ac72ff18db612"

LOWER_BODY_IL29_INDICES = (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18)
IL29_TO_HARDWARE29 = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)
HARDWARE29_TO_IL29 = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)
SOURCE_HARDWARE_JOINT_NAMES = tuple(SOURCE_IL29_JOINT_NAMES[index] for index in HARDWARE29_TO_IL29)

SOURCE_DEFAULT_Q_HARDWARE = np.asarray(
    (
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ),
    dtype=np.float64,
)

_MOTOR = {
    "5020": (14.25062309787429, 0.907222843292423, 0.43857731392336724, 0.003609725),
    "7520_14": (40.179238471373175, 2.5578897650279457, 0.5475464652142304, 0.010177520),
    "7520_22": (99.09842777666113, 6.3088018534966395, 0.3506614663788243, 0.025101925),
    "4010": (16.77832748089279, 1.06814150219, 0.07450087032950714, 0.00425),
}

_SEMANTIC_BINDINGS = {
    "left_foot": {
        "teacher_reference": "left_ankle_roll_link",
        "true23_expert": "left_ankle_roll_link",
        "offset": (0.0, 0.0, 0.0),
    },
    "right_foot": {
        "teacher_reference": "right_ankle_roll_link",
        "true23_expert": "right_ankle_roll_link",
        "offset": (0.0, 0.0, 0.0),
    },
    "left_hand": {
        "teacher_reference": "left_wrist_yaw_link",
        "true23_expert": "left_wrist_roll_rubber_hand",
        "offset": (0.18, -0.025, 0.0),
    },
    "right_hand": {
        "teacher_reference": "right_wrist_yaw_link",
        "true23_expert": "right_wrist_roll_rubber_hand",
        "offset": (0.18, 0.025, 0.0),
    },
    "head_proxy": {
        "teacher_reference": "torso_link",
        "true23_expert": "torso_link",
        "offset": (0.0, 0.0, 0.35),
    },
}
_SHIPPED_EE = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
)


def _finite_array(value: Any, shape: tuple[int, ...], context: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{context} must be finite with shape {shape}, got {result.shape}")
    return result


def _json_float(value: float) -> float:
    result = float(value)
    return result if math.isfinite(result) else 0.0


def _json_vector(value: Sequence[float] | np.ndarray) -> list[float]:
    return [_json_float(item) for item in np.asarray(value).reshape(-1)]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_ref(root: Path, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    _write_json_atomic(path, payload)
    return {
        "file": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _binary_ref(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "file": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _materialize_file(source: Path, destination: Path) -> None:
    """Contain an immutable input, preferring a same-filesystem hardlink."""

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_binary_ref(root: Path, source: Path, relative_destination: str) -> dict[str, Any]:
    destination = root / relative_destination
    _materialize_file(source, destination)
    return _binary_ref(root, destination)


def _copy_json_ref(root: Path, source: Path, relative_destination: str) -> dict[str, Any]:
    destination = root / relative_destination
    _materialize_file(source, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON evidence root must be an object: {source}")
    return {
        "file": destination.relative_to(root).as_posix(),
        "sha256": sha256_file(destination),
        "payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _repository_binding(relative_path: str) -> dict[str, Any]:
    path = REPOSITORY_ROOT / relative_path
    return {
        "relpath": relative_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def terminal_hold_indices(
    frame_count: int,
    horizon_steps: int = HORIZON_STEPS,
    *,
    start_frame: int = 0,
) -> tuple[int, ...]:
    """Return a monotone terminal-held schedule; looping is impossible."""

    if frame_count < 1 or horizon_steps < 1:
        raise ValueError("frame_count and horizon_steps must be positive")
    if start_frame < 0 or start_frame >= frame_count:
        raise ValueError("start_frame lies outside the reference")
    return tuple(min(start_frame + step, frame_count - 1) for step in range(horizon_steps))


def terminal_hold_forward_velocity(values: np.ndarray, control_hz: int = CONTROL_HZ) -> np.ndarray:
    """Forward derivative whose terminal/held frame is exactly stationary."""

    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 2 or len(value) < 1 or not np.isfinite(value).all():
        raise ValueError("reference values must be a non-empty finite matrix")
    result = np.zeros_like(value)
    if len(value) > 1:
        result[:-1] = (value[1:] - value[:-1]) * float(control_hz)
    return result


def term_major_history(frames: Sequence[Sequence[float]]) -> np.ndarray:
    """Flatten ten measured 93-D frames into the released 930-D term order."""

    array = _finite_array(frames, (HISTORY_LENGTH, 93), "measured proprio history")
    result = np.concatenate(
        (
            array[:, 0:3].reshape(-1),
            array[:, 3:32].reshape(-1),
            array[:, 32:61].reshape(-1),
            array[:, 61:90].reshape(-1),
            array[:, 90:93].reshape(-1),
        )
    ).astype(np.float32)
    if result.shape != (PROPRIO_DIM,):
        raise AssertionError("released proprio layout drift")
    return result


def true23_controller_descriptor() -> dict[str, Any]:
    """Return the exact honest two-layer schema-v6/live-PD controller label."""

    return dict(TRUE23_CONTROLLER_DESCRIPTOR)


def terminal_event_flags(
    *,
    already_latched: bool,
    gate_terminated: bool,
    final_step: bool,
) -> tuple[bool, bool, bool, bool]:
    """Return event, timeout, post-event, next-latch without changing physics."""

    post_termination = bool(already_latched)
    terminated = bool(gate_terminated and not already_latched)
    timed_out = bool(final_step and not already_latched and not terminated)
    return terminated, timed_out, post_termination, bool(already_latched or terminated or timed_out)


def shipped_time_out_term(*, already_latched: bool, final_step: bool) -> bool:
    """Reproduce mdp.time_out independently of other final-step terms."""

    return bool(final_step and not already_latched)


def build_input_action_law(arm: str) -> dict[str, Any]:
    if arm == "teacher_reference":
        action_law = TEACHER_CONTROLLER_SEMANTICS
        output_space = "native_il29_actions"
    elif arm == "true23_expert":
        action_law = TRUE23_CONTROLLER_SEMANTICS
        output_space = "joint_torque"
    else:
        raise ValueError(f"unknown campaign arm: {arm}")
    return {
        "schema_version": 1,
        "kind": INPUT_ACTION_LAW_SCHEMA,
        "arm": arm,
        "action_law": action_law,
        "state_dependence": "live_robot_state_feedback",
        "input_fields": ["robot_state", "reference"],
        "output_space": output_space,
    }


@dataclass(frozen=True)
class EpisodeSpec:
    pair_id: str
    clip_id: str
    seed: int
    episode: int
    root_position_delta_m: tuple[float, float, float]
    root_yaw_delta_rad: float
    joint_delta_retained_hardware_rad: tuple[float, ...]
    disturbance_step: int
    disturbance_delta_world: tuple[float, ...]


def _seed_uniform(seed: int, label: str, lower: float, upper: float) -> float:
    digest = hashlib.sha256(f"step1b:{seed}:{label}".encode("ascii")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return lower + (upper - lower) * unit


def deterministic_episode_specs(
    clip_id: str,
    seeds: Sequence[int],
    *,
    disturbance_delta: Sequence[float] = (0.0,) * 6,
    enable_initial_perturbation: bool = False,
) -> tuple[EpisodeSpec, ...]:
    """Generate frozen seed-driven shared perturbations without RNG state."""

    disturbance = tuple(float(value) for value in disturbance_delta)
    if len(disturbance) != 6 or not all(math.isfinite(value) for value in disturbance):
        raise ValueError("disturbance_delta must contain six finite values")
    result: list[EpisodeSpec] = []
    for episode, raw_seed in enumerate(seeds):
        seed = int(raw_seed)
        perturbed = enable_initial_perturbation and episode != 0
        retained = tuple(
            _seed_uniform(seed, f"q:{index}", -0.01, 0.01) if perturbed else 0.0 for index in range(23)
        )
        position = tuple(
            _seed_uniform(seed, f"root:{axis}", -bound, bound) if perturbed else 0.0
            for axis, bound in enumerate((0.005, 0.005, 0.002))
        )
        yaw = _seed_uniform(seed, "root:yaw", -0.01, 0.01) if perturbed else 0.0
        result.append(
            EpisodeSpec(
                pair_id=f"{clip_id}--{episode:02d}--{seed}",
                clip_id=clip_id,
                seed=seed,
                episode=episode,
                root_position_delta_m=position,  # type: ignore[arg-type]
                root_yaw_delta_rad=yaw,
                joint_delta_retained_hardware_rad=retained,
                disturbance_step=250,
                disturbance_delta_world=(0.0,) * 6 if episode == 0 else disturbance,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class LoadedClip:
    clip_id: str
    source_csv_path: Path
    motion_path: Path
    expert_path: Path
    report_path: Path
    source_root_pos: np.ndarray
    source_root_quat_wxyz: np.ndarray
    source_joint_pos_hardware: np.ndarray
    source_joint_vel_hardware: np.ndarray
    target_root_pos: np.ndarray
    target_root_quat_wxyz: np.ndarray
    target_joint_pos_hardware: np.ndarray
    target_joint_vel_hardware: np.ndarray
    target_action_native: np.ndarray
    target_contact_flags: np.ndarray
    target_body_pos_w: np.ndarray
    target_body_quat_wxyz: np.ndarray
    frame_count: int

    def frame_index(self, step: int) -> int:
        return min(step, self.frame_count - 1)

    def future_indices(self, step: int) -> np.ndarray:
        return np.minimum(np.arange(step, step + 10), self.frame_count - 1)

    def terminal_held(self, step: int) -> bool:
        return step >= self.frame_count - 1


def _npz_array(archive: Any, key: str, shape: tuple[int, ...], context: str) -> np.ndarray:
    if key not in archive.files:
        raise ValueError(f"{context} lacks {key!r}")
    return _finite_array(archive[key], shape, f"{context}.{key}")


def load_frozen_clip(
    *,
    clip_id: str,
    source_csv_path: Path,
    motion_path: Path,
    expert_path: Path,
    report_path: Path,
    source_model: Any,
    target_model: Any,
    expected_frame_count: int,
) -> LoadedClip:
    """Load and cross-bind one exact schema-v6 triplet and its source29 CSV."""

    for path in (source_csv_path, motion_path, expert_path, report_path):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 6 or report.get("clip_id") != clip_id:
        raise ValueError("retarget report is not the requested schema-v6 clip")
    if report.get("motion_output_sha256") != sha256_file(motion_path.resolve()):
        raise ValueError("retarget motion hash differs from schema-v6 report")
    if report.get("expert_output_sha256") != sha256_file(expert_path.resolve()):
        raise ValueError("retarget expert hash differs from schema-v6 report")
    if report.get("source_model_sha256") != SOURCE_MODEL_SHA256:
        raise ValueError("schema-v6 report does not bind exact source29 model")
    if report.get("target_model_sha256") != TARGET_MODEL_SHA256:
        raise ValueError("schema-v6 report does not bind exact true23 model")
    if report.get("serialization_constraint_audit_passed") is not True:
        raise ValueError("schema-v6 artifact lacks float32 serialization certificate")

    source_names = _model_joint_names(source_model)
    if source_names != SOURCE_HARDWARE_JOINT_NAMES:
        raise ValueError("source29 MuJoCo joint order differs from pinned hardware order")
    from gear_sonic.scripts.retarget_g1_29dof_to_23dof_task_space import (
        _load_csv_trajectory,
    )

    source_root, source_quat, source_q = _load_csv_trajectory(
        source_csv_path.resolve(),
        source_joint_names=source_names,
        fps_source=120.0,
        fps_target=float(CONTROL_HZ),
    )
    if source_q.shape != (expected_frame_count, 29):
        raise ValueError("source29 resampling differs from frozen frame count")
    source_dq = terminal_hold_forward_velocity(source_q)

    with (
        np.load(motion_path.resolve(), allow_pickle=False) as motion,
        np.load(expert_path.resolve(), allow_pickle=False) as expert,
    ):
        if float(np.asarray(motion["fps"]).reshape(-1)[0]) != CONTROL_HZ:
            raise ValueError("schema-v6 motion is not exactly 50 Hz")
        if int(np.asarray(expert["schema_version"]).reshape(-1)[0]) != 6:
            raise ValueError("expert NPZ is not schema version 6")
        target_q = _npz_array(motion, "joint_pos", (expected_frame_count, 23), "motion")
        target_dq = _npz_array(motion, "joint_vel", (expected_frame_count, 23), "motion")
        action = np.asarray(expert["action_target_native"])
        if action.shape != (expected_frame_count, 23) or action.dtype != np.float32:
            raise ValueError("schema-v6 action_target_native must be float32 [frames,23]")
        if not np.isfinite(action).all():
            raise ValueError("schema-v6 actions contain NaN or Inf")
        _, reconstructed_target = safe_target_transform_numpy(action)
        if not np.allclose(reconstructed_target, target_q, rtol=0.0, atol=2.0e-6):
            raise ValueError("schema-v6 safe-target transform does not reproduce motion q")
        target_root = _npz_array(expert, "root_pos_w", (expected_frame_count, 3), "expert")
        target_quat = _npz_array(expert, "root_quat_wxyz", (expected_frame_count, 4), "expert")
        contacts = np.asarray(expert["contact_flags"], dtype=np.bool_)
        if contacts.shape != (expected_frame_count, 2):
            raise ValueError("schema-v6 contact_flags must have shape [frames,2]")
        body_count = int(target_model.nbody) - 1
        body_pos = _npz_array(motion, "body_pos_w", (expected_frame_count, body_count, 3), "motion")
        body_quat = _npz_array(motion, "body_quat_w", (expected_frame_count, body_count, 4), "motion")
    return LoadedClip(
        clip_id=clip_id,
        source_csv_path=source_csv_path.resolve(),
        motion_path=motion_path.resolve(),
        expert_path=expert_path.resolve(),
        report_path=report_path.resolve(),
        source_root_pos=source_root,
        source_root_quat_wxyz=source_quat,
        source_joint_pos_hardware=source_q,
        source_joint_vel_hardware=source_dq,
        target_root_pos=target_root,
        target_root_quat_wxyz=target_quat,
        target_joint_pos_hardware=target_q,
        target_joint_vel_hardware=target_dq,
        target_action_native=action.copy(),
        target_contact_flags=contacts,
        target_body_pos_w=body_pos,
        target_body_quat_wxyz=body_quat,
        frame_count=expected_frame_count,
    )


def _model_joint_names(model: Any) -> tuple[str, ...]:
    import mujoco

    result: list[str] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise ValueError("MuJoCo model contains unnamed joint")
        result.append(str(name))
    return tuple(result)


def _quaternion_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion_wxyz, dtype=np.float64)
    value /= np.linalg.norm(value)
    w, x, y, z = value
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    )


def _quaternion_conjugate(value: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(value, dtype=np.float64)
    return np.asarray((w, -x, -y, -z), dtype=np.float64)


def _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64)
    return np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _rotation_6d(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    return _quaternion_matrix(quaternion_wxyz)[:, :2].reshape(-1)


def _projected_gravity(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    return _quaternion_matrix(quaternion_wxyz).T @ np.asarray((0.0, 0.0, -1.0))


def world_angular_velocity_to_body(
    quaternion_wxyz: Sequence[float], angular_velocity_world: Sequence[float]
) -> np.ndarray:
    """Express a world-frame angular velocity in the measured root frame."""

    return _quaternion_matrix(quaternion_wxyz).T @ np.asarray(angular_velocity_world, dtype=np.float64)


def _yaw_perturb(quaternion_wxyz: Sequence[float], yaw: float) -> np.ndarray:
    half = yaw * 0.5
    delta = np.asarray((math.cos(half), 0.0, 0.0, math.sin(half)))
    result = _quaternion_multiply(delta, quaternion_wxyz)
    return result / np.linalg.norm(result)


class ReferenceKinematics:
    """Exact source29 FK plus serialized schema-v6 true23 reference poses."""

    _SOURCE_BODIES = (
        "pelvis",
        "torso_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )

    def __init__(self, module: Any, source_model: Any, target_model: Any, clip: LoadedClip):
        self.module = module
        self.clip = clip
        self.source_model = source_model
        self.target_model = target_model
        self._source_positions: dict[str, np.ndarray] = {}
        self._source_quaternions: dict[str, np.ndarray] = {}
        data = module.MjData(source_model)
        body_ids = {name: self._body_id(source_model, name) for name in self._SOURCE_BODIES}
        for name in self._SOURCE_BODIES:
            self._source_positions[name] = np.empty((clip.frame_count, 3), dtype=np.float64)
            self._source_quaternions[name] = np.empty((clip.frame_count, 4), dtype=np.float64)
        for frame in range(clip.frame_count):
            data.qpos[:3] = clip.source_root_pos[frame]
            data.qpos[3:7] = clip.source_root_quat_wxyz[frame]
            data.qpos[7:] = clip.source_joint_pos_hardware[frame]
            data.qvel[:] = 0.0
            module.mj_forward(source_model, data)
            for name, body_id in body_ids.items():
                self._source_positions[name][frame] = data.xpos[body_id]
                self._source_quaternions[name][frame] = data.xquat[body_id]
        self._target_body_ids = {
            name: self._body_id(target_model, name)
            for name in (
                "pelvis",
                "torso_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_roll_rubber_hand",
                "right_wrist_roll_rubber_hand",
            )
        }

    def _body_id(self, model: Any, name: str) -> int:
        body_id = self.module.mj_name2id(model, self.module.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"model lacks required reference body {name!r}")
        return int(body_id)

    def source_body(self, name: str, frame: int) -> tuple[np.ndarray, np.ndarray]:
        index = self.clip.frame_index(frame)
        return (
            self._source_positions[name][index].copy(),
            self._source_quaternions[name][index].copy(),
        )

    def target_body(self, name: str, frame: int) -> tuple[np.ndarray, np.ndarray]:
        index = self.clip.frame_index(frame)
        body_index = self._target_body_ids[name] - 1
        return (
            self.clip.target_body_pos_w[index, body_index].copy(),
            self.clip.target_body_quat_wxyz[index, body_index].copy(),
        )

    def arm_body(self, arm: str, name: str, frame: int) -> tuple[np.ndarray, np.ndarray]:
        if arm == "teacher_reference":
            source_name = {
                "left_wrist_roll_rubber_hand": "left_wrist_yaw_link",
                "right_wrist_roll_rubber_hand": "right_wrist_yaw_link",
            }.get(name, name)
            return self.source_body(source_name, frame)
        if arm == "true23_expert":
            return self.target_body(name, frame)
        raise ValueError(f"unknown campaign arm: {arm}")

    def teleop_encoder_input(
        self,
        step: int,
        measured_pelvis_quaternion_wxyz: Sequence[float],
    ) -> np.ndarray:
        """Build exact 267 values from source29 future frames and live orientation."""

        indices = self.clip.future_indices(step)
        q_il29 = self.clip.source_joint_pos_hardware[:, np.asarray(IL29_TO_HARDWARE29, dtype=np.int64)]
        dq_il29 = self.clip.source_joint_vel_hardware[:, np.asarray(IL29_TO_HARDWARE29, dtype=np.int64)]
        lower = np.asarray(LOWER_BODY_IL29_INDICES, dtype=np.int64)
        future = np.concatenate((q_il29[indices][:, lower].reshape(-1), dq_il29[indices][:, lower].reshape(-1)))

        frame = self.clip.frame_index(step)
        anchor_pos = self.clip.source_root_pos[frame]
        anchor_quat = self.clip.source_root_quat_wxyz[frame]
        anchor_inverse_rotation = _quaternion_matrix(anchor_quat).T
        vr_positions: list[float] = []
        vr_quaternions: list[float] = []
        for body_name, offset in (
            ("left_wrist_yaw_link", (0.18, -0.025, 0.0)),
            ("right_wrist_yaw_link", (0.18, 0.025, 0.0)),
            ("torso_link", (0.0, 0.0, 0.35)),
        ):
            body_pos, body_quat = self.source_body(body_name, frame)
            point = body_pos + _quaternion_matrix(body_quat) @ np.asarray(offset)
            vr_positions.extend((anchor_inverse_rotation @ (point - anchor_pos)).tolist())
            local_quat = _quaternion_multiply(_quaternion_conjugate(anchor_quat), body_quat)
            vr_quaternions.extend(local_quat.tolist())
        relative_anchor = _quaternion_multiply(_quaternion_conjugate(measured_pelvis_quaternion_wxyz), anchor_quat)
        result = np.concatenate(
            (
                future,
                np.asarray(vr_positions),
                np.asarray(vr_quaternions),
                _rotation_6d(relative_anchor),
            )
        ).astype(np.float32)
        if result.shape != (TELEOP_INPUT_DIM,) or not np.isfinite(result).all():
            raise AssertionError("teleop reference did not produce exact 267-D input")
        return result


@dataclass(frozen=True)
class PhysicsParameters:
    timestep_s: float
    decimation: int
    kp: np.ndarray
    kd: np.ndarray
    effort: np.ndarray
    armature: np.ndarray
    action_scale: np.ndarray | None


def _source_motor_class(name: str) -> str:
    if "hip_pitch" in name or "hip_roll" in name or "knee" in name:
        return "7520_22"
    if "hip_yaw" in name or name == "waist_yaw_joint":
        return "7520_14"
    if "wrist_pitch" in name or "wrist_yaw" in name:
        return "4010"
    return "5020"


def source29_physics_parameters() -> PhysicsParameters:
    classes = tuple(_source_motor_class(name) for name in SOURCE_HARDWARE_JOINT_NAMES)
    kp = np.asarray([_MOTOR[name][0] for name in classes], dtype=np.float64)
    kd = np.asarray([_MOTOR[name][1] for name in classes], dtype=np.float64)
    armature = np.asarray([_MOTOR[name][3] for name in classes], dtype=np.float64)
    scale = np.asarray([_MOTOR[name][2] for name in classes], dtype=np.float64)
    doubled = np.asarray(
        [
            "ankle" in name or name in {"waist_roll_joint", "waist_pitch_joint"}
            for name in SOURCE_HARDWARE_JOINT_NAMES
        ]
    )
    kp[doubled] *= 2.0
    kd[doubled] *= 2.0
    armature[doubled] *= 2.0
    effort = np.asarray(
        [
            139.0
            if any(term in name for term in ("hip_pitch", "hip_roll", "knee"))
            else 88.0
            if "hip_yaw" in name or name == "waist_yaw_joint"
            else 50.0
            if "ankle" in name or name in {"waist_roll_joint", "waist_pitch_joint"}
            else 5.0
            if "wrist_pitch" in name or "wrist_yaw" in name
            else 25.0
            for name in SOURCE_HARDWARE_JOINT_NAMES
        ],
        dtype=np.float64,
    )
    return PhysicsParameters(0.005, TEACHER_DECIMATION, kp, kd, effort, armature, scale)


def prepare_source29_model(path: Path) -> tuple[Any, Any, PhysicsParameters]:
    """Compile exact source29 and apply the released IsaacLab actuator physics."""

    import mujoco

    resolved = path.resolve()
    if sha256_file(resolved) != SOURCE_MODEL_SHA256:
        raise ValueError("source29 XML differs from pinned model")
    model = mujoco.MjModel.from_xml_path(str(resolved))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("source29 compiled dimensions changed")
    if _model_joint_names(model) != SOURCE_HARDWARE_JOINT_NAMES:
        raise ValueError("source29 compiled joint order changed")
    names = tuple(str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)) for index in range(model.nu))
    if names != SOURCE_HARDWARE_JOINT_NAMES:
        raise ValueError("source29 actuator order changed")
    physics = source29_physics_parameters()
    model.opt.timestep = physics.timestep_s
    model.opt.gravity[:] = (0.0, 0.0, -9.81)
    joint_ids = np.arange(1, model.njnt, dtype=np.int64)
    dof_ids = model.jnt_dofadr[joint_ids]
    model.dof_armature[dof_ids] = physics.armature
    model.dof_damping[dof_ids] = 0.0
    model.dof_frictionloss[dof_ids] = 0.0
    model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:, 0] = -physics.effort
    model.actuator_forcerange[:, 1] = physics.effort
    return mujoco, model, physics


def prepare_true23_model(path: Path, config_path: Path) -> tuple[Any, Any, PhysicsParameters]:
    """Compile exact true23 with the repository's hash-bound 500-Hz physics."""

    from gear_sonic.utils.g1_23dof_mujoco_sim2sim import (
        load_sim2sim_config,
        prepare_mujoco_model,
    )

    resolved = path.resolve()
    if sha256_file(resolved) != TARGET_MODEL_SHA256:
        raise ValueError("true23 XML differs from pinned model")
    config = load_sim2sim_config(config_path.resolve())
    module, model, _ = prepare_mujoco_model(mjcf_path=resolved, config=config)
    physics = config["physics"]
    return (
        module,
        model,
        PhysicsParameters(
            float(physics["timestep_s"]),
            int(physics["control_decimation"]),
            np.asarray(physics["kp_hardware"], dtype=np.float64),
            np.asarray(physics["kd_hardware"], dtype=np.float64),
            np.asarray(physics["effort_limit_hardware_nm"], dtype=np.float64),
            np.asarray(physics["armature_hardware"], dtype=np.float64),
            None,
        ),
    )


def _body_id(module: Any, model: Any, name: str) -> int:
    result = int(module.mj_name2id(model, module.mjtObj.mjOBJ_BODY, name))
    if result < 0:
        raise ValueError(f"model lacks body {name!r}")
    return result


def _body_pose(module: Any, model: Any, data: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = _body_id(module, model, name)
    return np.asarray(data.xpos[body_id]).copy(), np.asarray(data.xquat[body_id]).copy()


def _body_velocity(
    module: Any,
    model: Any,
    data: Any,
    name: str,
    offset: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    body_id = _body_id(module, model, name)
    value = np.empty(6, dtype=np.float64)
    module.mj_objectVelocity(model, data, module.mjtObj.mjOBJ_BODY, body_id, value, 0)
    angular = value[:3].copy()
    linear = value[3:].copy()
    world_offset = _quaternion_matrix(data.xquat[body_id]) @ np.asarray(offset)
    linear += np.cross(angular, world_offset)
    return linear, angular


def _semantic_state(module: Any, model: Any, data: Any, arm: str) -> dict[str, Any]:
    points: dict[str, list[float]] = {}
    orientations: dict[str, list[float]] = {}
    velocities: dict[str, Any] = {}
    for logical_name, binding in _SEMANTIC_BINDINGS.items():
        body_name = str(binding[arm])
        offset = tuple(binding["offset"])
        body_pos, body_quat = _body_pose(module, model, data, body_name)
        point = body_pos + _quaternion_matrix(body_quat) @ np.asarray(offset)
        points[logical_name] = _json_vector(point)
        if logical_name in {"left_foot", "right_foot"}:
            orientations[logical_name] = _json_vector(body_quat[[1, 2, 3, 0]])
        linear, angular = _body_velocity(module, model, data, body_name, offset)
        velocities[logical_name] = {
            "linear_mps": _json_vector(linear),
            "angular_radps": _json_vector(angular),
        }
    pelvis_linear, pelvis_angular = _body_velocity(module, model, data, "pelvis", (0.0, 0.0, 0.0))
    velocities = {
        "pelvis": {
            "linear_mps": _json_vector(pelvis_linear),
            "angular_radps": _json_vector(pelvis_angular),
        },
        **velocities,
    }
    pelvis_pos, pelvis_quat = _body_pose(module, model, data, "pelvis")
    pelvis_id = _body_id(module, model, "pelvis")
    return {
        "contacts": _foot_contacts(module, model, data),
        "pelvis": {
            "position_m": _json_vector(pelvis_pos),
            "orientation_xyzw": _json_vector(pelvis_quat[[1, 2, 3, 0]]),
        },
        "com_position_m": _json_vector(data.subtree_com[pelvis_id]),
        "semantic_points_m": points,
        "semantic_orientations_xyzw": orientations,
        "link_velocities": velocities,
    }


def _body_descends_from(model: Any, body_id: int, ancestor: int) -> bool:
    current = int(body_id)
    while current > 0:
        if current == ancestor:
            return True
        current = int(model.body_parentid[current])
    return False


def _raw_contact_pairs(model: Any, data: Any) -> list[list[int]]:
    return [[int(data.contact[index].geom1), int(data.contact[index].geom2)] for index in range(int(data.ncon))]


def _foot_contacts(module: Any, model: Any, data: Any) -> dict[str, bool]:
    floor = int(module.mj_name2id(model, module.mjtObj.mjOBJ_GEOM, "floor"))
    if floor < 0:
        raise ValueError("model lacks floor geom")
    ancestors = {
        "left_foot": _body_id(module, model, "left_ankle_roll_link"),
        "right_foot": _body_id(module, model, "right_ankle_roll_link"),
    }
    result = dict.fromkeys(ancestors, False)
    for geom1, geom2 in _raw_contact_pairs(model, data):
        if floor not in (geom1, geom2):
            continue
        other = geom2 if geom1 == floor else geom1
        body_id = int(model.geom_bodyid[other])
        for name, ancestor in ancestors.items():
            if _body_descends_from(model, body_id, ancestor):
                result[name] = True
    return result


def evaluate_shipped_termination(
    *,
    measured_torso_position: Sequence[float],
    measured_torso_quaternion_wxyz: Sequence[float],
    reference_torso_position: Sequence[float],
    reference_torso_quaternion_wxyz: Sequence[float],
    measured_ee_positions: Mapping[str, Sequence[float]],
    reference_ee_positions: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Recompute exact shipped z-only / projected-gravity-z termination terms."""

    anchor_error = abs(float(measured_torso_position[2]) - float(reference_torso_position[2]))
    # Preserve the raw quaternion arithmetic used by the strict raw-sidecar
    # recomputation. Reference quaternions are serialized float32 and therefore
    # must not be silently renormalized here.
    _, measured_x_raw, measured_y_raw, _ = measured_torso_quaternion_wxyz
    _, reference_x_raw, reference_y_raw, _ = reference_torso_quaternion_wxyz
    measured_x, measured_y = float(measured_x_raw), float(measured_y_raw)
    reference_x, reference_y = float(reference_x_raw), float(reference_y_raw)
    measured_gravity_z = -(1.0 - 2.0 * (measured_x * measured_x + measured_y * measured_y))
    reference_gravity_z = -(1.0 - 2.0 * (reference_x * reference_x + reference_y * reference_y))
    orientation_error = abs(reference_gravity_z - measured_gravity_z)
    ee_errors = {
        name: abs(float(measured_ee_positions[name][2]) - float(reference_ee_positions[name][2]))
        for name in _SHIPPED_EE
    }
    anchor_bad = anchor_error > 0.25
    orientation_bad = orientation_error > 0.8
    ee_bad = any(value > 0.25 for value in ee_errors.values())
    return {
        "time_out": False,
        "anchor_pos": anchor_bad,
        "anchor_pos_error_z_m": anchor_error,
        "anchor_pos_threshold_m": 0.25,
        "anchor_ori": orientation_bad,
        "anchor_ori_projected_gravity_z_error": orientation_error,
        "anchor_ori_threshold": 0.8,
        "ee_body_pos": ee_bad,
        "ee_body_pos_error_z_m": ee_errors,
        "ee_body_pos_threshold_m": 0.25,
        "terminated": anchor_bad or orientation_bad or ee_bad,
        "timed_out": False,
    }


def _reference_gate_values(
    reference: ReferenceKinematics,
    arm: str,
    frame: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torso_pos, torso_quat = reference.arm_body(arm, "torso_link", frame)
    ee: dict[str, Any] = {}
    for name in _SHIPPED_EE:
        position, quaternion = reference.arm_body(arm, name, frame)
        ee[name] = {
            "reference_position_m": _json_vector(position),
            "reference_quaternion_wxyz": _json_vector(quaternion),
        }
    return (
        {
            "reference_position_m": _json_vector(torso_pos),
            "reference_quaternion_wxyz": _json_vector(torso_quat),
        },
        ee,
    )


def _geom_identities(module: Any, model: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        geom_name = module.mj_id2name(model, module.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = int(model.geom_bodyid[geom_id])
        body_name = module.mj_id2name(model, module.mjtObj.mjOBJ_BODY, body_id)
        result.append(
            {
                "geom_id": geom_id,
                "geom_name": str(geom_name) if geom_name is not None else f"unnamed_geom_{geom_id}",
                "body_id": body_id,
                "body_name": str(body_name) if body_name is not None else "world",
            }
        )
    return result


def _apply_world_disturbance(
    qvel: np.ndarray,
    quaternion_wxyz: Sequence[float],
    delta_world: Sequence[float],
) -> None:
    delta = np.asarray(delta_world, dtype=np.float64)
    if delta.shape != (6,):
        raise ValueError("world disturbance must contain six values")
    qvel[:3] += delta[:3]
    qvel[3:6] += _quaternion_matrix(quaternion_wxyz).T @ delta[3:]


@dataclass(frozen=True)
class EpisodeTracePair:
    trace: dict[str, Any]
    raw_trace: dict[str, Any]


def _initial_state(
    clip: LoadedClip,
    spec: EpisodeSpec,
    arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    # Both arms receive the identical source-root projection.  Only embodiment
    # joint state differs, and the same retained-joint perturbation is mapped.
    qpos_size = 36 if arm == "teacher_reference" else 30
    qvel_size = 35 if arm == "teacher_reference" else 29
    qpos = np.zeros(qpos_size, dtype=np.float64)
    qvel = np.zeros(qvel_size, dtype=np.float64)
    qpos[:3] = clip.source_root_pos[0] + np.asarray(spec.root_position_delta_m)
    qpos[3:7] = _yaw_perturb(clip.source_root_quat_wxyz[0], spec.root_yaw_delta_rad)
    delta23 = np.asarray(spec.joint_delta_retained_hardware_rad)
    if arm == "teacher_reference":
        retained = np.asarray(tuple(range(13)) + tuple(range(15, 20)) + tuple(range(22, 27)))
        qpos[7:] = clip.source_joint_pos_hardware[0]
        qpos[7 + retained] += delta23
        qvel[6:] = clip.source_joint_vel_hardware[0]
    elif arm == "true23_expert":
        qpos[7:] = clip.target_joint_pos_hardware[0] + delta23
        qvel[6:] = clip.target_joint_vel_hardware[0]
    else:
        raise ValueError(f"unknown campaign arm: {arm}")
    return qpos, qvel


def _physics_step_digest_update(
    digest: Any,
    q: np.ndarray,
    dq: np.ndarray,
    target: np.ndarray,
    torque: np.ndarray,
) -> None:
    for value in (q, dq, target, torque):
        digest.update(np.asarray(value, dtype="<f8").tobytes(order="C"))


def _teacher_proprio_frame(
    module: Any,
    model: Any,
    data: Any,
    previous_action_native: np.ndarray,
) -> np.ndarray:
    q_hardware = np.asarray(data.qpos[7:], dtype=np.float64)
    dq_hardware = np.asarray(data.qvel[6:], dtype=np.float64)
    q_native = q_hardware[np.asarray(IL29_TO_HARDWARE29)]
    dq_native = dq_hardware[np.asarray(IL29_TO_HARDWARE29)]
    default_native = SOURCE_DEFAULT_Q_HARDWARE[np.asarray(IL29_TO_HARDWARE29)]
    _, angular_world = _body_velocity(module, model, data, "pelvis", (0.0, 0.0, 0.0))
    angular = world_angular_velocity_to_body(data.qpos[3:7], angular_world)
    gravity = _projected_gravity(data.qpos[3:7])
    result = np.concatenate(
        (angular, q_native - default_native, dq_native, previous_action_native, gravity)
    ).astype(np.float32)
    if result.shape != (93,) or not np.isfinite(result).all():
        raise ValueError("teacher measured proprio frame is not finite 93-D")
    return result


def _measured_gate_evidence(
    module: Any,
    model: Any,
    data: Any,
    reference: ReferenceKinematics,
    arm: str,
    frame: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    measured_torso_pos, measured_torso_quat = _body_pose(module, model, data, "torso_link")
    torso_reference, ee_reference = _reference_gate_values(reference, arm, frame)
    torso = {
        "measured_position_m": _json_vector(measured_torso_pos),
        "measured_quaternion_wxyz": _json_vector(measured_torso_quat),
        **torso_reference,
    }
    measured_ee_positions: dict[str, np.ndarray] = {}
    reference_ee_positions: dict[str, np.ndarray] = {}
    ee: dict[str, Any] = {}
    for standard_name in _SHIPPED_EE:
        body_name = standard_name
        if arm == "teacher_reference":
            body_name = {
                "left_wrist_roll_rubber_hand": "left_wrist_yaw_link",
                "right_wrist_roll_rubber_hand": "right_wrist_yaw_link",
            }.get(standard_name, standard_name)
        position, quaternion = _body_pose(module, model, data, body_name)
        measured_ee_positions[standard_name] = position
        reference_position = np.asarray(ee_reference[standard_name]["reference_position_m"])
        reference_ee_positions[standard_name] = reference_position
        ee[standard_name] = {
            "measured_position_m": _json_vector(position),
            "measured_quaternion_wxyz": _json_vector(quaternion),
            **ee_reference[standard_name],
        }
    termination = evaluate_shipped_termination(
        measured_torso_position=measured_torso_pos,
        measured_torso_quaternion_wxyz=measured_torso_quat,
        reference_torso_position=torso_reference["reference_position_m"],
        reference_torso_quaternion_wxyz=torso_reference["reference_quaternion_wxyz"],
        measured_ee_positions=measured_ee_positions,
        reference_ee_positions=reference_ee_positions,
    )
    return torso, ee, termination


def run_arm_episode(
    *,
    module: Any,
    model: Any,
    model_relpath: str,
    model_sha256: str,
    physics: PhysicsParameters,
    reference: ReferenceKinematics,
    spec: EpisodeSpec,
    arm: str,
    teacher: ExactLowLatencyTeacher | None,
    reference_schedule_ref: Mapping[str, Any],
    common_initial_ref: Mapping[str, Any],
    arm_initial_ref: Mapping[str, Any],
    disturbance_ref: Mapping[str, Any],
    horizon_steps: int = HORIZON_STEPS,
) -> EpisodeTracePair:
    """Run one arm without resets, even after a shipped termination event."""

    if arm == "teacher_reference" and teacher is None:
        raise ValueError("teacher arm requires exact neural runtime")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    data = module.MjData(model)
    initial_qpos, initial_qvel = _initial_state(reference.clip, spec, arm)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    module.mj_forward(model, data)
    joint_names = _model_joint_names(model)
    joint_ids = np.arange(1, model.njnt, dtype=np.int64)
    hard_lower = np.asarray(model.jnt_range[joint_ids, 0], dtype=np.float64)
    hard_upper = np.asarray(model.jnt_range[joint_ids, 1], dtype=np.float64)
    expected_count = 29 if arm == "teacher_reference" else 23
    if len(joint_names) != expected_count:
        raise ValueError("arm model joint count changed")

    history: list[np.ndarray] = []
    previous_action = np.zeros(29, dtype=np.float32)
    trace_steps: list[dict[str, Any]] = []
    raw_steps: list[dict[str, Any]] = []
    termination_latched = False
    action_scale = physics.action_scale
    for step in range(horizon_steps):
        frame = reference.clip.frame_index(step)
        held = reference.clip.terminal_held(step)
        post_termination = termination_latched
        time_before = float(data.time)
        disturbance = np.zeros(6, dtype=np.float64)
        if step == spec.disturbance_step:
            disturbance = np.asarray(spec.disturbance_delta_world, dtype=np.float64)
            _apply_world_disturbance(data.qvel, data.qpos[3:7], disturbance)
            module.mj_forward(model, data)
        qpos_before = np.asarray(data.qpos, dtype=np.float64).copy()
        qvel_before = np.asarray(data.qvel, dtype=np.float64).copy()

        if arm == "teacher_reference":
            proprio_frame = _teacher_proprio_frame(module, model, data, previous_action)
            if not history:
                history = [proprio_frame.copy() for _ in range(HISTORY_LENGTH)]
            else:
                history = [*history[1:], proprio_frame]
            encoder = reference.teleop_encoder_input(step, data.qpos[3:7])
            raw_action = teacher.infer(encoder, term_major_history(history))
            safe_action = np.clip(raw_action, -TEACHER_ACTION_CLIP, TEACHER_ACTION_CLIP)
            if action_scale is None:
                raise AssertionError("teacher physics lacks source action scale")
            action_hardware = safe_action[np.asarray(HARDWARE29_TO_IL29)]
            target = SOURCE_DEFAULT_Q_HARDWARE + action_hardware * action_scale
            previous_action = safe_action.astype(np.float32, copy=True)
        else:
            raw_action = reference.clip.target_action_native[frame].copy()
            safe_action, target32 = safe_target_transform_numpy(raw_action.astype(np.float32, copy=False))
            target = target32.astype(np.float64)

        digest = hashlib.sha256()
        first_torque = np.zeros(expected_count, dtype=np.float64)
        last_torque = np.zeros(expected_count, dtype=np.float64)
        peak_abs_torque = np.zeros(expected_count, dtype=np.float64)
        for substep in range(physics.decimation):
            q = np.asarray(data.qpos[7:], dtype=np.float64).copy()
            dq = np.asarray(data.qvel[6:], dtype=np.float64).copy()
            torque = np.clip(
                physics.kp * (target - q) - physics.kd * dq,
                -physics.effort,
                physics.effort,
            )
            _physics_step_digest_update(digest, q, dq, target, torque)
            if substep == 0:
                first_torque = torque.copy()
            last_torque = torque.copy()
            peak_abs_torque = np.maximum(peak_abs_torque, np.abs(torque))
            data.ctrl[:] = torque
            module.mj_step(model, data)

        qpos_after = np.asarray(data.qpos, dtype=np.float64).copy()
        qvel_after = np.asarray(data.qvel, dtype=np.float64).copy()
        semantic = _semantic_state(module, model, data, arm)
        torso, ee, termination = _measured_gate_evidence(module, model, data, reference, arm, frame)
        reported_nonfinite = not all(
            np.isfinite(value).all()
            for value in (
                qpos_before,
                qvel_before,
                qpos_after,
                qvel_after,
                raw_action,
                safe_action,
                target,
                last_torque,
            )
        )
        time_out = shipped_time_out_term(
            already_latched=termination_latched,
            final_step=step == horizon_steps - 1,
        )
        terminated, timed_out, post_termination, termination_latched = terminal_event_flags(
            already_latched=termination_latched,
            gate_terminated=bool(termination["terminated"] or reported_nonfinite),
            final_step=step == horizon_steps - 1,
        )
        termination["time_out"] = time_out
        termination["terminated"] = terminated
        termination["timed_out"] = timed_out
        trace_step = {
            "step_index": step,
            "terminated": terminated,
            "timed_out": timed_out,
            "post_termination": post_termination,
            "episode_instance_id": spec.pair_id,
            "reset_generation": 0,
            "sim_advanced": float(data.time) > time_before,
            "reported_nonfinite": reported_nonfinite,
            "reference_frame_index": frame,
            "joint_positions_rad": _json_vector(qpos_after[7:]),
            "disturbance_delta": _json_vector(disturbance),
            **semantic,
        }
        if set(trace_step) != TRACE_STEP_KEYS:
            raise AssertionError("qualification trace step topology drift")
        trace_steps.append(trace_step)
        raw_step = {
            "step_index": step,
            "reference_frame_index": frame,
            "reference_terminal_held": held,
            "sim_time_before_s": time_before,
            "sim_time_after_s": float(data.time),
            "reset_generation": 0,
            "sim_advanced": float(data.time) > time_before,
            "qpos_before": _json_vector(qpos_before),
            "qvel_before": _json_vector(qvel_before),
            "qpos_after": _json_vector(qpos_after),
            "qvel_after": _json_vector(qvel_after),
            "raw_action_native": _json_vector(raw_action),
            "safe_action_native": _json_vector(safe_action),
            "q_target_hardware_rad": _json_vector(target),
            "disturbance_delta_world": _json_vector(disturbance),
            "pd_substeps_sha256": digest.hexdigest(),
            "applied_torque_first_hardware_nm": _json_vector(first_torque),
            "applied_torque_last_hardware_nm": _json_vector(last_torque),
            "applied_torque_peak_abs_hardware_nm": _json_vector(peak_abs_torque),
            "raw_contact_geom_pairs": _raw_contact_pairs(model, data),
            "torso_reference": torso,
            "ee_reference": ee,
            "termination_terms": termination,
            "reported_nonfinite": reported_nonfinite,
        }
        if set(raw_step) != RAW_STEP_KEYS:
            raise AssertionError("raw audit trace step topology drift")
        raw_steps.append(raw_step)

    controller_semantics = (
        TEACHER_CONTROLLER_SEMANTICS if arm == "teacher_reference" else TRUE23_CONTROLLER_SEMANTICS
    )
    trace = {
        "schema_version": 1,
        "kind": TRACE_SCHEMA,
        "pair_id": spec.pair_id,
        "arm": arm,
        "clip_id": spec.clip_id,
        "seed": spec.seed,
        "controller_semantics": controller_semantics,
        "post_termination_policy": POST_TERMINATION_POLICY,
        "episode_instance_id": spec.pair_id,
        "reset_generation": 0,
        "reference_schedule_sha256": reference_schedule_ref["sha256"],
        "reference_schedule_payload_sha256": reference_schedule_ref["payload_sha256"],
        "common_initial_state_sha256": common_initial_ref["sha256"],
        "common_initial_state_payload_sha256": common_initial_ref["payload_sha256"],
        "arm_initial_state_sha256": arm_initial_ref["sha256"],
        "arm_initial_state_payload_sha256": arm_initial_ref["payload_sha256"],
        "disturbance_schedule_sha256": disturbance_ref["sha256"],
        "disturbance_schedule_payload_sha256": disturbance_ref["payload_sha256"],
        "control_hz": CONTROL_HZ,
        "horizon_steps": horizon_steps,
        "joint_names": list(joint_names),
        "hard_joint_position_lower_rad": _json_vector(hard_lower),
        "hard_joint_position_upper_rad": _json_vector(hard_upper),
        "record_count": len(trace_steps),
        "steps": trace_steps,
    }
    if set(trace) != TRACE_HEADER_KEYS:
        raise AssertionError("qualification trace header topology drift")
    raw_trace = {
        "schema_version": 1,
        "kind": RAW_TRACE_KIND,
        "pair_id": spec.pair_id,
        "arm": arm,
        "clip_id": spec.clip_id,
        "seed": spec.seed,
        "controller_semantics": controller_semantics,
        "trace_sha256": "",
        "trace_payload_sha256": "",
        "model": {
            "repository_relpath": model_relpath,
            "sha256": model_sha256,
            "size_bytes": (REPOSITORY_ROOT / model_relpath).stat().st_size,
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
        },
        "control_hz": CONTROL_HZ,
        "physics_hz": round(1.0 / physics.timestep_s),
        "horizon_steps": horizon_steps,
        "episode_instance_id": spec.pair_id,
        "reset_generation": 0,
        "joint_names": list(joint_names),
        "hard_joint_position_lower_rad": _json_vector(hard_lower),
        "hard_joint_position_upper_rad": _json_vector(hard_upper),
        "actuator_names": [
            str(module.mj_id2name(model, module.mjtObj.mjOBJ_ACTUATOR, index)) for index in range(model.nu)
        ],
        "floor_geom_id": int(module.mj_name2id(model, module.mjtObj.mjOBJ_GEOM, "floor")),
        "geom_identities": _geom_identities(module, model),
        "record_count": len(raw_steps),
        "steps": raw_steps,
    }
    if set(raw_trace) != RAW_HEADER_KEYS:
        raise AssertionError("raw audit trace header topology drift")
    return EpisodeTracePair(trace=trace, raw_trace=raw_trace)


@dataclass(frozen=True)
class FrozenClipPaths:
    clip_id: str
    source_csv: Path
    motion: Path
    expert: Path
    report: Path
    expected_frame_count: int


def frozen_clip_paths(
    *,
    contract_path: Path,
    source_csv_root: Path,
    step1a_root: Path,
    selected_clip_ids: Sequence[str] | None = None,
) -> tuple[FrozenClipPaths, ...]:
    """Resolve only the two frozen contract clips; substitutions are rejected."""

    contract = json.loads(contract_path.resolve().read_text(encoding="utf-8"))
    frozen = contract.get("frozen_clips")
    if not isinstance(frozen, list) or len(frozen) != 2:
        raise ValueError("Step1B contract must freeze exactly two clips")
    selected = (
        tuple(str(value) for value in selected_clip_ids)
        if selected_clip_ids is not None
        else tuple(str(item["clip_id"]) for item in frozen)
    )
    allowed = tuple(str(item["clip_id"]) for item in frozen)
    if any(clip_id not in allowed for clip_id in selected) or len(set(selected)) != len(selected):
        raise ValueError("selected clips must be unique frozen Step1B clip IDs")
    by_id = {str(item["clip_id"]): item for item in frozen}
    result: list[FrozenClipPaths] = []
    for clip_id in selected:
        item = by_id[clip_id]
        result.append(
            FrozenClipPaths(
                clip_id=clip_id,
                source_csv=source_csv_root.resolve() / str(item["source_csv"]),
                motion=step1a_root.resolve() / "motions" / f"{clip_id}.npz",
                expert=step1a_root.resolve() / "experts" / f"{clip_id}.task_space.npz",
                report=step1a_root.resolve() / "reports" / f"{clip_id}.retarget.json",
                expected_frame_count=int(item["source_frame_count"]),
            )
        )
    return tuple(result)


def _external_file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _initial_payloads(
    clip: LoadedClip,
    spec: EpisodeSpec,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    teacher_qpos, teacher_qvel = _initial_state(clip, spec, "teacher_reference")
    target_qpos, target_qvel = _initial_state(clip, spec, "true23_expert")
    if not np.array_equal(teacher_qpos[:7], target_qpos[:7]):
        raise AssertionError("paired arms do not share exact initial base pose")
    if not np.array_equal(teacher_qvel[:6], target_qvel[:6]):
        raise AssertionError("paired arms do not share exact initial base velocity")
    common = {
        "schema_version": 1,
        "kind": INITIAL_STATE_SCHEMA,
        "scope": INITIAL_STATE_SCOPES["common"],
        "state": {
            "root_position_m": _json_vector(teacher_qpos[:3]),
            "root_orientation_xyzw": _json_vector(teacher_qpos[[4, 5, 6, 3]]),
            "root_linear_velocity_mps": _json_vector(teacher_qvel[:3]),
            "root_angular_velocity_radps": _json_vector(teacher_qvel[3:6]),
            "reference_frame_index": 0,
        },
    }
    teacher = {
        "schema_version": 1,
        "kind": INITIAL_STATE_SCHEMA,
        "scope": INITIAL_STATE_SCOPES["teacher_reference"],
        "state": {
            "root_position_m": _json_vector(teacher_qpos[:3]),
            "root_orientation_xyzw": _json_vector(teacher_qpos[[4, 5, 6, 3]]),
            "root_linear_velocity_mps": _json_vector(teacher_qvel[:3]),
            "root_angular_velocity_radps": _json_vector(teacher_qvel[3:6]),
            "reference_frame_index": 0,
            "joint_names": list(SOURCE_HARDWARE_JOINT_NAMES),
            "joint_positions_rad": _json_vector(teacher_qpos[7:]),
            "joint_velocities_radps": _json_vector(teacher_qvel[6:]),
        },
    }
    true23 = {
        "schema_version": 1,
        "kind": INITIAL_STATE_SCHEMA,
        "scope": INITIAL_STATE_SCOPES["true23_expert"],
        "state": {
            "root_position_m": _json_vector(target_qpos[:3]),
            "root_orientation_xyzw": _json_vector(target_qpos[[4, 5, 6, 3]]),
            "root_linear_velocity_mps": _json_vector(target_qvel[:3]),
            "root_angular_velocity_radps": _json_vector(target_qvel[3:6]),
            "reference_frame_index": 0,
            "joint_names": list(HARDWARE_23_JOINT_NAMES),
            "joint_positions_rad": _json_vector(target_qpos[7:]),
            "joint_velocities_radps": _json_vector(target_qvel[6:]),
        },
    }
    return common, teacher, true23


def _schedule_payloads(
    clip: LoadedClip,
    spec: EpisodeSpec,
    horizon_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    indices = terminal_hold_indices(clip.frame_count, horizon_steps)
    reference = {
        "schema_version": 1,
        "kind": REFERENCE_SCHEDULE_SCHEMA,
        "clip_id": clip.clip_id,
        "source_frame_count": clip.frame_count,
        "start_frame": 0,
        "continuation": REFERENCE_CONTINUATION,
        "frame_indices": list(indices),
    }
    deltas = [[0.0] * 6 for _ in range(horizon_steps)]
    if spec.disturbance_step < horizon_steps:
        deltas[spec.disturbance_step] = list(spec.disturbance_delta_world)
    disturbance = {
        "schema_version": 1,
        "kind": DISTURBANCE_SCHEMA,
        "control_hz": CONTROL_HZ,
        "deltas": deltas,
    }
    return reference, disturbance


def _controller_payloads(
    teacher: ExactLowLatencyTeacher,
    true23_physics: PhysicsParameters,
) -> dict[str, Any]:
    true23_resolved = {
        "controller_descriptor": true23_controller_descriptor(),
        "reference_transform": safe_target_transform_contract(),
        "physics_gains": {
            "kp": _json_vector(true23_physics.kp),
            "kd": _json_vector(true23_physics.kd),
            "effort_limits_nm": _json_vector(true23_physics.effort),
        },
    }
    return {
        "teacher_controller_config": {
            "schema_version": 1,
            "kind": "low_latency_policy_config",
            "resolved_config": teacher.descriptor(),
        },
        "teacher_input_action_law": build_input_action_law("teacher_reference"),
        "teacher_semantic_reference": {
            "schema_version": 1,
            "kind": "source29_terminal_hold_low_latency_semantic_reference",
            "input_dim": TELEOP_INPUT_DIM,
            "term_order": [
                "command_multi_future_lower_body",
                "vr_3point_local_target",
                "vr_3point_local_orn_target",
                "motion_anchor_ori_b",
            ],
            "term_dims": [240, 9, 12, 6],
            "future_frame_offsets_s": [index / CONTROL_HZ for index in range(10)],
            "continuation": REFERENCE_CONTINUATION,
        },
        "true23_controller_config": {
            "schema_version": 1,
            "kind": "state_feedback_controller_config",
            "resolved_config": true23_resolved,
        },
        "true23_input_action_law": build_input_action_law("true23_expert"),
    }


def _physics_parameters_payload(physics: PhysicsParameters) -> dict[str, Any]:
    return {
        "timestep_s": physics.timestep_s,
        "decimation": physics.decimation,
        "kp": _json_vector(physics.kp),
        "kd": _json_vector(physics.kd),
        "effort_limits_nm": _json_vector(physics.effort),
        "armature": _json_vector(physics.armature),
        "action_scale": (None if physics.action_scale is None else _json_vector(physics.action_scale)),
    }


def _runtime_config_payload(
    *,
    device: str,
    simulator_version: str,
    source_physics: PhysicsParameters,
    true23_physics: PhysicsParameters,
    sim_config_path: Path,
) -> dict[str, Any]:
    resolved = {
        "controller_descriptor": true23_controller_descriptor(),
        "device": device,
        "simulator_name": "MuJoCo",
        "simulator_version": simulator_version,
        "source29_model_sha256": SOURCE_MODEL_SHA256,
        "true23_model_sha256": TARGET_MODEL_SHA256,
        "sim2sim_config_sha256": sha256_file(sim_config_path.resolve()),
        "source29_physics": _physics_parameters_payload(source_physics),
        "true23_physics": _physics_parameters_payload(true23_physics),
    }
    return {
        "schema_version": 1,
        "kind": RUNTIME_CONFIG_SCHEMA,
        "control_hz": CONTROL_HZ,
        "horizon_steps": HORIZON_STEPS,
        "post_termination_policy": POST_TERMINATION_POLICY,
        "shipped_end_effector_termination_threshold_m": (SHIPPED_EE_TERMINATION_THRESHOLD_M),
        "teacher_controller_semantics": TEACHER_CONTROLLER_SEMANTICS,
        "true23_controller_semantics": TRUE23_CONTROLLER_SEMANTICS,
        "resolved_config": resolved,
        "resolved_config_sha256": sha256_bytes(canonical_json_bytes(resolved)),
    }


def _is_qualification_shape(
    clip_paths: Sequence[FrozenClipPaths],
    episodes_per_clip: int,
    horizon_steps: int,
) -> bool:
    return (
        tuple(paths.clip_id for paths in clip_paths) == FROZEN_CLIP_IDS
        and episodes_per_clip == EPISODES_PER_CLIP
        and horizon_steps == HORIZON_STEPS
    )


def _write_qualification_envelope(
    *,
    root: Path,
    clip_paths: Sequence[FrozenClipPaths],
    checkpoint_path: Path,
    contract_path: Path,
    source_model_path: Path,
    target_model_path: Path,
    sim_config_path: Path,
    simulator_version: str,
    device: str,
    source_physics: PhysicsParameters,
    true23_physics: PhysicsParameters,
    controller_payloads: Mapping[str, Any],
    schedule_records: Sequence[Mapping[str, Any]],
    pair_records: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Write the exact strict-validator campaign/report topology."""

    step1a_roots = {paths.motion.resolve().parent.parent for paths in clip_paths}
    if len(step1a_roots) != 1:
        raise ValueError("all frozen clips must come from one Step1A evidence root")
    step1a_root = next(iter(step1a_roots))

    support_ref = _copy_json_ref(
        root,
        REPOSITORY_ROOT / STEP1A_SUPPORT_RELPATH,
        "inputs/step1a/support_manifest.json",
    )
    qualification_ref = _copy_json_ref(
        root,
        step1a_root / "qualification.json",
        "inputs/step1a/qualification.json",
    )
    batch_ref = _copy_json_ref(
        root,
        step1a_root / "batch.json",
        "inputs/step1a/batch.json",
    )
    source_model_ref = _copy_binary_ref(root, source_model_path, "inputs/assets/g1_29dof.xml")
    target_model_ref = _copy_binary_ref(root, target_model_path, "inputs/assets/g1_23dof_rev_1_0.xml")
    checkpoint_ref = _copy_binary_ref(root, checkpoint_path, "inputs/assets/low_latency_last.pt")
    controller_ref = _copy_binary_ref(
        root, REPOSITORY_ROOT / RUNNER_RELPATH, "inputs/runtime/g1_true23_step1b_mujoco.py"
    )

    teacher_config_ref = _json_ref(
        root,
        root / "inputs/controllers/teacher_controller_config.json",
        controller_payloads["teacher_controller_config"],
    )
    teacher_action_ref = _json_ref(
        root,
        root / "inputs/controllers/teacher_input_action_law.json",
        controller_payloads["teacher_input_action_law"],
    )
    teacher_semantic_ref = _json_ref(
        root,
        root / "inputs/controllers/teacher_semantic_reference.json",
        controller_payloads["teacher_semantic_reference"],
    )
    true23_config_ref = _json_ref(
        root,
        root / "inputs/controllers/true23_controller_config.json",
        controller_payloads["true23_controller_config"],
    )
    true23_action_ref = _json_ref(
        root,
        root / "inputs/controllers/true23_input_action_law.json",
        controller_payloads["true23_input_action_law"],
    )
    runtime_config = _runtime_config_payload(
        device=device,
        simulator_version=simulator_version,
        source_physics=source_physics,
        true23_physics=true23_physics,
        sim_config_path=sim_config_path,
    )
    runtime_config_ref = _json_ref(root, root / "inputs/runtime/runtime_config.json", runtime_config)

    clip_refs: list[dict[str, Any]] = []
    for paths in clip_paths:
        prefix = f"inputs/schema6/{paths.clip_id}"
        clip_refs.append(
            {
                "clip_id": paths.clip_id,
                "motion": _copy_binary_ref(root, paths.motion, f"{prefix}/motion.npz"),
                "expert": _copy_binary_ref(root, paths.expert, f"{prefix}/expert.task_space.npz"),
                "report": _copy_json_ref(root, paths.report, f"{prefix}/retarget.json"),
            }
        )

    termination_paths = json.loads(contract_path.resolve().read_text(encoding="utf-8"))[
        "selected_termination_gate"
    ]["config_relpaths"]
    campaign = {
        "schema_version": 1,
        "kind": CAMPAIGN_SCHEMA,
        "declared_categories": ["idle"],
        "frozen_clip_ids": list(FROZEN_CLIP_IDS),
        "clip_substitutions": False,
        "row_masks": False,
        "step1a": {
            "support_manifest": support_ref,
            "qualification": qualification_ref,
            "batch": batch_ref,
        },
        "teacher_reference": {
            "kind": "exact_29dof_teacher_reference",
            "controller_semantics": TEACHER_CONTROLLER_SEMANTICS,
            "model": source_model_ref,
            "checkpoint": checkpoint_ref,
            "controller_config": teacher_config_ref,
            "input_action_law": teacher_action_ref,
            "semantic_reference": teacher_semantic_ref,
        },
        "true23_expert": {
            "kind": "schema_v6_true23_expert",
            "controller_semantics": TRUE23_CONTROLLER_SEMANTICS,
            "controller": controller_ref,
            "controller_config": true23_config_ref,
            "input_action_law": true23_action_ref,
            "target_model": target_model_ref,
            "clips": clip_refs,
        },
        "runtime": {
            "simulator_name": "MuJoCo",
            "simulator_version": simulator_version,
            "runner": controller_ref,
            "runtime_config": runtime_config_ref,
            "robot_assets": [{"role": "true23_model", "artifact": target_model_ref}],
            "termination_configs": [
                _repository_binding(str(relative_path)) for relative_path in termination_paths
            ],
        },
        "episode_schedule": [dict(record) for record in schedule_records],
    }
    campaign_ref = _json_ref(root, root / "campaign.json", campaign)
    report_pairs = [
        {
            "pair_id": record["pair_id"],
            "clip_id": record["clip_id"],
            "teacher_trace": record["teacher_trace"],
            "true23_trace": record["true23_trace"],
            "teacher_raw_trace": record["teacher_raw_trace"],
            "true23_raw_trace": record["true23_raw_trace"],
        }
        for record in pair_records
    ]
    contract_payload = json.loads(contract_path.resolve().read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "kind": REPORT_SCHEMA,
        "contract_sha256": sha256_file(contract_path.resolve()),
        "contract_payload_sha256": sha256_bytes(canonical_json_bytes(contract_payload)),
        "campaign_manifest": campaign_ref,
        "pairs": report_pairs,
    }
    report_path = root / "report.json"
    _write_json_atomic(report_path, report)

    suggested_bindings = {
        "step1a": {
            "support_manifest_sha256": support_ref["sha256"],
            "support_manifest_payload_sha256": support_ref["payload_sha256"],
            "qualification_sha256": qualification_ref["sha256"],
            "qualification_payload_sha256": qualification_ref["payload_sha256"],
            "batch_sha256": batch_ref["sha256"],
            "batch_payload_sha256": batch_ref["payload_sha256"],
        },
        "teacher_reference": {
            "model_sha256": source_model_ref["sha256"],
            "checkpoint_sha256": checkpoint_ref["sha256"],
            "controller_config_payload_sha256": teacher_config_ref["payload_sha256"],
            "input_action_law_payload_sha256": teacher_action_ref["payload_sha256"],
            "semantic_reference_payload_sha256": teacher_semantic_ref["payload_sha256"],
        },
        "true23_expert": {
            "target_model_sha256": target_model_ref["sha256"],
            "controller_sha256": controller_ref["sha256"],
            "controller_config_payload_sha256": true23_config_ref["payload_sha256"],
            "input_action_law_payload_sha256": true23_action_ref["payload_sha256"],
            "action_law": TRUE23_CONTROLLER_SEMANTICS,
            "clips": {
                clip["clip_id"]: {
                    "motion_sha256": clip["motion"]["sha256"],
                    "expert_sha256": clip["expert"]["sha256"],
                    "report_payload_sha256": clip["report"]["payload_sha256"],
                }
                for clip in clip_refs
            },
        },
        "runtime": {
            "simulator_name": "MuJoCo",
            "simulator_version": simulator_version,
            "runner_sha256": controller_ref["sha256"],
            "runtime_config_payload_sha256": runtime_config_ref["payload_sha256"],
            "robot_assets": {"true23_model": target_model_ref["sha256"]},
        },
    }
    pin_suggestions = {
        "schema_version": 1,
        "kind": "g1_true23_idle_step1b_identity_pin_suggestions_v1",
        "authorization": "none",
        "contract_sha256_before_pinning": report["contract_sha256"],
        "trusted_identity_bindings": suggested_bindings,
    }
    pin_ref = _json_ref(root, root / "identity_pin_suggestions.json", pin_suggestions)
    return report_path, campaign_ref, pin_ref, suggested_bindings


def run_paired_diagnostic_campaign(
    *,
    output_root: Path,
    checkpoint_path: Path,
    clip_paths: Sequence[FrozenClipPaths],
    contract_path: Path,
    source_model_path: Path = REPOSITORY_ROOT / SOURCE_MODEL_RELPATH,
    target_model_path: Path = REPOSITORY_ROOT / TARGET_MODEL_RELPATH,
    sim_config_path: Path = REPOSITORY_ROOT / SIM_CONFIG_RELPATH,
    device: str = "cpu",
    episodes_per_clip: int = 1,
    horizon_steps: int = 50,
) -> Path:
    """Run a bounded paired campaign and return its primary evidence path.

    A complete two-clip, one-episode, 500-step run emits the strict campaign and
    report schemas. Smaller runs emit only a diagnostic bundle. In either case,
    only the separate validator can grant authorization.
    """

    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    if not clip_paths:
        raise ValueError("at least one frozen clip is required")
    if episodes_per_clip < 1 or episodes_per_clip > EPISODES_PER_CLIP:
        raise ValueError(f"episodes_per_clip must be in [1,{EPISODES_PER_CLIP}]")
    if horizon_steps < 1 or horizon_steps > HORIZON_STEPS:
        raise ValueError("horizon_steps must be in [1,500]")
    contract = json.loads(contract_path.resolve().read_text(encoding="utf-8"))
    schedule_contract = contract.get("campaign_schedule", {})
    disturbance_delta = tuple(schedule_contract.get("disturbance_delta", (0.0,) * 6))
    # The currently checked qualification contract does not authorize initial
    # state randomization.  A later explicit contract field can enable it.
    enable_initial_perturbation = bool(schedule_contract.get("deterministic_initial_perturbation", False))

    source_module, source_model, source_physics = prepare_source29_model(source_model_path)
    target_module, target_model, target_physics = prepare_true23_model(target_model_path, sim_config_path)
    if source_module is not target_module:
        raise AssertionError("paired arms loaded different MuJoCo Python modules")
    teacher = ExactLowLatencyTeacher(checkpoint_path, device=device)
    controller_payloads = _controller_payloads(teacher, target_physics)

    parent = output_root.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{output_root.name}.{uuid.uuid4().hex}.partial"
    temporary.mkdir()
    pair_records: list[dict[str, Any]] = []
    schedule_records: list[dict[str, Any]] = []
    try:
        for paths in clip_paths:
            clip = load_frozen_clip(
                clip_id=paths.clip_id,
                source_csv_path=paths.source_csv,
                motion_path=paths.motion,
                expert_path=paths.expert,
                report_path=paths.report,
                source_model=source_model,
                target_model=target_model,
                expected_frame_count=paths.expected_frame_count,
            )
            reference = ReferenceKinematics(source_module, source_model, target_model, clip)
            contract_seeds = schedule_contract.get("seeds_by_clip", {}).get(paths.clip_id)
            if not isinstance(contract_seeds, list):
                contract_seeds = list(FIXED_SEEDS_BY_CLIP[paths.clip_id])
            specs = deterministic_episode_specs(
                paths.clip_id,
                contract_seeds[:episodes_per_clip],
                disturbance_delta=disturbance_delta,
                enable_initial_perturbation=enable_initial_perturbation,
            )
            for spec in specs:
                schedule_dir = temporary / "schedule" / spec.pair_id
                reference_payload, disturbance_payload = _schedule_payloads(clip, spec, horizon_steps)
                common_payload, teacher_payload, true23_payload = _initial_payloads(clip, spec)
                reference_ref = _json_ref(temporary, schedule_dir / "reference.json", reference_payload)
                disturbance_ref = _json_ref(temporary, schedule_dir / "disturbance.json", disturbance_payload)
                common_ref = _json_ref(temporary, schedule_dir / "common_initial.json", common_payload)
                teacher_initial_ref = _json_ref(temporary, schedule_dir / "teacher_initial.json", teacher_payload)
                true23_initial_ref = _json_ref(temporary, schedule_dir / "true23_initial.json", true23_payload)
                schedule_records.append(
                    {
                        "pair_id": spec.pair_id,
                        "clip_id": spec.clip_id,
                        "seed": spec.seed,
                        "reference_schedule": reference_ref,
                        "common_initial_state": common_ref,
                        "teacher_initial_state": teacher_initial_ref,
                        "true23_initial_state": true23_initial_ref,
                        "disturbance_schedule": disturbance_ref,
                    }
                )
                teacher_result = run_arm_episode(
                    module=source_module,
                    model=source_model,
                    model_relpath=SOURCE_MODEL_RELPATH,
                    model_sha256=SOURCE_MODEL_SHA256,
                    physics=source_physics,
                    reference=reference,
                    spec=spec,
                    arm="teacher_reference",
                    teacher=teacher,
                    reference_schedule_ref=reference_ref,
                    common_initial_ref=common_ref,
                    arm_initial_ref=teacher_initial_ref,
                    disturbance_ref=disturbance_ref,
                    horizon_steps=horizon_steps,
                )
                true23_result = run_arm_episode(
                    module=target_module,
                    model=target_model,
                    model_relpath=TARGET_MODEL_RELPATH,
                    model_sha256=TARGET_MODEL_SHA256,
                    physics=target_physics,
                    reference=reference,
                    spec=spec,
                    arm="true23_expert",
                    teacher=None,
                    reference_schedule_ref=reference_ref,
                    common_initial_ref=common_ref,
                    arm_initial_ref=true23_initial_ref,
                    disturbance_ref=disturbance_ref,
                    horizon_steps=horizon_steps,
                )
                trace_dir = temporary / "traces"
                refs: dict[str, Any] = {}
                for name, result in (
                    ("teacher", teacher_result),
                    ("true23", true23_result),
                ):
                    trace_ref = _json_ref(
                        temporary,
                        trace_dir / f"{spec.pair_id}-{name}.json",
                        result.trace,
                    )
                    refs[f"{name}_trace"] = trace_ref
                    result.raw_trace["trace_sha256"] = trace_ref["sha256"]
                    result.raw_trace["trace_payload_sha256"] = trace_ref["payload_sha256"]
                    refs[f"{name}_raw_trace"] = _json_ref(
                        temporary,
                        trace_dir / f"{spec.pair_id}-{name}.raw.json",
                        result.raw_trace,
                    )
                pair_records.append(
                    {
                        "pair_id": spec.pair_id,
                        "clip_id": spec.clip_id,
                        "seed": spec.seed,
                        "reference_schedule": reference_ref,
                        "common_initial_state": common_ref,
                        "teacher_initial_state": teacher_initial_ref,
                        "true23_initial_state": true23_initial_ref,
                        "disturbance_schedule": disturbance_ref,
                        **refs,
                    }
                )

        qualification_shape_complete = _is_qualification_shape(clip_paths, episodes_per_clip, horizon_steps)
        report_path: Path | None = None
        campaign_ref: dict[str, Any] | None = None
        pin_ref: dict[str, Any] | None = None
        if qualification_shape_complete:
            report_path, campaign_ref, pin_ref, _suggested_bindings = _write_qualification_envelope(
                root=temporary,
                clip_paths=clip_paths,
                checkpoint_path=checkpoint_path,
                contract_path=contract_path,
                source_model_path=source_model_path,
                target_model_path=target_model_path,
                sim_config_path=sim_config_path,
                simulator_version=str(source_module.__version__),
                device=device,
                source_physics=source_physics,
                true23_physics=target_physics,
                controller_payloads=controller_payloads,
                schedule_records=schedule_records,
                pair_records=pair_records,
            )

        manifest = {
            "schema_version": 1,
            "kind": DIAGNOSTIC_BUNDLE_SCHEMA,
            "authorization": "none",
            "qualification_claimed": False,
            "qualification_shape_complete": qualification_shape_complete,
            "post_termination_policy": POST_TERMINATION_POLICY,
            "reference_continuation": REFERENCE_CONTINUATION,
            "control_hz": CONTROL_HZ,
            "horizon_steps": horizon_steps,
            "episodes_per_clip": episodes_per_clip,
            "contract": _external_file_binding(contract_path),
            "runtime": {
                "runner": _external_file_binding(REPOSITORY_ROOT / RUNNER_RELPATH),
                "simulator_name": "MuJoCo",
                "simulator_version": str(source_module.__version__),
                "device": device,
                "source_physics_hz": TEACHER_PHYSICS_HZ,
                "true23_physics_hz": TRUE23_PHYSICS_HZ,
                "source_model": _external_file_binding(source_model_path),
                "target_model": _external_file_binding(target_model_path),
                "true23_sim_config": _external_file_binding(sim_config_path),
            },
            "teacher_reference": {
                "controller_semantics": TEACHER_CONTROLLER_SEMANTICS,
                "checkpoint": _external_file_binding(checkpoint_path),
                "controller_config": controller_payloads["teacher_controller_config"],
                "input_action_law": controller_payloads["teacher_input_action_law"],
                "semantic_reference": controller_payloads["teacher_semantic_reference"],
            },
            "true23_expert": {
                "controller_semantics": TRUE23_CONTROLLER_SEMANTICS,
                "controller_config": controller_payloads["true23_controller_config"],
                "input_action_law": controller_payloads["true23_input_action_law"],
                "online_task_space_expert": False,
                "clips": [
                    {
                        "clip_id": paths.clip_id,
                        "source_csv": _external_file_binding(paths.source_csv),
                        "motion": _external_file_binding(paths.motion),
                        "expert": _external_file_binding(paths.expert),
                        "report": _external_file_binding(paths.report),
                    }
                    for paths in clip_paths
                ],
            },
            "pairs": pair_records,
            "qualification_envelope": (
                None
                if report_path is None or campaign_ref is None or pin_ref is None
                else {
                    "report": {
                        "file": report_path.relative_to(temporary).as_posix(),
                        "sha256": sha256_file(report_path),
                        "payload_sha256": sha256_bytes(
                            canonical_json_bytes(json.loads(report_path.read_text(encoding="utf-8")))
                        ),
                    },
                    "campaign_manifest": campaign_ref,
                    "identity_pin_suggestions": pin_ref,
                }
            ),
            "training_authorized": False,
            "deployment_ready": False,
        }
        manifest_path = temporary / "diagnostic_manifest.json"
        _write_json_atomic(manifest_path, manifest)
        temporary.replace(output_root.resolve())
        return output_root.resolve() / (
            "report.json" if qualification_shape_complete else "diagnostic_manifest.json"
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_plan(
    *,
    checkpoint_path: Path,
    clip_paths: Sequence[FrozenClipPaths],
    contract_path: Path,
    device: str,
    episodes_per_clip: int,
    horizon_steps: int,
) -> dict[str, Any]:
    """Validate cheap immutable identities and describe work without simulation."""

    if sha256_file(checkpoint_path.resolve()) != LOW_LATENCY_RELEASE_SHA256:
        raise ValueError("teacher checkpoint hash differs from pinned low-latency release")
    if sha256_file(REPOSITORY_ROOT / SOURCE_MODEL_RELPATH) != SOURCE_MODEL_SHA256:
        raise ValueError("source29 model hash drift")
    if sha256_file(REPOSITORY_ROOT / TARGET_MODEL_RELPATH) != TARGET_MODEL_SHA256:
        raise ValueError("true23 model hash drift")
    for paths in clip_paths:
        for path in (paths.source_csv, paths.motion, paths.expert, paths.report):
            if not path.resolve().is_file():
                raise FileNotFoundError(path)
    return {
        "schema_version": 1,
        "kind": "g1_true23_idle_step1b_mujoco_plan_v1",
        "authorization": "none",
        "device": device,
        "control_hz": CONTROL_HZ,
        "horizon_steps": horizon_steps,
        "episodes_per_clip": episodes_per_clip,
        "pair_count": len(clip_paths) * episodes_per_clip,
        "teacher_trace_count": len(clip_paths) * episodes_per_clip,
        "true23_trace_count": len(clip_paths) * episodes_per_clip,
        "physics_steps": len(clip_paths)
        * episodes_per_clip
        * horizon_steps
        * (TEACHER_DECIMATION + TRUE23_DECIMATION),
        "contract": _external_file_binding(contract_path),
        "checkpoint": _external_file_binding(checkpoint_path),
        "source_model": _external_file_binding(REPOSITORY_ROOT / SOURCE_MODEL_RELPATH),
        "target_model": _external_file_binding(REPOSITORY_ROOT / TARGET_MODEL_RELPATH),
        "clips": [
            {
                "clip_id": paths.clip_id,
                "frame_count": paths.expected_frame_count,
                "source_csv": _external_file_binding(paths.source_csv),
                "motion": _external_file_binding(paths.motion),
                "expert": _external_file_binding(paths.expert),
                "report": _external_file_binding(paths.report),
            }
            for paths in clip_paths
        ],
        "controller_semantics": {
            "teacher_reference": TEACHER_CONTROLLER_SEMANTICS,
            "true23_expert": TRUE23_CONTROLLER_SEMANTICS,
        },
        "training_authorized": False,
        "deployment_ready": False,
    }
