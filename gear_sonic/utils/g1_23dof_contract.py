"""Authoritative G1 23-DoF embodiment and policy-shape contract.

Index spaces in this file are deliberately named.  The released SONIC policy is
29-DoF in IsaacLab order, while G1 ``mode_machine == 4`` exposes 23 physical
motors in hardware/MuJoCo order.  Mixing those spaces silently produces unsafe
joint commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

ROBOT_MODEL = "g1_23dof_rev_1_0"
REQUIRED_MODE_MACHINE = 4
TARGET_DOF = 23
SOURCE_DOF = 29
DEPLOYMENT_HISTORY_LENGTH = 10
TOKEN_DIM = 64
TELEOP_ENCODER_INPUT_DIM = 267
TELEOP_ENCODER_INPUT_TERM_ORDER = (
    "command_multi_future_lower_body",
    "vr_3point_local_target",
    "vr_3point_local_orn_target",
    "motion_anchor_ori_b",
)
TELEOP_ENCODER_INPUT_TERM_DIMS = (240, 9, 12, 6)
# The reference term is the first encoder term and is laid out as all future
# positions followed by all future velocities.  Each future frame contributes
# this many values to each of those two halves, so the whole term is
# ``2 * REFERENCE_FRAME_TERM_WIDTH * future_frame_count``.  The three trailing
# terms are independent of the future horizon.
REFERENCE_FRAME_TERM_WIDTH = 12
TELEOP_ENCODER_TRAILING_TERM_DIMS = TELEOP_ENCODER_INPUT_TERM_DIMS[1:]
TELEOP_TOKEN_COUNT = 2
TELEOP_TOKEN_WIDTH = 32
TELEOP_FSQ_LEVEL = 32
DEPLOYMENT_DECODER_INPUT_DIM = 994
SIM_VALIDATION_SCHEMA_VERSION = 2
ARTIFACT_SCHEMA_VERSION = 3

# Teleop encoder shape stays 267 for both released policy families. What
# changes is the temporal meaning of its 240-value lower-body reference term.
# Keep this contract here, alongside artifact metadata, so profile selection
# cannot be inferred from tensor shape or silently relabelled at deployment.
REFERENCE_SOURCE_SAMPLE_RATE_HZ = 50
REFERENCE_SOURCE_SAMPLE_PERIOD_S = 0.02
REFERENCE_FUTURE_FRAME_COUNT = 10
REFERENCE_PROFILE_NORMAL = "true23_step5_0p1s"
REFERENCE_PROFILE_LOW_LATENCY = "released_low_latency_step1_0p02s"
# Teleoperation profile.  Two frames at 20 ms put the intrinsic measured-future
# delay at 40 ms, inside the 60 ms active-policy freshness budget; the released
# ten-frame profiles need 200 ms and 920 ms respectively and so cannot be fed
# from live operator motion without synthesizing a future.  This profile has a
# different encoder input width and therefore requires its own trained weights.
REFERENCE_PROFILE_TELEOP_2F = "true23_teleop_step1_2frame_v1"
REFERENCE_TELEOP_2F_FRAME_COUNT = 2
DEFAULT_REFERENCE_PROFILE = REFERENCE_PROFILE_NORMAL
NORMAL_RELEASE_SHA256 = (
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
)
LOW_LATENCY_RELEASE_SHA256 = (
    "0031ae7db24747445d6eb7c27697640973a837546f0b8763e775143c47d4507c"
)
LOW_LATENCY_RELEASE_HF_REVISION = (
    "7c90a56cfe04788c4f041daeef5b1e12930675ad"
)
NORMAL_INITIAL_POLICY_STATE_SHA256 = (
    "c247e5cf8bf06bc954db314013cba5ed8b56b6fe4c9a952c19f053583714f0bc"
)
LOW_LATENCY_INITIAL_POLICY_STATE_SHA256 = (
    "39049d5018608f198dee1d2ea5e0f465d212dc4572c036d21378a36f80c1fc2f"
)
MINIMUM_TRAINING_UPDATES = 50


@dataclass(frozen=True)
class ReferenceProfile:
    """Immutable temporal meaning of the encoder reference term.

    The two released profiles carry ten future frames, giving the 240-value
    term the released checkpoints were trained with.  Teleoperation profiles may
    carry fewer frames: a future reference built only from measurements cannot
    be produced until those samples exist, so the horizon is a hard lower bound
    on producer delay (see ``measured_future_latency_proof``).
    """

    name: str
    future_frame_step: int
    future_frame_offsets_s: tuple[float, ...]

    @property
    def horizon_s(self) -> float:
        return self.future_frame_offsets_s[-1]

    @property
    def future_frame_count(self) -> int:
        return len(self.future_frame_offsets_s)

    @property
    def reference_term_dim(self) -> int:
        """Width of the leading ``command_multi_future_lower_body`` term."""
        return 2 * REFERENCE_FRAME_TERM_WIDTH * self.future_frame_count

    @property
    def encoder_input_term_dims(self) -> tuple[int, ...]:
        return (self.reference_term_dim, *TELEOP_ENCODER_TRAILING_TERM_DIMS)

    @property
    def encoder_input_dim(self) -> int:
        return sum(self.encoder_input_term_dims)

    @property
    def command_layout(self) -> str:
        half = REFERENCE_FRAME_TERM_WIDTH * self.future_frame_count
        return f"positions_{half}_then_velocities_{half}"


def _reference_profile(
    name: str,
    step: int,
    frame_count: int = REFERENCE_FUTURE_FRAME_COUNT,
) -> ReferenceProfile:
    if frame_count < 1:
        raise ValueError("reference profiles need at least one future frame")
    return ReferenceProfile(
        name=name,
        future_frame_step=step,
        future_frame_offsets_s=tuple(
            frame * step / REFERENCE_SOURCE_SAMPLE_RATE_HZ
            for frame in range(frame_count)
        ),
    )


REFERENCE_PROFILES: Mapping[str, ReferenceProfile] = MappingProxyType(
    {
        REFERENCE_PROFILE_NORMAL: _reference_profile(
            REFERENCE_PROFILE_NORMAL,
            5,
        ),
        REFERENCE_PROFILE_LOW_LATENCY: _reference_profile(
            REFERENCE_PROFILE_LOW_LATENCY,
            1,
        ),
    }
)
# Teleoperation profiles are deliberately kept out of REFERENCE_PROFILES.
# That mapping enumerates the released profiles that carry approved material
# config hashes and runtime contracts; every entry in it must have one. A
# teleop profile has a different encoder width, so it needs its own trained
# weights and its own approvals before it can join.
TELEOP_REFERENCE_PROFILES: Mapping[str, ReferenceProfile] = MappingProxyType(
    {
        REFERENCE_PROFILE_TELEOP_2F: _reference_profile(
            REFERENCE_PROFILE_TELEOP_2F,
            1,
            frame_count=REFERENCE_TELEOP_2F_FRAME_COUNT,
        ),
    }
)
ALL_REFERENCE_PROFILES: Mapping[str, ReferenceProfile] = MappingProxyType(
    {**REFERENCE_PROFILES, **TELEOP_REFERENCE_PROFILES}
)


def reference_profile_by_name(profile: str) -> ReferenceProfile:
    """Look up a released or teleoperation profile by name."""
    try:
        return ALL_REFERENCE_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported true23 reference profile: {profile}") from exc
APPROVED_WARM_START_RELEASES: Mapping[str, Mapping[str, Any]] = (
    MappingProxyType(
        {
            NORMAL_RELEASE_SHA256: MappingProxyType(
                {
                    "source_family": "sonic_release",
                    "reference_profile": REFERENCE_PROFILE_NORMAL,
                    "source_revision": None,
                    "initial_policy_state_sha256": (
                        NORMAL_INITIAL_POLICY_STATE_SHA256
                    ),
                }
            ),
            LOW_LATENCY_RELEASE_SHA256: MappingProxyType(
                {
                    "source_family": "sonic_low_latency",
                    "reference_profile": REFERENCE_PROFILE_LOW_LATENCY,
                    "source_revision": LOW_LATENCY_RELEASE_HF_REVISION,
                    "initial_policy_state_sha256": (
                        LOW_LATENCY_INITIAL_POLICY_STATE_SHA256
                    ),
                }
            ),
        }
    )
)


def reference_profile_contract(profile: str) -> dict[str, Any]:
    """Return JSON-safe exact contract for one approved temporal profile."""

    value = reference_profile_by_name(profile)
    return {
        "profile": value.name,
        "source_sample_rate_hz": REFERENCE_SOURCE_SAMPLE_RATE_HZ,
        "source_sample_period_s": REFERENCE_SOURCE_SAMPLE_PERIOD_S,
        "future_frame_count": value.future_frame_count,
        "future_frame_step": value.future_frame_step,
        "future_frame_offsets_s": list(value.future_frame_offsets_s),
        "horizon_s": value.horizon_s,
        "command_layout": value.command_layout,
    }

# Unitree LowState motor slots present on a mode_machine=4 G1.
HARDWARE_JOINT_IDS = tuple(range(13)) + tuple(range(15, 20)) + tuple(range(22, 27))
EXCLUDED_HARDWARE_JOINT_IDS = (13, 14, 20, 21, 27, 28)

# Live slots in the released canonical IsaacLab-29 order. This sorted selector
# describes padded observations only; it is not the native rev-1.0 PhysX order.
SOURCE_IL29_KEEP_INDICES = (
    0,
    1,
    2,
    3,
    4,
    6,
    7,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
)
SOURCE_IL29_EXCLUDED_INDICES = (5, 8, 25, 26, 27, 28)

# Native rev-1.0 PhysX/IsaacLab joint order is breadth-first over the reduced
# URDF tree. Removing waist roll/pitch changes where both arm chains appear;
# simply filtering the old 29-joint order is therefore wrong.
# Usage: canonical_il29[NATIVE_IL23_TO_CANONICAL_IL29[i]] = native_il23[i].
NATIVE_IL23_TO_CANONICAL_IL29 = (
    0,
    1,
    2,
    3,
    4,
    11,
    12,
    6,
    7,
    15,
    16,
    9,
    10,
    19,
    20,
    13,
    14,
    21,
    22,
    17,
    18,
    23,
    24,
)

# Exact order of the 12 lower-body channels seen by the causal tokenizer.
# Training motion stores compact MuJoCo hardware order: left leg, then right.
CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES = (
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
)

# Select compact MuJoCo-23 motion pose from a MuJoCo-29 source pose.
SOURCE_MJ29_KEEP_INDICES = HARDWARE_JOINT_IDS

# Rectangular selector: source MuJoCo-29 -> native PhysX/IsaacLab-23.
# This is not the inverse of either compact permutation below.
SOURCE_MJ29_TO_TARGET_IL23 = (
    0,
    6,
    12,
    1,
    7,
    15,
    22,
    2,
    8,
    16,
    23,
    3,
    9,
    17,
    24,
    4,
    10,
    18,
    25,
    5,
    11,
    19,
    26,
)

# Runtime permutations after both sides are already compact 23-DoF.
# Usage matches existing repo convention: x_mj = x_il[ISAACLAB_TO_MUJOCO_DOF].
ISAACLAB_TO_MUJOCO_DOF = (
    0,
    3,
    7,
    11,
    15,
    19,
    1,
    4,
    8,
    12,
    16,
    20,
    2,
    5,
    9,
    13,
    17,
    21,
    6,
    10,
    14,
    18,
    22,
)
MUJOCO_TO_ISAACLAB_DOF = (
    0,
    6,
    12,
    1,
    7,
    13,
    18,
    2,
    8,
    14,
    19,
    3,
    9,
    15,
    20,
    4,
    10,
    16,
    21,
    5,
    11,
    17,
    22,
)

SOURCE_IL29_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

CANONICAL_COMPACT_IL23_JOINT_NAMES = tuple(SOURCE_IL29_JOINT_NAMES[index] for index in SOURCE_IL29_KEEP_INDICES)

NATIVE_IL23_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
)
# Compatibility alias for artifact code; target output order is native, never
# the sorted canonical compact order above.
TARGET_IL23_JOINT_NAMES = NATIVE_IL23_JOINT_NAMES

HARDWARE_23_JOINT_NAMES = (
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
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)

# Unitree's rev-1.0 23-DoF deployment scales in hardware/MuJoCo order.
# Position scale is independent of the 35 Nm ankle torque clamp.
HARDWARE_23_ACTION_SCALE = (
    0.55,
    0.35,
    0.55,
    0.35,
    0.44,
    0.44,
    0.55,
    0.35,
    0.55,
    0.35,
    0.44,
    0.44,
    0.55,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
)
NATIVE_IL23_ACTION_SCALE = tuple(HARDWARE_23_ACTION_SCALE[index] for index in MUJOCO_TO_ISAACLAB_DOF)

OBSERVATION_TERM_ORDER = (
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel",
    "previous_action",
    "projected_gravity",
)
HISTORY_ORDER = "oldest_to_newest"
MISSING_OBSERVATION_FILL = {
    "joint_pos_rel": "fixed_default_relative_zero",
    "joint_vel": "zero",
    "previous_action": "zero_every_history_frame",
}
REQUIRED_SIM_SCENARIOS = ("nominal", "disturbance_50", "disturbance_100")

OBS_LAYOUT_PADDED_IL29 = "canonical_il29_fixed_slots_v1"
# Backward-compatible name for checkpoint tooling.  Trained policies may retain
# this exact padded layout; training stage, not padding itself, controls readiness.
OBS_LAYOUT_CHECKPOINT_INIT_29 = OBS_LAYOUT_PADDED_IL29
OBS_LAYOUT_NATIVE_23 = "native_il23_v1"
DECODER_OUTPUT_LAYOUT = "native_physx_il23_bfs_v1"


@dataclass(frozen=True)
class DecoderShape:
    """Decoder input/output dimensions for one observation layout."""

    history_length: int
    observation_dof: int
    token_dim: int = 64

    @property
    def proprioception_dim(self) -> int:
        # base angular velocity, q, dq, previous action, gravity direction.
        return self.history_length * (3 + self.observation_dof * 3 + 3)

    @property
    def input_dim(self) -> int:
        return self.token_dim + self.proprioception_dim

    @property
    def output_dim(self) -> int:
        return TARGET_DOF


def decoder_shape(
    history_length: int,
    observation_layout: str = OBS_LAYOUT_PADDED_IL29,
    token_dim: int = 64,
) -> DecoderShape:
    """Return decoder shape; reject unknown layouts instead of guessing."""
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    if observation_layout == OBS_LAYOUT_PADDED_IL29:
        observation_dof = SOURCE_DOF
    elif observation_layout == OBS_LAYOUT_NATIVE_23:
        observation_dof = TARGET_DOF
    else:
        raise ValueError(f"unsupported observation layout: {observation_layout}")
    return DecoderShape(history_length, observation_dof, token_dim)


def decoder_input_keep_indices_29_to_23(history_length: int, token_dim: int = 64) -> tuple[int, ...]:
    """Select native-23 decoder columns from canonical padded-29 input.

    Actor observation block order is:
    ``base_ang_vel, q, dq, previous_action, gravity_dir``.  Each term stores
    ``history_length`` frames contiguously, oldest to newest.
    """
    source = decoder_shape(history_length, OBS_LAYOUT_PADDED_IL29, token_dim)
    target = decoder_shape(history_length, OBS_LAYOUT_NATIVE_23, token_dim)
    keep = list(range(token_dim))
    offset = token_dim

    # base_ang_vel history
    keep.extend(range(offset, offset + 3 * history_length))
    offset += 3 * history_length

    # q, dq, previous action histories
    for _ in range(3):
        for frame in range(history_length):
            frame_start = offset + frame * SOURCE_DOF
            keep.extend(frame_start + index for index in NATIVE_IL23_TO_CANONICAL_IL29)
        offset += SOURCE_DOF * history_length

    # gravity history
    keep.extend(range(offset, offset + 3 * history_length))
    offset += 3 * history_length

    if offset != source.input_dim or len(keep) != target.input_dim:
        raise AssertionError("decoder column contract is internally inconsistent")
    return tuple(keep)


def expand_il23_to_il29(
    values: Sequence[float],
    *,
    excluded_values: Sequence[float] = (0.0,) * len(SOURCE_IL29_EXCLUDED_INDICES),
) -> tuple[float, ...]:
    """Expand one compact vector for checkpoint initialization only.

    For relative-q, use explicit fixed/default-relative values (normally zero).
    For dq and previous action, use zeros.  This function is not an action
    adapter and must never be used to turn a 29-output policy into deployment.
    """
    if len(values) != TARGET_DOF:
        raise ValueError(f"expected {TARGET_DOF} values, got {len(values)}")
    if len(excluded_values) != len(SOURCE_IL29_EXCLUDED_INDICES):
        raise ValueError("excluded_values must contain six values")
    expanded = [0.0] * SOURCE_DOF
    for target_index, source_index in enumerate(NATIVE_IL23_TO_CANONICAL_IL29):
        expanded[source_index] = values[target_index]
    for source_index, value in zip(SOURCE_IL29_EXCLUDED_INDICES, excluded_values, strict=True):
        expanded[source_index] = value
    return tuple(expanded)


def make_artifact_metadata(
    *,
    history_length: int,
    observation_layout: str,
    checkpoint_stage: str,
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
    deployment_ready: bool = False,
    sim_validation_passed: bool = False,
    simulation_evidence: Mapping[str, Any] | None = None,
    artifact_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build metadata consumed by offline export/deployment gates."""
    shape = decoder_shape(history_length, observation_layout)
    reference_contract = reference_profile_contract(reference_profile)
    if deployment_ready or sim_validation_passed:
        if checkpoint_stage != "trained":
            raise ValueError("ready metadata requires checkpoint_stage='trained'")
        if deployment_ready is not True or sim_validation_passed is not True:
            raise ValueError("deployment readiness and simulation validation must both pass")
        if simulation_evidence is None:
            raise ValueError("ready metadata requires validated simulation evidence")
        if artifact_hashes is None:
            raise ValueError("ready metadata requires artifact hashes")

    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "robot_model": ROBOT_MODEL,
        "mode_machine": REQUIRED_MODE_MACHINE,
        "action_dof": TARGET_DOF,
        "hardware_joint_ids": list(HARDWARE_JOINT_IDS),
        "excluded_hardware_joint_ids": list(EXCLUDED_HARDWARE_JOINT_IDS),
        "decoder_output_layout": DECODER_OUTPUT_LAYOUT,
        "observation_layout": observation_layout,
        "history_length": history_length,
        "decoder_input_dim": shape.input_dim,
        "decoder_output_dim": TARGET_DOF,
        "reference_profile": reference_profile,
        "reference_contract": reference_contract,
        "checkpoint_stage": checkpoint_stage,
        "deployment_ready": deployment_ready,
        "sim_validation_passed": sim_validation_passed,
        "naive_output_masking": False,
        "observation_contract": {
            "token_dim": TOKEN_DIM,
            "proprioception_dim": shape.proprioception_dim,
            "term_order": list(OBSERVATION_TERM_ORDER),
            "history_order": HISTORY_ORDER,
            "source_il29_keep_indices": list(SOURCE_IL29_KEEP_INDICES),
            "source_il29_excluded_indices": list(SOURCE_IL29_EXCLUDED_INDICES),
            "source_il29_joint_names": list(SOURCE_IL29_JOINT_NAMES),
            "native_il23_to_canonical_il29": list(NATIVE_IL23_TO_CANONICAL_IL29),
            "missing_fill": dict(MISSING_OBSERVATION_FILL),
        },
        "action_contract": {
            "native_il23_joint_names": list(NATIVE_IL23_JOINT_NAMES),
            "canonical_compact_il23_joint_names": list(CANONICAL_COMPACT_IL23_JOINT_NAMES),
            "hardware_joint_names": list(HARDWARE_23_JOINT_NAMES),
            "hardware_action_scale": list(HARDWARE_23_ACTION_SCALE),
            "native_il23_action_scale": list(NATIVE_IL23_ACTION_SCALE),
            "isaaclab_to_mujoco_dof": list(ISAACLAB_TO_MUJOCO_DOF),
            "mujoco_to_isaaclab_dof": list(MUJOCO_TO_ISAACLAB_DOF),
        },
    }
    if simulation_evidence is not None:
        metadata["simulation_evidence"] = dict(simulation_evidence)
    if artifact_hashes is not None:
        metadata["hashes"] = dict(artifact_hashes)
    return metadata


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
    )


def _validate_simulation_evidence(metadata: Mapping[str, Any], errors: list[str]) -> None:
    evidence = metadata.get("simulation_evidence")
    if not isinstance(evidence, Mapping):
        errors.append("simulation_evidence must be an evidence mapping")
        return

    expected = {
        "schema_version": SIM_VALIDATION_SCHEMA_VERSION,
        "computed_pass": True,
        "required_scenarios": list(REQUIRED_SIM_SCENARIOS),
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            errors.append(f"simulation_evidence.{key}: expected {value!r}, got {evidence.get(key)!r}")

    for key in (
        "report_sha256",
        "report_payload_sha256",
        "checkpoint_sha256",
    ):
        if not _is_sha256(evidence.get(key)):
            errors.append(f"simulation_evidence.{key} must be lowercase SHA-256")

    run_count = evidence.get("run_count")
    if isinstance(run_count, bool) or not isinstance(run_count, int):
        errors.append("simulation_evidence.run_count must be an integer")
    elif run_count < len(REQUIRED_SIM_SCENARIOS):
        errors.append("simulation_evidence.run_count does not cover required scenarios")


def _validate_artifact_hashes(metadata: Mapping[str, Any], errors: list[str]) -> None:
    hashes = metadata.get("hashes")
    if not isinstance(hashes, Mapping):
        errors.append("hashes must be an artifact hash mapping")
        return
    required_hashes = (
        "checkpoint_sha256",
        "policy_state_sha256",
        "encoder_state_sha256",
        "decoder_state_sha256",
        "encoder_onnx_sha256",
        "decoder_onnx_sha256",
        "sim_report_sha256",
        "sim_report_payload_sha256",
        "contract_sha256",
        "robot_asset_sha256",
        "robot_config_sha256",
        "sim_config_sha256",
        "encoder_config_sha256",
        "decoder_config_sha256",
        "policy_config_sha256",
        "encoder_embedded_metadata_sha256",
        "decoder_embedded_metadata_sha256",
    )
    for key in required_hashes:
        if not _is_sha256(hashes.get(key)):
            errors.append(f"hashes.{key} must be lowercase SHA-256")


def validate_artifact_contract(
    metadata: Mapping[str, Any],
    *,
    decoder_input_dim: int,
    decoder_output_dim: int,
    require_deployment_ready: bool = True,
) -> None:
    """Fail closed on wrong embodiment, shape, mode, or training stage."""
    expected_pairs = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "robot_model": ROBOT_MODEL,
        "mode_machine": REQUIRED_MODE_MACHINE,
        "action_dof": TARGET_DOF,
        "hardware_joint_ids": list(HARDWARE_JOINT_IDS),
        "excluded_hardware_joint_ids": list(EXCLUDED_HARDWARE_JOINT_IDS),
        "decoder_output_layout": DECODER_OUTPUT_LAYOUT,
        "decoder_output_dim": TARGET_DOF,
        "naive_output_masking": False,
    }
    errors = [
        f"{key}: expected {expected!r}, got {metadata.get(key)!r}"
        for key, expected in expected_pairs.items()
        if metadata.get(key) != expected
    ]

    reference_profile = metadata.get("reference_profile")
    try:
        expected_reference_contract = reference_profile_contract(
            str(reference_profile)
        )
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if metadata.get("reference_contract") != expected_reference_contract:
            errors.append(
                "reference_contract does not match the exact immutable temporal profile"
            )

    expected_observation_contract = {
        "token_dim": TOKEN_DIM,
        "proprioception_dim": decoder_shape(
            int(metadata.get("history_length", 0)),
            str(metadata.get("observation_layout", "")),
        ).proprioception_dim
        if isinstance(metadata.get("history_length"), int)
        and metadata.get("history_length", 0) > 0
        and metadata.get("observation_layout") in {OBS_LAYOUT_PADDED_IL29, OBS_LAYOUT_NATIVE_23}
        else None,
        "term_order": list(OBSERVATION_TERM_ORDER),
        "history_order": HISTORY_ORDER,
        "source_il29_keep_indices": list(SOURCE_IL29_KEEP_INDICES),
        "source_il29_excluded_indices": list(SOURCE_IL29_EXCLUDED_INDICES),
        "source_il29_joint_names": list(SOURCE_IL29_JOINT_NAMES),
        "native_il23_to_canonical_il29": list(NATIVE_IL23_TO_CANONICAL_IL29),
        "missing_fill": dict(MISSING_OBSERVATION_FILL),
    }
    if metadata.get("observation_contract") != expected_observation_contract:
        errors.append("observation_contract does not match the exact true23 contract")

    expected_action_contract = {
        "native_il23_joint_names": list(NATIVE_IL23_JOINT_NAMES),
        "canonical_compact_il23_joint_names": list(CANONICAL_COMPACT_IL23_JOINT_NAMES),
        "hardware_joint_names": list(HARDWARE_23_JOINT_NAMES),
        "hardware_action_scale": list(HARDWARE_23_ACTION_SCALE),
        "native_il23_action_scale": list(NATIVE_IL23_ACTION_SCALE),
        "isaaclab_to_mujoco_dof": list(ISAACLAB_TO_MUJOCO_DOF),
        "mujoco_to_isaaclab_dof": list(MUJOCO_TO_ISAACLAB_DOF),
    }
    if metadata.get("action_contract") != expected_action_contract:
        errors.append("action_contract does not match the exact native/hardware contract")

    try:
        shape = decoder_shape(
            int(metadata["history_length"]),
            str(metadata["observation_layout"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid observation metadata: {exc}")
    else:
        if metadata.get("decoder_input_dim") != shape.input_dim:
            errors.append("metadata decoder_input_dim does not match observation layout")
        if decoder_input_dim != shape.input_dim:
            errors.append(f"artifact decoder input: expected {shape.input_dim}, got {decoder_input_dim}")

    if decoder_output_dim != TARGET_DOF:
        errors.append(f"artifact decoder output: expected {TARGET_DOF}, got {decoder_output_dim}")

    if require_deployment_ready:
        if metadata.get("observation_layout") != OBS_LAYOUT_PADDED_IL29:
            errors.append("deployment contract requires canonical_il29_fixed_slots_v1 observations")
        if metadata.get("checkpoint_stage") != "trained":
            errors.append("checkpoint_stage must be 'trained'")
        if metadata.get("deployment_ready") is not True:
            errors.append("deployment_ready must be true")
        if metadata.get("sim_validation_passed") is not True:
            errors.append("sim_validation_passed must be true")
        if metadata.get("history_length") != DEPLOYMENT_HISTORY_LENGTH:
            errors.append(f"history_length must be exactly {DEPLOYMENT_HISTORY_LENGTH} for deployment")
        if decoder_input_dim != DEPLOYMENT_DECODER_INPUT_DIM:
            errors.append(f"deployment decoder input must be exactly {DEPLOYMENT_DECODER_INPUT_DIM}")
        _validate_simulation_evidence(metadata, errors)
        _validate_artifact_hashes(metadata, errors)

    if errors:
        raise ValueError("G1 23-DoF artifact contract failed: " + "; ".join(errors))
