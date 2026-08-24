"""Fail-closed contract for PICO full-body -> canonical G1 IL29 retargeting.

This module does not retarget poses by approximation.  It defines and verifies
the complete raw-capture and output contracts needed before the official
NVIDIA SOMA Retargeter may become a ``SOURCE_RETARGETED_DELAYED`` producer.

Current deployment remains blocked because:

* no live hardened BodyTracking snapshot has proved all 24 fused roles,
  reconstructed derivatives, timestamps, and two-band health;
* the checked-in PICO ``PoseStreamer`` writes wrist joints into a zero IL29
  vector and emits zero velocity;
* the validated XR24-to-SOMA coordinate/role adapter and exact replay
  attestation do not exist yet.

PICO BodyTracking and raw MotionTracking are mutually exclusive modes.  This
contract therefore uses the calibrated BodyTracking solution, where ankle
bands are fused into XR24 ``Left_Ankle``/``Right_Ankle`` roles 7/8.  It never
pairs a retained MotionTracking snapshot or invents raw ankle poses.

No function opens DDS, ZMQ, ADB, an XR service, or a robot channel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from importlib import metadata, util
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from gear_sonic.utils.g1_23dof_contract import SOURCE_DOF, SOURCE_IL29_JOINT_NAMES
from gear_sonic.utils.g1_23dof_semantic_reference import (
    SOURCE_RETARGETED_DELAYED,
    SOURCE_SAMPLE_PERIOD_NS,
    required_buffer_frames,
)

RAW_CAPTURE_SCHEMA_VERSION = 3
LEGACY_RAW_CAPTURE_SCHEMA_VERSION = 2
RAW_CAPTURE_KIND = "xrobotoolkit_pico_xr24_bodytracking_ankle_fused_raw_capture"
RAW_CAPTURE_FRAME_KIND = (
    "xrobotoolkit_pico_xr24_bodytracking_ankle_fused_atomic_frame"
)
RETARGET_TRACE_SCHEMA_VERSION = 1
RETARGET_TRACE_KIND = "g1_il29_soma_retargeted_trace"
RETARGET_PRODUCER_KIND = "nvidia_soma_retargeter_xr24_adapter"
RETARGET_REPLAY_STATUS_BLOCKED = "blocked_pending_exact_soma_replay"
RETARGET_REPLAY_STATUS_EXACT_NONPROMOTABLE = (
    "exact_soma_replay_verified_pending_live_xr24_approval"
)
ATOMIC_SNAPSHOT_CONTRACT = "xrt_xr24_body_tracker_fused_ankles_atomic_v1"
DERIVATIVE_LAYOUT_CONTRACT = "linear_xyz_then_angular_xyz_v1"
POSITION_DERIVED_CONTROL_DERIVATIVE_CONTRACT = (
    "soma_il29_q_50hz_forward_difference_dq_v1"
)
SOURCE_COHERENCE_CONTRACT = "same_packet_xr24_body_tracking_v1"
POSITIVE_JOINT_TIMESTAMP_CONTRACT = "pico_local_pose_timestamp_positive_v1"
HEAD_ONLY_JOINT_TIMESTAMP_CONTRACT = "pico_local_pose_head_only_timestamp_v1"
UNAVAILABLE_JOINT_TIMESTAMP_CONTRACT = (
    "pico_local_pose_timestamp_unavailable_zero_v1"
)
LOCAL_POSE_BODY_TIMESTAMP_CONTRACT = "latest_pico_local_pose_timestamp_v1"
HEAD_LOCAL_POSE_BODY_TIMESTAMP_CONTRACT = "pico_head_local_pose_timestamp_v1"
PACKET_HEALTH_BODY_TIMESTAMP_CONTRACT = "same_packet_health_timestamp_v1"
ANKLE_ROLE_SEMANTICS = "bodytracking_fused_roles_7_8_from_calibrated_ankle_bands_v1"
BODY_TRACKING_MODE_CODE = 0
BODY_TRACKING_SUCCESS_CODE = 0
BODY_TRACKING_VALID_STATE_CODE = 1
XR24_SOMA_ADAPTER_VERSION = "xr24_soma_roles_v1"

PINNED_SOMA_REPOSITORY = "https://github.com/NVIDIA/soma-retargeter.git"
PINNED_SOMA_COMMIT = "b3ef2708d84bfd1314ddb52d0db6c9c211df1f57"
PINNED_SOMA_PACKAGE_VERSION = "0.1.0"
PINNED_NEWTON_VERSION = "1.0.0"
PINNED_WARP_VERSION = "1.12.0"
PINNED_CONFIG_HASH_SEMANTICS = "utf8_lf_normalized_sha256"
PINNED_CONFIG_SHA256 = {
    "soma_to_g1_retargeter_config.json": (
        "befc09515e3b4f75f85561ec757fd795d03904abade21bb9ed9940363797953e"
    ),
    "soma_to_g1_scaler_config.json": (
        "1f9da8ae28500a27bea90d2aaf3949df4992e3db7ca0a92fcc58cf8080854efb"
    ),
    "g1_feet_stabilizer_config.json": (
        "68ab0dc2318eb91272d6c067de6c3b85c27de6cdab83a7d9d09013f9b75dd738"
    ),
}

# Approval deliberately cannot be enabled by an argument or environment
# variable.  Code and tests must change after an exact replay fixture exists.
XR24_SOMA_ADAPTER_APPROVED = False

XRT_BODY_JOINT_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)
XRT_BODY_PARENT_INDICES = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
)
ANKLE_ROLE_TO_BODY_INDEX = {"left_ankle": 7, "right_ankle": 8}

# Official SOMA Retargeter UnitreeG129DOF_CSVConfig order.
SOMA_MJ29_JOINT_NAMES = (
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
SOMA_MJ29_TO_CANONICAL_IL29 = tuple(
    SOURCE_IL29_JOINT_NAMES.index(name) for name in SOMA_MJ29_JOINT_NAMES
)
if sorted(SOMA_MJ29_TO_CANONICAL_IL29) != list(range(SOURCE_DOF)):
    raise RuntimeError("SOMA MJ29 -> canonical IL29 mapping is not a permutation")

_AUTHORIZATION = {
    "read_only": True,
    "dds_opened": False,
    "robot_channel_opened": False,
    "actuation_authorized": False,
    "robot_commands_published": False,
}
_CAPTURE_KEYS = {
    "schema_version",
    "kind",
    "session_id",
    "source",
    "frames",
    "authorization",
}
_CAPTURE_SOURCE_KEYS_V2 = {
    "atomic_snapshot_contract",
    "xrobotoolkit_sdk_sha256",
    "pc_service_sha256",
    "pico_client_build",
    "pico_client_apk_sha256",
    "derivative_layout_contract",
    "source_coherence_contract",
    "body_timestamp_contract",
    "joint_timestamp_contract",
    "ankle_role_semantics",
    "ankle_role_indices",
    "body_joint_order",
}
_CAPTURE_SOURCE_KEYS = {
    *_CAPTURE_SOURCE_KEYS_V2,
    "sdk_derivatives_control_usable",
    "control_derivative_contract",
}
_FRAME_KEYS = {
    "schema_version",
    "kind",
    "frame_index",
    "capture_monotonic_ns",
    "body_sample_timestamp_ns",
    "body_timestamp_contract",
    "joint_timestamp_contract",
    "body_sample_sequence",
    "body_poses",
    "body_velocities",
    "body_accelerations",
    "body_joint_timestamps_ns",
    "health_supported",
    "health_available",
    "health_valid",
    "health_schema_version",
    "health_sample_sequence",
    "health_timestamp_ns",
    "health_client_build",
    "health_calibration_result",
    "health_calibrated",
    "health_tracking_mode",
    "health_connect_state_result",
    "health_tracker_count",
    "health_unique_tracker_count",
    "health_body_state_result",
    "health_is_tracking",
    "health_tracking_state_code",
    "health_body_state_code",
    "health_body_error_code",
    "health_connected_band_count",
    "health_body_data_result",
    "health_body_role_count",
}
_TRACE_KEYS = {
    "schema_version",
    "kind",
    "source_kind",
    "source_session_id",
    "raw_capture_sha256",
    "producer",
    "sample_period_ns",
    "joint_order_mj29",
    "joint_order_il29",
    "mj29_to_il29",
    "samples",
    "certification",
    "authorization",
}
_PRODUCER_KEYS = {
    "kind",
    "adapter_sha256",
    "repository",
    "commit",
    "package_version",
    "python_requires",
    "newton_version",
    "warp_version",
    "config_hash_semantics",
    "config_sha256",
    "xr24_soma_adapter_version",
    "ankle_input_semantics",
}
_TRACE_SAMPLE_KEYS = {
    "source_frame_index",
    "reference_monotonic_ns",
    "capture_monotonic_ns",
    "raw_bracket_indices",
    "raw_interpolation_alpha",
    "joint_pos_mj29",
    "joint_pos_il29",
    "joint_vel_il29",
}
_CERTIFICATION_KEYS = {
    "exact_backend_replay_performed",
    "raw_capture_replayed",
    "full_il29_verified",
    "promotion_eligible",
    "status",
}

_MAX_BODY_SPAN_M = 3.0
_MAX_ABS_POSITION_M = 10.0
_MAX_BONE_LENGTH_CHANGE_M = 0.05
_LOWER_BODY_BONES = ((1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11))

_BODY_SNAPSHOT_REQUIRED_FIELDS = {
    "contract",
    "derivative_layout_contract",
    "source_coherence_contract",
    "body_timestamp_contract",
    "joint_timestamp_contract",
    "available",
    "timestamp_ns",
    "sample_sequence",
    "poses",
    "velocities",
    "accelerations",
    "joint_timestamps_ns",
    "health_supported",
    "health_available",
    "health_valid",
    "health_schema_version",
    "health_sample_sequence",
    "health_timestamp_ns",
    "health_client_build",
    "health_calibration_result",
    "health_calibrated",
    "health_tracking_mode",
    "health_connect_state_result",
    "health_tracker_count",
    "health_unique_tracker_count",
    "health_body_state_result",
    "health_is_tracking",
    "health_tracking_state_code",
    "health_body_state_code",
    "health_body_error_code",
    "health_connected_band_count",
    "health_body_data_result",
    "health_body_role_count",
}


class ExactRetargeterUnavailable(RuntimeError):
    """Raised when exact XR24 -> SOMA -> G1 replay is not approved."""


def _exact_mapping(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{context} must contain exact keys {sorted(keys)}")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _finite_vector(value: Any, size: int, context: str) -> list[float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != size
    ):
        raise ValueError(f"{context} must contain exactly {size} values")
    return [_finite(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _finite_matrix(
    value: Any,
    rows: int,
    columns: int,
    context: str,
) -> list[list[float]]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != rows
    ):
        raise ValueError(f"{context} must contain exactly {rows} rows")
    return [
        _finite_vector(row, columns, f"{context}[{index}]")
        for index, row in enumerate(value)
    ]


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_utf8_sha256(path: Path) -> str:
    """Hash pinned text identically for LF and CRLF Git worktrees."""

    payload = path.read_bytes().decode("utf-8")
    normalized = payload.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))


def _validate_quaternions(matrix: Sequence[Sequence[float]], context: str) -> None:
    for index, row in enumerate(matrix):
        norm = math.sqrt(sum(float(value) ** 2 for value in row[3:7]))
        if not 0.95 <= norm <= 1.05:
            raise ValueError(f"{context}[{index}] quaternion is not normalized")


def _body_bone_lengths(body_poses: Sequence[Sequence[float]]) -> tuple[float, ...]:
    result = tuple(
        _distance(body_poses[parent][:3], body_poses[child][:3])
        for parent, child in _LOWER_BODY_BONES
    )
    if any(not 0.02 <= length <= 1.2 for length in result):
        raise ValueError("XR24 lower-body bone length is implausible")
    return result


def validate_bodytracking_xrt_snapshot(
    snapshot: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate exact output from hardened ``xrt.get_body_snapshot()``.

    PICO BodyTracking fuses two calibrated ankle bands into XR24 roles 7/8.
    Raw MotionTracking data is neither required nor accepted because PICO
    exposes BodyTracking and MotionTracking as mutually exclusive modes.
    """

    root = _exact_mapping(
        snapshot,
        _BODY_SNAPSHOT_REQUIRED_FIELDS,
        "XR24 BodyTracking snapshot",
    )
    if root["contract"] != ATOMIC_SNAPSHOT_CONTRACT:
        raise ValueError("XR24 BodyTracking snapshot contract mismatch")
    if root["derivative_layout_contract"] != DERIVATIVE_LAYOUT_CONTRACT:
        raise ValueError(
            "XR24 BodyTracking derivative layout is unversioned or incompatible"
        )
    if root["source_coherence_contract"] != SOURCE_COHERENCE_CONTRACT:
        raise ValueError("XR24 BodyTracking source coherence contract mismatch")
    if root["available"] is not True:
        raise ValueError("XR24 BodyTracking snapshot is unavailable")

    body_timestamp_ns = _integer(
        root["timestamp_ns"],
        "XR24 BodyTracking snapshot.timestamp_ns",
        minimum=1,
    )
    sample_sequence = _integer(
        root["sample_sequence"],
        "XR24 BodyTracking snapshot.sample_sequence",
        minimum=1,
    )
    health_timestamp_ns = _integer(
        root["health_timestamp_ns"],
        "XR24 BodyTracking snapshot.health_timestamp_ns",
        minimum=1,
    )
    health_sample_sequence = _integer(
        root["health_sample_sequence"],
        "XR24 BodyTracking snapshot.health_sample_sequence",
        minimum=1,
    )
    if health_sample_sequence != sample_sequence:
        raise ValueError("XR24 body and hardened health sample sequences differ")
    exact_health_bools = {
        "health_supported": True,
        "health_available": True,
        "health_valid": True,
        "health_calibrated": True,
        "health_is_tracking": True,
    }
    for key, expected in exact_health_bools.items():
        if root[key] is not expected:
            raise ValueError(f"XR24 BodyTracking snapshot.{key} mismatch")
    if root["health_client_build"] != "xrobotoolkit-pico-health-v1":
        raise ValueError("XR24 BodyTracking snapshot.health_client_build mismatch")
    exact_health_integers = {
        "health_schema_version": 1,
        "health_calibration_result": 0,
        "health_tracking_mode": BODY_TRACKING_MODE_CODE,
        "health_connect_state_result": 0,
        "health_body_state_result": 0,
        "health_tracking_state_code": BODY_TRACKING_SUCCESS_CODE,
        "health_body_state_code": BODY_TRACKING_VALID_STATE_CODE,
        "health_body_error_code": 0,
        "health_body_data_result": 0,
        "health_body_role_count": len(XRT_BODY_JOINT_NAMES),
    }
    for key, expected in exact_health_integers.items():
        if _integer(
            root[key],
            f"XR24 BodyTracking snapshot.{key}",
        ) != expected:
            raise ValueError(f"XR24 BodyTracking snapshot.{key} mismatch")
    for key in (
        "health_tracker_count",
        "health_unique_tracker_count",
        "health_connected_band_count",
    ):
        if _integer(
            root[key],
            f"XR24 BodyTracking snapshot.{key}",
        ) < 2:
            raise ValueError(
                f"XR24 BodyTracking snapshot.{key} must prove two ankle bands"
            )

    body_poses = _finite_matrix(
        root["poses"],
        len(XRT_BODY_JOINT_NAMES),
        7,
        "XR24 BodyTracking snapshot.poses",
    )
    _validate_quaternions(body_poses, "XR24 BodyTracking snapshot.poses")
    _body_bone_lengths(body_poses)
    positions = [row[:3] for row in body_poses]
    if any(
        abs(coordinate) > _MAX_ABS_POSITION_M
        for row in positions
        for coordinate in row
    ):
        raise ValueError("XR24 body position is outside safe bounds")
    for axis in range(3):
        axis_values = [row[axis] for row in positions]
        if max(axis_values) - min(axis_values) > _MAX_BODY_SPAN_M:
            raise ValueError("XR24 body span is implausible")
    _finite_matrix(
        root["velocities"],
        len(XRT_BODY_JOINT_NAMES),
        6,
        "XR24 BodyTracking snapshot.velocities",
    )
    _finite_matrix(
        root["accelerations"],
        len(XRT_BODY_JOINT_NAMES),
        6,
        "XR24 BodyTracking snapshot.accelerations",
    )
    joint_timestamps = root["joint_timestamps_ns"]
    if (
        isinstance(joint_timestamps, (str, bytes))
        or not isinstance(joint_timestamps, Sequence)
        or len(joint_timestamps) != len(XRT_BODY_JOINT_NAMES)
    ):
        raise ValueError(
            "XR24 BodyTracking snapshot.joint_timestamps_ns must contain 24 values"
        )
    body_timestamp_contract = root["body_timestamp_contract"]
    joint_timestamp_contract = root["joint_timestamp_contract"]
    decoded_joint_timestamps = [
        _integer(
            value,
            f"XR24 BodyTracking snapshot.joint_timestamps_ns[{index}]",
            minimum=0,
        )
        for index, value in enumerate(joint_timestamps)
    ]
    if joint_timestamp_contract == UNAVAILABLE_JOINT_TIMESTAMP_CONTRACT:
        if any(value != 0 for value in decoded_joint_timestamps):
            raise ValueError(
                "unavailable PICO joint timestamp contract requires 24 exact zeros"
            )
        if body_timestamp_contract != PACKET_HEALTH_BODY_TIMESTAMP_CONTRACT:
            raise ValueError(
                "unavailable PICO joint timestamps require packet-health body time"
            )
        if body_timestamp_ns != health_timestamp_ns:
            raise ValueError(
                "packet-health body timestamp does not match hardened health"
            )
    elif joint_timestamp_contract == HEAD_ONLY_JOINT_TIMESTAMP_CONTRACT:
        if body_timestamp_contract != HEAD_LOCAL_POSE_BODY_TIMESTAMP_CONTRACT:
            raise ValueError(
                "HEAD-only PICO timestamp requires HEAD local-pose body time"
            )
        if decoded_joint_timestamps[15] <= 0 or any(
            value != 0
            for index, value in enumerate(decoded_joint_timestamps)
            if index != 15
        ):
            raise ValueError(
                "HEAD-only PICO timestamp requires index 15 positive and 23 zeros"
            )
        if body_timestamp_ns != decoded_joint_timestamps[15]:
            raise ValueError("body sample timestamp does not match PICO HEAD")
    elif joint_timestamp_contract == POSITIVE_JOINT_TIMESTAMP_CONTRACT:
        if body_timestamp_contract != LOCAL_POSE_BODY_TIMESTAMP_CONTRACT:
            raise ValueError(
                "positive PICO joint timestamps require local-pose body time"
            )
        if any(value <= 0 for value in decoded_joint_timestamps):
            raise ValueError(
                "positive PICO joint timestamp contract requires 24 positive values"
            )
        if any(
            abs(value - body_timestamp_ns) > SOURCE_SAMPLE_PERIOD_NS
            for value in decoded_joint_timestamps
        ):
            raise ValueError(
                "XR24 body joint timestamp is not bound to body frame"
            )
    else:
        raise ValueError("XR24 joint timestamp contract is unsupported")
    return root


def raw_frame_from_bodytracking_xrt_snapshot(
    snapshot: Mapping[str, Any],
    *,
    frame_index: int,
    capture_monotonic_ns: int,
) -> dict[str, Any]:
    """Normalize one exact BodyTracking snapshot into immutable raw evidence."""

    body = validate_bodytracking_xrt_snapshot(snapshot)
    result = {
        "schema_version": RAW_CAPTURE_SCHEMA_VERSION,
        "kind": RAW_CAPTURE_FRAME_KIND,
        "frame_index": _integer(frame_index, "frame_index"),
        "capture_monotonic_ns": _integer(
            capture_monotonic_ns,
            "capture_monotonic_ns",
            minimum=1,
        ),
        "body_sample_timestamp_ns": int(body["timestamp_ns"]),
        "body_timestamp_contract": body["body_timestamp_contract"],
        "joint_timestamp_contract": body["joint_timestamp_contract"],
        "body_sample_sequence": int(body["sample_sequence"]),
        "body_poses": [list(row) for row in body["poses"]],
        "body_velocities": [list(row) for row in body["velocities"]],
        "body_accelerations": [list(row) for row in body["accelerations"]],
        "body_joint_timestamps_ns": list(body["joint_timestamps_ns"]),
    }
    result.update(
        {
            key: body[key]
            for key in _BODY_SNAPSHOT_REQUIRED_FIELDS
            if key.startswith("health_")
        }
    )
    return result


def _validate_raw_frame(
    frame: Mapping[str, Any],
    *,
    schema_version: int,
) -> tuple[dict[str, Any], tuple[float, ...]]:
    value = _exact_mapping(frame, _FRAME_KEYS, "raw XR24 BodyTracking frame")
    if (
        value["schema_version"] != schema_version
        or value["kind"] != RAW_CAPTURE_FRAME_KIND
    ):
        raise ValueError("raw XR24 BodyTracking frame schema/kind mismatch")
    frame_index = _integer(value["frame_index"], "raw frame.frame_index")
    capture_ns = _integer(
        value["capture_monotonic_ns"],
        "raw frame.capture_monotonic_ns",
        minimum=1,
    )
    body_ns = _integer(
        value["body_sample_timestamp_ns"],
        "raw frame.body_sample_timestamp_ns",
        minimum=1,
    )
    sample_sequence = _integer(
        value["body_sample_sequence"],
        "raw frame.body_sample_sequence",
        minimum=1,
    )
    snapshot = {
        "contract": ATOMIC_SNAPSHOT_CONTRACT,
        "derivative_layout_contract": DERIVATIVE_LAYOUT_CONTRACT,
        "source_coherence_contract": SOURCE_COHERENCE_CONTRACT,
        "body_timestamp_contract": value["body_timestamp_contract"],
        "joint_timestamp_contract": value["joint_timestamp_contract"],
        "available": True,
        "timestamp_ns": body_ns,
        "sample_sequence": sample_sequence,
        "poses": value["body_poses"],
        "velocities": value["body_velocities"],
        "accelerations": value["body_accelerations"],
        "joint_timestamps_ns": value["body_joint_timestamps_ns"],
    }
    snapshot.update(
        {
            key: value[key]
            for key in _FRAME_KEYS
            if key.startswith("health_")
        }
    )
    validate_bodytracking_xrt_snapshot(snapshot)
    body_poses = snapshot["poses"]

    return (
        {
            "frame_index": frame_index,
            "capture_monotonic_ns": capture_ns,
            "body_sample_timestamp_ns": body_ns,
            "body_timestamp_contract": value["body_timestamp_contract"],
            "joint_timestamp_contract": value["joint_timestamp_contract"],
            "body_sample_sequence": sample_sequence,
            "health_timestamp_ns": int(value["health_timestamp_ns"]),
        },
        _body_bone_lengths(body_poses),
    )


def validate_raw_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable BodyTracking XR24/ankle-fused capture and digest."""

    root = _exact_mapping(capture, _CAPTURE_KEYS, "raw XR24 BodyTracking capture")
    schema_version = root["schema_version"]
    if schema_version not in {
        LEGACY_RAW_CAPTURE_SCHEMA_VERSION,
        RAW_CAPTURE_SCHEMA_VERSION,
    } or root["kind"] != RAW_CAPTURE_KIND:
        raise ValueError("raw XR24 BodyTracking capture schema/kind mismatch")
    if not isinstance(root["session_id"], str) or not root["session_id"]:
        raise ValueError("raw capture session_id is empty")
    if root["authorization"] != _AUTHORIZATION:
        raise ValueError("raw XR24 BodyTracking capture is not strictly read-only")
    source = _exact_mapping(
        root["source"],
        (
            _CAPTURE_SOURCE_KEYS
            if schema_version == RAW_CAPTURE_SCHEMA_VERSION
            else _CAPTURE_SOURCE_KEYS_V2
        ),
        "raw XR24 BodyTracking capture source",
    )
    if source["atomic_snapshot_contract"] != ATOMIC_SNAPSHOT_CONTRACT:
        raise ValueError("raw capture atomic snapshot contract mismatch")
    for key in (
        "xrobotoolkit_sdk_sha256",
        "pc_service_sha256",
        "pico_client_apk_sha256",
    ):
        if not _is_sha256(source[key]):
            raise ValueError(f"raw capture source.{key} is invalid")
    if source["pico_client_build"] != "xrobotoolkit-pico-health-v1":
        raise ValueError("raw capture PICO client build mismatch")
    if source["derivative_layout_contract"] != DERIVATIVE_LAYOUT_CONTRACT:
        raise ValueError("raw capture derivative layout contract mismatch")
    if schema_version == RAW_CAPTURE_SCHEMA_VERSION:
        if source["sdk_derivatives_control_usable"] is not False:
            raise ValueError(
                "raw capture SDK derivatives must be explicitly control-unusable"
            )
        if (
            source["control_derivative_contract"]
            != POSITION_DERIVED_CONTROL_DERIVATIVE_CONTRACT
        ):
            raise ValueError("raw capture control derivative contract mismatch")
    if source["source_coherence_contract"] != SOURCE_COHERENCE_CONTRACT:
        raise ValueError("raw capture BodyTracking source coherence contract mismatch")
    timestamp_contract_pair = (
        source["body_timestamp_contract"],
        source["joint_timestamp_contract"],
    )
    if timestamp_contract_pair not in {
        (
            PACKET_HEALTH_BODY_TIMESTAMP_CONTRACT,
            UNAVAILABLE_JOINT_TIMESTAMP_CONTRACT,
        ),
        (
            LOCAL_POSE_BODY_TIMESTAMP_CONTRACT,
            POSITIVE_JOINT_TIMESTAMP_CONTRACT,
        ),
        (
            HEAD_LOCAL_POSE_BODY_TIMESTAMP_CONTRACT,
            HEAD_ONLY_JOINT_TIMESTAMP_CONTRACT,
        ),
    }:
        raise ValueError("raw capture timestamp contracts are incompatible")
    if source["ankle_role_semantics"] != ANKLE_ROLE_SEMANTICS:
        raise ValueError("raw capture fused-ankle role semantics mismatch")
    ankle_indices = _exact_mapping(
        source["ankle_role_indices"],
        set(ANKLE_ROLE_TO_BODY_INDEX),
        "raw capture fused-ankle role indices",
    )
    for role, expected in ANKLE_ROLE_TO_BODY_INDEX.items():
        if _integer(
            ankle_indices[role],
            f"raw capture fused-ankle role indices.{role}",
        ) != expected:
            raise ValueError("raw capture fused-ankle role indices mismatch")
    if source["body_joint_order"] != list(XRT_BODY_JOINT_NAMES):
        raise ValueError("raw capture XR24 joint order mismatch")
    frames = root["frames"]
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("raw capture needs at least two complete frames")
    summaries: list[dict[str, Any]] = []
    previous_bone_lengths: tuple[float, ...] | None = None
    for index, frame in enumerate(frames):
        summary, bone_lengths = _validate_raw_frame(
            frame,
            schema_version=schema_version,
        )
        if (
            summary["body_timestamp_contract"]
            != source["body_timestamp_contract"]
            or summary["joint_timestamp_contract"]
            != source["joint_timestamp_contract"]
        ):
            raise ValueError("raw capture timestamp contract changed during capture")
        if index:
            previous = summaries[-1]
            if summary["frame_index"] != previous["frame_index"] + 1:
                raise ValueError("raw capture frame indices are not contiguous")
            for key in (
                "capture_monotonic_ns",
                "body_sample_timestamp_ns",
                "body_sample_sequence",
                "health_timestamp_ns",
            ):
                if summary[key] <= previous[key]:
                    raise ValueError(f"raw capture {key} did not advance")
            assert previous_bone_lengths is not None
            if any(
                abs(current - previous_length) > _MAX_BONE_LENGTH_CHANGE_M
                for current, previous_length in zip(
                    bone_lengths,
                    previous_bone_lengths,
                    strict=True,
                )
            ):
                raise ValueError("raw XR24 lower-body bone length changed discontinuously")
        summaries.append(summary)
        previous_bone_lengths = bone_lengths

    return {
        "session_id": root["session_id"],
        "frame_count": len(frames),
        "sha256": _canonical_sha256(root),
        "first_capture_monotonic_ns": summaries[0]["capture_monotonic_ns"],
        "last_capture_monotonic_ns": summaries[-1]["capture_monotonic_ns"],
        "sdk_derivatives_control_usable": False,
        "control_derivative_contract": (
            POSITION_DERIVED_CONTROL_DERIVATIVE_CONTRACT
        ),
        "legacy_v2_diagnostic_only": (
            schema_version == LEGACY_RAW_CAPTURE_SCHEMA_VERSION
        ),
        "promotion_eligible": False,
        "read_only": True,
    }


def reject_legacy_pose_payload_as_semantic_reference(payload: Mapping[str, Any]) -> None:
    """Reject current wrist-only/zero-velocity PICO pose payload explicitly."""

    if not isinstance(payload, Mapping):
        raise ValueError("legacy PICO pose payload is not a mapping")
    joint_pos = payload.get("joint_pos")
    joint_vel = payload.get("joint_vel")
    if not isinstance(joint_pos, Sequence) or not isinstance(joint_vel, Sequence):
        raise ValueError("legacy PICO pose payload has no complete joint_pos/joint_vel arrays")
    if len(joint_pos) < 47:
        raise ValueError(
            "legacy PICO pose payload lacks 47-frame delayed proof for normal profile"
        )
    if len(joint_vel) != len(joint_pos):
        raise ValueError(
            "legacy PICO pose payload joint_pos/joint_vel frame counts differ"
        )
    try:
        positions = [_finite_vector(row, SOURCE_DOF, "legacy joint_pos") for row in joint_pos]
        velocities = [_finite_vector(row, SOURCE_DOF, "legacy joint_vel") for row in joint_vel]
    except ValueError as exc:
        raise ValueError("legacy PICO pose payload is not complete canonical IL29") from exc
    lower_body_indices = (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18)
    if all(
        positions[frame_index][joint_index] == 0.0
        for frame_index in range(len(positions))
        for joint_index in lower_body_indices
    ) and all(
        velocities[frame_index][joint_index] == 0.0
        for frame_index in range(len(velocities))
        for joint_index in lower_body_indices
    ):
        raise ValueError("legacy PICO pose payload is wrist-only with zero lower body")
    raise ValueError(
        "legacy PICO pose payload lacks raw-capture hash and exact SOMA replay provenance"
    )


def _expected_il29_from_mj29(joint_pos_mj29: Sequence[float]) -> list[float]:
    result = [0.0] * SOURCE_DOF
    for source_index, target_index in enumerate(SOMA_MJ29_TO_CANONICAL_IL29):
        result[target_index] = float(joint_pos_mj29[source_index])
    return result


def _validate_producer(producer: Any) -> Mapping[str, Any]:
    value = _exact_mapping(producer, _PRODUCER_KEYS, "retarget trace producer")
    expected_scalars = {
        "kind": RETARGET_PRODUCER_KIND,
        "repository": PINNED_SOMA_REPOSITORY,
        "commit": PINNED_SOMA_COMMIT,
        "package_version": PINNED_SOMA_PACKAGE_VERSION,
        "python_requires": ">=3.12",
        "newton_version": PINNED_NEWTON_VERSION,
        "warp_version": PINNED_WARP_VERSION,
        "config_hash_semantics": PINNED_CONFIG_HASH_SEMANTICS,
        "xr24_soma_adapter_version": XR24_SOMA_ADAPTER_VERSION,
        "ankle_input_semantics": ANKLE_ROLE_SEMANTICS,
    }
    for key, expected in expected_scalars.items():
        if value[key] != expected:
            raise ValueError(f"retarget trace producer.{key} mismatch")
    if not _is_sha256(value["adapter_sha256"]):
        raise ValueError("retarget trace producer.adapter_sha256 is invalid")
    if value["config_sha256"] != PINNED_CONFIG_SHA256:
        raise ValueError("retarget trace producer config hashes mismatch")
    return value


def validate_retarget_trace_contract(
    capture: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    """Validate trace structure/binding without certifying unexecuted IK."""

    capture_summary = validate_raw_capture(capture)
    root = _exact_mapping(trace, _TRACE_KEYS, "G1 IL29 retarget trace")
    if (
        root["schema_version"] != RETARGET_TRACE_SCHEMA_VERSION
        or root["kind"] != RETARGET_TRACE_KIND
        or root["source_kind"] != SOURCE_RETARGETED_DELAYED
    ):
        raise ValueError("G1 IL29 retarget trace schema/kind/source mismatch")
    if root["source_session_id"] != capture_summary["session_id"]:
        raise ValueError("retarget trace source session does not match raw capture")
    if root["raw_capture_sha256"] != capture_summary["sha256"]:
        raise ValueError("retarget trace raw-capture hash mismatch")
    _validate_producer(root["producer"])
    if root["sample_period_ns"] != SOURCE_SAMPLE_PERIOD_NS:
        raise ValueError("retarget trace is not on exact 50 Hz reference grid")
    if root["joint_order_mj29"] != list(SOMA_MJ29_JOINT_NAMES):
        raise ValueError("retarget trace SOMA MJ29 order mismatch")
    if root["joint_order_il29"] != list(SOURCE_IL29_JOINT_NAMES):
        raise ValueError("retarget trace canonical IL29 order mismatch")
    if root["mj29_to_il29"] != list(SOMA_MJ29_TO_CANONICAL_IL29):
        raise ValueError("retarget trace MJ29 -> IL29 permutation mismatch")
    if root["authorization"] != _AUTHORIZATION:
        raise ValueError("retarget trace is not strictly read-only")
    certification = _exact_mapping(
        root["certification"],
        _CERTIFICATION_KEYS,
        "retarget trace certification",
    )
    blocked_certification = {
        "exact_backend_replay_performed": False,
        "raw_capture_replayed": False,
        "full_il29_verified": False,
        "promotion_eligible": False,
        "status": RETARGET_REPLAY_STATUS_BLOCKED,
    }
    exact_nonpromotable_certification = {
        "exact_backend_replay_performed": True,
        "raw_capture_replayed": True,
        "full_il29_verified": True,
        "promotion_eligible": False,
        "status": RETARGET_REPLAY_STATUS_EXACT_NONPROMOTABLE,
    }
    if dict(certification) not in (
        blocked_certification,
        exact_nonpromotable_certification,
    ):
        raise ValueError(
            "retarget trace must remain non-promotable unless exact replay "
            "and approved live XR24 evidence agree"
        )

    samples = root["samples"]
    needed_samples = required_buffer_frames(profile) + 1
    if not isinstance(samples, list) or len(samples) < needed_samples:
        raise ValueError(
            f"retarget trace needs at least {needed_samples} position samples "
            f"for {profile} delayed proof and forward velocity"
        )
    capture_frames = capture["frames"]
    summaries: list[dict[str, Any]] = []
    for index, raw_sample in enumerate(samples):
        sample = _exact_mapping(
            raw_sample,
            _TRACE_SAMPLE_KEYS,
            f"retarget trace.samples[{index}]",
        )
        source_frame_index = _integer(
            sample["source_frame_index"],
            f"retarget trace.samples[{index}].source_frame_index",
        )
        reference_ns = _integer(
            sample["reference_monotonic_ns"],
            f"retarget trace.samples[{index}].reference_monotonic_ns",
            minimum=1,
        )
        capture_ns = _integer(
            sample["capture_monotonic_ns"],
            f"retarget trace.samples[{index}].capture_monotonic_ns",
            minimum=1,
        )
        brackets = sample["raw_bracket_indices"]
        if (
            isinstance(brackets, (str, bytes))
            or not isinstance(brackets, Sequence)
            or len(brackets) != 2
        ):
            raise ValueError("retarget trace raw bracket must contain two indices")
        left_index = _integer(
            brackets[0],
            f"retarget trace.samples[{index}].raw_bracket_indices[0]",
        )
        right_index = _integer(
            brackets[1],
            f"retarget trace.samples[{index}].raw_bracket_indices[1]",
        )
        if (
            right_index != left_index + 1
            or right_index >= len(capture_frames)
        ):
            raise ValueError("retarget trace raw bracket is not contiguous/in range")
        left_ns = int(capture_frames[left_index]["capture_monotonic_ns"])
        right_ns = int(capture_frames[right_index]["capture_monotonic_ns"])
        if not left_ns <= reference_ns <= right_ns or right_ns <= left_ns:
            raise ValueError("retarget reference time is outside raw capture bracket")
        expected_alpha = (reference_ns - left_ns) / (right_ns - left_ns)
        alpha = _finite(
            sample["raw_interpolation_alpha"],
            f"retarget trace.samples[{index}].raw_interpolation_alpha",
        )
        if not math.isclose(alpha, expected_alpha, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("retarget trace interpolation alpha mismatches raw capture")
        if capture_ns < right_ns:
            raise ValueError("retarget trace claims capture before raw bracket existed")

        joint_pos_mj29 = _finite_vector(
            sample["joint_pos_mj29"],
            SOURCE_DOF,
            f"retarget trace.samples[{index}].joint_pos_mj29",
        )
        joint_pos_il29 = _finite_vector(
            sample["joint_pos_il29"],
            SOURCE_DOF,
            f"retarget trace.samples[{index}].joint_pos_il29",
        )
        if joint_pos_il29 != _expected_il29_from_mj29(joint_pos_mj29):
            raise ValueError("retarget trace IL29 position is not exact SOMA MJ29 reorder")
        joint_vel_raw = sample["joint_vel_il29"]
        if index == len(samples) - 1:
            if joint_vel_raw is not None:
                raise ValueError("final retarget trace velocity must be null proof terminator")
            joint_vel_il29 = None
        else:
            joint_vel_il29 = _finite_vector(
                joint_vel_raw,
                SOURCE_DOF,
                f"retarget trace.samples[{index}].joint_vel_il29",
            )
        summaries.append(
            {
                "source_frame_index": source_frame_index,
                "reference_monotonic_ns": reference_ns,
                "capture_monotonic_ns": capture_ns,
                "joint_pos_il29": joint_pos_il29,
                "joint_vel_il29": joint_vel_il29,
            }
        )

    for index in range(1, len(summaries)):
        previous = summaries[index - 1]
        current = summaries[index]
        if (
            current["source_frame_index"] != previous["source_frame_index"] + 1
            or current["reference_monotonic_ns"]
            != previous["reference_monotonic_ns"] + SOURCE_SAMPLE_PERIOD_NS
            or current["capture_monotonic_ns"] < previous["capture_monotonic_ns"]
        ):
            raise ValueError("retarget trace frame/time sequence is not contiguous 50 Hz")
    for index in range(len(summaries) - 1):
        current = summaries[index]
        following = summaries[index + 1]
        expected_velocity = [
            (following_value - current_value) * 50.0
            for current_value, following_value in zip(
                current["joint_pos_il29"],
                following["joint_pos_il29"],
                strict=True,
            )
        ]
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-6)
            for actual, expected in zip(
                current["joint_vel_il29"],
                expected_velocity,
                strict=True,
            )
        ):
            raise ValueError(
                "retarget trace velocity is not 50 Hz forward difference"
            )
    lower_body_indices = (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18)
    if all(
        math.isclose(
            summary["joint_pos_il29"][joint_index],
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        for summary in summaries
        for joint_index in lower_body_indices
    ) and all(
        math.isclose(
            summary["joint_vel_il29"][joint_index],
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        for summary in summaries[:-1]
        for joint_index in lower_body_indices
    ):
        raise ValueError(
            "retarget trace is wrist-only/all-zero with zero lower-body q/dq"
        )

    return {
        "source_kind": SOURCE_RETARGETED_DELAYED,
        "raw_capture_sha256": capture_summary["sha256"],
        "trace_sha256": _canonical_sha256(root),
        "raw_frame_count": capture_summary["frame_count"],
        "retarget_position_sample_count": len(samples),
        "derivable_semantic_frame_count": len(samples) - 1,
        "required_semantic_frame_count": required_buffer_frames(profile),
        "exact_backend_replay_performed": certification[
            "exact_backend_replay_performed"
        ],
        "promotion_eligible": False,
        "status": certification["status"],
    }


def _distribution_version(distribution_name: str) -> str | None:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def _git_head(source_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_is_clean(source_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return not result.stdout.strip()


def _module_origin(module_name: str) -> Path | None:
    try:
        spec = util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin in {None, "built-in", "frozen"}:
        return None
    return Path(spec.origin).resolve()


def probe_exact_retargeter(
    *,
    soma_source_root: Path | None = None,
    bodytracking_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report exact missing requirements; never claim current adapter approved."""

    blockers: list[str] = []
    python_ok = sys.version_info >= (3, 12)
    if not python_ok:
        blockers.append("Python >=3.12 required by pinned soma-retargeter")

    package_versions = {
        "soma-retargeter": _distribution_version("soma-retargeter"),
        "newton": _distribution_version("newton"),
        "warp-lang": _distribution_version("warp-lang"),
    }
    expected_versions = {
        "soma-retargeter": PINNED_SOMA_PACKAGE_VERSION,
        "newton": PINNED_NEWTON_VERSION,
        "warp-lang": PINNED_WARP_VERSION,
    }
    for name, expected in expected_versions.items():
        actual = package_versions[name]
        if actual != expected:
            blockers.append(f"{name}=={expected} required; found {actual or 'missing'}")
    soma_module_origin = _module_origin("soma_retargeter")
    if soma_module_origin is None:
        blockers.append("soma_retargeter import unavailable")

    source_commit = None
    source_clean: bool | None = None
    import_bound_to_source: bool | None = None
    config_hashes: dict[str, str | None] = {
        name: None for name in PINNED_CONFIG_SHA256
    }
    if soma_source_root is None:
        blockers.append("pinned soma-retargeter source checkout not supplied")
    else:
        root = soma_source_root.resolve()
        source_commit = _git_head(root)
        if source_commit != PINNED_SOMA_COMMIT:
            blockers.append(
                f"soma-retargeter commit {PINNED_SOMA_COMMIT} required; "
                f"found {source_commit or 'unreadable'}"
            )
        source_clean = _git_is_clean(root)
        if source_clean is not True:
            blockers.append("soma-retargeter source checkout is dirty or unreadable")
        if soma_module_origin is not None:
            import_bound_to_source = soma_module_origin.is_relative_to(
                root / "soma_retargeter"
            )
            if not import_bound_to_source:
                blockers.append(
                    "imported soma_retargeter is not bound to pinned source checkout"
                )
        config_root = root / "soma_retargeter" / "configs" / "unitree_g1"
        for name, expected in PINNED_CONFIG_SHA256.items():
            path = config_root / name
            actual = _normalized_utf8_sha256(path) if path.is_file() else None
            config_hashes[name] = actual
            if actual != expected:
                blockers.append(f"pinned SOMA config hash mismatch/missing: {name}")

    atomic_snapshot_valid = False
    if bodytracking_snapshot is not None:
        try:
            validate_bodytracking_xrt_snapshot(bodytracking_snapshot)
        except ValueError as exc:
            blockers.append(str(exc))
        else:
            atomic_snapshot_valid = True
    else:
        blockers.append("hardened XR24 BodyTracking snapshot not supplied")

    if not XR24_SOMA_ADAPTER_APPROVED:
        blockers.append(
            "XR24->SOMA coordinate/role adapter lacks approved exact replay fixture"
        )

    return {
        "ready": False,
        "promotion_eligible": False,
        "producer_kind": RETARGET_PRODUCER_KIND,
        "source_kind": SOURCE_RETARGETED_DELAYED,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "python_ok": python_ok,
        "package_versions": package_versions,
        "source_repository": PINNED_SOMA_REPOSITORY,
        "source_commit": source_commit,
        "source_clean": source_clean,
        "soma_module_origin": (
            str(soma_module_origin) if soma_module_origin is not None else None
        ),
        "import_bound_to_source": import_bound_to_source,
        "required_commit": PINNED_SOMA_COMMIT,
        "config_hash_semantics": PINNED_CONFIG_HASH_SEMANTICS,
        "config_hashes": config_hashes,
        "atomic_snapshot_valid": atomic_snapshot_valid,
        "adapter_approved": XR24_SOMA_ADAPTER_APPROVED,
        "blockers": blockers,
        "authorization": dict(_AUTHORIZATION),
    }


def materialize_semantic_frames(*_args: Any, **_kwargs: Any) -> None:
    """Fail closed until capture replay and XR24-to-SOMA adapter are approved."""

    raise ExactRetargeterUnavailable(
        "semantic retargeted frames blocked: hardened ankle-fused XR24 capture "
        "must replay through pinned soma-retargeter and approved XR24->SOMA adapter"
    )


__all__ = [
    "ANKLE_ROLE_SEMANTICS",
    "ANKLE_ROLE_TO_BODY_INDEX",
    "ATOMIC_SNAPSHOT_CONTRACT",
    "BODY_TRACKING_MODE_CODE",
    "DERIVATIVE_LAYOUT_CONTRACT",
    "HEAD_LOCAL_POSE_BODY_TIMESTAMP_CONTRACT",
    "HEAD_ONLY_JOINT_TIMESTAMP_CONTRACT",
    "LOCAL_POSE_BODY_TIMESTAMP_CONTRACT",
    "LEGACY_RAW_CAPTURE_SCHEMA_VERSION",
    "PACKET_HEALTH_BODY_TIMESTAMP_CONTRACT",
    "POSITIVE_JOINT_TIMESTAMP_CONTRACT",
    "POSITION_DERIVED_CONTROL_DERIVATIVE_CONTRACT",
    "ExactRetargeterUnavailable",
    "PINNED_CONFIG_HASH_SEMANTICS",
    "PINNED_CONFIG_SHA256",
    "PINNED_SOMA_COMMIT",
    "PINNED_SOMA_REPOSITORY",
    "RAW_CAPTURE_FRAME_KIND",
    "RAW_CAPTURE_KIND",
    "RAW_CAPTURE_SCHEMA_VERSION",
    "RETARGET_PRODUCER_KIND",
    "RETARGET_REPLAY_STATUS_BLOCKED",
    "RETARGET_REPLAY_STATUS_EXACT_NONPROMOTABLE",
    "RETARGET_TRACE_KIND",
    "RETARGET_TRACE_SCHEMA_VERSION",
    "SOURCE_COHERENCE_CONTRACT",
    "UNAVAILABLE_JOINT_TIMESTAMP_CONTRACT",
    "SOMA_MJ29_JOINT_NAMES",
    "SOMA_MJ29_TO_CANONICAL_IL29",
    "XR24_SOMA_ADAPTER_APPROVED",
    "XR24_SOMA_ADAPTER_VERSION",
    "XRT_BODY_JOINT_NAMES",
    "materialize_semantic_frames",
    "probe_exact_retargeter",
    "raw_frame_from_bodytracking_xrt_snapshot",
    "reject_legacy_pose_payload_as_semantic_reference",
    "validate_bodytracking_xrt_snapshot",
    "validate_raw_capture",
    "validate_retarget_trace_contract",
]
