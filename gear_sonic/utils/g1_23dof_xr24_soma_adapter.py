"""Pinned PICO XR24 -> NVIDIA SOMA -> canonical G1 IL29 replay adapter.

This module is deliberately transport-free.  It accepts an immutable hardened
XR24 BodyTracking capture, resamples the measured poses onto a bracketed 50 Hz
grid, runs the pinned NVIDIA SOMA retargeter, and materializes complete IL29
position/velocity semantic frames.  It never opens XR, ADB, ZMQ, DDS, or robot
channels.

PICO's Unity SDK calls the field ``BodyTrackingRoleData.localPose``, but the
24 positions share one body-tracking origin (they are not parent-local bone
offsets).  They are tracking-space poses in Unity's left-handed, Y-up
coordinate system.  The official SDK's human-body subsystem also publishes a
role-specific post-rotation table for those poses.  This adapter pins that
table and mirrors X into SOMA's right-handed, Y-up frame.
Only XR24 roles with an authoritative SOMA counterpart are written; all other
SOMA joints retain the pinned neutral skeleton transform and are not IK
effectors in the pinned G1 configuration.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from gear_sonic.utils.g1_23dof_contract import (
    SOURCE_DOF,
    SOURCE_IL29_EXCLUDED_INDICES,
    SOURCE_IL29_JOINT_NAMES,
)
from gear_sonic.utils.g1_23dof_pico_retargeted_producer import (
    ANKLE_ROLE_SEMANTICS,
    PINNED_CONFIG_HASH_SEMANTICS,
    PINNED_CONFIG_SHA256,
    PINNED_NEWTON_VERSION,
    PINNED_SOMA_COMMIT,
    PINNED_SOMA_PACKAGE_VERSION,
    PINNED_SOMA_REPOSITORY,
    PINNED_WARP_VERSION,
    RETARGET_PRODUCER_KIND,
    RETARGET_REPLAY_STATUS_EXACT_NONPROMOTABLE,
    RETARGET_TRACE_KIND,
    RETARGET_TRACE_SCHEMA_VERSION,
    SOMA_MJ29_JOINT_NAMES,
    SOMA_MJ29_TO_CANONICAL_IL29,
    XR24_SOMA_ADAPTER_VERSION,
    XRT_BODY_JOINT_NAMES,
    probe_exact_retargeter,
    validate_raw_capture,
    validate_retarget_trace_contract,
)
from gear_sonic.utils.g1_23dof_semantic_reference import (
    JOINT_ORDER,
    SEMANTIC_REFERENCE_FRAME_KIND,
    SEMANTIC_REFERENCE_SCHEMA_VERSION,
    SOURCE_RETARGETED_DELAYED,
    SOURCE_SAMPLE_PERIOD_NS,
    validate_semantic_reference_frame,
)

PICO_UNITY_OPENXR_REPOSITORY = "https://github.com/Pico-Developer/PICO-Unity-OpenXR-SDK.git"
PICO_UNITY_OPENXR_COMMIT = "3aa3e62bff41df618529eeb60ff02c29a515dafe"
PICO_UNITY_BODY_FEATURE_RELPATH = "Runtime/Features/PICO/BodyTrackingFeature.cs"
PICO_UNITY_HUMAN_BODY_RELPATH = "Runtime/Subsystem/PXR_HumanBodySubsystem.cs"
XR24_COORDINATE_CONTRACT = "xrobot_openxr_rh_xright_yup_zback_to_soma_yup_xleft_zforward_neutral_calibrated_v3"
XR24_NEUTRAL_STANDING_GATE_VERSION = 1
XR24_NEUTRAL_STANDING_HOLD_FRAMES = 10
XR24_NEUTRAL_CALIBRATION_VERSION = 1

# XRoboToolkit's PICO client reverses the Unity Integration SDK's Z/quaternion
# conversion before serializing body poses.  The host therefore receives the
# native OpenXR right-handed frame: +X right, +Y up, +Z backward.  SOMA's BVH
# source frame is +X left, +Y up, +Z forward.  This is a proper 180-degree
# world rotation about +Y, not Unity's left/right-handed X reflection.
_XROBOT_TO_SOMA_WORLD_QUATERNION = (0.0, 1.0, 0.0, 0.0)

# Official PICO Unity OpenXR SDK PXR_HumanBodySubsystem role post-rotations.
# Quaternion.Euler uses Unity's documented Z-X-Y application order.
_PICO_ROLE_POST_EULER_DEG: dict[int, tuple[float, float, float]] = {
    0: (0.0, 180.0, 0.0),
    1: (0.0, 0.0, -95.0),
    2: (0.0, 0.0, 95.0),
    3: (0.0, 0.0, 90.0),
    4: (0.0, 0.0, -90.0),
    5: (0.0, 0.0, 90.0),
    6: (0.0, 90.0, 90.0),
    7: (180.0, -90.0, 0.0),
    8: (0.0, 90.0, 0.0),
    9: (0.0, 90.0, 90.0),
    10: (0.0, 90.0, 0.0),
    11: (0.0, 90.0, 0.0),
    12: (0.0, 0.0, 90.0),
    13: (0.0, 0.0, 180.0),
    14: (0.0, 0.0, 180.0),
    15: (0.0, 90.0, 90.0),
    16: (0.0, 0.0, 180.0),
    17: (0.0, 0.0, 180.0),
    18: (0.0, 0.0, 180.0),
    19: (0.0, 0.0, 180.0),
    20: (0.0, 0.0, 180.0),
    21: (180.0, 0.0, 180.0),
}

# PICO roles describe joint location plus the outgoing bone orientation.  These
# are exact name/role counterparts used by SOMA's pinned scaler and IK map.
XR24_ROLE_TO_SOMA_JOINT = (
    (0, "Hips"),
    (3, "Spine1"),
    (6, "Spine2"),
    (9, "Chest"),
    (12, "Neck1"),
    (15, "Head"),
    (13, "LeftShoulder"),
    (16, "LeftArm"),
    (18, "LeftForeArm"),
    (20, "LeftHand"),
    (14, "RightShoulder"),
    (17, "RightArm"),
    (19, "RightForeArm"),
    (21, "RightHand"),
    (1, "LeftLeg"),
    (4, "LeftShin"),
    (7, "LeftFoot"),
    (10, "LeftToeBase"),
    (2, "RightLeg"),
    (5, "RightShin"),
    (8, "RightFoot"),
    (11, "RightToeBase"),
)

_READ_ONLY_AUTHORIZATION = {
    "read_only": True,
    "dds_opened": False,
    "robot_channel_opened": False,
    "actuation_authorized": False,
    "robot_commands_published": False,
}


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _normalize_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("quaternion must contain four values")
    q = tuple(_finite(value, "quaternion") for value in values)
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1.0e-8:
        raise ValueError("quaternion norm is zero")
    return tuple(value / norm for value in q)  # type: ignore[return-value]


def _quat_mul(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return _normalize_quaternion(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def _axis_quaternion(
    axis: tuple[float, float, float],
    degrees: float,
) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    sine = math.sin(half)
    return _normalize_quaternion((axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(half)))


def _unity_euler_zxy(
    euler_xyz_degrees: Sequence[float],
) -> tuple[float, float, float, float]:
    """Match ``UnityEngine.Quaternion.Euler(x, y, z)`` Z-X-Y order."""

    x, y, z = euler_xyz_degrees
    qx = _axis_quaternion((1.0, 0.0, 0.0), x)
    qy = _axis_quaternion((0.0, 1.0, 0.0), y)
    qz = _axis_quaternion((0.0, 0.0, 1.0), z)
    return _quat_mul(qy, _quat_mul(qx, qz))


def _quat_conjugate(
    values: Sequence[float],
) -> tuple[float, float, float, float]:
    x, y, z, w = _normalize_quaternion(values)
    return (-x, -y, -z, w)


def _quat_rotate(
    quaternion: Sequence[float],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = _normalize_quaternion(quaternion)
    vx, vy, vz = vector
    # Expanded q * [v, 0] * conjugate(q), avoiding normalization of [v, 0].
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _slerp(
    left: Sequence[float],
    right: Sequence[float],
    alpha: float,
) -> tuple[float, float, float, float]:
    q0 = _normalize_quaternion(left)
    q1 = _normalize_quaternion(right)
    dot = sum(a * b for a, b in zip(q0, q1, strict=True))
    if dot < 0.0:
        q1 = tuple(-value for value in q1)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(tuple(a + alpha * (b - a) for a, b in zip(q0, q1, strict=True)))
    theta = math.acos(dot)
    sine = math.sin(theta)
    left_weight = math.sin((1.0 - alpha) * theta) / sine
    right_weight = math.sin(alpha * theta) / sine
    return _normalize_quaternion(tuple(left_weight * a + right_weight * b for a, b in zip(q0, q1, strict=True)))


def pico_unity_pose_to_soma(
    role_index: int,
    pose_xyzw: Sequence[float],
) -> list[float]:
    """Apply base XRoboToolkit/OpenXR world conversion before neutral calibration.

    Function name remains for API compatibility.  Output orientation still
    needs capture-specific role calibration from the mandatory neutral hold.
    """

    if role_index < 0 or role_index >= len(XRT_BODY_JOINT_NAMES):
        raise ValueError("XR24 role index is out of range")
    if len(pose_xyzw) != 7:
        raise ValueError("XR24 pose must contain x,y,z,qx,qy,qz,qw")
    px, py, pz = (_finite(value, "XR24 position") for value in pose_xyzw[:3])
    raw_q = _normalize_quaternion(pose_xyzw[3:])
    soma_q = _quat_mul(_XROBOT_TO_SOMA_WORLD_QUATERNION, raw_q)
    return [-px, py, -pz, *soma_q]


def _quaternion_mean(
    values: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("quaternion mean requires at least one value")
    reference = _normalize_quaternion(values[0])
    aligned: list[tuple[float, float, float, float]] = []
    for value in values:
        quaternion = _normalize_quaternion(value)
        if sum(a * b for a, b in zip(reference, quaternion, strict=True)) < 0.0:
            quaternion = tuple(-component for component in quaternion)
        aligned.append(quaternion)
    return _normalize_quaternion(tuple(sum(row[index] for row in aligned) for index in range(4)))


def build_xr24_soma_neutral_calibration(
    neutral_body_pose_frames: Sequence[Sequence[Sequence[float]]],
    *,
    skeleton: Any,
    target_neutral_global: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Bind XR24 world/role frames to pinned SOMA from ten upright frames."""

    if len(neutral_body_pose_frames) != XR24_NEUTRAL_STANDING_HOLD_FRAMES:
        raise ValueError("XR24 SOMA calibration requires exactly 10 neutral frames")
    reports = [assess_xr24_neutral_standing(body_poses) for body_poses in neutral_body_pose_frames]
    if not all(report["pass"] for report in reports):
        raise ValueError("XR24 SOMA calibration frames must all be neutral standing")
    if len(target_neutral_global) != skeleton.num_joints:
        raise ValueError("SOMA neutral global transform count changed")

    soma_indices = {name: skeleton.joint_index(name) for _, name in XR24_ROLE_TO_SOMA_JOINT}
    if any(index < 0 for index in soma_indices.values()):
        raise RuntimeError("pinned SOMA neutral skeleton lacks required mapped joint")

    base_frames = [
        [pico_unity_pose_to_soma(role_index, body_poses[role_index]) for role_index, _ in XR24_ROLE_TO_SOMA_JOINT]
        for body_poses in neutral_body_pose_frames
    ]
    hips_target = target_neutral_global[soma_indices["Hips"]]
    hips_positions = [frame[0][:3] for frame in base_frames]
    hips_mean = [
        sum(position[axis] for position in hips_positions) / float(len(hips_positions)) for axis in range(3)
    ]
    translation = [float(hips_target[axis]) - hips_mean[axis] for axis in range(3)]

    role_offsets: list[list[float]] = []
    max_neutral_orientation_error_degrees = 0.0
    for mapped_index, (role_index, soma_name) in enumerate(XR24_ROLE_TO_SOMA_JOINT):
        base_mean = _quaternion_mean([frame[mapped_index][3:] for frame in base_frames])
        target = _normalize_quaternion(target_neutral_global[soma_indices[soma_name]][3:])
        offset = _quat_mul(_quat_conjugate(base_mean), target)
        role_offsets.append(list(offset))
        for frame in base_frames:
            calibrated = _quat_mul(frame[mapped_index][3:], offset)
            dot = abs(sum(a * b for a, b in zip(calibrated, target, strict=True)))
            error = math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))
            max_neutral_orientation_error_degrees = max(
                max_neutral_orientation_error_degrees,
                error,
            )
    if max_neutral_orientation_error_degrees > 15.0:
        raise ValueError("XR24 neutral orientation hold is not stable")

    body = {
        "version": XR24_NEUTRAL_CALIBRATION_VERSION,
        "coordinate_contract": XR24_COORDINATE_CONTRACT,
        "hold_frame_count": XR24_NEUTRAL_STANDING_HOLD_FRAMES,
        "translation": translation,
        "role_indices": [role for role, _ in XR24_ROLE_TO_SOMA_JOINT],
        "role_quaternion_offsets_xyzw": role_offsets,
        "max_neutral_orientation_error_degrees": (max_neutral_orientation_error_degrees),
    }
    return {**body, "calibration_sha256": _canonical_sha256(body)}


def apply_xr24_soma_neutral_calibration(
    role_index: int,
    pose_xyzw: Sequence[float],
    calibration: Mapping[str, Any],
) -> list[float]:
    if calibration.get("version") != XR24_NEUTRAL_CALIBRATION_VERSION:
        raise ValueError("XR24 SOMA neutral calibration version changed")
    roles = list(calibration["role_indices"])
    if role_index not in roles:
        raise ValueError("XR24 role is absent from SOMA neutral calibration")
    mapped_index = roles.index(role_index)
    base = pico_unity_pose_to_soma(role_index, pose_xyzw)
    translation = calibration["translation"]
    offset = calibration["role_quaternion_offsets_xyzw"][mapped_index]
    return [
        *[base[axis] + float(translation[axis]) for axis in range(3)],
        *_quat_mul(base[3:], offset),
    ]


def _distance3(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left[:3], right[:3], strict=True)))


def _joint_angle_degrees(
    proximal: Sequence[float],
    joint: Sequence[float],
    distal: Sequence[float],
) -> float:
    first = [float(proximal[i]) - float(joint[i]) for i in range(3)]
    second = [float(distal[i]) - float(joint[i]) for i in range(3)]
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm < 1.0e-4 or second_norm < 1.0e-4:
        raise ValueError("XR24 limb segment length is zero")
    cosine = sum(a * b for a, b in zip(first, second, strict=True)) / (first_norm * second_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def assess_xr24_neutral_standing(
    body_poses: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Fail-closed geometry gate for solver/policy acquisition posture.

    This is only a start/acquisition gate.  It does not limit later teleop
    motion.  Thresholds use tracking-space relative geometry, so tracking
    origin height cannot change the result.
    """

    if len(body_poses) != len(XRT_BODY_JOINT_NAMES):
        raise ValueError("neutral-standing gate requires exactly 24 XR roles")
    poses = [[_finite(value, f"XR24 role {index}") for value in pose] for index, pose in enumerate(body_poses)]
    if any(len(pose) != 7 for pose in poses):
        raise ValueError("neutral-standing gate requires 7-value XR poses")

    pelvis = poses[0]
    left_hip, right_hip = poses[1], poses[2]
    left_knee, right_knee = poses[4], poses[5]
    left_ankle, right_ankle = poses[7], poses[8]
    left_foot, right_foot = poses[10], poses[11]
    head = poses[15]
    foot_y = 0.5 * (left_foot[1] + right_foot[1])
    knee_y = 0.5 * (left_knee[1] + right_knee[1])
    pelvis_height = pelvis[1] - foot_y
    pelvis_above_knees = pelvis[1] - knee_y
    head_above_pelvis = head[1] - pelvis[1]
    left_knee_angle = _joint_angle_degrees(left_hip, left_knee, left_ankle)
    right_knee_angle = _joint_angle_degrees(right_hip, right_knee, right_ankle)
    left_leg_length = _distance3(left_hip, left_knee) + _distance3(left_knee, left_ankle)
    right_leg_length = _distance3(right_hip, right_knee) + _distance3(right_knee, right_ankle)

    checks = {
        "pelvis_height_m": pelvis_height >= 0.55,
        "pelvis_above_knees_m": pelvis_above_knees >= 0.12,
        "head_above_pelvis_m": head_above_pelvis >= 0.35,
        "left_knee_angle_deg": left_knee_angle >= 130.0,
        "right_knee_angle_deg": right_knee_angle >= 130.0,
        "left_leg_length_m": 0.55 <= left_leg_length <= 1.15,
        "right_leg_length_m": 0.55 <= right_leg_length <= 1.15,
    }
    return {
        "schema_version": XR24_NEUTRAL_STANDING_GATE_VERSION,
        "kind": "g1_true23_xr24_neutral_standing_gate",
        "pass": all(checks.values()),
        "checks": checks,
        "metrics": {
            "pelvis_height_m": pelvis_height,
            "pelvis_above_knees_m": pelvis_above_knees,
            "head_above_pelvis_m": head_above_pelvis,
            "left_knee_angle_deg": left_knee_angle,
            "right_knee_angle_deg": right_knee_angle,
            "left_leg_length_m": left_leg_length,
            "right_leg_length_m": right_leg_length,
        },
    }


def require_xr24_neutral_standing(
    body_poses: Sequence[Sequence[float]],
) -> dict[str, Any]:
    report = assess_xr24_neutral_standing(body_poses)
    if not report["pass"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise ValueError("XR24 acquisition pose is not neutral standing: " + ", ".join(failed))
    return report


def _interpolate_pose(
    left: Sequence[float],
    right: Sequence[float],
    alpha: float,
) -> list[float]:
    position = [float(a) + alpha * (float(b) - float(a)) for a, b in zip(left[:3], right[:3], strict=True)]
    return [*position, *_slerp(left[3:], right[3:], alpha)]


def resample_raw_capture_50hz(capture: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bracket and interpolate measured XR24 poses onto exact 50 Hz timestamps."""

    validate_raw_capture(capture)
    frames = capture["frames"]
    raw_times = [int(frame["capture_monotonic_ns"]) for frame in frames]
    first_ns = raw_times[0]
    last_ns = raw_times[-1]
    sample_count = (last_ns - first_ns) // SOURCE_SAMPLE_PERIOD_NS + 1
    if sample_count < 2:
        raise ValueError("raw XR24 capture is too short for two 50 Hz samples")

    result: list[dict[str, Any]] = []
    for source_frame_index in range(sample_count):
        reference_ns = first_ns + source_frame_index * SOURCE_SAMPLE_PERIOD_NS
        left_index = bisect_right(raw_times, reference_ns) - 1
        left_index = max(0, min(left_index, len(raw_times) - 2))
        right_index = left_index + 1
        left_ns = raw_times[left_index]
        right_ns = raw_times[right_index]
        alpha = (reference_ns - left_ns) / (right_ns - left_ns)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("50 Hz reference timestamp lies outside raw bracket")
        left_poses = frames[left_index]["body_poses"]
        right_poses = frames[right_index]["body_poses"]
        poses = [
            _interpolate_pose(left, right, alpha) for left, right in zip(left_poses, right_poses, strict=True)
        ]
        result.append(
            {
                "source_frame_index": source_frame_index,
                "reference_monotonic_ns": reference_ns,
                "capture_monotonic_ns": right_ns,
                "raw_bracket_indices": [left_index, right_index],
                "raw_interpolation_alpha": alpha,
                "body_poses": poses,
            }
        )
    return result


def _compose_transform(
    parent: Sequence[float],
    local: Sequence[float],
) -> list[float]:
    rotated = _quat_rotate(parent[3:], local[:3])
    position = [float(parent[i]) + rotated[i] for i in range(3)]
    return [*position, *_quat_mul(parent[3:], local[3:])]


def _relative_transform(
    parent: Sequence[float],
    child: Sequence[float],
) -> list[float]:
    inverse = _quat_conjugate(parent[3:])
    delta = [float(child[i]) - float(parent[i]) for i in range(3)]
    position = _quat_rotate(inverse, delta)
    return [*position, *_quat_mul(inverse, child[3:])]


def _global_transforms(
    parent_indices: Sequence[int],
    local_transforms: Sequence[Sequence[float]],
) -> list[list[float]]:
    result: list[list[float]] = []
    for index, local in enumerate(local_transforms):
        parent = int(parent_indices[index])
        result.append(list(local) if parent < 0 else _compose_transform(result[parent], local))
    return result


def _local_transforms(
    parent_indices: Sequence[int],
    global_transforms: Sequence[Sequence[float]],
) -> list[list[float]]:
    return [
        list(transform)
        if int(parent_indices[index]) < 0
        else _relative_transform(
            global_transforms[int(parent_indices[index])],
            transform,
        )
        for index, transform in enumerate(global_transforms)
    ]


def _validate_pinned_soma_runtime(soma_source_root: Path) -> dict[str, Any]:
    report = probe_exact_retargeter(soma_source_root=soma_source_root)
    required = {
        "python_ok": True,
        "source_commit": PINNED_SOMA_COMMIT,
        "source_clean": True,
        "import_bound_to_source": True,
        "package_versions": {
            "soma-retargeter": PINNED_SOMA_PACKAGE_VERSION,
            "newton": PINNED_NEWTON_VERSION,
            "warp-lang": PINNED_WARP_VERSION,
        },
        "config_hashes": PINNED_CONFIG_SHA256,
    }
    mismatches = [key for key, expected in required.items() if report[key] != expected]
    if mismatches:
        raise RuntimeError("pinned SOMA runtime mismatch: " + ", ".join(sorted(mismatches)))
    return report


def build_soma_animation_buffer(
    capture: Mapping[str, Any],
    *,
    soma_source_root: Path,
    neutral_body_pose_frames: Sequence[Sequence[Sequence[float]]] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Build exact SOMA ``AnimationBuffer`` from bracketed XR24 role poses."""

    _validate_pinned_soma_runtime(soma_source_root)
    samples = resample_raw_capture_50hz(capture)
    calibration_frames = (
        [sample["body_poses"] for sample in samples[:XR24_NEUTRAL_STANDING_HOLD_FRAMES]]
        if neutral_body_pose_frames is None
        else list(neutral_body_pose_frames)
    )
    hold_count = len(calibration_frames)
    if hold_count != XR24_NEUTRAL_STANDING_HOLD_FRAMES:
        raise ValueError("XR24 calibration requires exactly 10 neutral-standing frames")
    acquisition_reports = [assess_xr24_neutral_standing(body_poses) for body_poses in calibration_frames]
    if not all(report["pass"] for report in acquisition_reports):
        first_failure = next(report for report in acquisition_reports if not report["pass"])
        failed = [name for name, passed in first_failure["checks"].items() if not passed]
        raise ValueError("XR24 capture must begin with 10 neutral-standing frames: " + ", ".join(failed))

    import numpy as np
    from soma_retargeter.animation.animation_buffer import AnimationBuffer
    from soma_retargeter.assets.bvh import load_bvh

    neutral_path = soma_source_root / "soma_retargeter" / "configs" / "soma" / "soma_zero_frame0.bvh"
    skeleton, neutral_animation = load_bvh(str(neutral_path))
    reference_local = skeleton.reference_local_transforms.astype(np.float64)
    reference_global = _global_transforms(
        skeleton.parent_indices,
        reference_local.tolist(),
    )
    neutral_local = np.asarray(
        neutral_animation.local_transforms[0],
        dtype=np.float64,
    )
    target_neutral_global = _global_transforms(
        skeleton.parent_indices,
        neutral_local.tolist(),
    )
    calibration = build_xr24_soma_neutral_calibration(
        calibration_frames,
        skeleton=skeleton,
        target_neutral_global=target_neutral_global,
    )
    soma_indices = {name: skeleton.joint_index(name) for _, name in XR24_ROLE_TO_SOMA_JOINT}
    if any(index < 0 for index in soma_indices.values()):
        raise RuntimeError("pinned SOMA neutral skeleton lacks required mapped joint")

    local_frames: list[list[list[float]]] = []
    for sample in samples:
        globals_for_frame = [list(transform) for transform in reference_global]
        for role_index, soma_name in XR24_ROLE_TO_SOMA_JOINT:
            globals_for_frame[soma_indices[soma_name]] = apply_xr24_soma_neutral_calibration(
                role_index,
                sample["body_poses"][role_index],
                calibration,
            )
        local_frames.append(_local_transforms(skeleton.parent_indices, globals_for_frame))

    local_array = np.asarray(local_frames, dtype=np.float32)
    expected_shape = (len(samples), skeleton.num_joints, 7)
    if local_array.shape != expected_shape or not np.isfinite(local_array).all():
        raise RuntimeError("XR24 -> SOMA animation buffer shape/finite check failed")
    return (
        AnimationBuffer(
            skeleton,
            len(samples),
            50.0,
            local_transforms=local_array,
        ),
        samples,
    )


def _execute_soma(
    capture: Mapping[str, Any],
    *,
    soma_source_root: Path,
    neutral_body_pose_frames: Sequence[Sequence[Sequence[float]]] | None = None,
) -> tuple[list[list[float]], list[dict[str, Any]], Any]:
    """Execute pinned NVIDIA SOMA; return root7+MJ29, timing, pipeline."""

    import numpy as np
    from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
    import warp as wp

    animation, samples = build_soma_animation_buffer(
        capture,
        soma_source_root=soma_source_root,
        neutral_body_pose_frames=neutral_body_pose_frames,
    )
    pipeline = NewtonPipeline(animation.skeleton)
    pipeline.add_input_motions(
        [animation],
        [wp.transform_identity()],
        scale_animation=True,
    )
    outputs = pipeline.execute()
    if len(outputs) != 1 or outputs[0].num_frames != len(samples):
        raise RuntimeError("pinned SOMA returned unexpected motion/frame count")
    rows = np.stack(
        [np.asarray(row).reshape(-1) for row in outputs[0].data],
        axis=0,
    ).astype(np.float64, copy=False)
    if rows.shape != (len(samples), SOURCE_DOF + 7):
        raise RuntimeError("pinned SOMA output is not root7 + MJ29")
    if not np.isfinite(rows).all():
        raise RuntimeError("pinned SOMA output contains non-finite values")
    return rows.tolist(), samples, pipeline


def retarget_mj29(
    capture: Mapping[str, Any],
    *,
    soma_source_root: Path,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Execute pinned NVIDIA SOMA and return exact 50 Hz MJ29 radians."""

    rows, samples, _ = _execute_soma(
        capture,
        soma_source_root=soma_source_root,
    )
    return [row[7:] for row in rows], samples


def _adapter_sha256() -> str:
    payload = Path(__file__).read_bytes().decode("utf-8")
    normalized = payload.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mj29_to_il29(values: Sequence[float]) -> list[float]:
    if len(values) != SOURCE_DOF:
        raise ValueError("SOMA output must contain exactly 29 joint values")
    result = [0.0] * SOURCE_DOF
    for source_index, target_index in enumerate(SOMA_MJ29_TO_CANONICAL_IL29):
        result[target_index] = _finite(values[source_index], "MJ29 joint")
    return result


def _build_retarget_trace_from_rows(
    capture: Mapping[str, Any],
    *,
    mj29_rows: Sequence[Sequence[float]],
    resampled: Sequence[Mapping[str, Any]],
    profile: str,
) -> dict[str, Any]:
    """Build hash-bound trace from already executed pinned SOMA output."""

    capture_summary = validate_raw_capture(capture)
    if len(mj29_rows) != len(resampled):
        raise RuntimeError("SOMA output count differs from 50 Hz resample count")
    il29_rows = [_mj29_to_il29(row) for row in mj29_rows]
    samples: list[dict[str, Any]] = []
    for index, (mj29, il29, timing) in enumerate(zip(mj29_rows, il29_rows, resampled, strict=True)):
        velocity = None
        if index + 1 < len(il29_rows):
            velocity = [
                (following - current) * 50.0
                for current, following in zip(
                    il29,
                    il29_rows[index + 1],
                    strict=True,
                )
            ]
        samples.append(
            {
                "source_frame_index": timing["source_frame_index"],
                "reference_monotonic_ns": timing["reference_monotonic_ns"],
                "capture_monotonic_ns": timing["capture_monotonic_ns"],
                "raw_bracket_indices": timing["raw_bracket_indices"],
                "raw_interpolation_alpha": timing["raw_interpolation_alpha"],
                "joint_pos_mj29": mj29,
                "joint_pos_il29": il29,
                "joint_vel_il29": velocity,
            }
        )

    trace = {
        "schema_version": RETARGET_TRACE_SCHEMA_VERSION,
        "kind": RETARGET_TRACE_KIND,
        "source_kind": SOURCE_RETARGETED_DELAYED,
        "source_session_id": capture["session_id"],
        "raw_capture_sha256": capture_summary["sha256"],
        "producer": {
            "kind": RETARGET_PRODUCER_KIND,
            "adapter_sha256": _adapter_sha256(),
            "repository": PINNED_SOMA_REPOSITORY,
            "commit": PINNED_SOMA_COMMIT,
            "package_version": PINNED_SOMA_PACKAGE_VERSION,
            "python_requires": ">=3.12",
            "newton_version": PINNED_NEWTON_VERSION,
            "warp_version": PINNED_WARP_VERSION,
            "config_hash_semantics": PINNED_CONFIG_HASH_SEMANTICS,
            "config_sha256": dict(PINNED_CONFIG_SHA256),
            "xr24_soma_adapter_version": XR24_SOMA_ADAPTER_VERSION,
            "ankle_input_semantics": ANKLE_ROLE_SEMANTICS,
        },
        "sample_period_ns": SOURCE_SAMPLE_PERIOD_NS,
        "joint_order_mj29": list(SOMA_MJ29_JOINT_NAMES),
        "joint_order_il29": list(SOURCE_IL29_JOINT_NAMES),
        "mj29_to_il29": list(SOMA_MJ29_TO_CANONICAL_IL29),
        "samples": samples,
        "certification": {
            "exact_backend_replay_performed": True,
            "raw_capture_replayed": True,
            "full_il29_verified": True,
            "promotion_eligible": False,
            "status": RETARGET_REPLAY_STATUS_EXACT_NONPROMOTABLE,
        },
        "authorization": dict(_READ_ONLY_AUTHORIZATION),
    }
    validate_retarget_trace_contract(capture, trace, profile=profile)
    return trace


def build_retarget_trace(
    capture: Mapping[str, Any],
    *,
    soma_source_root: Path,
    profile: str,
) -> dict[str, Any]:
    """Run exact backend and build hash-bound MJ29/IL29 trace evidence."""

    rows, resampled, _ = _execute_soma(
        capture,
        soma_source_root=soma_source_root,
    )
    return _build_retarget_trace_from_rows(
        capture,
        mj29_rows=[row[7:] for row in rows],
        resampled=resampled,
        profile=profile,
    )


def semantic_frames_from_trace(
    trace: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Materialize complete delayed IL29 q/dq frames; omit velocity terminator."""

    producer_sha256 = _canonical_sha256(trace["producer"])
    frames = []
    for sample in trace["samples"]:
        if sample["joint_vel_il29"] is None:
            continue
        frame = {
            "schema_version": SEMANTIC_REFERENCE_SCHEMA_VERSION,
            "kind": SEMANTIC_REFERENCE_FRAME_KIND,
            "source_kind": SOURCE_RETARGETED_DELAYED,
            "source_session_id": trace["source_session_id"],
            "producer_sha256": producer_sha256,
            "source_frame_index": sample["source_frame_index"],
            "reference_monotonic_ns": sample["reference_monotonic_ns"],
            "capture_monotonic_ns": sample["capture_monotonic_ns"],
            "sample_period_ns": SOURCE_SAMPLE_PERIOD_NS,
            "joint_order": JOINT_ORDER,
            "complete_joint_mask_il29": [True] * SOURCE_DOF,
            "joint_values_semantics": "full_reference_no_fill",
            "velocity_semantics": "source_50hz_forward_difference",
            "temporal_semantics": "measured_delayed_reference",
            "joint_pos_il29": list(sample["joint_pos_il29"]),
            "joint_vel_il29": list(sample["joint_vel_il29"]),
        }
        validate_semantic_reference_frame(frame)
        frames.append(frame)
    return frames


def _g1_reference_body_terms(
    soma_rows: Sequence[Sequence[float]],
    timing: Sequence[Mapping[str, Any]],
    pipeline: Any,
) -> list[dict[str, Any]]:
    """FK locked True23 embodiment and derive exact 3-point reference terms."""

    import newton
    import numpy as np
    import warp as wp

    q_rows = np.asarray(soma_rows, dtype=np.float32).copy()
    if q_rows.shape != (len(timing), SOURCE_DOF + 7):
        raise RuntimeError("SOMA rows do not have root7 + MJ29 shape")
    for source_index, canonical_index in enumerate(SOMA_MJ29_TO_CANONICAL_IL29):
        if canonical_index in SOURCE_IL29_EXCLUDED_INDICES:
            q_rows[:, 7 + source_index] = 0.0

    model = pipeline._build_model(len(q_rows))  # noqa: SLF001 - pinned API
    state = model.state()
    joint_q = wp.array(q_rows.reshape(-1), dtype=wp.float32)
    joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32)
    newton.eval_fk(model, joint_q, joint_qd, state)
    body_q = np.asarray(state.body_q.numpy(), dtype=np.float64).reshape(
        len(q_rows),
        pipeline.num_body_count,
        7,
    )
    leaf_names = [str(label).rsplit("/", 1)[-1] for label in pipeline.robot_builder.body_label]
    required = (
        "pelvis",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
        "torso_link",
    )
    if any(name not in leaf_names for name in required):
        raise RuntimeError("pinned G1 model lacks required reference body")
    pelvis_index, left_index, right_index, torso_index = (leaf_names.index(name) for name in required)
    body_indices = (left_index, right_index, torso_index)
    offsets = (
        (0.18, -0.025, 0.0),
        (0.18, 0.025, 0.0),
        (0.0, 0.0, 0.35),
    )

    result: list[dict[str, Any]] = []
    for frame_index, sample_timing in enumerate(timing):
        pelvis = body_q[frame_index, pelvis_index]
        pelvis_inverse = _quat_conjugate(pelvis[3:])
        local_positions: list[float] = []
        local_orientations: list[float] = []
        for body_index, offset in zip(body_indices, offsets, strict=True):
            body = body_q[frame_index, body_index]
            point_offset = _quat_rotate(body[3:], offset)
            point = [body[axis] + point_offset[axis] for axis in range(3)]
            delta = [point[axis] - pelvis[axis] for axis in range(3)]
            local_positions.extend(_quat_rotate(pelvis_inverse, delta))
            local_orientations.extend(_quat_mul(pelvis_inverse, body[3:]))
        result.append(
            {
                "source_frame_index": sample_timing["source_frame_index"],
                "reference_monotonic_ns": sample_timing["reference_monotonic_ns"],
                "capture_monotonic_ns": sample_timing["capture_monotonic_ns"],
                "vr_3point_local_target": local_positions,
                "vr_3point_local_orn_target": local_orientations,
                "reference_anchor_quaternion_xyzw": pelvis[3:].tolist(),
            }
        )
    return result


def _rotation_6d_xyzw(quaternion: Sequence[float]) -> list[float]:
    x, y, z, w = _normalize_quaternion(quaternion)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    return [r00, r01, r10, r11, r20, r21]


def complete_encoder_terms(
    *,
    reference_window: Mapping[str, Any],
    body_term: Mapping[str, Any],
    robot_anchor_quaternion_wxyz: Sequence[float],
) -> dict[str, Any]:
    """Combine delayed reference with live robot orientation into exact 267 terms."""

    from gear_sonic.utils.g1_23dof_live_shadow import (
        ENCODER_TERMS_KIND,
        ENCODER_TERMS_SCHEMA_VERSION,
    )
    from gear_sonic.utils.g1_23dof_semantic_reference import (
        true23_command_from_window,
    )

    if len(robot_anchor_quaternion_wxyz) != 4:
        raise ValueError("robot anchor quaternion must be w,x,y,z")
    robot_xyzw = _normalize_quaternion(
        (
            robot_anchor_quaternion_wxyz[1],
            robot_anchor_quaternion_wxyz[2],
            robot_anchor_quaternion_wxyz[3],
            robot_anchor_quaternion_wxyz[0],
        )
    )
    reference_xyzw = _normalize_quaternion(body_term["reference_anchor_quaternion_xyzw"])
    relative = _quat_mul(_quat_conjugate(robot_xyzw), reference_xyzw)
    return {
        "schema_version": ENCODER_TERMS_SCHEMA_VERSION,
        "kind": ENCODER_TERMS_KIND,
        "pico_source_frame_index": body_term["source_frame_index"],
        "pico_source_monotonic_ns": body_term["reference_monotonic_ns"],
        "future_frame_offsets_s": list(reference_window["future_frame_offsets_s"]),
        "command_multi_future_lower_body": true23_command_from_window(reference_window),
        "vr_3point_local_target": list(body_term["vr_3point_local_target"]),
        "vr_3point_local_orn_target": list(body_term["vr_3point_local_orn_target"]),
        "motion_anchor_ori_b": _rotation_6d_xyzw(relative),
    }


def replay_capture(
    capture: Mapping[str, Any],
    *,
    soma_source_root: Path,
    profile: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Run replay; return trace, semantic frames, body terms, and audit."""

    soma_rows, timing, pipeline = _execute_soma(
        capture,
        soma_source_root=soma_source_root,
    )
    trace = _build_retarget_trace_from_rows(
        capture,
        mj29_rows=[row[7:] for row in soma_rows],
        resampled=timing,
        profile=profile,
    )
    frames = semantic_frames_from_trace(trace)
    body_terms = _g1_reference_body_terms(soma_rows, timing, pipeline)
    lower_body_indices = (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18)
    lower_q_nonzero = any(
        abs(frame["joint_pos_il29"][index]) > 1.0e-8 for frame in frames for index in lower_body_indices
    )
    lower_dq_nonzero = any(
        abs(frame["joint_vel_il29"][index]) > 1.0e-8 for frame in frames for index in lower_body_indices
    )
    if not lower_q_nonzero or not lower_dq_nonzero:
        raise RuntimeError("exact SOMA replay lacks nonzero lower-body q/dq")
    report = {
        "schema_version": 1,
        "kind": "g1_true23_xr24_soma_replay_report",
        "ready_for_semantic_replay": True,
        "promotion_eligible": False,
        "live_deployment_approved": False,
        "coordinate_contract": XR24_COORDINATE_CONTRACT,
        "pico_sdk_repository": PICO_UNITY_OPENXR_REPOSITORY,
        "pico_sdk_commit": PICO_UNITY_OPENXR_COMMIT,
        "soma_repository": PINNED_SOMA_REPOSITORY,
        "soma_commit": PINNED_SOMA_COMMIT,
        "raw_capture_sha256": trace["raw_capture_sha256"],
        "trace_sha256": _canonical_sha256(trace),
        "semantic_frames_sha256": _canonical_sha256(frames),
        "position_sample_count": len(trace["samples"]),
        "semantic_frame_count": len(frames),
        "sample_period_ns": SOURCE_SAMPLE_PERIOD_NS,
        "timestamp_brackets_verified": True,
        "full_mj29_verified": True,
        "full_il29_verified": True,
        "lower_body_q_nonzero": lower_q_nonzero,
        "lower_body_dq_nonzero": lower_dq_nonzero,
        "g1_fk_body_terms_verified": len(body_terms) == len(trace["samples"]),
        "authorization": dict(_READ_ONLY_AUTHORIZATION),
        "remaining_live_gate": (
            "capture calibrated advancing hardened XR24 BodyTracking from actual "
            "PICO Ultra plus two ankle bands, replay it through this pinned stack, "
            "then review neutral-pose axes and motion direction before approval"
        ),
    }
    return trace, frames, body_terms, report


__all__ = [
    "PICO_UNITY_OPENXR_COMMIT",
    "PICO_UNITY_OPENXR_REPOSITORY",
    "XR24_COORDINATE_CONTRACT",
    "XR24_NEUTRAL_CALIBRATION_VERSION",
    "XR24_NEUTRAL_STANDING_GATE_VERSION",
    "XR24_ROLE_TO_SOMA_JOINT",
    "apply_xr24_soma_neutral_calibration",
    "assess_xr24_neutral_standing",
    "build_xr24_soma_neutral_calibration",
    "build_retarget_trace",
    "build_soma_animation_buffer",
    "complete_encoder_terms",
    "pico_unity_pose_to_soma",
    "replay_capture",
    "require_xr24_neutral_standing",
    "resample_raw_capture_50hz",
    "retarget_mj29",
    "semantic_frames_from_trace",
]
