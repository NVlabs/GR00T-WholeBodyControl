"""Read-only semantic future-reference bridge for the true-23 teleop encoder.

The 240-value ``command_multi_future_lower_body`` term is not a pose history.
It is ten complete G1 lower-body reference frames followed by their ten
reference-velocity frames.  This module builds that term only from either:

* a complete, precomputed 50 Hz G1 reference motion; or
* an explicitly versioned stream of complete 50 Hz G1 reference frames.

Raw PICO poses are deliberately not an accepted source.  The current PICO pose
wire contains a past window, zero-filled lower-body joints, and zero velocity,
so relabelling it as a future command would violate the training semantics.

No function in this module opens ZMQ, DDS, ADB, or a robot channel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from gear_sonic.utils.g1_23dof_contract import (
    REFERENCE_FUTURE_FRAME_COUNT,
    REFERENCE_PROFILE_LOW_LATENCY,
    REFERENCE_PROFILE_NORMAL,
    ALL_REFERENCE_PROFILES,
    REFERENCE_PROFILES,
    REFERENCE_SOURCE_SAMPLE_PERIOD_S,
    REFERENCE_SOURCE_SAMPLE_RATE_HZ,
    SOURCE_DOF,
)

SEMANTIC_REFERENCE_SCHEMA_VERSION = 1
SEMANTIC_REFERENCE_WINDOW_KIND = "g1_true23_semantic_future_reference_window"
SEMANTIC_REFERENCE_FRAME_KIND = "g1_semantic_reference_frame"

JOINT_ORDER = "canonical_isaaclab_il29"
SOURCE_SAMPLE_RATE_HZ = REFERENCE_SOURCE_SAMPLE_RATE_HZ
SOURCE_SAMPLE_PERIOD_NS = round(REFERENCE_SOURCE_SAMPLE_PERIOD_S * 1_000_000_000)
FUTURE_FRAME_COUNT = REFERENCE_FUTURE_FRAME_COUNT
MAX_EMISSION_LAG_NS = 100_000_000
VELOCITY_MATCH_TOLERANCE = 1.0e-4

# Mirrors lower_body_joint_isaaclab_order_in_isaaclab_index in the native
# deployment's policy_parameters.hpp and MotionTrackingCommand in commands.py.
LOWER_BODY_IL29_INDICES = (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18)
LOWER_BODY_DOF = len(LOWER_BODY_IL29_INDICES)
COMMAND_DIM = FUTURE_FRAME_COUNT * LOWER_BODY_DOF * 2

PROFILE_TRUE23_STEP5 = REFERENCE_PROFILE_NORMAL
PROFILE_RELEASED_LOW_LATENCY_STEP1 = REFERENCE_PROFILE_LOW_LATENCY
_PROFILE_STEPS = {
    name: profile.future_frame_step
    for name, profile in ALL_REFERENCE_PROFILES.items()
}

SOURCE_RECORDED_MOTION = "recorded_g1_motion_csv"
SOURCE_PLANNER = "planner_generated_g1_reference"
SOURCE_RETARGETED_DELAYED = "retargeted_g1_delayed_reference"
_ALLOWED_SOURCE_KINDS = {
    SOURCE_RECORDED_MOTION,
    SOURCE_PLANNER,
    SOURCE_RETARGETED_DELAYED,
}
_ALLOWED_TEMPORAL_SEMANTICS = {
    SOURCE_RECORDED_MOTION: "precomputed_future",
    SOURCE_PLANNER: "precomputed_future",
    SOURCE_RETARGETED_DELAYED: "measured_delayed_reference",
}

_WINDOW_KEYS = {
    "schema_version",
    "kind",
    "profile",
    "source",
    "playback",
    "future_frame_step",
    "future_frame_indices",
    "future_frame_offsets_s",
    "future_reference_monotonic_ns",
    "lower_body_il29_indices",
    "joint_pos_lower_body",
    "joint_vel_lower_body",
    "command_multi_future_lower_body",
    "authorization",
}
_SOURCE_KEYS = {
    "kind",
    "source_id",
    "source_relpath",
    "identity_sha256",
    "joint_order",
    "sample_rate_hz",
    "sample_period_ns",
    "complete_joint_count",
    "joint_values_semantics",
    "velocity_semantics",
    "temporal_semantics",
    "joint_pos_sha256",
    "joint_vel_sha256",
}
_PLAYBACK_KEYS = {
    "epoch_monotonic_ns",
    "frame_index",
    "frame_monotonic_ns",
    "emitted_monotonic_ns",
    "emission_lag_ns",
    "terminal_clamped",
}
_AUTHORIZATION_KEYS = {
    "read_only",
    "dds_opened",
    "robot_channel_opened",
    "actuation_authorized",
    "robot_commands_published",
}
_FRAME_KEYS = {
    "schema_version",
    "kind",
    "source_kind",
    "source_session_id",
    "producer_sha256",
    "source_frame_index",
    "reference_monotonic_ns",
    "capture_monotonic_ns",
    "sample_period_ns",
    "joint_order",
    "complete_joint_mask_il29",
    "joint_values_semantics",
    "velocity_semantics",
    "temporal_semantics",
    "joint_pos_il29",
    "joint_vel_il29",
}


def _exact_mapping(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{context} must contain exact keys {sorted(keys)}")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _vector(value: Any, size: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{context} must contain exactly {size} values")
    return [_number(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _profile_step(profile: str) -> int:
    try:
        return _PROFILE_STEPS[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported semantic reference profile: {profile}") from exc


def future_offsets_s(profile: str) -> tuple[float, ...]:
    """Return exact reference offsets for one approved temporal profile."""

    _profile_step(profile)
    return ALL_REFERENCE_PROFILES[profile].future_frame_offsets_s


def future_frame_step(profile: str) -> int:
    """Return source-frame stride for one approved temporal profile."""

    return _profile_step(profile)


def required_buffer_frames(profile: str) -> int:
    """Frames needed for full horizon plus final velocity proof frame."""

    frame_count = ALL_REFERENCE_PROFILES[profile].future_frame_count
    return (frame_count - 1) * _profile_step(profile) + 2


def _read_csv_matrix(path: Path, prefix: str) -> list[list[float]]:
    expected_header = [f"{prefix}{index}" for index in range(SOURCE_DOF)]
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if header != expected_header:
                raise ValueError(f"{path.name} header is not exact canonical IL29 order")
            rows: list[list[float]] = []
            for row_index, row in enumerate(reader):
                if len(row) != SOURCE_DOF:
                    raise ValueError(
                        f"{path.name} row {row_index} must contain {SOURCE_DOF} values"
                    )
                values = [_number(float(item), f"{path.name}[{row_index}]") for item in row]
                rows.append(values)
    except (OSError, UnicodeError, StopIteration) as exc:
        raise ValueError(f"cannot load semantic reference CSV {path}: {exc}") from exc
    if len(rows) < 2:
        raise ValueError(f"{path.name} must contain at least two frames")
    return rows


def _load_recorded_motion(
    motion_dir: Path,
) -> tuple[list[list[float]], list[list[float]], str, str]:
    motion_dir = motion_dir.resolve()
    joint_pos_path = motion_dir / "joint_pos.csv"
    joint_vel_path = motion_dir / "joint_vel.csv"
    positions = _read_csv_matrix(joint_pos_path, "joint_")
    velocities = _read_csv_matrix(joint_vel_path, "joint_vel_")
    if len(positions) != len(velocities):
        raise ValueError("joint_pos.csv and joint_vel.csv frame counts differ")
    return positions, velocities, _sha256(joint_pos_path), _sha256(joint_vel_path)


def _selected_values(
    rows: Sequence[Sequence[float]], frame_indices: Sequence[int]
) -> list[float]:
    return [
        float(rows[frame_index][joint_index])
        for frame_index in frame_indices
        for joint_index in LOWER_BODY_IL29_INDICES
    ]


def _verify_velocity_semantics(
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
    frame_indices: Sequence[int],
) -> None:
    for frame_index in frame_indices:
        if frame_index + 1 >= len(positions):
            raise ValueError(
                "semantic reference needs one source frame beyond horizon "
                "to verify final velocity; terminal clamping is forbidden"
            )
        for joint_index in LOWER_BODY_IL29_INDICES:
            expected = (
                positions[frame_index + 1][joint_index]
                - positions[frame_index][joint_index]
            ) * SOURCE_SAMPLE_RATE_HZ
            actual = velocities[frame_index][joint_index]
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=VELOCITY_MATCH_TOLERANCE,
            ):
                raise ValueError(
                    "source joint velocity does not match 50 Hz forward difference: "
                    f"frame={frame_index}, canonical_il29={joint_index}"
                )


def _authorization() -> dict[str, bool | int]:
    return {
        "read_only": True,
        "dds_opened": False,
        "robot_channel_opened": False,
        "actuation_authorized": False,
        "robot_commands_published": 0,
    }


def build_recorded_reference_window(
    motion_dir: Path,
    *,
    profile: str,
    playback_frame_index: int,
    playback_epoch_monotonic_ns: int,
    emitted_monotonic_ns: int,
    source_relpath: str | None = None,
) -> dict[str, Any]:
    """Build exact lower-body future terms from a tracked reference motion.

    ``playback_epoch_monotonic_ns`` defines when source frame zero is scheduled
    as the policy reference.  Future frames are already materialized in the
    recorded motion, so they are genuine precomputed references rather than
    predictions or relabelled past measurements.
    """

    frame_index = _integer(playback_frame_index, "playback_frame_index")
    epoch_ns = _integer(
        playback_epoch_monotonic_ns,
        "playback_epoch_monotonic_ns",
        minimum=1,
    )
    emitted_ns = _integer(emitted_monotonic_ns, "emitted_monotonic_ns", minimum=1)
    step = _profile_step(profile)
    offsets = future_offsets_s(profile)
    frame_indices = [frame_index + index * step for index in range(FUTURE_FRAME_COUNT)]

    positions, velocities, pos_hash, vel_hash = _load_recorded_motion(motion_dir)
    if frame_indices[-1] + 1 >= len(positions):
        raise ValueError(
            "recorded reference lacks complete future horizon plus velocity proof frame"
        )
    _verify_velocity_semantics(positions, velocities, frame_indices)

    frame_ns = epoch_ns + frame_index * SOURCE_SAMPLE_PERIOD_NS
    if emitted_ns < frame_ns:
        raise ValueError("emitted_monotonic_ns precedes playback reference frame")
    emission_lag_ns = emitted_ns - frame_ns
    if emission_lag_ns > MAX_EMISSION_LAG_NS:
        raise ValueError("recorded reference playback binding is stale")

    joint_pos = _selected_values(positions, frame_indices)
    joint_vel = _selected_values(velocities, frame_indices)
    command = [*joint_pos, *joint_vel]
    source_identity = {
        "kind": SOURCE_RECORDED_MOTION,
        "name": motion_dir.resolve().name,
        "joint_pos_sha256": pos_hash,
        "joint_vel_sha256": vel_hash,
    }
    return {
        "schema_version": SEMANTIC_REFERENCE_SCHEMA_VERSION,
        "kind": SEMANTIC_REFERENCE_WINDOW_KIND,
        "profile": profile,
        "source": {
            "kind": SOURCE_RECORDED_MOTION,
            "source_id": motion_dir.resolve().name,
            "source_relpath": source_relpath,
            "identity_sha256": _canonical_sha256(source_identity),
            "joint_order": JOINT_ORDER,
            "sample_rate_hz": SOURCE_SAMPLE_RATE_HZ,
            "sample_period_ns": SOURCE_SAMPLE_PERIOD_NS,
            "complete_joint_count": SOURCE_DOF,
            "joint_values_semantics": "full_reference_no_fill",
            "velocity_semantics": "source_50hz_forward_difference_verified",
            "temporal_semantics": "precomputed_future",
            "joint_pos_sha256": pos_hash,
            "joint_vel_sha256": vel_hash,
        },
        "playback": {
            "epoch_monotonic_ns": epoch_ns,
            "frame_index": frame_index,
            "frame_monotonic_ns": frame_ns,
            "emitted_monotonic_ns": emitted_ns,
            "emission_lag_ns": emission_lag_ns,
            "terminal_clamped": False,
        },
        "future_frame_step": step,
        "future_frame_indices": frame_indices,
        "future_frame_offsets_s": list(offsets),
        "future_reference_monotonic_ns": [
            frame_ns + index * step * SOURCE_SAMPLE_PERIOD_NS
            for index in range(FUTURE_FRAME_COUNT)
        ],
        "lower_body_il29_indices": list(LOWER_BODY_IL29_INDICES),
        "joint_pos_lower_body": joint_pos,
        "joint_vel_lower_body": joint_vel,
        "command_multi_future_lower_body": command,
        "authorization": _authorization(),
    }


def validate_semantic_reference_window(
    window: Mapping[str, Any],
    *,
    motion_dir: Path | None = None,
) -> Mapping[str, Any]:
    """Validate exact semantics and optionally replay against source CSVs."""

    root = _exact_mapping(window, _WINDOW_KEYS, "semantic reference window")
    if (
        root["schema_version"] != SEMANTIC_REFERENCE_SCHEMA_VERSION
        or root["kind"] != SEMANTIC_REFERENCE_WINDOW_KIND
    ):
        raise ValueError("semantic reference window schema/kind mismatch")
    profile = str(root["profile"])
    step = _profile_step(profile)

    source = _exact_mapping(root["source"], _SOURCE_KEYS, "semantic reference source")
    if source["kind"] not in _ALLOWED_SOURCE_KINDS:
        raise ValueError("semantic reference source kind is not approved")
    if not isinstance(source["source_id"], str) or not source["source_id"]:
        raise ValueError("semantic reference source_id is empty")
    if source["source_relpath"] is not None and (
        not isinstance(source["source_relpath"], str)
        or not source["source_relpath"]
    ):
        raise ValueError("semantic reference source_relpath is invalid")
    if not _is_sha256(source["identity_sha256"]):
        raise ValueError("semantic reference identity_sha256 is invalid")
    if (
        source["joint_order"] != JOINT_ORDER
        or source["sample_rate_hz"] != SOURCE_SAMPLE_RATE_HZ
        or source["sample_period_ns"] != SOURCE_SAMPLE_PERIOD_NS
        or source["complete_joint_count"] != SOURCE_DOF
        or source["joint_values_semantics"] != "full_reference_no_fill"
        or source["velocity_semantics"]
        != "source_50hz_forward_difference_verified"
        or source["temporal_semantics"]
        != _ALLOWED_TEMPORAL_SEMANTICS[source["kind"]]
    ):
        raise ValueError("semantic reference source contract mismatch")
    for key in ("joint_pos_sha256", "joint_vel_sha256"):
        value = source[key]
        if value is not None and not _is_sha256(value):
            raise ValueError(f"semantic reference source.{key} is invalid")

    playback = _exact_mapping(root["playback"], _PLAYBACK_KEYS, "semantic reference playback")
    epoch_ns = _integer(
        playback["epoch_monotonic_ns"],
        "semantic reference playback.epoch_monotonic_ns",
        minimum=1,
    )
    frame_index = _integer(
        playback["frame_index"],
        "semantic reference playback.frame_index",
    )
    frame_ns = _integer(
        playback["frame_monotonic_ns"],
        "semantic reference playback.frame_monotonic_ns",
        minimum=1,
    )
    emitted_ns = _integer(
        playback["emitted_monotonic_ns"],
        "semantic reference playback.emitted_monotonic_ns",
        minimum=1,
    )
    lag_ns = _integer(
        playback["emission_lag_ns"],
        "semantic reference playback.emission_lag_ns",
    )
    horizon_ns = (FUTURE_FRAME_COUNT - 1) * step * SOURCE_SAMPLE_PERIOD_NS
    minimum_lag_ns = (
        horizon_ns if source["temporal_semantics"] == "measured_delayed_reference" else 0
    )
    maximum_lag_ns = minimum_lag_ns + MAX_EMISSION_LAG_NS
    if (
        frame_ns != epoch_ns + frame_index * SOURCE_SAMPLE_PERIOD_NS
        or emitted_ns < frame_ns
        or lag_ns != emitted_ns - frame_ns
        or not minimum_lag_ns <= lag_ns <= maximum_lag_ns
        or playback["terminal_clamped"] is not False
    ):
        raise ValueError("semantic reference playback timeline mismatch")

    if root["future_frame_step"] != step:
        raise ValueError("semantic reference future_frame_step mismatch")
    expected_indices = [
        frame_index + index * step for index in range(FUTURE_FRAME_COUNT)
    ]
    if root["future_frame_indices"] != expected_indices:
        raise ValueError("semantic reference future_frame_indices mismatch")
    expected_offsets = list(future_offsets_s(profile))
    offsets = _vector(
        root["future_frame_offsets_s"],
        FUTURE_FRAME_COUNT,
        "semantic reference future_frame_offsets_s",
    )
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-9)
        for actual, expected in zip(offsets, expected_offsets, strict=True)
    ):
        raise ValueError("semantic reference future_frame_offsets_s mismatch")
    expected_reference_ns = [
        frame_ns + index * step * SOURCE_SAMPLE_PERIOD_NS
        for index in range(FUTURE_FRAME_COUNT)
    ]
    if root["future_reference_monotonic_ns"] != expected_reference_ns:
        raise ValueError("semantic reference future timestamps mismatch")
    if root["lower_body_il29_indices"] != list(LOWER_BODY_IL29_INDICES):
        raise ValueError("semantic reference lower-body selector mismatch")

    joint_pos = _vector(
        root["joint_pos_lower_body"],
        FUTURE_FRAME_COUNT * LOWER_BODY_DOF,
        "semantic reference joint_pos_lower_body",
    )
    joint_vel = _vector(
        root["joint_vel_lower_body"],
        FUTURE_FRAME_COUNT * LOWER_BODY_DOF,
        "semantic reference joint_vel_lower_body",
    )
    command = _vector(
        root["command_multi_future_lower_body"],
        COMMAND_DIM,
        "semantic reference command_multi_future_lower_body",
    )
    if command != [*joint_pos, *joint_vel]:
        raise ValueError(
            "semantic reference command layout is not positions120 then velocities120"
        )
    authorization = _exact_mapping(
        root["authorization"],
        _AUTHORIZATION_KEYS,
        "semantic reference authorization",
    )
    if authorization != _authorization():
        raise ValueError("semantic reference is not strictly read-only")

    if motion_dir is not None:
        if source["kind"] != SOURCE_RECORDED_MOTION:
            raise ValueError("motion_dir replay is only valid for recorded motion source")
        positions, velocities, pos_hash, vel_hash = _load_recorded_motion(motion_dir)
        if (
            source["source_id"] != motion_dir.resolve().name
            or source["joint_pos_sha256"] != pos_hash
            or source["joint_vel_sha256"] != vel_hash
        ):
            raise ValueError("semantic reference source identity/hash mismatch")
        expected_identity = _canonical_sha256(
            {
                "kind": SOURCE_RECORDED_MOTION,
                "name": motion_dir.resolve().name,
                "joint_pos_sha256": pos_hash,
                "joint_vel_sha256": vel_hash,
            }
        )
        if source["identity_sha256"] != expected_identity:
            raise ValueError("semantic reference source identity digest mismatch")
        if expected_indices[-1] + 1 >= len(positions):
            raise ValueError("semantic reference source no longer covers full horizon")
        _verify_velocity_semantics(positions, velocities, expected_indices)
        if joint_pos != _selected_values(positions, expected_indices):
            raise ValueError("semantic reference positions disagree with source CSV")
        if joint_vel != _selected_values(velocities, expected_indices):
            raise ValueError("semantic reference velocities disagree with source CSV")

    return {
        "profile": profile,
        "source_kind": source["kind"],
        "playback_frame_index": frame_index,
        "horizon_s": expected_offsets[-1],
        "command_dim": len(command),
        "terminal_clamped": False,
        "read_only": True,
    }


def true23_command_from_window(
    window: Mapping[str, Any],
    *,
    motion_dir: Path | None = None,
) -> list[float]:
    """Return 240 true23 command values; reject low-latency time semantics."""

    validate_semantic_reference_window(window, motion_dir=motion_dir)
    if window["profile"] != PROFILE_TRUE23_STEP5:
        raise ValueError(
            "true23 checkpoint requires 0.0..0.9 s step5 reference semantics; "
            "same-shape low-latency step1 terms are not interchangeable"
        )
    return list(window["command_multi_future_lower_body"])


def validate_semantic_reference_frame(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate minimal upstream live planner/retargeter frame protocol."""

    value = _exact_mapping(frame, _FRAME_KEYS, "semantic reference frame")
    if (
        value["schema_version"] != SEMANTIC_REFERENCE_SCHEMA_VERSION
        or value["kind"] != SEMANTIC_REFERENCE_FRAME_KIND
    ):
        raise ValueError("semantic reference frame schema/kind mismatch")
    source_kind = value["source_kind"]
    if source_kind not in {SOURCE_PLANNER, SOURCE_RETARGETED_DELAYED}:
        raise ValueError("raw/tracker pose is not an approved semantic joint reference")
    if not isinstance(value["source_session_id"], str) or not value["source_session_id"]:
        raise ValueError("semantic reference source_session_id is empty")
    if not _is_sha256(value["producer_sha256"]):
        raise ValueError("semantic reference producer_sha256 is invalid")
    frame_index = _integer(
        value["source_frame_index"],
        "semantic reference frame.source_frame_index",
    )
    reference_ns = _integer(
        value["reference_monotonic_ns"],
        "semantic reference frame.reference_monotonic_ns",
        minimum=1,
    )
    capture_ns = _integer(
        value["capture_monotonic_ns"],
        "semantic reference frame.capture_monotonic_ns",
        minimum=1,
    )
    if (
        value["sample_period_ns"] != SOURCE_SAMPLE_PERIOD_NS
        or value["joint_order"] != JOINT_ORDER
        or value["complete_joint_mask_il29"] != [True] * SOURCE_DOF
        or value["joint_values_semantics"] != "full_reference_no_fill"
        or value["velocity_semantics"] != "source_50hz_forward_difference"
        or value["temporal_semantics"] != _ALLOWED_TEMPORAL_SEMANTICS[source_kind]
    ):
        raise ValueError("semantic reference frame source contract mismatch")
    if source_kind == SOURCE_RETARGETED_DELAYED and capture_ns < reference_ns:
        raise ValueError("measured delayed reference cannot be captured before reference time")
    _vector(value["joint_pos_il29"], SOURCE_DOF, "semantic reference frame.joint_pos_il29")
    _vector(value["joint_vel_il29"], SOURCE_DOF, "semantic reference frame.joint_vel_il29")
    return {
        "source_kind": source_kind,
        "source_frame_index": frame_index,
        "reference_monotonic_ns": reference_ns,
        "capture_monotonic_ns": capture_ns,
    }


def build_stream_reference_window(
    frames: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    emitted_monotonic_ns: int,
) -> dict[str, Any]:
    """Build a window from a pinned planner or delayed full-joint retargeter.

    For a delayed measured source, the oldest retained frame becomes playback
    reference time zero and all selected frames must already have been captured.
    For a planner source, future reference times may be later than emission, but
    every predicted frame must already be present in this input batch.
    """

    emitted_ns = _integer(emitted_monotonic_ns, "emitted_monotonic_ns", minimum=1)
    needed = required_buffer_frames(profile)
    if not isinstance(frames, Sequence) or len(frames) < needed:
        raise ValueError(
            f"semantic stream needs at least {needed} complete frames for {profile}"
        )
    selected_buffer = list(frames[-needed:])
    validated = [validate_semantic_reference_frame(frame) for frame in selected_buffer]
    first = selected_buffer[0]
    source_kind = first["source_kind"]
    session_id = first["source_session_id"]
    producer_sha = first["producer_sha256"]
    for index, (frame, summary) in enumerate(zip(selected_buffer, validated, strict=True)):
        if (
            frame["source_kind"] != source_kind
            or frame["source_session_id"] != session_id
            or frame["producer_sha256"] != producer_sha
        ):
            raise ValueError("semantic stream source identity changed inside window")
        if index:
            previous = validated[index - 1]
            if (
                summary["source_frame_index"] != previous["source_frame_index"] + 1
                or summary["reference_monotonic_ns"]
                != previous["reference_monotonic_ns"] + SOURCE_SAMPLE_PERIOD_NS
            ):
                raise ValueError("semantic stream frame/time sequence is not contiguous 50 Hz")
    for index in range(len(selected_buffer) - 1):
        current = selected_buffer[index]
        following = selected_buffer[index + 1]
        for joint_index in LOWER_BODY_IL29_INDICES:
            expected = (
                following["joint_pos_il29"][joint_index]
                - current["joint_pos_il29"][joint_index]
            ) * SOURCE_SAMPLE_RATE_HZ
            if not math.isclose(
                current["joint_vel_il29"][joint_index],
                expected,
                rel_tol=0.0,
                abs_tol=VELOCITY_MATCH_TOLERANCE,
            ):
                raise ValueError(
                    "semantic stream velocity does not match contiguous positions"
                )

    step = _profile_step(profile)
    chosen = [selected_buffer[index * step] for index in range(FUTURE_FRAME_COUNT)]
    if source_kind == SOURCE_RETARGETED_DELAYED:
        newest_reference_ns = validated[-1]["reference_monotonic_ns"]
        if newest_reference_ns > emitted_ns:
            raise ValueError("delayed measured reference contains uncaptured future time")
        if emitted_ns - newest_reference_ns > MAX_EMISSION_LAG_NS:
            raise ValueError("delayed measured reference newest frame is stale")
        if any(frame["capture_monotonic_ns"] > emitted_ns for frame in chosen):
            raise ValueError("delayed measured reference was not captured before emission")
    else:
        if any(frame["capture_monotonic_ns"] > emitted_ns for frame in chosen):
            raise ValueError("planner future frame was not materialized before emission")

    frame_index = chosen[0]["source_frame_index"]
    frame_ns = chosen[0]["reference_monotonic_ns"]
    epoch_ns = frame_ns - frame_index * SOURCE_SAMPLE_PERIOD_NS
    if epoch_ns < 1:
        raise ValueError("semantic stream epoch would be non-positive")
    lag_ns = emitted_ns - frame_ns
    if source_kind == SOURCE_PLANNER and abs(lag_ns) > MAX_EMISSION_LAG_NS:
        raise ValueError("planner playback base is not aligned to current emission")
    if lag_ns < 0:
        raise ValueError("semantic stream playback base is future-dated")

    joint_pos = [
        float(frame["joint_pos_il29"][joint_index])
        for frame in chosen
        for joint_index in LOWER_BODY_IL29_INDICES
    ]
    joint_vel = [
        float(frame["joint_vel_il29"][joint_index])
        for frame in chosen
        for joint_index in LOWER_BODY_IL29_INDICES
    ]
    return {
        "schema_version": SEMANTIC_REFERENCE_SCHEMA_VERSION,
        "kind": SEMANTIC_REFERENCE_WINDOW_KIND,
        "profile": profile,
        "source": {
            "kind": source_kind,
            "source_id": session_id,
            "source_relpath": None,
            "identity_sha256": producer_sha,
            "joint_order": JOINT_ORDER,
            "sample_rate_hz": SOURCE_SAMPLE_RATE_HZ,
            "sample_period_ns": SOURCE_SAMPLE_PERIOD_NS,
            "complete_joint_count": SOURCE_DOF,
            "joint_values_semantics": "full_reference_no_fill",
            "velocity_semantics": "source_50hz_forward_difference_verified",
            "temporal_semantics": _ALLOWED_TEMPORAL_SEMANTICS[source_kind],
            "joint_pos_sha256": None,
            "joint_vel_sha256": None,
        },
        "playback": {
            "epoch_monotonic_ns": epoch_ns,
            "frame_index": frame_index,
            "frame_monotonic_ns": frame_ns,
            "emitted_monotonic_ns": emitted_ns,
            "emission_lag_ns": lag_ns,
            "terminal_clamped": False,
        },
        "future_frame_step": step,
        "future_frame_indices": [frame["source_frame_index"] for frame in chosen],
        "future_frame_offsets_s": list(future_offsets_s(profile)),
        "future_reference_monotonic_ns": [
            frame["reference_monotonic_ns"] for frame in chosen
        ],
        "lower_body_il29_indices": list(LOWER_BODY_IL29_INDICES),
        "joint_pos_lower_body": joint_pos,
        "joint_vel_lower_body": joint_vel,
        "command_multi_future_lower_body": [*joint_pos, *joint_vel],
        "authorization": _authorization(),
    }
