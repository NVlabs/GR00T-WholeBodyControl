"""Strict evidence contract for native-23 read-only live inference.

This module deliberately does not open ZMQ, DDS, or a Unitree channel.  It is
the shared verifier for evidence emitted by the isolated live-shadow process
and consumed by readiness tooling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from gear_sonic.utils.g1_23dof_contract import (
    ARTIFACT_SCHEMA_VERSION,
    HARDWARE_JOINT_IDS,
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_TO_CANONICAL_IL29,
    REFERENCE_FUTURE_FRAME_COUNT,
    SOURCE_IL29_EXCLUDED_INDICES,
    TELEOP_ENCODER_INPUT_DIM,
    TELEOP_ENCODER_INPUT_TERM_DIMS,
    TELEOP_ENCODER_INPUT_TERM_ORDER,
    reference_profile_contract,
)
from gear_sonic.utils.g1_23dof_semantic_reference import (
    SEMANTIC_REFERENCE_SCHEMA_VERSION,
    SEMANTIC_REFERENCE_WINDOW_KIND,
    SOURCE_RECORDED_MOTION,
    SOURCE_SAMPLE_PERIOD_NS,
    validate_semantic_reference_window,
)

LIVE_EVIDENCE_SCHEMA_VERSION = 2
LIVE_EVIDENCE_KIND = "g1_true23_integrated_live_shadow_evidence"
LIVE_PRODUCER_KIND = "g1_true23_integrated_readonly_shadow_probe"
LIVE_PRODUCER_VERSION = 1
LIVE_PRODUCER_FILENAME = "g1_true23_live_shadow"
LIVE_PRODUCER_SOURCE = (
    "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/"
    "g1_true23_live_shadow.cpp"
)
LIVE_PRODUCER_CORE = (
    "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/"
    "true23_live_shadow_core.hpp"
)
LIVE_PRODUCER_AUDIT = (
    "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/"
    "check_true23_shadow_binary.cmake"
)
DEFAULT_APPROVAL_MANIFEST = (
    "gear_sonic/config/deployment/g1_true23_live_shadow_approval.json"
)
APPROVAL_SCHEMA_VERSION = 1
APPROVAL_KIND = "g1_true23_live_shadow_producer_approval"
ENCODER_TERMS_SCHEMA_VERSION = 2
ENCODER_TERMS_KIND = "g1_true23_two_source_encoder_terms"
MIN_LIVE_SAMPLES = 25
MAX_EVIDENCE_AGE_S = 30.0
MAX_COMMON_WINDOW_NS = 15_000_000_000
MAX_SOURCE_PAIR_SKEW_NS = 100_000_000
MIN_EFFECTIVE_HZ = 40.0
MAX_EFFECTIVE_HZ = 60.0
MAX_SAMPLE_GAP_NS = 100_000_000
ONNX_REPLAY_ATOL = 1.0e-6
ONNX_REPLAY_RTOL = 1.0e-6

PROPRIO_FRAME_DIM = 93
PROPRIO_HISTORY_LENGTH = 10
PROPRIO_HISTORY_DIM = PROPRIO_FRAME_DIM * PROPRIO_HISTORY_LENGTH
TOKEN_DIM = 64
DECODER_INPUT_DIM = TOKEN_DIM + PROPRIO_HISTORY_DIM
ACTION_DIM = 23
ANGULAR_VELOCITY_HISTORY_OFFSET = 0
ANGULAR_VELOCITY_HISTORY_DIM = 30
JOINT_POSITION_HISTORY_OFFSET = 30
JOINT_POSITION_HISTORY_DIM = 290
JOINT_VELOCITY_HISTORY_OFFSET = 320
JOINT_VELOCITY_HISTORY_DIM = 290
PREVIOUS_ACTION_HISTORY_OFFSET = 610
PREVIOUS_ACTION_HISTORY_DIM = 290
GRAVITY_HISTORY_OFFSET = 900
GRAVITY_HISTORY_DIM = 30

# Hardware-compact order follows HARDWARE_JOINT_IDS.
HARDWARE_DEFAULT_Q = (
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
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
)
HARDWARE_ACTION_SCALE = (
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
HARDWARE_LOWER_LIMIT = (
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
    -3.0892,
    -1.5882,
    -2.618,
    -1.0472,
    -1.972222054,
    -3.0892,
    -2.2515,
    -2.618,
    -1.0472,
    -1.972222054,
)
HARDWARE_UPPER_LIMIT = (
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
    2.6704,
    2.2515,
    2.618,
    2.0944,
    1.972222054,
    2.6704,
    1.5882,
    2.618,
    2.0944,
    1.972222054,
)
HARDWARE_VELOCITY_LIMIT = (
    32.0,
    20.0,
    32.0,
    20.0,
    30.0,
    30.0,
    32.0,
    20.0,
    32.0,
    20.0,
    30.0,
    30.0,
    32.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
    37.0,
)

_ARTIFACT_HASH_KEYS = {
    "checkpoint",
    "simulation_report",
    "encoder_onnx",
    "decoder_onnx",
    "metadata",
}
_ROOT_KEYS = {
    "schema_version",
    "kind",
    "captured_at_utc",
    "producer",
    "artifact_hashes",
    "source_contract",
    "window",
    "pico_samples",
    "lowstate_samples",
    "inference_samples",
    "summary",
    "authorization",
}
_ENCODER_TERM_KEYS = {
    "schema_version",
    "kind",
    "pico_source_frame_index",
    "pico_source_monotonic_ns",
    "future_frame_offsets_s",
    *TELEOP_ENCODER_INPUT_TERM_ORDER,
}
_INFERENCE_KEYS = {
    "monotonic_ns",
    "pico_body_source_ns",
    "lowstate_tick",
    "semantic_reference_window",
    "encoder_terms",
    "observation",
    "token",
    "proprio_history",
    "decoder_input",
    "previous_action_native",
    "native_action",
    "hardware_action",
    "output_bounds",
}


def _exact_mapping(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{context} must contain exact keys {sorted(keys)}")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
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


def _close(left: float, right: float, *, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _vectors_close(
    left: Sequence[float],
    right: Sequence[float],
    *,
    tolerance: float = 1e-6,
) -> bool:
    return len(left) == len(right) and all(
        _close(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_strict_json(path: Path, context: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{context} contains non-finite number: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {context}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} root must be an object")
    return value


def _validate_producer_approval(
    *,
    manifest_path: Path,
    repo_root: Path,
    producer_path: Path,
) -> Mapping[str, str]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ValueError(f"live producer approval manifest missing: {manifest_path}")
    manifest = _exact_mapping(
        _load_strict_json(manifest_path, "live producer approval manifest"),
        {
            "schema_version",
            "kind",
            "promotion_enabled",
            "evidence_schema_version",
            "producer_kind",
            "producer_filename",
            "binary_format",
            "source",
            "core",
            "binary_sha256",
            "audit",
            "review_state",
        },
        "live producer approval manifest",
    )
    if (
        manifest["schema_version"] != APPROVAL_SCHEMA_VERSION
        or manifest["kind"] != APPROVAL_KIND
        or manifest["evidence_schema_version"] != LIVE_EVIDENCE_SCHEMA_VERSION
        or manifest["producer_kind"] != LIVE_PRODUCER_KIND
        or manifest["producer_filename"] != LIVE_PRODUCER_FILENAME
        or manifest["binary_format"] != "ELF"
    ):
        raise ValueError("live producer approval manifest contract mismatch")
    root = repo_root.resolve()
    expected_files = {
        "source": (LIVE_PRODUCER_SOURCE, root / LIVE_PRODUCER_SOURCE),
        "core": (LIVE_PRODUCER_CORE, root / LIVE_PRODUCER_CORE),
        "audit": (LIVE_PRODUCER_AUDIT, root / LIVE_PRODUCER_AUDIT),
    }
    resolved_hashes: dict[str, str] = {}
    for role, (expected_relpath, path) in expected_files.items():
        record = _exact_mapping(
            manifest[role],
            {"relpath", "sha256"},
            f"live producer approval manifest.{role}",
        )
        resolved = path.resolve()
        if (
            record["relpath"] != expected_relpath
            or not resolved.is_file()
            or record["sha256"] != _sha256(resolved)
        ):
            raise ValueError(f"live producer approval {role} hash/path mismatch")
        resolved_hashes[f"{role}_sha256"] = record["sha256"]
    producer_sha256 = _sha256(producer_path)
    if manifest["binary_sha256"] != producer_sha256:
        raise ValueError("live producer approval binary hash mismatch")
    if manifest["promotion_enabled"] is not True:
        raise ValueError(
            "live producer promotion is disabled; semantic PICO future-term "
            "producer has not passed review"
        )
    if manifest["review_state"] != "approved":
        raise ValueError("live producer approval review_state is not approved")
    return {
        **resolved_hashes,
        "binary_sha256": producer_sha256,
        "manifest_sha256": _sha256(manifest_path),
    }


def _binary_format(path: Path) -> str | None:
    magic = path.read_bytes()[:4]
    if magic == b"\x7fELF":
        return "ELF"
    if magic[:2] == b"MZ":
        return "PE"
    return None


def _load_onnx_replay_sessions(
    encoder_path: Path,
    decoder_path: Path,
) -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
    except (ImportError, OSError) as exc:
        raise ValueError(
            "numpy and onnxruntime CPU execution are required for live evidence replay"
        ) from exc

    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        encoder = ort.InferenceSession(
            str(encoder_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        decoder = ort.InferenceSession(
            str(decoder_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise ValueError(f"cannot load paired ONNX for CPU evidence replay: {exc}") from exc

    def require_signature(
        session: Any,
        *,
        role: str,
        input_name: str,
        input_shape: list[int],
        output_name: str,
        output_shape: list[int],
    ) -> None:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if (
            session.get_providers() != ["CPUExecutionProvider"]
            or len(inputs) != 1
            or len(outputs) != 1
            or inputs[0].name != input_name
            or inputs[0].shape != input_shape
            or inputs[0].type != "tensor(float)"
            or outputs[0].name != output_name
            or outputs[0].shape != output_shape
            or outputs[0].type != "tensor(float)"
        ):
            raise ValueError(f"{role} ONNX replay signature/provider mismatch")

    require_signature(
        encoder,
        role="encoder",
        input_name="teleop_obs",
        input_shape=[1, TELEOP_ENCODER_INPUT_DIM],
        output_name="token",
        output_shape=[1, TOKEN_DIM],
    )
    require_signature(
        decoder,
        role="decoder",
        input_name="obs_dict",
        input_shape=[1, DECODER_INPUT_DIM],
        output_name="action",
        output_shape=[1, ACTION_DIM],
    )
    return np, encoder, decoder


def _onnx_replay_vector(
    *,
    np: Any,
    session: Any,
    input_name: str,
    output_name: str,
    input_values: Sequence[float],
    output_size: int,
    context: str,
) -> list[float]:
    try:
        output = session.run(
            [output_name],
            {
                input_name: np.asarray(
                    input_values,
                    dtype=np.float32,
                ).reshape(1, len(input_values))
            },
        )
    except Exception as exc:
        raise ValueError(f"{context} CPU ONNX replay failed: {exc}") from exc
    if (
        len(output) != 1
        or output[0].shape != (1, output_size)
        or output[0].dtype != np.float32
        or not np.isfinite(output[0]).all()
    ):
        raise ValueError(f"{context} CPU ONNX replay output invalid")
    return [float(value) for value in output[0][0]]


def _require_replay_match(
    recorded: Sequence[float],
    replayed: Sequence[float],
    context: str,
) -> None:
    if len(recorded) != len(replayed):
        raise ValueError(f"{context} CPU ONNX replay size mismatch")
    differences = [abs(left - right) for left, right in zip(recorded, replayed, strict=True)]
    if any(
        difference > ONNX_REPLAY_ATOL + ONNX_REPLAY_RTOL * abs(expected)
        for difference, expected in zip(differences, replayed, strict=True)
    ):
        raise ValueError(
            f"{context} disagrees with CPU ONNX replay "
            f"(max_abs_error={max(differences):.9g})"
        )


def _frames_to_term_major_history(frames: Sequence[Sequence[float]]) -> list[float]:
    if len(frames) != PROPRIO_HISTORY_LENGTH or any(
        len(frame) != PROPRIO_FRAME_DIM for frame in frames
    ):
        raise ValueError("internal sequential proprio history dimensions changed")
    return [
        *(value for frame in frames for value in frame[0:3]),
        *(value for frame in frames for value in frame[3:32]),
        *(value for frame in frames for value in frame[32:61]),
        *(value for frame in frames for value in frame[61:90]),
        *(value for frame in frames for value in frame[90:93]),
    ]


def build_encoder_observation(
    encoder_terms: Mapping[str, Any],
    *,
    expected_reference_contract: Mapping[str, Any],
) -> list[float]:
    """Validate two-source terms and concatenate exact 267-vector.

    Temporal semantics come from the hash-bound artifact metadata sidecar.
    They are never inferred from the 240-value tensor shape or trusted from a
    producer-selected command-line profile.
    """

    terms = _exact_mapping(encoder_terms, _ENCODER_TERM_KEYS, "encoder_terms")
    if terms["schema_version"] != ENCODER_TERMS_SCHEMA_VERSION:
        raise ValueError("encoder_terms.schema_version is unsupported")
    if terms["kind"] != ENCODER_TERMS_KIND:
        raise ValueError("encoder_terms.kind is unsupported")
    _integer(
        terms["pico_source_frame_index"],
        "encoder_terms.pico_source_frame_index",
        minimum=0,
    )
    _integer(
        terms["pico_source_monotonic_ns"],
        "encoder_terms.pico_source_monotonic_ns",
        minimum=1,
    )
    contract = _exact_mapping(
        expected_reference_contract,
        {
            "profile",
            "source_sample_rate_hz",
            "source_sample_period_s",
            "future_frame_count",
            "future_frame_step",
            "future_frame_offsets_s",
            "horizon_s",
            "command_layout",
        },
        "expected_reference_contract",
    )
    if contract["future_frame_count"] != REFERENCE_FUTURE_FRAME_COUNT:
        raise ValueError("artifact reference frame count is unsupported")
    expected_offsets = _vector(
        contract["future_frame_offsets_s"],
        REFERENCE_FUTURE_FRAME_COUNT,
        "expected_reference_contract.future_frame_offsets_s",
    )
    offsets = _vector(
        terms["future_frame_offsets_s"],
        REFERENCE_FUTURE_FRAME_COUNT,
        "encoder_terms.future_frame_offsets_s",
    )
    if not _vectors_close(offsets, expected_offsets, tolerance=1e-7):
        raise ValueError(
            "encoder_terms.future_frame_offsets_s disagrees with artifact "
            "reference profile"
        )
    result: list[float] = []
    for name, size in zip(
        TELEOP_ENCODER_INPUT_TERM_ORDER,
        TELEOP_ENCODER_INPUT_TERM_DIMS,
        strict=True,
    ):
        result.extend(_vector(terms[name], size, f"encoder_terms.{name}"))
    if len(result) != TELEOP_ENCODER_INPUT_DIM:
        raise AssertionError("internal encoder observation contract drift")
    return result


def native_to_hardware(native: Sequence[float]) -> list[float]:
    if len(native) != ACTION_DIM:
        raise ValueError(f"native vector must contain {ACTION_DIM} values")
    return [float(native[index]) for index in ISAACLAB_TO_MUJOCO_DOF]


def hardware_to_native(hardware: Sequence[float]) -> list[float]:
    if len(hardware) != ACTION_DIM:
        raise ValueError(f"hardware vector must contain {ACTION_DIM} values")
    return [float(hardware[index]) for index in MUJOCO_TO_ISAACLAB_DOF]


def native_to_padded_il29(native: Sequence[float]) -> list[float]:
    if len(native) != ACTION_DIM:
        raise ValueError(f"native vector must contain {ACTION_DIM} values")
    result = [0.0] * 29
    for source, destination in enumerate(NATIVE_IL23_TO_CANONICAL_IL29):
        result[destination] = float(native[source])
    return result


def _projected_gravity(quaternion_wxyz: Sequence[float]) -> list[float]:
    if len(quaternion_wxyz) != 4:
        raise ValueError("IMU quaternion must contain four values")
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("IMU quaternion is non-finite or degenerate")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    # R(q)^T * [0, 0, -1].
    return [
        2.0 * (w * y - x * z),
        -2.0 * (w * x + y * z),
        -(1.0 - 2.0 * (x * x + y * y)),
    ]


def build_proprio_frame(
    *,
    hardware_q: Sequence[float],
    hardware_dq: Sequence[float],
    imu_gyroscope: Sequence[float],
    imu_quaternion_wxyz: Sequence[float],
    previous_action_native: Sequence[float],
) -> list[float]:
    """Build one [angvel3,q29,dq29,prev29,gravity3] canonical frame."""

    q = [_number(value, "hardware_q") for value in hardware_q]
    dq = [_number(value, "hardware_dq") for value in hardware_dq]
    gyro = [_number(value, "imu_gyroscope") for value in imu_gyroscope]
    previous = [_number(value, "previous_action_native") for value in previous_action_native]
    if len(q) != 23 or len(dq) != 23 or len(previous) != 23 or len(gyro) != 3:
        raise ValueError("invalid true23 proprio source dimensions")
    q_rel_hardware = [
        measured - default for measured, default in zip(q, HARDWARE_DEFAULT_Q, strict=True)
    ]
    frame = [
        *gyro,
        *native_to_padded_il29(hardware_to_native(q_rel_hardware)),
        *native_to_padded_il29(hardware_to_native(dq)),
        *native_to_padded_il29(previous),
        *_projected_gravity(imu_quaternion_wxyz),
    ]
    if len(frame) != PROPRIO_FRAME_DIM:
        raise AssertionError("internal proprio frame contract drift")
    return frame


def validate_proprio_history(history: Sequence[float]) -> None:
    if len(history) != PROPRIO_HISTORY_DIM:
        raise ValueError(f"proprio_history must contain {PROPRIO_HISTORY_DIM} values")
    for frame_index in range(PROPRIO_HISTORY_LENGTH):
        for block_offset, block_name in (
            (JOINT_POSITION_HISTORY_OFFSET, "q_rel"),
            (JOINT_VELOCITY_HISTORY_OFFSET, "dq"),
            (PREVIOUS_ACTION_HISTORY_OFFSET, "previous_action"),
        ):
            for missing_index in SOURCE_IL29_EXCLUDED_INDICES:
                history_index = block_offset + frame_index * 29 + missing_index
                if history[history_index] != 0.0:
                    raise ValueError(
                        "proprio_history fixed slot is non-zero: "
                        f"frame={frame_index}, block={block_name}, "
                        f"canonical_il29={missing_index}"
                    )


def _ticks_advance(previous: int, current: int) -> bool:
    delta = (current - previous) & 0xFFFFFFFF
    return 0 < delta < 0x80000000


def _check_effective_rate(monotonic_ns: Sequence[int], context: str) -> float:
    if len(monotonic_ns) < 2:
        raise ValueError(f"{context} needs at least two timestamps")
    intervals = [
        right - left for left, right in zip(monotonic_ns, monotonic_ns[1:], strict=False)
    ]
    if any(interval <= 0 for interval in intervals):
        raise ValueError(f"{context} timestamps must strictly advance")
    if max(intervals) > MAX_SAMPLE_GAP_NS:
        raise ValueError(f"{context} contains >100 ms freshness gap")
    rate = 1e9 * (len(monotonic_ns) - 1) / (monotonic_ns[-1] - monotonic_ns[0])
    if not MIN_EFFECTIVE_HZ <= rate <= MAX_EFFECTIVE_HZ:
        raise ValueError(f"{context} effective rate {rate:.3f} Hz is outside 40..60 Hz")
    return rate


def _output_bounds(
    hardware_action: Sequence[float],
    previous_hardware_action: Sequence[float] | None,
    dt_s: float | None,
) -> dict[str, float | int | bool]:
    targets = [
        default + action * scale
        for default, action, scale in zip(
            HARDWARE_DEFAULT_Q,
            hardware_action,
            HARDWARE_ACTION_SCALE,
            strict=True,
        )
    ]
    margins = [
        min(target - lower, upper - target)
        for target, lower, upper in zip(
            targets,
            HARDWARE_LOWER_LIMIT,
            HARDWARE_UPPER_LIMIT,
            strict=True,
        )
    ]
    limit_violations = sum(margin < 0.0 for margin in margins)
    slew_checked = previous_hardware_action is not None and dt_s is not None and dt_s > 0.0
    slew_ratios: list[float] = []
    if slew_checked:
        previous_targets = [
            default + action * scale
            for default, action, scale in zip(
                HARDWARE_DEFAULT_Q,
                previous_hardware_action,
                HARDWARE_ACTION_SCALE,
                strict=True,
            )
        ]
        slew_ratios = [
            abs(target - previous) / (velocity_limit * dt_s)
            for target, previous, velocity_limit in zip(
                targets,
                previous_targets,
                HARDWARE_VELOCITY_LIMIT,
                strict=True,
            )
        ]
    return {
        "finite": all(math.isfinite(value) for value in hardware_action),
        "normalized_max_abs": max(abs(value) for value in hardware_action),
        "target_position_min_margin_rad": min(margins),
        "target_limit_violations": limit_violations,
        "slew_checked": slew_checked,
        "target_slew_ratio_max": max(slew_ratios, default=0.0),
        "target_slew_violations": sum(ratio > 1.0 for ratio in slew_ratios),
    }


def _validate_bounds(
    actual: Mapping[str, Any],
    expected: Mapping[str, float | int | bool],
    context: str,
) -> None:
    bounds = _exact_mapping(actual, set(expected), context)
    for key, expected_value in expected.items():
        actual_value = bounds[key]
        if isinstance(expected_value, bool):
            if actual_value is not expected_value:
                raise ValueError(f"{context}.{key} mismatch")
        elif isinstance(expected_value, int):
            if actual_value != expected_value:
                raise ValueError(f"{context}.{key} mismatch")
        elif not _close(_number(actual_value, f"{context}.{key}"), expected_value):
            raise ValueError(f"{context}.{key} mismatch")


def _load_artifact_reference_contract(
    metadata_path: Path,
) -> tuple[str, Mapping[str, Any]]:
    metadata = _load_strict_json(metadata_path, "true23 artifact metadata")
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("true23 artifact metadata schema_version mismatch")
    profile = metadata.get("reference_profile")
    if not isinstance(profile, str):
        raise ValueError("true23 artifact metadata reference_profile is missing")
    expected = reference_profile_contract(profile)
    if metadata.get("reference_contract") != expected:
        raise ValueError(
            "true23 artifact metadata reference_contract does not match profile"
        )
    return profile, expected


def _replay_semantic_reference_source(
    window: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Mapping[str, Any]:
    """Replay approved recorded source; reject unaudited live producers."""

    source = window.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("semantic reference source is missing")
    if source.get("kind") != SOURCE_RECORDED_MOTION:
        raise ValueError(
            "live planner/retargeter semantic reference requires approved "
            "process-bound producer evidence; promotion remains blocked"
        )
    relpath_value = source.get("source_relpath")
    if not isinstance(relpath_value, str) or not relpath_value:
        raise ValueError("recorded semantic reference source_relpath is missing")
    relpath = PurePosixPath(relpath_value.replace("\\", "/"))
    if (
        relpath.is_absolute()
        or ".." in relpath.parts
        or relpath.parts[:2] != ("gear_sonic_deploy", "reference")
    ):
        raise ValueError("recorded semantic reference must stay under approved reference path")
    root = repo_root.resolve()
    motion_dir = root.joinpath(*relpath.parts).resolve()
    try:
        motion_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("recorded semantic reference escapes repo root") from exc
    return validate_semantic_reference_window(window, motion_dir=motion_dir)


def _validate_source_contract(
    value: Any,
    *,
    reference_profile: str,
    expected_reference_contract: Mapping[str, Any],
) -> None:
    expected = {
        "encoder_terms_kind": ENCODER_TERMS_KIND,
        "encoder_terms_schema_version": ENCODER_TERMS_SCHEMA_VERSION,
        "semantic_reference_window_kind": SEMANTIC_REFERENCE_WINDOW_KIND,
        "semantic_reference_window_schema_version": (
            SEMANTIC_REFERENCE_SCHEMA_VERSION
        ),
        "reference_profile": reference_profile,
        "reference_contract": dict(expected_reference_contract),
        "command_semantics": "artifact_profile_positions120_then_velocities120",
        "encoder_term_order": list(TELEOP_ENCODER_INPUT_TERM_ORDER),
        "decoder_layout": (
            "token64_then_term_major_"
            "angvel_h10_qrel_h10_dq_h10_previous_action_h10_gravity_h10_"
            "each_term_oldest_to_newest"
        ),
        "missing_canonical_il29_slots": list(SOURCE_IL29_EXCLUDED_INDICES),
        "control_period_s": 0.02,
    }
    contract = _exact_mapping(value, set(expected), "source_contract")
    for key, expected_value in expected.items():
        if contract[key] != expected_value:
            raise ValueError(f"source_contract.{key} mismatch")


def validate_live_shadow_evidence(
    evidence: Mapping[str, Any],
    *,
    producer_path: Path,
    artifact_paths: Mapping[str, Path],
    repo_root: Path,
    now_utc: datetime,
    approval_manifest_path: Path | None = None,
) -> Mapping[str, Any]:
    """Validate complete read-only live-shadow evidence or raise ``ValueError``."""

    root = _exact_mapping(evidence, _ROOT_KEYS, "live evidence")
    if root["schema_version"] != LIVE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported live evidence schema_version")
    if root["kind"] != LIVE_EVIDENCE_KIND:
        raise ValueError("unsupported live evidence kind")
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    captured = datetime.fromisoformat(str(root["captured_at_utc"]).replace("Z", "+00:00"))
    if captured.tzinfo is None:
        raise ValueError("captured_at_utc must be timezone-aware")
    age_s = (now_utc - captured.astimezone(timezone.utc)).total_seconds()
    if age_s < -5.0 or age_s > MAX_EVIDENCE_AGE_S:
        raise ValueError(f"live evidence stale or future-dated (age={age_s:.3f}s)")

    producer_path = producer_path.resolve()
    source_path = (repo_root.resolve() / LIVE_PRODUCER_SOURCE).resolve()
    if (
        producer_path.name != LIVE_PRODUCER_FILENAME
        or not producer_path.is_file()
        or producer_path.stat().st_size < 1024
        or _binary_format(producer_path) != "ELF"
        or not source_path.is_file()
    ):
        raise ValueError("live producer must be materialized pinned ELF and source must exist")
    approval_path = (
        (repo_root.resolve() / DEFAULT_APPROVAL_MANIFEST)
        if approval_manifest_path is None
        else approval_manifest_path.resolve()
    )
    approval = _validate_producer_approval(
        manifest_path=approval_path,
        repo_root=repo_root,
        producer_path=producer_path,
    )
    producer = _exact_mapping(
        root["producer"],
        {
            "kind",
            "version",
            "filename",
            "sha256",
            "binary_format",
            "source_relpath",
            "source_sha256",
        },
        "producer",
    )
    expected_producer = {
        "kind": LIVE_PRODUCER_KIND,
        "version": LIVE_PRODUCER_VERSION,
        "filename": LIVE_PRODUCER_FILENAME,
        "sha256": _sha256(producer_path),
        "binary_format": "ELF",
        "source_relpath": LIVE_PRODUCER_SOURCE,
        "source_sha256": _sha256(source_path),
    }
    if producer != expected_producer:
        raise ValueError("producer identity, source, or binary hash mismatch")

    if set(artifact_paths) != _ARTIFACT_HASH_KEYS:
        raise ValueError(f"artifact_paths must contain exact keys {sorted(_ARTIFACT_HASH_KEYS)}")
    artifact_hashes = _exact_mapping(
        root["artifact_hashes"], _ARTIFACT_HASH_KEYS, "artifact_hashes"
    )
    for key, path in artifact_paths.items():
        resolved = path.resolve()
        if not resolved.is_file() or artifact_hashes[key] != _sha256(resolved):
            raise ValueError(f"artifact hash mismatch: {key}")
    reference_profile, expected_reference_contract = (
        _load_artifact_reference_contract(artifact_paths["metadata"].resolve())
    )
    np, encoder_session, decoder_session = _load_onnx_replay_sessions(
        artifact_paths["encoder_onnx"].resolve(),
        artifact_paths["decoder_onnx"].resolve(),
    )

    _validate_source_contract(
        root["source_contract"],
        reference_profile=reference_profile,
        expected_reference_contract=expected_reference_contract,
    )
    window = _exact_mapping(
        root["window"], {"start_monotonic_ns", "end_monotonic_ns"}, "window"
    )
    start_ns = _integer(window["start_monotonic_ns"], "window.start_monotonic_ns", minimum=1)
    end_ns = _integer(window["end_monotonic_ns"], "window.end_monotonic_ns", minimum=1)
    if end_ns <= start_ns or end_ns - start_ns > MAX_COMMON_WINDOW_NS:
        raise ValueError("invalid common monotonic window")

    pico_samples = root["pico_samples"]
    lowstate_samples = root["lowstate_samples"]
    inference_samples = root["inference_samples"]
    if not all(isinstance(records, list) for records in (pico_samples, lowstate_samples, inference_samples)):
        raise ValueError("sample collections must be lists")
    if min(len(pico_samples), len(lowstate_samples), len(inference_samples)) < MIN_LIVE_SAMPLES:
        raise ValueError(f"need at least {MIN_LIVE_SAMPLES} samples from every live source")

    pico_by_source_ns: dict[int, Mapping[str, Any]] = {}
    pico_times: list[int] = []
    pico_source_times: list[int] = []
    pico_source_frame_indices: list[int] = []
    pico_keys = {
        "monotonic_ns",
        "body_source_frame_index",
        "body_source_ns",
        "body_pose",
        "tracker_ids",
        "tracking_state",
        "calibrated",
        "left_source_ns",
        "left_pose",
        "left_tracking_bits",
        "right_source_ns",
        "right_pose",
        "right_tracking_bits",
    }
    for index, raw_sample in enumerate(pico_samples):
        sample = _exact_mapping(raw_sample, pico_keys, f"pico_samples[{index}]")
        monotonic_ns = _integer(
            sample["monotonic_ns"], f"pico_samples[{index}].monotonic_ns", minimum=1
        )
        body_source_ns = _integer(
            sample["body_source_ns"], f"pico_samples[{index}].body_source_ns", minimum=1
        )
        body_source_frame_index = _integer(
            sample["body_source_frame_index"],
            f"pico_samples[{index}].body_source_frame_index",
            minimum=0,
        )
        if sample["tracking_state"] != "BT_VALID" or sample["calibrated"] is not True:
            raise ValueError(f"pico_samples[{index}] tracking/calibration invalid")
        trackers = sample["tracker_ids"]
        if not isinstance(trackers, list) or len(set(trackers)) < 2:
            raise ValueError(f"pico_samples[{index}] needs two distinct trackers")
        _vector(sample["body_pose"], 168, f"pico_samples[{index}].body_pose")
        _vector(sample["left_pose"], 7, f"pico_samples[{index}].left_pose")
        _vector(sample["right_pose"], 7, f"pico_samples[{index}].right_pose")
        if sample["left_tracking_bits"] & 3 != 3 or sample["right_tracking_bits"] & 3 != 3:
            raise ValueError(f"pico_samples[{index}] controller tracking incomplete")
        if body_source_ns in pico_by_source_ns:
            raise ValueError("duplicate PICO body_source_ns")
        pico_by_source_ns[body_source_ns] = sample
        pico_times.append(monotonic_ns)
        pico_source_times.append(body_source_ns)
        pico_source_frame_indices.append(body_source_frame_index)
    pico_hz = _check_effective_rate(pico_times, "pico_samples")
    _check_effective_rate(pico_source_times, "pico_samples.body_source_ns")
    if any(
        current <= previous
        for previous, current in zip(
            pico_source_frame_indices,
            pico_source_frame_indices[1:],
            strict=False,
        )
    ):
        raise ValueError("pico_samples.body_source_frame_index did not advance")

    lowstate_by_tick: dict[int, Mapping[str, Any]] = {}
    lowstate_times: list[int] = []
    lowstate_ticks: list[int] = []
    lowstate_keys = {
        "monotonic_ns",
        "tick",
        "mode_machine",
        "crc_expected",
        "crc_computed",
        "hardware_joint_ids",
        "q",
        "dq",
        "imu_quaternion_wxyz",
        "imu_gyroscope",
    }
    for index, raw_sample in enumerate(lowstate_samples):
        sample = _exact_mapping(raw_sample, lowstate_keys, f"lowstate_samples[{index}]")
        monotonic_ns = _integer(
            sample["monotonic_ns"], f"lowstate_samples[{index}].monotonic_ns", minimum=1
        )
        tick = _integer(sample["tick"], f"lowstate_samples[{index}].tick", minimum=0)
        if tick > 0xFFFFFFFF:
            raise ValueError(f"lowstate_samples[{index}].tick is not uint32")
        if sample["mode_machine"] != 4:
            raise ValueError(f"lowstate_samples[{index}] mode_machine is not 4")
        if sample["crc_expected"] != sample["crc_computed"]:
            raise ValueError(f"lowstate_samples[{index}] CRC mismatch")
        if sample["hardware_joint_ids"] != list(HARDWARE_JOINT_IDS):
            raise ValueError(f"lowstate_samples[{index}] active-slot layout mismatch")
        _vector(sample["q"], 23, f"lowstate_samples[{index}].q")
        _vector(sample["dq"], 23, f"lowstate_samples[{index}].dq")
        _vector(
            sample["imu_quaternion_wxyz"],
            4,
            f"lowstate_samples[{index}].imu_quaternion_wxyz",
        )
        _vector(
            sample["imu_gyroscope"], 3, f"lowstate_samples[{index}].imu_gyroscope"
        )
        if tick in lowstate_by_tick:
            raise ValueError("duplicate LowState tick")
        lowstate_by_tick[tick] = sample
        lowstate_times.append(monotonic_ns)
        lowstate_ticks.append(tick)
    lowstate_hz = _check_effective_rate(lowstate_times, "lowstate_samples")
    if any(
        not _ticks_advance(previous, current)
        for previous, current in zip(lowstate_ticks, lowstate_ticks[1:], strict=False)
    ):
        raise ValueError("LowState tick did not advance under uint32 wrap rules")

    inference_times: list[int] = []
    inference_pico_source_times: list[int] = []
    inference_lowstate_ticks: list[int] = []
    inference_pico_source_frame_indices: list[int] = []
    inference_reference_frame_indices: list[int] = []
    inference_reference_frame_times: list[int] = []
    inference_reference_epochs: list[int] = []
    reference_source_identities: set[tuple[str, str, str]] = set()
    previous_native: list[float] | None = None
    previous_hardware: list[float] | None = None
    previous_inference_ns: int | None = None
    expected_history_frames: list[list[float]] = []
    observed_bounds: list[Mapping[str, float | int | bool]] = []
    for index, raw_sample in enumerate(inference_samples):
        sample = _exact_mapping(raw_sample, _INFERENCE_KEYS, f"inference_samples[{index}]")
        monotonic_ns = _integer(
            sample["monotonic_ns"], f"inference_samples[{index}].monotonic_ns", minimum=1
        )
        pico_source_ns = _integer(
            sample["pico_body_source_ns"],
            f"inference_samples[{index}].pico_body_source_ns",
            minimum=1,
        )
        tick = _integer(
            sample["lowstate_tick"], f"inference_samples[{index}].lowstate_tick", minimum=0
        )
        if pico_source_ns not in pico_by_source_ns:
            raise ValueError(f"inference_samples[{index}] is not bound to a PICO frame")
        if tick not in lowstate_by_tick:
            raise ValueError(f"inference_samples[{index}] is not bound to a LowState tick")
        lowstate = lowstate_by_tick[tick]
        pico = pico_by_source_ns[pico_source_ns]
        if abs(monotonic_ns - int(lowstate["monotonic_ns"])) > MAX_SOURCE_PAIR_SKEW_NS:
            raise ValueError(f"inference_samples[{index}] LowState pairing is stale")
        if abs(monotonic_ns - int(pico["monotonic_ns"])) > MAX_SOURCE_PAIR_SKEW_NS:
            raise ValueError(f"inference_samples[{index}] PICO pairing is stale")

        reference_window = sample["semantic_reference_window"]
        reference_summary = _replay_semantic_reference_source(
            reference_window,
            repo_root=repo_root,
        )
        if reference_summary["profile"] != reference_profile:
            raise ValueError(
                f"inference_samples[{index}] semantic reference profile "
                "disagrees with artifact metadata"
            )
        reference_playback = reference_window["playback"]
        if reference_playback["emitted_monotonic_ns"] != monotonic_ns:
            raise ValueError(
                f"inference_samples[{index}] semantic reference emission "
                "is not bound to inference time"
            )
        reference_source = reference_window["source"]
        reference_source_identities.add(
            (
                str(reference_source["kind"]),
                str(reference_source["source_id"]),
                str(reference_source["identity_sha256"]),
            )
        )
        inference_reference_frame_indices.append(
            int(reference_playback["frame_index"])
        )
        inference_reference_frame_times.append(
            int(reference_playback["frame_monotonic_ns"])
        )
        inference_reference_epochs.append(
            int(reference_playback["epoch_monotonic_ns"])
        )

        terms = sample["encoder_terms"]
        observation = build_encoder_observation(
            terms,
            expected_reference_contract=expected_reference_contract,
        )
        if (
            terms["pico_source_monotonic_ns"] != pico_source_ns
            or terms["pico_source_frame_index"]
            != pico["body_source_frame_index"]
        ):
            raise ValueError(
                f"inference_samples[{index}] PICO capture binding mismatch"
            )
        if terms["command_multi_future_lower_body"] != reference_window[
            "command_multi_future_lower_body"
        ]:
            raise ValueError(
                f"inference_samples[{index}] lower-body command is not bound "
                "to semantic reference window"
            )
        inference_pico_source_frame_indices.append(
            int(terms["pico_source_frame_index"])
        )
        recorded_observation = _vector(
            sample["observation"], TELEOP_ENCODER_INPUT_DIM, f"inference_samples[{index}].observation"
        )
        if not _vectors_close(recorded_observation, observation):
            raise ValueError(f"inference_samples[{index}] observation is not exact term concatenation")
        token = _vector(sample["token"], TOKEN_DIM, f"inference_samples[{index}].token")
        replayed_token = _onnx_replay_vector(
            np=np,
            session=encoder_session,
            input_name="teleop_obs",
            output_name="token",
            input_values=observation,
            output_size=TOKEN_DIM,
            context=f"inference_samples[{index}].token",
        )
        _require_replay_match(
            token,
            replayed_token,
            f"inference_samples[{index}].token",
        )
        history = _vector(
            sample["proprio_history"],
            PROPRIO_HISTORY_DIM,
            f"inference_samples[{index}].proprio_history",
        )
        validate_proprio_history(history)
        decoder_input = _vector(
            sample["decoder_input"], DECODER_INPUT_DIM, f"inference_samples[{index}].decoder_input"
        )
        if not _vectors_close(decoder_input, [*token, *history]):
            raise ValueError(f"inference_samples[{index}] decoder input is not token64+history930")

        previous_action = _vector(
            sample["previous_action_native"],
            ACTION_DIM,
            f"inference_samples[{index}].previous_action_native",
        )
        expected_previous_action = (
            [0.0] * ACTION_DIM if previous_native is None else previous_native
        )
        if not _vectors_close(previous_action, expected_previous_action):
            raise ValueError(f"inference_samples[{index}] previous action chain mismatch")
        expected_frame = build_proprio_frame(
            hardware_q=lowstate["q"],
            hardware_dq=lowstate["dq"],
            imu_gyroscope=lowstate["imu_gyroscope"],
            imu_quaternion_wxyz=lowstate["imu_quaternion_wxyz"],
            previous_action_native=previous_action,
        )
        if not expected_history_frames:
            expected_history_frames = [
                expected_frame.copy() for _ in range(PROPRIO_HISTORY_LENGTH)
            ]
        else:
            expected_history_frames = [
                *expected_history_frames[1:],
                expected_frame,
            ]
        expected_history = _frames_to_term_major_history(expected_history_frames)
        if not _vectors_close(history, expected_history):
            raise ValueError(
                f"inference_samples[{index}] sequential term-major proprio history mismatch"
            )

        native_action = _vector(
            sample["native_action"], ACTION_DIM, f"inference_samples[{index}].native_action"
        )
        replayed_native_action = _onnx_replay_vector(
            np=np,
            session=decoder_session,
            input_name="obs_dict",
            output_name="action",
            input_values=decoder_input,
            output_size=ACTION_DIM,
            context=f"inference_samples[{index}].native_action",
        )
        _require_replay_match(
            native_action,
            replayed_native_action,
            f"inference_samples[{index}].native_action",
        )
        hardware_action = _vector(
            sample["hardware_action"], ACTION_DIM, f"inference_samples[{index}].hardware_action"
        )
        expected_hardware = native_to_hardware(replayed_native_action)
        if not _vectors_close(hardware_action, expected_hardware):
            raise ValueError(f"inference_samples[{index}] native/hardware permutation mismatch")
        dt_s = (
            None
            if previous_inference_ns is None
            else (monotonic_ns - previous_inference_ns) / 1e9
        )
        expected_bounds = _output_bounds(expected_hardware, previous_hardware, dt_s)
        _validate_bounds(
            sample["output_bounds"],
            expected_bounds,
            f"inference_samples[{index}].output_bounds",
        )
        if expected_bounds["finite"] is not True:
            raise ValueError(f"inference_samples[{index}] action is non-finite")
        observed_bounds.append(expected_bounds)
        previous_native = replayed_native_action
        previous_hardware = expected_hardware
        previous_inference_ns = monotonic_ns
        inference_times.append(monotonic_ns)
        inference_pico_source_times.append(pico_source_ns)
        inference_lowstate_ticks.append(tick)

    inference_hz = _check_effective_rate(inference_times, "inference_samples")
    if any(
        current <= previous
        for previous, current in zip(
            inference_pico_source_times,
            inference_pico_source_times[1:],
            strict=False,
        )
    ):
        raise ValueError("inference-bound PICO source timestamps did not advance")
    if any(
        current <= previous
        for previous, current in zip(
            inference_pico_source_frame_indices,
            inference_pico_source_frame_indices[1:],
            strict=False,
        )
    ):
        raise ValueError("inference-bound PICO source frame indices did not advance")
    if len(reference_source_identities) != 1:
        raise ValueError("semantic reference source identity changed during live window")
    if len(set(inference_reference_epochs)) != 1:
        raise ValueError("semantic reference playback epoch changed during live window")
    for index in range(1, len(inference_reference_frame_indices)):
        frame_delta = (
            inference_reference_frame_indices[index]
            - inference_reference_frame_indices[index - 1]
        )
        if frame_delta <= 0:
            raise ValueError(
                "inference-bound semantic reference playback frames did not advance"
            )
        inference_delta_ns = inference_times[index] - inference_times[index - 1]
        reference_delta_ns = (
            inference_reference_frame_times[index]
            - inference_reference_frame_times[index - 1]
        )
        if (
            frame_delta * SOURCE_SAMPLE_PERIOD_NS != inference_delta_ns
            or reference_delta_ns != inference_delta_ns
        ):
            raise ValueError(
                "semantic reference playback delta does not match inference time"
            )
    if any(
        not _ticks_advance(previous, current)
        for previous, current in zip(
            inference_lowstate_ticks,
            inference_lowstate_ticks[1:],
            strict=False,
        )
    ):
        raise ValueError("inference-bound LowState ticks did not advance")
    for records, context in (
        (pico_times, "PICO"),
        (lowstate_times, "LowState"),
        (inference_times, "inference"),
    ):
        if records[0] < start_ns or records[-1] > end_ns:
            raise ValueError(f"{context} samples fall outside common window")

    reference_source_kind = next(iter(reference_source_identities))[0]
    computed_summary = {
        "sample_count": len(inference_samples),
        "semantic_reference_window_count": len(inference_samples),
        "reference_profile": reference_profile,
        "reference_horizon_s": float(expected_reference_contract["horizon_s"]),
        "reference_source_kind": reference_source_kind,
        "pico_effective_hz": pico_hz,
        "lowstate_effective_hz": lowstate_hz,
        "inference_effective_hz": inference_hz,
        "all_outputs_finite": True,
        "target_limit_violation_count": sum(
            int(bounds["target_limit_violations"]) for bounds in observed_bounds
        ),
        "target_slew_violation_count": sum(
            int(bounds["target_slew_violations"]) for bounds in observed_bounds
        ),
        "normalized_action_max_abs": max(
            float(bounds["normalized_max_abs"]) for bounds in observed_bounds
        ),
        "target_slew_ratio_max": max(
            float(bounds["target_slew_ratio_max"]) for bounds in observed_bounds
        ),
        "onnx_replay_sample_count": len(inference_samples),
        "onnx_replay_atol": ONNX_REPLAY_ATOL,
        "onnx_replay_rtol": ONNX_REPLAY_RTOL,
        "missing_required_fields": [],
    }
    summary = _exact_mapping(root["summary"], set(computed_summary), "summary")
    for key, expected_value in computed_summary.items():
        actual_value = summary[key]
        if isinstance(expected_value, float):
            if not _close(_number(actual_value, f"summary.{key}"), expected_value):
                raise ValueError(f"summary.{key} mismatch")
        elif actual_value != expected_value:
            raise ValueError(f"summary.{key} mismatch")
    if computed_summary["target_limit_violation_count"] != 0:
        raise ValueError("live shadow found physical target-limit violations")
    if computed_summary["target_slew_violation_count"] != 0:
        raise ValueError("live shadow found physical velocity/slew violations")

    authorization = _exact_mapping(
        root["authorization"],
        {
            "lowcmd_publisher_present",
            "command_writer_present",
            "motion_switcher_present",
            "robot_mutation_authorized",
        },
        "authorization",
    )
    if authorization != {
        "lowcmd_publisher_present": False,
        "command_writer_present": False,
        "motion_switcher_present": False,
        "robot_mutation_authorized": False,
    }:
        raise ValueError("live shadow evidence contains command authority")
    return {
        **computed_summary,
        "producer_sha256": expected_producer["sha256"],
        "source_sha256": expected_producer["source_sha256"],
        "core_sha256": approval["core_sha256"],
        "audit_sha256": approval["audit_sha256"],
        "approval_manifest_sha256": approval["manifest_sha256"],
        "age_s": age_s,
    }
