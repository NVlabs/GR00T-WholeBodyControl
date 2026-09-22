"""Wire schema shared by the robot publisher and host receiver."""

from __future__ import annotations

import msgpack
import numpy as np

SCHEMA_VERSION = 1
TOPIC = b"g1_upper_body_telemetry"

JOINT_INDICES = tuple(range(12, 29))
JOINT_NAMES = (
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)

ARRAY_FIELDS = (
    "q_cmd", "dq_cmd", "ddq_cmd", "tau_cmd_ff", "kp", "kd", "tau_cmd_pd",
    "q_est", "dq_est", "ddq_est", "tau_est",
    "q_residual", "dq_residual", "ddq_residual",
    "tau_residual", "tau_residual_ff",
)

SCALAR_FIELDS = (
    "schema_version", "sequence_id", "source_timestamp_ns",
    "source_monotonic_ns", "state_age_sec", "command_age_sec",
    "publish_hz", "ddq_cmd_valid",
)


def _array(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    expected = (len(JOINT_NAMES),)
    if result.shape != expected:
        raise ValueError(f"{name} has shape {result.shape}, expected {expected}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def validate_sample(sample: dict) -> None:
    missing = [key for key in (*SCALAR_FIELDS, *ARRAY_FIELDS) if key not in sample]
    if missing:
        raise ValueError(f"Telemetry sample is missing fields: {missing}")
    if int(sample["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported telemetry schema {sample['schema_version']}; "
            f"expected {SCHEMA_VERSION}"
        )
    for field in ARRAY_FIELDS:
        _array(sample[field], field)


def encode_sample(sample: dict) -> bytes:
    """Validate and encode one sample without requiring msgpack-numpy."""
    validate_sample(sample)
    payload = dict(sample)
    for field in ARRAY_FIELDS:
        payload[field] = _array(payload[field], field).tolist()
    return msgpack.packb(payload, use_bin_type=True)


def decode_sample(payload: bytes) -> dict:
    sample = msgpack.unpackb(payload, raw=False)
    validate_sample(sample)
    for field in ARRAY_FIELDS:
        sample[field] = _array(sample[field], field)
    sample["schema_version"] = int(sample["schema_version"])
    sample["sequence_id"] = int(sample["sequence_id"])
    sample["source_timestamp_ns"] = int(sample["source_timestamp_ns"])
    sample["source_monotonic_ns"] = int(sample["source_monotonic_ns"])
    sample["ddq_cmd_valid"] = bool(sample["ddq_cmd_valid"])
    return sample
