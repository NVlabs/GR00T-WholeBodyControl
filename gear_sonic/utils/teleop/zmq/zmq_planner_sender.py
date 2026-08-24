"""Builders for ZMQ wire-format messages on the 'command', 'planner', and 'pose' topics.

Message layout: [topic_bytes][1280-byte JSON header][packed binary payload].
The header describes field names, dtypes, and shapes so the receiver can
deserialize without out-of-band schema knowledge.
"""

import json
import struct
from typing import Sequence

import numpy as np

HEADER_SIZE = 1280
CONTROL_SESSION_TOKEN_SIZE = 16


def _build_header(fields: list, version: int = 1, count: int = 1) -> bytes:
    header = {
        "v": version,
        "endian": "le",
        "count": count,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def _control_session_token(value: bytes | bytearray | memoryview, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{name} must be exactly {CONTROL_SESSION_TOKEN_SIZE} bytes")
    token = bytes(value)
    if len(token) != CONTROL_SESSION_TOKEN_SIZE or not any(token):
        raise ValueError(f"{name} must be a non-zero {CONTROL_SESSION_TOKEN_SIZE}-byte token")
    return token


def _command_index(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < 0
        or value > np.iinfo(np.int64).max
    ):
        raise ValueError("command_index must be a non-negative int64")
    return int(value)


def build_command_message(
    start: bool,
    stop: bool,
    planner: bool,
    delta_heading: float | None = None,
    *,
    receiver_epoch: bytes | bytearray | memoryview | None = None,
    publisher_session: bytes | bytearray | memoryview | None = None,
    command_index: int | None = None,
    claim: bool = False,
) -> bytes:
    """
    Assemble a 'command' topic message:
      - start: u8 (1=start control)
      - stop: u8 (1=stop control)
      - planner: u8 (1=planner mode, 0=streamed motion)
      - delta_heading: f32 (optional, yaw relative to heading command in radians)
    Returns: bytes ready to send via socket.send()
    """
    session_values = (receiver_epoch, publisher_session, command_index)
    has_session_envelope = any(value is not None for value in session_values)
    if has_session_envelope and not all(value is not None for value in session_values):
        raise ValueError("receiver_epoch, publisher_session, and command_index must be supplied together")
    if claim and not has_session_envelope:
        raise ValueError("claim requires a control-session envelope")
    if claim and (start or stop):
        raise ValueError("claim cannot be combined with start or stop")
    if start and stop:
        raise ValueError("start and stop cannot both be true")
    fields = []
    payload_parts = []
    if has_session_envelope:
        epoch = _control_session_token(receiver_epoch, "receiver_epoch")
        session = _control_session_token(publisher_session, "publisher_session")
        index = _command_index(command_index)
        if not (claim or start or stop):
            raise ValueError("session command must contain exactly one action")
        if delta_heading is not None:
            raise ValueError("delta_heading is not supported by session commands")
        fields.extend(
            (
                {"name": "receiver_epoch", "dtype": "u8", "shape": [16]},
                {"name": "publisher_session", "dtype": "u8", "shape": [16]},
                {"name": "command_index", "dtype": "i64", "shape": [1]},
                {"name": "claim", "dtype": "u8", "shape": [1]},
            )
        )
        payload_parts.extend(
            (
                epoch,
                session,
                struct.pack("<q", index),
                struct.pack("B", 1 if claim else 0),
            )
        )

    fields.extend(
        [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ]
    )
    payload_parts.extend(
        (
            struct.pack("B", 1 if start else 0),
            struct.pack("B", 1 if stop else 0),
            struct.pack("B", 1 if planner else 0),
        )
    )
    payload = b"".join(payload_parts)

    if delta_heading is not None:
        # Append delta_heading field to header and payload
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        payload += struct.pack("<f", float(delta_heading))

    header = _build_header(fields, version=2 if has_session_envelope else 1, count=1)

    return b"command" + header + payload


def build_planner_message(
    frame_index: int,
    mode: int,
    movement: Sequence[float],
    facing: Sequence[float],
    speed: float = -1.0,
    height: float = -1.0,
    upper_body_position: Sequence[float] | None = None,
    upper_body_velocity: Sequence[float] | None = None,
    left_hand_position: Sequence[float] | None = None,
    right_hand_position: Sequence[float] | None = None,
    vr_3pt_position: Sequence[float] | None = None,
    vr_3pt_orientation: Sequence[float] | None = None,
    vr_3pt_compliance: Sequence[float] | None = None,
    *,
    receiver_epoch: bytes | bytearray | memoryview | None = None,
    publisher_session: bytes | bytearray | memoryview | None = None,
) -> bytes:
    """
    Assemble a 'planner' topic message:
      - frame_index: i64 (strictly increasing publisher sequence)
      - mode: i32 (LocomotionMode enum)
      - movement: f32[3] (x,y,z)
      - facing: f32[3] (x,y,z)
      - speed: f32 (optional, -1 for default)
      - height: f32 (optional, -1 for default)
    Returns: bytes ready to send via socket.send()
    """
    if len(movement) != 3:
        raise ValueError("movement must have length 3")
    if len(facing) != 3:
        raise ValueError("facing must have length 3")
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, (int, np.integer))
        or frame_index < 0
        or frame_index > np.iinfo(np.int64).max
    ):
        raise ValueError("frame_index must be a non-negative int64")

    has_session_envelope = receiver_epoch is not None or publisher_session is not None
    if has_session_envelope and (receiver_epoch is None or publisher_session is None):
        raise ValueError("receiver_epoch and publisher_session must be supplied together")

    fields = []
    payload_parts = []
    if has_session_envelope:
        epoch = _control_session_token(receiver_epoch, "receiver_epoch")
        session = _control_session_token(publisher_session, "publisher_session")
        fields.extend(
            (
                {"name": "receiver_epoch", "dtype": "u8", "shape": [16]},
                {"name": "publisher_session", "dtype": "u8", "shape": [16]},
            )
        )
        payload_parts.extend((epoch, session))

    fields.extend(
        [
            {"name": "frame_index", "dtype": "i64", "shape": [1]},
            {"name": "mode", "dtype": "i32", "shape": [1]},
            {"name": "movement", "dtype": "f32", "shape": [3]},
            {"name": "facing", "dtype": "f32", "shape": [3]},
            {"name": "speed", "dtype": "f32", "shape": [1]},
            {"name": "height", "dtype": "f32", "shape": [1]},
        ]
    )

    payload_parts.extend(
        (
            struct.pack("<q", int(frame_index)),
            struct.pack("<i", int(mode)),
            struct.pack("<fff", float(movement[0]), float(movement[1]), float(movement[2])),
            struct.pack("<fff", float(facing[0]), float(facing[1]), float(facing[2])),
            struct.pack("<f", float(speed)),
            struct.pack("<f", float(height)),
        )
    )
    payload = b"".join(payload_parts)

    # Add upper body position and velocity to payload, optionally
    if upper_body_position is not None:
        fields.append({"name": "upper_body_position", "dtype": "f32", "shape": [len(upper_body_position)]})
        for value in upper_body_position:
            payload += struct.pack("<f", float(value))

    if upper_body_velocity is not None:
        fields.append({"name": "upper_body_velocity", "dtype": "f32", "shape": [len(upper_body_velocity)]})
        for value in upper_body_velocity:
            payload += struct.pack("<f", float(value))

    if left_hand_position is not None:
        fields.append({"name": "left_hand_joints", "dtype": "f32", "shape": [len(left_hand_position)]})
        for value in left_hand_position:
            payload += struct.pack("<f", float(value))

    if right_hand_position is not None:
        fields.append({"name": "right_hand_joints", "dtype": "f32", "shape": [len(right_hand_position)]})
        for value in right_hand_position:
            payload += struct.pack("<f", float(value))

    if vr_3pt_position is not None:
        fields.append({"name": "vr_position", "dtype": "f32", "shape": [len(vr_3pt_position)]})
        for value in vr_3pt_position:
            payload += struct.pack("<f", float(value))

    if vr_3pt_orientation is not None:
        fields.append({"name": "vr_orientation", "dtype": "f32", "shape": [len(vr_3pt_orientation)]})
        for value in vr_3pt_orientation:
            payload += struct.pack("<f", float(value))

    if vr_3pt_compliance is not None:
        fields.append({"name": "vr_compliance", "dtype": "f32", "shape": [len(vr_3pt_compliance)]})
        for value in vr_3pt_compliance:
            payload += struct.pack("<f", float(value))

    header = _build_header(fields, version=2 if has_session_envelope else 1, count=1)

    return b"planner" + header + payload


def pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 3) -> bytes:
    """
    Pack pose/action data into ZMQ message format:
    [topic_prefix][1280-byte JSON header][concatenated binary fields]

    This is a general-purpose function for packing numpy arrays into ZMQ messages.
    Supports protocol versions 3 and 4.

    Args:
        pose_data: Dictionary containing numpy arrays to send
        topic: Topic prefix string (default: "pose")
        version: Protocol version (default: 3). Version 4 includes "count" field.

    Returns:
        Packed message as bytes

    Example:
        >>> data = {
        ...     "token_state": np.array([1.0, 2.0], dtype=np.float32),
        ...     "frame_index": np.array([0], dtype=np.int64)
        ... }
        >>> msg = pack_pose_message(data, topic="pose", version=4)
    """
    # Build fields list from pose_data
    fields = []
    binary_data = []

    for key, value in pose_data.items():
        if isinstance(value, np.ndarray):
            # Determine dtype string
            if value.dtype == np.float32:
                dtype_str = "f32"
            elif value.dtype == np.float64:
                dtype_str = "f64"
            elif value.dtype == np.int32:
                dtype_str = "i32"
            elif value.dtype == np.int64:
                dtype_str = "i64"
            elif value.dtype == np.uint8:
                dtype_str = "u8"
            elif value.dtype == bool:
                dtype_str = "bool"
            else:
                # Default to f32, cast if needed
                dtype_str = "f32"
                value = value.astype(np.float32)

            fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})

            # Ensure contiguous and little-endian
            if not value.flags["C_CONTIGUOUS"]:
                value = np.ascontiguousarray(value)
            if value.dtype.byteorder == ">":
                value = value.astype(value.dtype.newbyteorder("<"))

            binary_data.append(value.tobytes())

    # Build header using common utility
    header_bytes = _build_header(fields, version=version, count=1)

    # Pack message: [topic][1280-byte header][binary data]
    topic_bytes = topic.encode("utf-8")
    data_bytes = b"".join(binary_data)

    packed_message = topic_bytes + header_bytes + data_bytes
    return packed_message
