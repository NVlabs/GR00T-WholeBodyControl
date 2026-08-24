import json
import struct

import numpy as np
import pytest

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    HEADER_SIZE,
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

RECEIVER_EPOCH = bytes(range(1, 17))
PUBLISHER_SESSION = bytes(range(17, 33))


def test_planner_message_carries_strict_source_sequence():
    message = build_planner_message(
        7,
        0,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    )

    header_start = len(b"planner")
    header = json.loads(message[header_start : header_start + HEADER_SIZE].rstrip(b"\x00"))
    payload = message[header_start + HEADER_SIZE :]

    assert header["fields"][0] == {
        "name": "frame_index",
        "dtype": "i64",
        "shape": [1],
    }
    assert struct.unpack_from("<q", payload)[0] == 7


@pytest.mark.parametrize("frame_index", [-1, True, 2**63, 1.5])
def test_planner_message_rejects_invalid_source_sequence(frame_index):
    with pytest.raises(ValueError, match="frame_index"):
        build_planner_message(
            frame_index,
            0,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        )


def test_session_planner_message_carries_receiver_and_publisher_identity():
    message = build_planner_message(
        7,
        0,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        receiver_epoch=RECEIVER_EPOCH,
        publisher_session=PUBLISHER_SESSION,
    )
    header_start = len(b"planner")
    header = json.loads(message[header_start : header_start + HEADER_SIZE].rstrip(b"\x00"))
    payload = message[header_start + HEADER_SIZE :]

    assert header["v"] == 2
    assert header["fields"][:3] == [
        {"name": "receiver_epoch", "dtype": "u8", "shape": [16]},
        {"name": "publisher_session", "dtype": "u8", "shape": [16]},
        {"name": "frame_index", "dtype": "i64", "shape": [1]},
    ]
    assert payload[:16] == RECEIVER_EPOCH
    assert payload[16:32] == PUBLISHER_SESSION
    assert struct.unpack_from("<q", payload, 32)[0] == 7


def test_session_command_claim_has_strict_envelope():
    message = build_command_message(
        start=False,
        stop=False,
        planner=True,
        receiver_epoch=RECEIVER_EPOCH,
        publisher_session=PUBLISHER_SESSION,
        command_index=4,
        claim=True,
    )
    header_start = len(b"command")
    header = json.loads(message[header_start : header_start + HEADER_SIZE].rstrip(b"\x00"))
    payload = message[header_start + HEADER_SIZE :]

    assert header["v"] == 2
    assert [field["name"] for field in header["fields"]] == [
        "receiver_epoch",
        "publisher_session",
        "command_index",
        "claim",
        "start",
        "stop",
        "planner",
    ]
    assert payload[:16] == RECEIVER_EPOCH
    assert payload[16:32] == PUBLISHER_SESSION
    assert struct.unpack_from("<q", payload, 32)[0] == 4
    assert payload[40:] == bytes([1, 0, 0, 1])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"receiver_epoch": b"\x01" * 15}, "supplied together"),
        (
            {
                "receiver_epoch": b"\x00" * 16,
                "publisher_session": PUBLISHER_SESSION,
                "command_index": 0,
            },
            "non-zero",
        ),
        (
            {
                "receiver_epoch": RECEIVER_EPOCH,
                "publisher_session": PUBLISHER_SESSION,
                "command_index": -1,
            },
            "command_index",
        ),
        (
            {
                "receiver_epoch": RECEIVER_EPOCH,
                "publisher_session": PUBLISHER_SESSION,
                "command_index": 0,
            },
            "exactly one action",
        ),
    ],
)
def test_session_command_rejects_invalid_envelope(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_command_message(
            start=False,
            stop=False,
            planner=True,
            **kwargs,
        )


def test_session_command_rejects_unsupported_delta_heading():
    with pytest.raises(ValueError, match="delta_heading"):
        build_command_message(
            start=True,
            stop=False,
            planner=True,
            delta_heading=0.1,
            receiver_epoch=RECEIVER_EPOCH,
            publisher_session=PUBLISHER_SESSION,
            command_index=0,
        )


def test_pose_packer_preserves_session_tokens_as_u8():
    message = pack_pose_message(
        {
            "receiver_epoch": np.frombuffer(RECEIVER_EPOCH, dtype=np.uint8),
            "publisher_session": np.frombuffer(PUBLISHER_SESSION, dtype=np.uint8),
            "frame_index": np.array([0, 1], dtype=np.int64),
        },
        version=3,
    )
    header = json.loads(message[4 : 4 + HEADER_SIZE].rstrip(b"\x00"))
    payload = message[4 + HEADER_SIZE :]

    assert header["fields"][:2] == [
        {"name": "receiver_epoch", "dtype": "u8", "shape": [16]},
        {"name": "publisher_session", "dtype": "u8", "shape": [16]},
    ]
    assert payload[:16] == RECEIVER_EPOCH
    assert payload[16:32] == PUBLISHER_SESSION
