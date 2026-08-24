import msgpack
import pytest

from gear_sonic.utils.teleop.zmq import control_session as control_session_module
from gear_sonic.utils.teleop.zmq.control_session import (
    CONTROL_SESSION_TOPIC,
    ControlSessionClient,
    ControlSessionError,
    ControlSessionSnapshot,
    decode_control_session_message,
)

RECEIVER_EPOCH = bytes(range(1, 17))
PUBLISHER_SESSION = bytes(range(17, 33))


def _wire_message(*, epoch=RECEIVER_EPOCH, bound=()):
    return CONTROL_SESSION_TOPIC + msgpack.packb(
        {
            "protocol": 1,
            "receiver_epoch": list(epoch),
            "bound_publisher_session": list(bound),
        },
        use_bin_type=True,
    )


def test_decode_control_session_message_accepts_unclaimed_receiver():
    snapshot = decode_control_session_message(_wire_message())
    assert snapshot == ControlSessionSnapshot(RECEIVER_EPOCH, None)


def test_decode_control_session_message_accepts_claim_ack():
    snapshot = decode_control_session_message(_wire_message(bound=PUBLISHER_SESSION))
    assert snapshot == ControlSessionSnapshot(RECEIVER_EPOCH, PUBLISHER_SESSION)


@pytest.mark.parametrize(
    "raw",
    [
        b"wrong" + msgpack.packb({}),
        CONTROL_SESSION_TOPIC + msgpack.packb({"protocol": 2}),
        CONTROL_SESSION_TOPIC
        + msgpack.packb(
            {
                "protocol": True,
                "receiver_epoch": list(RECEIVER_EPOCH),
                "bound_publisher_session": [],
            }
        ),
        CONTROL_SESSION_TOPIC
        + msgpack.packb(
            {
                "protocol": 1,
                "receiver_epoch": list(RECEIVER_EPOCH),
                "bound_publisher_session": [],
                "extra": 1,
            }
        ),
        _wire_message(epoch=b"\x00" * 16),
        _wire_message(epoch=b"\x01" * 15),
    ],
)
def test_decode_control_session_message_rejects_malformed_state(raw):
    with pytest.raises(ControlSessionError):
        decode_control_session_message(raw)


class _Publisher:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


def _client_without_socket(snapshots):
    client = object.__new__(ControlSessionClient)
    client.publisher_session = PUBLISHER_SESSION
    client.receiver_epoch = None
    client._command_index = 0
    client._claim_acknowledged = False
    client._last_verified_feedback_time = None
    snapshots = iter(snapshots)
    client._receive_snapshot = lambda _timeout: next(snapshots, None)
    return client


def test_claim_sends_no_actuation_and_waits_for_matching_ack():
    client = _client_without_socket(
        [
            ControlSessionSnapshot(RECEIVER_EPOCH, None),
            ControlSessionSnapshot(RECEIVER_EPOCH, PUBLISHER_SESSION),
        ]
    )
    publisher = _Publisher()

    client.claim(publisher, timeout_s=0.2)

    assert len(publisher.messages) == 1
    assert publisher.messages[0].startswith(b"command")
    assert client.receiver_epoch == RECEIVER_EPOCH
    assert client._command_index == 1
    assert client._claim_acknowledged
    assert client._last_verified_feedback_time is not None


def test_claim_refuses_existing_owner_without_takeover():
    other_session = b"\xff" * 16
    client = _client_without_socket([ControlSessionSnapshot(RECEIVER_EPOCH, other_session)])

    with pytest.raises(ControlSessionError, match="already claimed"):
        client.claim(_Publisher(), timeout_s=0.2)


def test_control_command_is_blocked_before_claim_ack():
    client = _client_without_socket([])
    client.receiver_epoch = RECEIVER_EPOCH

    with pytest.raises(ControlSessionError, match="not acknowledged"):
        client.build_command(start=True, stop=False, planner=True)


def _acknowledged_client(snapshots):
    client = _client_without_socket(snapshots)
    client.receiver_epoch = RECEIVER_EPOCH
    client._claim_acknowledged = True
    client._last_verified_feedback_time = 10.0
    return client


def test_feedback_verification_refreshes_matching_heartbeat(monkeypatch):
    client = _acknowledged_client([ControlSessionSnapshot(RECEIVER_EPOCH, PUBLISHER_SESSION)])
    monkeypatch.setattr(control_session_module.time, "monotonic", lambda: 10.2)

    client.verify_feedback(timeout_s=0.5)

    assert client._last_verified_feedback_time == pytest.approx(10.2)


@pytest.mark.parametrize(
    ("snapshot", "match"),
    [
        (
            ControlSessionSnapshot(b"\xff" * 16, PUBLISHER_SESSION),
            "epoch changed",
        ),
        (
            ControlSessionSnapshot(RECEIVER_EPOCH, b"\xff" * 16),
            "different publisher",
        ),
        (
            ControlSessionSnapshot(RECEIVER_EPOCH, None),
            "lost publisher ownership",
        ),
    ],
)
def test_feedback_verification_rejects_epoch_or_owner_mismatch(monkeypatch, snapshot, match):
    client = _acknowledged_client([snapshot])
    monkeypatch.setattr(control_session_module.time, "monotonic", lambda: 10.1)

    with pytest.raises(ControlSessionError, match=match):
        client.verify_feedback(timeout_s=0.5)


def test_feedback_verification_rejects_expired_heartbeat(monkeypatch):
    client = _acknowledged_client([None])
    monkeypatch.setattr(control_session_module.time, "monotonic", lambda: 10.6)

    with pytest.raises(ControlSessionError, match="heartbeat expired"):
        client.verify_feedback(timeout_s=0.5)
