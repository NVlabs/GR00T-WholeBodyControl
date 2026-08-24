"""Fail-closed ownership handshake for the ZMQ teleoperation manager."""

from __future__ import annotations

from dataclasses import dataclass
import math
import secrets
import time

import msgpack
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    CONTROL_SESSION_TOKEN_SIZE,
    build_command_message,
)

CONTROL_SESSION_PROTOCOL = 1
CONTROL_SESSION_TOPIC = b"control_session"
CONTROL_SESSION_HEARTBEAT_TIMEOUT_S = 0.5


class ControlSessionError(RuntimeError):
    """Raised when command ownership cannot be established safely."""


def _decode_token(value, name: str, *, optional: bool = False) -> bytes | None:
    if optional and (value is None or value == [] or value == b""):
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        token = bytes(value)
    elif isinstance(value, (list, tuple)):
        if len(value) != CONTROL_SESSION_TOKEN_SIZE or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > 255 for item in value
        ):
            raise ControlSessionError(f"{name} is not a 16-byte token")
        token = bytes(value)
    else:
        raise ControlSessionError(f"{name} is not a 16-byte token")
    if len(token) != CONTROL_SESSION_TOKEN_SIZE or not any(token):
        raise ControlSessionError(f"{name} is not a non-zero 16-byte token")
    return token


@dataclass(frozen=True)
class ControlSessionSnapshot:
    receiver_epoch: bytes
    bound_publisher_session: bytes | None


def decode_control_session_message(raw: bytes) -> ControlSessionSnapshot:
    """Decode one ``control_session`` topic message."""
    if not isinstance(raw, bytes) or not raw.startswith(CONTROL_SESSION_TOPIC):
        raise ControlSessionError("invalid control_session topic")
    try:
        payload = msgpack.unpackb(raw[len(CONTROL_SESSION_TOPIC) :], raw=False)
    except Exception as exc:
        raise ControlSessionError(f"invalid control_session payload: {exc}") from exc
    expected_keys = {
        "protocol",
        "receiver_epoch",
        "bound_publisher_session",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ControlSessionError("invalid control_session schema")
    protocol = payload["protocol"]
    if type(protocol) is not int or protocol != CONTROL_SESSION_PROTOCOL:
        raise ControlSessionError("unsupported control_session protocol")
    epoch = _decode_token(payload.get("receiver_epoch"), "receiver_epoch")
    bound = _decode_token(
        payload.get("bound_publisher_session"),
        "bound_publisher_session",
        optional=True,
    )
    return ControlSessionSnapshot(epoch, bound)


class ControlSessionClient:
    """Claim one native deployment process before publishing control data."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        context: zmq.Context | None = None,
        publisher_session: bytes | None = None,
    ):
        self._owns_context = context is None
        self._context = context or zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, CONTROL_SESSION_TOPIC)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self.publisher_session = (
            secrets.token_bytes(CONTROL_SESSION_TOKEN_SIZE) if publisher_session is None else publisher_session
        )
        self.publisher_session = _decode_token(self.publisher_session, "publisher_session")
        self.receiver_epoch: bytes | None = None
        self._command_index = 0
        self._claim_acknowledged = False
        self._last_verified_feedback_time: float | None = None

    def close(self) -> None:
        self._socket.close(linger=0)
        if self._owns_context:
            self._context.term()

    def _receive_snapshot(self, timeout_ms: int) -> ControlSessionSnapshot | None:
        if not self._socket.poll(timeout=max(0, timeout_ms), flags=zmq.POLLIN):
            return None
        return decode_control_session_message(self._socket.recv())

    def next_command_index(self) -> int:
        value = self._command_index
        self._command_index += 1
        return value

    def build_command(self, *, start: bool, stop: bool, planner: bool, claim: bool = False) -> bytes:
        if self.receiver_epoch is None:
            raise ControlSessionError("receiver epoch is unavailable")
        if not claim and not self._claim_acknowledged:
            raise ControlSessionError("publisher claim is not acknowledged")
        return build_command_message(
            start=start,
            stop=stop,
            planner=planner,
            receiver_epoch=self.receiver_epoch,
            publisher_session=self.publisher_session,
            command_index=self.next_command_index(),
            claim=claim,
        )

    def _verify_claimed_snapshot(self, snapshot: ControlSessionSnapshot) -> None:
        if self.receiver_epoch is None or not self._claim_acknowledged:
            raise ControlSessionError("publisher claim is not acknowledged")
        if snapshot.receiver_epoch != self.receiver_epoch:
            raise ControlSessionError("native receiver epoch changed; deployment restarted")
        if snapshot.bound_publisher_session != self.publisher_session:
            if snapshot.bound_publisher_session is None:
                raise ControlSessionError("native deployment lost publisher ownership")
            raise ControlSessionError("native deployment is bound to a different publisher")
        self._last_verified_feedback_time = time.monotonic()

    def verify_feedback(
        self,
        *,
        timeout_s: float = CONTROL_SESSION_HEARTBEAT_TIMEOUT_S,
    ) -> None:
        """Require a continuing heartbeat from the claimed native receiver."""
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive and finite")
        if not self._claim_acknowledged or self._last_verified_feedback_time is None:
            raise ControlSessionError("publisher claim is not acknowledged")

        snapshot = self._receive_snapshot(0)
        if snapshot is not None:
            self._verify_claimed_snapshot(snapshot)

        if time.monotonic() - self._last_verified_feedback_time > timeout_s:
            raise ControlSessionError("native control_session heartbeat expired; stopping manager")

    def claim(self, publisher_socket, *, timeout_s: float = 5.0) -> None:
        """Wait for receiver epoch, claim it, and require native acknowledgement."""
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive and finite")
        deadline = time.monotonic() + timeout_s
        claim_message: bytes | None = None
        next_send = 0.0
        saw_receiver = False

        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            snapshot = self._receive_snapshot(min(50, remaining_ms))
            if snapshot is not None:
                saw_receiver = True
                if snapshot.receiver_epoch != self.receiver_epoch:
                    self.receiver_epoch = snapshot.receiver_epoch
                    claim_message = None
                    self._claim_acknowledged = False
                    self._last_verified_feedback_time = None
                if snapshot.bound_publisher_session is not None:
                    if snapshot.bound_publisher_session == self.publisher_session:
                        self._claim_acknowledged = True
                        self._last_verified_feedback_time = time.monotonic()
                        return
                    raise ControlSessionError(
                        "native deployment is already claimed by another publisher; "
                        "stop that owner and restart deployment"
                    )

            now = time.monotonic()
            if self.receiver_epoch is not None and now >= next_send:
                if claim_message is None:
                    claim_message = self.build_command(start=False, stop=False, planner=True, claim=True)
                publisher_socket.send(claim_message)
                next_send = now + 0.1

        if not saw_receiver:
            raise ControlSessionError(
                "native control_session challenge not received; start zmq_manager deployment with ZMQ output first"
            )
        raise ControlSessionError("native deployment did not acknowledge publisher claim")
