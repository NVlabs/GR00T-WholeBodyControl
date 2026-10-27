"""Versioned, device-native contracts for an Inspire RH56DFX hand pair."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

HAND_DOF = 6
JOINT_ORDER = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotation",
)
COMMAND_SCHEMA = "inspire.hand.command.v2"
STATE_SCHEMA = "inspire.hand.state.v2"

# Device coordinates in JOINT_ORDER; positions are radians.
# These bounds describe the conversion domain, not preset open/close commands.
CANONICAL_LOWER_RAD = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
CANONICAL_UPPER_RAD = np.asarray([1.7, 1.7, 1.7, 1.7, 0.6, 1.3], dtype=np.float32)
CANONICAL_VELOCITY_LIMIT_RAD_S = np.asarray([2.0, 2.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32)
# The serial bridge itself reports an open fraction, opposite to canonical
# joint closure.  This is the only representation boundary in the system.
HARDWARE_NORMALIZED_CLOSED = 0.0
HARDWARE_NORMALIZED_OPEN = 1.0


class ContractError(ValueError):
    """Raised when data violates the Inspire wire/domain contract."""


class HandStatus(str, Enum):
    DISARMED = "DISARMED"
    ARMED = "ARMED"
    FAULT = "FAULT"
    STALE = "STALE"


def hand_vector(value: Any, *, field_name: str) -> np.ndarray:
    """Return a finite, owned float32 vector with the canonical six values."""
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (1, HAND_DOF):
        array = array[0]
    if array.shape != (HAND_DOF,):
        raise ContractError(f"{field_name} must have shape ({HAND_DOF},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{field_name} contains NaN or infinity")
    return np.array(array, dtype=np.float32, copy=True)


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{field_name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field_name} must be a non-negative integer") from exc
    if result < 0 or result != value:
        raise ContractError(f"{field_name} must be a non-negative integer")
    return result


@dataclass(frozen=True)
class InspireHandCommand:
    """One synchronized, native 6D command for both Inspire hands."""

    sequence: int
    source_monotonic_ns: int
    left_q: np.ndarray
    right_q: np.ndarray
    frame_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequence", _non_negative_int(self.sequence, field_name="sequence")
        )
        object.__setattr__(
            self,
            "source_monotonic_ns",
            _non_negative_int(self.source_monotonic_ns, field_name="source_monotonic_ns"),
        )
        object.__setattr__(self, "left_q", hand_vector(self.left_q, field_name="left_q"))
        object.__setattr__(self, "right_q", hand_vector(self.right_q, field_name="right_q"))
        if self.frame_index is not None:
            object.__setattr__(
                self,
                "frame_index",
                _non_negative_int(self.frame_index, field_name="frame_index"),
            )

    def to_wire_dict(self) -> dict[str, Any]:
        payload = {
            "schema": COMMAND_SCHEMA,
            "sequence": self.sequence,
            "source_monotonic_ns": self.source_monotonic_ns,
            "joint_order": list(JOINT_ORDER),
            "left_q": self.left_q.tolist(),
            "right_q": self.right_q.tolist(),
        }
        if self.frame_index is not None:
            payload["frame_index"] = self.frame_index
        return payload

    @classmethod
    def from_wire_dict(cls, payload: Mapping[str, Any]) -> "InspireHandCommand":
        _validate_envelope(payload, COMMAND_SCHEMA)
        return cls(
            sequence=payload.get("sequence"),
            source_monotonic_ns=payload.get("source_monotonic_ns"),
            left_q=payload.get("left_q"),
            right_q=payload.get("right_q"),
            frame_index=payload.get("frame_index"),
        )


@dataclass(frozen=True)
class InspireHandState:
    """Measured state returned by the gateway for both Inspire hands."""

    sequence: int
    source_monotonic_ns: int
    gateway_monotonic_ns: int
    status: HandStatus
    left_q: np.ndarray
    right_q: np.ndarray
    left_dq: np.ndarray | None = None
    right_dq: np.ndarray | None = None
    accepted_command_sequence: int | None = None
    accepted_frame_index: int | None = None
    left_command_q: np.ndarray | None = None
    right_command_q: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequence", _non_negative_int(self.sequence, field_name="sequence")
        )
        object.__setattr__(
            self,
            "source_monotonic_ns",
            _non_negative_int(self.source_monotonic_ns, field_name="source_monotonic_ns"),
        )
        object.__setattr__(
            self,
            "gateway_monotonic_ns",
            _non_negative_int(self.gateway_monotonic_ns, field_name="gateway_monotonic_ns"),
        )
        try:
            object.__setattr__(self, "status", HandStatus(self.status))
        except ValueError as exc:
            raise ContractError(f"unknown hand status: {self.status!r}") from exc
        object.__setattr__(self, "left_q", hand_vector(self.left_q, field_name="left_q"))
        object.__setattr__(self, "right_q", hand_vector(self.right_q, field_name="right_q"))
        if (self.left_dq is None) != (self.right_dq is None):
            raise ContractError(
                "left_dq and right_dq must either both be present or both be absent"
            )
        if self.left_dq is not None:
            object.__setattr__(self, "left_dq", hand_vector(self.left_dq, field_name="left_dq"))
            object.__setattr__(self, "right_dq", hand_vector(self.right_dq, field_name="right_dq"))
        if (self.left_command_q is None) != (self.right_command_q is None):
            raise ContractError(
                "left_command_q and right_command_q must either both be present or both be absent"
            )
        if self.accepted_command_sequence is not None and self.left_command_q is None:
            raise ContractError(
                "accepted_command_sequence and commanded positions must be present together"
            )
        if self.accepted_frame_index is not None and self.accepted_command_sequence is None:
            raise ContractError("accepted_frame_index requires accepted_command_sequence")
        if self.left_command_q is not None:
            if self.accepted_command_sequence is None:
                raise ContractError(
                    "accepted_command_sequence is required with commanded positions"
                )
            object.__setattr__(
                self,
                "accepted_command_sequence",
                _non_negative_int(
                    self.accepted_command_sequence, field_name="accepted_command_sequence"
                ),
            )
            object.__setattr__(
                self,
                "left_command_q",
                hand_vector(self.left_command_q, field_name="left_command_q"),
            )
            object.__setattr__(
                self,
                "right_command_q",
                hand_vector(self.right_command_q, field_name="right_command_q"),
            )
        if self.accepted_frame_index is not None:
            object.__setattr__(
                self,
                "accepted_frame_index",
                _non_negative_int(self.accepted_frame_index, field_name="accepted_frame_index"),
            )

    def to_wire_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "sequence": self.sequence,
            "source_monotonic_ns": self.source_monotonic_ns,
            "gateway_monotonic_ns": self.gateway_monotonic_ns,
            "status": self.status.value,
            "joint_order": list(JOINT_ORDER),
            "left_q": self.left_q.tolist(),
            "right_q": self.right_q.tolist(),
        }
        if self.left_dq is not None:
            payload["left_dq"] = self.left_dq.tolist()
            payload["right_dq"] = self.right_dq.tolist()
        if self.left_command_q is not None:
            payload["accepted_command_sequence"] = self.accepted_command_sequence
            payload["left_command_q"] = self.left_command_q.tolist()
            payload["right_command_q"] = self.right_command_q.tolist()
            if self.accepted_frame_index is not None:
                payload["accepted_frame_index"] = self.accepted_frame_index
        return payload

    @classmethod
    def from_wire_dict(cls, payload: Mapping[str, Any]) -> "InspireHandState":
        _validate_envelope(payload, STATE_SCHEMA)
        return cls(
            sequence=payload.get("sequence"),
            source_monotonic_ns=payload.get("source_monotonic_ns"),
            gateway_monotonic_ns=payload.get("gateway_monotonic_ns"),
            status=payload.get("status"),
            left_q=payload.get("left_q"),
            right_q=payload.get("right_q"),
            left_dq=payload.get("left_dq"),
            right_dq=payload.get("right_dq"),
            accepted_command_sequence=payload.get("accepted_command_sequence"),
            accepted_frame_index=payload.get("accepted_frame_index"),
            left_command_q=payload.get("left_command_q"),
            right_command_q=payload.get("right_command_q"),
        )


def _validate_envelope(payload: Mapping[str, Any], expected_schema: str) -> None:
    if not isinstance(payload, Mapping):
        raise ContractError("wire payload must be a mapping")
    if payload.get("schema") != expected_schema:
        raise ContractError(f"expected schema {expected_schema!r}, got {payload.get('schema')!r}")
    if tuple(payload.get("joint_order", ())) != JOINT_ORDER:
        raise ContractError(
            f"joint_order must be {list(JOINT_ORDER)}, got {payload.get('joint_order')!r}"
        )
