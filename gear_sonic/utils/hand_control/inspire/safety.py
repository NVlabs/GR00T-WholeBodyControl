"""Fail-closed command validation shared by Inspire runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gear_sonic.utils.hand_control.inspire.config import InspireMapping
from gear_sonic.utils.hand_control.inspire.contract import (
    ContractError,
    HandStatus,
    InspireHandCommand,
)


@dataclass(frozen=True)
class SafetyConfig:
    command_timeout_ns: int = 200_000_000
    max_future_skew_ns: int = 20_000_000

    def __post_init__(self) -> None:
        if self.command_timeout_ns <= 0:
            raise ValueError("command_timeout_ns must be positive")
        if self.max_future_skew_ns < 0:
            raise ValueError("max_future_skew_ns cannot be negative")


class CommandGuard:
    """Validates sequence, freshness, limits, and slew before accepting a command.

    The guard never silently clips. A bad command changes the state to ``FAULT``;
    an absent/expired command changes it to ``STALE``. Rearming is always explicit.
    """

    def __init__(self, mapping: InspireMapping, config: SafetyConfig | None = None) -> None:
        self.mapping = mapping
        self.config = config or SafetyConfig()
        self.status = HandStatus.DISARMED
        self.last_command: InspireHandCommand | None = None
        self.last_error: str | None = None

    def arm(self) -> None:
        if self.status is HandStatus.FAULT:
            raise ContractError("clear the fault before arming")
        self.status = HandStatus.ARMED
        self.last_error = None

    def disarm(self) -> None:
        self.status = HandStatus.DISARMED
        # Keep replay/slew history across a temporary transport interruption.
        # Only an explicit fault clear starts a new command epoch.

    def clear_fault(self) -> None:
        self.status = HandStatus.DISARMED
        self.last_command = None
        self.last_error = None

    def accept(self, command: InspireHandCommand, *, now_ns: int) -> None:
        if self.status is not HandStatus.ARMED:
            raise ContractError(f"gateway is not armed (status={self.status.value})")
        try:
            self._validate(command, now_ns=now_ns)
        except ContractError as exc:
            self.status = HandStatus.FAULT
            self.last_error = str(exc)
            raise
        self.last_command = command

    def poll(self, *, now_ns: int) -> HandStatus:
        if self.status is HandStatus.ARMED and self.last_command is not None:
            age = now_ns - self.last_command.source_monotonic_ns
            if age > self.config.command_timeout_ns:
                self.status = HandStatus.STALE
                self.last_error = f"command is stale by {age} ns"
        return self.status

    def _validate(self, command: InspireHandCommand, *, now_ns: int) -> None:
        age = now_ns - command.source_monotonic_ns
        if age > self.config.command_timeout_ns:
            raise ContractError(f"command age {age} ns exceeds timeout")
        if age < -self.config.max_future_skew_ns:
            raise ContractError(f"command timestamp is {-age} ns in the future")
        if self.last_command is not None and command.sequence <= self.last_command.sequence:
            raise ContractError(
                f"sequence must increase: previous={self.last_command.sequence}, got={command.sequence}"
            )

        self._validate_side(command.left_q, self.mapping.left, "left_q")
        self._validate_side(command.right_q, self.mapping.right, "right_q")
        if self.last_command is not None:
            elapsed_s = (command.source_monotonic_ns - self.last_command.source_monotonic_ns) / 1e9
            if elapsed_s <= 0:
                raise ContractError("command timestamp must increase")
            self._validate_slew(
                command.left_q, self.last_command.left_q, self.mapping.left, elapsed_s, "left_q"
            )
            self._validate_slew(
                command.right_q, self.last_command.right_q, self.mapping.right, elapsed_s, "right_q"
            )

    @staticmethod
    def _validate_side(values, joints, field_name: str) -> None:
        lower = np.asarray([joint.lower for joint in joints], dtype=np.float32)
        upper = np.asarray([joint.upper for joint in joints], dtype=np.float32)
        invalid = np.flatnonzero((values < lower) | (values > upper))
        if invalid.size:
            index = int(invalid[0])
            raise ContractError(
                f"{field_name}[{index}]={values[index]:.6g} outside [{lower[index]:.6g}, {upper[index]:.6g}]"
            )

    @staticmethod
    def _validate_slew(values, previous, joints, elapsed_s: float, field_name: str) -> None:
        velocity = np.abs(values - previous) / elapsed_s
        limit = np.asarray([joint.velocity_limit for joint in joints], dtype=np.float32)
        invalid = np.flatnonzero(velocity > limit)
        if invalid.size:
            index = int(invalid[0])
            raise ContractError(
                f"{field_name}[{index}] slew {velocity[index]:.6g} rad/s exceeds {limit[index]:.6g} rad/s"
            )
