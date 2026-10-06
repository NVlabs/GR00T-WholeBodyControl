"""Small, deterministic Inspire hand plant used by the simulator and tests."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from gear_sonic.utils.hand_control.inspire.config import InspireMapping, load_mapping
from gear_sonic.utils.hand_control.inspire.contract import (
    HandStatus,
    InspireHandCommand,
    InspireHandState,
)
from gear_sonic.utils.hand_control.inspire.safety import CommandGuard, SafetyConfig
from gear_sonic.utils.mujoco_sim.inspire.layout import InspireMujocoLayout


@dataclass(frozen=True)
class SimControlConfig:
    kp: float = 2.0
    kd: float = 0.08
    effort_limit: float = 1.4

    def __post_init__(self) -> None:
        if self.kp <= 0 or self.kd < 0 or self.effort_limit <= 0:
            raise ValueError("invalid hand PD/effort configuration")


class InspireSimController:
    """Applies guarded native-6D commands to named MuJoCo actuators."""

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        mapping: InspireMapping | None = None,
        control: SimControlConfig | None = None,
        safety: SafetyConfig | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.mapping = mapping or load_mapping()
        self.layout = InspireMujocoLayout.resolve(model, self.mapping)
        self.control = control or SimControlConfig()
        self.guard = CommandGuard(self.mapping, safety)
        self._state_sequence = 0

    def arm(self) -> None:
        self.guard.arm()

    def submit(self, command: InspireHandCommand, *, now_ns: int | None = None) -> None:
        self.guard.accept(command, now_ns=time.monotonic_ns() if now_ns is None else now_ns)

    def step(self, *, now_ns: int | None = None) -> InspireHandState:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        status = self.guard.poll(now_ns=now_ns)
        left_q, right_q, left_dq, right_dq = self.layout.read(self.data)
        if status is HandStatus.ARMED and self.guard.last_command is not None:
            command = self.guard.last_command
            self._apply_pd(self.layout.left, command.left_q, left_q, left_dq)
            self._apply_pd(self.layout.right, command.right_q, right_q, right_dq)
        else:
            self.data.ctrl[self.layout.left.actuator_ids] = 0.0
            self.data.ctrl[self.layout.right.actuator_ids] = 0.0

        self._state_sequence += 1
        source_ns = self.guard.last_command.source_monotonic_ns if self.guard.last_command else 0
        return InspireHandState(
            sequence=self._state_sequence,
            source_monotonic_ns=source_ns,
            gateway_monotonic_ns=now_ns,
            status=status,
            left_q=left_q,
            right_q=right_q,
            left_dq=left_dq,
            right_dq=right_dq,
        )

    def _apply_pd(self, layout, target, q, dq) -> None:
        torque = self.control.kp * (target - q) - self.control.kd * dq
        self.data.ctrl[layout.actuator_ids] = np.clip(
            torque, -self.control.effort_limit, self.control.effort_limit
        )
