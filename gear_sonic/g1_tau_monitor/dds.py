"""DDS acquisition and torque-residual calculation for Unitree G1."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

import numpy as np

from .joints import UPPER_BODY_INDICES

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
except ImportError as exc:  # pragma: no cover - depends on robot environment
    ChannelFactoryInitialize = None
    ChannelSubscriber = None
    LowCmd_ = None
    LowState_ = None
    _SDK_IMPORT_ERROR = exc
else:
    _SDK_IMPORT_ERROR = None


@dataclass(frozen=True)
class TorqueSample:
    timestamp: float
    tau_est: np.ndarray
    tau_cmd: np.ndarray
    residual: np.ndarray
    state_age: float
    command_age: float
    q: Optional[np.ndarray] = None
    q_cmd: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SubscriberStatus:
    state_received: bool
    command_received: bool
    state_age: Optional[float]
    command_age: Optional[float]
    last_error: Optional[str]


class G1TorqueSubscriber:
    """Subscribe to LowState/LowCmd and expose synchronized upper-body samples."""

    def __init__(
        self,
        *,
        domain_id: int,
        network_interface: Optional[str],
        state_topic: str,
        command_topic: str,
        include_pd: bool,
        show_q: bool,
        stale_seconds: float,
    ):
        if _SDK_IMPORT_ERROR is not None:
            raise ImportError(
                "unitree_sdk2py with unitree_hg LowState_/LowCmd_ is required"
            ) from _SDK_IMPORT_ERROR
        if network_interface:
            ChannelFactoryInitialize(domain_id, networkInterface=network_interface)
        else:
            ChannelFactoryInitialize(domain_id)

        self.include_pd = bool(include_pd)
        self.show_q = bool(show_q)
        self._read_position = self.include_pd or self.show_q
        self.stale_seconds = float(stale_seconds)
        self._lock = threading.Lock()
        self._state: Optional[dict[str, np.ndarray]] = None
        self._command: Optional[dict[str, np.ndarray]] = None
        self._state_time: Optional[float] = None
        self._command_time: Optional[float] = None
        self._version = 0
        self._last_read_version = -1
        self._last_error: Optional[str] = None
        self._last_warning_time = 0.0

        self._subscribers = [
            ChannelSubscriber(state_topic, LowState_),
            ChannelSubscriber(command_topic, LowCmd_),
        ]
        self._subscribers[0].Init(self._state_callback, 10)
        self._subscribers[1].Init(self._command_callback, 10)
        print(f"[G1TauMonitor] DDS state:   {state_topic}")
        print(f"[G1TauMonitor] DDS command: {command_topic}")
        print(f"[G1TauMonitor] upper-body motor indices: {UPPER_BODY_INDICES}")

    @staticmethod
    def _finite_array(values, field_name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (len(UPPER_BODY_INDICES),):
            raise ValueError(
                f"{field_name} has shape {result.shape}; "
                f"expected ({len(UPPER_BODY_INDICES)},)"
            )
        if not np.isfinite(result).all():
            raise ValueError(f"{field_name} contains non-finite values")
        return result

    @staticmethod
    def state_values(msg, include_pd: bool = False) -> dict[str, np.ndarray]:
        motors = msg.motor_state
        if len(motors) <= max(UPPER_BODY_INDICES):
            raise ValueError(
                f"LowState has {len(motors)} motor states; "
                f"index {max(UPPER_BODY_INDICES)} is required"
            )
        selected = [motors[index] for index in UPPER_BODY_INDICES]
        values = {
            "tau_est": G1TorqueSubscriber._finite_array(
                [motor.tau_est for motor in selected], "LowState.tau_est"
            ),
        }
        if include_pd:
            values.update(
                {
                    "q": G1TorqueSubscriber._finite_array(
                        [motor.q for motor in selected], "LowState.q"
                    ),
                    "dq": G1TorqueSubscriber._finite_array(
                        [motor.dq for motor in selected], "LowState.dq"
                    ),
                }
            )
        return values

    @staticmethod
    def command_values(msg, include_pd: bool = False) -> dict[str, np.ndarray]:
        motors = msg.motor_cmd
        if len(motors) <= max(UPPER_BODY_INDICES):
            raise ValueError(
                f"LowCmd has {len(motors)} motor commands; "
                f"index {max(UPPER_BODY_INDICES)} is required"
            )
        selected = [motors[index] for index in UPPER_BODY_INDICES]
        values = {
            "tau": G1TorqueSubscriber._finite_array(
                [motor.tau for motor in selected], "LowCmd.tau"
            )
        }
        if include_pd:
            for field in ("q", "dq", "kp", "kd"):
                values[field] = G1TorqueSubscriber._finite_array(
                    [getattr(motor, field) for motor in selected],
                    f"LowCmd.{field}",
                )
        return values

    @staticmethod
    def calculate_command_torque(
        state: dict[str, np.ndarray],
        command: dict[str, np.ndarray],
        include_pd: bool,
    ) -> np.ndarray:
        tau_cmd = command["tau"].copy()
        if include_pd:
            tau_cmd += command["kp"] * (command["q"] - state["q"])
            tau_cmd += command["kd"] * (command["dq"] - state["dq"])
        return tau_cmd

    def _state_callback(self, msg) -> None:
        try:
            values = self.state_values(msg, self._read_position)
        except Exception as exc:
            self._set_error(f"invalid LowState: {exc}")
            return
        with self._lock:
            self._state = values
            self._state_time = time.monotonic()
            self._version += 1
            self._last_error = None

    def _command_callback(self, msg) -> None:
        try:
            values = self.command_values(msg, self._read_position)
        except Exception as exc:
            self._set_error(f"invalid LowCmd: {exc}")
            return
        with self._lock:
            self._command = values
            self._command_time = time.monotonic()
            self._version += 1
            self._last_error = None

    def read(self) -> Optional[TorqueSample]:
        now = time.monotonic()
        with self._lock:
            if self._version == self._last_read_version:
                return None
            if self._state is None or self._command is None:
                return None
            state_age = now - float(self._state_time)
            command_age = now - float(self._command_time)
            if state_age > self.stale_seconds or command_age > self.stale_seconds:
                return None
            state = {key: value.copy() for key, value in self._state.items()}
            command = {key: value.copy() for key, value in self._command.items()}
            self._last_read_version = self._version

        tau_est = state["tau_est"]
        tau_cmd = self.calculate_command_torque(state, command, self.include_pd)
        return TorqueSample(
            timestamp=now,
            tau_est=tau_est,
            tau_cmd=tau_cmd,
            residual=tau_est - tau_cmd,
            state_age=state_age,
            command_age=command_age,
            q=state.get("q"),
            q_cmd=command.get("q"),
        )

    def status(self) -> SubscriberStatus:
        now = time.monotonic()
        with self._lock:
            return SubscriberStatus(
                state_received=self._state is not None,
                command_received=self._command is not None,
                state_age=(
                    None if self._state_time is None else now - self._state_time
                ),
                command_age=(
                    None if self._command_time is None else now - self._command_time
                ),
                last_error=self._last_error,
            )

    def _set_error(self, message: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_error = message
        if now - self._last_warning_time >= 1.0:
            self._last_warning_time = now
            print(f"[G1TauMonitor] Warning: {message}")

    def close(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.Close()
            except Exception:
                pass
