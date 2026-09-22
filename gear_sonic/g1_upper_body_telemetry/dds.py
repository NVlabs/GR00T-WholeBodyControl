"""Thread-safe acquisition of Unitree G1 LowState and LowCmd."""

from __future__ import annotations

import threading
import time

import numpy as np

from .protocol import JOINT_INDICES

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
except ImportError as exc:  # pragma: no cover - robot-only dependency
    ChannelFactoryInitialize = ChannelSubscriber = LowCmd_ = LowState_ = None
    _SDK_IMPORT_ERROR = exc
else:
    _SDK_IMPORT_ERROR = None


def _finite(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(JOINT_INDICES),):
        raise ValueError(f"{name} has invalid shape {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


class G1DDSReader:
    def __init__(
        self,
        *,
        domain_id: int = 0,
        network_interface: str | None = None,
        state_topic: str = "rt/lowstate",
        command_topic: str = "rt/lowcmd",
    ) -> None:
        if _SDK_IMPORT_ERROR is not None:
            raise ImportError("unitree_sdk2py with unitree_hg messages is required") from _SDK_IMPORT_ERROR
        if network_interface:
            ChannelFactoryInitialize(domain_id, networkInterface=network_interface)
        else:
            ChannelFactoryInitialize(domain_id)

        self._lock = threading.Lock()
        self._state = None
        self._command = None
        self._state_time_ns = None
        self._command_time_ns = None
        self._last_error = None
        self._subscribers = (
            ChannelSubscriber(state_topic, LowState_),
            ChannelSubscriber(command_topic, LowCmd_),
        )
        self._subscribers[0].Init(self._state_callback, 10)
        self._subscribers[1].Init(self._command_callback, 10)
        print(f"[G1Telemetry] DDS state: {state_topic}")
        print(f"[G1Telemetry] DDS command: {command_topic}")

    @staticmethod
    def state_values(msg) -> dict[str, np.ndarray]:
        motors = msg.motor_state
        if len(motors) <= max(JOINT_INDICES):
            raise ValueError(f"LowState has only {len(motors)} motors")
        selected = [motors[index] for index in JOINT_INDICES]
        return {
            field: _finite([getattr(motor, field) for motor in selected], f"LowState.{field}")
            for field in ("q", "dq", "ddq", "tau_est")
        }

    @staticmethod
    def command_values(msg) -> dict[str, np.ndarray]:
        motors = msg.motor_cmd
        if len(motors) <= max(JOINT_INDICES):
            raise ValueError(f"LowCmd has only {len(motors)} motors")
        selected = [motors[index] for index in JOINT_INDICES]
        return {
            field: _finite([getattr(motor, field) for motor in selected], f"LowCmd.{field}")
            for field in ("q", "dq", "tau", "kp", "kd")
        }

    def _state_callback(self, msg) -> None:
        try:
            values = self.state_values(msg)
        except Exception as exc:
            self._set_error(f"invalid LowState: {exc}")
            return
        with self._lock:
            self._state = values
            self._state_time_ns = time.monotonic_ns()
            self._last_error = None

    def _command_callback(self, msg) -> None:
        try:
            values = self.command_values(msg)
        except Exception as exc:
            self._set_error(f"invalid LowCmd: {exc}")
            return
        with self._lock:
            self._command = values
            self._command_time_ns = time.monotonic_ns()
            self._last_error = None

    def snapshot(self) -> dict | None:
        with self._lock:
            if self._state is None or self._command is None:
                return None
            return {
                "state": {key: value.copy() for key, value in self._state.items()},
                "command": {key: value.copy() for key, value in self._command.items()},
                "state_time_ns": int(self._state_time_ns),
                "command_time_ns": int(self._command_time_ns),
            }

    def status(self) -> dict:
        with self._lock:
            return {
                "state_received": self._state is not None,
                "command_received": self._command is not None,
                "last_error": self._last_error,
            }

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
        print(f"[G1Telemetry] Warning: {message}")

    def close(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.Close()
            except Exception:
                pass
