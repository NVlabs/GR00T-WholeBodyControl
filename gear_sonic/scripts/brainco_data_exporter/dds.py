"""Thread-safe BrainCo DDS command and feedback subscriber."""

import threading
import time

import numpy as np

from .constants import BRAINCO_NUM_MOTORS

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
except ImportError as exc:  # pragma: no cover - depends on the robot environment
    ChannelFactoryInitialize = None
    ChannelSubscriber = None
    MotorCmds_ = None
    MotorStates_ = None
    _BRAINCO_DDS_IMPORT_ERROR = exc
else:
    _BRAINCO_DDS_IMPORT_ERROR = None

class BraincoHandDDSSubscriber:
    """Thread-safe latest-value subscriber for BrainCo command and feedback."""

    def __init__(self, domain_id: int = 0, network_interface: str | None = None):
        if _BRAINCO_DDS_IMPORT_ERROR is not None:
            raise ImportError(
                "unitree_sdk2py with MotorCmds_/MotorStates_ is required for BrainCo recording"
            ) from _BRAINCO_DDS_IMPORT_ERROR

        ChannelFactoryInitialize(domain_id, network_interface)
        self._lock = threading.Lock()
        self._values: dict[str, np.ndarray | float | None] = {
            "left_state": None,
            "right_state": None,
            "left_tau_est": None,
            "right_tau_est": None,
            "left_command": None,
            "right_command": None,
            "left_state_time": None,
            "right_state_time": None,
            "left_tau_est_time": None,
            "right_tau_est_time": None,
            "left_command_time": None,
            "right_command_time": None,
        }

        self._subscribers = [
            ChannelSubscriber("rt/brainco/left/state", MotorStates_),
            ChannelSubscriber("rt/brainco/right/state", MotorStates_),
            ChannelSubscriber("rt/brainco/left/cmd", MotorCmds_),
            ChannelSubscriber("rt/brainco/right/cmd", MotorCmds_),
        ]
        callbacks = [
            lambda msg: self._set_state("left", msg),
            lambda msg: self._set_state("right", msg),
            lambda msg: self._set("left_command", self._command_q(msg)),
            lambda msg: self._set("right_command", self._command_q(msg)),
        ]
        for subscriber, callback in zip(self._subscribers, callbacks, strict=True):
            subscriber.Init(callback, 10)

        print("[BrainCo] DDS subscriptions initialized for left/right cmd and state")

    @staticmethod
    def _validate(values: np.ndarray, source: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size != BRAINCO_NUM_MOTORS:
            raise ValueError(
                f"{source} contains {values.size} motors; expected {BRAINCO_NUM_MOTORS}"
            )
        return np.clip(values, 0.0, 1.0)

    @classmethod
    def _state_q(cls, msg) -> np.ndarray:
        return cls._validate([motor.q for motor in msg.states], "MotorStates_")

    @staticmethod
    def _state_tau_est(msg) -> np.ndarray:
        values = np.asarray(
            [motor.tau_est for motor in msg.states], dtype=np.float32
        ).reshape(-1)
        if values.size != BRAINCO_NUM_MOTORS:
            raise ValueError(
                f"MotorStates_ contains {values.size} tau values; expected "
                f"{BRAINCO_NUM_MOTORS}"
            )
        if not np.isfinite(values).all():
            raise ValueError("MotorStates_ contains non-finite tau_est values")
        return values

    @classmethod
    def _command_q(cls, msg) -> np.ndarray:
        return cls._validate([motor.q for motor in msg.cmds], "MotorCmds_")

    def _set(self, key: str, value: np.ndarray) -> None:
        with self._lock:
            self._values[key] = value
            self._values[f"{key}_time"] = time.monotonic()

    def _set_state(self, side: str, msg) -> None:
        q = self._state_q(msg)
        tau_est = self._state_tau_est(msg)
        received_at = time.monotonic()
        with self._lock:
            self._values[f"{side}_state"] = q
            self._values[f"{side}_state_time"] = received_at
            self._values[f"{side}_tau_est"] = tau_est
            self._values[f"{side}_tau_est_time"] = received_at

    def snapshot(self) -> dict[str, np.ndarray | float | None]:
        with self._lock:
            return {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._values.items()
            }

    def wait_until_ready(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec if timeout_sec > 0 else None
        # Command topics are event-driven: an idle hand may not publish a command
        # at all.  Only feedback is therefore required before recording starts.
        required = ("left_state", "right_state")
        print("[BrainCo] Waiting for left/right DDS state messages ...")
        while True:
            snapshot = self.snapshot()
  
            missing = [key for key in required if snapshot[key] is None]
            if not missing:
                print("[BrainCo] Receiving both required DDS state streams")
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for BrainCo DDS streams: {missing}")
            time.sleep(0.05)

    def close(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.Close()
            except Exception:
                pass
