"""Fail-safe helpers for teleoperation input managers."""

from enum import Enum
import math
import time
from typing import Any, Callable

from gear_sonic.utils.teleop.input_readers import get_fresh_body_sample


class InputWatchdogAction(Enum):
    CONTINUE = "continue"
    WAIT = "wait"
    STOP = "stop"


def evaluate_body_input(
    reader: Any,
    *,
    active: bool,
    timeout_s: float,
    monotonic_now: float | None = None,
) -> tuple[InputWatchdogAction, dict[str, Any] | None, str]:
    """Map tracking validity to a fail-closed manager action."""
    # Raw Motion and Full-body are mutually exclusive on PICO. Real control
    # instead requires the hardened Full-body sideband: authoritative
    # BT_VALID state, calibration, successful API calls, and at least two
    # distinct connected trackers in the same packet as the Body frame.
    sample, reason = get_fresh_body_sample(
        reader,
        max_age_s=timeout_s,
        monotonic_now=monotonic_now,
        require_controllers=True,
        require_body_tracker_health=True,
        require_controller_tracking_health=True,
    )
    if sample is not None:
        return InputWatchdogAction.CONTINUE, sample, ""
    action = InputWatchdogAction.STOP if active else InputWatchdogAction.WAIT
    return action, None, reason


def send_stop_burst(
    socket: Any,
    stop_message: bytes,
    *,
    count: int = 10,
    interval_s: float = 0.02,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Send a stop immediately, then repeat it across a short PUB/SUB window."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if not math.isfinite(interval_s) or interval_s < 0.0:
        raise ValueError("interval_s must be a non-negative finite value")

    for index in range(count):
        socket.send(stop_message)
        if index + 1 < count and interval_s:
            sleep_fn(interval_s)
