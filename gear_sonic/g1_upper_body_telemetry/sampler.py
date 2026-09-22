"""Build fixed-rate, derived upper-body telemetry samples."""

from __future__ import annotations

import time

import numpy as np

from .protocol import SCHEMA_VERSION


class TelemetrySampler:
    def __init__(self, publish_hz: float) -> None:
        if publish_hz <= 0:
            raise ValueError("publish_hz must be positive")
        self.publish_hz = float(publish_hz)
        self._previous_dq_cmd = None
        self._previous_time_ns = None
        self._sequence = 0

    def build(self, snapshot: dict, now_ns: int | None = None) -> dict:
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        state = snapshot["state"]
        command = snapshot["command"]

        ddq_valid = self._previous_dq_cmd is not None and now_ns > self._previous_time_ns
        if ddq_valid:
            dt = (now_ns - self._previous_time_ns) * 1e-9
            ddq_cmd = (command["dq"] - self._previous_dq_cmd) / dt
        else:
            ddq_cmd = np.zeros_like(command["dq"], dtype=np.float32)
        ddq_cmd = np.asarray(ddq_cmd, dtype=np.float32)
        self._previous_dq_cmd = command["dq"].copy()
        self._previous_time_ns = now_ns

        tau_cmd_ff = command["tau"]
        tau_cmd_pd = (
            tau_cmd_ff
            + command["kp"] * (command["q"] - state["q"])
            + command["kd"] * (command["dq"] - state["dq"])
        ).astype(np.float32)
        sample = {
            "schema_version": SCHEMA_VERSION,
            "sequence_id": self._sequence,
            "source_timestamp_ns": time.time_ns(),
            "source_monotonic_ns": now_ns,
            "state_age_sec": max(0.0, (now_ns - snapshot["state_time_ns"]) * 1e-9),
            "command_age_sec": max(0.0, (now_ns - snapshot["command_time_ns"]) * 1e-9),
            "publish_hz": self.publish_hz,
            "ddq_cmd_valid": ddq_valid,
            "q_cmd": command["q"],
            "dq_cmd": command["dq"],
            "ddq_cmd": ddq_cmd,
            "tau_cmd_ff": tau_cmd_ff,
            "kp": command["kp"],
            "kd": command["kd"],
            "tau_cmd_pd": tau_cmd_pd,
            "q_est": state["q"],
            "dq_est": state["dq"],
            "ddq_est": state["ddq"],
            "tau_est": state["tau_est"],
            "q_residual": state["q"] - command["q"],
            "dq_residual": state["dq"] - command["dq"],
            "ddq_residual": state["ddq"] - ddq_cmd,
            "tau_residual": state["tau_est"] - tau_cmd_pd,
            "tau_residual_ff": state["tau_est"] - tau_cmd_ff,
        }
        self._sequence += 1
        return sample
