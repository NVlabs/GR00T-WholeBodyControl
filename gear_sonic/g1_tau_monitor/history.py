"""Bounded rolling history for torque samples."""

from collections import deque
from typing import Optional

import numpy as np

from .dds import TorqueSample
from .joints import UPPER_BODY_JOINTS


class TorqueHistory:
    def __init__(self, window_seconds: float):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = float(window_seconds)
        self._timestamps: deque[float] = deque()
        self._tau_est: deque[np.ndarray] = deque()
        self._tau_cmd: deque[np.ndarray] = deque()
        self._residual: deque[np.ndarray] = deque()
        self._q: deque[np.ndarray] = deque()
        self._q_cmd: deque[np.ndarray] = deque()
        self.latest_sample: Optional[TorqueSample] = None

    def append(self, sample: TorqueSample) -> None:
        expected_shape = (len(UPPER_BODY_JOINTS),)
        for name in ("tau_est", "tau_cmd", "residual"):
            values = np.asarray(getattr(sample, name), dtype=np.float64)
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {values.shape}; expected {expected_shape}"
                )
        self._timestamps.append(float(sample.timestamp))
        self._tau_est.append(sample.tau_est.copy())
        self._tau_cmd.append(sample.tau_cmd.copy())
        self._residual.append(sample.residual.copy())
        self._q.append(self._optional_values(sample.q, "q", expected_shape))
        self._q_cmd.append(
            self._optional_values(sample.q_cmd, "q_cmd", expected_shape)
        )
        self.latest_sample = sample
        self._prune(float(sample.timestamp))

    def clear(self) -> None:
        self._timestamps.clear()
        self._tau_est.clear()
        self._tau_cmd.clear()
        self._residual.clear()
        self._q.clear()
        self._q_cmd.clear()
        self.latest_sample = None

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
            self._tau_est.popleft()
            self._tau_cmd.popleft()
            self._residual.popleft()
            self._q.popleft()
            self._q_cmd.popleft()

    def snapshot(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        self._prune(now)
        if not self._timestamps:
            return (
                np.empty(0, dtype=np.float64),
                np.empty((0, len(UPPER_BODY_JOINTS)), dtype=np.float64),
            )
        return np.asarray(self._timestamps), np.stack(self._residual)

    def position_snapshot(
        self, now: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._prune(now)
        if not self._timestamps:
            empty = np.empty((0, len(UPPER_BODY_JOINTS)), dtype=np.float64)
            return np.empty(0, dtype=np.float64), empty, empty.copy()
        return np.asarray(self._timestamps), np.stack(self._q), np.stack(self._q_cmd)

    @staticmethod
    def _optional_values(
        values: Optional[np.ndarray], name: str, expected_shape: tuple[int, ...]
    ) -> np.ndarray:
        if values is None:
            return np.full(expected_shape, np.nan, dtype=np.float64)
        result = np.asarray(values, dtype=np.float64)
        if result.shape != expected_shape:
            raise ValueError(f"{name} has shape {result.shape}; expected {expected_shape}")
        return result.copy()
