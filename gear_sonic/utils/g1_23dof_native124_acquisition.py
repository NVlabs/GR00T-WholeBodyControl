"""Fail-closed, simulation-only acquisition transition for native23 targets.

This module has no Unitree SDK or transport dependency.  It models the staged
transition that must be proven in MuJoCo before an equivalent mechanism can be
considered for a robot runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

REQUIRED_MODE_MACHINE = 4
MAXIMUM_LOWSTATE_AGE_MS = 40.0
CONTROL_HZ = 50
CONTROL_PERIOD_S = 1.0 / CONTROL_HZ
WARM_START_FRAMES = 25
REFERENCE_RAMP_FRAMES = 100
MAXIMUM_TARGET_STEP_RAD = 0.005
HARD_MAXIMUM_GANTRY_TARGET_STEP_RAD = 0.04


def _vector(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (23,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite [23]")
    return result.copy()


def require_local_state_gate(*, mode_machine: int, lowstate_age_ms: float) -> None:
    """Apply the same hard local gates expected before target acceptance."""

    if isinstance(mode_machine, bool) or mode_machine != REQUIRED_MODE_MACHINE:
        raise RuntimeError("acquisition requires mode_machine == 4")
    if not np.isfinite(lowstate_age_ms) or not (0.0 <= lowstate_age_ms <= MAXIMUM_LOWSTATE_AGE_MS):
        raise RuntimeError("acquisition requires LowState age in [0, 40] ms")


@dataclass(frozen=True)
class AcquisitionFrame:
    frame_index: int
    phase: str
    reference_alpha: float
    reference_position_native: np.ndarray
    reference_velocity_native: np.ndarray
    applied_target_hardware: np.ndarray
    raw_target_limit_clamps: int
    raw_target_maximum_excess_rad: float
    maximum_target_step_rad: float
    mode_machine: int
    lowstate_age_ms: float
    acquisition_qualified: bool
    fail_closed_reason: str | None


@dataclass
class Native23AcquisitionTransition:
    """Warm actor history, ramp its reference, and slew applied PD targets."""

    measured_start_hardware: Sequence[float] | np.ndarray
    reference_start_native: Sequence[float] | np.ndarray
    lower_limit_hardware: Sequence[float] | np.ndarray
    upper_limit_hardware: Sequence[float] | np.ndarray
    warm_start_frames: int = WARM_START_FRAMES
    reference_ramp_frames: int = REFERENCE_RAMP_FRAMES
    maximum_target_step_rad: float = MAXIMUM_TARGET_STEP_RAD
    maximum_raw_target_clamps: int = 0
    maximum_raw_target_excess_rad: float = 0.0
    _frame_index: int = field(init=False, default=0)
    _previous_target_hardware: np.ndarray = field(init=False, repr=False)
    _fail_closed_reason: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.measured_start_hardware = _vector(self.measured_start_hardware, "measured_start_hardware")
        self.reference_start_native = _vector(self.reference_start_native, "reference_start_native")
        self.lower_limit_hardware = _vector(self.lower_limit_hardware, "lower_limit_hardware")
        self.upper_limit_hardware = _vector(self.upper_limit_hardware, "upper_limit_hardware")
        if np.any(self.lower_limit_hardware >= self.upper_limit_hardware):
            raise ValueError("hardware limits must be strictly ordered")
        if np.any(self.measured_start_hardware < self.lower_limit_hardware) or np.any(
            self.measured_start_hardware > self.upper_limit_hardware
        ):
            raise ValueError("measured start lies outside hardware limits")
        if (
            isinstance(self.warm_start_frames, bool)
            or self.warm_start_frames < 1
            or isinstance(self.reference_ramp_frames, bool)
            or self.reference_ramp_frames < 1
        ):
            raise ValueError("warm-start and reference-ramp frames must be positive")
        if not np.isfinite(self.maximum_target_step_rad) or not (
            0.0
            < self.maximum_target_step_rad
            <= HARD_MAXIMUM_GANTRY_TARGET_STEP_RAD
        ):
            raise ValueError("maximum target step must be in (0, 0.04] rad/frame")
        if (
            isinstance(self.maximum_raw_target_clamps, bool)
            or not 0 <= self.maximum_raw_target_clamps <= 23
        ):
            raise ValueError("maximum raw target clamps must be in [0, 23]")
        if not np.isfinite(self.maximum_raw_target_excess_rad) or not (
            0.0 <= self.maximum_raw_target_excess_rad <= 2.0
        ):
            raise ValueError("maximum raw target excess must be in [0, 2] rad")
        self._previous_target_hardware = self.measured_start_hardware.copy()

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def fail_closed_reason(self) -> str | None:
        return self._fail_closed_reason

    def _reference(
        self,
        desired_position_native: Sequence[float] | np.ndarray,
        desired_velocity_native: Sequence[float] | np.ndarray,
    ) -> tuple[str, float, np.ndarray, np.ndarray]:
        desired_position = _vector(desired_position_native, "desired_position_native")
        desired_velocity = _vector(desired_velocity_native, "desired_velocity_native")
        if self._fail_closed_reason is not None:
            return (
                "fail_closed_hold",
                0.0,
                self.reference_start_native.copy(),
                np.zeros(23, dtype=np.float64),
            )
        if self._frame_index < self.warm_start_frames:
            return (
                "warm_start_hold",
                0.0,
                self.reference_start_native.copy(),
                np.zeros(23, dtype=np.float64),
            )

        ramp_index = self._frame_index - self.warm_start_frames + 1
        u = min(ramp_index / self.reference_ramp_frames, 1.0)
        alpha = u * u * (3.0 - 2.0 * u)
        if u >= 1.0:
            alpha_rate = 0.0
            phase = "tracking"
        else:
            alpha_rate = 6.0 * u * (1.0 - u) / (self.reference_ramp_frames * CONTROL_PERIOD_S)
            phase = "reference_ramp"
        displacement = desired_position - self.reference_start_native
        position = self.reference_start_native + alpha * displacement
        velocity = alpha * desired_velocity + alpha_rate * displacement
        return phase, alpha, position, velocity

    def build_reference(
        self,
        *,
        desired_position_native: Sequence[float] | np.ndarray,
        desired_velocity_native: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the reference for the current frame without advancing it."""

        _, _, position, velocity = self._reference(desired_position_native, desired_velocity_native)
        return position.astype(np.float32), velocity.astype(np.float32)

    def accept_target(
        self,
        *,
        raw_target_hardware: Sequence[float] | np.ndarray,
        desired_position_native: Sequence[float] | np.ndarray,
        desired_velocity_native: Sequence[float] | np.ndarray,
        mode_machine: int,
        lowstate_age_ms: float,
    ) -> AcquisitionFrame:
        """Fail closed on local state, then clamp and slew one target frame."""

        require_local_state_gate(mode_machine=mode_machine, lowstate_age_ms=lowstate_age_ms)
        raw_target = _vector(raw_target_hardware, "raw_target_hardware")
        phase, alpha, reference_position, reference_velocity = self._reference(
            desired_position_native, desired_velocity_native
        )
        below = self.lower_limit_hardware - raw_target
        above = raw_target - self.upper_limit_hardware
        excess = np.maximum(np.maximum(below, above), 0.0)
        raw_clamps = int(np.count_nonzero(excess > 0.0))
        maximum_excess = float(np.max(excess))
        if (
            (
                raw_clamps > self.maximum_raw_target_clamps
                or maximum_excess > self.maximum_raw_target_excess_rad
            )
            and self._fail_closed_reason is None
        ):
            self._fail_closed_reason = "raw_policy_target_outside_hardware_limits"
        bounded_target = np.clip(raw_target, self.lower_limit_hardware, self.upper_limit_hardware)
        if self._fail_closed_reason is not None:
            phase = "fail_closed_hold"
            alpha = 0.0
            reference_position = self.reference_start_native.copy()
            reference_velocity = np.zeros(23, dtype=np.float64)
            requested_target = self._previous_target_hardware
        elif self._frame_index < self.warm_start_frames:
            requested_target = self.measured_start_hardware
        else:
            requested_target = bounded_target
        delta = np.clip(
            requested_target - self._previous_target_hardware,
            -self.maximum_target_step_rad,
            self.maximum_target_step_rad,
        )
        applied = np.clip(
            self._previous_target_hardware + delta,
            self.lower_limit_hardware,
            self.upper_limit_hardware,
        )
        maximum_step = float(np.max(np.abs(applied - self._previous_target_hardware)))
        frame = AcquisitionFrame(
            frame_index=self._frame_index,
            phase=phase,
            reference_alpha=alpha,
            reference_position_native=reference_position.astype(np.float32),
            reference_velocity_native=reference_velocity.astype(np.float32),
            applied_target_hardware=applied.astype(np.float32),
            raw_target_limit_clamps=raw_clamps,
            raw_target_maximum_excess_rad=maximum_excess,
            maximum_target_step_rad=maximum_step,
            mode_machine=mode_machine,
            lowstate_age_ms=float(lowstate_age_ms),
            acquisition_qualified=self._fail_closed_reason is None,
            fail_closed_reason=self._fail_closed_reason,
        )
        self._previous_target_hardware = applied
        self._frame_index += 1
        return frame


__all__ = [
    "AcquisitionFrame",
    "CONTROL_HZ",
    "CONTROL_PERIOD_S",
    "HARD_MAXIMUM_GANTRY_TARGET_STEP_RAD",
    "MAXIMUM_LOWSTATE_AGE_MS",
    "MAXIMUM_TARGET_STEP_RAD",
    "Native23AcquisitionTransition",
    "REFERENCE_RAMP_FRAMES",
    "REQUIRED_MODE_MACHINE",
    "WARM_START_FRAMES",
    "require_local_state_gate",
]
