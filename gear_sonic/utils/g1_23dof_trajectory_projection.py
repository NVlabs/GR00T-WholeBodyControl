"""Whole-horizon projection onto joint position, velocity, and acceleration limits.

The projector solves the strictly convex problem

    minimize 0.5 * ||path - desired||_F**2

subject to global or per-frame position bounds and symmetric finite-difference
velocity/acceleration limits.  It uses only SciPy: scaled ADMM handles the
linear inequality bounds, while one sparse banded factorization is reused for
every dimension and iteration.

Acceleration follows the motion-export convention used by the true-23
pipeline.  Velocity at frame zero defaults to zero, so the first acceleration
row is ``((path[1] - path[0]) / dt - initial_velocity) / dt``.  Later rows use
the usual second difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


class TrajectoryProjectionError(RuntimeError):
    """Raised when the constrained projection cannot be certified."""


@dataclass(frozen=True)
class TrajectoryProjectionConfig:
    """Numerical controls for the sparse ADMM projection."""

    rho: float = 10.0
    max_iterations: int = 10_000
    check_interval: int = 10
    primal_tolerance: float = 1.0e-9
    dual_tolerance: float = 1.0e-9
    audit_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.rho) or self.rho <= 0.0:
            raise ValueError("rho must be finite and positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.check_interval <= 0:
            raise ValueError("check_interval must be positive")
        for name in ("primal_tolerance", "dual_tolerance", "audit_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class TrajectoryConstraintAudit:
    """Direct finite-difference audit in displacement units."""

    passed: bool
    position_violation_max: float
    velocity_step_violation_max: float
    acceleration_step_violation_max: float
    max_velocity_abs: float
    max_acceleration_abs: float
    tolerance: float

    @property
    def max_constraint_violation(self) -> float:
        return max(
            self.position_violation_max,
            self.velocity_step_violation_max,
            self.acceleration_step_violation_max,
        )


@dataclass(frozen=True)
class TrajectoryProjectionResult:
    """Certified projected path and convergence diagnostics."""

    projected_path: np.ndarray
    iterations: int
    objective: float
    primal_residual_max: float
    dual_residual_max: float
    audit: TrajectoryConstraintAudit


def _dimension_vector(
    value: float | np.ndarray,
    dimension_count: int,
    *,
    name: str,
    strictly_positive: bool,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(dimension_count, float(array), dtype=np.float64)
    if array.shape != (dimension_count,):
        raise ValueError(f"{name} must be scalar or have shape [{dimension_count}]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    return array


def _position_bound_matrix(
    value: np.ndarray,
    frame_count: int,
    dimension_count: int,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (dimension_count,):
        array = np.broadcast_to(array, (frame_count, dimension_count)).copy()
    elif array.shape == (frame_count, dimension_count):
        array = array.copy()
    else:
        raise ValueError(
            f"{name} must have shape [{dimension_count}] or "
            f"[{frame_count}, {dimension_count}]"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _validated_inputs(
    desired_path: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    dt: float,
    max_velocity: float | np.ndarray,
    max_acceleration: float | np.ndarray,
    initial_velocity: float | np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    desired = np.asarray(desired_path, dtype=np.float64)
    if desired.ndim != 2:
        raise ValueError("desired_path must have shape [frames, dimensions]")
    frame_count, dimension_count = desired.shape
    if frame_count < 2 or dimension_count < 1:
        raise ValueError("desired_path needs at least two frames and one dimension")
    if not np.all(np.isfinite(desired)):
        raise ValueError("desired_path contains NaN or Inf")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    lower = _position_bound_matrix(
        lower_bounds,
        frame_count,
        dimension_count,
        name="lower_bounds",
    )
    upper = _position_bound_matrix(
        upper_bounds,
        frame_count,
        dimension_count,
        name="upper_bounds",
    )
    if np.any(lower > upper):
        raise ValueError("lower_bounds must not exceed upper_bounds")
    velocity = _dimension_vector(
        max_velocity,
        dimension_count,
        name="max_velocity",
        strictly_positive=True,
    )
    acceleration = _dimension_vector(
        max_acceleration,
        dimension_count,
        name="max_acceleration",
        strictly_positive=True,
    )
    initial = _dimension_vector(
        initial_velocity,
        dimension_count,
        name="initial_velocity",
        strictly_positive=False,
    )
    if np.any(np.abs(initial) > velocity):
        raise ValueError("initial_velocity must not exceed max_velocity")
    return desired, lower, upper, velocity, acceleration, initial


def _difference_operators(
    frame_count: int,
) -> tuple[sparse.csc_matrix, sparse.csc_matrix, sparse.csc_matrix]:
    identity = sparse.eye(frame_count, format="csc", dtype=np.float64)
    first_difference = sparse.diags(
        (-np.ones(frame_count - 1), np.ones(frame_count - 1)),
        offsets=(0, 1),
        shape=(frame_count - 1, frame_count),
        format="csc",
    )
    if frame_count == 2:
        acceleration = first_difference.copy()
    else:
        second_difference = sparse.diags(
            (
                np.ones(frame_count - 2),
                -2.0 * np.ones(frame_count - 2),
                np.ones(frame_count - 2),
            ),
            offsets=(0, 1, 2),
            shape=(frame_count - 2, frame_count),
            format="csc",
        )
        acceleration = sparse.vstack(
            (first_difference[:1], second_difference),
            format="csc",
        )
    return identity, first_difference, acceleration


def _constraint_system(
    frame_count: int,
    lower: np.ndarray,
    upper: np.ndarray,
    velocity_step: np.ndarray,
    acceleration_step: np.ndarray,
    initial_displacement: np.ndarray,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    identity, first_difference, acceleration = _difference_operators(frame_count)
    operator = sparse.vstack(
        (identity, first_difference, acceleration),
        format="csc",
    )
    acceleration_lower = np.broadcast_to(
        -acceleration_step,
        (frame_count - 1, lower.shape[1]),
    ).copy()
    acceleration_upper = np.broadcast_to(
        acceleration_step,
        (frame_count - 1, lower.shape[1]),
    ).copy()
    acceleration_lower[0] += initial_displacement
    acceleration_upper[0] += initial_displacement
    constraint_lower = np.concatenate(
        (
            lower,
            np.broadcast_to(
                -velocity_step,
                (frame_count - 1, lower.shape[1]),
            ),
            acceleration_lower,
        ),
        axis=0,
    )
    constraint_upper = np.concatenate(
        (
            upper,
            np.broadcast_to(
                velocity_step,
                (frame_count - 1, upper.shape[1]),
            ),
            acceleration_upper,
        ),
        axis=0,
    )
    return operator, constraint_lower, constraint_upper


def _positive_max(value: np.ndarray) -> float:
    return max(0.0, float(np.max(value)))


def audit_trajectory_constraints(
    path: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    dt: float,
    max_velocity: float | np.ndarray,
    max_acceleration: float | np.ndarray,
    initial_velocity: float | np.ndarray = 0.0,
    tolerance: float = 0.0,
) -> TrajectoryConstraintAudit:
    """Audit position and finite-difference limits without trusting solver state."""

    desired, lower, upper, velocity, acceleration, initial = _validated_inputs(
        path,
        lower_bounds,
        upper_bounds,
        dt,
        max_velocity,
        max_acceleration,
        initial_velocity,
    )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    position_violation = max(
        _positive_max(lower - desired),
        _positive_max(desired - upper),
    )
    displacement = np.diff(desired, axis=0)
    velocity_step_violation = _positive_max(
        np.abs(displacement) - velocity[None, :] * dt
    )
    acceleration_displacement = np.empty_like(displacement)
    acceleration_displacement[0] = displacement[0] - initial * dt
    if len(displacement) > 1:
        acceleration_displacement[1:] = np.diff(displacement, axis=0)
    acceleration_step_violation = _positive_max(
        np.abs(acceleration_displacement) - acceleration[None, :] * dt * dt
    )
    max_violation = max(
        position_violation,
        velocity_step_violation,
        acceleration_step_violation,
    )
    return TrajectoryConstraintAudit(
        passed=bool(max_violation <= tolerance),
        position_violation_max=position_violation,
        velocity_step_violation_max=velocity_step_violation,
        acceleration_step_violation_max=acceleration_step_violation,
        max_velocity_abs=max(
            float(np.max(np.abs(initial))),
            float(np.max(np.abs(displacement)) / dt),
        ),
        max_acceleration_abs=float(
            np.max(np.abs(acceleration_displacement)) / (dt * dt)
        ),
        tolerance=float(tolerance),
    )


def project_nearest_trajectory(
    desired_path: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    dt: float,
    max_velocity: float | np.ndarray,
    max_acceleration: float | np.ndarray,
    initial_velocity: float | np.ndarray = 0.0,
    config: TrajectoryProjectionConfig = TrajectoryProjectionConfig(),
) -> TrajectoryProjectionResult:
    """Return Euclidean-nearest path satisfying whole-horizon linear limits.

    Raises:
        ValueError: Input contract is malformed.
        TrajectoryProjectionError: ADMM cannot converge and pass an independent
            constraint audit within configured limits.
    """

    desired, lower, upper, velocity, acceleration, initial = _validated_inputs(
        desired_path,
        lower_bounds,
        upper_bounds,
        dt,
        max_velocity,
        max_acceleration,
        initial_velocity,
    )
    initial_audit = audit_trajectory_constraints(
        desired,
        lower_bounds=lower,
        upper_bounds=upper,
        dt=dt,
        max_velocity=velocity,
        max_acceleration=acceleration,
        initial_velocity=initial,
        tolerance=0.0,
    )
    if initial_audit.passed:
        return TrajectoryProjectionResult(
            projected_path=desired.copy(),
            iterations=0,
            objective=0.0,
            primal_residual_max=0.0,
            dual_residual_max=0.0,
            audit=initial_audit,
        )

    frame_count = len(desired)
    operator, constraint_lower, constraint_upper = _constraint_system(
        frame_count,
        lower,
        upper,
        velocity * dt,
        acceleration * dt * dt,
        initial * dt,
    )
    identity = sparse.eye(frame_count, format="csc", dtype=np.float64)
    normal_matrix = (
        identity + config.rho * (operator.T @ operator)
    ).tocsc()
    try:
        factor = splu(normal_matrix)
    except RuntimeError as error:
        raise TrajectoryProjectionError(
            "failed to factor trajectory projection system"
        ) from error

    projected = desired.copy()
    operator_path = operator @ projected
    auxiliary = np.clip(operator_path, constraint_lower, constraint_upper)
    scaled_dual = np.zeros_like(auxiliary)
    primal_residual = float("inf")
    dual_residual = float("inf")
    final_audit: TrajectoryConstraintAudit | None = None

    for iteration in range(1, config.max_iterations + 1):
        right_hand_side = desired + config.rho * (
            operator.T @ (auxiliary - scaled_dual)
        )
        projected = factor.solve(right_hand_side)
        if not np.all(np.isfinite(projected)):
            raise TrajectoryProjectionError("trajectory projection produced NaN or Inf")

        operator_path = operator @ projected
        previous_auxiliary = auxiliary
        auxiliary = np.clip(
            operator_path + scaled_dual,
            constraint_lower,
            constraint_upper,
        )
        scaled_dual = scaled_dual + operator_path - auxiliary

        if iteration % config.check_interval and iteration != config.max_iterations:
            continue
        primal_residual = float(np.max(np.abs(operator_path - auxiliary)))
        dual_residual = float(
            np.max(
                np.abs(
                    config.rho
                    * (operator.T @ (auxiliary - previous_auxiliary))
                )
            )
        )
        if (
            primal_residual <= config.primal_tolerance
            and dual_residual <= config.dual_tolerance
        ):
            final_audit = audit_trajectory_constraints(
                projected,
                lower_bounds=lower,
                upper_bounds=upper,
                dt=dt,
                max_velocity=velocity,
                max_acceleration=acceleration,
                initial_velocity=initial,
                tolerance=config.audit_tolerance,
            )
            if final_audit.passed:
                break
    else:
        raise TrajectoryProjectionError(
            "trajectory projection did not converge: "
            f"iterations={config.max_iterations}, "
            f"primal_residual={primal_residual:.3e}, "
            f"dual_residual={dual_residual:.3e}"
        )

    if final_audit is None or not final_audit.passed:
        raise TrajectoryProjectionError("trajectory projection failed final constraint audit")
    return TrajectoryProjectionResult(
        projected_path=np.asarray(projected, dtype=np.float64),
        iterations=iteration,
        objective=0.5 * float(np.sum((projected - desired) ** 2)),
        primal_residual_max=primal_residual,
        dual_residual_max=dual_residual,
        audit=final_audit,
    )


__all__ = [
    "TrajectoryConstraintAudit",
    "TrajectoryProjectionConfig",
    "TrajectoryProjectionError",
    "TrajectoryProjectionResult",
    "audit_trajectory_constraints",
    "project_nearest_trajectory",
]
