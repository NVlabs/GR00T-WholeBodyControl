from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_trajectory_projection import (
    TrajectoryProjectionConfig,
    TrajectoryProjectionError,
    audit_trajectory_constraints,
    project_nearest_trajectory,
)


def test_feasible_path_is_returned_exactly_without_solver_drift() -> None:
    path = np.asarray(
        [
            [0.0, 0.1],
            [0.01, 0.09],
            [0.025, 0.08],
            [0.04, 0.07],
        ],
        dtype=np.float32,
    )
    result = project_nearest_trajectory(
        path,
        lower_bounds=np.asarray([-1.0, -0.5]),
        upper_bounds=np.asarray([1.0, 0.5]),
        dt=0.1,
        max_velocity=np.asarray([1.0, 1.0]),
        max_acceleration=np.asarray([2.0, 2.0]),
    )

    assert result.iterations == 0
    assert result.objective == 0.0
    np.testing.assert_array_equal(result.projected_path, path.astype(np.float64))
    assert result.audit.passed


def test_two_frame_solution_is_nearest_and_enforces_zero_initial_velocity() -> None:
    result = project_nearest_trajectory(
        np.asarray([[0.0], [1.0]]),
        lower_bounds=np.asarray([-10.0]),
        upper_bounds=np.asarray([10.0]),
        dt=1.0,
        max_velocity=10.0,
        max_acceleration=0.2,
    )

    np.testing.assert_allclose(
        result.projected_path[:, 0],
        np.asarray([0.4, 0.6]),
        rtol=0.0,
        atol=2.0e-8,
    )
    assert result.audit.max_acceleration_abs <= 0.2 + 1.0e-8
    assert result.audit.passed


def test_noisy_multidimensional_path_passes_independent_exact_audit() -> None:
    rng = np.random.default_rng(17)
    desired = rng.normal(scale=0.35, size=(80, 3))
    lower = np.asarray([-0.5, -0.25, -0.8])
    upper = np.asarray([0.4, 0.6, 0.3])
    velocity = np.asarray([1.0, 1.5, 0.8])
    acceleration = np.asarray([4.0, 3.0, 2.0])
    dt = 0.05

    result = project_nearest_trajectory(
        desired,
        lower_bounds=lower,
        upper_bounds=upper,
        dt=dt,
        max_velocity=velocity,
        max_acceleration=acceleration,
    )
    independent = audit_trajectory_constraints(
        result.projected_path,
        lower_bounds=lower,
        upper_bounds=upper,
        dt=dt,
        max_velocity=velocity,
        max_acceleration=acceleration,
        tolerance=1.0e-8,
    )

    assert result.iterations > 0
    assert result.objective > 0.0
    assert independent.passed
    assert independent.position_violation_max <= 1.0e-8
    assert independent.velocity_step_violation_max <= 1.0e-8
    assert independent.acceleration_step_violation_max <= 1.0e-8


def test_audit_catches_first_step_acceleration_violation() -> None:
    audit = audit_trajectory_constraints(
        np.asarray([[0.0], [0.2], [0.2]]),
        lower_bounds=np.asarray([-1.0]),
        upper_bounds=np.asarray([1.0]),
        dt=0.1,
        max_velocity=10.0,
        max_acceleration=1.0,
    )

    assert not audit.passed
    assert audit.velocity_step_violation_max == 0.0
    assert audit.acceleration_step_violation_max == pytest.approx(0.19)
    assert audit.max_acceleration_abs == pytest.approx(20.0)


def test_moving_start_requires_matching_initial_velocity() -> None:
    moving_path = np.asarray([[0.0], [0.1], [0.2], [0.3]])
    zero_start_audit = audit_trajectory_constraints(
        moving_path,
        lower_bounds=np.asarray([-1.0]),
        upper_bounds=np.asarray([1.0]),
        dt=0.1,
        max_velocity=2.0,
        max_acceleration=1.0,
    )
    moving_start_audit = audit_trajectory_constraints(
        moving_path,
        lower_bounds=np.asarray([-1.0]),
        upper_bounds=np.asarray([1.0]),
        dt=0.1,
        max_velocity=2.0,
        max_acceleration=1.0,
        initial_velocity=np.asarray([1.0]),
    )
    result = project_nearest_trajectory(
        moving_path,
        lower_bounds=np.asarray([-1.0]),
        upper_bounds=np.asarray([1.0]),
        dt=0.1,
        max_velocity=2.0,
        max_acceleration=1.0,
        initial_velocity=np.asarray([1.0]),
    )

    assert not zero_start_audit.passed
    assert moving_start_audit.passed
    assert result.iterations == 0
    np.testing.assert_array_equal(result.projected_path, moving_path)


def test_projection_shifts_first_acceleration_bound_for_initial_velocity() -> None:
    result = project_nearest_trajectory(
        np.asarray([[0.0], [0.0]]),
        lower_bounds=np.asarray([-10.0]),
        upper_bounds=np.asarray([10.0]),
        dt=1.0,
        max_velocity=2.0,
        max_acceleration=0.2,
        initial_velocity=np.asarray([1.0]),
    )

    np.testing.assert_allclose(
        result.projected_path[:, 0],
        np.asarray([-0.4, 0.4]),
        rtol=0.0,
        atol=2.0e-8,
    )
    assert result.audit.passed
    assert result.audit.max_acceleration_abs <= 0.2 + 1.0e-8


def test_time_varying_equality_bounds_freeze_complement_of_repair_window() -> None:
    desired = np.full((10, 1), 0.2)
    repair = np.zeros(10, dtype=bool)
    repair[3:7] = True
    desired[repair, 0] = np.asarray([1.0, -0.8, 1.0, -0.8])
    lower = np.full_like(desired, 0.2)
    upper = np.full_like(desired, 0.2)
    lower[repair] = -2.0
    upper[repair] = 2.0

    result = project_nearest_trajectory(
        desired,
        lower_bounds=lower,
        upper_bounds=upper,
        dt=1.0,
        max_velocity=1.0,
        max_acceleration=0.3,
    )

    np.testing.assert_allclose(
        result.projected_path[~repair],
        desired[~repair],
        rtol=0.0,
        atol=1.0e-8,
    )
    assert np.max(np.abs(result.projected_path[repair] - desired[repair])) > 0.1
    assert result.audit.passed


def test_malformed_time_varying_position_bound_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="lower_bounds must have shape"):
        project_nearest_trajectory(
            np.zeros((5, 2)),
            lower_bounds=np.zeros((4, 2)),
            upper_bounds=np.ones((5, 2)),
            dt=0.1,
            max_velocity=1.0,
            max_acceleration=1.0,
        )


@pytest.mark.parametrize(
    ("initial_velocity", "message"),
    [
        (np.asarray([0.0, 0.1]), "initial_velocity"),
        (np.asarray([np.nan]), "initial_velocity must be finite"),
        (np.asarray([2.0]), "initial_velocity must not exceed max_velocity"),
    ],
)
def test_malformed_initial_velocity_is_rejected(
    initial_velocity: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        project_nearest_trajectory(
            np.zeros((3, 1)),
            lower_bounds=np.asarray([-1.0]),
            upper_bounds=np.asarray([1.0]),
            dt=0.1,
            max_velocity=1.0,
            max_acceleration=1.0,
            initial_velocity=initial_velocity,
        )


def test_nonconvergence_fails_closed() -> None:
    desired = np.tile(np.asarray([[0.0], [1.0], [-1.0], [1.0]]), (8, 1))
    with pytest.raises(
        TrajectoryProjectionError,
        match="trajectory projection did not converge",
    ):
        project_nearest_trajectory(
            desired,
            lower_bounds=np.asarray([-0.2]),
            upper_bounds=np.asarray([0.2]),
            dt=0.02,
            max_velocity=0.1,
            max_acceleration=0.1,
            config=TrajectoryProjectionConfig(
                max_iterations=1,
                check_interval=1,
                primal_tolerance=0.0,
                dual_tolerance=0.0,
                audit_tolerance=0.0,
            ),
        )


@pytest.mark.parametrize(
    ("desired", "lower", "upper", "message"),
    [
        (np.zeros(3), np.asarray([-1.0]), np.asarray([1.0]), "desired_path"),
        (
            np.zeros((2, 2)),
            np.asarray([-1.0]),
            np.asarray([1.0, 1.0]),
            "lower_bounds",
        ),
        (
            np.zeros((2, 1)),
            np.asarray([1.0]),
            np.asarray([-1.0]),
            "lower_bounds",
        ),
    ],
)
def test_invalid_contract_is_rejected(
    desired: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        project_nearest_trajectory(
            desired,
            lower_bounds=lower,
            upper_bounds=upper,
            dt=0.02,
            max_velocity=1.0,
            max_acceleration=1.0,
        )
