from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from gear_sonic.data.unitree_conversion.quaternion import (
    quat_slerp_deployment,
    validate_wxyz,
)
from gear_sonic.data.unitree_conversion.resampling import (
    TimestampAuditReport,
    audit_source_timestamps,
    clamped_future_indices,
    future_windows,
    nearest_image_indices,
    resample_positions,
    resample_quaternions,
)


def _axis_quaternion(axis: str, angle: float) -> np.ndarray:
    half = angle / 2.0
    vectors = {
        "x": (math.sin(half), 0.0, 0.0),
        "y": (0.0, math.sin(half), 0.0),
        "z": (0.0, 0.0, math.sin(half)),
    }
    return np.array([math.cos(half), *vectors[axis]], dtype=np.float64)


def _scalar_position_reference(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    motion_seconds = float(source.shape[0]) / 30.0
    frame_count = math.floor(motion_seconds * 50.0)
    positions = np.empty((frame_count, source.shape[1]), dtype=np.float64)
    for target_index in range(frame_count):
        source_coordinate = (target_index / 50.0) * 30.0
        source_index_0 = math.floor(source_coordinate)
        source_index_1 = min(source_index_0 + 1, source.shape[0] - 1)
        weight_0 = 1.0 - (source_coordinate - source_index_0)
        weight_1 = 1.0 - weight_0
        for column in range(source.shape[1]):
            positions[target_index, column] = (
                weight_0 * source[source_index_0, column] + weight_1 * source[source_index_1, column]
            )
    velocities = np.empty_like(positions)
    for target_index in range(frame_count - 1):
        velocities[target_index] = 50.0 * (positions[target_index + 1] - positions[target_index])
    velocities[-1] = velocities[-2]
    return positions, velocities


def test_three_source_frames_produce_five_target_frames_and_deployment_velocity() -> None:
    positions = np.array([[0.0], [1.0], [2.0]])

    q50, dq50 = resample_positions(positions)

    np.testing.assert_allclose(q50[:, 0], [0.0, 0.6, 1.2, 1.8, 2.0])
    np.testing.assert_allclose(dq50[:, 0], [30.0, 30.0, 30.0, 10.0, 10.0])


def test_nonlinear_positions_use_forward_difference_and_copy_terminal_velocity() -> None:
    positions = np.array([[0.0], [1.0], [4.0]], dtype=np.float64)

    q50, dq50 = resample_positions(positions)

    np.testing.assert_allclose(q50[:, 0], [0.0, 0.6, 1.6, 3.4, 4.0])
    np.testing.assert_allclose(dq50[:, 0], [30.0, 50.0, 90.0, 30.0, 30.0])
    assert dq50[-1, 0] != 0.0


@pytest.mark.parametrize(("source_count", "dimension"), [(2, 1), (3, 3), (10, 7), (31, 29)])
def test_positions_match_literal_scalar_deployment_formula(
    source_count: int,
    dimension: int,
) -> None:
    source = np.random.default_rng(0).normal(size=(source_count, dimension))
    expected_q, expected_dq = _scalar_position_reference(source)

    actual_q, actual_dq = resample_positions(source)

    assert actual_q.shape == (math.floor((float(source_count) / 30.0) * 50.0), dimension)
    assert actual_dq.shape == actual_q.shape
    np.testing.assert_array_equal(actual_q, expected_q)
    np.testing.assert_array_equal(actual_dq, expected_dq)
    np.testing.assert_allclose(actual_q[-1], source[-1], rtol=0.0, atol=2e-16)


def test_frame_count_preserves_deployment_binary64_rounding_for_sixty_nine_frames() -> None:
    source = np.zeros((69, 1), dtype=np.float64)

    positions, velocities = resample_positions(source)

    assert positions.shape == (114, 1)
    assert velocities.shape == (114, 1)


def test_source_floor_preserves_deployment_binary64_rounding_at_target_205() -> None:
    source = np.zeros((124, 1), dtype=np.float64)
    source[122, 0] = 1.0
    source_coordinate = (205 / 50.0) * 30.0
    assert source_coordinate == np.nextafter(123.0, 0.0)

    positions, _ = resample_positions(source)

    expected_weight_0 = 1.0 - (source_coordinate - math.floor(source_coordinate))
    assert positions[205, 0] == expected_weight_0
    assert positions[205, 0] > 0.0


def test_position_resampling_owns_contiguous_float64_outputs_without_mutation() -> None:
    source = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(4, 3))
    snapshot = source.copy()

    positions, velocities = resample_positions(source)

    np.testing.assert_array_equal(source, snapshot)
    for output in (positions, velocities):
        assert output.dtype == np.dtype(np.float64)
        assert output.flags.owndata
        assert output.flags.c_contiguous
        assert not np.shares_memory(output, source)


@pytest.mark.parametrize(
    "source",
    [
        np.zeros(2, dtype=np.float64),
        np.zeros((2, 1, 1), dtype=np.float64),
        np.zeros((2, 0), dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        np.array([[0.0], [np.nan]]),
        np.array([[0.0], [np.inf]]),
        np.array([[0.0], [-np.inf]]),
        np.array([[False], [True]]),
        np.array([[0.0j], [1.0j]]),
        np.array([[0.0], [object()]], dtype=object),
        np.array([["0"], ["1"]]),
    ],
)
def test_position_resampling_rejects_invalid_shape_length_values_or_dtype(source: object) -> None:
    with pytest.raises(ValueError):
        resample_positions(source)


def test_quaternion_resampling_uses_identical_target_indices_and_slerp_alphas() -> None:
    source = np.stack(
        [
            _axis_quaternion("z", 0.0),
            _axis_quaternion("z", math.pi / 2.0),
            _axis_quaternion("z", math.pi),
        ]
    )

    result = resample_quaternions(source)

    expected = np.stack(
        [
            _axis_quaternion("z", 0.0),
            _axis_quaternion("z", 0.3 * math.pi),
            _axis_quaternion("z", 0.6 * math.pi),
            _axis_quaternion("z", 0.9 * math.pi),
            _axis_quaternion("z", math.pi),
        ]
    )
    assert result.shape == (5, 4)
    np.testing.assert_allclose(result, expected, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-15)


def test_quaternion_resampling_preserves_deployment_shortest_image_path() -> None:
    end = -_axis_quaternion("z", math.pi / 2.0)
    source = np.stack([_axis_quaternion("z", 0.0), end])

    result = resample_quaternions(source)

    np.testing.assert_allclose(result[1], _axis_quaternion("z", 0.3 * math.pi), atol=1e-15)
    np.testing.assert_allclose(result[2], end, atol=1e-15)
    assert np.isfinite(result).all()


def test_quaternion_resampling_normalizes_each_accepted_source_exactly_once() -> None:
    source = np.array(
        [
            [0.10786321854116837, -0.17541065077178888, 0.09501948660078273, 0.9739498954343064],
            [0.27093245598157867, 0.9106316038088241, -0.1927022168455699, -0.24540397981421042],
        ],
        dtype=np.float64,
    )
    expected_single_normalization = np.array(
        [-0.13725386254852537, -0.7251248288765898, 0.18183958653939178, 0.6498382295074597],
        dtype=np.float64,
    )
    normalized_start, _ = validate_wxyz(source[0])
    normalized_end, _ = validate_wxyz(source[1])
    old_double_normalized = quat_slerp_deployment(normalized_start, normalized_end, 0.6)

    result = resample_quaternions(source)[1]

    np.testing.assert_array_equal(
        result.view(np.uint64),
        expected_single_normalization.view(np.uint64),
    )
    assert not np.array_equal(result.view(np.uint64), old_double_normalized.view(np.uint64))
    np.testing.assert_array_equal(
        np.abs(result.view(np.int64) - old_double_normalized.view(np.int64)),
        [0, 0, 1, 1],
    )


@pytest.mark.parametrize("source_count", [2, 3, 10, 31])
def test_quaternion_resampling_matches_scalar_indices_and_slerp(source_count: int) -> None:
    source = np.stack([_axis_quaternion("y", float(angle)) for angle in np.linspace(-1.0, 1.0, source_count)])
    expected_rows = []
    for target_index in range(math.floor((float(source_count) / 30.0) * 50.0)):
        source_coordinate = (target_index / 50.0) * 30.0
        source_index_0 = math.floor(source_coordinate)
        source_index_1 = min(source_index_0 + 1, source_count - 1)
        expected_rows.append(
            quat_slerp_deployment(
                source[source_index_0],
                source[source_index_1],
                source_coordinate - source_index_0,
            )
        )

    result = resample_quaternions(source)

    np.testing.assert_array_equal(result, np.stack(expected_rows))


def test_quaternion_resampling_validates_even_an_invalid_terminal_source() -> None:
    source = np.stack(
        [
            _axis_quaternion("x", 0.0),
            _axis_quaternion("x", 0.5),
            _axis_quaternion("x", 1.0),
        ]
    )
    source[-1] *= 1.01

    with pytest.raises(ValueError, match="unit-norm tolerance"):
        resample_quaternions(source)


def test_quaternion_resampling_owns_normalized_float64_output_without_mutation() -> None:
    source = np.stack([_axis_quaternion("x", 0.0), _axis_quaternion("x", 0.4)]).astype(np.float32)
    snapshot = source.copy()

    result = resample_quaternions(source)

    np.testing.assert_array_equal(source, snapshot)
    assert result.dtype == np.dtype(np.float64)
    assert result.flags.owndata
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, source)


@pytest.mark.parametrize(
    "source",
    [
        np.zeros(4, dtype=np.float64),
        np.zeros((1, 4), dtype=np.float64),
        np.zeros((2, 3), dtype=np.float64),
        np.ones((2, 4, 1), dtype=np.float64),
        np.ones((2, 4), dtype=bool),
        np.ones((2, 4), dtype=np.complex128),
        np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, object()]], dtype=object),
        np.array([[1.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0, 0.0], [np.inf, 0.0, 0.0, 0.0]]),
    ],
)
def test_quaternion_resampling_rejects_invalid_shape_length_values_or_dtype(
    source: object,
) -> None:
    with pytest.raises(ValueError):
        resample_quaternions(source)


def test_clamped_future_indices_are_episode_local_owned_int64_rows() -> None:
    indices = clamped_future_indices(4, width=3)

    np.testing.assert_array_equal(indices, [[0, 1, 2], [1, 2, 3], [2, 3, 3], [3, 3, 3]])
    assert indices.dtype == np.dtype(np.int64)
    assert indices.flags.owndata
    assert indices.flags.c_contiguous


@pytest.mark.parametrize("frame_count", [True, 0, -1, 1.5, "3"])
def test_clamped_future_indices_rejects_invalid_frame_count(frame_count: object) -> None:
    with pytest.raises(ValueError, match="frame_count"):
        clamped_future_indices(frame_count)


@pytest.mark.parametrize("width", [True, 0, -1, 1.5, "3"])
def test_clamped_future_indices_rejects_invalid_width(width: object) -> None:
    with pytest.raises(ValueError, match="width"):
        clamped_future_indices(3, width=width)


def test_final_window_repeats_stored_terminal_pose_and_velocity() -> None:
    q50, dq50 = resample_positions(np.array([[0.0], [1.0], [2.0]]))

    q_window, dq_window = future_windows(q50, dq50, width=10)

    np.testing.assert_allclose(q_window[-1, :, 0], [2.0] * 10)
    np.testing.assert_allclose(dq_window[-1, :, 0], [10.0] * 10)
    assert q_window.shape == (5, 10, 1)
    assert dq_window.shape == (5, 10, 1)


def test_future_windows_follow_clamped_indices_and_own_float64_storage() -> None:
    positions = np.arange(12, dtype=np.float32).reshape(4, 3)
    velocities = positions + np.float32(100.0)
    snapshots = (positions.copy(), velocities.copy())

    position_windows, velocity_windows = future_windows(positions, velocities, width=3)

    expected_indices = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 3], [3, 3, 3]])
    np.testing.assert_array_equal(position_windows, snapshots[0][expected_indices])
    np.testing.assert_array_equal(velocity_windows, snapshots[1][expected_indices])
    np.testing.assert_array_equal(positions, snapshots[0])
    np.testing.assert_array_equal(velocities, snapshots[1])
    for output in (position_windows, velocity_windows):
        assert output.dtype == np.dtype(np.float64)
        assert output.flags.owndata
        assert output.flags.c_contiguous
        assert not np.shares_memory(output, positions)
        assert not np.shares_memory(output, velocities)


@pytest.mark.parametrize(
    ("positions", "velocities"),
    [
        (np.zeros(3), np.zeros(3)),
        (np.zeros((0, 2)), np.zeros((0, 2))),
        (np.zeros((3, 0)), np.zeros((3, 0))),
        (np.zeros((3, 2)), np.zeros((2, 2))),
        (np.zeros((3, 2)), np.zeros((3, 1))),
        (np.zeros((3, 2), dtype=np.float32), np.zeros((3, 2), dtype=np.float64)),
        (np.zeros((3, 2), dtype=bool), np.zeros((3, 2), dtype=bool)),
        (np.zeros((3, 2), dtype=complex), np.zeros((3, 2), dtype=complex)),
        (np.zeros((3, 2), dtype=object), np.zeros((3, 2), dtype=object)),
        (np.array([[0.0], [1.0], [np.nan]]), np.zeros((3, 1))),
        (np.zeros((3, 1)), np.array([[0.0], [1.0], [np.inf]])),
    ],
)
def test_future_windows_rejects_invalid_shape_dtype_or_values(
    positions: object,
    velocities: object,
) -> None:
    with pytest.raises(ValueError):
        future_windows(positions, velocities)


def test_image_indices_use_round_half_up_and_return_exact_target_count() -> None:
    result = nearest_image_indices(3)

    np.testing.assert_array_equal(result, [0, 1, 1, 2, 2])
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    assert result.flags.owndata
    assert result.flags.c_contiguous


@pytest.mark.parametrize("source_count", [2, 3, 10, 31])
def test_image_indices_match_literal_scalar_formula(source_count: int) -> None:
    expected = []
    for target_index in range(math.floor((float(source_count) / 30.0) * 50.0)):
        source_coordinate = (target_index / 50.0) * 30.0
        expected.append(min(math.floor(source_coordinate + 0.5), source_count - 1))

    result = nearest_image_indices(source_count)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("source_count", [True, 0, 1, -1, 2.0, "2", np.array(2)])
def test_image_indices_reject_invalid_or_short_source_counts(source_count: object) -> None:
    with pytest.raises(ValueError, match="source frame_count"):
        nearest_image_indices(source_count)


def test_nominal_float32_timestamps_pass_audit_and_are_not_mutated() -> None:
    timestamps = np.arange(20, dtype=np.float32) / np.float32(30.0)
    snapshot = timestamps.copy()

    report = audit_source_timestamps(timestamps)

    assert report.frame_count == 20
    assert report.max_grid_error_seconds <= 1e-6
    assert report.warning is False
    assert report.rejected is False
    np.testing.assert_array_equal(timestamps, snapshot)


def test_timestamp_audit_ignores_a_constant_start_shift() -> None:
    report = audit_source_timestamps(123.25 + np.arange(10, dtype=np.float64) / 30.0)

    assert report.max_grid_error_seconds <= np.finfo(np.float64).eps * 64
    assert report.warning is False
    assert report.rejected is False


@pytest.mark.parametrize(
    ("threshold", "direction", "warning", "rejected"),
    [
        (0.001, 0.0, False, False),
        (0.001, np.inf, True, False),
        (1.0 / 60.0, 0.0, True, False),
        (1.0 / 60.0, np.inf, True, True),
    ],
)
def test_timestamp_audit_thresholds_are_strictly_greater_than(
    threshold: float,
    direction: float,
    warning: bool,
    rejected: bool,
) -> None:
    timestamps = np.arange(2, dtype=np.float64) / 30.0
    timestamps[-1] = np.nextafter(timestamps[-1] + threshold, direction)

    report = audit_source_timestamps(timestamps)

    assert report.warning is warning
    assert report.rejected is rejected


def test_timestamp_audit_report_is_deterministic_and_frozen() -> None:
    timestamps = np.array([2.0, 2.0 + 1.0 / 30.0, 2.0 + 2.0 / 30.0])

    first = audit_source_timestamps(timestamps)
    second = audit_source_timestamps(timestamps.copy())

    assert first == second
    assert isinstance(first, TimestampAuditReport)
    with pytest.raises(FrozenInstanceError):
        first.warning = True  # type: ignore[misc]


@pytest.mark.parametrize("frame_count", [True, 0, 1, 2.0, np.int64(2)])
def test_timestamp_audit_report_rejects_invalid_direct_frame_count(frame_count: object) -> None:
    with pytest.raises(ValueError, match="frame_count"):
        TimestampAuditReport(
            frame_count=frame_count,  # type: ignore[arg-type]
            max_grid_error_seconds=0.0,
            warning=False,
            rejected=False,
        )


@pytest.mark.parametrize(
    "max_error",
    [True, -1.0, np.nan, np.inf, -np.inf, "0", 0.0j, np.array(0.0)],
)
def test_timestamp_audit_report_rejects_invalid_direct_max_error(max_error: object) -> None:
    with pytest.raises(ValueError, match="max_grid_error_seconds"):
        TimestampAuditReport(
            frame_count=2,
            max_grid_error_seconds=max_error,  # type: ignore[arg-type]
            warning=False,
            rejected=False,
        )


@pytest.mark.parametrize(("warning", "rejected"), [(0, False), (np.bool_(False), False), (False, 0)])
def test_timestamp_audit_report_requires_builtin_bool_flags(
    warning: object,
    rejected: object,
) -> None:
    with pytest.raises(ValueError, match="flags"):
        TimestampAuditReport(
            frame_count=2,
            max_grid_error_seconds=0.0,
            warning=warning,  # type: ignore[arg-type]
            rejected=rejected,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("max_error", "warning", "rejected"),
    [
        (0.0, False, False),
        (0.001, False, False),
        (np.nextafter(0.001, np.inf), True, False),
        (1.0 / 60.0, True, False),
        (np.nextafter(1.0 / 60.0, np.inf), True, True),
    ],
)
def test_timestamp_audit_report_accepts_exact_strict_threshold_flags(
    max_error: float,
    warning: bool,
    rejected: bool,
) -> None:
    report = TimestampAuditReport(
        frame_count=2,
        max_grid_error_seconds=max_error,
        warning=warning,
        rejected=rejected,
    )

    assert report.max_grid_error_seconds == max_error
    assert report.warning is warning
    assert report.rejected is rejected


@pytest.mark.parametrize(
    ("max_error", "warning", "rejected"),
    [
        (0.0, True, False),
        (0.002, False, False),
        (0.002, True, True),
        (0.02, True, False),
        (0.02, False, True),
    ],
)
def test_timestamp_audit_report_rejects_flags_inconsistent_with_strict_thresholds(
    max_error: float,
    warning: bool,
    rejected: bool,
) -> None:
    with pytest.raises(ValueError, match="agree"):
        TimestampAuditReport(
            frame_count=2,
            max_grid_error_seconds=max_error,
            warning=warning,
            rejected=rejected,
        )


@pytest.mark.parametrize(
    "timestamps",
    [
        np.array(0.0),
        np.zeros((2, 1)),
        np.array([0.0]),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([0.0, -np.inf]),
        np.array([0.0, 0.0]),
        np.array([0.1, 0.0]),
        np.array([False, True]),
        np.array([0.0j, 1.0j]),
        np.array([0.0, object()], dtype=object),
        np.array(["0", "1"]),
    ],
)
def test_timestamp_audit_rejects_invalid_shape_length_values_order_or_dtype(
    timestamps: object,
) -> None:
    with pytest.raises(ValueError):
        audit_source_timestamps(timestamps)
