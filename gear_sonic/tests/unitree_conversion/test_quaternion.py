import math

import numpy as np
import pytest

from gear_sonic.data.unitree_conversion import dex3_adapter as dex3_adapter_module
from gear_sonic.data.unitree_conversion.dex3_adapter import DEX3_FEATURE_NAMES, adapt_dex3_arrays
from gear_sonic.data.unitree_conversion.inspire_diagnostics import (
    INSPIRE_GATE_REASONS,
    diagnose_inspire_arrays,
)
from gear_sonic.data.unitree_conversion.quaternion import (
    heading_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp_deployment,
    quat_to_encoder_rot6d,
    validate_wxyz,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _axis_quat(axis: str, angle: float) -> np.ndarray:
    half = angle / 2.0
    vector = {
        "x": (math.sin(half), 0.0, 0.0),
        "y": (0.0, math.sin(half), 0.0),
        "z": (0.0, 0.0, math.sin(half)),
    }[axis]
    return np.array([math.cos(half), *vector], dtype=np.float64)


def _inspire_arrays(n: int = 3) -> dict[str, np.ndarray]:
    current = np.zeros((n, 36), dtype=np.float64)
    desired = np.zeros((n, 36), dtype=np.float64)
    current[:, 3] = 1.0
    desired[:, 3] = 1.0
    return {
        "current": current,
        "desired": desired,
        "hand_state": np.zeros((n, 12), dtype=np.float64),
        "hand_cmd": np.zeros((n, 12), dtype=np.float64),
        "timestamps": np.arange(n, dtype=np.float64) / 30.0,
    }


def _deployment_heading_reference(quaternion: np.ndarray) -> np.ndarray:
    normalized, _ = validate_wxyz(quaternion)
    w, x, y, z = normalized
    scale_a = 2.0 * w * w - 1.0
    a0 = scale_a
    b0 = 0.0
    c0 = x * x * 2.0
    rotated_x = (a0 + b0) + c0
    a1 = 0.0
    b1 = z * w * 2.0
    c1 = y * x * 2.0
    rotated_y = (a1 + b1) + c1
    result, _ = validate_wxyz(_axis_quat("z", math.atan2(rotated_y, rotated_x)))
    return result


def test_validate_wxyz_rejects_norm_outside_one_e_minus_five() -> None:
    with pytest.raises(ValueError, match="unit-norm tolerance"):
        validate_wxyz(np.array([1.00002, 0.0, 0.0, 0.0], dtype=np.float64))


def test_validate_wxyz_renormalizes_accepted_quaternion_and_reports_bitwise_change() -> None:
    source = np.array([1.000005, 0.0, -0.0, 0.0], dtype=np.float64)
    snapshot = source.copy()

    result, changed = validate_wxyz(source)

    np.testing.assert_array_equal(result, [1.0, 0.0, -0.0, 0.0])
    np.testing.assert_array_equal(source.view(np.uint64), snapshot.view(np.uint64))
    assert changed is True
    assert result.dtype == np.dtype(np.float64)
    assert result.flags.owndata
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, source)


def test_validate_wxyz_reports_no_bitwise_change_for_exact_identity() -> None:
    result, changed = validate_wxyz(IDENTITY)

    np.testing.assert_array_equal(result.view(np.uint64), IDENTITY.view(np.uint64))
    assert changed is False
    assert result.flags.owndata


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(3, dtype=np.float64),
        np.zeros((1, 4), dtype=np.float64),
        np.array([True, False, False, False]),
        np.array([1, 0, 0, object()], dtype=object),
        np.array([1 + 0j, 0j, 0j, 0j]),
        ["1", "0", "0", "0"],
    ],
)
def test_validate_wxyz_rejects_wrong_shape_or_nonreal_numeric_dtype(value: object) -> None:
    with pytest.raises(ValueError, match="numeric float64 quaternion|shape"):
        validate_wxyz(value)


@pytest.mark.parametrize(
    "value",
    [
        np.array([np.nan, 0.0, 0.0, 0.0]),
        np.array([np.inf, 0.0, 0.0, 0.0]),
        np.array([-np.inf, 0.0, 0.0, 0.0]),
    ],
)
def test_validate_wxyz_rejects_nonfinite_components(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_wxyz(value)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(4, dtype=np.float64),
        np.array([np.nextafter(1e-12, 0.0), 0.0, 0.0, 0.0]),
    ],
)
def test_validate_wxyz_rejects_degenerate_norm(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="degenerate"):
        validate_wxyz(value)


def test_hamilton_multiplication_order_and_normalized_conjugate_inverse() -> None:
    qx = _axis_quat("x", math.pi / 2.0)
    qz = _axis_quat("z", math.pi / 2.0)

    np.testing.assert_allclose(quat_mul(qz, qx), [0.5, 0.5, 0.5, 0.5], atol=1e-15)
    np.testing.assert_allclose(quat_mul(qx, qz), [0.5, 0.5, -0.5, 0.5], atol=1e-15)
    np.testing.assert_allclose(quat_mul(qx, quat_conjugate(qx)), IDENTITY, atol=1e-15)


def test_conjugate_normalizes_an_accepted_inverse_precondition_without_mutation() -> None:
    source = _axis_quat("y", 0.4) * 1.000005
    snapshot = source.copy()

    result = quat_conjugate(source)

    expected = _axis_quat("y", -0.4)
    np.testing.assert_allclose(result, expected, atol=1e-15)
    np.testing.assert_array_equal(source, snapshot)
    assert result.dtype == np.dtype(np.float64)
    assert result.flags.owndata
    assert result.flags.c_contiguous


def test_heading_extracts_deployment_projected_x_yaw_with_roll_and_pitch() -> None:
    yaw = 0.7
    full = quat_mul(
        quat_mul(_axis_quat("z", yaw), _axis_quat("y", -0.3)),
        _axis_quat("x", 0.4),
    )

    result = heading_quat(full)

    np.testing.assert_allclose(result, _axis_quat("z", yaw), atol=1e-15)
    assert result.flags.owndata
    assert result.flags.c_contiguous


@pytest.mark.parametrize("pitch_sign", [1.0, -1.0])
def test_heading_matches_deployment_arithmetic_at_ninety_degree_pitch_singularity(
    pitch_sign: float,
) -> None:
    half = math.sqrt(0.5)
    quaternion = np.array([half, 0.0, pitch_sign * half, 0.0], dtype=np.float64)
    expected = _deployment_heading_reference(quaternion)

    result = heading_quat(quaternion)

    np.testing.assert_array_equal(expected.view(np.uint64), IDENTITY.view(np.uint64))
    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize(
    ("pitch", "expected_heading"),
    [
        (math.pi / 2.0 - 1e-8, 0.0),
        (math.pi / 2.0 + 1e-8, math.pi),
    ],
)
def test_heading_matches_deployment_arithmetic_near_pitch_singularity(
    pitch: float,
    expected_heading: float,
) -> None:
    quaternion = _axis_quat("y", pitch)
    expected = _deployment_heading_reference(quaternion)

    result = heading_quat(quaternion)

    np.testing.assert_array_equal(
        expected.view(np.uint64),
        _axis_quat("z", expected_heading).view(np.uint64),
    )
    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))


def test_slerp_uses_shortest_quaternion_image_for_antipodes() -> None:
    result = quat_slerp_deployment(IDENTITY, -IDENTITY, 0.4)

    np.testing.assert_allclose(result, IDENTITY, atol=1e-12)


def test_slerp_nearby_branch_matches_normalized_linear_deployment_formula() -> None:
    q0 = _axis_quat("z", 0.01)
    q1 = _axis_quat("z", 0.03)
    alpha = 0.3
    linear = q0 + alpha * (q1 - q0)
    expected = linear / np.linalg.norm(linear)

    result = quat_slerp_deployment(q0, q1, alpha)

    np.testing.assert_array_equal(result, expected)


def test_slerp_spherical_branch_matches_golden_axis_angle_and_unit_contract() -> None:
    result = quat_slerp_deployment(IDENTITY, _axis_quat("z", math.pi / 2.0), 0.25)

    np.testing.assert_allclose(result, _axis_quat("z", math.pi / 8.0), atol=1e-15)
    assert abs(np.linalg.norm(result) - 1.0) <= 1e-10
    assert result.dtype == np.dtype(np.float64)
    assert result.flags.owndata
    assert result.flags.c_contiguous


def test_slerp_endpoints_are_owned_and_inputs_are_unchanged() -> None:
    q0 = _axis_quat("x", 0.2)
    q1 = _axis_quat("y", -0.6)
    snapshots = (q0.copy(), q1.copy())

    start = quat_slerp_deployment(q0, q1, 0.0)
    end = quat_slerp_deployment(q0, q1, 1.0)

    np.testing.assert_allclose(start, snapshots[0], atol=1e-15)
    np.testing.assert_allclose(end, snapshots[1], atol=1e-15)
    np.testing.assert_array_equal(q0, snapshots[0])
    np.testing.assert_array_equal(q1, snapshots[1])
    assert start.flags.owndata and end.flags.owndata
    assert not np.shares_memory(start, q0)
    assert not np.shares_memory(end, q1)


@pytest.mark.parametrize("alpha", [True, "0.5", np.array([0.5]), np.nan, np.inf, -0.1, 1.1])
def test_slerp_rejects_invalid_alpha(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        quat_slerp_deployment(IDENTITY, IDENTITY, alpha)


def test_all_quaternion_operations_reject_invalid_operands() -> None:
    bad = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="shape"):
        quat_conjugate(bad)
    with pytest.raises(ValueError, match="shape"):
        quat_mul(IDENTITY, bad)
    with pytest.raises(ValueError, match="shape"):
        heading_quat(bad)
    with pytest.raises(ValueError, match="shape"):
        quat_slerp_deployment(bad, IDENTITY, 0.5)
    with pytest.raises(ValueError, match="shape"):
        quat_to_encoder_rot6d(bad)


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        (None, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ("x", [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ("y", [0.0, 0.0, 0.0, 1.0, -1.0, 0.0]),
        ("z", [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
    ],
)
def test_encoder_rot6d_is_row_wise_for_identity_and_ninety_degree_axes(
    axis: str | None,
    expected: list[float],
) -> None:
    quaternion = IDENTITY if axis is None else _axis_quat(axis, math.pi / 2.0)

    packed = quat_to_encoder_rot6d(quaternion)

    np.testing.assert_allclose(packed, expected, atol=1e-15)
    assert packed.shape == (6,)
    assert packed.dtype == np.dtype(np.float64)
    assert packed.flags.owndata
    assert packed.flags.c_contiguous


def test_heading_alignment_golden_equation_preserves_written_multiplication_order() -> None:
    observed_initial = _axis_quat("z", 0.3)
    reference_initial = _axis_quat("z", -0.2)
    observed_current = _axis_quat("z", 0.1)
    reference_future = _axis_quat("z", 0.4)

    apply_heading = quat_mul(heading_quat(observed_initial), quat_conjugate(heading_quat(reference_initial)))
    aligned = quat_mul(apply_heading, reference_future)
    relative = quat_mul(quat_conjugate(observed_current), aligned)

    np.testing.assert_allclose(relative, _axis_quat("z", 0.8), atol=1e-15)
    np.testing.assert_allclose(
        quat_to_encoder_rot6d(relative),
        [math.cos(0.8), -math.sin(0.8), math.sin(0.8), math.cos(0.8), 0.0, 0.0],
        atol=1e-15,
    )


def test_dex3_constructs_each_identity_root_stream_through_quaternion_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[np.ndarray] = []

    def tracked_validate(value: object) -> tuple[np.ndarray, bool]:
        calls.append(np.array(value, copy=True))
        return validate_wxyz(value)

    monkeypatch.setattr(dex3_adapter_module, "validate_wxyz", tracked_validate, raising=False)
    observed = np.zeros((2, 28), dtype=np.float64)

    episode = adapt_dex3_arrays(
        source_repo_id="synthetic/dex3",
        source_revision="1" * 40,
        source_episode_id=0,
        observed=observed,
        desired=observed,
        feature_names=DEX3_FEATURE_NAMES,
        timestamps=np.arange(2, dtype=np.float64) / 30.0,
        task_indices=np.zeros(2, dtype=np.int64),
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(calls, [IDENTITY, IDENTITY])
    np.testing.assert_array_equal(episode.observed_root_wxyz, np.tile(IDENTITY, (2, 1)))
    np.testing.assert_array_equal(episode.reference_root_wxyz, np.tile(IDENTITY, (2, 1)))


@pytest.mark.parametrize("field_name", ["current", "desired"])
@pytest.mark.parametrize("bad_root", [[0.0, 0.0, 0.0, 0.0], [1.000011, 0.0, 0.0, 0.0]])
def test_inspire_finite_invalid_root_is_complete_source_schema_report(
    field_name: str,
    bad_root: list[float],
) -> None:
    arrays = _inspire_arrays()
    arrays[field_name][1, 3:7] = bad_root

    report = diagnose_inspire_arrays(**arrays)

    assert report.status == "source_schema_error"
    assert report.gate_reasons == INSPIRE_GATE_REASONS
    assert report.encoder_invoked is False
    assert report.frame_count == 3
    report_key = field_name.replace("current", "robot_q_current").replace("desired", "robot_q_desired")
    assert report.finite_counts[report_key] == 108
    assert len(report.column_statistics["robot_q_current"]) == 36
    extrema = (
        report.current_root_quaternion_norm if field_name == "current" else report.desired_root_quaternion_norm
    )
    raw_norm = np.linalg.norm(bad_root)
    assert extrema.minimum == pytest.approx(min(1.0, raw_norm))
    assert extrema.maximum == pytest.approx(max(1.0, raw_norm))


def test_inspire_accepted_normalization_candidate_stays_blocked_and_preserves_raw_norms() -> None:
    arrays = _inspire_arrays()
    inside = 1.000009
    arrays["current"][1, 3:7] = (inside, 0.0, 0.0, 0.0)
    arrays["desired"][2, 3:7] = (-inside, 0.0, 0.0, 0.0)

    report = diagnose_inspire_arrays(**arrays)

    assert report.status == "blocked_unverified"
    assert report.current_root_quaternion_norm.maximum == inside
    assert report.desired_root_quaternion_norm.maximum == inside


def test_inspire_just_outside_tolerance_is_source_schema_error() -> None:
    arrays = _inspire_arrays()
    outside = 1.0000101
    arrays["desired"][0, 3:7] = (outside, 0.0, 0.0, 0.0)

    report = diagnose_inspire_arrays(**arrays)

    assert report.status == "source_schema_error"
    assert report.desired_root_quaternion_norm.maximum == outside
