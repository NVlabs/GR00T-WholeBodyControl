from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_native124_acquisition import (
    MAXIMUM_TARGET_STEP_RAD,
    Native23AcquisitionTransition,
    require_local_state_gate,
)


def _transition(*, warm: int = 2, ramp: int = 4) -> Native23AcquisitionTransition:
    return Native23AcquisitionTransition(
        measured_start_hardware=np.zeros(23),
        reference_start_native=np.zeros(23),
        lower_limit_hardware=np.full(23, -1.0),
        upper_limit_hardware=np.full(23, 1.0),
        warm_start_frames=warm,
        reference_ramp_frames=ramp,
    )


def test_local_gate_accepts_exact_boundary_and_rejects_stale_or_wrong_mode() -> None:
    require_local_state_gate(mode_machine=4, lowstate_age_ms=40.0)
    with pytest.raises(RuntimeError, match="mode_machine == 4"):
        require_local_state_gate(mode_machine=3, lowstate_age_ms=0.0)
    with pytest.raises(RuntimeError, match=r"\[0, 40\]"):
        require_local_state_gate(mode_machine=4, lowstate_age_ms=40.0001)
    with pytest.raises(RuntimeError, match=r"\[0, 40\]"):
        require_local_state_gate(mode_machine=4, lowstate_age_ms=-0.001)


def test_warm_start_holds_measured_target_then_slew_limits() -> None:
    transition = _transition()
    desired = np.full(23, 0.5)
    raw_target = np.full(23, 0.75)
    for expected_index in range(2):
        frame = transition.accept_target(
            raw_target_hardware=raw_target,
            desired_position_native=desired,
            desired_velocity_native=np.zeros(23),
            mode_machine=4,
            lowstate_age_ms=20.0,
        )
        assert frame.frame_index == expected_index
        assert frame.phase == "warm_start_hold"
        assert frame.reference_alpha == 0.0
        assert np.array_equal(frame.applied_target_hardware, np.zeros(23))

    frame = transition.accept_target(
        raw_target_hardware=raw_target,
        desired_position_native=desired,
        desired_velocity_native=np.zeros(23),
        mode_machine=4,
        lowstate_age_ms=40.0,
    )
    assert frame.phase == "reference_ramp"
    assert frame.reference_alpha == pytest.approx(0.15625)
    assert frame.maximum_target_step_rad == pytest.approx(MAXIMUM_TARGET_STEP_RAD)
    assert np.allclose(frame.applied_target_hardware, 0.005)


def test_ramp_reaches_tracking_and_includes_reference_velocity() -> None:
    transition = _transition(warm=1, ramp=2)
    desired = np.full(23, 0.4)
    desired_velocity = np.full(23, 0.1)
    raw_target = np.zeros(23)
    frames = []
    for _ in range(3):
        reference = transition.build_reference(
            desired_position_native=desired,
            desired_velocity_native=desired_velocity,
        )
        frames.append(
            transition.accept_target(
                raw_target_hardware=raw_target,
                desired_position_native=desired,
                desired_velocity_native=desired_velocity,
                mode_machine=4,
                lowstate_age_ms=0.0,
            )
        )
        assert np.array_equal(reference[0], frames[-1].reference_position_native)
        assert np.array_equal(reference[1], frames[-1].reference_velocity_native)

    assert frames[0].phase == "warm_start_hold"
    assert frames[1].phase == "reference_ramp"
    assert frames[1].reference_alpha == pytest.approx(0.5)
    assert np.all(frames[1].reference_velocity_native > desired_velocity)
    assert frames[2].phase == "tracking"
    assert frames[2].reference_alpha == 1.0
    assert np.allclose(frames[2].reference_position_native, desired)
    assert np.allclose(frames[2].reference_velocity_native, desired_velocity)


def test_raw_target_clamp_latches_fail_closed_hold() -> None:
    transition = _transition(warm=1, ramp=1)
    kwargs = {
        "desired_position_native": np.zeros(23),
        "desired_velocity_native": np.zeros(23),
        "mode_machine": 4,
        "lowstate_age_ms": 39.999,
    }
    transition.accept_target(raw_target_hardware=np.zeros(23), **kwargs)
    frame = transition.accept_target(raw_target_hardware=np.full(23, 5.0), **kwargs)
    assert frame.raw_target_limit_clamps == 23
    assert frame.raw_target_maximum_excess_rad == pytest.approx(4.0)
    assert frame.phase == "fail_closed_hold"
    assert not frame.acquisition_qualified
    assert frame.fail_closed_reason == "raw_policy_target_outside_hardware_limits"
    assert frame.maximum_target_step_rad == 0.0
    assert np.array_equal(frame.applied_target_hardware, np.zeros(23))

    later = transition.accept_target(raw_target_hardware=np.zeros(23), **kwargs)
    assert later.phase == "fail_closed_hold"
    assert not later.acquisition_qualified
    assert np.array_equal(later.applied_target_hardware, np.zeros(23))


def test_explicit_bounded_external_envelope_allows_only_small_clamp() -> None:
    transition = Native23AcquisitionTransition(
        measured_start_hardware=np.zeros(23),
        reference_start_native=np.zeros(23),
        lower_limit_hardware=np.full(23, -1.0),
        upper_limit_hardware=np.full(23, 1.0),
        warm_start_frames=1,
        reference_ramp_frames=1,
        maximum_raw_target_clamps=1,
        maximum_raw_target_excess_rad=0.06,
    )
    raw = np.zeros(23)
    raw[3] = 1.05
    frame = transition.accept_target(
        raw_target_hardware=raw,
        desired_position_native=np.zeros(23),
        desired_velocity_native=np.zeros(23),
        mode_machine=4,
        lowstate_age_ms=40.0,
    )
    assert frame.acquisition_qualified
    assert frame.raw_target_limit_clamps == 1
    assert frame.raw_target_maximum_excess_rad == pytest.approx(0.05)

    raw[4] = 1.01
    rejected = transition.accept_target(
        raw_target_hardware=raw,
        desired_position_native=np.zeros(23),
        desired_velocity_native=np.zeros(23),
        mode_machine=4,
        lowstate_age_ms=40.0,
    )
    assert not rejected.acquisition_qualified


def test_failed_local_gate_does_not_advance_transition() -> None:
    transition = _transition()
    with pytest.raises(RuntimeError):
        transition.accept_target(
            raw_target_hardware=np.zeros(23),
            desired_position_native=np.zeros(23),
            desired_velocity_native=np.zeros(23),
            mode_machine=4,
            lowstate_age_ms=41.0,
        )
    assert transition.frame_index == 0
