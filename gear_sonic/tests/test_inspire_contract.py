import numpy as np
import pytest

from gear_sonic.utils.hand_control.inspire.contract import (
    CANONICAL_LOWER_RAD,
    CANONICAL_UPPER_RAD,
    CANONICAL_VELOCITY_LIMIT_RAD_S,
    COMMAND_SCHEMA,
    JOINT_ORDER,
    ContractError,
    HandStatus,
    InspireHandCommand,
    InspireHandState,
)


def test_canonical_native6_matches_lab_hardware_joint_domain() -> None:
    assert COMMAND_SCHEMA == "inspire.hand.command.v2"
    np.testing.assert_array_equal(CANONICAL_LOWER_RAD, [0.0] * 6)
    np.testing.assert_allclose(
        CANONICAL_UPPER_RAD,
        [1.7, 1.7, 1.7, 1.7, 0.6, 1.3],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        CANONICAL_VELOCITY_LIMIT_RAD_S,
        [2.0, 2.0, 2.0, 2.0, 1.0, 1.0],
        rtol=0,
        atol=1e-7,
    )


def test_command_round_trip_keeps_native_six_dof() -> None:
    command = InspireHandCommand(3, 42, np.arange(6), np.arange(6) + 10, frame_index=17)
    decoded = InspireHandCommand.from_wire_dict(command.to_wire_dict())
    assert command.to_wire_dict()["schema"] == COMMAND_SCHEMA
    np.testing.assert_array_equal(decoded.left_q, np.arange(6, dtype=np.float32))
    assert decoded.frame_index == 17
    assert tuple(command.to_wire_dict()["joint_order"]) == JOINT_ORDER


@pytest.mark.parametrize("bad", [[0] * 5, [0] * 7, [0, 0, np.nan, 0, 0, 0]])
def test_command_rejects_non_native_or_non_finite_vectors(bad) -> None:
    with pytest.raises(ContractError):
        InspireHandCommand(1, 1, bad, np.zeros(6))


def test_state_velocity_is_all_or_nothing() -> None:
    with pytest.raises(ContractError, match="both be present"):
        InspireHandState(
            sequence=1,
            source_monotonic_ns=1,
            gateway_monotonic_ns=2,
            status=HandStatus.ARMED,
            left_q=np.zeros(6),
            right_q=np.zeros(6),
            left_dq=np.zeros(6),
        )


def test_wire_rejects_wrong_joint_order() -> None:
    payload = InspireHandCommand(1, 1, np.zeros(6), np.zeros(6)).to_wire_dict()
    payload["joint_order"] = list(reversed(JOINT_ORDER))
    with pytest.raises(ContractError, match="joint_order"):
        InspireHandCommand.from_wire_dict(payload)


def test_state_roundtrip_preserves_gateway_accepted_command() -> None:
    state = InspireHandState(
        sequence=4,
        source_monotonic_ns=20,
        gateway_monotonic_ns=25,
        status=HandStatus.ARMED,
        left_q=np.zeros(6),
        right_q=np.zeros(6),
        accepted_command_sequence=3,
        accepted_frame_index=17,
        left_command_q=np.full(6, 0.1),
        right_command_q=np.full(6, 0.2),
    )
    decoded = InspireHandState.from_wire_dict(state.to_wire_dict())
    assert decoded.accepted_command_sequence == 3
    assert decoded.accepted_frame_index == 17
    np.testing.assert_allclose(decoded.left_command_q, 0.1)


def test_state_rejects_command_sequence_without_command_vectors() -> None:
    with pytest.raises(ContractError, match="present together"):
        InspireHandState(
            sequence=4,
            source_monotonic_ns=20,
            gateway_monotonic_ns=25,
            status=HandStatus.ARMED,
            left_q=np.zeros(6),
            right_q=np.zeros(6),
            accepted_command_sequence=3,
        )
