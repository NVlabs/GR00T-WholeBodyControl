import numpy as np

from gear_sonic.utils.inference.initial_pose_ramp import (
    build_calib_full_hold_message,
    build_calib_full_ramp_messages,
    build_calib_full_ramp_targets,
    build_standing_ramp_messages,
)
from gear_sonic.utils.teleop.xr_upperbody_bridge import (
    G1_CALIB_FULL_UPPER_BODY,
    G1_STANDING_UPPER_BODY,
    unpack_bridge_message,
)


def test_calib_full_ramp_targets_start_at_standing_and_end_at_calib_full() -> None:
    targets = build_calib_full_ramp_targets(duration_s=1.0, rate_hz=2.0)

    assert len(targets) == 3
    assert np.allclose(targets[0], G1_STANDING_UPPER_BODY)
    assert np.allclose(targets[1], 0.5 * G1_STANDING_UPPER_BODY)
    assert np.allclose(targets[-1], G1_CALIB_FULL_UPPER_BODY)


def test_calib_full_ramp_messages_include_planner_upper_body_and_hand_ramp() -> None:
    left_hand = np.arange(7, dtype=np.float32) * 0.1
    right_hand = -left_hand

    messages = build_calib_full_ramp_messages(
        duration_s=1.0,
        rate_hz=2.0,
        left_hand_joints=left_hand,
        right_hand_joints=right_hand,
    )
    first = unpack_bridge_message(messages[0], topic="planner")
    last = unpack_bridge_message(messages[-1], topic="planner")

    assert int(last["mode"][0]) == 0
    assert np.allclose(first["upper_body_position"], G1_STANDING_UPPER_BODY)
    assert np.allclose(last["upper_body_position"], G1_CALIB_FULL_UPPER_BODY)
    assert np.allclose(first["upper_body_velocity"], np.zeros(17, dtype=np.float32))
    assert np.allclose(last["upper_body_velocity"], np.zeros(17, dtype=np.float32))
    assert np.allclose(first["left_hand_joints"], np.zeros(7, dtype=np.float32))
    assert np.allclose(last["left_hand_joints"], left_hand)
    assert np.allclose(first["right_hand_joints"], np.zeros(7, dtype=np.float32))
    assert np.allclose(last["right_hand_joints"], right_hand)


def test_standing_ramp_messages_return_from_calib_full_to_standing() -> None:
    messages = build_standing_ramp_messages(duration_s=1.0, rate_hz=2.0)

    first = unpack_bridge_message(messages[0], topic="planner")
    last = unpack_bridge_message(messages[-1], topic="planner")

    assert np.allclose(first["upper_body_position"], G1_CALIB_FULL_UPPER_BODY)
    assert np.allclose(last["upper_body_position"], G1_STANDING_UPPER_BODY)
    assert np.allclose(last["left_hand_joints"], np.zeros(7, dtype=np.float32))
    assert np.allclose(last["right_hand_joints"], np.zeros(7, dtype=np.float32))


def test_calib_full_hold_message_refreshes_planner_upper_body_control() -> None:
    left_hand = np.arange(7, dtype=np.float32) * 0.1
    right_hand = -left_hand

    message = build_calib_full_hold_message(
        left_hand_joints=left_hand,
        right_hand_joints=right_hand,
    )

    decoded = unpack_bridge_message(message, topic="planner")

    assert int(decoded["mode"][0]) == 0
    assert np.allclose(decoded["movement"], np.zeros(3, dtype=np.float32))
    assert np.allclose(decoded["upper_body_position"], G1_CALIB_FULL_UPPER_BODY)
    assert np.allclose(decoded["upper_body_velocity"], np.zeros(17, dtype=np.float32))
    assert np.allclose(decoded["left_hand_joints"], left_hand)
    assert np.allclose(decoded["right_hand_joints"], right_hand)
