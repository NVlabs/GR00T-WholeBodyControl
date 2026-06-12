"""Initial-pose ramp helpers for SONIC VLA inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gear_sonic.utils.teleop.xr_upperbody_bridge import (
    G1_CALIB_FULL_UPPER_BODY,
    G1_STANDING_UPPER_BODY,
    HAND_DOF,
    UPPER_BODY_DOF,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message

_ZERO_MOVEMENT = (0.0, 0.0, 0.0)
_FORWARD_FACING = (1.0, 0.0, 0.0)


def _as_vector(
    values: Sequence[float] | np.ndarray | None,
    *,
    size: int,
    default: float = 0.0,
    name: str,
) -> np.ndarray:
    if values is None:
        return np.full(size, default, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


def _smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def build_calib_full_ramp_targets(
    *,
    duration_s: float,
    rate_hz: float,
    start_upper_body: Sequence[float] | np.ndarray = G1_STANDING_UPPER_BODY,
    target_upper_body: Sequence[float] | np.ndarray = G1_CALIB_FULL_UPPER_BODY,
) -> list[np.ndarray]:
    """Return upper-body targets from standing to teleop CALIB_FULL."""

    start = _as_vector(
        start_upper_body,
        size=UPPER_BODY_DOF,
        name="start_upper_body",
    )
    target = _as_vector(
        target_upper_body,
        size=UPPER_BODY_DOF,
        name="target_upper_body",
    )
    if duration_s <= 0.0 or rate_hz <= 0.0:
        return [target.copy()]

    frame_count = max(2, int(round(duration_s * rate_hz)) + 1)
    targets: list[np.ndarray] = []
    for index in range(frame_count):
        alpha = _smoothstep(index / (frame_count - 1))
        target_frame = (1.0 - alpha) * start + alpha * target
        targets.append(np.asarray(target_frame, dtype=np.float32))
    return targets


def build_upper_body_ramp_messages(
    *,
    duration_s: float,
    rate_hz: float,
    start_upper_body: Sequence[float] | np.ndarray,
    target_upper_body: Sequence[float] | np.ndarray,
    start_left_hand_joints: Sequence[float] | np.ndarray | None = None,
    target_left_hand_joints: Sequence[float] | np.ndarray | None = None,
    start_right_hand_joints: Sequence[float] | np.ndarray | None = None,
    target_right_hand_joints: Sequence[float] | np.ndarray | None = None,
) -> list[bytes]:
    """Return planner messages that ramp upper body and hand joints."""

    targets = build_calib_full_ramp_targets(
        duration_s=duration_s,
        rate_hz=rate_hz,
        start_upper_body=start_upper_body,
        target_upper_body=target_upper_body,
    )
    left_start = _as_vector(
        start_left_hand_joints,
        size=HAND_DOF,
        name="start_left_hand_joints",
    )
    left_target = _as_vector(
        target_left_hand_joints,
        size=HAND_DOF,
        name="target_left_hand_joints",
    )
    right_start = _as_vector(
        start_right_hand_joints,
        size=HAND_DOF,
        name="start_right_hand_joints",
    )
    right_target = _as_vector(
        target_right_hand_joints,
        size=HAND_DOF,
        name="target_right_hand_joints",
    )
    zero_velocity = np.zeros(UPPER_BODY_DOF, dtype=np.float32)

    messages: list[bytes] = []
    denominator = max(1, len(targets) - 1)
    for index, target in enumerate(targets):
        alpha = _smoothstep(index / denominator)
        left_hand = (1.0 - alpha) * left_start + alpha * left_target
        right_hand = (1.0 - alpha) * right_start + alpha * right_target
        messages.append(
            build_planner_message(
                mode=0,
                movement=_ZERO_MOVEMENT,
                facing=_FORWARD_FACING,
                speed=0.0,
                height=-1.0,
                upper_body_position=target.tolist(),
                upper_body_velocity=zero_velocity.tolist(),
                left_hand_position=left_hand.tolist(),
                right_hand_position=right_hand.tolist(),
            )
        )
    return messages


def build_calib_full_ramp_messages(
    *,
    duration_s: float,
    rate_hz: float,
    start_upper_body: Sequence[float] | np.ndarray = G1_STANDING_UPPER_BODY,
    left_hand_joints: Sequence[float] | np.ndarray | None = None,
    right_hand_joints: Sequence[float] | np.ndarray | None = None,
) -> list[bytes]:
    """Return planner messages that ramp upper body and hands to CALIB_FULL."""

    return build_upper_body_ramp_messages(
        duration_s=duration_s,
        rate_hz=rate_hz,
        start_upper_body=start_upper_body,
        target_upper_body=G1_CALIB_FULL_UPPER_BODY,
        target_left_hand_joints=left_hand_joints,
        target_right_hand_joints=right_hand_joints,
    )


def build_calib_full_hold_message(
    *,
    left_hand_joints: Sequence[float] | np.ndarray | None = None,
    right_hand_joints: Sequence[float] | np.ndarray | None = None,
) -> bytes:
    """Return one idle planner message that holds CALIB_FULL upper body."""

    left_hand = _as_vector(
        left_hand_joints,
        size=HAND_DOF,
        name="left_hand_joints",
    )
    right_hand = _as_vector(
        right_hand_joints,
        size=HAND_DOF,
        name="right_hand_joints",
    )
    zero_velocity = np.zeros(UPPER_BODY_DOF, dtype=np.float32)
    return build_planner_message(
        mode=0,
        movement=_ZERO_MOVEMENT,
        facing=_FORWARD_FACING,
        speed=0.0,
        height=-1.0,
        upper_body_position=G1_CALIB_FULL_UPPER_BODY.tolist(),
        upper_body_velocity=zero_velocity.tolist(),
        left_hand_position=left_hand.tolist(),
        right_hand_position=right_hand.tolist(),
    )


def build_standing_ramp_messages(
    *,
    duration_s: float,
    rate_hz: float,
    start_upper_body: Sequence[float] | np.ndarray = G1_CALIB_FULL_UPPER_BODY,
) -> list[bytes]:
    """Return planner messages that ramp from CALIB_FULL back to standing."""

    return build_upper_body_ramp_messages(
        duration_s=duration_s,
        rate_hz=rate_hz,
        start_upper_body=start_upper_body,
        target_upper_body=G1_STANDING_UPPER_BODY,
    )
