"""Single source of truth for the Inspire RH56 6-DOF dexterous hand.

The Inspire hand exposes 6 actuated drives. Their order here matches the
``angle_set`` / ``angle_act`` arrays used by the Inspire DDS SDK
(``inspire_hand_ctrl`` / ``inspire_hand_state``):

    index 0: little finger (pinky)
    index 1: ring finger
    index 2: middle finger
    index 3: index finger
    index 4: thumb bend (proximal pitch)
    index 5: thumb rotation (proximal yaw)

The SDK speaks in normalized integer drive units in ``[0, 1000]`` where
``1000`` corresponds to the fully *open* pose and ``0`` to the fully *closed*
pose (confirmed against the Inspire FTP user manual, registers ANGLE_SET @ 1486
and ANGLE_ACT @ 1546; a value of ``-1`` means "no action" and is never emitted
here). The rest of the control stack speaks in radians on the robot model
joints, so this module owns the linear mapping between the two.

We model each drive as a single "flexion-from-open" joint: ``q = 0`` is fully
open and ``q = Q_CLOSED`` is fully closed, where ``Q_CLOSED`` is the physical
travel from the manual:

    fingers (pinky/ring/middle/index): 20deg..176deg  -> 156deg = 2.72 rad
    thumb bend (proximal pitch):       -13deg..70deg   ->  83deg = 1.45 rad
    thumb rotation (proximal yaw):      90deg..165deg   ->  75deg = 1.31 rad

These joint names and limits must stay in sync with
``g1_29dof_with_inspire_hand.urdf`` and ``G1SupplementalInfo``.
"""

from __future__ import annotations

import numpy as np

# Number of actuated drives on a single Inspire hand.
NUM_INSPIRE_DOF = 6

# Drive value bounds used by the Inspire SDK (int16, normalized).
ANGLE_MIN = 0
ANGLE_MAX = 1000

# Ordered joint suffixes, matching the SDK angle_set index order above.
JOINT_ORDER = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rot",
)


def joint_names(is_left: bool) -> list[str]:
    """Return the 6 URDF joint names for a hand, in SDK drive order."""
    side = "left" if is_left else "right"
    return [f"{side}_hand_{name}_joint" for name in JOINT_ORDER]


# Per-joint radian range, ordered to match JOINT_ORDER. Positive angle is
# flexion (closing). q == Q_OPEN maps to drive value 1000 (open), q == Q_CLOSED
# maps to drive value 0 (closed). These mirror the URDF joint limits.
Q_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
Q_CLOSED = np.array([2.72, 2.72, 2.72, 2.72, 1.45, 1.31])

# Per-joint [lower, upper] limits in radians, for the robot model / URDF.
JOINT_LIMITS = {
    name: [float(min(o, c)), float(max(o, c))]
    for name, o, c in zip(JOINT_ORDER, Q_OPEN, Q_CLOSED)
}


def rad_to_drive(q: np.ndarray) -> list[int]:
    """Map joint angles (rad, SDK order) to Inspire drive units [0, 1000]."""
    q = np.asarray(q, dtype=np.float64).reshape(NUM_INSPIRE_DOF)
    span = Q_CLOSED - Q_OPEN
    # Avoid divide-by-zero for any degenerate joint.
    span = np.where(span == 0.0, 1.0, span)
    drive = ANGLE_MAX * (Q_CLOSED - q) / span
    drive = np.clip(drive, ANGLE_MIN, ANGLE_MAX)
    return [int(round(v)) for v in drive]


def drive_to_rad(drive) -> np.ndarray:
    """Map Inspire drive units [0, 1000] (SDK order) to joint angles (rad)."""
    drive = np.asarray(drive, dtype=np.float64).reshape(NUM_INSPIRE_DOF)
    drive = np.clip(drive, ANGLE_MIN, ANGLE_MAX)
    span = Q_CLOSED - Q_OPEN
    return Q_CLOSED - (drive / ANGLE_MAX) * span
