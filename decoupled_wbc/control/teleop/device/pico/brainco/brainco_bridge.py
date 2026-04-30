"""Bridges xrobotoolkit hand tracking into BraincoController's shared arrays.

xrobotoolkit's `get_{left,right}_hand_tracking_state()` returns a (26, 7) array
of `[x, y, z, qx, qy, qz, qw]` rows in the OpenXR world frame. The brainco
retargeter (ported from xr_teleoperate) consumes a flat (25, 3) hand-keypoint
cloud expressed in the unitree-hand (brainco URDF base_link) frame, indexed as

    0          : wrist
    1..4       : thumb metacarpal -> tip
    5..9       : index  metacarpal -> tip
    10..14     : middle metacarpal -> tip
    15..19     : ring   metacarpal -> tip
    20..24     : little metacarpal -> tip

== Coordinate-frame chain ==

We must produce the same wrist-relative vectors that xr_teleoperate feeds to
dex-retargeting. Their pipeline does:

    pos_world_openxr (vuer)
        -> R_ROBOT_OPENXR @ pos                        (basis: OpenXR -> robot)
        -> inv(wrist_T_robot) @ pos                    (express in wrist frame)
        -> T_TO_UNITREE_HAND @ pos                     (basis: robot -> unitree-hand)

In xr_teleoperate the "arm pose" is literally `hand_data[0:16]` -- the wrist
joint pose from hand tracking (see televuer/src/televuer/televuer.py:278).
xrobotoolkit's wrist quaternion is the same OpenXR EXT_hand_tracking joint
pose, so we use it directly.

For dex-retargeting only the differences `tip - wrist` matter. Substituting
`wrist_T_robot.R = R_ROBOT_OPENXR @ wrist_R_openxr @ R_OPENXR_ROBOT` and
simplifying, the chain collapses to a single rotation applied to the
wrist-local OpenXR vector:

    delta_unitree = R_HAND_TO_UNITREE @ wrist_R_openxr.T @ delta_world_openxr

    where R_HAND_TO_UNITREE = T_TO_UNITREE_HAND @ R_ROBOT_OPENXR (precomputed).

Skipping the R_ROBOT_OPENXR step (an earlier version of this file did) gives
hand-frame vectors in the *OpenXR* basis convention, not the *robot* basis the
brainco URDF expects -- the retargeter then converges to a 90deg-off pose.

== Data flow (single xrobotoolkit_sdk connection, no extra subprocess) ==

    PicoStreamer.xr_client.get_hand_tracking_state(...)   [main process, sync]
        -> hand_state_to_unitree_keypoints(...)
            -> push_to_shared(...)
                -> BraincoController control loop          [subprocess]
                    -> rt/brainco/{left,right}/cmd         [DDS]

== Joint layout ==

Per Khronos XR_EXT_hand_tracking (which the PICO PXREA service implements),
the 26-joint enum is fixed:

    0       XR_HAND_JOINT_PALM_EXT
    1       XR_HAND_JOINT_WRIST_EXT
    2..5    thumb metacarpal -> tip
    6..10   index  (5 joints, includes intermediate)
    11..15  middle
    16..20  ring
    21..25  little

We drop PALM (row 0) so the resulting 25-joint layout matches
xr_teleoperate/dex_retargeting: WRIST at 0, fingertips at 4/9/14/19/24.
"""

from multiprocessing import Array

import numpy as np
from scipy.spatial.transform import Rotation as R

# 3x3 rotation matrices copied verbatim from
# xr_teleoperate/teleop/televuer/src/televuer/tv_wrapper.py.
T_TO_UNITREE_HAND_3x3 = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ]
)
R_ROBOT_OPENXR = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

# Composed basis change: rotates a vector expressed in the wrist-local OpenXR
# hand-joint frame into the unitree-hand (brainco URDF base_link) frame. This
# is the one rotation we need because xr_teleoperate's chain collapses to it
# when only wrist-relative differences are kept (see module docstring).
R_HAND_TO_UNITREE = T_TO_UNITREE_HAND_3x3 @ R_ROBOT_OPENXR

NUM_RAW_JOINTS = 26  # XR_HAND_JOINT_COUNT_EXT
NUM_KEEP_JOINTS = 25  # 26 minus PALM
PALM_INDEX = 0  # XR_HAND_JOINT_PALM_EXT
WRIST_INDEX_AFTER_DROP = 0  # XR_HAND_JOINT_WRIST_EXT after dropping PALM


def hand_state_to_unitree_keypoints(hand_state) -> np.ndarray:
    """Convert a (26, 7) xrobotoolkit hand-tracking state into a (25, 3)
    keypoint array in the unitree-hand (brainco URDF base_link) frame,
    anchored at the wrist.

    Parameters
    ----------
    hand_state : (26, 7) ndarray or None
        Rows are [x, y, z, qx, qy, qz, qw] following XR_EXT_hand_tracking.
        Pass None when xrobotoolkit reports the hand as inactive; this
        returns a (25, 3) zero array (which the controller treats as 'no
        command').
    """
    if hand_state is None:
        return np.zeros((NUM_KEEP_JOINTS, 3))

    arr = np.asarray(hand_state, dtype=np.float64)
    if arr.shape != (NUM_RAW_JOINTS, 7):
        raise ValueError(
            f"Expected hand_state shape ({NUM_RAW_JOINTS}, 7), got {arr.shape}"
        )

    keep = np.delete(arr, PALM_INDEX, axis=0)  # (25, 7), now WRIST first
    pos_world = keep[:, :3]
    quat_world = keep[:, 3:]

    wrist_quat = quat_world[WRIST_INDEX_AFTER_DROP]
    if not np.any(wrist_quat):
        wrist_quat = np.array([0.0, 0.0, 0.0, 1.0])

    # 1) Express keypoints in the wrist's local OpenXR hand-joint frame:
    #        delta_world_openxr = pos_world - wrist_pos
    #        v_wrist_local      = wrist_R.T @ delta_world_openxr
    #    For an (N, 3) row-major array this is `delta @ wrist_R`.
    wrist_R = R.from_quat(wrist_quat).as_matrix()
    pos_local = pos_world - pos_world[WRIST_INDEX_AFTER_DROP]
    pos_wrist_frame = pos_local @ wrist_R

    # 2) Composed basis change: OpenXR hand-joint local -> unitree-hand
    #    (= R_ROBOT_OPENXR followed by T_TO_UNITREE_HAND, see module
    #    docstring). For row-major points: `points @ R.T`.
    pos_unitree = pos_wrist_frame @ R_HAND_TO_UNITREE.T
    return pos_unitree


def push_keypoints_to_shared(
    shared_array: Array, keypoints: np.ndarray
) -> None:
    """Write a (25, 3) keypoint cloud into a shared Array('d', 75)."""
    flat = np.asarray(keypoints, dtype=np.float64).reshape(-1)
    if flat.size != 75:
        raise ValueError(f"keypoints must flatten to 75 elements, got {flat.size}")
    with shared_array.get_lock():
        shared_array[:] = flat


def make_brainco_shared_arrays() -> tuple[Array, Array]:
    """Allocate the (75,) shared arrays the controller subprocess reads from.

    Caller is responsible for keeping these alive for the lifetime of the
    BraincoController process.
    """
    left = Array("d", 75, lock=True)
    right = Array("d", 75, lock=True)
    return left, right
