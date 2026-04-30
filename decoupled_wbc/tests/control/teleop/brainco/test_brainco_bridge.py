"""Unit tests for the brainco bridge math.

These tests are hardware-free: they exercise the (26, 7) -> (25, 3) frame
transform with synthetic xrobotoolkit hand-tracking states. They guard against
regressions in the chain that turns OpenXR hand-joint quaternions into the
unitree-hand-frame keypoints that dex_retargeting expects.

Run:
    pytest decoupled_wbc/tests/control/teleop/brainco/
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from decoupled_wbc.control.teleop.device.pico.brainco.brainco_bridge import (
    NUM_KEEP_JOINTS,
    NUM_RAW_JOINTS,
    PALM_INDEX,
    R_HAND_TO_UNITREE,
    R_ROBOT_OPENXR,
    T_TO_UNITREE_HAND_3x3,
    WRIST_INDEX_AFTER_DROP,
    hand_state_to_unitree_keypoints,
    make_brainco_shared_arrays,
    push_keypoints_to_shared,
)


# ---------------------------------------------------------------------------
# Pure-math sanity
# ---------------------------------------------------------------------------


def test_R_HAND_TO_UNITREE_is_proper_rotation():
    """R_HAND_TO_UNITREE must be orthogonal with det = +1."""
    assert np.allclose(R_HAND_TO_UNITREE @ R_HAND_TO_UNITREE.T, np.eye(3))
    assert np.isclose(np.linalg.det(R_HAND_TO_UNITREE), 1.0)


def test_R_HAND_TO_UNITREE_matches_composition():
    """Documented composition: T_TO_UNITREE_HAND @ R_ROBOT_OPENXR."""
    expected = T_TO_UNITREE_HAND_3x3 @ R_ROBOT_OPENXR
    assert np.array_equal(R_HAND_TO_UNITREE, expected)


def test_R_HAND_TO_UNITREE_exact_value():
    """Locks in the specific cyclic-permutation rotation we derived.
    Changing this matrix without an accompanying change to the math in
    `brainco_bridge.hand_state_to_unitree_keypoints` would silently break
    retargeting.
    """
    expected = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert np.array_equal(R_HAND_TO_UNITREE, expected)


# ---------------------------------------------------------------------------
# hand_state_to_unitree_keypoints
# ---------------------------------------------------------------------------


def _identity_state(num_joints=NUM_RAW_JOINTS, distal_offset=0.1):
    """Build a synthetic (26, 7) state where every joint sits along the
    OpenXR +X axis (distal direction) at increasing distance from origin,
    with identity orientation.
    """
    state = np.zeros((num_joints, 7))
    state[:, 0] = np.arange(num_joints) * distal_offset  # x increases
    state[:, 6] = 1.0  # qw = 1, so quaternion = identity
    return state


def test_keypoints_shape_and_finite():
    state = _identity_state()
    out = hand_state_to_unitree_keypoints(state)
    assert out.shape == (NUM_KEEP_JOINTS, 3)
    assert np.all(np.isfinite(out))


def test_none_input_returns_zeros():
    out = hand_state_to_unitree_keypoints(None)
    assert out.shape == (NUM_KEEP_JOINTS, 3)
    assert np.all(out == 0.0)


def test_wrong_shape_raises():
    bad = np.zeros((10, 7))
    with pytest.raises(ValueError, match="Expected hand_state shape"):
        hand_state_to_unitree_keypoints(bad)


def test_wrist_anchored_at_origin():
    """After dropping PALM, the WRIST is at index 0 of the kept array, and the
    transform expresses everything relative to it."""
    state = _identity_state()
    out = hand_state_to_unitree_keypoints(state)
    assert np.allclose(out[WRIST_INDEX_AFTER_DROP], 0.0)


def test_palm_is_dropped_not_wrist():
    """If PALM (raw index 0) and WRIST (raw index 1) have distinct positions,
    the wrist position should drive the anchor — not the palm position. We do
    this by giving palm an unreachable coordinate; if the bridge accidentally
    used palm as the anchor, the wrist would land at -palm in the output."""
    state = _identity_state()
    state[PALM_INDEX, :3] = [99.0, 99.0, 99.0]  # garbage palm position
    state[1, :3] = [0.0, 0.0, 0.0]  # wrist at origin
    state[2, :3] = [0.0, 0.0, 0.1]  # thumb metacarpal slightly distal

    out = hand_state_to_unitree_keypoints(state)
    # If palm had been the anchor, wrist would be at -[99,99,99] != 0.
    assert np.allclose(out[WRIST_INDEX_AFTER_DROP], 0.0)
    # Other points are at finite (non-99) magnitudes.
    assert np.linalg.norm(out[1]) < 1.0


def test_axis_mapping_with_identity_wrist():
    """With identity wrist orientation, R_HAND_TO_UNITREE alone determines
    where each OpenXR axis lands in unitree-hand basis. Verifies the matrix
    matches its documented effect on basis vectors.
    """
    state = np.zeros((NUM_RAW_JOINTS, 7))
    state[:, 6] = 1.0  # all identity rotations
    state[1] = [0.0, 0.0, 0.0, 0, 0, 0, 1]  # wrist at origin
    state[2] = [1.0, 0.0, 0.0, 0, 0, 0, 1]  # +X (OpenXR distal)
    state[3] = [0.0, 1.0, 0.0, 0, 0, 0, 1]  # +Y (OpenXR dorsal)
    state[4] = [0.0, 0.0, 1.0, 0, 0, 0, 1]  # +Z (OpenXR RH-complete)

    out = hand_state_to_unitree_keypoints(state)
    # After dropping PALM (raw 0), wrist is kept[0], and raw 2/3/4 -> kept 1/2/3.
    # Documented mapping under identity wrist:
    #   OpenXR +X -> unitree (0, 0, 1)
    #   OpenXR +Y -> unitree (1, 0, 0)
    #   OpenXR +Z -> unitree (0, 1, 0)
    assert np.allclose(out[1], [0, 0, 1])
    assert np.allclose(out[2], [1, 0, 0])
    assert np.allclose(out[3], [0, 1, 0])


def test_rotated_wrist_reanchors_frame():
    """If the wrist's orientation rotates 90deg about world-Z, a fingertip at
    world +X should land in the same unitree-frame position as it would for an
    identity wrist + a fingertip at world +Y. I.e. retargeting input is
    invariant to overall hand pose in space."""
    # Case A: identity wrist, fingertip at world +X.
    state_a = np.zeros((NUM_RAW_JOINTS, 7))
    state_a[:, 6] = 1.0
    state_a[1] = [0, 0, 0, 0, 0, 0, 1]
    state_a[2] = [1, 0, 0, 0, 0, 0, 1]
    out_a = hand_state_to_unitree_keypoints(state_a)

    # Case B: wrist rotated 90deg about Z, fingertip at world +Y.
    quat_z90 = R.from_euler("z", 90, degrees=True).as_quat()  # [x,y,z,w]
    state_b = np.zeros((NUM_RAW_JOINTS, 7))
    state_b[:, 6] = 1.0
    state_b[1] = [0, 0, 0, *quat_z90]
    state_b[2] = [0, 1, 0, 0, 0, 0, 1]
    out_b = hand_state_to_unitree_keypoints(state_b)

    assert np.allclose(out_a[1], out_b[1], atol=1e-9)


def test_distance_preserved_under_transform():
    """The transform is rigid (rotation only), so |tip - wrist| is invariant.
    A 10 cm fingertip in the world stays 10 cm in the unitree frame.
    """
    state = _identity_state(distal_offset=0.0)
    state[1, :3] = [0.0, 0.0, 0.0]      # wrist
    state[5, :3] = [0.10, 0.02, -0.03]  # an index joint
    state[1, 6] = 1.0
    state[5, 6] = 1.0

    out = hand_state_to_unitree_keypoints(state)
    expected_dist = np.linalg.norm([0.10, 0.02, -0.03])
    actual_dist = np.linalg.norm(out[4])  # raw 5 -> kept 4
    assert np.isclose(expected_dist, actual_dist)


# ---------------------------------------------------------------------------
# Shared-array helpers
# ---------------------------------------------------------------------------


def test_make_brainco_shared_arrays_size():
    left, right = make_brainco_shared_arrays()
    assert len(left) == 75
    assert len(right) == 75


def test_push_keypoints_to_shared_roundtrip():
    left, _ = make_brainco_shared_arrays()
    pts = np.arange(75, dtype=np.float64).reshape(25, 3)
    push_keypoints_to_shared(left, pts)
    assert np.allclose(np.array(left[:]).reshape(25, 3), pts)


def test_push_keypoints_wrong_size_raises():
    left, _ = make_brainco_shared_arrays()
    with pytest.raises(ValueError, match="must flatten to 75"):
        push_keypoints_to_shared(left, np.zeros((10, 3)))
