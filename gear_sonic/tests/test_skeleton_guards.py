import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from gear_sonic.utils.teleop import skeleton_guards as guards
from gear_sonic.utils.teleop.body_joints import BodyJoint

NUM_JOINTS = len(BodyJoint)


def _pose_with_signature(reference):
    """A (24, 3) / (24, 4) pose whose bone directions reproduce ``reference``.

    Identity orientations make each signature row the bone's own direction, so
    placing every child one unit along its reference row from its parent
    reproduces the reference exactly.
    """
    positions = np.zeros((NUM_JOINTS, 3))
    for (parent, child), direction in zip(guards.BONES, reference, strict=True):
        positions[child] = positions[parent] + direction
    orientations = np.tile([0.0, 0.0, 0.0, 1.0], (NUM_JOINTS, 1))
    return positions, orientations


def _rotated_rows(reference, degrees):
    """Each row of ``reference`` turned by exactly ``degrees``, about an axis perpendicular to it."""
    rows = []
    for v in reference:
        helper = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(v, helper)
        axis /= np.linalg.norm(axis)
        rows.append(R.from_rotvec(np.radians(degrees) * axis).apply(v))
    return np.array(rows)


@pytest.mark.parametrize("source", ["pico", "quest"])
def test_each_reference_skeleton_matches_its_own_convention(source):
    match = guards.match_orientation_convention(
        *_pose_with_signature(guards.REFERENCE_SIGNATURES[source])
    )
    assert match.outcome is guards.MatchOutcome.DECIDED
    assert match.source == source


def test_two_poor_scores_far_apart_are_not_a_match(monkeypatch):
    """The margin alone would call this decided; the best score is still poor."""
    frame = guards.REFERENCE_SIGNATURES["pico"]
    monkeypatch.setitem(guards.REFERENCE_SIGNATURES, "pico", _rotated_rows(frame, 45.0))
    monkeypatch.setitem(guards.REFERENCE_SIGNATURES, "quest", _rotated_rows(frame, 135.0))
    match = guards.match_orientation_convention(*_pose_with_signature(frame))
    assert match.margin_deg >= guards.MIN_MARGIN_DEG
    assert match.outcome is guards.MatchOutcome.NO_MATCH
    assert match.source is None


def test_a_few_non_finite_joints_are_left_out_of_the_score():
    positions, orientations = _pose_with_signature(guards.REFERENCE_SIGNATURES["pico"])
    positions[BodyJoint.LEFT_FOOT] = np.nan
    orientations[BodyJoint.NECK] = np.nan
    match = guards.match_orientation_convention(positions, orientations)
    assert match.outcome is guards.MatchOutcome.DECIDED
    assert match.source == "pico"
    assert all(np.isfinite(v) for v in match.scores.values())


def test_too_few_measurable_bones_is_unusable():
    positions, orientations = _pose_with_signature(guards.REFERENCE_SIGNATURES["pico"])
    lost = len(guards.BONES) - guards.MIN_USABLE_BONES + 1
    for parent, _child in guards.BONES[:lost]:
        orientations[parent] = np.nan
    match = guards.match_orientation_convention(positions, orientations)
    assert match.outcome is guards.MatchOutcome.UNUSABLE


def test_quat_angle_ignores_quaternion_sign():
    rng = np.random.default_rng(0)
    q = R.random(50, random_state=rng).as_quat()
    ref = R.random(random_state=rng).as_quat()
    expected = np.degrees((R.from_quat(ref).inv() * R.from_quat(q)).magnitude())
    np.testing.assert_allclose(guards._quat_angle_deg(-q, ref), expected, atol=1e-9)


def test_window_full_tracks_the_buffer():
    positions = np.zeros((NUM_JOINTS, 3))
    orientations = np.tile([0.0, 0.0, 0.0, 1.0], (NUM_JOINTS, 1))
    detector = guards.TrackersDisconnectedDetector()
    for _ in range(guards.FROZEN_WINDOW - 1):
        detector.push(positions, orientations)
    assert not detector.window_full
    detector.push(positions, orientations)
    assert detector.window_full
    detector.reset()
    assert not detector.window_full
