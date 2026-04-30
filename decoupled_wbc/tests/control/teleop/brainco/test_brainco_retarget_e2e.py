"""End-to-end retargeting test.

Exercises the full pipeline: synthetic xrobotoolkit (26, 7) hand state ->
hand_state_to_unitree_keypoints -> dex_retargeting -> brainco-ordered motor
angles. Skipped automatically on dev boxes without `dex_retargeting`.

Catches regressions in:
  - YAML normalisation (DexPilot vs vector key suffixes)
  - the (25, 3) keypoint convention vs `target_link_human_indices`
  - URDF loading
  - the `left_dex_to_hardware` / `right_dex_to_hardware` permutation
  - sane retargeted joint-angle ranges

These tests run dex_retargeting's optimiser, so each takes ~50-200 ms.
"""

import numpy as np
import pytest

from decoupled_wbc.control.teleop.device.pico.brainco.brainco_bridge import (
    NUM_RAW_JOINTS,
    hand_state_to_unitree_keypoints,
)

dex_retargeting = pytest.importorskip("dex_retargeting")

from decoupled_wbc.control.teleop.device.pico.brainco.hand_retargeting import (  # noqa: E402
    BraincoHandRetargeting,
)

# Joint range limits per the brainco API contract (radians, post-retarget,
# pre-normalisation). See robot_hand_brainco._control_loop comments.
JOINT_RANGES = {
    0: (0.0, 1.52),  # thumb
    1: (0.0, 1.05),  # thumb-aux
    2: (0.0, 1.47),  # index proximal
    3: (0.0, 1.47),  # middle proximal
    4: (0.0, 1.47),  # ring proximal
    5: (0.0, 1.47),  # pinky proximal
}


@pytest.fixture(scope="module")
def retargeter():
    """Module-scoped so we only build the URDF + optimisers once."""
    return BraincoHandRetargeting()


def _open_hand_state():
    """Synthetic state: wrist at origin, fingers straight along OpenXR +X
    (distal direction), thumb out to the side. Identity wrist orientation.
    """
    state = np.zeros((NUM_RAW_JOINTS, 7))
    state[:, 6] = 1.0  # qw = 1
    # Wrist
    state[1, :3] = [0.0, 0.0, 0.0]
    # Thumb: 4 joints (raw 2..5), splayed out toward +Y
    for k, x in enumerate([0.02, 0.045, 0.07, 0.09]):
        state[2 + k, :3] = [x, 0.02 + 0.015 * k, 0.0]
    # Index..little: 5 joints each (raw 6..10, 11..15, 16..20, 21..25)
    # Each finger straight along +X with small Y offset for separation.
    for finger_idx, base in enumerate([6, 11, 16, 21]):
        y_off = 0.01 - finger_idx * 0.012
        for k in range(5):
            state[base + k, :3] = [0.04 + 0.025 * k, y_off, 0.0]
    return state


def _closed_hand_state():
    """Synthetic state: every fingertip pulled close to the thumb tip
    (positions collapse near the wrist). Identity orientations.
    """
    state = _open_hand_state()
    thumb_tip = np.array([0.04, 0.02, 0.02])
    for tip_raw in (5, 10, 15, 20, 25):
        # Pull each fingertip to thumb_tip; intermediate joints toward wrist.
        state[tip_raw, :3] = thumb_tip
    # Pull intermediate joints inward
    for finger_idx, base in enumerate([6, 11, 16, 21]):
        y_off = 0.01 - finger_idx * 0.012
        for k in range(5):
            t = k / 4.0
            x = 0.0 * (1 - t) + thumb_tip[0] * t
            y = y_off * (1 - t) + thumb_tip[1] * t
            state[base + k, :3] = [x, y, 0.0]
    return state


def _retarget(retargeter, state):
    pts = hand_state_to_unitree_keypoints(state)
    ref_l = pts[retargeter.left_indices[1, :]] - pts[retargeter.left_indices[0, :]]
    q_full = retargeter.left_retargeting.retarget(ref_l)
    q_motor = q_full[retargeter.left_dex_to_hardware]
    return pts, q_full, q_motor


def test_retargeter_loads_with_brainco_yaml(retargeter):
    """Asset wiring: brainco URDFs + YAML normalise + load through dex_retargeting."""
    assert len(retargeter.left_joint_names) == 11
    assert len(retargeter.right_joint_names) == 11
    assert len(retargeter.left_dex_to_hardware) == 6
    assert len(retargeter.right_dex_to_hardware) == 6
    # left_indices is (2, N) where N is the number of human-index pairs the
    # retargeter consumes per call (15 for DexPilot mode in this YAML).
    assert retargeter.left_indices.shape == (2, 15)
    # All indices must address a 25-joint cloud (post-PALM-drop).
    assert retargeter.left_indices.max() < 25
    assert retargeter.left_indices.min() >= 0


def test_retarget_runs_on_synthetic_input(retargeter):
    """Smoke: retargeter accepts the bridge's output and returns finite values
    of the right shape."""
    _, q_full, q_motor = _retarget(retargeter, _open_hand_state())
    assert q_full.shape == (11,)
    assert q_motor.shape == (6,)
    assert np.all(np.isfinite(q_motor))


def test_retarget_responds_to_input_changes(retargeter):
    """Two visibly different hand poses should produce visibly different
    motor commands. This guards against the retargeter being stuck at a
    constant output (which would happen if the YAML normalisation produced
    a degenerate config).

    We deliberately don't assert WHICH way each motor moves -- the URDF's
    joint-zero pose may not correspond to our synthetic 'open' geometry, so
    a directional assertion would be hardware-correlated.
    """
    # Build a fresh retargeter so the brainco.yml `low_pass_alpha: 0.2`
    # filter doesn't carry state between the two calls.
    fresh = BraincoHandRetargeting()
    pts_o = hand_state_to_unitree_keypoints(_open_hand_state())
    ref_o = pts_o[fresh.left_indices[1, :]] - pts_o[fresh.left_indices[0, :]]
    q_open = fresh.left_retargeting.retarget(ref_o)[fresh.left_dex_to_hardware]

    fresh2 = BraincoHandRetargeting()
    pts_c = hand_state_to_unitree_keypoints(_closed_hand_state())
    ref_c = pts_c[fresh2.left_indices[1, :]] - pts_c[fresh2.left_indices[0, :]]
    q_closed = fresh2.left_retargeting.retarget(ref_c)[fresh2.left_dex_to_hardware]

    diff = np.linalg.norm(q_open - q_closed)
    assert diff > 0.3, (
        f"retargeter produced near-identical motor commands for "
        f"open ({q_open}) and closed ({q_closed}) inputs (diff={diff:.3f}); "
        "something is wrong with the retargeting config or input mapping."
    )


def test_motor_angles_within_brainco_ranges(retargeter):
    """Retargeted motor angles should fall within the published brainco API
    ranges for both extreme inputs. dex_retargeting uses the URDF joint
    limits, so this test mostly guards against the URDF/YAML drifting away
    from the brainco hardware spec.
    """
    for state in (_open_hand_state(), _closed_hand_state()):
        _, _, q_motor = _retarget(retargeter, state)
        for motor_id, q in enumerate(q_motor):
            lo, hi = JOINT_RANGES[motor_id]
            # Allow small overshoot at the limits (5e-3 rad ~= 0.3 deg) to
            # absorb the optimiser's tolerance. _normalize() clips to [0, 1]
            # before publishing so a small overshoot here is not a runtime
            # hazard, only a regression signal.
            assert lo - 5e-3 <= q <= hi + 5e-3, (
                f"motor {motor_id} q={q:.4f} outside [{lo}, {hi}]"
            )
