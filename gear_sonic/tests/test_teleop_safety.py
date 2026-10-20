"""Unit tests for gear_sonic/scripts/teleop_safety.py (webcam teleop safety layer)."""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from teleop_safety import (  # noqa: E402
    FULL_BODY_KEYPOINTS,
    SMPL_LEG_JOINTS,
    UPPER_BODY_KEYPOINTS,
    SafetyLimits,
    SafetyState,
    SafetyStateMachine,
    blend_frames,
    clamp_frame_step,
    keypoints_visible,
    pelvis_tilt_deg,
    pose_is_sane,
    quat_angle_wxyz,
    sample_is_jump,
    zero_legs_smpl,
)

TICK = 1.0 / 50.0


def make_frame(pose_val=0.0, quat=None, joints_val=0.0, jp_val=0.0):
    return {
        "smpl_pose": np.full((21, 3), pose_val, dtype=np.float32),
        "smpl_joints": np.full((24, 3), joints_val, dtype=np.float32),
        "body_quat": np.array(quat if quat is not None else [1, 0, 0, 0], dtype=np.float32),
        "joint_pos": np.full(29, jp_val, dtype=np.float64),
    }


NEUTRAL = make_frame()
LIMITS = SafetyLimits()


def max_pose_delta(a, b):
    from scipy.spatial.transform import Rotation as R
    rel = R.from_rotvec(a["smpl_pose"].reshape(-1, 3)).inv() * R.from_rotvec(
        b["smpl_pose"].reshape(-1, 3)
    )
    return float(np.linalg.norm(rel.as_rotvec(), axis=-1).max())


def assert_step_within_limits(prev, curr, limits=LIMITS, tol=1e-5):
    assert max_pose_delta(prev, curr) <= limits.max_pose_step_rad + tol
    assert quat_angle_wxyz(prev["body_quat"], curr["body_quat"]) <= (
        limits.max_quat_step_rad + tol
    )
    assert np.abs(curr["joint_pos"] - prev["joint_pos"]).max() <= (
        limits.max_joint_pos_step_rad + tol
    )
    assert np.linalg.norm(
        curr["smpl_joints"] - prev["smpl_joints"], axis=-1
    ).max() <= limits.max_joints_step_m + tol


# ---------------------------------------------------------------------------
#  Clamping
# ---------------------------------------------------------------------------


class TestClamp:
    def test_small_step_passes_through(self):
        target = make_frame(pose_val=0.05, joints_val=0.01, jp_val=0.05)
        out = clamp_frame_step(NEUTRAL, target, LIMITS)
        np.testing.assert_allclose(out["smpl_pose"], target["smpl_pose"], atol=1e-5)
        np.testing.assert_allclose(out["joint_pos"], target["joint_pos"], atol=1e-9)

    def test_large_step_is_clamped(self):
        target = make_frame(pose_val=1.0, joints_val=1.0, jp_val=2.0)
        out = clamp_frame_step(NEUTRAL, target, LIMITS)
        assert_step_within_limits(NEUTRAL, out)
        # And it must actually move toward the target
        assert max_pose_delta(NEUTRAL, out) > 0.9 * LIMITS.max_pose_step_rad

    def test_quat_step_clamped(self):
        # 90 deg pelvis flip in one tick must be limited to max_quat_step_rad
        target = make_frame(quat=[np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
        out = clamp_frame_step(NEUTRAL, target, LIMITS)
        ang = quat_angle_wxyz(NEUTRAL["body_quat"], out["body_quat"])
        assert ang == pytest.approx(LIMITS.max_quat_step_rad, abs=1e-4)

    def test_converges_to_target(self):
        target = make_frame(pose_val=0.8, jp_val=1.0)
        curr = NEUTRAL
        for _ in range(300):
            curr = clamp_frame_step(curr, target, LIMITS)
        np.testing.assert_allclose(curr["smpl_pose"], target["smpl_pose"], atol=1e-3)
        np.testing.assert_allclose(curr["joint_pos"], target["joint_pos"], atol=1e-6)


# ---------------------------------------------------------------------------
#  Validity checks
# ---------------------------------------------------------------------------


class TestValidity:
    def test_sane_pose(self):
        ok, _ = pose_is_sane(NEUTRAL["smpl_pose"], NEUTRAL["body_quat"], LIMITS)
        assert ok

    def test_nan_rejected(self):
        pose = NEUTRAL["smpl_pose"].copy()
        pose[3, 1] = np.nan
        ok, reason = pose_is_sane(pose, NEUTRAL["body_quat"], LIMITS)
        assert not ok and "non_finite" in reason

    def test_extreme_axis_angle_rejected(self):
        pose = NEUTRAL["smpl_pose"].copy()
        pose[5] = [3.0, 1.0, 0.5]  # |aa| > 2.9
        ok, reason = pose_is_sane(pose, NEUTRAL["body_quat"], LIMITS)
        assert not ok and "aa_too_large" in reason

    def test_pelvis_tilt_rejected(self):
        # 60 deg roll > 45 deg limit
        half = np.radians(60) / 2
        quat = np.array([np.cos(half), np.sin(half), 0, 0], dtype=np.float32)
        assert pelvis_tilt_deg(quat) == pytest.approx(60, abs=0.5)
        ok, reason = pose_is_sane(NEUTRAL["smpl_pose"], quat, LIMITS)
        assert not ok and "pelvis_tilt" in reason

    def test_yaw_is_not_tilt(self):
        half = np.radians(120) / 2
        quat = np.array([np.cos(half), 0, 0, np.sin(half)], dtype=np.float32)
        assert pelvis_tilt_deg(quat) == pytest.approx(0, abs=0.5)

    def test_jump_detection(self):
        a = make_frame()
        b = make_frame(pose_val=1.0)  # |aa| ~ 1.73 rad per joint
        is_jump, reason = sample_is_jump(
            a["smpl_pose"], b["smpl_pose"], a["body_quat"], b["body_quat"], LIMITS
        )
        assert is_jump and "pose_jump" in reason

    def test_no_jump_for_small_motion(self):
        a = make_frame()
        b = make_frame(pose_val=0.1)
        is_jump, _ = sample_is_jump(
            a["smpl_pose"], b["smpl_pose"], a["body_quat"], b["body_quat"], LIMITS
        )
        assert not is_jump


# ---------------------------------------------------------------------------
#  Upper body mode
# ---------------------------------------------------------------------------


class TestUpperBody:
    def test_zero_legs(self):
        pose = np.random.default_rng(0).uniform(-0.5, 0.5, 63).astype(np.float32)
        out = zero_legs_smpl(pose)
        for j in SMPL_LEG_JOINTS:
            np.testing.assert_array_equal(out[j * 3:j * 3 + 3], 0.0)
        # Arms untouched (joint 17 = L elbow)
        np.testing.assert_array_equal(out[17 * 3:17 * 3 + 3], pose[17 * 3:17 * 3 + 3])


# ---------------------------------------------------------------------------
#  Keypoint gating (F1)
# ---------------------------------------------------------------------------


class TestKeypointGating:
    W, H = 640, 480

    def full_kp(self):
        kp = np.zeros((17, 2))
        kp[:, 0] = 320
        kp[:, 1] = np.linspace(50, 430, 17)
        conf = np.full(17, 0.9)
        return kp, conf

    def test_full_body_ok(self):
        kp, conf = self.full_kp()
        ok, _ = keypoints_visible(kp, conf, self.W, self.H, FULL_BODY_KEYPOINTS)
        assert ok

    def test_ankle_cut_off(self):
        kp, conf = self.full_kp()
        kp[15, 1] = self.H - 2  # left ankle at the border
        ok, reason = keypoints_visible(kp, conf, self.W, self.H, FULL_BODY_KEYPOINTS)
        assert not ok and "out_of_frame" in reason

    def test_low_confidence(self):
        kp, conf = self.full_kp()
        conf[11] = 0.1  # left hip occluded
        ok, reason = keypoints_visible(kp, conf, self.W, self.H, FULL_BODY_KEYPOINTS)
        assert not ok and "low_conf" in reason

    def test_upper_body_ignores_legs(self):
        kp, conf = self.full_kp()
        conf[13:] = 0.0  # knees/ankles invisible
        ok, _ = keypoints_visible(kp, conf, self.W, self.H, UPPER_BODY_KEYPOINTS)
        assert ok
        ok, _ = keypoints_visible(kp, conf, self.W, self.H, FULL_BODY_KEYPOINTS)
        assert not ok


# ---------------------------------------------------------------------------
#  State machine
# ---------------------------------------------------------------------------


def run_machine(machine, script):
    """Run [(n_ticks, live_frame_or_None), ...], return list of SafetyOutput.

    Keeps a continuous clock on the machine across calls.
    """
    outs = []
    t = getattr(machine, "_test_clock", 0.0)
    for n_ticks, live in script:
        for _ in range(n_ticks):
            t += TICK
            outs.append(machine.tick(t, live))
    machine._test_clock = t
    return outs


class TestStateMachine:
    def make_machine(self):
        return SafetyStateMachine(limits=LIMITS, neutral_frame=NEUTRAL)

    def test_starts_idle_disarmed(self):
        m = self.make_machine()
        out = m.tick(TICK, make_frame(0.1))
        assert out.state == SafetyState.IDLE and out.frame is None

    def test_arm_dwell_ramp_track(self):
        m = self.make_machine()
        m.arm(0.0)
        live = make_frame(0.3)
        outs = run_machine(m, [(200, live)])
        # dwell (1 s = 50 ticks) idle, then RESUME ramp (1 s), then TRACKING
        assert outs[10].state == SafetyState.IDLE
        assert outs[60].state == SafetyState.RESUME
        assert outs[-1].state == SafetyState.TRACKING
        np.testing.assert_allclose(outs[-1].frame["smpl_pose"], live["smpl_pose"], atol=1e-3)

    def test_all_streamed_steps_within_limits(self):
        m = self.make_machine()
        m.arm(0.0)
        live_a = make_frame(0.5, jp_val=0.8)
        live_b = make_frame(-0.5, jp_val=-0.8)
        outs = run_machine(
            m, [(150, live_a), (30, live_b), (30, None), (200, None), (150, live_a)]
        )
        prev = NEUTRAL
        for o in outs:
            if o.frame is not None:
                assert_step_within_limits(prev, o.frame)
                prev = o.frame

    def test_hold_then_idle_on_loss(self):
        m = self.make_machine()
        m.arm(0.0)
        live = make_frame(0.2)
        outs = run_machine(m, [(150, live)])
        assert outs[-1].state == SafetyState.TRACKING
        held = outs[-1].frame
        # Loss: < 0.2 s -> still repeating last pose (TRACKING)
        outs = run_machine(m, [(8, None)])
        assert outs[-1].state == SafetyState.TRACKING
        np.testing.assert_array_equal(outs[-1].frame["smpl_pose"], held["smpl_pose"])
        # 0.2 - 1.5 s -> HOLD, frozen
        outs = run_machine(m, [(40, None)])
        assert outs[-1].state == SafetyState.HOLD
        np.testing.assert_array_equal(outs[-1].frame["smpl_pose"], held["smpl_pose"])
        # > 1.5 s -> blend to idle, then IDLE (blend 2 s)
        outs = run_machine(m, [(40, None)])
        assert outs[-1].state == SafetyState.BLEND_TO_IDLE
        outs = run_machine(m, [(120, None)])
        assert outs[-1].state == SafetyState.IDLE and outs[-1].frame is None

    def test_blend_reaches_neutral(self):
        m = self.make_machine()
        m.arm(0.0)
        outs = run_machine(m, [(150, make_frame(0.2)), (300, None)])
        last_frame = None
        for o in outs:
            if o.frame is not None:
                last_frame = o.frame
        np.testing.assert_allclose(
            last_frame["smpl_pose"], NEUTRAL["smpl_pose"], atol=2 * LIMITS.max_pose_step_rad
        )

    def test_resume_after_idle(self):
        m = self.make_machine()
        m.arm(0.0)
        live = make_frame(0.2)
        run_machine(m, [(150, live), (300, None)])  # track then fall to idle
        assert m.state == SafetyState.IDLE
        outs = run_machine(m, [(45, live)])  # dwell not yet complete (0.9 s)
        assert outs[-1].state == SafetyState.IDLE
        outs = run_machine(m, [(120, live)])
        assert outs[-1].state == SafetyState.TRACKING

    def test_flicker_resets_dwell(self):
        m = self.make_machine()
        m.arm(0.0)
        live = make_frame(0.2)
        # valid 0.6 s / invalid 1 tick, repeatedly: must never leave IDLE
        outs = run_machine(m, [(30, live), (1, None)] * 8)
        assert all(o.state == SafetyState.IDLE for o in outs)

    def test_force_idle_blends_down(self):
        m = self.make_machine()
        m.arm(0.0)
        live = make_frame(0.4)
        run_machine(m, [(150, live)])
        assert m.state == SafetyState.TRACKING
        m.force_idle(m._test_clock, "session_timeout")
        outs = run_machine(m, [(1, live)])
        assert outs[0].state == SafetyState.BLEND_TO_IDLE
        # Disarmed: valid frames must NOT resume
        outs = run_machine(m, [(300, live)])
        assert outs[-1].state == SafetyState.IDLE
        # Re-arm resumes
        m.arm(m._test_clock)
        outs = run_machine(m, [(150, live)])
        assert outs[-1].state == SafetyState.TRACKING

    def test_loss_during_resume_holds(self):
        m = self.make_machine()
        m.arm(0.0)
        live = make_frame(0.2)
        run_machine(m, [(60, live)])  # dwell done, in RESUME
        assert m.state == SafetyState.RESUME
        outs = run_machine(m, [(1, None)])
        assert outs[0].state == SafetyState.HOLD
