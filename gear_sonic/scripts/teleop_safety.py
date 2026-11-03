#!/usr/bin/env python3
"""Safety layer for the webcam teleoperation bridge.

Pure numpy/scipy (no torch, no zmq) so it can be unit-tested in isolation and
imported from both the bridge (.venv_teleop) and the GEM publisher (GENMO venv).

Components:
    - SafetyLimits: all tunable thresholds in one place.
    - Frame validity checks: pose sanity, pelvis tilt, inter-sample jumps.
    - clamp_frame_step: per-tick joint-velocity clamp on the outgoing stream.
    - SafetyStateMachine: TRACKING / HOLD / BLEND_TO_IDLE / IDLE / RESUME.
    - full_body_visible: tracking-validity gating from 2D keypoints (COCO-17).

A "frame" throughout this module is a dict with keys:
    smpl_pose   (21, 3) float32  axis-angle body pose (SMPL, no root)
    smpl_joints (24, 3) float32  local joint positions (FK output)
    body_quat   (4,)    float32  pelvis orientation quaternion, wxyz, z-up
    joint_pos   (29,)   float64  G1 joint targets (wrists populated)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation as R

FRAME_KEYS = ("smpl_pose", "smpl_joints", "body_quat", "joint_pos")

# SMPL body-pose joint indices (21 joints, pelvis excluded)
SMPL_LEG_JOINTS = (0, 1, 3, 4, 6, 7, 9, 10)  # hips, knees, ankles, feet

# COCO-17 keypoint indices (ViTPose output)
COCO_NOSE = 0
COCO_L_SHOULDER, COCO_R_SHOULDER = 5, 6
COCO_L_HIP, COCO_R_HIP = 11, 12
COCO_L_KNEE, COCO_R_KNEE = 13, 14
COCO_L_ANKLE, COCO_R_ANKLE = 15, 16

FULL_BODY_KEYPOINTS = (
    COCO_NOSE,
    COCO_L_SHOULDER, COCO_R_SHOULDER,
    COCO_L_HIP, COCO_R_HIP,
    COCO_L_KNEE, COCO_R_KNEE,
    COCO_L_ANKLE, COCO_R_ANKLE,
)
UPPER_BODY_KEYPOINTS = (
    COCO_NOSE,
    COCO_L_SHOULDER, COCO_R_SHOULDER,
    COCO_L_HIP, COCO_R_HIP,
)


@dataclass
class SafetyLimits:
    """All safety thresholds. Defaults assume a 50 Hz output tick."""

    # --- Per-tick output clamps (velocity limits) ---
    max_pose_step_rad: float = 0.20      # per SMPL joint per tick (10 rad/s @ 50 Hz)
    max_quat_step_rad: float = 0.10      # pelvis orientation per tick (5 rad/s)
    max_joint_pos_step_rad: float = 0.20  # G1 joint targets per tick
    max_joints_step_m: float = 0.06      # SMPL joint positions per tick (3 m/s)

    # --- Frame validity ---
    pose_aa_max_rad: float = 2.9          # any |axis-angle| above this = garbage
    pelvis_tilt_max_deg: float = 45.0     # pelvis tilt from vertical
    jump_pose_max_rad: float = 1.2        # per-joint jump between GEM samples
    jump_quat_max_rad: float = 1.0        # pelvis jump between GEM samples

    # --- State machine timing (seconds) ---
    watchdog_s: float = 0.3               # newest valid GEM frame older than this = stale
    hold_after_s: float = 0.2             # invalid for this long -> HOLD (freeze)
    idle_after_s: float = 1.5             # invalid for this long -> blend to idle
    blend_to_idle_s: float = 2.0          # duration of the blend to neutral
    resume_dwell_s: float = 1.0           # continuous valid frames required to resume
    resume_ramp_s: float = 1.0            # ramp from held/neutral pose to live pose

    # --- Burst instability filter ---
    unstable_window_s: float = 2.0        # window for counting rejected GEM samples
    unstable_max_rejects: int = 4         # rejects in window -> treat tracking invalid


# ---------------------------------------------------------------------------
#  Quaternion / rotation helpers (wxyz convention, matching the bridge)
# ---------------------------------------------------------------------------


def quat_angle_wxyz(q0: np.ndarray, q1: np.ndarray) -> float:
    """Rotation angle (rad) between two wxyz quaternions."""
    dot = abs(float(np.dot(q0, q1)))
    dot = min(dot / (np.linalg.norm(q0) * np.linalg.norm(q1) + 1e-12), 1.0)
    return 2.0 * float(np.arccos(dot))


def quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Proper slerp between wxyz quaternions (shortest path)."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = min(dot, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-6:
        out = (1.0 - t) * q0 + t * q1
    else:
        s0 = np.sin((1.0 - t) * theta) / np.sin(theta)
        s1 = np.sin(t * theta) / np.sin(theta)
        out = s0 * q0 + s1 * q1
    return (out / (np.linalg.norm(out) + 1e-12)).astype(q0.dtype)


def pelvis_tilt_deg(body_quat_wxyz: np.ndarray) -> float:
    """Tilt of the pelvis frame from vertical (deg). Identity quat = 0 deg."""
    q = np.asarray(body_quat_wxyz, dtype=np.float64)
    mat = R.from_quat(q[[1, 2, 3, 0]]).as_matrix()  # scipy wants xyzw
    cos_tilt = np.clip(mat[2, 2], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))


def _rotvec_relative_angles(prev_rv: np.ndarray, curr_rv: np.ndarray) -> np.ndarray:
    """Per-joint relative rotation angle (rad) between two (J,3) rotvec arrays."""
    r_rel = R.from_rotvec(prev_rv.reshape(-1, 3)).inv() * R.from_rotvec(curr_rv.reshape(-1, 3))
    return np.linalg.norm(r_rel.as_rotvec(), axis=-1)


def clamp_rotvec_step(prev_rv: np.ndarray, target_rv: np.ndarray, max_step: float) -> np.ndarray:
    """Move each joint from prev toward target, geodesic angle capped at max_step."""
    r_prev = R.from_rotvec(prev_rv.reshape(-1, 3))
    r_tgt = R.from_rotvec(target_rv.reshape(-1, 3))
    rel = (r_prev.inv() * r_tgt).as_rotvec()
    ang = np.linalg.norm(rel, axis=-1)
    scale = np.minimum(1.0, max_step / np.maximum(ang, 1e-9))
    rel = rel * scale[:, None]
    out = (r_prev * R.from_rotvec(rel)).as_rotvec()
    return out.reshape(prev_rv.shape).astype(prev_rv.dtype)


# ---------------------------------------------------------------------------
#  Frame validity
# ---------------------------------------------------------------------------


def pose_is_sane(smpl_pose: np.ndarray, body_quat_wxyz: np.ndarray,
                 limits: SafetyLimits) -> tuple[bool, str]:
    """Reject garbage poses (NaN, extreme axis-angles, pelvis far from upright)."""
    if not np.all(np.isfinite(smpl_pose)) or not np.all(np.isfinite(body_quat_wxyz)):
        return False, "non_finite"
    max_aa = float(np.linalg.norm(smpl_pose.reshape(-1, 3), axis=-1).max())
    if max_aa > limits.pose_aa_max_rad:
        return False, f"aa_too_large({max_aa:.2f})"
    tilt = pelvis_tilt_deg(body_quat_wxyz)
    if tilt > limits.pelvis_tilt_max_deg:
        return False, f"pelvis_tilt({tilt:.0f}deg)"
    return True, ""


def sample_is_jump(prev_pose: np.ndarray, curr_pose: np.ndarray,
                   prev_quat: np.ndarray, curr_quat: np.ndarray,
                   limits: SafetyLimits) -> tuple[bool, str]:
    """Detect an implausible jump between two consecutive GEM samples."""
    max_jump = float(_rotvec_relative_angles(prev_pose, curr_pose).max())
    if max_jump > limits.jump_pose_max_rad:
        return True, f"pose_jump({max_jump:.2f}rad)"
    q_jump = quat_angle_wxyz(prev_quat, curr_quat)
    if q_jump > limits.jump_quat_max_rad:
        return True, f"quat_jump({q_jump:.2f}rad)"
    return False, ""


# ---------------------------------------------------------------------------
#  Frame blending / clamping
# ---------------------------------------------------------------------------


def blend_frames(a: dict, b: dict, alpha: float) -> dict:
    """Interpolate frame a -> b (alpha in [0,1]). Rotations via slerp/quat-lerp."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    qa = R.from_rotvec(a["smpl_pose"].reshape(-1, 3))
    qb = R.from_rotvec(b["smpl_pose"].reshape(-1, 3))
    rel = (qa.inv() * qb).as_rotvec()
    pose = (qa * R.from_rotvec(alpha * rel)).as_rotvec().reshape(a["smpl_pose"].shape)
    return {
        "smpl_pose": pose.astype(np.float32),
        "smpl_joints": ((1 - alpha) * a["smpl_joints"] + alpha * b["smpl_joints"]).astype(
            np.float32
        ),
        "body_quat": quat_slerp_wxyz(
            a["body_quat"].astype(np.float64), b["body_quat"].astype(np.float64), alpha
        ).astype(np.float32),
        "joint_pos": (1 - alpha) * a["joint_pos"] + alpha * b["joint_pos"],
    }


def clamp_frame_step(prev: dict, target: dict, limits: SafetyLimits) -> dict:
    """Clamp the per-tick change from prev to target (joint velocity limit)."""
    pose = clamp_rotvec_step(prev["smpl_pose"], target["smpl_pose"], limits.max_pose_step_rad)

    q_ang = quat_angle_wxyz(prev["body_quat"], target["body_quat"])
    if q_ang > limits.max_quat_step_rad:
        body_quat = quat_slerp_wxyz(
            prev["body_quat"].astype(np.float64),
            target["body_quat"].astype(np.float64),
            limits.max_quat_step_rad / q_ang,
        ).astype(np.float32)
    else:
        body_quat = target["body_quat"]

    delta_j = target["smpl_joints"] - prev["smpl_joints"]
    dist = np.linalg.norm(delta_j, axis=-1, keepdims=True)
    scale = np.minimum(1.0, limits.max_joints_step_m / np.maximum(dist, 1e-9))
    smpl_joints = (prev["smpl_joints"] + delta_j * scale).astype(np.float32)

    delta_q = np.clip(
        target["joint_pos"] - prev["joint_pos"],
        -limits.max_joint_pos_step_rad,
        limits.max_joint_pos_step_rad,
    )
    joint_pos = prev["joint_pos"] + delta_q

    return {
        "smpl_pose": pose,
        "smpl_joints": smpl_joints,
        "body_quat": body_quat,
        "joint_pos": joint_pos,
    }


def zero_legs_smpl(body_pose_63: np.ndarray) -> np.ndarray:
    """Zero out leg joints of a flat (63,) SMPL body pose (upper body mode)."""
    out = np.asarray(body_pose_63, dtype=np.float32).copy()
    for j in SMPL_LEG_JOINTS:
        out[j * 3:j * 3 + 3] = 0.0
    return out


# ---------------------------------------------------------------------------
#  Tracking-validity gating from 2D keypoints (F1)
# ---------------------------------------------------------------------------


def keypoints_visible(
    kp_xy: np.ndarray,
    kp_conf: np.ndarray,
    img_w: int,
    img_h: int,
    required: tuple[int, ...] = FULL_BODY_KEYPOINTS,
    conf_thresh: float = 0.35,
    margin_px: float = 8.0,
) -> tuple[bool, str]:
    """Check that all required COCO-17 keypoints are confident and inside the frame.

    Args:
        kp_xy: (17, 2) pixel coordinates
        kp_conf: (17,) confidences
        img_w, img_h: image size
        required: keypoint indices that must be visible
        conf_thresh: minimum confidence
        margin_px: keypoints closer than this to the border count as cut off

    Returns:
        (ok, reason) — reason is "" when ok.
    """
    kp_xy = np.asarray(kp_xy, dtype=np.float64)
    kp_conf = np.asarray(kp_conf, dtype=np.float64)
    for j in required:
        if kp_conf[j] < conf_thresh:
            return False, f"kp{j}_low_conf({kp_conf[j]:.2f})"
        x, y = kp_xy[j]
        if not (margin_px <= x <= img_w - margin_px and margin_px <= y <= img_h - margin_px):
            return False, f"kp{j}_out_of_frame"
    return True, ""


# ---------------------------------------------------------------------------
#  Safety state machine
# ---------------------------------------------------------------------------


class SafetyState(Enum):
    TRACKING = "TRACKING"
    HOLD = "HOLD"
    BLEND_TO_IDLE = "BLEND_TO_IDLE"
    IDLE = "IDLE"
    RESUME = "RESUME"


@dataclass
class SafetyOutput:
    state: SafetyState
    frame: dict | None       # pose frame to stream, or None (stream planner-idle)
    reason: str = ""          # last transition reason, for logging


@dataclass
class SafetyStateMachine:
    """Drives what the bridge streams: live pose, held pose, blend, or idle.

    Call tick() once per output tick (50 Hz) with the current (already
    validated) live frame or None. The machine guarantees:
      - per-tick deltas of the streamed frames respect SafetyLimits clamps;
      - loss of tracking freezes, then blends to the neutral pose and idles;
      - resuming requires a dwell of continuous valid frames plus a ramp.
    """

    limits: SafetyLimits
    neutral_frame: dict
    state: SafetyState = SafetyState.IDLE
    armed: bool = False
    reason: str = "disarmed"

    _last_sent: dict = field(default=None, repr=False)  # type: ignore[assignment]
    _last_valid_t: float | None = None
    _valid_streak_start: float | None = None
    _blend_start: float = 0.0
    _blend_from: dict | None = None
    _resume_start: float = 0.0
    _resume_from: dict | None = None

    def __post_init__(self):
        self._last_sent = {k: np.copy(v) for k, v in self.neutral_frame.items()}

    # --- external controls -------------------------------------------------

    def arm(self, now: float):
        """Allow streaming. Robot will resume via dwell + ramp when tracking is valid."""
        self.armed = True
        self.reason = "armed"
        self._valid_streak_start = None

    def force_idle(self, now: float, reason: str = "forced"):
        """Disarm and return to idle (blending down first if streaming)."""
        self.armed = False
        self.reason = reason
        if self.state in (SafetyState.TRACKING, SafetyState.HOLD, SafetyState.RESUME):
            self._enter_blend_to_idle(now)
        # BLEND_TO_IDLE / IDLE: already on the way down

    # --- transitions --------------------------------------------------------

    def _enter_blend_to_idle(self, now: float):
        self.state = SafetyState.BLEND_TO_IDLE
        self._blend_start = now
        self._blend_from = {k: np.copy(v) for k, v in self._last_sent.items()}

    def _enter_hold(self, reason: str):
        self.state = SafetyState.HOLD
        self.reason = reason

    def _enter_resume(self, now: float):
        self.state = SafetyState.RESUME
        self.reason = "resume"
        self._resume_start = now
        self._resume_from = {k: np.copy(v) for k, v in self._last_sent.items()}

    # --- main tick ------------------------------------------------------

    def tick(self, now: float, live: dict | None) -> SafetyOutput:
        """Advance one output tick. live=None means no valid live frame right now."""
        if live is not None:
            self._last_valid_t = now
            if self._valid_streak_start is None:
                self._valid_streak_start = now
        else:
            self._valid_streak_start = None

        invalid_for = (
            float("inf") if self._last_valid_t is None else now - self._last_valid_t
        )
        valid_streak = (
            0.0 if self._valid_streak_start is None else now - self._valid_streak_start
        )

        out: dict | None = None

        if self.state == SafetyState.TRACKING:
            if not self.armed:
                self._enter_blend_to_idle(now)
                out = self._blend_tick(now)
            elif live is not None:
                out = clamp_frame_step(self._last_sent, live, self.limits)
            elif invalid_for > self.limits.idle_after_s:
                self._enter_blend_to_idle(now)
                out = self._blend_tick(now)
            elif invalid_for > self.limits.hold_after_s:
                self._enter_hold("tracking_lost")
                out = self._last_sent
            else:
                out = self._last_sent  # micro-gap: repeat last pose

        elif self.state == SafetyState.HOLD:
            if not self.armed or invalid_for > self.limits.idle_after_s:
                self._enter_blend_to_idle(now)
                out = self._blend_tick(now)
            elif live is not None and valid_streak >= self.limits.resume_dwell_s:
                self._enter_resume(now)
                out = self._resume_tick(now, live)
            else:
                out = self._last_sent

        elif self.state == SafetyState.BLEND_TO_IDLE:
            out = self._blend_tick(now)
            alpha = (now - self._blend_start) / self.limits.blend_to_idle_s
            if alpha >= 1.0:
                self.state = SafetyState.IDLE
                out = None

        elif self.state == SafetyState.IDLE:
            if self.armed and live is not None and valid_streak >= self.limits.resume_dwell_s:
                self._enter_resume(now)
                out = self._resume_tick(now, live)
            else:
                out = None

        elif self.state == SafetyState.RESUME:
            if not self.armed:
                self._enter_blend_to_idle(now)
                out = self._blend_tick(now)
            elif live is None:
                self._enter_hold("tracking_lost_in_resume")
                out = self._last_sent
            else:
                out = self._resume_tick(now, live)
                if (now - self._resume_start) >= self.limits.resume_ramp_s:
                    self.state = SafetyState.TRACKING
                    self.reason = "tracking"

        if out is not None:
            self._last_sent = {k: np.copy(v) for k, v in out.items()}
        return SafetyOutput(state=self.state, frame=out, reason=self.reason)

    # --- state helpers ------------------------------------------------------

    def _blend_tick(self, now: float) -> dict:
        alpha = (now - self._blend_start) / self.limits.blend_to_idle_s
        target = blend_frames(self._blend_from, self.neutral_frame, alpha)
        return clamp_frame_step(self._last_sent, target, self.limits)

    def _resume_tick(self, now: float, live: dict) -> dict:
        alpha = (now - self._resume_start) / self.limits.resume_ramp_s
        target = blend_frames(self._resume_from, live, alpha)
        return clamp_frame_step(self._last_sent, target, self.limits)
