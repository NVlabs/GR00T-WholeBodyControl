# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a KIMODO G1 qpos CSV (native 30fps) into GEAR-SONIC's ZMQ 'pose'
message format (50Hz), safety-check it, and stream it to the deploy binary.

KIMODO CSV layout (36 columns, MuJoCo qpos convention):
  cols 0:3  = root position xyz      (NOT sent -- 'pose' topic has no root
                                       world-position field)
  cols 3:7  = root quaternion (w,x,y,z)  -> body_quat
  cols 7:36 = 29 actuated joint angles   -> joint_pos

Usage:
    python3 kimodo_to_sonic.py <path_to_kimodo.csv> [--host localhost] [--port 5556]
"""
import argparse
import os
import sys
import time

# This file lives at gear_sonic/examples/speech_to_motion/kimodo_to_sonic.py,
# three directories below the repo root -- derive the tests/ path (where
# ZMQPublisher lives) from our own location rather than hardcoding a path.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(
    0, os.path.join(_REPO_ROOT, "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests")
)
import numpy as np
from test_zmq_manager import ZMQPublisher

KIMODO_FPS = 30
SONIC_FPS = 50
NUM_JOINTS = 29
BLEND_IN_SECONDS = 1.0

# GEAR-SONIC's own known standing pose (policy_parameters.hpp `default_angles`,
# same 29-joint order confirmed identical to KIMODO's g1.xml). Used to blend
# smoothly from wherever the robot actually is into the clip's first frame,
# instead of snapping there instantly -- an instant jump within one 20ms
# control tick is a physically impossible "teleport" and a likely cause of
# the twisting/falling seen when streaming raw KIMODO frame 0 directly.
SONIC_DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,       # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,       # right leg
    0.0, 0.0, 0.0,                              # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,          # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,         # right arm
], dtype=np.float32)

# GEAR-SONIC's ZMQ 'pose' topic (Protocol v1) requires joint_pos/joint_vel in
# IsaacLab order, NOT MuJoCo order -- see docs/source/tutorials/zmq.md
# ("Joint positions in IsaacLab order (all 29 joints)"). Everything above
# this point (KIMODO's CSV, SONIC_DEFAULT_ANGLES, g1_action_scale in
# policy_parameters.hpp) is MuJoCo order, so we must remap right before
# sending.
#
# policy_parameters.hpp defines two arrays, `isaaclab_to_mujoco` and
# `mujoco_to_isaaclab`, but their comments are self-contradictory (same issue
# found in test_zmq_manager.py) so the variable names can't be trusted
# blindly. Decoded both against the known MuJoCo joint labels (from
# `g1_action_scale`'s comments, the one reliable source in that file): the
# array actually named `isaaclab_to_mujoco` decodes to a scrambled,
# incoherent joint grouping when used as an IsaacLab-index table (tried
# first during development, caused the robot to fall over in sim). The array
# named `mujoco_to_isaaclab` decodes to a clean, sensible IsaacLab layout --
# (L,R) pairs grouped by joint type: hip_pitch, waist_yaw, hip_roll,
# waist_roll, hip_yaw, waist_pitch, knee, shoulder_pitch, ankle_pitch,
# shoulder_roll, ankle_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch,
# wrist_yaw -- so THIS is the one that is functionally
# isaaclab_to_mujoco[i] = MuJoCo index of IsaacLab position i, despite its
# misleading variable name in the C++ source.
ISAACLAB_TO_MUJOCO = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])


def mujoco_to_isaaclab_order(arr_mujoco_order: np.ndarray) -> np.ndarray:
    """arr_mujoco_order: [..., 29] in MuJoCo joint order -> returns IsaacLab order."""
    return arr_mujoco_order[..., ISAACLAB_TO_MUJOCO]


# Safety gate: per-joint position limits, MuJoCo order, read directly from
# this repo's gear_sonic_deploy/g1/g1_29dof.xml <joint ... range="lo hi">
# values -- the real, authoritative hardware limits, not estimated.
G1_JOINT_POSITION_LIMITS_MUJOCO = np.array([
    (-2.5307, 2.8798),    # left_hip_pitch
    (-0.5236, 2.9671),    # left_hip_roll
    (-2.7576, 2.7576),    # left_hip_yaw
    (-0.087267, 2.8798),  # left_knee
    (-0.87267, 0.5236),   # left_ankle_pitch
    (-0.2618, 0.2618),    # left_ankle_roll
    (-2.5307, 2.8798),    # right_hip_pitch
    (-2.9671, 0.5236),    # right_hip_roll
    (-2.7576, 2.7576),    # right_hip_yaw
    (-0.087267, 2.8798),  # right_knee
    (-0.87267, 0.5236),   # right_ankle_pitch
    (-0.2618, 0.2618),    # right_ankle_roll
    (-2.618, 2.618),      # waist_yaw
    (-0.52, 0.52),        # waist_roll
    (-0.52, 0.52),        # waist_pitch
    (-3.0892, 2.6704),    # left_shoulder_pitch
    (-1.5882, 2.2515),    # left_shoulder_roll
    (-2.618, 2.618),      # left_shoulder_yaw
    (-1.0472, 2.0944),    # left_elbow
    (-1.97222, 1.97222),  # left_wrist_roll
    (-1.61443, 1.61443),  # left_wrist_pitch
    (-1.61443, 1.61443),  # left_wrist_yaw
    (-3.0892, 2.6704),    # right_shoulder_pitch
    (-2.2515, 1.5882),    # right_shoulder_roll
    (-2.618, 2.618),      # right_shoulder_yaw
    (-1.0472, 2.0944),    # right_elbow
    (-1.97222, 1.97222),  # right_wrist_roll
    (-1.61443, 1.61443),  # right_wrist_pitch
    (-1.61443, 1.61443),  # right_wrist_yaw
], dtype=np.float32)

G1_JOINT_NAMES_MUJOCO = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# No official per-joint velocity limit spec exists in this repo (checked
# policy_parameters.hpp and g1_29dof.xml -- only position/torque limits are
# defined there). This threshold is empirically derived from our own test
# data instead: every validated-good leg/torso-driving clip topped out
# around ~10.1 rad/s, the one clip that actually caused a problem spiked to
# ~145 rad/s, and a "wave hello" clip visually confirmed safe in the MuJoCo
# viewer peaked at ~24.9 rad/s on right_wrist_roll (brief, 2/299 frames,
# low-inertia joint, not a leg). Set above that confirmed-safe peak with
# headroom. Still a single global threshold across all 29 joints, not
# per-joint-group -- a genuinely dangerous 25 rad/s hip/knee spike would be
# treated the same as a harmless 25 rad/s wrist flick. Revisit against real
# Unitree G1 motor datasheets (and consider per-joint-group limits) before
# relying on this for unsupervised hardware operation.
JOINT_VELOCITY_LIMIT_RAD_S = 27.0

# Small tolerance on the position-limit check. Measured on a known-good,
# physically-verified compound clip (played correctly on both sim and real
# hardware): left_ankle_roll genuinely reaches -0.28274 rad against a
# -0.2618 rad XML limit in 10 of 449 frames -- a real ~0.021 rad (1.2 deg)
# overshoot, not interpolation noise. KIMODO's kinematic generation isn't
# hard-constrained to the G1's precise hardware bounds, and this specific
# overshoot has already been confirmed safe in practice. Tolerance set above
# that measured value with headroom; still far too small to mask a genuine
# violation (the regression test for this gate injects a 10 rad overshoot).
POSITION_LIMIT_TOLERANCE_RAD = 0.05


def validate_motion_safety(joint_pos_mujoco: np.ndarray, joint_vel_mujoco: np.ndarray):
    """Check every frame of a MuJoCo-order clip against G1 position/velocity limits.

    Returns (True, "") if the whole clip is safe, else (False, reason) -- in
    which case the caller must not stream anything to SONIC (robot holds pose).
    """
    lo = G1_JOINT_POSITION_LIMITS_MUJOCO[:, 0] - POSITION_LIMIT_TOLERANCE_RAD
    hi = G1_JOINT_POSITION_LIMITS_MUJOCO[:, 1] + POSITION_LIMIT_TOLERANCE_RAD

    pos_violations = (joint_pos_mujoco < lo) | (joint_pos_mujoco > hi)
    if pos_violations.any():
        frame_idx, joint_idx = np.argwhere(pos_violations)[0]
        name = G1_JOINT_NAMES_MUJOCO[joint_idx]
        val = joint_pos_mujoco[frame_idx, joint_idx]
        return False, (
            f"position limit violated: joint '{name}' = {val:.3f} rad at frame {frame_idx} "
            f"(allowed [{lo[joint_idx]:.3f}, {hi[joint_idx]:.3f}])"
        )

    vel_violations = np.abs(joint_vel_mujoco) > JOINT_VELOCITY_LIMIT_RAD_S
    if vel_violations.any():
        frame_idx, joint_idx = np.argwhere(vel_violations)[0]
        name = G1_JOINT_NAMES_MUJOCO[joint_idx]
        val = joint_vel_mujoco[frame_idx, joint_idx]
        return False, (
            f"velocity limit violated: joint '{name}' = {val:.3f} rad/s at frame {frame_idx} "
            f"(limit {JOINT_VELOCITY_LIMIT_RAD_S:.1f} rad/s)"
        )

    return True, ""


def load_and_resample(csv_path: str):
    qpos = np.loadtxt(csv_path, delimiter=",")
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    n_frames_in = qpos.shape[0]
    if qpos.shape[1] != 36:
        raise ValueError(f"Expected 36 columns (7 root + 29 joints), got {qpos.shape[1]}")

    body_quat_30 = qpos[:, 3:7].astype(np.float32)
    joint_pos_30 = qpos[:, 7:36].astype(np.float32)

    duration_s = (n_frames_in - 1) / KIMODO_FPS
    t_30 = np.linspace(0.0, duration_s, n_frames_in)
    n_frames_out = int(duration_s * SONIC_FPS) + 1
    t_50 = np.linspace(0.0, duration_s, n_frames_out)

    joint_pos_50 = np.zeros((n_frames_out, NUM_JOINTS), dtype=np.float32)
    for j in range(NUM_JOINTS):
        joint_pos_50[:, j] = np.interp(t_50, t_30, joint_pos_30[:, j])

    body_quat_50 = np.zeros((n_frames_out, 4), dtype=np.float32)
    for k in range(4):
        body_quat_50[:, k] = np.interp(t_50, t_30, body_quat_30[:, k])
    # renormalize interpolated quaternions
    norms = np.linalg.norm(body_quat_50, axis=1, keepdims=True)
    body_quat_50 = body_quat_50 / np.clip(norms, 1e-6, None)

    # Prepend a blend-in ramp from GEAR-SONIC's known default standing pose to
    # the clip's actual first frame, instead of jumping there instantly.
    n_blend = int(BLEND_IN_SECONDS * SONIC_FPS)
    blend_t = np.linspace(0.0, 1.0, n_blend, endpoint=False)[:, None]  # [n_blend, 1]
    blend_joint_pos = SONIC_DEFAULT_ANGLES[None, :] * (1 - blend_t) + joint_pos_50[0][None, :] * blend_t
    blend_body_quat = np.tile(body_quat_50[0], (n_blend, 1))

    joint_pos_50 = np.concatenate([blend_joint_pos.astype(np.float32), joint_pos_50], axis=0)
    body_quat_50 = np.concatenate([blend_body_quat.astype(np.float32), body_quat_50], axis=0)

    dt = 1.0 / SONIC_FPS
    joint_vel_50 = np.zeros_like(joint_pos_50)
    joint_vel_50[1:] = (joint_pos_50[1:] - joint_pos_50[:-1]) / dt
    joint_vel_50[0] = joint_vel_50[1] if joint_pos_50.shape[0] > 1 else 0.0

    print(
        f"Loaded {csv_path}: {n_frames_in} frames @ {KIMODO_FPS}fps ({duration_s:.2f}s) "
        f"-> resampled to {n_frames_out} frames @ {SONIC_FPS}fps "
        f"+ {n_blend} blend-in frames from default pose ({BLEND_IN_SECONDS}s)"
    )
    return joint_pos_50, joint_vel_50, body_quat_50


def resample_and_reorder(csv_path: str, force_identity_quat: bool = False):
    """Load+resample a KIMODO CSV and return both the MuJoCo-order arrays
    (for safety validation, matching G1_JOINT_POSITION_LIMITS_MUJOCO) and the
    IsaacLab-order arrays (what GEAR-SONIC's ZMQ 'pose' topic requires)."""
    joint_pos_mujoco, joint_vel_mujoco, body_quat = load_and_resample(csv_path)
    if force_identity_quat:
        print("[diagnostic] Forcing body_quat to identity (w=1,x=y=z=0) for all frames")
        body_quat = np.zeros_like(body_quat)
        body_quat[:, 0] = 1.0

    joint_pos_isaaclab = mujoco_to_isaaclab_order(joint_pos_mujoco)
    joint_vel_isaaclab = mujoco_to_isaaclab_order(joint_vel_mujoco)
    return joint_pos_mujoco, joint_vel_mujoco, joint_pos_isaaclab, joint_vel_isaaclab, body_quat


def send_clip_to_sonic(joint_pos_isaaclab, joint_vel_isaaclab, body_quat, host="*", port=5556, verbose=True):
    """Stream an already-resampled, IsaacLab-order clip to GEAR-SONIC as a
    single ZMQ 'pose' burst (catch_up=False). Sent as one burst rather than
    chunked real-time pacing: this is a pre-generated clip, not a live feed,
    so there is no need to drip-feed it, and a single burst avoids
    StreamedMotionMerger's MAX_GAP_FRAMES window-reset path entirely."""
    frame_indices = np.arange(joint_pos_isaaclab.shape[0], dtype=np.int64)
    n_total = joint_pos_isaaclab.shape[0]

    pub = ZMQPublisher(host=host, port=port, verbose=verbose)
    if verbose:
        print("Waiting 2s for subscriber to fully connect...")
    time.sleep(2.0)

    if verbose:
        print("Sending pose stream as a single burst (requires ']', '#', and Enter already done in the deploy terminal)...")
    pub.send_pose(joint_pos_isaaclab, joint_vel_isaaclab, body_quat, frame_indices, catch_up=False)

    if verbose:
        print(f"Sent {n_total} frames. Waiting for playback ({n_total/SONIC_FPS:.1f}s)...")
    time.sleep(n_total / SONIC_FPS + 1.0)
    pub.close()
    if verbose:
        print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument(
        "--force-identity-quat",
        action="store_true",
        help="Diagnostic: send identity body_quat instead of KIMODO's, to isolate "
        "whether orientation data or joint-angle data is destabilizing the robot.",
    )
    args = parser.parse_args()

    joint_pos_mujoco, joint_vel_mujoco, joint_pos_isaaclab, joint_vel_isaaclab, body_quat = resample_and_reorder(
        args.csv_path, force_identity_quat=args.force_identity_quat
    )

    ok, reason = validate_motion_safety(joint_pos_mujoco, joint_vel_mujoco)
    if not ok:
        print(f"[SAFETY GATE] REJECTED -- {reason}")
        print("[SAFETY GATE] Nothing sent to SONIC. Robot holds its current pose.")
        sys.exit(1)
    print("[SAFETY GATE] passed.")

    send_clip_to_sonic(joint_pos_isaaclab, joint_vel_isaaclab, body_quat, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
