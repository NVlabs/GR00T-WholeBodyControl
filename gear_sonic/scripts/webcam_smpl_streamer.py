#!/usr/bin/env python3
"""Webcam SMPL streamer: bridges GEM (GENMO) webcam pose estimation to SONIC deployment.

Receives per-frame SMPL parameters from ``gem_webcam_zmq_publisher.py`` (ZMQ SUB,
default port 5558), converts them to the SONIC pose-streaming protocol (same wire
format as ``pico_manager_thread_server.py``), and publishes on the ZMQ PUB socket
(default port 5556) consumed by the C++ deployment (``deploy.sh --input-type
zmq_manager``).

Pipeline:
    GEM webcam demo (GENMO venv, GPU)  --ZMQ 5558-->  this bridge (.venv_teleop, CPU)
    --ZMQ 5556-->  C++ SONIC controller (sim or real G1)

Keyboard controls (this terminal):
    s : ARM streaming (robot resumes imitation after a short dwell + ramp)
    p : PAUSE (disarm; robot blends to neutral pose and idles)
    n : NEXT PILOT (same as pause; press 's' when the next person is ready)
    o : STOP control and exit
    q : same as 'o'

Safety layer (see teleop_safety.py):
    - per-tick joint velocity clamp on everything streamed;
    - invalid/lost tracking: freeze (HOLD) -> blend to neutral -> PLANNER-IDLE;
    - resume requires ~1 s of continuous valid tracking + 1 s ramp;
    - --upper_body: only arms/torso are imitated (legs + pelvis stay neutral);
    - session timer (--session_timeout) forces idle and requires re-arming.

Run (inside .venv_teleop, from repo root):
    python gear_sonic/scripts/webcam_smpl_streamer.py
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import defaultdict, deque
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from teleop_safety import (  # noqa: E402
    SafetyLimits,
    SafetyState,
    SafetyStateMachine,
    pose_is_sane,
    sample_is_jump,
    zero_legs_smpl,
)

from gear_sonic.isaac_utils.rotations import (  # noqa: E402
    remove_smpl_base_rot,
    smpl_root_ytoz_up,
)
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa  # noqa: E402
from gear_sonic.trl.utils.torch_transform import (  # noqa: E402
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

HUMAN_JOINTS_INFO_PATH = str(REPO_ROOT / "gear_sonic" / "data" / "human" / "human_joints_info.pkl")


class StreamMode(IntEnum):
    OFF = 0
    POSE = 1
    PLANNER_IDLE = 2


# ---------------------------------------------------------------------------
#  SMPL processing (mirrors pico_manager_thread_server.process_smpl_joints)
# ---------------------------------------------------------------------------


@torch.no_grad()
def process_smpl_joints(body_pose: torch.Tensor, global_orient: torch.Tensor) -> dict:
    """Process SMPL parameters to compute local joints.

    Args:
        body_pose: Body pose tensor (axis-angle), shape (T, 63) or (T, 69)
        global_orient: Global orientation tensor (axis-angle, y-up SMPL world), shape (T, 3)

    Returns:
        Dict with smpl_pose, smpl_joints_local, global_orient_quat (z-up, base rot removed).
    """
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
        human_joints_info_path=HUMAN_JOINTS_INFO_PATH,
    )  # (T, 24, 3)

    global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)
    global_orient_mat = quaternion_to_rotation_matrix(global_orient_quat)
    global_orient_6d = global_orient_mat[..., :2].reshape(1, 6)

    return {
        "smpl_pose": body_pose,
        "smpl_joints_local": smpl_joints_local,
        "global_orient_quat": global_orient_quat,
        "global_orient_6d": global_orient_6d,
    }


def _quat_lerp_normalized(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Lerp two quaternions (shape (4,)) with shortest-path sign flip and renormalize."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _yaw_inverse_wxyz(q: np.ndarray) -> np.ndarray:
    """Inverse of the z-up heading (yaw) component of a wxyz quaternion."""
    w, x, y, z = q
    heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array(
        [np.cos(-heading / 2.0), 0.0, 0.0, np.sin(-heading / 2.0)], dtype=np.float64
    )


def _yaw_wxyz(q: np.ndarray) -> np.ndarray:
    """Z-up heading (yaw) component of a wxyz quaternion."""
    w, x, y, z = q
    heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array(
        [np.cos(heading / 2.0), 0.0, 0.0, np.sin(heading / 2.0)], dtype=np.float64
    )


def _interp_pose_axis_angle(
    prev_pose: np.ndarray, curr_pose: np.ndarray, alpha: float
) -> np.ndarray:
    """Interpolate (21,3) axis-angle poses via per-joint quaternion lerp."""
    prev_quats = R.from_rotvec(prev_pose.reshape(-1, 3)).as_quat()
    curr_quats = R.from_rotvec(curr_pose.reshape(-1, 3)).as_quat()
    out_quats = np.empty_like(prev_quats)
    for i in range(prev_quats.shape[0]):
        out_quats[i] = _quat_lerp_normalized(prev_quats[i], curr_quats[i], alpha)
    return R.from_quat(out_quats).as_rotvec().reshape(prev_pose.shape)


def compute_g1_wrist_joint_pos(body_pose_21x3: np.ndarray) -> np.ndarray:
    """Map SMPL elbow/wrist rotations onto the G1 29-dof wrist joints.

    Mirrors the "From @Jiefeng" block in pico_manager_thread_server.SmplStream.run_once.

    Args:
        body_pose_21x3: (21, 3) axis-angle body pose (SMPL, no root)

    Returns:
        joint_pos: (29,) with only wrist entries populated
    """
    joint_pos = np.zeros(29)
    body_pose = body_pose_21x3.reshape(-1, 21, 3)

    SMPL_L_ELBOW_IDX = 17
    SMPL_L_WRIST_IDX = 19
    SMPL_R_ELBOW_IDX = 18
    SMPL_R_WRIST_IDX = 20

    G1_L_WRIST_ROLL_IDX = 23
    G1_L_WRIST_PITCH_IDX = 25
    G1_L_WRIST_YAW_IDX = 27
    G1_R_WRIST_ROLL_IDX = 24
    G1_R_WRIST_PITCH_IDX = 26
    G1_R_WRIST_YAW_IDX = 28

    smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
    smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
    smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
    smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

    # Guard against exactly-zero rotations (decompose_rotation_aa divides by the angle)
    def _safe_aa(aa: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        aa = aa.copy()
        norms = np.linalg.norm(aa, axis=-1)
        aa[norms < eps, 0] = eps
        return aa

    smpl_l_elbow_aa = _safe_aa(smpl_l_elbow_aa)
    smpl_r_elbow_aa = _safe_aa(smpl_r_elbow_aa)

    elbow_axis = np.array([0, 1, 0])
    _, g1_l_elbow_q_swing = decompose_rotation_aa(smpl_l_elbow_aa, elbow_axis)
    _, g1_r_elbow_q_swing = decompose_rotation_aa(smpl_r_elbow_aa, elbow_axis)

    l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )

    l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
    r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

    g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
    g1_l_wrist_pitch = -l_wrist_euler[:, 1]
    g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

    g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
    g1_r_wrist_pitch = -r_wrist_euler[:, 1]
    g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

    joint_pos[G1_L_WRIST_ROLL_IDX] = g1_l_wrist_roll[0]
    joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch[0]
    joint_pos[G1_L_WRIST_YAW_IDX] = g1_l_wrist_yaw[0]

    joint_pos[G1_R_WRIST_ROLL_IDX] = g1_r_wrist_roll[0]
    joint_pos[G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch[0]
    joint_pos[G1_R_WRIST_YAW_IDX] = g1_r_wrist_yaw[0]

    return joint_pos


# ---------------------------------------------------------------------------
#  GEM receiver (background thread)
# ---------------------------------------------------------------------------


class GemReceiver:
    """Background SUB that receives GEM frames and pre-computes SMPL joint outputs.

    Keeps the two most recent processed frames so the main loop can interpolate
    (GEM runs at ~15-30 fps; the controller consumes 50 Hz).
    """

    def __init__(self, host: str, port: int, limits: SafetyLimits, upper_body: bool = False):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{host}:{port}")
        self._limits = limits
        self._upper_body = upper_body
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._prev: dict | None = None
        self._curr: dict | None = None
        self._last_received: dict | None = None  # jump reference (even if rejected)
        self._n_received = 0
        self._n_rejected = 0
        self._last_reject_reason = ""
        self._reject_times: deque[float] = deque(maxlen=64)
        self._last_msg_ns = 0  # any message (incl. heartbeats) = GEM alive
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close(0)

    @property
    def n_received(self) -> int:
        return self._n_received

    @property
    def n_rejected(self) -> int:
        return self._n_rejected

    @property
    def last_reject_reason(self) -> str:
        return self._last_reject_reason

    @property
    def last_msg_ns(self) -> int:
        return self._last_msg_ns

    def _run(self):
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            events = dict(poller.poll(timeout=100))
            if self._sock not in events:
                continue
            try:
                sample = self._sock.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            processed = self._process(sample)
            if processed is None:
                continue
            with self._lock:
                self._prev = self._curr
                self._curr = processed
                self._n_received += 1

    def _process(self, sample: dict) -> dict | None:
        self._last_msg_ns = time.monotonic_ns()
        if not sample.get("tracking_valid", True):
            # GEM heartbeat: alive but no usable pose (warmup / no person / gating)
            self._last_reject_reason = str(sample.get("reason", "tracking_invalid"))
            return None
        try:
            body_pose_raw = np.asarray(sample["body_pose"], dtype=np.float32).reshape(-1)
            global_orient_raw = np.asarray(
                sample["global_orient"], dtype=np.float32
            ).reshape(-1)
        except (KeyError, ValueError) as e:
            print(f"[Bridge] Bad GEM sample: {e}")
            return None

        if self._upper_body:
            body_pose_raw = zero_legs_smpl(body_pose_raw)
            global_orient_raw = np.zeros(3, dtype=np.float32)

        body_pose = torch.from_numpy(body_pose_raw).reshape(1, -1)
        global_orient = torch.from_numpy(global_orient_raw).reshape(1, 3)

        out = process_smpl_joints(body_pose, global_orient)
        processed = {
            "timestamp_ns": int(sample.get("timestamp_ns", time.monotonic_ns())),
            "timestamp_realtime": float(sample.get("timestamp_realtime", time.time())),
            "dt": float(sample.get("dt", 0.0)),
            "fps": float(sample.get("fps", 0.0)),
            "smpl_pose_np": out["smpl_pose"].numpy()[:, :63].reshape(-1, 21, 3)[0].astype(
                np.float32
            ),
            "smpl_joints_np": out["smpl_joints_local"].numpy()[0].astype(np.float32),
            "body_quat_np": out["global_orient_quat"].numpy()[0].astype(np.float32),
        }

        # --- Safety validation (see teleop_safety) ---
        ok, reason = pose_is_sane(
            processed["smpl_pose_np"], processed["body_quat_np"], self._limits
        )
        if ok and self._last_received is not None:
            is_jump, jump_reason = sample_is_jump(
                self._last_received["smpl_pose_np"],
                processed["smpl_pose_np"],
                self._last_received["body_quat_np"],
                processed["body_quat_np"],
                self._limits,
            )
            if is_jump:
                ok, reason = False, jump_reason
        # Jump reference tracks the last *received* sample so a genuinely new,
        # stable pose becomes valid again on the following frame.
        self._last_received = processed
        if not ok:
            self._n_rejected += 1
            self._last_reject_reason = reason
            self._reject_times.append(time.monotonic())
            return None
        return processed

    def recent_rejects(self, window_s: float) -> int:
        cutoff = time.monotonic() - window_s
        return sum(1 for t in self._reject_times if t >= cutoff)

    def get_pair(self) -> tuple[dict | None, dict | None]:
        with self._lock:
            return self._prev, self._curr


# ---------------------------------------------------------------------------
#  Keyboard (raw, non-blocking, this terminal)
# ---------------------------------------------------------------------------


class KeyboardListener:
    """Non-blocking single-key reader using termios cbreak mode."""

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def restore(self):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def get_key(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def build_neutral_frame() -> dict:
    """Neutral standing frame (A-pose, arms lowered) used as the idle target."""
    body_pose = np.zeros(63, dtype=np.float32)
    body_pose[15 * 3 + 2] = -1.0  # left shoulder: lower arm from T-pose
    body_pose[16 * 3 + 2] = 1.0  # right shoulder
    out = process_smpl_joints(
        torch.from_numpy(body_pose).reshape(1, -1),
        torch.zeros(1, 3, dtype=torch.float32),
    )
    smpl_pose = out["smpl_pose"].numpy()[:, :63].reshape(21, 3).astype(np.float32)
    return {
        "smpl_pose": smpl_pose,
        "smpl_joints": out["smpl_joints_local"].numpy()[0].astype(np.float32),
        "body_quat": out["global_orient_quat"].numpy()[0].astype(np.float32),
        "joint_pos": compute_g1_wrist_joint_pos(smpl_pose),
    }


def main():
    parser = argparse.ArgumentParser(description="GEM webcam -> SONIC pose bridge")
    parser.add_argument("--gem_host", type=str, default="localhost", help="GEM publisher host")
    parser.add_argument("--gem_port", type=int, default=5558, help="GEM publisher port")
    parser.add_argument("--port", type=int, default=5556, help="SONIC ZMQ PUB port")
    parser.add_argument(
        "--bind_host", type=str, default="127.0.0.1",
        help="Interface to bind the SONIC PUB socket (default: localhost only)",
    )
    parser.add_argument("--target_fps", type=int, default=50, help="Output rate (Hz)")
    parser.add_argument(
        "--num_frames_to_send", type=int, default=5, help="Frames per pose message"
    )
    parser.add_argument(
        "--record_dir", type=str, default=None,
        help="Save sent batches as npz (default: teleop_recordings/<timestamp>)",
    )
    parser.add_argument(
        "--no_record", action="store_true", help="Disable recording of sent batches"
    )
    parser.add_argument(
        "--auto_start",
        action="store_true",
        help="Arm streaming automatically once GEM frames arrive",
    )
    parser.add_argument(
        "--upper_body",
        action="store_true",
        help="Upper body mode: only arms/torso are imitated; legs and pelvis stay neutral",
    )
    parser.add_argument(
        "--session_timeout", type=float, default=None,
        help="Force idle after this many seconds of session (default: 90 in "
        "--upper_body mode, disabled otherwise; 0 disables)",
    )
    args = parser.parse_args()

    session_timeout = args.session_timeout
    if session_timeout is None:
        session_timeout = 90.0 if args.upper_body else 0.0

    record_dir = None
    if not args.no_record:
        record_dir = args.record_dir or str(
            REPO_ROOT / "teleop_recordings" / time.strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(record_dir, exist_ok=True)
        print(f"[Bridge] Recording sent batches to {record_dir}")

    limits = SafetyLimits()
    neutral_frame = build_neutral_frame()
    neutral_yaw = _yaw_wxyz(neutral_frame["body_quat"].astype(np.float64))
    machine = SafetyStateMachine(limits=limits, neutral_frame=neutral_frame)

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{args.bind_host}:{args.port}")
    time.sleep(0.2)
    print(f"[Bridge] SONIC PUB bound to tcp://{args.bind_host}:{args.port}")

    receiver = GemReceiver(args.gem_host, args.gem_port, limits, args.upper_body)
    receiver.start()
    print(f"[Bridge] Listening for GEM frames on tcp://{args.gem_host}:{args.gem_port}")
    if args.upper_body:
        print("[Bridge] UPPER BODY mode: legs/pelvis stay neutral, session "
              f"timeout={session_timeout:.0f}s")
    print(
        "[Bridge] Keys: [s]=ARM  [p]=PAUSE  [n]=NEXT PILOT  [o]/[q]=STOP+exit"
    )

    started = False  # becomes True on first arm; before that nothing is sent
    wire_mode: str | None = None  # "pose" | "planner"
    session_armed_t = 0.0
    yaw_ref_inv: np.ndarray | None = None  # pilot heading calibration (wxyz)
    unstable_reported = False
    frame_buffer: dict[str, deque] = defaultdict(lambda: deque(maxlen=args.num_frames_to_send))
    buffer_cleared = True
    step = 0
    record_idx = 0
    frame_time = 1.0 / args.target_fps
    watchdog_ns = int(limits.watchdog_s * 1e9)
    left_hand_joints = np.zeros((1, 7), dtype=np.float32)
    right_hand_joints = np.zeros((1, 7), dtype=np.float32)
    fps_counter = 0
    last_fps_report = time.time()
    last_state = machine.state

    kb = KeyboardListener()

    def clear_buffer():
        nonlocal buffer_cleared
        for k in list(frame_buffer.keys()):
            frame_buffer[k].clear()
        buffer_cleared = True

    def send_command(new_mode: StreamMode):
        if new_mode == StreamMode.POSE:
            pub.send(build_command_message(start=True, stop=False, planner=False))
        elif new_mode == StreamMode.PLANNER_IDLE:
            pub.send(build_command_message(start=True, stop=False, planner=True))
        elif new_mode == StreamMode.OFF:
            pub.send(build_command_message(start=False, stop=True, planner=True))

    def arm(reason: str):
        nonlocal started, session_armed_t, yaw_ref_inv
        started = True
        session_armed_t = time.monotonic()
        yaw_ref_inv = None  # recalibrate pilot heading on each session
        machine.arm(time.monotonic())
        print(f"[Bridge] ARMED ({reason}) — robot will resume after ~"
              f"{limits.resume_dwell_s:.0f}s of valid tracking")

    try:
        frame_start = time.time()
        while True:
            now = time.monotonic()

            # --- Keyboard ---
            key = kb.get_key()
            if key in ("o", "q", "\x03"):
                print("\n[Bridge] STOP requested")
                send_command(StreamMode.OFF)
                break
            elif key == "s":
                if not machine.armed:
                    arm("key s")
            elif key == "p" and started:
                if machine.armed:
                    machine.force_idle(now, "paused")
                    print("[Bridge] PAUSED — blending to neutral, then idle. "
                          "[s] to resume.")
            elif key == "n" and started:
                machine.force_idle(now, "next_pilot")
                print("[Bridge] NEXT PILOT — blending to neutral. Press [s] when "
                      "the next person is in position.")

            if args.auto_start and not started and receiver.n_received > 0:
                arm("auto_start")

            if not started:
                time.sleep(frame_time)
                frame_start = time.time()
                continue

            # --- Session timer ---
            if (
                session_timeout > 0
                and machine.armed
                and now - session_armed_t > session_timeout
            ):
                machine.force_idle(now, "session_timeout")
                print(f"[Bridge] SESSION TIMEOUT ({session_timeout:.0f}s) — "
                      "blending to neutral. Press [s] for the next session.")

            # --- Compute live frame (interpolated, only if fresh + valid) ---
            live = None
            prev, curr = receiver.get_pair()
            now_ns = time.monotonic_ns()
            if (
                curr is not None
                and prev is not None
                and curr["timestamp_ns"] > prev["timestamp_ns"]
                and (now_ns - curr["timestamp_ns"]) <= watchdog_ns
            ):
                src_interval = curr["timestamp_ns"] - prev["timestamp_ns"]
                playback_ns = now_ns - src_interval
                alpha = (playback_ns - prev["timestamp_ns"]) / float(src_interval)
                alpha = min(max(alpha, 0.0), 1.0)

                use_pose = _interp_pose_axis_angle(
                    prev["smpl_pose_np"], curr["smpl_pose_np"], alpha
                ).astype(np.float32)
                use_joints = (
                    (1.0 - alpha) * prev["smpl_joints_np"] + alpha * curr["smpl_joints_np"]
                ).astype(np.float32)
                use_body_quat = _quat_lerp_normalized(
                    prev["body_quat_np"], curr["body_quat_np"], alpha
                ).astype(np.float32)
                # Keep recalibrating until tracking stabilizes; freeze while TRACKING.
                # Reference heading = the neutral frame's heading (SMPL forward
                # convention), so the pilot's facing at resume maps to "forward".
                if yaw_ref_inv is None or machine.state != SafetyState.TRACKING:
                    yaw_ref_inv = _quat_mul_wxyz(
                        neutral_yaw, _yaw_inverse_wxyz(use_body_quat)
                    )
                use_body_quat = _quat_mul_wxyz(yaw_ref_inv, use_body_quat).astype(
                    np.float32
                )
                live = {
                    "smpl_pose": use_pose,
                    "smpl_joints": use_joints,
                    "body_quat": use_body_quat,
                    "joint_pos": compute_g1_wrist_joint_pos(use_pose),
                }

            # --- Burst instability filter: too many rejected GEM samples in a
            # short window means the stream is erratic; treat tracking as
            # invalid so the machine holds/blends to neutral and only resumes
            # after a clean dwell.
            n_recent_rejects = receiver.recent_rejects(limits.unstable_window_s)
            if live is not None and n_recent_rejects >= limits.unstable_max_rejects:
                live = None
                if not unstable_reported:
                    unstable_reported = True
                    print(f"[Bridge] UNSTABLE tracking — {n_recent_rejects} rejects "
                          f"in {limits.unstable_window_s:.0f}s; holding until stable")
            elif unstable_reported and n_recent_rejects == 0:
                unstable_reported = False
                print("[Bridge] tracking stable again")

            # --- Safety state machine ---
            out = machine.tick(now, live)
            if out.state != last_state:
                print(f"[Bridge] safety: {last_state.name} -> {out.state.name} "
                      f"({out.reason})")
                last_state = out.state

            if out.frame is None:
                # IDLE: robot stands via the planner
                if wire_mode != "planner":
                    wire_mode = "planner"
                    send_command(StreamMode.PLANNER_IDLE)
                pub.send(
                    build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0)
                )
                fps_counter += 1
            else:
                if wire_mode != "pose":
                    wire_mode = "pose"
                    clear_buffer()

                frame_buffer["smpl_pose"].append(out.frame["smpl_pose"])
                frame_buffer["smpl_joints"].append(out.frame["smpl_joints"])
                frame_buffer["body_quat_w"].append(out.frame["body_quat"])
                frame_buffer["frame_index"].append(int(step))
                frame_buffer["joint_pos"].append(out.frame["joint_pos"])

                N = len(frame_buffer["frame_index"])
                buffer_is_full = N >= args.num_frames_to_send
                if buffer_is_full and buffer_cleared:
                    buffer_cleared = False
                    send_command(StreamMode.POSE)
                    print("[Bridge] Buffer full — streaming pose + START command sent")

                if buffer_is_full and not buffer_cleared:
                    src = curr if curr is not None else {}
                    numpy_data = {
                        "smpl_pose": np.stack(frame_buffer["smpl_pose"], axis=0),
                        "smpl_joints": np.stack(frame_buffer["smpl_joints"], axis=0),
                        "body_quat_w": np.stack(frame_buffer["body_quat_w"], axis=0),
                        "joint_pos": np.stack(frame_buffer["joint_pos"], axis=0),
                        "joint_vel": np.zeros((N, 29)),
                        "frame_index": np.array(
                            frame_buffer["frame_index"], dtype=np.int64
                        ),
                        "left_trigger": np.array([0.0], dtype=np.float32),
                        "right_trigger": np.array([0.0], dtype=np.float32),
                        "left_grip": np.array([0.0], dtype=np.float32),
                        "right_grip": np.array([0.0], dtype=np.float32),
                        "pico_dt": np.array(
                            [float(src.get("dt", 0.0))], dtype=np.float32
                        ),
                        "pico_fps": np.array(
                            [float(src.get("fps", 0.0))], dtype=np.float32
                        ),
                        "timestamp_realtime": np.array(
                            [float(src.get("timestamp_realtime", time.time()))],
                            dtype=np.float64,
                        ),
                        "timestamp_monotonic": np.array(
                            [float(src.get("timestamp_ns", now_ns)) * 1e-9],
                            dtype=np.float64,
                        ),
                        "left_hand_joints": left_hand_joints.reshape(-1),
                        "right_hand_joints": right_hand_joints.reshape(-1),
                        "toggle_data_collection": np.array([False], dtype=bool),
                        "toggle_data_abort": np.array([False], dtype=bool),
                        "heading_increment": np.array([0.0], dtype=np.float32),
                    }
                    pub.send(pack_pose_message(numpy_data, topic="pose"))

                    if record_dir:
                        out_path = os.path.join(
                            record_dir, f"pose_{record_idx:06d}.npz"
                        )
                        np.savez_compressed(out_path, **numpy_data)
                        record_idx += 1

                step += 1
                fps_counter += 1

            now_wall = time.time()
            if now_wall - last_fps_report >= 5.0:
                fps = fps_counter / (now_wall - last_fps_report)
                print(
                    f"[Bridge] mode={out.state.name} out_fps={fps:.1f} "
                    f"gem_frames={receiver.n_received} step={step} "
                    f"rejected={receiver.n_rejected}"
                    + (
                        f" last_reject={receiver.last_reject_reason}"
                        if receiver.last_reject_reason
                        else ""
                    )
                )
                fps_counter = 0
                last_fps_report = now_wall

            elapsed = time.time() - frame_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            frame_start = time.time()

    except KeyboardInterrupt:
        print("\n[Bridge] Interrupted — sending STOP")
        send_command(StreamMode.OFF)
    finally:
        kb.restore()
        receiver.stop()
        time.sleep(0.1)
        pub.close(0)
        print("[Bridge] Shutdown complete")


if __name__ == "__main__":
    main()
