"""
Quest 3 body reader — test connection and stream body pose data.

Use this to:
  1. Test ADB connection to Quest 3 (run with --test-adb).
  2. Stream body pose in the same format as PicoReader for pico_manager_thread_server.
  3. Stream VR 3-point pose (L-Wrist, R-Wrist, Head) from Meta Quest controllers via meta_quest_teleop.

Expected sample format (compatible with PicoReader):
  body_poses_np: np.ndarray shape (24, 7) — 24 SMPL-like joints, each [x, y, z, qx, qy, qz, qw]
  (Unity frame, scalar-last quaternion)

For source="meta_quest", sample also includes:
  vr_3pt_pose: np.ndarray shape (3, 7) — [x, y, z, qw, qx, qy, qz] per row (robot frame)
  Row 0: Left Wrist, Row 1: Right Wrist, Row 2: Head (default, Quest has no head tracking)

Usage:
  # Test ADB only
  python -m gear_sonic.utils.teleop.readers.quest_reader --test-adb

  # Run reader with synthetic data (no Quest needed) to test pipeline
  python -m gear_sonic.utils.teleop.readers.quest_reader --synthetic

  # Run reader with Meta Quest controllers (meta_quest_teleop)
  python -m gear_sonic.utils.teleop.readers.quest_reader --source meta_quest

  # Run reader using ADB (implement read_quest_body_via_adb for your app)
  python -m gear_sonic.utils.teleop.readers.quest_reader --source adb
"""

import argparse
import subprocess
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as sRot


# -----------------------------------------------------------------------------
# ADB connection test
# -----------------------------------------------------------------------------

def check_adb_connection() -> bool:
    """Check if ADB is available and at least one device is connected."""
    try:
        out = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            print("[QuestReader] adb devices failed")
            return False
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        # First line is "List of devices attached"
        devices = [l for l in lines[1:] if l and not l.startswith("*")]
        if not devices:
            print("[QuestReader] No devices found. Connect Quest 3 via USB and enable USB debugging.")
            return False
        print(f"[QuestReader] ADB devices: {devices}")
        return True
    except FileNotFoundError:
        print("[QuestReader] 'adb' not found. Install Android platform tools.")
        return False
    except subprocess.TimeoutExpired:
        print("[QuestReader] adb devices timed out")
        return False


def test_adb():
    """Standalone test: verify ADB and optionally run a shell command."""
    print("Checking ADB connection to Quest 3...")
    if not check_adb_connection():
        sys.exit(1)
    print("ADB connection OK.")
    # Optional: run a simple command on device
    try:
        out = subprocess.run(
            ["adb", "shell", "getprop", "ro.product.model"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            print(f"Device model: {out.stdout.strip()}")
    except Exception as e:
        print(f"Optional shell check: {e}")


# -----------------------------------------------------------------------------
# Default SMPL-like 24-joint pose (identity / T-pose style)
# Used when no real data is available (synthetic mode or fallback).
# -----------------------------------------------------------------------------

def default_smpl_pose_24() -> np.ndarray:
    """Return a default (24, 7) pose: zeros for position, identity quat for rotation."""
    pose = np.zeros((24, 7), dtype=np.float32)
    # Identity quaternion (scalar-last: qx, qy, qz, qw)
    pose[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    # Optional: set root height
    pose[0, 2] = 1.0
    return pose


# -----------------------------------------------------------------------------
# VR 3-point pose from Meta Quest controllers (meta_quest_teleop)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Head reference frame (row 2 of vr_3pt_pose) — Quest has no head tracking, so we use
# a fixed default. This should match the robot's head position for correct waist/neck
# calibration. Origin: robot pelvis. Frame: X=forward, Y=left, Z=up (ROS convention).
#
# VR_3PT_HEAD_POSITION: [x, y, z] in meters — head position relative to pelvis.
#   Default from C++ zmq_manager.hpp. For G1: ~0.4m above pelvis, slightly forward.
#   To use G1 FK: get_g1_key_frame_poses(robot_model)["torso"] + HEAD_LINK_LENGTH.
#
# VR_3PT_HEAD_ORIENTATION: [qw, qx, qy, qz] — head orientation (identity ≈ looking forward).
# -----------------------------------------------------------------------------
VR_3PT_HEAD_POSITION = np.array([0.0241, -0.0081, 0.4028], dtype=np.float32)
VR_3PT_HEAD_ORIENTATION = np.array([0.9991, 0.011, 0.0402, -0.0002], dtype=np.float32)  # wxyz

# Home positions for calibration (robot frame, meters) — fallback when G1 FK not available
META_QUEST_HOME_LEFT = np.array([0.15, 0.25, 0.45])
META_QUEST_HOME_RIGHT = np.array([0.15, -0.25, 0.45])


def _g1_pose_to_4x4(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build 4x4 transform from G1 FK pose (position + orientation_wxyz)."""
    # scipy expects [qx, qy, qz, qw]
    q_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    R = sRot.from_quat(q_xyzw).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _matrix_to_quat_wxyz(matrix_3x3: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz] (scalar-first)."""
    r = sRot.from_matrix(matrix_3x3)
    q_xyzw = r.as_quat()  # scipy returns [x, y, z, w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


# Permutation: meta_quest axes -> ROS (X=forward, Y=left, Z=up)
# meta_quest has forward in Z, up in Y, left in X -> we need [z,x,y]
_QUEST_TO_ROS_P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)


def _apply_quest_yz_swap(pose_4x4: np.ndarray) -> np.ndarray:
    """
    Correct meta_quest_teleop axes to ROS convention (X=forward, Y=left, Z=up).

    The APK maps: forward->Z, up->Y, left->X. We permute [x,y,z] -> [z,x,y]
    and negate position to fix inverted directions (forward->back, left->right).
    """
    out = pose_4x4.copy()
    # Position: [x,y,z] -> [z,-x,-y]
    p = np.array([pose_4x4[2, 3], pose_4x4[0, 3], pose_4x4[1, 3]])
    out[:3, 3] = [p[0], -p[1], -p[2]]
    # Rotation: P @ R @ P^T (orientation stays same; position sign flip is enough)
    R = pose_4x4[:3, :3]
    out[:3, :3] = _QUEST_TO_ROS_P @ R @ _QUEST_TO_ROS_P.T
    return out


def quest_controllers_to_vr_3pt_pose(
    left_4x4: np.ndarray | None,
    right_4x4: np.ndarray | None,
    head_pos: np.ndarray | None = None,
    head_quat_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert Meta Quest controller 4x4 poses to VR 3-point pose for SONIC.

    Args:
        left_4x4: 4x4 transform for left controller (ROS frame = robot frame)
        right_4x4: 4x4 transform for right controller
        head_pos: Optional head position [x,y,z]. Default: VR_3PT_HEAD_POSITION
        head_quat_wxyz: Optional head quat [qw,qx,qy,qz]. Default: VR_3PT_HEAD_ORIENTATION

    Returns:
        vr_3pt_pose: np.ndarray shape (3, 7) — each row [x, y, z, qw, qx, qy, qz]
        Row 0: Left Wrist, Row 1: Right Wrist, Row 2: Head (robot head frame, see VR_3PT_HEAD_*)
    """
    out = np.zeros((3, 7), dtype=np.float32)
    head_pos = head_pos if head_pos is not None else VR_3PT_HEAD_POSITION
    head_quat = head_quat_wxyz if head_quat_wxyz is not None else VR_3PT_HEAD_ORIENTATION

    # Row 0, 1: controller poses (wrists). Row 2: head — fixed default when Quest has no head tracking.
    # Head frame origin should be at robot head for correct waist/neck calibration.
    if left_4x4 is not None:
        out[0, :3] = left_4x4[:3, 3]
        out[0, 3:7] = _matrix_to_quat_wxyz(left_4x4[:3, :3])
    else:
        out[0, :3] = META_QUEST_HOME_LEFT
        out[0, 3:7] = [1.0, 0.0, 0.0, 0.0]

    if right_4x4 is not None:
        out[1, :3] = right_4x4[:3, 3]
        out[1, 3:7] = _matrix_to_quat_wxyz(right_4x4[:3, :3])
    else:
        out[1, :3] = META_QUEST_HOME_RIGHT
        out[1, 3:7] = [1.0, 0.0, 0.0, 0.0]

    # Row 2: Head (robot head frame, see VR_3PT_HEAD_POSITION / VR_3PT_HEAD_ORIENTATION above)
    out[2, :3] = head_pos
    out[2, 3:7] = head_quat
    return out


# -----------------------------------------------------------------------------
# Read body from Quest via ADB (stub — implement with your app)
# -----------------------------------------------------------------------------

def read_quest_body_via_adb() -> np.ndarray | None:
    """
    Read body pose from Quest 3 via ADB.

    Implement according to your setup, for example:
      - Quest app writes pose to /sdcard/body_pose.bin or .json; adb pull or exec-out cat
      - Quest app opens TCP server; adb forward tcp:LOCAL tcp:REMOTE, then socket read
      - Your PC script receives from Quest app over ADB or network

    Returns:
      np.ndarray shape (N, 7) with [x, y, z, qx, qy, qz, qw] per joint (Quest format).
      Or None if no data this frame.
    """
    # Stub: no real read yet
    return None


def quest_joints_to_smpl_24(quest_joints: np.ndarray) -> np.ndarray:
    """
    Map Quest joint array to SMPL-like 24 joints.

    If quest_joints already has 24 rows, use them (optionally reorder).
    If Quest uses 70 joints (Meta IOBT), map key indices to SMPL 24.
    """
    if quest_joints is None or quest_joints.size == 0:
        return default_smpl_pose_24()

    n = quest_joints.shape[0]
    smpl = np.zeros((24, 7), dtype=np.float32)
    smpl[:, 3:7] = [0.0, 0.0, 0.0, 1.0]

    if n >= 24:
        smpl[:24] = quest_joints[:24]
        return smpl

    # Minimal mapping for 70-joint Meta format (indices to adjust per Meta docs)
    # SMPL: 0=root, 12=neck, 22=left_wrist, 23=right_wrist
    meta_to_smpl = {
        0: 0,   # Root
        7: 12,  # Neck (example)
        24: 22, # Left wrist (example)
        25: 23, # Right wrist (example)
    }
    for meta_idx, smpl_idx in meta_to_smpl.items():
        if meta_idx < n:
            smpl[smpl_idx] = quest_joints[meta_idx]
    return smpl


# -----------------------------------------------------------------------------
# QuestReader — same interface as PicoReader for drop-in use
# -----------------------------------------------------------------------------

class QuestReader:
    """
    Background reader that produces body pose samples compatible with PicoReader.

    Use source="synthetic" to emit dummy poses (for testing pipeline without Quest).
    Use source="meta_quest" to read controller poses via meta_quest_teleop (VR 3-point format).
    Use source="adb" to read from Quest via ADB (implement read_quest_body_via_adb).
    """

    def __init__(
        self,
        source: str = "synthetic",
        max_queue_size: int = 15,
        ip_address: str | None = None,
        calibration_threshold: float = 0.5,
        home_left_pose: np.ndarray | None = None,
        home_right_pose: np.ndarray | None = None,
    ):
        """
        Args:
            home_left_pose: 4x4 transform for left wrist reference (G1 FK default).
                If None, uses META_QUEST_HOME_LEFT (position only).
            home_right_pose: 4x4 transform for right wrist reference (G1 FK default).
                If None, uses META_QUEST_HOME_RIGHT (position only).
        """
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._latest = None
        self._lock = threading.Lock()
        self._source = source.lower()
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._frame_count = 0
        self._ip_address = ip_address
        self._calibration_threshold = calibration_threshold
        self._home_left_pose = home_left_pose
        self._home_right_pose = home_right_pose

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        if hasattr(self, "_meta_quest_reader") and self._meta_quest_reader is not None:
            try:
                self._meta_quest_reader.stop()
            except Exception:
                pass

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run_meta_quest(self, t_realtime: float, t_monotonic: float, stamp_ns: int, device_dt: float):
        """Read from MetaQuestReader and produce vr_3pt_pose sample."""
        if not hasattr(self, "_meta_quest_reader") or self._meta_quest_reader is None:
            if getattr(self, "_meta_quest_init_failed", False):
                return None
            try:
                from meta_quest_teleop.reader import MetaQuestReader  # pyright: ignore[reportMissingImports]

                self._meta_quest_reader = MetaQuestReader(
                    ip_address=self._ip_address,
                    run=True,
                )
                self._left_offset = None
                self._right_offset = None
                self._is_calibrated = False
                self._prev_trigger_pressed = False
                print("[QuestReader] MetaQuestReader started. Press right trigger to calibrate.")
            except ImportError as e:
                print(f"[QuestReader] meta_quest_teleop not installed: {e}")
                self._meta_quest_init_failed = True
                return None
            except SystemExit:
                print("[QuestReader] MetaQuestReader failed (no device?). Connect Quest via USB/WiFi.")
                self._meta_quest_init_failed = True
                self._meta_quest_reader = None
                return None

        reader = self._meta_quest_reader
        left_raw = reader.get_hand_controller_transform_ros("left")
        right_raw = reader.get_hand_controller_transform_ros("right")
        # Correct Y/Z swap: some Quest APKs map up to Y; we need Z up (ROS convention)
        if left_raw is not None:
            left_raw = _apply_quest_yz_swap(left_raw)
        if right_raw is not None:
            right_raw = _apply_quest_yz_swap(right_raw)
        trigger = float(reader.get_trigger_value("right"))
        trigger_pressed = trigger >= self._calibration_threshold

        # Calibrate on rising edge of trigger
        if trigger_pressed and not self._prev_trigger_pressed and left_raw is not None and right_raw is not None:
            try:
                if self._home_left_pose is not None and self._home_right_pose is not None:
                    home_left = self._home_left_pose.copy()
                    home_right = self._home_right_pose.copy()
                    ref_source = "G1 FK default"
                else:
                    home_left = np.eye(4)
                    home_left[:3, 3] = META_QUEST_HOME_LEFT
                    home_right = np.eye(4)
                    home_right[:3, 3] = META_QUEST_HOME_RIGHT
                    ref_source = "META_QUEST_HOME"
                self._left_offset = home_left @ np.linalg.inv(left_raw)
                self._right_offset = home_right @ np.linalg.inv(right_raw)
                self._is_calibrated = True
                print(f"[QuestReader] Calibration complete (right trigger, ref={ref_source}).")
            except np.linalg.LinAlgError:
                pass
        self._prev_trigger_pressed = trigger_pressed

        if left_raw is None or right_raw is None:
            return None
        if not self._is_calibrated:
            return None

        left_pose = self._left_offset @ left_raw
        right_pose = self._right_offset @ right_raw
        vr_3pt_pose = quest_controllers_to_vr_3pt_pose(left_pose, right_pose)

        # Controller state for manager mode switching (A,B,X,Y, joysticks, triggers, etc.)
        left_axis = reader.get_joystick_value("left")
        right_axis = reader.get_joystick_value("right")
        left_axis = (float(left_axis[0]), float(left_axis[1])) if len(left_axis) >= 2 else (0.0, 0.0)
        right_axis = (float(right_axis[0]), float(right_axis[1])) if len(right_axis) >= 2 else (0.0, 0.0)

        def _btn(name: str) -> bool:
            try:
                return bool(reader.get_button_state(name))
            except Exception:
                return False

        # Also provide minimal body_poses_np for compatibility (not used in VR_3PT)
        body_poses_np = default_smpl_pose_24()
        body_poses_np[22, :3] = left_pose[:3, 3]
        body_poses_np[23, :3] = right_pose[:3, 3]
        q_l = _matrix_to_quat_wxyz(left_pose[:3, :3])
        q_r = _matrix_to_quat_wxyz(right_pose[:3, :3])
        body_poses_np[22, 3:7] = [q_l[1], q_l[2], q_l[3], q_l[0]]  # wxyz -> xyzw
        body_poses_np[23, 3:7] = [q_r[1], q_r[2], q_r[3], q_r[0]]

        return {
            "body_poses_np": body_poses_np,
            "vr_3pt_pose": vr_3pt_pose,
            "timestamp_realtime": t_realtime,
            "timestamp_monotonic": t_monotonic,
            "timestamp_ns": stamp_ns,
            "dt": device_dt,
            "fps": self._fps_ema,
            # Controller state for manager (Quest-only)
            "button_a": _btn("A"),
            "button_b": _btn("B"),
            "button_x": _btn("X"),
            "button_y": _btn("Y"),
            "left_trigger": float(reader.get_trigger_value("left")),
            "right_trigger": float(reader.get_trigger_value("right")),
            "left_grip": float(reader.get_grip_value("left")),
            "right_grip": float(reader.get_grip_value("right")),
            "left_axis": left_axis,
            "right_axis": right_axis,
            "left_axis_click": _btn("LJ"),
            "right_axis_click": _btn("RJ"),
            "left_menu_button": _btn("leftMenu") or _btn("menu") or False,
        }

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            t_realtime = time.time()
            t_monotonic = time.monotonic()
            stamp_ns = int(t_monotonic * 1e9)
            prev_stamp_ns = self._last_stamp_ns
            device_dt = (stamp_ns - prev_stamp_ns) * 1e-9 if prev_stamp_ns is not None else 0.0
            if device_dt > 0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns

            sample = None
            if self._source == "synthetic":
                body_poses_np = default_smpl_pose_24()
                self._frame_count += 1
                body_poses_np[0, 0] = 0.1 * np.sin(self._frame_count * 0.1)
                sample = {
                    "body_poses_np": body_poses_np,
                    "timestamp_realtime": t_realtime,
                    "timestamp_monotonic": t_monotonic,
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
            elif self._source == "meta_quest":
                sample = self._run_meta_quest(t_realtime, t_monotonic, stamp_ns, device_dt)
            elif self._source == "adb":
                raw = read_quest_body_via_adb()
                body_poses_np = quest_joints_to_smpl_24(raw) if raw is not None else None
                if body_poses_np is not None:
                    sample = {
                        "body_poses_np": body_poses_np,
                        "timestamp_realtime": t_realtime,
                        "timestamp_monotonic": t_monotonic,
                        "timestamp_ns": stamp_ns,
                        "dt": device_dt,
                        "fps": self._fps_ema,
                    }

            if sample is not None:
                with self._lock:
                    self._latest = sample

            if time.time() - last_report >= 5.0:
                print(f"[QuestReader] source={self._source} fps={self._fps_ema:.2f}")
                last_report = time.time()

            time.sleep(0.02)


# -----------------------------------------------------------------------------
# CLI for testing connection and reader
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quest 3 reader — test ADB or run reader")
    parser.add_argument("--test-adb", action="store_true", help="Only test ADB connection and exit")
    parser.add_argument("--synthetic", action="store_true", help="Run reader with synthetic data (no Quest)")
    parser.add_argument(
        "--source",
        choices=["adb", "synthetic", "meta_quest"],
        default="synthetic",
        help="Reader source (meta_quest uses meta_quest_teleop)",
    )
    parser.add_argument("--ip-address", type=str, default=None, help="Quest IP for WiFi (meta_quest only)")
    parser.add_argument(
        "--calibration-threshold",
        type=float,
        default=0.5,
        help="Right trigger threshold for calibration (meta_quest, 0-1)",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Run reader for N seconds (default 10)")
    args = parser.parse_args()

    if args.test_adb:
        test_adb()
        return

    source = "synthetic" if args.synthetic else args.source
    print(f"Starting QuestReader (source={source}) for {args.duration}s...")
    reader = QuestReader(
        source=source,
        ip_address=args.ip_address,
        calibration_threshold=args.calibration_threshold,
    )
    reader.start()
    try:
        time.sleep(args.duration)
        sample = reader.get_latest()
        if sample is not None:
            print(f"Last sample: body_poses_np shape {sample['body_poses_np'].shape}, fps={sample['fps']:.2f}")
            if "vr_3pt_pose" in sample:
                v = sample["vr_3pt_pose"]
                print(f"  vr_3pt_pose shape {v.shape}: L={v[0,:3]}, R={v[1,:3]}, H={v[2,:3]}")
    finally:
        reader.stop()
    print("Done.")


if __name__ == "__main__":
    main()
