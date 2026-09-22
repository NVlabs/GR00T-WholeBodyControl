"""Pico/XRT and Isaac Teleop controller input readers."""

import subprocess
import threading
import time

import numpy as np

from .constants import JOYSTICK_DEADZONE
from .runtime import input_readers, xrt

class YawAccumulator:
    """Accumulates yaw heading angle based on joystick input."""

    def __init__(self, yaw_gain: float = 1.5, deadzone: float = JOYSTICK_DEADZONE):
        self.yaw_gain = yaw_gain
        self.deadzone = deadzone
        self.reset()

    def reset(self):
        """Reset facing direction to default (1,0,0)."""
        self.heading = [1.0, 0.0, 0.0]
        self.yaw_angle_rad = 0.0
        self.dyaw = 0.0
        print("YawAccumulator: reset yaw angle to 0.0")

    def yaw_angle(self) -> float:
        """Get current yaw angle in radians."""
        return self.yaw_angle_rad

    def yaw_angle_change(self) -> float:
        """Get current yaw angle change in radians."""
        return self.dyaw

    def update(self, rx: float, dt: float) -> list[float]:
        """
        Update facing direction based on right stick x-axis input.

        Args:
            rx: Right stick x-axis value (-1 to 1)
            dt: Time delta in seconds

        Returns:
            Facing direction as [x, y, 0.0]
        """
        self.dyaw = self.yaw_gain * (-rx) * dt
        if abs(rx) >= self.deadzone:
            self.yaw_angle_rad += self.dyaw
            self.heading = [np.cos(self.yaw_angle_rad), np.sin(self.yaw_angle_rad), 0.0]
        return self.heading

# def compute_latest_frame(parent_indices: list, device) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Pull body data from XRoboToolkit, compute local SMPL joints and body orientation.
#     Returns (smpl_joints_local_np [24,3], global_orient_quat_np [4,])
#     """
#     body_poses = xrt.get_body_joints_pose()
#     body_poses_np = np.array(body_poses)
#     return compute_from_body_poses(parent_indices, device, body_poses_np)


# Readers that expose `get_controller_data()` returning the IsaacTeleop
# controller_data dict schema (left/right trigger/squeeze, thumbstick, clicks).
# Tuple form keeps the dispatch sites uniform if/when a second reader speaks
# the same schema.
_ISAAC_TELEOP_READERS = (input_readers.IsaacTeleopReader,)


def get_controller_inputs(reader=None):
    """Fetch controller button/trigger states from XRoboToolkit or IsaacTeleop."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return False, 0.0, 0.0, 0.0, 0.0
        return (
            False,
            float(ctrl.get("left_trigger_value", 0.0)),
            float(ctrl.get("right_trigger_value", 0.0)),
            float(ctrl.get("left_squeeze_value", 0.0)),
            float(ctrl.get("right_squeeze_value", 0.0)),
        )
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_grip = xrt.get_left_grip()
    right_grip = xrt.get_right_grip()
    left_menu_button = xrt.get_left_menu_button()
    return left_menu_button, left_trigger, right_trigger, left_grip, right_grip


def get_controller_axes(reader=None):
    """Fetch joystick axes (lx, ly, rx, ry). Falls back to zeros if not available."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return 0.0, 0.0, 0.0, 0.0
        left_thumbstick = ctrl.get("left_thumbstick", [0.0, 0.0])
        right_thumbstick = ctrl.get("right_thumbstick", [0.0, 0.0])
        return (
            float(left_thumbstick[0]),
            float(left_thumbstick[1]),
            float(right_thumbstick[0]),
            float(right_thumbstick[1]),
        )
    if xrt is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        left_axis = xrt.get_left_axis()  # expected [x, y]
        right_axis = xrt.get_right_axis()  # expected [x, y]
        lx = float(left_axis[0]) if len(left_axis) >= 1 else 0.0
        ly = float(left_axis[1]) if len(left_axis) >= 2 else 0.0
        rx = float(right_axis[0]) if len(right_axis) >= 1 else 0.0
        ry = float(right_axis[1]) if len(right_axis) >= 2 else 0.0
        return lx, ly, rx, ry
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def get_menu_buttons(reader=None):
    """Fetch both menu buttons (left, right). Falls back to False if not available."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        return False, False
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_menu_button")
    right = _safe_btn("get_right_menu_button")
    return left, right


def get_axis_clicks(reader=None):
    """Fetch both axis click buttons (left, right). Falls back to False if not available."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return False, False
        return (
            float(ctrl.get("left_thumbstick_click", 0.0)) > 0.5,
            float(ctrl.get("right_thumbstick_click", 0.0)) > 0.5,
        )
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_axis_click")
    right = _safe_btn("get_right_axis_click")
    return left, right


def get_face_buttons(reader=None):
    """Fetch primary face buttons A and X. Returns (a_pressed, x_pressed)."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return False, False
        return (
            float(ctrl.get("right_primary_click", 0.0)) > 0.5,
            float(ctrl.get("left_primary_click", 0.0)) > 0.5,
        )
    if xrt is None:
        return False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        x_pressed = bool(xrt.get_X_button())
        return a_pressed, x_pressed
    except Exception:
        return False, False


def get_abxy_buttons(reader=None):
    """Fetch A,B,X,Y face buttons as booleans (a,b,x,y)."""
    if isinstance(reader, _ISAAC_TELEOP_READERS):
        ctrl = reader.get_controller_data()
        if ctrl is None:
            return False, False, False, False
        return (
            float(ctrl.get("right_primary_click", 0.0)) > 0.5,
            float(ctrl.get("right_secondary_click", 0.0)) > 0.5,
            float(ctrl.get("left_primary_click", 0.0)) > 0.5,
            float(ctrl.get("left_secondary_click", 0.0)) > 0.5,
        )
    if xrt is None:
        return False, False, False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        b_pressed = bool(xrt.get_B_button())
        x_pressed = bool(xrt.get_X_button())
        y_pressed = bool(xrt.get_Y_button())
        return a_pressed, b_pressed, x_pressed, y_pressed
    except Exception:
        return False, False, False, False
class PicoReader:
    """
    Background reader that pulls Pico/XRT data as fast as possible and computes dt/FPS.
    """

    def __init__(self, max_queue_size: int = 15):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_t = None
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    @property
    def disconnected(self) -> bool:
        return False

    def clear_disconnect(self):
        pass

    def get_timestamp_ns(self) -> int:
        if xrt is None:
            return 0
        return int(xrt.get_time_stamp_ns())

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if not xrt.is_body_data_available():
                time.sleep(0.001)
                continue
            stamp_ns = xrt.get_time_stamp_ns()
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(0.000001)
                continue
            # Compute device-based dt/fps using timestamp deltas (ns -> s)
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns
            t_realtime = time.time()
            t_monotonic = time.monotonic()
            try:
                body_poses = xrt.get_body_joints_pose()

                sample = {
                    "body_poses_np": np.array(body_poses),
                    "timestamp_realtime": t_realtime,
                    "timestamp_monotonic": t_monotonic,
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample
                now = time.time()
                if now - last_report >= 5.0:
                    print(
                        f"[PicoReader] dt_ts: {device_dt*1000.0:.2f} ms, fps: {self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as e:
                print(f"[PicoReader] read error: {e}")
def _init_input_source(
    input_source: str,
    buffer_size: int,
) -> "PicoReader | input_readers.IsaacTeleopReader":
    """Create, start, and wait for readiness of the requested teleop input source."""
    if input_source == "isaac-teleop":
        reader = input_readers.IsaacTeleopReader(max_queue_size=buffer_size)
        reader.start()
        print("Using Isaac Teleop (in-process CloudXR / DeviceIO), waiting for data...")
        while reader.get_latest() is None:
            print("waiting for Isaac Teleop body data (connect the headset to CloudXR)...")
            time.sleep(1)
        return reader

    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run Pico streaming."
        )

    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()
    return reader
