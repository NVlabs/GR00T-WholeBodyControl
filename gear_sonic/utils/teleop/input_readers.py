"""Input source readers for body tracking data.

PicoReader         -- pulls data from XRoboToolkit SDK (Pico headset).
IsaacTeleopReader  -- in-process IsaacTeleop / CloudXR DeviceIO session.
"""

import logging
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

try:
    from gear_sonic.utils.teleop.isaac_teleop_client import IsaacTeleopClient
except ImportError:
    IsaacTeleopClient = None


class PicoReader:
    """Background reader that pulls Pico/XRT data and computes dt/FPS."""

    STALE_TIMEOUT = 5.0

    def __init__(self, max_queue_size: int = 15):
        del max_queue_size
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()
        self._last_new_data_time = time.monotonic()
        self._disconnected = threading.Event()

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    @property
    def disconnected(self) -> bool:
        return self._disconnected.is_set()

    def clear_disconnect(self):
        self._disconnected.clear()
        self._last_new_data_time = time.monotonic()
        self._last_stamp_ns = None
        self._fps_ema = 0.0

    def get_timestamp_ns(self) -> int:
        if xrt is None:
            return 0
        return int(xrt.get_time_stamp_ns())

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if xrt is None or not xrt.is_body_data_available():
                if (
                    time.monotonic() - self._last_new_data_time > self.STALE_TIMEOUT
                    and not self._disconnected.is_set()
                ):
                    logger.warning(
                        "[PicoReader] No new data for %.1fs, flagging disconnect",
                        self.STALE_TIMEOUT,
                    )
                    self._disconnected.set()
                time.sleep(0.001)
                continue

            stamp_ns = xrt.get_time_stamp_ns()
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                if (
                    time.monotonic() - self._last_new_data_time > self.STALE_TIMEOUT
                    and not self._disconnected.is_set()
                ):
                    logger.warning(
                        "[PicoReader] Timestamps stale for %.1fs, flagging disconnect",
                        self.STALE_TIMEOUT,
                    )
                    self._disconnected.set()
                time.sleep(0.000001)
                continue

            self._last_new_data_time = time.monotonic()
            if self._disconnected.is_set():
                logger.info("[PicoReader] Fresh data received, connection restored")
                self._disconnected.clear()

            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns

            try:
                body_poses = xrt.get_body_joints_pose()
                sample = {
                    "body_poses_np": np.array(body_poses),
                    "timestamp_realtime": time.time(),
                    "timestamp_monotonic": time.monotonic(),
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample

                now = time.time()
                if now - last_report >= 5.0:
                    logger.info(
                        "[PicoReader] dt_ts: %.2f ms, fps: %.2f",
                        device_dt * 1000.0,
                        self._fps_ema,
                    )
                    last_report = now
            except Exception:
                logger.exception("[PicoReader] read error")


def _attr_or_item(obj: Any, name: str, default: Any = None) -> Any:
    """Return ``obj.<name>`` if present, else ``obj[<name>]`` if dict-like, else ``default``."""
    if obj is None:
        return default
    sentinel = object()
    val = getattr(obj, name, sentinel)
    if val is not sentinel:
        return val
    if hasattr(obj, "get"):
        try:
            return obj.get(name, default)
        except Exception:
            return default
    return default


def _vec3(point: Any) -> tuple[float, float, float] | None:
    """Extract (x, y, z) from a point-like (.x/.y/.z attrs or 3-sequence)."""
    if point is None:
        return None
    x = _attr_or_item(point, "x")
    y = _attr_or_item(point, "y")
    z = _attr_or_item(point, "z")
    if x is not None and y is not None and z is not None:
        return float(x), float(y), float(z)
    try:
        return float(point[0]), float(point[1]), float(point[2])
    except Exception:
        return None


def _quat_xyzw(orientation: Any) -> tuple[float, float, float, float] | None:
    """Extract (qx, qy, qz, qw) from an orientation-like."""
    if orientation is None:
        return None
    qx = _attr_or_item(orientation, "x")
    qy = _attr_or_item(orientation, "y")
    qz = _attr_or_item(orientation, "z")
    qw = _attr_or_item(orientation, "w")
    if all(v is not None for v in (qx, qy, qz, qw)):
        return float(qx), float(qy), float(qz), float(qw)
    try:
        return (
            float(orientation[0]),
            float(orientation[1]),
            float(orientation[2]),
            float(orientation[3]),
        )
    except Exception:
        return None


# Number of joints in the Isaac Teleop FullBodyPoseT (XR_BD_body_tracking).
# Mirrors core.BodyJoint.NUM_JOINTS in Isaac Teleop's schema bindings.
_NUM_BODY_JOINTS = 24

_UNRECOGNISED_SCHEMA_LOGGED: set[str] = set()


def _log_unrecognised_schema_once(body_data: Any) -> None:
    """One-shot diagnostic if ``body_data`` doesn't look like either schema we
    expect. Logs the type once per process so it doesn't flood the streamer.
    """
    type_name = type(body_data).__name__
    if type_name in _UNRECOGNISED_SCHEMA_LOGGED:
        return
    _UNRECOGNISED_SCHEMA_LOGGED.add(type_name)
    attrs = sorted(a for a in dir(body_data) if not a.startswith("_"))[:25]
    logger.warning(
        "[IsaacTeleopReader] Unrecognised body_data schema: type=%s attrs=%s. "
        "Update _body_data_to_24x7() to handle this layout.",
        type_name,
        attrs,
    )


def _body_data_to_24x7(body_data: Any) -> np.ndarray | None:
    """Convert ``FullBodyTracker.get_body_pose().data`` to a (24, 7) array.

    Returns ``None`` while no joint is valid (typical when the headset isn't
    connected yet — every ``BodyJointPose.is_valid`` is False, the streamer
    keeps polling and the C++ deploy doesn't see fake zero pose).

    Two accepted schemas:

    Schema A — Isaac Teleop ``FullBodyPoseT`` (DeviceIO direct).
        Defined in IsaacTeleop's ``schema/full_body.fbs`` /
        ``schema/python/full_body_bindings.h``::

            FullBodyPoseT.joints                    → BodyJoints (attr)
            BodyJoints.joints(index)                → BodyJointPose (METHOD; index 0..23)
            BodyJointPose.is_valid                  → bool
            BodyJointPose.pose.position             → Point (.x .y .z)
            BodyJointPose.pose.orientation          → Quaternion (.x .y .z .w)

    Schema B — msgpack wire format published by ``teleop_ros2_ref`` (kept for
        compatibility with ROS2 bridges; consumed when ``body_data`` already
        looks like a dict with ``joint_positions`` / ``joint_orientations``).
    """
    if body_data is None:
        return None

    # Schema B: msgpack wire format (teleop_ros2_ref-compatible).
    positions = _attr_or_item(body_data, "joint_positions")
    orientations = _attr_or_item(body_data, "joint_orientations")
    if positions is not None and orientations is not None:
        n = min(len(positions), len(orientations), _NUM_BODY_JOINTS)
        if n == 0:
            return None
        body_poses = np.zeros((_NUM_BODY_JOINTS, 7), dtype=np.float32)
        for i in range(n):
            pos = _vec3(positions[i])
            quat = _quat_xyzw(orientations[i])
            if pos is None or quat is None:
                continue
            body_poses[i, :3] = pos
            body_poses[i, 3:] = quat
        return body_poses

    # Schema A: native FullBodyPoseT — joints exposed via
    # BodyJoints.joints(index) method (one BodyJointPose per call).
    joints_container = getattr(body_data, "joints", None)
    if joints_container is None:
        _log_unrecognised_schema_once(body_data)
        return None
    get_joint = getattr(joints_container, "joints", None)
    if not callable(get_joint):
        _log_unrecognised_schema_once(body_data)
        return None

    body_poses = np.zeros((_NUM_BODY_JOINTS, 7), dtype=np.float32)
    any_valid = False
    for i in range(_NUM_BODY_JOINTS):
        try:
            joint = get_joint(i)
        except Exception:
            continue
        if joint is None:
            continue
        # Older builds may omit is_valid — default to True so we don't drop
        # samples on schema drift; per-field validity falls out below.
        if not getattr(joint, "is_valid", True):
            continue
        pose = getattr(joint, "pose", None)
        if pose is None:
            continue
        pos = _vec3(getattr(pose, "position", None))
        quat = _quat_xyzw(getattr(pose, "orientation", None))
        if pos is None or quat is None:
            continue
        body_poses[i, :3] = pos
        body_poses[i, 3:] = quat
        any_valid = True

    return body_poses if any_valid else None


def _controller_inputs_to_dict_side(snapshot: Any) -> dict[str, Any] | None:
    """Project one ControllerSnapshot.inputs into the dict shape consumed by helpers."""
    if snapshot is None:
        return None
    inputs = _attr_or_item(snapshot, "inputs")
    if inputs is None:
        return None
    return {
        "trigger_value": float(_attr_or_item(inputs, "trigger_value", 0.0) or 0.0),
        "squeeze_value": float(_attr_or_item(inputs, "squeeze_value", 0.0) or 0.0),
        "thumbstick_x": float(_attr_or_item(inputs, "thumbstick_x", 0.0) or 0.0),
        "thumbstick_y": float(_attr_or_item(inputs, "thumbstick_y", 0.0) or 0.0),
        "thumbstick_click": float(_attr_or_item(inputs, "thumbstick_click", 0.0) or 0.0),
        "primary_click": float(_attr_or_item(inputs, "primary_click", 0.0) or 0.0),
        "secondary_click": float(_attr_or_item(inputs, "secondary_click", 0.0) or 0.0),
    }


def _build_controller_dict(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert ``IsaacTeleopClient._get_tracker_data()`` into the controller dict
    schema that ``pico_manager_thread_server`` consumes (left/right trigger,
    squeeze, thumbstick, click, primary/secondary click)."""
    if raw is None:
        return None

    left = _controller_inputs_to_dict_side(raw.get("left_controller"))
    right = _controller_inputs_to_dict_side(raw.get("right_controller"))
    if left is None and right is None:
        return None

    out: dict[str, Any] = {}
    if left is not None:
        out["left_trigger_value"] = left["trigger_value"]
        out["left_squeeze_value"] = left["squeeze_value"]
        out["left_thumbstick"] = [left["thumbstick_x"], left["thumbstick_y"]]
        out["left_thumbstick_click"] = left["thumbstick_click"]
        out["left_primary_click"] = left["primary_click"]
        out["left_secondary_click"] = left["secondary_click"]
    if right is not None:
        out["right_trigger_value"] = right["trigger_value"]
        out["right_squeeze_value"] = right["squeeze_value"]
        out["right_thumbstick"] = [right["thumbstick_x"], right["thumbstick_y"]]
        out["right_thumbstick_click"] = right["thumbstick_click"]
        out["right_primary_click"] = right["primary_click"]
        out["right_secondary_click"] = right["secondary_click"]
    return out


class IsaacTeleopReader:
    """Background reader using the in-process IsaacTeleop / CloudXR DeviceIO session.

    Drop-in alternative to ``PicoReader`` — same ``get_latest()`` /
    ``get_controller_data()`` contract. Hosts the CloudXR runtime in-process
    via :class:`IsaacTeleopClient` (no separate publisher container, no host
    ``~/.cloudxr`` sharing required).
    """

    STALE_TIMEOUT = 5.0

    def __init__(
        self,
        max_queue_size: int = 15,
        use_adb: bool = False,
        poll_hz: float = 90.0,
    ):
        del max_queue_size

        if IsaacTeleopClient is None:
            raise RuntimeError(
                "isaacteleop is required for --input-source isaac-teleop but was not "
                "found. Install via install_scripts/install_pico.sh, which runs:\n"
                "  uv pip install 'isaacteleop[cloudxr]~=1.4.0' --prerelease=allow "
                "--extra-index-url https://pypi.nvidia.com"
            )

        self._client = IsaacTeleopClient(use_adb=use_adb)
        self._period = 1.0 / max(1.0, float(poll_hz))

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._ctrl_lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._latest_controller: dict[str, Any] | None = None
        self._fps_ema = 0.0
        self._last_stamp_ns: int | None = None
        self._last_new_data_time = time.monotonic()
        self._disconnected = threading.Event()
        self._unrecognised_logged = False

    def start(self) -> None:
        self._client.start_streaming()
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self._client.close()
        except Exception:
            logger.exception("Failed to close IsaacTeleopClient cleanly")

    def get_latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest

    def get_controller_data(self) -> dict[str, Any] | None:
        with self._ctrl_lock:
            return self._latest_controller

    @property
    def disconnected(self) -> bool:
        return self._disconnected.is_set()

    def clear_disconnect(self) -> None:
        self._disconnected.clear()
        self._last_new_data_time = time.monotonic()
        self._last_stamp_ns = None
        self._fps_ema = 0.0

    def get_timestamp_ns(self) -> int:
        with self._lock:
            sample = self._latest
        return int(sample["timestamp_ns"]) if sample else 0

    def _run(self) -> None:
        last_report = time.time()
        while not self._stop.is_set():
            try:
                raw = self._client._get_tracker_data()  # noqa: SLF001 — internal API by design
            except Exception:
                logger.exception("[IsaacTeleopReader] DeviceIO update failed")
                time.sleep(self._period)
                continue

            if raw is None:
                if (
                    time.monotonic() - self._last_new_data_time > self.STALE_TIMEOUT
                    and not self._disconnected.is_set()
                ):
                    logger.warning(
                        "[IsaacTeleopReader] No DeviceIO data for %.1fs, flagging disconnect",
                        self.STALE_TIMEOUT,
                    )
                    self._disconnected.set()
                time.sleep(self._period)
                continue

            controller = _build_controller_dict(raw)
            if controller is not None:
                with self._ctrl_lock:
                    self._latest_controller = controller

            body_poses = _body_data_to_24x7(raw.get("full_body"))
            if body_poses is None:
                if not self._unrecognised_logged and not _attr_or_item(
                    raw.get("full_body"), "joint_positions"
                ):
                    self._unrecognised_logged = True
                time.sleep(self._period)
                continue

            stamp_ns = int(self._client.get_timestamp_ns())
            prev_stamp_ns = self._last_stamp_ns
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns
            self._last_new_data_time = time.monotonic()
            if self._disconnected.is_set():
                logger.info("[IsaacTeleopReader] Fresh data received, connection restored")
                self._disconnected.clear()

            sample = {
                "body_poses_np": body_poses,
                "timestamp_realtime": time.time(),
                "timestamp_monotonic": time.monotonic(),
                "timestamp_ns": stamp_ns,
                "dt": device_dt,
                "fps": self._fps_ema,
            }
            with self._lock:
                self._latest = sample

            now = time.time()
            if now - last_report >= 5.0:
                logger.info(
                    "[IsaacTeleopReader] dt: %.2f ms, fps: %.2f",
                    device_dt * 1000.0,
                    self._fps_ema,
                )
                last_report = now

            time.sleep(self._period)

