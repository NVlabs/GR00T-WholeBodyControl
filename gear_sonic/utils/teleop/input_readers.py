"""Input source readers for body tracking data.

PicoReader         -- pulls data from XRoboToolkit SDK (Pico headset).
IsaacTeleopReader  -- in-process IsaacCapture / CloudXR DeviceIO session.
"""

import logging
import threading
import time
from typing import Any, Callable, TYPE_CHECKING, TypedDict

import numpy as np

from gear_sonic.utils.teleop.skeleton_guards import (
    MatchOutcome,
    TrackersDisconnectedDetector,
    match_orientation_convention,
)

logger = logging.getLogger(__name__)

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

try:
    from gear_sonic.utils.teleop.isaac_teleop_client import IsaacTeleopClient
except ImportError:
    IsaacTeleopClient = None

if TYPE_CHECKING:  # import cycle at runtime; only needed for annotations
    from collections.abc import Mapping

    from gear_sonic.utils.teleop.isaac_teleop_client import FullBodyPose, TrackerSnapshot

# Minimum time between repeated "degenerate skeleton" log lines. Every degenerate
# frame is still refused regardless of this; it only throttles the ERROR spam.
_DEGENERATE_LOG_INTERVAL_NS = 2_000_000_000

#: Consecutive frames the geometry guard may fail to verify before it starts
#: refusing them. A healthy skeleton decides on every frame (the two classes
#: sit ~60 deg apart against a 20 deg threshold), so sustained inability to
#: tell is itself evidence something is wrong. Held long enough -- about half
#: a second at 60 Hz -- that transient noise cannot trip it, because a false
#: refusal stops teleop for a Pico operator who has nothing wrong.
_UNVERIFIED_REFUSE_FRAMES = 30

#: How far a joint quaternion's norm may sit from 1 before the frame is refused.
_QUAT_NORM_TOLERANCE = 0.1


class BodySample(TypedDict):
    """One body-tracking frame, as every reader's ``get_latest()`` returns it."""

    #: ``(24, 7)`` per joint: position xyz in metres, then orientation xyzw.
    body_poses_np: np.ndarray
    #: ``time.time()`` when the frame was read.
    timestamp_realtime: float
    #: ``time.monotonic()`` when the frame was read.
    timestamp_monotonic: float
    #: Device (or host monotonic) timestamp of the frame, in nanoseconds.
    timestamp_ns: int
    #: Seconds since the previous frame's ``timestamp_ns``; 0.0 for the first.
    dt: float
    #: Smoothed frame rate, in Hz.
    fps: float
    #: Bumped each time the reader withdraws its cached sample (a refused frame,
    #: a ``clear_latest()``). A consumer that sees it change must drop any
    #: history built from earlier samples, even if it never saw the cache empty.
    #: Readers that never withdraw a sample leave it at 0.
    epoch: int


class _ControllerSide(TypedDict):
    """One controller's inputs, before ``_build_controller_dict`` prefixes them by side."""

    trigger_value: float
    squeeze_value: float
    thumbstick_x: float
    thumbstick_y: float
    thumbstick_click: float
    primary_click: float
    secondary_click: float


class ControllerDict(TypedDict, total=False):
    """The controller schema ``pico_manager_thread_server`` consumes.

    Not total: a side is absent while its controller is not tracked.
    """

    left_trigger_value: float
    left_squeeze_value: float
    left_thumbstick: list[float]
    left_thumbstick_click: float
    left_primary_click: float
    left_secondary_click: float
    right_trigger_value: float
    right_squeeze_value: float
    right_thumbstick: list[float]
    right_thumbstick_click: float
    right_primary_click: float
    right_secondary_click: float


#: ``correct_body_orientations(orientations, profile)`` from isaacteleop.
CorrectionFn = Callable[[np.ndarray, str], np.ndarray]


class PicoReader:
    """Background reader that pulls Pico/XRT data and computes dt/FPS."""

    STALE_TIMEOUT = 5.0

    def __init__(self, max_queue_size: int = 15):
        del max_queue_size
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest: BodySample | None = None
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

    def get_latest(self) -> BodySample | None:
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
                sample: BodySample = {
                    "body_poses_np": np.array(body_poses),
                    "timestamp_realtime": time.time(),
                    "timestamp_monotonic": time.monotonic(),
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                    "epoch": 0,
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


# Number of joints in the IsaacCapture FullBodyPose (XR_BD_body_tracking).
# Mirrors core.BodyJoint.NUM_JOINTS in IsaacCapture's schema bindings.
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


def _body_data_to_24x7(
    body_data: "FullBodyPose | Mapping[str, Any] | None",
) -> np.ndarray | None:
    """Convert ``FullBodyTracker.get_body_pose()`` to a (24, 7) array.

    Returns ``None`` while no joint is valid (typical when the headset isn't
    connected yet — every ``BodyJointPose.is_valid`` is False, the streamer
    keeps polling and the C++ deploy doesn't see fake zero pose).

    Two accepted schemas:

    Schema A — IsaacCapture ``FullBodyPose`` (DeviceIO direct).
        Defined in IsaacCapture's ``schema/full_body.fbs`` /
        ``schema/python/full_body_bindings.h``::

            FullBodyPose.joints                     → BodyJoints (attr)
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

    # Schema A: native FullBodyPose — joints exposed via
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


def _controller_inputs_to_dict_side(snapshot: Any) -> _ControllerSide | None:
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


def _build_controller_dict(raw: "TrackerSnapshot | None") -> ControllerDict | None:
    """Convert ``IsaacTeleopClient._get_tracker_data()`` into the controller dict
    schema that ``pico_manager_thread_server`` consumes (left/right trigger,
    squeeze, thumbstick, click, primary/secondary click)."""
    if raw is None:
        return None

    left = _controller_inputs_to_dict_side(raw.left_controller)
    right = _controller_inputs_to_dict_side(raw.right_controller)
    if left is None and right is None:
        return None

    out: ControllerDict = {}
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


class SkeletonCorrectionUnavailable(RuntimeError):
    """The isaacteleop skeleton correction is needed but not importable."""


# Deliberately actionable rather than a bare ImportError: on an isaacteleop
# without the correction, this is the first thing a Quest operator hits.
_CORRECTION_UNAVAILABLE_MESSAGE = (
    "the isaacteleop skeleton correction is not available, so a {profile!r} "
    "skeleton cannot be corrected.\n"
    "  Driving a robot from an uncorrected Quest skeleton gives it permuted "
    "joint axes, so this refuses rather than continuing.\n"
    "  The correction (isaacteleop.retargeting_engine.utilities."
    "correct_body_orientations) comes with NVIDIA/IsaacTeleop#921; install an "
    "isaacteleop that includes it.\n"
    "  Or run with --skeleton-source pico if you are on a PICO headset.\n"
    "  (PICO needs no correction and is unaffected by this.)"
)


class IsaacTeleopReader:
    """Background reader using the in-process IsaacCapture / CloudXR DeviceIO session.

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
        skeleton_profile: str | None = None,
    ):
        del max_queue_size

        # The skeleton correction is applied here, by isaacteleop, on the raw
        # ByteDance orientations. Downstream code sees corrected data and must not
        # correct again.
        #
        # None means pass the skeleton through untouched, which is both what a
        # PICO needs and what --skeleton-source auto needs before the first frame
        # resolves it. _apply_resolved_source installs the correction once
        # resolved.
        #
        # The function and its profile live in one attribute, replaced whole,
        # so the reader thread can never pair a function with a profile from a
        # different call to set_skeleton_profile().
        self._correction: tuple[CorrectionFn, str] | None = None

        # Resolution happens here rather than in a consumer, and that placement is
        # load-bearing. The correction has to be settled before anything
        # downstream latches a reference from the skeleton --
        # ThreePointPose.calibrate_now() captures a neck frame once and never
        # recaptures it. Resolving in a consumer that only runs in one stream mode
        # meant calibration could happen first, on uncorrected data, and the
        # global orientation then jumped ~121 deg when the correction arrived.
        #
        # "auto" (or unset) resolves live, every frame, from the OpenXR Extension
        # Method: which full-body vendor tracker (body.pico-xr / body.quest-cloudxr)
        # actually delivered active data this frame is a fact the runtime states,
        # not an inference -- see IsaacTeleopClient._get_tracker_data()'s
        # "full_body_source". This supersedes the old geometry-based detector
        # which has been removed; geometry is repurposed below as a degeneracy
        # guard instead. An explicit --skeleton-source pico|quest is honored
        # regardless of what the extension reports, and never changes for the
        # life of this reader.
        self._manual_skeleton_source: str | None = None
        self._fatal: Exception | None = None
        self.resolved_skeleton_source = None
        self._last_degenerate_log_ns = 0
        self._last_seen_full_body_source: str | None = "<unset>"  # sentinel so the first frame always logs

        # Pico "motion trackers disconnected" guard (skeleton_guards.py):
        # a separate, temporal check from the geometry mismatch guard above --
        # see that guard's comment in _run() for why the two are independent.
        self._trackers_disconnected_detector = TrackersDisconnectedDetector()
        self._last_trackers_disconnected_log_ns = 0
        #: Consecutive frames the geometry guard could not verify; see
        #: _UNVERIFIED_REFUSE_FRAMES.
        self._unverified_streak = 0
        self._last_unverified_log_ns = 0
        #: Frames the guards have refused; see :attr:`refusal_count`.
        self._refusal_count = 0
        if skeleton_profile != "auto":
            self._manual_skeleton_source = skeleton_profile or "pico"
            self._apply_resolved_source(self._manual_skeleton_source)

        if IsaacTeleopClient is None:
            raise RuntimeError(
                "isaacteleop is required for --input-source isaac-teleop but was not "
                "found. Install via install_scripts/install_pico.sh, which runs:\n"
                "  uv pip install -e 'gear_sonic[teleop]' "
                "'isaacteleop[cloudxr]~=1.5.0' --prerelease=allow "
                "--extra-index-url https://pypi.nvidia.com"
            )

        self._client = IsaacTeleopClient(use_adb=use_adb)
        self._period = 1.0 / max(1.0, float(poll_hz))

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._ctrl_lock = threading.Lock()
        # Bumped on every invalidation so a frame already in flight when the
        # cache is cleared is dropped instead of published after the clear.
        self._epoch = 0
        self._latest: BodySample | None = None
        self._latest_controller: ControllerDict | None = None
        self._fps_ema = 0.0
        self._last_stamp_ns: int | None = None
        self._last_new_data_time = time.monotonic()
        self._disconnected = threading.Event()
        self._unrecognised_logged = False


    def set_skeleton_profile(self, profile: str | None) -> None:
        """Install (or clear) the skeleton correction applied to incoming frames.

        Safe to call while the reader thread is running: the function and its
        profile are swapped as one attribute and read once per frame, so a
        frame is corrected entirely before or entirely after the change.
        """
        if not profile:
            self._correction = None
            return
        try:
            from isaacteleop.retargeting_engine.utilities import correct_body_orientations
        # Not only ImportError: a broken install can fail to load its native
        # extension with other errors, and this runs on the reader thread.
        except Exception as exc:
            raise SkeletonCorrectionUnavailable(
                _CORRECTION_UNAVAILABLE_MESSAGE.format(profile=profile)
            ) from exc

        self._correction = (correct_body_orientations, profile)
        logger.info(
            "[IsaacTeleopReader] applying isaacteleop skeleton correction (profile=%s)",
            profile,
        )

    def _apply_resolved_source(self, source: str) -> None:
        """Install (or clear) the skeleton correction for ``source`` and record it.

        Called once for an explicit ``--skeleton-source``, and from ``_run()``
        whenever the OpenXR Extension Method's resolved source changes from the
        previously installed one -- including switching back and forth at
        runtime as the operator changes XR displays mid-session.

        Raises:
            SkeletonCorrectionUnavailable: if ``source`` needs a correction
                (i.e. is not "pico") and isaacteleop does not provide one.
        """
        if source == self.resolved_skeleton_source:
            return
        self.set_skeleton_profile(None if source == "pico" else source)
        self.resolved_skeleton_source = source
        # The streak counts frames against the *previous* source; a swap makes
        # it meaningless, so start it again rather than carrying it across.
        self._unverified_streak = 0
        # Likewise the frozen-pose window. It is reset here, at the switch, and
        # not only where a non-PICO frame is validated: a frame refused before
        # reaching that point would leave an earlier PICO session's motion in
        # the window, and a frozen skeleton after switching back would pass
        # until the old frames aged out. Gaps within one source keep it.
        self._trackers_disconnected_detector.reset()

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

    def _invalidate_latest(self) -> None:
        """Drop the last accepted sample so a refused frame stops the policy.

        The guards log "Refusing to forward this frame to the robot control
        policy", but a consumer reads poses by polling :meth:`get_latest`, which
        hands back whatever was accepted last. Without clearing it, refusing a
        frame only stops *new* poses arriving -- the policy keeps acting on the
        previous one, which is the stale pose it was refused to prevent.
        Consumers already treat None as "nothing to drive from" and wait.
        """
        with self._lock:
            self._latest = None
            self._epoch += 1

    def _refuse(self) -> None:
        """Withhold this frame from the robot: drop the cached pose and count it.

        Clearing the cache stops the pose being read again, but a consumer that
        already holds it -- the deploy side holds the last streamed pose -- keeps
        acting on it. :attr:`refusal_count` is what tells that consumer to move
        the robot somewhere safe.
        """
        with self._lock:
            self._refusal_count += 1
        self._invalidate_latest()

    @property
    def refusal_count(self) -> int:
        """How many frames the guards have refused since this reader started.

        Increases on every refused frame, and never on :meth:`clear_latest`.
        A consumer driving the robot from the stream should compare it with the
        value it last saw and leave that mode when it changes.
        """
        with self._lock:
            return self._refusal_count

    def clear_latest(self) -> None:
        """Public form of :meth:`_invalidate_latest`.

        For callers that invalidate the pose themselves -- after a calibration,
        say, where a sample captured against the old reference frame is no
        longer meaningful even though nothing is wrong with the stream.

        Clearing the cache is not enough on its own: the reader may already be
        midway through a frame it acquired *before* this call, and publishing
        that frame afterwards would restore exactly the pre-calibration pose
        this was meant to discard. The epoch bump closes that window -- any
        frame whose read began before this call is dropped at its publish site
        instead of being stored.
        """
        self._invalidate_latest()

    def get_latest(self) -> BodySample | None:
        # Raised here, not in the reader thread, so it reaches the caller rather
        # than dying silently in a background thread.
        if self._fatal is not None:
            raise self._fatal
        with self._lock:
            return self._latest

    def get_controller_data(self) -> ControllerDict | None:
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
        # Any failure the loop does not handle itself is re-raised from
        # get_latest(), so it stops the caller instead of silently leaving it
        # waiting on a reader thread that has died.
        try:
            self._run_loop()
        except Exception as exc:
            logger.exception("[IsaacTeleopReader] reader thread failed")
            self._fatal = exc
            self._stop.set()

    def _run_loop(self) -> None:
        last_report = time.time()
        while not self._stop.is_set():
            # Read before the frame is acquired, compared before it is
            # published: a clear_latest() landing anywhere in between bumps the
            # epoch, and this frame -- which describes the pose from before the
            # clear -- is dropped rather than refilling the cache the caller
            # just emptied.
            with self._lock:
                frame_epoch = self._epoch
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

            # OpenXR Extension Method: which full-body vendor tracker actually
            # delivered active data this frame, straight from the runtime -- see
            # IsaacTeleopClient._get_tracker_data(). Re-evaluated every frame (not
            # cached) so a headset swap mid-session is picked up on the next poll.
            # An explicit --skeleton-source never gets here (_manual_skeleton_source
            # is set) and is honored regardless of what the extension reports.
            if self._manual_skeleton_source is None:
                full_body_source = raw.full_body_source
                # Diagnostic: log every observed value, not just changes we act on,
                # so a stuck resolution shows up as a repeating line here instead of
                # only inferred from the degeneracy guard downstream.
                if full_body_source != self._last_seen_full_body_source:
                    logger.info(
                        "[IsaacTeleopReader] full_body_source: %r -> %r (resolved was %r)",
                        self._last_seen_full_body_source,
                        full_body_source,
                        self.resolved_skeleton_source,
                    )
                    self._last_seen_full_body_source = full_body_source
                if full_body_source is not None:
                    try:
                        self._apply_resolved_source(full_body_source)
                    except SkeletonCorrectionUnavailable as exc:
                        # Refuse rather than stream an uncorrected non-PICO
                        # skeleton. Logging and continuing was tried first and is
                        # not safe: the robot is driven with permuted joint axes
                        # and falls, which is precisely what this detects.
                        # Re-raised from get_latest() so it surfaces on the main
                        # thread and stops the process, matching what an explicit
                        # --skeleton-source quest does at startup.
                        logger.error("[IsaacTeleopReader] %s", exc)
                        self._fatal = exc
                        self._stop.set()
                        return

            body_poses = _body_data_to_24x7(raw.full_body)

            # NaN and inf fail every threshold comparison in the guards below,
            # so without this they would read as "not degenerate".
            if body_poses is not None and not np.isfinite(body_poses).all():
                now_ns = time.monotonic_ns()
                if now_ns - self._last_degenerate_log_ns > _DEGENERATE_LOG_INTERVAL_NS:
                    self._last_degenerate_log_ns = now_ns
                    logger.error(
                        "[IsaacTeleopReader] DEGENERATE SKELETON: non-finite joint values. "
                        "Refusing to forward this frame to the robot control policy."
                    )
                self._refuse()
                time.sleep(self._period)
                continue

            # A joint the runtime did not fill (is_valid False, or a missing
            # field) is left as a zero row by _body_data_to_24x7. The geometry
            # guard can still identify the vendor from the remaining bones, but
            # the pose consumer needs every joint: a zero quaternion is not a
            # rotation, and it raises there. Real frames are unit to within
            # 1e-6, so 0.1 only ever catches a row that is not a rotation.
            missing = (
                np.flatnonzero(
                    np.abs(np.linalg.norm(body_poses[:, 3:], axis=1) - 1.0) > _QUAT_NORM_TOLERANCE
                )
                if body_poses is not None
                else ()
            )
            if len(missing):
                now_ns = time.monotonic_ns()
                if now_ns - self._last_degenerate_log_ns > _DEGENERATE_LOG_INTERVAL_NS:
                    self._last_degenerate_log_ns = now_ns
                    logger.error(
                        "[IsaacTeleopReader] DEGENERATE SKELETON: joints %s have no valid "
                        "orientation. Refusing to forward this frame to the robot control "
                        "policy.",
                        missing.tolist(),
                    )
                self._refuse()
                time.sleep(self._period)
                continue

            # Geometry degeneracy guard. This used to be the primary detection
            # mechanism; that driver has been removed, but its comparison is kept;
            # now that the OpenXR Extension Method says definitively which vendor
            # produced this frame, geometry is repurposed as a safety check: does
            # this skeleton's orientation convention actually match the resolved
            # source -- the extension it arrived on in auto mode, or the source an
            # explicit --skeleton-source forced? A mismatch means something is
            # wrong -- a disconnected/misbehaving tracker, a skeleton sent over the
            # wrong extension, or an operator's manual override not matching the
            # headset that's actually connected -- and driving a robot from it
            # risks permuted joint axes, so this refuses the frame rather than risk
            # it. Applies in both manual and auto mode: a forced --skeleton-source
            # is honored for *which correction to apply*, never for skipping this
            # safety check.
            #
            if body_poses is not None and self.resolved_skeleton_source in ("pico", "quest"):
                # A fault in the matcher must not take the reader down with it.
                # Everything from here to the end of the loop body runs outside
                # the try that wraps _get_tracker_data(), so an exception would
                # otherwise kill this thread silently and stop all teleop.
                try:
                    match = match_orientation_convention(
                        body_poses[:, :3], body_poses[:, 3:]
                    )
                except Exception:
                    logger.exception(
                        "[IsaacTeleopReader] skeleton geometry check failed; "
                        "counting this frame as unverified"
                    )
                    match = None

                if match is not None and match.outcome is MatchOutcome.DECIDED:
                    if match.source != self.resolved_skeleton_source:
                        now_ns = time.monotonic_ns()
                        if now_ns - self._last_degenerate_log_ns > _DEGENERATE_LOG_INTERVAL_NS:
                            self._last_degenerate_log_ns = now_ns
                            logger.error(
                                "[IsaacTeleopReader] DEGENERATE SKELETON: resolved source is "
                                "%r but this frame's orientations read as %r (scores %s). "
                                "Refusing to forward this frame to the robot control policy.",
                                self.resolved_skeleton_source,
                                match.source,
                                match.scores,
                            )
                        self._refuse()
                        time.sleep(self._period)
                        continue
                    # Verified against the resolved source: clear the streak.
                    self._unverified_streak = 0
                elif match is not None and match.outcome is MatchOutcome.NO_MATCH:
                    # Neither reference fits, so the resolved source does not
                    # either. That is a positive finding, like a DECIDED
                    # mismatch, not a failure to tell: refuse now rather than
                    # give it the grace period meant for AMBIGUOUS frames.
                    now_ns = time.monotonic_ns()
                    if now_ns - self._last_degenerate_log_ns > _DEGENERATE_LOG_INTERVAL_NS:
                        self._last_degenerate_log_ns = now_ns
                        logger.error(
                            "[IsaacTeleopReader] DEGENERATE SKELETON: this frame's orientations "
                            "match no known convention (scores %s), so not resolved source %r. "
                            "Refusing to forward this frame to the robot control policy.",
                            match.scores,
                            self.resolved_skeleton_source,
                        )
                    self._refuse()
                    time.sleep(self._period)
                    continue
                else:
                    # Every other way of not being DECIDED lands here:
                    # AMBIGUOUS, UNUSABLE, and the matcher raising. They differ in cause but
                    # not in consequence -- this frame was not verified against
                    # the resolved source -- so they share one streak.
                    #
                    # The raising case used to fall through both branches and be
                    # forwarded, without even incrementing the streak. A matcher
                    # that threw on every frame therefore streamed every frame to
                    # the policy unverified, forever, which is the one failure
                    # mode this guard exists to prevent. Catching the exception
                    # keeps the reader thread alive; it must not also buy the
                    # frame a free pass.
                    self._unverified_streak += 1
                    if self._unverified_streak >= _UNVERIFIED_REFUSE_FRAMES:
                        now_ns = time.monotonic_ns()
                        if now_ns - self._last_unverified_log_ns > _DEGENERATE_LOG_INTERVAL_NS:
                            self._last_unverified_log_ns = now_ns
                            detail = (
                                "the geometry check raised"
                                if match is None
                                else f"{match.outcome.value}, scores {match.scores}, "
                                f"margin {match.margin_deg:.1f} deg"
                            )
                            logger.error(
                                "[IsaacTeleopReader] DEGENERATE SKELETON: %d consecutive "
                                "frames could not be verified against resolved source %r "
                                "(%s). Refusing to forward them to the robot control policy.",
                                self._unverified_streak,
                                self.resolved_skeleton_source,
                                detail,
                            )
                        self._refuse()
                        time.sleep(self._period)
                        continue

            # Pico "motion trackers disconnected" guard. Separate from the geometry
            # mismatch guard above: that one is a per-frame check of orientation
            # convention (does this frame look like pico or quest), this one is a
            # temporal check (has the torso been frozen to a constant pose for the
            # last several frames) -- see skeleton_guards.py for how the
            # signature was characterized and why it's BD-only for now. Only makes
            # sense while resolved to pico, so the window is reset whenever we're
            # not, to avoid a stale window from a previous pico session bleeding
            # into a fresh one.
            #
            # It is deliberately *not* reset on a frame with no body data. A
            # frozen skeleton interleaved with dropped frames would then never
            # fill the window, and so never be refused.
            if self.resolved_skeleton_source != "pico":
                self._trackers_disconnected_detector.reset()
            elif body_poses is not None:
                detector = self._trackers_disconnected_detector
                try:
                    frozen = detector.push(body_poses[:, :3], body_poses[:, 3:])
                except Exception:
                    logger.exception(
                        "[IsaacTeleopReader] frozen-skeleton check failed; refusing this frame"
                    )
                    frozen = True
                if not detector.window_full:
                    # Until the window fills, a frozen skeleton is
                    # indistinguishable from a live one, so the first frames of
                    # a PICO session wait (about half a second) rather than
                    # reach the robot unchecked.
                    self._invalidate_latest()
                    time.sleep(self._period)
                    continue
                if frozen:
                    now_ns = time.monotonic_ns()
                    if now_ns - self._last_trackers_disconnected_log_ns > _DEGENERATE_LOG_INTERVAL_NS:
                        self._last_trackers_disconnected_log_ns = now_ns
                        logger.error(
                            "[IsaacTeleopReader] DEGENERATE SKELETON: Pico motion trackers "
                            "appear disconnected (torso/pelvis frozen to a constant pose). "
                            "Refusing to forward this frame to the robot control policy.",
                        )
                    self._refuse()
                    time.sleep(self._period)
                    continue

            correction = self._correction
            if body_poses is not None and correction is not None:
                correct_fn, profile = correction
                # A frame that cannot be corrected must not reach the policy
                # uncorrected, and the failure must not end this thread.
                try:
                    body_poses[:, 3:] = correct_fn(body_poses[:, 3:], profile).astype(
                        body_poses.dtype
                    )
                except Exception:
                    now_ns = time.monotonic_ns()
                    if now_ns - self._last_degenerate_log_ns > _DEGENERATE_LOG_INTERVAL_NS:
                        self._last_degenerate_log_ns = now_ns
                        logger.exception(
                            "[IsaacTeleopReader] skeleton correction (profile=%s) failed; "
                            "refusing to forward this frame to the robot control policy",
                            profile,
                        )
                    self._refuse()
                    time.sleep(self._period)
                    continue
            if body_poses is None:
                if not self._unrecognised_logged and not _attr_or_item(
                    raw.full_body, "joint_positions"
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

            sample: BodySample = {
                "body_poses_np": body_poses,
                "timestamp_realtime": time.time(),
                "timestamp_monotonic": time.monotonic(),
                "timestamp_ns": stamp_ns,
                "dt": device_dt,
                "fps": self._fps_ema,
                "epoch": frame_epoch,
            }
            with self._lock:
                if self._epoch == frame_epoch:
                    self._latest = sample
                else:
                    # Invalidated while this frame was being read or converted.
                    # Dropping it leaves the cache empty, as the caller left it.
                    logger.debug("[IsaacTeleopReader] Dropped a frame that predates a cache clear")

            now = time.time()
            if now - last_report >= 5.0:
                logger.info(
                    "[IsaacTeleopReader] dt: %.2f ms, fps: %.2f",
                    device_dt * 1000.0,
                    self._fps_ema,
                )
                last_report = now

            time.sleep(self._period)


