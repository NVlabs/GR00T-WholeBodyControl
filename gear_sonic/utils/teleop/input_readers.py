"""Input source readers for body tracking data.

PicoReader         -- pulls data from XRoboToolkit SDK (Pico headset).
IsaacTeleopReader  -- in-process IsaacTeleop / CloudXR DeviceIO session.
"""

import logging
import threading
import time
from typing import Any, TYPE_CHECKING

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
    from gear_sonic.utils.teleop.isaac_teleop_client import TrackerSnapshot

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


# Number of joints in the IsaacTeleop FullBodyPoseT (XR_BD_body_tracking).
# Mirrors core.BodyJoint.NUM_JOINTS in IsaacTeleop's schema bindings.
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
    """Convert ``FullBodyTracker.get_body_pose()`` to a (24, 7) array.

    Returns ``None`` while no joint is valid (typical when the headset isn't
    connected yet — every ``BodyJointPose.is_valid`` is False, the streamer
    keeps polling and the C++ deploy doesn't see fake zero pose).

    Two accepted schemas:

    Schema A — IsaacTeleop ``FullBodyPoseT`` (DeviceIO direct).
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


def _build_controller_dict(raw: "TrackerSnapshot | None") -> dict[str, Any] | None:
    """Convert ``IsaacTeleopClient._get_tracker_data()`` into the controller dict
    schema that ``pico_manager_thread_server`` consumes (left/right trigger,
    squeeze, thumbstick, click, primary/secondary click)."""
    if raw is None:
        return None

    left = _controller_inputs_to_dict_side(raw.left_controller)
    right = _controller_inputs_to_dict_side(raw.right_controller)
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


class SkeletonCorrectionUnavailable(RuntimeError):
    """The isaacteleop skeleton correction is needed but not importable."""


# Deliberately actionable rather than a bare ImportError. The correction ships in
# isaacteleop 1.5+, while this repo pins 1.3.131, so on any machine that has not
# been bridged the import is the first thing a Quest operator hits.
_BRIDGE_MESSAGE = (
    "the isaacteleop skeleton correction is not available, so a {profile!r} "
    "skeleton cannot be corrected.\n"
    "  Driving a robot from an uncorrected Quest skeleton gives it permuted "
    "joint axes, so this refuses rather than continuing.\n"
    "  Fix it with:\n"
    "      git -C ~/IsaacTeleop checkout jsepulveda/quest_remapping\n"
    "      ./install_scripts/install_isaacteleop_skeleton_correction.sh\n"
    "  Or run with --skeleton-source pico if you are on a PICO headset.\n"
    "  (PICO needs no correction and is unaffected by this.)"
)


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
        self._skeleton_profile = None
        self._correct_fn = None

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
        if skeleton_profile != "auto":
            self._manual_skeleton_source = skeleton_profile or "pico"
            self._apply_resolved_source(self._manual_skeleton_source)

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


    def set_skeleton_profile(self, profile: str | None) -> None:
        """Install (or clear) the skeleton correction applied to incoming frames.

        Safe to call while the reader thread is running: the correction is a
        single attribute read per frame, so the worst case is one frame either
        side of the change.
        """
        if not profile:
            self._skeleton_profile, self._correct_fn = None, None
            return
        try:
            from isaacteleop.retargeting_engine.utilities import correct_body_orientations
        except ImportError as exc:
            raise SkeletonCorrectionUnavailable(_BRIDGE_MESSAGE.format(profile=profile)) from exc

        self._skeleton_profile = profile
        self._correct_fn = correct_body_orientations
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

    def clear_latest(self) -> None:
        """Public form of :meth:`_invalidate_latest`.

        For callers that invalidate the pose themselves -- after a calibration,
        say, where a sample captured against the old reference frame is no
        longer meaningful even though nothing is wrong with the stream.
        """
        self._invalidate_latest()

    def get_latest(self) -> dict[str, Any] | None:
        # Raised here, not in the reader thread, so it reaches the caller rather
        # than dying silently in a background thread.
        if self._fatal is not None:
            raise self._fatal
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
                        self._invalidate_latest()
                        time.sleep(self._period)
                        continue
                    # Verified against the resolved source: clear the streak.
                    self._unverified_streak = 0
                else:
                    # Every way of not being DECIDED lands here: AMBIGUOUS,
                    # UNUSABLE, and the matcher raising. They differ in cause but
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
                        self._invalidate_latest()
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
            if self.resolved_skeleton_source == "pico" and body_poses is not None:
                frozen = self._trackers_disconnected_detector.push(body_poses[:, :3], body_poses[:, 3:])
                if frozen:
                    now_ns = time.monotonic_ns()
                    if now_ns - self._last_trackers_disconnected_log_ns > _DEGENERATE_LOG_INTERVAL_NS:
                        self._last_trackers_disconnected_log_ns = now_ns
                        logger.error(
                            "[IsaacTeleopReader] DEGENERATE SKELETON: Pico motion trackers "
                            "appear disconnected (torso/pelvis frozen to a constant pose). "
                            "Refusing to forward this frame to the robot control policy.",
                        )
                    self._invalidate_latest()
                    time.sleep(self._period)
                    continue
            else:
                self._trackers_disconnected_detector.reset()

            correct_fn = self._correct_fn
            if body_poses is not None and correct_fn is not None:
                body_poses[:, 3:] = correct_fn(
                    body_poses[:, 3:], self._skeleton_profile
                ).astype(body_poses.dtype)
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


