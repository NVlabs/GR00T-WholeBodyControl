"""Input source readers for body tracking data.

PicoReader         -- pulls data from XRoboToolkit SDK (Pico headset).
IsaacTeleopReader  -- in-process IsaacTeleop / CloudXR DeviceIO session.
"""

from collections.abc import Mapping
import logging
import math
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MAX_BODY_POSITION_M = 10.0
MAX_BODY_SPAN_M = 3.0
MAX_BODY_FRAME_DISPLACEMENT_M = 0.35
MAX_BODY_JOINT_SPEED_M_S = 12.0
BODY_FRAME_POSITION_SLACK_M = 0.03
MAX_BODY_FRAME_ANGULAR_DISPLACEMENT_RAD = math.radians(75.0)
MAX_BODY_JOINT_ANGULAR_SPEED_RAD_S = 25.0
BODY_FRAME_ANGULAR_SLACK_RAD = math.radians(10.0)
XRT_ATOMIC_SNAPSHOT_CONTRACT = "xrt_xr24_plus_two_ankles_atomic_v1"
XRT_DERIVATIVE_LAYOUT_CONTRACT = "linear_xyz_then_angular_xyz_v1"
XRT_SOURCE_COHERENCE_CONTRACT = "same_packet_xr24_plus_raw_ankles_v1"
XRT_BODY_SNAPSHOT_CONTRACT = (
    "xrt_xr24_body_tracker_fused_ankles_atomic_v1"
)
XRT_BODY_SOURCE_COHERENCE_CONTRACT = (
    "same_packet_xr24_body_tracking_v1"
)


def validate_body_poses(body_poses: Any) -> str:
    """Return an empty string for a plausible 24-joint pose array, else a reason."""
    try:
        poses = np.asarray(body_poses)
    except (TypeError, ValueError):
        return "body pose array is not numeric"
    if poses.ndim != 2 or poses.shape[0] < 24 or poses.shape[1] < 7:
        return f"body pose shape is {poses.shape}, expected at least (24, 7)"
    try:
        if not bool(np.isfinite(poses).all()):
            return "body pose contains NaN or Inf"
        positions = poses[:24, :3].astype(np.float64)
        quaternion_norms = np.linalg.norm(
            poses[:24, 3:7].astype(np.float64),
            axis=1,
        )
    except (TypeError, ValueError):
        return "body pose array is not numeric"
    if np.any(np.abs(positions) > MAX_BODY_POSITION_M):
        return "body pose position is outside safe bounds"
    if np.any(np.ptp(positions, axis=0) > MAX_BODY_SPAN_M):
        return "body pose joint span is anatomically implausible"
    if np.any(quaternion_norms < 0.5) or np.any(quaternion_norms > 1.5):
        return "body pose contains invalid joint quaternion"
    return ""


try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

try:
    from gear_sonic.utils.teleop.isaac_teleop_client import IsaacTeleopClient
except ImportError:
    IsaacTeleopClient = None


def get_fresh_body_sample(
    reader: Any,
    max_age_s: float,
    *,
    monotonic_now: float | None = None,
    require_controllers: bool = False,
    require_motion_trackers: bool = False,
    require_body_tracker_health: bool = False,
    require_controller_tracking_health: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """Return a validated body sample, or ``(None, reason)``.

    Manager control must not engage from a missing, stale, malformed, or
    non-finite tracking frame. Readers used for robot control must derive
    freshness from a source frame timestamp, rather than from host polling.
    """
    if not math.isfinite(max_age_s) or max_age_s <= 0.0:
        raise ValueError("max_age_s must be a positive finite value")

    try:
        disconnected = bool(getattr(reader, "disconnected", False))
    except Exception as exc:
        return None, f"reader connection state failed: {exc}"
    if disconnected:
        return None, "reader reports disconnected"
    try:
        trusted_source_timestamps = getattr(
            reader,
            "supports_trusted_source_timestamps",
            False,
        )
    except Exception as exc:
        return None, f"reader timestamp capability check failed: {exc}"
    if trusted_source_timestamps is not True:
        return None, "reader does not expose trusted source frame timestamps"

    try:
        sample = reader.get_latest()
    except Exception as exc:
        return None, f"reader failed to return tracking: {exc}"
    if sample is None:
        try:
            fault_reason = str(getattr(reader, "tracking_fault_reason", "")).strip()
        except Exception:
            fault_reason = ""
        if fault_reason:
            return None, fault_reason
        return None, "no body-tracking sample"
    if not isinstance(sample, dict):
        return None, "body-tracking sample is not a mapping"

    try:
        sample_time = float(sample["timestamp_monotonic"])
    except (KeyError, TypeError, ValueError):
        return None, "sample has no valid monotonic timestamp"
    if not math.isfinite(sample_time):
        return None, "sample monotonic timestamp is not finite"

    try:
        now = time.monotonic() if monotonic_now is None else float(monotonic_now)
    except (TypeError, ValueError):
        return None, "manager monotonic time is invalid"
    if not math.isfinite(now):
        return None, "manager monotonic time is not finite"
    age_s = now - sample_time
    if age_s < -0.05:
        return None, f"sample timestamp is {abs(age_s):.3f}s in the future"
    if age_s > max_age_s:
        return None, f"sample is stale ({age_s:.3f}s > {max_age_s:.3f}s)"

    try:
        body_poses = np.asarray(sample["body_poses_np"])
    except (KeyError, TypeError, ValueError):
        return None, "sample has no valid body pose array"
    if body_reason := validate_body_poses(body_poses):
        return None, body_reason
    try:
        source_dt_s = float(sample["dt"])
    except (KeyError, TypeError, ValueError):
        return None, "sample has no valid body source frame interval"
    if not math.isfinite(source_dt_s) or source_dt_s <= 0.0:
        return None, "body source frame interval is invalid"
    try:
        frame_displacement_m = float(sample["body_frame_displacement_m"])
    except (KeyError, TypeError, ValueError):
        return None, "sample has no valid body frame displacement"
    if not math.isfinite(frame_displacement_m) or frame_displacement_m < 0.0:
        return None, "body frame displacement is invalid"
    allowed_displacement_m = min(
        MAX_BODY_FRAME_DISPLACEMENT_M,
        BODY_FRAME_POSITION_SLACK_M + MAX_BODY_JOINT_SPEED_M_S * source_dt_s,
    )
    if frame_displacement_m > allowed_displacement_m:
        return None, (
            "body tracking jumped "
            f"{frame_displacement_m:.3f}m between frames "
            f"(limit {allowed_displacement_m:.3f}m for dt={source_dt_s:.4f}s)"
        )
    try:
        frame_angular_displacement_rad = float(sample["body_frame_angular_displacement_rad"])
    except (KeyError, TypeError, ValueError):
        return None, "sample has no valid body frame angular displacement"
    if not math.isfinite(frame_angular_displacement_rad) or frame_angular_displacement_rad < 0.0:
        return None, "body frame angular displacement is invalid"
    allowed_angular_displacement_rad = min(
        MAX_BODY_FRAME_ANGULAR_DISPLACEMENT_RAD,
        BODY_FRAME_ANGULAR_SLACK_RAD + MAX_BODY_JOINT_ANGULAR_SPEED_RAD_S * source_dt_s,
    )
    if frame_angular_displacement_rad > allowed_angular_displacement_rad:
        return None, (
            "body tracking rotated "
            f"{math.degrees(frame_angular_displacement_rad):.1f} degrees "
            "between frames "
            f"(limit {math.degrees(allowed_angular_displacement_rad):.1f} "
            f"degrees for dt={source_dt_s:.4f}s)"
        )

    if require_body_tracker_health:
        if sample.get("body_tracker_health_supported") is not True:
            return None, "PICO client does not expose hardened body-tracker health telemetry"
        if sample.get("body_tracker_health_available") is not True:
            return None, "PICO body-tracker health telemetry is unavailable"
        try:
            health_schema = int(sample["body_tracker_health_schema_version"])
        except (KeyError, TypeError, ValueError):
            return None, "PICO body-tracker health schema is missing or malformed"
        if health_schema != 1:
            return None, f"unsupported PICO body-tracker health schema {health_schema}"
        if sample.get("body_tracker_health_calibrated") is not True:
            return None, "PICO body trackers are not calibrated"
        expected_zero_fields = (
            ("body_tracker_health_calibration_result", "calibration query"),
            ("body_tracker_health_connect_state_result", "tracker connection query"),
            ("body_tracker_health_body_state_result", "body tracking state query"),
            ("body_tracker_health_tracking_state_code", "motion tracking state"),
            ("body_tracker_health_body_data_result", "body tracking data query"),
        )
        for key, label in expected_zero_fields:
            try:
                value = int(sample[key])
            except (KeyError, TypeError, ValueError):
                return None, f"PICO {label} result is missing or malformed"
            if value != 0:
                return None, f"PICO {label} failed with code {value}"
        try:
            tracking_mode = int(sample["body_tracker_health_tracking_mode"])
        except (KeyError, TypeError, ValueError):
            return None, "PICO motion-tracker mode is missing or malformed"
        if tracking_mode != 0:
            return None, f"PICO motion-tracker mode is {tracking_mode}, expected BodyTracking (0)"
        if sample.get("body_tracker_health_is_tracking") is not True:
            return None, "PICO body tracking reports tracking lost"
        try:
            body_state_code = int(sample["body_tracker_health_body_state_code"])
        except (KeyError, TypeError, ValueError):
            return None, "PICO body-tracking status is missing or malformed"
        if body_state_code != 1:
            try:
                body_error = int(sample["body_tracker_health_body_error_code"])
            except (KeyError, TypeError, ValueError):
                body_error = -1
            return None, (
                f"PICO body tracking is not BT_VALID "
                f"(state={body_state_code}, error={body_error})"
            )
        count_fields = (
            ("body_tracker_health_tracker_count", "connected tracker count", 2),
            ("body_tracker_health_unique_tracker_count", "unique tracker count", 2),
            ("body_tracker_health_connected_band_count", "body-tracking band count", 2),
            ("body_tracker_health_body_role_count", "body role count", 24),
        )
        for key, label, minimum in count_fields:
            try:
                value = int(sample[key])
            except (KeyError, TypeError, ValueError):
                return None, f"PICO {label} is missing or malformed"
            if value < minimum:
                return None, f"PICO {label} is {value}; at least {minimum} required"
        try:
            sequence = int(sample["body_tracker_health_sample_sequence"])
        except (KeyError, TypeError, ValueError):
            return None, "PICO body-tracker health sequence is missing or malformed"
        if sequence <= 0:
            return None, "PICO body-tracker health sequence is not positive"
        if sample.get("body_tracker_health_valid") is not True:
            return None, "PICO body-tracker health gate reports invalid"
        try:
            health_time = float(sample["body_tracker_health_timestamp_monotonic"])
        except (KeyError, TypeError, ValueError):
            return None, "PICO body-tracker health has no fresh source sequence"
        if not math.isfinite(health_time):
            return None, "PICO body-tracker health timestamp is not finite"
        health_age_s = now - health_time
        if health_age_s < -0.05:
            return None, "PICO body-tracker health timestamp is in the future"
        if health_age_s > max_age_s:
            return None, f"PICO body-tracker health is stale ({health_age_s:.3f}s)"

    if require_motion_trackers:
        tracker_count_raw = sample.get("motion_tracker_count")
        if isinstance(tracker_count_raw, bool) or not isinstance(
            tracker_count_raw,
            (int, np.integer),
        ):
            return None, "motion tracker count is missing or malformed"
        tracker_count = int(tracker_count_raw)
        if tracker_count < 2:
            return None, f"only {tracker_count} motion tracker(s) are available; two are required"

        tracker_serials_raw = sample.get("motion_tracker_serial_numbers")
        if not isinstance(tracker_serials_raw, (list, tuple)):
            return None, "motion tracker serial numbers are missing or malformed"
        if len(tracker_serials_raw) < tracker_count:
            return None, "motion tracker serial list is shorter than the reported count"
        tracker_serials = [str(serial).strip() for serial in tracker_serials_raw[:tracker_count]]
        unique_serials = {serial for serial in tracker_serials if serial}
        if len(unique_serials) < 2:
            return None, "two unique non-empty motion tracker serial numbers are required"

        try:
            tracker_time = float(sample["motion_tracker_timestamp_monotonic"])
        except (KeyError, TypeError, ValueError):
            return None, "motion trackers have no fresh tracking timestamp"
        if not math.isfinite(tracker_time):
            return None, "motion tracker timestamp is not finite"
        tracker_age_s = now - tracker_time
        if tracker_age_s < -0.05:
            return None, "motion tracker timestamp is in the future"
        if tracker_age_s > max_age_s:
            return None, f"motion trackers are stale ({tracker_age_s:.3f}s)"

    if require_controllers:
        for side in ("left", "right"):
            key = f"{side}_controller_timestamp_monotonic"
            try:
                controller_time = float(sample[key])
            except (KeyError, TypeError, ValueError):
                return None, f"{side} controller has no fresh tracking timestamp"
            if not math.isfinite(controller_time):
                return None, f"{side} controller tracking timestamp is not finite"
            controller_age_s = now - controller_time
            if controller_age_s < -0.05:
                return None, f"{side} controller timestamp is in the future"
            if controller_age_s > max_age_s:
                return None, f"{side} controller is stale ({controller_age_s:.3f}s)"

        controller_data = sample.get("controller_data")
        if not isinstance(controller_data, Mapping):
            return None, "controller sample is missing"
        for side in ("left", "right"):
            try:
                pose = np.asarray(controller_data[f"{side}_pose"], dtype=np.float64)
                thumbstick = np.asarray(
                    controller_data[f"{side}_thumbstick"],
                    dtype=np.float64,
                )
                analog = np.asarray(
                    [
                        controller_data[f"{side}_trigger_value"],
                        controller_data[f"{side}_squeeze_value"],
                    ],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError):
                return None, f"{side} controller sample is malformed"
            if pose.shape != (7,) or thumbstick.shape != (2,):
                return None, f"{side} controller pose/axis shape is invalid"
            if not np.isfinite(pose).all() or not np.isfinite(thumbstick).all():
                return None, f"{side} controller pose/axis contains NaN or Inf"
            if not np.isfinite(analog).all():
                return None, f"{side} controller trigger/grip contains NaN or Inf"
            if np.any(np.abs(pose[:3]) > 10.0):
                return None, f"{side} controller position is outside safe bounds"
            if np.any(np.abs(thumbstick) > 1.25):
                return None, f"{side} controller axis is outside expected range"
            if np.any(analog < -0.05) or np.any(analog > 1.05):
                return None, f"{side} controller trigger/grip is outside expected range"
            quaternion_norm = float(np.linalg.norm(pose[3:7]))
            if quaternion_norm < 0.5 or quaternion_norm > 1.5:
                return None, f"{side} controller pose quaternion is invalid"
            for suffix in (
                "thumbstick_click",
                "primary_click",
                "secondary_click",
            ):
                key = f"{side}_{suffix}"
                if key not in controller_data:
                    return None, f"{side} controller is missing required {suffix}"
                try:
                    button_value = float(controller_data[key])
                except (TypeError, ValueError):
                    return None, f"{side} controller button state is malformed"
                if not math.isfinite(button_value) or not -0.05 <= button_value <= 1.05:
                    return None, f"{side} controller button state is outside expected range"
            menu_key = f"{side}_menu_button"
            if menu_key in controller_data:
                try:
                    menu_value = float(controller_data[menu_key])
                except (TypeError, ValueError):
                    return None, f"{side} controller menu button state is malformed"
                if not math.isfinite(menu_value) or not -0.05 <= menu_value <= 1.05:
                    return None, f"{side} controller menu button state is outside expected range"

    if require_controller_tracking_health:
        if sample.get("controller_tracking_health_supported") is not True:
            return None, "PICO client does not expose hardened controller-tracking health telemetry"
        if sample.get("controller_tracking_health_available") is not True:
            return None, "PICO controller-tracking health telemetry is unavailable"
        try:
            health_schema = int(sample["controller_tracking_health_schema_version"])
            sequence = int(sample["controller_tracking_health_sample_sequence"])
            health_time = float(sample["controller_tracking_health_timestamp_monotonic"])
            controller_packet_timestamp_ns = int(
                sample["controller_tracking_health_source_timestamp_ns"]
            )
            body_packet_timestamp_ns = int(
                sample["body_tracker_health_source_timestamp_ns"]
            )
        except (KeyError, TypeError, ValueError):
            return None, "PICO controller-tracking health telemetry is malformed"
        if health_schema != 1:
            return None, f"unsupported PICO controller-tracking health schema {health_schema}"
        if sequence <= 0:
            return None, "PICO controller-tracking health sequence is not positive"
        if not math.isfinite(health_time):
            return None, "PICO controller-tracking health timestamp is not finite"
        health_age_s = now - health_time
        if health_age_s < -0.05:
            return None, "PICO controller-tracking health timestamp is in the future"
        if health_age_s > max_age_s:
            return None, f"PICO controller-tracking health is stale ({health_age_s:.3f}s)"
        if controller_packet_timestamp_ns <= 0:
            return None, "PICO controller-tracking packet timestamp is not positive"
        if controller_packet_timestamp_ns != body_packet_timestamp_ns:
            return None, "PICO body and controller health did not come from the same packet"
        for side in ("left", "right"):
            if sample.get(f"controller_tracking_health_{side}_device_valid") is not True:
                return None, f"PICO {side} controller device is invalid"
            if sample.get(f"controller_tracking_health_{side}_is_tracked_available") is not True:
                return None, f"PICO {side} controller tracked-state query is unavailable"
            if sample.get(f"controller_tracking_health_{side}_is_tracked") is not True:
                return None, f"PICO {side} controller reports tracking lost"
            if sample.get(f"controller_tracking_health_{side}_tracking_state_available") is not True:
                return None, f"PICO {side} controller tracking-state query is unavailable"
            try:
                tracking_state = int(
                    sample[f"controller_tracking_health_{side}_tracking_state"]
                )
            except (KeyError, TypeError, ValueError):
                return None, f"PICO {side} controller tracking state is malformed"
            if tracking_state & 3 != 3:
                return None, (
                    f"PICO {side} controller lacks position+rotation tracking "
                    f"(state={tracking_state})"
                )
            if sample.get(f"controller_tracking_health_{side}_valid") is not True:
                return None, f"PICO {side} controller health gate reports invalid"
        if sample.get("controller_tracking_health_valid") is not True:
            return None, "PICO controller-tracking health gate reports invalid"

    return sample, ""


class PicoReader:
    """Background reader that pulls Pico/XRT data and computes dt/FPS."""

    STALE_TIMEOUT = 5.0
    supports_trusted_source_timestamps = True

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
        self._controller_last_stamp_ns = {"left": None, "right": None}
        self._controller_last_new_data_time = {"left": None, "right": None}
        self._controller_health_last_sequence: int | None = None
        self._controller_health_last_new_data_time: float | None = None
        self._motion_tracker_last_stamp_ns = None
        self._motion_tracker_last_new_data_time = None
        self._motion_tracker_count = 0
        self._motion_tracker_serials: tuple[str, ...] = ()
        self._body_tracker_health_last_sequence: int | None = None
        self._body_tracker_health_last_new_data_time: float | None = None
        self._tracking_fault_reason = ""
        self._last_body_positions: np.ndarray | None = None
        self._last_body_quaternions: np.ndarray | None = None

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
    def tracking_fault_reason(self) -> str:
        with self._lock:
            return self._tracking_fault_reason

    def get_controller_data(self) -> dict[str, Any] | None:
        with self._lock:
            sample = self._latest
            return sample.get("controller_data") if sample is not None else None

    @property
    def disconnected(self) -> bool:
        return self._disconnected.is_set()

    def clear_disconnect(self):
        self._disconnected.clear()
        self._last_new_data_time = time.monotonic()
        self._last_stamp_ns = None
        self._fps_ema = 0.0
        self._controller_last_stamp_ns = {"left": None, "right": None}
        self._controller_last_new_data_time = {"left": None, "right": None}
        self._controller_health_last_sequence = None
        self._controller_health_last_new_data_time = None
        self._motion_tracker_last_stamp_ns = None
        self._motion_tracker_last_new_data_time = None
        self._motion_tracker_count = 0
        self._motion_tracker_serials = ()
        self._body_tracker_health_last_sequence = None
        self._body_tracker_health_last_new_data_time = None
        with self._lock:
            self._tracking_fault_reason = ""
        self._last_body_positions = None
        self._last_body_quaternions = None

    def get_timestamp_ns(self) -> int:
        if xrt is None:
            return 0
        return int(xrt.get_body_timestamp_ns())

    def _update_controller_freshness(
        self,
        now: float,
        controller_snapshot: Mapping[str, Any],
    ) -> None:
        # XRoboToolkit currently exposes one source-frame timestamp for the
        # required left/right controller pair, not independent per-side clocks.
        for side in ("left", "right"):
            try:
                stamp_ns = int(controller_snapshot[f"{side}_timestamp_ns"])
            except (KeyError, TypeError, ValueError):
                continue
            if stamp_ns <= 0:
                continue
            previous = self._controller_last_stamp_ns[side]
            if previous is None:
                self._controller_last_stamp_ns[side] = stamp_ns
                self._controller_last_new_data_time[side] = None
            elif stamp_ns > previous:
                self._controller_last_stamp_ns[side] = stamp_ns
                self._controller_last_new_data_time[side] = now
            elif stamp_ns < previous:
                # Require a second advancing frame after a controller timestamp
                # epoch reset. Never bless the regressing frame itself.
                self._controller_last_stamp_ns[side] = stamp_ns
                self._controller_last_new_data_time[side] = None

    def _update_motion_tracker_freshness(
        self,
        now: float,
        motion_snapshot: Mapping[str, Any],
    ) -> None:
        self._motion_tracker_count = 0
        self._motion_tracker_serials = ()

        try:
            count = int(motion_snapshot["count"])
            serials = tuple(str(serial).strip() for serial in motion_snapshot["serial_numbers"][: max(count, 0)])
            stamp_ns = int(motion_snapshot["timestamp_ns"])
        except (KeyError, TypeError, ValueError):
            self._motion_tracker_last_new_data_time = None
            return

        self._motion_tracker_count = max(count, 0)
        self._motion_tracker_serials = serials
        unique_serials = {serial for serial in serials if serial}
        if count < 2 or len(serials) < count or len(unique_serials) < 2 or stamp_ns <= 0:
            self._motion_tracker_last_new_data_time = None
            return

        previous = self._motion_tracker_last_stamp_ns
        if previous is None:
            # Establish an epoch, then require an advancing source frame before
            # declaring the trackers live.
            self._motion_tracker_last_stamp_ns = stamp_ns
            self._motion_tracker_last_new_data_time = None
        elif stamp_ns > previous:
            self._motion_tracker_last_stamp_ns = stamp_ns
            self._motion_tracker_last_new_data_time = now
        elif stamp_ns < previous:
            self._motion_tracker_last_stamp_ns = stamp_ns
            self._motion_tracker_last_new_data_time = None

    def _update_controller_tracking_health(
        self,
        now: float,
        controller_snapshot: Mapping[str, Any],
    ) -> str:
        if controller_snapshot.get("health_supported") is not True:
            self._controller_health_last_new_data_time = None
            return "PICO client lacks hardened controller-tracking health telemetry"
        if controller_snapshot.get("health_available") is not True:
            self._controller_health_last_new_data_time = None
            return "PICO controller-tracking health telemetry is unavailable"
        if controller_snapshot.get("health_valid") is not True:
            self._controller_health_last_new_data_time = None
            invalid_sides = [
                side
                for side in ("left", "right")
                if controller_snapshot.get(f"health_{side}_valid") is not True
            ]
            suffix = f": {', '.join(invalid_sides)}" if invalid_sides else ""
            return f"PICO controller tracking invalid{suffix}"
        try:
            sequence = int(controller_snapshot["health_sample_sequence"])
        except (KeyError, TypeError, ValueError):
            self._controller_health_last_new_data_time = None
            return "PICO controller-tracking health sequence is malformed"
        if sequence <= 0:
            self._controller_health_last_new_data_time = None
            return "PICO controller-tracking health sequence is not positive"

        previous = self._controller_health_last_sequence
        if previous is None:
            self._controller_health_last_sequence = sequence
            self._controller_health_last_new_data_time = None
            return "waiting for advancing PICO controller-tracking health sequence"
        if sequence > previous:
            self._controller_health_last_sequence = sequence
            self._controller_health_last_new_data_time = now
            return ""
        if sequence < previous:
            self._controller_health_last_sequence = sequence
            self._controller_health_last_new_data_time = None
            return "PICO controller-tracking health sequence regressed"
        if self._controller_health_last_new_data_time is None:
            return "waiting for advancing PICO controller-tracking health sequence"
        return ""

    def _update_body_tracker_health(
        self,
        now: float,
        body_snapshot: Mapping[str, Any],
    ) -> str:
        if body_snapshot.get("health_supported") is not True:
            self._body_tracker_health_last_new_data_time = None
            return "PICO client lacks hardened body-tracker health telemetry"
        if body_snapshot.get("health_available") is not True:
            self._body_tracker_health_last_new_data_time = None
            return "PICO body-tracker health telemetry is unavailable"
        if body_snapshot.get("health_valid") is not True:
            self._body_tracker_health_last_new_data_time = None
            try:
                state = int(body_snapshot["health_body_state_code"])
                error = int(body_snapshot["health_body_error_code"])
                trackers = int(body_snapshot["health_tracker_count"])
            except (KeyError, TypeError, ValueError):
                return "PICO body-tracker health telemetry is malformed"
            return (
                "PICO body-tracker health invalid "
                f"(state={state}, error={error}, trackers={trackers})"
            )
        try:
            sequence = int(body_snapshot["health_sample_sequence"])
        except (KeyError, TypeError, ValueError):
            self._body_tracker_health_last_new_data_time = None
            return "PICO body-tracker health sequence is malformed"
        if sequence <= 0:
            self._body_tracker_health_last_new_data_time = None
            return "PICO body-tracker health sequence is not positive"

        previous = self._body_tracker_health_last_sequence
        if previous is None:
            self._body_tracker_health_last_sequence = sequence
            self._body_tracker_health_last_new_data_time = None
            return "waiting for advancing PICO body-tracker health sequence"
        if sequence > previous:
            self._body_tracker_health_last_sequence = sequence
            self._body_tracker_health_last_new_data_time = now
            return ""
        if sequence < previous:
            self._body_tracker_health_last_sequence = sequence
            self._body_tracker_health_last_new_data_time = None
            return "PICO body-tracker health sequence regressed"
        if self._body_tracker_health_last_new_data_time is None:
            return "waiting for advancing PICO body-tracker health sequence"
        return ""

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            body_snapshot: dict[str, Any] | None = None
            motion_snapshot: dict[str, Any] | None = None
            atomic_snapshot_contract: str | None = None
            body_derivative_layout_contract: str | None = None
            body_snapshot_contract: str | None = None
            body_source_coherence_contract: str | None = None
            if xrt is not None:
                try:
                    atomic_getter = getattr(
                        xrt,
                        "get_xr24_ankle_snapshot",
                        None,
                    )
                    if callable(atomic_getter):
                        atomic_snapshot = dict(atomic_getter())
                        candidate_contract = str(
                            atomic_snapshot.get("contract", "")
                        )
                        if (
                            candidate_contract == XRT_ATOMIC_SNAPSHOT_CONTRACT
                            and atomic_snapshot.get(
                                "derivative_layout_contract"
                            )
                            == XRT_DERIVATIVE_LAYOUT_CONTRACT
                            and atomic_snapshot.get(
                                "source_coherence_contract"
                            )
                            == XRT_SOURCE_COHERENCE_CONTRACT
                        ):
                            atomic_snapshot_contract = candidate_contract
                            body_derivative_layout_contract = (
                                XRT_DERIVATIVE_LAYOUT_CONTRACT
                            )
                            body_snapshot = dict(atomic_snapshot["body"])
                            motion_snapshot = dict(atomic_snapshot["trackers"])
                        else:
                            # Current PICO modes cannot provide body tracking
                            # and raw motion-tracker poses in one source frame.
                            # Keep generic body teleop available, but never
                            # promote retained cross-mode tracker state.
                            body_snapshot = dict(xrt.get_body_snapshot())
                    else:
                        body_snapshot = dict(xrt.get_body_snapshot())
                    if body_derivative_layout_contract is None:
                        candidate_layout = body_snapshot.get(
                            "derivative_layout_contract"
                        )
                        if candidate_layout == XRT_DERIVATIVE_LAYOUT_CONTRACT:
                            body_derivative_layout_contract = str(
                                candidate_layout
                            )
                    if (
                        body_snapshot.get("contract")
                        == XRT_BODY_SNAPSHOT_CONTRACT
                        and body_snapshot.get(
                            "source_coherence_contract"
                        )
                        == XRT_BODY_SOURCE_COHERENCE_CONTRACT
                    ):
                        body_snapshot_contract = (
                            XRT_BODY_SNAPSHOT_CONTRACT
                        )
                        body_source_coherence_contract = (
                            XRT_BODY_SOURCE_COHERENCE_CONTRACT
                        )
                except Exception:
                    body_snapshot = None
                    motion_snapshot = None
            health_reason = (
                "PICO body-tracker health snapshot is unavailable"
                if body_snapshot is None
                else self._update_body_tracker_health(time.monotonic(), body_snapshot)
            )
            hardened_health_failed = (
                body_snapshot is not None
                and body_snapshot.get("health_supported") is True
                and bool(health_reason)
            )
            if (
                hardened_health_failed
                or body_snapshot is None
                or not bool(body_snapshot.get("available"))
            ):
                with self._lock:
                    self._latest = None
                    self._tracking_fault_reason = health_reason or (
                        "PICO body data is unavailable"
                    )
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

            # Body freshness must use the body stream timestamp. The general
            # device timestamp can continue advancing from controller/head
            # traffic while full-body tracking is frozen.
            try:
                stamp_ns = int(body_snapshot["timestamp_ns"])
            except (KeyError, TypeError, ValueError):
                stamp_ns = 0
            prev_stamp_ns = self._last_stamp_ns
            if stamp_ns <= 0:
                with self._lock:
                    self._latest = None
                time.sleep(0.001)
                continue
            if prev_stamp_ns is None:
                # A service may retain one cached frame. Establish the source
                # timestamp epoch, but require a strictly advancing frame before
                # publishing trusted host freshness.
                self._last_stamp_ns = stamp_ns
                with self._lock:
                    self._latest = None
                time.sleep(0.000001)
                continue
            if prev_stamp_ns is not None and stamp_ns <= prev_stamp_ns:
                if (
                    time.monotonic() - self._last_new_data_time > self.STALE_TIMEOUT
                    and not self._disconnected.is_set()
                ):
                    logger.warning(
                        "[PicoReader] Timestamp stale or regressing for %.1fs, flagging disconnect",
                        self.STALE_TIMEOUT,
                    )
                    self._disconnected.set()
                    # Permit a restarted timestamp epoch to recover only after
                    # the manager watchdog has already failed closed. A frozen
                    # equal timestamp remains rejected.
                    if stamp_ns < prev_stamp_ns:
                        self._last_stamp_ns = stamp_ns
                    with self._lock:
                        self._latest = None
                time.sleep(0.000001)
                continue

            sample_time = time.monotonic()
            self._last_new_data_time = sample_time
            if self._disconnected.is_set():
                logger.info("[PicoReader] Fresh data received, connection restored")
                self._disconnected.clear()

            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns

            try:
                body_poses = np.asarray(body_snapshot["poses"])
                frame_displacement_m = float("inf")
                frame_angular_displacement_rad = float("inf")
                if body_poses.ndim == 2 and body_poses.shape[0] >= 24 and body_poses.shape[1] >= 7:
                    try:
                        body_positions = body_poses[:24, :3].astype(np.float64)
                        body_quaternions = body_poses[:24, 3:7].astype(np.float64)
                        quaternion_norms = np.linalg.norm(
                            body_quaternions,
                            axis=1,
                            keepdims=True,
                        )
                        if (
                            np.isfinite(body_positions).all()
                            and np.isfinite(body_quaternions).all()
                            and np.all(quaternion_norms > 0.0)
                        ):
                            body_quaternions /= quaternion_norms
                            if self._last_body_positions is None:
                                frame_displacement_m = 0.0
                            else:
                                frame_displacement_m = float(
                                    np.linalg.norm(
                                        body_positions - self._last_body_positions,
                                        axis=1,
                                    ).max()
                                )
                            if self._last_body_quaternions is None:
                                frame_angular_displacement_rad = 0.0
                            else:
                                # q and -q encode the same rotation. Absolute
                                # dot products make the discontinuity metric
                                # invariant to harmless quaternion sign flips.
                                quaternion_dots = np.abs(
                                    np.sum(
                                        body_quaternions * self._last_body_quaternions,
                                        axis=1,
                                    )
                                )
                                frame_angular_displacement_rad = float(
                                    (2.0 * np.arccos(np.clip(quaternion_dots, 0.0, 1.0))).max()
                                )
                            self._last_body_positions = body_positions.copy()
                            self._last_body_quaternions = body_quaternions.copy()
                    except (TypeError, ValueError):
                        pass
                controller_snapshot = dict(xrt.get_controller_snapshot())
                if motion_snapshot is None:
                    motion_snapshot = dict(xrt.get_motion_tracker_snapshot())
                self._update_controller_freshness(sample_time, controller_snapshot)
                controller_health_reason = self._update_controller_tracking_health(
                    sample_time,
                    controller_snapshot,
                )
                self._update_motion_tracker_freshness(
                    sample_time,
                    motion_snapshot,
                )
                hardened_controller_health_failed = (
                    body_snapshot.get("health_supported") is True
                    and bool(controller_health_reason)
                )
                if hardened_controller_health_failed:
                    with self._lock:
                        self._latest = None
                        self._tracking_fault_reason = controller_health_reason
                    time.sleep(0.001)
                    continue
                body_health_source_timestamp_ns = int(
                    body_snapshot.get("health_timestamp_ns", 0)
                )
                controller_health_source_timestamp_ns = int(
                    controller_snapshot.get("health_timestamp_ns", 0)
                )
                both_health_protocols_active = (
                    body_snapshot.get("health_supported") is True
                    and controller_snapshot.get("health_supported") is True
                )
                if (
                    both_health_protocols_active
                    and body_health_source_timestamp_ns
                    != controller_health_source_timestamp_ns
                ):
                    # The two getter calls straddled a packet commit. Keep the
                    # prior sample briefly and retry instead of combining data
                    # from different source frames.
                    time.sleep(0.001)
                    continue
                controller_data = {
                    "left_pose": np.asarray(controller_snapshot["left_pose"], dtype=np.float64),
                    "right_pose": np.asarray(controller_snapshot["right_pose"], dtype=np.float64),
                    "left_trigger_value": float(controller_snapshot["left_trigger_value"]),
                    "right_trigger_value": float(controller_snapshot["right_trigger_value"]),
                    "left_squeeze_value": float(controller_snapshot["left_squeeze_value"]),
                    "right_squeeze_value": float(controller_snapshot["right_squeeze_value"]),
                    "left_thumbstick": np.asarray(
                        controller_snapshot["left_thumbstick"],
                        dtype=np.float64,
                    ),
                    "right_thumbstick": np.asarray(
                        controller_snapshot["right_thumbstick"],
                        dtype=np.float64,
                    ),
                    "left_thumbstick_click": bool(controller_snapshot["left_thumbstick_click"]),
                    "right_thumbstick_click": bool(controller_snapshot["right_thumbstick_click"]),
                    "left_primary_click": bool(controller_snapshot["left_primary_click"]),
                    "left_secondary_click": bool(controller_snapshot["left_secondary_click"]),
                    "right_primary_click": bool(controller_snapshot["right_primary_click"]),
                    "right_secondary_click": bool(controller_snapshot["right_secondary_click"]),
                    "left_menu_button": bool(controller_snapshot["left_menu_button"]),
                    "right_menu_button": bool(controller_snapshot["right_menu_button"]),
                }
                sample = {
                    "body_poses_np": body_poses,
                    "body_velocities_np": (
                        np.asarray(body_snapshot["velocities"], dtype=np.float64)
                        if (
                            body_derivative_layout_contract
                            == XRT_DERIVATIVE_LAYOUT_CONTRACT
                            and "velocities" in body_snapshot
                        )
                        else None
                    ),
                    "body_accelerations_np": (
                        np.asarray(
                            body_snapshot["accelerations"],
                            dtype=np.float64,
                        )
                        if (
                            body_derivative_layout_contract
                            == XRT_DERIVATIVE_LAYOUT_CONTRACT
                            and "accelerations" in body_snapshot
                        )
                        else None
                    ),
                    "body_joint_timestamps_ns": (
                        tuple(
                            int(value)
                            for value in body_snapshot[
                                "joint_timestamps_ns"
                            ]
                        )
                        if "joint_timestamps_ns" in body_snapshot
                        else None
                    ),
                    "body_frame_displacement_m": frame_displacement_m,
                    "body_frame_angular_displacement_rad": (frame_angular_displacement_rad),
                    "timestamp_realtime": time.time(),
                    "timestamp_monotonic": sample_time,
                    "timestamp_ns": stamp_ns,
                    "left_controller_timestamp_monotonic": (self._controller_last_new_data_time["left"]),
                    "right_controller_timestamp_monotonic": (self._controller_last_new_data_time["right"]),
                    "motion_tracker_timestamp_monotonic": (self._motion_tracker_last_new_data_time),
                    "motion_tracker_source_timestamp_ns": int(
                        motion_snapshot.get("timestamp_ns", 0)
                    ),
                    "motion_tracker_count": self._motion_tracker_count,
                    "motion_tracker_serial_numbers": self._motion_tracker_serials,
                    "motion_tracker_poses_np": (
                        np.asarray(
                            motion_snapshot["poses"],
                            dtype=np.float64,
                        )
                        if "poses" in motion_snapshot
                        else None
                    ),
                    "motion_tracker_velocities_np": (
                        np.asarray(
                            motion_snapshot["velocities"],
                            dtype=np.float64,
                        )
                        if "velocities" in motion_snapshot
                        else None
                    ),
                    "motion_tracker_accelerations_np": (
                        np.asarray(
                            motion_snapshot["accelerations"],
                            dtype=np.float64,
                        )
                        if "accelerations" in motion_snapshot
                        else None
                    ),
                    "xrt_atomic_snapshot_contract": atomic_snapshot_contract,
                    "xrt_body_derivative_layout_contract": (
                        body_derivative_layout_contract
                    ),
                    "xrt_body_snapshot_contract": body_snapshot_contract,
                    "xrt_body_source_coherence_contract": (
                        body_source_coherence_contract
                    ),
                    "body_tracker_health_supported": bool(
                        body_snapshot.get("health_supported", False)
                    ),
                    "body_tracker_health_available": bool(
                        body_snapshot.get("health_available", False)
                    ),
                    "body_tracker_health_valid": bool(
                        body_snapshot.get("health_valid", False)
                    ),
                    "body_tracker_health_schema_version": int(
                        body_snapshot.get("health_schema_version", 0)
                    ),
                    "body_tracker_health_sample_sequence": int(
                        body_snapshot.get("health_sample_sequence", 0)
                    ),
                    "body_tracker_health_source_timestamp_ns": (
                        body_health_source_timestamp_ns
                    ),
                    "body_tracker_health_timestamp_monotonic": (
                        self._body_tracker_health_last_new_data_time
                    ),
                    "body_tracker_health_calibration_result": int(
                        body_snapshot.get("health_calibration_result", 0)
                    ),
                    "body_tracker_health_calibrated": bool(
                        body_snapshot.get("health_calibrated", False)
                    ),
                    "body_tracker_health_tracking_mode": int(
                        body_snapshot.get("health_tracking_mode", -1)
                    ),
                    "body_tracker_health_connect_state_result": int(
                        body_snapshot.get("health_connect_state_result", 0)
                    ),
                    "body_tracker_health_tracker_count": int(
                        body_snapshot.get("health_tracker_count", 0)
                    ),
                    "body_tracker_health_unique_tracker_count": int(
                        body_snapshot.get("health_unique_tracker_count", 0)
                    ),
                    "body_tracker_health_body_state_result": int(
                        body_snapshot.get("health_body_state_result", 0)
                    ),
                    "body_tracker_health_is_tracking": bool(
                        body_snapshot.get("health_is_tracking", False)
                    ),
                    "body_tracker_health_tracking_state_code": int(
                        body_snapshot.get("health_tracking_state_code", 0)
                    ),
                    "body_tracker_health_body_state_code": int(
                        body_snapshot.get("health_body_state_code", 0)
                    ),
                    "body_tracker_health_body_error_code": int(
                        body_snapshot.get("health_body_error_code", 0)
                    ),
                    "body_tracker_health_connected_band_count": int(
                        body_snapshot.get("health_connected_band_count", 0)
                    ),
                    "body_tracker_health_body_data_result": int(
                        body_snapshot.get("health_body_data_result", 0)
                    ),
                    "body_tracker_health_body_role_count": int(
                        body_snapshot.get("health_body_role_count", 0)
                    ),
                    "controller_tracking_health_supported": bool(
                        controller_snapshot.get("health_supported", False)
                    ),
                    "controller_tracking_health_available": bool(
                        controller_snapshot.get("health_available", False)
                    ),
                    "controller_tracking_health_valid": bool(
                        controller_snapshot.get("health_valid", False)
                    ),
                    "controller_tracking_health_schema_version": int(
                        controller_snapshot.get("health_schema_version", 0)
                    ),
                    "controller_tracking_health_sample_sequence": int(
                        controller_snapshot.get("health_sample_sequence", 0)
                    ),
                    "controller_tracking_health_source_timestamp_ns": (
                        controller_health_source_timestamp_ns
                    ),
                    "controller_tracking_health_timestamp_monotonic": (
                        self._controller_health_last_new_data_time
                    ),
                    "controller_tracking_health_left_device_valid": bool(
                        controller_snapshot.get("health_left_device_valid", False)
                    ),
                    "controller_tracking_health_left_is_tracked_available": bool(
                        controller_snapshot.get(
                            "health_left_is_tracked_available",
                            False,
                        )
                    ),
                    "controller_tracking_health_left_is_tracked": bool(
                        controller_snapshot.get("health_left_is_tracked", False)
                    ),
                    "controller_tracking_health_left_tracking_state_available": bool(
                        controller_snapshot.get(
                            "health_left_tracking_state_available",
                            False,
                        )
                    ),
                    "controller_tracking_health_left_tracking_state": int(
                        controller_snapshot.get("health_left_tracking_state", 0)
                    ),
                    "controller_tracking_health_left_valid": bool(
                        controller_snapshot.get("health_left_valid", False)
                    ),
                    "controller_tracking_health_right_device_valid": bool(
                        controller_snapshot.get("health_right_device_valid", False)
                    ),
                    "controller_tracking_health_right_is_tracked_available": bool(
                        controller_snapshot.get(
                            "health_right_is_tracked_available",
                            False,
                        )
                    ),
                    "controller_tracking_health_right_is_tracked": bool(
                        controller_snapshot.get("health_right_is_tracked", False)
                    ),
                    "controller_tracking_health_right_tracking_state_available": bool(
                        controller_snapshot.get(
                            "health_right_tracking_state_available",
                            False,
                        )
                    ),
                    "controller_tracking_health_right_tracking_state": int(
                        controller_snapshot.get("health_right_tracking_state", 0)
                    ),
                    "controller_tracking_health_right_valid": bool(
                        controller_snapshot.get("health_right_valid", False)
                    ),
                    "controller_data": controller_data,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                }
                with self._lock:
                    self._latest = sample
                    self._tracking_fault_reason = ""

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
                with self._lock:
                    self._latest = None


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


# Number of joints in the IsaacTeleop FullBodyPosePicoT (XR_BD_body_tracking).
# Mirrors core.BodyJointPico.NUM_JOINTS in IsaacTeleop's schema bindings.
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
    """Convert ``FullBodyTrackerPico.get_body_pose().data`` to a (24, 7) array.

    Returns ``None`` while no joint is valid (typical when the headset isn't
    connected yet — every ``BodyJointPose.is_valid`` is False, the streamer
    keeps polling and the C++ deploy doesn't see fake zero pose).

    Two accepted schemas:

    Schema A — IsaacTeleop ``FullBodyPosePicoT`` (DeviceIO direct).
        Defined in IsaacTeleop's ``schema/full_body.fbs`` /
        ``schema/python/full_body_bindings.h``::

            FullBodyPosePicoT.joints                → BodyJointsPico (attr)
            BodyJointsPico.joints(index)            → BodyJointPose  (METHOD; index 0..23)
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

    # Schema A: native FullBodyPosePicoT — joints exposed via
    # BodyJointsPico.joints(index) method (one BodyJointPose per call).
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
    fields = (
        "trigger_value",
        "squeeze_value",
        "thumbstick_x",
        "thumbstick_y",
        "thumbstick_click",
        "primary_click",
        "secondary_click",
    )
    sentinel = object()
    projected: dict[str, float] = {}
    for field in fields:
        value = _attr_or_item(inputs, field, sentinel)
        if value is sentinel or value is None:
            return None
        try:
            projected[field] = float(value)
        except (TypeError, ValueError):
            return None
    return projected


def _controller_snapshot_is_tracked(snapshot: Any) -> bool:
    if snapshot is None:
        return False
    for pose_name in ("aim_pose", "grip_pose"):
        pose = _attr_or_item(snapshot, pose_name)
        if pose is not None and bool(_attr_or_item(pose, "is_valid", False)):
            return True
    return False


def _controller_snapshot_pose(snapshot: Any) -> np.ndarray | None:
    if snapshot is None:
        return None
    for pose_name in ("aim_pose", "grip_pose"):
        tracked_pose = _attr_or_item(snapshot, pose_name)
        if tracked_pose is None or not bool(_attr_or_item(tracked_pose, "is_valid", False)):
            continue
        pose = _attr_or_item(tracked_pose, "pose")
        position = _vec3(_attr_or_item(pose, "position"))
        orientation = _quat_xyzw(_attr_or_item(pose, "orientation"))
        if position is not None and orientation is not None:
            return np.asarray((*position, *orientation), dtype=np.float64)
    return None


def _build_controller_dict(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert ``IsaacTeleopClient._get_tracker_data()`` into the controller dict
    schema that ``pico_manager_thread_server`` consumes (left/right trigger,
    squeeze, thumbstick, click, primary/secondary click)."""
    if raw is None:
        return None

    left = _controller_inputs_to_dict_side(raw.get("left_controller"))
    right = _controller_inputs_to_dict_side(raw.get("right_controller"))
    if left is None or right is None:
        return None

    out: dict[str, Any] = {
        "_left_tracking_valid": _controller_snapshot_is_tracked(raw.get("left_controller")),
        "_right_tracking_valid": _controller_snapshot_is_tracked(raw.get("right_controller")),
        "left_pose": _controller_snapshot_pose(raw.get("left_controller")),
        "right_pose": _controller_snapshot_pose(raw.get("right_controller")),
    }
    out["left_trigger_value"] = left["trigger_value"]
    out["left_squeeze_value"] = left["squeeze_value"]
    out["left_thumbstick"] = [left["thumbstick_x"], left["thumbstick_y"]]
    out["left_thumbstick_click"] = left["thumbstick_click"]
    out["left_primary_click"] = left["primary_click"]
    out["left_secondary_click"] = left["secondary_click"]
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
    # IsaacTeleop's live Tracked wrappers expose payloads but no source frame
    # timestamps. Host poll time cannot safely prove that a cached OpenXR frame
    # advanced, so the fail-closed robot manager rejects this reader.
    supports_trusted_source_timestamps = False

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
                "  uv pip install 'isaacteleop[cloudxr]~=1.3.0' --prerelease=allow "
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
        self._controller_last_new_data_time = {"left": None, "right": None}

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
        self._controller_last_new_data_time = {"left": None, "right": None}

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
                controller_time = time.monotonic()
                for side in ("left", "right"):
                    if controller.get(f"_{side}_tracking_valid", False):
                        self._controller_last_new_data_time[side] = controller_time
                with self._ctrl_lock:
                    self._latest_controller = controller

            body_poses = _body_data_to_24x7(raw.get("full_body"))
            if body_poses is None:
                if not self._unrecognised_logged and not _attr_or_item(raw.get("full_body"), "joint_positions"):
                    self._unrecognised_logged = True
                time.sleep(self._period)
                continue

            stamp_ns = int(self._client.get_timestamp_ns())
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns <= prev_stamp_ns:
                if (
                    time.monotonic() - self._last_new_data_time > self.STALE_TIMEOUT
                    and not self._disconnected.is_set()
                ):
                    logger.warning(
                        "[IsaacTeleopReader] Device timestamp stale for %.1fs, flagging disconnect",
                        self.STALE_TIMEOUT,
                    )
                    self._disconnected.set()
                    # Permit a restarted timestamp epoch to recover only after
                    # the manager watchdog has already failed closed. A frozen
                    # equal timestamp remains rejected.
                    if stamp_ns < prev_stamp_ns:
                        self._last_stamp_ns = stamp_ns
                    with self._lock:
                        self._latest = None
                time.sleep(self._period)
                continue

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
                "left_controller_timestamp_monotonic": (self._controller_last_new_data_time["left"]),
                "right_controller_timestamp_monotonic": (self._controller_last_new_data_time["right"]),
                "controller_data": controller,
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
