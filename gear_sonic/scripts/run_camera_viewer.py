"""
ROS-free camera viewer with optional recording.

Connects to a ZMQ camera server (MuJoCo sim SensorServer or real robot camera)
and displays live camera feeds using OpenCV. Supports recording to MP4.
Also subscribes to the data exporter's episode status ZMQ stream and overlays
the current episode index and duration. G1 upper-body and BrainCo hand
``tau_est`` feedback can be displayed as a live rolling plot alongside the
camera feeds.

Virtual environment setup (run from repo root):
    bash install_scripts/install_data_collection.sh
    source .venv_data_collection/bin/activate

Usage:
    python gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5555

Controls (OpenCV window must be focused):
    A / D - Previous/next tau preset
    R - Start/stop recording
    Q - Quit

Output structure:
    camera_recordings/
    └── rec_20260403_143052/
        ├── ego_view.mp4
        └── head_left_color_image.mp4
"""

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Optional, Sequence

import cv2
import numpy as np
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
except ImportError as exc:  # pragma: no cover - depends on the robot environment
    ChannelFactoryInitialize = None
    ChannelSubscriber = None
    MotorStates_ = None
    _BRAINCO_DDS_IMPORT_ERROR = exc
else:
    _BRAINCO_DDS_IMPORT_ERROR = None


EPISODE_STATUS_TOPIC = "episode_status"
EXTERNAL_VIEW_STREAM_NAME = "external-view-camera"
BRAINCO_MOTOR_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
BRAINCO_NUM_MOTORS = len(BRAINCO_MOTOR_NAMES)
G1_UPPER_BODY_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_NUM_UPPER_BODY_MOTORS = len(G1_UPPER_BODY_JOINT_NAMES)
TAU_JOINT_NAMES = (
    *G1_UPPER_BODY_JOINT_NAMES,
    *(f"left_brainco_{name}" for name in BRAINCO_MOTOR_NAMES),
    *(f"right_brainco_{name}" for name in BRAINCO_MOTOR_NAMES),
)
TAU_JOINT_INDEX = {name: index for index, name in enumerate(TAU_JOINT_NAMES)}
DEFAULT_TAU_PRESETS_PATH = (
    Path(__file__).resolve().parent / "viewer_tau_pressets" / "tau_presets.json"
)
BRAINCO_TAU_TOPICS = {
    "left": "rt/brainco/left/state",
    "right": "rt/brainco/right/state",
}
SONIC_TAU_TOPICS = ("pose", "manager_state")
SONIC_HEADER_SIZE = 1280
G1_UPPER_BODY_STATE_SIZE = 70
G1_UPPER_BODY_TAU_SLICE = slice(53, 70)
G1_UPPER_BODY_MAX_AGE_SECONDS = 0.25
TAU_STALE_SECONDS = 0.5
TAU_PLOT_COLORS = (
    (80, 180, 255),
    (255, 160, 70),
    (80, 230, 100),
    (220, 100, 255),
    (60, 220, 240),
    (255, 120, 150),
    (170, 230, 80),
    (240, 190, 80),
    (180, 120, 255),
    (80, 230, 210),
    (230, 130, 80),
    (160, 200, 255),
)


@dataclass
class CameraViewerConfig:
    """CLI config for the ROS-free camera viewer."""

    camera_host: str = "localhost"
    """Camera server hostname."""

    camera_port: int = 5555
    """Camera server port."""

    external_view_camera_host: Optional[str] = None
    """Host publishing external-view-camera preview; None disables the stream."""

    external_view_camera_port: int = 5582
    """ZMQ port published by the standalone external-view camera process."""

    fps: int = 50
    """Target display refresh rate (Hz)."""

    output_path: Optional[str] = None
    """Output directory for recordings. Auto-creates 'camera_recordings/' if not set."""

    codec: str = "mp4v"
    """Video codec for recording (e.g., 'mp4v', 'XVID')."""

    max_display_width: int = 0
    """Max width per camera tile in the display window. Set 0 to keep source size."""

    camera_streams: tuple[str, ...] = (
        "head",
        "head_depth",
        "ego_view",
        "ego_view_depth",
    )
    """Camera keys to display and record, in display order."""

    show_depth: bool = True
    """Display depth streams. Recording remains enabled for requested streams."""

    depth_camera: str = "ego_view"
    """Camera whose single depth stream is displayed: ego_view or head."""

    grid_columns: int = 2
    """Number of camera tiles per row when the tau plot is disabled."""

    depth_display_min_meters: float = 0.2
    """Near end of the depth display colour scale."""

    depth_display_max_meters: float = 5.0
    """Far end of the depth display colour scale."""

    episode_status_zmq_host: Optional[str] = "localhost"
    """ZMQ host for episode status messages. None uses camera_host."""

    episode_status_zmq_port: int = 5581
    """ZMQ port for episode status messages from run_data_exporter.py."""

    show_episode_status: bool = True
    """Show current data-collection episode status overlay."""

    show_tau_plot: bool = False
    """Show live G1 body and BrainCo hand tau_est plots."""

    tau_window_seconds: float = 20.0
    """Length of the rolling tau plot time window in seconds."""

    brainco_dds_domain_id: int = 0
    """DDS domain used by the BrainCo hand state topics."""

    brainco_network_interface: Optional[str] = "wlp128s20f3"
    """DDS network interface. Set None to let the SDK choose automatically."""

    sonic_zmq_host: Optional[str] = "localhost"
    """Host publishing pose/manager_state. None reuses camera_host."""

    sonic_zmq_port: int = 5556
    """Port publishing the pico manager pose/manager_state topics."""

    tau_presets_path: str = str(DEFAULT_TAU_PRESETS_PATH)
    """JSON file containing named lists of joints for the tau plot."""

    tau_preset: Optional[str] = None
    """Initial tau preset. None selects the first preset in the JSON file."""


def _optional_string(value: str) -> Optional[str]:
    return None if value.strip().lower() in {"", "auto", "none"} else value


def parse_args(argv: Optional[Sequence[str]] = None) -> CameraViewerConfig:
    """Parse command-line arguments using the standard-library argparse."""
    defaults = CameraViewerConfig()
    parser = argparse.ArgumentParser(
        description="Display and optionally record SONIC camera and motor tau streams.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera-host", default=defaults.camera_host)
    parser.add_argument("--camera-port", type=int, default=defaults.camera_port)
    parser.add_argument(
        "--external-view-camera-host",
        type=_optional_string,
        default=defaults.external_view_camera_host,
        help="Host publishing external-view-camera; use 'none' to disable",
    )
    parser.add_argument(
        "--external-view-camera-port",
        type=int,
        default=defaults.external_view_camera_port,
    )
    parser.add_argument("--fps", type=int, default=defaults.fps)
    parser.add_argument("--output-path", default=defaults.output_path)
    parser.add_argument("--codec", default=defaults.codec)
    parser.add_argument(
        "--max-display-width", type=int, default=defaults.max_display_width
    )
    parser.add_argument(
        "--camera-streams",
        nargs="+",
        default=list(defaults.camera_streams),
        metavar="STREAM",
    )
    parser.add_argument(
        "--no-depth",
        dest="show_depth",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Do not display depth streams; the tau plot takes the free grid cell",
    )
    parser.add_argument(
        "--depth-camera",
        default=defaults.depth_camera,
        metavar="CAMERA",
        help="Display only CAMERA_depth (for example ego_view or head)",
    )
    parser.add_argument("--grid-columns", type=int, default=defaults.grid_columns)
    parser.add_argument(
        "--depth-display-min-meters",
        type=float,
        default=defaults.depth_display_min_meters,
    )
    parser.add_argument(
        "--depth-display-max-meters",
        type=float,
        default=defaults.depth_display_max_meters,
    )
    parser.add_argument(
        "--episode-status-zmq-host",
        type=_optional_string,
        default=defaults.episode_status_zmq_host,
        help="Use 'none' to reuse --camera-host",
    )
    parser.add_argument(
        "--episode-status-zmq-port",
        type=int,
        default=defaults.episode_status_zmq_port,
    )
    parser.add_argument(
        "--show-episode-status",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_episode_status,
    )
    parser.add_argument(
        "--show-tau-plot",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_tau_plot,
    )
    parser.add_argument(
        "--no-motor-states",
        dest="show_tau_plot",
        action="store_false",
        help="Hide all motor-state/tau plots and skip their subscribers",
    )
    parser.add_argument(
        "--tau-window-seconds", type=float, default=defaults.tau_window_seconds
    )
    parser.add_argument(
        "--brainco-dds-domain-id", type=int, default=defaults.brainco_dds_domain_id
    )
    parser.add_argument(
        "--brainco-network-interface",
        type=_optional_string,
        default=defaults.brainco_network_interface,
        help="DDS interface; use 'auto' or 'none' for SDK auto-selection",
    )
    parser.add_argument(
        "--sonic-zmq-host",
        type=_optional_string,
        default=defaults.sonic_zmq_host,
        help="pose/manager_state host; use 'none' to reuse --camera-host",
    )
    parser.add_argument("--sonic-zmq-port", type=int, default=defaults.sonic_zmq_port)
    parser.add_argument("--tau-presets-path", default=defaults.tau_presets_path)
    parser.add_argument(
        "--tau-preset",
        default=defaults.tau_preset,
        help="Initial preset name; A/D switches presets at runtime",
    )

    args = parser.parse_args(argv)
    config = CameraViewerConfig(
        camera_host=args.camera_host,
        camera_port=args.camera_port,
        external_view_camera_host=args.external_view_camera_host,
        external_view_camera_port=args.external_view_camera_port,
        fps=args.fps,
        output_path=args.output_path,
        codec=args.codec,
        max_display_width=args.max_display_width,
        camera_streams=tuple(args.camera_streams),
        show_depth=getattr(args, "show_depth", defaults.show_depth),
        depth_camera=args.depth_camera.removesuffix("_depth"),
        grid_columns=args.grid_columns,
        depth_display_min_meters=args.depth_display_min_meters,
        depth_display_max_meters=args.depth_display_max_meters,
        episode_status_zmq_host=args.episode_status_zmq_host,
        episode_status_zmq_port=args.episode_status_zmq_port,
        show_episode_status=args.show_episode_status,
        show_tau_plot=args.show_tau_plot,
        tau_window_seconds=args.tau_window_seconds,
        brainco_dds_domain_id=args.brainco_dds_domain_id,
        brainco_network_interface=args.brainco_network_interface,
        sonic_zmq_host=args.sonic_zmq_host,
        sonic_zmq_port=args.sonic_zmq_port,
        tau_presets_path=args.tau_presets_path,
        tau_preset=args.tau_preset,
    )
    try:
        _validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    return config


def _validate_config(config: CameraViewerConfig) -> None:
    if not 1 <= config.camera_port <= 65535:
        raise ValueError("camera_port must be in the range 1..65535")
    if (
        config.external_view_camera_host is not None
        and not 1 <= config.external_view_camera_port <= 65535
    ):
        raise ValueError("external_view_camera_port must be in the range 1..65535")
    if config.fps <= 0:
        raise ValueError("fps must be positive")
    if config.max_display_width < 0:
        raise ValueError("max_display_width cannot be negative")
    if not config.camera_streams:
        raise ValueError("at least one camera stream must be requested")
    if not config.depth_camera:
        raise ValueError("depth_camera cannot be empty")
    if config.grid_columns <= 0:
        raise ValueError("grid_columns must be positive")
    if config.depth_display_max_meters <= config.depth_display_min_meters:
        raise ValueError(
            "depth_display_max_meters must exceed depth_display_min_meters"
        )
    if not 1 <= config.episode_status_zmq_port <= 65535:
        raise ValueError("episode_status_zmq_port must be in the range 1..65535")
    if not 1 <= config.sonic_zmq_port <= 65535:
        raise ValueError("sonic_zmq_port must be in the range 1..65535")
    if config.show_tau_plot and config.tau_window_seconds <= 0:
        raise ValueError("tau_window_seconds must be positive")


class EpisodeStatusSubscriber:
    """Non-blocking subscriber for data-exporter episode status messages."""

    def __init__(self, host: str, port: int, topic: str = EPISODE_STATUS_TOPIC):
        self.endpoint = f"tcp://{host}:{port}"
        self._topic_prefix = f"{topic} "
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic_prefix)
        self._socket.setsockopt(zmq.RCVHWM, 50)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)
        print(f"[EpisodeStatus] Subscribed to {self.endpoint} topic '{topic}'")

    def read(self) -> Optional[dict]:
        latest_status = None
        for _ in range(50):
            try:
                message = self._socket.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as e:
                print(f"[EpisodeStatus] Warning: failed to read status: {e}")
                break

            if not message.startswith(self._topic_prefix):
                continue

            try:
                latest_status = json.loads(message[len(self._topic_prefix) :])
                latest_status["_received_monotonic"] = time.monotonic()
            except json.JSONDecodeError as e:
                print(f"[EpisodeStatus] Warning: invalid status JSON: {e}")

        return latest_status

    def close(self):
        self._socket.close()
        self._ctx.term()


class TauCollector:
    """Merge upper-body ZMQ and hand DDS feedback into one 29-channel sample."""

    def __init__(self):
        self._lock = threading.Lock()
        self._values: dict[str, Optional[np.ndarray]] = {
            "body": None,
            "left": None,
            "right": None,
        }
        self._timestamps: dict[str, Optional[float]] = {
            "body": None,
            "left": None,
            "right": None,
        }
        self._version = 0
        self._last_read_version = 0

    def update(self, source: str, values: np.ndarray, expected_size: int) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size != expected_size:
            raise ValueError(
                f"received {values.size} motors; expected {expected_size}"
            )
        if not np.isfinite(values).all():
            raise ValueError("received non-finite tau_est values")

        with self._lock:
            self._values[source] = values.copy()
            self._timestamps[source] = time.monotonic()
            self._version += 1

    def read(self) -> Optional[tuple[float, np.ndarray]]:
        """Return one 29-channel sample when at least one source has new data."""
        now = time.monotonic()
        with self._lock:
            if self._version == self._last_read_version:
                return None
            self._last_read_version = self._version
            values = np.full(len(TAU_JOINT_NAMES), np.nan, dtype=np.float32)
            source_slices = {
                "body": slice(0, G1_NUM_UPPER_BODY_MOTORS),
                "left": slice(
                    G1_NUM_UPPER_BODY_MOTORS,
                    G1_NUM_UPPER_BODY_MOTORS + BRAINCO_NUM_MOTORS,
                ),
                "right": slice(G1_NUM_UPPER_BODY_MOTORS + BRAINCO_NUM_MOTORS, None),
            }
            for source, target_slice in source_slices.items():
                source_values = self._values[source]
                source_timestamp = self._timestamps[source]
                if (
                    source_values is not None
                    and source_timestamp is not None
                    and now - source_timestamp <= TAU_STALE_SECONDS
                ):
                    values[target_slice] = source_values

        if not np.isfinite(values).any():
            return None
        return now, values


class BraincoTauSubscriber:
    """Forward left/right BrainCo DDS ``tau_est`` into a TauCollector."""

    def __init__(
        self,
        collector: TauCollector,
        domain_id: int,
        network_interface: Optional[str],
    ):
        if _BRAINCO_DDS_IMPORT_ERROR is not None:
            raise ImportError(
                "unitree_sdk2py with MotorStates_ is required for hand tau"
            ) from _BRAINCO_DDS_IMPORT_ERROR

        if network_interface:
            ChannelFactoryInitialize(domain_id, network_interface)
        else:
            ChannelFactoryInitialize(domain_id)
        self._collector = collector
        self._last_warning_time = 0.0
        self._subscribers = []
        for side, topic in BRAINCO_TAU_TOPICS.items():
            subscriber = ChannelSubscriber(topic, MotorStates_)
            subscriber.Init(lambda msg, side=side: self._hand_state_callback(side, msg), 10)
            self._subscribers.append(subscriber)

        topics = ", ".join(BRAINCO_TAU_TOPICS.values())
        print(f"[BrainCoTau] DDS subscriptions initialized: {topics}")

    def _hand_state_callback(self, side: str, msg) -> None:
        try:
            values = [motor.tau_est for motor in msg.states]
            self._collector.update(side, values, BRAINCO_NUM_MOTORS)
        except Exception as exc:
            self._warn_invalid(side, exc)

    def _warn_invalid(self, source: str, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_warning_time >= 1.0:
            self._last_warning_time = now
            print(f"[BrainCoTau] Warning: invalid {source} state: {exc}")

    def close(self) -> None:
        for subscriber in self._subscribers:
            try:
                subscriber.Close()
            except Exception:
                pass


def _unpack_sonic_message(packed_data: bytes, topic: str) -> dict:
    """Decode the pico manager wire format used by run_data_exporter."""
    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"message does not start with topic '{topic}'")
    header_start = len(topic_bytes)
    payload_start = header_start + SONIC_HEADER_SIZE
    if len(packed_data) < payload_start:
        raise ValueError(
            f"packed data is too small: {len(packed_data)} < {payload_start}"
        )

    header_bytes = packed_data[header_start:payload_start].split(b"\x00", 1)[0]
    header = json.loads(header_bytes.decode("utf-8"))
    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": bool,
    }
    result = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
    current_offset = payload_start
    for field in header.get("fields", []):
        dtype = dtype_map.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        byte_count = int(np.prod(shape)) * np.dtype(dtype).itemsize
        field_end = current_offset + byte_count
        if field_end > len(packed_data):
            raise ValueError(f"truncated field '{field['name']}'")
        result[field["name"]] = (
            np.frombuffer(packed_data[current_offset:field_end], dtype=dtype)
            .reshape(shape)
            .copy()
        )
        current_offset = field_end
    return result


class G1UpperBodyTauSubscriber:
    """Read 17 upper-body torques from pico manager ``g1_upper_body_state``."""

    def __init__(self, collector: TauCollector, host: str, port: int):
        self._collector = collector
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 50)
        self._socket.setsockopt(zmq.LINGER, 0)
        for topic in SONIC_TAU_TOPICS:
            self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.endpoint = f"tcp://{host}:{port}"
        self._socket.connect(self.endpoint)
        self._last_warning_time = 0.0
        self.error_reason: Optional[str] = None
        topics = ", ".join(SONIC_TAU_TOPICS)
        print(f"[G1UpperBodyTau] Connected to {self.endpoint} ({topics})")

    def poll(self) -> None:
        latest_state = None
        received_message = False
        for _ in range(50):
            try:
                packed_data = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                topic = next(
                    topic
                    for topic in SONIC_TAU_TOPICS
                    if packed_data.startswith(topic.encode("utf-8"))
                )
                message = _unpack_sonic_message(packed_data, topic)
                received_message = True
                if "g1_upper_body_state" in message:
                    latest_state = message["g1_upper_body_state"]
            except Exception as exc:
                self._set_error(f"invalid pose/manager_state message: {exc}")

        if latest_state is None:
            if received_message:
                self._set_error(
                    "pose/manager_state has no 'g1_upper_body_state'; "
                    "start the pico manager variant that publishes G1 state"
                )
            return
        values = np.asarray(latest_state, dtype=np.float32).reshape(-1)
        if values.size != G1_UPPER_BODY_STATE_SIZE:
            self._set_error(
                f"g1_upper_body_state has {values.size} values; "
                f"expected {G1_UPPER_BODY_STATE_SIZE}"
            )
            return
        if values[0] != 1.0:
            self._set_error("g1_upper_body_state is marked invalid")
            return
        if values[1] > G1_UPPER_BODY_MAX_AGE_SECONDS * 1000.0:
            self._set_error(f"g1_upper_body_state is stale ({values[1]:.1f} ms)")
            return
        try:
            self._collector.update(
                "body",
                values[G1_UPPER_BODY_TAU_SLICE],
                G1_NUM_UPPER_BODY_MOTORS,
            )
            self.error_reason = None
        except Exception as exc:
            self._set_error(f"invalid upper-body tau_est: {exc}")

    def _set_error(self, message: str) -> None:
        self.error_reason = message
        now = time.monotonic()
        if now - self._last_warning_time >= 1.0:
            self._last_warning_time = now
            print(f"[G1UpperBodyTau] Warning: {message}")

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()


class TauHistory:
    """Bounded timestamped history used by the rolling tau plot."""

    def __init__(self, window_seconds: float):
        if window_seconds <= 0:
            raise ValueError("tau_window_seconds must be positive")
        self.window_seconds = float(window_seconds)
        self._timestamps: deque[float] = deque()
        self._values: deque[np.ndarray] = deque()

    def append(self, timestamp: float, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        expected_size = len(TAU_JOINT_NAMES)
        if values.size != expected_size:
            raise ValueError(
                f"Tau sample has {values.size} values; expected {expected_size}"
            )
        self._timestamps.append(float(timestamp))
        self._values.append(values.copy())
        self._prune(float(timestamp))

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
            self._values.popleft()

    def snapshot(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        self._prune(now)
        if not self._timestamps:
            return (
                np.empty(0, dtype=np.float64),
                np.empty((0, len(TAU_JOINT_NAMES)), dtype=np.float32),
            )
        return np.asarray(self._timestamps), np.stack(self._values)

    @property
    def latest_timestamp(self) -> Optional[float]:
        return self._timestamps[-1] if self._timestamps else None


def _get_screen_size() -> Optional[tuple[int, int]]:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None
    return width, height


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:04.1f}"


def _display_image(
    name: str,
    image: np.ndarray,
    depth_min_meters: float,
    depth_max_meters: float,
) -> np.ndarray:
    """Convert RGB or metric depth into a BGR image suitable for display/video."""
    if image.ndim == 2 or name.endswith("_depth"):
        if depth_max_meters <= depth_min_meters:
            raise ValueError("depth_display_max_meters must exceed depth_display_min_meters")
        depth = np.asarray(image, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.0)
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        normalized[valid] = np.clip(
            (depth[valid] - depth_min_meters)
            * 255.0
            / (depth_max_meters - depth_min_meters),
            0.0,
            255.0,
        ).astype(np.uint8)
        display = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        display[~valid] = 0
        cv2.putText(
            display,
            f"{depth_min_meters:.1f}-{depth_max_meters:.1f} m",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return display
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _compose_grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    """Place camera tiles in a fixed grid without stretching their aspect ratio."""
    if not tiles:
        raise ValueError("Cannot compose an empty camera grid")
    if columns <= 0:
        raise ValueError("grid_columns must be positive")

    cell_h = max(tile.shape[0] for tile in tiles)
    cell_w = max(tile.shape[1] for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    canvas = np.zeros((rows * cell_h, columns * cell_w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        y = row * cell_h + (cell_h - tile.shape[0]) // 2
        x = column * cell_w + (cell_w - tile.shape[1]) // 2
        canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return canvas


def _compose_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Place two image columns side by side, vertically centering the shorter one."""
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    left_y = (height - left.shape[0]) // 2
    right_y = (height - right.shape[0]) // 2
    canvas[left_y : left_y + left.shape[0], : left.shape[1]] = left
    canvas[
        right_y : right_y + right.shape[0],
        left.shape[1] :,
    ] = right
    return canvas


def _load_tau_presets(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Load and validate the ordered tau preset mapping."""
    preset_path = Path(path).expanduser()
    try:
        raw_presets = json.loads(preset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Tau preset file does not exist: {preset_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid tau preset JSON in {preset_path}: {exc}") from exc

    if not isinstance(raw_presets, dict) or not raw_presets:
        raise ValueError("Tau preset JSON must be a non-empty object")

    presets: dict[str, tuple[str, ...]] = {}
    for preset_name, joint_names in raw_presets.items():
        if not isinstance(preset_name, str) or not preset_name.strip():
            raise ValueError("Every tau preset must have a non-empty string name")
        if not isinstance(joint_names, list) or not joint_names:
            raise ValueError(f"Tau preset '{preset_name}' must be a non-empty array")
        if not all(isinstance(name, str) for name in joint_names):
            raise ValueError(f"Tau preset '{preset_name}' contains a non-string joint")
        if len(joint_names) != len(set(joint_names)):
            raise ValueError(f"Tau preset '{preset_name}' contains duplicate joints")
        unknown = [name for name in joint_names if name not in TAU_JOINT_INDEX]
        if unknown:
            raise ValueError(
                f"Tau preset '{preset_name}' contains unknown joints: {unknown}"
            )
        presets[preset_name] = tuple(joint_names)
    return presets


def _tau_source_for_joints(joint_names: Sequence[str]) -> str:
    if all(name.startswith("left_brainco_") for name in joint_names):
        return "left"
    if all(name.startswith("right_brainco_") for name in joint_names):
        return "right"
    return "body"


def _render_tau_plot(
    history: TauHistory,
    preset_name: str,
    joint_names: Sequence[str],
    width: int = 640,
    height: int = 480,
    unavailable_reason: Optional[str] = None,
) -> np.ndarray:
    """Render one named subset of the full 29-channel ``tau_est`` history."""
    selected_indices = np.asarray(
        [TAU_JOINT_INDEX[name] for name in joint_names], dtype=np.int64
    )
    width = max(520, int(width))
    height = max(400, int(height))
    canvas = np.full((height, width, 3), (20, 22, 27), dtype=np.uint8)
    now = time.monotonic()
    timestamps, all_values = history.snapshot(now)
    latest = (
        all_values[-1]
        if all_values.shape[0]
        else np.full(len(TAU_JOINT_NAMES), np.nan, dtype=np.float32)
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        f"tau_est | {preset_name} | {history.window_seconds:g} s | A/D: switch",
        (12, 20),
        font,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    selected_complete_rows = (
        np.isfinite(all_values[:, selected_indices]).all(axis=1)
        if timestamps.size
        else np.empty(0, dtype=bool)
    )
    if unavailable_reason:
        status_text = f"unavailable: {unavailable_reason}"
        status_color = (80, 100, 255)
    elif not selected_complete_rows.any():
        status_text = "waiting for selected tau source"
        status_color = (0, 210, 255)
    else:
        selected_timestamp = timestamps[np.flatnonzero(selected_complete_rows)[-1]]
        age = max(0.0, now - selected_timestamp)
        if age > TAU_STALE_SECONDS:
            status_text = f"stale {age:.1f} s"
            status_color = (0, 210, 255)
        else:
            status_text = "live"
            status_color = (80, 230, 100)
    (status_w, _), _ = cv2.getTextSize(status_text, font, 0.45, 1)
    cv2.putText(
        canvas,
        status_text,
        (max(12, width - status_w - 12), 20),
        font,
        0.45,
        status_color,
        1,
        cv2.LINE_AA,
    )

    plot_left = 55
    legend_width = 245
    plot_right = max(plot_left + 100, width - legend_width)
    panels_top = 29
    panel_height = height - panels_top
    window_start = now - history.window_seconds

    for panel_index, panel_label in enumerate((preset_name,)):
        values = all_values[:, selected_indices]
        panel_top = panels_top + panel_index * panel_height
        panel_bottom = min(height - 1, panel_top + panel_height)
        graph_top = panel_top + 23
        graph_bottom = panel_bottom - 24

        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            y_min = min(0.0, float(np.min(finite_values)))
            y_max = max(0.0, float(np.max(finite_values)))
            y_span = y_max - y_min
            if y_span < 1e-4:
                padding = max(0.1, abs(y_max) * 0.2)
            else:
                padding = max(0.05, y_span * 0.1)
            y_min -= padding
            y_max += padding
        else:
            y_min, y_max = -1.0, 1.0

        cv2.rectangle(
            canvas,
            (plot_left, graph_top),
            (plot_right, graph_bottom),
            (42, 46, 54),
            -1,
        )
        cv2.putText(
            canvas,
            panel_label,
            (12, panel_top + 17),
            font,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        for grid_index in range(5):
            fraction = grid_index / 4.0
            x = int(round(plot_left + fraction * (plot_right - plot_left)))
            y = int(round(graph_top + fraction * (graph_bottom - graph_top)))
            cv2.line(canvas, (x, graph_top), (x, graph_bottom), (65, 69, 78), 1)
            cv2.line(canvas, (plot_left, y), (plot_right, y), (65, 69, 78), 1)

        zero_fraction = (y_max - 0.0) / (y_max - y_min)
        if 0.0 <= zero_fraction <= 1.0:
            zero_y = int(round(graph_top + zero_fraction * (graph_bottom - graph_top)))
            cv2.line(
                canvas,
                (plot_left, zero_y),
                (plot_right, zero_y),
                (120, 125, 135),
                1,
            )

        cv2.putText(
            canvas,
            f"{y_max:.2f}",
            (4, graph_top + 5),
            font,
            0.34,
            (175, 175, 175),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{y_min:.2f}",
            (4, graph_bottom),
            font,
            0.34,
            (175, 175, 175),
            1,
            cv2.LINE_AA,
        )

        if timestamps.size:
            x_coordinates = plot_left + (
                (timestamps - window_start)
                / history.window_seconds
                * (plot_right - plot_left)
            )
            for motor_index, _ in enumerate(joint_names):
                color = TAU_PLOT_COLORS[motor_index % len(TAU_PLOT_COLORS)]
                channel = values[:, motor_index]
                y_coordinates = graph_top + (
                    (y_max - channel)
                    / (y_max - y_min)
                    * (graph_bottom - graph_top)
                )
                current_segment = []
                for x_value, y_value, tau_value in zip(
                    x_coordinates, y_coordinates, channel, strict=True
                ):
                    if np.isfinite(tau_value):
                        current_segment.append(
                            [int(round(x_value)), int(round(y_value))]
                        )
                    elif current_segment:
                        if len(current_segment) >= 2:
                            cv2.polylines(
                                canvas,
                                [np.asarray(current_segment, dtype=np.int32)],
                                False,
                                color,
                                1,
                                cv2.LINE_AA,
                            )
                        current_segment = []
                if len(current_segment) >= 2:
                    cv2.polylines(
                        canvas,
                        [np.asarray(current_segment, dtype=np.int32)],
                        False,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        time_labels = (
            (plot_left, f"-{history.window_seconds:g}s"),
            ((plot_left + plot_right) // 2, f"-{history.window_seconds / 2:g}s"),
            (plot_right, "now"),
        )
        for x, label in time_labels:
            (label_w, _), _ = cv2.getTextSize(label, font, 0.34, 1)
            cv2.putText(
                canvas,
                label,
                (max(plot_left, min(plot_right - label_w, x - label_w // 2)), panel_bottom - 5),
                font,
                0.34,
                (175, 175, 175),
                1,
                cv2.LINE_AA,
            )

        legend_x = plot_right + 10
        legend_y = graph_top + 9
        legend_step = max(16, min(28, (graph_bottom - graph_top - 8) // len(joint_names)))
        for motor_index, motor_name in enumerate(joint_names):
            color = TAU_PLOT_COLORS[motor_index % len(TAU_PLOT_COLORS)]
            y = legend_y + motor_index * legend_step
            cv2.line(canvas, (legend_x, y), (legend_x + 12, y), color, 2)
            value = latest[selected_indices[motor_index]]
            value_text = "--" if not np.isfinite(value) else f"{value:.3f}"
            label = motor_name.removesuffix("_joint").replace("brainco_", "")
            cv2.putText(
                canvas,
                f"{label} {value_text}",
                (legend_x + 17, y + 4),
                font,
                0.34,
                color,
                1,
                cv2.LINE_AA,
            )

    return canvas


def _fit_to_window(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Scale to fit inside the window and letterbox the unused area."""
    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        return image
    scale = min(target_w / image.shape[1], target_h / image.shape[0])
    resized_w = max(1, int(round(image.shape[1] * scale)))
    resized_h = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - resized_w) // 2
    y = (target_h - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def _episode_status_text(
    status: Optional[dict],
    endpoint: str = "",
) -> tuple[str, tuple[int, int, int]]:
    if status is None:
        suffix = f" | {endpoint}" if endpoint else ""
        return f"Episode: waiting for exporter{suffix}", (0, 220, 255)

    state = str(status.get("state", "unknown"))
    state_label = {
        "idle": "idle",
        "recording": "recording",
        "need_to_save": "saving",
    }.get(state, state)

    duration_sec = float(status.get("duration_sec", 0.0))
    received_at = status.get("_received_monotonic")
    status_age = None
    if received_at is not None:
        status_age = max(0.0, time.monotonic() - float(received_at))
    if state == "recording" and status_age is not None and status_age <= 2.0:
        duration_sec += status_age

    episode_index = status.get("episode_index", "?")
    frame_count = int(status.get("frame_count", 0))
    text = (
        f"Episode {episode_index} | {state_label} | "
        f"{_format_duration(duration_sec)} | {frame_count}f"
    )

    color = (0, 255, 0) if state == "recording" else (230, 230, 230)
    if status_age is not None and status_age > 2.0:
        text += f" | stale {status_age:.1f}s"
        color = (0, 220, 255)
    return text, color


def _draw_text_overlay(
    canvas: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    position: str = "bottom_left",
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75 * 5.0
    thickness = 2 * 5
    margin = 12 * 5
    padding_x = 8 * 5
    padding_y = 8 * 5

    def wrap_lines(scale: float, line_thickness: int, max_width: int) -> list[str]:
        parts = text.split(" | ")
        lines = []
        current = parts[0]
        for part in parts[1:]:
            candidate = f"{current} | {part}"
            (candidate_w, _), _ = cv2.getTextSize(candidate, font, scale, line_thickness)
            if candidate_w <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = part
        lines.append(current)
        return lines

    for _ in range(12):
        max_text_width = max(1, canvas.shape[1] - 2 * (margin + padding_x))
        lines = wrap_lines(font_scale, thickness, max_text_width)
        sizes = [cv2.getTextSize(line, font, font_scale, thickness) for line in lines]
        text_w = max(size[0][0] for size in sizes)
        text_h = sum(size[0][1] for size in sizes)
        baseline = max(size[1] for size in sizes)
        line_gap = max(8, int(8 * 5 * font_scale / (0.75 * 5.0)))
        box_w = text_w + 2 * padding_x
        box_h = text_h + baseline + line_gap * (len(lines) - 1) + 2 * padding_y
        if (
            box_w <= canvas.shape[1] - 2 * margin
            and box_h <= canvas.shape[0] - 2 * margin
        ) or font_scale <= 0.75:
            break
        font_scale *= 0.9
        thickness = max(2, int(round(thickness * 0.9)))
        margin = max(12, int(round(margin * 0.9)))
        padding_x = max(8, int(round(padding_x * 0.9)))
        padding_y = max(8, int(round(padding_y * 0.9)))

    x1 = max(0, margin)
    x2 = min(canvas.shape[1] - 1, x1 + box_w)
    if position == "bottom_left":
        y2 = min(canvas.shape[0] - 1, canvas.shape[0] - margin)
        y1 = max(0, y2 - box_h)
    else:
        y1 = max(0, margin)
        y2 = min(canvas.shape[0] - 1, y1 + box_h)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    y = y1 + padding_y
    for idx, line in enumerate(lines):
        line_h = sizes[idx][0][1]
        y += line_h
        cv2.putText(
            canvas,
            line,
            (x1 + padding_x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_gap


def main(config: CameraViewerConfig):
    _validate_config(config)
    tau_presets: dict[str, tuple[str, ...]] = {}
    tau_preset_names: list[str] = []
    tau_preset_index = 0
    if config.show_tau_plot:
        tau_presets = _load_tau_presets(config.tau_presets_path)
        tau_preset_names = list(tau_presets)
        if config.tau_preset is not None:
            if config.tau_preset not in tau_presets:
                raise ValueError(
                    f"Unknown tau preset '{config.tau_preset}'. Available presets: "
                    f"{', '.join(tau_preset_names)}"
                )
            tau_preset_index = tau_preset_names.index(config.tau_preset)

    client = ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port)
    episode_status_host = config.episode_status_zmq_host or config.camera_host
    episode_status_subscriber = (
        EpisodeStatusSubscriber(
            host=episode_status_host,
            port=config.episode_status_zmq_port,
        )
        if config.show_episode_status
        else None
    )
    latest_episode_status = None
    last_logged_episode_status_key = None

    print("Waiting for first camera frame...")
    sample = None
    for _ in range(100):
        sample = client.read(blocking=False)
        if sample and sample.get("images"):
            break
        time.sleep(0.1)

    if sample is None or not sample.get("images"):
        print("ERROR: No camera frames received after 10s. Check the camera server.")
        client.close()
        if episode_status_subscriber is not None:
            episode_status_subscriber.close()
        return

    external_view_client = None
    external_view_sample = None
    if config.external_view_camera_host is not None:
        external_view_client = ComposedCameraClientSensor(
            server_ip=config.external_view_camera_host,
            port=config.external_view_camera_port,
        )
        print(
            "Waiting for external-view-camera at "
            f"tcp://{config.external_view_camera_host}:"
            f"{config.external_view_camera_port}..."
        )
        for _ in range(100):
            external_view_sample = external_view_client.read(blocking=False)
            if (
                external_view_sample
                and EXTERNAL_VIEW_STREAM_NAME
                in external_view_sample.get("images", {})
            ):
                break
            time.sleep(0.1)
        if (
            external_view_sample is None
            or EXTERNAL_VIEW_STREAM_NAME
            not in external_view_sample.get("images", {})
        ):
            print(
                "WARNING: No external-view-camera frames received after 10s; "
                "continuing to wait for the standalone publisher"
            )

    available_camera_names = list(sample["images"].keys())
    if external_view_client is not None:
        available_camera_names.append(EXTERNAL_VIEW_STREAM_NAME)
    camera_names = [
        name for name in config.camera_streams if name in available_camera_names
    ]
    if (
        external_view_client is not None
        and EXTERNAL_VIEW_STREAM_NAME not in camera_names
    ):
        camera_names.append(EXTERNAL_VIEW_STREAM_NAME)
    selected_depth_stream = f"{config.depth_camera}_depth"
    display_camera_names = [
        name
        for name in camera_names
        if not name.endswith("_depth")
        or (config.show_depth and name == selected_depth_stream)
    ]
    missing_camera_names = [
        name for name in config.camera_streams if name not in available_camera_names
    ]
    print(f"Available camera streams: {available_camera_names}")
    print(f"Recording camera streams: {camera_names}")
    print(f"Displaying camera streams: {display_camera_names}")
    if config.show_depth and selected_depth_stream not in available_camera_names:
        print(
            f"WARNING: Selected depth stream '{selected_depth_stream}' is unavailable; "
            "only RGB streams will be displayed"
        )
    if missing_camera_names:
        print(f"WARNING: Requested camera streams are missing: {missing_camera_names}")
    if not camera_names:
        print("ERROR: None of the requested camera streams are available.")
        client.close()
        if external_view_client is not None:
            external_view_client.close()
        if episode_status_subscriber is not None:
            episode_status_subscriber.close()
        return

    tau_history = TauHistory(config.tau_window_seconds) if config.show_tau_plot else None
    tau_collector = TauCollector() if config.show_tau_plot else None
    brainco_tau_subscriber = None
    g1_upper_body_tau_subscriber = None
    tau_source_errors: dict[str, str] = {}
    sonic_zmq_host = config.sonic_zmq_host or config.camera_host
    if config.show_tau_plot:
        try:
            g1_upper_body_tau_subscriber = G1UpperBodyTauSubscriber(
                collector=tau_collector,
                host=sonic_zmq_host,
                port=config.sonic_zmq_port,
            )
        except Exception as exc:
            tau_source_errors["body"] = str(exc)
            print(f"[G1UpperBodyTau] Warning: body tau is unavailable: {exc}")
        try:
            brainco_tau_subscriber = BraincoTauSubscriber(
                collector=tau_collector,
                domain_id=config.brainco_dds_domain_id,
                network_interface=config.brainco_network_interface,
            )
        except Exception as exc:
            tau_source_errors["left"] = str(exc)
            tau_source_errors["right"] = str(exc)
            print(f"[BrainCoTau] Warning: hand tau is unavailable: {exc}")

    output_dir = Path(config.output_path) if config.output_path else Path("camera_recordings")

    is_recording = False
    video_writers: dict[str, cv2.VideoWriter] = {}
    frame_count = 0
    recording_start_time = 0.0
    recording_dir = Path(".")
    loop_period = 1.0 / config.fps

    window_name = "SONIC Camera Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )
    display_size = _get_screen_size()

    print(f"Target FPS: {config.fps}")
    if display_size is not None:
        print(f"Fullscreen display size: {display_size[0]}x{display_size[1]}")
    print(f"Recordings will be saved to: {output_dir}")
    if config.show_episode_status:
        print(
            "Episode status overlay: "
            f"tcp://{episode_status_host}:{config.episode_status_zmq_port}"
        )
    if config.show_tau_plot:
        selected_preset = tau_preset_names[tau_preset_index]
        print(
            f"Tau plot: {config.tau_window_seconds:g}s window, preset "
            f"'{selected_preset}', {len(TAU_JOINT_NAMES)} history channels, "
            f"upper-body ZMQ tcp://{sonic_zmq_host}:{config.sonic_zmq_port}, "
            f"hand DDS domain {config.brainco_dds_domain_id}, "
            f"interface {config.brainco_network_interface or 'auto'}"
        )
    else:
        print("Motor-state/tau display: disabled")
    if external_view_client is not None:
        print(
            "External view: "
            f"tcp://{config.external_view_camera_host}:"
            f"{config.external_view_camera_port}"
        )
    if config.show_depth:
        print(f"Depth display: {selected_depth_stream}")
    else:
        print("Depth display: disabled (--no-depth)")
    print("Controls: A/D = previous/next tau preset, R = record, Q = quit")

    try:
        while True:
            t_start = time.monotonic()

            if episode_status_subscriber is not None:
                status = episode_status_subscriber.read()
                if status is not None:
                    latest_episode_status = status
                    status_key = (
                        status.get("episode_index"),
                        status.get("state"),
                        status.get("is_recording"),
                    )
                    if status_key != last_logged_episode_status_key:
                        last_logged_episode_status_key = status_key
                        print(
                            "[EpisodeStatus] received "
                            f"episode={status.get('episode_index')} "
                            f"state={status.get('state')} "
                            f"duration={float(status.get('duration_sec', 0.0)):.1f}s "
                            f"frames={status.get('frame_count')}"
                        )

            if g1_upper_body_tau_subscriber is not None:
                g1_upper_body_tau_subscriber.poll()
            if tau_collector is not None and tau_history is not None:
                tau_sample = tau_collector.read()
                if tau_sample is not None:
                    tau_history.append(*tau_sample)

            robot_image_data = client.read(blocking=False)
            external_image_data = (
                external_view_client.read(blocking=False)
                if external_view_client is not None
                else None
            )
            combined_images = {}
            if robot_image_data is not None:
                combined_images.update(robot_image_data.get("images", {}))
            if external_image_data is not None:
                external_frame = external_image_data.get("images", {}).get(
                    EXTERNAL_VIEW_STREAM_NAME
                )
                if external_frame is not None:
                    combined_images[EXTERNAL_VIEW_STREAM_NAME] = external_frame
            if not combined_images:
                elapsed = time.monotonic() - t_start
                remaining = loop_period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                continue
            image_data = {"images": combined_images}

            camera_tiles = []
            for name in camera_names:
                img = image_data["images"].get(name)
                if img is None:
                    continue

                img_bgr = _display_image(
                    name,
                    img,
                    config.depth_display_min_meters,
                    config.depth_display_max_meters,
                )

                if is_recording and name in video_writers:
                    video_writers[name].write(img_bgr)

                if name not in display_camera_names:
                    continue

                h, w = img_bgr.shape[:2]
                if config.max_display_width > 0 and w > config.max_display_width:
                    scale = config.max_display_width / w
                    img_bgr = cv2.resize(
                        img_bgr, (config.max_display_width, int(h * scale))
                    )

                label = f"{name}"
                if is_recording:
                    label = f"[REC] {name}"
                cv2.putText(
                    img_bgr, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                camera_tiles.append(img_bgr)

            canvas = None
            if config.show_tau_plot and tau_history is not None:
                selected_preset = tau_preset_names[tau_preset_index]
                selected_joints = tau_presets[selected_preset]
                selected_source = _tau_source_for_joints(selected_joints)
                unavailable_reason = tau_source_errors.get(selected_source)
                if (
                    selected_source == "body"
                    and g1_upper_body_tau_subscriber is not None
                    and g1_upper_body_tau_subscriber.error_reason is not None
                ):
                    unavailable_reason = g1_upper_body_tau_subscriber.error_reason
                camera_column = (
                    _compose_grid(camera_tiles, columns=1) if camera_tiles else None
                )
                plot_width = max(
                    640, camera_column.shape[1] if camera_column is not None else 0
                )
                plot_height = max(
                    480, camera_column.shape[0] if camera_column is not None else 0
                )
                tau_plot = _render_tau_plot(
                    tau_history,
                    preset_name=selected_preset,
                    joint_names=selected_joints,
                    width=plot_width,
                    height=plot_height,
                    unavailable_reason=unavailable_reason,
                )
                canvas = (
                    _compose_columns(camera_column, tau_plot)
                    if camera_column is not None
                    else tau_plot
                )
            elif camera_tiles:
                canvas = _compose_grid(camera_tiles, config.grid_columns)

            if canvas is not None:

                if is_recording:
                    frame_count += 1
                    elapsed_rec = time.time() - recording_start_time
                    status = f"REC {frame_count}f / {elapsed_rec:.1f}s"
                    cv2.putText(
                        canvas, status, (canvas.shape[1] - 300, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2,
                    )

                target_size = display_size
                if target_size is None:
                    _, _, window_w, window_h = cv2.getWindowImageRect(window_name)
                    if window_w > 0 and window_h > 0:
                        target_size = (window_w, window_h)

                if target_size is not None:
                    canvas = _fit_to_window(canvas, target_size)

                if config.show_episode_status:
                    status_endpoint = (
                        episode_status_subscriber.endpoint
                        if episode_status_subscriber is not None
                        else ""
                    )
                    status_text, status_color = _episode_status_text(
                        latest_episode_status,
                        endpoint=status_endpoint,
                    )
                    _draw_text_overlay(canvas, status_text, status_color)

                cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quit requested.")
                break
            elif key in (ord("a"), ord("A")) and config.show_tau_plot:
                tau_preset_index = (tau_preset_index - 1) % len(tau_preset_names)
                print(f"Tau preset: {tau_preset_names[tau_preset_index]}")
            elif key in (ord("d"), ord("D")) and config.show_tau_plot:
                tau_preset_index = (tau_preset_index + 1) % len(tau_preset_names)
                print(f"Tau preset: {tau_preset_names[tau_preset_index]}")
            elif key == ord("r"):
                if not is_recording:
                    recording_dir = output_dir / f"rec_{time.strftime('%Y%m%d_%H%M%S')}"
                    recording_dir.mkdir(parents=True, exist_ok=True)

                    fourcc = cv2.VideoWriter_fourcc(*config.codec)
                    video_writers = {}
                    for name in camera_names:
                        img = image_data["images"].get(name)
                        if img is not None:
                            h, w = img.shape[:2]
                            path = recording_dir / f"{name}.mp4"
                            video_writers[name] = cv2.VideoWriter(
                                str(path), fourcc, config.fps, (w, h)
                            )

                    is_recording = True
                    recording_start_time = time.time()
                    frame_count = 0
                    print(f"Recording started: {recording_dir}")
                else:
                    is_recording = False
                    for writer in video_writers.values():
                        writer.release()
                    video_writers = {}
                    duration = time.time() - recording_start_time
                    print(
                        f"Recording stopped - {duration:.1f}s, {frame_count} frames "
                        f"-> {recording_dir}"
                    )

            elapsed = time.monotonic() - t_start
            remaining = loop_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if video_writers:
            for writer in video_writers.values():
                writer.release()
            if is_recording:
                duration = time.time() - recording_start_time
                print(f"Final recording: {duration:.1f}s, {frame_count} frames")

        client.close()
        if external_view_client is not None:
            external_view_client.close()
        if episode_status_subscriber is not None:
            episode_status_subscriber.close()
        if brainco_tau_subscriber is not None:
            brainco_tau_subscriber.close()
        if g1_upper_body_tau_subscriber is not None:
            g1_upper_body_tau_subscriber.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = parse_args()
    main(config)
