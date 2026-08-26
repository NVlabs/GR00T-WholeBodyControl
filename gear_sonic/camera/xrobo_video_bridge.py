"""Display TeleImager cameras in the XRoboToolkit PICO app.

The bridge implements only XRoboToolkit's Remote Vision control and H.264
transport.  It does not read or publish robot state, poses, or motor commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import socket
import struct
import threading
import time
from typing import Callable

import numpy as np


_MAX_CONTROL_BODY_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class CameraRequest:
    width: int
    height: int
    fps: int
    bitrate: int
    enable_mv_hevc: int
    render_mode: int
    port: int
    camera: str
    ip: str


def serialize_control_message(command: str, data: bytes = b"") -> bytes:
    """Serialize an XRoboToolkit length-framed control message."""

    command_bytes = command.encode("utf-8")
    body = (
        struct.pack("<I", len(command_bytes))
        + command_bytes
        + struct.pack("<I", len(data))
        + data
    )
    return struct.pack(">I", len(body)) + body


def deserialize_control_body(body: bytes) -> tuple[str, bytes]:
    """Deserialize the body after the outer big-endian length prefix."""

    if len(body) < 8:
        raise ValueError("XRoboToolkit control body is too short")
    command_len = struct.unpack_from("<I", body, 0)[0]
    data_len_offset = 4 + command_len
    if data_len_offset + 4 > len(body):
        raise ValueError("XRoboToolkit command length exceeds the control body")
    command = body[4:data_len_offset].decode("utf-8")
    data_len = struct.unpack_from("<I", body, data_len_offset)[0]
    data_offset = data_len_offset + 4
    if data_offset + data_len != len(body):
        raise ValueError("XRoboToolkit data length does not match the control body")
    return command, body[data_offset:]


def deserialize_camera_request(data: bytes) -> CameraRequest:
    """Parse XRoboToolkit CameraRequestSerializer protocol version 1."""

    fixed_size = 3 + 7 * 4
    if len(data) < fixed_size + 2:
        raise ValueError("XRoboToolkit camera request is too short")
    if data[:2] != b"\xca\xfe":
        raise ValueError("XRoboToolkit camera request has invalid magic bytes")
    if data[2] != 1:
        raise ValueError(f"Unsupported XRoboToolkit camera protocol: {data[2]}")

    values = struct.unpack_from("<7i", data, 3)
    offset = fixed_size

    def read_string() -> str:
        nonlocal offset
        if offset >= len(data):
            raise ValueError("Missing camera request string length")
        length = data[offset]
        offset += 1
        if offset + length > len(data):
            raise ValueError("Camera request string exceeds payload")
        value = data[offset : offset + length].decode("utf-8")
        offset += length
        return value

    camera = read_string()
    ip = read_string()
    if offset != len(data):
        raise ValueError("Unexpected trailing bytes in camera request")

    return CameraRequest(
        width=values[0],
        height=values[1],
        fps=values[2],
        bitrate=values[3],
        enable_mv_hevc=values[4],
        render_mode=values[5],
        port=values[6],
        camera=camera,
        ip=ip,
    )


def prepare_head_frame(
    bgr: np.ndarray,
    width: int,
    height: int,
    *,
    binocular: bool,
) -> np.ndarray:
    """Prepare a full side-by-side frame matching the PICO video profile."""

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected a BGR HxWx3 image, got {bgr.shape}")

    # A mono UVC source is duplicated rather than stretched when the requested
    # output is side-by-side.  TeleImager's standard binocular UVC source is
    # already side-by-side and passes through this branch unchanged.
    source = bgr
    source_ratio = bgr.shape[1] / bgr.shape[0]
    target_ratio = width / height
    if not binocular and target_ratio >= source_ratio * 1.8:
        source = np.concatenate((bgr, bgr), axis=1)

    if source.shape[1] != width or source.shape[0] != height:
        import cv2

        source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(source)


def compose_head_and_wrist_frame(
    head_bgr: np.ndarray,
    left_wrist_bgr: np.ndarray | None,
    right_wrist_bgr: np.ndarray | None,
    width: int,
    height: int,
    *,
    binocular: bool,
) -> np.ndarray:
    """Compose the head view above two wrist panels in each stereo eye.

    The binocular head image retains its aspect ratio, while the left and
    right wrist images share the panel below it. Wrist panels are duplicated
    into both eyes so they are readable without introducing artificial
    disparity.
    """

    if head_bgr.ndim != 3 or head_bgr.shape[2] != 3:
        raise ValueError(f"Expected a BGR HxWx3 head image, got {head_bgr.shape}")
    if width % 2 != 0:
        raise ValueError("XRoboToolkit side-by-side output width must be even")

    import cv2

    if binocular:
        if head_bgr.shape[1] % 2 != 0:
            raise ValueError("Binocular head image width must be even")
        source_eye_width = head_bgr.shape[1] // 2
        head_eyes = (
            head_bgr[:, :source_eye_width],
            head_bgr[:, source_eye_width:],
        )
    else:
        # The XRoboToolkit profile is side-by-side.  Preserve a mono camera's
        # aspect ratio by giving each eye the same image.
        head_eyes = (head_bgr, head_bgr)

    wrists = [
        frame
        for frame in (left_wrist_bgr, right_wrist_bgr)
        if frame is not None
    ]
    if not wrists:
        return prepare_head_frame(
            head_bgr,
            width,
            height,
            binocular=binocular,
        )
    for wrist in wrists:
        if wrist.ndim != 3 or wrist.shape[2] != 3:
            raise ValueError(f"Expected a BGR HxWx3 wrist image, got {wrist.shape}")

    eye_width = width // 2
    head_aspect = head_eyes[0].shape[1] / head_eyes[0].shape[0]
    aspect_sum = sum(wrist.shape[1] / wrist.shape[0] for wrist in wrists)

    # The source layout is one head image above a row of wrist images. Fit the
    # complete layout inside each eye's canvas without changing any camera's
    # aspect ratio. XRoboToolkit's stereo surface itself is fixed at 4:3, so a
    # tall head+wrist column needs pillar-boxing rather than UV compression.
    layout_aspect = 1.0 / (1.0 / head_aspect + 1.0 / aspect_sum)
    layout_width = max(1, min(eye_width, int(height * layout_aspect)))
    head_height = max(1, round(layout_width / head_aspect))
    wrist_height = max(1, round(layout_width / aspect_sum))
    while head_height + wrist_height > height and layout_width > 1:
        layout_width -= 1
        head_height = max(1, round(layout_width / head_aspect))
        wrist_height = max(1, round(layout_width / aspect_sum))

    wrist_widths = [
        max(1, round(wrist.shape[1] * wrist_height / wrist.shape[0]))
        for wrist in wrists
    ]
    while sum(wrist_widths) > layout_width and max(wrist_widths) > 1:
        widest = max(range(len(wrist_widths)), key=wrist_widths.__getitem__)
        wrist_widths[widest] -= 1

    output = np.zeros((height, width, 3), dtype=head_bgr.dtype)
    layout_x = (eye_width - layout_width) // 2
    layout_y = (height - head_height - wrist_height) // 2
    resized_wrists = [
        cv2.resize(wrist, (wrist_width, wrist_height), interpolation=cv2.INTER_AREA)
        for wrist, wrist_width in zip(wrists, wrist_widths)
    ]
    for eye_index, head_eye in enumerate(head_eyes):
        eye_x = eye_index * eye_width
        resized_head = cv2.resize(
            head_eye,
            (layout_width, head_height),
            interpolation=cv2.INTER_AREA,
        )
        output[
            layout_y : layout_y + head_height,
            eye_x + layout_x : eye_x + layout_x + layout_width,
        ] = resized_head

        x = eye_x + layout_x + (layout_width - sum(wrist_widths)) // 2
        panel_y = layout_y + head_height
        for wrist in resized_wrists:
            output[panel_y : panel_y + wrist_height, x : x + wrist.shape[1]] = wrist
            x += wrist.shape[1]
    return np.ascontiguousarray(output)


class H264Encoder:
    """Small low-latency PyAV encoder yielding Annex-B access units."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        encoder: str = "auto",
    ):
        import av

        candidates = ("h264_nvenc", "libx264") if encoder == "auto" else (encoder,)
        errors: list[str] = []
        self._codec = None
        self.name = ""
        for candidate in candidates:
            try:
                codec = av.CodecContext.create(candidate, "w")
                codec.width = width
                codec.height = height
                codec.pix_fmt = "yuv420p"
                codec.time_base = Fraction(1, fps)
                codec.framerate = Fraction(fps, 1)
                codec.bit_rate = bitrate
                if candidate == "h264_nvenc":
                    codec.options = {
                        "preset": "p1",
                        "tune": "ull",
                        "delay": "0",
                        "zerolatency": "1",
                        "g": str(fps),
                        "forced-idr": "1",
                    }
                else:
                    codec.options = {
                        "preset": "ultrafast",
                        "tune": "zerolatency",
                        "profile": "baseline",
                        "x264-params": (
                            f"keyint={fps}:min-keyint={fps}:scenecut=0:"
                            "repeat-headers=1"
                        ),
                    }
                codec.open()
                self._codec = codec
                self.name = candidate
                break
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
        if self._codec is None:
            raise RuntimeError("Could not initialize H.264 encoder: " + "; ".join(errors))
        self._pts = 0

    def encode(self, bgr: np.ndarray) -> list[bytes]:
        import av

        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = self._pts
        frame.time_base = self._codec.time_base
        self._pts += 1
        return [bytes(packet) for packet in self._codec.encode(frame)]

    def flush(self) -> list[bytes]:
        return [bytes(packet) for packet in self._codec.encode(None)]


class _VideoStream:
    def __init__(
        self,
        request: CameraRequest,
        target_ip: str,
        image_client,
        binocular: bool,
        encoder_name: str,
        wrist_camera_names: tuple[str, ...] = (),
    ):
        self.request = request
        self.target_ip = target_ip
        self.image_client = image_client
        self.binocular = binocular
        self.encoder_name = encoder_name
        self.wrist_camera_names = wrist_camera_names
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="xrobo-video")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=2.0)

    def _run(self):
        request = self.request
        try:
            encoder = H264Encoder(
                request.width,
                request.height,
                request.fps,
                request.bitrate,
                self.encoder_name,
            )
            print(
                f"[Video] Encoder={encoder.name}, "
                f"{request.width}x{request.height}@{request.fps}, "
                f"bitrate={request.bitrate}"
            )
            with socket.create_connection(
                (self.target_ip, request.port), timeout=3.0
            ) as stream_socket:
                stream_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[Video] Connected to PICO {self.target_ip}:{request.port}")
                period = 1.0 / request.fps
                next_frame_time = time.monotonic()
                sent_frames = 0
                started_at = next_frame_time
                while not self.stop_event.is_set():
                    now = time.monotonic()
                    if now < next_frame_time:
                        self.stop_event.wait(next_frame_time - now)
                        continue
                    next_frame_time = max(next_frame_time + period, now)

                    tele_image = self.image_client.get_camera_frame("head_camera")
                    bgr = None if tele_image is None else tele_image.bgr
                    if bgr is None:
                        self.stop_event.wait(0.01)
                        continue
                    if self.wrist_camera_names:
                        wrist_frames: dict[str, np.ndarray | None] = {}
                        for camera_name in self.wrist_camera_names:
                            wrist_image = self.image_client.get_camera_frame(camera_name)
                            wrist_frames[camera_name] = (
                                None if wrist_image is None else wrist_image.bgr
                            )
                        frame = compose_head_and_wrist_frame(
                            bgr,
                            wrist_frames.get("left_wrist_camera"),
                            wrist_frames.get("right_wrist_camera"),
                            request.width,
                            request.height,
                            binocular=self.binocular,
                        )
                    else:
                        frame = prepare_head_frame(
                            bgr,
                            request.width,
                            request.height,
                            binocular=self.binocular,
                        )
                    for packet in encoder.encode(frame):
                        stream_socket.sendall(struct.pack(">I", len(packet)) + packet)
                    sent_frames += 1
                    if sent_frames % max(request.fps * 10, 1) == 0:
                        elapsed = max(time.monotonic() - started_at, 1e-6)
                        print(f"[Video] Streaming {sent_frames / elapsed:.1f} fps")
                for packet in encoder.flush():
                    stream_socket.sendall(struct.pack(">I", len(packet)) + packet)
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"[Video ERROR] {exc}")
        finally:
            print("[Video] Stream stopped")


class XRoboTeleImagerBridge:
    """One-way camera bridge for XRoboToolkit Remote Vision."""

    def __init__(
        self,
        camera_host: str,
        camera_port: int,
        listen_host: str,
        listen_port: int,
        encoder: str = "auto",
        show_wrist_cameras: bool = False,
        image_client_factory: Callable | None = None,
    ):
        if image_client_factory is None:
            try:
                from teleimager import ImageClient
            except ImportError:
                try:
                    from teleimager.image_client import ImageClient
                except ImportError as exc:
                    raise ImportError(
                        "TeleImager is not installed. Activate .venv_data_collection "
                        "where the official teleimager package is installed."
                    ) from exc
            image_client_factory = ImageClient

        self.camera_host = camera_host
        self.camera_port = camera_port
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.encoder = encoder
        self.show_wrist_cameras = show_wrist_cameras
        self.stop_event = threading.Event()
        self._stream: _VideoStream | None = None
        self._send_lock = threading.Lock()
        self._control_socket: socket.socket | None = None
        self._audio_request_id = ""

        bgr_camera_names = {"head_camera"}
        if show_wrist_cameras:
            bgr_camera_names.update(("left_wrist_camera", "right_wrist_camera"))
        self.image_client = image_client_factory(
            host=camera_host,
            request_port=camera_port,
            request_bgr=False,
            bgr_camera_names=bgr_camera_names,
        )
        config = self.image_client.get_cam_config()
        head_config = config.get("head_camera", {})
        if not head_config.get("enable_zmq", False):
            raise RuntimeError("TeleImager head_camera must have enable_zmq: true")
        self.binocular = bool(head_config.get("binocular", False))
        self.wrist_camera_names: tuple[str, ...] = ()
        if show_wrist_cameras:
            wrist_names = ("left_wrist_camera", "right_wrist_camera")
            disabled = [
                name for name in wrist_names if not config.get(name, {}).get("enable_zmq", False)
            ]
            if disabled:
                raise RuntimeError(
                    "TeleImager wrist display requires enable_zmq: true for "
                    + ", ".join(disabled)
                )
            self.wrist_camera_names = wrist_names

    def close(self):
        self.stop_event.set()
        self._stop_stream()
        if self._control_socket is not None:
            try:
                self._control_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._control_socket.close()
            self._control_socket = None
        self.image_client.close()

    def serve_forever(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.listen_host, self.listen_port))
            server.listen(1)
            server.settimeout(0.5)
            print(
                f"[Bridge] TeleImager {self.camera_host}:{self.camera_port} -> "
                f"XRoboToolkit control {self.listen_host}:{self.listen_port}"
            )
            print("[Bridge] View-only bridge; no robot-control topics are used")
            if self.wrist_camera_names:
                print("[Bridge] XR layout: head view + left/right wrist panels")
            while not self.stop_event.is_set():
                try:
                    control_socket, address = server.accept()
                except socket.timeout:
                    continue
                print(f"[Control] XRoboToolkit connected from {address[0]}:{address[1]}")
                self._serve_control_client(control_socket, address[0])

    def _serve_control_client(self, control_socket: socket.socket, peer_ip: str):
        self._control_socket = control_socket
        self._audio_request_id = ""
        control_socket.settimeout(0.5)
        control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(control_socket, heartbeat_stop),
            daemon=True,
            name="xrobo-control-heartbeat",
        )
        heartbeat.start()
        try:
            with control_socket:
                while not self.stop_event.is_set():
                    try:
                        length_bytes = self._recv_exact(control_socket, 4)
                    except socket.timeout:
                        continue
                    if length_bytes is None:
                        break
                    body_len = struct.unpack(">I", length_bytes)[0]
                    if body_len <= 0 or body_len > _MAX_CONTROL_BODY_BYTES:
                        raise ValueError(f"Invalid control message length: {body_len}")
                    body = self._recv_exact(control_socket, body_len)
                    if body is None:
                        break
                    command, data = deserialize_control_body(body)
                    self._handle_command(command, data, peer_ip)
        except (OSError, ValueError) as exc:
            if not self.stop_event.is_set():
                print(f"[Control] Connection ended: {exc}")
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=1.0)
            self._stop_stream()
            self._control_socket = None
            print("[Control] XRoboToolkit disconnected")

    def _handle_command(self, command: str, data: bytes, peer_ip: str):
        if command == "AUDIO_SESSION":
            request_id = data.decode("ascii", errors="ignore").strip()
            self._audio_request_id = request_id
            self._send_audio_disabled_config()
        elif command == "OPEN_CAMERA":
            request = deserialize_camera_request(data)
            self._validate_request(request)
            if request.ip and request.ip != peer_ip:
                print(
                    f"[Control] Ignoring requested target {request.ip}; "
                    f"using connected PICO address {peer_ip}"
                )
            print(
                f"[Control] OPEN_CAMERA {request.width}x{request.height}@{request.fps} "
                f"camera={request.camera!r}, target={peer_ip}:{request.port}"
            )
            self._stop_stream()
            self._stream = _VideoStream(
                request,
                peer_ip,
                self.image_client,
                self.binocular,
                self.encoder,
                self.wrist_camera_names,
            )
            self._stream.start()
        elif command == "CLOSE_CAMERA":
            print("[Control] CLOSE_CAMERA")
            self._stop_stream()
        elif command in ("PONG", "PING"):
            if command == "PING":
                self._send_control("PONG", data)
        else:
            print(f"[Control] Ignoring unsupported command {command!r}")

    def _validate_request(self, request: CameraRequest):
        # Deliberately bound this display-only bridge to the modest profile
        # shipped with SONIC.  Selecting XRoboToolkit's built-in 2160x810 or
        # 2560x720 @ 60 fps profiles should fail instead of unexpectedly
        # consuming substantial CPU/GPU while real-robot control is active.
        if not (160 <= request.width <= 1920 and 120 <= request.height <= 1080):
            raise ValueError(
                f"Unsupported requested resolution {request.width}x{request.height}; "
                "select TELEIMAGER_HEAD in XRoboToolkit"
            )
        if not 1 <= request.fps <= 30:
            raise ValueError(
                f"Unsupported requested frame rate {request.fps}; "
                "select TELEIMAGER_HEAD in XRoboToolkit"
            )
        if not 100_000 <= request.bitrate <= 8_000_000:
            raise ValueError(
                f"Unsupported requested bitrate {request.bitrate}; "
                "select TELEIMAGER_HEAD in XRoboToolkit"
            )
        if not 1 <= request.port <= 65535:
            raise ValueError(f"Invalid requested stream port {request.port}")
        if self.wrist_camera_names:
            eye_aspect = request.width / (2.0 * request.height)
            if abs(eye_aspect - 4.0 / 3.0) > 0.02:
                raise ValueError(
                    f"Requested profile {request.width}x{request.height} has per-eye "
                    f"aspect {eye_aspect:.3f}; select TELEIMAGER_HEAD_WRISTS "
                    "(4:3 per eye) in XRoboToolkit"
                )

    def _send_audio_disabled_config(self):
        request_id = self._audio_request_id
        if not (16 <= len(request_id) <= 128):
            return
        payload = json.dumps(
            {
                "schema": "g1_wuji_audio_ports_v2",
                "audio_request_id": request_id,
                "audio_stream_port": 0,
                "microphone_upload_port": 0,
                "sample_rate": 16000,
                "channels": 1,
                "sample_format": "s16le",
                "video_projection": "flat",
                "video_stereo_layout": "side_by_side" if self.binocular else "mono",
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_control("AUDIO_CONFIG", payload)

    def _heartbeat_loop(self, control_socket: socket.socket, stop_event: threading.Event):
        # XRoboToolkit 1.1.1 closes an idle control connection after five
        # seconds.  PING is part of its control protocol and has no robot-side
        # effect.
        while not stop_event.wait(2.0) and not self.stop_event.is_set():
            if self._control_socket is not control_socket:
                return
            try:
                self._send_control("PING", struct.pack("<d", time.monotonic()))
            except OSError:
                return

    def _send_control(self, command: str, data: bytes = b""):
        control_socket = self._control_socket
        if control_socket is None:
            return
        packet = serialize_control_message(command, data)
        with self._send_lock:
            control_socket.sendall(packet)

    def _stop_stream(self):
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()

    def _recv_exact(self, sock: socket.socket, size: int) -> bytes | None:
        data = bytearray()
        while len(data) < size and not self.stop_event.is_set():
            try:
                chunk = sock.recv(size - len(data))
            except socket.timeout:
                if not data:
                    raise
                continue
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)
