"""Read-only dashboard helpers for XR + GEAR-SONIC teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from gear_sonic.utils.teleop.xr_upperbody_bridge import (
    BridgeConfig,
    UpperBodyFilter,
    normalize_live_source_payload,
    unpack_bridge_message,
    xr_g1_29_arm_to_sonic_upper_body,
)

XR_STALE_S = 0.5
BRIDGE_STALE_S = 0.5
CAMERA_STALE_S = 1.0
READ_ONLY_SOCKET_TYPES = {"SUB"}

DASHBOARD_HTML = b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XR GEAR-SONIC Teleop</title>
  <link rel="stylesheet" href="/static/dashboard.css">
</head>
<body>
  <header class="top">
    <div><strong>XR</strong><span id="xr-status">waiting</span></div>
    <div><strong>Bridge</strong><span id="bridge-status">waiting</span></div>
    <div><strong>Camera</strong><span id="camera-status">waiting</span></div>
    <div><strong>Mode</strong><span id="mode">-</span></div>
    <div><strong>Collection</strong><span id="collection">unknown inferred</span></div>
  </header>
  <main class="layout">
    <section class="vision"><h2>Robot Vision</h2><div id="camera-grid"></div></section>
    <section><h2>XR Command</h2><pre id="xr-command"></pre></section>
    <section><h2>Bridge Transform</h2><pre id="bridge-transform"></pre></section>
    <section><h2>Safety</h2><pre id="safety"></pre></section>
  </main>
  <script src="/static/dashboard.js"></script>
</body>
</html>"""

DASHBOARD_CSS = b"""body{margin:0;font-family:Inter,Arial,sans-serif;background:#101419;color:#e8edf2}.top{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;padding:10px 12px;background:#18212b;position:sticky;top:0;z-index:2}.top div,section{border:1px solid #2d3a47;background:#151c24;border-radius:6px;padding:10px}.top span{display:block;margin-top:4px;color:#9dd1ff}.layout{display:grid;grid-template-columns:2fr 1fr 1fr;gap:10px;padding:10px}.vision{grid-row:span 2}h2{font-size:15px;margin:0 0 8px;color:#c7d4e1}pre{white-space:pre-wrap;font-size:12px;line-height:1.35;margin:0}.cam{margin-bottom:8px}.cam img{width:100%;background:#05070a;border-radius:4px;aspect-ratio:16/9;object-fit:contain}.warn{color:#ffcf70}@media(max-width:900px){.top{grid-template-columns:1fr 1fr}.layout{grid-template-columns:1fr}}"""

DASHBOARD_JS = b"""async function fetchJson(url){const r=await fetch(url,{cache:'no-store'});return await r.json();}
function fmt(v){return JSON.stringify(v,null,2);}
function text(id,value){document.getElementById(id).textContent=value;}
async function tick(){
  try{
    const s=await fetchJson('/api/state');
    text('xr-status',`${s.xr.status} ${s.xr.age_s ?? '-'}s`);
    text('bridge-status',`${s.bridge.planner_status} ${s.bridge.planner_age_s ?? '-'}s`);
    text('camera-status',s.cameras.status);
    text('mode',s.xr.mode ?? '-');
    text('collection',`${s.collection.state} (inferred)`);
    text('xr-command',fmt({movement:s.xr.movement,facing:s.xr.facing,speed:s.xr.speed,height:s.xr.height,hands:s.hands,edge_state:s.edge_state}));
    text('bridge-transform',fmt(s.bridge_transform));
    text('safety',fmt({errors:s.errors,notice:'XR stop is source lifecycle. Bridge command.stop is planner stop intent.'}));
    const grid=document.getElementById('camera-grid');
    const names=Object.keys(s.cameras.items);
    grid.innerHTML=names.length?names.map(n=>`<div class="cam"><h2>${n}</h2><img src="/api/camera/${encodeURIComponent(n)}?t=${Date.now()}"><pre>${fmt(s.cameras.items[n])}</pre></div>`).join(''):'<pre>No camera frames</pre>';
  }catch(e){text('safety',String(e));}
}
setInterval(tick,200); tick();"""


def summarize_values(values: Any) -> dict[str, float | int | None]:
    if values is None:
        return {"count": 0, "min": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(array.size),
        "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6),
        "mean": round(float(np.mean(array)), 6),
    }


def assert_read_only_socket_type(source_name: str, socket_type: str) -> None:
    if socket_type not in READ_ONLY_SOCKET_TYPES:
        raise ValueError(
            f"{source_name} dashboard socket must be read-only SUB, got {socket_type}"
        )


def _list3(values: tuple[float, float, float] | np.ndarray | list[float]) -> list[float]:
    return [round(float(v), 6) for v in list(values)[:3]]


def _status(last_receive_time: float | None, now: float, stale_s: float) -> str:
    if last_receive_time is None:
        return "waiting"
    return "stale" if now - last_receive_time > stale_s else "live"


def _age(last_receive_time: float | None, now: float) -> float | None:
    if last_receive_time is None:
        return None
    return round(max(0.0, now - last_receive_time), 3)


def estimate_hand_aperture(joints: np.ndarray | None) -> dict[str, Any]:
    summary = summarize_values(joints)
    if joints is None:
        return {"estimate": None, "summary": summary}
    array = np.asarray(joints, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return {"estimate": None, "summary": summary}
    return {
        "estimate": round(float(np.mean(np.abs(array))), 6),
        "summary": summary,
        "label": "aperture_estimate",
    }


@dataclass
class CameraFrame:
    content: bytes
    content_type: str
    receive_time: float
    source_shape: tuple[int, ...] | None = None


@dataclass
class DashboardState:
    bridge_config: BridgeConfig = field(default_factory=BridgeConfig)
    lock: threading.RLock = field(default_factory=threading.RLock)
    filter: UpperBodyFilter = field(init=False)
    latest_xr_payload: dict[str, Any] | None = None
    latest_xr_frame: Any = None
    latest_planner: dict[str, np.ndarray] | None = None
    latest_manager_state: dict[str, np.ndarray] | None = None
    latest_command: dict[str, np.ndarray] | None = None
    latest_camera_frames: dict[str, CameraFrame] = field(default_factory=dict)
    xr_receive_time: float | None = None
    planner_receive_time: float | None = None
    manager_receive_time: float | None = None
    command_receive_time: float | None = None
    camera_receive_time: float | None = None
    xr_count: int = 0
    planner_count: int = 0
    manager_count: int = 0
    command_count: int = 0
    camera_count: int = 0
    error_counts: dict[str, int] = field(default_factory=dict)
    collection_state: str = "unknown"
    last_collection_toggle_time: float | None = None
    last_abort_toggle_time: float | None = None

    def __post_init__(self) -> None:
        self.filter = UpperBodyFilter(self.bridge_config)

    def _inc_error(self, key: str) -> None:
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

    def update_xr_payload(self, payload: dict[str, Any], receive_time: float | None = None) -> None:
        receive_time = time.monotonic() if receive_time is None else receive_time
        try:
            frame = normalize_live_source_payload(payload)
            raw_dual_arm = np.asarray(payload.get("dual_arm_position"), dtype=np.float32)
            mapped = xr_g1_29_arm_to_sonic_upper_body(raw_dual_arm)
            filtered = self.filter.apply(mapped)
        except Exception:
            with self.lock:
                self._inc_error("xr_payload")
            return

        with self.lock:
            self.latest_xr_payload = dict(payload)
            self.latest_xr_frame = frame
            self.latest_xr_payload["_raw_dual_arm_summary"] = summarize_values(raw_dual_arm)
            self.latest_xr_payload["_mapped_upper_summary"] = summarize_values(mapped)
            self.latest_xr_payload["_filtered_upper_summary"] = summarize_values(filtered)
            self.xr_receive_time = receive_time
            self.xr_count += 1
            if frame.toggle_data_collection:
                self.collection_state = (
                    "recording_inferred"
                    if self.collection_state != "recording_inferred"
                    else "idle_inferred"
                )
                self.last_collection_toggle_time = receive_time
            if frame.toggle_data_abort:
                self.last_abort_toggle_time = receive_time

    def update_bridge_message(self, raw: bytes, receive_time: float | None = None) -> None:
        receive_time = time.monotonic() if receive_time is None else receive_time
        try:
            if raw.startswith(b"planner"):
                decoded = unpack_bridge_message(raw, topic="planner")
                with self.lock:
                    self.latest_planner = decoded
                    self.planner_receive_time = receive_time
                    self.planner_count += 1
            elif raw.startswith(b"manager_state"):
                decoded = unpack_bridge_message(raw, topic="manager_state")
                with self.lock:
                    self.latest_manager_state = decoded
                    self.manager_receive_time = receive_time
                    self.manager_count += 1
            elif raw.startswith(b"command"):
                decoded = unpack_bridge_message(raw, topic="command")
                with self.lock:
                    self.latest_command = decoded
                    self.command_receive_time = receive_time
                    self.command_count += 1
            else:
                with self.lock:
                    self._inc_error("bridge_unknown_topic")
        except Exception:
            with self.lock:
                self._inc_error("bridge_message")

    def update_camera_frame(
        self,
        name: str,
        content: bytes,
        content_type: str = "image/jpeg",
        receive_time: float | None = None,
        source_shape: tuple[int, ...] | None = None,
    ) -> None:
        receive_time = time.monotonic() if receive_time is None else receive_time
        with self.lock:
            self.latest_camera_frames[name] = CameraFrame(
                content=content,
                content_type=content_type,
                receive_time=receive_time,
                source_shape=source_shape,
            )
            self.camera_receive_time = receive_time
            self.camera_count += 1

    def get_camera_frame(self, name: str) -> CameraFrame | None:
        with self.lock:
            return self.latest_camera_frames.get(name)

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        with self.lock:
            frame = self.latest_xr_frame
            payload = self.latest_xr_payload or {}
            planner = self.latest_planner or {}
            command = self.latest_command
            manager = self.latest_manager_state
            left_hand = None if frame is None else frame.left_hand_joints
            right_hand = None if frame is None else frame.right_hand_joints
            seconds_since_edge = (
                None
                if self.last_collection_toggle_time is None
                else round(max(0.0, now - self.last_collection_toggle_time), 3)
            )
            cameras = {
                name: {
                    "content_type": item.content_type,
                    "age_s": _age(item.receive_time, now),
                    "status": _status(item.receive_time, now, CAMERA_STALE_S),
                    "source_shape": list(item.source_shape) if item.source_shape else None,
                }
                for name, item in sorted(self.latest_camera_frames.items())
            }
            return {
                "time": now,
                "xr": {
                    "status": _status(self.xr_receive_time, now, XR_STALE_S),
                    "age_s": _age(self.xr_receive_time, now),
                    "count": self.xr_count,
                    "mode": None if frame is None else frame.mode,
                    "movement": None if frame is None else _list3(frame.movement),
                    "facing": None if frame is None else _list3(frame.facing),
                    "speed": None if frame is None else round(float(frame.speed), 6),
                    "height": None if frame is None else round(float(frame.height), 6),
                    "stop": bool(payload.get("stop", False)) if payload else None,
                },
                "bridge": {
                    "planner_status": _status(self.planner_receive_time, now, BRIDGE_STALE_S),
                    "planner_age_s": _age(self.planner_receive_time, now),
                    "planner_count": self.planner_count,
                    "manager_count": self.manager_count,
                    "command_count": self.command_count,
                    "latest_command": None
                    if command is None
                    else {k: v.astype(float).reshape(-1).tolist() for k, v in command.items()},
                    "latest_manager_state": None
                    if manager is None
                    else {k: v.reshape(-1).tolist() for k, v in manager.items()},
                    "planner_mode": None
                    if "mode" not in planner
                    else int(planner["mode"].reshape(-1)[0]),
                },
                "edge_state": {
                    "command": "not_yet_received" if self.command_receive_time is None else "received",
                    "collection_toggle": "not_yet_received"
                    if self.last_collection_toggle_time is None
                    else "received",
                },
                "collection": {
                    "inferred": True,
                    "state": self.collection_state,
                    "last_collection_toggle_age_s": _age(self.last_collection_toggle_time, now),
                    "last_abort_toggle_age_s": _age(self.last_abort_toggle_time, now),
                    "seconds_since_last_record_edge": seconds_since_edge,
                },
                "bridge_transform": {
                    "raw_dual_arm": payload.get("_raw_dual_arm_summary", summarize_values(None)),
                    "mapped_upper_body": payload.get("_mapped_upper_summary", summarize_values(None)),
                    "filtered_upper_body": payload.get(
                        "_filtered_upper_summary", summarize_values(None)
                    ),
                    "planner_upper_body": summarize_values(planner.get("upper_body_position")),
                },
                "hands": {
                    "left": estimate_hand_aperture(left_hand),
                    "right": estimate_hand_aperture(right_hand),
                },
                "cameras": {
                    "status": _status(self.camera_receive_time, now, CAMERA_STALE_S),
                    "count": self.camera_count,
                    "items": cameras,
                },
                "errors": dict(self.error_counts),
            }


class DashboardHttpHandler(BaseHTTPRequestHandler):
    state: DashboardState

    def _send_bytes(self, status: int, content_type: str, content: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send_bytes(
            status,
            "application/json",
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", DASHBOARD_HTML)
            return
        if path == "/static/dashboard.css":
            self._send_bytes(HTTPStatus.OK, "text/css; charset=utf-8", DASHBOARD_CSS)
            return
        if path == "/static/dashboard.js":
            self._send_bytes(HTTPStatus.OK, "application/javascript; charset=utf-8", DASHBOARD_JS)
            return
        if path == "/api/state":
            self._send_json(HTTPStatus.OK, self.state.snapshot())
            return
        if path == "/api/cameras":
            snapshot = self.state.snapshot()
            self._send_json(HTTPStatus.OK, snapshot["cameras"])
            return
        if path.startswith("/api/camera/"):
            name = unquote(path.removeprefix("/api/camera/"))
            frame = self.state.get_camera_frame(name)
            if frame is None:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "camera_frame_unavailable", "camera": name},
                )
                return
            self._send_bytes(HTTPStatus.OK, frame.content_type, frame.content)
            return
        self._send_bytes(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found")


def encode_color_image(image: np.ndarray) -> tuple[bytes, str] | None:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        return None
    import cv2

    ok, buffer = cv2.imencode(".jpg", array[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return buffer.tobytes(), "image/jpeg"


def run_xr_receiver(state: DashboardState, host: str, port: int, topic: str, stop_event) -> None:
    import zmq

    assert_read_only_socket_type("xr", "SUB")
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.setsockopt(zmq.RCVTIMEO, 100)
    socket.connect(f"tcp://{host}:{port}")
    topic_bytes = topic.encode("utf-8")
    try:
        while not stop_event.is_set():
            try:
                raw = socket.recv()
            except zmq.Again:
                continue
            payload_bytes = raw[len(topic_bytes) :] if raw.startswith(topic_bytes) else raw
            try:
                payload = json.loads(payload_bytes.decode("utf-8"))
            except Exception:
                with state.lock:
                    state._inc_error("xr_json")
                continue
            state.update_xr_payload(payload)
    finally:
        socket.close(linger=0)


def run_bridge_receiver(state: DashboardState, host: str, port: int, stop_event) -> None:
    import zmq

    assert_read_only_socket_type("bridge", "SUB")
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    for topic in ("planner", "manager_state", "command"):
        socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.setsockopt(zmq.RCVTIMEO, 100)
    socket.connect(f"tcp://{host}:{port}")
    try:
        while not stop_event.is_set():
            try:
                raw = socket.recv()
            except zmq.Again:
                continue
            state.update_bridge_message(raw)
    finally:
        socket.close(linger=0)


def run_camera_receiver(state: DashboardState, host: str, port: int, stop_event) -> None:
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

    client = None
    try:
        client = ComposedCameraClientSensor(server_ip=host, port=port)
        while not stop_event.is_set():
            sample = client.read(blocking=False)
            if not sample or not sample.get("images"):
                time.sleep(0.05)
                continue
            receive_time = time.monotonic()
            for name, image in sample["images"].items():
                encoded = encode_color_image(image)
                if encoded is None:
                    with state.lock:
                        state._inc_error("camera_non_color")
                    continue
                content, content_type = encoded
                state.update_camera_frame(
                    name,
                    content,
                    content_type,
                    receive_time=receive_time,
                    source_shape=tuple(np.asarray(image).shape),
                )
            time.sleep(0.02)
    except Exception:
        with state.lock:
            state._inc_error("camera_receiver")
    finally:
        if client is not None:
            client.close()


def make_handler(state: DashboardState):
    class Handler(DashboardHttpHandler):
        pass

    Handler.state = state
    return Handler


def serve_dashboard(state: DashboardState, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    return server
