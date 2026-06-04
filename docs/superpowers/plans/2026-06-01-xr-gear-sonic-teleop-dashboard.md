# XR + GEAR-SONIC Teleop Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only browser dashboard for XR controller + GEAR-SONIC teleoperation state, bridge transforms, camera frames, and collection-intent telemetry.

**Architecture:** Add a Python-only dashboard runtime that subscribes to XR JSON, bridge ZMQ messages, and the existing composed camera server. A `ThreadingHTTPServer` serves static HTML/CSS/JS plus JSON/image endpoints; all command/control sockets are read-only subscribers. The dashboard reuses `normalize_live_source_payload()`, `UpperBodyFrame`, and `unpack_bridge_message()` so it cannot drift from bridge semantics.

**Tech Stack:** Python standard library HTTP server, `threading`, `time.monotonic`, `json`, `zmq`, `numpy`, existing `gear_sonic.camera.composed_camera.ComposedCameraClientSensor`, existing `gear_sonic.utils.teleop.xr_upperbody_bridge`.

---

## File Structure

- Create `gear_sonic/utils/teleop/xr_dashboard.py`
  - Owns dashboard state, source receivers, bridge/XR normalization, camera image caching, HTTP handler, and static assets.
  - Must not create outbound control sockets. Only `zmq.SUB` is allowed for ZMQ inputs.
- Create `gear_sonic/scripts/xr_teleop_dashboard.py`
  - CLI entry point with defaults matching the spec: XR `5560`, bridge `5556`, camera `5555`, HTTP `8088`.
- Create `gear_sonic/tests/test_xr_teleop_dashboard.py`
  - Tests non-browser logic: normalization, edge-state unknown, inferred collection latch, camera 503 behavior, and read-only ZMQ socket invariant.

Do not commit unless the user explicitly asks. The repository instruction overrides the generic "frequent commits" habit.

---

### Task 1: Dashboard State Core

**Files:**
- Create: `gear_sonic/utils/teleop/xr_dashboard.py`
- Test: `gear_sonic/tests/test_xr_teleop_dashboard.py`

- [ ] **Step 1: Write failing tests for state defaults, XR normalization, edge unknown, and collection latch**

Add the initial test file:

```python
from __future__ import annotations

import time

import numpy as np

from gear_sonic.utils.teleop.xr_dashboard import DashboardState, summarize_values


def sample_xr_payload(**overrides):
    payload = {
        "timestamp": 1.0,
        "dual_arm_position": [0.1 * i for i in range(14)],
        "dual_hand_joints": [0.01 * i for i in range(14)],
        "mode": 2,
        "movement": [0.25, -0.5, 0.0],
        "facing": [0.9, 0.1, 0.0],
        "speed": 0.4,
        "height": 0.2,
        "toggle_data_collection": False,
        "toggle_data_abort": False,
        "stop": False,
    }
    payload.update(overrides)
    return payload


def test_initial_edge_state_is_unknown():
    state = DashboardState()
    snapshot = state.snapshot()
    assert snapshot["edge_state"]["command"] == "not_yet_received"
    assert snapshot["collection"]["inferred"] is True
    assert snapshot["collection"]["state"] == "unknown"


def test_xr_payload_uses_bridge_normalization_and_summarizes_transforms():
    state = DashboardState()
    state.update_xr_payload(sample_xr_payload(), receive_time=10.0)
    snapshot = state.snapshot(now=10.1)

    assert snapshot["xr"]["status"] == "live"
    assert snapshot["xr"]["mode"] == 2
    assert snapshot["xr"]["movement"] == [0.25, -0.5, 0.0]
    assert snapshot["xr"]["stop"] is False
    assert snapshot["bridge_transform"]["raw_dual_arm"]["count"] == 14
    assert snapshot["bridge_transform"]["mapped_upper_body"]["count"] == 17
    assert snapshot["hands"]["left"]["count"] == 7
    assert snapshot["hands"]["right"]["count"] == 7


def test_collection_latch_is_inferred_from_toggle_edges():
    state = DashboardState()
    state.update_xr_payload(sample_xr_payload(toggle_data_collection=True), receive_time=20.0)
    first = state.snapshot(now=21.0)
    assert first["collection"]["state"] == "recording_inferred"
    assert first["collection"]["seconds_since_last_record_edge"] == 1.0

    state.update_xr_payload(sample_xr_payload(toggle_data_collection=True), receive_time=22.0)
    second = state.snapshot(now=22.5)
    assert second["collection"]["state"] == "idle_inferred"
    assert second["collection"]["seconds_since_last_record_edge"] == 0.5


def test_summarize_values_handles_absent_and_numeric_values():
    assert summarize_values(None)["count"] == 0
    summary = summarize_values(np.array([1.0, -2.0, 3.0], dtype=np.float32))
    assert summary == {"count": 3, "min": -2.0, "max": 3.0, "mean": 0.666667}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 pytest gear_sonic/tests/test_xr_teleop_dashboard.py -q
```

Expected: import failure for `gear_sonic.utils.teleop.xr_dashboard`.

- [ ] **Step 3: Implement dashboard state core**

Create `gear_sonic/utils/teleop/xr_dashboard.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import threading
import time
from typing import Any

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
```

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 pytest gear_sonic/tests/test_xr_teleop_dashboard.py -q
```

Expected: all Task 1 tests pass.

---

### Task 2: Bridge Decode, Camera Cache, and Read-Only Invariant

**Files:**
- Modify: `gear_sonic/utils/teleop/xr_dashboard.py`
- Test: `gear_sonic/tests/test_xr_teleop_dashboard.py`

- [ ] **Step 1: Add failing tests for bridge decode, camera 503, and socket kind helper**

Append tests:

```python
import pytest

from gear_sonic.utils.teleop.xr_dashboard import DashboardHttpHandler, assert_read_only_socket_type
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message


def test_bridge_planner_and_command_decode():
    state = DashboardState()
    state.update_bridge_message(
        build_planner_message(
            mode=4,
            movement=[1.0, 0.0, 0.0],
            facing=[0.0, 1.0, 0.0],
            height=0.3,
            upper_body_position=[0.1] * 17,
        ),
        receive_time=30.0,
    )
    state.update_bridge_message(
        build_command_message(start=True, stop=False, planner=True),
        receive_time=31.0,
    )
    snapshot = state.snapshot(now=31.1)
    assert snapshot["bridge"]["planner_mode"] == 4
    assert snapshot["bridge_transform"]["planner_upper_body"]["count"] == 17
    assert snapshot["edge_state"]["command"] == "received"
    assert snapshot["bridge"]["latest_command"]["start"] == [1.0]
    assert snapshot["bridge"]["latest_command"]["stop"] == [0.0]


def test_camera_frame_lookup_absent_returns_none():
    state = DashboardState()
    assert state.get_camera_frame("ego_view") is None
    state.update_camera_frame("ego_view", b"jpeg-bytes", "image/jpeg", receive_time=40.0)
    frame = state.get_camera_frame("ego_view")
    assert frame is not None
    assert frame.content == b"jpeg-bytes"
    assert frame.content_type == "image/jpeg"


def test_read_only_socket_type_rejects_control_socket_types():
    assert_read_only_socket_type("xr", "SUB")
    with pytest.raises(ValueError, match="read-only"):
        assert_read_only_socket_type("bridge", "PUB")
    with pytest.raises(ValueError, match="read-only"):
        assert_read_only_socket_type("bridge", "REQ")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 pytest gear_sonic/tests/test_xr_teleop_dashboard.py -q
```

Expected: import failure for `DashboardHttpHandler` or `assert_read_only_socket_type`.

- [ ] **Step 3: Implement read-only helper and HTTP handler skeleton**

Append to `gear_sonic/utils/teleop/xr_dashboard.py`:

```python
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse


READ_ONLY_SOCKET_TYPES = {"SUB"}


def assert_read_only_socket_type(source_name: str, socket_type: str) -> None:
    if socket_type not in READ_ONLY_SOCKET_TYPES:
        raise ValueError(
            f"{source_name} dashboard socket must be read-only SUB, got {socket_type}"
        )


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
```

- [ ] **Step 4: Run tests to verify Task 2 passes**

Run:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 pytest gear_sonic/tests/test_xr_teleop_dashboard.py -q
```

Expected: all Task 1-2 tests pass.

---

### Task 3: Receiver Threads and Camera Encoding

**Files:**
- Modify: `gear_sonic/utils/teleop/xr_dashboard.py`
- Test: `gear_sonic/tests/test_xr_teleop_dashboard.py`

- [ ] **Step 1: Add tests for color image encoding**

Append tests:

```python
from gear_sonic.utils.teleop.xr_dashboard import encode_color_image


def test_encode_color_image_rejects_non_color():
    assert encode_color_image(np.zeros((4, 4), dtype=np.uint16)) is None


def test_encode_color_image_returns_jpeg_bytes_for_rgb():
    encoded = encode_color_image(np.zeros((4, 4, 3), dtype=np.uint8))
    assert encoded is not None
    content, content_type = encoded
    assert content.startswith(b"\xff\xd8")
    assert content_type == "image/jpeg"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 pytest gear_sonic/tests/test_xr_teleop_dashboard.py -q
```

Expected: import failure for `encode_color_image`.

- [ ] **Step 3: Implement receiver helpers and JPEG cache encoding**

Append to `gear_sonic/utils/teleop/xr_dashboard.py`:

```python
def encode_color_image(image: np.ndarray) -> tuple[bytes, str] | None:
    import cv2

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        return None
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
```

- [ ] **Step 4: Run tests to verify Task 3 passes**

Run:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 pytest gear_sonic/tests/test_xr_teleop_dashboard.py -q
```

Expected: all Task 1-3 tests pass.

---

### Task 4: Static Dashboard UI and Server Launcher

**Files:**
- Modify: `gear_sonic/utils/teleop/xr_dashboard.py`
- Create: `gear_sonic/scripts/xr_teleop_dashboard.py`

- [ ] **Step 1: Add HTML/CSS/JS and server factory**

Extend `DashboardHttpHandler.do_GET()` to serve `/`, `/static/dashboard.css`, and `/static/dashboard.js`. Add compact static assets to `xr_dashboard.py`:

```python
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

DASHBOARD_CSS = b"""body{margin:0;font-family:Inter,Arial,sans-serif;background:#101419;color:#e8edf2} .top{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;padding:10px 12px;background:#18212b;position:sticky;top:0;z-index:2}.top div,section{border:1px solid #2d3a47;background:#151c24;border-radius:6px;padding:10px}.top span{display:block;margin-top:4px;color:#9dd1ff}.layout{display:grid;grid-template-columns:2fr 1fr 1fr;gap:10px;padding:10px}.vision{grid-row:span 2}h2{font-size:15px;margin:0 0 8px;color:#c7d4e1}pre{white-space:pre-wrap;font-size:12px;line-height:1.35;margin:0}.cam{margin-bottom:8px}.cam img{width:100%;background:#05070a;border-radius:4px;aspect-ratio:16/9;object-fit:contain}.warn{color:#ffcf70}@media(max-width:900px){.top{grid-template-columns:1fr 1fr}.layout{grid-template-columns:1fr}}"""

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


def make_handler(state: DashboardState):
    class Handler(DashboardHttpHandler):
        pass

    Handler.state = state
    return Handler


def serve_dashboard(state: DashboardState, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    return server
```

Update `do_GET()` with these branches before API handling:

```python
if path == "/":
    self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", DASHBOARD_HTML)
    return
if path == "/static/dashboard.css":
    self._send_bytes(HTTPStatus.OK, "text/css; charset=utf-8", DASHBOARD_CSS)
    return
if path == "/static/dashboard.js":
    self._send_bytes(HTTPStatus.OK, "application/javascript; charset=utf-8", DASHBOARD_JS)
    return
```

- [ ] **Step 2: Add CLI launcher**

Create `gear_sonic/scripts/xr_teleop_dashboard.py`:

```python
from __future__ import annotations

import argparse
import threading

from gear_sonic.utils.teleop.xr_dashboard import (
    DashboardState,
    run_bridge_receiver,
    run_camera_receiver,
    run_xr_receiver,
    serve_dashboard,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only XR + GEAR-SONIC teleop dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--xr-source-host", default="localhost")
    parser.add_argument("--xr-source-port", type=int, default=5560)
    parser.add_argument("--xr-source-topic", default="xr_teleop")
    parser.add_argument("--bridge-host", default="localhost")
    parser.add_argument("--bridge-port", type=int, default=5556)
    parser.add_argument("--camera-host", default="localhost")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--no-camera", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    state = DashboardState()
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=run_xr_receiver,
            args=(state, args.xr_source_host, args.xr_source_port, args.xr_source_topic, stop_event),
            daemon=True,
        ),
        threading.Thread(
            target=run_bridge_receiver,
            args=(state, args.bridge_host, args.bridge_port, stop_event),
            daemon=True,
        ),
    ]
    if not args.no_camera:
        threads.append(
            threading.Thread(
                target=run_camera_receiver,
                args=(state, args.camera_host, args.camera_port, stop_event),
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()
    server = serve_dashboard(state, args.host, args.port)
    print(f"XR teleop dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run syntax checks**

Run:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python -B -m py_compile gear_sonic/utils/teleop/xr_dashboard.py gear_sonic/scripts/xr_teleop_dashboard.py
```

Expected: no output and exit code 0.

---

### Task 5: Final Verification

**Files:**
- Verify: `gear_sonic/utils/teleop/xr_dashboard.py`
- Verify: `gear_sonic/scripts/xr_teleop_dashboard.py`
- Verify: `gear_sonic/tests/test_xr_teleop_dashboard.py`

- [ ] **Step 1: Run targeted dashboard tests**

Run:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 pytest gear_sonic/tests/test_xr_teleop_dashboard.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run existing bridge tests to catch regressions**

Run:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 pytest gear_sonic/tests/test_xr_upperbody_bridge.py -q
```

Expected: existing bridge tests still pass.

- [ ] **Step 3: Start dashboard manually without sources**

Run:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python -B gear_sonic/scripts/xr_teleop_dashboard.py --host 127.0.0.1 --port 8088 --no-camera
```

Expected:

- terminal prints `XR teleop dashboard: http://127.0.0.1:8088`
- `GET /api/state` shows XR and bridge waiting
- edge state shows `not_yet_received`

Do not leave the server running after the check.

---

## Self-Review Checklist

- Spec coverage:
  - Read-only dashboard: Tasks 2, 4, 5.
  - XR JSON reuse via bridge normalizer: Task 1.
  - Bridge decode and command stop distinction: Task 2.
  - Inferred collection latch: Task 1.
  - Hand aperture estimate instead of binary gripper: Task 1.
  - Camera cached image endpoint: Tasks 2-4.
  - Threading HTTP server: Task 4.
  - No-control invariant: Task 2.
- No placeholders: no task contains TBD/TODO/fill-later instructions.
- Type consistency:
  - `DashboardState.snapshot()` is the single API consumed by HTTP and tests.
  - Camera endpoint uses `/api/camera/<camera_name>`, not `.jpg`.
  - `collection.state` remains explicitly inferred.
