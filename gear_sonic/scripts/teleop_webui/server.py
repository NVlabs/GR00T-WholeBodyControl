#!/usr/bin/env python3
"""Web UI server for one-button webcam teleoperation.

Run (inside .venv_teleop, from repo root):
    python gear_sonic/scripts/teleop_webui/server.py
Then open http://localhost:8080

Endpoints:
    GET  /                    UI page
    WS   /ws                  status stream (1 Hz JSON)
    POST /api/start?mode=sim  start the full stack
    POST /api/stop            graceful stop
    POST /api/estop           emergency stop ('o' to the C++ controller)
    POST /api/cleanup         kill leftovers from previous runs
    POST /api/key             {"component": "cpp"|"bridge", "key": "]"|"enter"|...}
    GET  /api/preflight       run pre-flight checks only
    GET  /preview.mjpg        webcam preview (MJPEG, relayed from GEM port 5559)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

import uvicorn
import zmq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from orchestrator import PREVIEW_PORT, Orchestrator

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Webcam Teleop")
orch = Orchestrator()


# --------------------------------------------------------------------- preview
class PreviewRelay:
    """SUB to the GEM preview socket (JPEG bytes) and fan out to HTTP clients."""

    def __init__(self, port: int):
        self.port = port
        self._latest: bytes | None = None
        self._cond = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.connect(f"tcp://127.0.0.1:{self.port}")
        while True:
            try:
                jpeg = sock.recv()
            except zmq.ZMQError:
                time.sleep(0.5)
                continue
            with self._cond:
                self._latest = jpeg
                self._cond.notify_all()

    def frames(self, timeout: float = 2.0):
        while True:
            with self._cond:
                self._cond.wait(timeout=timeout)
                jpeg = self._latest
            if jpeg is None:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            )


relay = PreviewRelay(PREVIEW_PORT)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/preview.mjpg")
async def preview():
    return StreamingResponse(
        relay.frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/start")
async def api_start(
    mode: str = "sim", camera_id: int = 0, upper_body: bool = False,
    video_source: str = "webcam",
):
    if mode not in ("sim", "real"):
        return JSONResponse({"ok": False, "error": "mode must be sim|real"}, status_code=400)
    if video_source not in ("webcam", "phone"):
        return JSONResponse({"ok": False, "error": "video_source must be webcam|phone"}, status_code=400)
    ok = orch.start(mode=mode, camera_id=camera_id, upper_body=upper_body, video_source=video_source)
    return {"ok": ok}


@app.post("/api/stop")
async def api_stop():
    return {"ok": orch.stop()}


@app.post("/api/estop")
async def api_estop():
    orch.estop()
    return {"ok": True}


@app.post("/api/cleanup")
async def api_cleanup():
    orch.cleanup()
    return {"ok": True}


@app.post("/api/key")
async def api_key(payload: dict):
    ok = orch.send_key(payload.get("component", ""), payload.get("key", ""))
    return {"ok": ok}


@app.get("/api/preflight")
async def api_preflight(mode: str = "sim", video_source: str = "webcam"):
    return {"checks": orch.preflight(mode, video_source)}


@app.get("/api/status")
async def api_status():
    return orch.status()


@app.websocket("/ws")
async def ws_status(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps(orch.status()))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass


def main():
    parser = argparse.ArgumentParser(description="Webcam teleop web UI")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind host (use 0.0.0.0 to access from another device)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
