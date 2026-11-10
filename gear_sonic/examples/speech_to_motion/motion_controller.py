# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent orchestrator sitting between the ASR client and
kimodo_server.py / kimodo_to_sonic.py.

Implements:
  - Safety gate: every generated clip is validated against G1 joint
    position/velocity limits before anything reaches SONIC (reuses
    validate_motion_safety() from kimodo_to_sonic.py). A clip that fails is
    never streamed -- the robot just holds its current pose.
  - Live interruption: every new voice command interrupts and replaces
    whatever motion is currently playing, immediately -- there is no queue.
    E.g. mid-"walk 5 steps and kick", saying "turn left, walk 5 steps, sit"
    cancels the kick sequence and starts the new one right away. Generation
    takes a few seconds; if an even newer command arrives while an older one
    is still generating, the older one is discarded once it finishes rather
    than streamed out of order (see `_generation_id`).
  - Closed-loop execution monitor: observes GEAR-SONIC's real executed
    motion against what was commanded, live, via GEAR-SONIC's own existing
    `g1_debug` telemetry (see closed_loop_monitor.py). Observation and
    logging only today -- see that module's docstring for why automatic
    corrective action is deliberately not wired in yet.

Protocol: newline-delimited JSON ({"prompt": "..."}), matching
kimodo_server.py's own protocol shape.

Examples:
    # all-defaults: everything on localhost, matching a single-machine setup
    python motion_controller.py

    # kimodo_server.py and GEAR-SONIC on different hosts
    python motion_controller.py --kimodo-host 10.0.0.5 --sonic-host 10.0.0.7
"""
import argparse
import json
import os
import socket
import socketserver
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimodo_to_sonic import (  # noqa: E402
    SONIC_FPS,
    resample_and_reorder,
    validate_motion_safety,
)
from test_zmq_manager import ZMQPublisher  # noqa: E402
from closed_loop_monitor import ClosedLoopMonitor  # noqa: E402

_state_lock = threading.Lock()
_generation_id = 0  # bumped on every incoming prompt; used to discard stale, superseded work
_busy_until = 0.0  # time.monotonic() timestamp; informational only (no queue waits on this)

# Set by main() before the server starts; module-level so process_prompt()
# and Handler can reach them without threading config through every call.
_pub = None
_kimodo_gen_server = None


def generate_via_kimodo_server(prompt: str) -> dict:
    s = socket.create_connection(_kimodo_gen_server, timeout=60)
    s.sendall((json.dumps({"prompt": prompt, "duration": 5}) + "\n").encode("utf-8"))
    resp = s.recv(65536)
    s.close()
    return json.loads(resp.decode("utf-8"))


def process_prompt(prompt: str) -> None:
    """Generate, safety-check, and stream one prompt on its own thread.
    Aborts (without streaming) if a newer prompt has arrived in the
    meantime, so a slow-to-generate stale command never plays after a
    command the user gave more recently."""
    global _busy_until

    with _state_lock:
        my_generation_id = _generation_id

    t0 = time.monotonic()

    print(f"[controller] generating: {prompt!r}")
    gen = generate_via_kimodo_server(prompt)
    print(f"[controller] [+{time.monotonic()-t0:.2f}s] generation call returned")

    with _state_lock:
        superseded = my_generation_id != _generation_id
    if superseded:
        print(f"[controller] {prompt!r} superseded by a newer command -- discarding, not streaming")
        return

    if gen.get("status") != "ok":
        print(f"[controller] generation FAILED for {prompt!r}: {gen.get('error')}")
        return

    joint_pos_mj, joint_vel_mj, joint_pos_il, joint_vel_il, body_quat = resample_and_reorder(
        gen["output_csv"]
    )
    print(f"[controller] [+{time.monotonic()-t0:.2f}s] resample+reorder done")

    ok, reason = validate_motion_safety(joint_pos_mj, joint_vel_mj)
    print(f"[controller] [+{time.monotonic()-t0:.2f}s] safety gate done")
    if not ok:
        print(f"[controller] SAFETY GATE REJECTED {prompt!r}: {reason}")
        print("[controller] Nothing sent to SONIC. Robot holds its current pose.")
        return

    with _state_lock:
        superseded = my_generation_id != _generation_id
    if superseded:
        print(f"[controller] {prompt!r} superseded by a newer command during safety check -- discarding")
        return

    n_frames = joint_pos_il.shape[0]
    duration_s = n_frames / SONIC_FPS
    frame_indices = np.arange(n_frames, dtype=np.int64)

    with _state_lock:
        _busy_until = time.monotonic() + duration_s
    _pub.send_pose(joint_pos_il, joint_vel_il, body_quat, frame_indices, catch_up=False)
    print(f"[controller] [+{time.monotonic()-t0:.2f}s] streamed {prompt!r} ({n_frames} frames, {duration_s:.1f}s), interrupting anything prior")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        global _generation_id
        try:
            line = self.rfile.readline()
            if not line:
                return
            req = json.loads(line.decode("utf-8"))
            prompt = req.get("prompt", "")
            print(f"[controller] received: {prompt!r}")

            if not prompt:
                self.wfile.write((json.dumps({"status": "error", "error": "empty prompt"}) + "\n").encode("utf-8"))
                return

            with _state_lock:
                _generation_id += 1
            threading.Thread(target=process_prompt, args=(prompt,), daemon=True).start()

            self.wfile.write((json.dumps({"status": "processing", "prompt": prompt}) + "\n").encode("utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[controller] handler error: {e}")


class TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    global _pub, _kimodo_gen_server

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="bind address for the controller's own TCP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8767, help="TCP port for the controller's own server (default: 8767)")
    parser.add_argument("--kimodo-host", default=os.environ.get("KIMODO_HOST", "localhost"), help="kimodo_server.py host (default: $KIMODO_HOST or localhost)")
    parser.add_argument("--kimodo-port", type=int, default=8766, help="kimodo_server.py port (default: 8766)")
    parser.add_argument("--sonic-host", default=os.environ.get("SONIC_HOST", "localhost"), help="GEAR-SONIC host (default: $SONIC_HOST or localhost)")
    parser.add_argument("--sonic-pose-port", type=int, default=5556, help="GEAR-SONIC ZMQ 'pose' topic port (default: 5556)")
    parser.add_argument("--sonic-debug-port", type=int, default=5557, help="GEAR-SONIC ZMQ 'g1_debug' telemetry port, used by the closed-loop monitor (default: 5557)")
    parser.add_argument("--monitor-log", default=os.path.expanduser("~/closed_loop_monitor.log"), help="closed-loop monitor log file path")
    args = parser.parse_args()

    _kimodo_gen_server = (args.kimodo_host, args.kimodo_port)

    # Single persistent publisher for the controller's lifetime, instead of
    # binding/closing a fresh ZMQPublisher per clip (kimodo_to_sonic.py's CLI
    # behavior). A new send_pose() on this socket immediately supersedes
    # whatever SONIC is currently tracking -- that's what makes live
    # interruption work: we don't need to "cancel" the robot's current
    # motion, we just start streaming the new one and SONIC picks it up
    # right away.
    _pub = ZMQPublisher(host=args.sonic_host, port=args.sonic_pose_port, verbose=False)

    # Closed-loop execution monitor: observes GEAR-SONIC's real executed
    # motion (body_q) against what we commanded (body_q_target) via
    # GEAR-SONIC's own existing g1_debug telemetry, logs divergence. Does
    # not take any action yet -- see closed_loop_monitor.py for why
    # auto-actuation is deliberately staged out until there is real
    # baseline divergence data to calibrate a threshold against.
    monitor = ClosedLoopMonitor(host=args.sonic_host, port=args.sonic_debug_port, log_path=args.monitor_log)
    monitor.start()

    with TCPServer((args.host, args.port), Handler) as server:
        print(f"Motion controller listening on {args.host}:{args.port}")
        print(f"  -> generation: {args.kimodo_host}:{args.kimodo_port} (kimodo_server.py)")
        print(f"  -> streaming:  {args.sonic_host}:{args.sonic_pose_port} (GEAR-SONIC ZMQ 'pose' topic)")
        print("  -> every new command interrupts and replaces whatever is currently playing")
        print(f"  -> closed-loop monitor: observing body_q vs body_q_target, log-only (see {args.monitor_log})")
        server.serve_forever()


if __name__ == "__main__":
    main()
