#!/usr/bin/env python3
"""Monitor the G1 base orientation via the C++ debug socket (port 5557).

Subscribes to the ``g1_debug`` msgpack stream published by the C++ controller
(``ZMQOutputHandler``), computes the base tilt from ``base_quat`` and reports
whether the robot is upright. Useful as a fall detector during autonomous
tests and as an operator display at events.

Run (any env with pyzmq + msgpack, e.g. .venv_teleop):
    python gear_sonic/scripts/robot_upright_monitor.py            # monitor forever
    python gear_sonic/scripts/robot_upright_monitor.py --watch 30 # exit 1 if the robot
                                                                  # falls within 30 s
Exit codes (--watch): 0 = upright the whole time, 1 = fall detected,
2 = no data received.
"""

from __future__ import annotations

import argparse
import sys
import time

import msgpack
import numpy as np
import zmq

TOPIC = b"g1_debug"


def base_tilt_deg(quat_wxyz) -> float:
    """Tilt of the base from vertical (deg) from a wxyz quaternion."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    # Rotation matrix element m[2][2] = 1 - 2(x^2 + y^2)
    m22 = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(m22, -1.0, 1.0))))


def main():
    parser = argparse.ArgumentParser(description="G1 upright monitor (debug port 5557)")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument(
        "--tilt_max_deg", type=float, default=60.0,
        help="Base tilt above this = fallen",
    )
    parser.add_argument(
        "--watch", type=float, default=0.0,
        help="Watch for N seconds and exit 0 (upright) / 1 (fell) / 2 (no data). "
        "0 = monitor forever.",
    )
    args = parser.parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, TOPIC)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(f"tcp://{args.host}:{args.port}")
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    t0 = time.monotonic()
    n_msgs = 0
    fell = False
    max_tilt = 0.0
    last_print = 0.0
    try:
        while True:
            if args.watch > 0 and time.monotonic() - t0 >= args.watch:
                break
            if not dict(poller.poll(timeout=200)):
                continue
            raw = sub.recv()
            payload = msgpack.unpackb(raw[len(TOPIC):], raw=False)
            quat = payload.get("base_quat")
            if quat is None:
                continue
            n_msgs += 1
            tilt = base_tilt_deg(quat)
            max_tilt = max(max_tilt, tilt)
            if tilt > args.tilt_max_deg and not fell:
                fell = True
                print(f"\n[UprightMonitor] FALL DETECTED: base tilt {tilt:.0f} deg "
                      f"(limit {args.tilt_max_deg:.0f})")
                if args.watch > 0:
                    break
            now = time.monotonic()
            if now - last_print >= 1.0:
                state = "FALLEN" if tilt > args.tilt_max_deg else "upright"
                print(f"\r[UprightMonitor] tilt={tilt:5.1f} deg ({state}) "
                      f"msgs={n_msgs}", end="")
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        sub.close(0)

    print(f"\n[UprightMonitor] done: msgs={n_msgs} max_tilt={max_tilt:.1f} deg "
          f"fell={fell}")
    if args.watch > 0:
        if n_msgs == 0:
            sys.exit(2)
        sys.exit(1 if fell else 0)


if __name__ == "__main__":
    main()
