# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop tracking-divergence monitor (observe + log only, no
automatic corrective action).

GEAR-SONIC's deploy binary (run with --output-type all, already the default
in this example's docs) publishes real-time robot state on the 'g1_debug'
ZMQ topic every control tick -- no changes to GEAR-SONIC needed. That stream
includes body_q_target (what was commanded) and body_q (actual measured
joint positions from real IMU/encoder feedback), both in MuJoCo order,
already matching this example's safety gate joint order
(kimodo_to_sonic.py).

The safety gate (kimodo_to_sonic.py's validate_motion_safety) is a
pre-flight check only: it validates a generated clip before streaming, then
trusts GEAR-SONIC to execute it faithfully. This module closes that loop by
comparing what was actually executed against what was commanded, in real
time, using GEAR-SONIC's own existing telemetry -- no new sensors, no new
instrumentation on the GEAR-SONIC side.

Deliberately observation-only for now: no automatic stop/corrective action.
There is no public baseline for what normal GEAR-SONIC tracking divergence
looks like across motion types, so picking a hard auto-stop threshold today
would be a guess, either too tight (false-positive stops on normal motion,
e.g. a stable walk can legitimately show close to 0.7 rad of peak divergence
on a leg joint) or too loose (misses a real problem). This logs real max/RMS
divergence numbers so a threshold can be set from measurement, the same way
this example's safety-gate velocity limit was calibrated from real test
data, not assumed.
"""
import argparse
import os
import sys
import threading
import time
from collections import deque

import numpy as np

# This file lives at gear_sonic/examples/speech_to_motion/closed_loop_monitor.py,
# two directories below gear_sonic/ -- derive the utils/ path from our own
# location rather than hardcoding a specific repo checkout path.
_GEAR_SONIC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_GEAR_SONIC_ROOT, "utils", "data_collection"))
from zmq_state_subscriber import ZMQStateSubscriber  # noqa: E402

POLL_INTERVAL_S = 0.1  # far below the 50Hz tick rate; enough to catch real divergence trends
LOG_THROTTLE_S = 1.0
ROLLING_WINDOW_S = 5.0

# Soft "watch" threshold only -- logs a [WATCH] line, does not trigger any
# action. Set conservatively wide since there is no real baseline data
# across motion types yet; tighten (or replace with a real action
# threshold) once this has produced real measured divergence numbers from
# a range of known-good clips.
WATCH_THRESHOLD_RAD = 0.5

G1_JOINT_NAMES_MUJOCO = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]


class ClosedLoopMonitor:
    def __init__(self, host="localhost", port=5557, log_path=None):
        self._sub = ZMQStateSubscriber(host=host, port=port)
        self._log_path = log_path
        self._log_f = open(log_path, "a") if log_path else None
        self._lock = threading.Lock()
        self._history = deque()  # (timestamp, max_div, rms_div, worst_joint)
        self._latest = None
        self._last_log_t = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _log(self, line: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{ts}] {line}"
        print(full, flush=True)
        if self._log_f:
            self._log_f.write(full + "\n")
            self._log_f.flush()

    def start(self) -> None:
        self._log("[monitor] starting, subscribing to g1_debug for body_q vs body_q_target")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def get_latest_divergence(self):
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while not self._stop.is_set():
            time.sleep(POLL_INTERVAL_S)
            msg = self._sub.get_msg()
            if msg is None:
                continue

            target = msg.get("body_q_target")
            measured = msg.get("body_q_measured", msg.get("body_q"))
            if target is None or measured is None:
                continue

            target = np.asarray(target, dtype=np.float64)
            measured = np.asarray(measured, dtype=np.float64)
            if target.shape != (29,) or measured.shape != (29,):
                continue

            diff = np.abs(measured - target)
            max_div = float(diff.max())
            rms_div = float(np.sqrt(np.mean(diff ** 2)))
            worst_idx = int(diff.argmax())
            worst_joint = G1_JOINT_NAMES_MUJOCO[worst_idx]

            now = time.monotonic()
            with self._lock:
                self._latest = {
                    "max_div": max_div,
                    "rms_div": rms_div,
                    "worst_joint": worst_joint,
                    "worst_val": max_div,
                }
                self._history.append((now, max_div, rms_div, worst_joint))
                while self._history and now - self._history[0][0] > ROLLING_WINDOW_S:
                    self._history.popleft()

            if max_div > WATCH_THRESHOLD_RAD:
                self._log(
                    f"[WATCH] divergence {max_div:.3f} rad on '{worst_joint}' "
                    f"(rms across all joints: {rms_div:.3f} rad) exceeds soft watch threshold "
                    f"{WATCH_THRESHOLD_RAD} rad -- observation only, no action taken"
                )

            if now - self._last_log_t >= LOG_THROTTLE_S:
                self._last_log_t = now
                with self._lock:
                    window = list(self._history)
                if window:
                    peak_max = max(w[1] for w in window)
                    peak_rms = max(w[2] for w in window)
                    self._log(
                        f"[status] current max={max_div:.3f} rad ('{worst_joint}') rms={rms_div:.3f} rad "
                        f"| last {ROLLING_WINDOW_S:.0f}s peak: max={peak_max:.3f} rad rms={peak_rms:.3f} rad"
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("SONIC_HOST", "localhost"), help="GEAR-SONIC host (default: $SONIC_HOST or localhost)")
    parser.add_argument("--port", type=int, default=5557, help="GEAR-SONIC ZMQ 'g1_debug' telemetry port (default: 5557)")
    parser.add_argument("--log-path", default=os.path.expanduser("~/closed_loop_monitor.log"), help="log file path")
    args = parser.parse_args()

    monitor = ClosedLoopMonitor(host=args.host, port=args.port, log_path=args.log_path)
    monitor.start()
    print(f"Closed-loop monitor running standalone, logging to {args.log_path}. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        monitor.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
