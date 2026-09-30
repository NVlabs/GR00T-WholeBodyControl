"""Open the BrainCo hands during a SONIC safe stop.

The deploy (g1_deploy_onnx_ref, --input-type zmq_manager) publishes its safe-stop
state on PUB tcp://*:5571, topic "safe_stop_state", message
    safe_stop_state {"active": 0|1, "epoch": n}
at 10 Hz and at once on change (see SAFE_STOP.md). The BrainCo hands are not driven
by the deploy but by the teleop (brainco.py) or VLA script, so each hand sender
passes its command through a SafeHandGuard just before it is written to DDS:

    guard = get_hand_guard("rearm")      # teleop; "hold" for the VLA script
    q, overridden = guard.apply("left", requested_q, last_sent_q)

While the stop is active every finger ramps from where it is to 0 (open) at
SAFE_STOP_HAND_OPEN_RATE (default 0.5 /s: fully closed -> open in ~2 s); the
operator's / VLA's finger commands are ignored. After the release (u):
  - "rearm" (teleop): the hands stay open until the operator releases the trigger,
    then the trigger works as usual (a trigger held since before the stop does
    not re-close the hand);
  - "hold" (VLA): the hands stay open until guard.resume() (called when the VLA
    is restarted / resumed from its keyboard), then they follow the VLA again,
    rate-limited at SAFE_STOP_HAND_RESUME_RATE (default 0.5 /s) until caught up.

Environment:
    SAFE_STOP_HOST (default localhost: the deploy runs on the same computer)
    SAFE_STOP_PORT (default 5571; 0 = guard off)
    SAFE_STOP_HAND_OPEN_RATE, SAFE_STOP_HAND_RESUME_RATE (finger q per second)

If the deploy is not running, nothing is received and the guard never acts. If the
deploy stops publishing during a stop, the hands stay open (the last known state
is kept; only an explicit active=0 releases).
"""

import json
import os
import threading
import time

import numpy as np

TOPIC = b"safe_stop_state"


class SafeStopState:
    """Background SUB of the deploy's safe-stop state."""

    def __init__(self, host: str, port: int):
        self.active = False
        self.epoch = 0
        self.connected = False
        self._lock = threading.Lock()
        self._host, self._port = host, port
        threading.Thread(target=self._loop, daemon=True, name="safe_stop_state").start()

    def _loop(self):
        try:
            import zmq
        except ImportError as e:
            print(f"[SafeStopHands] pyzmq missing ({e}); BrainCo hands will NOT open on a safe stop")
            return
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.setsockopt(zmq.SUBSCRIBE, TOPIC)
        sock.connect(f"tcp://{self._host}:{self._port}")
        while True:
            try:
                msg = sock.recv()
            except zmq.Again:
                continue
            except Exception as e:  # noqa: BLE001
                print(f"[SafeStopHands] receive failed: {e}")
                time.sleep(0.5)
                continue
            try:
                data = json.loads(msg[len(TOPIC):].decode().strip())
                active, epoch = bool(int(data["active"])), int(data["epoch"])
            except Exception:  # noqa: BLE001
                continue
            with self._lock:
                if not self.connected:
                    self.connected = True
                    print(f"[SafeStopHands] connected to the deploy safe-stop state "
                          f"({self._host}:{self._port}); BrainCo hands open on a safe stop")
                self.active, self.epoch = active, epoch

    def get(self):
        with self._lock:
            return self.active, self.epoch


class SafeHandGuard:
    """Per-hand override of the BrainCo command during / after a safe stop."""

    def __init__(self, mode: str = "rearm", state: SafeStopState | None = None,
                 open_rate: float | None = None, resume_rate: float | None = None):
        if mode not in ("rearm", "hold"):
            raise ValueError(f"mode must be 'rearm' or 'hold', got {mode}")
        self.mode = mode
        self.state = state
        self.open_rate = float(os.environ.get("SAFE_STOP_HAND_OPEN_RATE", "0.5")
                               if open_rate is None else open_rate)
        self.resume_rate = float(os.environ.get("SAFE_STOP_HAND_RESUME_RATE", "0.5")
                                 if resume_rate is None else resume_rate)
        self._lock = threading.Lock()
        # per hand: "normal" | "opening" (stop active) | "held" (released, still open) | "blend"
        self._phase = {"left": "normal", "right": "normal"}
        self._out = {"left": None, "right": None}
        self._t = {"left": None, "right": None}
        self._resume_requested = False

    def resume(self):
        """VLA restarted / resumed: let the hands follow it again (mode 'hold')."""
        with self._lock:
            self._resume_requested = True

    def active(self) -> bool:
        return self.state is not None and self.state.get()[0]

    def apply(self, hand: str, requested, last_sent=None):
        """Return (q, overridden). q is the command to send; overridden=False -> use requested."""
        requested = np.clip(np.asarray(requested, dtype=np.float32).reshape(-1), 0.0, 1.0)
        if self.state is None:
            return requested, False
        active, _ = self.state.get()
        now = time.monotonic()
        with self._lock:
            phase = self._phase[hand]
            dt = 0.0 if self._t[hand] is None else min(now - self._t[hand], 0.2)
            self._t[hand] = now
            prev = self._out[hand]
            if prev is None or prev.shape != requested.shape:
                base = last_sent if last_sent is not None else requested
                prev = np.clip(np.asarray(base, dtype=np.float32).reshape(-1), 0.0, 1.0)
                if prev.shape != requested.shape:
                    prev = requested.copy()

            if active:
                if phase != "opening":
                    print(f"[SafeStopHands] SAFE STOP: {hand} hand opening "
                          f"(q -> 0 at {self.open_rate:.2f}/s)")
                    phase = "opening"
                    if last_sent is not None:  # start from what the hand was last told
                        prev = np.clip(np.asarray(last_sent, dtype=np.float32).reshape(-1), 0.0, 1.0)
                out = np.maximum(prev - self.open_rate * dt, 0.0).astype(np.float32)
                overridden = True
            else:
                if phase == "opening":
                    phase = "held"
                    self._resume_requested = False
                    print(f"[SafeStopHands] released: {hand} hand stays open until "
                          + ("the trigger is released" if self.mode == "rearm"
                             else "the VLA is resumed (i / k / p)"))
                if phase == "held":
                    # finish opening if the release came mid-ramp
                    out = np.maximum(prev - self.open_rate * dt, 0.0).astype(np.float32)
                    if self.mode == "rearm":
                        done = bool(np.all(requested <= 1e-3))
                    else:
                        done = self._resume_requested
                    if done:
                        phase = "normal" if self.mode == "rearm" else "blend"
                        print(f"[SafeStopHands] {hand} hand follows "
                              f"{'the trigger' if self.mode == 'rearm' else 'the VLA'} again")
                    overridden = True
                if phase == "blend":
                    step = self.resume_rate * dt
                    out = (prev + np.clip(requested - prev, -step, step)).astype(np.float32)
                    if np.allclose(out, requested, atol=1e-3):
                        phase = "normal"
                    overridden = True
                if phase == "normal":
                    out, overridden = requested, False

            self._phase[hand] = phase
            self._out[hand] = out.copy()
            return out, overridden


_GUARD = None


def get_hand_guard(mode: str = "rearm") -> SafeHandGuard:
    """Shared guard (one SUB per process). SAFE_STOP_PORT=0 disables it."""
    global _GUARD
    if _GUARD is None:
        port = int(os.environ.get("SAFE_STOP_PORT", "5571"))
        host = os.environ.get("SAFE_STOP_HOST", "localhost")
        state = SafeStopState(host, port) if port > 0 else None
        if state is None:
            print("[SafeStopHands] disabled (SAFE_STOP_PORT=0)")
        else:
            print(f"[SafeStopHands] listening for the safe-stop state on {host}:{port} (mode {mode})")
        _GUARD = SafeHandGuard(mode, state)
    return _GUARD
