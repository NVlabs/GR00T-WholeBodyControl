"""Terminal keyboard input for controller-shaped teleop commands.

The SONIC manager still consumes the same controller semantics (A/B/X/Y and
left grip).  This source only replaces how those booleans are produced.  A
key press is held for a short TTL because a normal SSH terminal reports key
presses, not key-release events.  The TTL makes chords such as ``A+X`` and
``Grip+A`` deterministic without changing the manager state machine.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import select
import sys
import termios
import threading
import time


DEFAULT_KEY_MAPPING = {
    "a": "a",
    "b": "b",
    "x": "x",
    "y": "y",
    "grip": "g",
}

DEFAULT_ACTION_MAPPING = {
    "start_stop": "s",
    "mode": "m",
    "frozen_mode": "f",
    "record": "c",
    "discard": "d",
    "menu": "p",
    "axis_click": "v",
}

ACTION_CONTROLS = {
    "start_stop": ("a", "b", "x", "y"),
    "mode": ("a", "x"),
    "frozen_mode": ("b", "y"),
    "record": ("grip", "a"),
    "discard": ("grip", "b"),
    "menu": ("menu",),
    "axis_click": ("axis_click",),
}


class KeyboardControlSource:
    """Read keyboard presses and expose the existing controller-data schema.

    ``feed_key`` is intentionally public so the state machine can be tested
    without a terminal.  ``start`` is only needed for the live SSH/tmux path.
    """

    def __init__(
        self,
        ttl_sec: float = 0.32,
        key_mapping: Mapping[str, str] | None = None,
        action_mapping: Mapping[str, str] | None = None,
    ) -> None:
        if ttl_sec <= 0.0:
            raise ValueError("ttl_sec must be positive")

        mapping = dict(DEFAULT_KEY_MAPPING)
        if key_mapping is not None:
            mapping.update(key_mapping)
        actions = dict(DEFAULT_ACTION_MAPPING)
        if action_mapping is not None:
            actions.update(action_mapping)
        self._validate_mapping(mapping, actions)

        self.ttl_sec = float(ttl_sec)
        self.key_mapping = mapping
        self.action_mapping = actions
        self._active_until: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._terminal_attrs = None

    @staticmethod
    def _validate_mapping(
        mapping: Mapping[str, str], action_mapping: Mapping[str, str]
    ) -> None:
        required_controls = set(DEFAULT_KEY_MAPPING)
        required_actions = set(DEFAULT_ACTION_MAPPING)
        if set(mapping) != required_controls:
            missing = required_controls - set(mapping)
            extra = set(mapping) - required_controls
            raise ValueError(f"invalid control mapping; missing={missing}, extra={extra}")
        if set(action_mapping) != required_actions:
            missing = required_actions - set(action_mapping)
            extra = set(action_mapping) - required_actions
            raise ValueError(f"invalid action mapping; missing={missing}, extra={extra}")
        values = []
        for name, key in {**mapping, **action_mapping}.items():
            if not isinstance(key, str) or len(key) != 1 or not key.isprintable():
                raise ValueError(f"keyboard mapping for {name!r} must be one printable character")
            values.append(key.lower())
        if len(set(values)) != len(values):
            raise ValueError("keyboard mapping keys must be unique")

    def feed_key(self, key: str, now: float | None = None) -> bool:
        """Register one key press; return whether it is a mapped control key."""
        if not key:
            return False
        normalized = key.lower()
        controls = [
            name for name, mapped_key in self.key_mapping.items() if mapped_key.lower() == normalized
        ]
        action = next(
            (
                name
                for name, mapped_key in self.action_mapping.items()
                if mapped_key.lower() == normalized
            ),
            None,
        )
        if action is not None:
            controls.extend(ACTION_CONTROLS[action])
        if not controls:
            return False
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            for control in controls:
                self._active_until[control] = timestamp + self.ttl_sec
        return True

    def _pressed(self, name: str, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            active = self._active_until.get(name, 0.0) > timestamp
            if not active and name in self._active_until:
                del self._active_until[name]
            return active

    def get_controller_data(self, now: float | None = None) -> dict[str, object]:
        """Return the same fields consumed by ``pico_manager_thread_server``."""
        return {
            "left_trigger_value": 0.0,
            "right_trigger_value": 0.0,
            "left_squeeze_value": 1.0 if self._pressed("grip", now) else 0.0,
            "right_squeeze_value": 0.0,
            "left_thumbstick": [0.0, 0.0],
            "right_thumbstick": [0.0, 0.0],
            "left_thumbstick_click": 1.0 if self._pressed("axis_click", now) else 0.0,
            "right_thumbstick_click": 0.0,
            "left_menu_button": self._pressed("menu", now),
            "left_primary_click": 1.0 if self._pressed("x", now) else 0.0,
            "left_secondary_click": 1.0 if self._pressed("y", now) else 0.0,
            "right_primary_click": 1.0 if self._pressed("a", now) else 0.0,
            "right_secondary_click": 1.0 if self._pressed("b", now) else 0.0,
        }

    def start(self) -> None:
        """Start a cbreak-mode reader in the current terminal."""
        if self._thread is not None and self._thread.is_alive():
            return
        if not sys.stdin.isatty():
            raise RuntimeError(
                "keyboard control requires a TTY; run the manager in a terminal or tmux pane"
            )

        self._fd = sys.stdin.fileno()
        self._terminal_attrs = termios.tcgetattr(self._fd)
        try:
            attrs = termios.tcgetattr(self._fd)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 1
            termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
        except Exception:
            self._fd = None
            self._terminal_attrs = None
            raise

        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="keyboard-control", daemon=True)
        self._thread.start()
        print(
            "[Keyboard] atomic actions: s=start/stop, m=planner/pose, "
            "f=frozen mode, c=record, d=discard, p=pause, v=VR3PT; "
            "raw controls: a=A, b=B, x=X, y=Y, g=Grip"
        )

    def _read_loop(self) -> None:
        assert self._fd is not None
        while not self._stop.is_set():
            readable, _, _ = select.select([self._fd], [], [], 0.05)
            if not readable:
                continue
            try:
                data = os.read(self._fd, 1)
            except OSError:
                break
            if data:
                self.feed_key(data.decode("utf-8", errors="ignore"))

    def stop(self) -> None:
        """Stop the reader and restore the terminal exactly as it was."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._fd is not None and self._terminal_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._terminal_attrs)
            except (OSError, termios.error):
                pass
        self._thread = None
        self._fd = None
        self._terminal_attrs = None
