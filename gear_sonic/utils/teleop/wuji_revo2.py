"""Wuji Glove -> Revo2 hand retargeting boundary.

This module deliberately stops at a named, bounded Revo2 target.  It does not
open a serial/EtherCAT port or enable a hand.  A mature retargeting backend is
injected through ``retarget_side`` (for example, the existing DexPilot/
``dex_retargeting`` implementation), while this boundary owns the parts that
must be identical in data collection: MediaPipe keypoint validation, latest
frame semantics, stale-frame reset, target ordering, limits, and optional
velocity limiting.

Wuji Glove skeletons use the standard 21-point MediaPipe layout:
0 wrist, then thumb 1..4, index 5..8, middle 9..12, ring 13..16, and
pinky 17..20.  Revo2 has six named commands: thumb proximal, thumb
metacarpal, and one proximal command for each other finger.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import threading
import time
from typing import Any, Protocol

import numpy as np


WUJI_SKELETON_SHAPE = (21, 3)
WUJI_FINGERTIP_INDICES = (4, 8, 12, 16, 20)

REVO2_JOINT_NAMES = (
    "thumb_proximal_joint",
    "thumb_metacarpal_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)
REVO2_JOINT_LIMITS_RAD = (
    (0.0, 1.0472),
    (0.0, 1.5184),
    (0.0, 1.4661),
    (0.0, 1.4661),
    (0.0, 1.4661),
    (0.0, 1.4661),
)
_REVO2_LOWER = np.asarray([lower for lower, _ in REVO2_JOINT_LIMITS_RAD], dtype=np.float64)
_REVO2_UPPER = np.asarray([upper for _, upper in REVO2_JOINT_LIMITS_RAD], dtype=np.float64)


def _normalize_side(side: str) -> str:
    normalized = str(side).strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return normalized


@dataclass(frozen=True)
class WujiSkeletonFrame:
    """One validated Wuji frame with source and receive timing."""

    joints: np.ndarray
    side: str
    sequence: int
    timestamp_ns: int
    received_monotonic: float

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints, dtype=np.float32)
        if joints.shape != WUJI_SKELETON_SHAPE:
            raise ValueError(
                f"Wuji skeleton must have shape {WUJI_SKELETON_SHAPE}, got {joints.shape}"
            )
        if not np.isfinite(joints).all():
            raise ValueError("Wuji skeleton contains non-finite coordinates")
        _normalize_side(self.side)
        if int(self.sequence) < 0:
            raise ValueError("sequence must be non-negative")
        if int(self.timestamp_ns) < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not np.isfinite(float(self.received_monotonic)):
            raise ValueError("received_monotonic must be finite")
        joints = np.array(joints, dtype=np.float32, copy=True)
        joints.setflags(write=False)
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "side", _normalize_side(self.side))
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "received_monotonic", float(self.received_monotonic))

    @classmethod
    def from_joints(
        cls,
        joints: Any,
        side: str,
        sequence: int,
        timestamp_ns: int = 0,
        received_monotonic: float | None = None,
    ) -> "WujiSkeletonFrame":
        return cls(
            joints=np.asarray(joints, dtype=np.float32),
            side=side,
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            received_monotonic=time.monotonic()
            if received_monotonic is None
            else received_monotonic,
        )

    @property
    def fingertips(self) -> np.ndarray:
        """Return [thumb, index, middle, ring, pinky] fingertip positions."""
        return np.asarray(self.joints[list(WUJI_FINGERTIP_INDICES)], dtype=np.float32)


class LatestWujiSkeleton:
    """Thread-safe latest-frame store; old frames are intentionally discarded."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: WujiSkeletonFrame | None = None
        self._next_sequence = 0
        self._last_drained_sequence = -1

    def submit(
        self,
        joints: Any,
        side: str,
        *,
        sequence: int | None = None,
        timestamp_ns: int = 0,
        received_monotonic: float | None = None,
    ) -> WujiSkeletonFrame:
        with self._lock:
            if sequence is None:
                sequence = self._next_sequence
            frame = WujiSkeletonFrame.from_joints(
                joints,
                side,
                sequence=sequence,
                timestamp_ns=timestamp_ns,
                received_monotonic=received_monotonic,
            )
            self._next_sequence = max(self._next_sequence, frame.sequence + 1)
            self._latest = frame
            return frame

    def drain_latest(
        self,
        *,
        now: float | None = None,
        max_age_sec: float | None = None,
    ) -> WujiSkeletonFrame | None:
        """Return only a new, non-stale frame and mark it consumed."""
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            frame = self._latest
            if frame is None or frame.sequence <= self._last_drained_sequence:
                return None
            if max_age_sec is not None and now - frame.received_monotonic > max_age_sec:
                return None
            self._last_drained_sequence = frame.sequence
            return frame

    def is_stale(self, *, now: float | None = None, max_age_sec: float) -> bool:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            return self._latest is None or now - self._latest.received_monotonic > max_age_sec


class Revo2RetargetBackend(Protocol):
    def retarget_side(self, side: str, fingertip_positions: np.ndarray) -> Any:
        """Return six Revo2 joint targets in the named contract order."""


@dataclass(frozen=True)
class Revo2Target:
    """Named, bounded target ready for a separately-owned hardware adapter."""

    side: str
    positions_rad: np.ndarray
    source_sequence: int
    source_timestamp_ns: int
    received_monotonic: float

    def __post_init__(self) -> None:
        side = _normalize_side(self.side)
        positions = np.asarray(self.positions_rad, dtype=np.float64)
        if positions.shape != (len(REVO2_JOINT_NAMES),):
            raise ValueError(f"Revo2 target must have shape (6,), got {positions.shape}")
        if not np.isfinite(positions).all():
            raise ValueError("Revo2 target contains non-finite values")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "positions_rad", np.array(positions, copy=True))

    @property
    def joint_names(self) -> tuple[str, ...]:
        prefix = f"{self.side}_"
        return tuple(prefix + name for name in REVO2_JOINT_NAMES)

    def as_named_dict(self) -> dict[str, float]:
        return dict(zip(self.joint_names, self.positions_rad.tolist(), strict=True))


def _coerce_backend_output(output: Any, side: str) -> np.ndarray:
    if isinstance(output, Mapping):
        names = tuple(f"{side}_{name}" for name in REVO2_JOINT_NAMES)
        short_names = REVO2_JOINT_NAMES
        try:
            output = [output[name] for name in names]
        except KeyError:
            try:
                output = [output[name] for name in short_names]
            except KeyError as exc:
                raise ValueError("retarget backend mapping does not contain Revo2 joint names") from exc
    positions = np.asarray(output, dtype=np.float64)
    if positions.shape != (len(REVO2_JOINT_NAMES),):
        raise ValueError(f"retarget backend must return six values, got {positions.shape}")
    if not np.isfinite(positions).all():
        raise ValueError("retarget backend returned non-finite values")
    return positions


class WujiRevo2Bridge:
    """Convert Wuji frames to bounded Revo2 targets without driving hardware."""

    def __init__(
        self,
        backend: Revo2RetargetBackend | Callable[[str, np.ndarray], Any],
        *,
        stale_after_sec: float = 0.12,
        max_velocity_rad_per_sec: float | None = None,
    ) -> None:
        if stale_after_sec <= 0.0:
            raise ValueError("stale_after_sec must be positive")
        if max_velocity_rad_per_sec is not None and max_velocity_rad_per_sec <= 0.0:
            raise ValueError("max_velocity_rad_per_sec must be positive")
        self.backend = backend
        self.stale_after_sec = float(stale_after_sec)
        self.max_velocity_rad_per_sec = max_velocity_rad_per_sec
        self._previous: Revo2Target | None = None

    def reset(self) -> None:
        self._previous = None
        reset = getattr(self.backend, "reset", None)
        if reset is not None:
            reset()

    def _retarget(self, side: str, fingertips: np.ndarray) -> Any:
        method = getattr(self.backend, "retarget_side", None)
        if method is not None:
            return method(side, fingertips)
        if callable(self.backend):
            return self.backend(side, fingertips)
        raise TypeError("backend must implement retarget_side(side, fingertips) or be callable")

    def process(self, frame: WujiSkeletonFrame, *, now: float | None = None) -> Revo2Target | None:
        now = time.monotonic() if now is None else float(now)
        if now - frame.received_monotonic > self.stale_after_sec:
            self.reset()
            return None

        raw = _coerce_backend_output(self._retarget(frame.side, frame.fingertips), frame.side)
        positions = np.clip(raw, _REVO2_LOWER, _REVO2_UPPER)
        if self.max_velocity_rad_per_sec is not None and self._previous is not None:
            if self._previous.side != frame.side:
                self.reset()
            else:
                dt = max(frame.received_monotonic - self._previous.received_monotonic, 1e-3)
                max_delta = self.max_velocity_rad_per_sec * dt
                positions = np.clip(
                    positions,
                    self._previous.positions_rad - max_delta,
                    self._previous.positions_rad + max_delta,
                )

        target = Revo2Target(
            side=frame.side,
            positions_rad=positions,
            source_sequence=frame.sequence,
            source_timestamp_ns=frame.timestamp_ns,
            received_monotonic=frame.received_monotonic,
        )
        self._previous = target
        return target

    def step(self, source: LatestWujiSkeleton, *, now: float | None = None) -> Revo2Target | None:
        now = time.monotonic() if now is None else float(now)
        frame = source.drain_latest(now=now, max_age_sec=self.stale_after_sec)
        if frame is None:
            if source.is_stale(now=now, max_age_sec=self.stale_after_sec):
                self.reset()
            return None
        return self.process(frame, now=now)


class WujiGloveSource:
    """Optional Wuji SDK adapter using the SDK's latest-frame subscription API.

    The SDK import and device connection remain lazy so this module is usable in
    CI, replay tests, and on the robot before the vendor wheel is installed.
    """

    def __init__(self, glove: Any, *, side: str | None = None) -> None:
        self.glove = glove
        self.side = _normalize_side(side) if side is not None else None
        self.store = LatestWujiSkeleton()
        self._subscription = glove.hand_skeleton().subscribe()
        self._sequence = 0

    @classmethod
    def connect(cls, serial_number: str, *, side: str | None = None) -> "WujiGloveSource":
        from wuji_sdk import DeviceType, SdkManager

        manager = SdkManager.instance()
        devices = [
            device
            for device in manager.scan()
            if device.device_type == DeviceType.WujiGlove and device.sn == serial_number
        ]
        if not devices:
            raise RuntimeError(f"Wuji Glove not found: {serial_number}")
        glove = manager.connect(sn=serial_number, device_name=f"wuji_glove_{serial_number}")
        return cls(glove, side=side)

    def _resolve_side(self) -> str:
        if self.side is not None:
            return self.side
        value = self.glove.hand_side().get()
        text = str(value).lower()
        if "left" in text:
            return "left"
        if "right" in text:
            return "right"
        raise RuntimeError(f"Wuji Glove hand side is not left/right: {value!r}")

    @staticmethod
    def _skeleton_to_joints(skeleton: Any) -> np.ndarray:
        return np.asarray([joint.pose.position for joint in skeleton.joints], dtype=np.float32)

    def poll(self) -> WujiSkeletonFrame | None:
        """Drain the SDK subscription and submit only its newest skeleton."""
        latest = None
        while True:
            frame = self._subscription.recv()
            if frame is None:
                break
            latest = frame
        if latest is None:
            return None
        result = self.store.submit(
            self._skeleton_to_joints(latest),
            self._resolve_side(),
            sequence=self._sequence,
        )
        self._sequence += 1
        return result

    def close(self) -> None:
        self._subscription.close()
