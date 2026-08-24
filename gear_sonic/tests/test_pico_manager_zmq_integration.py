"""Synthetic PICO-to-manager integration test over real ZMQ sockets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import socket
import threading
import time
from typing import Any

import msgpack
import numpy as np
import zmq

from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.utils.teleop import input_readers
from gear_sonic.utils.teleop.zmq.control_session import (
    CONTROL_SESSION_PROTOCOL,
    CONTROL_SESSION_TOPIC,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE

RECEIVER_EPOCH = bytes(range(1, 17))
ACK_DELAY_S = 0.15
START_PRESS_BEGIN_S = 0.25
START_PRESS_END_S = 0.55
TRACKING_FREEZE_S = 1.20

_DTYPES = {
    "bool": np.dtype("?"),
    "u8": np.dtype("u1"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
}


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        return int(reservation.getsockname()[1])


def _decode_packed_message(raw: bytes, topic: bytes) -> tuple[int, dict[str, np.ndarray]]:
    assert raw.startswith(topic)
    packed = memoryview(raw)[len(topic) :]
    assert len(packed) >= HEADER_SIZE
    header_bytes = bytes(packed[:HEADER_SIZE]).split(b"\x00", 1)[0]
    header = json.loads(header_bytes)
    assert header["endian"] == "le"

    payload = packed[HEADER_SIZE:]
    offset = 0
    fields: dict[str, np.ndarray] = {}
    for field in header["fields"]:
        dtype = _DTYPES[field["dtype"]]
        shape = tuple(field["shape"])
        count = math.prod(shape)
        byte_count = count * dtype.itemsize
        assert offset + byte_count <= len(payload)
        fields[field["name"]] = (
            np.frombuffer(payload[offset : offset + byte_count], dtype=dtype, count=count).reshape(shape).copy()
        )
        offset += byte_count
    assert offset == len(payload)
    return int(header["v"]), fields


def _scalar(fields: dict[str, np.ndarray], name: str) -> int:
    return int(fields[name].reshape(-1)[0])


def _token(fields: dict[str, np.ndarray], name: str) -> bytes:
    return fields[name].astype(np.uint8, copy=False).tobytes()


class _SyntheticXRT:
    """Advancing atomic snapshots followed by a deliberate tracking freeze."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = 0
        self._last_stamp_ns = 1_000_000_000
        self._manager_start: float | None = None
        self.freeze_monotonic: float | None = None
        poses = np.zeros((24, 7), dtype=np.float64)
        poses[:, 3] = 1.0
        self._poses = poses

    def init(self) -> None:
        return None

    def is_body_data_available(self) -> bool:
        return True

    def mark_manager_poll(self) -> None:
        with self._lock:
            if self._manager_start is None:
                self._manager_start = time.monotonic()

    def _elapsed(self) -> float:
        with self._lock:
            start = self._manager_start
        return 0.0 if start is None else time.monotonic() - start

    def get_body_timestamp_ns(self) -> int:
        with self._lock:
            return self._last_stamp_ns

    def get_body_snapshot(self) -> dict[str, Any]:
        time.sleep(0.002)
        now = time.monotonic()
        with self._lock:
            elapsed = 0.0 if self._manager_start is None else now - self._manager_start
            if self._manager_start is None or elapsed < TRACKING_FREEZE_S:
                self._frame += 1
                self._last_stamp_ns += 2_000_000
            elif self.freeze_monotonic is None:
                self.freeze_monotonic = now
            stamp_ns = self._last_stamp_ns
            sequence = self._frame
        return {
            "available": True,
            "timestamp_ns": stamp_ns,
            "poses": self._poses.copy(),
            "health_supported": True,
            "health_available": True,
            "health_valid": True,
            "health_schema_version": 1,
            "health_sample_sequence": sequence,
            "health_timestamp_ns": stamp_ns,
            "health_calibration_result": 0,
            "health_calibrated": True,
            "health_tracking_mode": 0,
            "health_connect_state_result": 0,
            "health_tracker_count": 2,
            "health_unique_tracker_count": 2,
            "health_body_state_result": 0,
            "health_is_tracking": True,
            "health_tracking_state_code": 0,
            "health_body_state_code": 1,
            "health_body_error_code": 0,
            "health_connected_band_count": 2,
            "health_body_data_result": 0,
            "health_body_role_count": 24,
        }

    def get_controller_snapshot(self) -> dict[str, Any]:
        elapsed = self._elapsed()
        pressed = START_PRESS_BEGIN_S <= elapsed < START_PRESS_END_S
        with self._lock:
            stamp_ns = self._last_stamp_ns
            sequence = self._frame
        pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return {
            "available": True,
            "timestamp_ns": stamp_ns,
            "left_timestamp_ns": stamp_ns,
            "right_timestamp_ns": stamp_ns,
            "left_pose": pose.copy(),
            "right_pose": pose.copy(),
            "left_trigger_value": 0.0,
            "right_trigger_value": 0.0,
            "left_squeeze_value": 0.0,
            "right_squeeze_value": 0.0,
            "left_thumbstick": np.zeros(2, dtype=np.float64),
            "right_thumbstick": np.zeros(2, dtype=np.float64),
            "left_thumbstick_click": False,
            "right_thumbstick_click": False,
            "left_primary_click": pressed,
            "left_secondary_click": pressed,
            "right_primary_click": pressed,
            "right_secondary_click": pressed,
            "left_menu_button": False,
            "right_menu_button": False,
            "health_supported": True,
            "health_available": True,
            "health_valid": True,
            "health_schema_version": 1,
            "health_sample_sequence": sequence,
            "health_timestamp_ns": stamp_ns,
            "health_left_device_valid": True,
            "health_left_is_tracked_available": True,
            "health_left_is_tracked": True,
            "health_left_tracking_state_available": True,
            "health_left_tracking_state": 3,
            "health_left_valid": True,
            "health_right_device_valid": True,
            "health_right_is_tracked_available": True,
            "health_right_is_tracked": True,
            "health_right_tracking_state_available": True,
            "health_right_tracking_state": 3,
            "health_right_valid": True,
        }

    def get_motion_tracker_snapshot(self) -> dict[str, Any]:
        return {
            "available": True,
            "timestamp_ns": self.get_body_timestamp_ns(),
            "count": 2,
            "serial_numbers": ["SYNTH-LEFT-ANKLE", "SYNTH-RIGHT-ANKLE"],
        }


@dataclass(frozen=True)
class _WireEvent:
    topic: bytes
    received_at: float
    version: int
    fields: dict[str, np.ndarray]


class _FakeNativeControlSession:
    """Native-side challenge/ACK publisher plus command/planner observer."""

    def __init__(self, manager_port: int) -> None:
        self.manager_port = manager_port
        self.feedback_port: int | None = None
        self.events: list[_WireEvent] = []
        self.claimed_session: bytes | None = None
        self.claim_received_at: float | None = None
        self.ack_sent_at: float | None = None
        self.error: BaseException | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> int:
        self._thread.start()
        assert self._ready.wait(timeout=2.0)
        assert self.feedback_port is not None
        return self.feedback_port

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        assert not self._thread.is_alive()
        if self.error is not None:
            raise self.error

    def snapshot_events(self) -> list[_WireEvent]:
        with self._lock:
            return list(self.events)

    def _challenge(self, bound_session: bytes | None) -> bytes:
        return CONTROL_SESSION_TOPIC + msgpack.packb(
            {
                "protocol": CONTROL_SESSION_PROTOCOL,
                "receiver_epoch": list(RECEIVER_EPOCH),
                "bound_publisher_session": ([] if bound_session is None else list(bound_session)),
            },
            use_bin_type=True,
        )

    def _record(self, raw: bytes, topic: bytes) -> None:
        version, fields = _decode_packed_message(raw, topic)
        received_at = time.monotonic()
        with self._lock:
            self.events.append(_WireEvent(topic, received_at, version, fields))

        if topic != b"command" or _scalar(fields, "claim") != 1:
            return
        if _token(fields, "receiver_epoch") != RECEIVER_EPOCH:
            return
        if _scalar(fields, "start") != 0 or _scalar(fields, "stop") != 0:
            return
        session = _token(fields, "publisher_session")
        if self.claimed_session is None:
            self.claimed_session = session
            self.claim_received_at = received_at

    def _run(self) -> None:
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        subscriber = context.socket(zmq.SUB)
        publisher.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"command")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"planner")
        try:
            self.feedback_port = publisher.bind_to_random_port("tcp://127.0.0.1")
            subscriber.connect(f"tcp://127.0.0.1:{self.manager_port}")
            poller = zmq.Poller()
            poller.register(subscriber, zmq.POLLIN)
            self._ready.set()
            next_challenge = 0.0

            while not self._stop.is_set():
                if dict(poller.poll(10)).get(subscriber) == zmq.POLLIN:
                    raw = subscriber.recv()
                    if raw.startswith(b"command"):
                        self._record(raw, b"command")
                    elif raw.startswith(b"planner"):
                        self._record(raw, b"planner")

                now = time.monotonic()
                bound_session = None
                if (
                    self.claimed_session is not None
                    and self.claim_received_at is not None
                    and now - self.claim_received_at >= ACK_DELAY_S
                ):
                    bound_session = self.claimed_session

                if now >= next_challenge:
                    if bound_session is not None and self.ack_sent_at is None:
                        self.ack_sent_at = now
                    publisher.send(self._challenge(bound_session))
                    next_challenge = now + 0.02
        except BaseException as exc:
            self.error = exc
            self._ready.set()
        finally:
            subscriber.close(linger=0)
            publisher.close(linger=0)
            context.term()


class _FakeThreePointPose:
    enable_smpl_vis = False

    def __init__(self, **_kwargs) -> None:
        self.closed = False

    def calibrate_now(self, _body_poses: np.ndarray) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


class _FakeFeedbackReader:
    upper_body_position_target = None
    left_hand_position_target = None
    right_hand_position_target = None
    full_body_q_measured = None

    def __init__(self, **_kwargs) -> None:
        return None

    def poll_feedback(self) -> None:
        return None


def test_pico_manager_claims_v2_session_before_streaming_and_stops_on_freeze(
    monkeypatch,
    capsys,
):
    manager_port = _free_tcp_port()
    synthetic_xrt = _SyntheticXRT()
    native = _FakeNativeControlSession(manager_port)
    feedback_port = native.start()

    monkeypatch.setattr(input_readers, "xrt", synthetic_xrt)
    monkeypatch.setattr(manager, "xrt", synthetic_xrt)
    monkeypatch.setattr(manager, "_xrt_service_running", lambda: True)
    monkeypatch.setattr(manager, "ThreePointPose", _FakeThreePointPose)
    monkeypatch.setattr(manager, "FeedbackReader", _FakeFeedbackReader)
    monkeypatch.setattr(manager, "init_hand_ik_solvers", lambda: (None, None))

    original_evaluate = manager.evaluate_body_input

    def evaluate_with_phase_start(*args, **kwargs):
        synthetic_xrt.mark_manager_poll()
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(manager, "evaluate_body_input", evaluate_with_phase_start)

    try:
        manager.run_pico_manager(
            port=manager_port,
            buffer_size=15,
            num_frames_to_send=5,
            target_fps=50,
            use_cuda=False,
            zmq_feedback_host="127.0.0.1",
            zmq_feedback_port=feedback_port,
            enable_vis_vr3pt=False,
            with_g1_robot=False,
            input_source="xrt",
            input_timeout_s=0.5,
        )
        # Manager sends burst synchronously; give native SUB one poll cycle to
        # drain already-delivered messages before closing its sockets.
        time.sleep(0.10)
    finally:
        native.close()

    output = capsys.readouterr().out
    events = native.snapshot_events()
    command_events = [event for event in events if event.topic == b"command"]
    planner_events = [event for event in events if event.topic == b"planner"]
    start_events = [event for event in command_events if _scalar(event.fields, "start") == 1]
    stop_events = [event for event in command_events if _scalar(event.fields, "stop") == 1]
    claim_events = [event for event in command_events if _scalar(event.fields, "claim") == 1]

    assert native.claimed_session is not None
    assert native.claim_received_at is not None
    assert native.ack_sent_at is not None
    assert native.ack_sent_at - native.claim_received_at >= ACK_DELAY_S
    assert claim_events
    assert start_events
    assert len(stop_events) >= 5
    assert len(planner_events) >= 5

    # Claim is allowed before ACK; actuation and planner data are not.
    assert all(event.received_at >= native.ack_sent_at for event in start_events)
    assert all(event.received_at >= native.ack_sent_at for event in planner_events)

    for event in command_events + planner_events:
        assert event.version == 2
        assert _token(event.fields, "receiver_epoch") == RECEIVER_EPOCH
        assert _token(event.fields, "publisher_session") == native.claimed_session

    command_indices = [_scalar(event.fields, "command_index") for event in command_events]
    logical_command_indices = [
        value for index, value in enumerate(command_indices) if index == 0 or value != command_indices[index - 1]
    ]
    assert all(
        current > previous
        for previous, current in zip(
            logical_command_indices,
            logical_command_indices[1:],
        )
    )

    planner_indices = [_scalar(event.fields, "frame_index") for event in planner_events]
    assert all(
        current > previous
        for previous, current in zip(
            planner_indices,
            planner_indices[1:],
        )
    )

    assert synthetic_xrt.freeze_monotonic is not None
    stop_delay = stop_events[0].received_at - synthetic_xrt.freeze_monotonic
    assert 0.45 <= stop_delay <= 0.75
    assert "[Manager] Native deployment ownership claimed" in output
    assert "[Manager] StreamMode switch: OFF -> PLANNER" in output
    stale_match = re.search(
        r"\[Manager\] FAIL-SAFE: tracking unavailable while active "
        r"\(sample is stale \((\d+\.\d{3})s > 0\.500s\)\); sending stop",
        output,
    )
    assert stale_match is not None
    assert float(stale_match.group(1)) >= 0.5
