"""Host-side non-blocking ZMQ telemetry receiver."""

from __future__ import annotations

from collections import deque
import time

import zmq

from .protocol import TOPIC, decode_sample


class TelemetryReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        queue_seconds: float = 120.0,
        expected_hz: float = 50.0,
    ):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, max(1000, int(queue_seconds * expected_hz)))
        self._socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
        self._socket.connect(f"tcp://{host}:{port}")
        self._raw = deque()
        self._latest = None
        self._decode_errors = 0
        print(f"[G1Telemetry] connected to tcp://{host}:{port}")

    def poll(self, max_messages: int = 1000) -> int:
        count = 0
        while count < max_messages:
            try:
                topic, payload = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if topic != TOPIC:
                continue
            try:
                sample = decode_sample(payload)
            except Exception as exc:
                self._decode_errors += 1
                if self._decode_errors <= 3 or self._decode_errors % 100 == 0:
                    print(f"[G1Telemetry] invalid packet: {exc}")
                continue
            sample["host_receive_timestamp_ns"] = time.time_ns()
            sample["host_receive_monotonic_ns"] = time.monotonic_ns()
            self._latest = sample
            self._raw.append(sample)
            count += 1
        return count

    def latest(self, max_age_sec: float) -> dict | None:
        self.poll()
        if self._latest is None:
            return None
        age = (time.monotonic_ns() - self._latest["host_receive_monotonic_ns"]) * 1e-9
        return self._latest if age <= max_age_sec else None

    def drain_raw(self) -> list[dict]:
        samples = list(self._raw)
        self._raw.clear()
        return samples

    def wait_until_ready(self, timeout_sec: float) -> dict:
        deadline = None if timeout_sec == 0 else time.monotonic() + timeout_sec
        while deadline is None or time.monotonic() < deadline:
            sample = self.latest(max_age_sec=1.0)
            if sample is not None:
                return sample
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for G1 upper-body telemetry")

    def close(self) -> None:
        self._socket.close()
        self._context.term()
