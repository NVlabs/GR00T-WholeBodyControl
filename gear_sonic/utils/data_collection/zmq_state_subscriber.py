"""
ZMQ utilities for subscribing to robot state and config from the C++ deploy process.

Provides:
- ``ZMQStateSubscriber`` — non-blocking SUB on the ``g1_debug`` topic
- ``poll_robot_config_zmq`` — one-shot CONFIG topic reader
"""

import time

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

STATE_ZMQ_TOPIC = "g1_debug"
CONFIG_ZMQ_TOPIC = "robot_config"
DEFAULT_STATE_ZMQ_PORT = 5557


def _unpack_msgpack_zmq(raw: bytes, topic: str) -> dict:
    """Strip a ZMQ topic prefix and decode the msgpack payload."""
    payload = raw[len(topic):]
    return msgpack.unpackb(payload, raw=False)


def _convert_lists_to_numpy(data: dict) -> dict:
    """Convert list values in a dict to numpy arrays."""
    if not isinstance(data, dict):
        return data
    result = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            result[key] = np.array(value)
        elif isinstance(value, dict):
            result[key] = _convert_lists_to_numpy(value)
        else:
            result[key] = value
    return result


class ZMQStateSubscriber:
    """Non-blocking SUB on the ``g1_debug`` ZMQ topic for robot state.

    Uses ``zmq.CONFLATE`` so only the latest message is kept.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_STATE_ZMQ_PORT,
        topic: str = STATE_ZMQ_TOPIC,
    ):
        mnp.patch()
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self._topic = topic
        self._msg = None
        print(f"[ZMQStateSubscriber] Connected to tcp://{host}:{port} (topic: {topic})")

    def _poll(self):
        """Poll for latest message (non-blocking)."""
        try:
            raw = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return

        msg = _unpack_msgpack_zmq(raw, self._topic)
        msg = _convert_lists_to_numpy(msg)
        self._msg = msg

    def get_msg(self, clear: bool = True):
        """Return the latest state message (or ``None``)."""
        self._poll()
        msg = self._msg
        if clear:
            self._msg = None
        return msg

    def close(self):
        self._socket.close()
        self._ctx.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def poll_robot_config_zmq(host: str, port: int, timeout_sec: float = 0) -> dict:
    """Wait for the ``robot_config`` message from the C++ ZMQ publisher.

    The publisher re-sends the config every ~2 s, so this simply polls with a
    short receive timeout until a message arrives or *timeout_sec* elapses.

    Args:
        timeout_sec: Max seconds to wait. ``0`` means wait indefinitely.

    Returns the decoded config dict.

    Raises:
        TimeoutError: If *timeout_sec* > 0 and no message is received in time.
    """
    mnp.patch()
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, CONFIG_ZMQ_TOPIC)
    sub.setsockopt(zmq.RCVTIMEO, 500)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.connect(f"tcp://{host}:{port}")

    if timeout_sec > 0:
        print(f"[Config] Waiting up to {timeout_sec}s for robot_config on tcp://{host}:{port} ...")
    else:
        print(f"[Config] Waiting for robot_config on tcp://{host}:{port} ... is gear_sonic_deploy running?")
    deadline = (time.monotonic() + timeout_sec) if timeout_sec > 0 else None
    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"[Config] No robot_config received on tcp://{host}:{port} "
                    f"within {timeout_sec}s. Is the C++ deploy process running?"
                )
            try:
                raw = sub.recv()
                config = _unpack_msgpack_zmq(raw, CONFIG_ZMQ_TOPIC)
                print(f"[Config] Received robot_config ({len(config)} fields)")
                return config
            except zmq.Again:
                pass
    finally:
        sub.close()
        ctx.term()
