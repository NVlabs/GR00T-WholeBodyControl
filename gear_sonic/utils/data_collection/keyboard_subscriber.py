"""ZMQ-based keyboard subscriber for data collection control."""

import zmq

DEFAULT_ZMQ_KEYBOARD_PORT = 5580


class ZMQKeyboardSubscriber:
    """Receives keyboard events from ZMQ SUB socket (non-blocking)."""

    def __init__(self, port: int = DEFAULT_ZMQ_KEYBOARD_PORT, host: str = "localhost"):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self._data = None
        print(f"[ZMQKeyboardSubscriber] Connected to tcp://{host}:{port}")

    def read_msg(self):
        """Return the latest key press (or None)."""
        try:
            self._data = self._socket.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            pass
        data = self._data
        self._data = None
        return data

    def close(self):
        self._socket.close()
        self._ctx.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
