"""Non-blocking ZMQ subscriber that keeps only the latest message (CONFLATE mode)."""

import zmq


class ZMQPoller:
    """Simple ZMQ subscriber for sporadic non-blocking reads."""

    def __init__(self, host: str = "localhost", port: int = 5555, topic: str = ""):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(f"tcp://{host}:{port}")
        self._topic = topic

    def __del__(self):
        self.close()

    def get_data(self):
        """Get latest data or None if no data available."""
        if self._socket.poll(timeout=0):
            data = self._socket.recv(zmq.NOBLOCK)
            if data is None:
                print("ZMQPoller: no data received")
                return None

            # Strip topic prefix
            return data[len(self._topic) :]

        print("ZMQPoller: no data available")
        return None

    def close(self):
        self._socket.close()
        self._context.term()
