import pickle
import threading

import zmq

from decoupled_wbc.control.teleop.device.manus import Manus
from decoupled_wbc.control.teleop.streamers.base_streamer import BaseStreamer, StreamerOutput


class ManusStreamer(BaseStreamer):
    def __init__(self, port=5556):
        self.port = port
        self.context = None
        self.socket = None
        self.manus_server = None
        self.server_thread = None

    def request_data(self):
        """Request the latest data from the server."""
        if self.socket is None:
            raise RuntimeError("ManusStreamer not started. Call start_streaming() first.")

        # Send request to the server
        self.socket.send(b"request_data")  # Send a request message

        # Wait for the server's response
        message = self.socket.recv()  # Receive response
        data = pickle.loads(message)  # Deserialize the data

        return data

    def _run_server(self):
        """Run the manus server in a separate thread."""
        with self.manus_server.activate():
            print("Manus server activated")
            self.manus_server.run()

    def start_streaming(self):
        """Start the manus server and establish connection."""
        if self.manus_server is not None:
            return

        print(f"Starting manus server on port {self.port}...")

        # Create manus server instance
        self.manus_server = Manus(port=self.port)

        # Start server in separate thread
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

        # Establish ZMQ connection
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{self.port}")

    def get(self):
        """Return hand tracking data as StreamerOutput."""
        raw_data = self.request_data()

        # Initialize IK data (ik_keys) - Manus provides hand/finger tracking
        ik_data = {}
        if isinstance(raw_data, dict):
            # Extract finger and hand pose data
            for key, value in raw_data.items():
                if "finger" in key.lower() or "hand" in key.lower():
                    ik_data[key] = value

        # Return structured output - Manus only provides IK data
        return StreamerOutput(
            ik_data=ik_data,
            control_data={},  # No control commands from Manus
            teleop_data={},  # No teleop commands from Manus
            source="manus",
        )

    def stop_streaming(self):
        """Stop the manus server and close connections."""
        # Close ZMQ connection
        if self.socket:
            self.socket.close()
            self.socket = None

        if self.context:
            self.context.term()
            self.context = None

        # Reset server references
        self.manus_server = None
        self.server_thread = None
