import json
import threading

from flask import Flask
import socketio


class IPhoneDevice:
    """
    Returns the absolute pose of the iPhone device relative to the world frame.
    The coordinate system is defined when ARKit is initialized.

    Absolute pose coordinate system (at ARKit start):

    y (out of screen)
    ⊙---> x
    |
    |
    ↓ z


    If you compute the relative pose (T_0_inv @ T_now), the coordinate system becomes:

    z (out of screen)
    ⊙---> y
    |
    |
    ↓ x

    """

    def __init__(self, port: int = 5557, silent: bool = True):
        self._silent = silent
        self._port = port
        self._latest_transform: dict = {}
        self._latest_speed: dict = {}
        self._commands: list[str] = []

        # Use threading mode for socketio
        self._sio = socketio.Server(async_mode="threading", cors_allowed_origins="*")
        self._app = Flask(__name__)
        self._app.wsgi_app = socketio.WSGIApp(self._sio, self._app.wsgi_app)

        # Set up the event handler for updates
        @self._sio.event
        def connect(sid, environ):
            if not self._silent:
                print(f"===============>Client connected: {sid}")

        @self._sio.event
        def disconnect(sid):
            if not self._silent:
                print(f"===============>Client disconnected: {sid}")

        @self._sio.event
        def update(sid, data):
            try:
                data = json.loads(data)
                self._latest_transform = data
                self._sio.emit("commands", json.dumps(self._commands), to=sid)
            except Exception as e:
                if not self._silent:
                    print(f"Update failed: {e}")

    def _run_server(self):
        """Run the Flask server with threading."""
        self._app.run(host="0.0.0.0", port=self._port, threaded=True)

    def start(self):
        """Start the server in a background thread."""
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()
        if not self._silent:
            print(f"IPhone Device running at http://0.0.0.0:{self._port}")

    def stop(self):
        if not self._silent:
            print("IPhone Device stopped.")

    def get_cmd(self) -> dict:
        return self._latest_transform

    def send_cmd(self, enable: bool) -> None:
        self._commands = ["start_haptics" if enable else "stop_haptics"]


if __name__ == "__main__":
    import time

    device = IPhoneDevice()
    device.start()

    # or _ in range(100):
    try:
        while True:
            data = device.get_cmd()
            print("Latest data:", data)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping device...")
    finally:
        device.stop()
