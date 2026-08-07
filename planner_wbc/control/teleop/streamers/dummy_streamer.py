import numpy as np

from decoupled_wbc.control.teleop.streamers.base_streamer import BaseStreamer, StreamerOutput
from decoupled_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber


class DummyStreamer(BaseStreamer):
    """A dummy streamer that returns hardcoded structured data for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_streaming = False
        self.keyboard_listener = KeyboardListenerSubscriber()
        self.left_fingertips = np.zeros((25, 4, 4))
        self.right_fingertips = np.zeros((25, 4, 4))
        self.left_fingertips[4, 0, 3] = 1.0  # open
        self.right_fingertips[4, 0, 3] = 1.0  # open

    def start_streaming(self):
        self.is_streaming = True
        print("Dummy streamer started with hardcoded IK data")

    def get(self) -> StreamerOutput:
        """Return hardcoded dummy data with proper IK format."""
        if not self.is_streaming:
            return StreamerOutput()

        # Hardcoded dummy data - identity matrices and zeros
        left_wrist_pose = np.eye(4)
        right_wrist_pose = np.eye(4)
        left_fingers, right_fingers = self._generate_finger_data()

        return StreamerOutput(
            ik_data={
                "left_wrist": left_wrist_pose,
                "right_wrist": right_wrist_pose,
                "left_fingers": {"position": left_fingers},
                "right_fingers": {"position": right_fingers},
            },
            control_data={
                "navigate_cmd": [0.0, 0.0, 0.0],  # No movement
                "toggle_stand_command": False,
                "base_height_command": 0.78,  # Default standing height
            },
            teleop_data={
                "toggle_activation": False,
                "freeze_upper_body": False,
            },
            source="dummy",
        )

    def stop_streaming(self):
        self.is_streaming = False
        print("Dummy streamer stopped")

    def _generate_finger_data(self):
        """Generate finger position data similar to iPhone streamer."""

        # Control thumb based on shoulder button state (index 4 is thumb tip)
        key = self.keyboard_listener.read_msg()
        index = 5
        middle = 10
        ring = 15
        if key is not None:
            if key == "b":
                if self.left_fingertips[4 + index, 0, 3] == 1.0:
                    self.left_fingertips[4 + index, 0, 3] = 0.0  # open
                else:
                    self.left_fingertips[4 + index, 0, 3] = 1.0  # closed
            if key == "n":
                if self.left_fingertips[4 + middle, 0, 3] == 1.0:
                    self.left_fingertips[4 + middle, 0, 3] = 0.0  # open
                else:
                    self.left_fingertips[4 + middle, 0, 3] = 1.0  # closed
            if key == "m":
                if self.left_fingertips[4 + ring, 0, 3] == 1.0:
                    self.left_fingertips[4 + ring, 0, 3] = 0.0  # open
                else:
                    self.left_fingertips[4 + ring, 0, 3] = 1.0  # closed

        return self.left_fingertips, self.right_fingertips
