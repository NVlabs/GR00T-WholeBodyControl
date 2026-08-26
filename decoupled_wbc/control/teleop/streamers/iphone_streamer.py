import concurrent.futures
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from decoupled_wbc.control.teleop.device.iphone.iphone import IPhoneDevice
from decoupled_wbc.control.teleop.streamers.base_streamer import BaseStreamer, StreamerOutput


def get_data_with_timeout(obj, timeout=0.02):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(obj.data_collect.request_vive_data)
        try:
            combined_data = future.result(timeout=timeout)
            return combined_data
        except concurrent.futures.TimeoutError:
            print("Data request timed out.")
            return None


def get_transformation(vive_raw_data):
    """
    Turn the raw data from the Vive tracker into a transformation matrix.
    """
    position = np.array(
        [
            vive_raw_data["position"]["x"],
            vive_raw_data["position"]["y"],
            vive_raw_data["position"]["z"],
        ]
    )
    quat = np.array(
        [
            vive_raw_data["orientation"]["x"],
            vive_raw_data["orientation"]["y"],
            vive_raw_data["orientation"]["z"],
            vive_raw_data["orientation"]["w"],
        ]
    )
    T = np.identity(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = position
    return T


class IphoneStreamer(BaseStreamer):
    def __init__(self):
        self.left_device = IPhoneDevice(port=5557)
        self.right_device = IPhoneDevice(port=5558)
        self.left_prev_button_states = {"Reset": False, "Close": False}
        self.right_prev_button_states = {"Reset": False, "Close": False}

    def start_streaming(self):
        self.left_device.start()
        self.right_device.start()

    def __del__(self):
        self.stop_streaming()

    def get(self):
        """Request combined data and return transformations as StreamerOutput."""
        # Initialize data groups
        ik_data = {}  # For pose and joint data (ik_keys)
        control_data = {}  # For robot control commands (control_keys)
        teleop_data = {}  # For internal policy commands (teleop_keys)

        try:
            # Request combined data from the server and wait until we get data
            left_combined_data = None
            right_combined_data = None
            while not left_combined_data:
                left_combined_data = self.left_device.get_cmd()
                if not left_combined_data:
                    print("Waiting for left iPhone data...")
            while not right_combined_data:
                right_combined_data = self.right_device.get_cmd()
                if not right_combined_data:
                    print("Waiting for right iPhone data...")

            # IK data - wrist poses and finger positions (ik_keys)
            ik_data["left_wrist"] = np.array(left_combined_data.get("transformMatrix"))
            ik_data["right_wrist"] = np.array(right_combined_data.get("transformMatrix"))

            # left button states
            current_left_reset = left_combined_data.get("buttonStates").get("Reset")
            current_left_close = left_combined_data.get("buttonStates").get("Close")

            # Trigger logic: only True when button transitions from False to True
            left_reset = current_left_reset and not self.left_prev_button_states["Reset"]

            # Store current button states for next iteration
            self.left_prev_button_states["Reset"] = current_left_reset
            self.left_prev_button_states["Close"] = current_left_close

            # left fingers - IK data (ik_keys)
            fingertips = np.zeros([25, 4, 4])
            positions = fingertips[:, :3, 3]
            if current_left_close:
                positions[4, 0] = 0  # closed
            else:
                positions[4, 0] = 1  # open
            ik_data["left_fingers"] = {"position": fingertips}

            # right button states
            current_right_reset = right_combined_data.get("buttonStates").get("Reset")
            current_right_close = right_combined_data.get("buttonStates").get("Close")

            # Trigger logic: only True when button transitions from False to True
            right_reset = current_right_reset and not self.right_prev_button_states["Reset"]

            # Store current button states for next iteration
            self.right_prev_button_states["Reset"] = current_right_reset
            self.right_prev_button_states["Close"] = current_right_close

            # right fingers - IK data (ik_keys)
            fingertips = np.zeros([25, 4, 4])
            positions = fingertips[:, :3, 3]
            if current_right_close:
                positions[4, 0] = 0  # closed
            else:
                positions[4, 0] = 1  # open
            ik_data["right_fingers"] = {"position": fingertips}

            # Teleop commands (teleop_keys) - used by TeleopPolicy for activation
            teleop_data["toggle_activation"] = left_reset

            # Control commands (control_keys) - sent to robot
            control_data["toggle_stand_command"] = right_reset

        except Exception as e:
            print(f"Error while requesting iPhone data: {e}")

        # Return structured output
        return StreamerOutput(
            ik_data=ik_data, control_data=control_data, teleop_data=teleop_data, source="iphone"
        )

    def stop_streaming(self):
        self.left_device.stop()
        self.right_device.stop()


if __name__ == "__main__":
    streamer = IphoneStreamer()
    streamer.start_streaming()
    while True:
        data = streamer.get()
        print(data)
        time.sleep(0.1)
