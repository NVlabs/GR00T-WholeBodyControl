import concurrent.futures
import json

import numpy as np
from scipy.spatial.transform import Rotation as R
import zmq

from .base_streamer import BaseStreamer, StreamerOutput


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


class DataCollectorClient:
    def __init__(self, vive_tracker_address):
        # Create a ZeroMQ context
        self.context = zmq.Context()

        # Create a REQ socket to request data from the server
        self.vive_socket = self.context.socket(zmq.REQ)
        self.vive_socket.connect(vive_tracker_address)  # Connect to the server address
        self.latest_data = None

    def request_vive_data(self):
        """Request combined data for both left and right wrists."""
        try:
            # Send a request to the server asking for both left and right Vive tracker data
            self.vive_socket.send_string("get_vive_data")

            # Receive the response from the server
            message = self.vive_socket.recv_string()

            # Parse the received JSON string into a Python dictionary
            data = json.loads(message)

            # print(f"Received combined tracker data:", data)
            return data

        except zmq.ZMQError as e:
            print(f"ZMQ Error while requesting Vive data: {e}")

    def stop(self):
        """Stop the client and clean up resources."""
        try:
            # Close the socket and terminate the context
            self.vive_socket.close()  # Close the socket
            self.context.term()  # Terminate the context
            print("Client stopped successfully.")
        except zmq.ZMQError as e:
            print(f"Error while stopping client: {e}")


class ViveStreamer(BaseStreamer):
    def __init__(self, ip, port=5555, fps=20, keyword=None, **kwargs):
        self.ip = f"tcp://{ip}:{port}"
        self.fps = fps
        self.latest_data = None
        self.stop_thread = False
        self.update_thread = None
        self.keyword = keyword

    def reset_status(self):
        """Reset the cache of the streamer."""
        self.latest_data = None

    def start_streaming(self):
        self.data_collect = DataCollectorClient(self.ip)

    def __del__(self):
        self.stop_streaming()

    def get(self):
        """Request combined data and return transformations as StreamerOutput."""
        # Initialize IK data (ik_keys) - Vive only provides pose data
        ik_data = {}

        try:
            # Request combined data from the server
            combined_data = self.data_collect.request_vive_data()
            if combined_data:
                for dir in ["left", "right"]:
                    actual_name = f"{dir}_{self.keyword}"
                    if actual_name in combined_data and combined_data[actual_name] is not None:
                        ik_data[f"{dir}_wrist"] = get_transformation(combined_data[actual_name])

        except zmq.ZMQError as e:
            print(f"ZMQ Error while requesting Vive data: {e}")

        # Return structured output - Vive only provides IK data
        return StreamerOutput(
            ik_data=ik_data,
            control_data={},  # No control commands from Vive
            teleop_data={},  # No teleop commands from Vive
            source="vive",
        )

    def stop_streaming(self):
        self.data_collect.stop()
