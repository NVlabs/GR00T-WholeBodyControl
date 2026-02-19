from contextlib import contextmanager
import pickle
import time

import click
import numpy as np
from scipy.spatial.transform import Rotation as R
import zmq

from decoupled_wbc.control.teleop.device.SDKClient_Linux import ManusServer

manus_idx = {
    "left": ["3822396207", "3998055887", "432908014"],
    "right": ["3762867141", "831307785", "3585023564"],
}


class Manus:
    def __init__(self, port=5556):
        # Storage for the latest finger data
        self.latest_finger_data = {
            "left_fingers": {"angle": np.zeros([40]), "position": np.zeros([25, 4, 4])},
            "right_fingers": {"angle": np.zeros([40]), "position": np.zeros([25, 4, 4])},
        }
        self.manus_server = None
        self.port = port

    def process_finger_pose(self, raw_position, raw_orientation, dir):
        raw_position = np.asarray(raw_position).reshape([-1, 3])
        raw_orientation = np.asarray(raw_orientation).reshape([-1, 4])

        def reorder(data):
            return np.concatenate(
                [
                    data[0:1],  # root
                    data[21:25],  # thumb
                    data[1:6],  # index
                    data[6:11],  # middle
                    data[16:21],  # ring
                    data[11:16],  # pinky
                ]
            )

        raw_position = reorder(raw_position)
        raw_orientation = reorder(raw_orientation)

        transformation_matrix = np.zeros([25, 4, 4])
        rot_matrix = R.from_quat(raw_orientation[0][[1, 2, 3, 0]]).as_matrix()
        transformation_matrix[:, :3, :3] = rot_matrix
        transformation_matrix[:, :3, 3] = raw_position
        transformation_matrix[:, 3, 3] = 1.0

        T_root = transformation_matrix[0]
        T_root_inv = np.linalg.inv(T_root)
        transformation_matrix = np.matmul(T_root_inv[None], transformation_matrix)

        T_manus2avp = np.identity(4)
        if dir == "right":
            T_manus2avp[:3, :3] = R.from_euler("zx", [180, -90], degrees=True).as_matrix()
        else:
            T_manus2avp[:3, :3] = R.from_euler("x", [90], degrees=True).as_matrix()
        transformation_matrix = np.matmul(T_manus2avp[None], transformation_matrix)

        return transformation_matrix

    def process_finger_angle(self, raw_finger_data, dir):
        if dir == "right":
            non_zero_data = [value for value in raw_finger_data if value != 0.0]
            trailing_zeros_count = len(raw_finger_data) - len(non_zero_data)
            raw_finger_data = non_zero_data + [0.0] * trailing_zeros_count

        return raw_finger_data

    def request_finger_data(self):
        output = ManusServer.get_latest_state()
        for dir, val in manus_idx.items():
            for id in val:
                angle_name = f"{id}_angle"
                if angle_name in output and len(output[angle_name]) > 0:
                    self.latest_finger_data[f"{dir}_fingers"]["angle"] = self.process_finger_angle(
                        np.asarray(output[angle_name]), dir
                    )

                position_name = f"{id}_position"
                orientation_name = f"{id}_orientation"
                if (
                    position_name in output
                    and orientation_name in output
                    and len(output[position_name]) > 0
                    and len(output[orientation_name]) > 0
                ):
                    self.latest_finger_data[f"{dir}_fingers"]["position"] = (
                        self.process_finger_pose(
                            output[position_name], output[orientation_name], dir
                        )
                    )

        return self.latest_finger_data

    @contextmanager
    def activate(self):
        try:
            ManusServer.init()
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{self.port}")
            yield self
        finally:
            ManusServer.shutdown()

    def run(self):
        while True:
            # Wait for a request from the client
            _ = self.socket.recv()
            # Process finger data
            data = self.request_finger_data()
            data["timestamp"] = time.time()

            # Serialize the data to send back
            serialized_data = pickle.dumps(data)

            # Send the serialized data back to the client
            self.socket.send(serialized_data)


@click.command()
@click.option("--port", type=int, default=5556)
def main(port):
    print("... starting manus server ... at port", port, flush=True)
    manus = Manus(port=port)
    print("... manus server activating ...")
    with manus.activate():
        print("==> Manus server is running at port", port, flush=True)
        manus.run()


if __name__ == "__main__":
    main()
