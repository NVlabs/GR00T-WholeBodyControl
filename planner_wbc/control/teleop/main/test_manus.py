import pickle
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import zmq


def plot_fingertips(data, ax):
    """Plot specific fingertip positions of left and right fingers in 3D space."""

    left_fingertips = data["left_fingers"]["position"]
    right_fingertips = data["right_fingers"]["position"]

    # Extract X, Y, Z positions of fingertips from the transformation matrices
    left_positions = np.array([finger[:3, 3] for finger in left_fingertips])
    right_positions = np.array([finger[:3, 3] for finger in right_fingertips])

    # Ensure the positions are 2D arrays (N, 3)
    left_positions = np.reshape(left_positions, (-1, 3))  # Ensure 2D array with shape (N, 3)
    right_positions = np.reshape(right_positions, (-1, 3))  # Ensure 2D array with shape (N, 3)

    # Create a 3D plot
    ax.cla()  # Clear the previous plot

    # Plot selected left fingertips (use red color)
    ax.scatter(
        left_positions[:, 0],
        left_positions[:, 1],
        left_positions[:, 2],
        c="r",
        label="Left Fingers",
    )
    ax.scatter(
        right_positions[:, 0],
        right_positions[:, 1],
        right_positions[:, 2],
        c="b",
        label="Right Fingers",
    )
    # ax.scatter(avp_left_fingers[:, 0], avp_left_fingers[:, 1], avp_left_fingers[:, 2], c='b', label='AVP')

    # Set plot labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim([-0.3, 0.3])  # Set X-axis limits, adjust to your range
    ax.set_ylim([-0.3, 0.3])  # Set Y-axis limits, adjust to your range
    ax.set_zlim([-0.3, 0.3])  # Set Z-axis limits, adjust to your range

    # Add a legend
    ax.legend()

    # Display the plot (update it)
    plt.draw()
    plt.pause(0.00001)


class ManusVisClient:
    def __init__(self, port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)  # Use REQ instead of SUB
        self.socket.connect(f"tcp://localhost:{port}")

    def request_data(self):
        """Request the latest data from the server."""
        # Send request to the server
        self.socket.send(b"request_data")  # Send a request message

        # Wait for the server's response
        message = self.socket.recv()  # Receive response
        data = pickle.loads(message)  # Deserialize the data

        return data


@click.command()
@click.option("--port", type=int, default=5556)
def main(port):
    print("==>start test manus")
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    client = ManusVisClient(port=port)

    iter_idx = 0
    while True:
        data = client.request_data()
        plot_fingertips(data, ax)

        time.sleep(0.1)
        iter_idx += 1
        if iter_idx > 100:
            break


# Run the client
if __name__ == "__main__":
    main()
