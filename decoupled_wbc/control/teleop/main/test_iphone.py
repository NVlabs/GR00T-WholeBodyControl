import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from decoupled_wbc.control.teleop.device.iphone.iphone import IPhoneDevice


class IphonePublisher:
    def __init__(self, port: int = 5555, silent: bool = True):
        self._device = IPhoneDevice(port=port, silent=silent)

        # record the initial transform
        self._init_transform = None
        self._init_transform_inverse = None

        # record the latest transform and timestamp
        self._latest_transform = None
        self._latest_timestamp = None

        # publishing variables
        self._velocity = [0, 0, 0]
        self._position = [0, 0, 0]

        # visualization
        self._fig = None
        self._ax = None
        self._setup_visualization()

    def _draw_axes(self, ax):
        """Draw coordinate axes, labels, and legend on the given axes object."""
        origin = [0, 0, 0]
        axis_length = 0.3
        # X axis (right)
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axis_length,
            0,
            0,
            color="r",
            arrow_length_ratio=0.1,
            linewidth=2,
        )
        # Y axis (out/forward)
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            0,
            axis_length,
            0,
            color="b",
            arrow_length_ratio=0.1,
            linewidth=2,
        )
        # Z axis (up)
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            0,
            0,
            axis_length,
            color="g",
            arrow_length_ratio=0.1,
            linewidth=2,
        )
        ax.text(axis_length, 0, 0, "X", color="red")
        ax.text(0, axis_length, 0, "Y", color="blue")
        ax.text(0, 0, axis_length, "Z", color="green")
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="r", lw=2, label="X axis (right)"),
            Line2D([0], [0], color="g", lw=2, label="Y axis (out)"),
            Line2D([0], [0], color="b", lw=2, label="Z axis (up)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    def _draw_cube(self, ax):
        """Draw a 1x1x1 meter cube centered at the origin on the given axes object."""
        r = [-0.5, 0.5]
        corners = [
            [r[0], r[0], r[0]],
            [r[0], r[0], r[1]],
            [r[0], r[1], r[0]],
            [r[0], r[1], r[1]],
            [r[1], r[0], r[0]],
            [r[1], r[0], r[1]],
            [r[1], r[1], r[0]],
            [r[1], r[1], r[1]],
        ]
        edges = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]
        for i, j in edges:
            x = [corners[i][0], corners[j][0]]
            y = [corners[i][1], corners[j][1]]
            z = [corners[i][2], corners[j][2]]
            ax.plot3D(x, y, z, "gray")

    def _setup_visualization(self):
        """Setup matplotlib figure for 6D pose visualization"""
        plt.ion()  # Enable interactive mode
        self._fig = plt.figure(figsize=(8, 8))
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_xlim([-0.5, 0.5])
        self._ax.set_ylim([-0.5, 0.5])
        self._ax.set_zlim([-0.5, 0.5])
        self._ax.set_xlabel("X (right)")
        self._ax.set_ylabel("Y (out)")
        self._ax.set_zlabel("Z (up)")
        self._ax.set_title("iPhone 6D Pose")
        self._ax.view_init(elev=30, azim=-45)
        self._draw_cube(self._ax)
        self._draw_axes(self._ax)
        plt.tight_layout()
        self._fig.canvas.draw()
        plt.pause(0.001)

    def start(self):
        self._device.start()

    def stop(self):
        self._device.stop()
        if self._fig is not None:
            plt.close(self._fig)

    def reset_transform(self, *args, timeout=5.0, poll_interval=0.1):
        """
        Wait until device returns data, then set the initial transform.

        Args:
            timeout (float): Maximum time to wait in seconds.
            poll_interval (float): Time between checks in seconds.
        """
        start_time = time.time()
        while True:
            data = self._device.get_cmd()
            if data:
                self._init_transform = np.array(data.get("transformMatrix"))
                self._init_transform_inverse = np.linalg.inv(self._init_transform)
                print(
                    "initial position", [round(v, 4) for v in self._init_transform[:3, 3].tolist()]
                )
                self._latest_transform = self._init_transform.copy()
                self._latest_timestamp = data.get("timestamp")
                print("Initial transform set.")
                break
            elif time.time() - start_time > timeout:
                print("Timeout: Failed to get initial transform data after waiting.")
                break
            else:
                time.sleep(poll_interval)

        return {"success": True, "message": "Triggered!"}

    def update_transfrom(self) -> dict:
        data = self._device.get_cmd()

        if data:
            transform_matrix = np.array(data.get("transformMatrix"))
            position = transform_matrix[:3, 3]
            rotation = transform_matrix[:3, :3]

            # Create a single multiline string with position and rotation data
            output = f"\r\033[KPosition: {[round(v, 4) for v in position.tolist()]}\n"
            output += "\033[KRotation matrix:\n"
            for i in range(3):
                row = [round(v, 4) for v in rotation[i].tolist()]
                output += f"\033[K    {row}\n"

            # Move cursor up 5 lines (position + "rotation matrix:" + 3 rows)
            # and print the entire output at once
            print(f"\033[5A{output}", end="", flush=True)

            # draw 6d pose
            self._visualize_pose(position, rotation)

        if data:
            current_transform = np.array(data.get("transformMatrix")) @ self._init_transform_inverse
            # print("current_transform", [round(v, 4) for v in current_transform.flatten().tolist()])
            current_timestamp = data.get("timestamp")

            # Check if the current timestamp is the same as the latest timestamp (No update or disconnect)
            if current_timestamp == self._latest_timestamp:
                return {
                    "transform_matrix": self._latest_transform,
                    "velocity": [0, 0, 0],
                    "position": self._latest_transform[:3, 3].tolist(),
                    "timestamp": self._latest_timestamp,
                }

            if self._latest_transform is not None:
                current_position = current_transform[:3, 3]
                current_velocity = (current_position - self._latest_transform[:3, 3]) / (
                    current_timestamp - self._latest_timestamp
                )
                self._velocity = current_velocity.tolist()
                self._position = current_position.tolist()

            self._latest_transform = current_transform.copy()
            self._latest_timestamp = current_timestamp

        return {
            "transform_matrix": self._latest_transform,
            "velocity": self._velocity,
            "position": self._position,
            "timestamp": self._latest_timestamp,
        }

    def _visualize_pose(self, position, rotation):
        """Visualize the 6D pose using matplotlib"""
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._setup_visualization()
        self._ax.clear()
        self._ax.set_xlim([-0.5, 0.5])
        self._ax.set_ylim([-0.5, 0.5])
        self._ax.set_zlim([-0.5, 0.5])
        self._ax.set_xlabel("X (right)")
        self._ax.set_ylabel("Y (out)")
        self._ax.set_zlabel("Z (up)")
        self._ax.set_title("iPhone 6D Pose")
        self._ax.view_init(elev=30, azim=-45)
        self._draw_cube(self._ax)
        self._draw_axes(self._ax)
        # Draw position as a marker
        pos = np.clip(position, -0.5, 0.5)
        self._ax.scatter(pos[0], pos[1], pos[2], color="red", s=100, marker="o")
        # Draw orientation axes
        length = 0.2
        colors = ["r", "b", "g"]  # X=red, Y=blue, Z=green
        for i in range(3):
            direction = rotation[:, i] * length
            self._ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                direction[0],
                direction[1],
                direction[2],
                color=colors[i],
                arrow_length_ratio=0.2,
                linewidth=2,
            )
        self._fig.canvas.draw()
        plt.pause(0.001)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=20, help="Frames per second (default: 20)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for waiting for the device to connect (default: 30 seconds)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    fps = args.fps
    timeout = args.timeout

    # Initialize the iPhone publisher
    iphone_publisher = IphonePublisher(port=5555, silent=False)
    iphone_publisher.start()
    print("Waiting for device to connect...press the reset button on the device to start")
    iphone_publisher.reset_transform(
        timeout=timeout
    )  # wait x seconds for the device to return data

    def update_and_visualize():
        # Update the transform and get the latest data
        data = iphone_publisher.update_transfrom()

        # print the data for debugging
        if args.debug and np.random.rand() < 1 / fps:
            print("data", data)

    try:
        while True:
            update_and_visualize()
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        print("Stopping device...")
    finally:
        iphone_publisher.stop()


if __name__ == "__main__":
    main()
