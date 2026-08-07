import click
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from decoupled_wbc.control.teleop.streamers.vive_streamer import ViveStreamer


@click.command()
@click.option("--ip", default="192.168.0.182", help="IP address of the Vive Tracker")
@click.option("--port", default=5555, help="Port number of the Vive Tracker")
@click.option("--keyword", default="elbow", help="Keyword to filter the tracker data")
def main(ip, port, keyword):
    print("==>start test vive")
    streamer = ViveStreamer(ip=ip, port=port, fps=20, keyword=keyword)
    streamer.start_streaming()

    # Create figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    pos_data = {}
    pose_data_init = {}
    ax_line = {}
    for dir in ["left", "right"]:
        pose_data_init[dir] = None
        pos_data[dir] = list()
        # Initialize a line object that will be updated in the animation
        color = "r" if dir == "left" else "b"
        ax_line[dir] = ax.plot([], [], [], f"{color}-", marker="o")[0]

    # Set plot limits (adjust as needed)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Set axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Define the initialization function for the animation
    def init():
        for dir in ["left", "right"]:
            ax_line[dir].set_data([], [])
            ax_line[dir].set_3d_properties([])

        return ax_line["left"], ax_line["right"]

    # Define the update function that will update the trajectory in real-time
    def update(num):
        streamer_output = streamer.get()

        # Handle case where no data is available yet
        if not streamer_output or not streamer_output.ik_data:
            return ax_line["left"], ax_line["right"]

        for dir in ["left", "right"]:
            wrist_key = f"{dir}_wrist"
            if wrist_key not in streamer_output.ik_data:
                continue

            raw_pose = streamer_output.ik_data[wrist_key]
            if pose_data_init[dir] is None:
                pose_data_init[dir] = raw_pose
            relative_pose = np.linalg.inv(pose_data_init[dir]) @ raw_pose
            pos_data[dir].append(relative_pose[:3, 3])
            pos_data_np = np.array(pos_data[dir])
            ax_line[dir].set_data(pos_data_np[:, 0], pos_data_np[:, 1])
            ax_line[dir].set_3d_properties(pos_data_np[:, 2])
        return ax_line["left"], ax_line["right"]

    # Create the animation
    _ = animation.FuncAnimation(fig, update, init_func=init, frames=20, interval=50, blit=True)

    # Show the plot
    plt.show(block=False)
    plt.pause(5)


if __name__ == "__main__":
    main()
