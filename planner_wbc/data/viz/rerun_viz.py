#!/usr/bin/env python3
"""Rerun visualization utilities for plotting data and images."""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb


class RerunViz:
    """Class for visualizing data using Rerun."""

    def __init__(
        self,
        image_keys: List[str],
        tensor_keys: List[str],
        app_name: str = "rerun_visualization",
        memory_limit: str = "1GB",
        window_size: float = 5.0,
        port: int = 9876,
        in_docker: bool = False,
    ):
        """Initialize the RerunViz class.
        Args:
            app_name: Name of the Rerun application
            memory_limit: Memory limit for Rerun
            window_size: Size of the time window in seconds
            image_keys: List of image keys to plot
            tensor_keys: List of tensor keys to plot
            in_docker: Whether running inside Docker container. If in docker,
                forward data to outside of the container to be rendered.
                Use `rerun --port 9876` to visualize. Expecting rerun-cli 0.22.1 outside of docker.
                Tested with rerun-sdk 0.21.0 inside docker.
        """
        self.app_name = app_name
        self.memory_limit = memory_limit
        self.window_size = window_size
        self.tensor_keys = tensor_keys
        self.image_keys = image_keys
        self.port = port
        self.in_docker = in_docker
        # Initialize Rerun
        self._initialize_rerun()

    def _initialize_rerun(self):
        """Initialize Rerun and set up the blueprint."""
        rr.init(self.app_name)
        if not self.in_docker:
            # support for web visualization
            rr.spawn(memory_limit=self.memory_limit, port=self.port, connect=True)
        else:
            # forward data to outside of the docker container
            rr.connect(f"127.0.0.1:{self.port}")
        self._create_blueprint()

    def _create_blueprint(self):
        # Create a grid of plots
        contents = []

        # Add time series plots
        for tensor_key in self.tensor_keys:
            contents.append(
                rrb.TimeSeriesView(
                    origin=tensor_key,
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            "time",
                            start=rrb.TimeRangeBoundary.cursor_relative(seconds=-self.window_size),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                )
            )

        # Add image views
        for image_key in self.image_keys:
            contents.append(rrb.Spatial2DView(origin=image_key, name=image_key))

        # Send the blueprint with collapsed panels to hide side/bottom bars
        rr.send_blueprint(rrb.Blueprint(rrb.Grid(contents=contents), collapse_panels=True))

    def set_rerun_keys(self, image_keys: List[str], tensor_keys: List[str]):
        """Set the Rerun keys."""
        self.image_keys = image_keys
        self.tensor_keys = tensor_keys
        self._create_blueprint()

    def plot_images(self, images: Dict[str, np.ndarray], timestamp: Optional[float] = None):
        """Plot image data.

        Args:
            images: Dictionary mapping image names to image data
            timestamp: Timestamp for the data (if None, uses current time)
        """
        if timestamp is None:
            timestamp = time.time()

        rr.set_time_seconds("time", timestamp)

        for key, image in images.items():
            if image is None:
                continue

            if "depth" in key:
                # Color jet
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(image, alpha=0.03), cv2.COLORMAP_JET
                )
                rr.log(f"{key}", rr.Image(depth_colormap))
            else:
                # Convert to RGB
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rr.log(f"{key}", rr.Image(image))

    def plot_tensors(
        self, data: Optional[Dict[str, np.ndarray]] = None, timestamp: Optional[float] = None
    ):
        """Plot tensor data.

        Args:
            data: Dictionary mapping keys to tensor values
            timestamp: Timestamp for the data (if None, uses current time)
        """
        if timestamp is None:
            timestamp = time.time()

        rr.set_time_seconds("time", timestamp)

        # If no data provided, use random walk generators
        for tensor_key in self.tensor_keys:
            for i in range(data[tensor_key].shape[0]):
                rr.log(f"{tensor_key}/{i}", rr.Scalar(data[tensor_key][i]))

    def close(self):
        """Close the RerunViz instance."""
        rr.rerun_shutdown()


if __name__ == "__main__":
    """Main function to demonstrate the RerunViz class."""
    parser = argparse.ArgumentParser(description="Plot dashboard stress test")
    parser.add_argument(
        "--freq", type=float, default=20, help="Frequency of logging (applies to all series)"
    )
    parser.add_argument(
        "--window-size", type=float, default=5.0, help="Size of the window in seconds"
    )
    parser.add_argument("--duration", type=float, default=60, help="How long to log for in seconds")
    parser.add_argument("--use-rs", action="store_true", help="Use RealSense sensor")
    parser.add_argument("--use-zed", action="store_true", help="Use ZED sensor")
    parser.add_argument("--in-docker", action="store_true", help="Running inside Docker container")
    args = parser.parse_args()

    if args.use_rs:
        image_keys = ["color_image", "depth_image"]
        from decoupled_wbc.control.sensor.realsense import RealSenseClientSensor

        sensor = RealSenseClientSensor()
    elif args.use_zed:
        image_keys = ["left_image", "right_image"]
        from decoupled_wbc.control.sensor.zed import ZEDClientSensor

        sensor = ZEDClientSensor()
    else:
        from decoupled_wbc.control.sensor.dummy import DummySensor

        sensor = DummySensor()
        image_keys = ["color_image"]

    tensor_keys = ["left_arm_qpos", "left_hand_qpos", "right_arm_qpos", "right_hand_qpos"]

    # Initialize the RerunViz class
    viz = RerunViz(
        image_keys=image_keys,
        tensor_keys=tensor_keys,
        window_size=args.window_size,
        in_docker=args.in_docker,
    )

    # Run the visualization loop
    cur_time = time.time()
    end_time = cur_time + args.duration
    time_per_tick = 1.0 / args.freq

    while cur_time < end_time:
        # Advance time and sleep if necessary
        cur_time += time_per_tick
        sleep_for = cur_time - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)

        if sleep_for < -0.1:
            print(f"Warning: missed logging window by {-sleep_for:.2f} seconds")

        # Plot dummy tensor
        dummy_tensor = np.random.randn(5)
        dummy_tensor_dict = {key: dummy_tensor for key in tensor_keys}

        viz.plot_tensors(dummy_tensor_dict, cur_time)

        # Plot images if available
        images = sensor.read()
        if images is not None:
            img_to_show = {key: images[key] for key in image_keys}
            viz.plot_images(img_to_show, cur_time)

    rr.script_teardown(args)
