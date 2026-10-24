import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import pyrealsense2 as rs

from decoupled_wbc.control.base.sensor import Sensor
from decoupled_wbc.control.sensor.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


@dataclass
class RealSenseConfig:
    """Configuration for the Intel RealSense camera."""

    color_image_dim: Tuple[int, int] = (640, 480)  # (width, height)
    fps: int = 30
    enable_color: bool = True
    enable_depth: bool = False
    serial_number: Optional[str] = None  # None = first available device
    mount_position: str = CameraMountPosition.EGO_VIEW.value


class RealSenseSensor(Sensor, SensorServer):
    """Sensor for Intel RealSense cameras (e.g., D435i on Unitree G1)."""

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: RealSenseConfig = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        if config is None:
            config = RealSenseConfig(mount_position=mount_position)

        self.config = config
        self.mount_position = mount_position
        self._run_as_server = run_as_server

        self.pipeline = rs.pipeline()
        rs_config = rs.config()

        if config.serial_number is not None:
            rs_config.enable_device(config.serial_number)

        w, h = config.color_image_dim
        if config.enable_color:
            rs_config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, config.fps)
        if config.enable_depth:
            rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, config.fps)

        profile = self.pipeline.start(rs_config)
        device = profile.get_device()
        print(f"Connected to RealSense: {device.get_info(rs.camera_info.name)} "
              f"(SN: {device.get_info(rs.camera_info.serial_number)})")

        # Align depth to color if both enabled
        self._align = rs.align(rs.stream.color) if config.enable_depth and config.enable_color else None

        if run_as_server:
            self.start_server(port)

    def read(self) -> Optional[Dict[str, Any]]:
        """Read frames from the RealSense camera."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as e:
            print(f"[ERROR] RealSense failed to get frames for {self.mount_position}: {e}")
            return None

        if self._align is not None:
            frames = self._align.process(frames)

        timestamps = {}
        images = {}
        capture_time = time.time()

        if self.config.enable_color:
            color_frame = frames.get_color_frame()
            if not color_frame:
                print(f"[ERROR] RealSense no color frame for {self.mount_position}")
                return None
            bgr = np.asanyarray(color_frame.get_data())
            images[self.mount_position] = bgr[..., ::-1]  # BGR to RGB
            timestamps[self.mount_position] = capture_time

        if self.config.enable_depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                print(f"[ERROR] RealSense no depth frame for {self.mount_position}")
                return None
            key = f"{self.mount_position}_depth"
            images[key] = np.asanyarray(depth_frame.get_data())
            timestamps[key] = capture_time

        return {"timestamps": timestamps, "images": images}

    def serialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data using ImageMessageSchema."""
        return ImageMessageSchema(timestamps=data["timestamps"], images=data["images"]).serialize()

    def observation_space(self) -> gym.Space:
        spaces = {}
        w, h = self.config.color_image_dim
        if self.config.enable_color:
            spaces["color_image"] = gym.spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        if self.config.enable_depth:
            spaces["depth_image"] = gym.spaces.Box(low=0, high=65535, shape=(h, w), dtype=np.uint16)
        return gym.spaces.Dict(spaces)

    def close(self):
        if self._run_as_server:
            self.stop_server()
        if hasattr(self, "pipeline"):
            self.pipeline.stop()

    def run_server(self):
        if not self._run_as_server:
            raise ValueError("This function is only available when run_as_server is True")
        while True:
            frame = self.read()
            if frame is None:
                continue
            msg = self.serialize(frame)
            self.send_message({self.mount_position: msg})

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--serial", type=str, default=None, help="RealSense serial number")
    parser.add_argument("--mount-position", type=str, default="ego_view")
    parser.add_argument("--enable-depth", action="store_true")
    parser.add_argument("--show-image", action="store_true")
    args = parser.parse_args()

    config = RealSenseConfig(
        serial_number=args.serial,
        enable_depth=args.enable_depth,
        mount_position=args.mount_position,
    )

    sensor = RealSenseSensor(
        run_as_server=args.server,
        port=args.port,
        config=config,
        mount_position=args.mount_position,
    )

    if args.server:
        print(f"Starting RealSense server on port {args.port}")
        sensor.run_server()
    else:
        print("Running RealSense in standalone mode")
        while True:
            frame = sensor.read()
            if frame is None:
                time.sleep(0.1)
                continue

            for key, img in frame["images"].items():
                print(f"{key}: shape={img.shape}")
                if args.show_image:
                    display = img[..., ::-1] if img.ndim == 3 else img  # RGB back to BGR for cv2
                    cv2.imshow(key, display)

            if args.show_image and cv2.waitKey(1) == ord("q"):
                break

            time.sleep(0.01)

        cv2.destroyAllWindows()
        sensor.close()
