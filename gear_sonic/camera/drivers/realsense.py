"""Intel RealSense camera driver.

Requires the ``pyrealsense2`` SDK — install with::

    pip install pyrealsense2

See https://github.com/IntelRealSense/librealsense for hardware-specific instructions.
"""

import time
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

import pyrealsense2 as rs

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class RealSenseConfig:
    """Configuration for the RealSense camera."""

    depth_image_dim: tuple[int, int] = (640, 480)
    color_image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    mount_position: str = CameraMountPosition.EGO_VIEW.value
    align_depth_to_color: bool = True
    enable_depth: bool = True


class RealSenseSensor(Sensor, SensorServer):
    """Sensor for Intel RealSense depth cameras."""

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: RealSenseConfig = RealSenseConfig(),
        id: int = 0,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        device_id: str | None = None,
    ):
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found")

        for device in devices:
            print(f"Device: {device.get_info(rs.camera_info.name)}")
            print(f"    Serial number: {device.get_info(rs.camera_info.serial_number)}")
            print(f"    Firmware version: {device.get_info(rs.camera_info.firmware_version)}")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        devices = sorted(devices, key=lambda x: x.get_info(rs.camera_info.serial_number))
        serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
        if device_id is not None:
            requested_serial = str(device_id)
            if requested_serial not in serials:
                raise ValueError(
                    f"RealSense serial {requested_serial!r} was not found. "
                    f"Available serials: {serials}"
                )
            selected_device_index = serials.index(requested_serial)
        else:
            selected_device_index = id
            if not 0 <= selected_device_index < len(devices):
                raise ValueError(
                    f"RealSense device index {selected_device_index} is unavailable; "
                    f"found {len(devices)} device(s): {serials}"
                )
        selected_serial = serials[selected_device_index]
        self.config.enable_device(selected_serial)

        try:
            self.config.enable_stream(
                rs.stream.color,
                config.color_image_dim[0],
                config.color_image_dim[1],
                rs.format.rgb8,
                config.fps,
            )
            if config.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth,
                    config.depth_image_dim[0],
                    config.depth_image_dim[1],
                    rs.format.z16,
                    config.fps,
                )
            profile = self.pipeline.start(self.config)
        except Exception as e:
            depth_description = (
                f"{config.depth_image_dim[0]}x{config.depth_image_dim[1]}"
                if config.enable_depth
                else "disabled"
            )
            raise RuntimeError(
                "Failed to start RealSense pipeline with "
                f"RGB={config.color_image_dim[0]}x{config.color_image_dim[1]}, "
                f"depth={depth_description}, "
                f"fps={config.fps}: {e}"
            )

        self._realsense_config = config
        self._depth_scale = (
            profile.get_device().first_depth_sensor().get_depth_scale()
            if config.enable_depth
            else None
        )
        self._align = (
            rs.align(rs.stream.color)
            if config.enable_depth and config.align_depth_to_color
            else None
        )
        self._run_as_server = run_as_server
        self.mount_position = mount_position
        if self._run_as_server:
            self.start_server(port)
        print(
            f"Done initializing RealSense sensor: "
            f"{selected_serial} as {mount_position}"
        )

    def read(self) -> dict[str, Any] | None:
        try:
            frames = self.pipeline.wait_for_frames()
            if self._align is not None:
                frames = self._align.process(frames)
        except Exception as e:
            print(f"ERROR! Failed to wait for frames: {e}")
            return None

        color_frame = frames.get_color_frame()
        if not color_frame:
            print("WARNING! No color frame")
            return None

        depth_frame = (
            frames.get_depth_frame()
            if self._realsense_config.enable_depth
            else None
        )
        if self._realsense_config.enable_depth and not depth_frame:
            print("WARNING! No depth frame")
            return None

        try:
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = None
            if depth_frame is not None:
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_image = depth_raw.astype(np.float32) * self._depth_scale
        except Exception as e:
            print(f"ERROR! Failed to convert frames to numpy arrays: {e}")
            return None

        if color_image.size == 0:
            print("WARNING! Empty color image")
            return None
        if depth_image is not None and depth_image.size == 0:
            print("WARNING! Empty depth image")
            return None

        current_time = time.time()
        timestamps = {self.mount_position: current_time}
        images = {self.mount_position: color_image}
        if depth_image is not None:
            timestamps[f"{self.mount_position}_depth"] = current_time
            images[f"{self.mount_position}_depth"] = depth_image
        return {"timestamps": timestamps, "images": images}

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        spaces = {
            "color_image": gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    self._realsense_config.color_image_dim[1],
                    self._realsense_config.color_image_dim[0],
                    3,
                ),
                dtype=np.uint8,
            )
        }
        if self._realsense_config.enable_depth:
            spaces["depth_image"] = gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(
                    self._realsense_config.depth_image_dim[1],
                    self._realsense_config.depth_image_dim[0],
                ),
                dtype=np.float32,
            )
        return gym.spaces.Dict(spaces)

    def close(self):
        if self._run_as_server:
            self.stop_server()
        self.pipeline.stop()

    def run_server(self):
        if not self._run_as_server:
            raise ValueError("run_as_server must be True to call run_server()")
        while True:
            read_result = self.read()
            if read_result is None:
                continue
            self.send_message({self.mount_position: self.serialize(read_result)})
