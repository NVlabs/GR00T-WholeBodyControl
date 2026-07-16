"""Generic USB webcam driver using OpenCV.

No hardware SDK needed — works with any UVC-compatible camera visible as
``/dev/video*``.  Only requires ``opencv-python``.
"""

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import CameraMountPosition


class USBCameraConfig:
    """Configuration for generic USB camera."""

    image_dim: tuple = (640, 480)
    fps: int = 30
    device_index: int = 0


class USBCameraSensor(Sensor):
    """Sensor for generic USB cameras using OpenCV VideoCapture."""

    def __init__(
        self,
        config: USBCameraConfig = USBCameraConfig(),
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        device_index: int | None = None,
    ):
        self.config = config
        self.mount_position = mount_position

        idx = device_index if device_index is not None else config.device_index

        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera at index {idx}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.image_dim[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.image_dim[1])
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[{mount_position}] Warming up USB camera...")
        for _ in range(10):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.1)

        print(f"[{mount_position}] USB camera opened at index {idx}")
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

    def read(self) -> dict[str, Any] | None:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print(f"[{self.mount_position}] USB camera read failed: ret={ret}")
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {
            "timestamps": {self.mount_position: time.time()},
            "images": {self.mount_position: frame_rgb},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        from gear_sonic.camera.sensor_server import ImageMessageSchema

        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.config.image_dim[1], self.config.image_dim[0], 3),
                    dtype=np.uint8,
                ),
            }
        )

    def close(self):
        if self.cap is not None:
            self.cap.release()
