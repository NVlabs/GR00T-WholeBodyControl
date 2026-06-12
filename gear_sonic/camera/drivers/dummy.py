"""Dummy / replay sensor for testing without real camera hardware.

``DummySensor`` generates random images.
``ReplayDummySensor`` loops frames from a video file.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import ImageMessageSchema


class DummySensor(Sensor):
    """Produces random 640x480 images at each read() call."""

    def __init__(self):
        pass

    def read(self) -> dict[str, Any] | None:
        return {
            "color_image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "timestamp": time.time(),
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("DummySensor does not support serialize()")

    def close(self):
        pass

    def observation_space(self):
        if gym is None:
            return None
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                ),
            }
        )


class ReplayDummySensor(DummySensor):
    """Loops frames from a video file, useful for offline testing."""

    def __init__(self, video_path: str, mount_position: str = "color_image"):
        self.video_path = video_path
        self.mount_position = mount_position
        self.image_ctr = 0
        self.video_reader = cv2.VideoCapture(video_path)
        self.frames = []
        while self.video_reader.isOpened():
            ret, frame = self.video_reader.read()
            if not ret:
                break
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not self.frames:
            raise ValueError(f"No frames loaded from replay video: {video_path}")

    def read(self) -> dict[str, Any] | None:
        img = self.frames[self.image_ctr]
        self.image_ctr = (self.image_ctr + 1) % len(self.frames)
        img = cv2.resize(img, (640, 480))
        return {
            "timestamps": {self.mount_position: time.time()},
            "images": {self.mount_position: img},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(
            timestamps=data["timestamps"],
            images=data["images"],
        )
        return serialized_msg.serialize()

    def close(self):
        self.video_reader.release()
