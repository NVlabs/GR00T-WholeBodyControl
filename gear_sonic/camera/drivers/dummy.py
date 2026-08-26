"""Dummy / replay sensor for testing without real camera hardware.

``DummySensor`` generates random images.
``ReplayDummySensor`` loops frames from a video file.
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

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.image_ctr = 0
        self.video_reader = cv2.VideoCapture(video_path)
        self.frames = []
        while self.video_reader.isOpened():
            ret, frame = self.video_reader.read()
            if not ret:
                break
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def read(self) -> dict[str, Any] | None:
        self.image_ctr += 1
        if self.image_ctr >= len(self.frames):
            self.image_ctr = 0

        img = self.frames[self.image_ctr]
        img = cv2.resize(img, (640, 480))
        return {
            "color_image": img,
            "timestamp": {"color_image": time.time()},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(
            timestamps=data["timestamp"] if isinstance(data["timestamp"], dict) else {"color_image": data["timestamp"]},
            images={"color_image": data["color_image"]},
        )
        return serialized_msg.serialize()

    def close(self):
        self.video_reader.release()
