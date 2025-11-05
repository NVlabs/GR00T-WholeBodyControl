import base64
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq


@dataclass
class ImageMessageSchema:
    """
    This is a standardized message schema for image data.
    Any camera should use this schema to serialize (send to queue) and
    deserialize (receive from queue) the image data.

    """

    timestamps: Dict[str, float]
    """Dictionary of timestamps, keyed by image identifier (e.g., {"ego_view": 123.45})"""
    images: Dict[str, np.ndarray]
    """Dictionary of images, keyed by image identifier (e.g., {"ego_view": array, "ego_view_left_mono": array})"""

    def serialize(self) -> Dict[str, Any]:
        """Serialize the message for transmission."""
        serialized_msg = {"timestamps": self.timestamps, "images": {}}
        for key, image in self.images.items():
            serialized_msg["images"][key] = ImageUtils.encode_image(image)
        return serialized_msg

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> "ImageMessageSchema":
        """Deserialize received message data."""
        timestamps = data.get("timestamps", {})
        images = {}
        for key, value in data.get("images", {}).items():
            if isinstance(value, str):
                images[key] = ImageUtils.decode_image(value)
            else:
                images[key] = value
        return ImageMessageSchema(timestamps=timestamps, images=images)

    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {"timestamps": self.timestamps, "images": self.images}


class SensorServer:
    def start_server(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)  # high water mark
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")
        print(f"Sensor server running at tcp://*:{port}")

        self.message_sent = 0
        self.message_dropped = 0

    def stop_server(self):
        self.socket.close()
        self.context.term()

    def send_message(self, data: Dict[str, Any]):
        try:
            packed = msgpack.packb(data, use_bin_type=True)
            self.socket.send(packed, flags=zmq.NOBLOCK)
        except zmq.Again:
            self.message_dropped += 1
            print(f"[Warning] message dropped: {self.message_dropped}")
        self.message_sent += 1

        if self.message_sent % 100 == 0:
            print(
                f"[Sensor server] Message sent: {self.message_sent}, message dropped: {self.message_dropped}"
            )


class SensorClient:
    def start_client(self, server_ip: str, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.CONFLATE, True)  # last msg only.
        self.socket.setsockopt(zmq.RCVHWM, 3)  # queue size 3 for receive buffer
        self.socket.connect(f"tcp://{server_ip}:{port}")

    def stop_client(self):
        self.socket.close()
        self.context.term()

    def receive_message(self):
        packed = self.socket.recv()
        return msgpack.unpackb(packed, object_hook=m.decode)


class CameraMountPosition(Enum):
    EGO_VIEW = "ego_view"
    HEAD = "head"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"


class ImageUtils:
    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        _, color_buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(color_buffer).decode("utf-8")

    @staticmethod
    def encode_depth_image(image: np.ndarray) -> str:
        depth_compressed = cv2.imencode(".png", image)[1].tobytes()
        return base64.b64encode(depth_compressed).decode("utf-8")

    @staticmethod
    def decode_image(image: str) -> np.ndarray:
        color_data = base64.b64decode(image)
        color_array = np.frombuffer(color_data, dtype=np.uint8)
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

    @staticmethod
    def decode_depth_image(image: str) -> np.ndarray:
        depth_data = base64.b64decode(image)
        depth_array = np.frombuffer(depth_data, dtype=np.uint8)
        return cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
