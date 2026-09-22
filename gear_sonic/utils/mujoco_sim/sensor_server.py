"""ZMQ PUB server for streaming JPEG-encoded camera images as msgpack payloads."""

import base64
from dataclasses import dataclass, field
from typing import Any, Dict

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq


@dataclass
class ImageMessageSchema:
    """
    Standardized message schema for image data.
    """

    timestamps: Dict[str, float]
    images: Dict[str, np.ndarray]

    def serialize(self) -> Dict[str, Any]:
        serialized_msg = {"timestamps": self.timestamps, "images": {}}
        for key, image in self.images.items():
            if key.endswith("_depth"):
                serialized_msg["images"][key] = ImageUtils.encode_depth_image(image)
            else:
                serialized_msg["images"][key] = ImageUtils.encode_image(image)
        return serialized_msg

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> "ImageMessageSchema":
        timestamps = data.get("timestamps", {})
        images = {}
        for key, value in data.get("images", {}).items():
            if isinstance(value, dict) and value.get("encoding") == "uint16_mm_png":
                images[key] = ImageUtils.decode_depth_image(value["data"])
            elif isinstance(value, str):
                images[key] = ImageUtils.decode_image(value)
            else:
                images[key] = value
        return ImageMessageSchema(timestamps=timestamps, images=images)


class SensorServer:
    def start_server(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)
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
                f"[Sensor server] Message sent: {self.message_sent}, "
                f"message dropped: {self.message_dropped}"
            )


class ImageUtils:
    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        _, color_buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(color_buffer).decode("utf-8")

    @staticmethod
    def encode_depth_image(depth_m: np.ndarray) -> Dict[str, str]:
        """Encode float depth in metres as a lossless 16-bit millimetre PNG."""
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.0)
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        depth_mm[valid] = np.rint(np.clip(depth[valid] * 1000.0, 1.0, 65535.0)).astype(
            np.uint16
        )
        ok, buffer = cv2.imencode(".png", depth_mm)
        if not ok:
            raise RuntimeError("Failed to encode MuJoCo depth image")
        return {
            "encoding": "uint16_mm_png",
            "data": base64.b64encode(buffer).decode("utf-8"),
        }

    @staticmethod
    def decode_depth_image(image: str) -> np.ndarray:
        """Decode a lossless millimetre PNG to float32 metres."""
        depth_data = base64.b64decode(image)
        depth_array = np.frombuffer(depth_data, dtype=np.uint8)
        depth_mm = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise ValueError("Failed to decode depth image")
        return depth_mm.astype(np.float32) * 0.001

    @staticmethod
    def decode_image(image: str) -> np.ndarray:
        color_data = base64.b64decode(image)
        color_array = np.frombuffer(color_data, dtype=np.uint8)
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)
