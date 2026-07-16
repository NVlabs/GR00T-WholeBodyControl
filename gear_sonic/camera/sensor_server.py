"""ZMQ PUB/SUB transport and image serialisation for the camera server."""

import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq


# =============================================================================
# Pose Message Schema
# =============================================================================
@dataclass
class PoseData:
    """Single pose data point with quaternion orientation and translation."""

    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
            "tx": self.tx,
            "ty": self.ty,
            "tz": self.tz,
        }

    @staticmethod
    def from_dict(data: dict[str, float]) -> "PoseData":
        return PoseData(
            qx=data.get("qx", 0.0),
            qy=data.get("qy", 0.0),
            qz=data.get("qz", 0.0),
            qw=data.get("qw", 1.0),
            tx=data.get("tx", 0.0),
            ty=data.get("ty", 0.0),
            tz=data.get("tz", 0.0),
        )

    def to_array(self) -> np.ndarray:
        return np.array([self.qx, self.qy, self.qz, self.qw, self.tx, self.ty, self.tz])

    @staticmethod
    def from_array(arr: np.ndarray) -> "PoseData":
        return PoseData(
            qx=float(arr[0]),
            qy=float(arr[1]),
            qz=float(arr[2]),
            qw=float(arr[3]),
            tx=float(arr[4]),
            ty=float(arr[5]),
            tz=float(arr[6]),
        )


@dataclass
class PoseMessageSchema:
    """Standardized message schema for pose / positional data."""

    timestamp: float = 0.0
    device_id: str = "iphone"
    pose: PoseData = field(default_factory=PoseData)

    def serialize(self) -> bytes:
        data = {
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "pose": self.pose.to_dict(),
        }
        return msgpack.packb(data, use_bin_type=True)

    @staticmethod
    def deserialize(packed_data: bytes) -> "PoseMessageSchema":
        data = msgpack.unpackb(packed_data, object_hook=m.decode)
        return PoseMessageSchema(
            timestamp=data.get("timestamp", 0.0),
            device_id=data.get("device_id", "iphone"),
            pose=PoseData.from_dict(data.get("pose", {})),
        )

    def asdict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "pose": self.pose.to_dict(),
        }


# =============================================================================
# Image Message Schema
# =============================================================================
@dataclass
class ImageMessageSchema:
    """Standardized message schema for camera images.

    Handles two encodings on the wire:

    * **str** – legacy base64-encoded JPEG.
    * **bytes** – raw JPEG from on-device MJPEG encoder (e.g. OAK).
    """

    timestamps: dict[str, float]
    images: dict[str, np.ndarray]

    def serialize(self) -> dict[str, Any]:
        serialized_msg: dict[str, Any] = {"timestamps": self.timestamps, "images": {}}
        for key, image in self.images.items():
            if isinstance(image, bytes | bytearray):
                serialized_msg["images"][key] = image
            else:
                serialized_msg["images"][key] = ImageUtils.encode_image(image)
        return serialized_msg

    @staticmethod
    def deserialize(data: dict[str, Any]) -> "ImageMessageSchema":
        timestamps = data.get("timestamps", {})
        images = {}
        for key, value in data.get("images", {}).items():
            if isinstance(value, bytes | bytearray):
                mat = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
                images[key] = mat[..., ::-1]  # BGR -> RGB
            elif isinstance(value, str):
                images[key] = ImageUtils.decode_image(value)
            elif isinstance(value, np.ndarray):
                images[key] = value
            elif isinstance(value, dict) and b"nd" in value:
                images[key] = m.decode(value)
            else:
                images[key] = value
        return ImageMessageSchema(timestamps=timestamps, images=images)

    def asdict(self) -> dict[str, Any]:
        return {"timestamps": self.timestamps, "images": self.images}


# =============================================================================
# ZMQ Server / Client
# =============================================================================
class SensorServer:
    """ZMQ PUB server that streams msgpack-encoded sensor payloads."""

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

    def send_message(self, data: dict[str, Any]):
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


class SensorClient:
    """ZMQ SUB client that receives msgpack-encoded sensor payloads."""

    def start_client(self, server_ip: str, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.CONFLATE, True)
        self.socket.setsockopt(zmq.RCVHWM, 3)
        self.socket.connect(f"tcp://{server_ip}:{port}")

    def stop_client(self):
        self.socket.close()
        self.context.term()

    def receive_message(self):
        packed = self.socket.recv()
        return msgpack.unpackb(packed, object_hook=m.decode)

    def receive_message_nonblocking(self, timeout_ms: int = 0):
        if self.socket.poll(timeout_ms):
            packed = self.socket.recv()
            return msgpack.unpackb(packed, object_hook=m.decode)
        return None


# =============================================================================
# Helpers
# =============================================================================
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
