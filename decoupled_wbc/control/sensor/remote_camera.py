from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import zmq

from decoupled_wbc.control.base.sensor import Sensor
from decoupled_wbc.control.sensor.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorClient,
)


class RemoteCameraSensor(Sensor, SensorClient):
    """Camera sensor that connects to a remote ZMQ camera server (e.g., RealSense on Orin)."""

    def __init__(
        self,
        server_ip: str,
        port: int = 5556,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        self.mount_position = mount_position
        self.start_client(server_ip, port)
        # Set receive timeout so read() returns None instead of blocking forever
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        print(
            f"RemoteCameraSensor connected to tcp://{server_ip}:{port} "
            f"(mount_position={mount_position})"
        )

    def read(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Read a frame from the remote camera server."""
        try:
            message = self.receive_message()
        except zmq.Again:
            return None

        if not message:
            return None

        schema = ImageMessageSchema.deserialize(message)
        return schema.asdict()

    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                )
            }
        )

    def close(self):
        self.stop_client()
