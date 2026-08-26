"""Adapter from Unitree TeleImager's ImageClient to SONIC camera messages."""

from __future__ import annotations

from collections import deque
import time
from typing import Any

import numpy as np

from gear_sonic.camera.sensor import Sensor


class TeleImagerCameraClientSensor(Sensor):
    """Read head and wrist RGB images from an existing TeleImager server.

    This class does not launch or manage TeleImager. The server configuration
    determines which cameras exist and which ZMQ ports they use.
    """

    _CAMERA_NAMES = {
        "head_camera": "ego_view",
        "left_wrist_camera": "left_wrist",
        "right_wrist_camera": "right_wrist",
    }

    def __init__(self, server_ip: str = "localhost", request_port: int = 60000):
        try:
            from teleimager import ImageClient
        except ImportError:
            # Some TeleImager revisions leave ``teleimager.__init__`` empty and
            # expose the public client only from its implementation module.
            try:
                from teleimager.image_client import ImageClient
            except ImportError as exc:
                raise ImportError(
                    "TeleImager client is not installed in this environment. Install the "
                    "official teleimager package, for example: "
                    "uv pip install -e /path/to/xr_teleoperate/teleop/teleimager"
                ) from exc

        self._client = ImageClient(
            host=server_ip,
            request_port=request_port,
            request_bgr=True,
        )
        self._camera_config = self._client.get_cam_config()
        if hasattr(self._client, "get_zmq_camera_names"):
            available = set(self._client.get_zmq_camera_names())
        else:
            available = {
                name
                for name in self._CAMERA_NAMES
                if self._camera_config.get(name, {}).get("enable_zmq", False)
            }
        if "head_camera" not in available:
            raise RuntimeError(
                "TeleImager must publish head_camera over ZMQ for SONIC ego_view"
            )
        self._active_cameras = {
            source: destination
            for source, destination in self._CAMERA_NAMES.items()
            if source in available
        }
        self._frame_times: deque[float] = deque(maxlen=30)
        print(
            "Initialized TeleImager camera client: "
            + ", ".join(self._active_cameras)
        )

    def _get_frame(self, camera_name: str):
        if hasattr(self._client, "get_camera_frame"):
            return self._client.get_camera_frame(camera_name)
        getter_name = {
            "head_camera": "get_head_frame",
            "left_wrist_camera": "get_left_wrist_frame",
            "right_wrist_camera": "get_right_wrist_frame",
        }[camera_name]
        return getattr(self._client, getter_name)()

    def read(self, **_kwargs) -> dict[str, Any] | None:
        images: dict[str, np.ndarray] = {}
        timestamps: dict[str, float] = {}

        for source_name, destination_name in self._active_cameras.items():
            tele_image = self._get_frame(source_name)
            bgr = None if tele_image is None else tele_image.bgr
            if bgr is None:
                continue
            if (
                source_name == "head_camera"
                and self._camera_config.get("head_camera", {}).get("binocular", False)
            ):
                # Unitree's standard UVC head camera is side-by-side stereo.
                # SONIC's ego_view is one 640x480 view, matching color_0 in
                # xr_teleoperate's recorder.
                bgr = bgr[:, : bgr.shape[1] // 2]
            # TeleImager decodes to BGR; SONIC camera messages use RGB.
            images[destination_name] = np.ascontiguousarray(bgr[..., ::-1])
            timestamps[destination_name] = time.time()

        if "ego_view" not in images:
            return None

        self._frame_times.append(time.monotonic())
        return {"timestamps": timestamps, "images": images}

    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("TeleImager client does not serialize")

    def close(self):
        self._client.close()


def create_camera_client(
    backend: str,
    server_ip: str,
    port: int,
) -> Sensor:
    """Create a SONIC-compatible camera client for the selected backend."""
    if backend == "teleimager":
        return TeleImagerCameraClientSensor(server_ip=server_ip, request_port=port)
    if backend == "composed":
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

        return ComposedCameraClientSensor(server_ip=server_ip, port=port)
    raise ValueError("camera backend must be 'composed' or 'teleimager'")
