import time
from typing import Any, Dict, Optional, Tuple

import cv2
import depthai as dai
import gymnasium as gym
import numpy as np

from decoupled_wbc.control.base.sensor import Sensor
from decoupled_wbc.control.sensor.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class OAKConfig:
    """Configuration for the OAK camera."""

    color_image_dim: Tuple[int, int] = (640, 480)  # RGB camera resolution
    monochrome_image_dim: Tuple[int, int] = (640, 480)  # Monochrome camera resolution
    fps: int = 30
    enable_color: bool = True  # Enable CAM_A (RGB)
    enable_mono_cameras: bool = False  # Enable CAM_B & CAM_C (Monochrome stereo pair)
    mount_position: str = CameraMountPosition.EGO_VIEW.value


class OAKSensor(Sensor, SensorServer):
    """Sensor for the OAK camera family."""

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: OAKConfig = OAKConfig(),
        device_id: Optional[str] = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        """Initialize the OAK camera."""
        self.config = config
        self.mount_position = mount_position
        self._run_as_server = run_as_server

        device_infos = dai.Device.getAllAvailableDevices()
        assert len(device_infos) > 0, f"No OAK devices found for {mount_position}"
        print(f"Device infos: {device_infos}")
        if device_id is not None:
            device_found = False
            for device_info in device_infos:
                if device_info.getDeviceId() == device_id:
                    self.device = dai.Device(device_info)
                    device_found = True
                    break
            if not device_found:
                raise ValueError(f"Device with ID {device_id} not found")
        else:
            self.device = dai.Device()

        print(f"Connected to OAK device: {self.device.getDeviceName(), self.device.getDeviceId()}")
        print(f"Device ID: {self.device.getDeviceId()}")

        sockets: list[dai.CameraBoardSocket] = self.device.getConnectedCameras()
        print(f"Available cameras: {[str(s) for s in sockets]}")

        # Create pipeline (without context manager to persist across method calls)
        self.pipeline = dai.Pipeline(self.device)
        self.output_queues = {}

        # Configure RGB camera (CAM_A)
        if config.enable_color and dai.CameraBoardSocket.CAM_A in sockets:
            self.cam_rgb = self.pipeline.create(dai.node.Camera)
            cam_socket = dai.CameraBoardSocket.CAM_A
            self.cam_rgb = self.cam_rgb.build(cam_socket)
            # Create RGB output queue
            self.output_queues["color"] = self.cam_rgb.requestOutput(
                config.color_image_dim,
                fps=config.fps,
            ).createOutputQueue()
            print("Enabled CAM_A (RGB)")

        # Configure Monochrome cameras (CAM_B & CAM_C)
        if config.enable_mono_cameras:
            if dai.CameraBoardSocket.CAM_B in sockets:
                self.cam_mono_left = self.pipeline.create(dai.node.Camera)
                cam_socket = dai.CameraBoardSocket.CAM_B
                self.cam_mono_left = self.cam_mono_left.build(cam_socket)
                # Create mono left output queue
                self.output_queues["mono_left"] = self.cam_mono_left.requestOutput(
                    config.monochrome_image_dim,
                    fps=config.fps,
                ).createOutputQueue()
                print("Enabled CAM_B (Monochrome Left)")

            if dai.CameraBoardSocket.CAM_C in sockets:
                self.cam_mono_right = self.pipeline.create(dai.node.Camera)
                cam_socket = dai.CameraBoardSocket.CAM_C
                self.cam_mono_right = self.cam_mono_right.build(cam_socket)
                # Create mono right output queue
                self.output_queues["mono_right"] = self.cam_mono_right.requestOutput(
                    config.monochrome_image_dim,
                    fps=config.fps,
                ).createOutputQueue()
                print("Enabled CAM_C (Monochrome Right)")

        assert len(self.output_queues) > 0, "No output queues enabled"
        # auto exposure compensation, for CoRL demo
        # cam_q_in = self.cam_rgb.inputControl.createInputQueue()
        # ctrl = dai.CameraControl()
        # ctrl.setAutoExposureEnable()
        # ctrl.setAutoExposureCompensation(-2)
        # cam_q_in.send(ctrl)

        # Start pipeline on device
        self.pipeline.start()

        if run_as_server:
            self.start_server(port)

    def read(self) -> Optional[Dict[str, Any]]:
        """Read images from the camera."""
        if not self.pipeline.isRunning():
            print(f"[ERROR] OAK pipeline stopped for {self.mount_position}")
            return None

        # Check if device is still connected
        if not self.device.isPipelineRunning():
            print(f"[ERROR] OAK device disconnected for {self.mount_position}")
            return None

        timestamps = {}
        images = {}
        rgb_frame_time = None

        # Get color frame if enabled
        if "color" in self.output_queues:
            try:
                rgb_frame = self.output_queues["color"].get()
                rgb_frame_time = rgb_frame.getTimestamp()
                if rgb_frame is not None:
                    images[self.mount_position] = rgb_frame.getCvFrame()[..., ::-1]  # BGR to RGB
                    timestamps[self.mount_position] = (
                        rgb_frame_time - dai.Clock.now()
                    ).total_seconds() + time.time()
            except Exception as e:
                print(f"[ERROR] Failed to read color frame from {self.mount_position}: {e}")
                return None

        # Get mono frames if enabled
        if "mono_left" in self.output_queues:
            try:
                mono_left_frame = self.output_queues["mono_left"].get()
                mono_left_frame_time = mono_left_frame.getTimestamp()
                if mono_left_frame is not None:
                    key = f"{self.mount_position}_left_mono"
                    images[key] = mono_left_frame.getCvFrame()
                    timestamps[key] = (
                        mono_left_frame_time - dai.Clock.now()
                    ).total_seconds() + time.time()
            except Exception as e:
                print(f"[ERROR] Failed to read mono_left frame from {self.mount_position}: {e}")
                return None

        if "mono_right" in self.output_queues:
            try:
                mono_right_frame = self.output_queues["mono_right"].get()
                mono_right_frame_time = mono_right_frame.getTimestamp()
                if mono_right_frame is not None:
                    key = f"{self.mount_position}_right_mono"
                    images[key] = mono_right_frame.getCvFrame()
                    timestamps[key] = (
                        mono_right_frame_time - dai.Clock.now()
                    ).total_seconds() + time.time()
            except Exception as e:
                print(f"[ERROR] Failed to read mono_right frame from {self.mount_position}: {e}")
                return None

        if (
            rgb_frame_time is not None
            and (rgb_frame_time - dai.Clock.now()).total_seconds() <= -0.2
        ):
            print(
                f"[{self.mount_position}] OAK latency too large: "
                f"{(dai.Clock.now() - rgb_frame_time).total_seconds() * 1000}ms"
            )

        return {
            "timestamps": timestamps,
            "images": images,
        }

    def serialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data using ImageMessageSchema."""
        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self) -> gym.Space:
        spaces = {}

        if self.config.enable_color:
            spaces["color_image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.config.color_image_dim[1], self.config.color_image_dim[0], 3),
                dtype=np.uint8,
            )

        if self.config.enable_mono_cameras:
            spaces["mono_left_image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.config.monochrome_image_dim[1], self.config.monochrome_image_dim[0]),
                dtype=np.uint8,
            )
            spaces["mono_right_image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.config.monochrome_image_dim[1], self.config.monochrome_image_dim[0]),
                dtype=np.uint8,
            )

        return gym.spaces.Dict(spaces)

    def close(self):
        """Close the camera connection."""
        if self._run_as_server:
            self.stop_server()
        if hasattr(self, "pipeline") and self.pipeline.isRunning():
            self.pipeline.stop()
        self.device.close()

    def run_server(self):
        """Run the server."""
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
    """Test function for OAK camera."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Run as server")
    parser.add_argument("--client", action="store_true", help="Run as client")
    parser.add_argument("--host", type=str, default="localhost", help="Server IP address")
    parser.add_argument("--port", type=int, default=5555, help="Port number")
    parser.add_argument("--device-id", type=str, default=None, help="Specific device ID")
    parser.add_argument(
        "--enable-mono", action="store_true", help="Enable monochrome cameras (CAM_B & CAM_C)"
    )
    parser.add_argument("--mount-position", type=str, default="ego_view", help="Mount position")
    parser.add_argument("--show-image", action="store_true", help="Display images")
    args = parser.parse_args()

    oak_config = OAKConfig()
    if args.enable_mono:
        oak_config.enable_mono_cameras = True

    if args.server:
        # Run as server
        oak = OAKSensor(
            run_as_server=True,
            port=args.port,
            config=oak_config,
            device_id=args.device_id,
            mount_position=args.mount_position,
        )
        print(f"Starting OAK server on port {args.port}")
        oak.run_server()

    else:
        # Run standalone
        oak = OAKSensor(run_as_server=False, config=oak_config, device_id=args.device_id)
        print("Running OAK camera in standalone mode")

        while True:
            frame = oak.read()
            if frame is None:
                print("Waiting for frame...")
                time.sleep(0.5)
                continue

            if "color_image" in frame:
                print(f"Color image shape: {frame['color_image'].shape}")
            if "mono_left_image" in frame:
                print(f"Mono left image shape: {frame['mono_left_image'].shape}")
            if "mono_right_image" in frame:
                print(f"Mono right image shape: {frame['mono_right_image'].shape}")
            if "depth_image" in frame:
                print(f"Depth image shape: {frame['depth_image'].shape}")

            if args.show_image:
                if "color_image" in frame:
                    cv2.imshow("Color Image", frame["color_image"])

                if "mono_left_image" in frame:
                    cv2.imshow("Mono Left", frame["mono_left_image"])
                if "mono_right_image" in frame:
                    cv2.imshow("Mono Right", frame["mono_right_image"])

                if "depth_image" in frame:
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(frame["depth_image"], alpha=0.03), cv2.COLORMAP_JET
                    )
                    cv2.imshow("Depth Image", depth_colormap)

                if cv2.waitKey(1) == ord("q"):
                    break

            time.sleep(0.01)

        cv2.destroyAllWindows()
        oak.close()
