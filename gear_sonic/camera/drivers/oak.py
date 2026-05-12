"""OAK (DepthAI) camera driver.

Requires the ``depthai`` SDK — install with::

    pip install depthai

See https://docs.luxonis.com/ for hardware-specific instructions.
"""

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

import depthai as dai

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class OAKConfig:
    """Configuration for the OAK camera."""

    color_image_dim: tuple[int, int] = (640, 480)
    monochrome_image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    enable_color: bool = True
    enable_mono_cameras: bool = False
    mount_position: str = CameraMountPosition.EGO_VIEW.value
    autofocus: bool = False
    manual_focus: int = 130
    use_mjpeg: bool = False
    mjpeg_quality: int = 80


class OAKSensor(Sensor, SensorServer):
    """Sensor for the OAK camera family (OAK-D, OAK-1, etc.)."""

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: OAKConfig = OAKConfig(),
        device_id: str | None = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
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
                    self.device = dai.Device(device_info, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)
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

        self.pipeline = dai.Pipeline(self.device)
        self.output_queues = {}
        self._use_mjpeg = config.use_mjpeg

        # RGB camera (CAM_A)
        if config.enable_color and dai.CameraBoardSocket.CAM_A in sockets:
            self.cam_rgb = self.pipeline.create(dai.node.Camera)
            cam_socket = dai.CameraBoardSocket.CAM_A
            self.cam_rgb = self.cam_rgb.build(cam_socket)

            if config.use_mjpeg:
                cam_out = self.cam_rgb.requestOutput(
                    config.color_image_dim,
                    dai.ImgFrame.Type.NV12,
                    fps=config.fps,
                )
                encoder = self.pipeline.create(dai.node.VideoEncoder)
                encoder.setDefaultProfilePreset(
                    config.fps, dai.VideoEncoderProperties.Profile.MJPEG
                )
                encoder.setQuality(config.mjpeg_quality)
                cam_out.link(encoder.input)
                self.output_queues["color"] = encoder.out.createOutputQueue(
                    maxSize=3, blocking=False
                )
            else:
                self.output_queues["color"] = self.cam_rgb.requestOutput(
                    config.color_image_dim,
                    fps=config.fps,
                ).createOutputQueue(maxSize=3, blocking=False)
            print(f"Enabled CAM_A (RGB){' with MJPEG encoding' if config.use_mjpeg else ''}")

            if not config.autofocus:
                ctrl_in = self.cam_rgb.inputControl.createInputQueue()
                ctrl = dai.CameraControl()
                ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
                ctrl.setManualFocus(config.manual_focus)
                ctrl_in.send(ctrl)
                print(f"Autofocus disabled, manual focus set to {config.manual_focus}")

        # Monochrome cameras (CAM_B / CAM_C)
        if config.enable_mono_cameras:
            if dai.CameraBoardSocket.CAM_B in sockets:
                self.cam_mono_left = self.pipeline.create(dai.node.Camera)
                cam_socket = dai.CameraBoardSocket.CAM_B
                self.cam_mono_left = self.cam_mono_left.build(cam_socket)

                if config.use_mjpeg:
                    cam_out = self.cam_mono_left.requestOutput(
                        config.monochrome_image_dim,
                        dai.ImgFrame.Type.NV12,
                        fps=config.fps,
                    )
                    encoder = self.pipeline.create(dai.node.VideoEncoder)
                    encoder.setDefaultProfilePreset(
                        config.fps, dai.VideoEncoderProperties.Profile.MJPEG
                    )
                    encoder.setQuality(config.mjpeg_quality)
                    cam_out.link(encoder.input)
                    self.output_queues["mono_left"] = encoder.out.createOutputQueue(
                        maxSize=3, blocking=False
                    )
                else:
                    self.output_queues["mono_left"] = self.cam_mono_left.requestOutput(
                        config.monochrome_image_dim,
                        fps=config.fps,
                    ).createOutputQueue(maxSize=3, blocking=False)
                print("Enabled CAM_B (Monochrome Left)")

            if dai.CameraBoardSocket.CAM_C in sockets:
                self.cam_mono_right = self.pipeline.create(dai.node.Camera)
                cam_socket = dai.CameraBoardSocket.CAM_C
                self.cam_mono_right = self.cam_mono_right.build(cam_socket)

                if config.use_mjpeg:
                    cam_out = self.cam_mono_right.requestOutput(
                        config.monochrome_image_dim,
                        dai.ImgFrame.Type.NV12,
                        fps=config.fps,
                    )
                    encoder = self.pipeline.create(dai.node.VideoEncoder)
                    encoder.setDefaultProfilePreset(
                        config.fps, dai.VideoEncoderProperties.Profile.MJPEG
                    )
                    encoder.setQuality(config.mjpeg_quality)
                    cam_out.link(encoder.input)
                    self.output_queues["mono_right"] = encoder.out.createOutputQueue(
                        maxSize=3, blocking=False
                    )
                else:
                    self.output_queues["mono_right"] = self.cam_mono_right.requestOutput(
                        config.monochrome_image_dim,
                        fps=config.fps,
                    ).createOutputQueue(maxSize=3, blocking=False)
                print("Enabled CAM_C (Monochrome Right)")

        assert len(self.output_queues) > 0, "No output queues enabled"

        self.pipeline.start()

        print(f"[{mount_position}] Pipeline started, waiting for stabilization...")
        time.sleep(2.0)

        for _ in range(10):
            test_frame = None
            for queue_name, q in self.output_queues.items():
                test_frame = q.tryGet()
                if test_frame:
                    print(f"[{mount_position}] First frame received from {queue_name}")
                    break
            if test_frame:
                break
            time.sleep(0.3)
        else:
            print(f"[{mount_position}] Warning: No frames received during init verification")

        if run_as_server:
            self.start_server(port)

    def read(self) -> dict[str, Any] | None:
        if not self.pipeline.isRunning():
            print(f"[ERROR] OAK pipeline stopped for {self.mount_position}")
            return None
        if not self.device.isPipelineRunning():
            print(f"[ERROR] OAK device disconnected for {self.mount_position}")
            return None

        timestamps = {}
        images = {}
        rgb_frame_time = None

        def drain_queue_get_latest(queue):
            latest_frame = None
            while True:
                frame = queue.tryGet()
                if frame is None:
                    break
                latest_frame = frame
            return latest_frame

        expected_cameras = set(self.output_queues.keys())
        received_cameras = set()

        if "color" in self.output_queues:
            try:
                rgb_frame = drain_queue_get_latest(self.output_queues["color"])
                if rgb_frame is None:
                    return None
                rgb_frame_time = rgb_frame.getTimestamp()
                read_time = time.time()
                frame_age = (dai.Clock.now() - rgb_frame_time).total_seconds()
                capture_time = read_time - frame_age

                if self._use_mjpeg:
                    images[self.mount_position] = bytes(rgb_frame.getData())
                else:
                    images[self.mount_position] = rgb_frame.getCvFrame()[..., ::-1]
                timestamps[self.mount_position] = capture_time
                received_cameras.add("color")
            except Exception as e:
                print(f"[ERROR] Failed to read color frame from {self.mount_position}: {e}")
                return None

        if "mono_left" in self.output_queues:
            try:
                mono_left_frame = drain_queue_get_latest(self.output_queues["mono_left"])
                if mono_left_frame is None:
                    return None
                mono_left_frame_time = mono_left_frame.getTimestamp()
                read_time = time.time()
                frame_age = (dai.Clock.now() - mono_left_frame_time).total_seconds()
                capture_time = read_time - frame_age

                key = f"{self.mount_position}_left_mono"
                if self._use_mjpeg:
                    images[key] = bytes(mono_left_frame.getData())
                else:
                    images[key] = mono_left_frame.getCvFrame()
                timestamps[key] = capture_time
                received_cameras.add("mono_left")
            except Exception as e:
                print(f"[ERROR] Failed to read mono_left frame from {self.mount_position}: {e}")
                return None

        if "mono_right" in self.output_queues:
            try:
                mono_right_frame = drain_queue_get_latest(self.output_queues["mono_right"])
                if mono_right_frame is None:
                    return None
                mono_right_frame_time = mono_right_frame.getTimestamp()
                read_time = time.time()
                frame_age = (dai.Clock.now() - mono_right_frame_time).total_seconds()
                capture_time = read_time - frame_age

                key = f"{self.mount_position}_right_mono"
                if self._use_mjpeg:
                    images[key] = bytes(mono_right_frame.getData())
                else:
                    images[key] = mono_right_frame.getCvFrame()
                timestamps[key] = capture_time
                received_cameras.add("mono_right")
            except Exception as e:
                print(f"[ERROR] Failed to read mono_right frame from {self.mount_position}: {e}")
                return None

        if received_cameras != expected_cameras:
            missing = expected_cameras - received_cameras
            print(f"[ERROR] Missing frames from cameras: {missing} for {self.mount_position}")
            return None

        if rgb_frame_time is not None:
            frame_age = (dai.Clock.now() - rgb_frame_time).total_seconds()
            if frame_age > 0.1:
                print(
                    f"[{self.mount_position}] OAK frame age too large: {frame_age * 1000:.1f}ms"
                )

        return {"timestamps": timestamps, "images": images}

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
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
        if self._run_as_server:
            self.stop_server()
        if hasattr(self, "pipeline") and self.pipeline.isRunning():
            self.pipeline.stop()
        self.device.close()

    def run_server(self):
        if not self._run_as_server:
            raise ValueError("run_as_server must be True to call run_server()")
        while True:
            frame = self.read()
            if frame is None:
                continue
            msg = self.serialize(frame)
            self.send_message({self.mount_position: msg})

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Run as server")
    parser.add_argument("--host", type=str, default="localhost", help="Server IP address")
    parser.add_argument("--port", type=int, default=5555, help="Port number")
    parser.add_argument("--device-id", type=str, default=None, help="Specific device ID")
    parser.add_argument(
        "--enable-mono", action="store_true", help="Enable monochrome cameras (CAM_B & CAM_C)"
    )
    parser.add_argument("--mount-position", type=str, default="ego_view", help="Mount position")
    parser.add_argument("--show-image", action="store_true", help="Display images")
    parser.add_argument("--use-mjpeg", action="store_true", help="Use MJPEG encoding on-device")
    parser.add_argument(
        "--mjpeg-quality", type=int, default=80, help="MJPEG quality 1-100 (default: 80)"
    )
    args = parser.parse_args()

    oak_config = OAKConfig()
    if args.enable_mono:
        oak_config.enable_mono_cameras = True
    if args.use_mjpeg:
        oak_config.use_mjpeg = True
        oak_config.mjpeg_quality = args.mjpeg_quality

    if args.server:
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
        oak = OAKSensor(run_as_server=False, config=oak_config, device_id=args.device_id)
        print("Running OAK camera in standalone mode")

        while True:
            frame = oak.read()
            if frame is None:
                print("Waiting for frame...")
                time.sleep(0.5)
                continue

            if args.show_image:
                for key, img in frame.get("images", {}).items():
                    if isinstance(img, np.ndarray):
                        cv2.imshow(key, img[..., ::-1] if img.ndim == 3 and img.shape[2] == 3 else img)
                if cv2.waitKey(1) == ord("q"):
                    break

            time.sleep(0.01)

        cv2.destroyAllWindows()
        oak.close()
