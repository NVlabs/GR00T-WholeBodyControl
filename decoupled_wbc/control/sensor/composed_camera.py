from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Dict, Optional

# we need to import these first in this order to avoid TSL segmentation fault
# caused by zed and oak libraries
try:
    import cv2  # noqa
    import depthai as dai  # noqa
    import pyzed.sl as sl  # noqa
except ImportError:
    print(
        """
    Some of the camera specific dependencies are not installed. If you are
    not running this on the robot, having these libraries is optional.
    """
    )

import numpy as np  # noqa

from decoupled_wbc.control.base.sensor import Sensor
from decoupled_wbc.control.sensor.sensor_server import (
    ImageMessageSchema,
    SensorClient,
    SensorServer,
    CameraMountPosition,
)


def read_qr_code(data):
    current_time = time.monotonic()
    detector = cv2.QRCodeDetector()
    for key, img in data["images"].items():
        decoded_time, bbox, _ = detector.detectAndDecode(img)
        if bbox is not None and decoded_time:
            print(f"{key} latency: {(current_time - float(decoded_time)) * 1e3:.1f} ms")
        else:
            print(f"{key} QR code not detected.")


@dataclass
class ComposedCameraConfig:
    """Camera configuration for composed camera"""

    ego_view_camera: Optional[str] = "oak"
    """Camera type for ego view: oak, realsense, zed, or None"""

    ego_view_device_id: Optional[str] = None
    """Device ID for ego view camera (optional, used for OAK cameras)"""

    head_camera: Optional[str] = None
    """Camera type for head view: oak, oak_mono, realsense, zed or None"""

    head_device_id: Optional[str] = None
    """Device ID for head camera (optional, used for OAK cameras)"""

    left_wrist_camera: Optional[str] = None
    """Camera type for left wrist view: oak, realsense, zed or None"""

    left_wrist_device_id: Optional[str] = None
    """Device ID for left wrist camera (optional, used for OAK cameras)"""

    right_wrist_camera: Optional[str] = None
    """Camera type for right wrist view: oak, realsense, zed or None"""

    right_wrist_device_id: Optional[str] = None
    """Device ID for right wrist camera (optional, used for OAK cameras)"""

    fps: int = 30
    """Rate at which the composed camera will publish the images. Since composed camera
    can read from multiple cameras, it will publish all the images.
    Note that OAK can only run at 30 FPS. 20 FPS will cause large latency.
    """

    # Server configuration
    run_as_server: bool = True
    """Whether to run as server or client"""

    server: bool = True
    """Whether to run the camera as a server"""

    port: int = 5555
    """Port number for server/client communication"""

    test_latency: bool = False
    """Whether to test latency"""

    # Queue configuration
    queue_size: int = 3
    """Size of each camera's image queue"""

    def __post_init__(self):
        # runyu: Note that this is a hack to make the config work with G1 camera server in orin
        # we should not use this hack in the future
        self.run_as_server: bool = self.server


class ComposedCameraSensor(Sensor, SensorServer):

    def __init__(self, config: ComposedCameraConfig):
        self.config = config
        self.camera_queues: Dict[str, queue.Queue] = {}
        self.camera_threads: Dict[str, threading.Thread] = {}
        self.shutdown_events: Dict[str, threading.Event] = {}
        self.error_events: Dict[str, threading.Event] = {}
        self.error_messages: Dict[str, str] = {}
        self._observation_spaces: Dict[str, Any] = {}

        camera_configs = self._get_camera_configs()

        # Then create worker threads
        for mount_position, camera_config in camera_configs.items():
            # Create queue and shutdown event for this camera
            camera_queue = queue.Queue(maxsize=config.queue_size)
            shutdown_event = threading.Event()
            error_event = threading.Event()

            self.camera_queues[mount_position] = camera_queue
            self.shutdown_events[mount_position] = shutdown_event
            self.error_events[mount_position] = error_event

            # Start camera thread
            thread = threading.Thread(
                target=self._camera_worker_wrapper,
                args=(
                    mount_position,
                    camera_config["camera_type"],
                    camera_config["device_id"],
                    camera_queue,
                    shutdown_event,
                    error_event,
                ),
            )
            thread.start()
            self.camera_threads[mount_position] = thread

        if config.run_as_server:
            self.start_server(config.port)

    def _get_camera_configs(self) -> Dict[str, str]:
        """Get camera configurations as mount_position -> camera_type mapping"""
        camera_configs = {}

        if self.config.ego_view_camera is not None:
            camera_configs[CameraMountPosition.EGO_VIEW.value] = {
                "camera_type": self.config.ego_view_camera,
                "device_id": self.config.ego_view_device_id,
            }

        if self.config.head_camera is not None:
            camera_configs[CameraMountPosition.HEAD.value] = {
                "camera_type": self.config.head_camera,
                "device_id": self.config.head_device_id,
            }

        if self.config.left_wrist_camera is not None:
            camera_configs[CameraMountPosition.LEFT_WRIST.value] = {
                "camera_type": self.config.left_wrist_camera,
                "device_id": self.config.left_wrist_device_id,
            }

        if self.config.right_wrist_camera is not None:
            camera_configs[CameraMountPosition.RIGHT_WRIST.value] = {
                "camera_type": self.config.right_wrist_camera,
                "device_id": self.config.right_wrist_device_id,
            }

        return camera_configs

    def _camera_worker_wrapper(
        self,
        mount_position: str,
        camera_type: str,
        device_id: Optional[str],
        image_queue: queue.Queue,
        shutdown_event: threading.Event,
        error_event: threading.Event,
    ):
        """Worker thread that continuously captures from a single camera"""
        try:
            camera = self._instantiate_camera(mount_position, camera_type, device_id)
            self._observation_spaces[mount_position] = camera.observation_space()

            consecutive_failures = 0
            max_consecutive_failures = 5

            while not shutdown_event.is_set():
                frame = camera.read()
                if frame:
                    consecutive_failures = 0  # Reset on successful read
                    # Non-blocking queue put with frame dropping
                    try:
                        image_queue.put_nowait(frame)
                    except queue.Full:
                        # Remove oldest frame and add new one
                        try:
                            image_queue.get_nowait()
                            image_queue.put_nowait(frame)
                        except queue.Empty:
                            pass
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        error_msg = (
                            f"Camera {mount_position} ({camera_type}) dropped: "
                            f"failed to read {consecutive_failures} consecutive frames"
                        )
                        print(f"[ERROR] {error_msg}")
                        self.error_messages[mount_position] = error_msg
                        error_event.set()
                        break

            camera.close()

        except Exception as e:
            error_msg = f"Camera {mount_position} ({camera_type}) error: {str(e)}"
            print(f"[ERROR] {error_msg}")
            self.error_messages[mount_position] = error_msg
            error_event.set()

    def _instantiate_camera(
        self, mount_position: str, camera_type: str, device_id: Optional[str] = None
    ) -> Sensor:
        """
        Instantiate a camera sensor based on the camera type.

        Args:
            camera_type: Type of camera ("oak", "oak_mono", "realsense", "zed")
            device_id: Optional device ID for the camera (used for OAK cameras)

        Returns:
            Sensor instance for the specified camera type
        """
        if camera_type in ("oak", "oak_mono"):
            from decoupled_wbc.control.sensor.oak import OAKConfig, OAKSensor

            oak_config = OAKConfig()
            if camera_type == "oak_mono":
                oak_config.enable_mono_cameras = True
            print("Initializing OAK sensor for camera type: ", camera_type)
            return OAKSensor(config=oak_config, mount_position=mount_position, device_id=device_id)
        elif camera_type == "realsense":
            from decoupled_wbc.control.sensor.realsense import RealSenseSensor

            print("Initializing RealSense sensor for camera type: ", camera_type)
            return RealSenseSensor(mount_position=mount_position)
        elif camera_type == "zed":
            from decoupled_wbc.control.sensor.zed import ZEDSensor

            print("Initializing ZED sensor for camera type: ", camera_type)
            return ZEDSensor(mount_position=mount_position)
        elif camera_type.endswith(".mp4"):
            from decoupled_wbc.control.sensor.dummy import ReplayDummySensor

            print("Initializing Replay Dummy Sensor for camera type: ", camera_type)
            return ReplayDummySensor(video_path=camera_type)
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")

    def _check_for_errors(self):
        """Check if any camera thread has encountered an error and raise exception if so."""
        for mount_position, error_event in self.error_events.items():
            if error_event.is_set():
                error_msg = self.error_messages.get(
                    mount_position, f"Camera {mount_position} encountered an unknown error"
                )
                raise RuntimeError(error_msg)

    def read(self):
        """Read frames from all cameras."""
        # Check for errors from camera threads
        self._check_for_errors()

        message = {}
        for mount_position, camera_queue in self.camera_queues.items():
            frame = self._get_latest_from_queue(camera_queue)
            if frame is not None:
                message[mount_position] = frame
        return message

    def _get_latest_from_queue(self, camera_queue: queue.Queue) -> Optional[Dict[str, Any]]:
        """Get most recent frame, discard older ones"""
        latest = None
        try:
            while True:
                latest = camera_queue.get_nowait()
        except queue.Empty:
            pass
        return latest

    def close(self):
        """Close all cameras."""
        # Signal all worker threads to shutdown
        for shutdown_event in self.shutdown_events.values():
            shutdown_event.set()

        # Wait for all threads to finish
        for thread in self.camera_threads.values():
            thread.join(timeout=5.0)

        # Clear queues
        for camera_queue in self.camera_queues.values():
            try:
                while True:
                    camera_queue.get_nowait()
            except queue.Empty:
                pass

        # Stop server if running
        if self.config.run_as_server:
            self.stop_server()

    def serialize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Merge all camera data into a single ImageMessageSchema."""
        all_timestamps = {}
        all_images = {}

        for _, camera_data in message.items():
            all_timestamps.update(camera_data.get("timestamps", {}))
            all_images.update(camera_data.get("images", {}))

        # Create a single ImageMessageSchema with all data
        img_schema = ImageMessageSchema(timestamps=all_timestamps, images=all_images)
        return img_schema.serialize()

    def run_server(self):
        """Run the server."""
        idx = 0
        server_start_time = time.monotonic()
        fps_print_time = time.monotonic()
        frame_interval = 1.0 / self.config.fps

        while True:
            # Calculate when this frame should ideally complete
            target_time = server_start_time + (idx + 1) * frame_interval

            message = self.read()
            if message:
                if self.config.test_latency:
                    read_qr_code(message)

                serialized_message = self.serialize_message(message)
                self.send_message(serialized_message)
                idx += 1

                if idx % 10 == 0:
                    print(f"Image sending FPS: {10 / (time.monotonic() - fps_print_time):.2f}")
                    fps_print_time = time.monotonic()

            # Sleep to maintain precise timing
            current_time = time.monotonic()
            sleep_time = target_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we're behind, increment idx to stay on schedule
                if not message:
                    idx += 1

    def observation_space(self):
        """Return the observation space."""
        import gymnasium as gym

        return gym.spaces.Dict(self._observation_spaces)


class ComposedCameraClientSensor(Sensor, SensorClient):
    """Class that serves as client for multiple different cameras."""

    def __init__(self, server_ip: str = "localhost", port: int = 5555):
        self.start_client(server_ip, port)

        # Initialize tracking variables
        self._latest_message = {}
        self._avg_time_per_frame = deque(maxlen=20)
        self._msg_received_time = 0
        self._start_time = 0.0  # Initialize _start_time
        self.idx = 0

        print("Initialized composed camera client sensor")

    def read(self, **kwargs) -> Optional[Dict[str, Any]]:
        self._start_time = time.time()
        message = self.receive_message()
        if not message:
            return None
        self.idx += 1

        self._latest_message = ImageMessageSchema.deserialize(message).asdict()

        # if self.idx % 10 == 0:
        #     for image_key, image_time in self._latest_message["timestamps"].items():
        #         image_latency = (time.time() - image_time) * 1000
        # print(f"Image latency for {image_key}: {image_latency:.2f} ms")

        self._msg_received_time = time.time()
        self._avg_time_per_frame.append(self._msg_received_time - self._start_time)

        return self._latest_message

    def close(self):
        """Close the client connection."""
        self.stop_client()

    def fps(self) -> float:
        """Get the current FPS of the client."""
        if len(self._avg_time_per_frame) == 0:
            return 0.0
        return float(1 / np.mean(self._avg_time_per_frame))


if __name__ == "__main__":
    """Test function for ComposedCamera."""
    import tyro

    config = tyro.cli(ComposedCameraConfig)

    if config.run_as_server:
        composed_camera = ComposedCameraSensor(config)
        print("Running composed camera server...")
        composed_camera.run_server()

    else:
        # Client mode
        composed_client = ComposedCameraClientSensor(server_ip="localhost", port=config.port)

        try:
            while True:
                data = composed_client.read()
                if data is not None:
                    print(f"FPS: {composed_client.fps():.2f}")
                    if "timestamp" in data:
                        print(f"Timestamp: {data['timestamp']}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping client...")
            composed_client.close()
