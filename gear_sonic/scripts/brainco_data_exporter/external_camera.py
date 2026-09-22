"""External-view camera ZMQ publisher and dataset-side subscriber."""

from __future__ import annotations

import threading
import time

import msgpack
import numpy as np
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


EXTERNAL_VIEW_STREAM_NAME = "external-view-camera"


class ExternalViewCameraPublisher:
    """Capture a V4L2/OpenCV device and publish JPEG frames over ZMQ."""

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: float,
        fourcc: str | None = None,
        publish_port: int | None = None,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "The external-view camera publisher requires OpenCV. "
                "Install opencv-python in the data-collection environment."
            ) from exc

        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("External-view camera width, height, and fps must be positive")
        if fourcc is not None and len(fourcc) != 4:
            raise ValueError("fourcc must contain exactly four characters")
        if publish_port is not None and not 1 <= publish_port <= 65535:
            raise ValueError("publish port must be in 1..65535")

        self._cv2 = cv2
        self._device = device
        self._requested_shape = (height, width)
        self._publish_port = publish_port
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp: float | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: Exception | None = None

        # CAP_V4L2 makes /dev/video* selection deterministic on Linux.  Fall
        # back to OpenCV's automatic backend for non-V4L2 paths/backends.
        self._capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = cv2.VideoCapture(device)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open external-view camera device {device!r}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if fourcc is not None:
            self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="external-view-camera",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        context = None
        publisher = None
        try:
            if self._publish_port is not None:
                context = zmq.Context()
                publisher = context.socket(zmq.PUB)
                publisher.setsockopt(zmq.SNDHWM, 2)
                publisher.setsockopt(zmq.LINGER, 0)
                publisher.bind(f"tcp://*:{self._publish_port}")
                print(
                    "[ExternalViewCamera] Preview publishing on "
                    f"tcp://*:{self._publish_port}"
                )

            while not self._stop.is_set():
                ok, frame_bgr = self._capture.read()
                if not ok or frame_bgr is None:
                    if self._stop.wait(0.02):
                        return
                    continue
                if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
                    raise RuntimeError(
                        "External-view camera must return a three-channel colour frame, "
                        f"got shape {frame_bgr.shape}"
                    )
                if frame_bgr.shape[:2] != self._requested_shape:
                    raise RuntimeError(
                        "External-view camera did not apply the requested resolution: "
                        f"requested {self._requested_shape[1]}x{self._requested_shape[0]}, "
                        f"got {frame_bgr.shape[1]}x{frame_bgr.shape[0]}. "
                        "Use a supported V4L2 profile or pass its actual width and height."
                    )
                timestamp = time.time()
                frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._latest_frame = frame_rgb
                    self._latest_timestamp = timestamp
                self._ready.set()

                if publisher is not None:
                    ok, encoded = self._cv2.imencode(
                        ".jpg",
                        frame_bgr,
                        [int(self._cv2.IMWRITE_JPEG_QUALITY), 80],
                    )
                    if ok:
                        payload = msgpack.packb(
                            {
                                "timestamps": {EXTERNAL_VIEW_STREAM_NAME: timestamp},
                                "images": {
                                    EXTERNAL_VIEW_STREAM_NAME: encoded.tobytes()
                                },
                            },
                            use_bin_type=True,
                        )
                        try:
                            publisher.send(payload, flags=zmq.NOBLOCK)
                        except zmq.Again:
                            pass
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            if publisher is not None:
                publisher.close()
            if context is not None:
                context.term()

    def wait_until_ready(self, timeout_sec: float) -> None:
        if not self._ready.wait(timeout=None if timeout_sec == 0 else timeout_sec):
            self._raise_capture_error_or_timeout(timeout_sec)
        self._raise_capture_error_or_timeout(timeout_sec)

    def _raise_capture_error_or_timeout(self, timeout_sec: float) -> None:
        if self._error is not None:
            raise self._error
        if not self._ready.is_set():
            raise TimeoutError(
                f"No frame from external-view camera {self._device!r} within {timeout_sec} s"
            )

    def latest_frame(self) -> np.ndarray:
        self._raise_capture_error_or_timeout(0)
        with self._lock:
            if self._latest_frame is None:
                raise RuntimeError("External-view camera has no frame")
            return self._latest_frame.copy()

    def check_health(self) -> None:
        """Raise an asynchronous capture/publish error, if one occurred."""
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._capture.release()


class ExternalViewCameraSubscriber:
    """Read the latest external-view RGB frame from a standalone publisher."""

    def __init__(self, host: str, port: int) -> None:
        if not host:
            raise ValueError("external_view_camera_host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("external_view_camera_port must be in 1..65535")
        self._host = host
        self._port = port
        self._client = ComposedCameraClientSensor(server_ip=host, port=port)
        self._latest_frame: np.ndarray | None = None

    def _poll(self) -> bool:
        sample = self._client.read(blocking=False)
        if sample is None:
            return False
        frame = sample.get("images", {}).get(EXTERNAL_VIEW_STREAM_NAME)
        if frame is None:
            return False
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(
                "External-view publisher returned an invalid RGB frame: "
                f"shape={frame.shape}"
            )
        self._latest_frame = frame.copy()
        return True

    def wait_until_ready(self, timeout_sec: float) -> None:
        deadline = None if timeout_sec == 0 else time.monotonic() + timeout_sec
        while not self._poll():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "No external-view-camera frame from "
                    f"tcp://{self._host}:{self._port} within {timeout_sec} s"
                )
            time.sleep(0.02)

    def latest_frame(self) -> np.ndarray:
        self._poll()
        if self._latest_frame is None:
            raise RuntimeError(
                f"No external-view-camera frame from tcp://{self._host}:{self._port}"
            )
        return self._latest_frame.copy()

    def close(self) -> None:
        self._client.close()


# Backward-compatible name for code that used the original in-process camera.
ExternalViewCamera = ExternalViewCameraPublisher
