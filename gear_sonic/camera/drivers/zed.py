"""Stereolabs ZED camera driver.

Requires the ZED SDK and its Python API (``pyzed``) — install the SDK from
https://www.stereolabs.com/docs/installation, then run::

    python /usr/local/zed/get_python_api.py

Streams the left RGB image, rectified by default, and optionally metric depth (uint16
millimetres) as a second ``<mount>_depth`` stream. Depth is off by default (``DEPTH_MODE.NONE``)
because it costs GPU time and the data-collection pipeline consumes RGB only.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

import pyzed.sl as sl

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    DEPTH_KEY_SUFFIX,
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)

_NS_PER_S = 1_000_000_000


def timestamp_ns_to_s(timestamp_ns: int) -> float:
    """Convert an epoch timestamp from integer nanoseconds to float seconds."""
    return timestamp_ns / _NS_PER_S


def retrieve_resolution(native: tuple[int, int], dim: tuple[int, int]) -> tuple[int, int]:
    """Smallest resolution >= ``dim`` (width, height) preserving the native aspect.

    Passed to ``retrieve_image`` so the SDK downscales on the GPU and the CPU
    only crops/converts near-target-size images (full-res conversion saturates
    a Jetson CPU at 2x30 fps).
    """
    native_w, native_h = native
    target_w, target_h = dim
    scale = max(target_w / native_w, target_h / native_h)
    return round(native_w * scale), round(native_h * scale)


def fit_to_dim(
    image: np.ndarray, dim: tuple[int, int], interpolation: int | None = None
) -> np.ndarray:
    """Center-crop ``image`` to the aspect ratio of ``dim`` (width, height), then resize.

    ZED sensors are 16:9 / 16:10 while the pipeline expects 4:3; cropping the
    sides preserves the aspect ratio instead of squashing the image.
    """
    target_w, target_h = dim
    h, w = image.shape[:2]
    if w * target_h > h * target_w:
        new_w = h * target_w // target_h
        x0 = (w - new_w) // 2
        image = image[:, x0 : x0 + new_w]
    elif w * target_h < h * target_w:
        new_h = w * target_h // target_w
        y0 = (h - new_h) // 2
        image = image[y0 : y0 + new_h, :]
    if image.shape[:2] != (target_h, target_w):
        kwargs = {} if interpolation is None else {"interpolation": interpolation}
        image = cv2.resize(image, (target_w, target_h), **kwargs)
    return image


@dataclass
class ZEDConfig:
    """Configuration for the ZED camera.

    ``resolution`` is the name of an ``sl.RESOLUTION`` member (``AUTO`` keeps each model's
    native mode). ``fps`` is the camera's grab rate; the SDK falls back to the nearest rate
    the selected mode supports. ``color_image_dim`` is the (width, height) the frames are
    center-cropped and resized to before publishing — the data pipeline expects 640x480.
    ``rectified`` selects the rectified image (lens distortion removed) or the raw sensor
    image. ``use_depth`` additionally publishes metric depth as a ``<mount>_depth`` stream
    (uint16 millimetres), computed with ``depth_mode``; it costs GPU time, and monocular
    cameras cannot provide it.
    """

    color_image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    resolution: str = "AUTO"
    rectified: bool = True
    use_depth: bool = False
    depth_mode: str = "NEURAL"

    def __post_init__(self) -> None:
        # depth measures are always in the rectified left frame; an unrectified RGB would not align
        if self.use_depth and not self.rectified:
            raise ValueError("use_depth requires rectified=True: depth is only defined there")

    def sl_view(self) -> "sl.VIEW":
        return sl.VIEW.LEFT if self.rectified else sl.VIEW.LEFT_UNRECTIFIED

    def sl_depth_mode(self) -> "sl.DEPTH_MODE":
        if not self.use_depth:
            return sl.DEPTH_MODE.NONE
        try:
            return getattr(sl.DEPTH_MODE, self.depth_mode)
        except AttributeError as e:
            valid = [m.name for m in sl.DEPTH_MODE if m.name != "LAST"]
            raise RuntimeError(f"Unknown ZED depth mode {self.depth_mode!r}; valid: {valid}") from e

    def sl_resolution(self) -> "sl.RESOLUTION":
        try:
            return getattr(sl.RESOLUTION, self.resolution)
        except AttributeError as e:
            valid = [r.name for r in sl.RESOLUTION if r.name != "LAST"]
            raise RuntimeError(f"Unknown ZED resolution {self.resolution!r}; valid: {valid}") from e


class ZEDSensorBase(Sensor, SensorServer):
    """Shared implementation behind :class:`ZEDSensor` and :class:`ZEDOneSensor`.

    The SDK drives stereo cameras (``sl.Camera``) and single-sensor cameras (``sl.CameraOne``)
    through two similar but separate APIs, so a subclass implements only the three calls that
    differ between them:

    * ``list_devices()`` - enumerate the cameras that API can see
    * ``_open()`` - open one and assign it to ``self.zed``
    * ``_grab()`` - advance the camera one frame

    Everything else lives here and is inherited unchanged: ``read()`` with its 4:3 crop and SDK
    timestamps, ``serialize()``, ``observation_space()``, ``close()`` and ``run_server()``.
    """

    _model_label = "ZED"

    @staticmethod
    def list_devices():
        raise NotImplementedError

    def _open(self, config: "ZEDConfig", device_id: str | None) -> None:
        raise NotImplementedError

    def _grab(self):
        raise NotImplementedError

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: ZEDConfig | None = None,
        device_id: str | None = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        config = config if config is not None else ZEDConfig()
        devices = self.list_devices()
        if len(devices) == 0:
            raise RuntimeError(f"No {self._model_label} devices found")

        for device in devices:
            print(f"Device: {device.camera_model}")
            print(f"    Serial number: {device.serial_number}")

        # With several cameras attached, a serial is mandatory: otherwise every worker would
        # race for the first device and all but one open() would fail.
        if device_id is None and len(devices) > 1:
            serials = [str(d.serial_number) for d in devices]
            raise RuntimeError(
                f"{len(devices)} {self._model_label} cameras are attached ({', '.join(serials)}); "
                "pass the serial number of the one to use via --<mount>-device-id."
            )

        # Always name the device explicitly. Left to itself the SDK opens the first camera of its
        # own, unfiltered list, which for the monocular API can be a sensor belonging to a stereo
        # camera — i.e. not the camera `list_devices()` just reported.
        if device_id is None:
            device_id = str(devices[0].serial_number)

        self._open(config, device_id)
        self._config = config
        self._image = sl.Mat()
        self._depth = sl.Mat() if config.use_depth else None
        native = self.zed.get_camera_information().camera_configuration.resolution
        retrieve_w, retrieve_h = retrieve_resolution(
            (native.width, native.height), config.color_image_dim
        )
        self._retrieve_res = sl.Resolution(retrieve_w, retrieve_h)
        self._run_as_server = run_as_server
        self.mount_position = mount_position
        if self._run_as_server:
            self.start_server(port)
        serial = self.zed.get_camera_information().serial_number
        print(f"Done initializing {self._model_label} sensor: {serial}")

    def read(self) -> dict[str, Any] | None:
        status = self._grab()
        if status > sl.ERROR_CODE.SUCCESS:
            print(f"WARNING! ZED grab failed: {status}")
            return None
        if status != sl.ERROR_CODE.SUCCESS:  # negative (e.g. CORRUPTED_FRAME): frame still usable
            print(f"WARNING! ZED grab: {status!r}")

        status = self.zed.retrieve_image(
            self._image, self._config.sl_view(), sl.MEM.CPU, self._retrieve_res
        )
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"WARNING! ZED retrieve_image failed: {status}")
            return None

        bgra = self._image.get_data()
        if bgra is None or bgra.size == 0:
            print("WARNING! Empty ZED image")
            return None

        image = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        image = fit_to_dim(image, self._config.color_image_dim)
        timestamp = timestamp_ns_to_s(
            self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        )
        timestamps = {self.mount_position: timestamp}
        images = {self.mount_position: image}

        if self._depth is not None:
            # native resolution: asking the SDK for a smaller Mat resamples bilinearly (~+5 ms/frame)
            status = self.zed.retrieve_measure(self._depth, sl.MEASURE.DEPTH_U16_MM, sl.MEM.CPU)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"WARNING! ZED retrieve_measure failed: {status}")
                return None
            depth_mm = self._depth.get_data()
            if depth_mm is None or depth_mm.size == 0:
                print("WARNING! Empty ZED depth measure")
                return None
            # uint16 mm, invalid already 0. Nearest-neighbour: interpolating depth would invent
            # distances at object edges. The resize also detaches the result from the SDK's Mat.
            depth = fit_to_dim(
                depth_mm, self._config.color_image_dim, interpolation=cv2.INTER_NEAREST
            )
            key = f"{self.mount_position}{DEPTH_KEY_SUFFIX}"
            timestamps[key] = timestamp
            images[key] = depth

        return {"timestamps": timestamps, "images": images}

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        w, h = self._config.color_image_dim
        spaces = {"color_image": gym.spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)}
        if self._config.use_depth:
            # DEPTH_U16_MM clamps at 65000 mm; 0 marks an invalid pixel
            spaces["depth_image"] = gym.spaces.Box(low=0, high=65000, shape=(h, w), dtype=np.uint16)
        return gym.spaces.Dict(spaces)

    def close(self):
        if self._run_as_server:
            self.stop_server()
        self.zed.close()

    def run_server(self):
        if not self._run_as_server:
            raise ValueError("run_as_server must be True to call run_server()")
        while True:
            read_result = self.read()
            if read_result is None:
                continue
            self.send_message({self.mount_position: self.serialize(read_result)})


class ZEDSensor(ZEDSensorBase):
    """Sensor for Stereolabs ZED stereo cameras (``sl.Camera``)."""

    _model_label = "ZED"

    @staticmethod
    def list_devices():
        return sl.Camera.get_device_list()

    def _open(self, config: "ZEDConfig", device_id: str | None) -> None:
        init_params = sl.InitParameters()
        init_params.camera_resolution = config.sl_resolution()
        init_params.camera_fps = config.fps
        init_params.depth_mode = config.sl_depth_mode()
        init_params.enable_image_validity_check = 1  # else CORRUPTED_FRAME never fires
        if device_id is not None:
            try:
                init_params.set_from_serial_number(int(device_id))
            except ValueError:
                raise RuntimeError(f"Invalid ZED serial number: {device_id!r}")

        self.zed = sl.Camera()
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")
        self._runtime_params = sl.RuntimeParameters()

    def _grab(self):
        return self.zed.grab(self._runtime_params)


class ZEDOneSensor(ZEDSensorBase):
    """Sensor for the monocular Stereolabs ZED X One.

    The SDK drives single-sensor cameras through ``sl.CameraOne``: its own device list, its own
    init parameters (no depth mode) and a ``grab()`` that takes no runtime parameters. Only those
    three calls are overridden; ``read()``, ``serialize()``, ``observation_space()``, ``close()``
    and ``run_server()`` are inherited from :class:`ZEDSensorBase` unchanged, since this pipeline
    consumes RGB only.
    """

    _model_label = "ZED X One"

    @staticmethod
    def list_devices():
        return sl.CameraOne.get_device_list()

    def __init__(self, *args, config: "ZEDConfig | None" = None, **kwargs):
        # Validated before device enumeration, so a misconfiguration is not masked by a
        # "no devices found" error when no X One happens to be attached.
        if config is not None and config.use_depth:
            raise RuntimeError(
                "The ZED X One is monocular and cannot produce depth: a single sensor has nothing "
                "to triangulate against. Use a stereo ZED, or set use_depth=False."
            )
        super().__init__(*args, config=config, **kwargs)

    def _open(self, config: "ZEDConfig", device_id: str | None) -> None:
        init_params = sl.InitParametersOne()
        init_params.camera_resolution = config.sl_resolution()
        init_params.camera_fps = config.fps
        if device_id is not None:
            try:
                init_params.set_from_serial_number(int(device_id))
            except ValueError:
                raise RuntimeError(f"Invalid ZED X One serial number: {device_id!r}")

        self.zed = sl.CameraOne()
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED X One camera: {status}")

    def _grab(self):
        return self.zed.grab()  # CameraOne.grab() takes no RuntimeParameters
