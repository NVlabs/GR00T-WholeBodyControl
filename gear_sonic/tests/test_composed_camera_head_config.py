import sys
from types import ModuleType

import numpy as np

from gear_sonic.camera.composed_camera import ComposedCameraConfig, ComposedCameraSensor
from gear_sonic.camera.sensor_server import CameraMountPosition, ImageUtils


def _sensor_without_hardware(config: ComposedCameraConfig) -> ComposedCameraSensor:
    sensor = ComposedCameraSensor.__new__(ComposedCameraSensor)
    sensor.config = config
    return sensor


def test_usb_head_ignores_realsense_depth_dimensions():
    config = ComposedCameraConfig(
        ego_view_camera=None,
        head_camera="usb",
        head_camera_width=1280,
        head_camera_height=960,
        head_camera_fps=15,
        realsense_depth=True,
        realsense_depth_width=0,
        realsense_depth_height=0,
    )

    assert config.head_image_dim == (1280, 960)
    assert config.effective_head_fps == 15


def test_usb_head_receives_its_own_capture_settings(monkeypatch):
    captured = {}

    class FakeUsbSensor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import gear_sonic.camera.drivers.usb_camera as usb_module

    monkeypatch.setattr(usb_module, "USBCameraSensor", FakeUsbSensor)
    config = ComposedCameraConfig(
        ego_view_camera=None,
        head_camera="usb",
        head_device_id="/dev/video4",
        head_camera_width=1280,
        head_camera_height=960,
        head_camera_fps=15,
        head_camera_fourcc="MJPG",
    )
    sensor = _sensor_without_hardware(config)

    sensor._instantiate_camera(CameraMountPosition.HEAD.value, "usb", "/dev/video4")

    assert captured["device_index"] == "/dev/video4"
    assert captured["config"].image_dim == (1280, 960)
    assert captured["config"].fps == 15
    assert captured["config"].fourcc == "MJPG"


def test_realsense_head_receives_head_specific_rgb_settings(monkeypatch):
    captured = {}
    fake_module = ModuleType("gear_sonic.camera.drivers.realsense")

    class FakeRealSenseConfig:
        pass

    class FakeRealSenseSensor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module.RealSenseConfig = FakeRealSenseConfig
    fake_module.RealSenseSensor = FakeRealSenseSensor
    monkeypatch.setitem(sys.modules, "gear_sonic.camera.drivers.realsense", fake_module)

    config = ComposedCameraConfig(
        ego_view_camera=None,
        head_camera="realsense",
        head_device_id="1234",
        head_camera_width=1280,
        head_camera_height=960,
        head_camera_fps=15,
        realsense_depth=False,
    )
    sensor = _sensor_without_hardware(config)

    sensor._instantiate_camera(CameraMountPosition.HEAD.value, "realsense", "1234")

    camera_config = captured["config"]
    assert camera_config.color_image_dim == (1280, 960)
    assert camera_config.fps == 15
    assert camera_config.enable_depth is False


def test_head_quality_only_changes_head_jpeg(monkeypatch):
    qualities = {}

    def fake_encode(image, quality=80):
        qualities[int(image[0, 0, 0])] = quality
        return "encoded"

    monkeypatch.setattr(ImageUtils, "encode_image", staticmethod(fake_encode))
    config = ComposedCameraConfig(
        ego_view_camera="usb",
        head_camera="usb",
        head_camera_quality=93,
        realsense_depth_width=0,
        realsense_depth_height=0,
    )
    sensor = _sensor_without_hardware(config)
    image_ego = np.full((2, 2, 3), 1, dtype=np.uint8)
    image_head = np.full((2, 2, 3), 2, dtype=np.uint8)

    sensor.serialize_message(
        {
            "ego_view": {
                "timestamps": {"ego_view": 1.0},
                "images": {"ego_view": image_ego},
            },
            "head": {
                "timestamps": {"head": 1.0},
                "images": {"head": image_head},
            },
        }
    )

    assert qualities == {1: 80, 2: 93}
