"""Tests for the ZED camera driver.

Contract tests run without hardware; the live test is skipped unless a ZED
camera is connected. The whole module is skipped when pyzed (ZED SDK Python
API) is not installed.
"""

import time

import cv2
import numpy as np
import pytest

sl = pytest.importorskip("pyzed.sl")

from gear_sonic.camera.drivers.zed import (  # noqa: E402
    ZEDConfig,
    ZEDOneSensor,
    ZEDSensor,
    fit_to_dim,
    retrieve_resolution,
    timestamp_ns_to_s,
)
from gear_sonic.camera.sensor_server import (  # noqa: E402
    CameraMountPosition,
    ImageMessageSchema,
    is_depth_key,
)


def _zed_connected() -> bool:
    return len(sl.Camera.get_device_list()) > 0


def _first_zed_serial() -> str:
    # the sensor refuses to guess when several cameras are attached, so name one
    return str(sl.Camera.get_device_list()[0].serial_number)


def test_timestamp_ns_to_s():
    assert timestamp_ns_to_s(1_700_000_000_500_000_000) == pytest.approx(1_700_000_000.5)


@pytest.mark.parametrize(
    "input_hw",
    [
        (720, 1280),  # ZED 2i HD720, 16:9
        (1200, 1920),  # ZED X HD1200, 16:10
        (376, 672),  # ZED 2i VGA (upscale)
        (480, 640),  # already target
        (600, 400),  # taller than 4:3 (height crop)
    ],
)
def test_fit_to_dim_output_shape(input_hw):
    image = np.zeros((*input_hw, 3), dtype=np.uint8)
    out = fit_to_dim(image, (640, 480))
    assert out.shape == (480, 640, 3)
    assert out.dtype == np.uint8


@pytest.mark.parametrize(
    "native,expected",
    [
        ((1920, 1200), (768, 480)),  # ZED X HD1200, 16:10 -> crop width
        ((1920, 1536), (640, 512)),  # ZED X HDR HD1536, 5:4 -> crop height
        ((1280, 720), (853, 480)),  # ZED 2i HD720, 16:9
        ((640, 480), (640, 480)),  # exact
    ],
)
def test_retrieve_resolution(native, expected):
    out = retrieve_resolution(native, (640, 480))
    assert out == expected
    assert out[0] >= 640 and out[1] >= 480


def test_retrieve_then_fit_yields_target():
    for native in [(1920, 1200), (1920, 1536), (1280, 720), (672, 376)]:
        w, h = retrieve_resolution(native, (640, 480))
        image = np.zeros((h, w, 3), dtype=np.uint8)
        assert fit_to_dim(image, (640, 480)).shape == (480, 640, 3)


def test_fit_to_dim_center_crops_width():
    # 16:9 frame with red side margins and a green 4:3 center: the margins
    # must be cropped away, not squashed into the output.
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    x0 = (1280 - 960) // 2
    image[:, x0 : x0 + 960] = (0, 255, 0)
    out = fit_to_dim(image, (640, 480))
    assert (out[:, :, 1] == 255).all()
    assert (out[:, :, 0] == 0).all()


def test_zed_config_defaults():
    config = ZEDConfig()
    assert config.color_image_dim == (640, 480)  # the data pipeline's fixed contract
    assert config.fps == 30
    assert config.resolution == "AUTO"


def test_zed_config_resolves_resolution_name():
    assert ZEDConfig(resolution="HD720").sl_resolution() is sl.RESOLUTION.HD720
    assert ZEDConfig().sl_resolution() is sl.RESOLUTION.AUTO


def test_zed_config_rejects_unknown_resolution():
    with pytest.raises(RuntimeError, match="Unknown ZED resolution"):
        ZEDConfig(resolution="HD9000").sl_resolution()


def test_zed_config_selects_the_view_from_rectified():
    assert ZEDConfig().sl_view() is sl.VIEW.LEFT
    assert ZEDConfig(rectified=False).sl_view() is sl.VIEW.LEFT_UNRECTIFIED


def test_zed_config_depth_mode_off_by_default():
    assert ZEDConfig().sl_depth_mode() is sl.DEPTH_MODE.NONE
    assert ZEDConfig(use_depth=True).sl_depth_mode() is sl.DEPTH_MODE.NEURAL
    assert (
        ZEDConfig(use_depth=True, depth_mode="NEURAL_LIGHT").sl_depth_mode()
        is sl.DEPTH_MODE.NEURAL_LIGHT
    )


def test_zed_config_rejects_unknown_depth_mode():
    with pytest.raises(RuntimeError, match="Unknown ZED depth mode"):
        ZEDConfig(use_depth=True, depth_mode="MAGIC").sl_depth_mode()


def test_zed_config_rejects_depth_with_unrectified_rgb():
    """Depth lives in the rectified frame; pairing it with LEFT_UNRECTIFIED would misalign the streams."""
    with pytest.raises(ValueError, match="rectified=True"):
        ZEDConfig(use_depth=True, rectified=False)
    ZEDConfig(use_depth=False, rectified=False)  # RGB-only unrectified stays allowed


def _sensor_with_config(config: ZEDConfig) -> ZEDSensor:
    # observation_space() only reads the config, so skip the constructor's device open
    sensor = ZEDSensor.__new__(ZEDSensor)
    sensor._config = config
    return sensor


def test_observation_space_matches_the_published_streams():
    gym = pytest.importorskip("gymnasium")
    rgb_only = _sensor_with_config(ZEDConfig()).observation_space()
    assert set(rgb_only.spaces) == {"color_image"}
    assert rgb_only["color_image"].shape == (480, 640, 3)

    with_depth = _sensor_with_config(ZEDConfig(use_depth=True)).observation_space()
    assert set(with_depth.spaces) == {"color_image", "depth_image"}
    assert with_depth["depth_image"].shape == (480, 640)  # same (h, w) as read() returns
    assert with_depth["depth_image"].dtype == np.uint16
    assert with_depth["depth_image"].high.max() == 65000
    assert isinstance(with_depth, gym.spaces.Dict)


def test_depth_key_convention():
    assert is_depth_key("ego_view_depth")
    assert not is_depth_key("ego_view")


def test_depth_survives_serialization_losslessly():
    """Depth must not go through the JPEG path: OpenCV would silently down-convert it to 8 bit."""
    key = CameraMountPosition.EGO_VIEW.value
    depth_key = f"{key}_depth"
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = (np.random.rand(480, 640) * 9000).astype(np.uint16)

    schema = ImageMessageSchema(
        timestamps={key: 1.0, depth_key: 1.0}, images={key: rgb, depth_key: depth}
    )
    decoded = ImageMessageSchema.deserialize(schema.serialize())

    assert decoded.images[depth_key].dtype == np.uint16
    assert (decoded.images[depth_key] == depth).all()  # bit-exact millimetres
    assert decoded.images[key].shape == (480, 640, 3)


def test_fit_to_dim_nearest_keeps_exact_depth_values():
    depth = (np.arange(480 * 640, dtype=np.uint32).reshape(480, 640) % 9000).astype(np.uint16)
    out = fit_to_dim(depth, (320, 240), interpolation=cv2.INTER_NEAREST)
    assert out.shape == (240, 320)
    assert set(np.unique(out)).issubset(set(np.unique(depth)))  # no invented in-between distances


def test_monocular_sensor_rejects_depth_without_needing_hardware():
    """A single sensor cannot triangulate; say so before hunting for devices."""
    with pytest.raises(RuntimeError, match="monocular and cannot produce depth"):
        ZEDOneSensor(config=ZEDConfig(use_depth=True))


def test_serialization_roundtrip():
    key = CameraMountPosition.EGO_VIEW.value
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    t = time.time()
    schema = ImageMessageSchema(timestamps={key: t}, images={key: frame})
    decoded = ImageMessageSchema.deserialize(schema.serialize())
    assert decoded.timestamps == {key: t}
    assert set(decoded.images.keys()) == {key}
    assert decoded.images[key].shape == (480, 640, 3)
    assert decoded.images[key].dtype == np.uint8


@pytest.mark.skipif(not _zed_connected(), reason="no ZED camera connected")
def test_live_zed_sensor():
    sensor = ZEDSensor(
        mount_position=CameraMountPosition.EGO_VIEW.value, device_id=_first_zed_serial()
    )
    try:
        frames = []
        for _ in range(50):
            result = sensor.read()
            if result is not None:
                frames.append(result)
            if len(frames) >= 10:
                break
        assert len(frames) >= 10, "could not read 10 frames"

        timestamps = []
        for result in frames:
            image = result["images"]["ego_view"]
            assert image.shape == (480, 640, 3)
            assert image.dtype == np.uint8
            timestamps.append(result["timestamps"]["ego_view"])

        assert timestamps == sorted(timestamps)
        assert abs(timestamps[-1] - time.time()) < 5.0

        serialized = sensor.serialize(frames[-1])
        decoded = ImageMessageSchema.deserialize(serialized)
        assert decoded.images["ego_view"].shape == (480, 640, 3)
    finally:
        sensor.close()


@pytest.mark.skipif(not _zed_connected(), reason="no ZED camera connected")
def test_live_zed_sensor_depth():
    """The depth stream is only reachable with hardware, so assert its contract here.

    MEASURE.DEPTH_U16_MM must arrive as uint16 millimetres with 0 for invalid pixels,
    aligned with the colour stream and independent of InitParameters.coordinate_units.
    """
    sensor = ZEDSensor(
        mount_position=CameraMountPosition.EGO_VIEW.value,
        device_id=_first_zed_serial(),
        config=ZEDConfig(use_depth=True),
    )
    try:
        frames = []
        for _ in range(50):
            result = sensor.read()
            if result is not None:
                frames.append(result)
            if len(frames) >= 5:
                break
        assert len(frames) >= 5, "could not read 5 frames with depth enabled"

        depth_key = f"{CameraMountPosition.EGO_VIEW.value}_depth"
        for result in frames:
            assert is_depth_key(depth_key)
            depth = result["images"][depth_key]
            assert depth.dtype == np.uint16, "depth must stay uint16 millimetres"
            # same dimensions as the colour stream, so the two are pixel-aligned
            assert depth.shape[:2] == result["images"][CameraMountPosition.EGO_VIEW.value].shape[:2]
            assert depth.max() <= 65000, "the SDK clamps DEPTH_U16_MM at 65000"

        # at least one frame must carry real measurements, otherwise this proves nothing
        assert any(f["images"][depth_key].max() > 0 for f in frames), "every depth frame was empty"

        # the frames must not alias the Mat the SDK reuses
        first, last = frames[0]["images"][depth_key], frames[-1]["images"][depth_key]
        assert first is not last
        assert not np.shares_memory(first, last)

        # depth is retrieved at native resolution and downsampled nearest-neighbour, so every
        # published value is one the SDK produced: a bilinear path would create in-between values
        native = sensor.zed.get_camera_information().camera_configuration.resolution
        assert sensor._depth.get_width() == native.width
        assert sensor._depth.get_height() == native.height
        assert set(np.unique(last)) <= set(np.unique(sensor._depth.get_data()))
    finally:
        sensor.close()
