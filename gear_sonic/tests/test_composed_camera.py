"""Configuration-level tests for ComposedCameraSensor (no camera needed)."""

import pytest

from gear_sonic.camera.composed_camera import ComposedCameraConfig


def test_config_rejects_zed_depth_with_unrectified_rgb():
    with pytest.raises(ValueError, match="no-zed-rectified"):
        ComposedCameraConfig(ego_view_camera="zed", zed_use_depth=True, zed_rectified=False)
    # only a ZED rig is concerned; other rigs ignore the zed flags
    ComposedCameraConfig(ego_view_camera="oak", zed_use_depth=True, zed_rectified=False)
