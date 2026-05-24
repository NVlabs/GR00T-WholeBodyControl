from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sensor_msgs.msg")
pytest.importorskip("cv2")

from decoupled_wbc.control.utils.cv_bridge import CvBridge


def test_cv_bridge_reports_uint16_for_16uc1():
    bridge = CvBridge()

    assert bridge.encoding_to_dtype_with_channels("16UC1") == ("uint16", 1)


def test_cv_bridge_preserves_16uc1_depth_values():
    bridge = CvBridge()
    depth = np.array([[1000, 2000, 3000], [4000, 5000, 6000]], dtype=np.uint16)

    msg = type("Msg", (), {})()
    msg.encoding = "16UC1"
    msg.is_bigendian = False
    msg.height = depth.shape[0]
    msg.width = depth.shape[1]
    msg.data = depth.tobytes()

    decoded = bridge.imgmsg_to_cv2(msg)

    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, depth)
