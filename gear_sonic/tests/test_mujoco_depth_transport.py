import msgpack
import msgpack_numpy as m
import numpy as np

from gear_sonic.camera.sensor_server import ImageMessageSchema as ClientImageMessageSchema
from gear_sonic.utils.mujoco_sim.sensor_server import ImageMessageSchema


def test_mujoco_depth_round_trip_is_float_metres():
    depth_m = np.array([[0.0, 0.321], [1.234, 12.5]], dtype=np.float32)
    serialized = ImageMessageSchema(
        timestamps={"ego_view_depth": 1.0},
        images={"ego_view_depth": depth_m},
    ).serialize()
    packed = msgpack.packb(serialized, use_bin_type=True)
    message = msgpack.unpackb(packed, object_hook=m.decode)

    decoded = ClientImageMessageSchema.deserialize(message).images["ego_view_depth"]

    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, depth_m, atol=0.00051)
