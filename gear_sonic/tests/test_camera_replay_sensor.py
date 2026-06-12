import sys
import types

import numpy as np


class _FakeVideoCapture:
    def __init__(self, _path: str):
        self._frames = [np.full((2, 3, 3), 7, dtype=np.uint8)]
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        return self._index < len(self._frames)

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def test_replay_dummy_sensor_emits_composed_camera_schema(monkeypatch) -> None:
    fake_cv2 = types.SimpleNamespace(
        COLOR_BGR2RGB=1,
        VideoCapture=_FakeVideoCapture,
        cvtColor=lambda frame, _code: frame,
        resize=lambda frame, _size: frame,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "msgpack_numpy", types.SimpleNamespace(decode=lambda value: value))

    from gear_sonic.camera.drivers import dummy

    monkeypatch.setattr(dummy.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(dummy.cv2, "cvtColor", lambda frame, _code: frame)
    monkeypatch.setattr(dummy.cv2, "resize", lambda frame, _size: frame)

    sensor = dummy.ReplayDummySensor("episode.mp4", mount_position="ego_view")

    data = sensor.read()

    assert data is not None
    assert set(data) == {"timestamps", "images"}
    assert "ego_view" in data["timestamps"]
    assert "ego_view" in data["images"]
    assert np.allclose(data["images"]["ego_view"], np.full((2, 3, 3), 7, dtype=np.uint8))
