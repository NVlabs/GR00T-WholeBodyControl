import struct

import numpy as np

from gear_sonic.camera.xrobo_video_bridge import (
    compose_head_and_wrist_frame,
    deserialize_camera_request,
    deserialize_control_body,
    prepare_head_frame,
    serialize_control_message,
)


def test_control_protocol_round_trip():
    packet = serialize_control_message("OPEN_CAMERA", b"payload")
    body_len = struct.unpack(">I", packet[:4])[0]
    assert body_len == len(packet) - 4
    assert deserialize_control_body(packet[4:]) == ("OPEN_CAMERA", b"payload")


def test_camera_request_deserialization():
    camera = b"VR"
    ip = b"192.168.1.42"
    payload = (
        b"\xca\xfe\x01"
        + struct.pack("<7i", 1280, 480, 15, 1_000_000, 0, 2, 12345)
        + bytes([len(camera)])
        + camera
        + bytes([len(ip)])
        + ip
    )
    request = deserialize_camera_request(payload)
    assert request.width == 1280
    assert request.height == 480
    assert request.fps == 15
    assert request.camera == "VR"
    assert request.ip == "192.168.1.42"
    assert request.port == 12345


def test_prepare_head_frame_duplicates_mono_for_sbs():
    mono = np.zeros((4, 4, 3), dtype=np.uint8)
    mono[:, :, 1] = np.arange(4, dtype=np.uint8)
    output = prepare_head_frame(mono, 8, 4, binocular=False)
    np.testing.assert_array_equal(output[:, :4], mono)
    np.testing.assert_array_equal(output[:, 4:], mono)


def test_prepare_head_frame_keeps_binocular_frame():
    stereo = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    output = prepare_head_frame(stereo, 8, 4, binocular=True)
    np.testing.assert_array_equal(output, stereo)


def test_compose_head_and_wrist_frame_repeats_panels_for_both_eyes():
    head = np.zeros((4, 8, 3), dtype=np.uint8)
    head[:, :, 0] = 10
    left = np.zeros((2, 2, 3), dtype=np.uint8)
    left[:, :, 1] = 20
    right = np.zeros((2, 2, 3), dtype=np.uint8)
    right[:, :, 2] = 30

    output = compose_head_and_wrist_frame(
        head,
        left,
        right,
        8,
        6,
        binocular=True,
    )

    assert output.shape == (6, 8, 3)
    np.testing.assert_array_equal(output[:4], head)
    np.testing.assert_array_equal(output[4:, 0:2], left)
    np.testing.assert_array_equal(output[4:, 2:4], right)
    np.testing.assert_array_equal(output[4:, 4:6], left)
    np.testing.assert_array_equal(output[4:, 6:8], right)


def test_compose_head_and_wrist_frame_pillarboxes_without_distortion():
    head = np.zeros((6, 16, 3), dtype=np.uint8)
    head[:, :8, 0] = 10
    head[:, 8:, 0] = 11
    left = np.zeros((3, 4, 3), dtype=np.uint8)
    left[:, :, 1] = 20
    right = np.zeros((3, 4, 3), dtype=np.uint8)
    right[:, :, 2] = 30

    output = compose_head_and_wrist_frame(
        head,
        left,
        right,
        24,
        9,
        binocular=True,
    )

    assert output.shape == (9, 24, 3)
    # Each 12x9 eye is 4:3. The undistorted 8x9 composite is centered with
    # two black columns on either side.
    assert np.all(output[:, :2] == 0)
    assert np.all(output[:, 10:14] == 0)
    assert np.all(output[:, 22:] == 0)
    assert np.all(output[:6, 2:10, 0] == 10)
    assert np.all(output[:6, 14:22, 0] == 11)
    assert np.all(output[6:, 2:6, 1] == 20)
    assert np.all(output[6:, 6:10, 2] == 30)
    assert np.all(output[6:, 14:18, 1] == 20)
    assert np.all(output[6:, 18:22, 2] == 30)
