#!/usr/bin/env python3
"""
Standalone ZMQ server that captures frames from a RealSense camera and publishes
them over ZMQ PUB socket. Designed to run on the Orin (G1 robot) and be consumed
by RemoteCameraSensor on the PC side.

No decoupled_wbc imports — only needs: pyrealsense2, zmq, msgpack, cv2, numpy.

Usage:
    python3 realsense_zmq_server.py --port 5556 --mount-position ego_view

Copy this script to the Orin via scp:
    scp realsense_zmq_server.py unitree@192.168.123.164:/tmp/
"""

import argparse
import base64
import time

import cv2
import msgpack
import numpy as np
import pyrealsense2 as rs
import zmq


def encode_image(image: np.ndarray) -> str:
    """Encode image as base64 JPEG (matches ImageMessageSchema.serialize format)."""
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="RealSense ZMQ streaming server")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB port")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument(
        "--mount-position",
        type=str,
        default="ego_view",
        help="Camera mount position identifier",
    )
    args = parser.parse_args()

    # Set up ZMQ PUB socket
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 20)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://*:{args.port}")
    print(f"ZMQ PUB socket bound to tcp://*:{args.port}")

    # Set up RealSense pipeline
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )

    profile = pipeline.start(rs_config)
    device = profile.get_device()
    print(
        f"Connected to RealSense: {device.get_info(rs.camera_info.name)} "
        f"(SN: {device.get_info(rs.camera_info.serial_number)})"
    )

    frame_count = 0
    fps_start = time.monotonic()

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            rgb = bgr[..., ::-1]  # BGR to RGB
            capture_time = time.time()

            # Build message matching ImageMessageSchema.serialize() format
            message = {
                "timestamps": {args.mount_position: capture_time},
                "images": {args.mount_position: encode_image(rgb)},
            }

            packed = msgpack.packb(message, use_bin_type=True)
            try:
                sock.send(packed, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - fps_start
                print(f"Sent {frame_count} frames, FPS: {100 / elapsed:.1f}")
                fps_start = time.monotonic()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        pipeline.stop()
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
