#!/usr/bin/python3

import argparse
import os
import re

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as error:
    raise SystemExit(
        "pyrealsense2 is unavailable. Run this script with /usr/bin/python3."
    ) from error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "camera_test_output")

CAMERAS = {
    "head": {
        "serial": "344522070363",
        "name": "Intel RealSense D435I",
    },
    "leftwrist": {
        "serial": "230422271038",
        "name": "Intel RealSense D405",
    },
    "rightwrist": {
        "serial": "230322273171",
        "name": "Intel RealSense D405",
    },
}


def detected_serials():
    context = rs.context()
    return {
        device.get_info(rs.camera_info.serial_number)
        for device in context.query_devices()
    }


def start_camera(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color, 640, 480, rs.format.bgr8, 30
    )
    config.enable_stream(
        rs.stream.depth, 640, 480, rs.format.z16, 30
    )
    pipeline.start(config)
    return pipeline


def save_image(path, image):
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to save {path}")


def next_capture_dir(role):
    pattern = re.compile(rf"^{re.escape(role)}(\d+)$")
    capture_numbers = []

    for name in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, name)
        match = pattern.match(name)

        if match and os.path.isdir(path):
            capture_numbers.append(int(match.group(1)))

    next_number = max(capture_numbers, default=0) + 1
    capture_dir = os.path.join(
        OUTPUT_DIR, f"{role}{next_number}"
    )
    os.makedirs(capture_dir)
    return capture_dir


def make_depth_grayscale(depth_image):
    grayscale = np.zeros(depth_image.shape, dtype=np.uint8)
    valid_depth = depth_image[depth_image > 0]

    if valid_depth.size == 0:
        return grayscale

    near, far = np.percentile(valid_depth, (2, 98))

    if far <= near:
        return grayscale

    scaled = (
        (depth_image.astype(np.float32) - near)
        * (255.0 / (far - near))
    )
    grayscale = np.clip(scaled, 0, 255).astype(np.uint8)
    grayscale[depth_image == 0] = 0
    return grayscale


def capture_camera(role):
    camera = CAMERAS[role]
    serial = camera["serial"]
    pipeline = None

    print(f"Capturing {role}: {camera['name']} ({serial})")

    try:
        pipeline = start_camera(serial)
        frames = None

        # Discard initial frames while auto-exposure settles.
        for _ in range(30):
            frames = pipeline.wait_for_frames(5000)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame:
            raise RuntimeError("No color frame received")
        if not depth_frame:
            raise RuntimeError("No depth frame received")

        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_visible = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_raw, alpha=0.03),
            cv2.COLORMAP_JET,
        )
        depth_grayscale = make_depth_grayscale(depth_raw)

        capture_dir = next_capture_dir(role)

        paths = {
            "color": os.path.join(capture_dir, "color.jpg"),
            "depth_raw": os.path.join(
                capture_dir, "depth_raw.png"
            ),
            "depth": os.path.join(capture_dir, "depth.jpg"),
            "depth_gray": os.path.join(
                capture_dir, "depth_gray.jpg"
            ),
        }

        save_image(paths["color"], color)
        save_image(paths["depth_raw"], depth_raw)
        save_image(paths["depth"], depth_visible)
        save_image(paths["depth_gray"], depth_grayscale)

        for image_type, path in paths.items():
            print(f"Saved {image_type}: {path}")

    finally:
        if pipeline is not None:
            pipeline.stop()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Capture still color and depth images from selected cameras"
    )
    parser.add_argument(
        "--head", action="store_true", help="Capture the head camera"
    )
    parser.add_argument(
        "--leftwrist",
        action="store_true",
        help="Capture the left wrist camera",
    )
    parser.add_argument(
        "--rightwrist",
        action="store_true",
        help="Capture the right wrist camera",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    selected = [
        role
        for role in CAMERAS
        if getattr(args, role)
    ]

    if not selected:
        raise SystemExit(
            "Select at least one camera: "
            "--head, --leftwrist, or --rightwrist"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    available = detected_serials()
    failed = False

    for role in selected:
        serial = CAMERAS[role]["serial"]

        if serial not in available:
            print(f"ERROR: {role} camera {serial} was not detected")
            failed = True
            continue

        try:
            capture_camera(role)
        except Exception as error:
            print(f"ERROR capturing {role}: {error}")
            failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
