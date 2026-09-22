#!/usr/bin/env python3
"""Publish a host-local external-view camera independently over ZMQ."""

from __future__ import annotations

import argparse
import time

from gear_sonic.scripts.brainco_data_exporter.external_camera import (
    ExternalViewCameraPublisher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture external-view-camera and publish it for exporter/viewer."
    )
    parser.add_argument("--device", default="/dev/video4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--port", type=int, default=5582)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the first frame; 0 waits forever.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.startup_timeout < 0:
        raise ValueError("--startup-timeout must be non-negative")
    camera = ExternalViewCameraPublisher(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc or None,
        publish_port=args.port,
    )
    try:
        camera.wait_until_ready(args.startup_timeout)
        print(
            "[ExternalViewCamera] Ready: "
            f"device={args.device}, {args.width}x{args.height}@{args.fps:g}, "
            f"tcp://*:{args.port}"
        )
        while True:
            camera.check_health()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[ExternalViewCamera] Stopping")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
