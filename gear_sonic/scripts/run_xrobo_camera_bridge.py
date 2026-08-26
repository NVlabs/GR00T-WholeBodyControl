#!/usr/bin/env python3
"""Run the display-only TeleImager -> XRoboToolkit camera bridge."""

from __future__ import annotations

import argparse

from gear_sonic.camera.xrobo_video_bridge import XRoboTeleImagerBridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show TeleImager cameras in XRoboToolkit Remote Vision. "
            "This process does not interact with robot control."
        )
    )
    parser.add_argument("--camera-host", default="localhost")
    parser.add_argument("--camera-port", type=int, default=60000)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=13579)
    parser.add_argument(
        "--show-wrist-cameras",
        action="store_true",
        help=(
            "show the left and right wrist cameras below the head view; "
            "select TELEIMAGER_HEAD_WRISTS in XRoboToolkit"
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=("auto", "h264_nvenc", "libx264"),
        default="auto",
        help="H.264 encoder; auto tries NVENC then falls back to libx264",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bridge = XRoboTeleImagerBridge(
        camera_host=args.camera_host,
        camera_port=args.camera_port,
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        encoder=args.encoder,
        show_wrist_cameras=args.show_wrist_cameras,
    )
    try:
        bridge.serve_forever()
    except KeyboardInterrupt:
        print("\n[Bridge] Stopping")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
