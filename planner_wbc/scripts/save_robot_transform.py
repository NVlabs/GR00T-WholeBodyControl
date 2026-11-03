#!/usr/bin/env python3
"""Save one transform from the live ROS 2 TF tree to JSON."""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener


def parse_args():
    parser = argparse.ArgumentParser(description="Save one live TF2 transform to JSON.")
    parser.add_argument("--parent", default="pelvis")
    parser.add_argument("--child", default="d435_link")
    parser.add_argument("--output", type=Path, default=Path("pelvis_to_d435.json"))
    parser.add_argument("--timeout", type=float, default=15.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("save_robot_transform")
    tf_buffer = Buffer()
    listener = TransformListener(tf_buffer, node)
    deadline = time.monotonic() + args.timeout

    try:
        transform = None
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if tf_buffer.can_transform(
                args.parent,
                args.child,
                Time(),
                timeout=Duration(seconds=0.1),
            ):
                transform = tf_buffer.lookup_transform(args.parent, args.child, Time())
                break

        if transform is None:
            raise TimeoutError(
                f"Transform {args.parent} -> {args.child} was not available "
                f"within {args.timeout:g} seconds"
            )

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        quaternion_xyzw = [rotation.x, rotation.y, rotation.z, rotation.w]

        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
        matrix[:3, 3] = [translation.x, translation.y, translation.z]

        result = {
            "parent_frame": args.parent,
            "child_frame": args.child,
            "timestamp": {
                "sec": transform.header.stamp.sec,
                "nanosec": transform.header.stamp.nanosec,
            },
            "translation_xyz": [translation.x, translation.y, translation.z],
            "quaternion_xyzw": quaternion_xyzw,
            "matrix": matrix.tolist(),
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {args.parent} -> {args.child} to {args.output.resolve()}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
