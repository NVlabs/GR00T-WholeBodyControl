#!/usr/bin/env python3
"""Publish FoundationPose results as a ROS 2 TF transform."""

import argparse

from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from tf2_ros import TransformBroadcaster
import zmq


def quaternion_from_rotation_matrix(rotation):
    """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrix)

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = np.sqrt(
            1.0
            + matrix[index, index]
            - matrix[next_index, next_index]
            - matrix[last_index, last_index]
        ) * 2.0
        quaternion = np.zeros(4)
        quaternion[index] = 0.25 * scale
        quaternion[next_index] = (
            matrix[next_index, index] + matrix[index, next_index]
        ) / scale
        quaternion[last_index] = (
            matrix[last_index, index] + matrix[index, last_index]
        ) / scale
        quaternion[3] = (
            matrix[last_index, next_index] - matrix[next_index, last_index]
        ) / scale

    return quaternion / np.linalg.norm(quaternion)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose-endpoint",
        default="tcp://127.0.0.1:5561",
    )
    parser.add_argument("--parent-frame", default="d435_link")
    parser.add_argument("--object-frame", default="detected_object")
    return parser.parse_args()


def main():
    args = parse_args()
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.bind(args.pose_endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    rclpy.init()
    node = rclpy.create_node("foundationpose_object_tf_publisher")
    broadcaster = TransformBroadcaster(node)
    node.get_logger().info(
        f"Publishing {args.parent_frame} -> {args.object_frame} "
        f"from poses received at {args.pose_endpoint}"
    )

    try:
        while rclpy.ok():
            events = dict(poller.poll(100))
            if socket not in events:
                rclpy.spin_once(node, timeout_sec=0.0)
                continue

            pose = np.asarray(socket.recv_json(), dtype=np.float64)
            if pose.shape != (4, 4):
                node.get_logger().warning(
                    f"Ignoring pose with invalid shape {pose.shape}"
                )
                continue

            # TODO: Before using this TF for grasping, quantitatively verify the
            # FoundationPose optical-frame axes against d435_link at multiple
            # known 3D positions and apply an explicit frame conversion if needed.
            transform = TransformStamped()
            transform.header.stamp = node.get_clock().now().to_msg()
            transform.header.frame_id = args.parent_frame
            transform.child_frame_id = args.object_frame
            transform.transform.translation.x = float(pose[0, 3])
            transform.transform.translation.y = float(pose[1, 3])
            transform.transform.translation.z = float(pose[2, 3])

            quaternion = quaternion_from_rotation_matrix(pose[:3, :3])
            transform.transform.rotation.x = float(quaternion[0])
            transform.transform.rotation.y = float(quaternion[1])
            transform.transform.rotation.z = float(quaternion[2])
            transform.transform.rotation.w = float(quaternion[3])
            broadcaster.sendTransform(transform)
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        poller.unregister(socket)
        socket.close()
        context.term()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
