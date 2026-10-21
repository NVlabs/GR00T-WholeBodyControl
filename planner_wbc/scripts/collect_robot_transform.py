#!/usr/bin/env python3
"""Publish the Planner-WBC G1 robot state as a standard ROS 2 TF tree."""

import argparse
from pathlib import Path
import subprocess
import time

import numpy as np
from sensor_msgs.msg import JointState

from planner_wbc.control.main.constants import STATE_TOPIC_NAME
from planner_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from planner_wbc.control.utils.ros_utils import ROSManager, ROSMsgSubscriber


DEFAULT_URDF = (
    Path(__file__).resolve().parents[1]
    / "control"
    / "robot_model"
    / "model_data"
    / "g1"
    / "g1_29dof_with_hand.urdf"
)
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Bridge measured Planner-WBC G1 state to /joint_states and publish "
            "the complete URDF tree on /tf and /tf_static."
        )
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="G1 URDF used by robot_state_publisher.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the first G1 control-loop state message.",
    )
    return parser.parse_args()


def start_robot_state_publisher(urdf_path: Path) -> subprocess.Popen:
    if not urdf_path.is_file():
        raise FileNotFoundError(f"G1 URDF not found: {urdf_path}")

    robot_description = urdf_path.read_text()
    return subprocess.Popen(
        [
            "ros2",
            "run",
            "robot_state_publisher",
            "robot_state_publisher",
            "--ros-args",
            "-p",
            f"robot_description:={robot_description}",
        ]
    )


def main():
    args = parse_args()
    ros_manager = ROSManager(node_name="planner_wbc_joint_state_bridge")
    state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    joint_state_publisher = ros_manager.node.create_publisher(JointState, "/joint_states", 10)
    robot_model = instantiate_g1_robot_model()
    joint_names = robot_model.joint_names
    joint_indices = [robot_model.dof_index(name) for name in joint_names]
    robot_state_publisher = start_robot_state_publisher(args.urdf)
    deadline = time.monotonic() + args.timeout
    received_state = False
    try:
        while ros_manager.ok():
            if robot_state_publisher.poll() is not None:
                raise RuntimeError(
                    "robot_state_publisher exited unexpectedly with code "
                    f"{robot_state_publisher.returncode}"
                )

            state = state_subscriber.get_msg()
            if state is None:
                if not received_state and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"No state received from {STATE_TOPIC_NAME!r} "
                        f"within {args.timeout:g} seconds"
                    )
                time.sleep(0.01)
                continue

            received_state = True
            q = np.asarray(state["q"], dtype=np.float64)
            if q.shape != (robot_model.num_dofs,):
                raise ValueError(
                    f"Expected measured q shape ({robot_model.num_dofs},), got {q.shape}"
                )

            message = JointState()
            message.header.stamp = ros_manager.node.get_clock().now().to_msg()
            message.name = joint_names
            message.position = [float(q[index]) for index in joint_indices]

            dq = state.get("dq")
            if dq is not None:
                dq = np.asarray(dq, dtype=np.float64)
                if dq.shape == q.shape:
                    message.velocity = [float(dq[index]) for index in joint_indices]

            joint_state_publisher.publish(message)
    finally:
        if robot_state_publisher.poll() is None:
            robot_state_publisher.terminate()
            try:
                robot_state_publisher.wait(timeout=3)
            except subprocess.TimeoutExpired:
                robot_state_publisher.kill()
                robot_state_publisher.wait()
        ros_manager.shutdown()


if __name__ == "__main__":
    main()
