import os
import numpy as np
import mujoco
from pathlib import Path
import rclpy

from decoupled_wbc.control.main.planner.utils.ros_utils import (
    ROSDictServiceClient,
)
from decoupled_wbc.control.main.planner.simulation.robot import G1Up

PLANNER_PLAN_SERVICE = "PlannerServer/plan"
PLANNER_DIR = Path(__file__).resolve().parent


def random_goal(waist=True):
    xml_path = os.path.join(PLANNER_DIR, "simulation", "g1_up.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    robot = G1Up(model=model)
    in_contact = True
    while in_contact:
        goal = robot.sample_qpos()
        if not waist:
            goal[:3] = 0
        robot.set_joint_qpos(goal)
        in_contact = robot.in_contact()
    return goal.copy()


def upper_body_goal():
    """
    17-DoF goal for the planner server in `JOINT_NAMES_UP` order:
        0 waist_yaw_joint
        1 waist_roll_joint
        2 waist_pitch_joint
        3 left_shoulder_pitch_joint
        4 left_shoulder_roll_joint
        5 left_shoulder_yaw_joint
        6 left_elbow_joint
        7 left_wrist_roll_joint
        8 left_wrist_pitch_joint
        9 left_wrist_yaw_joint
        10 right_shoulder_pitch_joint
        11 right_shoulder_roll_joint
        12 right_shoulder_yaw_joint
        13 right_elbow_joint
        14 right_wrist_roll_joint
        15 right_wrist_pitch_joint
        16 right_wrist_yaw_joint
    """

    return np.array(
        [
            -0.1,
            0.0,
            0.2,
            -1.5,
            1.0,
            0.7,
            -1.0,
            0.0,
            0.0,
            0.0,
            -0.3,
            -0.2,
            0.3,
            0.3,
            0.0,
            0.0,
            0.5,
        ],
        dtype=np.float32,
    )


def main():
    execute_immediately = True
    goal_type = "upper_body"  # "upper_body", "bimanual", "left", "right"

    start = None
    # start = np.zeros(17, dtype=np.float32)
    # goal = upper_body_goal()
    # goal = random_goal(waist=True)
    goal = random_goal(waist=False)

    # One-shot client: own node only. Do not also create ROSManager here —
    # that would start a second node + background spin and abort on shutdown.
    if not rclpy.ok():
        rclpy.init()
    client = ROSDictServiceClient(
        PLANNER_PLAN_SERVICE, node_name="ExamplePlannerRequest"
    )
    try:
        req = {
            "goal_qpos": goal,
            "start_qpos": start,
            "goal_type": goal_type,
            "execute_immediately": execute_immediately,
        }
        res = client.call(req)
        client.get_logger().info(f"Planner response: {res}")
    except KeyboardInterrupt:
        client.get_logger().info("Interrupted by user")
    finally:
        client.get_logger().info("Cleaning up...")
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
