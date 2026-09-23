import os
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import mujoco
import rclpy
import tyro

from decoupled_wbc.control.main.planner.utils.ros_utils import (
    ROSDictServiceClient,
)
from decoupled_wbc.control.main.planner.simulation.robot import (
    G1Up,
    JOINT_NAMES_UP,
)

PLANNER_PLAN_SERVICE = "PlannerServer/plan"
PLANNER_DIR = Path(__file__).resolve().parent


@dataclass
class RequestConfig:
    trajectory_path: str = "dataset/retar_pour_handover_22.npz"
    execute_immediately: bool = True
    wait_for_execution: bool = False
    planner_frequency: float = 20.0
    initial_transition_time: float = 2.0
    execution_margin: float = 0.5


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


def load_test_trajectory(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = PLANNER_DIR / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as trajectory:
        required = {"qpos", "joint_names"}
        missing_keys = sorted(required.difference(trajectory.files))
        if missing_keys:
            raise KeyError(f"Trajectory is missing keys: {missing_keys}")
        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        joint_names = [
            str(name) for name in trajectory["joint_names"].tolist()
        ]

    if qpos.ndim != 2 or qpos.shape[1] != len(joint_names):
        raise ValueError("Trajectory qpos width does not match joint_names")
    if len(joint_names) != len(set(joint_names)):
        raise ValueError("Trajectory joint_names contains duplicates")
    name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    missing_joints = [
        name for name in JOINT_NAMES_UP if name not in name_to_index
    ]
    if missing_joints:
        raise ValueError(f"Trajectory is missing planning joints: {missing_joints}")
    qpos = qpos[:, [name_to_index[name] for name in JOINT_NAMES_UP]]
    if qpos.shape[0] < 2 or not np.isfinite(qpos).all():
        raise ValueError("Trajectory must contain at least two finite frames")
    return qpos


def report_path_diagnostics(client, path, start, goal, reference):
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != start.shape[0]:
        raise ValueError(f"Unexpected planned path shape: {path.shape}")

    start_error = float(np.max(np.abs(path[0] - start)))
    goal_error = float(np.max(np.abs(path[-1] - goal)))
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    path_length = float(np.sum(segment_lengths))
    nearest_reference_distance = np.asarray(
        [
            np.min(np.linalg.norm(reference - waypoint, axis=1))
            for waypoint in path
        ],
        dtype=np.float64,
    )
    reference_cost = float(
        np.sum(
            0.5
            * (
                nearest_reference_distance[:-1]
                + nearest_reference_distance[1:]
            )
            * segment_lengths
        )
    )

    client.get_logger().info(
        "Path diagnostics: "
        f"start_error={start_error:.6f}, goal_error={goal_error:.6f}, "
        f"path_length={path_length:.6f}, "
        f"approx_reference_cost={reference_cost:.6f}, "
        f"max_reference_distance={np.max(nearest_reference_distance):.6f}"
    )
    if start_error > 1e-4 or goal_error > 1e-4:
        client.get_logger().error(
            "INVALID PLANNER ENDPOINTS: returned path does not match the "
            "requested trajectory start and goal"
        )
        raise RuntimeError("Planner returned invalid path endpoints")


def main(config):
    if (
        not np.isfinite(config.planner_frequency)
        or config.planner_frequency <= 0
    ):
        raise ValueError("planner_frequency must be > 0")
    if (
        not np.isfinite(config.initial_transition_time)
        or config.initial_transition_time < 0
    ):
        raise ValueError("initial_transition_time must be >= 0")
    if (
        not np.isfinite(config.execution_margin)
        or config.execution_margin < 0
    ):
        raise ValueError("execution_margin must be >= 0")

    goal_type = "upper_body"  # "upper_body", "bimanual", "left", "right"

    # Live-state planning start disabled for this controlled endpoint test.
    # start = None
    # start = np.zeros(17, dtype=np.float32)
    # goal = upper_body_goal()
    # Random goal disabled; use the dataset endpoints instead.
    # goal = random_goal(waist=False)
    reference = load_test_trajectory(config.trajectory_path)
    start = reference[0].astype(np.float32)
    goal = reference[-1].astype(np.float32)
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
            "execute_immediately": config.execute_immediately,
        }
        res = client.call(req)
        client.get_logger().info(f"Planner response: {res}")
        report_path_diagnostics(
            client, res["qpos"], start, goal, reference
        )
        if config.wait_for_execution and res.get("executed", False):
            execution_time = (
                config.initial_transition_time
                + max(0, int(res["num_waypoints"]) - 1)
                / config.planner_frequency
                + config.execution_margin
            )
            client.get_logger().info(
                "Waiting approximately "
                f"{execution_time:.2f}s for trajectory execution"
            )
            time.sleep(execution_time)
    except KeyboardInterrupt:
        client.get_logger().info("Interrupted by user")
    finally:
        client.get_logger().info("Cleaning up...")
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.cli(RequestConfig))
