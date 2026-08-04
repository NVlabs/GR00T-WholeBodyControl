from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import rclpy
import tyro
import mujoco

from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.main.planner.configs.configs import PlannerConfig
from decoupled_wbc.control.main.planner.utils.ros_utils import (
    ROSDictServiceServer,
)
from decoupled_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from decoupled_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
    ROSMsgSubscriber,
)
from decoupled_wbc.control.main.planner.utils.ompl_planning import (
    OMPLGeometricPlanner,
)
from decoupled_wbc.control.main.planner.simulation.robot import (
    G1Up,
    JOINT_NAMES_UP,
    JOINT_NAMES_BIMANUAL,
    JOINT_NAMES_LEFT,
    JOINT_NAMES_RIGHT,
)

PLANNER_DIR = Path(__file__).resolve().parent
PLANNER_NODE_NAME = "PlannerServer"
PLANNER_PLAN_SERVICE = "PlannerServer/plan"


@dataclass
class ControllerUpperBodyInterface:
    indices: list[int]
    joint_names: list[str]
    default_qpos: np.ndarray
    name_to_index: dict[str, int]


@dataclass
class ActiveTrajectory:
    qpos: np.ndarray
    frame_idx: int
    is_first_publish: bool
    hold_final_pose_printed: bool


def densify_waypoints(
    waypoints: np.ndarray, max_joint_step: float
) -> np.ndarray:
    """Insert intermediate waypoints so consecutive joints jump by at most max_joint_step."""
    if waypoints.ndim != 2 or waypoints.shape[0] == 0:
        raise ValueError(
            f"Expected waypoints shape (N, DoF), got {waypoints.shape}"
        )
    if waypoints.shape[0] == 1:
        return waypoints.astype(np.float32, copy=True)
    if max_joint_step <= 0:
        raise ValueError("max_joint_step must be > 0")

    densified = [waypoints[0]]
    for nxt in waypoints[1:]:
        prev = densified[-1]
        delta = nxt - prev
        n_steps = int(np.ceil(np.max(np.abs(delta)) / max_joint_step))
        n_steps = max(1, n_steps)
        for step in range(1, n_steps + 1):
            densified.append(prev + delta * (step / n_steps))
    return np.asarray(densified, dtype=np.float32)


def upper_body_pose_to_planning_qpos(
    upper_body_pose: np.ndarray,
    controller_interface: ControllerUpperBodyInterface,
) -> np.ndarray:
    """Map controller-ordered upper-body joints into the 17-DoF planning configuration."""
    upper_body_pose = np.asarray(upper_body_pose, dtype=np.float32).reshape(-1)
    if upper_body_pose.shape[0] != len(controller_interface.joint_names):
        raise ValueError(
            f"Upper-body pose length {upper_body_pose.shape[0]} does not match "
            f"controller upper_body size {len(controller_interface.joint_names)}"
        )

    planning_q = np.zeros(len(JOINT_NAMES_UP), dtype=np.float32)
    for plan_idx, joint_name in enumerate(JOINT_NAMES_UP):
        controller_idx = controller_interface.name_to_index.get(joint_name)
        if controller_idx is None:
            raise ValueError(
                f"Planning joint '{joint_name}' missing from controller upper_body"
            )
        planning_q[plan_idx] = upper_body_pose[controller_idx]
    return planning_q


def full_q_to_planning_qpos(
    full_q: np.ndarray,
    controller_interface: ControllerUpperBodyInterface,
) -> np.ndarray:
    """Map a full-body joint vector into the 17-DoF planning configuration."""
    full_q = np.asarray(full_q, dtype=np.float32).reshape(-1)
    if full_q.shape[0] < max(controller_interface.indices) + 1:
        raise ValueError(
            f"Robot state q length {full_q.shape[0]} is too short for upper-body indices"
        )
    upper_body_pose = full_q[controller_interface.indices]
    return upper_body_pose_to_planning_qpos(
        upper_body_pose, controller_interface
    )


def planning_qpos_to_controller_pose(
    planning_q: np.ndarray,
    controller_interface: ControllerUpperBodyInterface,
) -> np.ndarray:
    """Map a planning configuration into the controller upper-body target vector."""
    planning_q = np.asarray(planning_q, dtype=np.float32).reshape(-1)
    if planning_q.shape[0] != len(JOINT_NAMES_UP):
        raise ValueError(
            f"Expected planning qpos length {len(JOINT_NAMES_UP)}, got {planning_q.shape[0]}"
        )

    target = controller_interface.default_qpos.copy()
    for plan_idx, joint_name in enumerate(JOINT_NAMES_UP):
        controller_idx = controller_interface.name_to_index.get(joint_name)
        if controller_idx is None:
            continue
        target[controller_idx] = planning_q[plan_idx]
    return target.astype(np.float32, copy=False)


def parse_plan_request(
    request: dict, default_goal_type: str, default_execute_immediately: bool
) -> tuple[np.ndarray, Optional[np.ndarray], str, bool]:
    """
    Parse a plan-service request into (goal, start, goal_type, execute_immediately).

    Accepted keys:
      - goal_qpos: 17-DoF planning joints in JOINT_NAMES_UP order
      - start_qpos: optional start overrides
      - goal_type: optional "upper_body" | "bimanual" | "left" | "right"
      - execute_immediately: optional bool; if true, stream path to the control loop
    """
    goal_type = str(request.get("goal_type", default_goal_type))
    execute_immediately = bool(
        request.get("execute_immediately", default_execute_immediately)
    )

    if "goal_qpos" in request:
        goal = np.asarray(request["goal_qpos"], dtype=np.float32).reshape(-1)
        if goal.shape[0] != len(JOINT_NAMES_UP):
            raise ValueError(
                f"goal_qpos length {goal.shape[0]} != {len(JOINT_NAMES_UP)}"
            )
    else:
        raise KeyError("Plan request must include goal_qpos")

    start = None
    if "start_qpos" in request:
        if request["start_qpos"] is not None:
            start = np.asarray(
                request["start_qpos"], dtype=np.float32
            ).reshape(-1)
            if start.shape[0] != len(JOINT_NAMES_UP):
                raise ValueError(
                    f"start_qpos length {start.shape[0]} != {len(JOINT_NAMES_UP)}"
                )
    return goal, start, goal_type, execute_immediately


def interruptible_sleep(duration: float, keep_running) -> None:
    end_time = time.monotonic() + duration
    while keep_running():
        remaining = end_time - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


class PlannerServer:
    def __init__(
        self,
        config: PlannerConfig,
        logger,
        state_subscriber: ROSMsgSubscriber,
    ):
        if config.planner_frequency <= 0:
            raise ValueError("planner_frequency must be > 0")
        if config.planning_timeout <= 0:
            raise ValueError("planning_timeout must be > 0")

        self.config = config
        self.logger = logger
        self.state_subscriber = state_subscriber
        self.controller_interface = self.build_controller_interface(config)
        self.ref_traj = None
        # self.ref_traj = np.load(
        #     os.path.join(PLANNER_DIR, "dataset", "reference_col.npy")
        # )

        xml_path = self.resolve_planning_xml(config)
        model = self.load_planning_model(xml_path)
        self.robot = G1Up(model=model, visualize=config.visualize_planning)
        self.planner = OMPLGeometricPlanner(
            self.robot,
            planner=config.ompl_planner,
            validity_resolution=config.validity_resolution,
            log=True,
        )

        self._lock = threading.Lock()
        self._active: Optional[ActiveTrajectory] = None
        self._cancel_execution = False
        self._planning = False

        missing = [
            name
            for name in JOINT_NAMES_UP
            if name not in self.controller_interface.name_to_index
        ]
        if missing:
            raise ValueError(
                f"Controller upper_body missing planning joints: {missing}"
            )

        self.logger.info(f"Planning XML: {xml_path}")
        self.logger.info(f"OMPL planner: {config.ompl_planner}")
        self.logger.info(f"Planning joints: {JOINT_NAMES_UP}")
        self.logger.info(
            f"Controller upper-body joints: {self.controller_interface.joint_names}"
        )
        self.logger.info(f"Plan service: {config.plan_service}")
        self.logger.info(f"Trajectory topic: {config.trajectory_topic}")
        self.logger.info(
            f"Default execute_immediately: {config.execute_immediately}"
        )

    def close(self) -> None:
        self.robot.close()

    def read_start_from_state(self) -> np.ndarray:
        state = self.state_subscriber.get_msg()
        if state is None or "q" not in state:
            self.logger.warn(
                "No robot state available; using planning-model qpos as start"
            )
            return self.robot.get_joint_qpos().astype(np.float32)
        return full_q_to_planning_qpos(
            np.asarray(state["q"], dtype=np.float32).reshape(-1),
            self.controller_interface,
        )

    def plan_to_goal(
        self,
        goal: np.ndarray,
        start: Optional[np.ndarray],
        goal_type: str,
    ) -> tuple[np.ndarray, float]:
        if start is None:
            start = self.read_start_from_state()
            self.robot.set_joint_qpos(start)

        start = np.asarray(start, dtype=np.float64).reshape(-1)
        goal = np.asarray(goal, dtype=np.float64).reshape(-1)
        self.logger.info(
            f"Planning Requested \nstart:\n {start} \ngoal:\n {goal}"
        )

        if start.shape[0] != len(JOINT_NAMES_UP) or goal.shape[0] != len(
            JOINT_NAMES_UP
        ):
            raise ValueError(
                "Start/goal must match the 17-DoF planning joint set"
            )

        self.logger.info(
            f"Planning with {self.config.ompl_planner} "
            f"(timeout={self.config.planning_timeout}s, goal_type={goal_type})"
        )
        t0 = time.monotonic()
        solution = self.planner.plan(
            start,
            goal,
            goal_type,
            self.ref_traj,
            timeout=self.config.planning_timeout,
            smooth_path=self.config.smooth_path,
            shortcut_path=self.config.shortcut_path,
        )
        elapsed = time.monotonic() - t0

        if solution is None or len(solution) < 2:
            raise RuntimeError(f"OMPL failed to find a path in {elapsed:.3f}s")

        waypoints = densify_waypoints(
            np.asarray(solution), self.config.max_joint_step
        )
        self.logger.info(
            f"Plan ready in {elapsed:.3f}s: "
            f"{len(solution)} raw -> {len(waypoints)} densified waypoints"
        )
        return waypoints, elapsed

    def set_active_trajectory(self, waypoints: np.ndarray) -> None:
        with self._lock:
            self._cancel_execution = True
            self._active = ActiveTrajectory(
                qpos=waypoints.astype(np.float32, copy=False),
                frame_idx=0,
                is_first_publish=True,
                hold_final_pose_printed=False,
            )
            self._cancel_execution = False

    def handle_plan_request(self, request: dict) -> dict:
        """ROS2 service handler: plan once and optionally begin execution."""
        with self._lock:
            if self._planning:
                raise RuntimeError("Planner is busy with another request")
            self._planning = True

        try:
            goal, start, goal_type, execute_immediately = parse_plan_request(
                request, self.config.goal_type, self.config.execute_immediately
            )
            waypoints, planning_time = self.plan_to_goal(
                goal, start, goal_type
            )

            if execute_immediately:
                self.set_active_trajectory(waypoints)
                self.logger.info(
                    "execute_immediately=True; streaming trajectory to control loop"
                )
            else:
                self.logger.info(
                    "execute_immediately=False; returning path only"
                )

            return {
                "qpos": waypoints.astype(np.float32, copy=False),
                "joint_names": list(JOINT_NAMES_UP),
                "num_waypoints": int(waypoints.shape[0]),
                "planning_time": float(planning_time),
                "executed": bool(execute_immediately),
                "goal_type": goal_type,
            }
        finally:
            with self._lock:
                self._planning = False

    def publish_step(
        self, control_publisher: ROSMsgPublisher, keep_running
    ) -> str:
        """
        Publish one trajectory frame.

        Returns:
            "idle" - nothing to publish
            "initial" - published first waypoint (already waited transition time)
            "streaming" - published a normal waypoint
        """
        with self._lock:
            active = self._active
            if active is None:
                return "idle"
            frame_idx = active.frame_idx
            is_first = active.is_first_publish
            qpos = active.qpos

        target_upper_body_pose = planning_qpos_to_controller_pose(
            qpos[frame_idx], self.controller_interface
        )
        t_now = time.monotonic()
        publish_period = 1.0 / self.config.planner_frequency
        target_time = (
            t_now + self.config.initial_transition_time
            if is_first
            else t_now + publish_period
        )
        control_publisher.publish(
            {
                "target_upper_body_pose": target_upper_body_pose,
                "timestamp": t_now,
                "target_time": target_time,
            }
        )

        with self._lock:
            if self._active is None or self._cancel_execution:
                return "idle"
            active = self._active
            if is_first:
                active.is_first_publish = False
                if active.qpos.shape[0] > 1:
                    active.frame_idx = 1
            elif active.frame_idx < active.qpos.shape[0] - 1:
                active.frame_idx += 1
            else:
                if self.config.hold_final_pose:
                    if not active.hold_final_pose_printed:
                        self.logger.info(
                            "Reached final planned waypoint; holding final pose."
                        )
                        active.hold_final_pose_printed = True
                    active.frame_idx = active.qpos.shape[0] - 1
                else:
                    self._active = None

        if is_first:
            self.logger.info(
                f"Publishing initial planned waypoint for {self.config.initial_transition_time}s"
            )
            interruptible_sleep(
                self.config.initial_transition_time, keep_running
            )
            return "initial"
        return "streaming"

    # Helper functions
    def build_controller_interface(
        self,
        config: PlannerConfig,
    ) -> ControllerUpperBodyInterface:
        waist_location = (
            "lower_and_upper_body" if config.enable_waist else "lower_body"
        )
        robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=config.high_elbow_pose,
        )
        upper_body_indices = robot_model.get_joint_group_indices("upper_body")
        upper_body_joint_names = [
            robot_model.joint_names[idx] for idx in upper_body_indices
        ]
        default_qpos = robot_model.default_body_pose[
            upper_body_indices
        ].astype(np.float32, copy=True)
        name_to_index = {
            name: idx for idx, name in enumerate(upper_body_joint_names)
        }
        return ControllerUpperBodyInterface(
            indices=upper_body_indices,
            joint_names=upper_body_joint_names,
            default_qpos=default_qpos,
            name_to_index=name_to_index,
        )

    def resolve_planning_xml(self, config: PlannerConfig) -> Path:
        xml_path = Path(config.planning_xml)
        if not xml_path.is_absolute():
            xml_path = PLANNER_DIR / xml_path
        xml_path = xml_path.resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"Planning XML not found: {xml_path}")
        return xml_path

    def load_planning_model(self, xml_path: Path):
        """Load MuJoCo model with includes resolved from the planner package root."""
        prev_cwd = os.getcwd()
        os.chdir(PLANNER_DIR)
        try:
            return mujoco.MjModel.from_xml_path(str(xml_path))
        finally:
            os.chdir(prev_cwd)


def main(config: PlannerConfig):
    if not config.plan_service:
        config.plan_service = PLANNER_PLAN_SERVICE
    if not config.trajectory_topic:
        config.trajectory_topic = CONTROL_GOAL_TOPIC

    ros_manager = ROSManager(node_name=PLANNER_NODE_NAME)
    node = ros_manager.node
    logger = node.get_logger()
    server = None

    try:
        state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
        control_publisher = ROSMsgPublisher(config.trajectory_topic)
        server = PlannerServer(config, logger, state_subscriber)
        ROSDictServiceServer(config.plan_service, server.handle_plan_request)
        rate = node.create_rate(config.planner_frequency)

        logger.info(
            f"Planner service ready at '{config.plan_service}'. "
            f"Set execute_immediately=true to stream to '{config.trajectory_topic}'."
        )

        while rclpy.ok():
            status = server.publish_step(control_publisher, rclpy.ok)
            if status != "initial":
                rate.sleep()

    except ros_manager.exceptions() as e:
        logger.info(f"ROSManager interrupted by user: {e}")
    finally:
        logger.info("Cleaning up...")
        if server is not None:
            server.close()
        ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(PlannerConfig)
    main(config)
