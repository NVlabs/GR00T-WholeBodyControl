from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import tyro

from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.main.planner.simulation.robot import JOINT_NAMES_UP
from decoupled_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from decoupled_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
    ROSMsgSubscriber,
)

PLANNER_DIR = Path(__file__).resolve().parent


@dataclass
class PrepositionConfig:
    """Slowly move the live G1 upper body to a trajectory's first frame."""

    trajectory_path: str = "dataset/retar_pour_handover_22.npz"
    duration: float = 8.0
    frequency: float = 20.0
    settle_time: float = 1.0
    state_timeout: float = 10.0
    verification_timeout: float = 5.0
    position_tolerance: float = 0.05
    enable_waist: bool = True
    high_elbow_pose: bool = False


def resolve_trajectory_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PLANNER_DIR / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_planning_start(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        required = {"qpos", "joint_names"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"Trajectory is missing keys: {missing}")
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        joint_names = [str(name) for name in data["joint_names"].tolist()]

    if qpos.ndim != 2 or qpos.shape[0] < 2:
        raise ValueError(
            f"Expected trajectory qpos shape (N >= 2, D), got {qpos.shape}"
        )
    if qpos.shape[1] != len(joint_names):
        raise ValueError("Trajectory qpos width does not match joint_names")
    if len(joint_names) != len(set(joint_names)):
        raise ValueError("Trajectory joint_names contains duplicates")
    if not np.isfinite(qpos).all():
        raise ValueError("Trajectory contains non-finite values")

    name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    missing_joints = [
        name for name in JOINT_NAMES_UP if name not in name_to_index
    ]
    if missing_joints:
        raise ValueError(f"Trajectory is missing planning joints: {missing_joints}")
    return np.asarray(
        [qpos[0, name_to_index[name]] for name in JOINT_NAMES_UP],
        dtype=np.float32,
    )


def wait_for_state(
    subscriber: ROSMsgSubscriber, timeout: float
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = subscriber.get_msg()
        if state is not None and "q" in state:
            return state
        time.sleep(0.02)
    raise TimeoutError(
        f"No robot state received from '{STATE_TOPIC_NAME}' within {timeout}s"
    )


def smoothstep(alpha: float) -> float:
    return alpha * alpha * (3.0 - 2.0 * alpha)


def planning_q_from_state(
    state: dict,
    controller_indices: list[int],
    controller_name_to_index: dict[str, int],
) -> np.ndarray:
    full_q = np.asarray(state["q"], dtype=np.float32).reshape(-1)
    if full_q.shape[0] <= max(controller_indices):
        raise ValueError(f"Robot state q length {full_q.shape[0]} is too short")
    controller_q = full_q[controller_indices]
    if not np.isfinite(controller_q).all():
        raise ValueError("Robot state contains non-finite upper-body values")
    return np.asarray(
        [
            controller_q[controller_name_to_index[name]]
            for name in JOINT_NAMES_UP
        ],
        dtype=np.float32,
    )


def main(config: PrepositionConfig) -> None:
    if not np.isfinite(config.duration) or config.duration <= 0:
        raise ValueError("duration must be > 0")
    if not np.isfinite(config.frequency) or config.frequency <= 0:
        raise ValueError("frequency must be > 0")
    if not np.isfinite(config.settle_time) or config.settle_time < 0:
        raise ValueError("settle_time must be >= 0")
    if not np.isfinite(config.state_timeout) or config.state_timeout <= 0:
        raise ValueError("state_timeout must be > 0")
    if (
        not np.isfinite(config.verification_timeout)
        or config.verification_timeout <= 0
    ):
        raise ValueError("verification_timeout must be > 0")
    if (
        not np.isfinite(config.position_tolerance)
        or config.position_tolerance <= 0
    ):
        raise ValueError("position_tolerance must be > 0")

    trajectory_path = resolve_trajectory_path(config.trajectory_path)
    target_planning_q = load_planning_start(trajectory_path)

    waist_location = (
        "lower_and_upper_body" if config.enable_waist else "lower_body"
    )
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location,
        high_elbow_pose=config.high_elbow_pose,
    )
    controller_indices = robot_model.get_joint_group_indices("upper_body")
    controller_names = [
        robot_model.joint_names[idx] for idx in controller_indices
    ]
    controller_name_to_index = {
        name: idx for idx, name in enumerate(controller_names)
    }
    missing_controller_joints = [
        name for name in JOINT_NAMES_UP if name not in controller_name_to_index
    ]
    if missing_controller_joints:
        raise ValueError(
            "Controller upper-body group is missing planning joints: "
            f"{missing_controller_joints}"
        )

    ros_manager = ROSManager(node_name="TrajectoryStartPrepositioner")
    subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)

    try:
        state = wait_for_state(subscriber, config.state_timeout)
        full_q = np.asarray(state["q"], dtype=np.float32).reshape(-1)
        if full_q.shape[0] <= max(controller_indices):
            raise ValueError(
                f"Robot state q length {full_q.shape[0]} is too short"
            )

        current_controller_q = full_q[controller_indices].copy()
        current_planning_q = planning_q_from_state(
            state,
            controller_indices,
            controller_name_to_index,
        )

        period = 1.0 / config.frequency
        steps = max(1, int(np.ceil(config.duration * config.frequency)))
        print(
            f"Slowly moving to trajectory start over {config.duration:.1f}s "
            f"({steps} commands at {config.frequency:.1f} Hz)."
        )
        print(
            "Initial max joint difference: "
            f"{np.max(np.abs(target_planning_q - current_planning_q)):.4f} rad"
        )

        next_tick = time.monotonic()
        for step in range(1, steps + 1):
            alpha = smoothstep(step / steps)
            planning_q = current_planning_q + alpha * (
                target_planning_q - current_planning_q
            )
            controller_target = current_controller_q.copy()
            for plan_idx, joint_name in enumerate(JOINT_NAMES_UP):
                controller_target[
                    controller_name_to_index[joint_name]
                ] = planning_q[plan_idx]

            now = time.monotonic()
            publisher.publish(
                {
                    "target_upper_body_pose": controller_target,
                    "timestamp": now,
                    "target_time": now + period,
                }
            )
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))

        if config.settle_time > 0:
            now = time.monotonic()
            final_controller_target = current_controller_q.copy()
            for plan_idx, joint_name in enumerate(JOINT_NAMES_UP):
                final_controller_target[
                    controller_name_to_index[joint_name]
                ] = target_planning_q[plan_idx]
            publisher.publish(
                {
                    "target_upper_body_pose": final_controller_target,
                    "timestamp": now,
                    "target_time": now + config.settle_time,
                }
            )
            time.sleep(config.settle_time)

        verification_deadline = (
            time.monotonic() + config.verification_timeout
        )
        final_error = float("inf")
        while time.monotonic() < verification_deadline:
            verification_state = subscriber.get_msg()
            if verification_state is None or "q" not in verification_state:
                time.sleep(0.02)
                continue
            achieved_q = planning_q_from_state(
                verification_state,
                controller_indices,
                controller_name_to_index,
            )
            final_error = float(
                np.max(np.abs(achieved_q - target_planning_q))
            )
            if final_error <= config.position_tolerance:
                break

        if final_error > config.position_tolerance:
            raise RuntimeError(
                "Robot did not reach trajectory frame 0: "
                f"max_error={final_error:.4f} rad, "
                f"tolerance={config.position_tolerance:.4f} rad"
            )
        print(
            "Pre-positioning verified: trajectory frame 0 reached "
            f"(max error {final_error:.4f} rad)."
        )
    finally:
        ros_manager.shutdown()


if __name__ == "__main__":
    main(tyro.cli(PrepositionConfig))
