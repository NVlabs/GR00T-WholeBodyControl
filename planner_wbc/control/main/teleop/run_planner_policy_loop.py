from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import rclpy
import tyro

import planner_wbc

from planner_wbc.control.main.constants import CONTROL_GOAL_TOPIC
from planner_wbc.control.main.teleop.configs.configs import PlannerConfig
from planner_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from planner_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher

PLANNER_NODE_NAME = "PlannerPolicy"
REQUIRED_UPPER_BODY_DATASET_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@dataclass
class ControllerUpperBodyInterface:
    indices: list[int]
    joint_names: list[str]
    default_qpos: np.ndarray


@dataclass
class TrajectoryRuntime:
    qpos: np.ndarray
    dataset_joint_names: list[str]
    controller_interface: ControllerUpperBodyInterface
    publish_period: float


def resolve_trajectory_path(trajectory_path: str) -> Path:
    package_dir = Path(planner_wbc.__file__).resolve().parent
    resolved_path = (package_dir / trajectory_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(resolved_path)
    return resolved_path


def _assert_unique(names: list[str], label: str) -> None:
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{label} contains duplicate names: {duplicates}")


def load_trajectory_dataset(trajectory_path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(trajectory_path, allow_pickle=False) as data:
        required_keys = {"qpos", "joint_names"}
        missing_keys = sorted(required_keys.difference(data.files))
        if missing_keys:
            raise KeyError(f"Trajectory dataset is missing required keys: {missing_keys}")

        qpos = data["qpos"]
        joint_names = data["joint_names"]

    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos.ndim == 2, got {qpos.ndim}")
    if qpos.shape[0] <= 0:
        raise ValueError("Trajectory dataset must contain at least one frame")
    if qpos.shape[1] != len(joint_names):
        raise ValueError(
            f"Trajectory qpos width {qpos.shape[1]} does not match joint_names length {len(joint_names)}"
        )
    if not np.isfinite(qpos).all():
        raise ValueError("Trajectory qpos contains non-finite values")

    qpos = qpos.astype(np.float32, copy=False)
    joint_names = [str(name) for name in joint_names.tolist()]
    _assert_unique(joint_names, "Dataset joint_names")
    return qpos, joint_names


def _build_controller_interface_from_robot_model(
    config: PlannerConfig,
) -> ControllerUpperBodyInterface:
    robot_model = instantiate_g1_robot_model(
        waist_location="lower_and_upper_body", high_elbow_pose=config.high_elbow_pose
    )
    upper_body_indices = robot_model.get_joint_group_indices("upper_body")
    upper_body_joint_names = [robot_model.joint_names[idx] for idx in upper_body_indices]
    default_qpos = robot_model.default_body_pose[upper_body_indices].astype(np.float32, copy=True)
    return ControllerUpperBodyInterface(
        indices=upper_body_indices,
        joint_names=upper_body_joint_names,
        default_qpos=default_qpos,
    )


def build_trajectory_runtime(config: PlannerConfig, logger) -> TrajectoryRuntime:
    if config.planner_frequency <= 0:
        raise ValueError("planner_frequency must be > 0")

    resolved_path = resolve_trajectory_path(config.trajectory_path)
    qpos, dataset_joint_names = load_trajectory_dataset(resolved_path)
    controller_interface = _build_controller_interface_from_robot_model(config)

    if len(controller_interface.indices) != len(controller_interface.joint_names):
        raise ValueError("Controller upper-body indices and names length mismatch")
    if controller_interface.default_qpos.shape[0] != len(controller_interface.indices):
        raise ValueError("Controller default upper-body pose length mismatch")

    dataset_name_to_index = {name: idx for idx, name in enumerate(dataset_joint_names)}
    controller_name_to_index = {
        name: idx for idx, name in enumerate(controller_interface.joint_names)
    }
    _assert_unique(list(controller_name_to_index.keys()), "Controller upper-body joint names")

    missing_required_dataset_joints = [
        joint_name
        for joint_name in REQUIRED_UPPER_BODY_DATASET_JOINTS
        if joint_name not in dataset_name_to_index
    ]
    if missing_required_dataset_joints:
        raise ValueError(
            f"Required upper-body dataset joints missing from dataset: {missing_required_dataset_joints}"
        )

    missing_required_controller_joints = [
        joint_name
        for joint_name in REQUIRED_UPPER_BODY_DATASET_JOINTS
        if joint_name not in controller_name_to_index
    ]
    if missing_required_controller_joints:
        raise ValueError(
            "Required upper-body dataset joints missing from controller upper_body group: "
            f"{missing_required_controller_joints}"
        )

    ignored_dataset_joints = [
        joint_name for joint_name in dataset_joint_names if joint_name not in controller_name_to_index
    ]
    preserved_default_joints = [
        joint_name
        for joint_name in controller_interface.joint_names
        if joint_name not in dataset_name_to_index
    ]

    logger.info(f"Resolved trajectory path: {resolved_path}")
    logger.info(f"qpos shape: {qpos.shape}")
    logger.info(f"Dataset joint order: {dataset_joint_names}")
    logger.info(f"Planner publishing frequency: {config.planner_frequency}")
    logger.info(f"Loop trajectory: {config.loop_trajectory}")
    logger.info(f"Controller upper-body joints: {controller_interface.joint_names}")
    logger.info(f"Dataset joints ignored by current upper_body group: {ignored_dataset_joints}")
    logger.info(f"Controller joints preserved at default values: {preserved_default_joints}")

    publish_period = 1.0 / config.planner_frequency

    return TrajectoryRuntime(
        qpos=qpos,
        dataset_joint_names=dataset_joint_names,
        controller_interface=controller_interface,
        publish_period=publish_period,
    )


def build_target_upper_body_pose(
    dataset_frame: np.ndarray,
    dataset_joint_names: list[str],
    controller_interface: ControllerUpperBodyInterface,
) -> np.ndarray:
    dataset_name_to_index = {name: idx for idx, name in enumerate(dataset_joint_names)}
    target_upper_body_pose = controller_interface.default_qpos.copy()

    for controller_idx, joint_name in enumerate(controller_interface.joint_names):
        dataset_idx = dataset_name_to_index.get(joint_name)
        if dataset_idx is None:
            continue
        target_upper_body_pose[controller_idx] = dataset_frame[dataset_idx]

    if len(target_upper_body_pose) != len(controller_interface.indices):
        raise ValueError("target_upper_body_pose length does not match controller upper_body size")
    return target_upper_body_pose


def interruptible_sleep(duration: float, keep_running) -> None:
    end_time = time.monotonic() + duration
    while keep_running():
        remaining = end_time - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def main(config: PlannerConfig):
    ros_manager = ROSManager(node_name=PLANNER_NODE_NAME)
    node = ros_manager.node
    logger = node.get_logger()
    control_publisher = None

    try:
        runtime = build_trajectory_runtime(config, logger)
        control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)

        rate = node.create_rate(config.planner_frequency)

        frame_idx = 0
        frame_direction = 1
        hold_final_pose_printed = False
        is_first_publish = True

        while rclpy.ok():
            t_now = time.monotonic()
            target_upper_body_pose = build_target_upper_body_pose(
                runtime.qpos[frame_idx], runtime.dataset_joint_names, runtime.controller_interface
            )
            target_time = (
                t_now + config.initial_transition_time
                if is_first_publish
                else t_now + runtime.publish_period
            )

            control_publisher.publish(
                {
                    "target_upper_body_pose": target_upper_body_pose,
                    "timestamp": t_now,
                    "target_time": target_time,
                }
            )

            if is_first_publish:
                logger.info(
                    f"Publishing initial trajectory frame for {config.initial_transition_time} seconds"
                )
                is_first_publish = False
                if runtime.qpos.shape[0] > 1:
                    frame_idx = 1
                interruptible_sleep(config.initial_transition_time, rclpy.ok)
                continue

            if config.loop_trajectory and runtime.qpos.shape[0] > 1:
                next_frame_idx = frame_idx + frame_direction
                if next_frame_idx >= runtime.qpos.shape[0]:
                    frame_direction = -1
                    next_frame_idx = runtime.qpos.shape[0] - 2
                elif next_frame_idx < 0:
                    frame_direction = 1
                    next_frame_idx = 1
                frame_idx = next_frame_idx
            elif frame_idx < runtime.qpos.shape[0] - 1:
                frame_idx += 1
            else:
                if not hold_final_pose_printed:
                    logger.info("Reached the final trajectory frame; holding the final trajectory pose.")
                    hold_final_pose_printed = True
                frame_idx = runtime.qpos.shape[0] - 1

            rate.sleep()

    except ros_manager.exceptions() as e:
        logger.info(f"ROSManager interrupted by user: {e}")
    finally:
        logger.info("Cleaning up...")
        ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(PlannerConfig)
    main(config)
