from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import threading
import time
from typing import Literal

import msgpack
import msgpack_numpy as mnp
import numpy as np
from std_msgs.msg import ByteMultiArray
import tyro

from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.main.planner.simulation.robot import JOINT_NAMES_UP
from decoupled_wbc.control.main.planner.utils.ros_utils import (
    ROSDictServiceClient,
)
from decoupled_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from decoupled_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
)

PLANNER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLANNER_DIR.parents[3]
PLANNER_PLAN_SERVICE = "PlannerServer/plan"


@dataclass
class ComparisonMotionConfig:
    """Execute and record one planner or direct-dataset upper-body motion."""

    mode: Literal["planner", "dataset"] = "planner"
    trajectory_path: str = "dataset/windex_l_place16.npz"
    output_dir: str = "outputs/planner_reference_comparison"
    planner_name: str = "BITstar"
    plan_service: str = PLANNER_PLAN_SERVICE
    reference_mode: Literal["with", "without"] = "with"
    planner_frequency: float = 20.0
    initial_transition_time: float = 2.0
    execution_margin: float = 0.5
    state_timeout: float = 10.0


@dataclass
class ControllerInterface:
    indices: list[int]
    joint_names: list[str]
    name_to_index: dict[str, int]
    default_qpos: np.ndarray
    planning_full_indices: list[int]


class MeasuredTrajectoryRecorder:
    """Record measured robot state in the canonical 17-joint planner order."""

    def __init__(self, node, planning_full_indices: list[int]):
        self._planning_full_indices = planning_full_indices
        self._lock = threading.Lock()
        self._latest_state_event = threading.Event()
        self._new_sample_event = threading.Event()
        self._recording = False
        self._start_monotonic = 0.0
        self._qpos: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._wall_timestamps: list[float] = []
        self._callback_error: Exception | None = None
        self.subscription = node.create_subscription(
            ByteMultiArray,
            STATE_TOPIC_NAME,
            self._callback,
            100,
        )

    def _callback(self, msg) -> None:
        try:
            payload = bytes([byte for item in msg.data for byte in item])
            state = msgpack.unpackb(payload, object_hook=mnp.decode)
            if "q" not in state:
                return
            full_q = np.asarray(state["q"], dtype=np.float32).reshape(-1)
            if full_q.shape[0] <= max(self._planning_full_indices):
                raise ValueError(
                    f"Robot state q length {full_q.shape[0]} is too short"
                )
            planning_q = full_q[self._planning_full_indices].copy()
            if not np.isfinite(planning_q).all():
                raise ValueError("Robot state contains non-finite joint values")

            receive_monotonic = time.monotonic()
            state_timestamps = state.get("timestamps", {})
            wall_timestamp = float(
                state_timestamps.get("proprio", time.time())
            )
            with self._lock:
                self._latest_state_event.set()
                if self._recording:
                    self._qpos.append(planning_q)
                    self._timestamps.append(
                        receive_monotonic - self._start_monotonic
                    )
                    self._wall_timestamps.append(wall_timestamp)
                    self._new_sample_event.set()
        except Exception as exc:
            with self._lock:
                self._callback_error = exc
                self._latest_state_event.set()
                self._new_sample_event.set()

    def _raise_callback_error(self) -> None:
        with self._lock:
            error = self._callback_error
        if error is not None:
            raise RuntimeError("Failed to decode robot state") from error

    def wait_for_state(self, timeout: float) -> None:
        if not self._latest_state_event.wait(timeout):
            raise TimeoutError(
                f"No robot state received from '{STATE_TOPIC_NAME}' "
                f"within {timeout:.1f}s"
            )
        self._raise_callback_error()

    def start(self, timeout: float) -> None:
        self.wait_for_state(timeout)
        with self._lock:
            self._qpos.clear()
            self._timestamps.clear()
            self._wall_timestamps.clear()
            self._callback_error = None
            self._new_sample_event.clear()
            self._start_monotonic = time.monotonic()
            self._recording = True
        if not self._new_sample_event.wait(timeout):
            with self._lock:
                self._recording = False
            raise TimeoutError("No measured state arrived after recording started")
        self._raise_callback_error()

    def stop(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            self._recording = False
            qpos = np.asarray(self._qpos, dtype=np.float32)
            timestamps = np.asarray(self._timestamps, dtype=np.float64)
            wall_timestamps = np.asarray(
                self._wall_timestamps, dtype=np.float64
            )
        self._raise_callback_error()
        if qpos.ndim != 2 or qpos.shape[1] != len(JOINT_NAMES_UP):
            raise RuntimeError(f"Invalid measured qpos shape: {qpos.shape}")
        if qpos.shape[0] == 0:
            raise RuntimeError("No measured states were recorded")
        if not np.isfinite(qpos).all() or not np.isfinite(timestamps).all():
            raise RuntimeError("Recorded trajectory contains non-finite values")
        if np.any(np.diff(timestamps) < 0):
            raise RuntimeError("Recorded timestamps are not monotonic")
        return qpos, timestamps, wall_timestamps


def resolve_path(path_value: str, relative_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = relative_root / path
    return path.resolve()


def load_trajectory(path: Path) -> tuple[np.ndarray, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"qpos", "joint_names"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"Trajectory is missing keys: {missing}")
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        joint_names = [str(name) for name in data["joint_names"].tolist()]
        source_fps = (
            float(np.asarray(data["fps"]).reshape(-1)[0])
            if "fps" in data.files
            else 20.0
        )

    if qpos.ndim != 2 or qpos.shape[0] < 2:
        raise ValueError(f"Expected qpos shape (N >= 2, D), got {qpos.shape}")
    if qpos.shape[1] != len(joint_names):
        raise ValueError("Trajectory qpos width does not match joint_names")
    if len(joint_names) != len(set(joint_names)):
        raise ValueError("Trajectory joint_names contains duplicates")
    if not np.isfinite(qpos).all():
        raise ValueError("Trajectory contains non-finite values")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid trajectory fps: {source_fps}")

    name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    missing_joints = [name for name in JOINT_NAMES_UP if name not in name_to_index]
    if missing_joints:
        raise ValueError(f"Trajectory is missing planning joints: {missing_joints}")
    planning_qpos = qpos[
        :, [name_to_index[name] for name in JOINT_NAMES_UP]
    ]
    return planning_qpos.astype(np.float32, copy=False), source_fps


def build_controller_interface() -> ControllerInterface:
    robot_model = instantiate_g1_robot_model(waist_location="upper_body")
    controller_indices = robot_model.get_joint_group_indices("upper_body")
    controller_names = [
        robot_model.joint_names[index] for index in controller_indices
    ]
    name_to_index = {name: idx for idx, name in enumerate(controller_names)}
    missing = [name for name in JOINT_NAMES_UP if name not in name_to_index]
    if missing:
        raise ValueError(f"Controller upper-body group is missing joints: {missing}")

    full_name_to_index = {
        name: idx for idx, name in enumerate(robot_model.joint_names)
    }
    missing_full = [name for name in JOINT_NAMES_UP if name not in full_name_to_index]
    if missing_full:
        raise ValueError(f"Robot model is missing planning joints: {missing_full}")
    return ControllerInterface(
        indices=controller_indices,
        joint_names=controller_names,
        name_to_index=name_to_index,
        default_qpos=robot_model.default_body_pose[controller_indices].astype(
            np.float32, copy=True
        ),
        planning_full_indices=[full_name_to_index[name] for name in JOINT_NAMES_UP],
    )


def planning_to_controller_pose(
    planning_qpos: np.ndarray, interface: ControllerInterface
) -> np.ndarray:
    target = interface.default_qpos.copy()
    for planning_idx, joint_name in enumerate(JOINT_NAMES_UP):
        target[interface.name_to_index[joint_name]] = planning_qpos[planning_idx]
    return target


def replay_dataset(
    trajectory: np.ndarray,
    source_fps: float,
    interface: ControllerInterface,
    publisher: ROSMsgPublisher,
    execution_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    period = 1.0 / source_fps
    print(
        f"Replaying {trajectory.shape[0]} dataset frames at "
        f"{source_fps:.1f} Hz."
    )
    start_time = time.monotonic()
    for frame_index, frame in enumerate(trajectory):
        now = time.monotonic()
        publisher.publish(
            {
                "target_upper_body_pose": planning_to_controller_pose(
                    frame, interface
                ),
                "timestamp": now,
                "target_time": now + period,
            }
        )
        deadline = start_time + (frame_index + 1) * period
        time.sleep(max(0.0, deadline - time.monotonic()))
    time.sleep(execution_margin)
    command_timestamps = np.arange(trajectory.shape[0], dtype=np.float64) * period
    return trajectory.copy(), command_timestamps


def request_planner_motion(
    trajectory: np.ndarray,
    plan_service: str,
    planner_frequency: float,
    initial_transition_time: float,
    execution_margin: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    client = ROSDictServiceClient(
        plan_service,
        node_name="ComparisonPlannerRequest",
    )
    try:
        response = client.call(
            {
                "start_qpos": trajectory[0],
                "goal_qpos": trajectory[-1],
                "goal_type": "upper_body",
                "execute_immediately": True,
            }
        )
    finally:
        client.destroy_node()

    commanded_qpos = np.asarray(response["qpos"], dtype=np.float32)
    if commanded_qpos.ndim != 2 or commanded_qpos.shape[1] != len(
        JOINT_NAMES_UP
    ):
        raise RuntimeError(
            f"Planner returned invalid path shape: {commanded_qpos.shape}"
        )
    start_error = float(np.max(np.abs(commanded_qpos[0] - trajectory[0])))
    goal_error = float(np.max(np.abs(commanded_qpos[-1] - trajectory[-1])))
    if start_error > 1e-4 or goal_error > 1e-4:
        raise RuntimeError(
            "Planner returned invalid endpoints: "
            f"start_error={start_error:.6f}, goal_error={goal_error:.6f}"
        )

    command_timestamps = np.zeros(commanded_qpos.shape[0], dtype=np.float64)
    if commanded_qpos.shape[0] > 1:
        command_timestamps[1:] = initial_transition_time + np.arange(
            commanded_qpos.shape[0] - 1, dtype=np.float64
        ) / planner_frequency
    execution_time = (
        initial_transition_time
        + max(0, commanded_qpos.shape[0] - 1) / planner_frequency
        + execution_margin
    )
    print(
        f"Planner returned {commanded_qpos.shape[0]} executable waypoints; "
        f"recording execution for {execution_time:.2f}s."
    )
    return commanded_qpos, command_timestamps, response


def estimate_fps(timestamps: np.ndarray) -> float:
    positive_deltas = np.diff(timestamps)
    positive_deltas = positive_deltas[positive_deltas > 0]
    if positive_deltas.size == 0:
        return 0.0
    return float(1.0 / np.median(positive_deltas))


def save_recording(
    output_dir: Path,
    config: ComparisonMotionConfig,
    trajectory_path: Path,
    measured_qpos: np.ndarray,
    timestamps: np.ndarray,
    wall_timestamps: np.ndarray,
    commanded_qpos: np.ndarray,
    commanded_timestamps: np.ndarray,
    completed: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    mode_label = "planner" if config.mode == "planner" else "dataset"
    output_path = output_dir / f"{timestamp_label}_{mode_label}.npz"
    temporary_path = output_dir / f".{output_path.name}.tmp.npz"
    np.savez_compressed(
        temporary_path,
        qpos=measured_qpos,
        measured_qpos=measured_qpos,
        joint_names=np.asarray(JOINT_NAMES_UP),
        timestamps=timestamps,
        wall_timestamps=wall_timestamps,
        fps=np.asarray([estimate_fps(timestamps)], dtype=np.float64),
        commanded_qpos=commanded_qpos,
        commanded_timestamps=commanded_timestamps,
        mode=np.asarray(config.mode),
        planner=np.asarray(config.planner_name if config.mode == "planner" else ""),
        plan_service=np.asarray(config.plan_service),
        reference_mode=np.asarray(config.reference_mode),
        source_trajectory=np.asarray(str(trajectory_path)),
        completed=np.asarray(completed),
        schema_version=np.asarray(1, dtype=np.int64),
    )
    os.replace(temporary_path, output_path)
    return output_path


def main(config: ComparisonMotionConfig) -> None:
    if not np.isfinite(config.planner_frequency) or config.planner_frequency <= 0:
        raise ValueError("planner_frequency must be > 0")
    if (
        not np.isfinite(config.initial_transition_time)
        or config.initial_transition_time < 0
    ):
        raise ValueError("initial_transition_time must be >= 0")
    if not np.isfinite(config.execution_margin) or config.execution_margin < 0:
        raise ValueError("execution_margin must be >= 0")

    trajectory_path = resolve_path(config.trajectory_path, PLANNER_DIR)
    output_dir = resolve_path(config.output_dir, REPO_ROOT)
    trajectory, source_fps = load_trajectory(trajectory_path)
    interface = build_controller_interface()

    ros_manager = ROSManager(node_name="ComparisonMotionRecorder")
    recorder = MeasuredTrajectoryRecorder(
        ros_manager.node, interface.planning_full_indices
    )
    publisher = None
    commanded_qpos = np.empty((0, len(JOINT_NAMES_UP)), dtype=np.float32)
    commanded_timestamps = np.empty((0,), dtype=np.float64)
    measured_qpos = np.empty((0, len(JOINT_NAMES_UP)), dtype=np.float32)
    timestamps = np.empty((0,), dtype=np.float64)
    wall_timestamps = np.empty((0,), dtype=np.float64)
    recording_started = False
    completed = False
    planner_response: dict = {}

    try:
        recorder.wait_for_state(config.state_timeout)
        if config.mode == "planner":
            (
                commanded_qpos,
                commanded_timestamps,
                planner_response,
            ) = request_planner_motion(
                trajectory,
                config.plan_service,
                config.planner_frequency,
                config.initial_transition_time,
                config.execution_margin,
            )
            recorder.start(config.state_timeout)
            recording_started = True
            execution_time = (
                config.initial_transition_time
                + max(0, commanded_qpos.shape[0] - 1)
                / config.planner_frequency
                + config.execution_margin
            )
            time.sleep(execution_time)
        else:
            publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
            recorder.start(config.state_timeout)
            recording_started = True
            commanded_qpos, commanded_timestamps = replay_dataset(
                trajectory,
                source_fps,
                interface,
                publisher,
                config.execution_margin,
            )
        completed = True
    finally:
        try:
            if recording_started:
                measured_qpos, timestamps, wall_timestamps = recorder.stop()
                output_path = save_recording(
                    output_dir,
                    config,
                    trajectory_path,
                    measured_qpos,
                    timestamps,
                    wall_timestamps,
                    commanded_qpos,
                    commanded_timestamps,
                    completed,
                )
                final_error = float(
                    np.max(np.abs(measured_qpos[-1] - trajectory[-1]))
                )
                print(
                    f"Saved measured upper-body trajectory: {output_path}\n"
                    f"Samples: {measured_qpos.shape[0]}, "
                    f"measured FPS: {estimate_fps(timestamps):.1f}, "
                    f"final max error: {final_error:.4f} rad"
                )
                if planner_response:
                    print(
                        "Planner time: "
                        f"{float(planner_response['planning_time']):.3f}s"
                    )
        finally:
            ros_manager.shutdown()
            spin_thread = getattr(ros_manager, "thread", None)
            if spin_thread is not None:
                spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main(tyro.cli(ComparisonMotionConfig))
