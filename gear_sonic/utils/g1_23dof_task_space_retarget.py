"""Constrained task-space retargeting from G1 29-DoF to rev-1.0 23-DoF.

The released SONIC decoder can be sliced exactly to 23 outputs, but that slice
cannot compensate for the six joints missing on the physical rev-1.0 robot.
This module builds a geometric expert instead: it matches task-space poses with
the missing joints absent, while enforcing the target robot's trajectory
constraints.  The result is offline training material, never deployment
authorization.

All joint vectors at this boundary use MuJoCo/hardware order.  Policy action
targets are converted explicitly to native IsaacLab-23 order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Literal, Sequence

import mujoco
import numpy as np
from scipy.spatial import transform

from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    HARDWARE_23_JOINT_NAMES,
    MUJOCO_TO_ISAACLAB_DOF,
    TARGET_DOF,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
    SAFE_TARGET_INNER_LOWER_HARDWARE,
    SAFE_TARGET_INNER_UPPER_HARDWARE,
    SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
)
from gear_sonic.utils.g1_23dof_trajectory_projection import (
    TrajectoryProjectionConfig,
    audit_trajectory_constraints,
    project_nearest_trajectory,
)

DEFAULT_SOURCE_MODEL = "gear_sonic/data/robots/g1/g1_29dof.xml"
DEFAULT_TARGET_MODEL = "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
TASK_SPACE_RETARGET_SCHEMA_VERSION = 6
FREE_QPOS_WIDTH = 7
KINEMATIC_GATE_MAX_STANCE_FOOT_ERROR_M = 0.005
KINEMATIC_GATE_ACTION_CLIP = SAFE_TARGET_RAW_ACTION_CLIP
KINEMATIC_GATE_MIN_VALID_FRAME_FRACTION = 0.95
TRAJECTORY_SERIALIZATION_LIMIT_FRACTION = 0.995
TRAJECTORY_DERIVATIVE_CONVENTION = "free_initial_velocity_equal_first_interval"


@dataclass(frozen=True)
class TaskSpec:
    """One desired source feature and its corresponding target feature."""

    name: str
    kind: Literal["body", "subtree_com"]
    source_body: str
    target_body: str
    source_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    priority: int = 0
    orientation_priority: int | None = None
    position_weight: float = 0.0
    orientation_weight: float = 0.0
    contact_side: Literal["left", "right"] | None = None

    def __post_init__(self) -> None:
        if self.position_weight < 0.0 or self.orientation_weight < 0.0:
            raise ValueError(f"task {self.name!r} has a negative weight")
        if self.priority < 0:
            raise ValueError(f"task {self.name!r} has a negative priority")
        if self.orientation_priority is not None and self.orientation_priority < 0:
            raise ValueError(
                f"task {self.name!r} has a negative orientation priority"
            )
        if self.kind == "subtree_com" and self.orientation_weight:
            raise ValueError("subtree COM tasks cannot carry orientation weight")
        if not self.position_weight and not self.orientation_weight:
            raise ValueError(f"task {self.name!r} has no active objective")


# Priority is lexicographic: feet/contact, whole-robot COM, torso/head, hands,
# then elbow shaping. Weights only balance objectives inside the same tier.
# These are the existing SONIC command/reward semantics: both embodiments use
# the same local 18 cm hand offsets even though their wrist chains differ.
DEFAULT_TASKS = (
    TaskSpec(
        "left_foot",
        "body",
        "left_ankle_roll_link",
        "left_ankle_roll_link",
        priority=0,
        position_weight=32.0,
        orientation_weight=16.0,
        contact_side="left",
    ),
    TaskSpec(
        "right_foot",
        "body",
        "right_ankle_roll_link",
        "right_ankle_roll_link",
        priority=0,
        position_weight=32.0,
        orientation_weight=16.0,
        contact_side="right",
    ),
    TaskSpec(
        "whole_robot_com",
        "subtree_com",
        "pelvis",
        "pelvis",
        priority=1,
        position_weight=18.0,
    ),
    TaskSpec(
        "torso_orientation",
        "body",
        "torso_link",
        "torso_link",
        priority=2,
        orientation_weight=12.0,
    ),
    TaskSpec(
        "head_proxy",
        "body",
        "torso_link",
        "torso_link",
        source_point=(0.0, 0.0, 0.35),
        target_point=(0.0, 0.0, 0.35),
        priority=2,
        position_weight=8.0,
    ),
    TaskSpec(
        "left_hand",
        "body",
        "left_wrist_yaw_link",
        "left_wrist_roll_rubber_hand",
        source_point=(0.18, -0.025, 0.0),
        target_point=(0.18, -0.025, 0.0),
        priority=3,
        orientation_priority=5,
        position_weight=6.0,
        orientation_weight=0.2,
    ),
    TaskSpec(
        "right_hand",
        "body",
        "right_wrist_yaw_link",
        "right_wrist_roll_rubber_hand",
        source_point=(0.18, 0.025, 0.0),
        target_point=(0.18, 0.025, 0.0),
        priority=3,
        orientation_priority=5,
        position_weight=6.0,
        orientation_weight=0.2,
    ),
    TaskSpec(
        "left_elbow",
        "body",
        "left_elbow_link",
        "left_elbow_link",
        priority=4,
        position_weight=1.5,
    ),
    TaskSpec(
        "right_elbow",
        "body",
        "right_elbow_link",
        "right_elbow_link",
        priority=4,
        position_weight=1.5,
    ),
)


@dataclass(frozen=True)
class RetargetConfig:
    """Numerical and trajectory constraints for the bounded IK solver."""

    max_iterations: int = 16
    damping: float = 2.0e-3
    posture_weight: float = 0.08
    smoothness_weight: float = 0.16
    max_iteration_step_rad: float = 0.18
    max_velocity_rad_s: float = 8.0
    max_acceleration_rad_s2: float = 80.0
    contact_weight_multiplier: float = 2.5
    contact_height_tolerance_m: float = 0.035
    contact_speed_tolerance_m_s: float = 0.45
    objective_tolerance: float = 1.0e-10
    priority_relative_tolerance: float = 1.0e-7
    protected_priority_tiers: int = 2
    safe_limit_guard_rad: float = 1.0e-5
    native_action_clip: float = KINEMATIC_GATE_ACTION_CLIP
    valid_max_foot_position_error_m: float = KINEMATIC_GATE_MAX_STANCE_FOOT_ERROR_M
    valid_max_foot_orientation_regression_rad: float = 0.005
    valid_max_com_regression_m: float = 0.001
    allow_acceleration_constraint_relaxation: bool = False
    optimize_lower_body: bool = False
    enable_lower_root_feasibility: bool = True
    lower_root_max_iterations: int = 8
    lower_root_max_step_m: float = 0.02
    lower_root_joint_step_rad: float = 0.18
    max_root_offset_m: float = 0.08
    max_root_offset_velocity_m_s: float = 0.75
    max_root_offset_acceleration_m_s2: float = 6.0
    lower_root_posture_weight: float = 0.02
    lower_root_smoothness_weight: float = 0.04
    step_tolerance_rad: float = 1.0e-7

    def __post_init__(self) -> None:
        positive = {
            "max_iterations": self.max_iterations,
            "damping": self.damping,
            "max_iteration_step_rad": self.max_iteration_step_rad,
            "max_velocity_rad_s": self.max_velocity_rad_s,
            "max_acceleration_rad_s2": self.max_acceleration_rad_s2,
            "contact_weight_multiplier": self.contact_weight_multiplier,
            "native_action_clip": self.native_action_clip,
            "lower_root_max_iterations": self.lower_root_max_iterations,
            "lower_root_max_step_m": self.lower_root_max_step_m,
            "lower_root_joint_step_rad": self.lower_root_joint_step_rad,
            "max_root_offset_m": self.max_root_offset_m,
            "max_root_offset_velocity_m_s": self.max_root_offset_velocity_m_s,
            "max_root_offset_acceleration_m_s2": (
                self.max_root_offset_acceleration_m_s2
            ),
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"retarget configuration must be positive: {invalid}")
        if self.safe_limit_guard_rad <= 0.0:
            raise ValueError("safe_limit_guard_rad must be positive")
        nonnegative = {
            "valid_max_foot_position_error_m": self.valid_max_foot_position_error_m,
            "valid_max_foot_orientation_regression_rad": (
                self.valid_max_foot_orientation_regression_rad
            ),
            "valid_max_com_regression_m": self.valid_max_com_regression_m,
            "lower_root_posture_weight": self.lower_root_posture_weight,
            "lower_root_smoothness_weight": self.lower_root_smoothness_weight,
        }
        invalid_nonnegative = [
            name for name, value in nonnegative.items() if value < 0.0
        ]
        if invalid_nonnegative:
            raise ValueError(
                "retarget validity tolerances must be nonnegative: "
                f"{invalid_nonnegative}"
            )
        if self.protected_priority_tiers < 1:
            raise ValueError("protected_priority_tiers must be positive")


@dataclass(frozen=True)
class _ModelLayout:
    joint_names: tuple[str, ...]
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class _TaskTarget:
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class _JointGroups:
    lower_body: np.ndarray
    upper_body: np.ndarray


@dataclass(frozen=True)
class _LowerRootMetrics:
    foot_position_error_m: np.ndarray
    foot_orientation_error_rad: np.ndarray
    com_position_error_m: np.ndarray
    weighted_error: np.ndarray


@dataclass(frozen=True)
class _LowerRootPath:
    root_offset_w: np.ndarray
    joint_pos_hardware: np.ndarray
    metrics_before: _LowerRootMetrics
    metrics_after: _LowerRootMetrics
    solver_iterations: int
    accepted_step_count: int
    projection_iteration_count: int


@dataclass
class RetargetResult:
    """Task-space expert trajectory and auditable per-frame measurements."""

    source_root_pos_w: np.ndarray
    root_pos_w: np.ndarray
    root_offset_w: np.ndarray
    root_quat_wxyz: np.ndarray
    direct_joint_pos_hardware: np.ndarray
    joint_pos_hardware: np.ndarray
    direct_action_native: np.ndarray
    action_target_native: np.ndarray
    contact_flags: np.ndarray
    desired_task_pos_w: np.ndarray
    achieved_task_pos_w: np.ndarray
    desired_task_quat_wxyz: np.ndarray
    achieved_task_quat_wxyz: np.ndarray
    task_has_orientation: np.ndarray
    task_names: tuple[str, ...]
    diagnostics: dict[str, np.ndarray]
    fps: float
    config: RetargetConfig

    def expert_valid_mask(self) -> np.ndarray:
        """Frames that improve total cost within physical protected-task bounds."""

        before = self.diagnostics["weighted_task_error_before"]
        after = self.diagnostics["weighted_task_error_after"]
        tolerance = self.config.priority_relative_tolerance * np.maximum(
            1.0, before
        )
        valid = after <= before + tolerance
        for foot_name in ("left_foot", "right_foot"):
            position_after = self.diagnostics[
                f"task_{foot_name}_position_error_after_m"
            ]
            orientation_before = self.diagnostics[
                f"task_{foot_name}_orientation_error_before_rad"
            ]
            orientation_after = self.diagnostics[
                f"task_{foot_name}_orientation_error_after_rad"
            ]
            valid &= position_after <= self.config.valid_max_foot_position_error_m
            valid &= (
                orientation_after
                <= orientation_before
                + self.config.valid_max_foot_orientation_regression_rad
            )
        com_before = self.diagnostics[
            "task_whole_robot_com_position_error_before_m"
        ]
        com_after = self.diagnostics[
            "task_whole_robot_com_position_error_after_m"
        ]
        valid &= com_after <= com_before + self.config.valid_max_com_regression_m
        valid &= self.diagnostics["constraint_relaxation_count"] == 0
        valid &= np.all(
            np.abs(self.action_target_native) <= self.config.native_action_clip,
            axis=1,
        )
        return valid

    def adaptation_arrays(self) -> dict[str, np.ndarray]:
        """Return array-only expert material suitable for an ``npz`` file."""

        dt = 1.0 / self.fps
        stored_joint_pos = self.joint_pos_hardware.astype(np.float32)
        joint_vel, joint_acc = trajectory_derivatives(
            stored_joint_pos.astype(np.float64), dt
        )
        stored_root_offset = self.root_offset_w.astype(np.float32)
        root_offset_vel, root_offset_acc = trajectory_derivatives(
            stored_root_offset.astype(np.float64), dt
        )
        serialized_diagnostics = {
            name: values.astype(np.float32)
            for name, values in self.diagnostics.items()
        }
        serialized_diagnostics.update(
            {
                "trajectory_velocity_abs_max": np.max(
                    np.abs(joint_vel), axis=1
                ).astype(np.float32),
                "trajectory_acceleration_abs_max": np.max(
                    np.abs(joint_acc), axis=1
                ).astype(np.float32),
                "root_offset_velocity_abs_max": np.max(
                    np.abs(root_offset_vel), axis=1
                ).astype(np.float32),
                "root_offset_acceleration_abs_max": np.max(
                    np.abs(root_offset_acc), axis=1
                ).astype(np.float32),
            }
        )
        return {
            "schema_version": np.asarray(
                [TASK_SPACE_RETARGET_SCHEMA_VERSION], dtype=np.int64
            ),
            "fps": np.asarray([self.fps], dtype=np.float64),
            "source_root_pos_w": self.source_root_pos_w.astype(np.float32),
            "root_pos_w": self.root_pos_w.astype(np.float32),
            "root_offset_w": stored_root_offset,
            "root_offset_vel_w": root_offset_vel.astype(np.float32),
            "root_quat_wxyz": self.root_quat_wxyz.astype(np.float32),
            "direct_joint_pos_hardware": self.direct_joint_pos_hardware.astype(
                np.float32
            ),
            "joint_pos_hardware": stored_joint_pos,
            "joint_vel_hardware": joint_vel.astype(np.float32),
            "joint_acc_hardware": joint_acc.astype(np.float32),
            "residual_joint_pos_hardware": (
                self.joint_pos_hardware - self.direct_joint_pos_hardware
            ).astype(np.float32),
            "direct_action_native": self.direct_action_native.astype(np.float32),
            "action_target_native": self.action_target_native.astype(np.float32),
            "contact_flags": self.contact_flags.astype(np.bool_),
            "desired_task_pos_w": self.desired_task_pos_w.astype(np.float32),
            "achieved_task_pos_w": self.achieved_task_pos_w.astype(np.float32),
            "desired_task_quat_wxyz": self.desired_task_quat_wxyz.astype(
                np.float32
            ),
            "achieved_task_quat_wxyz": self.achieved_task_quat_wxyz.astype(
                np.float32
            ),
            "task_has_orientation": self.task_has_orientation.astype(np.bool_),
            "expert_valid": self.expert_valid_mask().astype(np.bool_),
            "trajectory_derivative_convention": np.asarray(
                [TRAJECTORY_DERIVATIVE_CONVENTION], dtype=np.str_
            ),
            **serialized_diagnostics,
        }

    def summary(self) -> dict[str, object]:
        before = self.diagnostics["weighted_task_error_before"]
        after = self.diagnostics["weighted_task_error_after"]
        denominator = np.maximum(before, 1.0e-12)
        per_task = {
            task_name: {
                "position_error_before_mean_m": float(
                    np.mean(
                        self.diagnostics[
                            f"task_{task_name}_position_error_before_m"
                        ]
                    )
                ),
                "position_error_after_mean_m": float(
                    np.mean(
                        self.diagnostics[
                            f"task_{task_name}_position_error_after_m"
                        ]
                    )
                ),
                "position_error_before_p95_m": float(
                    np.percentile(
                        self.diagnostics[
                            f"task_{task_name}_position_error_before_m"
                        ],
                        95.0,
                    )
                ),
                "position_error_after_p95_m": float(
                    np.percentile(
                        self.diagnostics[
                            f"task_{task_name}_position_error_after_m"
                        ],
                        95.0,
                    )
                ),
                "position_error_before_max_m": float(
                    np.max(
                        self.diagnostics[
                            f"task_{task_name}_position_error_before_m"
                        ]
                    )
                ),
                "position_error_after_max_m": float(
                    np.max(
                        self.diagnostics[
                            f"task_{task_name}_position_error_after_m"
                        ]
                    )
                ),
                "orientation_error_before_mean_rad": float(
                    np.mean(
                        self.diagnostics[
                            f"task_{task_name}_orientation_error_before_rad"
                        ]
                    )
                ),
                "orientation_error_after_mean_rad": float(
                    np.mean(
                        self.diagnostics[
                            f"task_{task_name}_orientation_error_after_rad"
                        ]
                    )
                ),
            }
            for task_name in self.task_names
        }
        gate_failures: list[str] = []
        valid_fraction = float(np.mean(self.expert_valid_mask()))
        if valid_fraction < KINEMATIC_GATE_MIN_VALID_FRAME_FRACTION:
            gate_failures.append(
                "valid expert-frame fraction is below "
                f"{KINEMATIC_GATE_MIN_VALID_FRAME_FRACTION:.0%}"
            )
        if float(np.mean(after)) >= float(np.mean(before)):
            gate_failures.append("no mean task-space improvement")
        for foot_name in ("left_foot", "right_foot"):
            if foot_name in per_task:
                foot_error = self.diagnostics[
                    f"task_{foot_name}_position_error_after_m"
                ]
                if float(np.max(foot_error)) > KINEMATIC_GATE_MAX_STANCE_FOOT_ERROR_M:
                    gate_failures.append(
                        f"{foot_name} error exceeds "
                        f"{KINEMATIC_GATE_MAX_STANCE_FOOT_ERROR_M:.3f} m"
                    )
        saturation_fraction = float(np.mean(np.abs(self.action_target_native) > 1.0))
        clipped_fraction = float(
            np.mean(np.abs(self.action_target_native) > self.config.native_action_clip)
        )
        if clipped_fraction:
            gate_failures.append(
                "expert action exceeds configured "
                f"+/-{self.config.native_action_clip:g} action clip"
            )
        if np.sum(self.diagnostics["constraint_relaxation_count"]):
            gate_failures.append("trajectory constraints were relaxed")
        return {
            "schema": "g1_29dof_to_true23_task_space_retarget_v6",
            "schema_version": TASK_SPACE_RETARGET_SCHEMA_VERSION,
            "deployment_ready": False,
            "authorization": "offline_training_material_only",
            "frame_count": int(len(self.joint_pos_hardware)),
            "fps": float(self.fps),
            "task_names": list(self.task_names),
            "per_task_error": per_task,
            "kinematic_gate_passed": not gate_failures,
            "kinematic_gate_failures": gate_failures,
            "weighted_task_error_before_mean": float(np.mean(before)),
            "weighted_task_error_after_mean": float(np.mean(after)),
            "mean_relative_improvement": float(np.mean((before - after) / denominator)),
            "frames_improved_fraction": float(np.mean(after < before)),
            "expert_valid_frame_fraction": valid_fraction,
            "solver_zero_accepted_step_frame_count": int(
                np.sum(self.diagnostics["solver_accepted_step_count"] == 0)
            ),
            "solver_feasible_seed_fallback_frame_count": int(
                np.sum(self.diagnostics["solver_used_feasible_seed_fallback"])
            ),
            "position_limit_hit_count": int(
                np.sum(self.diagnostics["position_limit_hit_count"])
            ),
            "safe_envelope_hit_count": int(
                np.sum(self.diagnostics["position_limit_hit_count"])
            ),
            "hard_position_limit_violation_count": int(
                np.sum(
                    (
                        self.joint_pos_hardware
                        < np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) - 1.0e-9
                    )
                    | (
                        self.joint_pos_hardware
                        > np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) + 1.0e-9
                    )
                )
            ),
            "trajectory_constraint_relaxation_count": int(
                np.sum(self.diagnostics["constraint_relaxation_count"])
            ),
            "native_action_abs_gt_one_fraction": saturation_fraction,
            "native_action_abs_gt_ten_fraction": clipped_fraction,
            "native_action_abs_max": float(np.max(np.abs(self.action_target_native))),
            "direct_native_action_abs_gt_one_fraction": float(
                np.mean(np.abs(self.direct_action_native) > 1.0)
            ),
            "direct_native_action_abs_gt_ten_fraction": float(
                np.mean(
                    np.abs(self.direct_action_native) > self.config.native_action_clip
                )
            ),
            "direct_native_action_abs_max": float(
                np.max(np.abs(self.direct_action_native))
            ),
            "root_offset_norm_max_m": float(
                np.max(np.linalg.norm(self.root_offset_w, axis=1))
            ),
            "root_offset_component_abs_max_m": float(
                np.max(np.abs(self.root_offset_w))
            ),
            "lower_root_solver_iterations": int(
                np.max(self.diagnostics["lower_root_solver_iterations"])
            ),
            "lower_root_accepted_step_count": int(
                np.max(self.diagnostics["lower_root_accepted_step_count"])
            ),
            "lower_root_projection_iteration_count": int(
                np.max(
                    self.diagnostics["lower_root_projection_iteration_count"]
                )
            ),
            "constraints": {
                "max_velocity_rad_s": self.config.max_velocity_rad_s,
                "max_acceleration_rad_s2": self.config.max_acceleration_rad_s2,
                "max_iteration_step_rad": self.config.max_iteration_step_rad,
                "native_action_clip": self.config.native_action_clip,
                "serialization_limit_fraction": (
                    TRAJECTORY_SERIALIZATION_LIMIT_FRACTION
                ),
                "derivative_convention": TRAJECTORY_DERIVATIVE_CONVENTION,
                "max_root_offset_m": self.config.max_root_offset_m,
                "max_root_offset_velocity_m_s": (
                    self.config.max_root_offset_velocity_m_s
                ),
                "max_root_offset_acceleration_m_s2": (
                    self.config.max_root_offset_acceleration_m_s2
                ),
                "measured_velocity_abs_max_rad_s": float(
                    np.max(self.diagnostics["trajectory_velocity_abs_max"])
                ),
                "measured_acceleration_abs_max_rad_s2": float(
                    np.max(self.diagnostics["trajectory_acceleration_abs_max"])
                ),
                "measured_root_offset_velocity_abs_max_m_s": float(
                    np.max(self.diagnostics["root_offset_velocity_abs_max"])
                ),
                "measured_root_offset_acceleration_abs_max_m_s2": float(
                    np.max(
                        self.diagnostics["root_offset_acceleration_abs_max"]
                    )
                ),
            },
        }


def load_models(
    source_model_path: str | Path = DEFAULT_SOURCE_MODEL,
    target_model_path: str | Path = DEFAULT_TARGET_MODEL,
) -> tuple[mujoco.MjModel, mujoco.MjModel]:
    """Load and validate the exact source and rev-1.0 target MuJoCo models."""

    source = mujoco.MjModel.from_xml_path(str(Path(source_model_path).resolve()))
    target = mujoco.MjModel.from_xml_path(str(Path(target_model_path).resolve()))
    source_layout = _model_layout(source)
    target_layout = _model_layout(target)
    if len(source_layout.joint_names) != 29:
        raise ValueError(f"source model must have 29 actuated joints, got {source.nu}")
    if target_layout.joint_names != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError(
            "target model joint order is not the rev-1.0 hardware contract: "
            f"{target_layout.joint_names}"
        )
    missing = sorted(set(target_layout.joint_names) - set(source_layout.joint_names))
    if missing:
        raise ValueError(f"target joints absent from source model: {missing}")
    return source, target


def safe_target_joint_bounds(
    target_model: mujoco.MjModel,
    *,
    safe_limit_guard_rad: float = RetargetConfig.safe_limit_guard_rad,
    native_action_clip: float = RetargetConfig.native_action_clip,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact hardware-order q bounds reachable through deployed clipping."""

    layout = _safe_target_layout(
        _model_layout(target_model),
        safe_limit_guard_rad,
        native_action_clip,
    )
    return layout.lower.copy(), layout.upper.copy()


def _model_layout(model: mujoco.MjModel) -> _ModelLayout:
    joint_ids: list[int] = []
    names: list[str] = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if name is None:
            raise ValueError(f"model joint {joint_id} is unnamed")
        joint_ids.append(joint_id)
        names.append(name)
    ids = np.asarray(joint_ids, dtype=np.int64)
    limited = np.asarray(model.jnt_limited[ids], dtype=bool)
    lower = np.where(limited, model.jnt_range[ids, 0], -np.inf).astype(np.float64)
    upper = np.where(limited, model.jnt_range[ids, 1], np.inf).astype(np.float64)
    return _ModelLayout(
        joint_names=tuple(names),
        joint_ids=ids,
        qpos_addresses=np.asarray(model.jnt_qposadr[ids], dtype=np.int64),
        dof_addresses=np.asarray(model.jnt_dofadr[ids], dtype=np.int64),
        lower=lower,
        upper=upper,
    )


def _safe_target_layout(
    layout: _ModelLayout,
    guard_rad: float,
    native_action_clip: float,
) -> _ModelLayout:
    """Tighten bounds to targets reachable through deployed action clipping."""

    if layout.joint_names != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError("safe target bounds require hardware-order true23 joints")
    safe_lower = np.asarray(SAFE_TARGET_INNER_LOWER_HARDWARE, dtype=np.float64)
    safe_upper = np.asarray(SAFE_TARGET_INNER_UPPER_HARDWARE, dtype=np.float64)
    default = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float64)
    scale = np.asarray(HARDWARE_23_ACTION_SCALE, dtype=np.float64)
    positive = np.asarray(SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE, dtype=np.float64)
    negative = np.asarray(SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE, dtype=np.float64)
    reachable_lower = default - negative * np.tanh(
        native_action_clip * scale / negative
    )
    reachable_upper = default + positive * np.tanh(
        native_action_clip * scale / positive
    )
    lower = np.maximum.reduce(
        (layout.lower, safe_lower + guard_rad, reachable_lower + guard_rad)
    )
    upper = np.minimum.reduce(
        (layout.upper, safe_upper - guard_rad, reachable_upper - guard_rad)
    )
    if np.any(lower >= upper):
        raise ValueError("safe-target guard makes one or more joint ranges empty")
    return _ModelLayout(
        joint_names=layout.joint_names,
        joint_ids=layout.joint_ids,
        qpos_addresses=layout.qpos_addresses,
        dof_addresses=layout.dof_addresses,
        lower=lower,
        upper=upper,
    )


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"model lacks required body {name!r}")
    return body_id


def validate_tasks(
    source_model: mujoco.MjModel,
    target_model: mujoco.MjModel,
    tasks: Sequence[TaskSpec],
) -> tuple[TaskSpec, ...]:
    result = tuple(tasks)
    if not result:
        raise ValueError("at least one task-space objective is required")
    names = [task.name for task in result]
    if len(names) != len(set(names)):
        raise ValueError("task-space objective names must be unique")
    for task in result:
        _body_id(source_model, task.source_body)
        _body_id(target_model, task.target_body)
    return result


def _normalize_root_quaternions(value: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(value, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("root_quat_wxyz must have shape [frames, 4]")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1.0e-12):
        raise ValueError("root quaternion contains a zero-norm frame")
    return quaternions / norms


def _set_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout: _ModelLayout,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joints: np.ndarray,
) -> None:
    data.qpos[:3] = root_pos
    data.qpos[3:FREE_QPOS_WIDTH] = root_quat_wxyz
    data.qpos[layout.qpos_addresses] = joints
    mujoco.mj_forward(model, data)


def _point_pose(
    data: mujoco.MjData,
    body_id: int,
    local_point: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()
    point = np.asarray(data.xpos[body_id], dtype=np.float64) + rotation @ np.asarray(
        local_point, dtype=np.float64
    )
    return point, rotation


def _task_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tasks: Sequence[TaskSpec],
) -> tuple[_TaskTarget, ...]:
    targets: list[_TaskTarget] = []
    for task in tasks:
        body_id = _body_id(model, task.source_body)
        if task.kind == "subtree_com":
            position = np.asarray(data.subtree_com[body_id], dtype=np.float64).copy()
            rotation = np.eye(3, dtype=np.float64)
        else:
            position, rotation = _point_pose(data, body_id, task.source_point)
        targets.append(_TaskTarget(position=position, rotation=rotation))
    return tuple(targets)


def _joint_groups(layout: _ModelLayout) -> _JointGroups:
    lower_body = np.asarray(
        [
            index
            for index, name in enumerate(layout.joint_names)
            if name.startswith("left_hip_")
            or name.startswith("right_hip_")
            or "knee" in name
            or "ankle" in name
        ],
        dtype=np.int64,
    )
    upper_body = np.setdiff1d(
        np.arange(len(layout.joint_names), dtype=np.int64),
        lower_body,
        assume_unique=True,
    )
    if len(lower_body) != 12 or len(upper_body) != TARGET_DOF - 12:
        raise ValueError(
            "true23 lower/upper joint split must contain 12/11 joints, got "
            f"{len(lower_body)}/{len(upper_body)}"
        )
    return _JointGroups(lower_body=lower_body, upper_body=upper_body)


def _lower_root_task_indices(tasks: Sequence[TaskSpec]) -> tuple[int, int, int]:
    by_name = {task.name: index for index, task in enumerate(tasks)}
    required = ("left_foot", "right_foot", "whole_robot_com")
    missing = [name for name in required if name not in by_name]
    if missing:
        raise ValueError(
            "lower/root feasibility stage requires task objectives "
            f"{missing}"
        )
    return tuple(by_name[name] for name in required)  # type: ignore[return-value]


def _lower_root_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout: _ModelLayout,
    tasks: Sequence[TaskSpec],
    targets_by_frame: Sequence[Sequence[_TaskTarget]],
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joints: np.ndarray,
    contacts: np.ndarray,
    lower_joint_indices: np.ndarray,
    config: RetargetConfig,
) -> _LowerRootMetrics:
    left_foot, right_foot, com = _lower_root_task_indices(tasks)
    frame_count = len(joints)
    foot_position = np.empty((frame_count, 2), dtype=np.float64)
    foot_orientation = np.empty_like(foot_position)
    com_position = np.empty(frame_count, dtype=np.float64)
    weighted_error = np.empty(frame_count, dtype=np.float64)
    for frame in range(frame_count):
        _set_configuration(
            model,
            data,
            layout,
            root_pos[frame],
            root_quat[frame],
            joints[frame],
        )
        jacobian, residual, position, orientation, _, _ = _task_linearization(
            model,
            data,
            layout,
            tasks,
            targets_by_frame[frame],
            (bool(contacts[frame, 0]), bool(contacts[frame, 1])),
            config.contact_weight_multiplier,
            lower_joint_indices,
        )
        foot_position[frame] = position[[left_foot, right_foot]]
        foot_orientation[frame] = orientation[[left_foot, right_foot]]
        com_position[frame] = position[com]
        weighted_error[frame] = _weighted_task_error(jacobian, residual)
    return _LowerRootMetrics(
        foot_position_error_m=foot_position,
        foot_orientation_error_rad=foot_orientation,
        com_position_error_m=com_position,
        weighted_error=weighted_error,
    )


def _lower_root_frame_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout: _ModelLayout,
    tasks: Sequence[TaskSpec],
    targets: Sequence[_TaskTarget],
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joints: np.ndarray,
    contacts: tuple[bool, bool],
    lower_joint_indices: np.ndarray,
    variable_scale: np.ndarray,
    current_variable: np.ndarray,
    reference: np.ndarray,
    smooth_reference: np.ndarray,
    config: RetargetConfig,
) -> np.ndarray:
    """One normalized Gauss-Newton step with foot position as hard first tier."""

    _set_configuration(model, data, layout, root_pos, root_quat, joints)
    variable_dofs = np.concatenate(
        (
            np.arange(3, dtype=np.int64),
            layout.dof_addresses[lower_joint_indices],
        )
    )
    foot_position_rows: list[np.ndarray] = []
    foot_position_residuals: list[np.ndarray] = []
    secondary_rows: list[np.ndarray] = []
    secondary_residuals: list[np.ndarray] = []
    contact_by_side = {"left": contacts[0], "right": contacts[1]}
    for task, target in zip(tasks, targets, strict=True):
        body_id = _body_id(model, task.target_body)
        multiplier = (
            config.contact_weight_multiplier
            if task.contact_side is not None and contact_by_side[task.contact_side]
            else 1.0
        )
        if task.kind == "subtree_com":
            current_position = np.asarray(
                data.subtree_com[body_id], dtype=np.float64
            ).copy()
            jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacSubtreeCom(model, data, jacobian_position, body_id)
            current_rotation = np.eye(3, dtype=np.float64)
            jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
        else:
            current_position, current_rotation = _point_pose(
                data, body_id, task.target_point
            )
            jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                current_position,
                body_id,
            )
        if task.position_weight:
            scale = math.sqrt(task.position_weight * multiplier)
            row = (
                scale
                * jacobian_position[:, variable_dofs]
                * variable_scale[None, :]
            )
            residual = scale * (target.position - current_position)
            if task.name in {"left_foot", "right_foot"}:
                foot_position_rows.append(row)
                foot_position_residuals.append(residual)
            else:
                secondary_rows.append(row)
                secondary_residuals.append(residual)
        if task.orientation_weight:
            scale = math.sqrt(task.orientation_weight * multiplier)
            secondary_rows.append(
                scale
                * jacobian_rotation[:, variable_dofs]
                * variable_scale[None, :]
            )
            secondary_residuals.append(
                scale * _rotation_error(target.rotation, current_rotation)
            )
    if not foot_position_rows or not secondary_rows:
        raise ValueError("lower/root feasibility system is missing required task rows")
    priority_systems = (
        (
            np.concatenate(foot_position_rows, axis=0),
            np.concatenate(foot_position_residuals, axis=0),
        ),
        (
            np.concatenate(secondary_rows, axis=0),
            np.concatenate(secondary_residuals, axis=0),
        ),
    )
    regularization_rows: list[np.ndarray] = []
    regularization_residuals: list[np.ndarray] = []
    identity = np.eye(len(variable_scale), dtype=np.float64)
    if config.lower_root_posture_weight:
        scale = math.sqrt(config.lower_root_posture_weight)
        regularization_rows.append(scale * identity)
        regularization_residuals.append(
            scale * (reference - current_variable) / variable_scale
        )
    if config.lower_root_smoothness_weight:
        scale = math.sqrt(config.lower_root_smoothness_weight)
        regularization_rows.append(scale * identity)
        regularization_residuals.append(
            scale * (smooth_reference - current_variable) / variable_scale
        )
    normalized_step = _hierarchical_step(
        priority_systems,
        regularization_rows,
        regularization_residuals,
        config.damping,
        1.0,
    )
    return variable_scale * normalized_step


def _project_lower_root_variables(
    desired: np.ndarray,
    layout: _ModelLayout,
    lower_joint_indices: np.ndarray,
    variable_scale: np.ndarray,
    dt: float,
    config: RetargetConfig,
    *,
    fixed_mask: np.ndarray | None = None,
    fixed_values: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    lower = np.concatenate(
        (
            np.full(3, -config.max_root_offset_m, dtype=np.float64),
            layout.lower[lower_joint_indices],
        )
    )
    upper = np.concatenate(
        (
            np.full(3, config.max_root_offset_m, dtype=np.float64),
            layout.upper[lower_joint_indices],
        )
    )
    max_velocity = np.concatenate(
        (
            np.full(3, config.max_root_offset_velocity_m_s, dtype=np.float64),
            np.full(len(lower_joint_indices), config.max_velocity_rad_s),
        )
    )
    max_acceleration = np.concatenate(
        (
            np.full(
                3, config.max_root_offset_acceleration_m_s2, dtype=np.float64
            ),
            np.full(len(lower_joint_indices), config.max_acceleration_rad_s2),
        )
    )
    if fixed_mask is not None:
        if fixed_values is None:
            raise ValueError("fixed lower/root mask requires fixed values")
        mask = np.asarray(fixed_mask, dtype=bool)
        values = np.asarray(fixed_values, dtype=np.float64)
        if mask.shape != (len(desired),):
            raise ValueError("fixed lower/root mask must have shape [frames]")
        if values.shape != desired.shape:
            raise ValueError("fixed lower/root values must match desired path")
        if not np.all(np.isfinite(values[mask])):
            raise ValueError("fixed lower/root values contain NaN or Inf")
        if np.any(values[mask] < lower[None, :] - 1.0e-10) or np.any(
            values[mask] > upper[None, :] + 1.0e-10
        ):
            raise ValueError("fixed lower/root values violate safe position bounds")
        lower_by_frame = np.broadcast_to(lower, desired.shape).copy()
        upper_by_frame = np.broadcast_to(upper, desired.shape).copy()
        lower_by_frame[mask] = values[mask]
        upper_by_frame[mask] = values[mask]
    else:
        lower_by_frame = lower
        upper_by_frame = upper
    safety_fraction = TRAJECTORY_SERIALIZATION_LIMIT_FRACTION
    initial_velocity = np.clip(
        (desired[1] - desired[0]) / dt,
        -safety_fraction * max_velocity,
        safety_fraction * max_velocity,
    )
    projection = project_nearest_trajectory(
        desired / variable_scale[None, :],
        lower_bounds=lower_by_frame / variable_scale,
        upper_bounds=upper_by_frame / variable_scale,
        dt=dt,
        max_velocity=safety_fraction * max_velocity / variable_scale,
        max_acceleration=safety_fraction * max_acceleration / variable_scale,
        initial_velocity=initial_velocity / variable_scale,
        config=TrajectoryProjectionConfig(
            max_iterations=6_000,
            check_interval=20,
            primal_tolerance=1.0e-7,
            dual_tolerance=1.0e-7,
            audit_tolerance=2.0e-7,
        ),
    )
    result = projection.projected_path * variable_scale[None, :]
    audit = audit_trajectory_constraints(
        result,
        lower_bounds=lower_by_frame,
        upper_bounds=upper_by_frame,
        dt=dt,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        initial_velocity=initial_velocity,
        tolerance=1.0e-7,
    )
    if not audit.passed:
        raise RuntimeError(
            "lower/root whole-horizon projection failed independent audit: "
            f"max_violation={audit.max_constraint_violation:.3e}"
        )
    return result, projection.iterations


def _lower_root_validity_excess(
    metrics: _LowerRootMetrics,
    baseline: _LowerRootMetrics,
    config: RetargetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    weighted_limit = baseline.weighted_error + (
        config.priority_relative_tolerance
        * np.maximum(1.0, baseline.weighted_error)
    )
    weighted_excess = np.maximum(metrics.weighted_error - weighted_limit, 0.0) / np.maximum(
        1.0, baseline.weighted_error
    )
    foot_scale = max(config.valid_max_foot_position_error_m, 1.0e-9)
    foot_excess = np.maximum(
        metrics.foot_position_error_m - config.valid_max_foot_position_error_m,
        0.0,
    ) / foot_scale
    orientation_scale = max(
        config.valid_max_foot_orientation_regression_rad, 1.0e-9
    )
    orientation_excess = np.maximum(
        metrics.foot_orientation_error_rad
        - (
            baseline.foot_orientation_error_rad
            + config.valid_max_foot_orientation_regression_rad
        ),
        0.0,
    ) / orientation_scale
    com_scale = max(config.valid_max_com_regression_m, 1.0e-9)
    com_excess = np.maximum(
        metrics.com_position_error_m
        - (baseline.com_position_error_m + config.valid_max_com_regression_m),
        0.0,
    ) / com_scale
    per_frame = (
        weighted_excess**2
        + np.sum(foot_excess**2, axis=1)
        + np.sum(orientation_excess**2, axis=1)
        + com_excess**2
    )
    return per_frame > 1.0e-18, per_frame


def _lower_root_candidate_better(
    candidate_metrics: _LowerRootMetrics,
    candidate_variables: np.ndarray,
    current_metrics: _LowerRootMetrics,
    current_variables: np.ndarray,
    baseline_metrics: _LowerRootMetrics,
    reference_variables: np.ndarray,
    variable_scale: np.ndarray,
    config: RetargetConfig,
) -> bool:
    candidate_invalid, candidate_excess = _lower_root_validity_excess(
        candidate_metrics, baseline_metrics, config
    )
    current_invalid, current_excess = _lower_root_validity_excess(
        current_metrics, baseline_metrics, config
    )
    if np.any(candidate_invalid & ~current_invalid):
        return False
    candidate_count = int(np.sum(candidate_invalid))
    current_count = int(np.sum(current_invalid))
    if candidate_count != current_count:
        return candidate_count < current_count
    candidate_violation = float(np.sum(candidate_excess))
    current_violation = float(np.sum(current_excess))
    if candidate_violation + 1.0e-10 < current_violation:
        return True
    if candidate_violation > current_violation + 1.0e-10:
        return False
    candidate_weighted = float(np.mean(candidate_metrics.weighted_error))
    current_weighted = float(np.mean(current_metrics.weighted_error))
    if candidate_weighted + config.objective_tolerance < current_weighted:
        return True
    if candidate_weighted > current_weighted + config.objective_tolerance:
        return False
    candidate_regularization = float(
        np.mean(
            ((candidate_variables - reference_variables) / variable_scale[None, :])
            ** 2
        )
    )
    current_regularization = float(
        np.mean(
            ((current_variables - reference_variables) / variable_scale[None, :])
            ** 2
        )
    )
    return candidate_regularization + config.objective_tolerance < current_regularization


def _solve_lower_root_path(
    model: mujoco.MjModel,
    layout: _ModelLayout,
    tasks: Sequence[TaskSpec],
    targets_by_frame: Sequence[Sequence[_TaskTarget]],
    source_root_pos: np.ndarray,
    root_quat: np.ndarray,
    direct: np.ndarray,
    contacts: np.ndarray,
    fps: float,
    config: RetargetConfig,
) -> _LowerRootPath:
    groups = _joint_groups(layout)
    frame_count = len(direct)
    dt = 1.0 / fps
    variable_scale = np.concatenate(
        (
            np.full(3, config.lower_root_max_step_m, dtype=np.float64),
            np.full(
                len(groups.lower_body),
                config.lower_root_joint_step_rad,
                dtype=np.float64,
            ),
        )
    )
    reference_variables = np.concatenate(
        (
            np.zeros((frame_count, 3), dtype=np.float64),
            direct[:, groups.lower_body],
        ),
        axis=1,
    )
    current_variables, projection_iterations = _project_lower_root_variables(
        reference_variables,
        layout,
        groups.lower_body,
        variable_scale,
        dt,
        config,
    )

    def configuration(variables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        root = source_root_pos + variables[:, :3]
        joints = direct.copy()
        joints[:, groups.lower_body] = variables[:, 3:]
        return root, joints

    data = mujoco.MjData(model)
    baseline_metrics = _lower_root_metrics(
        model,
        data,
        layout,
        tasks,
        targets_by_frame,
        source_root_pos,
        root_quat,
        direct,
        contacts,
        groups.lower_body,
        config,
    )
    current_root, current_joints = configuration(current_variables)
    current_metrics = _lower_root_metrics(
        model,
        data,
        layout,
        tasks,
        targets_by_frame,
        current_root,
        root_quat,
        current_joints,
        contacts,
        groups.lower_body,
        config,
    )
    baseline_invalid, _ = _lower_root_validity_excess(
        baseline_metrics, baseline_metrics, config
    )
    current_invalid, _ = _lower_root_validity_excess(
        current_metrics, baseline_metrics, config
    )
    new_seed_invalid = current_invalid & ~baseline_invalid
    if np.any(new_seed_invalid):
        frames = np.flatnonzero(new_seed_invalid).tolist()
        raise RuntimeError(
            "whole-horizon lower/root seed creates new invalid frames; "
            f"retime exact safe-clipped path first: {frames[:16]}"
        )
    accepted_step_count = 0
    completed_iterations = 0
    for iteration in range(config.lower_root_max_iterations):
        completed_iterations = iteration + 1
        current_invalid, _ = _lower_root_validity_excess(
            current_metrics, baseline_metrics, config
        )
        lower_task_signal = (
            np.any(
                current_metrics.foot_position_error_m
                > config.valid_max_foot_position_error_m,
                axis=1,
            )
            | np.any(
                current_metrics.foot_orientation_error_rad > 1.0e-3,
                axis=1,
            )
            | (
                current_metrics.com_position_error_m
                > baseline_metrics.com_position_error_m
                + config.valid_max_com_regression_m
            )
        )
        repair_core = current_invalid & lower_task_signal
        if not np.any(repair_core):
            break
        stopping_halo = max(
            2,
            int(
                math.ceil(
                    config.max_root_offset_velocity_m_s
                    / (config.max_root_offset_acceleration_m_s2 * dt)
                )
            ),
            int(
                math.ceil(
                    config.max_velocity_rad_s
                    / (config.max_acceleration_rad_s2 * dt)
                )
            ),
        )
        repair_window = repair_core.copy()
        for offset in range(1, stopping_halo + 1):
            repair_window[offset:] |= repair_core[:-offset]
            repair_window[:-offset] |= repair_core[offset:]
        proposed = current_variables.copy()
        for frame in range(frame_count):
            if not repair_core[frame]:
                continue
            if frame == 0:
                smooth_reference = current_variables[1]
            elif frame == frame_count - 1:
                smooth_reference = current_variables[-2]
            else:
                smooth_reference = 0.5 * (
                    current_variables[frame - 1] + current_variables[frame + 1]
                )
            proposed[frame] += _lower_root_frame_step(
                model,
                data,
                layout,
                tasks,
                targets_by_frame[frame],
                current_root[frame],
                root_quat[frame],
                current_joints[frame],
                (bool(contacts[frame, 0]), bool(contacts[frame, 1])),
                groups.lower_body,
                variable_scale,
                current_variables[frame],
                reference_variables[frame],
                smooth_reference,
                config,
            )
        projected, projection_count = _project_lower_root_variables(
            proposed,
            layout,
            groups.lower_body,
            variable_scale,
            dt,
            config,
            fixed_mask=~repair_window,
            fixed_values=current_variables,
        )
        projection_iterations += projection_count
        best_variables = current_variables
        best_metrics = current_metrics
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate_variables = current_variables + alpha * (
                projected - current_variables
            )
            candidate_root, candidate_joints = configuration(candidate_variables)
            candidate_metrics = _lower_root_metrics(
                model,
                data,
                layout,
                tasks,
                targets_by_frame,
                candidate_root,
                root_quat,
                candidate_joints,
                contacts,
                groups.lower_body,
                config,
            )
            if _lower_root_candidate_better(
                candidate_metrics,
                candidate_variables,
                best_metrics,
                best_variables,
                baseline_metrics,
                reference_variables,
                variable_scale,
                config,
            ):
                best_variables = candidate_variables
                best_metrics = candidate_metrics
        normalized_change = float(
            np.max(
                np.abs(best_variables - current_variables)
                / variable_scale[None, :]
            )
        )
        if best_variables is current_variables:
            break
        current_variables = best_variables
        current_metrics = best_metrics
        current_root, current_joints = configuration(current_variables)
        accepted_step_count += 1
        if normalized_change <= config.step_tolerance_rad:
            break
    return _LowerRootPath(
        root_offset_w=current_variables[:, :3].copy(),
        joint_pos_hardware=current_joints,
        metrics_before=baseline_metrics,
        metrics_after=current_metrics,
        solver_iterations=completed_iterations,
        accepted_step_count=accepted_step_count,
        projection_iteration_count=projection_iterations,
    )


def _rotation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """World-axis rotation taking ``current`` to ``desired``."""

    return transform.Rotation.from_matrix(desired @ current.T).as_rotvec()


def _task_linearization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout: _ModelLayout,
    tasks: Sequence[TaskSpec],
    targets: Sequence[_TaskTarget],
    contacts: tuple[bool, bool],
    contact_weight_multiplier: float,
    active_joint_indices: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[tuple[np.ndarray, np.ndarray], ...],
]:
    rows: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    priority_count = (
        max(
            max(
                task.priority,
                task.orientation_priority
                if task.orientation_priority is not None
                else task.priority,
            )
            for task in tasks
        )
        + 1
    )
    priority_errors = np.zeros(priority_count)
    priority_rows: list[list[np.ndarray]] = [[] for _ in range(priority_count)]
    priority_residuals: list[list[np.ndarray]] = [
        [] for _ in range(priority_count)
    ]
    contact_by_side = {"left": contacts[0], "right": contacts[1]}
    for task, target in zip(tasks, targets, strict=True):
        body_id = _body_id(model, task.target_body)
        multiplier = (
            contact_weight_multiplier
            if task.contact_side is not None and contact_by_side[task.contact_side]
            else 1.0
        )
        if task.kind == "subtree_com":
            current_position = np.asarray(
                data.subtree_com[body_id], dtype=np.float64
            ).copy()
            jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacSubtreeCom(model, data, jacobian_position, body_id)
            current_rotation = np.eye(3, dtype=np.float64)
            jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
        else:
            current_position, current_rotation = _point_pose(
                data, body_id, task.target_point
            )
            jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                current_position,
                body_id,
            )
        position_error = target.position - current_position
        position_errors.append(float(np.linalg.norm(position_error)))
        priority_errors[task.priority] += (
            task.position_weight * multiplier * float(np.dot(position_error, position_error))
        )
        if task.position_weight:
            scale = math.sqrt(task.position_weight * multiplier)
            row = (
                scale
                * jacobian_position[
                    :, layout.dof_addresses[active_joint_indices]
                ]
            )
            scaled_error = scale * position_error
            rows.append(row)
            residuals.append(scaled_error)
            priority_rows[task.priority].append(row)
            priority_residuals[task.priority].append(scaled_error)
        if task.orientation_weight:
            orientation_error = _rotation_error(target.rotation, current_rotation)
            orientation_errors.append(float(np.linalg.norm(orientation_error)))
            orientation_priority = (
                task.orientation_priority
                if task.orientation_priority is not None
                else task.priority
            )
            priority_errors[orientation_priority] += (
                task.orientation_weight
                * multiplier
                * float(np.dot(orientation_error, orientation_error))
            )
            scale = math.sqrt(task.orientation_weight * multiplier)
            row = (
                scale
                * jacobian_rotation[
                    :, layout.dof_addresses[active_joint_indices]
                ]
            )
            scaled_error = scale * orientation_error
            rows.append(row)
            residuals.append(scaled_error)
            priority_rows[orientation_priority].append(row)
            priority_residuals[orientation_priority].append(scaled_error)
        else:
            orientation_errors.append(0.0)
    return (
        np.concatenate(rows, axis=0),
        np.concatenate(residuals, axis=0),
        np.asarray(position_errors, dtype=np.float64),
        np.asarray(orientation_errors, dtype=np.float64),
        priority_errors,
        tuple(
            (
                (
                    np.concatenate(tier_rows, axis=0)
                    if tier_rows
                    else np.zeros(
                        (0, len(active_joint_indices)), dtype=np.float64
                    )
                ),
                (
                    np.concatenate(tier_residuals, axis=0)
                    if tier_residuals
                    else np.zeros(0, dtype=np.float64)
                ),
            )
            for tier_rows, tier_residuals in zip(
                priority_rows, priority_residuals, strict=True
            )
        ),
    )


def _weighted_task_error(jacobian: np.ndarray, residual: np.ndarray) -> float:
    del jacobian
    return float(np.dot(residual, residual))


def _damped_pseudoinverse(value: np.ndarray, damping: float) -> np.ndarray:
    if not np.any(value):
        return np.zeros((value.shape[1], value.shape[0]), dtype=np.float64)
    left, singular, right = np.linalg.svd(value, full_matrices=False)
    inverse = singular / (singular**2 + damping**2)
    return (right.T * inverse) @ left.T


def _hierarchical_step(
    priority_systems: Sequence[tuple[np.ndarray, np.ndarray]],
    regularization_rows: Sequence[np.ndarray],
    regularization_residuals: Sequence[np.ndarray],
    damping: float,
    max_abs_step: float,
) -> np.ndarray:
    """Solve task tiers in null spaces of every higher-priority tier."""

    variable_count = priority_systems[0][0].shape[1]
    step = np.zeros(variable_count, dtype=np.float64)
    null_space = np.eye(variable_count, dtype=np.float64)

    def add_bounded(increment: np.ndarray) -> None:
        nonlocal step
        alpha = 1.0
        for current_value, increment_value in zip(step, increment, strict=True):
            if increment_value > 0.0:
                alpha = min(
                    alpha,
                    (max_abs_step - current_value) / increment_value,
                )
            elif increment_value < 0.0:
                alpha = min(
                    alpha,
                    (-max_abs_step - current_value) / increment_value,
                )
        step += max(0.0, alpha) * increment

    for jacobian, residual in priority_systems:
        effective = jacobian @ null_space
        if not np.any(np.abs(effective) > 1.0e-12):
            continue
        right_hand_side = residual - jacobian @ step
        damped_inverse = _damped_pseudoinverse(effective, damping)
        add_bounded(null_space @ (damped_inverse @ right_hand_side))
        projector_inverse = np.linalg.pinv(effective, rcond=1.0e-7)
        null_space = null_space @ (
            np.eye(variable_count, dtype=np.float64)
            - projector_inverse @ effective
        )

    if regularization_rows:
        jacobian = np.concatenate(regularization_rows, axis=0)
        residual = np.concatenate(regularization_residuals, axis=0)
        effective = jacobian @ null_space
        if np.any(np.abs(effective) > 1.0e-12):
            right_hand_side = residual - jacobian @ step
            add_bounded(
                null_space
                @ (_damped_pseudoinverse(effective, damping) @ right_hand_side)
            )
    return step


def _weighted_step(
    task_jacobian: np.ndarray,
    task_residual: np.ndarray,
    regularization_rows: Sequence[np.ndarray],
    regularization_residuals: Sequence[np.ndarray],
    damping: float,
    max_abs_step: float,
) -> np.ndarray:
    rows = [task_jacobian, *regularization_rows]
    residuals = [task_residual, *regularization_residuals]
    system = np.concatenate(rows, axis=0)
    target = np.concatenate(residuals, axis=0)
    normal = system.T @ system
    normal.flat[:: normal.shape[0] + 1] += damping**2
    step = np.linalg.solve(normal, system.T @ target)
    maximum = float(np.max(np.abs(step)))
    if maximum > max_abs_step:
        step *= max_abs_step / maximum
    return step


def _protected_priorities_not_worse(
    candidate: np.ndarray,
    reference: np.ndarray,
    protected_tiers: int,
    relative_tolerance: float,
) -> bool:
    if protected_tiers > len(reference):
        raise ValueError("protected task-tier count exceeds configured priorities")
    tolerance = relative_tolerance * np.maximum(
        1.0, np.abs(reference[:protected_tiers])
    )
    return bool(
        np.all(candidate[:protected_tiers] <= reference[:protected_tiers] + tolerance)
    )


def _trajectory_bounds(
    layout: _ModelLayout,
    previous: np.ndarray | None,
    previous_velocity: np.ndarray | None,
    dt: float,
    config: RetargetConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    lower = layout.lower.copy()
    upper = layout.upper.copy()
    if previous is None:
        return lower, upper, 0

    velocity_step = config.max_velocity_rad_s * dt
    lower = np.maximum(lower, previous - velocity_step)
    upper = np.minimum(upper, previous + velocity_step)
    # Keep the newly implied velocity inside the one-dimensional braking
    # viability envelope. Without this, a myopic frame can reach a joint limit
    # too fast for any acceleration-bounded next frame to remain feasible.
    acceleration_dt2 = config.max_acceleration_rad_s2 * dt * dt
    distance_to_lower = np.maximum(previous - layout.lower, 0.0)
    distance_to_upper = np.maximum(layout.upper - previous, 0.0)
    lower_displacement = -acceleration_dt2 + np.sqrt(
        acceleration_dt2**2 + 2.0 * acceleration_dt2 * distance_to_lower
    )
    upper_displacement = -acceleration_dt2 + np.sqrt(
        acceleration_dt2**2 + 2.0 * acceleration_dt2 * distance_to_upper
    )
    lower = np.maximum(lower, previous - lower_displacement)
    upper = np.minimum(upper, previous + upper_displacement)
    relaxations = 0
    if previous_velocity is not None:
        acceleration_step = config.max_acceleration_rad_s2 * dt * dt
        acceleration_lower = previous + previous_velocity * dt - acceleration_step
        acceleration_upper = previous + previous_velocity * dt + acceleration_step
        proposed_lower = np.maximum(lower, acceleration_lower)
        proposed_upper = np.minimum(upper, acceleration_upper)
        impossible = proposed_lower > proposed_upper
        relaxations = int(np.sum(impossible))
        if relaxations and not config.allow_acceleration_constraint_relaxation:
            raise RuntimeError(
                "acceleration constraint became infeasible; refusing to relax "
                f"{relaxations} joint bounds"
            )
        proposed_lower[impossible] = lower[impossible]
        proposed_upper[impossible] = upper[impossible]
        lower, upper = proposed_lower, proposed_upper
    if np.any(lower > upper):
        raise RuntimeError("position/velocity trajectory constraints are infeasible")
    return lower, upper, relaxations


def _solve_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout: _ModelLayout,
    tasks: Sequence[TaskSpec],
    targets: Sequence[_TaskTarget],
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    direct: np.ndarray,
    previous: np.ndarray | None,
    contacts: tuple[bool, bool],
    lower: np.ndarray,
    upper: np.ndarray,
    config: RetargetConfig,
    *,
    seed: np.ndarray | None = None,
    fallback_seed: np.ndarray | None = None,
    active_joint_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if active_joint_indices is None:
        active_joint_indices = np.asarray(
            [
                index
                for index, name in enumerate(layout.joint_names)
                if config.optimize_lower_body
                or (
                    not (
                        name.startswith("left_hip_")
                        or name.startswith("right_hip_")
                    )
                    and "knee" not in name
                    and "ankle" not in name
                )
            ],
            dtype=np.int64,
        )
    else:
        active_joint_indices = np.asarray(active_joint_indices, dtype=np.int64)
        if active_joint_indices.ndim != 1:
            raise ValueError("active_joint_indices must be one-dimensional")
        if np.any(active_joint_indices < 0) or np.any(
            active_joint_indices >= len(layout.joint_names)
        ):
            raise ValueError("active_joint_indices contains an out-of-range joint")
        if len(np.unique(active_joint_indices)) != len(active_joint_indices):
            raise ValueError("active_joint_indices must not contain duplicates")
    if not len(active_joint_indices):
        raise ValueError("retarget solver has no active joints")

    def bounded_configuration(
        value: np.ndarray | None,
        *,
        default: np.ndarray,
        name: str,
    ) -> np.ndarray:
        candidate = np.asarray(default if value is None else value, dtype=np.float64)
        if candidate.shape != direct.shape:
            raise ValueError(f"{name} must have shape {direct.shape}")
        if not np.all(np.isfinite(candidate)):
            raise ValueError(f"{name} contains NaN or Inf")
        return np.clip(candidate, lower, upper)

    feasible_seed = bounded_configuration(seed, default=direct, name="seed")
    fallback = bounded_configuration(
        fallback_seed,
        default=feasible_seed,
        name="fallback_seed",
    )
    inactive_joint_indices = np.setdiff1d(
        np.arange(len(layout.joint_names), dtype=np.int64),
        active_joint_indices,
        assume_unique=True,
    )
    # Inactive joints are owned by the preceding solver stage. A fallback may
    # replace active values, but it must never silently discard that frozen path.
    fallback[inactive_joint_indices] = feasible_seed[inactive_joint_indices]
    baseline = np.clip(direct, layout.lower, layout.upper)
    _set_configuration(model, data, layout, root_pos, root_quat, baseline)
    (
        baseline_jacobian,
        baseline_residual,
        position_before,
        orientation_before,
        priority_before,
        _,
    ) = _task_linearization(
        model,
        data,
        layout,
        tasks,
        targets,
        contacts,
        config.contact_weight_multiplier,
        active_joint_indices,
    )
    before = _weighted_task_error(baseline_jacobian, baseline_residual)

    current = feasible_seed.copy()
    _set_configuration(model, data, layout, root_pos, root_quat, fallback)
    (
        fallback_jacobian,
        fallback_residual,
        _,
        _,
        fallback_priority,
        _,
    ) = _task_linearization(
        model,
        data,
        layout,
        tasks,
        targets,
        contacts,
        config.contact_weight_multiplier,
        active_joint_indices,
    )
    fallback_task_error = _weighted_task_error(
        fallback_jacobian, fallback_residual
    )

    def objective(candidate: np.ndarray) -> tuple[float, float, np.ndarray]:
        _set_configuration(model, data, layout, root_pos, root_quat, candidate)
        local_jacobian, local_residual, _, _, priority_errors, _ = (
            _task_linearization(
                model,
                data,
                layout,
                tasks,
                targets,
                contacts,
                config.contact_weight_multiplier,
                active_joint_indices,
            )
        )
        task_error = _weighted_task_error(local_jacobian, local_residual)
        result = task_error + config.posture_weight * float(
            np.sum(
                (
                    candidate[active_joint_indices]
                    - direct[active_joint_indices]
                )
                ** 2
            )
        )
        if previous is not None:
            result += config.smoothness_weight * float(
                np.sum(
                    (
                        candidate[active_joint_indices]
                        - previous[active_joint_indices]
                    )
                    ** 2
                )
            )
        return result, task_error, priority_errors

    current_objective, current_task_error, current_priority_errors = objective(current)
    iterations = 0
    accepted_step_count = 0
    for iteration in range(config.max_iterations):
        iterations = iteration + 1
        _set_configuration(model, data, layout, root_pos, root_quat, current)
        jacobian, residual, _, _, _, priority_systems = _task_linearization(
            model,
            data,
            layout,
            tasks,
            targets,
            contacts,
            config.contact_weight_multiplier,
            active_joint_indices,
        )
        regularization_rows: list[np.ndarray] = []
        regularization_residuals: list[np.ndarray] = []
        identity = np.eye(len(active_joint_indices), dtype=np.float64)
        if config.posture_weight:
            scale = math.sqrt(config.posture_weight)
            regularization_rows.append(scale * identity)
            regularization_residuals.append(
                scale * (direct[active_joint_indices] - current[active_joint_indices])
            )
        if previous is not None and config.smoothness_weight:
            scale = math.sqrt(config.smoothness_weight)
            regularization_rows.append(scale * identity)
            regularization_residuals.append(
                scale
                * (
                    previous[active_joint_indices]
                    - current[active_joint_indices]
                )
            )
        directions = (
            _hierarchical_step(
                priority_systems,
                regularization_rows,
                regularization_residuals,
                config.damping,
                config.max_iteration_step_rad,
            ),
            _weighted_step(
                jacobian,
                residual,
                regularization_rows,
                regularization_residuals,
                config.damping,
                config.max_iteration_step_rad,
            ),
        )
        best: tuple[np.ndarray, float, float, np.ndarray] | None = None
        for step in directions:
            if float(np.max(np.abs(step))) <= config.step_tolerance_rad:
                continue
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = current.copy()
                candidate[active_joint_indices] += alpha * step
                candidate = np.clip(candidate, lower, upper)
                (
                    candidate_objective,
                    candidate_task_error,
                    candidate_priority_errors,
                ) = objective(candidate)
                task_non_regression = (
                    candidate_task_error
                    <= current_task_error
                    + config.priority_relative_tolerance
                    * max(1.0, current_task_error)
                )
                protected_non_regression = _protected_priorities_not_worse(
                    candidate_priority_errors,
                    current_priority_errors,
                    config.protected_priority_tiers,
                    config.priority_relative_tolerance,
                )
                if (
                    task_non_regression
                    and protected_non_regression
                    and candidate_objective + config.objective_tolerance
                    < current_objective
                    and (best is None or candidate_objective < best[1])
                ):
                    best = (
                        candidate,
                        candidate_objective,
                        candidate_task_error,
                        candidate_priority_errors,
                    )
        if best is None:
            break
        current, current_objective, current_task_error, current_priority_errors = best
        accepted_step_count += 1

    _set_configuration(model, data, layout, root_pos, root_quat, current)
    (
        jacobian,
        residual,
        position_after,
        orientation_after,
        priority_after,
        _,
    ) = _task_linearization(
            model,
            data,
            layout,
            tasks,
            targets,
            contacts,
            config.contact_weight_multiplier,
            active_joint_indices,
        )
    after = _weighted_task_error(jacobian, residual)
    # The solver is an expert only when it is at least as good as the selected
    # feasible fallback under both aggregate and protected task metrics.
    used_feasible_seed_fallback = False
    if (
        after
        > fallback_task_error
        + config.priority_relative_tolerance * max(1.0, fallback_task_error)
        or not _protected_priorities_not_worse(
            priority_after,
            fallback_priority,
            config.protected_priority_tiers,
            config.priority_relative_tolerance,
        )
    ):
        used_feasible_seed_fallback = True
        current = fallback.copy()
        _set_configuration(model, data, layout, root_pos, root_quat, current)
        (
            jacobian,
            residual,
            position_after,
            orientation_after,
            priority_after,
            _,
        ) = _task_linearization(
                model,
                data,
                layout,
                tasks,
                targets,
                contacts,
                config.contact_weight_multiplier,
                active_joint_indices,
            )
        after = _weighted_task_error(jacobian, residual)
    finite_lower = np.isfinite(layout.lower)
    finite_upper = np.isfinite(layout.upper)
    limit_hits = np.sum(
        (finite_lower & np.isclose(current, layout.lower, rtol=0.0, atol=1.0e-6))
        | (
            finite_upper
            & np.isclose(current, layout.upper, rtol=0.0, atol=1.0e-6)
        )
    )
    diagnostics = {
        "weighted_task_error_before": before,
        "weighted_task_error_feasible_seed": fallback_task_error,
        "weighted_task_error_after": after,
        "position_error_before_mean": float(np.mean(position_before)),
        "position_error_after_mean": float(np.mean(position_after)),
        "orientation_error_before_mean": float(np.mean(orientation_before)),
        "orientation_error_after_mean": float(np.mean(orientation_after)),
        "solver_iterations": float(iterations),
        "solver_accepted_step_count": float(accepted_step_count),
        "solver_used_feasible_seed_fallback": float(used_feasible_seed_fallback),
        "position_limit_hit_count": float(limit_hits),
    }
    for priority, (before_value, after_value) in enumerate(
        zip(priority_before, priority_after, strict=True)
    ):
        diagnostics[f"priority_{priority}_error_before"] = float(before_value)
        diagnostics[f"priority_{priority}_error_after"] = float(after_value)
        diagnostics[f"priority_{priority}_error_feasible_seed"] = float(
            fallback_priority[priority]
        )
    for index, task in enumerate(tasks):
        diagnostics[f"task_{task.name}_position_error_before_m"] = float(
            position_before[index]
        )
        diagnostics[f"task_{task.name}_position_error_after_m"] = float(
            position_after[index]
        )
        diagnostics[f"task_{task.name}_orientation_error_before_rad"] = float(
            orientation_before[index]
        )
        diagnostics[f"task_{task.name}_orientation_error_after_rad"] = float(
            orientation_after[index]
        )
    return current, diagnostics


def _task_pose_arrays(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tasks: Sequence[TaskSpec],
    source: bool,
) -> tuple[np.ndarray, np.ndarray]:
    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    for task in tasks:
        body_name = task.source_body if source else task.target_body
        point = task.source_point if source else task.target_point
        body_id = _body_id(model, body_name)
        if task.kind == "subtree_com":
            position = np.asarray(data.subtree_com[body_id], dtype=np.float64).copy()
            rotation = np.eye(3, dtype=np.float64)
        else:
            position, rotation = _point_pose(data, body_id, point)
        quaternion_xyzw = transform.Rotation.from_matrix(rotation).as_quat()
        positions.append(position)
        quaternions.append(quaternion_xyzw[[3, 0, 1, 2]])
    return np.stack(positions), np.stack(quaternions)


def _source_foot_positions(
    model: mujoco.MjModel,
    layout: _ModelLayout,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joints: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    result = np.empty((len(joints), 2, 3), dtype=np.float64)
    body_ids = (
        _body_id(model, "left_ankle_roll_link"),
        _body_id(model, "right_ankle_roll_link"),
    )
    for frame in range(len(joints)):
        _set_configuration(
            model, data, layout, root_pos[frame], root_quat[frame], joints[frame]
        )
        for side, body_id in enumerate(body_ids):
            result[frame, side] = data.xpos[body_id]
    return result


def infer_foot_contacts(
    foot_positions_w: np.ndarray,
    *,
    fps: float,
    height_tolerance_m: float = 0.035,
    speed_tolerance_m_s: float = 0.45,
) -> np.ndarray:
    """Infer a conservative two-foot contact mask from a kinematic clip."""

    positions = np.asarray(foot_positions_w, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (2, 3):
        raise ValueError("foot_positions_w must have shape [frames, 2, 3]")
    if fps <= 0 or len(positions) < 2:
        raise ValueError("contact inference needs at least two frames and positive fps")
    ground = float(np.quantile(positions[:, :, 2], 0.02))
    speed = np.linalg.norm(_finite_difference(positions, 1.0 / fps), axis=2)
    return (positions[:, :, 2] <= ground + height_tolerance_m) & (
        speed <= speed_tolerance_m_s
    )


def _hardware_targets_to_raw_native(targets_hardware: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets_hardware, dtype=np.float64)
    if targets.shape[-1] != TARGET_DOF:
        raise ValueError("hardware targets must end in 23 joints")
    default_q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float64)
    delta = targets - default_q
    positive_capacity = np.asarray(
        SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE, dtype=np.float64
    )
    negative_capacity = np.asarray(
        SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE, dtype=np.float64
    )
    capacity = np.where(delta >= 0.0, positive_capacity, negative_capacity)
    ratio = delta / capacity
    if np.any(np.abs(ratio) >= 1.0):
        raise RuntimeError("hardware target cannot be inverted through safe-target tanh")
    raw_hardware_delta = capacity * np.arctanh(ratio)
    hardware_action = raw_hardware_delta / np.asarray(
        HARDWARE_23_ACTION_SCALE, dtype=np.float64
    )
    return hardware_action[..., np.asarray(MUJOCO_TO_ISAACLAB_DOF)]


def retarget_trajectory(
    *,
    source_model: mujoco.MjModel,
    target_model: mujoco.MjModel,
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    source_joint_pos_hardware: np.ndarray,
    fps: float,
    tasks: Sequence[TaskSpec] = DEFAULT_TASKS,
    config: RetargetConfig = RetargetConfig(),
    contact_flags: np.ndarray | None = None,
) -> RetargetResult:
    """Retarget a 29-DoF kinematic trajectory onto the constrained 23-DoF body."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    source_layout = _model_layout(source_model)
    target_layout = _safe_target_layout(
        _model_layout(target_model),
        config.safe_limit_guard_rad,
        config.native_action_clip,
    )
    task_tuple = validate_tasks(source_model, target_model, tasks)
    root_pos = np.asarray(root_pos_w, dtype=np.float64)
    root_quat = _normalize_root_quaternions(root_quat_wxyz)
    source_joints = np.asarray(source_joint_pos_hardware, dtype=np.float64)
    frame_count = len(source_joints)
    if root_pos.shape != (frame_count, 3):
        raise ValueError("root_pos_w must have shape [frames, 3]")
    if source_joints.shape != (frame_count, len(source_layout.joint_names)):
        raise ValueError(
            "source_joint_pos_hardware shape does not match source model: "
            f"{source_joints.shape}"
        )
    if frame_count < 2:
        raise ValueError("retargeting needs at least two frames")
    if not all(
        np.all(np.isfinite(value)) for value in (root_pos, root_quat, source_joints)
    ):
        raise ValueError("source trajectory contains NaN or Inf")

    source_index = {name: index for index, name in enumerate(source_layout.joint_names)}
    direct = np.stack(
        [source_joints[:, source_index[name]] for name in target_layout.joint_names],
        axis=1,
    )
    direct = np.clip(direct, target_layout.lower, target_layout.upper)

    if contact_flags is None:
        feet = _source_foot_positions(
            source_model, source_layout, root_pos, root_quat, source_joints
        )
        contacts = infer_foot_contacts(
            feet,
            fps=fps,
            height_tolerance_m=config.contact_height_tolerance_m,
            speed_tolerance_m_s=config.contact_speed_tolerance_m_s,
        )
    else:
        contacts = np.asarray(contact_flags, dtype=bool)
        if contacts.shape != (frame_count, 2):
            raise ValueError("contact_flags must have shape [frames, 2]")

    source_data = mujoco.MjData(source_model)
    target_data = mujoco.MjData(target_model)
    result = np.empty_like(direct)
    desired_positions = np.empty((frame_count, len(task_tuple), 3), dtype=np.float64)
    desired_quaternions = np.empty((frame_count, len(task_tuple), 4), dtype=np.float64)
    targets_by_frame: list[tuple[_TaskTarget, ...]] = []
    for frame in range(frame_count):
        _set_configuration(
            source_model,
            source_data,
            source_layout,
            root_pos[frame],
            root_quat[frame],
            source_joints[frame],
        )
        targets_by_frame.append(_task_targets(source_model, source_data, task_tuple))
        desired_positions[frame], desired_quaternions[frame] = _task_pose_arrays(
            source_model, source_data, task_tuple, source=True
        )

    groups = _joint_groups(target_layout)
    if config.enable_lower_root_feasibility:
        lower_root_path = _solve_lower_root_path(
            target_model,
            target_layout,
            task_tuple,
            targets_by_frame,
            root_pos,
            root_quat,
            direct,
            contacts,
            fps,
            config,
        )
        adapted_root_pos = root_pos + lower_root_path.root_offset_w
    else:
        baseline_metrics = _lower_root_metrics(
            target_model,
            target_data,
            target_layout,
            task_tuple,
            targets_by_frame,
            root_pos,
            root_quat,
            direct,
            contacts,
            groups.lower_body,
            config,
        )
        lower_root_path = _LowerRootPath(
            root_offset_w=np.zeros_like(root_pos),
            joint_pos_hardware=direct.copy(),
            metrics_before=baseline_metrics,
            metrics_after=baseline_metrics,
            solver_iterations=0,
            accepted_step_count=0,
            projection_iteration_count=0,
        )
        adapted_root_pos = root_pos.copy()

    achieved_positions = np.empty_like(desired_positions)
    achieved_quaternions = np.empty_like(desired_quaternions)
    diagnostic_lists: dict[str, list[float]] = {
        "weighted_task_error_before": [],
        "weighted_task_error_after": [],
        "position_error_before_mean": [],
        "position_error_after_mean": [],
        "orientation_error_before_mean": [],
        "orientation_error_after_mean": [],
        "solver_iterations": [],
        "solver_accepted_step_count": [],
        "solver_used_feasible_seed_fallback": [],
        "position_limit_hit_count": [],
        "constraint_relaxation_count": [],
    }
    previous: np.ndarray | None = None
    previous_velocity: np.ndarray | None = None
    dt = 1.0 / fps
    trajectory_constraint_config = replace(
        config,
        max_velocity_rad_s=(
            TRAJECTORY_SERIALIZATION_LIMIT_FRACTION * config.max_velocity_rad_s
        ),
        max_acceleration_rad_s2=(
            TRAJECTORY_SERIALIZATION_LIMIT_FRACTION
            * config.max_acceleration_rad_s2
        ),
    )
    trajectory_layout = target_layout
    if config.enable_lower_root_feasibility:
        trajectory_lower = target_layout.lower.copy()
        trajectory_upper = target_layout.upper.copy()
        trajectory_lower[groups.lower_body] = -np.inf
        trajectory_upper[groups.lower_body] = np.inf
        trajectory_layout = _ModelLayout(
            joint_names=target_layout.joint_names,
            joint_ids=target_layout.joint_ids,
            qpos_addresses=target_layout.qpos_addresses,
            dof_addresses=target_layout.dof_addresses,
            lower=trajectory_lower,
            upper=trajectory_upper,
        )
    for frame in range(frame_count):
        targets = targets_by_frame[frame]
        lower, upper, relaxations = _trajectory_bounds(
            trajectory_layout,
            previous,
            previous_velocity,
            dt,
            trajectory_constraint_config,
        )
        seed: np.ndarray | None = None
        active_joint_indices: np.ndarray | None = None
        if config.enable_lower_root_feasibility:
            frozen_lower = lower_root_path.joint_pos_hardware[
                frame, groups.lower_body
            ]
            if np.any(
                frozen_lower < target_layout.lower[groups.lower_body] - 1.0e-9
            ) or np.any(
                frozen_lower > target_layout.upper[groups.lower_body] + 1.0e-9
            ):
                raise RuntimeError("lower/root stage produced an out-of-range joint")
            lower[groups.lower_body] = frozen_lower
            upper[groups.lower_body] = frozen_lower
            seed = np.clip(direct[frame], lower, upper)
            seed[groups.lower_body] = frozen_lower
            active_joint_indices = groups.upper_body
        solved, diagnostics = _solve_frame(
            target_model,
            target_data,
            target_layout,
            task_tuple,
            targets,
            adapted_root_pos[frame],
            root_quat[frame],
            direct[frame],
            previous,
            (bool(contacts[frame, 0]), bool(contacts[frame, 1])),
            lower,
            upper,
            config,
            seed=seed,
            fallback_seed=seed,
            active_joint_indices=active_joint_indices,
        )

        _set_configuration(
            target_model,
            target_data,
            target_layout,
            root_pos[frame],
            root_quat[frame],
            direct[frame],
        )
        (
            baseline_jacobian,
            baseline_residual,
            baseline_position,
            baseline_orientation,
            baseline_priority,
            _,
        ) = _task_linearization(
            target_model,
            target_data,
            target_layout,
            task_tuple,
            targets,
            (bool(contacts[frame, 0]), bool(contacts[frame, 1])),
            config.contact_weight_multiplier,
            groups.upper_body,
        )
        diagnostics["weighted_task_error_before"] = _weighted_task_error(
            baseline_jacobian, baseline_residual
        )
        diagnostics["position_error_before_mean"] = float(
            np.mean(baseline_position)
        )
        diagnostics["orientation_error_before_mean"] = float(
            np.mean(baseline_orientation)
        )
        for priority, value in enumerate(baseline_priority):
            diagnostics[f"priority_{priority}_error_before"] = float(value)
        for index, task in enumerate(task_tuple):
            diagnostics[f"task_{task.name}_position_error_before_m"] = float(
                baseline_position[index]
            )
            diagnostics[
                f"task_{task.name}_orientation_error_before_rad"
            ] = float(baseline_orientation[index])
        result[frame] = solved
        _set_configuration(
            target_model,
            target_data,
            target_layout,
            adapted_root_pos[frame],
            root_quat[frame],
            solved,
        )
        achieved_positions[frame], achieved_quaternions[frame] = _task_pose_arrays(
            target_model, target_data, task_tuple, source=False
        )
        diagnostics["constraint_relaxation_count"] = float(relaxations)
        for name, value in diagnostics.items():
            diagnostic_lists.setdefault(name, []).append(value)
        velocity = (
            None if previous is None else (solved - previous) / dt
        )
        previous = solved.copy()
        previous_velocity = velocity

    trajectory_velocity, trajectory_acceleration = trajectory_derivatives(
        result, dt
    )
    velocity_max = np.max(np.abs(trajectory_velocity), axis=1)
    acceleration_max = np.max(np.abs(trajectory_acceleration), axis=1)
    if np.any(velocity_max > config.max_velocity_rad_s + 1.0e-8):
        raise RuntimeError("retarget result violates configured joint velocity limit")
    if (
        not config.allow_acceleration_constraint_relaxation
        and np.any(acceleration_max > config.max_acceleration_rad_s2 + 1.0e-6)
    ):
        raise RuntimeError("retarget result violates configured joint acceleration limit")
    diagnostic_lists["trajectory_velocity_abs_max"] = velocity_max.tolist()
    diagnostic_lists["trajectory_acceleration_abs_max"] = acceleration_max.tolist()

    root_offset_velocity, root_offset_acceleration = trajectory_derivatives(
        lower_root_path.root_offset_w,
        dt,
    )
    root_velocity_max = np.max(np.abs(root_offset_velocity), axis=1)
    root_acceleration_max = np.max(np.abs(root_offset_acceleration), axis=1)
    if np.any(
        root_velocity_max > config.max_root_offset_velocity_m_s + 1.0e-7
    ):
        raise RuntimeError("retarget result violates root-offset velocity limit")
    if np.any(
        root_acceleration_max
        > config.max_root_offset_acceleration_m_s2 + 1.0e-6
    ):
        raise RuntimeError("retarget result violates root-offset acceleration limit")
    before_invalid, before_excess = _lower_root_validity_excess(
        lower_root_path.metrics_before,
        lower_root_path.metrics_before,
        config,
    )
    after_invalid, after_excess = _lower_root_validity_excess(
        lower_root_path.metrics_after,
        lower_root_path.metrics_before,
        config,
    )
    diagnostic_lists.update(
        {
            "root_offset_velocity_abs_max": root_velocity_max.tolist(),
            "root_offset_acceleration_abs_max": root_acceleration_max.tolist(),
            "lower_root_solver_iterations": np.full(
                frame_count,
                lower_root_path.solver_iterations,
                dtype=np.float64,
            ).tolist(),
            "lower_root_accepted_step_count": np.full(
                frame_count,
                lower_root_path.accepted_step_count,
                dtype=np.float64,
            ).tolist(),
            "lower_root_projection_iteration_count": np.full(
                frame_count,
                lower_root_path.projection_iteration_count,
                dtype=np.float64,
            ).tolist(),
            "lower_root_invalid_before": before_invalid.astype(np.float64).tolist(),
            "lower_root_invalid_after": after_invalid.astype(np.float64).tolist(),
            "lower_root_validity_excess_before": before_excess.tolist(),
            "lower_root_validity_excess_after": after_excess.tolist(),
        }
    )

    native_action = _hardware_targets_to_raw_native(result)
    direct_native_action = _hardware_targets_to_raw_native(direct)
    return RetargetResult(
        source_root_pos_w=root_pos,
        root_pos_w=adapted_root_pos,
        root_offset_w=lower_root_path.root_offset_w,
        root_quat_wxyz=root_quat,
        direct_joint_pos_hardware=direct,
        joint_pos_hardware=result,
        direct_action_native=direct_native_action,
        action_target_native=native_action,
        contact_flags=contacts,
        desired_task_pos_w=desired_positions,
        achieved_task_pos_w=achieved_positions,
        desired_task_quat_wxyz=desired_quaternions,
        achieved_task_quat_wxyz=achieved_quaternions,
        task_has_orientation=np.asarray(
            [task.orientation_weight > 0.0 for task in task_tuple], dtype=bool
        ),
        task_names=tuple(task.name for task in task_tuple),
        diagnostics={
            name: np.asarray(values, dtype=np.float64)
            for name, values in diagnostic_lists.items()
        },
        fps=float(fps),
        config=config,
    )


def trajectory_derivatives(
    values: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interval velocity and second difference with a free initial velocity."""

    path = np.asarray(values, dtype=np.float64)
    if path.ndim != 2 or len(path) < 2:
        raise ValueError("trajectory derivatives require shape [frames>=2, dims]")
    interval_velocity = np.diff(path, axis=0) / dt
    velocity = np.vstack((interval_velocity[:1], interval_velocity))
    acceleration = np.vstack(
        (
            np.zeros((1, path.shape[1]), dtype=np.float64),
            np.diff(velocity, axis=0) / dt,
        )
    )
    return velocity, acceleration


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(np.asarray(values, dtype=np.float64), dt, axis=0, edge_order=1)


def _angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    rotations = transform.Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    velocity = np.zeros((len(rotations), 3), dtype=np.float64)
    for index in range(len(rotations)):
        lower = max(index - 1, 0)
        upper = min(index + 1, len(rotations) - 1)
        if lower == upper:
            continue
        velocity[index] = (
            rotations[upper] * rotations[lower].inv()
        ).as_rotvec() / (dt * (upper - lower))
    return velocity


def build_mjlab_motion_arrays(
    target_model: mujoco.MjModel,
    result: RetargetResult,
) -> dict[str, np.ndarray]:
    """Build the standard MJLab tracking-motion arrays from the expert path."""

    layout = _model_layout(target_model)
    if layout.joint_names != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError("motion output requires the exact rev-1.0 model")
    data = mujoco.MjData(target_model)
    frame_count = len(result.joint_pos_hardware)
    body_count = target_model.nbody - 1
    body_pos = np.empty((frame_count, body_count, 3), dtype=np.float64)
    body_quat = np.empty((frame_count, body_count, 4), dtype=np.float64)
    for frame in range(frame_count):
        _set_configuration(
            target_model,
            data,
            layout,
            result.root_pos_w[frame],
            result.root_quat_wxyz[frame],
            result.joint_pos_hardware[frame],
        )
        body_pos[frame] = data.xpos[1:]
        body_quat[frame] = data.xquat[1:]
    dt = 1.0 / result.fps
    body_ang_vel = np.stack(
        [_angular_velocity(body_quat[:, body], dt) for body in range(body_count)],
        axis=1,
    )
    stored_joint_pos = result.joint_pos_hardware.astype(np.float32)
    stored_joint_vel, _ = trajectory_derivatives(
        stored_joint_pos.astype(np.float64), dt
    )
    return {
        "fps": np.asarray([result.fps], dtype=np.float64),
        "joint_pos": stored_joint_pos,
        "joint_vel": stored_joint_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": _finite_difference(body_pos, dt).astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
    }


def build_residual_supervision(
    *,
    result: RetargetResult,
    decoder_input: np.ndarray,
    base_action_native: np.ndarray,
) -> dict[str, np.ndarray]:
    """Bind expert actions to captured student states for residual imitation.

    This function does not claim DAgger provenance.  The caller must record
    whether inputs came from teacher, mixed, or student on-policy rollouts.
    """

    inputs = np.asarray(decoder_input, dtype=np.float32)
    base = np.asarray(base_action_native, dtype=np.float32)
    frame_count = len(result.action_target_native)
    if inputs.shape != (frame_count, 994):
        raise ValueError("decoder_input must have shape [frames, 994]")
    if base.shape != (frame_count, TARGET_DOF):
        raise ValueError("base_action_native must have shape [frames, 23]")
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(base)):
        raise ValueError("residual supervision inputs contain NaN or Inf")
    expert = result.action_target_native.astype(np.float32)
    return {
        "decoder_input": inputs,
        "base_action_native": base,
        "expert_action_native": expert,
        "residual_action_native": expert - base,
        "expert_valid": result.expert_valid_mask(),
    }
