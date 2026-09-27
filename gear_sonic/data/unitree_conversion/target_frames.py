"""Build exact non-video SONIC VLA frames from one resampled Dex3 episode."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation as R

from gear_sonic.data.features_sonic_vla import get_g1_robot_model
from gear_sonic.data.unitree_conversion.contracts import ResampledEpisode
from gear_sonic.data.unitree_conversion.joint_mapping import (
    G1_MUJOCO_NAMES,
    dex3_dds_to_robot_model,
)

_LEFT_HAND_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
)
_RIGHT_HAND_NAMES = tuple(name.replace("left_", "right_", 1) for name in _LEFT_HAND_NAMES)
_EXPECTED_JOINT_NAMES = frozenset((*G1_MUJOCO_NAMES, *_LEFT_HAND_NAMES, *_RIGHT_HAND_NAMES))


def _owned_array(
    values: object,
    *,
    dtype: np.dtype | type,
    field_name: str,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(values, dtype=dtype, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values after casting to {array.dtype}")
    return array


def _validate_robot_model(robot_model: object) -> tuple[tuple[str, ...], dict[str, str]]:
    try:
        joint_names = tuple(robot_model.joint_names)
    except (AttributeError, TypeError) as error:
        raise ValueError("RobotModel must expose exactly 43 semantic joint names") from error
    if len(joint_names) != 43:
        raise ValueError(f"RobotModel must expose exactly 43 semantic joint names; got {len(joint_names)}")
    if len(set(joint_names)) != 43:
        raise ValueError("RobotModel semantic joint names must be unique")
    if set(joint_names) != _EXPECTED_JOINT_NAMES:
        missing = sorted(_EXPECTED_JOINT_NAMES - set(joint_names))
        extra = sorted(set(joint_names) - _EXPECTED_JOINT_NAMES)
        raise ValueError(f"RobotModel semantic joint set differs from G1 + Dex3: missing={missing}, extra={extra}")

    try:
        hand_frame_names = robot_model.supplemental_info.hand_frame_names
    except AttributeError as error:
        raise ValueError("RobotModel hand_frame_names must define left and right") from error
    if not isinstance(hand_frame_names, Mapping) or set(hand_frame_names) != {"left", "right"}:
        raise ValueError("RobotModel hand_frame_names must define exactly left and right")
    if any(
        not isinstance(hand_frame_names[side], str) or not hand_frame_names[side].strip()
        for side in ("left", "right")
    ):
        raise ValueError("RobotModel hand_frame_names must define nonempty left and right names")
    return joint_names, {side: hand_frame_names[side] for side in ("left", "right")}


def _semantic_configuration(
    *,
    field_name: str,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    body_values: np.ndarray,
    left_hand_dds: np.ndarray,
    right_hand_dds: np.ndarray,
) -> np.ndarray:
    semantic_values = dict(zip(body_names, body_values, strict=True))
    semantic_values.update(dex3_dds_to_robot_model("left", left_hand_dds))
    semantic_values.update(dex3_dds_to_robot_model("right", right_hand_dds))
    if set(semantic_values) != set(joint_names):
        missing = sorted(set(joint_names) - set(semantic_values))
        extra = sorted(set(semantic_values) - set(joint_names))
        raise ValueError(f"episode semantic joint set differs from RobotModel: missing={missing}, extra={extra}")
    return _owned_array(
        [semantic_values[name] for name in joint_names],
        dtype=np.float64,
        field_name=field_name,
    )


class TargetFrameBuilder:
    """Create exact schema values for one isolated 50 Hz target frame."""

    def __init__(self, robot_model: object | None = None) -> None:
        if robot_model is None:
            robot_model = get_g1_robot_model(waist_location="lower_and_upper_body")
        self.robot_model = robot_model
        self._joint_names, self._hand_frame_names = _validate_robot_model(robot_model)

    def build(
        self,
        episode: ResampledEpisode,
        frame_index: int,
        token: np.ndarray,
    ) -> dict[str, object]:
        """Build one owned non-video frame without crossing episode boundaries."""
        if not isinstance(episode, ResampledEpisode):
            raise TypeError("episode must be a ResampledEpisode")
        if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
            raise TypeError("frame_index must be an integer")
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= episode.frame_count:
            raise ValueError(f"frame_index must be in [0, {episode.frame_count}); got {frame_index}")
        if not isinstance(token, np.ndarray):
            raise TypeError("motion token must be a numpy.ndarray")
        if token.dtype != np.dtype(np.float32):
            raise ValueError(f"motion token must have dtype float32; got {token.dtype}")
        if token.shape != (64,):
            raise ValueError(f"motion token must have shape (64,); got {token.shape}")
        if not np.isfinite(token).all():
            raise ValueError("motion token must contain only finite values")

        observed_state = _semantic_configuration(
            field_name="observation.state",
            joint_names=self._joint_names,
            body_names=episode.body_joint_names,
            body_values=episode.observed_body_q[frame_index],
            left_hand_dds=episode.observed_left_hand[frame_index],
            right_hand_dds=episode.observed_right_hand[frame_index],
        )
        desired_state = _semantic_configuration(
            field_name="action.wbc",
            joint_names=self._joint_names,
            body_names=episode.body_joint_names,
            body_values=episode.desired_body_q[frame_index],
            left_hand_dds=episode.desired_left_hand[frame_index],
            right_hand_dds=episode.desired_right_hand[frame_index],
        )

        self.robot_model.cache_forward_kinematics(observed_state, auto_clip=False)
        eef_parts: list[np.ndarray] = []
        for side in ("left", "right"):
            placement = self.robot_model.frame_placement(self._hand_frame_names[side])
            translation = np.asarray(placement.translation, dtype=np.float64)
            rotation = np.asarray(placement.rotation, dtype=np.float64)
            if translation.shape != (3,) or rotation.shape != (3, 3):
                raise ValueError("RobotModel wrist placement must contain translation (3,) and rotation (3, 3)")
            if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
                raise ValueError("RobotModel wrist placement must contain only finite values")
            quaternion = R.from_matrix(rotation).as_quat(scalar_first=True)
            eef_parts.append(np.concatenate((translation, quaternion)))
        eef_state = _owned_array(
            np.concatenate(eef_parts),
            dtype=np.float64,
            field_name="observation.eef_state",
        )

        root_orientation = _owned_array(
            episode.observed_root_wxyz[frame_index],
            dtype=np.float64,
            field_name="observation.root_orientation",
        )
        projected_gravity = _owned_array(
            R.from_quat(root_orientation, scalar_first=True)
            .inv()
            .apply(np.array([0.0, 0.0, -1.0], dtype=np.float64)),
            dtype=np.float64,
            field_name="observation.projected_gravity",
        )

        return {
            "observation.state": observed_state,
            "observation.eef_state": eef_state,
            "action.wbc": desired_state,
            "observation.root_orientation": root_orientation,
            "observation.projected_gravity": projected_gravity,
            "observation.cpp_rotation_offset": _owned_array(
                episode.reference_root_wxyz[0],
                dtype=np.float64,
                field_name="observation.cpp_rotation_offset",
            ),
            "observation.init_base_quat": _owned_array(
                episode.observed_root_wxyz[0],
                dtype=np.float64,
                field_name="observation.init_base_quat",
            ),
            "teleop.delta_heading": np.zeros(1, dtype=np.float64),
            "action.motion_token": _owned_array(
                token,
                dtype=np.float64,
                field_name="action.motion_token",
            ),
            "teleop.smpl_joints": np.zeros(72, dtype=np.float32),
            "teleop.smpl_pose": np.zeros(63, dtype=np.float32),
            "teleop.body_quat_w": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "teleop.target_body_orientation": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "teleop.left_hand_joints": _owned_array(
                episode.desired_left_hand[frame_index],
                dtype=np.float32,
                field_name="teleop.left_hand_joints",
            ),
            "teleop.right_hand_joints": _owned_array(
                episode.desired_right_hand[frame_index],
                dtype=np.float32,
                field_name="teleop.right_hand_joints",
            ),
            "teleop.smpl_frame_index": np.array([frame_index], dtype=np.int64),
            "teleop.left_wrist_joints": np.zeros(3, dtype=np.float32),
            "teleop.right_wrist_joints": np.zeros(3, dtype=np.float32),
            "teleop.stream_mode": np.zeros(1, dtype=np.int32),
            "teleop.planner_mode": np.zeros(1, dtype=np.int32),
            "teleop.planner_movement": np.zeros(3, dtype=np.float32),
            "teleop.planner_facing": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "teleop.planner_speed": np.array([-1.0], dtype=np.float32),
            "teleop.planner_height": np.array([-1.0], dtype=np.float32),
            "teleop.vr_3pt_position": np.zeros(9, dtype=np.float32),
            "teleop.vr_3pt_orientation": np.zeros(18, dtype=np.float32),
            "timestamp": np.float32(frame_index / 50.0),
            "task": episode.task_texts[frame_index],
        }
