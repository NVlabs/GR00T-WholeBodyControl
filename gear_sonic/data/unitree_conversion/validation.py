"""Structural validation for immutable Unitree-to-SONIC episode stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import io
from pathlib import Path
import threading
from types import MappingProxyType
from typing import BinaryIO

import av
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class FrameFieldSpec:
    """One exact target-frame scalar or fixed-width array field."""

    dtype: np.dtype
    shape: tuple[int, ...]


def _spec(dtype: str, width: int) -> FrameFieldSpec:
    return FrameFieldSpec(np.dtype(dtype), (width,))


TARGET_FRAME_SCHEMA: Mapping[str, FrameFieldSpec | None] = MappingProxyType(
    {
        "observation.state": _spec("float64", 43),
        "observation.eef_state": _spec("float64", 14),
        "action.wbc": _spec("float64", 43),
        "observation.root_orientation": _spec("float64", 4),
        "observation.projected_gravity": _spec("float64", 3),
        "observation.cpp_rotation_offset": _spec("float64", 4),
        "observation.init_base_quat": _spec("float64", 4),
        "teleop.delta_heading": _spec("float64", 1),
        "action.motion_token": _spec("float64", 64),
        "teleop.smpl_joints": _spec("float32", 72),
        "teleop.smpl_pose": _spec("float32", 63),
        "teleop.body_quat_w": _spec("float32", 4),
        "teleop.target_body_orientation": _spec("float32", 6),
        "teleop.left_hand_joints": _spec("float32", 7),
        "teleop.right_hand_joints": _spec("float32", 7),
        "teleop.smpl_frame_index": _spec("int64", 1),
        "teleop.left_wrist_joints": _spec("float32", 3),
        "teleop.right_wrist_joints": _spec("float32", 3),
        "teleop.stream_mode": _spec("int32", 1),
        "teleop.planner_mode": _spec("int32", 1),
        "teleop.planner_movement": _spec("float32", 3),
        "teleop.planner_facing": _spec("float32", 3),
        "teleop.planner_speed": _spec("float32", 1),
        "teleop.planner_height": _spec("float32", 1),
        "teleop.vr_3pt_position": _spec("float32", 9),
        "teleop.vr_3pt_orientation": _spec("float32", 18),
        "timestamp": None,
        "task": None,
    }
)
TARGET_VIDEO_KEYS = frozenset(
    {
        "observation.images.ego_view",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
)
TARGET_VIDEO_SIZE = (640, 480)
_QUATERNION_NORM_TOLERANCE = 1e-5
_JOINT_LIMIT_TOLERANCE = 1e-9
_DEX3_MEASUREMENT_TOLERANCE = 1e-2
_EEF_POSITION_TOLERANCE = 1e-9
_EEF_QUATERNION_TOLERANCE = 1e-8
TARGET_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
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
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)
# ``observation.state`` and ``action.wbc`` use RobotModel *ordering*, but the
# hand slots retain the raw DDS motor coordinates used by deployment and the
# existing data exporter.  They therefore cannot use the symmetric hand limits
# from the visualization URDF.  These float32-exact command domains match the
# pinned Unitree Dex3 dataset/controller convention.
_DEX3_DDS_COMMAND_LIMITS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "left_hand_index_0_joint": (-1.832595705986023, 0.19198620319366455),
        "left_hand_index_1_joint": (-2.094395160675049, 0.0),
        "left_hand_middle_0_joint": (-1.832595705986023, 0.19198620319366455),
        "left_hand_middle_1_joint": (-2.094395160675049, 0.0),
        "left_hand_thumb_0_joint": (-1.0471975803375244, 1.0471975803375244),
        "left_hand_thumb_1_joint": (-1.0471975803375244, 1.0471975803375244),
        "left_hand_thumb_2_joint": (0.0, 1.7453292608261108),
        "right_hand_index_0_joint": (-0.19198620319366455, 1.832595705986023),
        "right_hand_index_1_joint": (0.0, 2.094395160675049),
        "right_hand_middle_0_joint": (-0.19198620319366455, 1.832595705986023),
        "right_hand_middle_1_joint": (0.0, 2.094395160675049),
        "right_hand_thumb_0_joint": (-1.0471975803375244, 1.0471975803375244),
        "right_hand_thumb_1_joint": (-1.0471975803375244, 1.0471975803375244),
        "right_hand_thumb_2_joint": (-1.7453292608261108, 0.0),
    }
)
_DEX3_DDS_INDICES = np.array(
    [index for index, name in enumerate(TARGET_JOINT_NAMES) if name in _DEX3_DDS_COMMAND_LIMITS],
    dtype=np.intp,
)
_BODY_JOINT_INDICES = np.array(
    [index for index, name in enumerate(TARGET_JOINT_NAMES) if name not in _DEX3_DDS_COMMAND_LIMITS],
    dtype=np.intp,
)
_EXACT_NEUTRAL_FIELDS: Mapping[str, np.ndarray] = MappingProxyType(
    {
        "teleop.delta_heading": np.zeros(1, dtype=np.float64),
        "teleop.smpl_joints": np.zeros(72, dtype=np.float32),
        "teleop.smpl_pose": np.zeros(63, dtype=np.float32),
        "teleop.body_quat_w": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "teleop.target_body_orientation": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
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
    }
)


@dataclass(frozen=True)
class _TargetKinematics:
    model: object
    lower: np.ndarray
    upper: np.ndarray
    left_frame_id: int
    right_frame_id: int


_KINEMATICS_THREAD_LOCAL = threading.local()


@lru_cache(maxsize=1)
def _target_kinematics() -> _TargetKinematics:
    import pinocchio as pin

    from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
        G1SupplementalInfo,
    )

    urdf = Path(__file__).resolve().parents[1] / "robot_model" / "model_data" / "g1" / "g1_29dof_with_hand.urdf"
    model = pin.buildModelFromUrdf(str(urdf))
    joint_names = tuple(model.names)[1:]
    if model.nq != 43 or model.nv != 43 or joint_names != TARGET_JOINT_NAMES:
        raise RuntimeError("asset-free G1 URDF does not match the exact 43-joint target order")
    supplemental = G1SupplementalInfo()
    if set(supplemental.joint_limits) != set(TARGET_JOINT_NAMES):
        raise RuntimeError("G1 supplemental limits do not match the exact 43-joint target model")
    lower = np.array(model.lowerPositionLimit, dtype=np.float64, order="C", copy=True)
    upper = np.array(model.upperPositionLimit, dtype=np.float64, order="C", copy=True)
    for name, (lower_bound, upper_bound) in _DEX3_DDS_COMMAND_LIMITS.items():
        index = TARGET_JOINT_NAMES.index(name)
        lower[index] = lower_bound
        upper[index] = upper_bound
    if lower.shape != (43,) or upper.shape != (43,) or not np.all(lower <= upper):
        raise RuntimeError("asset-free G1 URDF contains invalid target joint limits")
    lower.setflags(write=False)
    upper.setflags(write=False)
    return _TargetKinematics(
        model=model,
        lower=lower,
        upper=upper,
        left_frame_id=int(model.getFrameId("left_wrist_yaw_link")),
        right_frame_id=int(model.getFrameId("right_wrist_yaw_link")),
    )


def target_joint_limits() -> tuple[np.ndarray, np.ndarray]:
    """Return raw body-URDF and Dex3 DDS command limits in target field order."""
    kinematics = _target_kinematics()
    return kinematics.lower.copy(), kinematics.upper.copy()


def regenerate_eef_state(state: np.ndarray) -> np.ndarray:
    """Regenerate wrist XYZ + WXYZ from an observed 43D target state."""
    if not isinstance(state, np.ndarray) or state.dtype != np.dtype(np.float64) or state.shape != (43,):
        raise ValueError("observation.state must be a float64 ndarray with shape (43,)")
    if not np.isfinite(state).all():
        raise ValueError("observation.state must contain only finite values")
    import pinocchio as pin

    kinematics = _target_kinematics()
    data_by_model = getattr(_KINEMATICS_THREAD_LOCAL, "data_by_model", None)
    if data_by_model is None:
        data_by_model = {}
        _KINEMATICS_THREAD_LOCAL.data_by_model = data_by_model
    data = data_by_model.get(id(kinematics.model))
    if data is None:
        data = kinematics.model.createData()
        data_by_model[id(kinematics.model)] = data
    pin.framesForwardKinematics(kinematics.model, data, state)
    parts: list[np.ndarray] = []
    for frame_id in (kinematics.left_frame_id, kinematics.right_frame_id):
        placement = data.oMf[frame_id]
        quaternion = R.from_matrix(np.asarray(placement.rotation)).as_quat(scalar_first=True)
        parts.extend((np.asarray(placement.translation, dtype=np.float64), quaternion))
    return np.array(np.concatenate(parts), dtype=np.float64, order="C", copy=True)


@dataclass(frozen=True)
class ValidationReport:
    """A complete successful structural validation result."""

    success: bool
    row_count: int
    field_names: tuple[str, ...]
    video_frame_counts: Mapping[str, int] = field(default_factory=dict)
    video_dimensions: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.success is not True:
            raise ValueError("published validation reports must represent success")
        if type(self.row_count) is not int or self.row_count < 1:
            raise ValueError("validation row_count must be positive")
        if self.field_names != tuple(TARGET_FRAME_SCHEMA):
            raise ValueError("validation field_names must match the exact target schema")
        counts = dict(sorted(self.video_frame_counts.items()))
        dimensions = dict(sorted(self.video_dimensions.items()))
        if set(counts) != set(dimensions):
            raise ValueError("video validation counts and dimensions must have identical keys")
        if "observation.images.ego_view" not in counts:
            raise ValueError("validation requires observation.images.ego_view")
        if any(type(count) is not int or count != self.row_count for count in counts.values()):
            raise ValueError("every video frame count must equal validation row_count")
        if any(size != TARGET_VIDEO_SIZE for size in dimensions.values()):
            raise ValueError(f"video dimensions must be exactly {TARGET_VIDEO_SIZE}")
        object.__setattr__(self, "video_frame_counts", MappingProxyType(counts))
        object.__setattr__(self, "video_dimensions", MappingProxyType(dimensions))

    def to_dict(self) -> dict[str, object]:
        return {
            "success": True,
            "row_count": self.row_count,
            "field_names": list(self.field_names),
            "video_frame_counts": dict(self.video_frame_counts),
            "video_dimensions": {key: list(value) for key, value in self.video_dimensions.items()},
        }


def validate_target_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    start_index: int = 0,
    initial_root_fields: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject rows that cannot be represented by the exact target schema."""
    if isinstance(rows, (str, bytes)) or len(rows) < 1:
        raise ValueError("stage rows must contain at least one target frame")
    if type(start_index) is not int or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")
    expected_keys = tuple(TARGET_FRAME_SCHEMA)
    for local_index, row in enumerate(rows):
        index = start_index + local_index
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be a target-frame mapping")
        if set(row) != set(expected_keys):
            missing = sorted(set(expected_keys) - set(row))
            extra = sorted(set(row) - set(expected_keys))
            raise ValueError(f"row {index} target keys differ: missing={missing}, extra={extra}")
        for key, spec in TARGET_FRAME_SCHEMA.items():
            value = row[key]
            if key == "task":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"row {index} task must be a nonempty UTF-8 string")
                try:
                    value.encode("utf-8", errors="strict")
                except UnicodeEncodeError as error:
                    raise ValueError(f"row {index} task must be a nonempty UTF-8 string") from error
                continue
            if key == "timestamp":
                if not isinstance(value, np.float32):
                    raise ValueError(f"row {index} timestamp must be a numpy.float32 scalar")
                expected = np.float32(index / 50.0)
                if value.tobytes() != expected.tobytes():
                    raise ValueError(f"row {index} timestamp must be bitwise np.float32(j/50)")
                continue
            assert spec is not None
            if not isinstance(value, np.ndarray):
                raise ValueError(f"row {index} {key} must be a numpy.ndarray")
            if value.dtype != spec.dtype or value.shape != spec.shape:
                raise ValueError(f"row {index} {key} must have dtype {spec.dtype} and shape {spec.shape}")
            if value.dtype.kind in "iuf" and not np.isfinite(value).all():
                raise ValueError(f"row {index} {key} must contain only finite values")
        frame_index = row["teleop.smpl_frame_index"]
        assert isinstance(frame_index, np.ndarray)
        if int(frame_index[0]) != index:
            raise ValueError(f"row {index} teleop.smpl_frame_index must equal its episode-local index")
        if initial_root_fields is None:
            initial_root_fields = (
                np.array(row["observation.cpp_rotation_offset"], copy=True),
                np.array(row["observation.init_base_quat"], copy=True),
            )
        for key, expected in _EXACT_NEUTRAL_FIELDS.items():
            if not np.array_equal(row[key], expected):
                raise ValueError(f"row {index} {key} must equal the exact neutral target value")
        kinematics = _target_kinematics()
        for key in ("observation.state", "action.wbc"):
            joint_values = row[key]
            assert isinstance(joint_values, np.ndarray)
            body_values = joint_values[_BODY_JOINT_INDICES]
            if np.any(body_values < kinematics.lower[_BODY_JOINT_INDICES] - _JOINT_LIMIT_TOLERANCE) or np.any(
                body_values > kinematics.upper[_BODY_JOINT_INDICES] + _JOINT_LIMIT_TOLERANCE
            ):
                raise ValueError(f"row {index} {key} must remain within exact target body joint limits")
            hand_tolerance = _DEX3_MEASUREMENT_TOLERANCE if key == "observation.state" else _JOINT_LIMIT_TOLERANCE
            hand_values = joint_values[_DEX3_DDS_INDICES]
            if np.any(hand_values < kinematics.lower[_DEX3_DDS_INDICES] - hand_tolerance) or np.any(
                hand_values > kinematics.upper[_DEX3_DDS_INDICES] + hand_tolerance
            ):
                qualifier = "measurement" if key == "observation.state" else "command"
                raise ValueError(f"row {index} {key} must remain within exact Dex3 DDS {qualifier} limits")
        for key in (
            "observation.root_orientation",
            "observation.cpp_rotation_offset",
            "observation.init_base_quat",
        ):
            quaternion = row[key]
            assert isinstance(quaternion, np.ndarray)
            norm = float(np.linalg.norm(quaternion))
            if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
                raise ValueError(f"row {index} {key} must be a unit WXYZ quaternion")
        eef_state = row["observation.eef_state"]
        assert isinstance(eef_state, np.ndarray)
        for start in (3, 10):
            if abs(float(np.linalg.norm(eef_state[start : start + 4])) - 1.0) > _QUATERNION_NORM_TOLERANCE:
                raise ValueError(f"row {index} observation.eef_state wrist quaternion must be unit WXYZ")
        regenerated_eef = regenerate_eef_state(row["observation.state"])
        for start in (0, 7):
            if not np.allclose(
                eef_state[start : start + 3],
                regenerated_eef[start : start + 3],
                rtol=0.0,
                atol=_EEF_POSITION_TOLERANCE,
            ):
                raise ValueError(f"row {index} observation.eef_state translation differs from forward kinematics")
            actual_quaternion = eef_state[start + 3 : start + 7]
            expected_quaternion = regenerated_eef[start + 3 : start + 7]
            quaternion_error = min(
                np.max(np.abs(actual_quaternion - expected_quaternion)),
                np.max(np.abs(actual_quaternion + expected_quaternion)),
            )
            if quaternion_error > _EEF_QUATERNION_TOLERANCE:
                raise ValueError(f"row {index} observation.eef_state quaternion differs from forward kinematics")
        root = row["observation.root_orientation"]
        gravity = row["observation.projected_gravity"]
        assert isinstance(root, np.ndarray) and isinstance(gravity, np.ndarray)
        normalized = root / np.linalg.norm(root)
        w = normalized[0]
        vector = normalized[1:]
        down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        cross = np.cross(vector, down)
        expected_gravity = down - 2.0 * w * cross + 2.0 * np.cross(vector, cross)
        if not np.allclose(gravity, expected_gravity, rtol=0.0, atol=1e-10):
            raise ValueError(f"row {index} observation.projected_gravity must match inverse root rotation")
        if index and (
            not np.array_equal(row["observation.cpp_rotation_offset"], initial_root_fields[0])
            or not np.array_equal(row["observation.init_base_quat"], initial_root_fields[1])
        ):
            raise ValueError("episode initial root fields must remain constant across every row")
    assert initial_root_fields is not None
    return np.array(initial_root_fields[0], copy=True), np.array(initial_root_fields[1], copy=True)


def probe_video_50fps_stream(stream: BinaryIO, *, field_name: str) -> tuple[int, tuple[int, int]]:
    """Decode one seekable MP4 stream and return its exact 50 Hz count and size."""
    try:
        stream.seek(0)
        with av.open(stream, mode="r") as container:
            streams = tuple(container.streams.video)
            if len(container.streams) != 1 or len(streams) != 1:
                raise ValueError(f"{field_name} must contain exactly one video stream and no other streams")
            stream = streams[0]
            if stream.codec_context.name != "h264":
                raise ValueError(f"{field_name} video codec must be H264")
            if stream.pix_fmt != "yuv420p":
                raise ValueError(f"{field_name} pixel format must be yuv420p")
            if stream.average_rate is None or Fraction(stream.average_rate) != 50:
                raise ValueError(f"{field_name} must have nominal fps 50")
            count = 0
            dimensions: tuple[int, int] | None = None
            expected_position = 0
            for frame in container.decode(stream):
                current = (int(frame.width), int(frame.height))
                if current != TARGET_VIDEO_SIZE:
                    raise ValueError(f"{field_name} frame size {current} != required {TARGET_VIDEO_SIZE}")
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(f"{field_name} frames require PTS and time_base")
                position = Fraction(frame.pts) * Fraction(frame.time_base) * 50
                if position != expected_position:
                    raise ValueError(f"{field_name} PTS must be consecutive on the episode-local 50 Hz grid")
                if dimensions is None:
                    dimensions = current
                elif dimensions != current:
                    raise ValueError(f"{field_name} changes dimensions within the episode")
                count += 1
                expected_position += 1
    except (av.error.FFmpegError, OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(field_name):
            raise
        raise ValueError(f"{field_name} must be a readable 50 fps MP4") from error
    if count < 1 or dimensions is None:
        raise ValueError(f"{field_name} must contain at least one video frame")
    return count, dimensions


def probe_video_50fps(data: bytes, *, field_name: str) -> tuple[int, tuple[int, int]]:
    """Decode an owned MP4 blob and return its exact 50 Hz count and size."""
    return probe_video_50fps_stream(io.BytesIO(data), field_name=field_name)


def validate_stage_payload(
    rows: Sequence[Mapping[str, object]],
    videos: Mapping[str, bytes],
) -> ValidationReport:
    """Validate an entire detached stage payload before publication."""
    inspections: dict[str, tuple[int, tuple[int, int]]] = {}
    for key, data in sorted(videos.items()):
        inspections[key] = probe_video_50fps(data, field_name=f"video {key}")
    return validation_report_from_inspections(rows, inspections)


def validation_report_from_inspections(
    rows: Sequence[Mapping[str, object]],
    inspections: Mapping[str, tuple[int, tuple[int, int]]],
) -> ValidationReport:
    """Combine exact row validation with already-decoded video inspections."""
    validate_target_rows(rows)
    if "observation.images.ego_view" not in inspections:
        raise ValueError("stage videos require observation.images.ego_view")
    unknown = sorted(set(inspections) - TARGET_VIDEO_KEYS)
    if unknown:
        raise ValueError(f"stage videos contain unsupported target keys: {unknown}")
    counts: dict[str, int] = {}
    dimensions: dict[str, tuple[int, int]] = {}
    for key, (count, size) in sorted(inspections.items()):
        if count != len(rows):
            raise ValueError(f"video frame count for {key} is {count}; expected target row count {len(rows)}")
        counts[key] = count
        dimensions[key] = size
    return ValidationReport(
        success=True,
        row_count=len(rows),
        field_names=tuple(TARGET_FRAME_SCHEMA),
        video_frame_counts=counts,
        video_dimensions=dimensions,
    )
