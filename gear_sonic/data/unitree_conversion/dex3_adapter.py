"""Strict adapter for revision-pinned Unitree G1 Dex3 LeRobot episodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
import numbers
from pathlib import Path
from typing import Any

import numpy as np

from gear_sonic.data.unitree_conversion.contracts import (
    CanonicalEpisode,
    SourceSpec,
    SourceVideoSegment,
)
from gear_sonic.data.unitree_conversion.joint_mapping import G1_MUJOCO_NAMES, NOMINAL_G1_MUJOCO
from gear_sonic.data.unitree_conversion.lerobot_v3_source import (
    default_lerobot_cache_base,
    load_pinned_v3_episode,
    revision_scoped_root,
)
from gear_sonic.data.unitree_conversion.quaternion import validate_wxyz

_default_lerobot_cache_base = default_lerobot_cache_base
_revision_scoped_root = revision_scoped_root

DEX3_ARM_NAMES = (
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
)

DEX3_HAND_NAMES = (
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
)

DEX3_FEATURE_NAMES = DEX3_ARM_NAMES + DEX3_HAND_NAMES

_ARM_TARGET_BY_SOURCE = {
    "kLeftShoulderPitch": "left_shoulder_pitch_joint",
    "kLeftShoulderRoll": "left_shoulder_roll_joint",
    "kLeftShoulderYaw": "left_shoulder_yaw_joint",
    "kLeftElbow": "left_elbow_joint",
    "kLeftWristRoll": "left_wrist_roll_joint",
    "kLeftWristPitch": "left_wrist_pitch_joint",
    "kLeftWristYaw": "left_wrist_yaw_joint",
    "kRightShoulderPitch": "right_shoulder_pitch_joint",
    "kRightShoulderRoll": "right_shoulder_roll_joint",
    "kRightShoulderYaw": "right_shoulder_yaw_joint",
    "kRightElbow": "right_elbow_joint",
    "kRightWristRoll": "right_wrist_roll_joint",
    "kRightWristPitch": "right_wrist_pitch_joint",
    "kRightWristYaw": "right_wrist_yaw_joint",
}

_ROW_KEYS = (
    "observation.state",
    "action",
    "timestamp",
    "task_index",
    "frame_index",
    "episode_index",
)
_MISSING = object()


def _finite_float_array(value: object, *, field_name: str) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a numeric array") from error
    if source.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must be a numeric array")
    array = np.asarray(source, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def _exact_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(feature_names, (str, bytes)):
        raise ValueError("feature_names must equal the exact Dex3 feature names")
    try:
        names = tuple(feature_names)
    except TypeError as error:
        raise ValueError("feature_names must equal the exact Dex3 feature names") from error
    if names != DEX3_FEATURE_NAMES:
        raise ValueError("feature_names must equal the exact Dex3 feature names in documented source order")
    return names


def adapt_dex3_arrays(
    *,
    source_repo_id: str,
    source_revision: str,
    source_episode_id: int,
    observed: object,
    desired: object,
    feature_names: Sequence[str],
    timestamps: object,
    task_indices: object,
    video_segments: Mapping[str, SourceVideoSegment] | None = None,
) -> CanonicalEpisode:
    """Map documented 28D Dex3 state/action arrays into one canonical episode."""
    names = _exact_feature_names(feature_names)
    observed_array = _finite_float_array(observed, field_name="observed")
    desired_array = _finite_float_array(desired, field_name="desired")
    if observed_array.ndim != 2 or observed_array.shape[1] != 28:
        raise ValueError(f"observed must have shape (N, 28); got {observed_array.shape}")
    if desired_array.ndim != 2 or desired_array.shape[1] != 28:
        raise ValueError(f"desired must have shape (N, 28); got {desired_array.shape}")
    if desired_array.shape[0] != observed_array.shape[0]:
        raise ValueError("observed and desired must have the same number of rows")

    row_count = observed_array.shape[0]
    if row_count < 2:
        raise ValueError("Dex3 episodes require at least two rows")
    timestamps_array = _finite_float_array(timestamps, field_name="timestamps")
    if timestamps_array.shape != (row_count,):
        raise ValueError(f"timestamps must have shape ({row_count},); got {timestamps_array.shape}")
    try:
        task_indices_array = np.asarray(task_indices)
    except (TypeError, ValueError) as error:
        raise ValueError(f"task_indices must have shape ({row_count},)") from error
    if task_indices_array.shape != (row_count,):
        raise ValueError(f"task_indices must have shape ({row_count},); got {task_indices_array.shape}")

    observed_body_q = np.tile(NOMINAL_G1_MUJOCO, (row_count, 1))
    desired_body_q = np.tile(NOMINAL_G1_MUJOCO, (row_count, 1))
    for source_name, target_name in _ARM_TARGET_BY_SOURCE.items():
        source_index = names.index(source_name)
        target_index = G1_MUJOCO_NAMES.index(target_name)
        observed_body_q[:, target_index] = observed_array[:, source_index]
        desired_body_q[:, target_index] = desired_array[:, source_index]

    left_hand_indices = tuple(names.index(name) for name in DEX3_HAND_NAMES[:7])
    right_hand_indices = tuple(names.index(name) for name in DEX3_HAND_NAMES[7:])
    observed_identity, _ = validate_wxyz(np.array([1.0, 0.0, 0.0, 0.0]))
    reference_identity, _ = validate_wxyz(np.array([1.0, 0.0, 0.0, 0.0]))
    observed_identity_roots = np.tile(observed_identity, (row_count, 1))
    reference_identity_roots = np.tile(reference_identity, (row_count, 1))
    return CanonicalEpisode(
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        source_episode_id=source_episode_id,
        source_fps=30,
        timestamps=timestamps_array,
        task_indices=task_indices_array,
        observed_root_wxyz=observed_identity_roots,
        reference_root_wxyz=reference_identity_roots,
        observed_body_q=observed_body_q,
        desired_body_q=desired_body_q,
        observed_left_hand=observed_array[:, left_hand_indices],
        observed_right_hand=observed_array[:, right_hand_indices],
        desired_left_hand=desired_array[:, left_hand_indices],
        desired_right_hand=desired_array[:, right_hand_indices],
        video_segments={} if video_segments is None else video_segments,
    )


def _to_numpy(value: object, *, field_name: str) -> np.ndarray:
    converted = value
    for method_name in ("detach", "cpu"):
        method = getattr(converted, method_name, None)
        if callable(method):
            converted = method()
    numpy_method = getattr(converted, "numpy", None)
    if callable(numpy_method):
        converted = numpy_method()
    try:
        return np.asarray(converted)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} cannot be converted to a NumPy array") from error


def _integer_scalar(value: object, *, field_name: str) -> int:
    array = _to_numpy(value, field_name=field_name)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{field_name} must be an integer scalar")
    integer = int(array)
    if integer < np.iinfo(np.int64).min or integer > np.iinfo(np.int64).max:
        raise ValueError(f"{field_name} must be an int64-compatible integer scalar")
    return integer


def _numeric_scalar(value: object, *, field_name: str) -> float:
    array = _to_numpy(value, field_name=field_name)
    if array.shape != () or array.dtype.kind not in "iuf":
        raise ValueError(f"{field_name} must be a numeric scalar")
    scalar = float(array)
    if not math.isfinite(scalar):
        raise ValueError(f"{field_name} must be finite")
    return scalar


def _feature_names_from_meta(features: object, key: str) -> tuple[str, ...]:
    if not isinstance(features, Mapping) or key not in features or not isinstance(features[key], Mapping):
        raise ValueError(
            "dataset meta.features must be a mapping containing mapping entries for observation.state and action"
        )
    metadata = features[key]
    if "names" not in metadata:
        raise ValueError(f"dataset meta.features entry {key} must contain names")
    names = metadata["names"]
    if isinstance(names, (str, bytes)):
        raise ValueError(f"{key} metadata must contain the exact Dex3 feature names")
    try:
        names_tuple = tuple(names)
    except TypeError as error:
        raise ValueError(f"{key} metadata must contain the exact Dex3 feature names") from error
    if len(names_tuple) == 1 and isinstance(names_tuple[0], (list, tuple)):
        names_tuple = tuple(names_tuple[0])
    if names_tuple != DEX3_FEATURE_NAMES or not all(isinstance(name, str) for name in names_tuple):
        raise ValueError(f"{key} metadata must contain the exact Dex3 feature names")
    return names_tuple


def load_dex3_episode(
    source_spec: SourceSpec,
    episode_id: int,
    root: str | Path | None = None,
    *,
    dataset_factory: Callable[..., Any] | None = None,
    snapshot_downloader: Callable[..., object] | None = None,
) -> CanonicalEpisode:
    """Load one selected episode from an approved, revision-pinned Dex3 source."""
    if not isinstance(source_spec, SourceSpec) or not source_spec.approved:
        raise ValueError("source_spec must be an approved SourceSpec")
    if source_spec.episode_count is None:
        raise ValueError("source_spec must have a pinned episode_count")
    if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id < 0:
        raise ValueError("episode_id must be a nonnegative integer")
    if episode_id >= source_spec.episode_count:
        raise ValueError("episode_id must be smaller than pinned episode_count")
    if episode_id not in source_spec.episodes:
        raise ValueError("episode_id must be selected in source_spec.episodes")

    cache_base = _default_lerobot_cache_base() if root is None else root
    direct_dataset = load_pinned_v3_episode(
        source_spec,
        episode_id,
        cache_base=cache_base,
        snapshot_downloader=snapshot_downloader,
    )
    if dataset_factory is None:
        dataset = direct_dataset
    else:
        dataset = dataset_factory(
            repo_id=source_spec.repo_id,
            root=direct_dataset.root,
            episodes=[episode_id],
            revision=source_spec.revision,
            download_videos=True,
            video_backend="pyav",
        )
    meta = getattr(dataset, "meta", None)
    if getattr(dataset, "revision", None) != source_spec.revision:
        raise ValueError("dataset.revision must equal pinned source revision")
    if getattr(meta, "revision", None) != source_spec.revision:
        raise ValueError("dataset.meta.revision must equal pinned source revision")
    total_episodes = getattr(meta, "total_episodes", None)
    if (
        isinstance(total_episodes, bool)
        or not isinstance(total_episodes, numbers.Integral)
        or total_episodes != source_spec.episode_count
    ):
        raise ValueError("dataset metadata total_episodes must exactly match pinned episode_count")
    fps = getattr(meta, "fps", None)
    if isinstance(fps, bool) or not isinstance(fps, numbers.Real) or not math.isfinite(float(fps)) or fps != 30:
        raise ValueError("dataset metadata fps must be numeric 30")

    features = getattr(meta, "features", None)
    observation_names = _feature_names_from_meta(features, "observation.state")
    action_names = _feature_names_from_meta(features, "action")
    if observation_names != action_names:
        raise ValueError("observation.state and action feature names must match exactly")
    expected_video_keys = tuple(
        key
        for key, feature in direct_dataset.meta.features.items()
        if isinstance(feature, Mapping) and feature.get("dtype") == "video"
    )
    video_segments = direct_dataset.video_segments
    if set(video_segments) != set(expected_video_keys):
        raise ValueError("approved Dex3 source must carry exact segment metadata for every video key")

    observed_rows: list[np.ndarray] = []
    desired_rows: list[np.ndarray] = []
    timestamps: list[float] = []
    task_indices: list[int] = []
    raw_rows = getattr(dataset, "hf_dataset", _MISSING)
    if raw_rows is _MISSING or isinstance(raw_rows, (str, bytes, Mapping)):
        raise ValueError("dataset.hf_dataset must be a non-mapping iterable of raw rows")
    try:
        row_iterator = iter(raw_rows)
    except TypeError as error:
        raise ValueError("dataset.hf_dataset must be a non-mapping iterable of raw rows") from error

    for row_number, row in enumerate(row_iterator):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_number} must be a mapping")
        for key in _ROW_KEYS:
            if key not in row:
                raise ValueError(f"row {row_number} is missing required key {key}")

        source_episode_id = _integer_scalar(
            row["episode_index"],
            field_name=f"row {row_number} episode_index",
        )
        if source_episode_id != episode_id:
            raise ValueError(
                f"source episode_index {source_episode_id} does not match requested episode {episode_id}"
            )
        frame_index = _integer_scalar(
            row["frame_index"],
            field_name=f"row {row_number} frame_index",
        )
        if frame_index != row_number:
            raise ValueError(f"expected frame_index {row_number}, got {frame_index}")

        observed = _to_numpy(row["observation.state"], field_name=f"row {row_number} observation.state")
        desired = _to_numpy(row["action"], field_name=f"row {row_number} action")
        if observed.shape != (28,):
            raise ValueError(f"row {row_number} observation.state must have shape (28,); got {observed.shape}")
        if desired.shape != (28,):
            raise ValueError(f"row {row_number} action must have shape (28,); got {desired.shape}")
        observed_rows.append(observed)
        desired_rows.append(desired)
        timestamps.append(_numeric_scalar(row["timestamp"], field_name=f"row {row_number} timestamp"))
        task_indices.append(_integer_scalar(row["task_index"], field_name=f"row {row_number} task_index"))

    if not observed_rows:
        raise ValueError(f"no rows for requested episode {episode_id}")
    return adapt_dex3_arrays(
        source_repo_id=source_spec.repo_id,
        source_revision=source_spec.revision,
        source_episode_id=episode_id,
        observed=np.stack(observed_rows),
        desired=np.stack(desired_rows),
        feature_names=observation_names,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        task_indices=np.asarray(task_indices),
        video_segments=video_segments,
    )
