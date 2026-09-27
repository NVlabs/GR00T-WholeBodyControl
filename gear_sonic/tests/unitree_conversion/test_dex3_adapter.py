from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gear_sonic.data.unitree_conversion import dex3_adapter as dex3_adapter_module
from gear_sonic.data.unitree_conversion.contracts import (
    CanonicalEpisode,
    SourceSpec,
    SourceVideoSegment,
)
from gear_sonic.data.unitree_conversion.dex3_adapter import adapt_dex3_arrays, load_dex3_episode
from gear_sonic.data.unitree_conversion.joint_mapping import G1_MUJOCO_NAMES, NOMINAL_G1_MUJOCO
from gear_sonic.data.unitree_conversion.lerobot_v3_source import (
    V3DataSchema,
    load_pinned_v3_episode,
    revision_scoped_root,
)

ARM_NAMES = (
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
HAND_NAMES = (
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
FEATURE_NAMES = ARM_NAMES + HAND_NAMES
ARM_TARGET_NAMES = (
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
)
CANONICAL_ARRAY_FIELDS = (
    "timestamps",
    "task_indices",
    "observed_root_wxyz",
    "reference_root_wxyz",
    "observed_body_q",
    "desired_body_q",
    "observed_left_hand",
    "observed_right_hand",
    "desired_left_hand",
    "desired_right_hand",
)


@pytest.fixture(autouse=True)
def _isolated_lerobot_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lerobot-cache"))


def _adapter_kwargs(n: int = 3) -> dict[str, object]:
    observed = np.arange(n * 28, dtype=np.float64).reshape(n, 28)
    return {
        "source_repo_id": "synthetic/dex3",
        "source_revision": "1" * 40,
        "source_episode_id": 7,
        "observed": observed,
        "desired": observed + 1000.0,
        "feature_names": FEATURE_NAMES,
        "timestamps": np.arange(n, dtype=np.float64) / 30.0,
        "task_indices": np.arange(n, dtype=np.int64),
    }


def _canonical_kwargs(n: int = 3) -> dict[str, object]:
    return {
        "source_repo_id": "synthetic/dex3",
        "source_revision": "a" * 40,
        "source_episode_id": 2,
        "source_fps": 30,
        "timestamps": np.arange(n, dtype=np.float64) / 30.0,
        "task_indices": np.arange(n, dtype=np.int64),
        "observed_root_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "reference_root_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "observed_body_q": np.tile(NOMINAL_G1_MUJOCO, (n, 1)),
        "desired_body_q": np.tile(NOMINAL_G1_MUJOCO, (n, 1)),
        "observed_left_hand": np.zeros((n, 7)),
        "observed_right_hand": np.zeros((n, 7)),
        "desired_left_hand": np.ones((n, 7)),
        "desired_right_hand": np.ones((n, 7)),
    }


def test_adapts_arms_by_semantic_name_and_preserves_exact_dds_hand_orders() -> None:
    kwargs = _adapter_kwargs()
    observed = kwargs["observed"]
    desired = kwargs["desired"]

    episode = adapt_dex3_arrays(**kwargs)

    expected_observed_body = np.tile(NOMINAL_G1_MUJOCO, (3, 1))
    expected_desired_body = np.tile(NOMINAL_G1_MUJOCO, (3, 1))
    for source_name, target_name in zip(ARM_NAMES, ARM_TARGET_NAMES, strict=True):
        source_index = FEATURE_NAMES.index(source_name)
        target_index = G1_MUJOCO_NAMES.index(target_name)
        expected_observed_body[:, target_index] = observed[:, source_index]
        expected_desired_body[:, target_index] = desired[:, source_index]

    np.testing.assert_array_equal(episode.observed_body_q, expected_observed_body)
    np.testing.assert_array_equal(episode.desired_body_q, expected_desired_body)
    np.testing.assert_array_equal(episode.observed_left_hand, observed[:, 14:21])
    np.testing.assert_array_equal(episode.desired_left_hand, desired[:, 14:21])
    np.testing.assert_array_equal(episode.observed_right_hand, observed[:, 21:28])
    np.testing.assert_array_equal(episode.desired_right_hand, desired[:, 21:28])
    assert HAND_NAMES[3:7] == (
        "kLeftHandMiddle0",
        "kLeftHandMiddle1",
        "kLeftHandIndex0",
        "kLeftHandIndex1",
    )
    assert HAND_NAMES[10:14] == (
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    )
    np.testing.assert_array_equal(episode.observed_root_wxyz, [[1.0, 0.0, 0.0, 0.0]] * 3)
    np.testing.assert_array_equal(episode.reference_root_wxyz, [[1.0, 0.0, 0.0, 0.0]] * 3)


def test_adapter_returns_mutable_owned_contiguous_arrays_isolated_from_every_source_buffer() -> None:
    observed_storage = np.arange(3 * 56, dtype=np.float32).reshape(3, 56)
    desired_storage = observed_storage + 500.0
    timestamp_storage = np.arange(6, dtype=np.float32) / 60.0
    task_storage = np.arange(6, dtype=np.int32)
    observed = observed_storage[:, ::2]
    desired = desired_storage[:, ::2]
    timestamps = timestamp_storage[::2]
    task_indices = task_storage[::2]
    kwargs = _adapter_kwargs()
    kwargs.update(
        observed=observed,
        desired=desired,
        timestamps=timestamps,
        task_indices=task_indices,
    )

    episode = adapt_dex3_arrays(**kwargs)
    snapshots = {name: getattr(episode, name).copy() for name in CANONICAL_ARRAY_FIELDS}
    observed_storage[:] = -1
    desired_storage[:] = -2
    timestamp_storage[:] = -3
    task_storage[:] = -4

    for name in CANONICAL_ARRAY_FIELDS:
        array = getattr(episode, name)
        np.testing.assert_array_equal(array, snapshots[name])
        assert array.flags.owndata
        assert array.flags.c_contiguous
        assert array.flags.writeable
    assert episode.task_indices.dtype == np.dtype(np.int64)
    assert all(
        getattr(episode, name).dtype == np.dtype(np.float64)
        for name in CANONICAL_ARRAY_FIELDS
        if name != "task_indices"
    )


@pytest.mark.parametrize(
    "feature_names",
    [
        tuple(f"feature_{index}" for index in range(28)),
        (ARM_NAMES[1], ARM_NAMES[0]) + ARM_NAMES[2:] + HAND_NAMES,
        ARM_NAMES + HAND_NAMES[:-2] + (HAND_NAMES[-1], HAND_NAMES[-2]),
        FEATURE_NAMES[:-1],
    ],
)
def test_adapter_rejects_nonexact_or_swapped_feature_names(feature_names: tuple[str, ...]) -> None:
    kwargs = _adapter_kwargs()
    kwargs["feature_names"] = feature_names

    with pytest.raises(ValueError, match="exact Dex3 feature names"):
        adapt_dex3_arrays(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("observed", np.zeros((3, 27))),
        ("observed", np.zeros((3, 28, 1))),
        ("desired", np.zeros((3, 29))),
        ("desired", np.zeros(28)),
    ],
)
def test_adapter_rejects_wrong_state_and_action_shapes(field_name: str, value: np.ndarray) -> None:
    kwargs = _adapter_kwargs()
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=rf"{field_name} must have shape"):
        adapt_dex3_arrays(**kwargs)


def test_adapter_rejects_mismatched_observed_and_desired_row_counts() -> None:
    kwargs = _adapter_kwargs()
    kwargs["desired"] = np.zeros((2, 28))

    with pytest.raises(ValueError, match="same number of rows"):
        adapt_dex3_arrays(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("timestamps", np.zeros(2)),
        ("timestamps", np.zeros((3, 1))),
        ("task_indices", np.zeros(2, dtype=np.int64)),
        ("task_indices", np.zeros((3, 1), dtype=np.int64)),
    ],
)
def test_adapter_rejects_wrong_timeline_shapes(field_name: str, value: np.ndarray) -> None:
    kwargs = _adapter_kwargs()
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=rf"{field_name} must have shape"):
        adapt_dex3_arrays(**kwargs)


@pytest.mark.parametrize("field_name", ["observed", "desired", "timestamps"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_adapter_rejects_nonfinite_sources(field_name: str, bad_value: float) -> None:
    kwargs = _adapter_kwargs()
    value = np.array(kwargs[field_name], copy=True)
    value.flat[-1] = bad_value
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=rf"{field_name}.*finite"):
        adapt_dex3_arrays(**kwargs)


def test_adapter_rejects_episodes_shorter_than_two_rows() -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        adapt_dex3_arrays(**_adapter_kwargs(n=1))


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("source_repo_id", "", "nonempty"),
        ("source_repo_id", "   ", "nonempty"),
        ("source_revision", "main", "40-character immutable revision"),
        ("source_episode_id", True, "nonnegative integer"),
        ("source_episode_id", -1, "nonnegative integer"),
    ],
)
def test_adapter_rejects_invalid_source_identity(field_name: str, value: object, message: str) -> None:
    kwargs = _adapter_kwargs()
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=message):
        adapt_dex3_arrays(**kwargs)


def test_canonical_episode_has_exact_mutable_contract_fields() -> None:
    assert tuple(field.name for field in fields(CanonicalEpisode)) == (
        "source_repo_id",
        "source_revision",
        "source_episode_id",
        "source_fps",
        "timestamps",
        "task_indices",
        "observed_root_wxyz",
        "reference_root_wxyz",
        "observed_body_q",
        "desired_body_q",
        "observed_left_hand",
        "observed_right_hand",
        "desired_left_hand",
        "desired_right_hand",
        "video_segments",
    )
    episode = CanonicalEpisode(**_canonical_kwargs())
    episode.observed_body_q[0, 0] = 123.0
    assert episode.observed_body_q[0, 0] == 123.0


@pytest.mark.parametrize("source_fps", [29, 30.1, "30", True, np.int64(30)])
def test_canonical_episode_requires_builtin_integer_30_hz(source_fps: object) -> None:
    kwargs = _canonical_kwargs()
    kwargs["source_fps"] = source_fps

    with pytest.raises(ValueError, match="source_fps must be exactly integer 30"):
        CanonicalEpisode(**kwargs)


def test_source_video_segment_matches_actual_pinned_episode_78_rounding(tmp_path: Path) -> None:
    path = tmp_path / "shared.mp4"
    path.write_bytes(b"pinned")

    segment = SourceVideoSegment(
        source_key=VIDEO_KEYS[0],
        path=path,
        from_timestamp=1033.3666666666668,
        to_timestamp=1045.6666666666667,
        start_frame=31001,
        end_frame=31370,
        frame_count=369,
    )

    assert segment.path == path.resolve()
    assert segment.start_frame == 31001
    assert segment.end_frame == 31370


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"from_timestamp": -1 / 30}, "from_timestamp"),
        ({"to_timestamp": 0.0}, "strictly larger"),
        ({"from_timestamp": 0.01}, "30 Hz frame index"),
        ({"to_timestamp": 3 / 30 + 0.01}, "30 Hz frame index"),
        ({"start_frame": 1}, "start_frame"),
        ({"end_frame": 4}, "end_frame"),
        ({"frame_count": 2}, "frame_count"),
    ],
)
def test_source_video_segment_rejects_invalid_offset_duration_or_count(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "shared.mp4"
    path.write_bytes(b"pinned")
    values: dict[str, object] = {
        "source_key": VIDEO_KEYS[0],
        "path": path,
        "from_timestamp": 0.0,
        "to_timestamp": 3 / 30,
        "start_frame": 0,
        "end_frame": 3,
        "frame_count": 3,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        SourceVideoSegment(**values)


def test_canonical_episode_owns_frozen_deterministic_video_segments(tmp_path: Path) -> None:
    path = tmp_path / "shared.mp4"
    path.write_bytes(b"pinned")
    first = SourceVideoSegment(
        source_key="camera.z",
        path=path,
        from_timestamp=0.0,
        to_timestamp=3 / 30,
        start_frame=0,
        end_frame=3,
        frame_count=3,
    )
    second = SourceVideoSegment(
        source_key="camera.a",
        path=path,
        from_timestamp=0.0,
        to_timestamp=3 / 30,
        start_frame=0,
        end_frame=3,
        frame_count=3,
    )
    caller = {first.source_key: first, second.source_key: second}
    kwargs = _canonical_kwargs()
    kwargs["video_segments"] = caller

    episode = CanonicalEpisode(**kwargs)
    caller.clear()

    assert tuple(episode.video_segments) == ("camera.a", "camera.z")
    with pytest.raises(TypeError):
        episode.video_segments["new"] = first


def test_canonical_episode_rejects_video_segment_count_or_key_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "shared.mp4"
    path.write_bytes(b"pinned")
    segment = SourceVideoSegment(
        source_key="camera.actual",
        path=path,
        from_timestamp=0.0,
        to_timestamp=2 / 30,
        start_frame=0,
        end_frame=2,
        frame_count=2,
    )
    kwargs = _canonical_kwargs()
    kwargs["video_segments"] = {"camera.wrong": segment}

    with pytest.raises(ValueError, match="video segment key"):
        CanonicalEpisode(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("source_repo_id", None, "nonempty"),
        ("source_revision", "f" * 39, "40-character immutable revision"),
        ("source_episode_id", 1.5, "nonnegative integer"),
        ("source_episode_id", True, "nonnegative integer"),
        ("source_episode_id", -5, "nonnegative integer"),
    ],
)
def test_canonical_episode_rejects_invalid_source_identity(
    field_name: str,
    value: object,
    message: str,
) -> None:
    kwargs = _canonical_kwargs()
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=message):
        CanonicalEpisode(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "shape"),
    [
        ("timestamps", (3, 1)),
        ("task_indices", (2,)),
        ("observed_root_wxyz", (3, 3)),
        ("reference_root_wxyz", (2, 4)),
        ("observed_body_q", (3, 28)),
        ("desired_body_q", (2, 29)),
        ("observed_left_hand", (3, 6)),
        ("observed_right_hand", (2, 7)),
        ("desired_left_hand", (3, 8)),
        ("desired_right_hand", (3, 7, 1)),
    ],
)
def test_canonical_episode_rejects_every_wrong_array_shape(field_name: str, shape: tuple[int, ...]) -> None:
    kwargs = _canonical_kwargs()
    dtype = np.int64 if field_name == "task_indices" else np.float64
    kwargs[field_name] = np.zeros(shape, dtype=dtype)

    with pytest.raises(ValueError, match=rf"{field_name} must have shape"):
        CanonicalEpisode(**kwargs)


def test_canonical_episode_rejects_fewer_than_two_timestamps() -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        CanonicalEpisode(**_canonical_kwargs(n=1))


@pytest.mark.parametrize("timestamps", [np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.2, 0.1])])
def test_canonical_episode_requires_strictly_increasing_timestamps(timestamps: np.ndarray) -> None:
    kwargs = _canonical_kwargs()
    kwargs["timestamps"] = timestamps

    with pytest.raises(ValueError, match="strictly increasing"):
        CanonicalEpisode(**kwargs)


@pytest.mark.parametrize(
    "task_indices",
    [
        np.array([0.0, 1.0, 2.0]),
        np.array([False, False, True]),
        np.array([0, -1, 2], dtype=np.int64),
        np.array([0, 1, np.iinfo(np.uint64).max], dtype=np.uint64),
        [0, 1.0, 2],
    ],
)
def test_canonical_episode_rejects_noninteger_or_nonsensical_task_values(task_indices: object) -> None:
    kwargs = _canonical_kwargs()
    kwargs["task_indices"] = task_indices

    with pytest.raises(ValueError, match="task_indices must contain nonnegative int64-compatible integers"):
        CanonicalEpisode(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    [
        "timestamps",
        "observed_root_wxyz",
        "reference_root_wxyz",
        "observed_body_q",
        "desired_body_q",
        "observed_left_hand",
        "observed_right_hand",
        "desired_left_hand",
        "desired_right_hand",
    ],
)
def test_canonical_episode_rejects_nonfinite_values_in_every_float_array(field_name: str) -> None:
    kwargs = _canonical_kwargs()
    value = np.array(kwargs[field_name], copy=True)
    value.flat[-1] = np.nan
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=rf"{field_name}.*finite"):
        CanonicalEpisode(**kwargs)


def test_canonical_episode_copies_noncontiguous_callers_into_owned_storage() -> None:
    kwargs = _canonical_kwargs()
    caller_arrays: dict[str, np.ndarray] = {}
    for field_name in CANONICAL_ARRAY_FIELDS:
        value = np.asarray(kwargs[field_name])
        storage = np.empty((value.shape[0] * 2, *value.shape[1:]), dtype=value.dtype)
        storage[::2] = value
        storage[1::2] = value
        view = storage[::2]
        caller_arrays[field_name] = view
        kwargs[field_name] = view

    episode = CanonicalEpisode(**kwargs)
    snapshots = {name: getattr(episode, name).copy() for name in CANONICAL_ARRAY_FIELDS}
    for value in caller_arrays.values():
        value[...] = 99

    for field_name, expected in snapshots.items():
        array = getattr(episode, field_name)
        np.testing.assert_array_equal(array, expected)
        assert array.flags.owndata
        assert array.flags.c_contiguous
        assert array.flags.writeable


class TorchLike:
    def __init__(self, value: object) -> None:
        self._value = np.asarray(value)

    def detach(self) -> "TorchLike":
        return self

    def cpu(self) -> "TorchLike":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class FakeDataset:
    def __init__(
        self,
        rows: list[dict[str, object]],
        *,
        total_episodes: object = 3,
        fps: object = 30,
        features: object | None = None,
        dataset_revision: object = "b" * 40,
        meta_revision: object = "b" * 40,
    ) -> None:
        if features is None:
            features = {
                "observation.state": {"names": [list(FEATURE_NAMES)]},
                "action": {"names": [list(FEATURE_NAMES)]},
            }
        self.meta = type(
            "FakeMeta",
            (),
            {
                "total_episodes": total_episodes,
                "fps": fps,
                "features": features,
                "revision": meta_revision,
            },
        )()
        self.revision = dataset_revision
        self.hf_dataset = rows
        self.top_level_iteration_attempts = 0

    def __iter__(self):
        self.top_level_iteration_attempts += 1
        raise AssertionError("top-level LeRobotDataset iteration may decode video or task strings")


VIDEO_KEYS = (
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
    "observation.images.cam_right_high",
)


def _metadata_info(**changes: object) -> dict[str, object]:
    info: dict[str, object] = {
        "codebase_version": "v3.0",
        "chunks_size": 1000,
        "fps": 30,
        "total_episodes": 3,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "names": [list(FEATURE_NAMES)]},
            "action": {"dtype": "float32", "names": [list(FEATURE_NAMES)]},
            "timestamp": {"dtype": "float32"},
            "task_index": {"dtype": "int64"},
            "frame_index": {"dtype": "int64"},
            "episode_index": {"dtype": "int64"},
            **{key: {"dtype": "video"} for key in VIDEO_KEYS},
        },
    }
    info.update(changes)
    return info


def _episode_row(
    episode_id: object = 1,
    *,
    data_chunk_index: object = 0,
    data_file_index: object = 0,
    video_locations: dict[str, tuple[object, object]] | None = None,
    length: object = 2,
    video_timestamps: dict[str, tuple[object, object]] | None = None,
) -> dict[str, object]:
    locations = {key: (0, 0) for key in VIDEO_KEYS} if video_locations is None else video_locations
    timestamps = (
        {key: (0.0, float(length) / 30.0) for key in VIDEO_KEYS}
        if video_timestamps is None and isinstance(length, int) and not isinstance(length, bool)
        else video_timestamps
    )
    row: dict[str, object] = {
        "episode_index": episode_id,
        "data/chunk_index": data_chunk_index,
        "data/file_index": data_file_index,
        "length": length,
    }
    for video_key in VIDEO_KEYS:
        chunk_index, file_index = locations[video_key]
        row[f"videos/{video_key}/chunk_index"] = chunk_index
        row[f"videos/{video_key}/file_index"] = file_index
        from_timestamp, to_timestamp = (0.0, 2 / 30) if timestamps is None else timestamps[video_key]
        row[f"videos/{video_key}/from_timestamp"] = from_timestamp
        row[f"videos/{video_key}/to_timestamp"] = to_timestamp
    return row


def _v3_data_rows(episode_id: int = 1, n: int = 2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame_index in range(n):
        observed = (np.arange(28, dtype=np.float32) + frame_index * 100).tolist()
        rows.append(
            {
                "observation.state": observed,
                "action": (np.asarray(observed) + 1000).tolist(),
                "timestamp": frame_index / 30,
                "task_index": frame_index,
                "frame_index": frame_index,
                "episode_index": episode_id,
            }
        )
    return rows


class FakeSnapshotDownloader:
    def __init__(
        self,
        *,
        info: dict[str, object] | None = None,
        raw_info: str | None = None,
        episode_rows: list[dict[str, object]] | None = None,
        raw_episode_metadata: bytes | None = None,
        write_episode_metadata: bool = True,
        data_rows: list[dict[str, object]] | None = None,
        raw_data: bytes | None = None,
        missing_patterns: tuple[str, ...] = (),
        wrong_return_call: int | None = None,
        dataset_path: str = ".",
    ) -> None:
        self.info = _metadata_info() if info is None else info
        self.raw_info = raw_info
        self.episode_rows = [_episode_row()] if episode_rows is None else episode_rows
        self.raw_episode_metadata = raw_episode_metadata
        self.write_episode_metadata = write_episode_metadata
        self.data_rows = _v3_data_rows() if data_rows is None else data_rows
        self.raw_data = raw_data
        self.missing_patterns = missing_patterns
        self.wrong_return_call = wrong_return_call
        self.dataset_path = dataset_path
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        allow_patterns = kwargs["allow_patterns"]
        dataset_root = local_dir if self.dataset_path == "." else local_dir / self.dataset_path
        prefix = "" if self.dataset_path == "." else f"{self.dataset_path}/"
        if len(self.calls) == 1:
            assert allow_patterns == [f"{prefix}meta/**"]
            info_path = dataset_root / "meta/info.json"
            info_path.parent.mkdir(parents=True, exist_ok=True)
            text = self.raw_info if self.raw_info is not None else json.dumps(self.info)
            info_path.write_text(text)
            if self.write_episode_metadata:
                episodes_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
                episodes_path.parent.mkdir(parents=True, exist_ok=True)
                if self.raw_episode_metadata is not None:
                    episodes_path.write_bytes(self.raw_episode_metadata)
                else:
                    pq.write_table(pa.Table.from_pylist(self.episode_rows), episodes_path)
        else:
            assert isinstance(allow_patterns, list)
            for pattern in allow_patterns:
                if pattern in self.missing_patterns:
                    continue
                path = local_dir / pattern
                path.parent.mkdir(parents=True, exist_ok=True)
                if pattern.startswith(f"{prefix}data/") and pattern.endswith(".parquet"):
                    if self.raw_data is not None:
                        path.write_bytes(self.raw_data)
                    else:
                        pq.write_table(pa.Table.from_pylist(self.data_rows), path)
                else:
                    path.write_bytes(b"pinned")
        if self.wrong_return_call == len(self.calls):
            return str(local_dir.parent / "wrong-snapshot")
        return str(local_dir)


def _load_episode(*args: object, snapshot_downloader: FakeSnapshotDownloader | None = None, **kwargs: object):
    downloader = FakeSnapshotDownloader() if snapshot_downloader is None else snapshot_downloader
    return load_dex3_episode(*args, snapshot_downloader=downloader, **kwargs)


def _source_spec(**changes: object) -> SourceSpec:
    values = {
        "approved": True,
        "repo_id": "unitreerobotics/synthetic-dex3",
        "revision": "b" * 40,
        "dataset_path": ".",
        "episode_count": 3,
        "episodes": (1,),
        "primary_camera": "observation.images.primary",
        "camera_map": {"observation.images.primary": "observation.images.ego_view"},
        "label": "dex3",
    }
    values.update(changes)
    return SourceSpec(**values)


def _corrupted_unpinned_source_spec() -> SourceSpec:
    source_spec = _source_spec()
    object.__setattr__(source_spec, "episode_count", None)
    return source_spec


def _rows(episode_id: int = 1, n: int = 2) -> list[dict[str, object]]:
    rows = []
    for frame_index in range(n):
        observed = np.arange(28, dtype=np.float32) + frame_index * 100
        desired = observed + 1000
        rows.append(
            {
                "observation.state": TorchLike(observed),
                "action": TorchLike(desired),
                "timestamp": TorchLike(frame_index / 30),
                "task_index": TorchLike(np.int32(frame_index)),
                "frame_index": TorchLike(np.int64(frame_index)),
                "episode_index": TorchLike(np.int64(episode_id)),
            }
        )
    return rows


def _recording_factory(dataset: FakeDataset, calls: list[dict[str, object]]):
    def factory(**kwargs: object) -> FakeDataset:
        calls.append(kwargs)
        return dataset

    return factory


EXPECTED_EPISODE_PATHS = [
    "data/chunk-000/file-000.parquet",
    "videos/observation.images.cam_left_high/chunk-000/file-000.mp4",
    "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
    "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
    "videos/observation.images.cam_right_high/chunk-000/file-000.mp4",
]


def test_direct_v3_reader_supports_pinned_nested_dataset_root_without_videos(tmp_path: Path) -> None:
    dataset_path = "G1_WB_Dex5_Pickup_Pillow"
    data_rows = [
        {
            "observation.state.robot_q_current": [0.0] * 36,
            "action.robot_q_desired": [1.0] * 36,
            "observation.state.hand_state": [2.0] * 12,
            "action.hand_cmd": [3.0] * 12,
            "timestamp": frame_index / 30,
            "task_index": 0,
            "frame_index": frame_index,
            "episode_index": 1,
        }
        for frame_index in range(2)
    ]
    info = _metadata_info(
        features={
            "observation.state.robot_q_current": {"dtype": "float32"},
            "action.robot_q_desired": {"dtype": "float32"},
            "observation.state.hand_state": {"dtype": "float32"},
            "action.hand_cmd": {"dtype": "float32"},
            "timestamp": {"dtype": "float32"},
            "task_index": {"dtype": "int64"},
            "frame_index": {"dtype": "int64"},
            "episode_index": {"dtype": "int64"},
            "observation.images.cam_0": {"dtype": "video"},
            "observation.images.cam_1": {"dtype": "video"},
        }
    )
    downloader = FakeSnapshotDownloader(
        info=info,
        episode_rows=[
            {
                "episode_index": 1,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "length": 2,
                "videos/observation.images.cam_0/chunk_index": 0,
                "videos/observation.images.cam_0/file_index": 0,
                "videos/observation.images.cam_0/from_timestamp": 0.0,
                "videos/observation.images.cam_0/to_timestamp": 2 / 30,
                "videos/observation.images.cam_1/chunk_index": 0,
                "videos/observation.images.cam_1/file_index": 0,
                "videos/observation.images.cam_1/from_timestamp": 0.0,
                "videos/observation.images.cam_1/to_timestamp": 2 / 30,
            }
        ],
        data_rows=data_rows,
        dataset_path=dataset_path,
    )
    source = _source_spec(dataset_path=dataset_path)
    schema = V3DataSchema(
        float_vector_columns=(
            "observation.state.robot_q_current",
            "action.robot_q_desired",
            "observation.state.hand_state",
            "action.hand_cmd",
        ),
        float_scalar_columns=("timestamp",),
        integer_scalar_columns=("task_index", "frame_index", "episode_index"),
    )

    dataset = load_pinned_v3_episode(
        source,
        1,
        cache_base=tmp_path,
        snapshot_downloader=downloader,
        schema=schema,
        download_videos=False,
    )

    scoped_root = revision_scoped_root(tmp_path, source.repo_id, source.revision)
    assert dataset.root == scoped_root / dataset_path
    assert downloader.calls[0]["allow_patterns"] == [f"{dataset_path}/meta/**"]
    assert downloader.calls[1]["allow_patterns"] == [f"{dataset_path}/data/chunk-000/file-000.parquet"]
    assert tuple(dataset.hf_dataset[0]) == schema.columns
    assert dict(dataset.video_segments) == {}


def test_direct_v3_reader_rejects_invalid_or_overlapping_schema_columns() -> None:
    with pytest.raises(ValueError, match="column names must be unique"):
        V3DataSchema(
            float_vector_columns=("duplicate",),
            float_scalar_columns=("duplicate",),
            integer_scalar_columns=("episode_index",),
        )


def test_loader_default_reads_prefetched_v3_data_directly_without_v21_lerobot(tmp_path: Path) -> None:
    downloader = FakeSnapshotDownloader(data_rows=[*_v3_data_rows(0), *_v3_data_rows(1)])

    episode = load_dex3_episode(
        _source_spec(),
        1,
        root=tmp_path,
        snapshot_downloader=downloader,
    )

    assert len(downloader.calls) == 2
    assert downloader.calls[1]["allow_patterns"] == EXPECTED_EPISODE_PATHS
    np.testing.assert_array_equal(episode.timestamps, [0.0, 1 / 30])
    np.testing.assert_array_equal(episode.task_indices, [0, 1])
    np.testing.assert_array_equal(episode.observed_root_wxyz, [[1.0, 0.0, 0.0, 0.0]] * 2)
    np.testing.assert_array_equal(episode.reference_root_wxyz, [[1.0, 0.0, 0.0, 0.0]] * 2)
    np.testing.assert_array_equal(episode.observed_left_hand[1], np.arange(100, 128)[14:21])
    np.testing.assert_array_equal(episode.desired_right_hand[1], np.arange(1100, 1128)[21:28])
    assert tuple(episode.video_segments) == tuple(sorted(VIDEO_KEYS))
    for source_key, segment in episode.video_segments.items():
        assert segment.source_key == source_key
        assert segment.path.is_absolute()
        assert segment.path.is_file()
        assert segment.from_timestamp == 0.0
        assert segment.to_timestamp == 2 / 30
        assert segment.start_frame == 0
        assert segment.end_frame == segment.frame_count == 2


def test_loader_default_rejects_corrupt_selected_v3_data_parquet(tmp_path: Path) -> None:
    downloader = FakeSnapshotDownloader(raw_data=b"not parquet")

    with pytest.raises(ValueError, match="selected v3 data parquet must be readable"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            snapshot_downloader=downloader,
        )


@pytest.mark.parametrize(
    "missing_column",
    ["observation.state", "action", "timestamp", "task_index", "frame_index", "episode_index"],
)
def test_loader_default_rejects_missing_selected_v3_data_columns(
    tmp_path: Path,
    missing_column: str,
) -> None:
    rows = _v3_data_rows()
    for row in rows:
        del row[missing_column]
    downloader = FakeSnapshotDownloader(data_rows=rows)

    with pytest.raises(ValueError, match="selected v3 data parquet.*required columns"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            snapshot_downloader=downloader,
        )


@pytest.mark.parametrize("source_episode_id", [0, 2])
def test_loader_default_rejects_selected_shard_without_requested_episode_rows(
    tmp_path: Path,
    source_episode_id: int,
) -> None:
    downloader = FakeSnapshotDownloader(data_rows=_v3_data_rows(episode_id=source_episode_id))

    with pytest.raises(ValueError, match="selected v3 data contains no rows for episode 1"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            snapshot_downloader=downloader,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("observation.state", ["bad"] * 28),
        ("action", list(range(28))),
        ("timestamp", "0.0"),
        ("task_index", 0.5),
        ("frame_index", 0.5),
        ("episode_index", True),
    ],
)
def test_loader_default_rejects_wrong_selected_v3_data_column_types(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    rows = _v3_data_rows()
    for row in rows:
        row[field_name] = value
    downloader = FakeSnapshotDownloader(data_rows=rows)

    with pytest.raises(ValueError, match=rf"selected v3 data column {field_name} has incompatible type"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            snapshot_downloader=downloader,
        )


def test_loader_prefetches_exact_commit_files_before_cold_cache_factory(tmp_path: Path) -> None:
    downloader = FakeSnapshotDownloader()
    factory_calls: list[dict[str, object]] = []

    def pinned_cold_cache_factory(**kwargs: object) -> FakeDataset:
        factory_calls.append(kwargs)
        assert len(downloader.calls) == 2, "factory was called before exact-revision prefetch completed"
        root = Path(kwargs["root"])
        for relative_path in EXPECTED_EPISODE_PATHS:
            if not (root / relative_path).is_file():
                raise ValueError("InvalidVersion: pinned SHA reached LeRobot's cold-cache compatibility path")
        return FakeDataset(_rows())

    episode = load_dex3_episode(
        _source_spec(),
        1,
        root=tmp_path,
        dataset_factory=pinned_cold_cache_factory,
        snapshot_downloader=downloader,
    )

    assert episode.source_episode_id == 1
    assert len(factory_calls) == 1


def test_loader_prefetch_uses_exact_two_revision_pinned_snapshot_calls(tmp_path: Path) -> None:
    downloader = FakeSnapshotDownloader()

    load_dex3_episode(
        _source_spec(),
        1,
        root=tmp_path,
        dataset_factory=lambda **_: FakeDataset(_rows()),
        snapshot_downloader=downloader,
    )

    scoped_root = (
        tmp_path / f"repo-{hashlib.sha256(b'unitreerobotics/synthetic-dex3').hexdigest()}" / f"revision-{'b' * 40}"
    )
    assert downloader.calls == [
        {
            "repo_id": "unitreerobotics/synthetic-dex3",
            "repo_type": "dataset",
            "revision": "b" * 40,
            "local_dir": scoped_root,
            "allow_patterns": ["meta/**"],
        },
        {
            "repo_id": "unitreerobotics/synthetic-dex3",
            "repo_type": "dataset",
            "revision": "b" * 40,
            "local_dir": scoped_root,
            "allow_patterns": EXPECTED_EPISODE_PATHS,
        },
    ]


def test_loader_prefetch_uses_selected_v3_row_and_per_video_shard_coordinates(tmp_path: Path) -> None:
    video_locations = {
        VIDEO_KEYS[0]: (4, 11),
        VIDEO_KEYS[1]: (4, 12),
        VIDEO_KEYS[2]: (5, 1),
        VIDEO_KEYS[3]: (5, 9),
    }
    downloader = FakeSnapshotDownloader(
        episode_rows=[
            _episode_row(0, data_chunk_index=9, data_file_index=9),
            _episode_row(
                310,
                data_chunk_index=3,
                data_file_index=7,
                video_locations=video_locations,
            ),
        ],
        data_rows=_v3_data_rows(episode_id=310),
    )
    source = _source_spec(episode_count=311, episodes=(310,))

    load_dex3_episode(
        source,
        310,
        root=tmp_path,
        dataset_factory=lambda **_: FakeDataset(
            _rows(episode_id=310),
            total_episodes=311,
        ),
        snapshot_downloader=downloader,
    )

    assert downloader.calls[1]["allow_patterns"] == [
        "data/chunk-003/file-007.parquet",
        "videos/observation.images.cam_left_high/chunk-004/file-011.mp4",
        "videos/observation.images.cam_left_wrist/chunk-004/file-012.mp4",
        "videos/observation.images.cam_right_wrist/chunk-005/file-001.mp4",
        "videos/observation.images.cam_right_high/chunk-005/file-009.mp4",
    ]


@pytest.mark.parametrize(
    "missing_field",
    [
        "codebase_version",
        "chunks_size",
        "fps",
        "total_episodes",
        "data_path",
        "video_path",
        "features",
    ],
)
def test_loader_prefetch_rejects_missing_metadata_fields(tmp_path: Path, missing_field: str) -> None:
    info = _metadata_info()
    del info[missing_field]

    with pytest.raises(ValueError, match=rf"metadata.*{missing_field}"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=FakeSnapshotDownloader(info=info),
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("chunks_size", 0),
        ("chunks_size", -1),
        ("chunks_size", True),
        ("chunks_size", 1000.0),
        ("codebase_version", "v2.1"),
        ("codebase_version", 3),
        ("data_path", None),
        ("video_path", 7),
        ("features", []),
        ("features", {"camera": []}),
        ("features", {"camera": {}}),
        ("features", {"camera": {"dtype": 1}}),
    ],
)
def test_loader_prefetch_rejects_malformed_metadata_fields(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    info = _metadata_info(**{field_name: value})

    with pytest.raises(ValueError, match=rf"metadata.*{field_name}"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=FakeSnapshotDownloader(info=info),
        )


def test_loader_prefetch_rejects_duplicate_json_metadata_keys(tmp_path: Path) -> None:
    raw_info = json.dumps(_metadata_info()).replace(
        '"chunks_size": 1000',
        '"chunks_size": 1000, "chunks_size": 1000',
        1,
    )

    with pytest.raises(ValueError, match="duplicate metadata key.*chunks_size"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=FakeSnapshotDownloader(raw_info=raw_info),
        )


@pytest.mark.parametrize(
    ("field_name", "template"),
    [
        ("data_path", "data/file-{file_index:03d}.parquet"),
        (
            "data_path",
            "data/{chunk_index:03d}/file-{file_index:03d}-{file_index:03d}.parquet",
        ),
        ("data_path", "data/{unknown}/file-{file_index:03d}.parquet"),
        ("data_path", "data/{chunk_index/file-{file_index}.parquet"),
        ("video_path", "videos/{chunk_index:03d}/file-{file_index:03d}.mp4"),
        (
            "video_path",
            "videos/{chunk_index:03d}/{video_key}/{video_key}/file-{file_index:03d}.mp4",
        ),
        ("video_path", "videos/{unknown}/{video_key}/file-{file_index:03d}.mp4"),
        ("video_path", "videos/{chunk_index}/{video_key/{file_index}.mp4"),
    ],
)
def test_loader_prefetch_rejects_missing_duplicate_unknown_or_malformed_template_fields(
    tmp_path: Path,
    field_name: str,
    template: str,
) -> None:
    info = _metadata_info(**{field_name: template})

    with pytest.raises(ValueError, match=rf"metadata {field_name} template"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=FakeSnapshotDownloader(info=info),
        )


@pytest.mark.parametrize(
    ("field_name", "template"),
    [
        ("data_path", "/absolute/chunk-{chunk_index}/file-{file_index}.parquet"),
        ("data_path", "data/../../escape-{chunk_index}-{file_index}.parquet"),
        (
            "video_path",
            "/absolute/{chunk_index}/{video_key}/file-{file_index}.mp4",
        ),
        (
            "video_path",
            "videos/../../../escape-{chunk_index}-{video_key}-{file_index}.mp4",
        ),
    ],
)
def test_loader_prefetch_rejects_absolute_or_escaping_rendered_paths(
    tmp_path: Path,
    field_name: str,
    template: str,
) -> None:
    info = _metadata_info(**{field_name: template})

    with pytest.raises(ValueError, match="must render a safe relative path"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=FakeSnapshotDownloader(info=info),
        )


@pytest.mark.parametrize("unsafe", ["*", "?", "[", "]", "\x00", "\x1f", "\x7f"])
def test_loader_rejects_unsafe_literal_data_template_before_payload_download(
    tmp_path: Path,
    unsafe: str,
) -> None:
    info = _metadata_info(
        data_path=f"data/unsafe{unsafe}/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.parquet"
    )
    downloader = FakeSnapshotDownloader(info=info)

    with pytest.raises(ValueError, match="safe literal relative path"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            snapshot_downloader=downloader,
        )

    assert len(downloader.calls) == 1


@pytest.mark.parametrize("unsafe", ["*", "?", "[", "]", "\x00", "\x1f", "\x7f"])
def test_loader_rejects_unsafe_literal_video_template_before_payload_download(
    tmp_path: Path,
    unsafe: str,
) -> None:
    info = _metadata_info(
        video_path=f"videos/unsafe{unsafe}/{{video_key}}/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.mp4"
    )
    downloader = FakeSnapshotDownloader(info=info)

    with pytest.raises(ValueError, match="safe literal relative path"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            snapshot_downloader=downloader,
        )

    assert len(downloader.calls) == 1


@pytest.mark.parametrize("unsafe", ["*", "?", "[", "]", "\x00", "\x1f", "\x7f"])
def test_loader_rejects_unsafe_interpolated_video_key_before_payload_download(
    tmp_path: Path,
    unsafe: str,
) -> None:
    unsafe_key = f"observation.images.unsafe{unsafe}camera"
    info = _metadata_info()
    features = info["features"]
    del features[VIDEO_KEYS[-1]]
    features[unsafe_key] = {"dtype": "video"}
    episode_row = _episode_row()
    del episode_row[f"videos/{VIDEO_KEYS[-1]}/chunk_index"]
    del episode_row[f"videos/{VIDEO_KEYS[-1]}/file_index"]
    del episode_row[f"videos/{VIDEO_KEYS[-1]}/from_timestamp"]
    del episode_row[f"videos/{VIDEO_KEYS[-1]}/to_timestamp"]
    episode_row[f"videos/{unsafe_key}/chunk_index"] = 0
    episode_row[f"videos/{unsafe_key}/file_index"] = 0
    episode_row[f"videos/{unsafe_key}/from_timestamp"] = 0.0
    episode_row[f"videos/{unsafe_key}/to_timestamp"] = 2 / 30
    downloader = FakeSnapshotDownloader(info=info, episode_rows=[episode_row])

    with pytest.raises(ValueError, match="safe literal relative path"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            snapshot_downloader=downloader,
        )

    assert len(downloader.calls) == 1


def test_loader_prefetch_rejects_missing_episode_metadata_parquet(tmp_path: Path) -> None:
    downloader = FakeSnapshotDownloader(write_episode_metadata=False)

    with pytest.raises(ValueError, match="episode metadata must contain at least one parquet file"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=downloader,
        )


def test_loader_prefetch_rejects_unreadable_episode_metadata_parquet(tmp_path: Path) -> None:
    downloader = FakeSnapshotDownloader(raw_episode_metadata=b"not parquet")

    with pytest.raises(ValueError, match="episode metadata parquet.*readable"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=downloader,
        )


def test_loader_prefetch_rejects_no_selected_episode_metadata_row(tmp_path: Path) -> None:
    downloader = FakeSnapshotDownloader(episode_rows=[_episode_row(0), _episode_row(2)])

    with pytest.raises(ValueError, match="no episode metadata row for selected episode 1"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=downloader,
        )


@pytest.mark.parametrize(
    "episode_rows",
    [
        [_episode_row(1), _episode_row(1)],
        [_episode_row(0), _episode_row(0), _episode_row(1)],
    ],
)
def test_loader_prefetch_rejects_duplicate_episode_metadata_rows(
    tmp_path: Path,
    episode_rows: list[dict[str, object]],
) -> None:
    downloader = FakeSnapshotDownloader(episode_rows=episode_rows)

    with pytest.raises(ValueError, match="duplicate episode_index in episode metadata"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=downloader,
        )


@pytest.mark.parametrize(
    "missing_column",
    [
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "length",
        f"videos/{VIDEO_KEYS[0]}/chunk_index",
        f"videos/{VIDEO_KEYS[0]}/file_index",
        f"videos/{VIDEO_KEYS[0]}/from_timestamp",
        f"videos/{VIDEO_KEYS[0]}/to_timestamp",
    ],
)
def test_loader_prefetch_rejects_missing_episode_metadata_columns(
    tmp_path: Path,
    missing_column: str,
) -> None:
    row = _episode_row()
    del row[missing_column]
    downloader = FakeSnapshotDownloader(episode_rows=[row])

    with pytest.raises(ValueError, match="episode metadata parquet.*required columns"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=downloader,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("episode_index", True),
        ("episode_index", -1),
        ("episode_index", 1.0),
        ("data/chunk_index", True),
        ("data/chunk_index", -1),
        ("data/file_index", "0"),
        (f"videos/{VIDEO_KEYS[0]}/chunk_index", -1),
        (f"videos/{VIDEO_KEYS[0]}/file_index", False),
    ],
)
def test_loader_prefetch_rejects_invalid_episode_metadata_values(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    row = _episode_row()
    row[field_name] = value
    downloader = FakeSnapshotDownloader(episode_rows=[row])

    with pytest.raises(ValueError, match=rf"episode metadata {field_name} must be a nonnegative integer"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=downloader,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("length", 3, "length.*selected data row count"),
        (f"videos/{VIDEO_KEYS[0]}/from_timestamp", -0.1, "from_timestamp"),
        (f"videos/{VIDEO_KEYS[0]}/from_timestamp", float("nan"), "finite nonnegative"),
        (f"videos/{VIDEO_KEYS[0]}/to_timestamp", 0.0, "strictly larger"),
        (f"videos/{VIDEO_KEYS[0]}/to_timestamp", 0.07, "30 Hz frame index"),
    ],
)
def test_loader_rejects_invalid_video_interval_metadata(
    tmp_path: Path,
    field_name: str,
    value: object,
    message: str,
) -> None:
    row = _episode_row()
    row[field_name] = value

    with pytest.raises(ValueError, match=message):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            snapshot_downloader=FakeSnapshotDownloader(episode_rows=[row]),
        )


def test_loader_accepts_pinned_ep155_camera_local_video_shard_offsets(tmp_path: Path) -> None:
    expected_offsets = {
        VIDEO_KEYS[0]: (3563, 4013),
        VIDEO_KEYS[1]: (63971, 64421),
        VIDEO_KEYS[2]: (7246, 7696),
        VIDEO_KEYS[3]: (63971, 64421),
    }
    timestamps = {
        source_key: (start_frame / 30, end_frame / 30)
        for source_key, (start_frame, end_frame) in expected_offsets.items()
    }
    source = _source_spec(episode_count=156, episodes=(155,))
    downloader = FakeSnapshotDownloader(
        info=_metadata_info(total_episodes=156),
        episode_rows=[
            _episode_row(
                episode_id=155,
                length=450,
                video_timestamps=timestamps,
            )
        ],
        data_rows=_v3_data_rows(episode_id=155, n=450),
    )

    episode = load_dex3_episode(
        source,
        155,
        root=tmp_path,
        snapshot_downloader=downloader,
    )

    assert tuple(episode.video_segments) == tuple(sorted(VIDEO_KEYS))
    assert {
        source_key: (segment.start_frame, segment.end_frame)
        for source_key, segment in episode.video_segments.items()
    } == expected_offsets
    assert {segment.frame_count for segment in episode.video_segments.values()} == {450}


@pytest.mark.parametrize("missing_path", EXPECTED_EPISODE_PATHS)
def test_loader_prefetch_rejects_missing_materialized_episode_files(
    tmp_path: Path,
    missing_path: str,
) -> None:
    downloader = FakeSnapshotDownloader(missing_patterns=(missing_path,))

    with pytest.raises(ValueError, match=rf"materialized file.*{missing_path}"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=downloader,
        )


@pytest.mark.parametrize("wrong_return_call", [1, 2])
def test_loader_prefetch_rejects_wrong_downloader_return_location(
    tmp_path: Path,
    wrong_return_call: int,
) -> None:
    downloader = FakeSnapshotDownloader(wrong_return_call=wrong_return_call)

    with pytest.raises(ValueError, match="snapshot downloader must return the revision-scoped root"):
        load_dex3_episode(
            _source_spec(),
            1,
            root=tmp_path,
            dataset_factory=lambda **_: pytest.fail("factory must not be called"),
            snapshot_downloader=downloader,
        )


def test_loader_uses_exact_pinned_lerobot_constructor_and_adapts_torch_like_rows() -> None:
    calls: list[dict[str, object]] = []
    rows = _rows()
    dataset = FakeDataset(rows)

    episode = _load_episode(
        _source_spec(),
        1,
        root=Path("/tmp/pinned-dex3"),
        dataset_factory=_recording_factory(dataset, calls),
    )

    assert calls == [
        {
            "repo_id": "unitreerobotics/synthetic-dex3",
            "root": Path("/tmp/pinned-dex3")
            / f"repo-{hashlib.sha256(b'unitreerobotics/synthetic-dex3').hexdigest()}"
            / f"revision-{'b' * 40}",
            "episodes": [1],
            "revision": "b" * 40,
            "download_videos": True,
            "video_backend": "pyav",
        }
    ]
    assert episode.source_repo_id == "unitreerobotics/synthetic-dex3"
    assert episode.source_revision == "b" * 40
    assert episode.source_episode_id == 1
    assert episode.source_fps == 30
    np.testing.assert_array_equal(episode.timestamps, [0.0, 1 / 30])
    np.testing.assert_array_equal(episode.task_indices, [0, 1])
    np.testing.assert_array_equal(episode.observed_left_hand[1], rows[1]["observation.state"].numpy()[14:21])
    np.testing.assert_array_equal(episode.desired_right_hand[1], rows[1]["action"].numpy()[21:28])
    assert dataset.top_level_iteration_attempts == 0


def test_revision_scoped_root_is_deterministic_repo_and_revision_identity(tmp_path: Path) -> None:
    same = dex3_adapter_module._revision_scoped_root(tmp_path, "org/repo", "a" * 40)
    same_again = dex3_adapter_module._revision_scoped_root(tmp_path, "org/repo", "a" * 40)
    other_revision = dex3_adapter_module._revision_scoped_root(tmp_path, "org/repo", "b" * 40)
    other_repo = dex3_adapter_module._revision_scoped_root(tmp_path, "other/repo", "a" * 40)

    assert same == same_again
    assert len({same, other_revision, other_repo}) == 3
    assert same.parent == other_revision.parent
    assert same.parent != other_repo.parent
    assert same.name != other_revision.name
    assert same.name == other_repo.name


@pytest.mark.parametrize("repo_id", ["../escape", "/absolute/repo", "a/../../outside", "odd ☃/repo"])
def test_revision_scoped_root_encodes_repo_as_one_safe_segment(tmp_path: Path, repo_id: str) -> None:
    scoped = dex3_adapter_module._revision_scoped_root(tmp_path, repo_id, "b" * 40)

    relative = scoped.relative_to(tmp_path)
    assert relative.parts == (
        f"repo-{hashlib.sha256(repo_id.encode()).hexdigest()}",
        f"revision-{'b' * 40}",
    )


def test_default_lerobot_cache_base_matches_hf_lerobot_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))

    assert dex3_adapter_module._default_lerobot_cache_base() == tmp_path


def test_loader_scopes_roots_by_both_repo_identity_and_full_revision(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> FakeDataset:
        calls.append(kwargs)
        revision = kwargs["revision"]
        return FakeDataset(_rows(), dataset_revision=revision, meta_revision=revision)

    sources = (
        _source_spec(revision="a" * 40),
        _source_spec(revision="c" * 40),
        _source_spec(repo_id="other/repository", revision="a" * 40),
    )
    for source in sources:
        _load_episode(source, 1, root=tmp_path, dataset_factory=factory)

    roots = [call["root"] for call in calls]
    assert len(set(roots)) == 3
    for root in roots:
        relative = Path(root).relative_to(tmp_path)
        assert len(relative.parts) == 2
        assert relative.parts[0].startswith("repo-")
        assert relative.parts[1].startswith("revision-")
    assert roots[0].parent == roots[1].parent
    assert roots[0].parent != roots[2].parent
    assert roots[0].name != roots[1].name
    assert roots[0].name == roots[2].name


@pytest.mark.parametrize("repo_id", ["../escape", "/absolute/repo", "a/../../outside", "odd ☃/repo"])
def test_loader_repo_identity_cannot_escape_or_add_segments_to_cache_base(
    tmp_path: Path,
    repo_id: str,
) -> None:
    calls: list[dict[str, object]] = []
    source = _source_spec(repo_id=repo_id)

    _load_episode(
        source,
        1,
        root=tmp_path,
        dataset_factory=_recording_factory(FakeDataset(_rows()), calls),
    )

    relative = Path(calls[0]["root"]).relative_to(tmp_path)
    assert len(relative.parts) == 2
    assert relative.parts[0] == f"repo-{hashlib.sha256(repo_id.encode()).hexdigest()}"
    assert relative.parts[1] == f"revision-{'b' * 40}"


def test_loader_none_root_uses_lerobot_cache_convention_without_importing_lerobot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))

    _load_episode(
        _source_spec(),
        1,
        dataset_factory=_recording_factory(FakeDataset(_rows()), calls),
    )

    assert Path(calls[0]["root"]).relative_to(tmp_path).parts == (
        f"repo-{hashlib.sha256(b'unitreerobotics/synthetic-dex3').hexdigest()}",
        f"revision-{'b' * 40}",
    )


@pytest.mark.parametrize(
    ("dataset_revision", "meta_revision", "message"),
    [
        ("c" * 40, "b" * 40, "dataset.revision must equal pinned source revision"),
        ("b" * 40, "c" * 40, "dataset.meta.revision must equal pinned source revision"),
        (None, "b" * 40, "dataset.revision must equal pinned source revision"),
        ("b" * 40, None, "dataset.meta.revision must equal pinned source revision"),
    ],
)
def test_loader_rejects_dataset_or_metadata_revision_mismatch(
    dataset_revision: object,
    meta_revision: object,
    message: str,
) -> None:
    dataset = FakeDataset(
        _rows(),
        dataset_revision=dataset_revision,
        meta_revision=meta_revision,
    )

    with pytest.raises(ValueError, match=message):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


def test_loader_iterates_only_raw_hf_dataset_rows() -> None:
    dataset = FakeDataset(_rows())

    episode = _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)

    assert episode.source_episode_id == 1
    assert dataset.top_level_iteration_attempts == 0


def test_loader_rejects_missing_raw_hf_dataset() -> None:
    dataset = FakeDataset(_rows())
    del dataset.hf_dataset

    with pytest.raises(ValueError, match="dataset.hf_dataset must be a non-mapping iterable of raw rows"):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize("hf_dataset", [None, "rows", b"rows", {"frame": 0}, 7])
def test_loader_rejects_malformed_raw_hf_dataset(hf_dataset: object) -> None:
    dataset = FakeDataset(_rows())
    dataset.hf_dataset = hf_dataset

    with pytest.raises(ValueError, match="dataset.hf_dataset must be a non-mapping iterable of raw rows"):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize("total_episodes", [2, 4, True, 3.0])
def test_loader_rejects_total_episode_metadata_mismatch(total_episodes: object) -> None:
    dataset = FakeDataset(_rows(), total_episodes=total_episodes)

    with pytest.raises(ValueError, match="total_episodes must exactly match pinned episode_count"):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize("fps", [29, 30.5, True, "30", np.nan])
def test_loader_rejects_non_30_hz_metadata(fps: object) -> None:
    dataset = FakeDataset(_rows(), fps=fps)

    with pytest.raises(ValueError, match="dataset metadata fps must be numeric 30"):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize("feature_key", ["observation.state", "action"])
def test_loader_rejects_mismatched_feature_names(feature_key: str) -> None:
    features = {
        "observation.state": {"names": [list(FEATURE_NAMES)]},
        "action": {"names": [list(FEATURE_NAMES)]},
    }
    features[feature_key] = {"names": [[f"generic_{index}" for index in range(28)]]}
    dataset = FakeDataset(_rows(), features=features)

    with pytest.raises(ValueError, match=rf"{feature_key} metadata must contain the exact Dex3 feature names"):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


def test_loader_accepts_exact_flat_feature_names_if_lerobot_flattens_v3_metadata() -> None:
    features = {
        "observation.state": {"names": list(FEATURE_NAMES)},
        "action": {"names": list(FEATURE_NAMES)},
    }
    dataset = FakeDataset(_rows(), features=features)

    episode = _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)

    assert episode.source_episode_id == 1


@pytest.mark.parametrize(
    "names",
    [
        [list(FEATURE_NAMES), list(FEATURE_NAMES)],
        [[list(FEATURE_NAMES)]],
        [list(FEATURE_NAMES[:-1])],
        [[FEATURE_NAMES[1], FEATURE_NAMES[0], *FEATURE_NAMES[2:]]],
    ],
)
def test_loader_rejects_all_other_nested_feature_name_shapes(names: object) -> None:
    features = {
        "observation.state": {"names": names},
        "action": {"names": [list(FEATURE_NAMES)]},
    }
    dataset = FakeDataset(_rows(), features=features)

    with pytest.raises(ValueError, match="observation.state metadata must contain the exact Dex3 feature names"):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize(
    "features",
    [
        None,
        [],
        {"observation.state": {"names": list(FEATURE_NAMES)}},
        {"observation.state": list(FEATURE_NAMES), "action": {"names": list(FEATURE_NAMES)}},
    ],
)
def test_loader_rejects_undocumented_feature_metadata_schemas(features: object) -> None:
    dataset = FakeDataset(_rows())
    dataset.meta.features = features

    with pytest.raises(ValueError, match="dataset meta.features"):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


def test_loader_rejects_episode_id_remapped_to_zero() -> None:
    dataset = FakeDataset(_rows(episode_id=0))

    with pytest.raises(ValueError, match="source episode_index 0 does not match requested episode 1"):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


def test_loader_rejects_no_requested_rows() -> None:
    dataset = FakeDataset([])

    with pytest.raises(ValueError, match="no rows for requested episode 1"):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize(
    ("frame_indices", "message"),
    [
        ([0, 0], "expected frame_index 1, got 0"),
        ([0, 2], "expected frame_index 1, got 2"),
        ([1, 0], "expected frame_index 0, got 1"),
    ],
)
def test_loader_rejects_duplicate_gapped_or_out_of_order_frame_indices(
    frame_indices: list[int],
    message: str,
) -> None:
    rows = _rows()
    for row, frame_index in zip(rows, frame_indices, strict=True):
        row["frame_index"] = frame_index
    dataset = FakeDataset(rows)

    with pytest.raises(ValueError, match=message):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize(
    "missing_key",
    ["observation.state", "action", "timestamp", "task_index", "frame_index", "episode_index"],
)
def test_loader_requires_every_documented_row_key(missing_key: str) -> None:
    rows = _rows()
    del rows[0][missing_key]
    dataset = FakeDataset(rows)

    with pytest.raises(ValueError, match=rf"row 0 is missing required key {missing_key}"):
        _load_episode(_source_spec(), 1, dataset_factory=lambda **_: dataset)


@pytest.mark.parametrize(
    ("source_spec", "episode_id", "message"),
    [
        (object(), 1, "source_spec must be an approved SourceSpec"),
        (_source_spec(approved=False), 1, "source_spec must be an approved SourceSpec"),
        (_corrupted_unpinned_source_spec(), 1, "pinned episode_count"),
        (_source_spec(), True, "episode_id must be a nonnegative integer"),
        (_source_spec(), -1, "episode_id must be a nonnegative integer"),
        (_source_spec(), 3, "smaller than pinned episode_count"),
        (_source_spec(), 0, "selected in source_spec.episodes"),
    ],
)
def test_loader_rejects_unapproved_unpinned_or_ineligible_sources(
    source_spec: object,
    episode_id: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _load_episode(source_spec, episode_id, dataset_factory=lambda **_: FakeDataset(_rows()))
