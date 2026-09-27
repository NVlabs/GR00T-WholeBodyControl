from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from gear_sonic.data.features_sonic_vla import get_features_sonic_vla
from gear_sonic.data.unitree_conversion import target_frames as target_frames_module
from gear_sonic.data.unitree_conversion.contracts import ResampledEpisode
from gear_sonic.data.unitree_conversion.joint_mapping import G1_MUJOCO_NAMES
from gear_sonic.data.unitree_conversion.target_frames import TargetFrameBuilder

PRODUCTION_G1_JOINT_NAMES = (
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
G1_MESH_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "robot_model"
    / "model_data"
    / "g1"
    / "meshes"
    / "left_hip_pitch_link.STL"
)
GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class FakeRobotModel:
    def __init__(self, joint_names: tuple[str, ...] = PRODUCTION_G1_JOINT_NAMES) -> None:
        self.joint_names = list(joint_names)
        self.num_joints = len(self.joint_names)
        self.num_dofs = len(self.joint_names)
        self.supplemental_info = SimpleNamespace(
            hand_frame_names={"left": "left_wrist_frame", "right": "right_wrist_frame"}
        )
        self.cached_q: np.ndarray | None = None
        self.cache_auto_clip: bool | None = None

    def cache_forward_kinematics(self, q: np.ndarray, auto_clip: bool = True) -> None:
        self.cached_q = q.copy()
        self.cache_auto_clip = auto_clip

    def frame_placement(self, frame_name: str) -> SimpleNamespace:
        assert self.cached_q is not None
        if frame_name == "left_wrist_frame":
            return SimpleNamespace(
                translation=np.array([self.cached_q[0], 2.0, 3.0], dtype=np.float64),
                rotation=np.eye(3, dtype=np.float64),
            )
        if frame_name == "right_wrist_frame":
            return SimpleNamespace(
                translation=np.array([4.0, self.cached_q[1], 6.0], dtype=np.float64),
                rotation=R.from_euler("z", 90.0, degrees=True).as_matrix(),
            )
        raise AssertionError(f"unexpected frame {frame_name!r}")


def _episode(frame_count: int = 3) -> ResampledEpisode:
    source_body_names = tuple(reversed(G1_MUJOCO_NAMES))
    observed_body = np.stack(
        [np.arange(100.0 + frame * 100.0, 129.0 + frame * 100.0) for frame in range(frame_count)]
    )
    desired_body = observed_body + 1000.0
    observed_left = np.stack([np.arange(10.0 + frame * 10.0, 17.0 + frame * 10.0) for frame in range(frame_count)])
    observed_right = observed_left + 10.0
    desired_left = observed_left + 200.0
    desired_right = observed_right + 300.0
    observed_root = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frame_count, 1))
    observed_root[1] = R.from_euler("x", 90.0, degrees=True).as_quat(scalar_first=True)
    reference_root = np.tile(
        R.from_euler("z", 30.0, degrees=True).as_quat(scalar_first=True),
        (frame_count, 1),
    )
    return ResampledEpisode(
        source_repo_id="synthetic/dex3",
        source_revision="a" * 40,
        source_episode_id=5,
        body_joint_names=source_body_names,
        task_indices=np.arange(frame_count, dtype=np.int64),
        task_texts=("pour", "place", "pour")[:frame_count],
        observed_root_wxyz=observed_root,
        reference_root_wxyz=reference_root,
        observed_body_q=observed_body,
        desired_body_q=desired_body,
        desired_body_velocity=np.zeros_like(desired_body),
        observed_left_hand=observed_left,
        observed_right_hand=observed_right,
        desired_left_hand=desired_left,
        desired_right_hand=desired_right,
    )


def _token() -> np.ndarray:
    return np.linspace(-0.125, 0.125, 64, dtype=np.float32).copy()


def test_builder_emits_exact_nonvideo_schema_dtypes_shapes_and_owned_arrays() -> None:
    robot_model = FakeRobotModel()
    frame = TargetFrameBuilder(robot_model=robot_model).build(_episode(), frame_index=1, token=_token())
    target_features = get_features_sonic_vla(robot_model)
    expected_keys = {key for key, feature in target_features.items() if feature["dtype"] != "video"} | {
        "timestamp",
        "task",
    }

    assert set(frame) == expected_keys
    for key, feature in target_features.items():
        if feature["dtype"] == "video":
            continue
        value = frame[key]
        assert isinstance(value, np.ndarray), key
        assert value.dtype == np.dtype(feature["dtype"]), key
        assert value.shape == tuple(feature["shape"]), key
        assert value.flags.owndata, key
        assert value.flags.c_contiguous, key
        assert np.isfinite(value).all(), key
    assert isinstance(frame["timestamp"], np.float32)
    assert frame["timestamp"].tobytes() == np.float32(1.0 / 50.0).tobytes()
    assert frame["task"] == "place"


def test_builder_assembles_observed_and_desired_states_by_semantic_joint_name() -> None:
    episode = _episode()
    robot_model = FakeRobotModel()
    frame = TargetFrameBuilder(robot_model=robot_model).build(episode, 0, _token())

    # Pinocchio interleaves the left hand before the right arm and orders each
    # hand index,middle,thumb. The source body order is independently reversed.
    observed_body_by_name = dict(zip(episode.body_joint_names, episode.observed_body_q[0], strict=True))
    desired_body_by_name = dict(zip(episode.body_joint_names, episode.desired_body_q[0], strict=True))
    for name in G1_MUJOCO_NAMES:
        target_index = PRODUCTION_G1_JOINT_NAMES.index(name)
        assert frame["observation.state"][target_index] == observed_body_by_name[name]
        assert frame["action.wbc"][target_index] == desired_body_by_name[name]

    observed_hands_by_name = {
        "left_hand_thumb_0_joint": 10.0,
        "left_hand_thumb_1_joint": 11.0,
        "left_hand_thumb_2_joint": 12.0,
        "left_hand_middle_0_joint": 13.0,
        "left_hand_middle_1_joint": 14.0,
        "left_hand_index_0_joint": 15.0,
        "left_hand_index_1_joint": 16.0,
        "right_hand_thumb_0_joint": 20.0,
        "right_hand_thumb_1_joint": 21.0,
        "right_hand_thumb_2_joint": 22.0,
        "right_hand_index_0_joint": 23.0,
        "right_hand_index_1_joint": 24.0,
        "right_hand_middle_0_joint": 25.0,
        "right_hand_middle_1_joint": 26.0,
    }
    desired_hands_by_name = {
        "left_hand_thumb_0_joint": 210.0,
        "left_hand_thumb_1_joint": 211.0,
        "left_hand_thumb_2_joint": 212.0,
        "left_hand_middle_0_joint": 213.0,
        "left_hand_middle_1_joint": 214.0,
        "left_hand_index_0_joint": 215.0,
        "left_hand_index_1_joint": 216.0,
        "right_hand_thumb_0_joint": 320.0,
        "right_hand_thumb_1_joint": 321.0,
        "right_hand_thumb_2_joint": 322.0,
        "right_hand_index_0_joint": 323.0,
        "right_hand_index_1_joint": 324.0,
        "right_hand_middle_0_joint": 325.0,
        "right_hand_middle_1_joint": 326.0,
    }
    for name, expected in observed_hands_by_name.items():
        target_index = PRODUCTION_G1_JOINT_NAMES.index(name)
        assert frame["observation.state"][target_index] == expected
        assert frame["action.wbc"][target_index] == desired_hands_by_name[name]

    # Raw teleop values remain the unchanged desired DDS arrays.
    np.testing.assert_array_equal(frame["teleop.left_hand_joints"], episode.desired_left_hand[0])
    np.testing.assert_array_equal(frame["teleop.right_hand_joints"], episode.desired_right_hand[0])
    assert frame["teleop.left_hand_joints"].dtype == np.float32
    assert frame["teleop.right_hand_joints"].dtype == np.float32


def test_eef_fk_uses_observed_state_without_silent_joint_clipping() -> None:
    robot_model = FakeRobotModel()
    episode = _episode()
    frame = TargetFrameBuilder(robot_model=robot_model).build(episode, 0, _token())

    np.testing.assert_array_equal(robot_model.cached_q, frame["observation.state"])
    assert not np.array_equal(robot_model.cached_q, frame["action.wbc"])
    assert robot_model.cache_auto_clip is False
    np.testing.assert_allclose(
        frame["observation.eef_state"],
        [128.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 127.0, 6.0, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        atol=1e-15,
    )


def test_root_metadata_gravity_token_and_neutral_fields_are_exact() -> None:
    episode = _episode()
    token = _token()
    frame = TargetFrameBuilder(robot_model=FakeRobotModel()).build(episode, 1, token)

    np.testing.assert_array_equal(frame["observation.root_orientation"], episode.observed_root_wxyz[1])
    np.testing.assert_allclose(frame["observation.projected_gravity"], [0.0, -1.0, 0.0], atol=1e-15)
    np.testing.assert_array_equal(frame["observation.cpp_rotation_offset"], episode.reference_root_wxyz[0])
    np.testing.assert_array_equal(frame["observation.init_base_quat"], episode.observed_root_wxyz[0])
    np.testing.assert_array_equal(frame["action.motion_token"], token.astype(np.float64))
    assert not np.shares_memory(frame["action.motion_token"], token)

    expected_neutral = {
        "teleop.delta_heading": np.zeros(1, dtype=np.float64),
        "teleop.smpl_joints": np.zeros(72, dtype=np.float32),
        "teleop.smpl_pose": np.zeros(63, dtype=np.float32),
        "teleop.body_quat_w": np.array([1, 0, 0, 0], dtype=np.float32),
        "teleop.target_body_orientation": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
        "teleop.left_wrist_joints": np.zeros(3, dtype=np.float32),
        "teleop.right_wrist_joints": np.zeros(3, dtype=np.float32),
        "teleop.stream_mode": np.zeros(1, dtype=np.int32),
        "teleop.planner_mode": np.zeros(1, dtype=np.int32),
        "teleop.planner_movement": np.zeros(3, dtype=np.float32),
        "teleop.planner_facing": np.array([1, 0, 0], dtype=np.float32),
        "teleop.planner_speed": np.array([-1], dtype=np.float32),
        "teleop.planner_height": np.array([-1], dtype=np.float32),
        "teleop.vr_3pt_position": np.zeros(9, dtype=np.float32),
        "teleop.vr_3pt_orientation": np.zeros(18, dtype=np.float32),
    }
    for key, expected in expected_neutral.items():
        np.testing.assert_array_equal(frame[key], expected)
    np.testing.assert_array_equal(frame["teleop.smpl_frame_index"], np.array([1], dtype=np.int64))


@pytest.mark.parametrize("frame_index", [True, -1, 3, 1.5])
def test_builder_rejects_invalid_frame_indices(frame_index: object) -> None:
    with pytest.raises((TypeError, ValueError), match="frame_index"):
        TargetFrameBuilder(robot_model=FakeRobotModel()).build(_episode(), frame_index, _token())


@pytest.mark.parametrize(
    ("token", "message"),
    [
        (list(np.zeros(64, dtype=np.float32)), "numpy.ndarray"),
        (np.zeros(64, dtype=np.float64), "float32"),
        (np.zeros((1, 64), dtype=np.float32), r"\(64,\)"),
        (np.full(64, np.nan, dtype=np.float32), "finite"),
    ],
)
def test_builder_rejects_invalid_tokens(token: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        TargetFrameBuilder(robot_model=FakeRobotModel()).build(_episode(), 0, token)


def test_builder_accepts_encoder_row_views_but_owns_the_float64_output() -> None:
    encoded_batch = np.arange(64, dtype=np.float32).reshape(1, 64)
    token_view = encoded_batch[0]
    assert not token_view.flags.owndata

    output = TargetFrameBuilder(robot_model=FakeRobotModel()).build(_episode(), 0, token_view)[
        "action.motion_token"
    ]

    np.testing.assert_array_equal(output, token_view)
    assert output.dtype == np.float64
    assert output.flags.owndata
    assert not np.shares_memory(output, encoded_batch)


@pytest.mark.parametrize(
    ("source_field", "target_field"),
    [
        ("desired_left_hand", "teleop.left_hand_joints"),
        ("desired_right_hand", "teleop.right_hand_joints"),
    ],
)
def test_builder_rejects_finite_desired_hand_values_that_overflow_float32_without_warning(
    source_field: str,
    target_field: str,
) -> None:
    kwargs = _episode().__dict__.copy()
    desired_hand = getattr(_episode(), source_field).copy()
    desired_hand[0, 0] = 1e300
    kwargs[source_field] = desired_hand
    episode = ResampledEpisode(**kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match=rf"{target_field}.*finite"):
            TargetFrameBuilder(robot_model=FakeRobotModel()).build(episode, 0, _token())


@pytest.mark.parametrize(
    ("robot_model", "message"),
    [
        (FakeRobotModel((*PRODUCTION_G1_JOINT_NAMES[:-1], "unexpected_joint")), "semantic joint set"),
        (
            FakeRobotModel((*PRODUCTION_G1_JOINT_NAMES[:-1], PRODUCTION_G1_JOINT_NAMES[0])),
            "unique",
        ),
        (FakeRobotModel(PRODUCTION_G1_JOINT_NAMES[:-1]), "exactly 43"),
    ],
)
def test_builder_rejects_robot_models_without_the_exact_semantic_joint_contract(
    robot_model: FakeRobotModel,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TargetFrameBuilder(robot_model=robot_model)


def test_builder_requires_both_named_hand_frames() -> None:
    robot_model = FakeRobotModel()
    robot_model.supplemental_info.hand_frame_names = {"left": "left_wrist_frame"}
    with pytest.raises(ValueError, match="hand_frame_names.*left.*right"):
        TargetFrameBuilder(robot_model=robot_model)


def test_default_builder_requests_the_lower_and_upper_body_model(monkeypatch: pytest.MonkeyPatch) -> None:
    robot_model = FakeRobotModel()
    calls: list[str] = []

    def fake_factory(*, waist_location: str) -> FakeRobotModel:
        calls.append(waist_location)
        return robot_model

    monkeypatch.setattr(target_frames_module, "get_g1_robot_model", fake_factory)
    builder = TargetFrameBuilder()

    assert builder.robot_model is robot_model
    assert calls == ["lower_and_upper_body"]


def test_real_g1_model_builds_a_finite_frame_when_assets_are_available() -> None:
    with G1_MESH_PATH.open("rb") as mesh_file:
        if mesh_file.read(len(GIT_LFS_POINTER_PREFIX)) == GIT_LFS_POINTER_PREFIX:
            pytest.skip("G1 mesh is an unmaterialized Git-LFS pointer")

    robot_model = target_frames_module.get_g1_robot_model(waist_location="lower_and_upper_body")

    frame = TargetFrameBuilder(robot_model=robot_model).build(_episode(), 0, _token())

    assert frame["observation.state"].shape == (43,)
    assert np.isfinite(frame["observation.eef_state"]).all()


def test_exporter_registers_episode_tasks_in_first_occurrence_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    # Dataset-wide lexicographic task preregistration belongs to the Task 10
    # deterministic merge. This locks the exporter's episode-local order.
    from gear_sonic.data import exporter as exporter_module

    class FakeMeta:
        total_episodes = 0
        total_frames = 0
        video_keys: list[str] = []
        episodes = {"episode_index": np.array([0])}

        def __init__(self) -> None:
            self.task_ids: dict[str, int] = {}
            self.saved_tasks: list[str] | None = None

        def get_task_index(self, task: str) -> int | None:
            return self.task_ids.get(task)

        def add_task(self, task: str) -> None:
            self.task_ids[task] = len(self.task_ids)

        def save_episode(self, _index, _length, tasks, _stats) -> None:
            self.saved_tasks = list(tasks)

        def get_data_file_path(self, _episode_index: int) -> str:
            return "episode.parquet"

    meta = FakeMeta()
    episode_buffer = {
        "size": 4,
        "task": ["pour", "place", "pour", "grasp"],
        "episode_index": 0,
        "timestamp": [np.float32(index / 50.0) for index in range(4)],
    }
    fake_exporter = SimpleNamespace(
        episode_buffer=episode_buffer,
        meta=meta,
        features={"timestamp": {"dtype": "float32"}},
        root=tmp_path,
        fps=50,
        tolerance_s=1e-4,
        num_episodes=1,
        _wait_image_writer=lambda: None,
        _save_episode_table=lambda _buffer, _index: None,
        encode_episode_videos=lambda _index: {},
        create_episode_buffer=lambda: {},
        create_video_writer=lambda: {},
    )
    (tmp_path / "episode.parquet").touch()
    monkeypatch.setattr(exporter_module, "validate_episode_buffer", lambda *_args: None)
    monkeypatch.setattr(exporter_module, "compute_episode_stats", lambda *_args: {})
    monkeypatch.setattr(
        exporter_module,
        "get_episode_data_index",
        lambda *_args: {"from": SimpleNamespace(numpy=lambda: np.array([0]))},
    )
    monkeypatch.setattr(exporter_module, "check_timestamps_sync", lambda *_args: None)

    exporter_module.Gr00tDataExporter.save_episode(fake_exporter)

    assert list(meta.task_ids) == ["pour", "place", "grasp"]
    assert meta.saved_tasks == ["pour", "place", "grasp"]
