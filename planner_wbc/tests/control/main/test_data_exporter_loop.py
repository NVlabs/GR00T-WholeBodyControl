import glob
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    from decoupled_wbc.control.main.teleop.run_g1_data_exporter import Gr00tDataCollector
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from decoupled_wbc.data.constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features
except ModuleNotFoundError as e:
    if "No module named 'rclpy'" in str(e):
        pytestmark = pytest.mark.skip(reason="ROS (rclpy) is not installed")
    else:
        raise e


import json

# How does mocking ROS work?
#
# This test file uses mocking to simulate a ROS environment without requiring actual ROS hardware:
#
# 1. ros_ok_side_effect: Controls how long the ROS loop runs by returning a sequence of
#    True/False values. [True, True, False] means "run for 2 iterations then stop"
#
# 2. MockROSMsgSubscriber: Simulates sensors (camera/state) by returning pre-defined data:
#
# 3. MockKeyboardListenerSubscriber: Simulates user input:
#    - 'c' = start/stop recording
#    - 'd' = discard episode
#    - KeyboardInterrupt = simulate Ctrl+C
#    - None = no input
#
# 4. MockROSEnvironment: A context manager that patches all ROS dependencies to use our mocks,
#    allowing us to test ROS-dependent code without actual ROS running.


class MockROSMsgSubscriber:
    def __init__(self, return_value: list[dict]):
        self.return_value = return_value
        self.counter = 0

    def get_image(self):
        if self.counter < len(self.return_value):
            self.counter += 1
            return self.return_value[self.counter - 1]
        else:
            return None

    def get_msg(self):
        if self.counter < len(self.return_value):
            self.counter += 1
            return self.return_value[self.counter - 1]
        else:
            return None


class MockKeyboardListenerSubscriber:
    def __init__(self, return_value: list[str]):
        self.return_value = return_value
        self.counter = 0

    def get_keyboard_input(self):
        return self.return_value[self.counter]

    def read_msg(self):
        if self.counter < len(self.return_value):
            result = self.return_value[self.counter]
            if isinstance(result, KeyboardInterrupt):
                raise result
            self.counter += 1
            return result
        return None


class MockROSEnvironment:
    """Context manager for mocking ROS environment and subscribers."""

    def __init__(self, ok_side_effect, keyboard_listener, img_subscriber, state_subscriber):
        self.ok_side_effect = ok_side_effect
        self.keyboard_listener = keyboard_listener
        self.img_subscriber = img_subscriber
        self.state_subscriber = state_subscriber
        self.patches = []

    def __enter__(self):
        self.patches = [
            patch("rclpy.init"),
            patch("rclpy.create_node"),
            patch("rclpy.spin"),
            patch("rclpy.ok", side_effect=self.ok_side_effect),
            patch("rclpy.shutdown"),
            patch(
                "decoupled_wbc.control.main.teleop.run_g1_data_exporter.KeyboardListenerSubscriber",
                return_value=self.keyboard_listener,
            ),
            patch(
                "decoupled_wbc.control.main.teleop.run_g1_data_exporter.ROSImgMsgSubscriber",
                return_value=self.img_subscriber,
            ),
            patch(
                "decoupled_wbc.control.main.teleop.run_g1_data_exporter.ROSMsgSubscriber",
                return_value=self.state_subscriber,
            ),
        ]

        for p in self.patches:
            p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in reversed(self.patches):
            p.stop()
        return False


def verify_parquet_files_exist(file_path: str, num_episodes: int):
    parquet_files = glob.glob(os.path.join(file_path, "data/chunk-*/episode_*.parquet"))
    assert (
        len(parquet_files) == num_episodes
    ), f"Expected {num_episodes} parquet files, but found {len(parquet_files)}"


def verify_video_files_exist(file_path: str, observation_keys: list[str], num_episodes: int):
    for observation_key in observation_keys:
        video_files = glob.glob(
            os.path.join(file_path, f"videos/chunk-*/{observation_key}/episode_*.mp4")
        )
        assert (
            len(video_files) == num_episodes
        ), f"Expected {num_episodes} video files, but found {len(video_files)}"


def verify_metadata_files(file_path: str):
    files_to_check = ["episodes.jsonl", "info.json", "tasks.jsonl", "modality.json"]
    for file in files_to_check:
        assert os.path.exists(os.path.join(file_path, "meta", file)), f"meta/{file} not created"


@pytest.fixture
def lerobot_features():
    robot_model = instantiate_g1_robot_model()
    return get_dataset_features(robot_model)


@pytest.fixture
def modality_config():
    return {
        "state": {"feature1": {"start": 0, "end": 4}, "feature2": {"start": 4, "end": 9}},
        "action": {"feature1": {"start": 0, "end": 4}, "feature2": {"start": 4, "end": 9}},
        "video": {"rs_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def _get_image_stream_data(episode_length: int, frame_rate: int, img_height: int, img_width: int):
    return [
        {
            "image": np.zeros((img_height, img_width, 3), dtype=np.uint8),
            "timestamp": (i * 1 / frame_rate),
        }
        for i in range(episode_length)
    ]


def _get_state_act_stream_data(
    episode_length: int, frame_rate: int, state_dim: int, action_dim: int
):
    return [
        {
            "q": np.zeros(state_dim),
            "action": np.zeros(action_dim),
            "timestamp": (i * 1 / frame_rate),
            "navigate_command": np.zeros(3, dtype=np.float64),
            "base_height_command": 0.0,
            "wrist_pose": np.zeros(14, dtype=np.float64),
            "action.eef": np.zeros(14, dtype=np.float64),
        }
        for i in range(episode_length)
    ]


def test_control_loop_happy_path_workflow(lerobot_features, modality_config):
    """
    This test records a single episode and saves it to disk.
    """
    episode_length = 10
    frame_rate = 20
    img_stream_data = _get_image_stream_data(
        episode_length, frame_rate, RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH
    )
    robot_model = instantiate_g1_robot_model()
    state_act_stream_data = _get_state_act_stream_data(
        episode_length, frame_rate, robot_model.num_joints, robot_model.num_joints
    )

    keyboard_sub_output = [None for _ in range(episode_length)]
    keyboard_sub_output[0] = "c"  # Start recording
    keyboard_sub_output[-1] = "c"  # Stop recording and save

    # --------- Save the first episode ---------
    mock_img_sub = MockROSMsgSubscriber(img_stream_data)
    mock_state_sub = MockROSMsgSubscriber(state_act_stream_data)
    mock_keyboard_listner = MockKeyboardListenerSubscriber(keyboard_sub_output)

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_dir = os.path.join(temp_dir, "dataset")

        data_exporter = Gr00tDataExporter.create(
            save_root=dataset_dir,
            fps=frame_rate,
            features=lerobot_features,
            modality_config=modality_config,
            task="test",
        )

        ros_ok_side_effect = [True] * (episode_length + 1) + [False]
        with MockROSEnvironment(
            ros_ok_side_effect, mock_keyboard_listner, mock_img_sub, mock_state_sub
        ):
            data_collector = Gr00tDataCollector(
                camera_topic_name="mock_camera_topic",
                state_topic_name="mock_state_topic",
                data_exporter=data_exporter,
                frequency=frame_rate,
            )

            # mocking to avoid actual sleeping
            data_collector.rate = MagicMock()

            data_collector.run()

        verify_parquet_files_exist(dataset_dir, 1)
        verify_video_files_exist(dataset_dir, data_exporter.meta.video_keys, 1)
        verify_metadata_files(dataset_dir)

        # --------- Save the second episode ---------
        # we reset the mock subscribers and re-run the control loop
        # This immitates the case where the user starts recording a new episode on an existing dataset
        mock_img_sub = MockROSMsgSubscriber(img_stream_data)
        mock_state_sub = MockROSMsgSubscriber(state_act_stream_data)
        ros_ok_side_effect = [True] * (episode_length + 1) + [False]
        mock_keyboard_listner = MockKeyboardListenerSubscriber(keyboard_sub_output)
        with MockROSEnvironment(
            ros_ok_side_effect, mock_keyboard_listner, mock_img_sub, mock_state_sub
        ):
            data_collector = Gr00tDataCollector(
                camera_topic_name="mock_camera_topic",
                state_topic_name="mock_state_topic",
                data_exporter=data_exporter,
                frequency=frame_rate,
            )

            # mocking to avoid actual sleeping
            data_collector.rate = MagicMock()

            data_collector.run()

        # now there should be 2 episodes in the dataset
        verify_parquet_files_exist(dataset_dir, 2)
        verify_video_files_exist(dataset_dir, data_exporter.meta.video_keys, 2)
        verify_metadata_files(dataset_dir)


def test_control_loop_keyboard_interrupt_workflow(lerobot_features, modality_config):
    """
    This test simulates a keyboard interruption in the middle of recording.
    Expected behavior:
    - The episode is saved to disk
    - The episode is marked as discarded
    """
    episode_length = 15
    frame_rate = 20
    img_stream_data = _get_image_stream_data(
        episode_length, frame_rate, RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH
    )
    robot_model = instantiate_g1_robot_model()
    state_act_stream_data = _get_state_act_stream_data(
        episode_length, frame_rate, robot_model.num_joints, robot_model.num_joints
    )

    keyboard_sub_output = [None for _ in range(episode_length)]
    keyboard_sub_output[0] = "c"  # Start recording
    keyboard_sub_output[5] = KeyboardInterrupt()  # keyboard interruption in the middle of recording

    mock_img_sub = MockROSMsgSubscriber(img_stream_data)
    mock_state_sub = MockROSMsgSubscriber(state_act_stream_data)
    mock_keyboard_listener = MockKeyboardListenerSubscriber(keyboard_sub_output)

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_dir = os.path.join(temp_dir, "dataset")

        data_exporter = Gr00tDataExporter.create(
            save_root=dataset_dir,
            fps=frame_rate,
            features=lerobot_features,
            modality_config=modality_config,
            task="test",
        )

        ros_ok_side_effect = [True] * episode_length + [False]
        with MockROSEnvironment(
            ros_ok_side_effect, mock_keyboard_listener, mock_img_sub, mock_state_sub
        ):
            data_collector = Gr00tDataCollector(
                camera_topic_name="mock_camera_topic",
                state_topic_name="mock_state_topic",
                data_exporter=data_exporter,
                frequency=frame_rate,
            )

            data_collector.rate = MagicMock()
            # try:
            data_collector.run()
            # except KeyboardInterrupt:
            #     pass

        verify_parquet_files_exist(dataset_dir, 1)
        verify_video_files_exist(dataset_dir, data_exporter.meta.video_keys, 1)
        verify_metadata_files(dataset_dir)

        # verify that the episode is marked as discarded
        ep_info = json.load(open(os.path.join(dataset_dir, "meta", "info.json")))
        assert ep_info["discarded_episode_indices"][0] == 0
        assert ep_info["total_frames"] == 5
        assert ep_info["total_episodes"] == 1


def test_discarded_episode_workflow(lerobot_features, modality_config):
    """
    This test simulates a case where the user discards an episode in the middle of recording.
    Expected behavior:
    - Record 3 episodes, discard episode 0 and 2
    - There should be 3 episodes saved to disk
    - Episode 0 and 2 should be flagged as discarded
    """
    episode_length = 17
    frame_rate = 20
    robot_model = instantiate_g1_robot_model()
    state_dim = robot_model.num_joints
    action_dim = robot_model.num_joints
    img_stream_data = _get_image_stream_data(
        episode_length, frame_rate, RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH
    )
    state_act_stream_data = _get_state_act_stream_data(
        episode_length, frame_rate, state_dim, action_dim
    )

    keyboard_sub_output = [None for _ in range(episode_length)]
    keyboard_sub_output[0] = "c"  # Start recording episode index 0
    keyboard_sub_output[5] = "x"  # Discard episode index 0
    keyboard_sub_output[7] = "c"  # Start recording episode index 1
    keyboard_sub_output[10] = "c"  # stop recording and save episode index 1
    keyboard_sub_output[12] = "c"  # start recording episode index 2
    keyboard_sub_output[15] = "x"  # discard episode index 2

    mock_img_sub = MockROSMsgSubscriber(img_stream_data)
    mock_state_sub = MockROSMsgSubscriber(state_act_stream_data)
    mock_keyboard_listener = MockKeyboardListenerSubscriber(keyboard_sub_output)

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_dir = os.path.join(temp_dir, "dataset")

        data_exporter = Gr00tDataExporter.create(
            save_root=dataset_dir,
            fps=frame_rate,
            features=lerobot_features,
            modality_config=modality_config,
            task="test",
        )

        ros_ok_side_effect = [True] * episode_length + [False]
        with MockROSEnvironment(
            ros_ok_side_effect, mock_keyboard_listener, mock_img_sub, mock_state_sub
        ):
            data_collector = Gr00tDataCollector(
                camera_topic_name="mock_camera_topic",
                state_topic_name="mock_state_topic",
                data_exporter=data_exporter,
                frequency=frame_rate,
            )

            data_collector.rate = MagicMock()
            try:
                data_collector.run()
            except Exception:
                pass

            # vrify if the episode is marked as discarded
            ep_info = json.load(open(os.path.join(dataset_dir, "meta", "info.json")))
            assert len(ep_info["discarded_episode_indices"]) == 2
            assert ep_info["discarded_episode_indices"][0] == 0
            assert ep_info["discarded_episode_indices"][1] == 2

            # verify that all episodes are saved regardless of being discarded
            verify_parquet_files_exist(dataset_dir, 3)
            verify_video_files_exist(dataset_dir, data_exporter.meta.video_keys, 3)
            verify_metadata_files(dataset_dir)
