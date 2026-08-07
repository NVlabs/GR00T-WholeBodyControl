import json
from pathlib import Path
import shutil
import tempfile
import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pytest

from decoupled_wbc.data.exporter import DataCollectionInfo, Gr00tDataExporter


@pytest.fixture
def test_features():
    """Fixture providing test features dict."""
    return {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [64, 64, 3],  # Small images for faster tests
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
        },
    }


@pytest.fixture
def test_modality_config():
    return {
        "state": {"feature1": {"start": 0, "end": 4}, "feature2": {"start": 4, "end": 9}},
        "action": {"feature1": {"start": 0, "end": 4}, "feature2": {"start": 4, "end": 9}},
        "video": {"rs_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


@pytest.fixture
def test_data_collection_info():
    return DataCollectionInfo(
        teleoperator_username="test_user",
        support_operator_username="test_user",
        robot_type="test_robot",
        lower_body_policy="test_policy",
        wbc_model_path="test_path",
    )


def get_test_frame(step: int):
    """Generate a test frame with data that varies by step."""
    # Create a simple, small image that will encode quickly
    img = np.ones((64, 64, 3), dtype=np.uint8) * (step % 255)
    # Add a pattern to make each frame unique and verifiable
    img[step % 64, :, :] = 255 - (step % 255)

    return {
        "observation.images.ego_view": img,
        "observation.state": np.ones(8, dtype=np.float32) * step,
        "action": np.ones(8, dtype=np.float32) * step,
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data that's cleaned up after tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir) / "dataset"
    shutil.rmtree(temp_dir)


class TestInterruptAndResume:
    """Test class for simulating interruption and resumption of recording."""

    # Skip this test if ffmpeg is not installed
    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None, reason="ffmpeg not installed, skipping test"
    )
    def test_interrupted_mid_episode(
        self, temp_dir, test_features, test_modality_config, test_data_collection_info
    ):
        """
        Test that simulates a recording session that gets interrupted and then resumes.

        This test uses the actual Gr00tDataExporter implementation with no mocks.
        """
        # Constants for the test
        NUM_EPISODES = 2
        FRAMES_PER_EPISODE = 5

        # Pick a random episode and frame to interrupt at
        interrupt_episode = 1
        interrupt_frame = 3

        print(f"Will interrupt at episode {interrupt_episode}, frame {interrupt_frame}")

        # Track what we've added to verify later
        completed_episodes = []
        frames_added_first_session = 0

        # Initial recording session
        try:
            # Start recording with real Gr00tDataExporter
            exporter1 = Gr00tDataExporter.create(
                save_root=temp_dir,
                fps=30,
                features=test_features,
                modality_config=test_modality_config,
                task="test_task",
                robot_type="test_robot",
                vcodec="libx264",  # Use a common codec that should be available
                data_collection_info=test_data_collection_info,
            )

            # Record episodes until interruption
            for episode in range(NUM_EPISODES):
                for frame in range(FRAMES_PER_EPISODE):
                    # Simulate interruption
                    if episode == interrupt_episode and frame == interrupt_frame:
                        print(f"Simulating interruption at episode {episode}, frame {frame}")
                        raise KeyboardInterrupt("Simulated interruption")

                    # Add frame
                    exporter1.add_frame(get_test_frame(frame))
                    frames_added_first_session += 1

                # Save episode
                exporter1.save_episode()
                completed_episodes.append(episode)

        except KeyboardInterrupt:
            print(f"Recording interrupted at episode {interrupt_episode}, frame {interrupt_frame}")
            print(f"Completed episodes: {completed_episodes}")
            # Don't consolidate since we're interrupted
            pass

        # Verify what was recorded before interruption
        assert len(completed_episodes) == interrupt_episode
        assert (
            frames_added_first_session == interrupt_episode * FRAMES_PER_EPISODE + interrupt_frame
        )

        # Let file system operations complete
        time.sleep(0.5)

        # Resume recording - create a new exporter pointing to the same directory
        exporter2 = Gr00tDataExporter.create(
            save_root=temp_dir,
            fps=30,
            features=test_features,
            modality_config=test_modality_config,
            task="test_task",
            robot_type="test_robot",
            vcodec="libx264",
        )

        # The interrupted episode had frames added but wasn't saved
        # In a real scenario with the current implementation, we need to restart that episode

        # Record all episodes from the beginning
        frames_added_second_session = 0
        episodes_saved_second_session = 0

        for episode in range(NUM_EPISODES):
            for frame in range(FRAMES_PER_EPISODE):
                exporter2.add_frame(get_test_frame(frame))
                frames_added_second_session += 1

            # Save episode
            exporter2.save_episode()
            episodes_saved_second_session += 1

        # Verify the result
        assert frames_added_second_session == NUM_EPISODES * FRAMES_PER_EPISODE
        assert episodes_saved_second_session == NUM_EPISODES

        # Verify actual files were created
        for episode_idx in range(NUM_EPISODES):
            video_path = exporter2.root / exporter2.meta.get_video_file_path(
                episode_idx, "observation.images.ego_view"
            )
            assert video_path.exists(), f"Video file not found: {video_path}"

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None, reason="ffmpeg not installed, skipping test"
    )
    def test_interrupted_after_episode_completion(
        self, temp_dir, test_features, test_modality_config, test_data_collection_info
    ):
        """
        Test specifically for the case when interruption happens after an episode is completed.
        Uses the real Gr00tDataExporter implementation.
        """
        # First session - record 1 complete episode and then interrupt
        exporter1 = Gr00tDataExporter.create(
            save_root=temp_dir,
            fps=30,
            features=test_features,
            modality_config=test_modality_config,
            task="test_task",
            data_collection_info=test_data_collection_info,
            vcodec="libx264",
        )

        # Record 1 complete episode
        for frame in range(5):
            exporter1.add_frame(get_test_frame(frame))
        exporter1.save_episode()

        # Let file system operations complete
        time.sleep(0.5)

        # Verify the first episode was saved
        video_path = exporter1.root / exporter1.meta.get_video_file_path(
            0, "observation.images.ego_view"
        )
        assert video_path.exists(), f"First episode video file not found: {video_path}"

        # Second session - resume and record another episode
        exporter2 = Gr00tDataExporter.create(
            save_root=temp_dir,
            fps=30,
            features=test_features,
            modality_config=test_modality_config,
            task="test_task",
            vcodec="libx264",
        )

        # Record the second episode
        for frame in range(5):
            exporter2.add_frame(get_test_frame(frame))
        exporter2.save_episode()

        # Verify the second episode was saved
        video_path = exporter2.root / exporter2.meta.get_video_file_path(
            1, "observation.images.ego_view"
        )
        assert video_path.exists(), f"Second episode video file not found: {video_path}"

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None, reason="ffmpeg not installed, skipping test"
    )
    def test_interrupted_no_episode_completion(
        self, temp_dir, test_features, test_modality_config, test_data_collection_info
    ):
        """
        Test specifically for the case when interruption happens in the middle of recording an episode.
        Uses the real Gr00tDataExporter implementation.
        """
        # First session - add some frames and interrupt before saving
        exporter1 = Gr00tDataExporter.create(
            save_root=temp_dir,
            fps=30,
            features=test_features,
            modality_config=test_modality_config,
            task="test_task",
            data_collection_info=test_data_collection_info,
            vcodec="libx264",
        )

        # Add 3 frames but don't save
        for frame in range(3):
            exporter1.add_frame(get_test_frame(frame))
        # Don't save episode or consolidate to simulate interruption
        # The episode buffer is only in memory and will be lost on interruption

        # Let file system operations complete
        time.sleep(0.5)

        # Verify no episode was saved
        video_path = exporter1.root / exporter1.meta.get_video_file_path(
            0, "observation.images.ego_view"
        )
        assert not video_path.exists(), f"Episode should not have been saved: {video_path}"

        # Second session - will raise an error because no meta file exist, so we can't resume
        with pytest.raises(ValueError):
            _ = Gr00tDataExporter.create(
                save_root=temp_dir,
                fps=30,
                features=test_features,
                modality_config=test_modality_config,
                task="test_task",
                vcodec="libx264",
            )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed, skipping test")
def test_full_workflow(temp_dir, test_features, test_modality_config, test_data_collection_info):
    """
    Test that simulates the complete workflow from the record_session.py example.
    """
    NUM_EPISODES = 2
    FRAMES_PER_EPISODE = 3

    # Create the exporter
    exporter = Gr00tDataExporter.create(
        save_root=temp_dir,
        fps=20,
        features=test_features,
        modality_config=test_modality_config,
        task="test_task",
        data_collection_info=test_data_collection_info,
        robot_type="dummy",
    )

    # Create a small dataset
    for episode_index in range(NUM_EPISODES):
        for frame_index in range(FRAMES_PER_EPISODE):
            exporter.add_frame(get_test_frame(frame_index))
        exporter.save_episode()

    # check modality config
    modality_config_path = exporter.root / "meta" / "modality.json"
    assert modality_config_path.exists(), f"{modality_config_path} does not exists."
    with open(modality_config_path, "rb") as f:
        actual_modality_config = json.load(f)

    assert (
        actual_modality_config == test_modality_config
    ), f"Modality configs don't match.\nActual: {actual_modality_config}\nExpected: {test_modality_config}"

    # Verify results
    for episode_idx in range(NUM_EPISODES):
        video_path = exporter.root / exporter.meta.get_video_file_path(
            episode_idx, "observation.images.ego_view"
        )
        assert video_path.exists(), f"Video file not found: {video_path}"

    # Check that the expected number of episodes exists
    episode_count = 0
    for path in exporter.root.glob("**/*.mp4"):
        episode_count += 1
    assert episode_count == NUM_EPISODES, f"Expected {NUM_EPISODES} episodes, found {episode_count}"

    # Check the values of the dataset
    dataset = LeRobotDataset(
        repo_id="dataset",
        root=temp_dir,
    )
    for episode_idx in range(NUM_EPISODES):
        for frame_idx in range(FRAMES_PER_EPISODE):
            expected_frame = get_test_frame(frame_idx)
            actual_frame = dataset[episode_idx * FRAMES_PER_EPISODE + frame_idx]
            print(actual_frame["observation.images.ego_view"])
            actual_image_frame = actual_frame["observation.images.ego_view"].permute(1, 2, 0) * 255
            assert np.allclose(
                actual_image_frame.numpy(), expected_frame["observation.images.ego_view"], atol=10
            )  # Allow some tolerance for video compression
            assert np.allclose(
                actual_frame["observation.state"], expected_frame["observation.state"]
            )
            assert np.allclose(actual_frame["action"], expected_frame["action"])

        # validate data_collection_info
        assert dataset.meta.info["data_collection_info"] == test_data_collection_info.to_dict()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed, skipping test")
def test_overwrite_existing_dataset_false(
    temp_dir, test_features, test_modality_config, test_data_collection_info
):
    """
    Test that appends to the existing dataset when overwrite_existing is set to false.
    """
    # first dataset
    FIRST_NUM_EPISODES = 2
    FIRST_FRAMES_PER_EPISODE = 3

    exporter = Gr00tDataExporter.create(
        save_root=temp_dir,
        fps=20,
        features=test_features,
        modality_config=test_modality_config,
        task="test_task",
        data_collection_info=test_data_collection_info,
        robot_type="dummy",
    )
    # !! `overwrite_existing` should always be set to false by default
    # So we're deliberately not setting the overwrite_existing argument here.
    # This test ensures that
    # i. the default behavior is overwrite_existing=False
    # ii. the dataset appends to the existing dataset (instead of overwriting)

    # Create a first dataset
    for episode_index in range(FIRST_NUM_EPISODES):
        for frame_index in range(FIRST_FRAMES_PER_EPISODE):
            exporter.add_frame(get_test_frame(frame_index))
        exporter.save_episode()

    # second dataset
    del exporter
    SECOND_NUM_EPISODES = 3
    SECOND_FRAMES_PER_EPISODE = 2

    exporter = Gr00tDataExporter.create(
        save_root=temp_dir,
        fps=20,
        features=test_features,
        modality_config=test_modality_config,
        task="test_task",
        robot_type="dummy",
    )
    for episode_index in range(SECOND_NUM_EPISODES):
        for frame_index in range(SECOND_FRAMES_PER_EPISODE):
            exporter.add_frame(get_test_frame(frame_index))
        exporter.save_episode()

    # verify that there are
    EXPECTED_NUM_EPISODES = FIRST_NUM_EPISODES + SECOND_NUM_EPISODES
    assert len(list(exporter.root.glob("**/*.mp4"))) == EXPECTED_NUM_EPISODES
    assert len(list(exporter.root.glob("**/*.parquet"))) == EXPECTED_NUM_EPISODES


def test_overwrite_existing_dataset_true(
    temp_dir, test_features, test_modality_config, test_data_collection_info
):
    """
    Test that overwrites to an existing dataset when overwrite_existing=True.
    """
    # first dataset
    FIRST_NUM_EPISODES = 2
    FIRST_FRAMES_PER_EPISODE = 3

    exporter = Gr00tDataExporter.create(
        save_root=temp_dir,
        fps=20,
        features=test_features,
        modality_config=test_modality_config,
        task="test_task",
        data_collection_info=test_data_collection_info,
        robot_type="dummy",
    )

    # Create a first dataset
    for episode_index in range(FIRST_NUM_EPISODES):
        for frame_index in range(FIRST_FRAMES_PER_EPISODE):
            exporter.add_frame(get_test_frame(frame_index))
        exporter.save_episode()

    # verify that the dataset is written to the disk
    assert len(list(exporter.root.glob("**/*.mp4"))) == FIRST_NUM_EPISODES
    assert len(list(exporter.root.glob("**/*.parquet"))) == FIRST_NUM_EPISODES

    # second dataset
    SECOND_NUM_EPISODES = 3
    SECOND_FRAMES_PER_EPISODE = 2

    # re-initialize the exporter
    del exporter
    exporter = Gr00tDataExporter.create(
        save_root=temp_dir,
        fps=20,
        features=test_features,
        modality_config=test_modality_config,
        task="test_task",
        data_collection_info=test_data_collection_info,
        robot_type="dummy",
        overwrite_existing=True,
    )
    for episode_index in range(SECOND_NUM_EPISODES):
        for frame_index in range(SECOND_FRAMES_PER_EPISODE):
            exporter.add_frame(get_test_frame(frame_index))
        exporter.save_episode()

    # verify that the dataset is overwritten
    assert len(list(exporter.root.glob("**/*.mp4"))) == SECOND_NUM_EPISODES
    assert len(list(exporter.root.glob("**/*.parquet"))) == SECOND_NUM_EPISODES


def test_save_episode_as_discarded_and_skip(
    temp_dir, test_features, test_modality_config, test_data_collection_info
):
    """
    Test that verifies the functionality of saving an episode as discarded and skipping an episode.
    """
    FIRST_NUM_EPISODES = 10
    FIRST_FRAMES_PER_EPISODE = 3

    exporter = Gr00tDataExporter.create(
        save_root=temp_dir,
        fps=20,
        features=test_features,
        modality_config=test_modality_config,
        task="test_task",
        data_collection_info=test_data_collection_info,
        robot_type="dummy",
    )

    # Create a first dataset
    saved_episodes = 0
    discarded_episode_indices = []
    for episode_index in range(FIRST_NUM_EPISODES):
        for frame_index in range(FIRST_FRAMES_PER_EPISODE):
            exporter.add_frame(get_test_frame(frame_index))
        if episode_index % 3 == 0:
            exporter.save_episode_as_discarded()
            discarded_episode_indices.append(saved_episodes)
            saved_episodes += 1
        elif episode_index % 3 == 1:
            exporter.skip_and_start_new_episode()
        else:
            exporter.save_episode()
            saved_episodes += 1

    # verify that the dataset is written to the disk
    assert len(list(exporter.root.glob("**/*.mp4"))) == saved_episodes
    assert len(list(exporter.root.glob("**/*.parquet"))) == saved_episodes

    dataset = LeRobotDataset(
        repo_id="dataset",
        root=temp_dir,
    )

    assert dataset.meta.info["discarded_episode_indices"] == discarded_episode_indices
