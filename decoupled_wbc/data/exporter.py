import copy
from dataclasses import dataclass
from functools import partial
import json
import os
from pathlib import Path
import shutil
from typing import Optional

import datasets
from datasets import load_dataset
from datasets.utils import disable_progress_bars
from huggingface_hub.errors import RepositoryNotFoundError
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    compute_episode_stats,
)
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
    validate_frame,
)
import numpy as np
from PIL import Image as PILImage
import torch
from torchvision import transforms

from decoupled_wbc.control.main.config_template import ArgsConfig
from decoupled_wbc.data.video_writer import VideoWriter

disable_progress_bars()  # Disable HuggingFace progress bars


@dataclass
class DataCollectionInfo:
    """
    This dataclass stores additional information that is relevant to the data collection process.
    """

    lower_body_policy: Optional[str] = None
    wbc_model_path: Optional[str] = None
    teleoperator_username: Optional[str] = None
    support_operator_username: Optional[str] = None
    robot_type: Optional[str] = None
    robot_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary for JSON serialization."""
        return {
            "lower_body_policy": self.lower_body_policy,
            "wbc_model_path": self.wbc_model_path,
            "teleoperator_username": self.teleoperator_username,
            "support_operator_username": self.support_operator_username,
            "robot_type": self.robot_type,
            "robot_id": self.robot_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DataCollectionInfo":
        """Create a DataCollectionInfo instance from a dictionary."""
        return cls(**data)


class Gr00tDatasetMetadata(LeRobotDatasetMetadata):
    """
    Additional metadata on top of LeRobotDatasetMetadata:
    - modality_config: Written to `meta/modality.json`
    - discarded_episode_indices: List of episode indices that were discarded. Written to `meta/info.json`
    """

    MODALITY_CONFIG_REL_PATH = Path("meta/modality.json")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.root / self.MODALITY_CONFIG_REL_PATH, "rb") as f:
            self.modality_config = json.load(f)

    @classmethod
    def create(
        cls,
        modality_config: dict,
        script_config: dict,
        data_collection_info: DataCollectionInfo,
        *args,
        **kwargs,
    ):
        cls.validate_modality_config(modality_config)

        # Create base metadata object using parent class
        obj = super().create(*args, **kwargs)

        # we also need to initialize the discarded_episode_indices
        obj.info["script_config"] = script_config
        obj.info["discarded_episode_indices"] = []
        obj.info["data_collection_info"] = data_collection_info.to_dict()
        with open(obj.root / "meta" / "info.json", "w") as f:
            json.dump(obj.info, f, indent=4)

        obj.__class__ = cls
        with open(obj.root / cls.MODALITY_CONFIG_REL_PATH, "w") as f:
            json.dump(modality_config, f, indent=4)
        obj.modality_config = modality_config
        return obj

    @staticmethod
    def validate_modality_config(modality_config: dict) -> None:
        # verify if it contains all state, action, video, annotation keys
        valid_keys = ["state", "action", "video", "annotation"]
        if not all(key in modality_config for key in valid_keys):
            raise ValueError(
                f"Modality config must contain all of the following keys: {valid_keys}"
            )

        # verify that each key has a modality_config dict
        for key in valid_keys:
            if key not in modality_config:
                raise ValueError(f"Modality config must contain a '{key}' key")


class Gr00tDataExporter(LeRobotDataset):
    """
    A class for exporting data collected for a single session to LeRobot Dataset.

    Intended life cycle:
    1. Create a Gr00tDataExporter object
    2. Add frames using add_frame()
    3. Save the episode using save_episode()
        - This will flush the episode buffer to disk
        - This will also close the video writers
        - Create a new video writer and ep buffer to start new episode

    If interrupted, here's the indented behavior:
        - Interruption before save_episode() is called: loses the current episode
        - Interruption after save_episode() is called: keeps completed episodes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_writers = self.create_video_writer()

    @property
    def repo_id(self):
        return self.meta.repo_id

    @property
    def root(self):
        return self.meta.root

    @property
    def local_files_only(self):
        return self.meta.local_files_only

    @property
    def video_keys(self):
        return self.meta.video_keys

    @classmethod
    def create(
        cls,
        save_root: str | Path,
        fps: int,
        features: dict,
        modality_config: dict,
        task: str,
        script_config: ArgsConfig = ArgsConfig(),
        data_collection_info: DataCollectionInfo = DataCollectionInfo(),
        robot_type: str | None = None,
        tolerance_s: float = 1e-4,
        vcodec: str = "h264",
        overwrite_existing: bool = False,
        upload_bucket_path: str | None = None,
    ) -> "Gr00tDataExporter":
        """
        Create a Gr00tDataExporter object.

        Args:
            save_root: The root directory to save the dataset.
            fps: The frame rate of the dataset.
            features: The features of the dataset.
            modality_config: The modality config of the dataset.
            task: The task performed during the data collection session.
            data_collection_info: The data collection info.
                If the dataset already exists, this argument will be ignored.
                If data_collection_info is not provided, it will be set to an empty DataCollectionInfo object.
            robot_type: The type of robot.
            tolerance_s: The tolerance for the dataset.
            image_writer_processes: The number of processes to use for the image writer.
            image_writer_threads: The number of threads to use for the image writer.
            vcodec: The codec to use for the video writer.
        """

        obj = cls.__new__(cls)
        repo_id = (
            "tmp/tmp_dataset"  # NOTE(fengyuanh): Not relevant since we are not pushing to the hub
        )
        if overwrite_existing and (Path(save_root)).exists():
            print(
                f"Found existing dataset at {save_root}",
                "Cleaning up this directory since overwrite_existing is True.",
            )
            shutil.rmtree(save_root)

        if (Path(save_root)).exists():
            # Try to resume from existing dataset
            try:
                # Load the metadata
                obj.meta = Gr00tDatasetMetadata(
                    repo_id=repo_id,
                    root=save_root,
                )

            except RepositoryNotFoundError as e:
                raise ValueError(
                    f"Failed to resume from corrupted dataset. Please manually check the dataset at {save_root}"
                ) from e
        else:
            if not isinstance(script_config, dict):
                script_config = script_config.to_dict()
            obj.meta = Gr00tDatasetMetadata.create(
                repo_id=repo_id,
                fps=fps,
                root=save_root,
                # NOTE(fengyuanh): We use "robot_type" instead of this field which requires a Robot object
                robot=None,
                robot_type=robot_type,
                features=features,
                modality_config=modality_config,
                script_config=script_config,
                # NOTE(fengyuanh): Always use videos for exporting
                use_videos=True,
                data_collection_info=data_collection_info,
            )
        obj.tolerance_s = tolerance_s
        obj.video_backend = (
            "pyav"  # NOTE(fengyuanh): Only used in training, not relevant for exporting
        )
        obj.vcodec = vcodec
        obj.task = task
        obj.image_writer = None

        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.upload_bucket_path = upload_bucket_path
        obj.video_writers = obj.create_video_writer()
        return obj

    def create_video_writer(self) -> dict[str, VideoWriter]:
        video_writers = {}
        for key in self.meta.video_keys:
            video_writers[key] = VideoWriter(
                self.root
                / self.meta.get_video_file_path(self.episode_buffer["episode_index"], key),
                self.meta.shapes[key][1],
                self.meta.shapes[key][0],
                self.fps,
                self.vcodec,
            )
        return video_writers

    # @note (k2): This function is copied from LeRobotDataset.add_frame.
    # This is done because we want to bypass lerobot's
    # image_writer and use our own VideoWriter class.
    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Videos are handled by the video_writer,
        which uses a stream writer to write to disk.
        """
        frame = copy.deepcopy(frame)
        frame["task"] = frame.get("task", self.task)

        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        # Add frame features to episode_buffer
        for key in frame:
            if key == "task":
                # Note: we associate the task in natural language to its task index during `save_episode`
                self.episode_buffer["task"].append(frame["task"])
                continue

            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"],
                    image_key=key,
                    frame_index=frame_index,
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)

                # @note (k2): using our own VideoWriter class, bypassing the image_writer
                self.video_writers[key].add_frame(frame[key])
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def stop_video_writers(self):
        if not hasattr(self, "video_writers"):
            raise RuntimeError(
                "Can't stop video writers because they haven't been initialized. Call create() first."
            )
        for key in self.video_writers:
            self.video_writers[key].stop()

    def skip_and_start_new_episode(
        self,
    ) -> None:
        """
        Skip the current episode and start a new one.
        """
        self.stop_video_writers()
        self.episode_buffer = self.create_episode_buffer()
        self.video_writers = self.create_video_writer()

    # @note (k2): Code copied from LeRobotDataset.save_episode
    # We override this function because we want to bypass lerobot's `compute_episode_stats` on video features
    # since `compute_episode_stats` only works when images are written to disk.
    def save_episode(self, episode_data: dict | None = None) -> None:
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        # @note (k2): computing only non-video features stats
        non_video_features = {k: v for k, v in self.features.items() if v["dtype"] not in ["video"]}
        non_vid_ep_buffer = {
            k: v for k, v in episode_buffer.items() if k in non_video_features.keys()
        }
        ep_stats = compute_episode_stats(non_vid_ep_buffer, non_video_features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer and create new video writers
            self.episode_buffer = self.create_episode_buffer()
            self.video_writers = self.create_video_writer()

        # check if all video and parquet files exist
        for key in self.meta.video_keys:
            video_path = os.path.join(self.root, self.meta.get_video_file_path(episode_index, key))
            if not os.path.exists(video_path):
                raise FileNotFoundError(
                    f"Video path: {video_path} does not exist for episode {episode_index}"
                )

        parquet_path = os.path.join(self.root, self.meta.get_data_file_path(episode_index))
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Parquet path: {parquet_path} does not exist for episode {episode_index}"
            )

    # @note (k2): Overriding LeRobotDataset.encode_episode_videos to use our own VideoWriter class
    def encode_episode_videos(self, episode_index: int) -> dict:
        video_paths = {}
        for key in self.meta.video_keys:
            video_paths[key] = self.video_writers[key].stop()
        return video_paths

    def save_episode_as_discarded(self) -> None:
        """
        Flag ongoing episode as discarded and save it to disk. Failed manipulations (grasp, manipulation) are
        flagged as discarded. It will add the episode index to the discarded episode indices list in info.json.
        """
        self.meta.info["discarded_episode_indices"] = self.meta.info.get(
            "discarded_episode_indices", []
        ) + [self.episode_buffer["episode_index"]]
        self.save_episode()


def hf_transform_to_torch_by_features(
    features: datasets.Sequence, items_dict: dict[torch.Tensor | None]
):
    """Get a transform function that convert items from Hugging Face dataset (pyarrow)
    to torch tensors. Importantly, images are converted from PIL, which corresponds to
    a channel last representation (h w c) of uint8 type, to a torch image representation
    with channel first (c h w) of float32 type in range [0,1].
    """
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is None:
            pass
        else:
            if isinstance(features[key], datasets.Value):
                dtype_str = features[key].dtype
            elif isinstance(features[key], datasets.Sequence):
                assert isinstance(features[key].feature, datasets.Value)
                dtype_str = features[key].feature.dtype
            else:
                raise ValueError(f"Unsupported feature type for key '{key}': {features[key]}")
            dtype_mapping = {
                "float32": torch.float32,
                "float64": torch.float64,
                "int32": torch.int32,
                "int64": torch.int64,
            }
            items_dict[key] = [
                torch.tensor(x, dtype=dtype_mapping[dtype_str]) for x in items_dict[key]
            ]
    return items_dict


# This is a subclass of LeRobotDataset that only fixes the data type when loading
# By default, LeRobotDataset will automatically convert float64 to float32
class TypedLeRobotDataset(LeRobotDataset):
    def __init__(self, load_video=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not load_video:
            video_keys = []
            for key in self.meta.features.keys():
                if self.meta.features[key]["dtype"] == "video":
                    video_keys.append(key)
            for key in video_keys:
                self.meta.features.pop(key)

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [
                str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes
            ]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(partial(hf_transform_to_torch_by_features, hf_dataset.features))
        return hf_dataset
