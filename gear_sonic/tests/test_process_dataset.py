import json

import pytest


def test_write_output_dataset_remaps_task_indices_by_task_text(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("av")
    pytest.importorskip("tyro")
    from gear_sonic.scripts.process_dataset import write_output_dataset

    episodes = [
        {
            "df": pd.DataFrame(
                {
                    "episode_index": [10, 10],
                    "index": [0, 1],
                    "frame_index": [0, 1],
                    "timestamp": [0.0, 0.02],
                    "task_index": [0, 0],
                }
            ),
            "source_video_paths": {},
            "valid_indices": None,
            "episode_meta": {"episode_index": 10, "tasks": ["point blue"], "length": 2},
            "fps": 50,
        },
        {
            "df": pd.DataFrame(
                {
                    "episode_index": [0, 0, 0],
                    "index": [0, 1, 2],
                    "frame_index": [0, 1, 2],
                    "timestamp": [0.0, 0.02, 0.04],
                    "task_index": [0, 0, 0],
                }
            ),
            "source_video_paths": {},
            "valid_indices": None,
            "episode_meta": {"episode_index": 0, "tasks": ["point green"], "length": 3},
            "fps": 50,
        },
    ]
    info = {
        "fps": 50,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"task_index": {"dtype": "int64", "shape": [1], "names": None}},
    }

    write_output_dataset(
        tmp_path / "merged",
        episodes,
        info,
        [
            {"task_index": 0, "task": "point blue"},
            {"task_index": 0, "task": "point green"},
        ],
        script_config=None,
    )

    tasks = [
        json.loads(line)
        for line in (tmp_path / "merged/meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first = pd.read_parquet(tmp_path / "merged/data/chunk-000/episode_000000.parquet")
    second = pd.read_parquet(tmp_path / "merged/data/chunk-000/episode_000001.parquet")
    merged_info = json.loads((tmp_path / "merged/meta/info.json").read_text(encoding="utf-8"))

    assert tasks == [
        {"task_index": 0, "task": "point blue"},
        {"task_index": 1, "task": "point green"},
    ]
    assert first["task_index"].tolist() == [0, 0]
    assert second["task_index"].tolist() == [1, 1, 1]
    assert merged_info["total_tasks"] == 2
