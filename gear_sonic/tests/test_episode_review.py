import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.data.episode_review import (
    build_episode_catalog,
    build_joint_toggle_groups,
    compute_joint_velocity_from_arrays,
    default_trim_dataset_path,
    format_trash_confirmation_message,
    gradio_allowed_paths,
    make_disabled_joint_chips_html,
    make_joint_velocity_svg_html,
    make_joint_velocity_toggle_html,
    make_velocity_payload,
    move_episode_to_trash,
    navigate_episode_index,
    prompt_for_episode,
    resolve_episode_video,
    review_sync_head,
    selected_catalog_index_from_table_event,
    trim_episode_to_dataset,
    trim_frame_indices,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "point_green_block_clean"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "videos/chunk-000/observation.images.ego_view").mkdir(parents=True)
    (dataset / "data/chunk-000/episode_000000.parquet").write_bytes(b"parquet")
    (dataset / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4").write_bytes(
        b"video"
    )
    (dataset / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": 100,
                "fps": 50,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": (
                    "videos/chunk-{episode_chunk:03d}/{video_key}/"
                    "episode_{episode_index:06d}.mp4"
                ),
                "features": {
                    "observation.images.ego_view": {"dtype": "video"},
                    "observation.state": {
                        "dtype": "float64",
                        "shape": [2],
                        "names": ["joint_a", "joint_b"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        dataset / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["point finger to green block"], "length": 100}],
    )
    _write_jsonl(dataset / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "fallback"}])
    return dataset


class _FakeColumn:
    def __init__(self, values: list) -> None:
        self._values = values

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._values)


class _FakeIloc:
    def __init__(self, table: "_FakeTrimTable") -> None:
        self._table = table

    def __getitem__(self, indices: np.ndarray) -> "_FakeTrimTable":
        selected = [int(index) for index in indices]
        return _FakeTrimTable(
            {
                key: [values[index] for index in selected]
                for key, values in self._table.rows.items()
            }
        )


class _FakeTrimTable:
    def __init__(self, rows: dict[str, list]) -> None:
        self.rows = {key: list(values) for key, values in rows.items()}
        self.iloc = _FakeIloc(self)

    @property
    def columns(self) -> list[str]:
        return list(self.rows)

    def __len__(self) -> int:
        return len(next(iter(self.rows.values()))) if self.rows else 0

    def __getitem__(self, column: str) -> _FakeColumn:
        return _FakeColumn(self.rows[column])

    def __setitem__(self, column: str, values) -> None:
        try:
            self.rows[column] = list(values)
        except TypeError:
            self.rows[column] = [values] * len(self)

    def copy(self) -> "_FakeTrimTable":
        return _FakeTrimTable(self.rows)

    def reset_index(self, drop: bool = False) -> "_FakeTrimTable":
        return self

    def to_parquet(self, path: Path) -> None:
        path.write_text(json.dumps({"rows": len(self)}), encoding="utf-8")


def test_build_episode_catalog_includes_prompt_duration_and_video(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)

    catalog = build_episode_catalog([dataset])

    assert len(catalog) == 1
    assert catalog[0].dataset_name == "point_green_block_clean"
    assert catalog[0].episode_index == 0
    assert catalog[0].prompt == "point finger to green block"
    assert catalog[0].duration_s == 2.0
    assert catalog[0].video_path.name == "episode_000000.mp4"


def test_resolve_episode_video_prefers_ego_view(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))

    video = resolve_episode_video(dataset, info, 0)

    assert video == dataset / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4"


def test_prompt_for_episode_uses_episode_task_strings_before_task_table() -> None:
    assert (
        prompt_for_episode(
            {"tasks": ["episode prompt"]},
            [{"task_index": 0, "task": "task table prompt"}],
        )
        == "episode prompt"
    )


def test_prompt_for_episode_resolves_integer_task_indices() -> None:
    prompt = prompt_for_episode(
        {"tasks": [1]},
        [
            {"task_index": 0, "task": "zero"},
            {"task_index": 1, "task": "one"},
        ],
    )

    assert prompt == "one"


def test_compute_joint_velocity_from_arrays_uses_timestamp_deltas() -> None:
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 1.0],
        ]
    )
    timestamps = np.array([0.0, 0.5, 1.0])

    velocity = compute_joint_velocity_from_arrays(positions, timestamps, fps=50)

    np.testing.assert_allclose(
        velocity,
        np.array(
            [
                [2.0, 0.0],
                [2.0, 0.0],
                [4.0, 2.0],
            ]
        ),
    )


def test_trim_frame_indices_uses_half_open_interval() -> None:
    timestamps = np.arange(0.0, 2.5, 0.5)

    indices = trim_frame_indices(timestamps, start_s=0.5, end_s=1.5)

    np.testing.assert_array_equal(indices, np.array([1, 2]))


def test_default_trim_dataset_path_is_stable_per_source_episode(tmp_path: Path) -> None:
    output_root = tmp_path / "trims"

    first_path = default_trim_dataset_path(
        output_root,
        dataset_name="point_green_block_clean",
        episode_index=3,
        start_s=2.0,
        end_s=22.0,
    )
    second_path = default_trim_dataset_path(
        output_root,
        dataset_name="point_green_block_clean",
        episode_index=3,
        start_s=0.0,
        end_s=6.0,
    )

    assert first_path.name == "point_green_block_clean_episode_000003"
    assert second_path == first_path


def test_trim_episode_to_dataset_replaces_previous_final_from_original(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    output_root = tmp_path / "trims"
    legacy_version = output_root / "point_green_block_clean_episode_000000_0p000_5p000"
    legacy_version.mkdir(parents=True)
    original_table = _FakeTrimTable(
        {
            "episode_index": [0] * 70,
            "frame_index": list(range(70)),
            "timestamp": list(np.arange(70, dtype=np.float64) / 10.0),
            "observation.state": [np.array([float(idx), 0.0]) for idx in range(70)],
        }
    )

    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["fps"] = 10
    info_path.write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(
        dataset / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["point finger to green block"], "length": 70}],
    )
    load_calls: list[Path] = []
    trim_calls: list[tuple[Path, Path, float, float]] = []

    def load_original(path: Path) -> _FakeTrimTable:
        load_calls.append(path)
        return original_table.copy()

    def fake_trim_video(src: Path, dst: Path, start_s: float, end_s: float) -> None:
        trim_calls.append((src, dst, start_s, end_s))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(f"{start_s}-{end_s}", encoding="utf-8")

    first = trim_episode_to_dataset(
        dataset,
        0,
        0.0,
        5.0,
        output_root=output_root,
        dataframe_loader=load_original,
        video_trimmer=fake_trim_video,
    )
    (first.dataset_path / "stale.txt").write_text("old", encoding="utf-8")

    second = trim_episode_to_dataset(
        dataset,
        0,
        0.0,
        6.0,
        output_root=output_root,
        dataframe_loader=load_original,
        video_trimmer=fake_trim_video,
    )

    assert second.dataset_path == first.dataset_path
    assert not (second.dataset_path / "stale.txt").exists()
    assert not legacy_version.exists()
    assert first.dataset_path.name == "point_green_block_clean_episode_000000"
    assert load_calls == [
        dataset / "data/chunk-000/episode_000000.parquet",
        dataset / "data/chunk-000/episode_000000.parquet",
    ]
    assert first.frames == 50
    assert second.frames == 60
    assert trim_calls == [
        (
            dataset / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
            first.dataset_path
            / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
            0.0,
            5.0,
        ),
        (
            dataset / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
            second.dataset_path
            / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
            0.0,
            6.0,
        ),
    ]



def test_navigate_episode_index_supports_button_jumps() -> None:
    assert navigate_episode_index(5, "first", total=10) == 0
    assert navigate_episode_index(5, "previous", total=10) == 4
    assert navigate_episode_index(5, "next", total=10) == 6
    assert navigate_episode_index(5, "last", total=10) == 9
    assert navigate_episode_index(0, "previous", total=10) == 0
    assert navigate_episode_index(9, "next", total=10) == 9


def test_selected_catalog_index_from_table_event_uses_row_index() -> None:
    rows = [
        ["point_block/episode_000000 (7.80s)", "point_block", 0, 390, 7.8, "point"],
        ["point_block/episode_000001 (4.68s)", "point_block", 1, 234, 4.68, "point"],
    ]

    index = selected_catalog_index_from_table_event((1, 4), rows, total=2)

    assert index == 1


def test_selected_catalog_index_from_table_event_can_match_row_value() -> None:
    rows = [
        ["point_block/episode_000000 (7.80s)", "point_block", 0, 390, 7.8, "point"],
        ["point_yellow_block_clean/episode_000001 (5.56s)", "point_yellow_block_clean", 1, 278, 5.56, "point"],
    ]
    event_value = ["point_yellow_block_clean/episode_000001 (5.56s)", "point_yellow_block_clean", 1]

    index = selected_catalog_index_from_table_event(None, rows, total=2, event_value=event_value)

    assert index == 1


def test_review_app_data_index_select_collects_event_data(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    from gear_sonic.scripts.review_lerobot_episodes import build_app

    dataset = _make_dataset(tmp_path)
    app = build_app([dataset], tmp_path / "trims", max_plot_joints=8)
    dependencies = app.config["dependencies"]
    select_dependencies = [
        dependency
        for dependency in dependencies
        if dependency.get("api_name") == "select_from_table"
    ]

    assert len(select_dependencies) == 1
    assert select_dependencies[0]["collects_event_data"] is True


def test_joint_velocity_svg_html_exposes_video_sync_marker() -> None:
    timestamps = np.array([0.0, 0.5, 1.0])
    velocity = np.array(
        [
            [0.0, 1.0],
            [2.0, 0.5],
            [1.0, 0.0],
        ]
    )

    html = make_joint_velocity_svg_html(timestamps, velocity, ["joint_a", "joint_b"], max_joints=2)

    assert 'class="episode-velocity-plot"' in html
    assert 'data-duration="1.000000"' in html
    assert "episode-velocity-marker" in html
    assert "joint_a" in html
    assert "joint_b" in html


def test_joint_velocity_svg_html_honors_selected_joints() -> None:
    timestamps = np.array([0.0, 0.5, 1.0])
    velocity = np.array(
        [
            [0.0, 1.0],
            [2.0, 0.5],
            [1.0, 0.0],
        ]
    )

    html = make_joint_velocity_svg_html(
        timestamps,
        velocity,
        ["left_elbow_joint", "right_elbow_joint"],
        selected_joint_names=["right_elbow_joint"],
    )

    assert "right_elbow_joint" in html
    assert "left_elbow_joint" not in html


def test_joint_velocity_toggle_html_renders_client_side_controls() -> None:
    timestamps = np.array([0.0, 0.5, 1.0])
    velocity = np.zeros((3, 4), dtype=np.float64)
    velocity[:, 0] = [0.0, 1.0, 0.5]
    velocity[:, 1] = [0.0, 0.2, 0.1]
    names = [
        "left_shoulder_pitch_joint",
        "right_elbow_joint",
        "left_hand_index_0_joint",
        "right_hand_index_0_joint",
    ]

    html = make_joint_velocity_toggle_html(
        timestamps,
        velocity,
        names,
        selected_joint_names=["right_elbow_joint"],
        duration_s=1.0,
    )

    assert "episode-joint-checkbox" in html
    assert 'data-joint="right_elbow_joint"' in html
    assert 'data-joint="left_shoulder_pitch_joint"' in html
    assert 'name="right_elbow_joint" checked' in html
    assert "left_hand_index_0_joint" in html
    assert "episode-disabled-joint-chip" in html


def test_joint_toggle_groups_mark_zero_velocity_joints_unavailable() -> None:
    names = [
        "left_shoulder_pitch_joint",
        "left_hand_index_0_joint",
        "right_elbow_joint",
    ]
    velocity = np.zeros((4, 3), dtype=np.float64)
    velocity[:, 2] = [0.0, 0.2, 0.3, 0.0]

    groups = build_joint_toggle_groups(names, velocity)

    assert groups["right_arm"].enabled == ["right_elbow_joint"]
    assert groups["right_arm"].selected == ["right_elbow_joint"]
    assert groups["left_arm"].disabled == ["left_shoulder_pitch_joint"]
    assert groups["left_hand"].disabled == ["left_hand_index_0_joint"]


def test_disabled_joint_chips_html_renders_grey_buttons() -> None:
    html = make_disabled_joint_chips_html(["left_hand_index_0_joint"])

    assert "episode-disabled-joint-chip" in html
    assert "left_hand_index_0_joint" in html


def test_trash_confirmation_message_names_current_episode(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    item = build_episode_catalog([dataset])[0]

    message = format_trash_confirmation_message(item)

    assert message.startswith("Delete current episode?")
    assert item.label in message


def test_gradio_allowed_paths_include_external_dataset_roots_and_output_root(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "external" / "point_block"
    output_root = tmp_path / "trims"
    dataset.mkdir(parents=True)

    allowed_paths = gradio_allowed_paths([dataset, dataset], output_root)

    assert allowed_paths == [str(dataset.resolve()), str(output_root.resolve())]


def test_review_sync_head_updates_marker_from_video_time() -> None:
    head = review_sync_head()

    assert "timeupdate" in head
    assert "currentTime" in head
    assert "episode-velocity-marker" in head
    assert "episode-joint-checkbox" in head
    assert "toggleJointVisibility" in head


def test_review_sync_head_formats_video_time_with_tenths() -> None:
    head = review_sync_head()

    assert "formatVideoTime" in head
    assert 'return minutes + ":" + secondsText + "." + tenths;' in head
    assert "formatVideoTime(boundedTime)" in head


def test_review_sync_head_adds_precise_video_timestamp_readout() -> None:
    head = review_sync_head()

    assert "episode-video-current" in head
    assert "ensureVideoTimestamp" in head
    assert "videoTimeLabel.textContent = formatVideoTime(time)" in head


def test_joint_velocity_svg_html_initial_time_uses_minute_second_tenths() -> None:
    timestamps = np.array([0.0, 0.5, 1.0])
    velocity = np.array(
        [
            [0.0],
            [2.0],
            [1.0],
        ]
    )

    html = make_joint_velocity_svg_html(timestamps, velocity, ["joint_a"], max_joints=1)

    assert '<span class="episode-velocity-current">0:00.0</span>' in html


def test_review_sync_head_forces_readable_widget_text_in_dark_mode() -> None:
    head = review_sync_head()

    assert "color-scheme: light" in head
    assert "color: #111827 !important" in head
    assert "fill: #111827 !important" in head


def test_velocity_payload_is_json_serializable() -> None:
    payload = make_velocity_payload(
        np.array([0.0, 0.5]),
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        ["joint_a", "joint_b"],
        duration_s=0.5,
    )

    encoded = json.dumps(payload)

    assert "joint_a" in encoded
    assert payload["timestamps"] == [0.0, 0.5]
    assert payload["velocity"] == [[0.0, 1.0], [2.0, 3.0]]


def test_move_episode_to_trash_updates_metadata_and_moves_files(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    (dataset / "data/chunk-000/episode_000001.parquet").write_bytes(b"parquet-1")
    (dataset / "videos/chunk-000/observation.images.ego_view/episode_000001.mp4").write_bytes(
        b"video-1"
    )
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] = 2
    info["total_frames"] = 150
    info["total_videos"] = 2
    info["splits"] = {"train": "0:2"}
    info_path.write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(
        dataset / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["point finger to green block"], "length": 100},
            {"episode_index": 1, "tasks": ["point finger to green block"], "length": 50},
        ],
    )

    result = move_episode_to_trash(dataset, 0, reason="bad pointing")

    assert result.episode_index == 0
    assert result.trash_dir.exists()
    assert not (dataset / "data/chunk-000/episode_000000.parquet").exists()
    assert not (dataset / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4").exists()
    assert (result.trash_dir / "data/chunk-000/episode_000000.parquet").exists()
    assert (result.trash_dir / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4").exists()

    remaining = [
        json.loads(line)
        for line in (dataset / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    updated_info = json.loads(info_path.read_text(encoding="utf-8"))
    assert [episode["episode_index"] for episode in remaining] == [1]
    assert updated_info["total_episodes"] == 1
    assert updated_info["total_frames"] == 50
    assert updated_info["total_videos"] == 1
    assert updated_info["splits"] == {"train": "0:1"}
