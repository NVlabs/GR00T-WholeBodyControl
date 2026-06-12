import json
from pathlib import Path

import numpy as np

from gear_sonic.data.cleanse_lerobot_dataset import (
    check_dataset_structure,
    check_frame_count,
    check_pointing_activity,
    compute_tier,
    run_episode_checks,
    score_episode,
    select_episode_indices,
    summarise_report,
)


def test_compute_tier_caps_continuity_failure() -> None:
    assert compute_tier(90.0, continuity_passed=False) == "review"
    assert compute_tier(70.0, continuity_passed=True) == "keep"
    assert compute_tier(40.0, continuity_passed=True) == "review"
    assert compute_tier(39.9, continuity_passed=True) == "discard"


def test_score_episode_uses_pointing_profile_weights() -> None:
    checks = {
        "frame_count": {"passed": True},
        "action_range": {"passed": True},
        "state_continuity": {"passed": True},
        "pointing_activity": {"passed": True},
        "video_integrity": {"passed": True},
        "teleop_tracking_quality": {"passed": False, "bad_ratio": 0.25},
        "task_motion": {
            "active_frame": 12,
            "movement_detected": True,
            "hold_frames": 20,
        },
    }

    score, details = score_episode(checks)

    assert score == 97.5
    assert details["teleop_tracking_quality"] == 7.5
    assert details["task_motion"] == 25.0


def test_pointing_activity_requires_nonzero_signal_and_motion() -> None:
    frozen = np.zeros((20, 9), dtype=np.float32)
    failed = check_pointing_activity([frozen], movement_threshold=0.05)
    assert failed["passed"] is False

    moving = frozen.copy()
    moving[5:, 0] = np.linspace(0.0, 0.2, 15)
    passed = check_pointing_activity([moving], movement_threshold=0.05)
    assert passed["passed"] is True
    assert passed["active_frame"] == 6
    assert passed["movement_detected"] is True


def test_episode_checks_use_robot_motion_when_teleop_columns_are_zero(tmp_path: Path) -> None:
    dataset = tmp_path / "point_green_block"
    video_dir = dataset / "videos/chunk-000/observation.images.ego_view"
    video_dir.mkdir(parents=True)
    (video_dir / "episode_000000.mp4").write_bytes(b"not-a-real-video-but-nonempty")

    state = np.zeros((40, 4), dtype=np.float32)
    state[:, 0] = np.linspace(0.0, 0.2, 40)
    action = state.copy()
    table = {
        "observation.state": list(state),
        "action.wbc": list(action),
        "teleop.vr_3pt_position": [np.zeros(9, dtype=np.float32) for _ in range(40)],
        "teleop.left_hand_joints": [np.zeros(7, dtype=np.float32) for _ in range(40)],
        "teleop.right_hand_joints": [np.zeros(7, dtype=np.float32) for _ in range(40)],
        "teleop.smpl_pose": [np.zeros(63, dtype=np.float32) for _ in range(40)],
    }
    info = {
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"observation.images.ego_view": {"dtype": "video"}},
    }

    checks = run_episode_checks(
        dataset,
        info,
        {"episode_index": 0},
        dataframe_loader=lambda _path: table,
    )

    assert checks["pointing_activity"]["passed"] is True
    assert checks["pointing_activity"]["source"] == "robot_motion"
    assert checks["teleop_tracking_quality"]["passed"] is True


def test_check_dataset_structure_reports_metadata_only_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "point_block"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 0,
                "total_frames": 0,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.images.ego_view": {"dtype": "video"},
                },
            }
        ),
        encoding="utf-8",
    )

    result = check_dataset_structure(dataset)

    assert result["passed"] is False
    assert result["reason"] == "No episodes recorded"
    assert result["total_episodes"] == 0


def test_select_episode_indices_honors_keep_include_and_exclude() -> None:
    report = {
        "episodes": {
            "episode_000000": {"episode_index": 0, "tier": "keep"},
            "episode_000001": {"episode_index": 1, "tier": "review"},
            "episode_000002": {"episode_index": 2, "tier": "keep"},
        }
    }

    selected = select_episode_indices(
        report,
        include=["episode_000001"],
        exclude=["episode_000002"],
    )

    assert selected == [0, 1]


def test_select_episode_indices_filters_combined_report_by_dataset() -> None:
    report = {
        "episodes": {
            "point_green_block/episode_000000": {
                "dataset": "outputs/point_green_block",
                "episode_index": 0,
                "tier": "keep",
            },
            "point_yellow_block/episode_000000": {
                "dataset": "outputs/point_yellow_block",
                "episode_index": 0,
                "tier": "keep",
            },
            "point_green_block/episode_000001": {
                "dataset": "outputs/point_green_block",
                "episode_index": 1,
                "tier": "review",
            },
        }
    }

    selected = select_episode_indices(
        report,
        include=["episode_000001"],
        dataset="outputs/point_green_block",
    )

    assert selected == [0, 1]


def test_summarise_report_counts_invalid_dataset() -> None:
    report = {
        "episodes": {
            "episode_000000": {"tier": "keep"},
            "episode_000001": {"tier": "review"},
            "episode_000002": {"tier": "discard"},
        },
        "datasets": {
            "outputs/point_block": {"passed": False},
            "outputs/point_green_block": {"passed": True},
        },
    }

    summary = summarise_report(report)

    assert summary == {
        "total": 3,
        "keep": 1,
        "review": 1,
        "discard": 1,
        "keep_rate": 33.3,
        "datasets_total": 2,
        "datasets_invalid": 1,
    }


def test_frame_count_default_accepts_pointing_episode_lengths() -> None:
    assert check_frame_count(244)["passed"] is True
    assert check_frame_count(29)["passed"] is False
    assert "Too short" in check_frame_count(29)["reason"]
