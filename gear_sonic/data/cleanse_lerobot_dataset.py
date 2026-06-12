"""Cleansing and verification utilities for GEAR-SONIC LeRobot datasets.

The collection path in this repository writes LeRobot-format datasets directly:
``meta/*.json(l)``, ``data/chunk-*/episode_*.parquet``, and
``videos/chunk-*``.  This module keeps the score/tier/report workflow from the
older raw-episode cleanser, but adapts the checks for pointing demonstrations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable

import numpy as np

WEIGHTS: dict[str, int] = {
    "frame_count": 10,
    "action_range": 10,
    "state_continuity": 15,
    "pointing_activity": 15,
    "video_integrity": 15,
    "teleop_tracking_quality": 10,
    "task_motion": 25,
}

KEEP_THRESHOLD = 70
DISCARD_THRESHOLD = 40

DEFAULT_MIN_FRAMES = 30
DEFAULT_MAX_FRAMES = 900
DEFAULT_ACTION_ABS_LIMIT = 8.0
DEFAULT_STATE_MAX_DELTA = 0.8
DEFAULT_POINTING_MOVEMENT = 0.05
DEFAULT_MAX_BAD_RATIO = 0.1
DEFAULT_HOLD_MIN_FRAMES = 15

_ACTIVE_WEIGHT = 10
_MOVEMENT_WEIGHT = 10
_HOLD_WEIGHT = 5

DataFrameLoader = Callable[[Path], Any]


class MissingParquetDependency(RuntimeError):
    """Raised when a parquet operation needs optional data-collection deps."""


def episode_name(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_info(dataset_path: Path) -> dict[str, Any]:
    return _read_json(Path(dataset_path) / "meta" / "info.json")


def load_episodes_meta(dataset_path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(Path(dataset_path) / "meta" / "episodes.jsonl")


def get_video_keys(info: dict[str, Any]) -> list[str]:
    keys = list(info.get("video_keys", []))
    if keys:
        return keys
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") in {"video", "image"}
    ]


def get_parquet_path(dataset_path: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return Path(dataset_path) / pattern.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )


def get_video_paths(dataset_path: Path, info: dict[str, Any], episode_index: int) -> dict[str, Path]:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    return {
        video_key: Path(dataset_path)
        / pattern.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=video_key,
        )
        for video_key in get_video_keys(info)
    }


def load_episode_table(parquet_path: Path) -> Any:
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on operator env
        raise MissingParquetDependency(
            "Reading parquet episodes requires pandas plus a parquet engine. "
            "Install the data-collection extras, for example: "
            "pip install -e 'gear_sonic[data_collection]'."
        ) from exc
    try:
        return pd.read_parquet(parquet_path)
    except Exception as exc:
        message = str(exc).lower()
        if isinstance(exc, ImportError) or "pyarrow" in message or "parquet" in message:
            raise MissingParquetDependency(
                "Reading parquet episodes requires pandas plus a parquet engine. "
                "Install the data-collection extras, for example: "
                "pip install -e 'gear_sonic[data_collection]', or the review UI extras: "
                "pip install -e 'gear_sonic[review]'."
            ) from exc
        raise


def check_dataset_structure(dataset_path: Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return {"passed": False, "reason": "Missing meta/info.json"}

    info = _read_json(info_path)
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    result = {
        "passed": True,
        "reason": "Dataset structure OK",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
    }

    if total_episodes <= 0:
        return {**result, "passed": False, "reason": "No episodes recorded"}

    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return {**result, "passed": False, "reason": "Missing meta/episodes.jsonl"}

    episodes = load_episodes_meta(dataset_path)
    if len(episodes) != total_episodes:
        return {
            **result,
            "passed": False,
            "reason": f"Episode metadata count mismatch: {len(episodes)} != {total_episodes}",
        }

    missing_parquet = []
    missing_video = []
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        if not get_parquet_path(dataset_path, info, ep_idx).exists():
            missing_parquet.append(episode_name(ep_idx))
        for key, video_path in get_video_paths(dataset_path, info, ep_idx).items():
            if not video_path.exists():
                missing_video.append(f"{episode_name(ep_idx)}:{key}")

    if missing_parquet:
        return {
            **result,
            "passed": False,
            "reason": f"Missing parquet files: {missing_parquet[:3]}",
            "missing_parquet": missing_parquet,
        }
    if missing_video:
        return {
            **result,
            "passed": False,
            "reason": f"Missing video files: {missing_video[:3]}",
            "missing_video": missing_video,
        }
    return result


def check_frame_count(
    num_frames: int,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> dict[str, Any]:
    if num_frames < min_frames:
        return {"passed": False, "reason": f"Too short: {num_frames} frames (min {min_frames})"}
    if num_frames > max_frames:
        return {"passed": False, "reason": f"Too long: {num_frames} frames (max {max_frames})"}
    return {"passed": True, "reason": f"{num_frames} frames OK"}


def _stack_column(table: Any, column: str) -> np.ndarray:
    values = table[column]
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.vstack([np.asarray(value) for value in values])


def _table_has_column(table: Any, column: str) -> bool:
    columns = getattr(table, "columns", None)
    if columns is not None:
        return column in columns
    return column in table


def _available_stacked_columns(table: Any, columns: Iterable[str]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for column in columns:
        if _table_has_column(table, column):
            arrays.append(_stack_column(table, column).astype(np.float64))
    return arrays


def check_action_range(
    actions: np.ndarray,
    max_abs: float = DEFAULT_ACTION_ABS_LIMIT,
) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.size == 0:
        return {"passed": False, "reason": "No action values"}
    if not np.isfinite(actions).all():
        return {"passed": False, "reason": "Action contains NaN or inf"}
    max_value = float(np.max(np.abs(actions)))
    if max_value > max_abs:
        idx = np.unravel_index(np.argmax(np.abs(actions)), actions.shape)
        return {
            "passed": False,
            "reason": f"Action out of range at frame {idx[0]}, joint {idx[1]}: {actions[idx]:.3f}",
            "max_abs": max_value,
        }
    return {"passed": True, "reason": f"Actions in range (max abs {max_value:.3f})", "max_abs": max_value}


def check_state_continuity(
    states: np.ndarray,
    max_delta: float = DEFAULT_STATE_MAX_DELTA,
) -> dict[str, Any]:
    states = np.asarray(states, dtype=np.float64)
    if len(states) < 2:
        return {"passed": True, "reason": "Single frame, skipping continuity check"}
    if not np.isfinite(states).all():
        return {"passed": False, "reason": "State contains NaN or inf"}
    deltas = np.abs(np.diff(states, axis=0))
    max_value = float(np.max(deltas))
    if max_value > max_delta:
        idx = np.unravel_index(np.argmax(deltas), deltas.shape)
        return {
            "passed": False,
            "reason": f"State jump at frame {idx[0]}, joint {idx[1]}: delta={deltas[idx]:.3f}",
            "max_delta": max_value,
        }
    return {"passed": True, "reason": f"State continuous (max delta {max_value:.3f})", "max_delta": max_value}


def _combined_signal(signals: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(signal, dtype=np.float64) for signal in signals if len(signal) > 0]
    if not arrays:
        return np.empty((0, 0), dtype=np.float64)
    min_len = min(len(array) for array in arrays)
    return np.hstack([array[:min_len].reshape(min_len, -1) for array in arrays])


def _max_stable_run_after(signal: np.ndarray, start: int, delta: float) -> int:
    if len(signal) < 2 or start >= len(signal) - 1:
        return 0
    run = 0
    best = 0
    for frame in range(start, len(signal) - 1):
        if float(np.max(np.abs(signal[frame + 1] - signal[frame]))) < delta:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def check_pointing_activity(
    signals: Iterable[np.ndarray],
    movement_threshold: float = DEFAULT_POINTING_MOVEMENT,
    hold_min_frames: int = DEFAULT_HOLD_MIN_FRAMES,
) -> dict[str, Any]:
    signal = _combined_signal(signals)
    if signal.size == 0:
        return {
            "passed": False,
            "reason": "No teleop signal columns available",
            "active_frame": -1,
            "movement_detected": False,
            "hold_frames": 0,
        }

    norms = np.linalg.norm(signal, axis=1)
    active_frames = np.flatnonzero(norms > 1e-6)
    if len(active_frames) == 0:
        return {
            "passed": False,
            "reason": "Teleop pointing signal is all zero",
            "active_frame": -1,
            "movement_detected": False,
            "hold_frames": 0,
        }

    diffs = np.abs(np.diff(signal, axis=0))
    total_motion = float(np.sum(diffs))
    movement_detected = total_motion >= movement_threshold
    active_frame = int(active_frames[0])
    hold_frames = _max_stable_run_after(signal, active_frame, movement_threshold)
    passed = movement_detected
    reason = (
        f"Pointing signal active at frame {active_frame}, motion={total_motion:.3f}, hold={hold_frames}"
        if passed
        else f"Pointing signal active but motion too small: {total_motion:.3f} < {movement_threshold}"
    )
    return {
        "passed": passed,
        "reason": reason,
        "active_frame": active_frame,
        "movement_detected": movement_detected,
        "hold_frames": hold_frames,
        "hold_min_frames": hold_min_frames,
        "total_motion": total_motion,
    }


def check_teleop_tracking_quality(
    signals: Iterable[np.ndarray],
    max_bad_ratio: float = DEFAULT_MAX_BAD_RATIO,
) -> dict[str, Any]:
    signal = _combined_signal(signals)
    if signal.size == 0:
        return {"passed": True, "reason": "Skipped (no teleop tracking columns)", "bad_ratio": 0.0}
    all_zero = np.linalg.norm(signal, axis=1) <= 1e-6
    bad_ratio = float(np.sum(all_zero) / max(len(signal), 1))
    if bad_ratio > max_bad_ratio:
        return {
            "passed": False,
            "reason": f"Teleop tracking: {bad_ratio:.0%} zero frames (threshold {max_bad_ratio:.0%})",
            "bad_ratio": bad_ratio,
        }
    return {"passed": True, "reason": f"Teleop tracking OK ({bad_ratio:.0%} zero frames)", "bad_ratio": bad_ratio}


def check_video_integrity(video_paths: Iterable[Path]) -> dict[str, Any]:
    paths = list(video_paths)
    if not paths:
        return {"passed": True, "reason": "No video keys declared", "bad_ratio": 0.0}
    bad = [str(path) for path in paths if not path.exists() or path.stat().st_size <= 0]
    bad_ratio = len(bad) / max(len(paths), 1)
    if bad:
        return {
            "passed": False,
            "reason": f"Missing or empty video files: {bad[:3]}",
            "bad_ratio": bad_ratio,
        }
    return {"passed": True, "reason": "Video files present", "bad_ratio": 0.0}


def compute_tier(score: float, continuity_passed: bool) -> str:
    if score >= KEEP_THRESHOLD:
        tier = "keep"
    elif score >= DISCARD_THRESHOLD:
        tier = "review"
    else:
        tier = "discard"
    if not continuity_passed and tier == "keep":
        return "review"
    return tier


def score_episode(checks: dict[str, dict[str, Any]]) -> tuple[float, dict[str, float]]:
    details: dict[str, float] = {}

    for name in (
        "frame_count",
        "action_range",
        "state_continuity",
        "pointing_activity",
        "video_integrity",
    ):
        result = checks.get(name)
        details[name] = float(WEIGHTS[name]) if result is None or result.get("passed", True) else 0.0

    tracking = checks.get("teleop_tracking_quality")
    if tracking is None:
        details["teleop_tracking_quality"] = float(WEIGHTS["teleop_tracking_quality"])
    else:
        bad_ratio = min(max(float(tracking.get("bad_ratio", 0.0)), 0.0), 1.0)
        details["teleop_tracking_quality"] = WEIGHTS["teleop_tracking_quality"] * (1.0 - bad_ratio)

    task_motion = checks.get("task_motion")
    if task_motion is None:
        details["task_motion"] = float(WEIGHTS["task_motion"])
    else:
        points = 0.0
        if task_motion.get("active_frame", -1) >= 0:
            points += _ACTIVE_WEIGHT
        if task_motion.get("movement_detected", False):
            points += _MOVEMENT_WEIGHT
        if task_motion.get("hold_frames", 0) >= DEFAULT_HOLD_MIN_FRAMES:
            points += _HOLD_WEIGHT
        details["task_motion"] = points

    total = max(0.0, min(100.0, sum(details.values())))
    return round(total, 4), details


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(val) for val in value]
    return value


def run_episode_checks(
    dataset_path: Path,
    info: dict[str, Any],
    episode_meta: dict[str, Any],
    dataframe_loader: DataFrameLoader | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> dict[str, dict[str, Any]]:
    dataset_path = Path(dataset_path)
    loader = dataframe_loader or load_episode_table
    ep_idx = int(episode_meta["episode_index"])
    parquet_path = get_parquet_path(dataset_path, info, ep_idx)
    table = loader(parquet_path)
    num_frames = len(table)

    checks: dict[str, dict[str, Any]] = {}
    checks["frame_count"] = check_frame_count(num_frames, max_frames=max_frames)

    if _table_has_column(table, "action.wbc"):
        checks["action_range"] = check_action_range(_stack_column(table, "action.wbc"))
    else:
        checks["action_range"] = {"passed": False, "reason": "Missing action.wbc column"}

    if _table_has_column(table, "observation.state"):
        checks["state_continuity"] = check_state_continuity(_stack_column(table, "observation.state"))
    else:
        checks["state_continuity"] = {"passed": False, "reason": "Missing observation.state column"}

    checks["video_integrity"] = check_video_integrity(get_video_paths(dataset_path, info, ep_idx).values())

    teleop_arrays = _available_stacked_columns(
        table,
        [
            "teleop.vr_3pt_position",
            "teleop.left_hand_joints",
            "teleop.right_hand_joints",
            "teleop.smpl_pose",
        ],
    )
    pointing_arrays = _available_stacked_columns(
        table,
        [
            "teleop.vr_3pt_position",
            "teleop.left_hand_joints",
            "teleop.right_hand_joints",
        ],
    )
    robot_motion_arrays = _available_stacked_columns(
        table,
        [
            "action.wbc",
            "observation.state",
            "observation.eef_state",
        ],
    )
    pointing = check_pointing_activity(pointing_arrays)
    tracking = check_teleop_tracking_quality(teleop_arrays)
    if not pointing.get("passed", False):
        robot_pointing = check_pointing_activity(robot_motion_arrays)
        if robot_pointing.get("passed", False):
            pointing = {
                **robot_pointing,
                "source": "robot_motion",
                "reason": f"Robot motion fallback: {robot_pointing['reason']}",
            }
            tracking = {
                "passed": True,
                "reason": "Teleop columns are zero; robot state/action motion is active",
                "bad_ratio": 0.0,
            }
        else:
            pointing["source"] = "teleop"
    else:
        pointing["source"] = "teleop"
    checks["pointing_activity"] = pointing
    checks["task_motion"] = {
        key: pointing[key]
        for key in ("active_frame", "movement_detected", "hold_frames", "reason")
        if key in pointing
    }
    checks["teleop_tracking_quality"] = tracking
    return checks


def summarise_report(report: dict[str, Any]) -> dict[str, Any]:
    episodes = report.get("episodes", {})
    counts = {"keep": 0, "review": 0, "discard": 0}
    for episode in episodes.values():
        tier = episode.get("tier", "discard")
        if tier in counts:
            counts[tier] += 1

    total = sum(counts.values())
    datasets = report.get("datasets", {})
    datasets_total = len(datasets)
    datasets_invalid = sum(1 for dataset in datasets.values() if not dataset.get("passed", False))
    return {
        "total": total,
        **counts,
        "keep_rate": round(counts["keep"] / max(total, 1) * 100.0, 1),
        "datasets_total": datasets_total,
        "datasets_invalid": datasets_invalid,
    }


def score_dataset(
    dataset_path: Path,
    dataframe_loader: DataFrameLoader | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    dataset_key = str(dataset_path)
    report: dict[str, Any] = {
        "datasets": {dataset_key: check_dataset_structure(dataset_path)},
        "episodes": {},
    }
    if not report["datasets"][dataset_key]["passed"]:
        report["summary"] = summarise_report(report)
        return report

    info = load_info(dataset_path)
    for episode_meta in load_episodes_meta(dataset_path):
        ep_idx = int(episode_meta["episode_index"])
        ep_name = episode_name(ep_idx)
        try:
            checks = run_episode_checks(
                dataset_path,
                info,
                episode_meta,
                dataframe_loader=dataframe_loader,
                max_frames=max_frames,
            )
            score, details = score_episode(checks)
            continuity_passed = checks.get("state_continuity", {}).get("passed", True)
            tier = compute_tier(score, continuity_passed)
            report["episodes"][ep_name] = {
                "dataset": dataset_key,
                "episode_index": ep_idx,
                "score": round(score, 2),
                "tier": tier,
                "checks": _jsonable(checks),
                "details": {key: round(value, 4) for key, value in details.items()},
            }
        except MissingParquetDependency:
            raise
        except Exception as exc:
            report["episodes"][ep_name] = {
                "dataset": dataset_key,
                "episode_index": ep_idx,
                "score": 0.0,
                "tier": "discard",
                "error": str(exc),
                "checks": {},
                "details": {},
            }

    report["summary"] = summarise_report(report)
    return report


def score_datasets(
    dataset_paths: Iterable[Path],
    dataframe_loader: DataFrameLoader | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> dict[str, Any]:
    combined: dict[str, Any] = {"datasets": {}, "episodes": {}}
    for dataset_path in dataset_paths:
        report = score_dataset(dataset_path, dataframe_loader=dataframe_loader, max_frames=max_frames)
        combined["datasets"].update(report.get("datasets", {}))
        for ep_name, ep in report.get("episodes", {}).items():
            key = f"{Path(dataset_path).name}/{ep_name}"
            combined["episodes"][key] = ep
    combined["summary"] = summarise_report(combined)
    return combined


def _normalise_episode_selector(selector: str) -> str:
    selector = selector.strip()
    if not selector:
        return selector
    if selector.isdigit():
        return episode_name(int(selector))
    if "/" in selector:
        return selector.rsplit("/", 1)[-1]
    return selector


def _same_dataset(left: str | None, right: str | Path | None) -> bool:
    if left is None or right is None:
        return False
    left_path = Path(left)
    right_path = Path(right)
    return left == str(right) or left_path == right_path or left_path.name == right_path.name


def select_episode_indices(
    report: dict[str, Any],
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    dataset: str | None = None,
) -> list[int]:
    include_set = {_normalise_episode_selector(item) for item in include if item}
    exclude_set = {_normalise_episode_selector(item) for item in exclude if item}

    selected: set[int] = set()
    for key, episode in report.get("episodes", {}).items():
        if dataset is not None and not _same_dataset(episode.get("dataset"), dataset):
            continue
        ep_idx = int(episode.get("episode_index", -1))
        ep_name = _normalise_episode_selector(key)
        if ep_name in exclude_set or key in exclude_set:
            continue
        if episode.get("tier") == "keep" or ep_name in include_set or key in include_set:
            selected.add(ep_idx)
    return sorted(selected)


def _load_report(report: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    return _read_json(Path(report))


def export_clean_dataset(
    dataset_path: Path,
    report: dict[str, Any] | str | Path,
    output_dir: Path,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    loaded_report = _load_report(report)
    selected_indices = select_episode_indices(
        loaded_report,
        include=include,
        exclude=exclude,
        dataset=str(dataset_path),
    )

    metadata = {
        "source_dataset": str(dataset_path.resolve()),
        "selected_episode_indices": selected_indices,
        "include_overrides": sorted(include),
        "exclude_overrides": sorted(exclude),
        "mapping": {},
    }

    if dry_run:
        return {"dry_run": True, **metadata}

    try:
        import pandas as pd  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - depends on operator env
        raise MissingParquetDependency(
            "Exporting a clean LeRobot dataset requires pandas plus a parquet engine. "
            "Install the data-collection extras, for example: "
            "pip install -e 'gear_sonic[data_collection]'."
        ) from exc

    info = load_info(dataset_path)
    episodes = {int(ep["episode_index"]): ep for ep in load_episodes_meta(dataset_path)}
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    tasks = _read_jsonl(tasks_path)
    fps = int(info.get("fps", 50))
    chunks_size = int(info.get("chunks_size", 1000))

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    new_episodes: list[dict[str, Any]] = []
    for new_idx, old_idx in enumerate(selected_indices):
        old_ep = episodes[old_idx]
        old_parquet = get_parquet_path(dataset_path, info, old_idx)
        table = load_episode_table(old_parquet).copy()
        ep_len = len(table)
        table["episode_index"] = new_idx
        table["frame_index"] = np.arange(ep_len)
        table["index"] = np.arange(total_frames, total_frames + ep_len)
        if "timestamp" in table:
            table["timestamp"] = np.arange(ep_len) / fps

        episode_chunk = new_idx // chunks_size
        parquet_rel = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        ).format(episode_chunk=episode_chunk, episode_index=new_idx)
        dst_parquet = output_dir / parquet_rel
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(dst_parquet)

        for video_key, src_video in get_video_paths(dataset_path, info, old_idx).items():
            video_rel = info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            ).format(
                episode_chunk=episode_chunk,
                episode_index=new_idx,
                video_key=video_key,
            )
            dst_video = output_dir / video_rel
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, dst_video)

        metadata["mapping"][episode_name(new_idx)] = episode_name(old_idx)
        new_episodes.append(
            {
                "episode_index": new_idx,
                "tasks": old_ep.get("tasks", []),
                "length": ep_len,
            }
        )
        total_frames += ep_len

    new_info = dict(info)
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = len(new_episodes) * len(get_video_keys(info))
    new_info["total_chunks"] = (len(new_episodes) + chunks_size - 1) // chunks_size
    new_info["splits"] = {"train": f"0:{len(new_episodes)}"} if new_episodes else {}
    new_info["discarded_episode_indices"] = []

    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(new_info, f, indent=4)
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for episode in new_episodes:
            f.write(json.dumps(episode) + "\n")
    if tasks:
        with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
    modality = dataset_path / "meta" / "modality.json"
    if modality.exists():
        shutil.copy2(modality, meta_dir / "modality.json")
    with open(output_dir / "cleanse_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score and export clean GEAR-SONIC LeRobot episodes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score one or more LeRobot datasets.")
    score_parser.add_argument("--dataset-path", action="append", required=True, type=Path)
    score_parser.add_argument("--output", required=True, type=Path)
    score_parser.add_argument("--max-frames", default=DEFAULT_MAX_FRAMES, type=int)

    export_parser = subparsers.add_parser("export", help="Export keep-tier episodes to a new dataset.")
    export_parser.add_argument("--dataset-path", required=True, type=Path)
    export_parser.add_argument("--report", required=True, type=Path)
    export_parser.add_argument("--output-dir", required=True, type=Path)
    export_parser.add_argument("--include", default="")
    export_parser.add_argument("--exclude", default="")
    export_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "score":
            report = score_datasets(args.dataset_path, max_frames=args.max_frames)
            write_report(report, args.output)
            print(json.dumps(report["summary"], indent=2))
            print(f"Report saved to {args.output}")
        elif args.command == "export":
            metadata = export_clean_dataset(
                args.dataset_path,
                args.report,
                args.output_dir,
                include=_split_csv(args.include),
                exclude=_split_csv(args.exclude),
                dry_run=args.dry_run,
            )
            print(json.dumps(metadata, indent=2))
    except MissingParquetDependency as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
