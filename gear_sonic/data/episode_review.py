"""Episode review and trim helpers for LeRobot-format GEAR-SONIC datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html as html_lib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Iterable

import numpy as np

from gear_sonic.data.cleanse_lerobot_dataset import (
    MissingParquetDependency,
    get_parquet_path,
    get_video_keys,
    get_video_paths,
    load_episode_table,
    load_episodes_meta,
    load_info,
)

DEFAULT_VIDEO_KEY = "observation.images.ego_view"
DEFAULT_STATE_COLUMN = "observation.state"
JOINT_GROUPS: tuple[tuple[str, str], ...] = (
    ("left_arm", "Left arm"),
    ("right_arm", "Right arm"),
    ("left_hand", "Left hand"),
    ("right_hand", "Right hand"),
    ("waist", "Waist"),
    ("left_leg", "Left leg"),
    ("right_leg", "Right leg"),
    ("other", "Other"),
)


@dataclass(frozen=True)
class EpisodeReference:
    """A UI-friendly pointer to one episode inside one LeRobot dataset."""

    label: str
    dataset_path: Path
    dataset_name: str
    episode_index: int
    length_frames: int
    fps: float
    duration_s: float
    prompt: str
    video_path: Path | None


@dataclass(frozen=True)
class TrimResult:
    """Result metadata from exporting one trimmed episode as a new dataset."""

    dataset_path: Path
    parquet_path: Path
    video_paths: dict[str, Path]
    frames: int
    duration_s: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class JointToggleGroup:
    """Available and unavailable joint toggles for one semantic group."""

    key: str
    label: str
    enabled: list[str]
    disabled: list[str]
    selected: list[str]


@dataclass(frozen=True)
class TrashResult:
    """Result metadata from moving one episode into a dataset-local trash folder."""

    dataset_path: Path
    trash_dir: Path
    episode_index: int
    moved_paths: tuple[Path, ...]


DataFrameLoader = Callable[[Path], Any]


def format_trash_confirmation_message(item: EpisodeReference) -> str:
    """Return the confirmation prompt shown before moving an episode to trash."""

    return f"Delete current episode? {item.label}"


def gradio_allowed_paths(dataset_paths: Iterable[Path], output_root: Path) -> list[str]:
    """Return minimal directories Gradio may serve files from."""

    allowed: list[str] = []
    seen: set[str] = set()
    for path in [*dataset_paths, output_root]:
        resolved = str(Path(path).resolve())
        if resolved not in seen:
            allowed.append(resolved)
            seen.add(resolved)
    return allowed


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


def load_tasks_meta(dataset_path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(Path(dataset_path) / "meta" / "tasks.jsonl")


def _task_lookup(tasks_meta: Iterable[dict[str, Any]]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for task in tasks_meta:
        if "task_index" in task and "task" in task:
            lookup[int(task["task_index"])] = str(task["task"])
    return lookup


def prompt_for_episode(
    episode_meta: dict[str, Any],
    tasks_meta: Iterable[dict[str, Any]],
) -> str:
    """Return the best available natural-language prompt for an episode."""

    tasks = episode_meta.get("tasks", [])
    if isinstance(tasks, str):
        return tasks
    if isinstance(tasks, list) and tasks:
        if all(isinstance(task, str) for task in tasks):
            return " | ".join(tasks)
        lookup = _task_lookup(tasks_meta)
        resolved = [lookup.get(int(task), str(task)) for task in tasks]
        return " | ".join(resolved)

    if "task_index" in episode_meta:
        lookup = _task_lookup(tasks_meta)
        prompt = lookup.get(int(episode_meta["task_index"]))
        if prompt:
            return prompt

    for task in tasks_meta:
        if "task" in task:
            return str(task["task"])
    return ""


def resolve_episode_video(
    dataset_path: Path,
    info: dict[str, Any],
    episode_index: int,
    preferred_video_key: str = DEFAULT_VIDEO_KEY,
) -> Path | None:
    """Resolve the best video path for an episode, preferring ego_view."""

    paths = get_video_paths(dataset_path, info, episode_index)
    if not paths:
        return None
    preferred = paths.get(preferred_video_key)
    if preferred is not None:
        return preferred
    existing = [path for path in paths.values() if path.exists()]
    return existing[0] if existing else next(iter(paths.values()))


def build_episode_catalog(dataset_paths: Iterable[Path]) -> list[EpisodeReference]:
    """Collect episodes from one or more LeRobot dataset directories."""

    catalog: list[EpisodeReference] = []
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        info = load_info(dataset_path)
        fps = float(info.get("fps", 50))
        tasks_meta = load_tasks_meta(dataset_path)

        for episode_meta in load_episodes_meta(dataset_path):
            ep_idx = int(episode_meta["episode_index"])
            length_frames = int(episode_meta.get("length", 0))
            duration_s = round(length_frames / fps, 6) if fps > 0 else 0.0
            prompt = prompt_for_episode(episode_meta, tasks_meta)
            label = f"{dataset_path.name}/episode_{ep_idx:06d} ({duration_s:.2f}s)"
            catalog.append(
                EpisodeReference(
                    label=label,
                    dataset_path=dataset_path,
                    dataset_name=dataset_path.name,
                    episode_index=ep_idx,
                    length_frames=length_frames,
                    fps=fps,
                    duration_s=duration_s,
                    prompt=prompt,
                    video_path=resolve_episode_video(dataset_path, info, ep_idx),
                )
            )
    return catalog


def find_dataset_paths(dataset_root: Path, dataset_names: Iterable[str] | None = None) -> list[Path]:
    """Find LeRobot dataset directories under a root or from explicit names."""

    dataset_root = Path(dataset_root)
    names = [name for name in dataset_names or [] if name]
    if names:
        candidates = [Path(name) if Path(name).is_absolute() else dataset_root / name for name in names]
    elif (dataset_root / "meta" / "info.json").exists():
        candidates = [dataset_root]
    else:
        candidates = sorted(path for path in dataset_root.iterdir() if path.is_dir())

    return [path for path in candidates if (path / "meta" / "info.json").exists()]


def _table_has_column(table: Any, column: str) -> bool:
    columns = getattr(table, "columns", None)
    if columns is not None:
        return column in columns
    return column in table


def _column_values(table: Any, column: str) -> Any:
    values = table[column]
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return values


def stack_table_column(table: Any, column: str) -> np.ndarray:
    values = _column_values(table, column)
    return np.vstack([np.asarray(value, dtype=np.float64) for value in values])


def episode_timestamps(table: Any, fps: float) -> np.ndarray:
    if _table_has_column(table, "timestamp"):
        values = np.asarray(_column_values(table, "timestamp"), dtype=np.float64)
        if len(values) == len(table):
            return values
    step = 1.0 / fps if fps > 0 else 1.0
    return np.arange(len(table), dtype=np.float64) * step


def compute_joint_velocity_from_arrays(
    positions: np.ndarray,
    timestamps: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Compute per-joint velocity from position rows and timestamps."""

    positions = np.asarray(positions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if positions.ndim != 2:
        raise ValueError("positions must be a 2D array")
    if len(positions) != len(timestamps):
        raise ValueError("positions and timestamps must have the same length")
    velocity = np.zeros_like(positions, dtype=np.float64)
    if len(positions) < 2:
        return velocity

    fallback_dt = 1.0 / fps if fps > 0 else 1.0
    dt = np.diff(timestamps)
    dt = np.where(dt > 0.0, dt, fallback_dt)
    velocity[1:] = np.diff(positions, axis=0) / dt[:, None]
    velocity[0] = velocity[1]
    return velocity


def compute_joint_velocity(
    table: Any,
    fps: float,
    state_column: str = DEFAULT_STATE_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    if not _table_has_column(table, state_column):
        raise KeyError(f"Missing {state_column} column")
    timestamps = episode_timestamps(table, fps)
    positions = stack_table_column(table, state_column)
    return timestamps, compute_joint_velocity_from_arrays(positions, timestamps, fps)


def joint_names(info: dict[str, Any], state_column: str = DEFAULT_STATE_COLUMN) -> list[str]:
    names = info.get("features", {}).get(state_column, {}).get("names", [])
    if isinstance(names, list) and names and all(isinstance(name, str) for name in names):
        return list(names)
    shape = info.get("features", {}).get(state_column, {}).get("shape", [])
    joint_count = int(shape[0]) if shape else 0
    return [f"joint_{idx}" for idx in range(joint_count)]


def joint_group_key(joint_name: str) -> str:
    if joint_name.startswith("left_hand_"):
        return "left_hand"
    if joint_name.startswith("right_hand_"):
        return "right_hand"
    if joint_name.startswith(("left_shoulder_", "left_elbow_", "left_wrist_")):
        return "left_arm"
    if joint_name.startswith(("right_shoulder_", "right_elbow_", "right_wrist_")):
        return "right_arm"
    if joint_name.startswith("waist_"):
        return "waist"
    if joint_name.startswith(("left_hip_", "left_knee_", "left_ankle_")):
        return "left_leg"
    if joint_name.startswith(("right_hip_", "right_knee_", "right_ankle_")):
        return "right_leg"
    return "other"


def build_joint_toggle_groups(
    names: list[str],
    velocity: np.ndarray,
    selected_joint_names: Iterable[str] | None = None,
    default_limit: int = 8,
    active_epsilon: float = 1e-8,
) -> dict[str, JointToggleGroup]:
    """Group joint names for UI toggles and mark zero-velocity joints unavailable."""

    velocity = np.asarray(velocity, dtype=np.float64)
    selected_names = set(selected_joint_names or [])
    selected_provided = selected_joint_names is not None
    max_abs = np.zeros(len(names), dtype=np.float64)
    if velocity.ndim == 2 and velocity.shape[1] > 0:
        usable = min(len(names), velocity.shape[1])
        max_abs[:usable] = np.max(np.abs(velocity[:, :usable]), axis=0)

    enabled_by_group: dict[str, list[str]] = {key: [] for key, _label in JOINT_GROUPS}
    disabled_by_group: dict[str, list[str]] = {key: [] for key, _label in JOINT_GROUPS}
    for idx, name in enumerate(names):
        group = joint_group_key(name)
        if idx < len(max_abs) and max_abs[idx] > active_epsilon:
            enabled_by_group[group].append(name)
        else:
            disabled_by_group[group].append(name)

    if selected_provided:
        default_names = selected_names
    else:
        active_indices = np.flatnonzero(max_abs > active_epsilon)
        ranked_indices = active_indices[np.argsort(max_abs[active_indices])[::-1]]
        default_names = {names[idx] for idx in ranked_indices[:default_limit]}

    groups: dict[str, JointToggleGroup] = {}
    for key, label in JOINT_GROUPS:
        enabled = enabled_by_group[key]
        groups[key] = JointToggleGroup(
            key=key,
            label=label,
            enabled=enabled,
            disabled=disabled_by_group[key],
            selected=[name for name in enabled if name in default_names],
        )
    return groups


def make_disabled_joint_chips_html(joint_names: Iterable[str]) -> str:
    names = list(joint_names)
    if not names:
        return '<div class="episode-disabled-joint-list"></div>'
    chips = "\n".join(
        f'<span class="episode-disabled-joint-chip">{html_lib.escape(name)}</span>' for name in names
    )
    return f'<div class="episode-disabled-joint-list">{chips}</div>'


def make_velocity_payload(
    timestamps: np.ndarray,
    velocity: np.ndarray,
    names: Iterable[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build a JSON-safe velocity payload for UI state or diagnostics."""

    return {
        "timestamps": np.asarray(timestamps, dtype=np.float64).tolist(),
        "velocity": np.asarray(velocity, dtype=np.float64).tolist(),
        "names": list(names),
        "duration_s": float(duration_s),
    }


def navigate_episode_index(current_index: int, action: str, total: int) -> int:
    """Return the catalog index selected by a navigation button action."""

    if total <= 0:
        return 0
    current_index = max(0, min(int(current_index), total - 1))
    if action == "first":
        return 0
    if action == "previous":
        return max(0, current_index - 1)
    if action == "next":
        return min(total - 1, current_index + 1)
    if action == "last":
        return total - 1
    raise ValueError(f"Unknown navigation action: {action}")


def selected_catalog_index_from_table_event(
    event_index: Any,
    catalog_rows: list[list[Any]],
    total: int,
    event_value: Any = None,
) -> int:
    """Resolve a Gradio Dataframe selection event to a catalog row index."""

    if total <= 0:
        return 0
    if isinstance(event_index, (list, tuple)) and event_index:
        return max(0, min(int(event_index[0]), total - 1))
    if isinstance(event_index, int):
        return max(0, min(event_index, total - 1))

    if isinstance(event_value, (list, tuple)) and event_value:
        selected_label = str(event_value[0])
        selected_dataset = str(event_value[1]) if len(event_value) > 1 else None
        selected_episode = event_value[2] if len(event_value) > 2 else None
        for row_index, row in enumerate(catalog_rows[:total]):
            if not row or str(row[0]) != selected_label:
                continue
            if selected_dataset is not None and len(row) > 1 and str(row[1]) != selected_dataset:
                continue
            if selected_episode is not None and len(row) > 2 and row[2] != selected_episode:
                continue
            return row_index

    return 0


def make_joint_velocity_plot(
    timestamps: np.ndarray,
    velocity: np.ndarray,
    names: list[str] | None = None,
    max_joints: int = 8,
) -> Any:
    """Create a matplotlib figure with the most active joint velocities."""

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on operator env
        raise RuntimeError("Joint velocity plots require matplotlib.") from exc

    fig, ax = plt.subplots(figsize=(10, 4))
    if velocity.size == 0 or len(timestamps) == 0:
        ax.set_title("Joint velocity")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("rad/s")
        ax.text(0.5, 0.5, "No velocity data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    mean_abs = np.mean(np.abs(velocity), axis=0)
    count = min(max_joints, velocity.shape[1])
    indices = np.argsort(mean_abs)[::-1][:count]
    names = names or [f"joint_{idx}" for idx in range(velocity.shape[1])]

    for idx in indices:
        label = names[idx] if idx < len(names) else f"joint_{idx}"
        ax.plot(timestamps, velocity[:, idx], linewidth=1.2, label=label)
    ax.set_title("Joint velocity by time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rad/s")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _svg_points(
    timestamps: np.ndarray,
    values: np.ndarray,
    duration_s: float,
    min_value: float,
    max_value: float,
    left: float,
    top: float,
    plot_width: float,
    plot_height: float,
    max_points: int,
) -> str:
    if len(timestamps) == 0:
        return ""
    sample_count = min(len(timestamps), max_points)
    sample_indices = np.unique(np.linspace(0, len(timestamps) - 1, sample_count, dtype=int))
    span = max(max_value - min_value, 1e-9)
    duration_s = max(duration_s, 1e-9)
    points: list[str] = []
    for idx in sample_indices:
        x = left + (float(timestamps[idx]) / duration_s) * plot_width
        y = top + ((max_value - float(values[idx])) / span) * plot_height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def make_joint_velocity_svg_html(
    timestamps: np.ndarray,
    velocity: np.ndarray,
    names: list[str] | None = None,
    selected_joint_names: Iterable[str] | None = None,
    max_joints: int = 8,
    duration_s: float | None = None,
    max_points: int = 600,
) -> str:
    """Render a self-contained SVG plot with a video-time marker target."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 2:
        raise ValueError("velocity must be a 2D array")
    if len(timestamps) != len(velocity):
        raise ValueError("timestamps and velocity must have the same length")

    duration = float(duration_s if duration_s is not None else (timestamps[-1] if len(timestamps) else 0.0))
    names = names or [f"joint_{idx}" for idx in range(velocity.shape[1] if velocity.ndim == 2 else 0)]

    width = 860.0
    height = 340.0
    left = 64.0
    right = 18.0
    top = 24.0
    bottom = 48.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    plot_bottom = top + plot_height

    if velocity.size == 0 or len(timestamps) == 0:
        polylines = ""
        legend = ""
        min_value = -1.0
        max_value = 1.0
    else:
        mean_abs = np.mean(np.abs(velocity), axis=0)
        if selected_joint_names is None:
            count = max(1, min(int(max_joints), velocity.shape[1]))
            active_indices = np.argsort(mean_abs)[::-1][:count]
        else:
            selected = set(selected_joint_names)
            active_indices = np.asarray(
                [idx for idx, name in enumerate(names) if name in selected and idx < velocity.shape[1]],
                dtype=int,
            )
        if len(active_indices) == 0:
            active_indices = np.asarray([], dtype=int)
        selected = velocity[:, active_indices]
        min_value = float(np.min(selected)) if selected.size else -1.0
        max_value = float(np.max(selected)) if selected.size else 1.0
        if abs(max_value - min_value) < 1e-9:
            min_value -= 1.0
            max_value += 1.0
        colors = [
            "#2563eb",
            "#dc2626",
            "#16a34a",
            "#9333ea",
            "#ea580c",
            "#0891b2",
            "#be123c",
            "#4d7c0f",
        ]
        line_parts: list[str] = []
        legend_parts: list[str] = []
        for color_idx, joint_idx in enumerate(active_indices):
            label = names[joint_idx] if joint_idx < len(names) else f"joint_{joint_idx}"
            safe_label = html_lib.escape(label)
            color = colors[color_idx % len(colors)]
            points = _svg_points(
                timestamps,
                velocity[:, joint_idx],
                duration,
                min_value,
                max_value,
                left,
                top,
                plot_width,
                plot_height,
                max_points,
            )
            line_parts.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="2" stroke-linejoin="round" stroke-linecap="round" />'
            )
            legend_parts.append(
                f'<span class="episode-velocity-legend-item">'
                f'<span class="episode-velocity-swatch" style="background:{color}"></span>'
                f"{safe_label}</span>"
            )
        polylines = "\n".join(line_parts)
        legend = "\n".join(legend_parts)

    y_zero = top + ((max_value - 0.0) / max(max_value - min_value, 1e-9)) * plot_height
    y_zero = max(top, min(plot_bottom, y_zero))
    marker_x = left
    y_ticks = "\n".join(
        f'<line x1="{left:.1f}" x2="{left + plot_width:.1f}" y1="{(top + i * plot_height / 4):.1f}" '
        f'y2="{(top + i * plot_height / 4):.1f}" class="episode-velocity-grid" />'
        for i in range(5)
    )
    left_axis = (
        f'<line x1="{left:.1f}" x2="{left:.1f}" y1="{top:.1f}" '
        f'y2="{plot_bottom:.1f}" class="episode-velocity-axis" />'
    )
    bottom_axis = (
        f'<line x1="{left:.1f}" x2="{left + plot_width:.1f}" y1="{plot_bottom:.1f}" '
        f'y2="{plot_bottom:.1f}" class="episode-velocity-axis" />'
    )
    zero_line = (
        f'<line x1="{left:.1f}" x2="{left + plot_width:.1f}" y1="{y_zero:.1f}" '
        f'y2="{y_zero:.1f}" class="episode-velocity-zero" />'
    )
    marker_line = (
        f'<line class="episode-velocity-marker" x1="{marker_x:.1f}" x2="{marker_x:.1f}" '
        f'y1="{top:.1f}" y2="{plot_bottom:.1f}" />'
    )
    start_label = (
        f'<text x="{left:.1f}" y="{height - 14:.1f}" '
        'class="episode-velocity-axis-label">0.0 s</text>'
    )
    end_label = (
        f'<text x="{left + plot_width:.1f}" y="{height - 14:.1f}" '
        f'class="episode-velocity-axis-label episode-velocity-axis-label-end">{duration:.1f} s</text>'
    )
    return f"""
<div class="episode-velocity-plot"
     data-duration="{duration:.6f}"
     data-left="{left:.1f}"
     data-plot-width="{plot_width:.1f}">
  <div class="episode-velocity-header">
    <span>Joint velocity by time</span>
    <span class="episode-velocity-current">0:00.0</span>
  </div>
  <svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Joint velocity by time">
    {left_axis}
    {bottom_axis}
    {y_ticks}
    {zero_line}
    {polylines}
    {marker_line}
    {start_label}
    {end_label}
    <text x="14" y="{top + 8:.1f}" class="episode-velocity-axis-label">{max_value:.2f}</text>
    <text x="14" y="{plot_bottom:.1f}" class="episode-velocity-axis-label">{min_value:.2f}</text>
  </svg>
  <div class="episode-velocity-legend">{legend}</div>
</div>
"""


def _joint_line_parts(
    timestamps: np.ndarray,
    velocity: np.ndarray,
    names: list[str],
    active_indices: np.ndarray,
    selected_names: set[str],
    duration: float,
    min_value: float,
    max_value: float,
    left: float,
    top: float,
    plot_width: float,
    plot_height: float,
    max_points: int,
) -> tuple[str, str]:
    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4d7c0f",
        "#7c3aed",
        "#0f766e",
    ]
    line_parts: list[str] = []
    legend_parts: list[str] = []
    for color_idx, joint_idx in enumerate(active_indices):
        label = names[joint_idx] if joint_idx < len(names) else f"joint_{joint_idx}"
        safe_label = html_lib.escape(label)
        safe_attr = html_lib.escape(label, quote=True)
        color = colors[color_idx % len(colors)]
        visible = label in selected_names
        hidden_attr = "" if visible else ' style="display:none"'
        points = _svg_points(
            timestamps,
            velocity[:, joint_idx],
            duration,
            min_value,
            max_value,
            left,
            top,
            plot_width,
            plot_height,
            max_points,
        )
        line_parts.append(
            f'<polyline class="episode-velocity-line" data-joint="{safe_attr}" points="{points}" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" '
            f'stroke-linecap="round"{hidden_attr} />'
        )
        legend_parts.append(
            f'<span class="episode-velocity-legend-item" data-joint="{safe_attr}"{hidden_attr}>'
            f'<span class="episode-velocity-swatch" style="background:{color}"></span>'
            f"{safe_label}</span>"
        )
    return "\n".join(line_parts), "\n".join(legend_parts)


def make_joint_velocity_toggle_html(
    timestamps: np.ndarray,
    velocity: np.ndarray,
    names: list[str] | None = None,
    selected_joint_names: Iterable[str] | None = None,
    default_limit: int = 8,
    duration_s: float | None = None,
    max_points: int = 300,
) -> str:
    """Render one browser-local velocity plot with grouped joint toggles."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 2:
        raise ValueError("velocity must be a 2D array")
    if len(timestamps) != len(velocity):
        raise ValueError("timestamps and velocity must have the same length")

    names = names or [f"joint_{idx}" for idx in range(velocity.shape[1])]
    duration = float(duration_s if duration_s is not None else (timestamps[-1] if len(timestamps) else 0.0))
    groups = build_joint_toggle_groups(
        names,
        velocity,
        selected_joint_names=selected_joint_names,
        default_limit=default_limit,
    )
    selected_names = {name for group in groups.values() for name in group.selected}
    active_names = {name for group in groups.values() for name in group.enabled}
    active_indices = np.asarray(
        [idx for idx, name in enumerate(names) if name in active_names and idx < velocity.shape[1]],
        dtype=int,
    )

    width = 860.0
    height = 340.0
    left = 64.0
    right = 18.0
    top = 24.0
    bottom = 48.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    plot_bottom = top + plot_height

    selected_velocity = velocity[:, active_indices] if len(active_indices) else np.empty((len(velocity), 0))
    min_value = float(np.min(selected_velocity)) if selected_velocity.size else -1.0
    max_value = float(np.max(selected_velocity)) if selected_velocity.size else 1.0
    if abs(max_value - min_value) < 1e-9:
        min_value -= 1.0
        max_value += 1.0

    polylines, legend = _joint_line_parts(
        timestamps,
        velocity,
        names,
        active_indices,
        selected_names,
        duration,
        min_value,
        max_value,
        left,
        top,
        plot_width,
        plot_height,
        max_points,
    )
    y_zero = top + ((max_value - 0.0) / max(max_value - min_value, 1e-9)) * plot_height
    y_zero = max(top, min(plot_bottom, y_zero))
    y_ticks = "\n".join(
        f'<line x1="{left:.1f}" x2="{left + plot_width:.1f}" y1="{(top + i * plot_height / 4):.1f}" '
        f'y2="{(top + i * plot_height / 4):.1f}" class="episode-velocity-grid" />'
        for i in range(5)
    )
    left_axis = (
        f'<line x1="{left:.1f}" x2="{left:.1f}" y1="{top:.1f}" '
        f'y2="{plot_bottom:.1f}" class="episode-velocity-axis" />'
    )
    bottom_axis = (
        f'<line x1="{left:.1f}" x2="{left + plot_width:.1f}" y1="{plot_bottom:.1f}" '
        f'y2="{plot_bottom:.1f}" class="episode-velocity-axis" />'
    )
    zero_line = (
        f'<line x1="{left:.1f}" x2="{left + plot_width:.1f}" y1="{y_zero:.1f}" '
        f'y2="{y_zero:.1f}" class="episode-velocity-zero" />'
    )
    marker_line = (
        f'<line class="episode-velocity-marker" x1="{left:.1f}" x2="{left:.1f}" '
        f'y1="{top:.1f}" y2="{plot_bottom:.1f}" />'
    )
    start_label = (
        f'<text x="{left:.1f}" y="{height - 14:.1f}" '
        'class="episode-velocity-axis-label">0.0 s</text>'
    )
    end_label = (
        f'<text x="{left + plot_width:.1f}" y="{height - 14:.1f}" '
        f'class="episode-velocity-axis-label episode-velocity-axis-label-end">{duration:.1f} s</text>'
    )

    controls: list[str] = []
    for group_key, group_label in JOINT_GROUPS:
        group = groups[group_key]
        open_attr = " open" if group_key in {"right_arm", "left_arm"} else ""
        enabled_controls = []
        for joint_name in group.enabled:
            safe_name = html_lib.escape(joint_name)
            safe_attr = html_lib.escape(joint_name, quote=True)
            checked = " checked" if joint_name in selected_names else ""
            enabled_controls.append(
                f'<label class="episode-joint-toggle">'
                f'<input class="episode-joint-checkbox" type="checkbox" '
                f'name="{safe_attr}"{checked} data-joint="{safe_attr}">'
                f"<span>{safe_name}</span></label>"
            )
        disabled_controls = [
            f'<span class="episode-disabled-joint-chip">{html_lib.escape(name)}</span>'
            for name in group.disabled
        ]
        controls.append(
            f'<details class="episode-joint-group" data-group="{group_key}"{open_attr}>'
            f"<summary>{html_lib.escape(group_label)}</summary>"
            f'<div class="episode-joint-toggle-list">{"".join(enabled_controls)}</div>'
            f'<div class="episode-disabled-joint-list">{"".join(disabled_controls)}</div>'
            f"</details>"
        )

    return f"""
<div class="episode-velocity-widget">
  <div class="episode-velocity-controls">{"".join(controls)}</div>
  <div class="episode-velocity-plot"
       data-duration="{duration:.6f}"
       data-left="{left:.1f}"
       data-plot-width="{plot_width:.1f}">
    <div class="episode-velocity-header">
      <span>Joint velocity by time</span>
      <span class="episode-velocity-current">0:00.0</span>
    </div>
    <svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Joint velocity by time">
      {left_axis}
      {bottom_axis}
      {y_ticks}
      {zero_line}
      {polylines}
      {marker_line}
      {start_label}
      {end_label}
      <text x="14" y="{top + 8:.1f}" class="episode-velocity-axis-label">{max_value:.2f}</text>
      <text x="14" y="{plot_bottom:.1f}" class="episode-velocity-axis-label">{min_value:.2f}</text>
    </svg>
    <div class="episode-velocity-legend">{legend}</div>
  </div>
</div>
"""


def review_sync_head() -> str:
    """Return CSS/JS that keeps the velocity marker synced to video playback."""

    return """
<style>
.episode-velocity-widget {
  display: grid;
  gap: 10px;
  color: #111827 !important;
  color-scheme: light;
}
.episode-video-current {
  margin: 6px 0 10px;
  color: #111827 !important;
  font-size: 13px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  text-align: right;
}
.episode-velocity-controls {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 8px;
  color: #111827 !important;
}
.episode-joint-group {
  border: 1px solid #e4e4e7;
  border-radius: 8px;
  padding: 6px 8px;
  background: #fafafa;
  color: #111827 !important;
}
.episode-joint-group summary {
  cursor: pointer;
  font-weight: 600;
  color: #111827 !important;
}
.episode-joint-toggle-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 7px;
}
.episode-joint-toggle {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 1px solid #d4d4d8;
  border-radius: 6px;
  padding: 3px 7px;
  background: #fff;
  color: #111827 !important;
  font-size: 12px;
  line-height: 1.25;
}
.episode-joint-checkbox {
  margin: 0;
}
.episode-velocity-plot {
  border: 1px solid #d4d4d8;
  border-radius: 8px;
  padding: 10px 12px;
  background: #fff;
  color: #111827 !important;
  color-scheme: light;
}
.episode-velocity-header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
  font-weight: 600;
  margin-bottom: 6px;
  color: #111827 !important;
}
.episode-velocity-current {
  font-variant-numeric: tabular-nums;
  color: #111827 !important;
}
.episode-velocity-plot svg {
  display: block;
  width: 100%;
  height: auto;
}
.episode-velocity-axis {
  stroke: #71717a;
  stroke-width: 1.2;
}
.episode-velocity-grid {
  stroke: #e4e4e7;
  stroke-width: 1;
}
.episode-velocity-zero {
  stroke: #a1a1aa;
  stroke-width: 1;
  stroke-dasharray: 4 4;
}
.episode-velocity-marker {
  stroke: #111827;
  stroke-width: 2;
}
.episode-velocity-axis-label {
  fill: #111827 !important;
  font-size: 13px;
}
.episode-velocity-axis-label-end {
  text-anchor: end;
}
.episode-velocity-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  margin-top: 6px;
  font-size: 12px;
  color: #111827 !important;
}
.episode-velocity-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: #111827 !important;
}
.episode-velocity-swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  display: inline-block;
}
.episode-disabled-joint-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  min-height: 12px;
  margin: 4px 0 10px;
}
.episode-disabled-joint-chip {
  display: inline-flex;
  align-items: center;
  border: 1px solid #d4d4d8;
  border-radius: 6px;
  padding: 3px 7px;
  background: #f4f4f5;
  color: #71717a;
  font-size: 12px;
}
</style>
<script>
(function () {
  if (window.__gearSonicEpisodeReviewSyncInstalled) {
    return;
  }
  window.__gearSonicEpisodeReviewSyncInstalled = true;

  function formatVideoTime(value) {
    var time = Number(value || 0);
    if (!Number.isFinite(time) || time < 0) {
      time = 0;
    }
    var totalTenths = Math.round(time * 10);
    var minutes = Math.floor(totalTenths / 600);
    var seconds = Math.floor((totalTenths % 600) / 10);
    var tenths = totalTenths % 10;
    var secondsText = seconds < 10 ? "0" + seconds : String(seconds);
    return minutes + ":" + secondsText + "." + tenths;
  }

  function ensureVideoTimestamp(video) {
    if (!video) {
      return null;
    }
    var videoTimeLabel = document.querySelector(".episode-video-current");
    if (!videoTimeLabel) {
      videoTimeLabel = document.createElement("div");
      videoTimeLabel.className = "episode-video-current";
      videoTimeLabel.setAttribute("aria-live", "polite");
      videoTimeLabel.textContent = "0:00.0";
    }
    if (video.nextElementSibling !== videoTimeLabel) {
      video.insertAdjacentElement("afterend", videoTimeLabel);
    }
    return videoTimeLabel;
  }

  function updateVideoTimestamp(video, time) {
    var videoTimeLabel = ensureVideoTimestamp(video);
    if (videoTimeLabel) {
      videoTimeLabel.textContent = formatVideoTime(time);
    }
  }

  function updateVelocityMarker() {
    var video = document.querySelector("video");
    var time = video ? Number(video.currentTime || 0) : 0;
    if (!Number.isFinite(time)) {
      time = 0;
    }
    updateVideoTimestamp(video, time);
    var plot = document.querySelector(".episode-velocity-plot");
    if (!plot) {
      return;
    }
    var marker = plot.querySelector(".episode-velocity-marker");
    var label = plot.querySelector(".episode-velocity-current");
    if (!marker) {
      return;
    }
    var duration = Number(plot.dataset.duration || 0);
    var left = Number(plot.dataset.left || 0);
    var plotWidth = Number(plot.dataset.plotWidth || 1);
    if (!Number.isFinite(duration) || duration <= 0) {
      duration = video && Number.isFinite(video.duration) ? Number(video.duration) : 0;
    }
    var boundedTime = duration > 0 ? Math.max(0, Math.min(time, duration)) : 0;
    var x = left + (duration > 0 ? boundedTime / duration : 0) * plotWidth;
    marker.setAttribute("x1", String(x));
    marker.setAttribute("x2", String(x));
    if (label) {
      label.textContent = formatVideoTime(boundedTime);
    }
  }

  function setJointVisibility(widget, joint, visible) {
    var targets = widget.querySelectorAll(
      ".episode-velocity-line, .episode-velocity-legend-item"
    );
    targets.forEach(function (target) {
      if (target.dataset.joint === joint) {
        target.style.display = visible ? "" : "none";
      }
    });
  }

  function toggleJointVisibility(input) {
    var widget = input.closest(".episode-velocity-widget");
    if (!widget) {
      return;
    }
    setJointVisibility(widget, input.dataset.joint || input.name, input.checked);
  }

  document.addEventListener("change", function (event) {
    var input = event.target;
    if (!input || !input.classList || !input.classList.contains("episode-joint-checkbox")) {
      return;
    }
    toggleJointVisibility(input);
  });

  var attachedVideo = null;
  function attachVideo() {
    var video = document.querySelector("video");
    if (video && video !== attachedVideo) {
      attachedVideo = video;
      ["timeupdate", "play", "pause", "seeked", "loadedmetadata"].forEach(function (eventName) {
        video.addEventListener(eventName, updateVelocityMarker);
      });
    }
    updateVelocityMarker();
  }

  document.addEventListener("DOMContentLoaded", attachVideo);
  setInterval(attachVideo, 400);
})();
</script>
"""


def trim_frame_indices(timestamps: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Return frame indices in the half-open interval [start_s, end_s)."""

    if start_s < 0:
        raise ValueError("trim start must be >= 0")
    if end_s <= start_s:
        raise ValueError("trim end must be greater than trim start")
    timestamps = np.asarray(timestamps, dtype=np.float64)
    indices = np.flatnonzero((timestamps >= start_s) & (timestamps < end_s))
    if len(indices) == 0:
        raise ValueError("trim interval contains no frames")
    return indices


def trim_episode_table(
    table: Any,
    start_s: float,
    end_s: float,
    fps: float,
    output_episode_index: int = 0,
) -> tuple[Any, np.ndarray]:
    """Trim a pandas-like table and reset LeRobot episode columns."""

    timestamps = episode_timestamps(table, fps)
    indices = trim_frame_indices(timestamps, start_s, end_s)
    if not hasattr(table, "iloc"):
        raise TypeError("trim_episode_table requires a pandas-like table with .iloc")

    trimmed = table.iloc[indices].copy().reset_index(drop=True)
    frame_count = len(trimmed)
    if "episode_index" in trimmed.columns:
        trimmed["episode_index"] = output_episode_index
    if "frame_index" in trimmed.columns:
        trimmed["frame_index"] = range(frame_count)
    if "index" in trimmed.columns:
        trimmed["index"] = range(frame_count)
    if "timestamp" in trimmed.columns:
        step = 1.0 / fps if fps > 0 else 1.0
        trimmed["timestamp"] = np.arange(frame_count, dtype=np.float64) * step
    return trimmed, indices


def _format_seconds(value: float) -> str:
    return f"{value:.3f}".replace(".", "p").replace("-", "m")


def default_trim_dataset_path(
    output_root: Path,
    dataset_name: str,
    episode_index: int,
    start_s: float,
    end_s: float,
) -> Path:
    output_root = Path(output_root)
    return output_root / f"{dataset_name}_episode_{episode_index:06d}"


def replace_trim_output_path(dest: Path) -> None:
    """Remove an existing trim output directory before writing the final trim."""

    if not dest.exists() and not dest.is_symlink():
        return
    if dest.is_dir() and not dest.is_symlink():
        shutil.rmtree(dest)
    else:
        dest.unlink()


def replace_previous_trim_outputs(
    output_root: Path,
    dataset_name: str,
    episode_index: int,
    final_dest: Path,
) -> None:
    """Remove previous final and legacy versioned trims for one source episode."""

    output_root = Path(output_root)
    final_dest = Path(final_dest)
    candidates = [final_dest]
    legacy_prefix = f"{dataset_name}_episode_{episode_index:06d}_"
    if output_root.exists():
        candidates.extend(path for path in output_root.iterdir() if path.name.startswith(legacy_prefix))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        replace_trim_output_path(candidate)


def trim_video_with_ffmpeg(
    src_video: Path,
    dst_video: Path,
    start_s: float,
    end_s: float,
) -> str | None:
    """Trim video with ffmpeg; copy the full source video if ffmpeg fails."""

    if not src_video.exists():
        return f"Missing source video: {src_video}"
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy2(src_video, dst_video)
        return "ffmpeg was not found; copied the full source video instead"

    duration_s = end_s - start_s
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_video),
        "-ss",
        f"{start_s:.6f}",
        "-t",
        f"{duration_s:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dst_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0 and dst_video.exists() and dst_video.stat().st_size > 0:
        return None

    shutil.copy2(src_video, dst_video)
    stderr = result.stderr.strip().replace("\n", " ")
    return f"ffmpeg trim failed; copied the full source video instead: {stderr[:240]}"


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _unique_trash_dir(dataset_path: Path, episode_index: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(dataset_path) / ".trash" / f"episode_{episode_index:06d}_{timestamp}"
    if not base.exists():
        return base
    for suffix in range(1, 1000):
        candidate = base.with_name(f"{base.name}_{suffix:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique trash directory under {dataset_path / '.trash'}")


def move_episode_to_trash(
    dataset_path: Path,
    episode_index: int,
    reason: str = "",
) -> TrashResult:
    """Move one episode's parquet/videos to dataset-local trash and remove metadata."""

    dataset_path = Path(dataset_path)
    info = load_info(dataset_path)
    episodes = load_episodes_meta(dataset_path)
    removed_episode = next(
        (episode for episode in episodes if int(episode["episode_index"]) == episode_index),
        None,
    )
    if removed_episode is None:
        raise ValueError(f"Episode {episode_index} not found in {dataset_path}")

    trash_dir = _unique_trash_dir(dataset_path, episode_index)
    trash_dir.mkdir(parents=True)
    moved_paths: list[Path] = []

    candidate_paths = [get_parquet_path(dataset_path, info, episode_index)]
    candidate_paths.extend(get_video_paths(dataset_path, info, episode_index).values())
    for src in candidate_paths:
        if not src.exists():
            continue
        rel = src.relative_to(dataset_path)
        dst = trash_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved_paths.append(dst)

    remaining_episodes = [
        episode for episode in episodes if int(episode["episode_index"]) != episode_index
    ]
    _write_jsonl(dataset_path / "meta" / "episodes.jsonl", remaining_episodes)

    updated_info = dict(info)
    updated_info["total_episodes"] = len(remaining_episodes)
    updated_info["total_frames"] = int(sum(int(ep.get("length", 0)) for ep in remaining_episodes))
    updated_info["total_videos"] = len(remaining_episodes) * len(get_video_keys(info))
    updated_info["total_chunks"] = 1 if remaining_episodes else 0
    updated_info["splits"] = {"train": f"0:{len(remaining_episodes)}"}
    _write_json(dataset_path / "meta" / "info.json", updated_info)

    trash_metadata = {
        "source_dataset": str(dataset_path),
        "episode_index": episode_index,
        "reason": reason,
        "episode_meta": removed_episode,
        "moved_paths": [str(path.relative_to(trash_dir)) for path in moved_paths],
    }
    _write_json(trash_dir / "trash_metadata.json", trash_metadata)
    with open(dataset_path / ".trash" / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({**trash_metadata, "trash_dir": str(trash_dir)}) + "\n")

    return TrashResult(
        dataset_path=dataset_path,
        trash_dir=trash_dir,
        episode_index=episode_index,
        moved_paths=tuple(moved_paths),
    )


def trim_episode_to_dataset(
    dataset_path: Path,
    episode_index: int,
    start_s: float,
    end_s: float,
    output_root: Path,
    output_path: Path | None = None,
    dataframe_loader: DataFrameLoader | None = None,
    video_trimmer: Callable[[Path, Path, float, float], str | None] = trim_video_with_ffmpeg,
) -> TrimResult:
    """Export one episode interval as a new one-episode LeRobot dataset."""

    dataset_path = Path(dataset_path)
    info = load_info(dataset_path)
    fps = float(info.get("fps", 50))
    episodes = load_episodes_meta(dataset_path)
    episode_meta = next(
        (episode for episode in episodes if int(episode["episode_index"]) == episode_index),
        None,
    )
    if episode_meta is None:
        raise ValueError(f"Episode {episode_index} not found in {dataset_path}")

    loader = dataframe_loader or load_episode_table
    table = loader(get_parquet_path(dataset_path, info, episode_index))
    trimmed, _indices = trim_episode_table(table, start_s, end_s, fps, output_episode_index=0)

    dest = Path(output_path) if output_path else default_trim_dataset_path(
        output_root,
        dataset_name=dataset_path.name,
        episode_index=episode_index,
        start_s=start_s,
        end_s=end_s,
    )
    replace_previous_trim_outputs(output_root, dataset_path.name, episode_index, dest)

    meta_dir = dest / "meta"
    meta_dir.mkdir(parents=True)

    chunks_size = int(info.get("chunks_size", 1000))
    data_path_pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    parquet_rel = data_path_pattern.format(episode_chunk=0, episode_index=0)
    parquet_path = dest / parquet_rel
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        trimmed.to_parquet(parquet_path)
    except Exception as exc:  # pragma: no cover - depends on operator env
        message = str(exc).lower()
        if isinstance(exc, ImportError) or "pyarrow" in message or "parquet" in message:
            raise MissingParquetDependency(
                "Writing trimmed episodes requires pandas plus a parquet engine "
                "(for example pandas + pyarrow)."
            ) from exc
        raise

    video_paths: dict[str, Path] = {}
    warnings: list[str] = []
    source_videos = get_video_paths(dataset_path, info, episode_index)
    video_path_pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    for video_key in get_video_keys(info):
        dst_rel = video_path_pattern.format(
            episode_chunk=0,
            episode_index=0,
            video_key=video_key,
        )
        dst_video = dest / dst_rel
        video_paths[video_key] = dst_video
        warning = video_trimmer(source_videos[video_key], dst_video, start_s, end_s)
        if warning:
            warnings.append(f"{video_key}: {warning}")

    out_info = json.loads(json.dumps(info))
    out_info["total_episodes"] = 1
    out_info["total_frames"] = int(len(trimmed))
    out_info["total_videos"] = len(video_paths)
    out_info["total_chunks"] = 1
    out_info["chunks_size"] = chunks_size
    out_info["splits"] = {"train": "0:1"}
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(out_info, f, indent=4)

    tasks_meta = load_tasks_meta(dataset_path)
    _write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {
                "episode_index": 0,
                "tasks": episode_meta.get("tasks", []),
                "length": int(len(trimmed)),
            }
        ],
    )
    if tasks_meta:
        _write_jsonl(meta_dir / "tasks.jsonl", tasks_meta)

    modality_src = dataset_path / "meta" / "modality.json"
    if modality_src.exists():
        shutil.copy2(modality_src, meta_dir / "modality.json")

    with open(dest / "trim_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_dataset": str(dataset_path),
                "source_episode_index": episode_index,
                "start_s": start_s,
                "end_s": end_s,
                "source_prompt": prompt_for_episode(episode_meta, tasks_meta),
                "frames": int(len(trimmed)),
            },
            f,
            indent=4,
        )

    return TrimResult(
        dataset_path=dest,
        parquet_path=parquet_path,
        video_paths=video_paths,
        frames=int(len(trimmed)),
        duration_s=round(len(trimmed) / fps, 6) if fps > 0 else 0.0,
        warnings=tuple(warnings),
    )
