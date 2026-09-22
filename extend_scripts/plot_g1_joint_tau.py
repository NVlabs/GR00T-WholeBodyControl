#!/usr/bin/env python3
"""Record ``tau_est`` for Unitree G1 upper-body and BrainCo hand motors.

The tool is read-only: it can subscribe to ``rt/lowstate`` or read an exported
Parquet episode. It never publishes a robot command. Upper-body and hand values
are written to separate plots; the hand plot has one column per hand.

Example:

    python gear_sonic_deploy/scripts/plot_g1_joint_tau.py enP8p1s0 \
        --joint-indices 12 15 18 22 25 --duration 20 \
        --output g1_tau_upper_body.png
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


G1_JOINT_NAMES = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
G1_UPPER_BODY_INDICES = tuple(range(12, 29))
PARQUET_TAU_COLUMN = "observation.upper_body.tau_est"
PARQUET_LEFT_HAND_TAU_COLUMN = "observation.left_hand.tau_est"
PARQUET_RIGHT_HAND_TAU_COLUMN = "observation.right_hand.tau_est"
BRAINCO_MOTOR_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
PLOT_GROUPS = (
    ("WAIST", (12, 13, 14)),
    ("RIGHT HAND", tuple(range(22, 29))),
    ("LEFT HAND", tuple(range(15, 22))),
)


def _import_unitree_sdk() -> tuple[Any, Any, Any, Any]:
    """Import an installed SDK, or the checkout bundled with this repository."""
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
    except ImportError:
        repository_root = Path(__file__).resolve().parents[2]
        bundled_sdk = repository_root / "external_dependencies" / "unitree_sdk2_python"
        if bundled_sdk.is_dir():
            sys.path.insert(0, str(bundled_sdk))
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
        except ImportError as error:
            raise SystemExit(
                "unitree_sdk2py is unavailable. Install it, or initialize "
                "external_dependencies/unitree_sdk2_python (including cyclonedds)."
            ) from error
    return ChannelFactoryInitialize, ChannelSubscriber, LowState_, MotorStates_


@dataclass(frozen=True)
class TauStatistics:
    count: int
    mean: float
    mean_abs: float
    rms: float
    std: float
    minimum: float
    maximum: float


class TauRecorder:
    """Thread-safe recorder fed directly by the DDS callback."""

    def __init__(self, joint_indices: tuple[int, ...]) -> None:
        self.joint_indices = joint_indices
        self._lock = threading.Lock()
        self.ready = threading.Event()
        self.motor_count: int | None = None
        self.message_count = 0
        self.last_message_time: float | None = None
        self._recording = False
        self._start_time = 0.0
        self._times: list[float] = []
        self._values: dict[int, list[float]] = {
            index: [] for index in self.joint_indices
        }

    def callback(self, message: Any) -> None:
        now = time.monotonic()
        with self._lock:
            self.motor_count = len(message.motor_state)
            self.message_count += 1
            self.last_message_time = now
            self.ready.set()
            if not self._recording:
                return

            # Indices are validated against the first received message before
            # start() is called, so callback-side indexing is safe.
            self._times.append(now - self._start_time)
            for index in self.joint_indices:
                self._values[index].append(float(message.motor_state[index].tau_est))

    def start(self) -> None:
        with self._lock:
            self._times.clear()
            for values in self._values.values():
                values.clear()
            self._start_time = time.monotonic()
            self._recording = True

    def stop(self) -> tuple[list[float], dict[int, list[float]]]:
        with self._lock:
            self._recording = False
            return self._times.copy(), {
                index: values.copy() for index, values in self._values.items()
            }


class HandTauRecorder:
    """Record asynchronous left/right BrainCo hand state streams."""

    SIDES = ("left", "right")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ready = {side: threading.Event() for side in self.SIDES}
        self.motor_count: dict[str, int] = {}
        self._recording = False
        self._start_time = 0.0
        self._times = {side: [] for side in self.SIDES}
        self._values = {
            side: {index: [] for index in range(len(BRAINCO_MOTOR_NAMES))}
            for side in self.SIDES
        }

    def callback(self, side: str):
        def record(message: Any) -> None:
            now = time.monotonic()
            with self._lock:
                self.motor_count[side] = len(message.states)
                self.ready[side].set()
                if not self._recording:
                    return
                self._times[side].append(now - self._start_time)
                for index in range(len(BRAINCO_MOTOR_NAMES)):
                    self._values[side][index].append(
                        float(message.states[index].tau_est)
                    )

        return record

    def start(self) -> None:
        with self._lock:
            for side in self.SIDES:
                self._times[side].clear()
                for values in self._values[side].values():
                    values.clear()
            self._start_time = time.monotonic()
            self._recording = True

    def stop(self) -> dict[str, tuple[list[float], dict[int, list[float]]]]:
        with self._lock:
            self._recording = False
            return {
                side: (
                    self._times[side].copy(),
                    {
                        index: values.copy()
                        for index, values in self._values[side].items()
                    },
                )
                for side in self.SIDES
            }


def _joint_name(index: int) -> str:
    if index < len(G1_JOINT_NAMES):
        return G1_JOINT_NAMES[index]
    return f"motor_{index}"


def _statistics(values: list[float]) -> TauStatistics:
    if not values:
        raise ValueError("Cannot calculate statistics for an empty signal")
    mean = statistics.fmean(values)
    return TauStatistics(
        count=len(values),
        mean=mean,
        mean_abs=statistics.fmean(abs(value) for value in values),
        rms=math.sqrt(statistics.fmean(value * value for value in values)),
        std=statistics.pstdev(values, mu=mean),
        minimum=min(values),
        maximum=max(values),
    )


def _write_csv(
    path: Path,
    times: list[float],
    values: dict[int, list[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = tuple(values)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["time_s"]
            + [f"tau_est_nm_{index}_{_joint_name(index)}" for index in indices]
        )
        for row_index, sample_time in enumerate(times):
            writer.writerow(
                [f"{sample_time:.9f}"]
                + [f"{values[index][row_index]:.9f}" for index in indices]
            )


def _write_hand_csv(
    path: Path,
    hand_data: dict[str, tuple[list[float], dict[int, list[float]]]],
) -> None:
    """Write both asynchronous hand streams in a single long-form table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["time_s", "hand", "motor_index", "finger", "tau_est"])
        for side in HandTauRecorder.SIDES:
            times, values = hand_data[side]
            for row_index, sample_time in enumerate(times):
                for motor_index, motor_name in enumerate(BRAINCO_MOTOR_NAMES):
                    writer.writerow(
                        [
                            f"{sample_time:.9f}",
                            side,
                            motor_index,
                            motor_name,
                            f"{values[motor_index][row_index]:.9f}",
                        ]
                    )


def _read_parquet_episode(
    parquet_path: Path,
    requested_indices: tuple[int, ...] | None,
) -> tuple[list[float], dict[int, list[float]]]:
    """Read the exported 17-DoF upper-body tau array from one episode."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit(
            "Offline Parquet input requires pyarrow. Install it with: pip install pyarrow"
        ) from error

    try:
        table = parquet.read_table(
            parquet_path,
            columns=["timestamp", PARQUET_TAU_COLUMN],
        )
    except Exception as error:
        raise SystemExit(f"Could not read {parquet_path}: {error}") from error
    if table.num_rows == 0:
        raise SystemExit(f"Parquet episode is empty: {parquet_path}")

    tau_rows = table[PARQUET_TAU_COLUMN].to_pylist()
    width = len(tau_rows[0])
    if width != len(G1_UPPER_BODY_INDICES):
        raise SystemExit(
            f"{PARQUET_TAU_COLUMN} has width {width}; expected "
            f"{len(G1_UPPER_BODY_INDICES)} for G1 motor indices 12..28"
        )
    if any(len(row) != width for row in tau_rows):
        raise SystemExit(f"Inconsistent tau vector widths in {parquet_path}")

    indices = requested_indices or G1_UPPER_BODY_INDICES
    invalid = [index for index in indices if index not in G1_UPPER_BODY_INDICES]
    if invalid:
        raise SystemExit(
            f"Parquet upper-body tau contains only motor indices 12..28; "
            f"invalid requested indices: {invalid}"
        )

    raw_times = [float(value) for value in table["timestamp"].to_pylist()]
    first_time = raw_times[0]
    times = [value - first_time for value in raw_times]
    values = {
        index: [float(row[index - G1_UPPER_BODY_INDICES[0]]) for row in tau_rows]
        for index in indices
    }
    if not all(math.isfinite(value) for value in times):
        raise SystemExit(f"Non-finite timestamps found in {parquet_path}")
    if not all(
        math.isfinite(value)
        for joint_values in values.values()
        for value in joint_values
    ):
        raise SystemExit(f"Non-finite tau values found in {parquet_path}")
    return times, values


def _read_parquet_hand_tau(
    parquet_path: Path,
) -> dict[str, tuple[list[float], dict[int, list[float]]]] | None:
    """Read optional six-motor left/right hand tau arrays from an episode."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit(
            "Offline Parquet input requires pyarrow. Install it with: pip install pyarrow"
        ) from error

    parquet_file = parquet.ParquetFile(parquet_path)
    available = set(parquet_file.schema_arrow.names)
    required = {PARQUET_LEFT_HAND_TAU_COLUMN, PARQUET_RIGHT_HAND_TAU_COLUMN}
    present = required & available
    if not present:
        print(
            "Hand tau is not stored in this episode; skipping the hand plot. "
            f"Expected columns: {', '.join(sorted(required))}",
            file=sys.stderr,
        )
        return None
    if present != required:
        missing = sorted(required - present)
        raise SystemExit(f"Incomplete hand tau data in {parquet_path}; missing: {missing}")

    table = parquet.read_table(
        parquet_path,
        columns=[
            "timestamp",
            PARQUET_LEFT_HAND_TAU_COLUMN,
            PARQUET_RIGHT_HAND_TAU_COLUMN,
        ],
    )
    raw_times = [float(value) for value in table["timestamp"].to_pylist()]
    first_time = raw_times[0]
    times = [value - first_time for value in raw_times]
    result: dict[str, tuple[list[float], dict[int, list[float]]]] = {}
    for side, column in (
        ("left", PARQUET_LEFT_HAND_TAU_COLUMN),
        ("right", PARQUET_RIGHT_HAND_TAU_COLUMN),
    ):
        rows = table[column].to_pylist()
        widths = {len(row) for row in rows}
        if widths != {len(BRAINCO_MOTOR_NAMES)}:
            raise SystemExit(
                f"{column} must contain {len(BRAINCO_MOTOR_NAMES)} values per frame; "
                f"found widths: {sorted(widths)}"
            )
        values = {
            index: [float(row[index]) for row in rows]
            for index in range(len(BRAINCO_MOTOR_NAMES))
        }
        if not all(math.isfinite(value) for series in values.values() for value in series):
            raise SystemExit(f"Non-finite tau values found in {column} of {parquet_path}")
        result[side] = (times.copy(), values)
    return result


def _plot(
    output_path: Path,
    times: list[float],
    values: dict[int, list[float]],
    stats: dict[int, TauStatistics],
    columns: int,
    dpi: int,
    title: str,
    threshold: float | None,
    show: bool,
) -> None:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from error

    if columns != len(PLOT_GROUPS):
        raise ValueError(f"Grouped upper-body layout requires {len(PLOT_GROUPS)} columns")

    unknown_indices = [
        index
        for index in values
        if not any(index in group_indices for _, group_indices in PLOT_GROUPS)
    ]
    if unknown_indices:
        raise SystemExit(
            "The WAIST / RIGHT HAND / LEFT HAND layout supports G1 upper-body "
            f"indices 12..28; unsupported indices: {unknown_indices}"
        )

    grouped_indices = [
        [index for index in group_indices if index in values]
        for _, group_indices in PLOT_GROUPS
    ]
    row_count = max(len(indices) for indices in grouped_indices)
    column_count = len(PLOT_GROUPS)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7.0 * column_count, 3.6 * row_count),
        sharex=True,
        squeeze=False,
    )

    for column, ((group_title, _), indices) in enumerate(
        zip(PLOT_GROUPS, grouped_indices, strict=True)
    ):
        for row in range(row_count):
            axis = axes[row, column]
            if row >= len(indices):
                axis.set_visible(False)
                continue

            index = indices[row]
            tau_values = values[index]
            joint_stats = stats[index]
            axis.plot(
                times, tau_values, linewidth=0.9, color="tab:blue", label="tau_est"
            )
            axis.axhline(0.0, linewidth=0.8, color="black", alpha=0.55)
            axis.axhline(
                joint_stats.mean,
                linewidth=1.2,
                linestyle="--",
                color="tab:orange",
                label=f"mean={joint_stats.mean:.3f}",
            )
            if threshold is not None:
                axis.axhline(
                    threshold,
                    linewidth=1.0,
                    linestyle=":",
                    color="tab:red",
                    label=f"threshold=±{threshold:g}",
                )
                axis.axhline(
                    -threshold,
                    linewidth=1.0,
                    linestyle=":",
                    color="tab:red",
                )
            axis.set_title(
                f"{index}: {_joint_name(index)}\n"
                f"mean|tau|={joint_stats.mean_abs:.3f}, RMS={joint_stats.rms:.3f}, "
                f"range=[{joint_stats.minimum:.3f}, {joint_stats.maximum:.3f}]"
            )
            axis.set_ylabel("tau_est [N·m]")
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right", fontsize="small")
            if row == len(indices) - 1:
                axis.set_xlabel("time [s]")

    observed_duration = times[-1] - times[0] if len(times) > 1 else 0.0
    sample_rate = (len(times) - 1) / observed_duration if observed_duration > 0 else 0.0
    figure.suptitle(
        f"{title}\n{len(times)} samples, {observed_duration:.2f} s, "
        f"average sample rate {sample_rate:.1f} Hz",
        fontsize=14,
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    for column, (group_title, _) in enumerate(PLOT_GROUPS):
        # Place headings after tight_layout so each one remains centered over
        # its final semantic column.
        column_box = axes[0, column].get_position()
        figure.text(
            (column_box.x0 + column_box.x1) / 2.0,
            0.935,
            group_title,
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    print(f"Saved plot: {output_path.resolve()}")
    if show:
        plt.show()
    plt.close(figure)


def _plot_hands(
    output_path: Path,
    hand_data: dict[str, tuple[list[float], dict[int, list[float]]]],
    dpi: int,
    title: str,
    threshold: float | None,
    show: bool,
) -> None:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from error

    sides = (("left", "LEFT HAND"), ("right", "RIGHT HAND"))
    row_count = len(BRAINCO_MOTOR_NAMES)
    figure, axes = plt.subplots(
        row_count,
        len(sides),
        figsize=(14.0, 3.6 * row_count),
        sharex="col",
        squeeze=False,
    )

    sample_summaries = []
    for column, (side, group_title) in enumerate(sides):
        times, values = hand_data[side]
        observed_duration = times[-1] - times[0] if len(times) > 1 else 0.0
        sample_rate = (
            (len(times) - 1) / observed_duration if observed_duration > 0 else 0.0
        )
        sample_summaries.append(
            f"{side}: {len(times)} samples, {observed_duration:.2f} s, {sample_rate:.1f} Hz"
        )
        for row, motor_name in enumerate(BRAINCO_MOTOR_NAMES):
            axis = axes[row, column]
            tau_values = values[row]
            motor_stats = _statistics(tau_values)
            axis.plot(times, tau_values, linewidth=0.9, color="tab:blue", label="tau_est")
            axis.axhline(0.0, linewidth=0.8, color="black", alpha=0.55)
            axis.axhline(
                motor_stats.mean,
                linewidth=1.2,
                linestyle="--",
                color="tab:orange",
                label=f"mean={motor_stats.mean:.3f}",
            )
            if threshold is not None:
                axis.axhline(
                    threshold,
                    linewidth=1.0,
                    linestyle=":",
                    color="tab:red",
                    label=f"threshold=±{threshold:g}",
                )
                axis.axhline(-threshold, linewidth=1.0, linestyle=":", color="tab:red")
            axis.set_title(
                f"{row}: {motor_name}\n"
                f"mean|tau|={motor_stats.mean_abs:.3f}, RMS={motor_stats.rms:.3f}, "
                f"range=[{motor_stats.minimum:.3f}, {motor_stats.maximum:.3f}]"
            )
            axis.set_ylabel("tau_est")
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right", fontsize="small")
            if row == row_count - 1:
                axis.set_xlabel("time [s]")

    figure.suptitle(f"{title}\n{' | '.join(sample_summaries)}", fontsize=14, y=0.99)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    for column, (_, group_title) in enumerate(sides):
        column_box = axes[0, column].get_position()
        figure.text(
            (column_box.x0 + column_box.x1) / 2.0,
            0.935,
            group_title,
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    print(f"Saved hand plot: {output_path.resolve()}")
    if show:
        plt.show()
    plt.close(figure)


def _derived_output_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _robust_contact_score(
    times: list[float],
    values: list[float],
    baseline_window: float,
    smoothing_window: float,
) -> list[float]:
    """Return a smoothed residual divided by a rolling robust noise scale."""
    if len(times) != len(values) or not times:
        raise ValueError("times and values must be non-empty and have equal length")
    if any(current < previous for previous, current in zip(times, times[1:])):
        raise ValueError("timestamps must be monotonically non-decreasing")

    baselines: list[float] = []
    scales: list[float] = []
    baseline_start = 0
    nonzero_changes = [
        abs(current - previous)
        for previous, current in zip(values, values[1:])
        if current != previous
    ]
    resolution = statistics.median(nonzero_changes) if nonzero_changes else 0.0
    scale_floor = max(1e-6, 0.25 * resolution)

    for index, sample_time in enumerate(times):
        cutoff = sample_time - baseline_window
        while baseline_start < index and times[baseline_start] < cutoff:
            baseline_start += 1
        window_values = values[baseline_start : index + 1]
        baseline = statistics.median(window_values)
        mad = statistics.median(abs(value - baseline) for value in window_values)
        baselines.append(baseline)
        scales.append(max(1.4826 * mad, scale_floor))

    residuals = [value - baseline for value, baseline in zip(values, baselines)]
    smoothed: list[float] = []
    smoothing_start = 0
    running_sum = 0.0
    for index, (sample_time, residual) in enumerate(zip(times, residuals)):
        running_sum += residual
        cutoff = sample_time - smoothing_window
        while smoothing_start < index and times[smoothing_start] < cutoff:
            running_sum -= residuals[smoothing_start]
            smoothing_start += 1
        smoothed.append(running_sum / (index - smoothing_start + 1))

    return [value / scale for value, scale in zip(smoothed, scales)]


def _plot_normalized_groups(
    output_path: Path,
    groups: tuple[
        tuple[str, tuple[tuple[str, list[float], list[float]], ...]], ...
    ],
    dpi: int,
    title: str,
    baseline_window: float,
    smoothing_window: float,
    event_threshold: float,
    show: bool,
) -> None:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from error

    row_count = max(len(series) for _, series in groups)
    figure, axes = plt.subplots(
        row_count,
        len(groups),
        figsize=(7.0 * len(groups), 3.6 * row_count),
        sharex="col",
        squeeze=False,
    )

    for column, (_, group_series) in enumerate(groups):
        for row in range(row_count):
            axis = axes[row, column]
            if row >= len(group_series):
                axis.set_visible(False)
                continue

            label, times, values = group_series[row]
            scores = _robust_contact_score(
                times,
                values,
                baseline_window=baseline_window,
                smoothing_window=smoothing_window,
            )
            positive_events = [score > event_threshold for score in scores]
            negative_events = [score < -event_threshold for score in scores]
            event_count = sum(
                positive or negative
                for positive, negative in zip(positive_events, negative_events)
            )
            max_abs_score = max(abs(score) for score in scores)

            axis.plot(times, scores, linewidth=0.9, color="tab:blue", label="score")
            axis.axhline(0.0, linewidth=0.8, color="black", alpha=0.55)
            axis.axhline(
                event_threshold,
                linewidth=1.0,
                linestyle="--",
                color="tab:red",
                label=f"event threshold=±{event_threshold:g}",
            )
            axis.axhline(
                -event_threshold,
                linewidth=1.0,
                linestyle="--",
                color="tab:red",
            )
            axis.fill_between(
                times,
                event_threshold,
                scores,
                where=positive_events,
                color="tab:red",
                alpha=0.3,
                interpolate=True,
            )
            axis.fill_between(
                times,
                -event_threshold,
                scores,
                where=negative_events,
                color="tab:red",
                alpha=0.3,
                interpolate=True,
            )
            event_percent = 100.0 * event_count / len(scores)
            axis.set_title(
                f"{label}\nmax|score|={max_abs_score:.2f}, "
                f"event samples={event_percent:.1f}%"
            )
            axis.set_ylabel("robust contact score")
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right", fontsize="small")
            if row == len(group_series) - 1:
                axis.set_xlabel("time [s]")

    figure.suptitle(
        f"{title}\nbaseline={baseline_window:g} s, "
        f"smoothing={smoothing_window:g} s, threshold=±{event_threshold:g}",
        fontsize=14,
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    for column, (group_title, _) in enumerate(groups):
        column_box = axes[0, column].get_position()
        figure.text(
            (column_box.x0 + column_box.x1) / 2.0,
            0.935,
            group_title,
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    print(f"Saved normalized plot: {output_path.resolve()}")
    if show:
        plt.show()
    plt.close(figure)


def _save_upper_body_normalized(
    args: argparse.Namespace,
    times: list[float],
    values: dict[int, list[float]],
    title: str,
) -> None:
    groups = tuple(
        (
            group_title,
            tuple(
                (f"{index}: {_joint_name(index)}", times, values[index])
                for index in group_indices
                if index in values
            ),
        )
        for group_title, group_indices in PLOT_GROUPS
    )
    output = args.normalized_output or _derived_output_path(args.output, "_normalized")
    _plot_normalized_groups(
        output_path=output,
        groups=groups,
        dpi=args.dpi,
        title=title,
        baseline_window=args.baseline_window,
        smoothing_window=args.smoothing_window,
        event_threshold=args.event_threshold,
        show=args.show,
    )


def _save_hands_normalized(
    args: argparse.Namespace,
    hand_data: dict[str, tuple[list[float], dict[int, list[float]]]],
    hands_output: Path,
    title: str,
) -> None:
    groups = tuple(
        (
            f"{side.upper()} HAND",
            tuple(
                (f"{index}: {motor_name}", hand_data[side][0], hand_data[side][1][index])
                for index, motor_name in enumerate(BRAINCO_MOTOR_NAMES)
            ),
        )
        for side in HandTauRecorder.SIDES
    )
    output = args.hands_normalized_output or _derived_output_path(
        hands_output, "_normalized"
    )
    _plot_normalized_groups(
        output_path=output,
        groups=groups,
        dpi=args.dpi,
        title=title,
        baseline_window=args.baseline_window,
        smoothing_window=args.smoothing_window,
        event_threshold=args.event_threshold,
        show=args.show,
    )


def _print_hand_statistics(
    hand_data: dict[str, tuple[list[float], dict[int, list[float]]]],
) -> None:
    print(
        "\nhand  idx finger       samples    mean  mean|tau|     RMS     std      min      max"
    )
    print(
        "----- --- ------------ ------- ------- ---------- ------- ------- -------- --------"
    )
    for side in HandTauRecorder.SIDES:
        _, values = hand_data[side]
        for index, motor_name in enumerate(BRAINCO_MOTOR_NAMES):
            result = _statistics(values[index])
            print(
                f"{side:<5} {index:>3} {motor_name:<12} {result.count:>7} "
                f"{result.mean:>7.3f} {result.mean_abs:>10.3f} "
                f"{result.rms:>7.3f} {result.std:>7.3f} "
                f"{result.minimum:>8.3f} {result.maximum:>8.3f}"
            )


def _save_hand_outputs(
    args: argparse.Namespace,
    hand_data: dict[str, tuple[list[float], dict[int, list[float]]]],
    title: str,
) -> None:
    _print_hand_statistics(hand_data)
    hands_output = args.hands_output or _derived_output_path(args.output, "_hands")
    if not args.no_csv:
        hands_csv = args.hands_csv or hands_output.with_suffix(".csv")
        _write_hand_csv(hands_csv, hand_data)
        print(f"Saved hand samples: {hands_csv.resolve()}")
    _plot_hands(
        output_path=hands_output,
        hand_data=hand_data,
        dpi=args.dpi,
        title=title,
        threshold=args.hand_threshold,
        show=args.show,
    )
    if not args.no_normalized:
        _save_hands_normalized(
            args,
            hand_data,
            hands_output=hands_output,
            title=f"{title} — normalized contact score",
        )


def _check_plot_dependency() -> None:
    """Fail before DDS capture when the plotting dependency is unavailable."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from error


def parse_args() -> argparse.Namespace:
    default_output = Path(
        f"g1_joint_tau_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Plot upper-body and hand tau_est from live DDS or an exported "
            "Parquet episode. Hands are saved as a separate two-column image."
        )
    )
    parser.add_argument(
        "interface",
        nargs="?",
        default="wlxfc23cd997021",
        help="network interface used for DDS, e.g. eth0 or enP8p1s0",
    )
    parser.add_argument(
        "--joint-indices",
        "--joints",
        type=int,
        nargs="+",
        metavar="INDEX",
        help=(
            "G1 motor indices to plot, for example: 12 15 18 25. Required for "
            "live DDS; Parquet mode defaults to every stored index (12..28)."
        ),
    )
    parser.add_argument(
        "--parquet-file",
        type=Path,
        default=None,
        help=(
            "read observation.upper_body.tau_est from an exported episode "
            "instead of live DDS"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="recording duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"output image path (default: {default_output.name})",
    )
    parser.add_argument(
        "--hands-output",
        type=Path,
        default=None,
        help="hand plot path (default: <output stem>_hands<output suffix>)",
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        default=None,
        help="normalized upper-body plot (default: <output stem>_normalized)",
    )
    parser.add_argument(
        "--hands-normalized-output",
        type=Path,
        default=None,
        help="normalized hand plot (default: <hands-output stem>_normalized)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV output path (default: same basename as --output)",
    )
    parser.add_argument(
        "--hands-csv",
        type=Path,
        default=None,
        help="hand CSV path (default: same basename as the hand plot)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="do not save raw time/tau samples as CSV",
    )
    parser.add_argument(
        "--no-hands",
        action="store_true",
        help="skip hand DDS subscriptions and hand Parquet columns",
    )
    parser.add_argument(
        "--no-normalized",
        action="store_true",
        help="do not create normalized contact-score plots",
    )
    parser.add_argument(
        "--domain-id", type=int, default=0, help="DDS domain id (default: 0)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="seconds to wait for the first rt/lowstate message (default: 5)",
    )
    parser.add_argument(
        "--columns",
        type=int,
        choices=[3],
        default=3,
        help="fixed grouped subplot layout uses 3 columns (default: 3)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="image DPI (default: 150)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="draw optional horizontal threshold lines at ±VALUE N·m",
    )
    parser.add_argument(
        "--hand-threshold",
        type=float,
        default=None,
        help="draw optional horizontal threshold lines at ±VALUE on hand plots",
    )
    parser.add_argument(
        "--baseline-window",
        type=float,
        default=1.0,
        help="rolling median/MAD baseline window in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=float,
        default=0.1,
        help="residual moving-average window in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=3.0,
        help="highlight normalized |score| above this value (default: 3.0)",
    )
    parser.add_argument(
        "--title", default="Unitree G1 joint tau_est", help="figure title"
    )
    parser.add_argument(
        "--show", action="store_true", help="also open the interactive plot window"
    )
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.columns <= 0:
        parser.error("--columns must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.domain_id < 0:
        parser.error("--domain-id must be non-negative")
    if args.threshold is not None and args.threshold < 0:
        parser.error("--threshold must be non-negative")
    if args.hand_threshold is not None and args.hand_threshold < 0:
        parser.error("--hand-threshold must be non-negative")
    if args.baseline_window <= 0:
        parser.error("--baseline-window must be positive")
    if args.smoothing_window <= 0:
        parser.error("--smoothing-window must be positive")
    if args.event_threshold <= 0:
        parser.error("--event-threshold must be positive")
    if args.joint_indices is None and args.parquet_file is None:
        parser.error("--joint-indices is required for live DDS capture")
    if args.joint_indices is not None:
        if any(index < 0 for index in args.joint_indices):
            parser.error("joint indices must be non-negative")
        if len(set(args.joint_indices)) != len(args.joint_indices):
            parser.error("joint indices must not contain duplicates")
    if args.parquet_file is not None and not args.parquet_file.is_file():
        parser.error(f"Parquet file does not exist: {args.parquet_file}")
    if args.output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        parser.error("--output must end in .png, .pdf, or .svg")
    if args.hands_output is not None and args.hands_output.suffix.lower() not in {
        ".png",
        ".pdf",
        ".svg",
    }:
        parser.error("--hands-output must end in .png, .pdf, or .svg")
    for option, path in (
        ("--normalized-output", args.normalized_output),
        ("--hands-normalized-output", args.hands_normalized_output),
    ):
        if path is not None and path.suffix.lower() not in {".png", ".pdf", ".svg"}:
            parser.error(f"{option} must end in .png, .pdf, or .svg")
    return args


def main() -> int:
    args = parse_args()
    _check_plot_dependency()
    requested_indices = (
        tuple(args.joint_indices) if args.joint_indices is not None else None
    )

    if args.parquet_file is not None:
        times, values = _read_parquet_episode(args.parquet_file, requested_indices)
        hand_data = (
            None if args.no_hands else _read_parquet_hand_tau(args.parquet_file)
        )
        indices = tuple(values)
        joint_stats = {index: _statistics(values[index]) for index in indices}
        print(
            f"Loaded {len(times)} samples from {args.parquet_file}: "
            + ", ".join(f"{index}:{_joint_name(index)}" for index in indices)
        )
        print("\nidx joint                       samples    mean  mean|tau|     RMS     std      min      max")
        print("--- --------------------------- ------- ------- ---------- ------- ------- -------- --------")
        for index in indices:
            result = joint_stats[index]
            print(
                f"{index:>3} {_joint_name(index):<27} {result.count:>7} "
                f"{result.mean:>7.3f} {result.mean_abs:>10.3f} "
                f"{result.rms:>7.3f} {result.std:>7.3f} "
                f"{result.minimum:>8.3f} {result.maximum:>8.3f}"
            )
        if not args.no_csv:
            csv_path = args.csv if args.csv is not None else args.output.with_suffix(".csv")
            _write_csv(csv_path, times, values)
            print(f"Saved samples: {csv_path.resolve()}")
        _plot(
            output_path=args.output,
            times=times,
            values=values,
            stats=joint_stats,
            columns=args.columns,
            dpi=args.dpi,
            title=f"{args.title} — {args.parquet_file.stem}",
            threshold=args.threshold,
            show=args.show,
        )
        if not args.no_normalized:
            _save_upper_body_normalized(
                args,
                times,
                values,
                title=(
                    f"{args.title} — {args.parquet_file.stem} — "
                    "normalized contact score"
                ),
            )
        if hand_data is not None:
            _save_hand_outputs(
                args,
                hand_data,
                title=f"Unitree G1 hand tau_est — {args.parquet_file.stem}",
            )
        return 0

    assert requested_indices is not None
    indices = requested_indices
    factory_initialize, subscriber_type, lowstate_type, hand_state_type = (
        _import_unitree_sdk()
    )

    if args.interface:
        factory_initialize(args.domain_id, args.interface)
    else:
        factory_initialize(args.domain_id)

    recorder = TauRecorder(indices)
    hand_recorder = None if args.no_hands else HandTauRecorder()
    subscribers = []
    lowstate_subscriber = subscriber_type("rt/lowstate", lowstate_type)
    lowstate_subscriber.Init(recorder.callback, 10)
    subscribers.append(lowstate_subscriber)
    if hand_recorder is not None:
        for side in HandTauRecorder.SIDES:
            subscriber = subscriber_type(
                f"rt/brainco/{side}/state", hand_state_type
            )
            subscriber.Init(hand_recorder.callback(side), 10)
            subscribers.append(subscriber)

    try:
        if not recorder.ready.wait(args.timeout):
            interface = args.interface or "DDS auto-detection"
            print(
                f"No rt/lowstate received in {args.timeout:g} s via {interface}. "
                "Check the robot connection and network interface.",
                file=sys.stderr,
            )
            return 1

        if hand_recorder is not None:
            missing_hands = [
                side
                for side, ready in hand_recorder.ready.items()
                if not ready.wait(args.timeout)
            ]
            if missing_hands:
                print(
                    "No BrainCo hand state received for: "
                    f"{', '.join(missing_hands)}. Expected DDS topics "
                    "rt/brainco/{left,right}/state. Use --no-hands to record "
                    "upper body only.",
                    file=sys.stderr,
                )
                return 1
            invalid_hands = {
                side: hand_recorder.motor_count[side]
                for side in HandTauRecorder.SIDES
                if hand_recorder.motor_count[side] < len(BRAINCO_MOTOR_NAMES)
            }
            if invalid_hands:
                print(
                    f"Hand state has too few motors: {invalid_hands}; expected at least "
                    f"{len(BRAINCO_MOTOR_NAMES)} per hand.",
                    file=sys.stderr,
                )
                return 2

        assert recorder.motor_count is not None
        invalid = [index for index in indices if index >= recorder.motor_count]
        if invalid:
            print(
                f"Invalid motor indices {invalid}; received LowState has "
                f"{recorder.motor_count} motor states (valid range: "
                f"0..{recorder.motor_count - 1}).",
                file=sys.stderr,
            )
            return 2

        selected = ", ".join(f"{index}:{_joint_name(index)}" for index in indices)
        print(f"Recording rt/lowstate tau_est for {args.duration:g} s: {selected}")
        if hand_recorder is not None:
            print(
                "Recording BrainCo hand tau_est from "
                "rt/brainco/{left,right}/state"
            )
        recorder.start()
        if hand_recorder is not None:
            hand_recorder.start()
        deadline = time.monotonic() + args.duration
        interrupted = False
        try:
            while time.monotonic() < deadline:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        except KeyboardInterrupt:
            interrupted = True

        times, values = recorder.stop()
        hand_data = hand_recorder.stop() if hand_recorder is not None else None
        if not times:
            print("No samples were recorded after capture started.", file=sys.stderr)
            return 3
        if interrupted:
            print(f"Capture interrupted; using {len(times)} collected samples.")

        joint_stats = {index: _statistics(values[index]) for index in indices}
        print("\nidx joint                       samples    mean  mean|tau|     RMS     std      min      max")
        print("--- --------------------------- ------- ------- ---------- ------- ------- -------- --------")
        for index in indices:
            result = joint_stats[index]
            print(
                f"{index:>3} {_joint_name(index):<27} {result.count:>7} "
                f"{result.mean:>7.3f} {result.mean_abs:>10.3f} "
                f"{result.rms:>7.3f} {result.std:>7.3f} "
                f"{result.minimum:>8.3f} {result.maximum:>8.3f}"
            )

        if not args.no_csv:
            csv_path = args.csv if args.csv is not None else args.output.with_suffix(".csv")
            _write_csv(csv_path, times, values)
            print(f"Saved samples: {csv_path.resolve()}")

        _plot(
            output_path=args.output,
            times=times,
            values=values,
            stats=joint_stats,
            columns=args.columns,
            dpi=args.dpi,
            title=args.title,
            threshold=args.threshold,
            show=args.show,
        )
        if not args.no_normalized:
            _save_upper_body_normalized(
                args,
                times,
                values,
                title=f"{args.title} — normalized contact score",
            )
        if hand_data is not None:
            empty_hands = [side for side, (side_times, _) in hand_data.items() if not side_times]
            if empty_hands:
                print(
                    f"No hand samples recorded for: {', '.join(empty_hands)}.",
                    file=sys.stderr,
                )
                return 3
            _save_hand_outputs(
                args,
                hand_data,
                title="Unitree G1 BrainCo hand tau_est",
            )
        return 0
    finally:
        for subscriber in subscribers:
            try:
                subscriber.Close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
