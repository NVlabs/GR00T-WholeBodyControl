"""OpenCV rendering of torso, left-arm, and right-arm torque residuals."""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .dds import SubscriberStatus
from .history import TorqueHistory
from .joints import PLOT_GROUPS, UPPER_BODY_NAMES


COLORS = (
    (80, 180, 255),
    (255, 160, 70),
    (80, 230, 100),
    (220, 100, 255),
    (60, 220, 240),
    (255, 120, 150),
    (170, 230, 80),
)
BACKGROUND = (22, 24, 29)
GRID_COLOR = (62, 65, 72)
TEXT_COLOR = (225, 225, 225)


def _status_text(status: SubscriberStatus, stale_seconds: float) -> tuple[str, tuple]:
    if status.last_error:
        return status.last_error, (80, 100, 255)
    if not status.state_received:
        return "waiting for rt/lowstate", (0, 210, 255)
    if not status.command_received:
        return "waiting for rt/lowcmd", (0, 210, 255)
    if status.state_age is None or status.command_age is None:
        return "waiting for synchronized data", (0, 210, 255)
    if status.state_age > stale_seconds or status.command_age > stale_seconds:
        return (
            f"stale: state {status.state_age:.2f}s, cmd {status.command_age:.2f}s",
            (0, 210, 255),
        )
    return (
        f"LIVE | state {status.state_age*1000:.0f} ms | cmd {status.command_age*1000:.0f} ms",
        (80, 230, 100),
    )


def _short_name(name: str) -> str:
    return (
        name.replace("left_", "L_")
        .replace("right_", "R_")
        .replace("shoulder_", "sh_")
        .replace("wrist_", "wr_")
        .replace("waist_", "")
    )


def render(
    history: TorqueHistory,
    status: SubscriberStatus,
    *,
    width: int,
    height: int,
    stale_seconds: float,
    include_pd: bool,
    show_q: bool,
    y_limit: float,
    paused: bool,
) -> np.ndarray:
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    now = time.monotonic()
    timestamps, residuals = history.snapshot(now)
    _, q_values, q_cmd_values = history.position_snapshot(now)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if include_pd:
        formula = "delta tau = tau_est - [tau + kp*(q_cmd-q) + kd*(dq_cmd-dq)]"
    else:
        formula = "delta tau = LowState.tau_est - LowCmd.tau"
    if paused:
        formula += " | PAUSED"
    if show_q:
        formula += " | q: solid, q_cmd: dashed"
    cv2.putText(canvas, formula, (18, 27), font, 0.68, TEXT_COLOR, 2, cv2.LINE_AA)
    status_text, status_color = _status_text(status, stale_seconds)
    cv2.putText(
        canvas, status_text, (18, 53), font, 0.52, status_color, 1, cv2.LINE_AA
    )
    controls = "Q/Esc: quit | Space: pause | C: clear"
    (controls_width, _), _ = cv2.getTextSize(controls, font, 0.48, 1)
    cv2.putText(
        canvas,
        controls,
        (max(18, width - controls_width - 18), 27),
        font,
        0.48,
        (170, 175, 185),
        1,
        cv2.LINE_AA,
    )

    outer_margin = 12
    panel_gap = 10
    panel_top = 68
    panel_width = (width - 2 * outer_margin - 2 * panel_gap) // 3
    panel_rows = 2 if show_q else 1
    row_gap = 10
    panel_height = (
        height - panel_top - 12 - row_gap * (panel_rows - 1)
    ) // panel_rows
    for group_number, (title, indices) in enumerate(PLOT_GROUPS):
        x0 = outer_margin + group_number * (panel_width + panel_gap)
        _render_panel(
            canvas,
            f"{title} | delta tau [Nm]",
            indices,
            timestamps,
            residuals,
            now,
            history.window_seconds,
            x0,
            panel_top,
            panel_width,
            panel_height,
            y_limit,
        )
        if show_q:
            _render_position_panel(
                canvas,
                f"{title} | q / q_cmd [rad]",
                indices,
                timestamps,
                q_values,
                q_cmd_values,
                now,
                history.window_seconds,
                x0,
                panel_top + panel_height + row_gap,
                panel_width,
                panel_height,
            )
    return canvas


def _render_panel(
    canvas: np.ndarray,
    title: str,
    indices: tuple[int, ...],
    timestamps: np.ndarray,
    residuals: np.ndarray,
    now: float,
    window_seconds: float,
    x0: int,
    y0: int,
    width: int,
    height: int,
    fixed_y_limit: float,
) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (38, 41, 48), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (75, 78, 86), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (x0 + 12, y0 + 24), font, 0.62, TEXT_COLOR, 2, cv2.LINE_AA)

    graph_left = x0 + 48
    graph_right = x0 + width - 14
    graph_top = y0 + 116
    graph_bottom = y0 + height - 35
    selected = residuals[:, indices] if residuals.size else np.empty((0, len(indices)))
    if fixed_y_limit > 0:
        limit = float(fixed_y_limit)
    else:
        finite = np.abs(selected[np.isfinite(selected)])
        limit = max(0.5, float(np.max(finite)) * 1.1) if finite.size else 1.0

    for grid_index in range(5):
        fraction = grid_index / 4.0
        y = int(round(graph_top + fraction * (graph_bottom - graph_top)))
        value = limit * (1.0 - 2.0 * fraction)
        color = (105, 108, 115) if grid_index == 2 else GRID_COLOR
        cv2.line(canvas, (graph_left, y), (graph_right, y), color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{value:+.1f}",
            (x0 + 4, y + 4),
            font,
            0.34,
            (165, 168, 175),
            1,
            cv2.LINE_AA,
        )
    for grid_index in range(5):
        fraction = grid_index / 4.0
        x = int(round(graph_left + fraction * (graph_right - graph_left)))
        cv2.line(canvas, (x, graph_top), (x, graph_bottom), GRID_COLOR, 1)
        seconds = -window_seconds * (1.0 - fraction)
        cv2.putText(
            canvas,
            f"{seconds:.0f}s",
            (x - 10, graph_bottom + 20),
            font,
            0.32,
            (150, 153, 160),
            1,
            cv2.LINE_AA,
        )

    latest = selected[-1] if selected.shape[0] else np.full(len(indices), np.nan)
    legend_x = x0 + 12
    legend_y = y0 + 48
    columns = 1 if len(indices) <= 3 else 2
    column_width = max(150, (width - 24) // columns)
    for local_index, joint_index in enumerate(indices):
        color = COLORS[local_index % len(COLORS)]
        column = local_index % columns
        row = local_index // columns
        lx = legend_x + column * column_width
        ly = legend_y + row * 18
        value = latest[local_index]
        value_text = "--" if not np.isfinite(value) else f"{value:+.2f}"
        cv2.line(canvas, (lx, ly - 4), (lx + 15, ly - 4), color, 2)
        cv2.putText(
            canvas,
            f"{_short_name(UPPER_BODY_NAMES[joint_index])} {value_text}",
            (lx + 20, ly),
            font,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )

    if timestamps.size < 2:
        cv2.putText(
            canvas,
            "waiting for samples",
            (graph_left + 20, (graph_top + graph_bottom) // 2),
            font,
            0.55,
            (0, 210, 255),
            1,
            cv2.LINE_AA,
        )
        return

    time_start = now - window_seconds
    x_values = graph_left + np.clip(
        (timestamps - time_start) / window_seconds, 0.0, 1.0
    ) * (graph_right - graph_left)
    for local_index in range(len(indices)):
        values = selected[:, local_index]
        valid = np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            continue
        y_values = graph_bottom - np.clip(
            (values[valid] + limit) / (2.0 * limit), 0.0, 1.0
        ) * (graph_bottom - graph_top)
        points = np.column_stack((x_values[valid], y_values)).astype(np.int32)
        cv2.polylines(
            canvas,
            [points.reshape(-1, 1, 2)],
            False,
            COLORS[local_index % len(COLORS)],
            2,
            cv2.LINE_AA,
        )


def _render_position_panel(
    canvas: np.ndarray,
    title: str,
    indices: tuple[int, ...],
    timestamps: np.ndarray,
    q_values: np.ndarray,
    q_cmd_values: np.ndarray,
    now: float,
    window_seconds: float,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (38, 41, 48), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (75, 78, 86), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (x0 + 12, y0 + 24), font, 0.56, TEXT_COLOR, 2, cv2.LINE_AA)

    graph_left = x0 + 48
    graph_right = x0 + width - 14
    graph_top = y0 + 116
    graph_bottom = y0 + height - 35
    selected_q = q_values[:, indices] if q_values.size else np.empty((0, len(indices)))
    selected_cmd = (
        q_cmd_values[:, indices]
        if q_cmd_values.size
        else np.empty((0, len(indices)))
    )
    finite_parts = [
        values[np.isfinite(values)]
        for values in (selected_q, selected_cmd)
        if values.size
    ]
    finite = np.concatenate(finite_parts) if finite_parts else np.empty(0)
    limit = max(0.25, float(np.max(np.abs(finite))) * 1.1) if finite.size else 1.0

    for grid_index in range(5):
        fraction = grid_index / 4.0
        y = int(round(graph_top + fraction * (graph_bottom - graph_top)))
        value = limit * (1.0 - 2.0 * fraction)
        color = (105, 108, 115) if grid_index == 2 else GRID_COLOR
        cv2.line(canvas, (graph_left, y), (graph_right, y), color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{value:+.1f}",
            (x0 + 4, y + 4),
            font,
            0.34,
            (165, 168, 175),
            1,
            cv2.LINE_AA,
        )
    for grid_index in range(5):
        fraction = grid_index / 4.0
        x = int(round(graph_left + fraction * (graph_right - graph_left)))
        cv2.line(canvas, (x, graph_top), (x, graph_bottom), GRID_COLOR, 1)
        seconds = -window_seconds * (1.0 - fraction)
        cv2.putText(
            canvas,
            f"{seconds:.0f}s",
            (x - 10, graph_bottom + 20),
            font,
            0.32,
            (150, 153, 160),
            1,
            cv2.LINE_AA,
        )

    latest_q = selected_q[-1] if selected_q.shape[0] else np.full(len(indices), np.nan)
    latest_cmd = (
        selected_cmd[-1] if selected_cmd.shape[0] else np.full(len(indices), np.nan)
    )
    legend_x = x0 + 12
    legend_y = y0 + 48
    columns = 1 if len(indices) <= 3 else 2
    column_width = max(150, (width - 24) // columns)
    for local_index, joint_index in enumerate(indices):
        color = COLORS[local_index % len(COLORS)]
        column = local_index % columns
        row = local_index // columns
        lx = legend_x + column * column_width
        ly = legend_y + row * 18
        q_text = "--" if not np.isfinite(latest_q[local_index]) else f"{latest_q[local_index]:+.2f}"
        cmd_text = "--" if not np.isfinite(latest_cmd[local_index]) else f"{latest_cmd[local_index]:+.2f}"
        cv2.line(canvas, (lx, ly - 4), (lx + 15, ly - 4), color, 2)
        cv2.putText(
            canvas,
            f"{_short_name(UPPER_BODY_NAMES[joint_index])} {q_text}/{cmd_text}",
            (lx + 20, ly),
            font,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )

    if timestamps.size < 2 or not finite.size:
        cv2.putText(
            canvas,
            "waiting for q / q_cmd",
            (graph_left + 20, (graph_top + graph_bottom) // 2),
            font,
            0.55,
            (0, 210, 255),
            1,
            cv2.LINE_AA,
        )
        return

    time_start = now - window_seconds
    x_values = graph_left + np.clip(
        (timestamps - time_start) / window_seconds, 0.0, 1.0
    ) * (graph_right - graph_left)
    for local_index in range(len(indices)):
        color = COLORS[local_index % len(COLORS)]
        _draw_position_curve(
            canvas,
            x_values,
            selected_q[:, local_index],
            limit,
            graph_top,
            graph_bottom,
            color,
            dashed=False,
        )
        _draw_position_curve(
            canvas,
            x_values,
            selected_cmd[:, local_index],
            limit,
            graph_top,
            graph_bottom,
            color,
            dashed=True,
        )


def _draw_position_curve(
    canvas: np.ndarray,
    x_values: np.ndarray,
    values: np.ndarray,
    limit: float,
    graph_top: int,
    graph_bottom: int,
    color: tuple[int, int, int],
    *,
    dashed: bool,
) -> None:
    valid = np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return
    y_values = graph_bottom - np.clip(
        (values[valid] + limit) / (2.0 * limit), 0.0, 1.0
    ) * (graph_bottom - graph_top)
    points = np.column_stack((x_values[valid], y_values)).astype(np.int32)
    if not dashed:
        cv2.polylines(
            canvas, [points.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA
        )
        return
    for segment_index in range(0, len(points) - 1, 2):
        cv2.line(
            canvas,
            tuple(points[segment_index]),
            tuple(points[segment_index + 1]),
            color,
            1,
            cv2.LINE_AA,
        )
