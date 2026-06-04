"""Compute PICO controller-to-G1 wrist orientation offsets from a calibration log.

The controller-only VR_3PT path applies CLI offsets as:

    R_controller_ee = R_controller_robot * R_offset

For a marked CALIB_FULL sample, this script solves the local offset that makes
the controller frame match the simulated G1 wrist frame:

    R_controller_robot * R_offset = R_g1_wrist

therefore:

    R_offset = inv(R_controller_robot) * R_g1_wrist
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def _quat_wxyz(row: dict, dotted_path: str) -> np.ndarray:
    value = row
    for key in dotted_path.split("."):
        value = value[key]
    quat = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-9:
        raise ValueError(f"Zero quaternion at {dotted_path}")
    return quat / norm


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_extrinsic_xyz_deg(matrix: np.ndarray) -> np.ndarray:
    """Return extrinsic xyz Euler angles matching scipy Rotation.as_euler('xyz')."""
    sy = -float(matrix[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = math.atan2(-float(matrix[1, 2]), float(matrix[1, 1]))
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def _choose_row(rows: list[dict], label: str, event: str | None) -> dict:
    candidates = [
        row
        for row in rows
        if row.get("posture_marker", {}).get("label") == label
        and (event is None or row.get("event") == event)
        and row.get("g1_ee_fk")
    ]
    if not candidates:
        available = [
            (
                row.get("_line_no"),
                row.get("event"),
                row.get("posture_marker", {}).get("label"),
                row.get("g1_ee_fk", {}).get("source"),
            )
            for row in rows
        ]
        raise SystemExit(
            f"No row found for label={label!r}, event={event!r}. "
            f"Available rows: {available}"
        )
    return candidates[-1]


def _frame_prefix(
    row: dict,
    controller_convention: str | None,
    headset_convention: str | None,
    side: str,
) -> str:
    if (
        (controller_convention is not None or headset_convention is not None)
        and "controller_pose_convention_debug" not in row
    ):
        raise SystemExit(
            f"Row {row.get('_line_no')} does not contain side-by-side convention logs. "
            "Use --convention active for old logs, or collect a new log with the updated manager."
        )
    if controller_convention is not None or headset_convention is not None:
        debug = row["controller_pose_convention_debug"]
        for convention in (controller_convention, headset_convention):
            if convention is not None and convention not in debug:
                raise SystemExit(
                    f"Convention {convention!r} not found in row. "
                    f"Available: {list(debug.keys())}"
                )
        controller_convention = controller_convention or row["pico"].get(
            "active_controller_pose_convention", "xrobotoolkit_unity"
        )
        headset_convention = headset_convention or row["pico"].get(
            "active_headset_pose_convention",
            row["pico"].get("active_controller_pose_convention", "xrobotoolkit_unity"),
        )
        return f"controller_pose_convention_debug.{controller_convention}.{side}_controller_robot_frame"
    return f"pico.{side}_controller_robot_frame"


def _offset_for_side(
    row: dict,
    side: str,
    controller_convention: str | None,
    headset_convention: str | None,
) -> np.ndarray:
    controller_prefix = _frame_prefix(
        row, controller_convention, headset_convention, side
    )
    controller = _quat_wxyz(row, f"{controller_prefix}.orientation.quat_wxyz")
    g1 = _quat_wxyz(row, f"g1_ee_fk.{side}_wrist.orientation.quat_wxyz")
    controller_matrix = _quat_wxyz_to_matrix(controller)
    g1_matrix = _quat_wxyz_to_matrix(g1)
    return controller_matrix.T @ g1_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument(
        "--label",
        default="calib_full",
        help="Posture marker label to use. Default: calib_full",
    )
    parser.add_argument(
        "--event",
        default="posture_marker",
        help=(
            "Event to use. Default: posture_marker. Use an empty string to allow any event "
            "with the requested label."
        ),
    )
    parser.add_argument(
        "--convention",
        choices=("active", "xrobotoolkit_unity", "openxr_unitree"),
        default="active",
        help=(
            "Which logged controller basis convention to use. 'active' uses the active "
            "pico frame in the row and also works on old logs."
        ),
    )
    parser.add_argument(
        "--headset-convention",
        choices=("active", "xrobotoolkit_unity", "openxr_unitree"),
        default="active",
        help=(
            "Which logged headset basis convention to use. For mixed mode, use "
            "--convention openxr_unitree --headset-convention xrobotoolkit_unity."
        ),
    )
    args = parser.parse_args()

    rows = _load_rows(args.log_path)
    event = args.event or None
    row = _choose_row(rows, args.label, event)

    controller_convention = None if args.convention == "active" else args.convention
    headset_convention = (
        None if args.headset_convention == "active" else args.headset_convention
    )
    left_offset = _offset_for_side(row, "left", controller_convention, headset_convention)
    right_offset = _offset_for_side(row, "right", controller_convention, headset_convention)
    left_rpy = _matrix_to_extrinsic_xyz_deg(left_offset)
    right_rpy = _matrix_to_extrinsic_xyz_deg(right_offset)

    print(
        f"Using line {row['_line_no']}: "
        f"event={row.get('event')} label={row.get('posture_marker', {}).get('label')} "
        f"g1_source={row.get('g1_ee_fk', {}).get('source')} "
        f"controller_convention={args.convention} "
        f"headset_convention={args.headset_convention}"
    )
    print("Computed local controller-to-G1 wrist offsets, extrinsic xyz degrees:")
    print(f"  left_controller_offset_rpy  {left_rpy[0]:.3f} {left_rpy[1]:.3f} {left_rpy[2]:.3f}")
    print(
        f"  right_controller_offset_rpy {right_rpy[0]:.3f} "
        f"{right_rpy[1]:.3f} {right_rpy[2]:.3f}"
    )
    print()
    print("Command args:")
    print(
        "  --left_controller_offset_rpy "
        f"{left_rpy[0]:.3f} {left_rpy[1]:.3f} {left_rpy[2]:.3f} "
        "--right_controller_offset_rpy "
        f"{right_rpy[0]:.3f} {right_rpy[1]:.3f} {right_rpy[2]:.3f}"
    )
    if args.convention != "active":
        print(f"  --controller_pose_convention {args.convention}")
    if args.headset_convention != "active":
        print(f"  --headset_pose_convention {args.headset_convention}")


if __name__ == "__main__":
    main()
