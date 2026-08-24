"""Retarget released 29-DoF SONIC planner motions onto true23 G1.

Produces standard MJLab motion NPZ files.  Uses task-space IK to compensate
for missing waist roll/pitch and wrist pitch/yaw joints; never slices policy
outputs or authorizes hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gear_sonic.scripts.render_g1_sonic_library_motions import PLANNER_MODES
from gear_sonic.scripts.retarget_g1_29dof_to_23dof_task_space import (
    _auto_retime_trajectory,
    _model_joint_names,
)
from gear_sonic.utils.g1_23dof_task_space_retarget import (
    DEFAULT_SOURCE_MODEL,
    DEFAULT_TARGET_MODEL,
    RetargetConfig,
    build_mjlab_motion_arrays,
    load_models,
    retarget_trajectory,
    safe_target_joint_bounds,
)

SOURCE_FPS = 30.0
TARGET_FPS = 50.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resample_qpos(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(qpos, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 36 or values.shape[0] < 2:
        raise ValueError(f"released planner qpos must be [frames, 36], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("released planner qpos contains NaN or Inf")
    source_time = np.arange(values.shape[0], dtype=np.float64) / SOURCE_FPS
    frame_count = int(math.floor(source_time[-1] * TARGET_FPS)) + 1
    target_time = np.arange(frame_count, dtype=np.float64) / TARGET_FPS
    root_pos = np.stack(
        [np.interp(target_time, source_time, values[:, index]) for index in range(3)],
        axis=1,
    )
    quaternion_xyzw = values[:, [4, 5, 6, 3]].copy()
    for index in range(1, len(quaternion_xyzw)):
        if float(np.dot(quaternion_xyzw[index - 1], quaternion_xyzw[index])) < 0.0:
            quaternion_xyzw[index] *= -1.0
    root_quat = Slerp(
        source_time,
        Rotation.from_quat(quaternion_xyzw),
    )(target_time).as_quat()[:, [3, 0, 1, 2]]
    joints = np.stack(
        [np.interp(target_time, source_time, values[:, 7 + index]) for index in range(29)],
        axis=1,
    )
    return root_pos, root_quat, joints


def _write_npz_exclusive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--planner-motion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--motions",
        nargs="+",
        choices=tuple(PLANNER_MODES),
        default=("hand_crawling", "happy_dance"),
    )
    parser.add_argument("--velocity-fraction", type=float, default=0.8)
    parser.add_argument("--acceleration-fraction", type=float, default=0.8)
    parser.add_argument("--maximum-time-scale", type=float, default=8.0)
    parser.add_argument("--disable-retiming", action="store_true")
    parser.add_argument("--retarget-max-velocity-rad-s", type=float, default=8.0)
    parser.add_argument("--retarget-max-acceleration-rad-s2", type=float, default=80.0)
    parser.add_argument("--optimize-lower-body", action="store_true")
    parser.add_argument(
        "--source-kind",
        choices=("planner_30hz", "physical_50hz", "physical_targets_50hz"),
        default="planner_30hz",
    )
    parser.add_argument("--allow-failed-kinematic-diagnostic", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.velocity_fraction <= 1.0:
        raise ValueError("velocity fraction must be in (0, 1]")
    if not 0.0 < args.acceleration_fraction <= 1.0:
        raise ValueError("acceleration fraction must be in (0, 1]")
    if args.maximum_time_scale < 1.0:
        raise ValueError("maximum time scale must be >= 1")
    root = args.repository_root.resolve(strict=True)
    input_dir = args.planner_motion_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {report_path}")

    source_model_path = (root / DEFAULT_SOURCE_MODEL).resolve(strict=True)
    target_model_path = (root / DEFAULT_TARGET_MODEL).resolve(strict=True)
    source_model, target_model = load_models(source_model_path, target_model_path)
    config = RetargetConfig(
        max_velocity_rad_s=args.retarget_max_velocity_rad_s,
        max_acceleration_rad_s2=args.retarget_max_acceleration_rad_s2,
        optimize_lower_body=args.optimize_lower_body,
    )
    lower, upper = safe_target_joint_bounds(
        target_model,
        safe_limit_guard_rad=config.safe_limit_guard_rad,
        native_action_clip=config.native_action_clip,
    )
    source_names = _model_joint_names(source_model)
    target_names = _model_joint_names(target_model)

    records: list[dict[str, Any]] = []
    for name in args.motions:
        source_filename = f"{name}.npz" if args.source_kind == "planner_30hz" else f"{name}.physical.npz"
        source_path = (input_dir / source_filename).resolve(strict=True)
        with np.load(source_path, allow_pickle=False) as archive:
            if args.source_kind == "planner_30hz":
                root_pos, root_quat, source_joints = _resample_qpos(archive["qpos"])
                source_fps = SOURCE_FPS
                source_frame_count = int(archive["qpos"].shape[0])
            else:
                qpos = np.asarray(archive["qpos"], dtype=np.float64)
                if qpos.ndim != 2 or qpos.shape[1] != 36 or not np.isfinite(qpos).all():
                    raise ValueError("physical SONIC rollout qpos must be finite [frames, 36]")
                if "control_dt" not in archive.files or not math.isclose(
                    float(archive["control_dt"][0]), 1.0 / TARGET_FPS, rel_tol=0.0, abs_tol=1.0e-12
                ):
                    raise ValueError("physical SONIC rollout must use exact 50-Hz control")
                root_pos = qpos[:, :3].copy()
                root_quat = qpos[:, 3:7].copy()
                if args.source_kind == "physical_targets_50hz":
                    if "target_positions_hardware29" not in archive.files:
                        raise ValueError("physical target source lacks original SONIC actuator targets")
                    source_joints = np.asarray(
                        archive["target_positions_hardware29"],
                        dtype=np.float64,
                    ).copy()
                    if source_joints.shape != (qpos.shape[0], 29):
                        raise ValueError("original SONIC actuator targets shape mismatch")
                else:
                    source_joints = qpos[:, 7:].copy()
                source_fps = TARGET_FPS
                source_frame_count = int(qpos.shape[0])
        base_frame_count = int(source_joints.shape[0])
        if args.disable_retiming:
            retiming = {
                "enabled": False,
                "reason": "preserve_original_sonic_contact_timing",
                "base_frame_count": base_frame_count,
                "retimed_frame_count": base_frame_count,
                "time_scale": 1.0,
                "retarget_max_velocity_rad_s": config.max_velocity_rad_s,
                "retarget_max_acceleration_rad_s2": config.max_acceleration_rad_s2,
            }
        else:
            root_pos, root_quat, source_joints, retiming = _auto_retime_trajectory(
                root_pos,
                root_quat,
                source_joints,
                source_joint_names=source_names,
                target_joint_names=target_names,
                fps=TARGET_FPS,
                max_velocity_rad_s=config.max_velocity_rad_s,
                max_acceleration_rad_s2=config.max_acceleration_rad_s2,
                retained_lower_bounds=lower,
                retained_upper_bounds=upper,
                velocity_fraction=args.velocity_fraction,
                acceleration_fraction=args.acceleration_fraction,
                max_time_scale=args.maximum_time_scale,
            )
        result = retarget_trajectory(
            source_model=source_model,
            target_model=target_model,
            root_pos_w=root_pos,
            root_quat_wxyz=root_quat,
            source_joint_pos_hardware=source_joints,
            fps=TARGET_FPS,
            config=config,
        )
        summary = result.summary()
        if not summary["kinematic_gate_passed"] and not args.allow_failed_kinematic_diagnostic:
            raise RuntimeError(f"{name} failed true23 kinematic gate: {summary['kinematic_gate_failures']}")
        arrays = build_mjlab_motion_arrays(target_model, result)
        output_path = output_dir / f"{name}.true23.npz"
        _write_npz_exclusive(output_path, arrays)
        records.append(
            {
                "name": name,
                "mode": PLANNER_MODES[name],
                "source_path": str(source_path),
                "source_kind": args.source_kind,
                "source_fps": source_fps,
                "source_frame_count": source_frame_count,
                "source_sha256": _sha256(source_path),
                "output": output_path.name,
                "output_sha256": _sha256(output_path),
                "source_30hz_frame_count": (source_frame_count if args.source_kind == "planner_30hz" else None),
                "resampled_50hz_frame_count": base_frame_count,
                "retargeted_50hz_frame_count": int(arrays["joint_pos"].shape[0]),
                "retiming": retiming,
                "summary": summary,
            }
        )

    _write_json_exclusive(
        report_path,
        {
            "schema_version": 1,
            "kind": "g1_released_sonic_library_to_true23_task_space_retarget",
            "passed": all(record["summary"]["kinematic_gate_passed"] for record in records),
            "authorization": {
                "offline_motion_material_only": True,
                "simulator_qualification_complete": False,
                "hardware_authorized": False,
                "robot_commands_published": False,
            },
            "source_model_sha256": _sha256(source_model_path),
            "target_model_sha256": _sha256(target_model_path),
            "source_fps": SOURCE_FPS,
            "target_fps": TARGET_FPS,
            "records": records,
        },
    )
    print(report_path)
    return 0 if all(record["summary"]["kinematic_gate_passed"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
