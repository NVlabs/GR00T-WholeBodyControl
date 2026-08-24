"""Convert BONES-SEED G1 CSV motions into the mjlab tracking npz format.

The repository already converts these CSVs into the IsaacLab motion_lib pickle
format. The mjlab causal trainer consumes a different layout: per-frame joint
state plus world-frame pose and velocity for all 24 robot bodies. Those body
terms are not in the CSV, so this converter runs MuJoCo forward kinematics on
the 23-DoF rev-1.0 model to derive them.

Source CSV layout is ``Frame``, ``root_translate{X,Y,Z}`` in centimetres,
``root_rotate{X,Y,Z}`` as intrinsic-xyz Euler degrees, then 29 ``*_dof`` columns
in MJCF actuator order in degrees. The rev-1.0 body keeps 23 of those 29 joints;
the six absent ones are dropped by name rather than by index.

Output arrays match the existing mjlab motions:

    fps             (1,)
    joint_pos       (T, 23)      joint_vel        (T, 23)
    body_pos_w      (T, 24, 3)   body_quat_w      (T, 24, 4)   wxyz
    body_lin_vel_w  (T, 24, 3)   body_ang_vel_w   (T, 24, 3)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import traceback

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial import transform

DEFAULT_MODEL = "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
ROOT_POSITION_SCALE = 0.01  # centimetres to metres
FREE_JOINT_QPOS = 7


def _model(model_path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(model_path))


def model_joint_names(model: mujoco.MjModel) -> list[str]:
    """Actuated joint names in MJCF order, excluding the floating base."""
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]
    return [name for name in names if name != "floating_base_joint"]


def model_body_names(model: mujoco.MjModel) -> list[str]:
    """Robot body names in MJCF order, excluding the world body."""
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(model.nbody)
    ]
    return names[1:]


def read_bones_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return root position (m), root quaternion (wxyz) and joint angles (rad)."""
    frame = pd.read_csv(csv_path)

    root_pos = (
        np.stack(
            [
                frame["root_translateX"].to_numpy(),
                frame["root_translateY"].to_numpy(),
                frame["root_translateZ"].to_numpy(),
            ],
            axis=1,
        ).astype(np.float64)
        * ROOT_POSITION_SCALE
    )

    euler_deg = np.stack(
        [
            frame["root_rotateX"].to_numpy(),
            frame["root_rotateY"].to_numpy(),
            frame["root_rotateZ"].to_numpy(),
        ],
        axis=1,
    ).astype(np.float64)
    quat_xyzw = transform.Rotation.from_euler(
        "xyz", euler_deg, degrees=True
    ).as_quat()
    root_quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    joints = {
        column[: -len("_dof")]: np.deg2rad(frame[column].to_numpy().astype(np.float64))
        for column in frame.columns
        if column.endswith("_dof")
    }
    return root_pos, root_quat_wxyz, joints


def resample_indices(count: int, fps_source: float, fps_target: float) -> np.ndarray:
    """Nearest-sample indices mapping a source clip onto the target rate."""
    if fps_target > fps_source:
        raise ValueError("upsampling is not supported; pick fps_target <= fps_source")
    duration_s = count / fps_source
    target_count = int(np.floor(duration_s * fps_target))
    if target_count < 2:
        raise ValueError("clip is too short to resample")
    times = np.arange(target_count, dtype=np.float64) / fps_target
    return np.clip(np.round(times * fps_source).astype(np.int64), 0, count - 1)


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """Central differences with one-sided ends, preserving the input shape."""
    return np.gradient(values, dt, axis=0, edge_order=1)


def _angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a quaternion sequence."""
    rotations = transform.Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    count = len(rotations)
    velocity = np.zeros((count, 3), dtype=np.float64)
    for index in range(count):
        lo = max(index - 1, 0)
        hi = min(index + 1, count - 1)
        if hi == lo:
            continue
        delta = rotations[hi] * rotations[lo].inv()
        velocity[index] = delta.as_rotvec() / (dt * (hi - lo))
    return velocity


def convert_clip(
    csv_path: Path,
    model: mujoco.MjModel,
    *,
    fps_source: float,
    fps_target: float,
) -> dict[str, np.ndarray]:
    root_pos, root_quat, joints = read_bones_csv(csv_path)
    joint_names = model_joint_names(model)

    missing = [name for name in joint_names if name not in joints]
    if missing:
        raise ValueError(f"CSV lacks joints required by the model: {missing}")

    indices = resample_indices(len(root_pos), fps_source, fps_target)
    root_pos = root_pos[indices]
    root_quat = root_quat[indices]
    joint_pos = np.stack([joints[name][indices] for name in joint_names], axis=1)

    data = mujoco.MjData(model)
    count = len(indices)
    body_count = model.nbody - 1
    body_pos = np.zeros((count, body_count, 3), dtype=np.float64)
    body_quat = np.zeros((count, body_count, 4), dtype=np.float64)

    for step in range(count):
        data.qpos[:3] = root_pos[step]
        data.qpos[3:FREE_JOINT_QPOS] = root_quat[step]
        data.qpos[FREE_JOINT_QPOS:] = joint_pos[step]
        mujoco.mj_kinematics(model, data)
        body_pos[step] = data.xpos[1:]
        body_quat[step] = data.xquat[1:]

    dt = 1.0 / fps_target
    body_ang_vel = np.stack(
        [_angular_velocity(body_quat[:, body], dt) for body in range(body_count)],
        axis=1,
    )

    return {
        # The existing mjlab motions store fps as float64; every other array is
        # float32. Match that exactly so loaders see an identical schema.
        "fps": np.asarray([fps_target], dtype=np.float64),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": _finite_difference(joint_pos, dt).astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": _finite_difference(body_pos, dt).astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
    }


def _convert_one(
    csv_path: Path,
    output_dir: Path,
    model_path: Path,
    fps_source: float,
    fps_target: float,
    min_frames: int,
) -> tuple[str, str]:
    output = output_dir / f"{csv_path.stem}.npz"
    if output.exists():
        return csv_path.name, "skipped (exists)"
    # tar creates every file in a directory before filling it, so a concurrent
    # extraction leaves zero-length placeholders. Treat those as pending rather
    # than as conversion failures; a later pass picks them up.
    if csv_path.stat().st_size < 1024:
        return csv_path.name, "pending (empty on disk)"
    try:
        model = _model(model_path)
        arrays = convert_clip(
            csv_path, model, fps_source=fps_source, fps_target=fps_target
        )
        frames = arrays["joint_pos"].shape[0]
        if frames < min_frames:
            return csv_path.name, f"rejected (only {frames} frames)"
        for name, value in arrays.items():
            if not np.all(np.isfinite(value)):
                return csv_path.name, f"rejected (non-finite {name})"
        # np.savez appends ".npz" to a path that lacks it, which would rename
        # the temp file out from under the replace(). Write through a handle so
        # the name on disk is exactly what we asked for.
        temporary = output.with_name(f"{output.name}.partial")
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary.replace(output)
        return csv_path.name, f"ok ({frames} frames)"
    except Exception as exc:  # noqa: BLE001
        return csv_path.name, f"failed: {type(exc).__name__}: {exc}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="directory of CSVs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--fps-source", type=float, default=120.0)
    parser.add_argument("--fps-target", type=float, default=50.0)
    parser.add_argument("--min-frames", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")

    csv_paths = sorted(args.input.rglob("*.csv"))
    if args.limit:
        csv_paths = csv_paths[: args.limit]
    if not csv_paths:
        raise SystemExit(f"no CSV files under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"converting {len(csv_paths)} clips -> {args.output}", flush=True)
    results: dict[str, str] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _convert_one,
                path,
                args.output,
                args.model,
                args.fps_source,
                args.fps_target,
                args.min_frames,
            ): path
            for path in csv_paths
        }
        done = 0
        for future in concurrent.futures.as_completed(futures):
            name, status = future.result()
            results[name] = status
            done += 1
            if done % 250 == 0 or done == len(csv_paths):
                ok = sum(1 for value in results.values() if value.startswith("ok"))
                print(f"  {done}/{len(csv_paths)}  ok={ok}", flush=True)

    counts: dict[str, int] = {}
    for status in results.values():
        counts[status.split(" ")[0].rstrip(":")] = (
            counts.get(status.split(" ")[0].rstrip(":"), 0) + 1
        )
    print(f"\nsummary: {counts}")
    if args.report:
        args.report.write_text(json.dumps(results, indent=1, sort_keys=True))
        print(f"report -> {args.report}")
    return 0 if counts.get("ok", 0) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(130) from None
