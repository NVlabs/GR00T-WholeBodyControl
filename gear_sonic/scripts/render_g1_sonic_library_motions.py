"""Render motions from the pinned, released GEAR-SONIC planner.

This utility is deliberately kinematic and simulator-only.  It runs the
official ``planner_sonic.onnx`` recurrently, validates every generated qpos,
and renders the resulting G1 sequence with MuJoCo.  It never opens DDS or
sends robot commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

PLANNER_MODES = {
    "hand_crawling": 8,
    "elbow_crawling": 14,
    "happy_dance": 23,
    "zombie_walk": 24,
}

# Exact MuJoCo/hardware order from policy_parameters.hpp in the released
# deployment stack.
DEFAULT_ANGLES = np.asarray(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
DEFAULT_HEIGHT_M = np.float32(0.788740)
ALLOWED_PRED_NUM_TOKENS = np.asarray([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} must be a nonsymlink regular file")
    return resolved


def _canonical_write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _initial_context() -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[2] = DEFAULT_HEIGHT_M
    qpos[3] = 1.0
    qpos[7:] = DEFAULT_ANGLES
    return np.repeat(qpos[None, None, :], 4, axis=1)


def _validate_planner_abi(session: ort.InferenceSession) -> None:
    expected_inputs = {
        "context_mujoco_qpos": ("tensor(float)", [1, 4, 36]),
        "target_vel": ("tensor(float)", [1]),
        "mode": ("tensor(int64)", [1]),
        "movement_direction": ("tensor(float)", [1, 3]),
        "facing_direction": ("tensor(float)", [1, 3]),
        "random_seed": ("tensor(int64)", [1]),
        "height": ("tensor(float)", [1]),
        "has_specific_target": ("tensor(int64)", [1, 1]),
        "specific_target_positions": ("tensor(float)", [1, 4, 3]),
        "specific_target_headings": ("tensor(float)", [1, 4]),
        "allowed_pred_num_tokens": ("tensor(int64)", [1, 11]),
    }
    actual = {item.name: (item.type, list(item.shape)) for item in session.get_inputs()}
    if actual != expected_inputs:
        raise ValueError(f"released planner input ABI drift: {actual!r}")
    outputs = {item.name: (item.type, list(item.shape)) for item in session.get_outputs()}
    if set(outputs) != {"mujoco_qpos", "num_pred_frames"}:
        raise ValueError(f"released planner output ABI drift: {outputs!r}")


def _infer(
    session: ort.InferenceSession,
    context: np.ndarray,
    *,
    mode: int,
    seed: int,
) -> np.ndarray:
    outputs = session.run(
        ["mujoco_qpos", "num_pred_frames"],
        {
            "context_mujoco_qpos": np.ascontiguousarray(context, dtype=np.float32),
            "target_vel": np.asarray([-1.0], dtype=np.float32),
            "mode": np.asarray([mode], dtype=np.int64),
            "movement_direction": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            "facing_direction": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            "random_seed": np.asarray([seed], dtype=np.int64),
            "height": np.asarray([-1.0], dtype=np.float32),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": ALLOWED_PRED_NUM_TOKENS.copy(),
        },
    )
    padded = np.asarray(outputs[0], dtype=np.float32)
    num_frames = int(np.asarray(outputs[1]).reshape(-1)[0])
    if padded.ndim != 3 or padded.shape[0] != 1 or padded.shape[2] != 36:
        raise ValueError(f"planner qpos output shape drift: {padded.shape}")
    if num_frames < 24 or num_frames > padded.shape[1] or num_frames % 4:
        raise ValueError(f"planner num_pred_frames invalid: {num_frames}")
    valid = np.ascontiguousarray(padded[0, :num_frames])
    if not np.isfinite(valid).all():
        raise ValueError("planner emitted nonfinite qpos")
    quat_norm = np.linalg.norm(valid[:, 3:7], axis=1)
    if float(np.max(np.abs(quat_norm - 1.0))) > 2.0e-4:
        raise ValueError("planner emitted invalid root quaternion")
    if float(np.max(np.abs(valid[:, 7:]))) >= 10.0:
        raise ValueError("planner emitted implausible joint position")
    return valid


def generate_motion(
    session: ort.InferenceSession,
    *,
    mode: int,
    replans: int,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    context = _initial_context()
    frames: list[np.ndarray] = [context[0].copy()]
    lengths: list[int] = []

    # Match the deployment initialization: populate an IDLE trajectory first.
    idle = _infer(session, context, mode=0, seed=seed)
    frames.append(idle[4:])
    context = idle[-4:][None]
    lengths.append(int(idle.shape[0]))

    for index in range(replans):
        valid = _infer(session, context, mode=mode, seed=seed + index + 1)
        frames.append(valid[4:])
        context = valid[-4:][None]
        lengths.append(int(valid.shape[0]))

    motion = np.ascontiguousarray(np.concatenate(frames, axis=0), dtype=np.float32)
    if not np.isfinite(motion).all() or motion.shape[1] != 36:
        raise ValueError("assembled motion invalid")
    return motion, lengths


def _motion_metrics(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, Any]:
    limited = np.flatnonzero(model.jnt_limited)
    violations = 0
    maximum_excess = 0.0
    for joint_id in limited:
        address = int(model.jnt_qposadr[joint_id])
        if address < 7:
            continue
        low, high = model.jnt_range[joint_id]
        values = qpos[:, address]
        excess = np.maximum(low - values, values - high)
        maximum_excess = max(maximum_excess, float(np.max(excess)))
        violations += int(np.count_nonzero(excess > 1.0e-5))
    return {
        "frame_count": int(qpos.shape[0]),
        "minimum_root_height_m": float(np.min(qpos[:, 2])),
        "maximum_root_height_m": float(np.max(qpos[:, 2])),
        "horizontal_displacement_m": float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])),
        "maximum_absolute_joint_position_rad": float(np.max(np.abs(qpos[:, 7:]))),
        "joint_limit_violation_coordinate_count": violations,
        "maximum_joint_limit_excess_rad": maximum_excess,
        "maximum_quaternion_norm_error": float(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.0))),
    }


def render_motion(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    output: Path,
    *,
    fps: int,
) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 960)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 720)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.7
    camera.azimuth = 135.0
    camera.elevation = -18.0
    writer = imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for pose in qpos:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            camera.lookat[:] = pose[:3]
            camera.lookat[2] = max(0.35, float(pose[2]) * 0.55)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--motions",
        nargs="+",
        choices=tuple(PLANNER_MODES),
        default=tuple(PLANNER_MODES),
    )
    parser.add_argument("--replans", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.replans < 1 or args.replans > 100:
        raise ValueError("replans must be in [1, 100]")
    if args.fps < 1 or args.fps > 120:
        raise ValueError("fps must be in [1, 120]")

    root = args.repository_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    planner_path = _regular_file(
        root / "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
        "released planner",
    )
    model_path = _regular_file(root / "gear_sonic_deploy/g1/scene_29dof.xml", "G1 MuJoCo scene")
    report_path = output_dir / "report.json"
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {report_path}")

    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in ort.get_available_providers()
    ]
    session = ort.InferenceSession(str(planner_path), providers=providers)
    _validate_planner_abi(session)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq != 36:
        raise ValueError(f"G1 scene nq drift: {model.nq}")

    records = []
    for offset, name in enumerate(args.motions):
        qpos, lengths = generate_motion(
            session,
            mode=PLANNER_MODES[name],
            replans=args.replans,
            seed=args.seed + 1000 * offset,
        )
        npz_path = output_dir / f"{name}.npz"
        video_path = output_dir / f"{name}.mp4"
        if npz_path.exists() or npz_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {npz_path}")
        np.savez_compressed(
            npz_path,
            qpos=qpos,
            mode=np.asarray([PLANNER_MODES[name]], dtype=np.int64),
            fps=np.asarray([args.fps], dtype=np.int64),
        )
        render_motion(model, qpos, video_path, fps=args.fps)
        records.append(
            {
                "name": name,
                "mode": PLANNER_MODES[name],
                "replan_output_frame_counts": lengths,
                "metrics": _motion_metrics(model, qpos),
                "npz": npz_path.name,
                "npz_sha256": _sha256(npz_path),
                "video": video_path.name,
                "video_sha256": _sha256(video_path),
            }
        )

    _canonical_write(
        report_path,
        {
            "schema_version": 1,
            "kind": "g1_released_sonic_planner_motion_suite",
            "passed": all(record["metrics"]["joint_limit_violation_coordinate_count"] == 0 for record in records),
            "authorization": {
                "simulator_only": True,
                "dds_opened": False,
                "robot_commands_published": False,
                "hardware_authorized": False,
            },
            "repository_revision": "7c90a56cfe04788c4f041daeef5b1e12930675ad",
            "planner_sha256": _sha256(planner_path),
            "scene_sha256": _sha256(model_path),
            "onnxruntime_providers": session.get_providers(),
            "seed": args.seed,
            "replans": args.replans,
            "fps": args.fps,
            "records": records,
        },
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
