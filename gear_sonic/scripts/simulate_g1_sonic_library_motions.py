"""Run released GEAR-SONIC planner motions through released 29-DoF policy.

This is an offline MuJoCo-only reproduction of the released C++ deployment
observation and action contracts.  It never opens DDS or publishes commands.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

from gear_sonic.scripts.render_g1_sonic_library_motions import (
    DEFAULT_ANGLES,
    PLANNER_MODES,
)

CONTROL_DT = 0.02
PLANNER_FPS = 30.0
HISTORY_LENGTH = 10

# C++ ``mujoco_to_isaaclab``: for each IsaacLab coordinate, source MuJoCo
# coordinate.  C++ names predate the current comments and are easy to invert.
ISAAC_TO_MUJOCO_INDEX = np.asarray(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)
# C++ ``isaaclab_to_mujoco``: for each MuJoCo coordinate, source IsaacLab
# coordinate when mapping policy action to a hardware target.
MUJOCO_TO_ISAAC_INDEX = np.asarray(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQUENCY = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQUENCY**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQUENCY


STIFFNESS_5020 = _stiffness(ARMATURE_5020)
STIFFNESS_7520_14 = _stiffness(ARMATURE_7520_14)
STIFFNESS_7520_22 = _stiffness(ARMATURE_7520_22)
STIFFNESS_4010 = _stiffness(ARMATURE_4010)
DAMPING_5020 = _damping(ARMATURE_5020)
DAMPING_7520_14 = _damping(ARMATURE_7520_14)
DAMPING_7520_22 = _damping(ARMATURE_7520_22)
DAMPING_4010 = _damping(ARMATURE_4010)

KPS = np.asarray(
    [
        STIFFNESS_7520_22,
        STIFFNESS_7520_22,
        STIFFNESS_7520_14,
        STIFFNESS_7520_22,
        2.0 * STIFFNESS_5020,
        2.0 * STIFFNESS_5020,
        STIFFNESS_7520_22,
        STIFFNESS_7520_22,
        STIFFNESS_7520_14,
        STIFFNESS_7520_22,
        2.0 * STIFFNESS_5020,
        2.0 * STIFFNESS_5020,
        STIFFNESS_7520_14,
        2.0 * STIFFNESS_5020,
        2.0 * STIFFNESS_5020,
        *([STIFFNESS_5020] * 5),
        STIFFNESS_4010,
        STIFFNESS_4010,
        *([STIFFNESS_5020] * 5),
        STIFFNESS_4010,
        STIFFNESS_4010,
    ],
    dtype=np.float64,
)
KDS = np.asarray(
    [
        DAMPING_7520_22,
        DAMPING_7520_22,
        DAMPING_7520_14,
        DAMPING_7520_22,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_7520_22,
        DAMPING_7520_22,
        DAMPING_7520_14,
        DAMPING_7520_22,
        DAMPING_5020,
        DAMPING_5020,
        DAMPING_7520_14,
        DAMPING_5020,
        DAMPING_5020,
        *([DAMPING_5020] * 5),
        DAMPING_4010,
        DAMPING_4010,
        *([DAMPING_5020] * 5),
        DAMPING_4010,
        DAMPING_4010,
    ],
    dtype=np.float64,
)
EFFORT_LIMITS = np.asarray(
    [
        139.0,
        139.0,
        88.0,
        139.0,
        25.0,
        25.0,
        139.0,
        139.0,
        88.0,
        139.0,
        25.0,
        25.0,
        88.0,
        25.0,
        25.0,
        *([25.0] * 5),
        5.0,
        5.0,
        *([25.0] * 5),
        5.0,
        5.0,
    ],
    dtype=np.float64,
)
ACTION_SCALE = (
    0.25
    * EFFORT_LIMITS
    / np.asarray(
        [
            STIFFNESS_7520_22,
            STIFFNESS_7520_22,
            STIFFNESS_7520_14,
            STIFFNESS_7520_22,
            STIFFNESS_5020,
            STIFFNESS_5020,
            STIFFNESS_7520_22,
            STIFFNESS_7520_22,
            STIFFNESS_7520_14,
            STIFFNESS_7520_22,
            STIFFNESS_5020,
            STIFFNESS_5020,
            STIFFNESS_7520_14,
            STIFFNESS_5020,
            STIFFNESS_5020,
            *([STIFFNESS_5020] * 5),
            STIFFNESS_4010,
            STIFFNESS_4010,
            *([STIFFNESS_5020] * 5),
            STIFFNESS_4010,
            STIFFNESS_4010,
        ],
        dtype=np.float64,
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return q * np.asarray([1.0, -1.0, -1.0, -1.0])


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pure = np.concatenate(([0.0], vector))
    return _quat_multiply(_quat_multiply(q, pure), _quat_conjugate(q))[1:]


def _heading_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _rotation_6d(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    matrix = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return matrix[:, :2].reshape(-1)


def _slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    return (math.sin((1.0 - fraction) * theta) * q0 + math.sin(fraction * theta) * q1) / math.sin(theta)


def _resample_reference(qpos_30hz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(math.floor(qpos_30hz.shape[0] / PLANNER_FPS / CONTROL_DT))
    root_q = np.empty((count, 4), dtype=np.float64)
    joints = np.empty((count, 29), dtype=np.float64)
    root_position = np.empty((count, 3), dtype=np.float64)
    for frame in range(count):
        source = frame * CONTROL_DT * PLANNER_FPS
        f0 = int(math.floor(source))
        f1 = min(f0 + 1, qpos_30hz.shape[0] - 1)
        weight = source - f0
        root_position[frame] = (1.0 - weight) * qpos_30hz[f0, :3] + weight * qpos_30hz[f1, :3]
        root_q[frame] = _slerp(qpos_30hz[f0, 3:7], qpos_30hz[f1, 3:7], weight)
        joint_mj = (1.0 - weight) * qpos_30hz[f0, 7:] + weight * qpos_30hz[f1, 7:]
        joints[frame] = joint_mj[ISAAC_TO_MUJOCO_INDEX]
    velocity = np.empty_like(joints)
    velocity[:-1] = np.diff(joints, axis=0) / CONTROL_DT
    velocity[-1] = velocity[-2]
    return root_position, root_q, joints, velocity


def _validate_policy_abi(encoder: ort.InferenceSession, decoder: ort.InferenceSession) -> None:
    encoder_inputs = [(value.name, value.type, list(value.shape)) for value in encoder.get_inputs()]
    encoder_outputs = [(value.name, value.type, list(value.shape)) for value in encoder.get_outputs()]
    decoder_inputs = [(value.name, value.type, list(value.shape)) for value in decoder.get_inputs()]
    decoder_outputs = [(value.name, value.type, list(value.shape)) for value in decoder.get_outputs()]
    if encoder_inputs != [("obs_dict", "tensor(float)", [1, 1762])]:
        raise ValueError(f"encoder ABI drift: {encoder_inputs!r}")
    if encoder_outputs != [("encoded_tokens", "tensor(float)", [1, 64])]:
        raise ValueError(f"encoder ABI drift: {encoder_outputs!r}")
    if decoder_inputs != [("obs_dict", "tensor(float)", [1, 994])]:
        raise ValueError(f"decoder ABI drift: {decoder_inputs!r}")
    if decoder_outputs != [("action", "tensor(float)", [1, 29])]:
        raise ValueError(f"decoder ABI drift: {decoder_outputs!r}")


def _encoder_observation(
    ref_quaternion: np.ndarray,
    ref_joint: np.ndarray,
    ref_velocity: np.ndarray,
    current_frame: int,
    robot_quaternion: np.ndarray,
    apply_delta_heading: np.ndarray,
) -> np.ndarray:
    observation = np.zeros(1762, dtype=np.float32)
    # encoder_mode_4 is [0, 0, 0, 0] for G1 mode.
    indices = np.minimum(current_frame + 5 * np.arange(10), ref_joint.shape[0] - 1)
    observation[4:294] = ref_joint[indices].reshape(-1)
    observation[294:584] = ref_velocity[indices].reshape(-1)
    orientation = []
    for index in indices:
        aligned_reference = _quat_multiply(apply_delta_heading, ref_quaternion[index])
        relative = _quat_multiply(_quat_conjugate(robot_quaternion), aligned_reference)
        orientation.append(_rotation_6d(relative))
    observation[601:661] = np.asarray(orientation).reshape(-1)
    return observation


def _policy_observation(history: deque[dict[str, np.ndarray]]) -> np.ndarray:
    frames = list(history)
    return np.concatenate(
        [
            np.stack([frame["angular_velocity"] for frame in frames]).reshape(-1),
            np.stack([frame["joint_position"] for frame in frames]).reshape(-1),
            np.stack([frame["joint_velocity"] for frame in frames]).reshape(-1),
            np.stack([frame["last_action"] for frame in frames]).reshape(-1),
            np.stack([frame["gravity"] for frame in frames]).reshape(-1),
        ]
    ).astype(np.float32)


def _zero_history_frame() -> dict[str, np.ndarray]:
    return {
        "angular_velocity": np.zeros(3),
        "joint_position": np.zeros(29),
        "joint_velocity": np.zeros(29),
        "last_action": np.zeros(29),
        "gravity": np.zeros(3),
    }


def _current_history_frame(data: mujoco.MjData, last_action: np.ndarray) -> dict[str, np.ndarray]:
    quaternion = np.asarray(data.qpos[3:7], dtype=np.float64)
    return {
        "angular_velocity": np.asarray(data.qvel[3:6], dtype=np.float64).copy(),
        "joint_position": (np.asarray(data.qpos[7:]) - DEFAULT_ANGLES)[ISAAC_TO_MUJOCO_INDEX],
        "joint_velocity": np.asarray(data.qvel[6:])[ISAAC_TO_MUJOCO_INDEX].copy(),
        "last_action": last_action.copy(),
        "gravity": _quat_rotate(_quat_conjugate(quaternion), np.asarray([0.0, 0.0, -1.0])),
    }


def _render(model: mujoco.MjModel, qpos: np.ndarray, output: Path, fps: int) -> None:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 960)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 720)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.8
    camera.azimuth = 135.0
    camera.elevation = -18.0
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    try:
        sample_count = max(1, int(round(qpos.shape[0] / (qpos.shape[0] * CONTROL_DT * fps))))
        for pose in qpos[::sample_count]:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            camera.lookat[:] = pose[:3]
            camera.lookat[2] = max(0.3, float(pose[2]) * 0.55)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()


def simulate_motion(
    model: mujoco.MjModel,
    encoder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    qpos_30hz: np.ndarray,
    diagnostic_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if qpos_30hz.ndim != 2 or qpos_30hz.shape[1] != 36:
        raise ValueError(f"planner qpos shape invalid: {qpos_30hz.shape}")
    _, ref_quaternion, ref_joint, ref_velocity = _resample_reference(qpos_30hz)
    data = mujoco.MjData(model)
    data.qpos[:7] = qpos_30hz[0, :7]
    data.qpos[7:] = DEFAULT_ANGLES
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    history: deque[dict[str, np.ndarray]] = deque(maxlen=HISTORY_LENGTH)
    for _ in range(HISTORY_LENGTH - 1):
        history.append(_zero_history_frame())
    last_action = np.zeros(29, dtype=np.float64)
    initial_robot_heading = _heading_quaternion(np.asarray(data.qpos[3:7]))
    initial_reference_heading = _heading_quaternion(ref_quaternion[0])
    apply_delta_heading = _quat_multiply(
        initial_robot_heading,
        _quat_conjugate(initial_reference_heading),
    )

    poses = []
    raw_actions = []
    target_positions = []
    maximum_velocity = 0.0
    maximum_torque_ratio = 0.0
    minimum_height = math.inf
    nonfinite = False
    physics_steps_per_control = int(round(CONTROL_DT / model.opt.timestep))
    if physics_steps_per_control < 1 or not math.isclose(
        physics_steps_per_control * model.opt.timestep,
        CONTROL_DT,
        abs_tol=1.0e-12,
    ):
        raise ValueError("MuJoCo timestep does not divide 20 ms control period")

    for current_frame in range(ref_joint.shape[0]):
        history.append(_current_history_frame(data, last_action))
        encoder_obs = _encoder_observation(
            ref_quaternion,
            ref_joint,
            ref_velocity,
            current_frame,
            np.asarray(data.qpos[3:7], dtype=np.float64),
            apply_delta_heading,
        )
        token = np.asarray(
            encoder.run(["encoded_tokens"], {"obs_dict": encoder_obs[None]})[0],
            dtype=np.float32,
        )
        policy_obs = _policy_observation(history)
        decoder_obs = np.concatenate([token.reshape(-1), policy_obs]).astype(np.float32)
        action = np.asarray(
            decoder.run(["action"], {"obs_dict": decoder_obs[None]})[0],
            dtype=np.float64,
        ).reshape(-1)
        if action.shape != (29,) or not np.isfinite(action).all() or np.max(np.abs(action)) >= 10.0:
            nonfinite = True
            break
        target = DEFAULT_ANGLES.astype(np.float64) + action[MUJOCO_TO_ISAAC_INDEX] * ACTION_SCALE
        for _ in range(physics_steps_per_control):
            torque = KPS * (target - data.qpos[7:]) - KDS * data.qvel[6:]
            torque = np.clip(torque, -EFFORT_LIMITS, EFFORT_LIMITS)
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            maximum_torque_ratio = max(
                maximum_torque_ratio,
                float(np.max(np.abs(torque) / EFFORT_LIMITS)),
            )
        poses.append(np.asarray(data.qpos).copy())
        raw_actions.append(action.copy())
        target_positions.append(target.copy())
        last_action = action
        maximum_velocity = max(maximum_velocity, float(np.max(np.abs(data.qvel[6:]))))
        minimum_height = min(minimum_height, float(data.qpos[2]))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2] < 0.08:
            nonfinite = True
            break

    pose_array = np.asarray(poses, dtype=np.float64)
    action_array = np.asarray(raw_actions, dtype=np.float64)
    target_array = np.asarray(target_positions, dtype=np.float64)
    if diagnostic_arrays is not None:
        diagnostic_arrays["raw_actions"] = action_array.copy()
        diagnostic_arrays["target_positions_hardware29"] = target_array.copy()
    completed = int(pose_array.shape[0])
    reference_mj = ref_joint[:completed, MUJOCO_TO_ISAAC_INDEX]
    joint_rmse = float(np.sqrt(np.mean(np.square(pose_array[:, 7:] - reference_mj))))
    metrics = {
        "passed": completed == ref_joint.shape[0] and not nonfinite,
        "requested_control_steps": int(ref_joint.shape[0]),
        "completed_control_steps": completed,
        "physics_steps_per_control": physics_steps_per_control,
        "minimum_root_height_m": minimum_height,
        "maximum_absolute_joint_velocity_rad_s": maximum_velocity,
        "maximum_torque_limit_ratio": maximum_torque_ratio,
        "joint_reference_rmse_rad": joint_rmse,
        "maximum_absolute_raw_action": float(np.max(np.abs(action_array))),
        "maximum_absolute_target_position_rad": float(np.max(np.abs(target_array))),
        "horizontal_displacement_m": float(np.linalg.norm(pose_array[-1, :2] - pose_array[0, :2])),
    }
    return pose_array, metrics


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
    parser.add_argument("--video-fps", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve(strict=True)
    motion_dir = args.planner_motion_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {report_path}")

    scene_path = (root / "gear_sonic_deploy/g1/scene_29dof.xml").resolve(strict=True)
    encoder_path = (root / "gear_sonic_deploy/policy/release/model_encoder.onnx").resolve(strict=True)
    decoder_path = (root / "gear_sonic_deploy/policy/release/model_decoder.onnx").resolve(strict=True)
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in ort.get_available_providers()
    ]
    encoder = ort.InferenceSession(str(encoder_path), providers=providers)
    decoder = ort.InferenceSession(str(decoder_path), providers=providers)
    _validate_policy_abi(encoder, decoder)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = 0.002
    if model.nq != 36 or model.nv != 35 or model.nu != 29:
        raise ValueError(f"G1 model ABI drift: nq={model.nq}, nv={model.nv}, nu={model.nu}")

    records = []
    for name in args.motions:
        source_path = (motion_dir / f"{name}.npz").resolve(strict=True)
        with np.load(source_path, allow_pickle=False) as archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
        diagnostic_arrays: dict[str, np.ndarray] = {}
        poses, metrics = simulate_motion(
            model,
            encoder,
            decoder,
            qpos,
            diagnostic_arrays=diagnostic_arrays,
        )
        npz_path = output_dir / f"{name}.physical.npz"
        video_path = output_dir / f"{name}.physical.mp4"
        if npz_path.exists() or video_path.exists():
            raise FileExistsError(f"refusing to overwrite output for {name}")
        np.savez_compressed(
            npz_path,
            qpos=poses,
            control_dt=np.asarray([CONTROL_DT]),
            **diagnostic_arrays,
        )
        _render(model, poses, video_path, args.video_fps)
        records.append(
            {
                "name": name,
                "mode": PLANNER_MODES[name],
                "source_planner_npz": str(source_path),
                "source_planner_npz_sha256": _sha256(source_path),
                "physical_npz": npz_path.name,
                "physical_npz_sha256": _sha256(npz_path),
                "physical_video": video_path.name,
                "physical_video_sha256": _sha256(video_path),
                "metrics": metrics,
            }
        )

    payload = {
        "schema_version": 1,
        "kind": "g1_released_sonic_policy_mujoco_motion_suite",
        "passed": all(record["metrics"]["passed"] for record in records),
        "authorization": {
            "simulator_only": True,
            "dds_opened": False,
            "robot_commands_published": False,
            "hardware_authorized": False,
        },
        "scene_sha256": _sha256(scene_path),
        "encoder_sha256": _sha256(encoder_path),
        "decoder_sha256": _sha256(decoder_path),
        "onnxruntime_providers": encoder.get_providers(),
        "control_dt_s": CONTROL_DT,
        "records": records,
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(report_path)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
