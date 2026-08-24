"""Run released SONIC features on the physical 23-DoF G1 model.

Released encoder/decoder remain a compatibility teacher.  Physics, state,
targets, actuators, safe transform, and saved trajectory are true23 only.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

from gear_sonic.scripts.render_g1_sonic_library_motions import DEFAULT_ANGLES, PLANNER_MODES
from gear_sonic.scripts.simulate_g1_sonic_library_motions import (
    ACTION_SCALE,
    CONTROL_DT,
    HISTORY_LENGTH,
    ISAAC_TO_MUJOCO_INDEX,
    KDS,
    KPS,
    MUJOCO_TO_ISAAC_INDEX,
    _encoder_observation,
    _heading_quaternion,
    _policy_observation,
    _quat_conjugate,
    _quat_multiply,
    _quat_rotate,
    _resample_reference,
    _validate_policy_abi,
    _zero_history_frame,
    simulate_motion,
)
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_sonic_library_replay import ReferenceActionPolicy, sha256_file
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def _joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(1, model.njnt))


def _history_frame(
    data: mujoco.MjData,
    retained_source_indices: np.ndarray,
    last_action29: np.ndarray,
) -> dict[str, np.ndarray]:
    q29 = DEFAULT_ANGLES.astype(np.float64).copy()
    dq29 = np.zeros(29, dtype=np.float64)
    q29[retained_source_indices] = np.asarray(data.qpos[7:], dtype=np.float64)
    dq29[retained_source_indices] = np.asarray(data.qvel[6:], dtype=np.float64)
    quaternion = np.asarray(data.qpos[3:7], dtype=np.float64)
    return {
        "angular_velocity": np.asarray(data.qvel[3:6], dtype=np.float64).copy(),
        "joint_position": (q29 - DEFAULT_ANGLES)[ISAAC_TO_MUJOCO_INDEX],
        "joint_velocity": dq29[ISAAC_TO_MUJOCO_INDEX],
        "last_action": last_action29.copy(),
        "gravity": _quat_rotate(_quat_conjugate(quaternion), np.asarray([0.0, 0.0, -1.0])),
    }


def _render(model_path: Path, qpos: np.ndarray, output: Path, fps: int) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 960)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 720)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.8
    camera.azimuth = 135.0
    camera.elevation = -18.0
    sample_count = max(1, int(round(1.0 / (CONTROL_DT * fps))))
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    try:
        for pose in qpos[::sample_count]:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            camera.lookat[:] = pose[:3]
            camera.lookat[2] = max(0.25, float(pose[2]) * 0.55)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()


def simulate_true23(
    *,
    root: Path,
    qpos_30hz: np.ndarray,
    encoder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    teacher_targets_hardware29: np.ndarray | None = None,
    diagnostic_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    _, ref_quaternion, ref_joint, ref_velocity = _resample_reference(qpos_30hz)
    model_path = root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
    module, model, physics = prepare_true23_model(
        model_path,
        root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
    )
    source_model = mujoco.MjModel.from_xml_path(str(root / "gear_sonic_deploy/g1/scene_29dof.xml"))
    source_names = _joint_names(source_model)
    retained = np.asarray([source_names.index(name) for name in HARDWARE_23_JOINT_NAMES], dtype=np.int64)
    compatibility_kp = np.asarray(KPS, dtype=np.float64)[retained]
    compatibility_kd = np.asarray(KDS, dtype=np.float64)[retained]
    data = module.MjData(model)
    data.qpos[:7] = qpos_30hz[0, :7]
    reference_action = ReferenceActionPolicy()
    reference_action.set_target(DEFAULT_ANGLES[retained].astype(np.float32))
    _, initial_target = __import__(
        "gear_sonic.utils.g1_23dof_safe_target_transform",
        fromlist=["safe_target_transform_numpy"],
    ).safe_target_transform_numpy(reference_action.raw)
    data.qpos[7:] = initial_target
    data.qvel[:] = 0.0
    module.mj_forward(model, data)

    history: deque[dict[str, np.ndarray]] = deque(maxlen=HISTORY_LENGTH)
    for _ in range(HISTORY_LENGTH - 1):
        history.append(_zero_history_frame())
    last_action = np.zeros(29, dtype=np.float64)
    initial_robot_heading = _heading_quaternion(np.asarray(data.qpos[3:7]))
    initial_reference_heading = _heading_quaternion(ref_quaternion[0])
    apply_delta_heading = _quat_multiply(initial_robot_heading, _quat_conjugate(initial_reference_heading))
    poses: list[np.ndarray] = []
    min_height = math.inf
    max_tilt = 0.0
    max_raw = 0.0
    max_torque_ratio = 0.0
    projection_coordinate_count = 0
    max_projection_error = 0.0
    source_action_clip_coordinate_count = 0
    max_source_raw_action = 0.0
    failure: dict[str, object] | None = None
    pre_qpos: list[np.ndarray] = []
    pre_qvel: list[np.ndarray] = []
    raw23_rows: list[np.ndarray] = []
    target23_rows: list[np.ndarray] = []
    for frame in range(ref_joint.shape[0]):
        pre_qpos.append(np.asarray(data.qpos, dtype=np.float64).copy())
        pre_qvel.append(np.asarray(data.qvel, dtype=np.float64).copy())
        if teacher_targets_hardware29 is None:
            history.append(_history_frame(data, retained, last_action))
            encoder_obs = _encoder_observation(
                ref_quaternion,
                ref_joint,
                ref_velocity,
                frame,
                np.asarray(data.qpos[3:7], dtype=np.float64),
                apply_delta_heading,
            )
            token = np.asarray(
                encoder.run(["encoded_tokens"], {"obs_dict": encoder_obs[None]})[0],
                dtype=np.float32,
            )
            decoder_obs = np.concatenate([token.reshape(-1), _policy_observation(history)]).astype(np.float32)
            action29 = np.asarray(
                decoder.run(["action"], {"obs_dict": decoder_obs[None]})[0],
                dtype=np.float32,
            ).reshape(29)
            if not np.isfinite(action29).all():
                failure = {"frame": frame, "reason": "released_policy_action_nonfinite"}
                break
            max_source_raw_action = max(max_source_raw_action, float(np.max(np.abs(action29))))
            source_clip_mask = np.abs(action29) >= np.float32(10.0)
            source_action_clip_coordinate_count += int(np.count_nonzero(source_clip_mask))
            action29 = np.clip(action29, np.float32(-10.0), np.float32(10.0))
            target29 = DEFAULT_ANGLES.astype(np.float64) + action29[MUJOCO_TO_ISAAC_INDEX] * ACTION_SCALE
            last_action = action29.astype(np.float64)
        else:
            target29 = np.asarray(teacher_targets_hardware29[frame], dtype=np.float64)
        reference_action.set_target(target29[retained], project_to_raw_clip=True)
        projection_coordinate_count += reference_action.raw_clip_coordinate_count
        max_projection_error = max(
            max_projection_error,
            reference_action.target_projection_max_abs_rad,
        )
        _, target23 = __import__(
            "gear_sonic.utils.g1_23dof_safe_target_transform",
            fromlist=["safe_target_transform_numpy"],
        ).safe_target_transform_numpy(reference_action.raw)
        raw23_rows.append(reference_action.raw.copy())
        target23_rows.append(target23.copy())
        for _ in range(physics.decimation):
            torque = np.clip(
                compatibility_kp * (target23.astype(np.float64) - data.qpos[7:])
                - compatibility_kd * data.qvel[6:],
                -physics.effort,
                physics.effort,
            )
            data.ctrl[:] = torque
            module.mj_step(model, data)
            max_torque_ratio = max(max_torque_ratio, float(np.max(np.abs(torque) / physics.effort)))
        poses.append(np.asarray(data.qpos, dtype=np.float64).copy())
        min_height = min(min_height, float(data.qpos[2]))
        gravity = _quat_rotate(_quat_conjugate(data.qpos[3:7]), np.asarray([0.0, 0.0, -1.0]))
        tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
        max_tilt = max(max_tilt, tilt)
        max_raw = max(max_raw, float(np.max(np.abs(reference_action.raw))))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2] < 0.08:
            failure = {"frame": frame, "reason": "true23_physical_state_failed"}
            break
    result = np.asarray(poses, dtype=np.float64)
    if diagnostic_arrays is not None:
        diagnostic_arrays["pre_qpos"] = np.asarray(pre_qpos, dtype=np.float32)
        diagnostic_arrays["pre_qvel"] = np.asarray(pre_qvel, dtype=np.float32)
        diagnostic_arrays["applied_raw_native23"] = np.asarray(raw23_rows, dtype=np.float32)
        diagnostic_arrays["applied_target_hardware23"] = np.asarray(target23_rows, dtype=np.float32)
    completed = len(result)
    metrics: dict[str, object] = {
        "requested_control_steps": int(ref_joint.shape[0]),
        "completed_control_steps": completed,
        "minimum_root_height_m": None if completed == 0 else min_height,
        "maximum_root_tilt_rad": max_tilt,
        "maximum_absolute_true23_raw_action": max_raw,
        "maximum_torque_limit_ratio": max_torque_ratio,
        "true23_raw_projection_coordinate_count": projection_coordinate_count,
        "maximum_true23_target_projection_error_rad": max_projection_error,
        "maximum_absolute_released_teacher_raw_action_before_clip": max_source_raw_action,
        "released_teacher_raw_clip_coordinate_count": source_action_clip_coordinate_count,
        "horizontal_displacement_m": (
            0.0 if completed == 0 else float(np.linalg.norm(result[-1, :2] - result[0, :2]))
        ),
        "failure": failure,
        "passed": failure is None and completed == ref_joint.shape[0],
        "physical_dof": 23,
        "actuator_count": int(model.nu),
        "removed_source_outputs_discarded": 6,
        "teacher_mode": (
            "live_released_policy_on_true23_state"
            if teacher_targets_hardware29 is None
            else "exact_released_policy_recorded_targets"
        ),
        "gain_profile": "released_retained_joint_kp_kd_with_true23_effort_limits",
    }
    return result, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--planner-motion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motions", nargs="+", choices=tuple(PLANNER_MODES), default=("hand_crawling",))
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--teacher-mode", choices=("recorded", "live"), default="recorded")
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    motion_dir = args.planner_motion_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if os.path.lexists(report_path):
        raise FileExistsError(report_path)
    encoder_path = root / "gear_sonic_deploy/policy/release/model_encoder.onnx"
    decoder_path = root / "gear_sonic_deploy/policy/release/model_decoder.onnx"
    encoder = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    _validate_policy_abi(encoder, decoder)
    source_model = mujoco.MjModel.from_xml_path(str(root / "gear_sonic_deploy/g1/scene_29dof.xml"))
    source_model.opt.timestep = 0.002
    records = []
    for name in args.motions:
        source = (motion_dir / f"{name}.npz").resolve(strict=True)
        with np.load(source, allow_pickle=False) as archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
        teacher_targets = None
        if args.teacher_mode == "recorded":
            teacher_arrays: dict[str, np.ndarray] = {}
            _, teacher_metrics = simulate_motion(
                source_model,
                encoder,
                decoder,
                qpos,
                diagnostic_arrays=teacher_arrays,
            )
            if teacher_metrics["passed"] is not True:
                raise RuntimeError(f"released source teacher failed for {name}")
            teacher_targets = teacher_arrays["target_positions_hardware29"]
        true23_diagnostics: dict[str, np.ndarray] = {}
        poses, metrics = simulate_true23(
            root=root,
            qpos_30hz=qpos,
            encoder=encoder,
            decoder=decoder,
            teacher_targets_hardware29=teacher_targets,
            diagnostic_arrays=true23_diagnostics,
        )
        npz_path = output_dir / f"{name}.true23.physical.npz"
        video_path = output_dir / f"{name}.true23.physical.mp4"
        if os.path.lexists(npz_path) or os.path.lexists(video_path):
            raise FileExistsError(name)
        with npz_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                qpos=poses,
                control_dt=np.asarray([CONTROL_DT]),
                **true23_diagnostics,
            )
        _render(root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml", poses, video_path, args.video_fps)
        records.append(
            {
                "name": name,
                "mode": PLANNER_MODES[name],
                "source_sha256": sha256_file(source),
                "physical_npz": npz_path.name,
                "physical_npz_sha256": sha256_file(npz_path),
                "physical_video": video_path.name,
                "physical_video_sha256": sha256_file(video_path),
                "metrics": metrics,
            }
        )
    report = {
        "schema_version": 1,
        "kind": "g1_released_sonic_feature_teacher_true23_physical_adapter",
        "released_encoder_sha256": sha256_file(encoder_path),
        "released_decoder_sha256": sha256_file(decoder_path),
        "physical_model": "g1_23dof_rev_1_0",
        "physical_dof": 23,
        "passed": all(record["metrics"]["passed"] is True for record in records),
        "records": records,
        "authorization": {
            "simulator_only": True,
            "compatibility_teacher_not_native_true23": True,
            "dds_opened": False,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
