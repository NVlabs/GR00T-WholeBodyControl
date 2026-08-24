"""Diagnostic-only external-reference ONNX replay for true-23 MuJoCo.

This intentionally does not import or alter Step1B evidence/qualification code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_JOINT_IDS
from gear_sonic.utils.g1_23dof_native124_policy import (
    OBSERVATION_DIM,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import (
    CONTROL_HZ,
    REPOSITORY_ROOT,
    ReferenceKinematics,
    _body_pose,
    _initial_state,
    _measured_gate_evidence,
    _model_joint_names,
    _quaternion_conjugate,
    _quaternion_multiply,
    _semantic_state,
    deterministic_episode_specs,
    load_frozen_clip,
    prepare_source29_model,
    prepare_true23_model,
)

SCHEMA = "g1_true23_step1c_external_reference_onnx_diagnostic_v1"
MODES = frozenset(
    (
        "static_frame0",
        "static_frame0_zero_velocity",
        "full_terminal_hold",
        "full_terminal_hold_zero_velocity",
    )
)
STATIC_FRAME0_MODES = frozenset(("static_frame0", "static_frame0_zero_velocity"))
OBSERVATION_LAYOUT = (
    ("qref_unitree_mjlab_hardware_order", 23),
    ("qdref_unitree_mjlab_hardware_order", 23),
    ("torso_relative_rotation_6d", 6),
    ("base_angular_velocity", 3),
    ("joint_position_minus_home_unitree_mjlab_hardware_order", 23),
    ("joint_velocity_unitree_mjlab_hardware_order", 23),
    ("previous_raw_action_unitree_mjlab_hardware_order", 23),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(value: object, name: str, size: int = 23) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite shape ({size},)")
    return result


@dataclass(frozen=True)
class UnitreeEnvParams:
    path: Path
    sha256: str
    home_hardware: np.ndarray
    action_scale_hardware: np.ndarray


def load_unitree_env_params(path: Path, *, expected_sha256: str | None = None) -> UnitreeEnvParams:
    """Load exact saved Unitree ``params/env.yaml`` topology, fail closed."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyYAML is required to parse Unitree env.yaml") from error
    path = path.resolve()
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("env.yaml SHA256 mismatch")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("env.yaml must be a mapping")
    if raw.get("joint_ids_map") != list(HARDWARE_JOINT_IDS):
        raise ValueError("env.yaml joint_ids_map differs from pinned 23-joint contract")
    home = _vector(raw.get("default_joint_pos"), "env.yaml.default_joint_pos")
    actions = raw.get("actions")
    if not isinstance(actions, dict) or set(actions) != {"JointPositionAction"}:
        raise ValueError("env.yaml requires exactly actions.JointPositionAction")
    action = actions["JointPositionAction"]
    if not isinstance(action, dict):
        raise ValueError("env.yaml JointPositionAction must be a mapping")
    scale = _vector(action.get("scale"), "env.yaml action scale")
    offset = _vector(action.get("offset"), "env.yaml action offset")
    if not np.array_equal(home, offset):
        raise ValueError("env.yaml default_joint_pos and action offset differ")
    observations = raw.get("observations")
    required = {
        "motion_command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    }
    if not isinstance(observations, dict) or set(observations) != required:
        raise ValueError("env.yaml observation topology differs from saved 124-D actor")
    expected_sizes = {
        "motion_command": 46,
        "motion_anchor_ori_b": 6,
        "base_ang_vel": 3,
        "joint_pos_rel": 23,
        "joint_vel_rel": 23,
        "last_action": 23,
    }
    for name, size in expected_sizes.items():
        node = observations[name]
        if not isinstance(node, dict) or len(node.get("scale", ())) != size or node.get("history_length") != 1:
            raise ValueError(f"env.yaml observation {name} topology differs")
    return UnitreeEnvParams(path, actual, home, scale)


def _rotation_6d(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    matrix = np.asarray(
        (
            (1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w),
            (2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w),
            (2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y),
        )
    )
    return matrix[:, :2].reshape(-1).astype(np.float32)


def build_observation(
    *,
    qref_native: Sequence[float],
    qdref_native: Sequence[float],
    torso_relative_quaternion_wxyz: Sequence[float],
    base_angular_velocity: Sequence[float],
    q_hardware: Sequence[float],
    qd_hardware: Sequence[float],
    previous_raw_action_native: Sequence[float],
    env: UnitreeEnvParams,
) -> np.ndarray:
    """Build exact float32 actor observation in saved Unitree field order."""
    result = np.concatenate(
        (
            _vector(qref_native, "qref_native"),
            _vector(qdref_native, "qdref_native"),
            _rotation_6d(torso_relative_quaternion_wxyz),
            _vector(base_angular_velocity, "base_angular_velocity", 3),
            _vector(q_hardware, "q_hardware") - env.home_hardware,
            _vector(qd_hardware, "qd_hardware"),
            _vector(previous_raw_action_native, "previous_raw_action_native"),
        )
    ).astype(np.float32, copy=False)
    if result.shape != (OBSERVATION_DIM,) or not np.isfinite(result).all():
        raise RuntimeError("124-D observation layout drift")
    return result[None, :]


class ExternalReferenceOnnxPolicy:
    def __init__(self, path: Path, *, expected_sha256: str | None = None) -> None:
        self.path = path.resolve()
        self.sha256 = sha256_file(self.path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("ONNX SHA256 mismatch")
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("onnxruntime is required") from error
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or inputs[0].shape != [1, 124]
            or outputs[0].shape != [1, 23]
            or inputs[0].type != "tensor(float)"
            or outputs[0].type != "tensor(float)"
        ):
            raise ValueError("ONNX topology must be exactly one float [1,124] input and one float [1,23] output")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def run(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape != (1, 124) or observation.dtype != np.float32 or not np.isfinite(observation).all():
            raise ValueError("ONNX observation must be finite float32 [1,124]")
        result = np.asarray(
            self.session.run([self.output_name], {self.input_name: observation})[0], dtype=np.float32
        )
        if result.shape != (1, 23) or not np.isfinite(result).all():
            raise RuntimeError("ONNX action must be finite float32 [1,23]")
        return result[0]


def _reference_frame(clip: Any, index: int, mode: str) -> int:
    return 0 if mode in STATIC_FRAME0_MODES else min(index, clip.frame_count - 1)


def _reference(clip: Any, index: int, mode: str, env: UnitreeEnvParams) -> tuple[np.ndarray, np.ndarray]:
    frame = _reference_frame(clip, index, mode)
    qref = clip.target_joint_pos_hardware[frame].astype(np.float32)
    zero_velocity = mode == "static_frame0_zero_velocity" or (
        mode == "full_terminal_hold_zero_velocity" and index >= clip.frame_count
    )
    qdref = (
        np.zeros_like(clip.target_joint_vel_hardware[frame], dtype=np.float32)
        if zero_velocity
        else clip.target_joint_vel_hardware[frame].astype(np.float32)
    )
    return qref, qdref


def run_clip(
    *,
    clip: Any,
    reference: ReferenceKinematics,
    module: Any,
    model: Any,
    physics: Any,
    policy: ExternalReferenceOnnxPolicy,
    env: UnitreeEnvParams,
    mode: str,
    seed: int,
    horizon_steps: int,
    stop_on_first_termination: bool = False,
) -> dict[str, Any]:
    if mode not in MODES or horizon_steps < 1:
        raise ValueError("unsupported mode or horizon")
    spec = deterministic_episode_specs(clip.clip_id, (seed,))[0]
    data = module.MjData(model)
    qpos, qvel = _initial_state(clip, spec, "true23_expert")
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    module.mj_forward(model, data)
    joint_ids = np.arange(1, model.njnt, dtype=np.int64)
    lower, upper = model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1]
    previous = np.zeros(23, dtype=np.float32)
    first_term = None
    max_error = 0.0
    peak_torque = 0.0
    saturation_steps = 0
    hard_violations = 0
    contact_steps = {"left_foot": 0, "right_foot": 0}
    steps_executed = 0
    for step in range(horizon_steps):
        qref, qdref = _reference(clip, step, mode, env)
        torso_pos, torso_quat = _body_pose(module, model, data, "torso_link")
        del torso_pos
        frame = _reference_frame(clip, step, mode)
        _, reference_torso_quat = reference.arm_body("true23_expert", "torso_link", frame)
        relative_torso_quat = _quaternion_multiply(_quaternion_conjugate(torso_quat), reference_torso_quat)
        obs = build_observation(
            qref_native=qref,
            qdref_native=qdref,
            torso_relative_quaternion_wxyz=relative_torso_quat,
            base_angular_velocity=data.qvel[3:6],
            q_hardware=data.qpos[7:],
            qd_hardware=data.qvel[6:],
            previous_raw_action_native=previous,
            env=env,
        )
        raw = policy.run(obs)
        target = (env.home_hardware + env.action_scale_hardware * raw).astype(np.float64)
        step_peak = np.zeros(23)
        for _ in range(physics.decimation):
            torque = np.clip(
                physics.kp * (target - data.qpos[7:]) - physics.kd * data.qvel[6:], -physics.effort, physics.effort
            )
            step_peak = np.maximum(step_peak, np.abs(torque))
            data.ctrl[:] = torque
            module.mj_step(model, data)
        previous = raw
        q = data.qpos[7:]
        max_error = max(max_error, float(np.max(np.abs(q - target))))
        peak_torque = max(peak_torque, float(np.max(step_peak)))
        saturation_steps += int(np.any(step_peak >= physics.effort - 1e-10))
        hard_violations += int(np.any((q < lower) | (q > upper)))
        _, _, term = _measured_gate_evidence(module, model, data, reference, "true23_expert", frame)
        state = _semantic_state(module, model, data, "true23_expert")
        for foot, value in state["contacts"].items():
            contact_steps[foot] += int(value)
        bad = [name for name in ("anchor_pos", "anchor_ori", "ee_body_pos") if term[name]]
        if first_term is None and bad:
            first_term = {"step": step, "terms": bad}
        steps_executed = step + 1
        if first_term is not None and stop_on_first_termination:
            break
    return {
        "mode": mode,
        "clip_id": clip.clip_id,
        "seed": seed,
        "horizon_steps": horizon_steps,
        "steps_executed": steps_executed,
        "first_shipped_termination": first_term,
        "joint_hard_limit_violation_steps": hard_violations,
        "contact_steps": contact_steps,
        "torque_saturation_control_steps": saturation_steps,
        "peak_abs_torque_nm": peak_torque,
        "max_abs_joint_tracking_error_rad": max_error,
    }


def run_diagnostic(
    *,
    policy_path: Path,
    env_path: Path,
    source_csv_root: Path,
    step1a_root: Path,
    modes: Sequence[str],
    horizon_steps: int,
    selected_clip_ids: Sequence[str] | None = None,
    stop_on_first_termination: bool = False,
    policy_sha256: str | None = None,
    env_sha256: str | None = None,
) -> dict[str, Any]:
    from gear_sonic.utils.g1_true23_step1b_mujoco import frozen_clip_paths

    if not set(modes) <= MODES or not modes:
        raise ValueError("modes must be non-empty supported modes")
    env = load_unitree_env_params(env_path, expected_sha256=env_sha256)
    policy = ExternalReferenceOnnxPolicy(policy_path, expected_sha256=policy_sha256)
    contract = REPOSITORY_ROOT / "gear_sonic/config/sim_validation/g1_true23_idle_step1b_qualification_v1.json"
    paths = frozen_clip_paths(
        contract_path=contract,
        source_csv_root=source_csv_root,
        step1a_root=step1a_root,
        selected_clip_ids=selected_clip_ids,
    )
    module, source_model, _ = prepare_source29_model(REPOSITORY_ROOT / "gear_sonic/data/robots/g1/g1_29dof.xml")
    module, target_model, physics = prepare_true23_model(
        REPOSITORY_ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        REPOSITORY_ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
    )
    seeds = {
        "idle__220721__hands_on_back_loop_a036m": 1729,
        "idle__220713__change_idle_left_a021": 2729,
    }
    runs = []
    for path in paths:
        seed = seeds[path.clip_id]
        clip = load_frozen_clip(
            clip_id=path.clip_id,
            source_csv_path=path.source_csv,
            motion_path=path.motion,
            expert_path=path.expert,
            report_path=path.report,
            source_model=source_model,
            target_model=target_model,
            expected_frame_count=path.expected_frame_count,
        )
        reference = ReferenceKinematics(module, source_model, target_model, clip)
        runs.extend(
            run_clip(
                clip=clip,
                reference=reference,
                module=module,
                model=target_model,
                physics=physics,
                policy=policy,
                env=env,
                mode=mode,
                seed=seed,
                horizon_steps=horizon_steps,
                stop_on_first_termination=stop_on_first_termination,
            )
            for mode in modes
        )
    return {
        "schema_version": 1,
        "kind": SCHEMA,
        "diagnostic_only": True,
        "training_authorized": False,
        "deployment_ready": False,
        "robot_commands_performed": False,
        "control_hz": CONTROL_HZ,
        "policy": {
            "path": str(policy.path),
            "sha256": policy.sha256,
            "input_shape": [1, 124],
            "output_shape": [1, 23],
        },
        "env_params": {
            "path": str(env.path),
            "sha256": env.sha256,
            "home_hardware": env.home_hardware.tolist(),
            "action_scale_hardware": env.action_scale_hardware.tolist(),
        },
        "observation_layout": list(OBSERVATION_LAYOUT),
        "actor_joint_order": "unitree_mjlab_hardware_order",
        "stop_on_first_termination": stop_on_first_termination,
        "joint_names_hardware": list(_model_joint_names(target_model)),
        "runs": runs,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
