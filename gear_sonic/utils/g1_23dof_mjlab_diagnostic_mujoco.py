"""Strict offline MuJoCo diagnostics for MJLab true23 ONNX bundles.

This module is intentionally separate from the promotion validator.  A pass is
diagnostic evidence only: every report permanently denies deployment,
promotion, active motor control, and robot/network command use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_PROFILE,
    causal_history_profile_contract,
)
from gear_sonic.utils import g1_23dof_mujoco_sim2sim as sim2sim
from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from gear_sonic.utils.g1_23dof_contract import (
    DEPLOYMENT_DECODER_INPUT_DIM,
    HARDWARE_23_ACTION_SCALE,
    ROBOT_MODEL,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
    TOKEN_DIM,
)
from gear_sonic.utils.g1_23dof_live_shadow import (
    HARDWARE_DEFAULT_Q,
    build_proprio_frame,
    native_to_hardware,
)
from gear_sonic.utils.g1_23dof_mjlab_diagnostic_onnx import (
    DIAGNOSTIC_ALLOWED_USES,
    DIAGNOSTIC_BUNDLE_KIND,
    DIAGNOSTIC_FORBIDDEN_USES,
    verify_mjlab_diagnostic_onnx,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_GUARANTEED_GUARD_RAD,
    safe_target_transform_contract,
)

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "g1_true23_mjlab_diagnostic_mujoco_report"
PRODUCER_KIND = "g1_true23_mjlab_diagnostic_mujoco_runner"
PRODUCER_VERSION = 1
AGGREGATOR_KIND = "g1_true23_mjlab_diagnostic_mujoco_aggregator"
DEFAULT_DOMAIN_SOURCE_RELPATH = (
    "external_dependencies/unitree_rl_mjlab/src/tasks/tracking/"
    "tracking_env_cfg.py"
)
DEFAULT_MJLAB_ENV_RELPATH = "gear_sonic/envs/mjlab/sonic_true23.py"
DEFAULT_RUNTIME_RELPATH = (
    "gear_sonic/utils/g1_23dof_mjlab_diagnostic_mujoco.py"
)
DEFAULT_RUNNER_RELPATH = (
    "gear_sonic/scripts/run_g1_23dof_mjlab_diagnostic_mujoco.py"
)

_ROOT = Path(__file__).resolve().parents[2]
_SAFETY_FLAGS = {
    "diagnostic_only": True,
    "deployment_ready": False,
    "promotion_eligible": False,
    "active_motor_control_authorized": False,
}
_DOMAIN_CONTRACT = {
    "kind": "unitree_mjlab_true23_training_domain_randomization",
    "base_com_offset_m": {
        "body_name": "torso_link",
        "x": [-0.05, 0.05],
        "y": [-0.05, 0.05],
        "z": [-0.05, 0.05],
        "operation": "add",
    },
    "encoder_bias_rad": {
        "range": [-0.01, 0.01],
        "operation": "subtract_from_position_target",
        "joint_count": TARGET_DOF,
    },
    "foot_tangential_friction": {
        "range": [0.3, 1.2],
        "operation": "absolute",
        "shared_random": True,
        "axis": 0,
    },
}
_PROFILE_CONTRACT = {
    "smoke": {
        "deterministic_seeds": [1729],
        "episodes_per_seed": 1,
        "steps_per_episode": 100,
        "seconds_per_episode": 2.0,
    },
    "full": {
        "deterministic_seeds": [1729, 2718, 3141],
        "episodes_per_seed": 22,
        "steps_per_episode": 250,
        "seconds_per_episode": 5.0,
    },
}
_SCENARIOS = {
    "nominal": {"disturbance_scale": 0.0, "domain_randomization": False},
    "push_50": {"disturbance_scale": 0.5, "domain_randomization": False},
    "push_100": {"disturbance_scale": 1.0, "domain_randomization": False},
    "domain_push_100": {
        "disturbance_scale": 1.0,
        "domain_randomization": True,
    },
}


def _resolve_campaign_seeds(
    profile: str, seed_subset: Sequence[int] | None
) -> tuple[list[int], bool]:
    """Return an ordered, duplicate-free subset and exact completeness flag."""

    expected = list(_PROFILE_CONTRACT[profile]["deterministic_seeds"])
    if seed_subset is None:
        return expected, profile == "full"
    selected = [int(seed) for seed in seed_subset]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("diagnostic seed subset must be non-empty and unique")
    if any(seed not in expected for seed in selected):
        raise ValueError("diagnostic seed subset is outside the profile contract")
    selected.sort(key=expected.index)
    return selected, profile == "full" and selected == expected


def _validate_output_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError("diagnostic report may not be a symlink")
    resolved = expanded.resolve()
    if resolved.suffix.lower() != ".json" or "diagnostic" not in resolved.stem.lower():
        raise ValueError("output must be a .json file containing 'diagnostic'")
    lowered = resolved.name.lower()
    if "promotion" in lowered or "deployment" in lowered:
        raise ValueError("diagnostic output name may not claim promotion/deployment")
    trace_dir = resolved.with_name(f"{resolved.stem}.traces")
    if resolved.exists() or trace_dir.exists():
        raise FileExistsError("refusing to overwrite diagnostic report or traces")
    return resolved


def _atomic_write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite diagnostic report: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def diagnostic_bundle_paths(prefix: str | Path) -> tuple[Path, Path, Path]:
    """Resolve exact encoder, decoder, and metadata paths from one prefix."""

    value = Path(prefix).expanduser()
    if value.suffix:
        raise ValueError("diagnostic prefix must be extensionless")
    return (
        value.with_name(f"{value.name}.encoder.onnx").resolve(),
        value.with_name(f"{value.name}.decoder.onnx").resolve(),
        value.with_name(f"{value.name}.diagnostic.json").resolve(),
    )


def deterministic_domain_sample(
    *, seed: int, episode: int, enabled: bool
) -> dict[str, Any]:
    """Sample exact MJLab training-domain ranges without global RNG state."""

    if not enabled:
        return {
            "enabled": False,
            "base_com_offset_m": [0.0, 0.0, 0.0],
            "encoder_bias_hardware_rad": [0.0] * TARGET_DOF,
            "foot_tangential_friction": None,
        }

    def sample(label: str, index: int, lower: float, upper: float) -> float:
        digest = hashlib.sha256(
            (
                f"{ROBOT_MODEL}:mjlab-diagnostic-domain:{seed}:"
                f"{episode}:{label}:{index}"
            ).encode("ascii")
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        return round(lower + unit * (upper - lower), 12)

    return {
        "enabled": True,
        "base_com_offset_m": [
            sample("base_com", index, -0.05, 0.05) for index in range(3)
        ],
        "encoder_bias_hardware_rad": [
            sample("encoder_bias", index, -0.01, 0.01)
            for index in range(TARGET_DOF)
        ],
        "foot_tangential_friction": sample("foot_friction", 0, 0.3, 1.2),
    }


class DiagnosticOnnxRuntime:
    """Verified CPU-only inference for diagnostic MJLab ONNX pairs."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        encoder_path: Path,
        decoder_path: Path,
        metadata_path: Path,
        expected_reference_profile: str | None = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.encoder_path = encoder_path.expanduser().resolve()
        self.decoder_path = decoder_path.expanduser().resolve()
        self.metadata_path = metadata_path.expanduser().resolve()
        self.metadata = verify_mjlab_diagnostic_onnx(
            self.encoder_path,
            self.decoder_path,
            self.metadata_path,
            checkpoint_path=self.checkpoint_path,
            expected_reference_profile=expected_reference_profile,
        )
        self.output_transform = self.metadata["contract"].get(
            "safe_target_transform"
        )
        if self.output_transform is not None and (
            self.output_transform != safe_target_transform_contract()
            or self.metadata["contract"].get("decoder_output_semantics")
            != "applied_safe_native_action"
            or self.metadata["contract"].get(
                "external_safe_target_transform_allowed"
            )
            is not False
        ):
            raise ValueError("diagnostic safe-target output contract mismatch")
        if (
            self.metadata.get("kind") != DIAGNOSTIC_BUNDLE_KIND
            or any(
                self.metadata.get(key) is not expected
                for key, expected in _SAFETY_FLAGS.items()
            )
            or self.metadata.get("allowed_uses")
            != list(DIAGNOSTIC_ALLOWED_USES)
            or self.metadata.get("forbidden_uses")
            != list(DIAGNOSTIC_FORBIDDEN_USES)
        ):
            raise ValueError("verified bundle diagnostic safety flags differ")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for diagnostic inference") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.encoder_session = ort.InferenceSession(
            str(self.encoder_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.decoder_session = ort.InferenceSession(
            str(self.decoder_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._assert_session(
            self.encoder_session,
            TELEOP_ENCODER_INPUT_DIM,
            TOKEN_DIM,
            "encoder",
        )
        self._assert_session(
            self.decoder_session,
            DEPLOYMENT_DECODER_INPUT_DIM,
            TARGET_DOF,
            "decoder",
        )
        self.inference_count = 0
        self.nonfinite_output_count = 0

    @staticmethod
    def _assert_session(
        session: Any,
        input_dim: int,
        output_dim: int,
        context: str,
    ) -> None:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or list(inputs[0].shape) != [1, input_dim]
            or list(outputs[0].shape) != [1, output_dim]
            or inputs[0].type != "tensor(float)"
            or outputs[0].type != "tensor(float)"
        ):
            raise ValueError(f"diagnostic {context} ONNX contract mismatch")

    def infer(
        self,
        encoder_input: Sequence[float],
        proprio_history: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        encoder = np.asarray(encoder_input, dtype=np.float32).reshape(
            1, TELEOP_ENCODER_INPUT_DIM
        )
        history = np.asarray(proprio_history, dtype=np.float32).reshape(
            1, sim2sim.PROPRIO_HISTORY_DIM
        )
        token = self.encoder_session.run(
            None,
            {self.encoder_session.get_inputs()[0].name: encoder},
        )[0]
        decoder_input = np.concatenate((token, history), axis=1)
        action = self.decoder_session.run(
            None,
            {self.decoder_session.get_inputs()[0].name: decoder_input},
        )[0]
        if token.shape != (1, TOKEN_DIM) or action.shape != (1, TARGET_DOF):
            raise ValueError("diagnostic ONNX output shape changed")
        self.inference_count += 1
        if not (np.isfinite(token).all() and np.isfinite(action).all()):
            self.nonfinite_output_count += 1
        return token[0].astype(np.float64), action[0].astype(np.float64)

    def source_evidence(self) -> dict[str, Any]:
        hashes = self.metadata["hashes"]
        evidence = {
            "kind": DIAGNOSTIC_BUNDLE_KIND,
            "checkpoint_filename": self.checkpoint_path.name,
            "checkpoint_update_count": self.metadata["source"][
                "checkpoint_update_count"
            ],
            "reference_profile": self.metadata["source"]["reference_profile"],
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "lineage_sha256": hashes["lineage_sha256"],
            "policy_state_sha256": hashes["policy_state_sha256"],
            "encoder_onnx_sha256": sha256_file(self.encoder_path),
            "decoder_onnx_sha256": sha256_file(self.decoder_path),
            "diagnostic_metadata_sha256": sha256_file(self.metadata_path),
            "diagnostic_metadata_payload_sha256": self.metadata[
                "metadata_payload_sha256"
            ],
            "verification": {
                "strict_manifest_schema_and_flags": True,
                "checkpoint_hash_and_lineage": True,
                "policy_and_component_state_hashes": True,
                "onnx_file_and_embedded_metadata_hashes": True,
                "checkpoint_to_onnx_cpu_parity": True,
                "static_float32_shapes": True,
            },
            "inference_runtime": "onnxruntime_cpu",
            "inference_threads": 1,
        }
        if self.output_transform is not None:
            evidence.update(
                {
                    "decoder_output_semantics": (
                        "applied_safe_native_action"
                    ),
                    "safe_target_transform": self.output_transform,
                    "external_safe_target_transform_applied": False,
                }
            )
        return evidence


def _campaign_config(base: Mapping[str, Any], profile: str) -> Mapping[str, Any]:
    if profile not in _PROFILE_CONTRACT:
        raise ValueError(f"unsupported diagnostic profile: {profile}")
    config = copy.deepcopy(dict(base))
    config["coverage"] = dict(_PROFILE_CONTRACT[profile])
    config["scenarios"] = {
        "nominal": {"disturbance_scale": 0.0},
        "disturbance_50": {"disturbance_scale": 0.5},
        "disturbance_100": {"disturbance_scale": 1.0},
    }
    return config


def _model_domain_contract(module: Any, model: Any) -> dict[str, Any]:
    torso_id = module.mj_name2id(
        model,
        module.mjtObj.mjOBJ_BODY,
        "torso_link",
    )
    foot_body_ids = [
        module.mj_name2id(model, module.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    if torso_id < 0 or any(body_id < 0 for body_id in foot_body_ids):
        raise ValueError("diagnostic domain-randomization bodies missing")
    foot_geom_ids = [
        index
        for index in range(model.ngeom)
        if int(model.geom_bodyid[index]) in foot_body_ids
        and int(model.geom_contype[index]) != 0
        and int(model.geom_conaffinity[index]) != 0
    ]
    if not foot_geom_ids:
        raise ValueError("diagnostic model has no contact-enabled foot geoms")
    return {
        "torso_body_id": int(torso_id),
        "foot_body_ids": [int(value) for value in foot_body_ids],
        "foot_geom_ids": [int(value) for value in foot_geom_ids],
        "foot_geom_count": len(foot_geom_ids),
    }


def _apply_domain_sample(
    *,
    module: Any,
    model: Any,
    baseline: Mapping[str, np.ndarray],
    model_contract: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> None:
    model.body_ipos[:] = baseline["body_ipos"]
    model.geom_friction[:] = baseline["geom_friction"]
    if sample["enabled"]:
        torso_id = int(model_contract["torso_body_id"])
        model.body_ipos[torso_id] += np.asarray(
            sample["base_com_offset_m"], dtype=np.float64
        )
        model.geom_friction[
            np.asarray(model_contract["foot_geom_ids"], dtype=np.int32), 0
        ] = float(sample["foot_tangential_friction"])
    module.mj_setConst(model, module.MjData(model))


def _run_episode(
    *,
    module: Any,
    model: Any,
    runtime: DiagnosticOnnxRuntime,
    reference: Any,
    config: Mapping[str, Any],
    scenario: str,
    seed: int,
    episode: int,
    disturbance_scale: float,
    encoder_bias_hardware: Sequence[float],
) -> list[dict[str, Any]]:
    """Exact Sim2Sim loop plus MJLab encoder-bias action semantics."""

    data = module.MjData(model)
    data.qpos[:] = model.qpos0
    initial_state = config["initial_state"]
    data.qpos[:3] = np.asarray(initial_state["base_position_m"])
    data.qpos[3:7] = np.asarray(initial_state["base_quaternion_wxyz"])
    data.qpos[7:] = np.asarray(initial_state["joint_position_hardware_rad"])
    data.qvel[:] = 0.0
    data.qvel[6:] = np.asarray(initial_state["joint_velocity_hardware_radps"])
    module.mj_forward(model, data)
    physics = config["physics"]
    kp = np.asarray(physics["kp_hardware"], dtype=np.float64)
    kd = np.asarray(physics["kd_hardware"], dtype=np.float64)
    effort = np.asarray(physics["effort_limit_hardware_nm"], dtype=np.float64)
    velocity_limits = np.asarray(
        physics["velocity_limit_hardware_radps"], dtype=np.float64
    )
    action_scale = np.asarray(HARDWARE_23_ACTION_SCALE, dtype=np.float64)
    default_q = np.asarray(HARDWARE_DEFAULT_Q, dtype=np.float64)
    encoder_bias = np.asarray(encoder_bias_hardware, dtype=np.float64)
    if encoder_bias.shape != (TARGET_DOF,) or not np.isfinite(encoder_bias).all():
        raise ValueError("encoder bias must contain 23 finite values")
    hard_ranges = np.asarray(model.jnt_range[1:], dtype=np.float64)
    midpoint = np.mean(hard_ranges, axis=1)
    half_range = (
        (hard_ranges[:, 1] - hard_ranges[:, 0])
        * float(physics["soft_joint_pos_limit_factor"])
        * 0.5
    )
    soft_ranges = np.column_stack((midpoint - half_range, midpoint + half_range))
    previous_action = np.zeros(TARGET_DOF, dtype=np.float64)
    history_frames: list[list[float]] = []
    records: list[dict[str, Any]] = []
    latched_reason = ""
    apply_step = int(config["disturbance_schedule"]["apply_step"])
    disturbance = sim2sim.deterministic_disturbance(
        config=config,
        seed=seed,
        episode=episode,
        scale=disturbance_scale,
    )
    for step in range(int(config["coverage"]["steps_per_episode"])):
        step_disturbance = [0.0] * 6
        if step == apply_step and disturbance_scale > 0:
            sim2sim.apply_world_velocity_disturbance(
                data.qvel,
                data.qpos[3:7],
                disturbance,
            )
            step_disturbance = disturbance
            module.mj_forward(model, data)
        q_before = np.asarray(data.qpos[7:], dtype=np.float64).copy()
        dq_before = np.asarray(data.qvel[6:], dtype=np.float64).copy()
        quat_before = np.asarray(data.qpos[3:7], dtype=np.float64).copy()
        gyro = sim2sim._sensor_vector(  # noqa: SLF001
            module, model, data, "imu-pelvis-angular-velocity"
        )
        frame = build_proprio_frame(
            hardware_q=q_before.tolist(),
            hardware_dq=dq_before.tolist(),
            imu_gyroscope=gyro.tolist(),
            imu_quaternion_wxyz=quat_before.tolist(),
            previous_action_native=previous_action.tolist(),
        )
        if not history_frames:
            history_frames = [
                frame.copy() for _ in range(sim2sim.PROPRIO_HISTORY_LENGTH)
            ]
        else:
            history_frames = [*history_frames[1:], frame]
        history = sim2sim._term_major_history(history_frames)  # noqa: SLF001
        token, raw_action = runtime.infer(
            reference.encoder_input(quat_before), history
        )
        output_nonfinite = not (
            np.isfinite(token).all() and np.isfinite(raw_action).all()
        )
        finite_action = np.zeros(TARGET_DOF) if output_nonfinite else raw_action
        safe_action = np.clip(
            finite_action,
            -float(physics["action_clip_value"]),
            float(physics["action_clip_value"]),
        )
        saturation_count = int(np.count_nonzero(safe_action != finite_action))
        action_hardware = np.asarray(native_to_hardware(safe_action.tolist()))
        target = default_q + action_hardware * action_scale - encoder_bias
        if getattr(runtime, "output_transform", None) is not None:
            target_guard = np.minimum(
                target - soft_ranges[:, 0],
                soft_ranges[:, 1] - target,
            )
            if float(np.min(target_guard)) < (
                SAFE_TARGET_GUARANTEED_GUARD_RAD - 2.5e-7
            ):
                raise ValueError(
                    "embedded safe-target decoder output violates guaranteed "
                    "post-bias joint-position guard"
                )
        peak_torque = np.zeros(TARGET_DOF, dtype=np.float64)
        for _ in range(int(physics["control_decimation"])):
            q = np.asarray(data.qpos[7:], dtype=np.float64)
            dq = np.asarray(data.qvel[6:], dtype=np.float64)
            torque = np.clip(kp * (target - q) - kd * dq, -effort, effort)
            replace = np.abs(torque) >= np.abs(peak_torque)
            peak_torque[replace] = torque[replace]
            data.ctrl[:] = torque
            module.mj_step(model, data)
        q = np.asarray(data.qpos[7:], dtype=np.float64).copy()
        dq = np.asarray(data.qvel[6:], dtype=np.float64).copy()
        root_pos = np.asarray(data.qpos[:3], dtype=np.float64).copy()
        root_quat = np.asarray(data.qpos[3:7], dtype=np.float64).copy()
        root_lin_vel = np.asarray(data.qvel[:3], dtype=np.float64).copy()
        root_ang_vel = sim2sim._sensor_vector(  # noqa: SLF001
            module, model, data, "imu-pelvis-angular-velocity"
        )
        gravity = sim2sim._projected_gravity(root_quat)  # noqa: SLF001
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        tracking_rmse = float(np.sqrt(np.mean((target - q) ** 2)))
        recovery_metric = (
            tilt
            + abs(float(root_pos[2]) - float(reference.root_position[2]))
            + tracking_rmse
        )
        state_nonfinite = not all(
            np.isfinite(value).all()
            for value in (
                q,
                dq,
                root_pos,
                root_quat,
                root_lin_vel,
                root_ang_vel,
                peak_torque,
            )
        )
        nonfinite = output_nonfinite or state_nonfinite
        joint_limit_violation = bool(
            np.any(q < soft_ranges[:, 0])
            or np.any(q > soft_ranges[:, 1])
            or np.any(target < soft_ranges[:, 0])
            or np.any(target > soft_ranges[:, 1])
        )
        velocity_ratio = float(np.max(np.abs(dq) / velocity_limits))
        if not latched_reason:
            latched_reason = sim2sim._termination_reason(  # noqa: SLF001
                base_height=float(root_pos[2]),
                tilt=tilt,
                velocity_ratio=velocity_ratio,
                nonfinite=nonfinite,
                config=config,
            )
        record = {
            "schema_version": sim2sim.TRACE_SCHEMA_VERSION,
            "kind": sim2sim.TRACE_KIND,
            "scenario": scenario,
            "seed": seed,
            "episode": episode,
            "step": step,
            "time_s": sim2sim._canonical_float(  # noqa: SLF001
                (step + 1) / config["control_hz"]
            ),
            "disturbance_delta": sim2sim._canonical_vector(  # noqa: SLF001
                step_disturbance
            ),
            "base_position_m": sim2sim._canonical_vector(root_pos),  # noqa: SLF001
            "base_quaternion_wxyz": sim2sim._canonical_vector(  # noqa: SLF001
                root_quat
            ),
            "base_linear_velocity_mps": sim2sim._canonical_vector(  # noqa: SLF001
                root_lin_vel
            ),
            "base_angular_velocity_radps": sim2sim._canonical_vector(  # noqa: SLF001
                root_ang_vel
            ),
            "projected_gravity": sim2sim._canonical_vector(gravity),  # noqa: SLF001
            "joint_position_hardware_rad": sim2sim._canonical_vector(q),  # noqa: SLF001
            "joint_velocity_hardware_radps": sim2sim._canonical_vector(  # noqa: SLF001
                dq
            ),
            "action_native_raw": sim2sim._canonical_vector(raw_action),  # noqa: SLF001
            "action_native": sim2sim._canonical_vector(safe_action),  # noqa: SLF001
            "action_saturated_count": saturation_count,
            "target_position_hardware_rad": sim2sim._canonical_vector(  # noqa: SLF001
                target
            ),
            "applied_torque_hardware_nm": sim2sim._canonical_vector(  # noqa: SLF001
                peak_torque
            ),
            "base_height_m": sim2sim._canonical_float(root_pos[2]),  # noqa: SLF001
            "tilt_rad": sim2sim._canonical_float(tilt),  # noqa: SLF001
            "tracking_rmse_rad": sim2sim._canonical_float(  # noqa: SLF001
                tracking_rmse
            ),
            "recovery_metric": sim2sim._canonical_float(  # noqa: SLF001
                recovery_metric
            ),
            "nonfinite": nonfinite,
            "joint_limit_violation": joint_limit_violation,
            "terminated": bool(latched_reason),
            "termination_reason": latched_reason,
        }
        sim2sim._exact_keys(  # noqa: SLF001
            record, sim2sim._TRACE_RECORD_KEYS, "diagnostic trace record"  # noqa: SLF001
        )
        records.append(record)
        previous_action = safe_action
    return records


def _bound_metrics(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    velocity_limits = np.asarray(
        config["physics"]["velocity_limit_hardware_radps"], dtype=np.float64
    )
    effort_limits = np.asarray(
        config["physics"]["effort_limit_hardware_nm"], dtype=np.float64
    )
    velocity_ratios: list[float] = []
    effort_ratios: list[float] = []
    termination_reasons: dict[str, int] = {}
    velocity_violation_count = 0
    effort_violation_count = 0
    for record in records:
        velocity = np.abs(
            np.asarray(record["joint_velocity_hardware_radps"], dtype=np.float64)
        )
        effort = np.abs(
            np.asarray(record["applied_torque_hardware_nm"], dtype=np.float64)
        )
        velocity_ratio = float(np.max(velocity / velocity_limits))
        effort_ratio = float(np.max(effort / effort_limits))
        velocity_ratios.append(velocity_ratio)
        effort_ratios.append(effort_ratio)
        velocity_violation_count += int(velocity_ratio > 1.0 + 1e-9)
        effort_violation_count += int(effort_ratio > 1.0 + 1e-9)
    for episode in sorted({int(record["episode"]) for record in records}):
        reason = next(
            (
                str(record["termination_reason"])
                for record in records
                if int(record["episode"]) == episode
                and record["termination_reason"]
            ),
            "",
        )
        if reason:
            termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
    return {
        "max_joint_velocity_ratio": sim2sim._canonical_float(  # noqa: SLF001
            max(velocity_ratios)
        ),
        "joint_velocity_bound_violation_count": velocity_violation_count,
        "max_effort_ratio": sim2sim._canonical_float(max(effort_ratios)),  # noqa: SLF001
        "effort_bound_violation_count": effort_violation_count,
        "fall_count": sum(
            termination_reasons.get(reason, 0)
            for reason in ("base_height", "base_tilt")
        ),
        "termination_reasons": termination_reasons,
    }


def _source_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def run_mjlab_diagnostic_mujoco(
    *,
    checkpoint_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
    output_path: Path,
    profile: str,
    config_path: Path | None = None,
    mjcf_path: Path | None = None,
    expected_reference_profile: str | None = None,
    seed_subset: Sequence[int] | None = None,
) -> Mapping[str, Any]:
    """Run smoke/full CPU diagnostics and always emit a non-authorizing report."""

    output = _validate_output_path(output_path)
    if profile not in _PROFILE_CONTRACT:
        raise ValueError(f"unsupported diagnostic profile: {profile}")
    selected_seeds, full_campaign_complete = _resolve_campaign_seeds(
        profile, seed_subset
    )
    config = (
        config_path.expanduser().resolve()
        if config_path is not None
        else (_ROOT / sim2sim.DEFAULT_CONFIG_RELPATH).resolve()
    )
    mjcf = (
        mjcf_path.expanduser().resolve()
        if mjcf_path is not None
        else (_ROOT / sim2sim.DEFAULT_MJCF_RELPATH).resolve()
    )
    causal_contract = (
        causal_history_profile_contract()
        if expected_reference_profile == CAUSAL_HISTORY_PROFILE
        else None
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        **_SAFETY_FLAGS,
        "robot_or_network_commands_performed": False,
        "allowed_uses": ["offline_mujoco_diagnostic_review"],
        "forbidden_uses": list(DIAGNOSTIC_FORBIDDEN_USES),
        "profile": profile,
        "campaign_shard": {
            "is_shard": not full_campaign_complete,
            "selected_seeds": selected_seeds,
            "expected_full_seeds": list(
                _PROFILE_CONTRACT["full"]["deterministic_seeds"]
            ),
            "full_campaign_complete": full_campaign_complete,
        },
        "computed_pass": False,
        "promotion_assessment": {
            "required_reference_profile": expected_reference_profile,
            "required_semantic_contract_sha256": (
                causal_contract["contract_sha256"]
                if causal_contract is not None
                else None
            ),
            "required_scenarios": list(_SCENARIOS),
            "full_campaign_required": True,
            "computed_pass": False,
            "authorization_created": False,
        },
        "error": None,
        "source_artifact": None,
        "producer": {
            "kind": PRODUCER_KIND,
            "version": PRODUCER_VERSION,
            "runtime_sha256": _source_hash(_ROOT / DEFAULT_RUNTIME_RELPATH),
            "runner_sha256": _source_hash(_ROOT / DEFAULT_RUNNER_RELPATH),
        },
        "configuration": {
            "control_hz": 50,
            "base_sim2sim_config_path": str(config),
            "base_sim2sim_config_sha256": _source_hash(config),
            "mjcf_path": str(mjcf),
            "mjcf_sha256": _source_hash(mjcf),
            "profile_contract": dict(_PROFILE_CONTRACT[profile]),
            "scenarios": copy.deepcopy(_SCENARIOS),
            "required_reference_profile": expected_reference_profile,
            "causal_semantic_contract": causal_contract,
            "domain_randomization": copy.deepcopy(_DOMAIN_CONTRACT),
            "domain_source_sha256": _source_hash(
                _ROOT / DEFAULT_DOMAIN_SOURCE_RELPATH
            ),
            "sonic_true23_env_sha256": _source_hash(
                _ROOT / DEFAULT_MJLAB_ENV_RELPATH
            ),
        },
        "simulator": None,
        "runs": [],
        "summary": None,
    }
    try:
        base_config = sim2sim.load_sim2sim_config(config)
        campaign_config = _campaign_config(base_config, profile)
        campaign_config["coverage"]["deterministic_seeds"] = selected_seeds
        runtime = DiagnosticOnnxRuntime(
            checkpoint_path=checkpoint_path,
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            metadata_path=metadata_path,
            expected_reference_profile=expected_reference_profile,
        )
        report["source_artifact"] = runtime.source_evidence()
        module, model, physics_contract = sim2sim.prepare_mujoco_model(
            mjcf_path=mjcf,
            config=campaign_config,
        )
        reference = sim2sim.NeutralReference(
            model, module, campaign_config["initial_state"]
        )
        model_domain = _model_domain_contract(module, model)
        baseline = {
            "body_ipos": np.asarray(model.body_ipos).copy(),
            "geom_friction": np.asarray(model.geom_friction).copy(),
        }
        report["simulator"] = {
            "name": "MuJoCo",
            "version": module.__version__,
            "host": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "compiled_model": sim2sim._compiled_model_contract(  # noqa: SLF001
                module, model
            ),
            "physics_contract": physics_contract,
            "domain_model_contract": model_domain,
            "policy_control_hz": campaign_config["control_hz"],
            "physics_timestep_s": campaign_config["physics"]["timestep_s"],
            "control_decimation": campaign_config["physics"][
                "control_decimation"
            ],
        }
        runs: list[dict[str, Any]] = []
        for scenario, scenario_cfg in _SCENARIOS.items():
            for seed in campaign_config["coverage"]["deterministic_seeds"]:
                records: list[dict[str, Any]] = []
                domain_samples: list[dict[str, Any]] = []
                nonfinite_before = runtime.nonfinite_output_count
                for episode in range(
                    campaign_config["coverage"]["episodes_per_seed"]
                ):
                    sample = deterministic_domain_sample(
                        seed=seed,
                        episode=episode,
                        enabled=bool(scenario_cfg["domain_randomization"]),
                    )
                    _apply_domain_sample(
                        module=module,
                        model=model,
                        baseline=baseline,
                        model_contract=model_domain,
                        sample=sample,
                    )
                    records.extend(
                        _run_episode(
                            module=module,
                            model=model,
                            runtime=runtime,
                            reference=reference,
                            config=campaign_config,
                            scenario=scenario,
                            seed=seed,
                            episode=episode,
                            disturbance_scale=float(
                                scenario_cfg["disturbance_scale"]
                            ),
                            encoder_bias_hardware=sample[
                                "encoder_bias_hardware_rad"
                            ],
                        )
                    )
                    domain_samples.append(sample)
                trace = sim2sim._write_trace(  # noqa: SLF001
                    output_path=output,
                    scenario=scenario,
                    seed=seed,
                    records=records,
                )
                metrics = sim2sim.recompute_metrics(
                    records,
                    config=campaign_config,
                    disturbance_scale=float(scenario_cfg["disturbance_scale"]),
                )
                bounds = _bound_metrics(records, campaign_config)
                output_nonfinite_count = (
                    runtime.nonfinite_output_count - nonfinite_before
                )
                run_pass = bool(
                    sim2sim.metrics_pass(metrics, campaign_config)
                    and output_nonfinite_count == 0
                    and bounds["joint_velocity_bound_violation_count"] == 0
                    and bounds["effort_bound_violation_count"] == 0
                    and bounds["fall_count"] == 0
                )
                runs.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "episodes": campaign_config["coverage"][
                            "episodes_per_seed"
                        ],
                        "steps_per_episode": campaign_config["coverage"][
                            "steps_per_episode"
                        ],
                        "disturbance_scale": scenario_cfg["disturbance_scale"],
                        "domain_randomization": scenario_cfg[
                            "domain_randomization"
                        ],
                        "domain_samples_sha256": sha256_bytes(
                            canonical_json_bytes(domain_samples)
                        ),
                        "computed_pass": run_pass,
                        "policy_output_nonfinite_count": output_nonfinite_count,
                        "metrics": metrics,
                        "bounds": bounds,
                        "trace": trace,
                    }
                )
        report["runs"] = runs
        computed_pass = all(run["computed_pass"] for run in runs)
        report["computed_pass"] = computed_pass
        report["promotion_assessment"]["computed_pass"] = bool(
            computed_pass and full_campaign_complete
        )
        report["summary"] = {
            "computed_pass": computed_pass,
            "full_campaign_complete": full_campaign_complete,
            "run_count": len(runs),
            "episode_count": sum(run["episodes"] for run in runs),
            "record_count": sum(
                run["metrics"]["record_count"] for run in runs
            ),
            "termination_count": sum(
                run["metrics"]["termination_count"] for run in runs
            ),
            "fall_count": sum(run["bounds"]["fall_count"] for run in runs),
            "policy_output_nonfinite_count": sum(
                run["policy_output_nonfinite_count"] for run in runs
            ),
            "joint_limit_violation_count": sum(
                run["metrics"]["joint_limit_violation_count"] for run in runs
            ),
            "joint_velocity_bound_violation_count": sum(
                run["bounds"]["joint_velocity_bound_violation_count"]
                for run in runs
            ),
            "effort_bound_violation_count": sum(
                run["bounds"]["effort_bound_violation_count"] for run in runs
            ),
            "max_joint_velocity_ratio": max(
                run["bounds"]["max_joint_velocity_ratio"] for run in runs
            ),
            "max_effort_ratio": max(
                run["bounds"]["max_effort_ratio"] for run in runs
            ),
            "max_abs_native_action_raw": max(
                run["metrics"]["max_abs_native_action_raw"] for run in runs
            ),
            "max_action_saturation_fraction": max(
                run["metrics"]["action_saturation_fraction"] for run in runs
            ),
            "minimum_base_height_m": min(
                run["metrics"]["min_base_height_m"] for run in runs
            ),
            "maximum_tilt_rad": max(
                run["metrics"]["max_tilt_rad"] for run in runs
            ),
            "minimum_recovery_fraction": min(
                run["metrics"]["recovery_fraction"] for run in runs
            ),
            **_SAFETY_FLAGS,
        }
    except Exception as exc:  # Fail closed and preserve diagnostic evidence.
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["computed_pass"] = False
        report["summary"] = {
            "computed_pass": False,
            "full_campaign_complete": False,
            **_SAFETY_FLAGS,
        }
    _atomic_write_new(output, canonical_json_bytes(report))
    return report


def aggregate_mjlab_diagnostic_shards(
    *, shard_paths: Sequence[Path], output_path: Path
) -> Mapping[str, Any]:
    """Fail closed unless exact full-profile seed shards and traces agree."""

    output = _validate_output_path(output_path)
    expected_seeds = list(_PROFILE_CONTRACT["full"]["deterministic_seeds"])
    if len(shard_paths) != len(expected_seeds):
        raise ValueError("exactly three full-profile seed shards are required")
    by_seed: dict[int, tuple[Path, dict[str, Any]]] = {}
    common: dict[str, Any] | None = None
    shard_manifest: list[dict[str, Any]] = []
    expected_records = (
        _PROFILE_CONTRACT["full"]["episodes_per_seed"]
        * _PROFILE_CONTRACT["full"]["steps_per_episode"]
    )
    for raw_path in shard_paths:
        path = raw_path.expanduser()
        if path.is_symlink() or not path.is_file():
            raise ValueError("each diagnostic shard must be a regular file")
        path = path.resolve()
        try:
            shard = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("diagnostic shard is not valid JSON") from exc
        manifest = shard.get("campaign_shard")
        selected = manifest.get("selected_seeds") if isinstance(manifest, dict) else None
        if (
            shard.get("schema_version") != REPORT_SCHEMA_VERSION
            or shard.get("kind") != REPORT_KIND
            or shard.get("profile") != "full"
            or shard.get("computed_pass") is not True
            or shard.get("error") is not None
            or any(shard.get(key) is not value for key, value in _SAFETY_FLAGS.items())
            or not isinstance(manifest, dict)
            or manifest.get("is_shard") is not True
            or manifest.get("full_campaign_complete") is not False
            or manifest.get("expected_full_seeds") != expected_seeds
            or not isinstance(selected, list)
            or len(selected) != 1
            or shard.get("summary", {}).get("full_campaign_complete") is not False
            or shard.get("promotion_assessment", {}).get("computed_pass") is not False
        ):
            raise ValueError("report is not a passing incomplete full-profile shard")
        seed = selected[0]
        if seed not in expected_seeds or seed in by_seed:
            raise ValueError("diagnostic shard seeds must be exact and unique")
        comparable = {
            "source_artifact": shard.get("source_artifact"),
            "configuration": shard.get("configuration"),
            "simulator": shard.get("simulator"),
            "producer": shard.get("producer"),
        }
        if common is None:
            common = comparable
        elif comparable != common:
            raise ValueError("diagnostic shard artifact/configuration hashes differ")
        runs = shard.get("runs")
        if not isinstance(runs, list) or len(runs) != len(_SCENARIOS):
            raise ValueError("each diagnostic shard must contain four runs")
        run_keys: set[tuple[str, int]] = set()
        trace_manifest: list[dict[str, Any]] = []
        trace_dir_candidate = path.with_name(f"{path.stem}.traces")
        if trace_dir_candidate.is_symlink():
            raise ValueError("diagnostic shard trace directory may not be a symlink")
        expected_trace_dir = trace_dir_candidate.resolve()
        for run in runs:
            key = (run.get("scenario"), run.get("seed"))
            if (
                key[0] not in _SCENARIOS
                or key[1] != seed
                or key in run_keys
                or run.get("episodes")
                != _PROFILE_CONTRACT["full"]["episodes_per_seed"]
                or run.get("steps_per_episode")
                != _PROFILE_CONTRACT["full"]["steps_per_episode"]
                or run.get("computed_pass") is not True
                or run.get("metrics", {}).get("record_count") != expected_records
            ):
                raise ValueError("diagnostic shard run coverage is not exact")
            run_keys.add(key)
            trace = run.get("trace")
            if not isinstance(trace, dict) or trace.get("record_count") != expected_records:
                raise ValueError("diagnostic shard trace record count is not exact")
            trace_candidate = path.parent / str(trace.get("file", ""))
            if trace_candidate.is_symlink():
                raise ValueError("diagnostic shard trace may not be a symlink")
            trace_path = trace_candidate.resolve()
            if trace_path.parent != expected_trace_dir:
                raise ValueError("diagnostic shard trace path escaped its trace directory")
            if not trace_path.is_file() or sha256_file(trace_path) != trace.get("sha256"):
                raise ValueError("diagnostic shard trace hash mismatch")
            with trace_path.open("rb") as stream:
                line_count = sum(1 for _ in stream)
            if line_count != expected_records:
                raise ValueError("diagnostic shard trace line count is not exact")
            trace_manifest.append(
                {
                    "scenario": key[0],
                    "file": str(trace_path),
                    "sha256": trace["sha256"],
                    "payload_sha256": trace.get("payload_sha256"),
                    "record_count": expected_records,
                }
            )
        if run_keys != {(scenario, seed) for scenario in _SCENARIOS}:
            raise ValueError("diagnostic shard scenarios are not exact")
        by_seed[seed] = (path, shard)
        shard_manifest.append(
            {
                "seed": seed,
                "report_file": str(path),
                "report_sha256": sha256_file(path),
                "traces": trace_manifest,
            }
        )
    if set(by_seed) != set(expected_seeds) or common is None:
        raise ValueError("diagnostic shard set does not cover exact full seeds")
    runs = [
        copy.deepcopy(run)
        for scenario in _SCENARIOS
        for seed in expected_seeds
        for run in by_seed[seed][1]["runs"]
        if run["scenario"] == scenario
    ]
    first = by_seed[expected_seeds[0]][1]
    summary = {
        "computed_pass": True,
        "full_campaign_complete": True,
        "run_count": len(runs),
        "episode_count": sum(run["episodes"] for run in runs),
        "record_count": sum(run["metrics"]["record_count"] for run in runs),
        "termination_count": sum(run["metrics"]["termination_count"] for run in runs),
        "fall_count": sum(run["bounds"]["fall_count"] for run in runs),
        "policy_output_nonfinite_count": sum(
            run["policy_output_nonfinite_count"] for run in runs
        ),
        "joint_limit_violation_count": sum(
            run["metrics"]["joint_limit_violation_count"] for run in runs
        ),
        "joint_velocity_bound_violation_count": sum(
            run["bounds"]["joint_velocity_bound_violation_count"] for run in runs
        ),
        "effort_bound_violation_count": sum(
            run["bounds"]["effort_bound_violation_count"] for run in runs
        ),
        "max_joint_velocity_ratio": max(
            run["bounds"]["max_joint_velocity_ratio"] for run in runs
        ),
        "max_effort_ratio": max(run["bounds"]["max_effort_ratio"] for run in runs),
        "max_abs_native_action_raw": max(
            run["metrics"]["max_abs_native_action_raw"] for run in runs
        ),
        "max_action_saturation_fraction": max(
            run["metrics"]["action_saturation_fraction"] for run in runs
        ),
        "minimum_base_height_m": min(
            run["metrics"]["min_base_height_m"] for run in runs
        ),
        "maximum_tilt_rad": max(run["metrics"]["max_tilt_rad"] for run in runs),
        "minimum_recovery_fraction": min(
            run["metrics"]["recovery_fraction"] for run in runs
        ),
        **_SAFETY_FLAGS,
    }
    assessment = copy.deepcopy(first["promotion_assessment"])
    assessment["computed_pass"] = True
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        **_SAFETY_FLAGS,
        "robot_or_network_commands_performed": False,
        "allowed_uses": list(first["allowed_uses"]),
        "forbidden_uses": list(first["forbidden_uses"]),
        "profile": "full",
        "campaign_shard": {
            "is_shard": False,
            "selected_seeds": expected_seeds,
            "expected_full_seeds": expected_seeds,
            "full_campaign_complete": True,
            "aggregated_from_seed_shards": True,
        },
        "computed_pass": True,
        "promotion_assessment": assessment,
        "error": None,
        "source_artifact": common["source_artifact"],
        "producer": {
            "kind": AGGREGATOR_KIND,
            "version": PRODUCER_VERSION,
            "runtime_sha256": _source_hash(_ROOT / DEFAULT_RUNTIME_RELPATH),
            "runner_sha256": _source_hash(_ROOT / DEFAULT_RUNNER_RELPATH),
        },
        "configuration": common["configuration"],
        "simulator": common["simulator"],
        "aggregation": {
            "exact_seed_order": expected_seeds,
            "exact_scenario_order": list(_SCENARIOS),
            "shards": sorted(shard_manifest, key=lambda item: expected_seeds.index(item["seed"])),
        },
        "runs": runs,
        "summary": summary,
    }
    _atomic_write_new(output, canonical_json_bytes(report))
    return report


def verify_full_mjlab_diagnostic_report(report_path: Path) -> Mapping[str, Any]:
    """Verify frozen full-report coverage and every trace without recomputing metrics."""

    path = report_path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("full diagnostic report must be a regular file")
    path = path.resolve()
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("full diagnostic report is not valid JSON") from exc
    expected_seeds = list(_PROFILE_CONTRACT["full"]["deterministic_seeds"])
    expected_records = (
        _PROFILE_CONTRACT["full"]["episodes_per_seed"]
        * _PROFILE_CONTRACT["full"]["steps_per_episode"]
    )
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("kind") != REPORT_KIND
        or report.get("profile") != "full"
        or report.get("computed_pass") is not True
        or report.get("error") is not None
        or report.get("robot_or_network_commands_performed") is not False
        or any(report.get(key) is not value for key, value in _SAFETY_FLAGS.items())
        or report.get("summary", {}).get("full_campaign_complete") is not True
        or report.get("promotion_assessment", {}).get("computed_pass") is not True
        or report.get("configuration", {}).get("profile_contract")
        != _PROFILE_CONTRACT["full"]
        or report.get("configuration", {}).get("scenarios") != _SCENARIOS
        or not isinstance(report.get("source_artifact"), dict)
    ):
        raise ValueError("report is not a passing complete full-profile diagnostic")
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != len(_SCENARIOS) * len(expected_seeds):
        raise ValueError("full diagnostic report must contain exactly twelve runs")
    expected_keys = {
        (scenario, seed) for scenario in _SCENARIOS for seed in expected_seeds
    }
    actual_keys: set[tuple[str, int]] = set()
    trace_manifest: list[dict[str, Any]] = []
    trace_dir_candidate = path.with_name(f"{path.stem}.traces")
    if trace_dir_candidate.is_symlink():
        raise ValueError("full diagnostic trace directory may not be a symlink")
    trace_dir = trace_dir_candidate.resolve()
    for run in runs:
        key = (run.get("scenario"), run.get("seed"))
        metrics = run.get("metrics")
        bounds = run.get("bounds")
        trace = run.get("trace")
        if (
            key not in expected_keys
            or key in actual_keys
            or run.get("episodes") != _PROFILE_CONTRACT["full"]["episodes_per_seed"]
            or run.get("steps_per_episode")
            != _PROFILE_CONTRACT["full"]["steps_per_episode"]
            or run.get("computed_pass") is not True
            or run.get("policy_output_nonfinite_count") != 0
            or not isinstance(metrics, dict)
            or metrics.get("record_count") != expected_records
            or metrics.get("termination_count") != 0
            or metrics.get("nonfinite_count") != 0
            or metrics.get("joint_limit_violation_count") != 0
            or not isinstance(bounds, dict)
            or bounds.get("joint_velocity_bound_violation_count") != 0
            or bounds.get("effort_bound_violation_count") != 0
            or bounds.get("fall_count") != 0
            or not isinstance(trace, dict)
            or trace.get("record_count") != expected_records
        ):
            raise ValueError("full diagnostic run coverage or strict gates failed")
        actual_keys.add(key)
        trace_candidate = path.parent / str(trace.get("file", ""))
        if trace_candidate.is_symlink():
            raise ValueError("full diagnostic trace may not be a symlink")
        trace_path = trace_candidate.resolve()
        if trace_path.parent != trace_dir:
            raise ValueError("full diagnostic trace path escaped its trace directory")
        if not trace_path.is_file() or sha256_file(trace_path) != trace.get("sha256"):
            raise ValueError("full diagnostic trace hash mismatch")
        with trace_path.open("rb") as stream:
            line_count = sum(1 for _ in stream)
        if line_count != expected_records:
            raise ValueError("full diagnostic trace line count is not exact")
        trace_manifest.append(
            {
                "scenario": key[0],
                "seed": key[1],
                "file": str(trace_path),
                "sha256": trace["sha256"],
                "payload_sha256": trace.get("payload_sha256"),
                "record_count": expected_records,
            }
        )
    if actual_keys != expected_keys:
        raise ValueError("full diagnostic scenario/seed matrix is incomplete")
    summary = report["summary"]
    expected_summary = {
        "run_count": 12,
        "episode_count": 264,
        "record_count": 66_000,
        "termination_count": 0,
        "fall_count": 0,
        "policy_output_nonfinite_count": 0,
        "joint_limit_violation_count": 0,
        "joint_velocity_bound_violation_count": 0,
        "effort_bound_violation_count": 0,
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ValueError("full diagnostic summary totals are not exact")
    return {
        "report_path": str(path),
        "report_sha256": sha256_file(path),
        "report": report,
        "traces": sorted(
            trace_manifest,
            key=lambda item: (
                list(_SCENARIOS).index(item["scenario"]),
                expected_seeds.index(item["seed"]),
            ),
        ),
    }
