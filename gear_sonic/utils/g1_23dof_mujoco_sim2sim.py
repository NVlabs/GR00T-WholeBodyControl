"""Fail-closed MuJoCo Sim2Sim runner for native G1 rev-1.0 true23 policies.

This module is intentionally offline.  It imports neither IsaacLab nor Unitree
SDK code and has no network, DDS, PICO, or robot command surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np

from gear_sonic.utils.g1_23dof_artifact import (
    build_true23_policy_pair,
    canonical_json_bytes,
    canonical_true23_robot_asset_manifest,
    inspect_true23_policy_state,
    sha256_bytes,
    sha256_file,
    validate_training_checkpoint_records,
)
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    TRAINED_STAGE,
    checkpoint_stage,
    extract_global_step,
    load_safe_true23_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import (
    DECODER_OUTPUT_LAYOUT,
    DEPLOYMENT_DECODER_INPUT_DIM,
    HARDWARE_23_ACTION_SCALE,
    HARDWARE_23_JOINT_NAMES,
    HARDWARE_JOINT_IDS,
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_JOINT_NAMES,
    NATIVE_IL23_TO_CANONICAL_IL29,
    OBS_LAYOUT_PADDED_IL29,
    REQUIRED_MODE_MACHINE,
    ROBOT_MODEL,
    SOURCE_IL29_EXCLUDED_INDICES,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
    TELEOP_ENCODER_INPUT_TERM_DIMS,
    TELEOP_ENCODER_INPUT_TERM_ORDER,
    TOKEN_DIM,
)
from gear_sonic.utils.g1_23dof_live_shadow import (
    HARDWARE_DEFAULT_Q,
    build_proprio_frame,
    native_to_hardware,
)
from gear_sonic.utils.g1_23dof_mujoco_promotion import (
    CANDIDATE_KIND,
    PINNED_ASSET_SOURCE,
    verify_true23_mujoco_candidate,
)

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "g1_true23_mujoco_sim2sim_report"
TRACE_SCHEMA_VERSION = 1
TRACE_KIND = "g1_true23_mujoco_sim2sim_jsonl_trace"
PRODUCER_KIND = "g1_true23_mujoco_sim2sim_runner"
PRODUCER_VERSION = 1
REFERENCE_KIND = "g1_true23_neutral_fk_reference"
PHYSICS_CONTRACT_KIND = "g1_true23_mujoco_isaac_matched_physics"
EXPECTED_MUJOCO_DIMS = (30, 29, 23)
PROPRIO_HISTORY_LENGTH = 10
PROPRIO_FRAME_DIM = 93
PROPRIO_HISTORY_DIM = 930
DEFAULT_CONFIG_RELPATH = (
    "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
)
DEFAULT_MJCF_RELPATH = (
    "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
)
DEFAULT_URDF_RELPATH = (
    "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.urdf"
)
RUNNER_RELPATH = "gear_sonic/scripts/run_g1_23dof_mujoco_sim2sim.py"
RUNTIME_RELPATH = "gear_sonic/utils/g1_23dof_mujoco_sim2sim.py"
APPROVED_CONFIG_SHA256 = (
    "1ffce4afaeb82e20323e24cf4645a98259d9378ba5936e9d9d52e426241a25cf"
)

_ROOT = Path(__file__).resolve().parents[2]
_EXACT_CONFIG_KEYS = {
    "schema_version",
    "kind",
    "robot_model",
    "control_hz",
    "initial_state",
    "physics",
    "coverage",
    "disturbance_schedule",
    "disturbance_envelope",
    "scenarios",
    "termination",
    "promotion_thresholds",
}
_TRACE_RECORD_KEYS = {
    "schema_version",
    "kind",
    "scenario",
    "seed",
    "episode",
    "step",
    "time_s",
    "disturbance_delta",
    "base_position_m",
    "base_quaternion_wxyz",
    "base_linear_velocity_mps",
    "base_angular_velocity_radps",
    "projected_gravity",
    "joint_position_hardware_rad",
    "joint_velocity_hardware_radps",
    "action_native_raw",
    "action_native",
    "action_saturated_count",
    "target_position_hardware_rad",
    "applied_torque_hardware_nm",
    "base_height_m",
    "tilt_rad",
    "tracking_rmse_rad",
    "recovery_metric",
    "nonfinite",
    "joint_limit_violation",
    "terminated",
    "termination_reason",
}


def _strict_json(path: Path, context: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{context} contains non-finite number {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {context}: {exc}") from exc
    if not isinstance(result, Mapping):
        raise ValueError(f"{context} root must be an object")
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _finite_vector(value: Any, size: int, context: str) -> np.ndarray:
    if (
        not isinstance(value, (Sequence, np.ndarray))
        or isinstance(value, (str, bytes))
    ):
        raise ValueError(f"{context} must be an array")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{context} must contain {size} finite values")
    return result


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _canonical_float(value: float) -> float:
    if not math.isfinite(float(value)):
        return 1.0e30
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def _canonical_vector(values: Sequence[float]) -> list[float]:
    return [_canonical_float(value) for value in values]


def _atomic_write(path: Path, payload: bytes) -> None:
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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_sim2sim_config(path: Path) -> Mapping[str, Any]:
    """Load and validate immutable simulation coverage and physics settings."""

    config = _strict_json(path.resolve(), "MuJoCo Sim2Sim config")
    _exact_keys(config, _EXACT_CONFIG_KEYS, "MuJoCo Sim2Sim config")
    if (
        config["schema_version"] != 1
        or config["kind"] != "g1_true23_mujoco_sim2sim_config"
        or config["robot_model"] != ROBOT_MODEL
        or config["control_hz"] != 50
    ):
        raise ValueError("MuJoCo Sim2Sim config identity/control rate mismatch")
    initial_state = config["initial_state"]
    if not isinstance(initial_state, Mapping):
        raise ValueError("config.initial_state must be an object")
    _exact_keys(
        initial_state,
        {
            "base_position_m",
            "base_quaternion_wxyz",
            "joint_position_hardware_rad",
            "joint_velocity_hardware_radps",
        },
        "config.initial_state",
    )
    if not np.array_equal(
        _finite_vector(
            initial_state["base_position_m"],
            3,
            "config.initial_state.base_position_m",
        ),
        np.asarray((0.0, 0.0, 0.76)),
    ) or not np.array_equal(
        _finite_vector(
            initial_state["base_quaternion_wxyz"],
            4,
            "config.initial_state.base_quaternion_wxyz",
        ),
        np.asarray((1.0, 0.0, 0.0, 0.0)),
    ):
        raise ValueError("config initial root state differs from true23 Isaac config")
    if not np.array_equal(
        _finite_vector(
            initial_state["joint_position_hardware_rad"],
            TARGET_DOF,
            "config.initial_state.joint_position_hardware_rad",
        ),
        np.asarray(HARDWARE_DEFAULT_Q),
    ) or np.any(
        _finite_vector(
            initial_state["joint_velocity_hardware_radps"],
            TARGET_DOF,
            "config.initial_state.joint_velocity_hardware_radps",
        )
    ):
        raise ValueError("config initial joint state differs from true23 Isaac config")
    physics = config["physics"]
    if not isinstance(physics, Mapping):
        raise ValueError("config.physics must be an object")
    _exact_keys(
        physics,
        {
            "timestep_s",
            "control_decimation",
            "integrator",
            "gravity_mps2",
            "soft_joint_pos_limit_factor",
            "action_clip_value",
            "armature_hardware",
            "joint_damping_hardware",
            "joint_frictionloss_hardware",
            "kp_hardware",
            "kd_hardware",
            "effort_limit_hardware_nm",
            "velocity_limit_hardware_radps",
        },
        "config.physics",
    )
    if (
        physics["timestep_s"] != 0.002
        or physics["control_decimation"] != 10
        or physics["integrator"] != "Euler"
        or physics["soft_joint_pos_limit_factor"] != 0.9
        or physics["action_clip_value"] != 20.0
        or config["control_hz"] * physics["timestep_s"] * physics["control_decimation"] != 1
    ):
        raise ValueError("config must bind 0.002 s physics and exact 50 Hz policy loop")
    if not np.array_equal(
        _finite_vector(
            physics["gravity_mps2"],
            3,
            "config.physics.gravity_mps2",
        ),
        np.asarray((0.0, 0.0, -9.81)),
    ):
        raise ValueError("config gravity differs from exact MJCF gravity")
    for key in (
        "armature_hardware",
        "joint_damping_hardware",
        "joint_frictionloss_hardware",
        "kp_hardware",
        "kd_hardware",
        "effort_limit_hardware_nm",
        "velocity_limit_hardware_radps",
    ):
        values = _finite_vector(physics[key], TARGET_DOF, f"config.physics.{key}")
        if key not in {"joint_damping_hardware", "joint_frictionloss_hardware"} and (
            values <= 0
        ).any():
            raise ValueError(f"config.physics.{key} must be positive")
    coverage = config["coverage"]
    if not isinstance(coverage, Mapping):
        raise ValueError("config.coverage must be an object")
    _exact_keys(
        coverage,
        {
            "deterministic_seeds",
            "episodes_per_seed",
            "seconds_per_episode",
            "steps_per_episode",
        },
        "config.coverage",
    )
    seeds = coverage["deterministic_seeds"]
    if seeds != [1729, 2718, 3141]:
        raise ValueError("config.coverage deterministic seeds differ from approved set")
    episodes = _positive_int(coverage["episodes_per_seed"], "episodes_per_seed")
    steps = _positive_int(coverage["steps_per_episode"], "steps_per_episode")
    if episodes != 22 or steps != 250 or coverage["seconds_per_episode"] != 5.0:
        raise ValueError("config coverage must bind 66 episodes/scenario and 5 seconds")
    scenarios = config["scenarios"]
    if not isinstance(scenarios, Mapping) or dict(scenarios) != {
        "nominal": {"disturbance_scale": 0.0},
        "disturbance_50": {"disturbance_scale": 0.5},
        "disturbance_100": {"disturbance_scale": 1.0},
    }:
        raise ValueError("config scenarios must be exact nominal/50%/100% set")
    schedule = config["disturbance_schedule"]
    if not isinstance(schedule, Mapping) or dict(schedule) != {
        "apply_step": 50,
        "recovery_baseline_steps": 10,
        "recovery_stable_steps": 5,
        "recovery_margin": 0.1,
    }:
        raise ValueError("config disturbance schedule differs from approved values")
    envelope = config["disturbance_envelope"]
    if not isinstance(envelope, Mapping) or dict(envelope) != {
        "linear_velocity_delta_mps": {
            "x": [-0.5, 0.5],
            "y": [-0.5, 0.5],
            "z": [-0.2, 0.2],
        },
        "angular_velocity_delta_radps": {
            "roll": [-0.52, 0.52],
            "pitch": [-0.52, 0.52],
            "yaw": [-0.78, 0.78],
        },
    }:
        raise ValueError("config disturbance envelope differs from approved values")
    termination = config["termination"]
    if not isinstance(termination, Mapping) or dict(termination) != {
        "minimum_base_height_m": 0.45,
        "maximum_tilt_rad": 1.0,
        "maximum_joint_velocity_ratio": 1.0,
    }:
        raise ValueError("config termination contract differs from approved values")
    thresholds = config["promotion_thresholds"]
    if not isinstance(thresholds, Mapping) or dict(thresholds) != {
        "termination_count": 0,
        "nonfinite_count": 0,
        "joint_limit_violation_count": 0,
        "minimum_recovery_fraction": 1.0,
        "maximum_recovery_time_s": 2.0,
        "maximum_tracking_rmse_rad": 0.75,
        "maximum_native_action_abs": 20.0,
        "maximum_action_saturation_fraction": 0.1,
    }:
        raise ValueError("config promotion thresholds differ from approved values")
    return config


def _quaternion_conjugate(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = _finite_vector(quaternion, 4, "quaternion")
    return np.asarray((w, -x, -y, -z), dtype=np.float64)


def _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = _finite_vector(left, 4, "left quaternion")
    w2, x2, y2, z2 = _finite_vector(right, 4, "right quaternion")
    return np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float64,
    )


def _quaternion_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    quaternion = _finite_vector(quaternion, 4, "quaternion")
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _projected_gravity(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    return _quaternion_rotation_matrix(quaternion_wxyz).T @ np.asarray((0.0, 0.0, -1.0))


def _rotation_6d(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    return _quaternion_rotation_matrix(quaternion_wxyz)[:, :2].reshape(-1)


def _term_major_history(frames: Sequence[Sequence[float]]) -> list[float]:
    if len(frames) != PROPRIO_HISTORY_LENGTH or any(
        len(frame) != PROPRIO_FRAME_DIM for frame in frames
    ):
        raise ValueError("proprio history requires ten exact 93-value frames")
    result = [
        *(value for frame in frames for value in frame[0:3]),
        *(value for frame in frames for value in frame[3:32]),
        *(value for frame in frames for value in frame[32:61]),
        *(value for frame in frames for value in frame[61:90]),
        *(value for frame in frames for value in frame[90:93]),
    ]
    if len(result) != PROPRIO_HISTORY_DIM:
        raise AssertionError("term-major proprio history shape drift")
    for block_offset in (30, 320, 610):
        for frame_index in range(PROPRIO_HISTORY_LENGTH):
            for missing in SOURCE_IL29_EXCLUDED_INDICES:
                if result[block_offset + frame_index * 29 + missing] != 0.0:
                    raise ValueError("canonical IL29 missing observation slot became non-zero")
    return result


class True23PolicyRuntime:
    """CPU policy-pair runtime, optionally replaying the exact paired ONNX files."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        encoder_onnx_path: Path | None = None,
        decoder_onnx_path: Path | None = None,
        metadata_path: Path | None = None,
    ):
        import torch

        # Tiny MLP inference is slower and less deterministic with a large
        # intra-op thread pool. ONNX sessions below use the same single-thread
        # contract.
        torch.set_num_threads(1)
        self.checkpoint_path = checkpoint_path.resolve()
        self.checkpoint = load_safe_true23_checkpoint(
            self.checkpoint_path,
            map_location="cpu",
        )
        self.stage = checkpoint_stage(self.checkpoint)
        self.encoder, self.decoder, self.policy_state_sha256 = build_true23_policy_pair(
            self.checkpoint
        )
        inspected = inspect_true23_policy_state(self.checkpoint)
        if inspected != self.policy_state_sha256:
            raise ValueError("policy-state identity differs across exact validators")
        if self.stage == TRAINED_STAGE:
            validate_training_checkpoint_records(
                self.checkpoint,
                global_step=extract_global_step(self.checkpoint),
                policy_state_sha256=self.policy_state_sha256,
            )
        self._torch = torch
        paths = (encoder_onnx_path, decoder_onnx_path, metadata_path)
        if any(path is not None for path in paths) and not all(path is not None for path in paths):
            raise ValueError("paired ONNX runtime requires encoder, decoder, and metadata")
        self.encoder_onnx_path = (
            encoder_onnx_path.resolve() if encoder_onnx_path is not None else None
        )
        self.decoder_onnx_path = (
            decoder_onnx_path.resolve() if decoder_onnx_path is not None else None
        )
        self.metadata_path = metadata_path.resolve() if metadata_path is not None else None
        self.metadata: Mapping[str, Any] | None = None
        self.encoder_session: Any | None = None
        self.decoder_session: Any | None = None
        if self.encoder_onnx_path is not None:
            self._load_onnx_pair()

    def _load_onnx_pair(self) -> None:
        if self.stage != TRAINED_STAGE:
            raise ValueError("initialization checkpoints cannot be relabelled as paired ONNX")
        if (
            self.encoder_onnx_path is None
            or self.decoder_onnx_path is None
            or self.metadata_path is None
        ):
            raise AssertionError("paired ONNX paths disappeared")
        for path in (
            self.encoder_onnx_path,
            self.decoder_onnx_path,
            self.metadata_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.metadata = verify_true23_mujoco_candidate(
            self.encoder_onnx_path,
            self.decoder_onnx_path,
            self.metadata_path,
            checkpoint_path=self.checkpoint_path,
        )
        if self.metadata.get("artifact_kind") != CANDIDATE_KIND:
            raise ValueError("verified ONNX pair is not a true23 MuJoCo candidate")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for paired ONNX Sim2Sim") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.encoder_session = ort.InferenceSession(
            str(self.encoder_onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.decoder_session = ort.InferenceSession(
            str(self.decoder_onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._assert_session(self.encoder_session, TELEOP_ENCODER_INPUT_DIM, TOKEN_DIM, "encoder")
        self._assert_session(
            self.decoder_session,
            DEPLOYMENT_DECODER_INPUT_DIM,
            TARGET_DOF,
            "decoder",
        )

    @staticmethod
    def _assert_session(session: Any, input_dim: int, output_dim: int, context: str) -> None:
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
            raise ValueError(f"{context} ONNX static float32 structure mismatch")

    @property
    def uses_onnx(self) -> bool:
        return self.encoder_session is not None

    def infer(
        self,
        encoder_input: Sequence[float],
        proprio_history: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        encoder_array = np.asarray(encoder_input, dtype=np.float32).reshape(
            1, TELEOP_ENCODER_INPUT_DIM
        )
        history_array = np.asarray(proprio_history, dtype=np.float32).reshape(
            1, PROPRIO_HISTORY_DIM
        )
        if self.uses_onnx:
            token = self.encoder_session.run(
                None,
                {self.encoder_session.get_inputs()[0].name: encoder_array},
            )[0]
            decoder_input = np.concatenate((token, history_array), axis=1)
            action = self.decoder_session.run(
                None,
                {self.decoder_session.get_inputs()[0].name: decoder_input},
            )[0]
        else:
            with self._torch.no_grad():
                token_tensor = self.encoder(self._torch.from_numpy(encoder_array))
                decoder_input_tensor = self._torch.cat(
                    (token_tensor, self._torch.from_numpy(history_array)),
                    dim=1,
                )
                action_tensor = self.decoder(decoder_input_tensor)
            token = token_tensor.detach().cpu().numpy()
            action = action_tensor.detach().cpu().numpy()
        if token.shape != (1, TOKEN_DIM) or action.shape != (1, TARGET_DOF):
            raise ValueError("policy pair output shapes changed")
        return token[0].astype(np.float64), action[0].astype(np.float64)

    def source_artifact(self) -> dict[str, Any]:
        metadata_payload = (
            sha256_bytes(canonical_json_bytes(self.metadata))
            if self.metadata is not None
            else None
        )
        return {
            "artifact_kind": (
                "paired_onnx_trained_candidate"
                if self.uses_onnx
                else (
                    "trained_checkpoint_diagnostic"
                    if self.stage == TRAINED_STAGE
                    else "checkpoint_initialization_diagnostic"
                )
            ),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "policy_state_sha256": self.policy_state_sha256,
            "encoder_onnx_sha256": (
                sha256_file(self.encoder_onnx_path)
                if self.encoder_onnx_path is not None
                else None
            ),
            "decoder_onnx_sha256": (
                sha256_file(self.decoder_onnx_path)
                if self.decoder_onnx_path is not None
                else None
            ),
            "candidate_manifest_sha256": (
                sha256_file(self.metadata_path) if self.metadata_path is not None else None
            ),
            "candidate_manifest_payload_sha256": metadata_payload,
            "candidate_claimed_payload_sha256": (
                self.metadata.get("metadata_payload_sha256")
                if self.metadata is not None
                else None
            ),
            "inference_runtime": (
                "onnxruntime_cpu"
                if self.uses_onnx
                else "pytorch_checkpoint_cpu"
            ),
            "inference_threads": 1,
        }

    @property
    def promotion_source_complete(self) -> bool:
        artifact = self.source_artifact()
        return (
            self.stage == TRAINED_STAGE
            and self.uses_onnx
            and self.metadata is not None
            and self.metadata.get("artifact_kind") == CANDIDATE_KIND
            and self.metadata.get("deployment_authorized") is False
            and self.metadata.get("deployment_ready") is False
            and self.metadata.get("sim_validation_passed") is False
            and all(
                artifact[key] is not None
                for key in (
                    "checkpoint_sha256",
                    "policy_state_sha256",
                    "encoder_onnx_sha256",
                    "decoder_onnx_sha256",
                    "candidate_manifest_sha256",
                    "candidate_manifest_payload_sha256",
                    "candidate_claimed_payload_sha256",
                )
            )
        )


class NeutralReference:
    """Exact FK-derived, semantically valid stand reference for encoder input."""

    _BODY_NAMES = (
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand",
        "torso_link",
    )
    _BODY_OFFSETS = (
        (0.18, -0.025, 0.0),
        (0.18, 0.025, 0.0),
        (0.0, 0.0, 0.35),
    )

    def __init__(
        self,
        model: Any,
        mujoco_module: Any,
        initial_state: Mapping[str, Any],
    ):
        data = mujoco_module.MjData(model)
        data.qpos[:] = model.qpos0
        data.qpos[:3] = np.asarray(initial_state["base_position_m"])
        data.qpos[3:7] = np.asarray(initial_state["base_quaternion_wxyz"])
        data.qpos[7:] = np.asarray(initial_state["joint_position_hardware_rad"])
        data.qvel[:] = 0.0
        data.qvel[6:] = np.asarray(initial_state["joint_velocity_hardware_radps"])
        mujoco_module.mj_forward(model, data)
        self.root_position = np.asarray(data.qpos[:3], dtype=np.float64).copy()
        self.root_quaternion = np.asarray(data.qpos[3:7], dtype=np.float64).copy()
        root_rotation_inverse = _quaternion_rotation_matrix(
            self.root_quaternion
        ).T
        local_positions: list[float] = []
        local_orientations: list[float] = []
        for name, offset in zip(self._BODY_NAMES, self._BODY_OFFSETS, strict=True):
            body_id = mujoco_module.mj_name2id(
                model,
                mujoco_module.mjtObj.mjOBJ_BODY,
                name,
            )
            if body_id < 0:
                raise ValueError(f"neutral reference body missing: {name}")
            body_quaternion = np.asarray(data.xquat[body_id], dtype=np.float64)
            body_rotation = _quaternion_rotation_matrix(body_quaternion)
            point_world = (
                np.asarray(data.xpos[body_id], dtype=np.float64)
                + body_rotation @ np.asarray(offset, dtype=np.float64)
            )
            local_positions.extend(
                (root_rotation_inverse @ (point_world - self.root_position)).tolist()
            )
            local_orientations.extend(
                _quaternion_multiply(
                    _quaternion_conjugate(self.root_quaternion),
                    body_quaternion,
                ).tolist()
            )
        lower_positions = list(HARDWARE_DEFAULT_Q[:12]) * 10
        lower_velocities = [0.0] * 120
        self.fixed_terms = [
            *lower_positions,
            *lower_velocities,
            *local_positions,
            *local_orientations,
        ]
        if len(self.fixed_terms) != 261:
            raise AssertionError("neutral encoder fixed-term shape drift")

    def encoder_input(self, measured_root_quaternion: Sequence[float]) -> list[float]:
        relative = _quaternion_multiply(
            _quaternion_conjugate(measured_root_quaternion),
            self.root_quaternion,
        )
        result = [*self.fixed_terms, *_rotation_6d(relative).tolist()]
        if len(result) != TELEOP_ENCODER_INPUT_DIM or not np.isfinite(result).all():
            raise ValueError("neutral semantic encoder input is invalid")
        return result

    def descriptor(self) -> dict[str, Any]:
        vector = self.encoder_input(self.root_quaternion)
        return {
            "kind": REFERENCE_KIND,
            "body_names": list(self._BODY_NAMES),
            "body_offsets": [list(value) for value in self._BODY_OFFSETS],
            "encoder_term_order": list(TELEOP_ENCODER_INPUT_TERM_ORDER),
            "encoder_term_dims": list(TELEOP_ENCODER_INPUT_TERM_DIMS),
            "encoder_input_sha256_at_neutral": sha256_bytes(
                canonical_json_bytes(_canonical_vector(vector))
            ),
            "zeros_only_command": False,
        }


def _mujoco_name(module: Any, model: Any, object_type: Any, index: int) -> str:
    value = module.mj_id2name(model, object_type, index)
    if value is None:
        raise ValueError(f"unnamed MuJoCo object at index {index}")
    return str(value)


def _asset_provenance() -> dict[str, Any]:
    manifest = canonical_true23_robot_asset_manifest(repo_root=_ROOT)
    expected = {
        "schema_version": 1,
        "file_count": PINNED_ASSET_SOURCE["file_count"],
        "total_bytes": PINNED_ASSET_SOURCE["total_bytes"],
        "manifest_sha256": PINNED_ASSET_SOURCE["manifest_sha256"],
    }
    if {key: manifest[key] for key in expected} != expected:
        raise ValueError("local true23 assets differ from pinned Unitree manifest")
    return {
        "source": dict(PINNED_ASSET_SOURCE),
        "verified": True,
        "urdf_sha256": sha256_file(_ROOT / DEFAULT_URDF_RELPATH),
        "mjcf_sha256": sha256_file(_ROOT / DEFAULT_MJCF_RELPATH),
        "local_manifest": manifest,
    }


def _compiled_model_contract(module: Any, model: Any) -> dict[str, Any]:
    joint_ids = np.arange(1, model.njnt, dtype=np.int32)
    floor_id = module.mj_name2id(
        model,
        module.mjtObj.mjOBJ_GEOM,
        "floor",
    )
    if floor_id < 0:
        raise ValueError("exact MJCF floor geom is missing")
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "njnt": int(model.njnt),
        "joint_qpos_addresses": [
            int(value) for value in model.jnt_qposadr[joint_ids]
        ],
        "joint_dof_addresses": [
            int(value) for value in model.jnt_dofadr[joint_ids]
        ],
        "actuator_joint_ids": [
            int(value) for value in model.actuator_trnid[:, 0]
        ],
        "floor": {
            "geom_id": int(floor_id),
            "type": int(model.geom_type[floor_id]),
            "contype": int(model.geom_contype[floor_id]),
            "conaffinity": int(model.geom_conaffinity[floor_id]),
            "condim": int(model.geom_condim[floor_id]),
            "friction": _canonical_vector(model.geom_friction[floor_id]),
            "solref": _canonical_vector(model.geom_solref[floor_id]),
            "solimp": _canonical_vector(model.geom_solimp[floor_id]),
            "margin": _canonical_float(model.geom_margin[floor_id]),
            "gap": _canonical_float(model.geom_gap[floor_id]),
        },
    }


def prepare_mujoco_model(
    *,
    mjcf_path: Path,
    config: Mapping[str, Any],
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Load exact rev-1.0 MJCF and apply hash-bound Isaac actuator physics."""

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("mujoco Python package is required") from exc
    path = mjcf_path.resolve()
    if path.name != "g1_23dof_rev_1_0.xml":
        raise ValueError("only exact g1_23dof_rev_1_0.xml is accepted")
    model = mujoco.MjModel.from_xml_path(str(path))
    if (model.nq, model.nv, model.nu) != EXPECTED_MUJOCO_DIMS:
        raise ValueError(
            f"MuJoCo dimensions must be {EXPECTED_MUJOCO_DIMS}; "
            f"got {(model.nq, model.nv, model.nu)}"
        )
    if model.njnt != TARGET_DOF + 1:
        raise ValueError("MuJoCo model must contain one free joint plus 23 motors")
    joint_names = [
        _mujoco_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    ]
    actuator_names = [
        _mujoco_name(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ]
    if joint_names != list(HARDWARE_23_JOINT_NAMES) or actuator_names != joint_names:
        raise ValueError("MuJoCo joint/actuator order differs from hardware true23")
    joint_ids = np.arange(1, model.njnt, dtype=np.int32)
    if not np.array_equal(model.actuator_trnid[:, 0], joint_ids):
        raise ValueError("MuJoCo actuator-to-joint mapping is not one-to-one true23")
    physics = config["physics"]
    if (
        not math.isclose(float(model.opt.timestep), float(physics["timestep_s"]))
        or int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_EULER)
        or not np.array_equal(model.opt.gravity, np.asarray(physics["gravity_mps2"]))
    ):
        raise ValueError("MJCF timestep/integrator/gravity differs from pinned config")
    dof_addresses = model.jnt_dofadr[joint_ids]
    armature = _finite_vector(
        physics["armature_hardware"],
        TARGET_DOF,
        "config.physics.armature_hardware",
    )
    damping = _finite_vector(
        physics["joint_damping_hardware"],
        TARGET_DOF,
        "config.physics.joint_damping_hardware",
    )
    friction = _finite_vector(
        physics["joint_frictionloss_hardware"],
        TARGET_DOF,
        "config.physics.joint_frictionloss_hardware",
    )
    effort = _finite_vector(
        physics["effort_limit_hardware_nm"],
        TARGET_DOF,
        "config.physics.effort_limit_hardware_nm",
    )
    model.dof_armature[dof_addresses] = armature
    model.dof_damping[dof_addresses] = damping
    model.dof_frictionloss[dof_addresses] = friction
    model.jnt_actfrclimited[joint_ids] = 1
    model.jnt_actfrcrange[joint_ids, 0] = -effort
    model.jnt_actfrcrange[joint_ids, 1] = effort
    if (
        not np.array_equal(model.dof_armature[dof_addresses], armature)
        or not np.array_equal(model.dof_damping[dof_addresses], damping)
        or not np.array_equal(model.dof_frictionloss[dof_addresses], friction)
        or not np.all(model.jnt_actfrclimited[joint_ids] == 1)
        or not np.array_equal(model.jnt_actfrcrange[joint_ids, 1], effort)
    ):
        raise ValueError("MuJoCo Isaac-matched physics override did not apply exactly")
    physics_contract = {
        "kind": PHYSICS_CONTRACT_KIND,
        "timestep_s": physics["timestep_s"],
        "control_decimation": physics["control_decimation"],
        "control_hz": config["control_hz"],
        "integrator": physics["integrator"],
        "gravity_mps2": list(physics["gravity_mps2"]),
        "soft_joint_pos_limit_factor": physics["soft_joint_pos_limit_factor"],
        "action_clip_value": physics["action_clip_value"],
        "cone": int(model.opt.cone),
        "jacobian": int(model.opt.jacobian),
        "solver": int(model.opt.solver),
        "iterations": int(model.opt.iterations),
        "ls_iterations": int(model.opt.ls_iterations),
        "noslip_iterations": int(model.opt.noslip_iterations),
        "armature_hardware": list(physics["armature_hardware"]),
        "joint_damping_hardware": list(physics["joint_damping_hardware"]),
        "joint_frictionloss_hardware": list(physics["joint_frictionloss_hardware"]),
        "kp_hardware": list(physics["kp_hardware"]),
        "kd_hardware": list(physics["kd_hardware"]),
        "effort_limit_hardware_nm": list(physics["effort_limit_hardware_nm"]),
        "velocity_limit_hardware_radps": list(
            physics["velocity_limit_hardware_radps"]
        ),
    }
    physics_contract["payload_sha256"] = sha256_bytes(
        canonical_json_bytes(physics_contract)
    )
    return mujoco, model, physics_contract


def deterministic_disturbance(
    *,
    config: Mapping[str, Any],
    seed: int,
    episode: int,
    scale: float,
) -> list[float]:
    envelope = config["disturbance_envelope"]
    bounds = (
        envelope["linear_velocity_delta_mps"]["x"],
        envelope["linear_velocity_delta_mps"]["y"],
        envelope["linear_velocity_delta_mps"]["z"],
        envelope["angular_velocity_delta_radps"]["roll"],
        envelope["angular_velocity_delta_radps"]["pitch"],
        envelope["angular_velocity_delta_radps"]["yaw"],
    )
    result: list[float] = []
    for axis, (lower, upper) in enumerate(bounds):
        digest = hashlib.sha256(
            f"{ROBOT_MODEL}:mujoco:{seed}:{episode}:{axis}".encode("ascii")
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        result.append(
            round((float(lower) + unit * (float(upper) - float(lower))) * scale, 12)
        )
    return result


def apply_world_velocity_disturbance(
    qvel: np.ndarray,
    quaternion_wxyz: Sequence[float],
    disturbance_world: Sequence[float],
) -> None:
    """Apply Isaac-style world velocity delta to MuJoCo free-joint qvel."""

    disturbance = _finite_vector(
        disturbance_world,
        6,
        "world-frame disturbance",
    )
    if qvel.ndim != 1 or qvel.shape[0] < 6:
        raise ValueError("MuJoCo qvel must contain a six-axis free joint")
    qvel[:3] += disturbance[:3]
    # MuJoCo free-joint angular qvel uses child-body local coordinates.
    qvel[3:6] += (
        _quaternion_rotation_matrix(quaternion_wxyz).T @ disturbance[3:6]
    )


def approved_offline_input_paths(config_path: Path, mjcf_path: Path) -> bool:
    """Return whether paths/bytes match immutable promotion inputs."""

    approved_config_path = (_ROOT / DEFAULT_CONFIG_RELPATH).resolve()
    approved_mjcf_path = (_ROOT / DEFAULT_MJCF_RELPATH).resolve()
    return bool(
        config_path.resolve() == approved_config_path
        and sha256_file(config_path.resolve()) == APPROVED_CONFIG_SHA256
        and mjcf_path.resolve() == approved_mjcf_path
    )


def _sensor_vector(module: Any, model: Any, data: Any, name: str) -> np.ndarray:
    sensor_id = module.mj_name2id(model, module.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0 or model.sensor_dim[sensor_id] != 3:
        raise ValueError(f"required three-axis MuJoCo sensor missing: {name}")
    address = int(model.sensor_adr[sensor_id])
    return np.asarray(data.sensordata[address : address + 3], dtype=np.float64).copy()


def _termination_reason(
    *,
    base_height: float,
    tilt: float,
    velocity_ratio: float,
    nonfinite: bool,
    config: Mapping[str, Any],
) -> str:
    termination = config["termination"]
    if nonfinite:
        return "nonfinite"
    if base_height < float(termination["minimum_base_height_m"]):
        return "base_height"
    if tilt > float(termination["maximum_tilt_rad"]):
        return "base_tilt"
    if velocity_ratio > float(termination["maximum_joint_velocity_ratio"]):
        return "joint_velocity"
    return ""


def run_episode(
    *,
    module: Any,
    model: Any,
    runtime: True23PolicyRuntime,
    reference: NeutralReference,
    config: Mapping[str, Any],
    scenario: str,
    seed: int,
    episode: int,
    disturbance_scale: float,
) -> list[dict[str, Any]]:
    """Run one deterministic closed-loop episode and return raw trace records."""

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
        physics["velocity_limit_hardware_radps"],
        dtype=np.float64,
    )
    action_scale = np.asarray(HARDWARE_23_ACTION_SCALE, dtype=np.float64)
    default_q = np.asarray(HARDWARE_DEFAULT_Q, dtype=np.float64)
    hard_joint_ranges = np.asarray(model.jnt_range[1:], dtype=np.float64)
    joint_midpoint = np.mean(hard_joint_ranges, axis=1)
    joint_half_range = (
        (hard_joint_ranges[:, 1] - hard_joint_ranges[:, 0])
        * float(physics["soft_joint_pos_limit_factor"])
        * 0.5
    )
    soft_joint_ranges = np.column_stack(
        (joint_midpoint - joint_half_range, joint_midpoint + joint_half_range)
    )
    previous_action_native = np.zeros(TARGET_DOF, dtype=np.float64)
    history_frames: list[list[float]] = []
    records: list[dict[str, Any]] = []
    latched_reason = ""
    apply_step = int(config["disturbance_schedule"]["apply_step"])
    steps = int(config["coverage"]["steps_per_episode"])
    disturbance = deterministic_disturbance(
        config=config,
        seed=seed,
        episode=episode,
        scale=disturbance_scale,
    )
    for step in range(steps):
        step_disturbance = [0.0] * 6
        if step == apply_step and disturbance_scale > 0:
            apply_world_velocity_disturbance(
                data.qvel,
                data.qpos[3:7],
                disturbance,
            )
            step_disturbance = disturbance
            module.mj_forward(model, data)
        hardware_q_before = np.asarray(data.qpos[7:], dtype=np.float64).copy()
        hardware_dq_before = np.asarray(data.qvel[6:], dtype=np.float64).copy()
        root_quaternion_before = np.asarray(data.qpos[3:7], dtype=np.float64).copy()
        gyro = _sensor_vector(
            module,
            model,
            data,
            "imu-pelvis-angular-velocity",
        )
        frame = build_proprio_frame(
            hardware_q=hardware_q_before.tolist(),
            hardware_dq=hardware_dq_before.tolist(),
            imu_gyroscope=gyro.tolist(),
            imu_quaternion_wxyz=root_quaternion_before.tolist(),
            previous_action_native=previous_action_native.tolist(),
        )
        if not history_frames:
            history_frames = [frame.copy() for _ in range(PROPRIO_HISTORY_LENGTH)]
        else:
            history_frames = [*history_frames[1:], frame]
        history = _term_major_history(history_frames)
        encoder_input = reference.encoder_input(root_quaternion_before)
        token, action_native_raw = runtime.infer(encoder_input, history)
        output_nonfinite = not (
            np.isfinite(token).all() and np.isfinite(action_native_raw).all()
        )
        finite_action = (
            np.zeros(TARGET_DOF, dtype=np.float64)
            if output_nonfinite
            else action_native_raw
        )
        safe_action = np.clip(
            finite_action,
            -float(physics["action_clip_value"]),
            float(physics["action_clip_value"]),
        )
        action_saturated_count = int(np.count_nonzero(safe_action != finite_action))
        action_hardware = np.asarray(native_to_hardware(safe_action.tolist()))
        target_hardware = default_q + action_hardware * action_scale
        peak_torque = np.zeros(TARGET_DOF, dtype=np.float64)
        for _ in range(int(physics["control_decimation"])):
            q = np.asarray(data.qpos[7:], dtype=np.float64)
            dq = np.asarray(data.qvel[6:], dtype=np.float64)
            torque = np.clip(kp * (target_hardware - q) - kd * dq, -effort, effort)
            replace = np.abs(torque) >= np.abs(peak_torque)
            peak_torque[replace] = torque[replace]
            data.ctrl[:] = torque
            module.mj_step(model, data)
        hardware_q = np.asarray(data.qpos[7:], dtype=np.float64).copy()
        hardware_dq = np.asarray(data.qvel[6:], dtype=np.float64).copy()
        root_position = np.asarray(data.qpos[:3], dtype=np.float64).copy()
        root_quaternion = np.asarray(data.qpos[3:7], dtype=np.float64).copy()
        root_linear_velocity = np.asarray(data.qvel[:3], dtype=np.float64).copy()
        root_angular_velocity = _sensor_vector(
            module,
            model,
            data,
            "imu-pelvis-angular-velocity",
        )
        gravity = _projected_gravity(root_quaternion)
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        tracking_rmse = float(np.sqrt(np.mean((target_hardware - hardware_q) ** 2)))
        recovery_metric = (
            tilt
            + abs(float(root_position[2]) - float(reference.root_position[2]))
            + tracking_rmse
        )
        state_nonfinite = not all(
            np.isfinite(value).all()
            for value in (
                hardware_q,
                hardware_dq,
                root_position,
                root_quaternion,
                root_linear_velocity,
                root_angular_velocity,
                peak_torque,
            )
        )
        nonfinite = output_nonfinite or state_nonfinite
        joint_limit_violation = bool(
            np.any(hardware_q < soft_joint_ranges[:, 0])
            or np.any(hardware_q > soft_joint_ranges[:, 1])
            or np.any(target_hardware < soft_joint_ranges[:, 0])
            or np.any(target_hardware > soft_joint_ranges[:, 1])
        )
        velocity_ratio = float(
            np.max(np.abs(hardware_dq) / velocity_limits)
        )
        if not latched_reason:
            latched_reason = _termination_reason(
                base_height=float(root_position[2]),
                tilt=tilt,
                velocity_ratio=velocity_ratio,
                nonfinite=nonfinite,
                config=config,
            )
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "kind": TRACE_KIND,
            "scenario": scenario,
            "seed": seed,
            "episode": episode,
            "step": step,
            "time_s": _canonical_float((step + 1) / config["control_hz"]),
            "disturbance_delta": _canonical_vector(step_disturbance),
            "base_position_m": _canonical_vector(root_position),
            "base_quaternion_wxyz": _canonical_vector(root_quaternion),
            "base_linear_velocity_mps": _canonical_vector(root_linear_velocity),
            "base_angular_velocity_radps": _canonical_vector(root_angular_velocity),
            "projected_gravity": _canonical_vector(gravity),
            "joint_position_hardware_rad": _canonical_vector(hardware_q),
            "joint_velocity_hardware_radps": _canonical_vector(hardware_dq),
            "action_native_raw": _canonical_vector(action_native_raw),
            "action_native": _canonical_vector(safe_action),
            "action_saturated_count": action_saturated_count,
            "target_position_hardware_rad": _canonical_vector(target_hardware),
            "applied_torque_hardware_nm": _canonical_vector(peak_torque),
            "base_height_m": _canonical_float(root_position[2]),
            "tilt_rad": _canonical_float(tilt),
            "tracking_rmse_rad": _canonical_float(tracking_rmse),
            "recovery_metric": _canonical_float(recovery_metric),
            "nonfinite": nonfinite,
            "joint_limit_violation": joint_limit_violation,
            "terminated": bool(latched_reason),
            "termination_reason": latched_reason,
        }
        _exact_keys(record, _TRACE_RECORD_KEYS, "internal trace record")
        records.append(record)
        previous_action_native = safe_action
    return records


def recompute_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    disturbance_scale: float,
) -> dict[str, Any]:
    """Recompute all run metrics from raw records; trust no pass boolean."""

    if not records:
        raise ValueError("trace has no records")
    by_episode: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("trace record must be an object")
        _exact_keys(record, _TRACE_RECORD_KEYS, "trace record")
        episode = int(record["episode"])
        by_episode.setdefault(episode, []).append(record)
    expected_steps = int(config["coverage"]["steps_per_episode"])
    for episode, episode_records in by_episode.items():
        if [int(record["step"]) for record in episode_records] != list(
            range(expected_steps)
        ):
            raise ValueError(f"episode {episode} trace steps are not contiguous")
    termination_count = sum(
        any(bool(record["terminated"]) for record in episode_records)
        for episode_records in by_episode.values()
    )
    nonfinite_count = sum(bool(record["nonfinite"]) for record in records)
    joint_limit_count = sum(
        bool(record["joint_limit_violation"]) for record in records
    )
    tracking = [float(record["tracking_rmse_rad"]) for record in records]
    base_heights = [float(record["base_height_m"]) for record in records]
    tilts = [float(record["tilt_rad"]) for record in records]
    max_joint_velocity = max(
        abs(float(value))
        for record in records
        for value in record["joint_velocity_hardware_radps"]
    )
    max_torque = max(
        abs(float(value))
        for record in records
        for value in record["applied_torque_hardware_nm"]
    )
    max_action = max(
        abs(float(value))
        for record in records
        for value in record["action_native"]
    )
    max_action_raw = max(
        abs(float(value))
        for record in records
        for value in record["action_native_raw"]
    )
    saturation_fraction = sum(
        int(record["action_saturated_count"]) for record in records
    ) / (len(records) * TARGET_DOF)
    recovered = 0
    recovery_times: list[float] = []
    if disturbance_scale > 0:
        schedule = config["disturbance_schedule"]
        apply_step = int(schedule["apply_step"])
        baseline_steps = int(schedule["recovery_baseline_steps"])
        stable_steps = int(schedule["recovery_stable_steps"])
        margin = float(schedule["recovery_margin"])
        for episode_records in by_episode.values():
            baseline = [
                float(record["recovery_metric"])
                for record in episode_records[
                    apply_step - baseline_steps : apply_step
                ]
            ]
            threshold = sum(baseline) / len(baseline) + margin
            consecutive = 0
            recovered_step: int | None = None
            for record in episode_records[apply_step:]:
                if (
                    not record["terminated"]
                    and float(record["recovery_metric"]) <= threshold
                ):
                    consecutive += 1
                    if consecutive >= stable_steps:
                        recovered_step = int(record["step"]) - stable_steps + 1
                        break
                else:
                    consecutive = 0
            if recovered_step is None:
                recovery_times.append(
                    expected_steps / float(config["control_hz"])
                )
            else:
                recovered += 1
                recovery_times.append(
                    max(0.0, (recovered_step - apply_step) / config["control_hz"])
                )
    else:
        recovered = len(by_episode)
        recovery_times = [0.0] * len(by_episode)
    return {
        "episode_count": len(by_episode),
        "record_count": len(records),
        "termination_count": termination_count,
        "nonfinite_count": nonfinite_count,
        "joint_limit_violation_count": joint_limit_count,
        "min_base_height_m": _canonical_float(min(base_heights)),
        "max_tilt_rad": _canonical_float(max(tilts)),
        "max_tracking_rmse_rad": _canonical_float(max(tracking)),
        "mean_tracking_rmse_rad": _canonical_float(
            sum(tracking) / len(tracking)
        ),
        "max_abs_joint_velocity_radps": _canonical_float(max_joint_velocity),
        "max_abs_applied_torque_nm": _canonical_float(max_torque),
        "max_abs_native_action": _canonical_float(max_action),
        "max_abs_native_action_raw": _canonical_float(max_action_raw),
        "action_saturation_fraction": _canonical_float(saturation_fraction),
        "recovery_fraction": _canonical_float(recovered / len(by_episode)),
        "max_recovery_time_s": _canonical_float(max(recovery_times)),
    }


def metrics_pass(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    thresholds = config["promotion_thresholds"]
    return bool(
        metrics["termination_count"] <= thresholds["termination_count"]
        and metrics["nonfinite_count"] <= thresholds["nonfinite_count"]
        and metrics["joint_limit_violation_count"]
        <= thresholds["joint_limit_violation_count"]
        and metrics["recovery_fraction"]
        >= thresholds["minimum_recovery_fraction"]
        and metrics["max_recovery_time_s"]
        <= thresholds["maximum_recovery_time_s"]
        and metrics["max_tracking_rmse_rad"]
        <= thresholds["maximum_tracking_rmse_rad"]
        and metrics["max_abs_native_action"]
        <= thresholds["maximum_native_action_abs"]
        and metrics["action_saturation_fraction"]
        <= thresholds["maximum_action_saturation_fraction"]
    )


def _write_trace(
    *,
    output_path: Path,
    scenario: str,
    seed: int,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trace_dir = output_path.with_name(f"{output_path.stem}.traces")
    trace_path = trace_dir / f"{scenario}-seed-{seed}.jsonl"
    payload = b"".join(canonical_json_bytes(record) for record in records)
    _atomic_write(trace_path, payload)
    return {
        "file": f"{trace_dir.name}/{trace_path.name}",
        "sha256": sha256_bytes(payload),
        "payload_sha256": sha256_bytes(canonical_json_bytes(list(records))),
        "record_count": len(records),
    }


def _contract_descriptor() -> dict[str, Any]:
    return {
        "robot_model": ROBOT_MODEL,
        "required_mode_machine": REQUIRED_MODE_MACHINE,
        "action_dof": TARGET_DOF,
        "encoder_input_dim": TELEOP_ENCODER_INPUT_DIM,
        "token_dim": TOKEN_DIM,
        "decoder_input_dim": DEPLOYMENT_DECODER_INPUT_DIM,
        "decoder_output_dim": TARGET_DOF,
        "decoder_output_layout": DECODER_OUTPUT_LAYOUT,
        "observation_layout": OBS_LAYOUT_PADDED_IL29,
        "hardware_joint_ids": list(HARDWARE_JOINT_IDS),
        "hardware_joint_names": list(HARDWARE_23_JOINT_NAMES),
        "native_joint_names": list(NATIVE_IL23_JOINT_NAMES),
        "native_to_hardware": list(ISAACLAB_TO_MUJOCO_DOF),
        "hardware_to_native": list(MUJOCO_TO_ISAACLAB_DOF),
        "native_to_canonical_il29": list(NATIVE_IL23_TO_CANONICAL_IL29),
        "missing_canonical_il29": list(SOURCE_IL29_EXCLUDED_INDICES),
        "missing_q_rel_fill": "fixed_default_relative_zero",
        "missing_velocity_fill": "zero",
        "missing_previous_action_fill": "zero",
        "history_layout": "term_major_oldest_to_newest",
        "hardware_action_scale": list(HARDWARE_23_ACTION_SCALE),
        "naive_29_output_masking": False,
    }


def run_sim2sim_validation(
    *,
    checkpoint_path: Path,
    output_path: Path,
    config_path: Path | None = None,
    mjcf_path: Path | None = None,
    encoder_onnx_path: Path | None = None,
    decoder_onnx_path: Path | None = None,
    metadata_path: Path | None = None,
) -> Mapping[str, Any]:
    """Run deterministic native-23 MuJoCo evidence and write raw JSONL traces."""

    output_path = output_path.resolve()
    trace_dir = output_path.with_name(f"{output_path.stem}.traces")
    if output_path.exists() or trace_dir.exists():
        raise FileExistsError("refusing to overwrite Sim2Sim report or trace directory")
    config_path = (
        config_path.resolve()
        if config_path is not None
        else (_ROOT / DEFAULT_CONFIG_RELPATH).resolve()
    )
    mjcf_path = (
        mjcf_path.resolve()
        if mjcf_path is not None
        else (_ROOT / DEFAULT_MJCF_RELPATH).resolve()
    )
    config = load_sim2sim_config(config_path)
    runtime = True23PolicyRuntime(
        checkpoint_path=checkpoint_path,
        encoder_onnx_path=encoder_onnx_path,
        decoder_onnx_path=decoder_onnx_path,
        metadata_path=metadata_path,
    )
    module, model, physics_contract = prepare_mujoco_model(
        mjcf_path=mjcf_path,
        config=config,
    )
    reference = NeutralReference(model, module, config["initial_state"])
    runs: list[dict[str, Any]] = []
    trace_manifest: list[dict[str, Any]] = []
    for scenario, scenario_config in config["scenarios"].items():
        scale = float(scenario_config["disturbance_scale"])
        for seed in config["coverage"]["deterministic_seeds"]:
            records: list[dict[str, Any]] = []
            for episode in range(config["coverage"]["episodes_per_seed"]):
                records.extend(
                    run_episode(
                        module=module,
                        model=model,
                        runtime=runtime,
                        reference=reference,
                        config=config,
                        scenario=scenario,
                        seed=seed,
                        episode=episode,
                        disturbance_scale=scale,
                    )
                )
            trace = _write_trace(
                output_path=output_path,
                scenario=scenario,
                seed=seed,
                records=records,
            )
            metrics = recompute_metrics(
                records,
                config=config,
                disturbance_scale=scale,
            )
            run = {
                "scenario": scenario,
                "seed": seed,
                "episodes": config["coverage"]["episodes_per_seed"],
                "steps_per_episode": config["coverage"]["steps_per_episode"],
                "disturbance_scale": scale,
                "computed_pass": metrics_pass(metrics, config),
                "metrics": metrics,
                "trace": trace,
            }
            runs.append(run)
            trace_manifest.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    **trace,
                }
            )
    summary_metrics = {
        "run_count": len(runs),
        "episode_count": sum(run["metrics"]["episode_count"] for run in runs),
        "record_count": sum(run["metrics"]["record_count"] for run in runs),
        "termination_count": sum(
            run["metrics"]["termination_count"] for run in runs
        ),
        "nonfinite_count": sum(run["metrics"]["nonfinite_count"] for run in runs),
        "joint_limit_violation_count": sum(
            run["metrics"]["joint_limit_violation_count"] for run in runs
        ),
        "min_base_height_m": min(
            run["metrics"]["min_base_height_m"] for run in runs
        ),
        "max_tilt_rad": max(run["metrics"]["max_tilt_rad"] for run in runs),
        "max_tracking_rmse_rad": max(
            run["metrics"]["max_tracking_rmse_rad"] for run in runs
        ),
        "max_abs_joint_velocity_radps": max(
            run["metrics"]["max_abs_joint_velocity_radps"] for run in runs
        ),
        "max_abs_applied_torque_nm": max(
            run["metrics"]["max_abs_applied_torque_nm"] for run in runs
        ),
        "max_abs_native_action": max(
            run["metrics"]["max_abs_native_action"] for run in runs
        ),
        "max_abs_native_action_raw": max(
            run["metrics"]["max_abs_native_action_raw"] for run in runs
        ),
        "max_action_saturation_fraction": max(
            run["metrics"]["action_saturation_fraction"] for run in runs
        ),
        "minimum_recovery_fraction": min(
            run["metrics"]["recovery_fraction"] for run in runs
        ),
        "max_recovery_time_s": max(
            run["metrics"]["max_recovery_time_s"] for run in runs
        ),
    }
    computed_pass = all(run["computed_pass"] for run in runs)
    approved_offline_inputs = approved_offline_input_paths(
        config_path,
        mjcf_path,
    )
    promotion_eligible = (
        runtime.promotion_source_complete
        and computed_pass
        and approved_offline_inputs
    )
    runner_path = (_ROOT / RUNNER_RELPATH).resolve()
    runtime_path = (_ROOT / RUNTIME_RELPATH).resolve()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "robot_model": ROBOT_MODEL,
        "checkpoint_stage": runtime.stage,
        "diagnostic_only": not promotion_eligible,
        "promotion_eligible": promotion_eligible,
        "computed_pass": computed_pass,
        "source_artifact": runtime.source_artifact(),
        "producer": {
            "kind": PRODUCER_KIND,
            "version": PRODUCER_VERSION,
            "runner_sha256": (
                sha256_file(runner_path) if runner_path.is_file() else None
            ),
            "runtime_sha256": sha256_file(runtime_path),
        },
        "simulator": {
            "name": "MuJoCo",
            "version": module.__version__,
            "mjcf_sha256": sha256_file(mjcf_path),
            "config_sha256": sha256_file(config_path),
            "approved_offline_inputs": approved_offline_inputs,
            "host": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": platform.python_version(),
            },
            "compiled_model": _compiled_model_contract(module, model),
            "physics_contract": physics_contract,
            "asset_provenance": _asset_provenance(),
        },
        "contract": _contract_descriptor(),
        "reference_command": reference.descriptor(),
        "trace_manifest_sha256": sha256_bytes(
            canonical_json_bytes(trace_manifest)
        ),
        "runs": runs,
        "summary": {
            "computed_pass": computed_pass,
            "promotion_eligible": promotion_eligible,
            "thresholds": dict(config["promotion_thresholds"]),
            "metrics": summary_metrics,
        },
    }
    _atomic_write(output_path, canonical_json_bytes(report))
    return report
