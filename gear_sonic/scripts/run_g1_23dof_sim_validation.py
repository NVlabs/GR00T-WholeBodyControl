"""Hash-bound IsaacLab disturbance validation for the true G1 23-DoF policy.

This runner is intentionally separate from deployment.  It imports no Unitree
DDS code and never creates a robot command publisher.  The normal CLI launches
``eval_agent_trl.py`` so environment and policy construction stay identical to
the project's existing evaluation path.  Once that runtime is loaded,
``run_loaded_isaaclab_validation`` executes the fixed contract in
``config/sim_validation/g1_23dof_rev_1_0.json``.

Every scenario/seed run writes a raw, per-step/per-environment trace.  The
promotion validator re-opens those traces, verifies their hashes and
deterministic disturbance vectors, and recomputes every reported metric.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Protocol

from gear_sonic.utils.g1_23dof_artifact import (
    SIM_REPORT_KIND,
    SIM_TRACE_SCHEMA_VERSION,
    build_true23_policy_pair,
    canonical_json_bytes,
    inspect_true23_policy_state,
    load_strict_json,
    sha256_bytes,
    sha256_file,
    simulation_material_provenance,
    true23_reference_profile_from_config,
    validate_runtime_config_snapshot,
    validate_simulation_report,
    validate_training_checkpoint_records,
    validate_true23_policy_module,
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
    DEPLOYMENT_HISTORY_LENGTH,
    NATIVE_IL23_JOINT_NAMES,
    OBS_LAYOUT_PADDED_IL29,
    ROBOT_MODEL,
    SIM_VALIDATION_SCHEMA_VERSION,
    SOURCE_IL29_EXCLUDED_INDICES,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
    TOKEN_DIM,
    reference_profile_contract,
)

TRACE_KIND = "g1_23dof_raw_isaaclab_trace"
TRACE_SCHEMA_VERSION = SIM_TRACE_SCHEMA_VERSION
EXPECTED_ISAACLAB_VERSION = "2.3.2"
NON_PROMOTABLE_TEST_REPORT_KIND = "g1_23dof_non_promotable_test_sim_validation"
NON_PROMOTABLE_TEST_PRODUCER_KIND = "gear_sonic_non_promotable_test_validation"
OUTPUT_ENV_VAR = "GEAR_SONIC_TRUE23_SIM_REPORT"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _PACKAGE_ROOT.parent
_CONFIG_PATH = _PACKAGE_ROOT / "config/sim_validation/g1_23dof_rev_1_0.json"
_ROBOT_ASSET_PATH = _PACKAGE_ROOT / "data/robots/g1/g1_23dof_rev_1_0.urdf"
_ROBOT_CONFIG_PATH = _PACKAGE_ROOT / "envs/manager_env/robots/g1_23dof.py"
_RUNNER_PATH = Path(__file__).resolve()
_VERIFIED_ISAACLAB_RUNTIME_TOKEN = object()


class _EvidenceIdentity:
    __slots__ = ("producer_kind", "report_kind", "simulator_name")

    def __init__(self, *, report_kind: str, producer_kind: str, simulator_name: str | None):
        self.report_kind = report_kind
        self.producer_kind = producer_kind
        self.simulator_name = simulator_name


_OFFICIAL_EVIDENCE = _EvidenceIdentity(
    report_kind=SIM_REPORT_KIND,
    producer_kind="",
    simulator_name=None,
)
_NON_PROMOTABLE_TEST_EVIDENCE = _EvidenceIdentity(
    report_kind=NON_PROMOTABLE_TEST_REPORT_KIND,
    producer_kind=NON_PROMOTABLE_TEST_PRODUCER_KIND,
    simulator_name="SyntheticTestBackend",
)


class ValidationBackend(Protocol):
    """Narrow interface used by the deterministic evidence loop."""

    num_envs: int
    control_hz: int
    simulator_name: str
    simulator_version: str
    resolved_config: Mapping[str, Any]

    def start_run(self, seed: int) -> None:
        """Seed and reset all environments for one scenario/seed run."""

    def apply_velocity_delta(self, deltas: Sequence[Sequence[float]]) -> None:
        """Add world-frame [vx, vy, vz, wx, wy, wz] to each root velocity."""

    def step(self) -> Mapping[str, Sequence[Any]]:
        """Advance once and return one value per environment for each raw metric."""


def _validation_config() -> Mapping[str, Any]:
    config = load_strict_json(_CONFIG_PATH)
    if config.get("robot_model") != ROBOT_MODEL:
        raise ValueError("validation config robot_model is not true23")
    return config


def _require_supported_isaaclab_version(version: Any) -> str:
    if version != EXPECTED_ISAACLAB_VERSION:
        raise RuntimeError(
            "official true23 evidence requires Isaac Lab "
            f"{EXPECTED_ISAACLAB_VERSION}; got {version!r}"
        )
    return version


def _canonical_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        # Evidence remains strict JSON while the explicit nonfinite flag fails
        # promotion.  A large sentinel also prevents a misleading metric pass.
        return 1.0e30
    return round(result, 12)


def deterministic_disturbance_vector(
    *,
    seed: int,
    episode_index: int,
    scale: float,
    config: Mapping[str, Any],
) -> list[float]:
    """Generate a platform-stable in-envelope six-axis velocity delta."""
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
            f"{ROBOT_MODEL}:{seed}:{episode_index}:{axis}".encode("ascii")
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        value = (float(lower) + unit * (float(upper) - float(lower))) * scale
        result.append(round(value, 12))
    return result


def _zero_disturbances(episodes: int) -> list[list[float]]:
    return [[0.0] * 6 for _ in range(episodes)]


def _trace_filename(scenario: str, seed: int) -> str:
    return f"{scenario}-seed-{seed}.json"


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


def _require_backend_metric(
    telemetry: Mapping[str, Sequence[Any]],
    key: str,
    episodes: int,
) -> Sequence[Any]:
    value = telemetry.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"backend metric {key!r} must be a sequence")
    if len(value) != episodes:
        raise ValueError(
            f"backend metric {key!r} has {len(value)} values; expected {episodes}"
        )
    return value


def _normalize_step_telemetry(
    telemetry: Mapping[str, Sequence[Any]],
    *,
    episodes: int,
    step: int,
    disturbance_delta: list[list[float]],
) -> dict[str, Any]:
    bool_keys = (
        "terminated",
        "timed_out",
        "nonfinite",
        "soft_limit_violation",
    )
    float_keys = (
        "phantom_observation_max_abs",
        "recovery_metric",
        "mpjpe_m",
    )
    int_keys = ("action_saturated_count", "action_count")
    record: dict[str, Any] = {
        "step": step,
        "disturbance_delta": disturbance_delta,
    }
    for key in bool_keys:
        values = _require_backend_metric(telemetry, key, episodes)
        record[key] = [bool(value) for value in values]
    for key in float_keys:
        values = _require_backend_metric(telemetry, key, episodes)
        record[key] = [
            _canonical_float(value, f"telemetry.{key}[{index}]")
            for index, value in enumerate(values)
        ]
    for key in int_keys:
        values = _require_backend_metric(telemetry, key, episodes)
        normalized: list[int] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"telemetry.{key}[{index}] must be integer >= 0")
            normalized.append(value)
        record[key] = normalized
    return record


def recompute_trace_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    episodes: int,
    disturbance_scale: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute promotion metrics from normalized raw trace records."""
    schedule = config["disturbance_schedule"]
    apply_step = int(schedule["apply_step"])
    baseline_steps = int(schedule["recovery_baseline_steps"])
    stable_steps = int(schedule["recovery_stable_steps"])
    recovery_margin = float(schedule["recovery_margin"])
    control_hz = int(config["control_hz"])

    termination_count = 0
    nonfinite_count = 0
    soft_limit_violation_count = 0
    phantom_max = 0.0
    saturated_total = 0
    action_total = 0
    mpjpe_total = 0.0
    mpjpe_count = 0
    recovery_by_episode = [0.0] * episodes

    for record in records:
        termination_count += sum(
            bool(terminated) or bool(timed_out)
            for terminated, timed_out in zip(
                record["terminated"],
                record["timed_out"],
                strict=True,
            )
        )
        nonfinite_count += sum(bool(value) for value in record["nonfinite"])
        soft_limit_violation_count += sum(
            bool(value) for value in record["soft_limit_violation"]
        )
        phantom_max = max(
            phantom_max,
            max(float(value) for value in record["phantom_observation_max_abs"]),
        )
        saturated_total += sum(int(value) for value in record["action_saturated_count"])
        action_total += sum(int(value) for value in record["action_count"])
        mpjpe_total += sum(float(value) for value in record["mpjpe_m"])
        mpjpe_count += episodes

    if disturbance_scale > 0.0:
        for episode in range(episodes):
            baseline_values = [
                float(records[step]["recovery_metric"][episode])
                for step in range(apply_step - baseline_steps, apply_step)
            ]
            threshold = sum(baseline_values) / len(baseline_values) + recovery_margin
            consecutive = 0
            recovered_at: int | None = None
            for step in range(apply_step, len(records)):
                if float(records[step]["recovery_metric"][episode]) <= threshold:
                    consecutive += 1
                    if consecutive >= stable_steps:
                        recovered_at = step - stable_steps + 1
                        break
                else:
                    consecutive = 0
            if recovered_at is None:
                recovery_by_episode[episode] = len(records) / control_hz
            else:
                recovery_by_episode[episode] = max(
                    0.0,
                    (recovered_at - apply_step) / control_hz,
                )

    return {
        "termination_count": termination_count,
        "nonfinite_count": nonfinite_count,
        "soft_limit_violation_count": soft_limit_violation_count,
        "phantom_observation_max_abs": round(phantom_max, 12),
        "max_recovery_time_s": round(max(recovery_by_episode, default=0.0), 12),
        "action_saturation_fraction": round(
            saturated_total / action_total if action_total else 1.0,
            12,
        ),
        "mpjpe_m": round(
            mpjpe_total / mpjpe_count if mpjpe_count else 1.0e30,
            12,
        ),
    }


def _write_validation_report(
    backend: ValidationBackend,
    *,
    checkpoint_path: Path,
    output_path: Path,
    config: Mapping[str, Any],
    identity: _EvidenceIdentity,
) -> Mapping[str, Any]:
    """Write official evidence or an explicitly non-promotable synthetic fixture."""
    if identity is _OFFICIAL_EVIDENCE:
        _require_verified_isaaclab_backend(backend)
    elif identity is not _NON_PROMOTABLE_TEST_EVIDENCE:
        raise TypeError("unsupported simulation evidence identity")
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    trace_dir = output_path.with_name(f"{output_path.stem}.traces")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if output_path.exists() or trace_dir.exists():
        raise FileExistsError("refusing to overwrite simulation report or trace directory")

    coverage = config["minimum_coverage"]
    seeds = tuple(int(seed) for seed in config["deterministic_seeds"])
    episodes = int(coverage["episodes_per_seed"])
    steps_per_episode = int(coverage["steps_per_episode"])
    if backend.num_envs != episodes:
        raise ValueError(
            f"validation requires exactly {episodes} parallel environments; "
            f"runtime has {backend.num_envs}"
        )
    if backend.control_hz != int(config["control_hz"]):
        raise ValueError(
            f"validation requires {config['control_hz']}Hz; runtime has "
            f"{backend.control_hz}Hz"
        )
    if identity is _OFFICIAL_EVIDENCE and backend.simulator_name != "IsaacLab":
        raise ValueError("validation backend must identify as IsaacLab")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    runner_sha256 = sha256_file(_RUNNER_PATH)
    runtime_hashes = validate_runtime_config_snapshot(
        backend.resolved_config,
        validation_config=config,
    )
    reference_profile = true23_reference_profile_from_config(
        backend.resolved_config
    )
    if identity is _OFFICIAL_EVIDENCE:
        checkpoint = load_safe_true23_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
        training_evidence = checkpoint["g1_23dof_training_evidence"]
        material_provenance = simulation_material_provenance(
            backend.resolved_config,
            checkpoint_motion_dataset=training_evidence["motion_dataset"],
            validation_config=config,
            repo_root=_REPOSITORY_ROOT,
        )
    else:
        material_provenance = {
            "schema_version": 1,
            "runtime_source": None,
            "motion_dataset": None,
        }
    producer = {
        "kind": (
            config["producer"]["kind"]
            if identity is _OFFICIAL_EVIDENCE
            else identity.producer_kind
        ),
        "version": (
            config["producer"]["version"] if identity is _OFFICIAL_EVIDENCE else 0
        ),
        "runner_sha256": runner_sha256,
    }
    simulator_name = (
        backend.simulator_name
        if identity is _OFFICIAL_EVIDENCE
        else identity.simulator_name
    )
    simulator = {
        "name": simulator_name,
        "version": backend.simulator_version,
        "asset_sha256": sha256_file(_ROBOT_ASSET_PATH),
        "robot_config_sha256": sha256_file(_ROBOT_CONFIG_PATH),
        "config_sha256": sha256_file(_CONFIG_PATH),
        "runtime_config_sha256": runtime_hashes["resolved_config_sha256"],
    }

    temporary_parent = Path(
        tempfile.mkdtemp(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.validation.",
        )
    )
    temporary_traces = temporary_parent / trace_dir.name
    temporary_traces.mkdir(parents=True)
    runs: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    try:
        for scenario, scenario_config in config["scenarios"].items():
            scale = float(scenario_config["disturbance_scale"])
            for seed in seeds:
                backend.start_run(seed)
                records: list[dict[str, Any]] = []
                for step in range(steps_per_episode):
                    if step == int(config["disturbance_schedule"]["apply_step"]):
                        deltas = [
                            deterministic_disturbance_vector(
                                seed=seed,
                                episode_index=episode,
                                scale=scale,
                                config=config,
                            )
                            for episode in range(episodes)
                        ]
                        backend.apply_velocity_delta(deltas)
                    else:
                        deltas = _zero_disturbances(episodes)
                    records.append(
                        _normalize_step_telemetry(
                            backend.step(),
                            episodes=episodes,
                            step=step,
                            disturbance_delta=deltas,
                        )
                    )

                trace = {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "kind": TRACE_KIND,
                    "checkpoint_sha256": checkpoint_sha256,
                    "producer": producer,
                    "simulator": simulator,
                    "material_provenance": material_provenance,
                    "scenario": scenario,
                    "seed": seed,
                    "disturbance_scale": scale,
                    "control_hz": backend.control_hz,
                    "episodes": episodes,
                    "steps_per_episode": steps_per_episode,
                    "records": records,
                }
                trace_bytes = canonical_json_bytes(trace)
                trace_name = _trace_filename(scenario, seed)
                trace_path = temporary_traces / trace_name
                _atomic_write(trace_path, trace_bytes)
                trace_record = {
                    "file": f"{trace_dir.name}/{trace_name}",
                    "sha256": sha256_bytes(trace_bytes),
                    "payload_sha256": sha256_bytes(canonical_json_bytes(trace)),
                    "record_count": len(records),
                }
                metrics = recompute_trace_metrics(
                    records,
                    episodes=episodes,
                    disturbance_scale=scale,
                    config=config,
                )
                runs.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "episodes": episodes,
                        "steps": episodes * steps_per_episode,
                        "disturbance_scale": scale,
                        **metrics,
                        "trace": trace_record,
                    }
                )
                manifest.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        **trace_record,
                    }
                )

        report = {
            "schema_version": SIM_VALIDATION_SCHEMA_VERSION,
            "kind": identity.report_kind,
            "robot_model": ROBOT_MODEL,
            "checkpoint_sha256": checkpoint_sha256,
            "producer": producer,
            "reference_profile": reference_profile,
            "reference_contract": reference_profile_contract(reference_profile),
            "observation_layout": OBS_LAYOUT_PADDED_IL29,
            "history_length": DEPLOYMENT_HISTORY_LENGTH,
            "decoder_input_dim": DEPLOYMENT_DECODER_INPUT_DIM,
            "decoder_output_dim": TARGET_DOF,
            "decoder_output_layout": DECODER_OUTPUT_LAYOUT,
            "runtime_config": {
                "resolved": backend.resolved_config,
                **runtime_hashes,
            },
            "simulator": simulator,
            "material_provenance": material_provenance,
            "trace_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
            "runs": runs,
        }
        report_bytes = canonical_json_bytes(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_traces, trace_dir)
        try:
            _atomic_write(output_path, report_bytes)
        except Exception:
            shutil.rmtree(trace_dir, ignore_errors=True)
            raise
        return report
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


class IsaacLabBackend:
    """Adapter over the already-loaded ``eval_agent_trl`` runtime."""

    simulator_name = "IsaacLab"

    def __init__(
        self,
        env: Any,
        model: Any,
        *,
        simulator_version: str,
        resolved_config: Mapping[str, Any],
    ):
        from isaaclab.assets import Articulation
        from isaaclab.envs import ManagerBasedRLEnv
        import torch

        from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand
        from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
        from gear_sonic.trl.trainer.ppo_trainer import PolicyAndValueWrapper

        if type(env) is not ManagerEnvWrapper:
            raise TypeError(
                "official evidence requires the exact gear_sonic ManagerEnvWrapper"
            )
        if type(env.env) is not ManagerBasedRLEnv:
            raise TypeError(
                "official evidence requires the exact IsaacLab ManagerBasedRLEnv"
            )
        if type(model) is not PolicyAndValueWrapper:
            raise TypeError(
                "official evidence requires the exact gear_sonic PolicyAndValueWrapper"
            )
        validate_true23_policy_module(
            model.policy,
            reference_profile=true23_reference_profile_from_config(
                resolved_config
            ),
        )

        self._torch = torch
        self.env = env
        self.model = model
        self.num_envs = int(env.num_envs)
        raw_env = env.env
        step_dt = float(getattr(raw_env, "step_dt", 0.0))
        if step_dt <= 0.0:
            raise ValueError("IsaacLab environment does not expose positive step_dt")
        reciprocal = 1.0 / step_dt
        self.control_hz = int(round(reciprocal))
        if not math.isclose(reciprocal, self.control_hz, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("IsaacLab control frequency is not integral")
        self.simulator_version = simulator_version
        self.resolved_config = resolved_config
        self.robot = raw_env.scene["robot"]
        if type(self.robot) is not Articulation:
            raise TypeError("official evidence requires the exact IsaacLab Articulation")
        if self.robot.num_joints != TARGET_DOF:
            raise ValueError("IsaacLab articulation must have exactly 23 joints")
        if tuple(self.robot.joint_names) != tuple(NATIVE_IL23_JOINT_NAMES):
            raise ValueError("IsaacLab joint order does not match native true23 contract")
        if raw_env.action_space.shape[-1] != TARGET_DOF:
            raise ValueError("IsaacLab action space must have exactly 23 outputs")
        if type(env.motion_command) is not TrackingCommand:
            raise TypeError(
                "official evidence requires the exact gear_sonic TrackingCommand"
            )
        robot_config = env.config.get("robot", {})
        if robot_config.get("type") != ROBOT_MODEL:
            raise ValueError("loaded environment is not g1_23dof_rev_1_0")
        self._clip = float(env.config.get("action_clip_value", 0.0))
        if self._clip <= 0.0:
            raise ValueError("true23 validation requires a positive action clip")
        self._obs: Mapping[str, Any] | None = None
        self._verified_runtime_token = _VERIFIED_ISAACLAB_RUNTIME_TOKEN

    def start_run(self, seed: int) -> None:
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_method = getattr(self.env.env, "seed", None)
        if callable(seed_method):
            seed_method(seed)
        self.env.set_is_evaluating(True, global_rank=0)
        self._obs = self.env.reset_all(global_rank=0)
        self.model.eval()
        self.model.policy.init_rollout()

    def apply_velocity_delta(self, deltas: Sequence[Sequence[float]]) -> None:
        torch = self._torch
        delta_tensor = torch.as_tensor(
            deltas,
            dtype=self.robot.data.root_vel_w.dtype,
            device=self.robot.data.root_vel_w.device,
        )
        if tuple(delta_tensor.shape) != (self.num_envs, 6):
            raise ValueError("velocity delta must have shape [num_envs, 6]")
        env_ids = torch.arange(self.num_envs, device=delta_tensor.device)
        velocity = self.robot.data.root_vel_w.clone()
        velocity += delta_tensor
        self.robot.write_root_velocity_to_sim(velocity, env_ids=env_ids)

    def _finite_by_env(self, value: Any) -> Any:
        torch = self._torch
        result = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.robot.data.joint_pos.device,
        )

        def visit(candidate: Any) -> None:
            nonlocal result
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
                if tensor.ndim > 0 and tensor.shape[0] == self.num_envs:
                    result &= torch.isfinite(tensor.reshape(self.num_envs, -1)).all(dim=1)
                else:
                    result &= torch.isfinite(tensor).all()
            elif isinstance(candidate, Mapping):
                for nested in candidate.values():
                    visit(nested)

        visit(value)
        return result

    def step(self) -> Mapping[str, Sequence[Any]]:
        torch = self._torch
        if self._obs is None:
            raise RuntimeError("start_run must be called before step")
        with torch.no_grad():
            actions = self.model.policy.rollout(obs_dict=self._obs)
            actor_state = {
                "actions": self.model.policy.action_mean.detach(),
                "obs_dict": actions["obs_dict"],
            }
            obs, _rewards, dones, extras = self.env.step(actor_state)
        self._obs = obs

        timeout = extras["time_outs"].bool().reshape(self.num_envs)
        terminated = dones.bool().reshape(self.num_envs) & ~timeout
        env_actions = extras["env_actions"].to(self.robot.data.joint_pos.device)
        if tuple(env_actions.shape) != (self.num_envs, TARGET_DOF):
            raise ValueError(
                f"true23 environment actions must be [{self.num_envs}, {TARGET_DOF}]"
            )
        saturated = (env_actions.abs() >= self._clip - 1.0e-6).sum(dim=1)

        limits = self.robot.data.soft_joint_pos_limits
        joint_pos = self.robot.data.joint_pos
        soft_limit = ((joint_pos < limits[..., 0]) | (joint_pos > limits[..., 1])).any(dim=1)

        from gear_sonic.envs.manager_env.mdp.observations import (
            g1_23dof_padded_joint_pos_rel,
            g1_23dof_padded_joint_vel_rel,
            g1_23dof_padded_last_action,
        )

        phantom_indices = list(SOURCE_IL29_EXCLUDED_INDICES)
        phantom_terms = (
            g1_23dof_padded_joint_pos_rel(self.env.env)[:, phantom_indices],
            g1_23dof_padded_joint_vel_rel(self.env.env)[:, phantom_indices],
            g1_23dof_padded_last_action(self.env.env)[:, phantom_indices],
        )
        phantom = torch.stack(
            [term.abs().max(dim=1).values for term in phantom_terms],
            dim=1,
        ).max(dim=1).values

        motion = self.env.motion_command
        current_body = motion.robot_body_pos_w
        reference_body = motion.body_pos_relative_w
        mpjpe = torch.linalg.vector_norm(current_body - reference_body, dim=-1).mean(dim=1)

        current_lin = motion.robot_anchor_lin_vel_w
        current_ang = motion.robot_anchor_ang_vel_w
        reference_lin = motion.anchor_lin_vel_w
        reference_ang = motion.anchor_ang_vel_w
        recovery_metric = torch.sqrt(
            (current_lin - reference_lin).square().sum(dim=1)
            + 0.25 * (current_ang - reference_ang).square().sum(dim=1)
        )

        finite = self._finite_by_env(
            {
                "obs": obs,
                "actions": env_actions,
                "joint_pos": joint_pos,
                "joint_vel": self.robot.data.joint_vel,
                "body_pos": current_body,
                "reference_body_pos": reference_body,
                "recovery_metric": recovery_metric,
            }
        )

        def values(tensor: Any) -> list[Any]:
            return tensor.detach().cpu().tolist()

        return {
            "terminated": values(terminated),
            "timed_out": values(timeout),
            "nonfinite": values(~finite),
            "soft_limit_violation": values(soft_limit),
            "phantom_observation_max_abs": values(phantom),
            "recovery_metric": values(recovery_metric),
            "action_saturated_count": values(saturated),
            "action_count": [TARGET_DOF] * self.num_envs,
            "mpjpe_m": values(mpjpe),
        }


def _require_verified_isaaclab_backend(backend: ValidationBackend) -> None:
    if (
        type(backend) is not IsaacLabBackend
        or getattr(backend, "_verified_runtime_token", None)
        is not _VERIFIED_ISAACLAB_RUNTIME_TOKEN
    ):
        raise TypeError(
            "official simulation evidence requires the exact verified "
            "IsaacLabBackend; synthetic or duck-typed backends are non-promotable"
        )
    if backend.env.env.scene["robot"] is not backend.robot:
        raise TypeError("verified IsaacLab backend robot identity changed")
    if backend.num_envs != int(backend.env.num_envs):
        raise TypeError("verified IsaacLab backend environment identity changed")


def _validate_official_checkpoint_binding(
    backend: IsaacLabBackend,
    checkpoint_path: Path,
) -> None:
    checkpoint = load_safe_true23_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint_stage(checkpoint) != TRAINED_STAGE:
        raise ValueError("official simulation evidence requires a trained safe checkpoint")
    _, _, policy_state_sha256 = build_true23_policy_pair(checkpoint)
    live_policy_state_sha256 = inspect_true23_policy_state(
        {"policy_state_dict": backend.model.policy.state_dict()}
    )
    if live_policy_state_sha256 != policy_state_sha256:
        raise ValueError(
            "loaded IsaacLab policy weights do not match the checkpoint file "
            "bound to simulation evidence"
        )
    global_step = extract_global_step(checkpoint)
    validate_training_checkpoint_records(
        checkpoint,
        global_step=global_step,
        policy_state_sha256=policy_state_sha256,
    )
    checkpoint_profile = checkpoint["g1_23dof_metadata"]["reference_profile"]
    runtime_profile = true23_reference_profile_from_config(backend.resolved_config)
    if runtime_profile != checkpoint_profile:
        raise ValueError(
            "IsaacLab runtime reference profile does not match trained checkpoint"
        )


def generate_validation_report(
    backend: IsaacLabBackend,
    *,
    checkpoint_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Generate official evidence only from a verified, exact IsaacLab runtime."""
    _require_verified_isaaclab_backend(backend)
    _validate_official_checkpoint_binding(backend, checkpoint_path)
    return _write_validation_report(
        backend,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        config=_validation_config(),
        identity=_OFFICIAL_EVIDENCE,
    )


def _generate_non_promotable_test_report(
    backend: ValidationBackend,
    *,
    checkpoint_path: Path,
    output_path: Path,
    config: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Generate synthetic fixtures that the promotion validator must reject."""
    return _write_validation_report(
        backend,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        config=_validation_config() if config is None else config,
        identity=_NON_PROMOTABLE_TEST_EVIDENCE,
    )


def run_loaded_isaaclab_validation(
    *,
    env: Any,
    model: Any,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    output_path: Path,
    runtime_config: Any,
) -> Mapping[str, Any]:
    """Entry used by ``eval_agent_trl`` after constructing its normal runtime."""
    _, _, policy_state_sha256 = build_true23_policy_pair(checkpoint)
    live_policy_state_sha256 = inspect_true23_policy_state(
        {"policy_state_dict": model.policy.state_dict()}
    )
    if live_policy_state_sha256 != policy_state_sha256:
        raise ValueError(
            "loaded IsaacLab policy weights do not match the checkpoint bound "
            "to simulation evidence"
        )
    global_step = extract_global_step(checkpoint)
    validate_training_checkpoint_records(
        checkpoint,
        global_step=global_step,
        policy_state_sha256=policy_state_sha256,
    )
    try:
        import isaaclab

        simulator_version = _require_supported_isaaclab_version(
            getattr(isaaclab, "__version__", None)
        )
    except ImportError as exc:  # pragma: no cover - guarded by eval_agent_trl
        raise RuntimeError("IsaacLab is required for true23 validation") from exc
    from omegaconf import OmegaConf

    resolved_config = OmegaConf.to_container(runtime_config, resolve=True)
    if not isinstance(resolved_config, Mapping):
        raise ValueError("resolved Hydra runtime config must be an object")
    backend = IsaacLabBackend(
        env,
        model,
        simulator_version=simulator_version,
        resolved_config=resolved_config,
    )
    return generate_validation_report(
        backend,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )


def _dry_run_payload(
    checkpoint_path: Path,
    output_path: Path,
    *,
    reference_profile: str,
    checkpoint_output: Mapping[str, Any],
) -> Mapping[str, Any]:
    config = _validation_config()
    return {
        "dry_run": True,
        "will_launch_isaaclab": False,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "reference_profile": reference_profile,
        "reference_contract": reference_profile_contract(reference_profile),
        "output": str(output_path.resolve()),
        "runner_sha256": sha256_file(_RUNNER_PATH),
        "config_sha256": sha256_file(_CONFIG_PATH),
        "promotion_enabled": config["producer"]["promotion_enabled"],
        "scenarios": list(config["scenarios"]),
        "seeds": list(config["deterministic_seeds"]),
        "episodes_per_seed": config["minimum_coverage"]["episodes_per_seed"],
        "steps_per_episode": config["minimum_coverage"]["steps_per_episode"],
        "checkpoint_output": dict(checkpoint_output),
    }


def _checkpoint_output_dry_run(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Execute the trained CPU policy pair before any simulator is launched."""
    import torch

    encoder, decoder, policy_state_sha256 = build_true23_policy_pair(checkpoint)
    global_step = extract_global_step(checkpoint)
    validate_training_checkpoint_records(
        checkpoint,
        global_step=global_step,
        policy_state_sha256=policy_state_sha256,
    )
    teleop_input = torch.zeros(
        (1, TELEOP_ENCODER_INPUT_DIM),
        dtype=torch.float32,
    )
    decoder_input = torch.zeros(
        (1, DEPLOYMENT_DECODER_INPUT_DIM),
        dtype=torch.float32,
    )
    with torch.no_grad():
        token = encoder(teleop_input)
        if tuple(token.shape) != (1, TOKEN_DIM) or not torch.isfinite(token).all():
            raise ValueError(
                "pre-simulation encoder dry-run did not produce finite [1,64]"
            )
        decoder_input[:, :TOKEN_DIM] = token
        action = decoder(decoder_input)
    if (
        tuple(action.shape) != (1, TARGET_DOF)
        or action.dtype != torch.float32
        or not torch.isfinite(action).all()
    ):
        raise ValueError(
            "pre-simulation decoder dry-run did not produce finite float32 [1,23]"
        )

    token_values = token.detach().cpu().tolist()
    action_values = action.detach().cpu().tolist()
    return {
        "performed": True,
        "before_simulator_launch": True,
        "device": "cpu",
        "dtype": "float32",
        "encoder_input_shape": [1, TELEOP_ENCODER_INPUT_DIM],
        "token_shape": [1, TOKEN_DIM],
        "decoder_input_shape": [1, DEPLOYMENT_DECODER_INPUT_DIM],
        "action_shape": [1, TARGET_DOF],
        "policy_state_sha256": policy_state_sha256,
        "token_sha256": sha256_bytes(canonical_json_bytes(token_values)),
        "action_sha256": sha256_bytes(canonical_json_bytes(action_values)),
        "action_values": action_values[0],
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hash-bound IsaacLab validation for G1 true23."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/contract and print the exact run plan without IsaacLab.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Re-open an existing report and raw traces; do not launch IsaacLab.",
    )
    args = parser.parse_args(argv)
    if args.dry_run and args.validate_only:
        parser.error("--dry-run and --validate-only are mutually exclusive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = load_safe_true23_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint_stage(checkpoint) != TRAINED_STAGE:
        raise ValueError(
            "simulation validation requires the separate trained *.promotion.pt "
            "checkpoint, not initialization or trainer-resume state"
        )
    reference_profile = str(
        checkpoint["g1_23dof_metadata"]["reference_profile"]
    )
    if args.dry_run:
        checkpoint_output = _checkpoint_output_dry_run(checkpoint)
        print(
            json.dumps(
                _dry_run_payload(
                    checkpoint_path,
                    output_path,
                    reference_profile=reference_profile,
                    checkpoint_output=checkpoint_output,
                ),
                indent=2,
            )
        )
        return 0
    if args.validate_only:
        report_bytes = output_path.read_bytes()
        report = load_strict_json(output_path)
        validate_simulation_report(
            report,
            checkpoint_sha256=sha256_file(checkpoint_path),
            report_sha256=sha256_bytes(report_bytes),
            report_payload_sha256=sha256_bytes(canonical_json_bytes(report)),
            report_path=output_path,
            reference_profile=reference_profile,
            checkpoint_motion_dataset=checkpoint[
                "g1_23dof_training_evidence"
            ]["motion_dataset"],
        )
        print("true23 simulation evidence valid")
        return 0

    config = _validation_config()
    episodes = int(config["minimum_coverage"]["episodes_per_seed"])
    if output_path.exists() or output_path.with_name(f"{output_path.stem}.traces").exists():
        raise FileExistsError("refusing to overwrite simulation report or trace directory")
    _checkpoint_output_dry_run(checkpoint)
    env = os.environ.copy()
    env[OUTPUT_ENV_VAR] = str(output_path)
    command = [
        sys.executable,
        str(_PACKAGE_ROOT / "eval_agent_trl.py"),
        f"checkpoint={checkpoint_path.as_posix()}",
        f"+num_envs={episodes}",
        "+headless=true",
        "+run_eval_loop=false",
        "+eval_callbacks=[]",
        "+true23_sim_validation=true",
    ]
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY_ROOT,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"IsaacLab validation subprocess failed with exit code "
            f"{completed.returncode}"
        )
    if not output_path.is_file():
        raise RuntimeError("IsaacLab validation exited without producing a report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
