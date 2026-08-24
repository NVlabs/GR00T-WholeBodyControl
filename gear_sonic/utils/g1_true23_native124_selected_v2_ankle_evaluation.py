"""Deterministic DadDance evaluator for one hash-bound Stage-1 warm restart.

This module is deliberately narrower than training.  It constructs the exact
nominal selected-V2 ankle task, loads one actual RSL warm restart through the
same row-masked integration used by training, and calls the actor with
``stochastic_output=False``.  It never calls PPO ``act``/``update`` and exposes
no hardware, deployment, upload, or network surface.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
from types import MethodType
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.native124_selected_v2_ankle_task import (
    ACTUATOR_SATURATION_THRESHOLD_RATIO,
    EPISODE_STEPS,
    TARGET_SOFT_LIMIT_MARGIN_FRACTION,
    audit_native124_selected_v2_ankle_task_env_cfg,
)
from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
    CAUSAL_HISTORY_ANCHOR_INDEX,
    DAD_DANCE_FRAME_COUNT,
    DAD_DANCE_RELATIVE_PATH,
    DAD_DANCE_SHA256,
    validate_dad_dance_motion_file,
)
from gear_sonic.scripts.train_g1_true23_native124_selected_v2_ankle import (
    FIXED_GPU,
    FIXED_LEARNING_RATE,
    FIXED_SEED,
    REPO_ROOT,
    SMOKE_ITERATIONS,
    SMOKE_NUM_ENVS,
    Stage1LaunchPlan,
    resolve_rsl_runtime_binding,
    stage1_agent_config,
)
from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
    ACTION_DIM,
    ACTOR_STATE_SCHEMA,
    ANKLE_HARDWARE_ROWS,
    SELECTED_ACTOR_STATE_SHA256,
    AnkleRowConfig,
    load_selected_v2_ankle_adaptation,
    safe_tree_sha256,
    sha256_file,
    tensor_state_sha256,
)
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes
from gear_sonic.utils.g1_23dof_native124_21204_adapter import load_checkpoint21204_binding
from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import (
    load_composite_mjlab_contract,
)
from gear_sonic.utils.g1_true23_native124_selected_v2_causal_parity import (
    EXPECTED_DONE_Q9 as SELECTED_SOURCE_FAILURE_Q9,
    WARMUP_STEPS,
    WARMUP_TARGET_ATOL,
    prove_warmup_action_equivalence,
)

EVALUATION_KIND = "g1_true23_native124_selected_v2_ankle_warm_daddance_evaluation_v1"
EVALUATION_SCHEMA_VERSION = 1
DEVICE = "cuda:0"
MAX_POLICY_TRANSITIONS = 500
EXPECTED_TIMEOUT_POLICY_TRANSITION = EPISODE_STEPS - WARMUP_STEPS - 1
EXPECTED_TIMEOUT_Q9 = CAUSAL_HISTORY_ANCHOR_INDEX + EPISODE_STEPS - 1
EXPECTED_POLICY_TRANSITIONS_ATTEMPTED = EPISODE_STEPS - WARMUP_STEPS

_FULLY_FROZEN_ACTOR_KEYS = tuple(key for key in ACTOR_STATE_SCHEMA if key not in {"mlp.6.weight", "mlp.6.bias"})
_SAFETY_SCALAR_KEYS = (
    "max_abs_selected_raw_action",
    "max_abs_candidate_target_hardware_rad",
    "max_abs_plain_sonic_raw_native",
    "max_abs_safe_native_action",
    "max_abs_final_target_hardware_rad",
    "max_abs_encoder_bias_rad",
    "full_v2_projection_linf_rad",
    "ankle_v2_projection_linf_rad",
    "maximum_joint_velocity_ratio",
    "maximum_actuator_force_ratio",
    "base_height_m",
    "base_tilt_rad",
    "target_tracking_rmse_rad",
)
_SAFETY_COUNT_KEYS = (
    "selected_action_abs_gt_one_coordinate_count",
    "plain_sonic_action_abs_gt_one_coordinate_count",
    "safe_action_abs_gt_one_coordinate_count",
    "raw_clip_coordinate_count",
    "target_inner_margin_violation_count",
    "target_soft_limit_violation_count",
    "actuator_target_soft_limit_violation_count",
    "measured_soft_limit_violation_count",
    "joint_velocity_limit_violation_count",
    "actuator_force_penalty_coordinate_count",
    "actuator_force_hard_limit_coordinate_count",
)
_HARD_SAFETY_COUNT_KEYS = (
    "raw_clip_coordinate_count",
    "target_soft_limit_violation_count",
    "actuator_target_soft_limit_violation_count",
    "measured_soft_limit_violation_count",
    "joint_velocity_limit_violation_count",
    "actuator_force_hard_limit_coordinate_count",
)
_SOFT_SAFETY_COUNT_KEYS = (
    "target_inner_margin_violation_count",
    "actuator_force_penalty_coordinate_count",
)


def _require_lower_sha256(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def _existing_regular_file(path: Path, context: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{context} must not be a symlink: {path}")
    try:
        result = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{context} does not exist: {path}") from error
    if not result.is_file():
        raise ValueError(f"{context} must be a regular file: {result}")
    return result


def _evaluation_output_path(repository_root: Path, output: Path) -> Path:
    evidence_root = (repository_root / "artifacts" / "g1_true23").resolve(strict=True)
    candidate = output if output.is_absolute() else repository_root / output
    result = candidate.resolve(strict=False)
    try:
        result.relative_to(evidence_root)
    except ValueError as error:
        raise ValueError("evaluation output must stay under artifacts/g1_true23") from error
    if result.suffix.lower() != ".json":
        raise ValueError("evaluation output must end in .json")
    if not result.parent.is_dir():
        raise ValueError("evaluation output parent directory must already exist")
    if os.path.lexists(result):
        raise FileExistsError(f"refusing to overwrite evaluation report: {result}")
    return result


@dataclass(frozen=True)
class WarmEvaluationRequest:
    """Fully resolved, immutable Stage-1 warm evaluation request."""

    repository_root: Path
    warm_checkpoint: Path
    expected_warm_sha256: str
    output: Path

    def __post_init__(self) -> None:
        if any(not isinstance(value, Path) for value in (self.repository_root, self.warm_checkpoint, self.output)):
            raise TypeError("evaluation paths must be pathlib.Path values")
        _require_lower_sha256(self.expected_warm_sha256, "expected_warm_sha256")

    @property
    def root(self) -> Path:
        result = self.repository_root.expanduser().resolve(strict=True)
        if not result.is_dir():
            raise ValueError("repository_root must be a directory")
        return result

    @property
    def checkpoint(self) -> Path:
        candidate = self.warm_checkpoint.expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        result = _existing_regular_file(candidate, "Stage-1 warm checkpoint")
        if result.suffix.lower() != ".pt":
            raise ValueError("Stage-1 warm checkpoint must end in .pt")
        return result

    @property
    def output_path(self) -> Path:
        return _evaluation_output_path(self.root, self.output.expanduser())


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _evaluation_agent_config(root: Path, motion: Path, output: Path) -> dict[str, Any]:
    if root != REPO_ROOT.resolve():
        # Launcher constants and manifests are repository-relative.  A copied
        # tree must import its own module rather than point this process at it.
        raise ValueError("evaluation repository root must match imported Stage-1 launcher root")
    plan = Stage1LaunchPlan(
        mode="smoke",
        motion_file=motion,
        run_dir=output.parent / ".evaluation_runner_unused",
        num_envs=SMOKE_NUM_ENVS,
        iterations=SMOKE_ITERATIONS,
    )
    return stage1_agent_config(plan)


def _bind_evaluation_runtime_sources(root: Path) -> dict[str, str]:
    """Bind simulator/task imports to this repository's source trees."""

    unitree_root = (root / "external_dependencies" / "unitree_rl_mjlab").resolve(strict=True)
    mjlab_root = (root / "external_dependencies" / "mjlab").resolve(strict=True)
    unitree_init = (unitree_root / "src" / "__init__.py").resolve(strict=True)
    unitree_text = str(unitree_root)
    if unitree_text not in sys.path:
        sys.path.insert(0, unitree_text)
    existing_src = sys.modules.get("src")
    if existing_src is not None:
        existing_file = getattr(existing_src, "__file__", None)
        if existing_file is None or Path(existing_file).resolve() != unitree_init:
            raise RuntimeError("unbound src package was imported before warm evaluation")
    src_module = importlib.import_module("src")
    if getattr(src_module, "__file__", None) is None or Path(src_module.__file__).resolve() != unitree_init:
        raise RuntimeError("warm evaluation task source binding mismatch")

    mjlab_module = importlib.import_module("mjlab")
    expected_mjlab = (mjlab_root / "src" / "mjlab" / "__init__.py").resolve(strict=True)
    if Path(mjlab_module.__file__).resolve() != expected_mjlab:
        raise RuntimeError("warm evaluation MJLab source binding mismatch")
    return {
        "unitree_task_init": str(unitree_init),
        "mjlab_init": str(expected_mjlab),
    }


def preflight_warm_evaluation(request: WarmEvaluationRequest) -> dict[str, Any]:
    """Hash all immutable inputs without constructing or stepping a simulator."""

    if type(request) is not WarmEvaluationRequest:
        raise TypeError("request must be exact WarmEvaluationRequest")
    root = request.root
    checkpoint = request.checkpoint
    output = request.output_path
    actual_warm_hash = sha256_file(checkpoint)
    if actual_warm_hash != request.expected_warm_sha256:
        raise ValueError(
            "Stage-1 warm checkpoint SHA-256 mismatch: "
            f"expected {request.expected_warm_sha256}, got {actual_warm_hash}"
        )
    motion = validate_dad_dance_motion_file(root / DAD_DANCE_RELATIVE_PATH)
    if sha256_file(motion) != DAD_DANCE_SHA256:
        raise RuntimeError("DadDance changed after validation")
    binding = load_checkpoint21204_binding(root)
    agent_cfg = _evaluation_agent_config(root, motion, output)
    rsl_runtime = resolve_rsl_runtime_binding()
    return {
        "schema": "g1_true23_native124_selected_v2_ankle_warm_evaluation_preflight_v1",
        "ready": True,
        "warm_checkpoint": {
            "path": str(checkpoint),
            "sha256": actual_warm_hash,
        },
        "selected_source": {
            "checkpoint_path": str(binding.checkpoint_path),
            "checkpoint_sha256": binding.checkpoint_sha256,
            "actor_state_sha256": binding.actor_state_sha256,
            "onnx_sha256": binding.onnx_sha256,
        },
        "motion": {
            "path": str(motion),
            "sha256": DAD_DANCE_SHA256,
            "frame_count": DAD_DANCE_FRAME_COUNT,
        },
        "output": str(output),
        "runner_agent_config_sha256": _canonical_sha256(agent_cfg),
        "rsl_runtime": rsl_runtime,
        "fixed": {
            "seed": FIXED_SEED,
            "device": DEVICE,
            "num_envs": 1,
            "warmup_steps": WARMUP_STEPS,
            "max_policy_transitions": MAX_POLICY_TRANSITIONS,
            "stochastic_output": False,
            "runner_profile": "smoke_for_inference_only",
            "warm_contract_profile_independent": True,
        },
        "safety": {
            "simulator_constructed": False,
            "simulator_steps": 0,
            "training_updates": 0,
            "hardware_authorized": False,
            "network_used": False,
            "deployment_authorized": False,
        },
    }


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes(order="C")


def compare_actor_to_selected_source(
    adapted_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    *,
    trainable_rows: tuple[int, ...] = ANKLE_HARDWARE_ROWS,
) -> dict[str, Any]:
    """Prove frozen tensors/rows are exact and quantify permitted row deltas."""

    expected_keys = set(ACTOR_STATE_SCHEMA)
    if set(adapted_state) != expected_keys or set(source_state) != expected_keys:
        raise ValueError("actor state keys do not match exact selected actor schema")
    if (
        type(trainable_rows) is not tuple
        or not trainable_rows
        or any(type(row) is not int or not 0 <= row < ACTION_DIM for row in trainable_rows)
        or len(set(trainable_rows)) != len(trainable_rows)
    ):
        raise ValueError("trainable_rows must be unique actor output rows")
    for key, spec in ACTOR_STATE_SCHEMA.items():
        for label, state in (("adapted", adapted_state), ("source", source_state)):
            value = state[key]
            if (
                type(value) is not torch.Tensor
                or tuple(value.shape) != spec.shape
                or value.dtype != spec.dtype
                or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
            ):
                raise ValueError(f"{label} actor tensor contract mismatch: {key}")
    for key in _FULLY_FROZEN_ACTOR_KEYS:
        if _tensor_bytes(adapted_state[key]) != _tensor_bytes(source_state[key]):
            raise ValueError(f"adapted warm checkpoint changed frozen actor tensor: {key}")

    frozen_rows = tuple(row for row in range(ACTION_DIM) if row not in trainable_rows)
    frozen_index = torch.tensor(frozen_rows, dtype=torch.long)
    trainable_index = torch.tensor(trainable_rows, dtype=torch.long)
    result: dict[str, Any] = {}
    total_changed = 0
    total_elements = 0
    total_squared_delta = 0.0
    total_max = 0.0
    for key in ("mlp.6.weight", "mlp.6.bias"):
        adapted = adapted_state[key].detach().cpu()
        source = source_state[key].detach().cpu()
        if not torch.equal(
            adapted.index_select(0, frozen_index),
            source.index_select(0, frozen_index),
        ):
            raise ValueError(f"adapted warm checkpoint changed frozen actor rows: {key}")
        delta = adapted.index_select(0, trainable_index) - source.index_select(0, trainable_index)
        changed = int(torch.count_nonzero(delta).item())
        squared = float(torch.sum(torch.square(delta.to(torch.float64))).item())
        maximum = float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0
        result[key] = {
            "changed_element_count": changed,
            "element_count": int(delta.numel()),
            "l2_delta": math.sqrt(squared),
            "maximum_absolute_delta": maximum,
        }
        total_changed += changed
        total_elements += int(delta.numel())
        total_squared_delta += squared
        total_max = max(total_max, maximum)

    return {
        "adapted_actor_state_sha256": tensor_state_sha256(adapted_state),
        "source_actor_state_sha256": tensor_state_sha256(source_state),
        "source_identity_matches_selected": tensor_state_sha256(source_state) == SELECTED_ACTOR_STATE_SHA256,
        "trainable_hardware_rows": list(trainable_rows),
        "frozen_hardware_rows": list(frozen_rows),
        "fully_frozen_tensor_keys": list(_FULLY_FROZEN_ACTOR_KEYS),
        "frozen_state_byte_exact": True,
        "trainable_rows_changed": total_changed > 0,
        "trainable_changed_element_count": total_changed,
        "trainable_element_count": total_elements,
        "trainable_l2_delta": math.sqrt(total_squared_delta),
        "trainable_maximum_absolute_delta": total_max,
        "by_tensor": result,
    }


@dataclass(frozen=True)
class StepEvidence:
    """Finite scalar evidence captured for one executed simulator transition."""

    reward: float
    scalars: Mapping[str, float]
    counts: Mapping[str, int]
    reward_rates: Mapping[str, float]

    def __post_init__(self) -> None:
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise ValueError("step reward must be finite float")
        if set(self.scalars) != set(_SAFETY_SCALAR_KEYS):
            raise ValueError("step safety scalar schema mismatch")
        if set(self.counts) != set(_SAFETY_COUNT_KEYS):
            raise ValueError("step safety count schema mismatch")
        if not self.reward_rates or any(type(key) is not str or not key for key in self.reward_rates):
            raise ValueError("step reward-rate mapping must be nonempty")
        for key, value in self.scalars.items():
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"step safety scalar is nonfinite: {key}")
        for key, value in self.counts.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"step safety count is invalid: {key}")
        for key, value in self.reward_rates.items():
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"step reward rate is nonfinite: {key}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "scalars": dict(self.scalars),
            "counts": dict(self.counts),
            "reward_rates": dict(self.reward_rates),
        }


class _ScalarStatistics:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value: float) -> None:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("statistics value must be finite float")
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def report(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "sum": 0.0, "mean": None, "minimum": None, "maximum": None}
        return {
            "count": self.count,
            "sum": self.total,
            "mean": self.total / self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


class RolloutEvidenceAccumulator:
    """Compact online aggregate; stores no full trajectory."""

    def __init__(self) -> None:
        self.step_count = 0
        self.reward = _ScalarStatistics()
        self.scalars = {key: _ScalarStatistics() for key in _SAFETY_SCALAR_KEYS}
        self.count_totals = {key: 0 for key in _SAFETY_COUNT_KEYS}
        self.count_step_maxima = {key: 0 for key in _SAFETY_COUNT_KEYS}
        self.reward_rates: dict[str, _ScalarStatistics] | None = None

    def add(self, evidence: StepEvidence) -> None:
        if type(evidence) is not StepEvidence:
            raise TypeError("evidence must be exact StepEvidence")
        names = set(evidence.reward_rates)
        if self.reward_rates is None:
            self.reward_rates = {key: _ScalarStatistics() for key in sorted(names)}
        elif names != set(self.reward_rates):
            raise ValueError("reward term set changed during evaluation")
        self.step_count += 1
        self.reward.add(evidence.reward)
        for key, value in evidence.scalars.items():
            self.scalars[key].add(value)
        for key, value in evidence.counts.items():
            self.count_totals[key] += value
            self.count_step_maxima[key] = max(self.count_step_maxima[key], value)
        for key, value in evidence.reward_rates.items():
            self.reward_rates[key].add(value)

    def report(self) -> dict[str, Any]:
        rates = (
            {} if self.reward_rates is None else {key: value.report() for key, value in self.reward_rates.items()}
        )
        hard_total = sum(self.count_totals[key] for key in _HARD_SAFETY_COUNT_KEYS)
        soft_total = sum(self.count_totals[key] for key in _SOFT_SAFETY_COUNT_KEYS)
        return {
            "transition_count": self.step_count,
            "all_values_finite": True,
            "reward": self.reward.report(),
            "safety_scalars": {key: value.report() for key, value in self.scalars.items()},
            "safety_count_totals": dict(self.count_totals),
            "safety_count_maximum_per_transition": dict(self.count_step_maxima),
            "hard_safety_violation_count": hard_total,
            "soft_safety_warning_count": soft_total,
            "reward_rate_by_term": rates,
        }


def _single_numpy_tensor(
    value: Any,
    shape: tuple[int, ...],
    context: str,
    *,
    floating: bool = True,
) -> np.ndarray:
    if type(value) is not torch.Tensor or tuple(value.shape) != shape:
        raise ValueError(f"{context} must be tensor with shape {shape}")
    if floating and (not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{context} must be finite floating tensor")
    return value.detach().to(device="cpu").contiguous().numpy().copy()


def _reward_rates(raw_env: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, values in raw_env.reward_manager.get_active_iterable_terms(0):
        if type(name) is not str or len(values) != 1:
            raise ValueError("reward manager iterable term contract drift")
        value = float(values[0])
        if not math.isfinite(value):
            raise ValueError(f"reward rate is nonfinite: {name}")
        result[name] = value
    required = {
        "motion_ankle_pos",
        "motion_ankle_ori",
        "motion_global_root_pos",
        "motion_global_root_ori",
        "joint_torques_l2",
        "actuator_saturation",
        "action_target_soft_limit_barrier",
        "joint_limit",
        "v2_ankle_projection_l2",
        "feet_slip",
        "soft_landing",
        "alive",
        "non_timeout_termination",
    }
    if not required.issubset(result):
        raise ValueError(f"Stage-1 reward terms missing: {sorted(required - set(result))}")
    return result


def _require_finite_runtime_tensor(value: Any, context: str) -> None:
    if type(value) is not torch.Tensor or value.numel() == 0:
        raise ValueError(f"{context} must be a nonempty tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{context} contains NaN or Inf")


def _validate_task_runtime_inputs(raw_env: Any) -> None:
    """Catch nonfinite task inputs that RewardManager would otherwise zero."""

    command = raw_env.command_manager.get_term("motion")
    for name in (
        "anchor_pos_w",
        "anchor_quat_w",
        "robot_anchor_pos_w",
        "robot_anchor_quat_w",
        "body_pos_relative_w",
        "body_quat_relative_w",
        "robot_body_pos_w",
        "robot_body_quat_w",
    ):
        _require_finite_runtime_tensor(getattr(command, name, None), f"motion command {name}")
    robot = raw_env.scene["robot"]
    _require_finite_runtime_tensor(robot.data.site_lin_vel_w, "robot site linear velocity")
    contact = raw_env.scene["feet_ground_contact"].data
    for name in (
        "found",
        "force",
        "current_air_time",
        "last_air_time",
        "current_contact_time",
        "last_contact_time",
        "force_history",
    ):
        _require_finite_runtime_tensor(getattr(contact, name, None), f"foot contact {name}")


def _step_evidence(raw_env: Any, velocity_limits: np.ndarray) -> StepEvidence:
    """Capture current transition state before any terminal autoreset."""

    if (
        velocity_limits.shape != (ACTION_DIM,)
        or not np.isfinite(velocity_limits).all()
        or np.any(velocity_limits <= 0.0)
    ):
        raise ValueError("velocity limits must be finite positive [23]")
    _validate_task_runtime_inputs(raw_env)
    action = raw_env.action_manager.get_term("joint_pos")
    robot = raw_env.scene["robot"]
    selected_raw = _single_numpy_tensor(action.raw_action, (1, ACTION_DIM), "selected raw action")[0]
    candidate = _single_numpy_tensor(
        action.candidate_target_hardware,
        (1, ACTION_DIM),
        "candidate target",
    )[0]
    plain = _single_numpy_tensor(
        action.plain_sonic_raw_action_native,
        (1, ACTION_DIM),
        "plain SONIC action",
    )[0]
    safe = _single_numpy_tensor(action.safe_native_action, (1, ACTION_DIM), "safe native action")[0]
    target = _single_numpy_tensor(action.processed_action, (1, ACTION_DIM), "final target")[0]
    clip_mask = _single_numpy_tensor(
        action.raw_clip_mask_native,
        (1, ACTION_DIM),
        "raw clip mask",
        floating=False,
    )[0]
    if clip_mask.dtype != np.bool_:
        raise ValueError("raw clip mask must be boolean")

    measured = _single_numpy_tensor(robot.data.joint_pos, (1, ACTION_DIM), "measured position")[0]
    velocity = _single_numpy_tensor(robot.data.joint_vel, (1, ACTION_DIM), "measured velocity")[0]
    encoder_bias = _single_numpy_tensor(robot.data.encoder_bias, (1, ACTION_DIM), "encoder bias")[0]
    limits = _single_numpy_tensor(
        robot.data.soft_joint_pos_limits,
        (1, ACTION_DIM, 2),
        "soft joint limits",
    )[0]
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("soft joint limits are unordered")
    actuator_target = target - encoder_bias
    span = limits[:, 1] - limits[:, 0]
    inner_lower = limits[:, 0] + TARGET_SOFT_LIMIT_MARGIN_FRACTION * span
    inner_upper = limits[:, 1] - TARGET_SOFT_LIMIT_MARGIN_FRACTION * span

    force = _single_numpy_tensor(robot.data.actuator_force, (1, ACTION_DIM), "actuator force")[0]
    ctrl_ids_value = robot.data.indexing.ctrl_ids
    if type(ctrl_ids_value) is torch.Tensor:
        ctrl_ids = _single_numpy_tensor(
            ctrl_ids_value,
            (ACTION_DIM,),
            "actuator control indices",
            floating=False,
        ).astype(np.int64, copy=False)
    else:
        ctrl_ids = np.asarray(ctrl_ids_value, dtype=np.int64)
        if ctrl_ids.shape != (ACTION_DIM,):
            raise ValueError("actuator control indices must be [23]")
    from mjlab.sim import TorchArray

    force_range = raw_env.sim.model.actuator_forcerange
    if (
        type(force_range) is not TorchArray
        or tuple(force_range.shape) != (1, ACTION_DIM, 2)
        or force_range.ndim != 3
        or force_range.dtype != torch.float32
        or force_range.device != robot.data.actuator_force.device
    ):
        raise ValueError("actuator force range must be exact MJLab TorchArray float32 [1,23,2] on robot device")
    if not np.array_equal(ctrl_ids, np.arange(ACTION_DIM, dtype=np.int64)):
        raise ValueError("compact hardware-23 actuator control indices must be exact 0..22")
    ctrl_index = robot.data.indexing.ctrl_ids.to(dtype=torch.long)
    force_limit_tensor = torch.abs(force_range[:, ctrl_index, 1])
    force_limits = _single_numpy_tensor(
        force_limit_tensor,
        (1, ACTION_DIM),
        "actuator force limits",
    )[0]
    if not np.isfinite(force_limits).all() or np.any(force_limits <= 0.0):
        raise ValueError("actuator force limits must be finite positive")
    force_ratio = np.abs(force) / force_limits

    root_position = _single_numpy_tensor(robot.data.root_link_pos_w, (1, 3), "root position")[0]
    projected_gravity = _single_numpy_tensor(
        robot.data.projected_gravity_b,
        (1, 3),
        "projected gravity",
    )[0]
    gravity_z = float(np.clip(projected_gravity[2], -1.0, 1.0))
    reward_value = float(_single_numpy_tensor(raw_env.reward_buf, (1,), "reward")[0])
    velocity_ratio = np.abs(velocity) / velocity_limits
    ankle_indices = np.asarray(ANKLE_HARDWARE_ROWS, dtype=np.int64)
    projection = candidate - target
    counts = {
        "selected_action_abs_gt_one_coordinate_count": int(np.count_nonzero(np.abs(selected_raw) > 1.0)),
        "plain_sonic_action_abs_gt_one_coordinate_count": int(np.count_nonzero(np.abs(plain) > 1.0)),
        "safe_action_abs_gt_one_coordinate_count": int(np.count_nonzero(np.abs(safe) > 1.0)),
        "raw_clip_coordinate_count": int(np.count_nonzero(clip_mask)),
        "target_inner_margin_violation_count": int(
            np.count_nonzero((target < inner_lower) | (target > inner_upper))
        ),
        "target_soft_limit_violation_count": int(
            np.count_nonzero((target < limits[:, 0]) | (target > limits[:, 1]))
        ),
        "actuator_target_soft_limit_violation_count": int(
            np.count_nonzero((actuator_target < limits[:, 0]) | (actuator_target > limits[:, 1]))
        ),
        "measured_soft_limit_violation_count": int(
            np.count_nonzero((measured < limits[:, 0]) | (measured > limits[:, 1]))
        ),
        "joint_velocity_limit_violation_count": int(np.count_nonzero(velocity_ratio > 1.0)),
        "actuator_force_penalty_coordinate_count": int(
            np.count_nonzero(force_ratio > ACTUATOR_SATURATION_THRESHOLD_RATIO)
        ),
        "actuator_force_hard_limit_coordinate_count": int(np.count_nonzero(force_ratio > 1.0)),
    }
    scalars = {
        "max_abs_selected_raw_action": float(np.max(np.abs(selected_raw))),
        "max_abs_candidate_target_hardware_rad": float(np.max(np.abs(candidate))),
        "max_abs_plain_sonic_raw_native": float(np.max(np.abs(plain))),
        "max_abs_safe_native_action": float(np.max(np.abs(safe))),
        "max_abs_final_target_hardware_rad": float(np.max(np.abs(target))),
        "max_abs_encoder_bias_rad": float(np.max(np.abs(encoder_bias))),
        "full_v2_projection_linf_rad": float(np.max(np.abs(projection))),
        "ankle_v2_projection_linf_rad": float(np.max(np.abs(projection[ankle_indices]))),
        "maximum_joint_velocity_ratio": float(np.max(velocity_ratio)),
        "maximum_actuator_force_ratio": float(np.max(force_ratio)),
        "base_height_m": float(root_position[2]),
        "base_tilt_rad": math.acos(float(np.clip(-gravity_z, -1.0, 1.0))),
        "target_tracking_rmse_rad": float(
            np.sqrt(np.mean(np.square(target.astype(np.float64) - measured.astype(np.float64))))
        ),
    }
    return StepEvidence(
        reward=reward_value,
        scalars=scalars,
        counts=counts,
        reward_rates=_reward_rates(raw_env),
    )


def _q9(command: Any) -> int:
    value = getattr(command, "time_steps", None)
    if type(value) is not torch.Tensor or value.shape != (1,) or value.dtype != torch.long:
        raise ValueError("motion command q9 must be int64 [1]")
    return int(value.detach().cpu().item())


def _termination_names(raw_env: Any) -> list[str]:
    result = []
    for name, values in raw_env.termination_manager.get_active_iterable_terms(0):
        if len(values) != 1:
            raise ValueError("termination iterable term contract drift")
        value = float(values[0])
        if not math.isfinite(value) or value not in {0.0, 1.0}:
            raise ValueError(f"termination value is invalid: {name}")
        if value == 1.0:
            result.append(name)
    return sorted(result)


def _extra_termination_names(extras: Mapping[str, Any]) -> list[str]:
    log = extras.get("log")
    if not isinstance(log, Mapping):
        return []
    prefix = "Episode_Termination/"
    result = []
    for key, value in log.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        if hasattr(value, "detach"):
            numeric = float(value.detach().cpu().item())
        else:
            numeric = float(value)
        if numeric != 0.0:
            result.append(key[len(prefix) :])
    return sorted(result)


class _TerminalPreResetRecorder:
    """Observe terminal state immediately before MJLab autoreset mutates it."""

    def __init__(self, raw_env: Any, velocity_limits: np.ndarray) -> None:
        self.raw_env = raw_env
        self.velocity_limits = velocity_limits
        self._original_reset = raw_env._reset_idx
        self._armed: dict[str, Any] | None = None
        self.captured: dict[str, Any] | None = None

        def observed_reset(_env: Any, env_ids: torch.Tensor | None = None) -> None:
            if self._armed is not None and int(_env.common_step_counter) > 0:
                if self.captured is not None:
                    raise RuntimeError("terminal pre-reset recorder captured more than once")
                if env_ids is None or int(env_ids.numel()) != 1 or int(env_ids.detach().cpu().item()) != 0:
                    raise RuntimeError("terminal pre-reset recorder expected only environment zero")
                self.captured = {
                    **self._armed,
                    "episode_length_pre_reset": int(_env.episode_length_buf[0].detach().cpu().item()),
                    "termination_names": _termination_names(_env),
                    "is_timeout": bool(_env.termination_manager.time_outs[0].detach().cpu().item()),
                    "is_terminated": bool(_env.termination_manager.terminated[0].detach().cpu().item()),
                    "evidence": _step_evidence(_env, self.velocity_limits),
                }
            self._original_reset(env_ids)

        raw_env._reset_idx = MethodType(observed_reset, raw_env)

    def arm(self, *, phase: str, index: int, q9_before: int) -> None:
        if self._armed is not None:
            raise RuntimeError("terminal pre-reset recorder already armed")
        self._armed = {"phase": phase, "index": index, "q9_before": q9_before}
        self.captured = None

    def finish(self, *, done: bool) -> dict[str, Any] | None:
        if self._armed is None:
            raise RuntimeError("terminal pre-reset recorder was not armed")
        captured = self.captured
        self._armed = None
        self.captured = None
        if done != (captured is not None):
            raise RuntimeError("terminal pre-reset capture disagrees with wrapper done")
        return captured

    def restore(self) -> None:
        self.raw_env._reset_idx = self._original_reset


def _deterministic_actor_action(actor: Any, observations: Any) -> torch.Tensor:
    actor_observation = observations["actor"]
    if (
        type(actor_observation) is not torch.Tensor
        or actor_observation.shape != (1, 124)
        or actor_observation.dtype != torch.float32
        or not bool(torch.isfinite(actor_observation).all())
    ):
        raise ValueError("evaluation actor observation must be finite float32 [1,124]")
    critic_observation = observations["critic"]
    policy_observation = observations["policy"]
    for name, value, width in (
        ("critic", critic_observation, 256),
        ("policy", policy_observation, 930),
    ):
        if (
            type(value) is not torch.Tensor
            or value.shape != (1, width)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"evaluation {name} observation must be finite float32 [1,{width}]")

    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(actor_observation.device).clone()
    with torch.no_grad():
        first = actor(observations, stochastic_output=False)
        second = actor(observations, stochastic_output=False)
    if not torch.equal(first, second):
        raise RuntimeError("deterministic actor mean changed on identical observation")
    if not torch.equal(torch.random.get_rng_state(), cpu_rng) or not torch.equal(
        torch.cuda.get_rng_state(actor_observation.device), cuda_rng
    ):
        raise RuntimeError("deterministic actor mean consumed RNG state")
    if (
        type(first) is not torch.Tensor
        or first.shape != (1, ACTION_DIM)
        or first.dtype != torch.float32
        or not bool(torch.isfinite(first).all())
    ):
        raise ValueError("deterministic actor action must be finite float32 [1,23]")
    return first.detach().clone()


def configure_hash_locked_warm_evaluation_runner(
    runner: Any,
    *,
    repository_root: Path,
    warm_checkpoint: Path,
    expected_warm_sha256: str,
) -> Any:
    """No-sim-loadable boundary used by both evaluator and integration test."""

    expected = _require_lower_sha256(expected_warm_sha256, "expected_warm_sha256")
    checkpoint = _existing_regular_file(warm_checkpoint, "Stage-1 warm checkpoint")
    if sha256_file(checkpoint) != expected:
        raise ValueError("Stage-1 warm checkpoint changed before actual RSL configuration")
    from gear_sonic.trl.mjlab.native124_selected_v2_ankle_rsl import (
        configure_selected_v2_ankle_rsl_runner,
    )

    integration = configure_selected_v2_ankle_rsl_runner(
        runner,
        repository_root=repository_root,
        config=AnkleRowConfig(),
        restart_path=checkpoint,
        expected_restart_sha256=expected,
    )
    integration.assert_frozen_invariants()
    storage_step = getattr(getattr(runner.alg, "storage", None), "step", None)
    if (
        runner.current_learning_iteration != 0
        or integration.optimizer_steps != 0
        or storage_step != 0
        or runner.alg.optimizer.state
    ):
        raise RuntimeError("warm evaluation runner contains training state")
    return integration


def _json_safe(value: Any) -> Any:
    if isinstance(value, StepEvidence):
        return _json_safe(value.to_dict())
    if type(value) is torch.Tensor:
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("report contains nonfinite array")
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("report contains nonfinite float")
        return value
    raise TypeError(f"unsupported report value: {type(value).__qualname__}")


def _qualification(
    *,
    first_done: Mapping[str, Any] | None,
    policy_summary: Mapping[str, Any],
    actor_delta: Mapping[str, Any],
    attempted: int,
    deterministic_actor_checks: int,
) -> dict[str, Any]:
    full_timeout = bool(
        first_done is not None
        and first_done.get("policy_transition") == EXPECTED_TIMEOUT_POLICY_TRANSITION
        and first_done.get("q9_before") == EXPECTED_TIMEOUT_Q9
        and first_done.get("episode_length_pre_reset") == EPISODE_STEPS
        and first_done.get("termination_names") == ["time_out"]
        and first_done.get("is_timeout") is True
        and first_done.get("is_terminated") is False
        and attempted == EXPECTED_POLICY_TRANSITIONS_ATTEMPTED
    )
    hard_safe = policy_summary.get("hard_safety_violation_count") == 0
    soft_safe = policy_summary.get("soft_safety_warning_count") == 0
    adapted = actor_delta.get("trainable_rows_changed") is True
    frozen = actor_delta.get("frozen_state_byte_exact") is True
    deterministic = deterministic_actor_checks == attempted
    improvement = bool(first_done is None or int(first_done.get("q9_before", -1)) > SELECTED_SOURCE_FAILURE_Q9)
    gate = full_timeout and hard_safe and adapted and frozen and deterministic
    strict_gate = gate and soft_safe
    return {
        "candidate_gate_passed": gate,
        "strict_nominal_gate_passed": strict_gate,
        "full_500_step_episode_including_two_warmups": full_timeout,
        "hard_safety_gate_passed": hard_safe,
        "soft_margin_and_saturation_warning_gate_passed": soft_safe,
        "trainable_rows_differ_from_selected_source": adapted,
        "frozen_actor_state_byte_exact": frozen,
        "deterministic_actor_checks_passed": deterministic,
        "survived_beyond_selected_source_q9_58_failure": improvement,
        "task_space_reward_thresholds_predeclared": False,
        "scope": "DadDance Stage-1 survival/safety characterization only",
        "promotion_or_deployment_authorized": False,
    }


def run_warm_daddance_evaluation(request: WarmEvaluationRequest) -> dict[str, Any]:
    """Run one exact deterministic warm policy until first done or fixed cap."""

    preflight = preflight_warm_evaluation(request)
    root = request.root
    checkpoint = request.checkpoint
    output = request.output_path
    motion = (root / DAD_DANCE_RELATIVE_PATH).resolve(strict=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FIXED_GPU)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    runtime_sources = _bind_evaluation_runtime_sources(root)
    if resolve_rsl_runtime_binding() != preflight["rsl_runtime"]:
        raise RuntimeError("RSL runtime binding changed after warm evaluation preflight")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.utils.torch import configure_torch_backends

    from gear_sonic.envs.mjlab.native124_selected_v2_ankle_task import (
        make_native124_selected_v2_ankle_task_env_cfg,
    )
    from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
        Native124SelectedV2CausalAdaptationWrapper,
        prime_native124_selected_v2_causal_adaptation_environment,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("warm evaluator requires exactly fixed visible CUDA device 0")
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED % (2**32))
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)

    velocity_contract = load_composite_mjlab_contract(root)
    velocity_limits = np.asarray(
        velocity_contract["nominal_gate"]["velocity_limit_hardware_radps"],
        dtype=np.float32,
    )
    agent_cfg = _evaluation_agent_config(root, motion, output)
    env_cfg = make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=str(motion),
        num_envs=1,
        play=False,
    )
    env_cfg.seed = FIXED_SEED
    task_audit = audit_native124_selected_v2_ankle_task_env_cfg(env_cfg)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    recorder: _TerminalPreResetRecorder | None = None
    try:
        wrapped = Native124SelectedV2CausalAdaptationWrapper(env, clip_actions=None)
        prime = prime_native124_selected_v2_causal_adaptation_environment(wrapped)
        observations = wrapped.get_observations()
        # Match launcher: simulator reset draws must not affect fresh runner
        # construction before hash-locked warm weights replace actor/critic.
        torch.manual_seed(FIXED_SEED)
        torch.cuda.manual_seed_all(FIXED_SEED)
        runner = MjlabOnPolicyRunner(wrapped, copy.deepcopy(agent_cfg), None, DEVICE)
        integration = configure_hash_locked_warm_evaluation_runner(
            runner,
            repository_root=root,
            warm_checkpoint=checkpoint,
            expected_warm_sha256=request.expected_warm_sha256,
        )
        actor = integration.actor
        actor.eval()
        integration.assert_frozen_invariants()

        binding = load_checkpoint21204_binding(root)
        source_bundle = load_selected_v2_ankle_adaptation(
            binding.checkpoint_path,
            learning_rate=FIXED_LEARNING_RATE,
            config=AnkleRowConfig(),
            load_critic=False,
            device="cpu",
        )
        source_state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in source_bundle.actor.state_dict().items()
        }
        actor_state_before = {
            key: value.detach().cpu().contiguous().clone() for key, value in actor.state_dict().items()
        }
        actor_delta = compare_actor_to_selected_source(
            actor_state_before,
            source_state,
            trainable_rows=integration.config.trainable_hardware_rows,
        )
        if actor_delta["source_identity_matches_selected"] is not True:
            raise RuntimeError("selected source actor state identity drift")
        critic_hash_before = tensor_state_sha256(integration.critic.state_dict())
        optimizer_hash_before = safe_tree_sha256(
            runner.alg.optimizer.state_dict(),
            context="evaluation optimizer before rollout",
        )

        recorder = _TerminalPreResetRecorder(env, velocity_limits)
        warmup_proof = prove_warmup_action_equivalence(device=DEVICE)
        warmup_action = torch.as_tensor(
            warmup_proof["selected_raw_action_hardware"],
            dtype=torch.float32,
            device=DEVICE,
        ).reshape(1, ACTION_DIM)
        warmup_accumulator = RolloutEvidenceAccumulator()
        warmup_records: list[dict[str, Any]] = []
        command = env.command_manager.get_term("motion")
        if _q9(command) != CAUSAL_HISTORY_ANCHOR_INDEX:
            raise RuntimeError("warm evaluation did not prime at q9=9")
        for index in range(WARMUP_STEPS):
            q9_before = _q9(command)
            recorder.arm(phase="selected_space_warmup", index=index, q9_before=q9_before)
            observations, _, dones, extras = wrapped.step(warmup_action)
            done = bool(int(dones[0].detach().cpu().item()))
            terminal = recorder.finish(done=done)
            if done or terminal is not None:
                raise RuntimeError("selected-space evaluator warmup terminated")
            q9_after = _q9(command)
            if q9_after != q9_before + 1 or _extra_termination_names(extras):
                raise RuntimeError("selected-space evaluator warmup q9/termination drift")
            evidence = _step_evidence(env, velocity_limits)
            warmup_accumulator.add(evidence)
            actual_target = _single_numpy_tensor(
                env.action_manager.get_term("joint_pos").processed_action,
                (1, ACTION_DIM),
                "warmup final target",
            )[0]
            old_target = np.asarray(warmup_proof["old_final_target_hardware"], dtype=np.float32)
            target_delta = float(np.max(np.abs(actual_target - old_target)))
            if target_delta > WARMUP_TARGET_ATOL:
                raise RuntimeError("actual warmup target differs from proven old warmup target")
            warmup_records.append(
                {
                    "index": index,
                    "q9_before": q9_before,
                    "q9_after": q9_after,
                    "old_final_target_linf_rad": target_delta,
                }
            )
        if _q9(command) != CAUSAL_HISTORY_ANCHOR_INDEX + WARMUP_STEPS:
            raise RuntimeError("two selected-space warmups did not end at q9=11")

        policy_accumulator = RolloutEvidenceAccumulator()
        attempted = 0
        completed_nonterminal = 0
        deterministic_actor_checks = 0
        first_done: dict[str, Any] | None = None
        for transition in range(MAX_POLICY_TRANSITIONS):
            q9_before = _q9(command)
            action = _deterministic_actor_action(actor, observations)
            deterministic_actor_checks += 1
            recorder.arm(phase="deterministic_actor_mean", index=transition, q9_before=q9_before)
            observations, _, dones, extras = wrapped.step(action)
            attempted += 1
            done = bool(int(dones[0].detach().cpu().item()))
            terminal = recorder.finish(done=done)
            if done:
                if terminal is None:
                    raise RuntimeError("terminal transition lacks pre-reset evidence")
                evidence = terminal.pop("evidence")
                if type(evidence) is not StepEvidence:
                    raise RuntimeError("terminal pre-reset evidence type drift")
                policy_accumulator.add(evidence)
                names = _extra_termination_names(extras)
                if names != terminal["termination_names"]:
                    raise RuntimeError("terminal extras disagree with pre-reset termination terms")
                first_done = {
                    "policy_transition": transition,
                    "q9_before": q9_before,
                    "q10_proof_before": q9_before + 1,
                    "q9_after_autoreset": _q9(command),
                    **terminal,
                    "evidence": evidence.to_dict(),
                }
                break
            if terminal is not None or _extra_termination_names(extras):
                raise RuntimeError("nonterminal transition contains termination evidence")
            if _q9(command) != q9_before + 1:
                raise RuntimeError("deterministic evaluation q9 discontinuity")
            policy_accumulator.add(_step_evidence(env, velocity_limits))
            completed_nonterminal += 1

        integration.assert_frozen_invariants()
        actor_state_after = {
            key: value.detach().cpu().contiguous().clone() for key, value in actor.state_dict().items()
        }
        if tensor_state_sha256(actor_state_after) != tensor_state_sha256(actor_state_before):
            raise RuntimeError("actor state changed during deterministic evaluation")
        critic_hash_after = tensor_state_sha256(integration.critic.state_dict())
        optimizer_hash_after = safe_tree_sha256(
            runner.alg.optimizer.state_dict(),
            context="evaluation optimizer after rollout",
        )
        storage_step = getattr(getattr(runner.alg, "storage", None), "step", None)
        if (
            critic_hash_after != critic_hash_before
            or optimizer_hash_after != optimizer_hash_before
            or integration.optimizer_steps != 0
            or runner.current_learning_iteration != 0
            or storage_step != 0
        ):
            raise RuntimeError("training state changed during deterministic evaluation")
        if resolve_rsl_runtime_binding() != preflight["rsl_runtime"]:
            raise RuntimeError("RSL runtime binding changed during deterministic evaluation")

        policy_summary = policy_accumulator.report()
        qualification = _qualification(
            first_done=first_done,
            policy_summary=policy_summary,
            actor_delta=actor_delta,
            attempted=attempted,
            deterministic_actor_checks=deterministic_actor_checks,
        )
        passed = qualification["candidate_gate_passed"] is True
        report = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "kind": EVALUATION_KIND,
            "passed": passed,
            "verdict": (
                "stage1_daddance_survival_safety_gate_passed"
                if passed
                else "stage1_daddance_candidate_quarantined"
            ),
            "preflight": preflight,
            "runtime_sources": runtime_sources,
            "task_audit": task_audit,
            "prime": prime,
            "runner": {
                "profile": "smoke_for_inference_only",
                "profile_effect_on_deterministic_mean_or_warm_load": False,
                "agent_config_sha256": _canonical_sha256(agent_cfg),
                "actor_call": "MLPModel.forward(stochastic_output=False)",
                "deterministic_actor_replay_and_rng_checks": deterministic_actor_checks,
                "optimizer_steps": integration.optimizer_steps,
                "current_learning_iteration": runner.current_learning_iteration,
                "rollout_storage_step": storage_step,
                "critic_state_sha256_before": critic_hash_before,
                "critic_state_sha256_after": critic_hash_after,
                "optimizer_state_sha256_before": optimizer_hash_before,
                "optimizer_state_sha256_after": optimizer_hash_after,
                "training_performed": False,
            },
            "actor_delta_from_selected_source": actor_delta,
            "warmup_equivalence": warmup_proof,
            "warmup": {
                "records": warmup_records,
                "summary": warmup_accumulator.report(),
            },
            "rollout": {
                "max_policy_transitions": MAX_POLICY_TRANSITIONS,
                "attempted_policy_transitions_including_done": attempted,
                "completed_nonterminal_policy_transitions": completed_nonterminal,
                "episode_steps_including_warmups": WARMUP_STEPS + attempted,
                "first_done": first_done,
                "summary_including_terminal_transition": policy_summary,
            },
            "selected_source_boundary": {
                "known_failure_q9": SELECTED_SOURCE_FAILURE_Q9,
                "known_failure_reason": "ee_body_pos",
                "same_two_selected_space_warmups": True,
            },
            "qualification": qualification,
            "safety": {
                "simulator_only": True,
                "hardware_or_network_commands_performed": False,
                "training_performed": False,
                "deployment_authorized": False,
                "promotion_authorized": False,
                "task_space_reward_thresholds_predeclared": False,
                "terminal_pre_reset_state_captured": first_done is not None,
            },
        }
        return _json_safe(report)
    finally:
        if recorder is not None:
            recorder.restore()
        env.close()


def write_warm_evaluation_report_new(
    request: WarmEvaluationRequest,
    report: Mapping[str, Any],
) -> Path:
    """Publish one finite UTF-8 JSON report with exclusive creation."""

    if type(request) is not WarmEvaluationRequest:
        raise TypeError("request must be exact WarmEvaluationRequest")
    output = request.output_path
    encoded = (json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return output


def failure_report(error: BaseException, request: WarmEvaluationRequest) -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": EVALUATION_KIND,
        "passed": False,
        "verdict": "stage1_daddance_evaluator_runtime_error",
        "error": f"{type(error).__name__}: {error}",
        "request": {
            "warm_checkpoint": str(request.warm_checkpoint),
            "expected_warm_sha256": request.expected_warm_sha256,
            "output": str(request.output),
        },
        "safety": {
            "training_performed": False,
            "hardware_or_network_commands_performed": False,
            "deployment_authorized": False,
            "candidate_quarantined": True,
        },
    }


__all__ = [
    "DEVICE",
    "EVALUATION_KIND",
    "EXPECTED_POLICY_TRANSITIONS_ATTEMPTED",
    "EXPECTED_TIMEOUT_POLICY_TRANSITION",
    "EXPECTED_TIMEOUT_Q9",
    "MAX_POLICY_TRANSITIONS",
    "RolloutEvidenceAccumulator",
    "StepEvidence",
    "WarmEvaluationRequest",
    "compare_actor_to_selected_source",
    "configure_hash_locked_warm_evaluation_runner",
    "failure_report",
    "preflight_warm_evaluation",
    "run_warm_daddance_evaluation",
    "write_warm_evaluation_report_new",
]
