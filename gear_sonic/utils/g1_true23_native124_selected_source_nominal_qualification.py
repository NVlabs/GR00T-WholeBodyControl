"""Canonical nominal-slice qualification for exact selected checkpoint 21204.

This is deliberately separate from warm-checkpoint evaluation.  It loads the
original selected PT actor and selected CPU ONNX, applies the actor immediately
at the training reset seam (q9=9), and checks both runtimes on every executed
observation.  Passing qualifies only this fixed-start DadDance simulator slice.
It never admits labels or authorizes promotion, deployment, or hardware use.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
from types import MethodType
from typing import Any, Mapping

import numpy as np
import torch

from gear_sonic.envs.mjlab.native124_selected_v2_ankle_task import (
    EPISODE_STEPS,
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
    FIXED_SEED,
    resolve_rsl_runtime_binding,
)
from gear_sonic.trl.mjlab.native124_selected_v2_ankle_rsl import (
    configure_selected_v2_ankle_rsl_runner,
)
from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
    safe_tree_sha256,
    sha256_file,
    tensor_state_sha256,
)
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTOR_STATE_SHA256,
    CHECKPOINT_SHA256,
    ONNX_SHA256,
    Native124Checkpoint21204Policy,
    load_checkpoint21204_binding,
)
from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import (
    CONTRACT_RELATIVE_PATH,
    CONTRACT_SHA256,
    load_composite_mjlab_contract,
)
import gear_sonic.utils.g1_true23_native124_selected_v2_ankle_evaluation as evaluation
import gear_sonic.utils.g1_true23_native124_selected_v2_reset_seam_diagnostic as reset_diagnostic
from gear_sonic.utils.g1_true23_teacher_support import (
    SUPPORT_CONFIG_RELATIVE_PATH,
    SUPPORT_CONFIG_SHA256,
    _compose_checkpoint21204_teacher_action,
    load_teacher_support_contract,
)

QUALIFICATION_KIND = "g1_true23_native124_selected_source_daddance_nominal_slice_qualification_v1"
QUALIFICATION_SCHEMA_VERSION = 1
DEVICE = evaluation.DEVICE
MAX_TRANSITIONS = EPISODE_STEPS
EXPECTED_DONE_TRANSITION = EPISODE_STEPS - 1
EXPECTED_DONE_Q9 = CAUSAL_HISTORY_ANCHOR_INDEX + EPISODE_STEPS - 1


@dataclass(frozen=True)
class SelectedSourceNominalQualificationRequest:
    """Immutable request with no derivative checkpoint or threshold controls."""

    repository_root: Path
    output: Path

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path) or not isinstance(self.output, Path):
            raise TypeError("qualification paths must be pathlib.Path values")

    @property
    def root(self) -> Path:
        result = self.repository_root.expanduser().resolve(strict=True)
        if not result.is_dir():
            raise ValueError("repository_root must be a directory")
        return result

    @property
    def output_path(self) -> Path:
        return evaluation._evaluation_output_path(  # noqa: SLF001
            self.root,
            self.output.expanduser(),
        )


def qualification_scope() -> dict[str, Any]:
    """Permanent narrow claim boundary for both passing and failing reports."""

    return {
        "classification": "selected_source_daddance_nominal_slice_candidate_only",
        "original_selected_actor_required": True,
        "adaptation_delta_required": False,
        "teacher_support_qualification_performed": False,
        "teacher_labels_admitted": False,
        "generalization_claimed": False,
        "disturbance_robustness_claimed": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }


def selected_source_nominal_gate_contract(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load exact frozen support-derived thresholds; invent no new metric cap."""

    root = Path(repository_root).resolve(strict=True)
    support = load_teacher_support_contract(root)
    composite = load_composite_mjlab_contract(root)
    support_identity = support["artifact_identity"]
    composite_identity = composite["artifact_identity"]
    if (
        support_identity["checkpoint_sha256"] != CHECKPOINT_SHA256
        or support_identity["actor_state_sha256"] != ACTOR_STATE_SHA256
        or support_identity["onnx_sha256"] != ONNX_SHA256
        or composite_identity["support_config_sha256"] != SUPPORT_CONFIG_SHA256
    ):
        raise RuntimeError("selected-source support identity drift")

    gate = composite["nominal_gate"]
    qualification = support["qualification"]
    expected = {
        "minimum_base_height_m": qualification["minimum_base_height_m"],
        "maximum_base_tilt_rad": qualification["maximum_base_tilt_rad"],
        "maximum_joint_velocity_ratio": qualification["maximum_joint_velocity_ratio"],
        "maximum_tracking_rmse_rad": qualification["tracking_rmse_rad_max"],
    }
    if any(gate[name] != value for name, value in expected.items()):
        raise RuntimeError("support/composite nominal threshold mismatch")
    parity_limit = float(support["global_gate"]["parity_max_absolute_error_max"])
    raw_limit = float(composite["action_chain"]["plain_sonic_raw_abs_strict_max"])
    if parity_limit != 1.0e-5 or raw_limit != 10.0:
        raise RuntimeError("selected-source parity/raw threshold drift")

    return {
        "schema": "g1_true23_selected_source_nominal_gate_contract_v1",
        "threshold_sources": {
            "support_config": {
                "path": str((root / SUPPORT_CONFIG_RELATIVE_PATH).resolve(strict=True)),
                "sha256": SUPPORT_CONFIG_SHA256,
            },
            "composite_mjlab_contract": {
                "path": str((root / CONTRACT_RELATIVE_PATH).resolve(strict=True)),
                "sha256": CONTRACT_SHA256,
            },
        },
        "required_transitions": EPISODE_STEPS,
        "initial_actor_q9": CAUSAL_HISTORY_ANCHOR_INDEX,
        "expected_done_transition": EXPECTED_DONE_TRANSITION,
        "expected_done_q9": EXPECTED_DONE_Q9,
        "minimum_base_height_m": float(gate["minimum_base_height_m"]),
        "maximum_base_tilt_rad": float(gate["maximum_base_tilt_rad"]),
        "maximum_joint_velocity_ratio": float(gate["maximum_joint_velocity_ratio"]),
        "maximum_tracking_rmse_rad": float(gate["maximum_tracking_rmse_rad"]),
        "plain_sonic_raw_abs_strict_max": raw_limit,
        "actor_onnx_parity_max_absolute_error": parity_limit,
        "required_zero_counts": list(gate["required_zero_counts"]),
        "stage1_hard_safety_violation_count_max": 0,
        "stage1_soft_safety_warning_count_max": 0,
        "inference_duration_threshold_applied": False,
        "inference_duration_exclusion": (
            "dual-runtime parity witness is not the frozen single-runtime inference seam"
        ),
        "projection_threshold_applied": False,
        "task_reward_rate_thresholds_applied": False,
    }


def preflight_selected_source_nominal_qualification(
    request: SelectedSourceNominalQualificationRequest,
) -> dict[str, Any]:
    """Bind selected PT, ONNX, motion, thresholds, runtime, and new output."""

    if type(request) is not SelectedSourceNominalQualificationRequest:
        raise TypeError("request must be exact SelectedSourceNominalQualificationRequest")
    root = request.root
    output = request.output_path
    binding = load_checkpoint21204_binding(root)
    motion = validate_dad_dance_motion_file(root / DAD_DANCE_RELATIVE_PATH)
    if sha256_file(motion) != DAD_DANCE_SHA256:
        raise RuntimeError("DadDance changed after validation")
    gate = selected_source_nominal_gate_contract(root)
    agent_cfg = evaluation._evaluation_agent_config(root, motion, output)  # noqa: SLF001
    return {
        "schema": "g1_true23_selected_source_nominal_qualification_preflight_v1",
        "ready": True,
        "selected_source": {
            "checkpoint_path": str(binding.checkpoint_path),
            "checkpoint_sha256": binding.checkpoint_sha256,
            "actor_state_sha256": binding.actor_state_sha256,
            "onnx_path": str(binding.onnx_path),
            "onnx_sha256": binding.onnx_sha256,
        },
        "motion": {
            "path": str(motion),
            "sha256": DAD_DANCE_SHA256,
            "frame_count": DAD_DANCE_FRAME_COUNT,
        },
        "gate_contract": gate,
        "output": str(output),
        "runner_agent_config_sha256": evaluation._canonical_sha256(  # noqa: SLF001
            agent_cfg
        ),
        "rsl_runtime": resolve_rsl_runtime_binding(),
        "fixed": {
            "seed": FIXED_SEED,
            "device": DEVICE,
            "num_envs": 1,
            "prime_q9": CAUSAL_HISTORY_ANCHOR_INDEX,
            "fixed_warmup_steps": 0,
            "action_substitution": False,
            "max_transitions": MAX_TRANSITIONS,
            "rsl_actor": "original_selected_pt_deterministic_mean",
            "onnx_provider": "CPUExecutionProvider",
        },
        "scope": qualification_scope(),
        "safety": {
            "simulator_constructed": False,
            "simulator_steps": 0,
            "training_updates": 0,
            "network_used": False,
            "hardware_authorized": False,
        },
    }


class ActorOnnxParityAccumulator:
    """Compact strict-absolute parity witness over the actual actor states."""

    def __init__(self, maximum_absolute_error: float) -> None:
        if (
            isinstance(maximum_absolute_error, bool)
            or not isinstance(maximum_absolute_error, (float, int))
            or not math.isfinite(float(maximum_absolute_error))
            or float(maximum_absolute_error) <= 0.0
        ):
            raise ValueError("parity threshold must be finite positive")
        self.threshold = float(maximum_absolute_error)
        self.check_count = 0
        self.violation_count = 0
        self.maximum_error = 0.0
        self.worst_transition: int | None = None
        self.worst_q9: int | None = None

    def add(
        self,
        *,
        transition: int,
        q9: int,
        rsl_action: np.ndarray,
        onnx_action: np.ndarray,
    ) -> None:
        if transition != self.check_count or q9 != CAUSAL_HISTORY_ANCHOR_INDEX + transition:
            raise ValueError("parity transition/q9 sequence drift")
        for value, context in ((rsl_action, "RSL"), (onnx_action, "ONNX")):
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or value.shape != (23,)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"{context} parity action must be finite float32 [23]")
        error = float(np.max(np.abs(rsl_action.astype(np.float64) - onnx_action.astype(np.float64))))
        if self.worst_transition is None or error > self.maximum_error:
            self.maximum_error = error
            self.worst_transition = transition
            self.worst_q9 = q9
        self.check_count += 1
        self.violation_count += int(error > self.threshold)

    def report(self) -> dict[str, Any]:
        return {
            "comparison": "cuda_rsl_original_selected_pt_vs_cpu_selected_onnx",
            "absolute_tolerance": self.threshold,
            "relative_tolerance": 0.0,
            "check_count": self.check_count,
            "violation_count": self.violation_count,
            "maximum_absolute_error": self.maximum_error,
            "worst_transition": self.worst_transition,
            "worst_q9": self.worst_q9,
            "passed": (
                self.check_count == EPISODE_STEPS
                and self.violation_count == 0
                and self.maximum_error <= self.threshold
            ),
        }


def _runtime_action_semantics(
    raw_env: Any,
    selected_raw_action: np.ndarray,
    support_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare actual selected-to-V2 runtime links with frozen composite formula."""

    if (
        not isinstance(selected_raw_action, np.ndarray)
        or selected_raw_action.dtype != np.float32
        or selected_raw_action.shape != (23,)
        or not np.isfinite(selected_raw_action).all()
    ):
        raise ValueError("selected raw action must be finite float32 [23]")
    expected = _compose_checkpoint21204_teacher_action(  # noqa: SLF001
        selected_raw_action,
        support_contract,
    )
    action = raw_env.action_manager.get_term("joint_pos")

    def vector(name: str) -> np.ndarray:
        return evaluation._single_numpy_tensor(  # noqa: SLF001
            getattr(action, name),
            (1, 23),
            f"qualification action {name}",
        )[0]

    actual = {
        "raw": vector("raw_action"),
        "candidate": vector("candidate_target_hardware"),
        "plain": vector("plain_sonic_raw_action_native"),
        "safe": vector("safe_native_action"),
        "target": vector("processed_action"),
    }
    expected_values = {
        "raw": expected.teacher_raw_action_hardware,
        "candidate": expected.teacher_candidate_target_hardware,
        "plain": expected.teacher_action_native,
        "safe": expected.teacher_applied_safe_action_native,
        "target": expected.teacher_target_hardware,
    }
    composite = support_contract["teacher_composite_contract"]
    tolerances = {
        "raw": 0.0,
        "candidate": float(composite["candidate_link_atol"]),
        "plain": float(composite["composite_link_atol"]),
        "safe": float(composite["composite_link_atol"]),
        "target": float(composite["composite_link_atol"]),
    }
    errors = {
        name: float(np.max(np.abs(actual[name].astype(np.float64) - expected_values[name].astype(np.float64))))
        for name in actual
    }
    return {
        "match": all(errors[name] <= tolerances[name] for name in errors),
        "maximum_absolute_error_by_link": errors,
        "absolute_tolerance_by_link": tolerances,
    }


class _ActionSemanticsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.mismatch_count = 0
        self.maxima = {name: 0.0 for name in ("raw", "candidate", "plain", "safe", "target")}

    def add(self, evidence: Mapping[str, Any]) -> None:
        errors = evidence.get("maximum_absolute_error_by_link")
        if not isinstance(errors, Mapping) or set(errors) != set(self.maxima):
            raise ValueError("action semantics evidence schema drift")
        for name, value in errors.items():
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError("action semantics evidence must be finite nonnegative")
            self.maxima[name] = max(self.maxima[name], value)
        self.count += 1
        self.mismatch_count += int(evidence.get("match") is not True)

    def report(self) -> dict[str, Any]:
        return {
            "check_count": self.count,
            "mismatch_count": self.mismatch_count,
            "maximum_absolute_error_by_link": dict(self.maxima),
            "passed": self.count == EPISODE_STEPS and self.mismatch_count == 0,
        }


class _TerminalQualificationRecorder:
    """Capture terminal state and action links before MJLab autoreset."""

    def __init__(
        self,
        raw_env: Any,
        velocity_limits: np.ndarray,
        support_contract: Mapping[str, Any],
    ) -> None:
        self.raw_env = raw_env
        self.velocity_limits = velocity_limits
        self.support_contract = support_contract
        self._original_reset = raw_env._reset_idx
        self._armed: dict[str, Any] | None = None
        self.captured: dict[str, Any] | None = None

        def observed_reset(_env: Any, env_ids: torch.Tensor | None = None) -> None:
            if self._armed is not None and int(_env.common_step_counter) > 0:
                if self.captured is not None:
                    raise RuntimeError("qualification recorder captured more than once")
                if env_ids is None or int(env_ids.numel()) != 1 or int(env_ids.detach().cpu().item()) != 0:
                    raise RuntimeError("qualification recorder expected environment zero")
                selected = self._armed["selected_raw_action"]
                self.captured = {
                    "transition": self._armed["transition"],
                    "q9_before": self._armed["q9_before"],
                    "episode_length_pre_reset": int(_env.episode_length_buf[0].detach().cpu().item()),
                    "termination_names": evaluation._termination_names(_env),  # noqa: SLF001
                    "is_timeout": bool(_env.termination_manager.time_outs[0].detach().cpu().item()),
                    "is_terminated": bool(_env.termination_manager.terminated[0].detach().cpu().item()),
                    "ee_body_position_errors": (reset_diagnostic.capture_exact_ee_position_errors(_env)),
                    "action_semantics": _runtime_action_semantics(
                        _env,
                        selected,
                        self.support_contract,
                    ),
                    "evidence": evaluation._step_evidence(  # noqa: SLF001
                        _env,
                        self.velocity_limits,
                    ),
                }
            self._original_reset(env_ids)

        raw_env._reset_idx = MethodType(observed_reset, raw_env)

    def arm(
        self,
        *,
        transition: int,
        q9_before: int,
        selected_raw_action: np.ndarray,
    ) -> None:
        if self._armed is not None:
            raise RuntimeError("qualification recorder already armed")
        self._armed = {
            "transition": transition,
            "q9_before": q9_before,
            "selected_raw_action": selected_raw_action.copy(),
        }
        self.captured = None

    def finish(self, *, done: bool) -> dict[str, Any] | None:
        if self._armed is None:
            raise RuntimeError("qualification recorder was not armed")
        captured = self.captured
        self._armed = None
        self.captured = None
        if done != (captured is not None):
            raise RuntimeError("qualification terminal capture disagrees with done")
        return captured

    def restore(self) -> None:
        self.raw_env._reset_idx = self._original_reset


def _support_nominal_summary(
    *,
    rollout_summary: Mapping[str, Any],
    termination_count: int,
    q9_discontinuity_count: int,
    action_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    scalars = rollout_summary["safety_scalars"]
    counts = rollout_summary["safety_count_totals"]
    support_counts = {
        "termination_count": termination_count,
        "q9_discontinuity_count": q9_discontinuity_count,
        "nonfinite_count": 0,
        "raw_clip_required_count": int(counts["raw_clip_coordinate_count"]),
        "action_semantics_mismatch_count": int(action_semantics["mismatch_count"]),
        "target_soft_limit_violation_count": int(counts["target_soft_limit_violation_count"]),
        "actuator_target_soft_limit_violation_count": int(counts["actuator_target_soft_limit_violation_count"]),
        "measured_soft_limit_violation_count": int(counts["measured_soft_limit_violation_count"]),
        "joint_velocity_limit_violation_count": int(counts["joint_velocity_limit_violation_count"]),
    }
    return {
        **support_counts,
        "minimum_base_height_m": float(scalars["base_height_m"]["minimum"]),
        "maximum_base_tilt_rad": float(scalars["base_tilt_rad"]["maximum"]),
        "maximum_joint_velocity_ratio": float(scalars["maximum_joint_velocity_ratio"]["maximum"]),
        "maximum_tracking_rmse_rad": float(scalars["target_tracking_rmse_rad"]["maximum"]),
        "maximum_plain_sonic_raw_native_abs": float(scalars["max_abs_plain_sonic_raw_native"]["maximum"]),
        "maximum_actuator_force_ratio_evidence_only": float(scalars["maximum_actuator_force_ratio"]["maximum"]),
        "maximum_ankle_projection_linf_rad_evidence_only": float(
            scalars["ankle_v2_projection_linf_rad"]["maximum"]
        ),
    }


def assess_selected_source_nominal_slice(
    *,
    gate: Mapping[str, Any],
    reset_seam: Mapping[str, Any],
    first_done: Mapping[str, Any] | None,
    attempted: int,
    support_summary: Mapping[str, Any],
    rollout_summary: Mapping[str, Any],
    parity: Mapping[str, Any],
    action_semantics: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    frozen_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure fail-closed decision for this one explicitly bounded slice."""

    reset_pass = bool(
        reset_seam.get("prime_q9") == CAUSAL_HISTORY_ANCHOR_INDEX
        and reset_seam.get("first_deterministic_actor_q9") == CAUSAL_HISTORY_ANCHOR_INDEX
        and reset_seam.get("fixed_warmup_steps") == 0
        and reset_seam.get("action_substitution") is False
        and reset_seam.get("reset_buffer_proof", {}).get("reset_virtual_torso_mask") is True
        and reset_seam.get("reset_buffer_proof", {}).get("actor_previous_action_slice_is_zero") is True
    )
    full_timeout = bool(
        first_done is not None
        and first_done.get("transition") == EXPECTED_DONE_TRANSITION
        and first_done.get("q9_before") == EXPECTED_DONE_Q9
        and first_done.get("q9_after_autoreset") == CAUSAL_HISTORY_ANCHOR_INDEX
        and first_done.get("episode_length_pre_reset") == EPISODE_STEPS
        and first_done.get("termination_names") == ["time_out"]
        and first_done.get("is_timeout") is True
        and first_done.get("is_terminated") is False
        and attempted == EPISODE_STEPS
        and rollout_summary.get("transition_count") == EPISODE_STEPS
    )
    required_zero = all(support_summary.get(name) == 0 for name in gate["required_zero_counts"])
    support_thresholds = bool(
        float(support_summary["minimum_base_height_m"]) >= float(gate["minimum_base_height_m"])
        and float(support_summary["maximum_base_tilt_rad"]) <= float(gate["maximum_base_tilt_rad"])
        and float(support_summary["maximum_joint_velocity_ratio"]) <= float(gate["maximum_joint_velocity_ratio"])
        and float(support_summary["maximum_tracking_rmse_rad"]) <= float(gate["maximum_tracking_rmse_rad"])
        and float(support_summary["maximum_plain_sonic_raw_native_abs"])
        < float(gate["plain_sonic_raw_abs_strict_max"])
    )
    stage1_safety = bool(
        rollout_summary.get("hard_safety_violation_count") <= gate["stage1_hard_safety_violation_count_max"]
        and rollout_summary.get("soft_safety_warning_count") <= gate["stage1_soft_safety_warning_count_max"]
    )
    source_exact = bool(
        source_identity.get("checkpoint_sha256") == CHECKPOINT_SHA256
        and source_identity.get("actor_state_sha256") == ACTOR_STATE_SHA256
        and source_identity.get("onnx_sha256") == ONNX_SHA256
        and source_identity.get("restart_or_derivative_loaded") is False
        and source_identity.get("adaptation_delta_required") is False
    )
    frozen = bool(
        frozen_state.get("actor_unchanged") is True
        and frozen_state.get("critic_unchanged") is True
        and frozen_state.get("optimizer_unchanged") is True
        and frozen_state.get("optimizer_steps") == 0
        and frozen_state.get("current_learning_iteration") == 0
        and frozen_state.get("rollout_storage_step") == 0
    )
    parity_pass = bool(
        parity.get("passed") is True
        and parity.get("check_count") == EPISODE_STEPS
        and parity.get("violation_count") == 0
        and float(parity.get("maximum_absolute_error", math.inf))
        <= float(gate["actor_onnx_parity_max_absolute_error"])
    )
    action_pass = bool(
        action_semantics.get("passed") is True
        and action_semantics.get("check_count") == EPISODE_STEPS
        and action_semantics.get("mismatch_count") == 0
    )
    qualified = all(
        (
            reset_pass,
            full_timeout,
            required_zero,
            support_thresholds,
            stage1_safety,
            source_exact,
            frozen,
            parity_pass,
            action_pass,
        )
    )
    return {
        "qualified_nominal_slice": qualified,
        "reset_seam_gate_passed": reset_pass,
        "full_500_transition_timeout_gate_passed": full_timeout,
        "support_required_zero_counts_gate_passed": required_zero,
        "support_nominal_threshold_gate_passed": support_thresholds,
        "stage1_hard_and_soft_safety_gate_passed": stage1_safety,
        "selected_source_identity_gate_passed": source_exact,
        "frozen_training_state_gate_passed": frozen,
        "per_step_rsl_onnx_parity_gate_passed": parity_pass,
        "runtime_action_semantics_gate_passed": action_pass,
        "adaptation_delta_required": False,
        "scope": qualification_scope(),
    }


def run_selected_source_nominal_qualification(
    request: SelectedSourceNominalQualificationRequest,
) -> dict[str, Any]:
    """Run exact selected source immediately from q9=9 through first done."""

    preflight = preflight_selected_source_nominal_qualification(request)
    root = request.root
    output = request.output_path
    motion = (root / DAD_DANCE_RELATIVE_PATH).resolve(strict=True)
    gate = preflight["gate_contract"]
    support_contract = load_teacher_support_contract(root)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FIXED_GPU)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    runtime_sources = evaluation._bind_evaluation_runtime_sources(root)  # noqa: SLF001
    if resolve_rsl_runtime_binding() != preflight["rsl_runtime"]:
        raise RuntimeError("RSL runtime changed after qualification preflight")

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
        raise RuntimeError("qualification requires exactly fixed visible CUDA device 0")
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED % (2**32))
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)

    composite_contract = load_composite_mjlab_contract(root)
    velocity_limits = np.asarray(
        composite_contract["nominal_gate"]["velocity_limit_hardware_radps"],
        dtype=np.float32,
    )
    agent_cfg = evaluation._evaluation_agent_config(root, motion, output)  # noqa: SLF001
    env_cfg = make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=str(motion),
        num_envs=1,
        play=False,
        fixed_anchor_index=CAUSAL_HISTORY_ANCHOR_INDEX,
    )
    env_cfg.seed = FIXED_SEED
    task_audit = audit_native124_selected_v2_ankle_task_env_cfg(env_cfg)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    recorder: _TerminalQualificationRecorder | None = None
    try:
        wrapped = Native124SelectedV2CausalAdaptationWrapper(env, clip_actions=None)
        prime = prime_native124_selected_v2_causal_adaptation_environment(wrapped)
        command = env.command_manager.get_term("motion")
        if (
            evaluation._q9(command) != CAUSAL_HISTORY_ANCHOR_INDEX  # noqa: SLF001
            or int(env.common_step_counter) != 0
            or int(env._sim_step_counter) != 0
        ):
            raise RuntimeError("qualification did not prime without physics at q9=9")

        torch.manual_seed(FIXED_SEED)
        torch.cuda.manual_seed_all(FIXED_SEED)
        runner = MjlabOnPolicyRunner(wrapped, copy.deepcopy(agent_cfg), None, DEVICE)
        integration = configure_selected_v2_ankle_rsl_runner(
            runner,
            repository_root=root,
            restart_path=None,
            expected_restart_sha256=None,
        )
        actor = integration.actor
        actor.eval()
        integration.assert_frozen_invariants()
        binding = load_checkpoint21204_binding(root)
        onnx_policy = Native124Checkpoint21204Policy(binding)

        observations = wrapped.get_observations()
        reset_buffer_proof = reset_diagnostic.prove_reset_actor_observation(
            observations,
            wrapped.diagnostics,
        )
        if (
            evaluation._q9(command) != CAUSAL_HISTORY_ANCHOR_INDEX  # noqa: SLF001
            or int(env.common_step_counter) != 0
            or int(env._sim_step_counter) != 0
        ):
            raise RuntimeError("runner construction changed q9 or stepped physics")

        actor_hash_before = tensor_state_sha256(actor.state_dict())
        if actor_hash_before != ACTOR_STATE_SHA256:
            raise RuntimeError("actual RSL actor is not exact selected source")
        critic_hash_before = tensor_state_sha256(integration.critic.state_dict())
        optimizer_hash_before = safe_tree_sha256(
            runner.alg.optimizer.state_dict(),
            context="selected-source qualification optimizer before",
        )
        source_identity = {
            "checkpoint_path": str(binding.checkpoint_path),
            "checkpoint_sha256": binding.checkpoint_sha256,
            "actor_state_sha256": actor_hash_before,
            "onnx_path": str(binding.onnx_path),
            "onnx_sha256": binding.onnx_sha256,
            "restart_or_derivative_loaded": False,
            "adaptation_delta_required": False,
        }

        recorder = _TerminalQualificationRecorder(
            env,
            velocity_limits,
            support_contract,
        )
        accumulator = evaluation.RolloutEvidenceAccumulator()
        parity_accumulator = ActorOnnxParityAccumulator(gate["actor_onnx_parity_max_absolute_error"])
        semantics_accumulator = _ActionSemanticsAccumulator()
        attempted = 0
        q9_discontinuity_count = 0
        first_actor_q9: int | None = None
        first_done: dict[str, Any] | None = None

        for transition in range(MAX_TRANSITIONS):
            q9_before = evaluation._q9(command)  # noqa: SLF001
            if first_actor_q9 is None:
                first_actor_q9 = q9_before
            actor_observation = observations["actor"].detach().cpu().contiguous().numpy().copy()
            rsl_action_tensor = evaluation._deterministic_actor_action(  # noqa: SLF001
                actor,
                observations,
            )
            rsl_action = rsl_action_tensor.detach().cpu().contiguous().numpy().copy()[0]
            onnx_action = onnx_policy.run(actor_observation)
            parity_accumulator.add(
                transition=transition,
                q9=q9_before,
                rsl_action=rsl_action,
                onnx_action=onnx_action,
            )

            recorder.arm(
                transition=transition,
                q9_before=q9_before,
                selected_raw_action=rsl_action,
            )
            observations, _, dones, extras = wrapped.step(rsl_action_tensor)
            attempted += 1
            done = bool(int(dones[0].detach().cpu().item()))
            terminal = recorder.finish(done=done)
            if done:
                if terminal is None:
                    raise RuntimeError("qualification terminal lacks pre-reset evidence")
                evidence = terminal.pop("evidence")
                semantics = terminal.pop("action_semantics")
                if type(evidence) is not evaluation.StepEvidence:
                    raise RuntimeError("qualification terminal evidence type drift")
                accumulator.add(evidence)
                semantics_accumulator.add(semantics)
                names = evaluation._extra_termination_names(extras)  # noqa: SLF001
                if names != terminal["termination_names"]:
                    raise RuntimeError("terminal extras disagree with pre-reset terms")
                first_done = {
                    **terminal,
                    "q10_proof_before": q9_before + 1,
                    "q9_after_autoreset": evaluation._q9(command),  # noqa: SLF001
                    "evidence": evidence.to_dict(),
                }
                break
            if terminal is not None or evaluation._extra_termination_names(extras):  # noqa: SLF001
                raise RuntimeError("qualification nonterminal contains termination evidence")
            q9_after = evaluation._q9(command)  # noqa: SLF001
            if q9_after != q9_before + 1:
                q9_discontinuity_count += 1
                raise RuntimeError("qualification q9 discontinuity")
            accumulator.add(evaluation._step_evidence(env, velocity_limits))  # noqa: SLF001
            semantics_accumulator.add(
                _runtime_action_semantics(
                    env,
                    rsl_action,
                    support_contract,
                )
            )

        integration.assert_frozen_invariants()
        actor_hash_after = tensor_state_sha256(actor.state_dict())
        critic_hash_after = tensor_state_sha256(integration.critic.state_dict())
        optimizer_hash_after = safe_tree_sha256(
            runner.alg.optimizer.state_dict(),
            context="selected-source qualification optimizer after",
        )
        storage_step = getattr(getattr(runner.alg, "storage", None), "step", None)
        frozen_state = {
            "actor_state_sha256_before": actor_hash_before,
            "actor_state_sha256_after": actor_hash_after,
            "actor_unchanged": actor_hash_after == actor_hash_before,
            "critic_state_sha256_before": critic_hash_before,
            "critic_state_sha256_after": critic_hash_after,
            "critic_unchanged": critic_hash_after == critic_hash_before,
            "optimizer_state_sha256_before": optimizer_hash_before,
            "optimizer_state_sha256_after": optimizer_hash_after,
            "optimizer_unchanged": optimizer_hash_after == optimizer_hash_before,
            "optimizer_steps": integration.optimizer_steps,
            "current_learning_iteration": runner.current_learning_iteration,
            "rollout_storage_step": storage_step,
        }
        if not (
            frozen_state["actor_unchanged"]
            and frozen_state["critic_unchanged"]
            and frozen_state["optimizer_unchanged"]
            and integration.optimizer_steps == 0
            and runner.current_learning_iteration == 0
            and storage_step == 0
        ):
            raise RuntimeError("training state changed during qualification")
        if first_actor_q9 != CAUSAL_HISTORY_ANCHOR_INDEX:
            raise RuntimeError("selected source actor did not begin at q9=9")
        if resolve_rsl_runtime_binding() != preflight["rsl_runtime"]:
            raise RuntimeError("RSL runtime changed during qualification")

        rollout_summary = accumulator.report()
        parity = parity_accumulator.report()
        action_semantics = semantics_accumulator.report()
        unexpected_termination_count = int(
            first_done is None
            or first_done.get("termination_names") != ["time_out"]
            or first_done.get("is_terminated") is not False
        )
        support_summary = _support_nominal_summary(
            rollout_summary=rollout_summary,
            termination_count=unexpected_termination_count,
            q9_discontinuity_count=q9_discontinuity_count,
            action_semantics=action_semantics,
        )
        reset_seam = {
            "prime_q9": CAUSAL_HISTORY_ANCHOR_INDEX,
            "first_deterministic_actor_q9": first_actor_q9,
            "fixed_warmup_steps": 0,
            "action_substitution": False,
            "first_action": "original_selected_pt_deterministic_actor_mean",
            "reset_buffer_proof": reset_buffer_proof,
        }
        qualification = assess_selected_source_nominal_slice(
            gate=gate,
            reset_seam=reset_seam,
            first_done=first_done,
            attempted=attempted,
            support_summary=support_summary,
            rollout_summary=rollout_summary,
            parity=parity,
            action_semantics=action_semantics,
            source_identity=source_identity,
            frozen_state=frozen_state,
        )
        passed = qualification["qualified_nominal_slice"] is True
        report = {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "kind": QUALIFICATION_KIND,
            "qualified_nominal_slice": passed,
            "verdict": (
                "selected_source_daddance_nominal_slice_qualified"
                if passed
                else "selected_source_daddance_nominal_slice_rejected"
            ),
            "scope": qualification_scope(),
            "preflight": preflight,
            "runtime_sources": runtime_sources,
            "task_audit": task_audit,
            "prime": prime,
            "source_identity": source_identity,
            "reset_seam": reset_seam,
            "rollout": {
                "max_transitions": MAX_TRANSITIONS,
                "attempted_transitions_including_done": attempted,
                "first_done": first_done,
                "summary_including_terminal_transition": rollout_summary,
                "support_nominal_summary": support_summary,
            },
            "per_step_actor_onnx_parity": parity,
            "runtime_action_semantics": action_semantics,
            "frozen_training_state_proof": frozen_state,
            "qualification": qualification,
            "old_evidence": {
                "warm_evaluator_modified": False,
                "reset_seam_diagnostic_reclassified": False,
                "existing_reports_superseded": False,
            },
            "safety": {
                "simulator_only": True,
                "training_performed": False,
                "hardware_or_network_commands_performed": False,
                "teacher_labels_admitted": False,
                "promotion_authorized": False,
                "deployment_authorized": False,
            },
        }
        return evaluation._json_safe(report)  # noqa: SLF001
    finally:
        if recorder is not None:
            recorder.restore()
        env.close()


def write_selected_source_nominal_qualification_new(
    request: SelectedSourceNominalQualificationRequest,
    report: Mapping[str, Any],
) -> Path:
    """Publish one finite UTF-8 report with exclusive creation."""

    if type(request) is not SelectedSourceNominalQualificationRequest:
        raise TypeError("request must be exact SelectedSourceNominalQualificationRequest")
    output = request.output_path
    encoded = (
        json.dumps(
            evaluation._json_safe(report),  # noqa: SLF001
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return output


def failure_report(
    error: BaseException,
    request: SelectedSourceNominalQualificationRequest,
) -> dict[str, Any]:
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "kind": QUALIFICATION_KIND,
        "qualified_nominal_slice": False,
        "verdict": "selected_source_nominal_qualification_runtime_error",
        "error": f"{type(error).__name__}: {error}",
        "request": {
            "repository_root": str(request.repository_root),
            "output": str(request.output),
        },
        "scope": qualification_scope(),
        "safety": {
            "training_performed": False,
            "hardware_or_network_commands_performed": False,
            "teacher_labels_admitted": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
        },
    }


__all__ = [
    "ActorOnnxParityAccumulator",
    "EXPECTED_DONE_Q9",
    "EXPECTED_DONE_TRANSITION",
    "QUALIFICATION_KIND",
    "SelectedSourceNominalQualificationRequest",
    "assess_selected_source_nominal_slice",
    "failure_report",
    "preflight_selected_source_nominal_qualification",
    "qualification_scope",
    "run_selected_source_nominal_qualification",
    "selected_source_nominal_gate_contract",
    "write_selected_source_nominal_qualification_new",
]
