"""Pinned full-clip and phase-restart qualification for selected source 21204.

Only two schedule families exist: the one continuous q9=9..2088 window, or
one named pinned phase window w0..w4.  No caller supplies an anchor or horizon.
Both modes load the original selected PT actor and CPU ONNX witness directly.
Passing remains simulator-only evidence and never admits labels or deployment.
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
    CONTROL_DT_S,
    EPISODE_STEPS,
    audit_native124_selected_v2_ankle_task_env_cfg,
)
from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
    DAD_DANCE_FRAME_COUNT,
    DAD_DANCE_RELATIVE_PATH,
    DAD_DANCE_SHA256,
    causal_episode_last_q9,
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
    load_composite_mjlab_contract,
)
import gear_sonic.utils.g1_true23_native124_selected_source_nominal_qualification as nominal
import gear_sonic.utils.g1_true23_native124_selected_v2_ankle_evaluation as evaluation
from gear_sonic.utils.g1_true23_native124_selected_v2_full_clip_schedule import (
    CONTINUOUS_CONTRACT_SHA256,
    FULL_CLIP_CONTINUOUS_WINDOW,
    PHASE_SCHEDULE_SHA256,
    PHASE_WINDOWS,
    QUALIFICATION_SCHEDULE_SHA256,
    CausalQualificationWindow,
    qualification_schedule_contract,
    validate_qualification_schedule_contract,
)
import gear_sonic.utils.g1_true23_native124_selected_v2_reset_seam_diagnostic as reset_diagnostic
from gear_sonic.utils.g1_true23_teacher_support import (
    load_teacher_support_contract,
)

QUALIFICATION_KIND = "g1_true23_native124_selected_source_daddance_full_clip_qualification_v1"
QUALIFICATION_SCHEMA_VERSION = 1
DEVICE = evaluation.DEVICE
MODES = ("continuous", "phase")
PHASE_WINDOW_IDS = tuple(window.window_id for window in PHASE_WINDOWS)
_EXPECTED_ACTION_CONTRACT_FAILURES = {
    "teacher composite requires plain SONIC raw clipping": ("plain_sonic_raw_clip_required"),
    "teacher composite safe inverse does not recover plain SONIC raw label": ("safe_inverse_round_trip_failure"),
    "teacher composite safe native action does not recover V2 forward transform": (
        "safe_inverse_round_trip_failure"
    ),
    "teacher composite final hardware target does not recover V2 forward transform": (
        "safe_inverse_round_trip_failure"
    ),
}


def resolve_qualification_window(
    mode: str,
    phase_window_id: str | None,
) -> CausalQualificationWindow:
    """Resolve only pinned schedule objects; reject custom window semantics."""

    validate_qualification_schedule_contract(qualification_schedule_contract())
    if mode == "continuous":
        if phase_window_id is not None:
            raise ValueError("continuous mode forbids phase_window_id")
        return FULL_CLIP_CONTINUOUS_WINDOW
    if mode != "phase":
        raise ValueError("mode must be continuous or phase")
    if phase_window_id not in PHASE_WINDOW_IDS:
        raise ValueError(f"phase_window_id must be one of {PHASE_WINDOW_IDS}")
    return next(window for window in PHASE_WINDOWS if window.window_id == phase_window_id)


@dataclass(frozen=True)
class FullClipQualificationRequest:
    repository_root: Path
    output: Path
    mode: str
    phase_window_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path) or not isinstance(self.output, Path):
            raise TypeError("qualification paths must be pathlib.Path values")
        if type(self.mode) is not str:
            raise TypeError("mode must be string")
        if self.phase_window_id is not None and type(self.phase_window_id) is not str:
            raise TypeError("phase_window_id must be string or None")
        resolve_qualification_window(self.mode, self.phase_window_id)

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

    @property
    def window(self) -> CausalQualificationWindow:
        return resolve_qualification_window(self.mode, self.phase_window_id)


def qualification_scope(
    mode: str,
    phase_window_id: str | None,
) -> dict[str, Any]:
    window = resolve_qualification_window(mode, phase_window_id)
    return {
        "classification": "selected_source_daddance_full_clip_candidate_only",
        "requested_mode": mode,
        "phase_window_id": None if mode == "continuous" else window.window_id,
        "continuous_full_clip_reachability_claim_permitted": mode == "continuous",
        "phase_restartability_claim_permitted": mode == "phase",
        "cross_mode_claim_permitted": False,
        "teacher_support_qualification_performed": False,
        "teacher_labels_admitted": False,
        "adaptation_delta_required": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "hardware_authorized": False,
    }


def full_clip_gate_contract(
    repository_root: str | Path,
    window: CausalQualificationWindow,
) -> dict[str, Any]:
    """Bind pinned schedule plus unchanged support-derived nominal thresholds."""

    if type(window) is not CausalQualificationWindow:
        raise TypeError("window must be exact CausalQualificationWindow")
    schedule_contract = validate_qualification_schedule_contract(qualification_schedule_contract())
    base = nominal.selected_source_nominal_gate_contract(repository_root)
    return {
        "schema": "g1_true23_selected_source_full_clip_gate_contract_v1",
        "schedule": schedule_contract,
        "schedule_sha256": QUALIFICATION_SCHEDULE_SHA256,
        "continuous_contract_sha256": CONTINUOUS_CONTRACT_SHA256,
        "phase_schedule_sha256": PHASE_SCHEDULE_SHA256,
        "window": {
            "id": window.window_id,
            "anchor_q9": window.anchor_q9,
            "transitions": window.transitions,
            "burn_in_transitions": window.burn_in_transitions,
            "first_scored_q9": window.first_scored_q9,
            "last_q9": window.last_q9,
            "q10_proof_last": window.last_q9 + 1,
        },
        "threshold_sources": base["threshold_sources"],
        "minimum_base_height_m": base["minimum_base_height_m"],
        "maximum_base_tilt_rad": base["maximum_base_tilt_rad"],
        "maximum_joint_velocity_ratio": base["maximum_joint_velocity_ratio"],
        "maximum_tracking_rmse_rad": base["maximum_tracking_rmse_rad"],
        "plain_sonic_raw_abs_strict_max": base["plain_sonic_raw_abs_strict_max"],
        "actor_onnx_parity_max_absolute_error": base["actor_onnx_parity_max_absolute_error"],
        "required_zero_counts": list(base["required_zero_counts"]),
        "stage1_hard_safety_violation_count_max": 0,
        "stage1_soft_safety_warning_count_max": 0,
        "inference_duration_threshold_applied": False,
        "projection_threshold_applied": False,
        "task_reward_rate_thresholds_applied": False,
    }


def preflight_full_clip_qualification(
    request: FullClipQualificationRequest,
) -> dict[str, Any]:
    if type(request) is not FullClipQualificationRequest:
        raise TypeError("request must be exact FullClipQualificationRequest")
    root = request.root
    output = request.output_path
    window = request.window
    binding = load_checkpoint21204_binding(root)
    motion = validate_dad_dance_motion_file(root / DAD_DANCE_RELATIVE_PATH)
    if sha256_file(motion) != DAD_DANCE_SHA256:
        raise RuntimeError("DadDance changed after validation")
    gate = full_clip_gate_contract(root, window)
    agent_cfg = evaluation._evaluation_agent_config(root, motion, output)  # noqa: SLF001
    return {
        "schema": "g1_true23_selected_source_full_clip_preflight_v1",
        "ready": True,
        "mode": request.mode,
        "window": gate["window"],
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
            "fixed_anchor_q9": window.anchor_q9,
            "transitions": window.transitions,
            "burn_in_transitions": window.burn_in_transitions,
            "fixed_warmup_or_action_substitution_steps": 0,
            "rsl_actor": "original_selected_pt_deterministic_mean",
            "onnx_provider": "CPUExecutionProvider",
        },
        "scope": qualification_scope(request.mode, request.phase_window_id),
        "safety": {
            "simulator_constructed": False,
            "simulator_steps": 0,
            "training_updates": 0,
            "network_used": False,
            "hardware_authorized": False,
        },
    }


def configure_scheduled_task_horizon(
    env_cfg: Any,
    window: CausalQualificationWindow,
) -> dict[str, Any]:
    """Audit standard task first, then change only its scheduled horizon."""

    if type(window) is not CausalQualificationWindow:
        raise TypeError("window must be exact CausalQualificationWindow")
    base_audit = audit_native124_selected_v2_ankle_task_env_cfg(env_cfg)
    expected_base_steps = min(
        EPISODE_STEPS,
        DAD_DANCE_FRAME_COUNT - window.anchor_q9 - 1,
    )
    if base_audit["episode_steps"] != expected_base_steps:
        raise RuntimeError("base Stage-1 task did not begin at its causal-safe nominal horizon")
    command = env_cfg.commands["motion"]
    if getattr(command, "fixed_anchor_index", None) != window.anchor_q9:
        raise RuntimeError("task command lacks pinned qualification anchor")
    before = float(env_cfg.episode_length_s)
    env_cfg.episode_length_s = window.transitions * CONTROL_DT_S
    last_q9 = causal_episode_last_q9(
        reset_anchor_index=window.anchor_q9,
        episode_steps=window.transitions,
    )
    if (
        last_q9 != window.last_q9
        or abs(float(env_cfg.episode_length_s) - window.transitions * CONTROL_DT_S) > 1.0e-12
    ):
        raise RuntimeError("scheduled qualification horizon drift")
    return {
        "schema": "g1_true23_selected_source_scheduled_task_horizon_v1",
        "base_task_audit_before_horizon_override": base_audit,
        "base_episode_length_s": before,
        "horizon_override_only": True,
        "fixed_anchor_q9": window.anchor_q9,
        "scheduled_episode_steps": window.transitions,
        "scheduled_episode_length_s": float(env_cfg.episode_length_s),
        "scheduled_last_q9": last_q9,
        "scheduled_last_q10_proof": last_q9 + 1,
    }


class ScheduledActorOnnxParityAccumulator:
    def __init__(self, window: CausalQualificationWindow, threshold: float) -> None:
        if type(window) is not CausalQualificationWindow:
            raise TypeError("window must be exact CausalQualificationWindow")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (float, int))
            or not math.isfinite(float(threshold))
            or float(threshold) <= 0.0
        ):
            raise ValueError("parity threshold must be finite positive")
        self.window = window
        self.threshold = float(threshold)
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
        if transition != self.check_count or q9 != self.window.anchor_q9 + transition:
            raise ValueError("scheduled parity transition/q9 sequence drift")
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
            "required_check_count": self.window.transitions,
            "check_count": self.check_count,
            "violation_count": self.violation_count,
            "maximum_absolute_error": self.maximum_error,
            "worst_transition": self.worst_transition,
            "worst_q9": self.worst_q9,
            "passed": (
                self.check_count == self.window.transitions
                and self.violation_count == 0
                and self.maximum_error <= self.threshold
            ),
        }


class _ScheduledActionSemanticsAccumulator:
    def __init__(self, transitions: int) -> None:
        if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions <= 0:
            raise ValueError("transitions must be positive integer")
        self.transitions = transitions
        self.count = 0
        self.mismatch_count = 0
        self.maxima = {name: 0.0 for name in ("raw", "candidate", "plain", "safe", "target")}
        self.available = {name: 0 for name in self.maxima}
        self.unavailable = {name: 0 for name in self.maxima}
        self.contract_failure_count = 0

    def add(self, evidence: Mapping[str, Any]) -> None:
        errors = evidence.get("maximum_absolute_error_by_link")
        if not isinstance(errors, Mapping) or set(errors) != set(self.maxima):
            raise ValueError("action semantics evidence schema drift")
        for name, value in errors.items():
            if value is None:
                self.unavailable[name] += 1
                continue
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError("action semantics evidence must be finite nonnegative or None")
            self.available[name] += 1
            self.maxima[name] = max(self.maxima[name], value)
        self.count += 1
        self.mismatch_count += int(evidence.get("match") is not True)
        self.contract_failure_count += int(evidence.get("contract_failure") is True)

    def report(self) -> dict[str, Any]:
        return {
            "required_check_count": self.transitions,
            "check_count": self.count,
            "mismatch_count": self.mismatch_count,
            "contract_failure_count": self.contract_failure_count,
            "maximum_absolute_error_by_link": {
                name: self.maxima[name] if self.available[name] else None for name in self.maxima
            },
            "unavailable_error_count_by_link": dict(self.unavailable),
            "passed": self.count == self.transitions and self.mismatch_count == 0,
        }


def _action_chain_snapshot(
    raw_env: Any,
    selected_raw_action: np.ndarray,
) -> dict[str, Any]:
    """Capture exact runtime action spaces before any autoreset can erase them."""

    if (
        not isinstance(selected_raw_action, np.ndarray)
        or selected_raw_action.dtype != np.float32
        or selected_raw_action.shape != (23,)
        or not np.isfinite(selected_raw_action).all()
    ):
        raise ValueError("selected raw action must be finite float32 [23]")
    action = raw_env.action_manager.get_term("joint_pos")

    def vector(name: str) -> np.ndarray:
        return evaluation._single_numpy_tensor(  # noqa: SLF001
            getattr(action, name),
            (1, 23),
            f"full-clip action {name}",
        )[0]

    clip_mask = evaluation._single_numpy_tensor(  # noqa: SLF001
        action.raw_clip_mask_native,
        (1, 23),
        "full-clip raw clip mask",
        floating=False,
    )[0]
    if clip_mask.dtype != np.bool_:
        raise ValueError("full-clip raw clip mask must be boolean")
    return {
        "selected_actor_raw_action_hardware": selected_raw_action.tolist(),
        "actual_selected_raw_action_hardware": vector("raw_action").tolist(),
        "candidate_target_hardware": vector("candidate_target_hardware").tolist(),
        "plain_sonic_raw_action_native": vector("plain_sonic_raw_action_native").tolist(),
        "safe_native_action": vector("safe_native_action").tolist(),
        "final_target_hardware": vector("processed_action").tolist(),
        "raw_clip_mask_native": clip_mask.tolist(),
        "raw_clip_coordinate_count": int(np.count_nonzero(clip_mask)),
    }


def capture_action_semantics_or_expected_failure(
    raw_env: Any,
    selected_raw_action: np.ndarray,
    support_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return expected action-contract failure as evidence, never as lost context."""

    snapshot = _action_chain_snapshot(raw_env, selected_raw_action)
    try:
        semantics = nominal._runtime_action_semantics(  # noqa: SLF001
            raw_env,
            selected_raw_action,
            support_contract,
        )
    except ValueError as error:
        message = str(error)
        reason = _EXPECTED_ACTION_CONTRACT_FAILURES.get(message)
        if reason is None:
            raise
        return {
            "semantics": {
                "match": False,
                "contract_failure": True,
                "maximum_absolute_error_by_link": {
                    "raw": float(
                        np.max(
                            np.abs(
                                np.asarray(
                                    snapshot["selected_actor_raw_action_hardware"],
                                    dtype=np.float64,
                                )
                                - np.asarray(
                                    snapshot["actual_selected_raw_action_hardware"],
                                    dtype=np.float64,
                                )
                            )
                        )
                    ),
                    "candidate": None,
                    "plain": None,
                    "safe": None,
                    "target": None,
                },
            },
            "failure": {
                "schema": "g1_true23_selected_source_action_contract_failure_v1",
                "reason": reason,
                "exception_type": type(error).__name__,
                "exception_message": message,
                "action_chain": snapshot,
            },
        }
    return {
        "semantics": {
            **semantics,
            "contract_failure": False,
        },
        "failure": None,
    }


def build_partial_action_contract_failure(
    *,
    transition: int,
    q9_before: int,
    q9_after: int,
    window: CausalQualificationWindow,
    environment_done: bool,
    parity_max_absolute_error: float,
    action_observation: Mapping[str, Any],
    step_evidence: evaluation.StepEvidence,
) -> dict[str, Any]:
    """Attach exact control location and state evidence to one contract stop."""

    failure = action_observation.get("failure")
    if not isinstance(failure, Mapping):
        raise ValueError("partial action-contract evidence requires failure mapping")
    if (
        isinstance(transition, bool)
        or not isinstance(transition, int)
        or not 0 <= transition < window.transitions
        or q9_before != window.anchor_q9 + transition
        or isinstance(q9_after, bool)
        or not isinstance(q9_after, int)
        or type(environment_done) is not bool
        or type(parity_max_absolute_error) is not float
        or not math.isfinite(parity_max_absolute_error)
        or parity_max_absolute_error < 0.0
        or type(step_evidence) is not evaluation.StepEvidence
    ):
        raise ValueError("partial action-contract location/evidence drift")
    return {
        **dict(failure),
        "transition": transition,
        "q9_before": q9_before,
        "q10_proof_before": q9_before + 1,
        "q9_after": q9_after,
        "window_id": window.window_id,
        "partition": ("burn_in" if transition < window.burn_in_transitions else "scored"),
        "partial_terminal": True,
        "qualification_stop_reason": "action_contract_failure",
        "simulator_termination": environment_done,
        "environment_done": environment_done,
        "rsl_onnx_max_absolute_error_at_failure": parity_max_absolute_error,
        "step_evidence": step_evidence.to_dict(),
        "gate_weakened": False,
        "qualification_must_fail": True,
    }


class _FullClipTerminalRecorder:
    """Capture terminal action-contract and safety evidence before autoreset."""

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
                    raise RuntimeError("full-clip recorder captured more than once")
                if env_ids is None or int(env_ids.numel()) != 1 or int(env_ids.detach().cpu().item()) != 0:
                    raise RuntimeError("full-clip recorder expected environment zero")
                selected = self._armed["selected_raw_action"]
                self.captured = {
                    "transition": self._armed["transition"],
                    "q9_before": self._armed["q9_before"],
                    "parity_max_absolute_error": self._armed["parity_max_absolute_error"],
                    "episode_length_pre_reset": int(_env.episode_length_buf[0].detach().cpu().item()),
                    "termination_names": evaluation._termination_names(_env),  # noqa: SLF001
                    "is_timeout": bool(_env.termination_manager.time_outs[0].detach().cpu().item()),
                    "is_terminated": bool(_env.termination_manager.terminated[0].detach().cpu().item()),
                    "ee_body_position_errors": (reset_diagnostic.capture_exact_ee_position_errors(_env)),
                    "action_observation": (
                        capture_action_semantics_or_expected_failure(
                            _env,
                            selected,
                            self.support_contract,
                        )
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
        parity_max_absolute_error: float,
    ) -> None:
        if self._armed is not None:
            raise RuntimeError("full-clip recorder already armed")
        self._armed = {
            "transition": transition,
            "q9_before": q9_before,
            "selected_raw_action": selected_raw_action.copy(),
            "parity_max_absolute_error": parity_max_absolute_error,
        }
        self.captured = None

    def finish(self, *, done: bool) -> dict[str, Any] | None:
        if self._armed is None:
            raise RuntimeError("full-clip recorder was not armed")
        captured = self.captured
        self._armed = None
        self.captured = None
        if done != (captured is not None):
            raise RuntimeError("full-clip terminal capture disagrees with done")
        return captured

    def restore(self) -> None:
        self.raw_env._reset_idx = self._original_reset


def _claims(
    *,
    mode: str,
    window: CausalQualificationWindow,
    qualified: bool,
) -> dict[str, Any]:
    return {
        "continuous_full_clip_reachability": {
            "performed": mode == "continuous",
            "qualified": qualified if mode == "continuous" else None,
        },
        "phase_restartability": {
            "performed": mode == "phase",
            "window_id": window.window_id if mode == "phase" else None,
            "qualified": qualified if mode == "phase" else None,
        },
        "cross_mode_inference_permitted": False,
        "teacher_support_or_label_admission": False,
        "promotion_or_deployment": False,
    }


def assess_full_clip_qualification(
    *,
    mode: str,
    window: CausalQualificationWindow,
    gate: Mapping[str, Any],
    reset_seam: Mapping[str, Any],
    first_done: Mapping[str, Any] | None,
    attempted: int,
    partition_counts: Mapping[str, Any],
    support_summary: Mapping[str, Any],
    rollout_summary: Mapping[str, Any],
    parity: Mapping[str, Any],
    action_semantics: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    frozen_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure fail-closed decision with disjoint continuous/phase claims."""

    resolved = resolve_qualification_window(
        mode,
        None if mode == "continuous" else window.window_id,
    )
    if resolved != window or gate.get("window", {}).get("id") != window.window_id:
        raise ValueError("qualification window differs from pinned request/gate")
    reset_pass = bool(
        reset_seam.get("prime_q9") == window.anchor_q9
        and reset_seam.get("first_deterministic_actor_q9") == window.anchor_q9
        and reset_seam.get("fixed_warmup_or_action_substitution_steps") == 0
        and reset_seam.get("action_substitution") is False
        and reset_seam.get("reset_buffer_proof", {}).get("reset_virtual_torso_mask") is True
        and reset_seam.get("reset_buffer_proof", {}).get("actor_previous_action_slice_is_zero") is True
    )
    expected_transition = window.transitions - 1
    full_timeout = bool(
        first_done is not None
        and first_done.get("transition") == expected_transition
        and first_done.get("q9_before") == window.last_q9
        and first_done.get("q9_after_autoreset") == window.anchor_q9
        and first_done.get("episode_length_pre_reset") == window.transitions
        and first_done.get("termination_names") == ["time_out"]
        and first_done.get("is_timeout") is True
        and first_done.get("is_terminated") is False
        and attempted == window.transitions
        and rollout_summary.get("transition_count") == window.transitions
    )
    partition_pass = bool(
        partition_counts.get("burn_in_transition_count") == window.burn_in_transitions
        and partition_counts.get("scored_transition_count") == window.transitions - window.burn_in_transitions
        and partition_counts.get("unexpected_done_before_final_count") == 0
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
        rollout_summary.get("hard_safety_violation_count") == 0
        and rollout_summary.get("soft_safety_warning_count") == 0
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
        and parity.get("check_count") == window.transitions
        and parity.get("violation_count") == 0
        and float(parity.get("maximum_absolute_error", math.inf))
        <= float(gate["actor_onnx_parity_max_absolute_error"])
    )
    action_pass = bool(
        action_semantics.get("passed") is True
        and action_semantics.get("check_count") == window.transitions
        and action_semantics.get("mismatch_count") == 0
    )
    qualified = all(
        (
            reset_pass,
            full_timeout,
            partition_pass,
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
        "qualified_requested_mode": qualified,
        "mode": mode,
        "window_id": window.window_id,
        "reset_seam_gate_passed": reset_pass,
        "exact_terminal_timeout_gate_passed": full_timeout,
        "burn_in_and_score_partition_gate_passed": partition_pass,
        "support_required_zero_counts_gate_passed": required_zero,
        "support_nominal_threshold_gate_passed": support_thresholds,
        "stage1_hard_and_soft_safety_gate_passed": stage1_safety,
        "selected_source_identity_gate_passed": source_exact,
        "frozen_training_state_gate_passed": frozen,
        "per_step_rsl_onnx_parity_gate_passed": parity_pass,
        "runtime_action_semantics_gate_passed": action_pass,
        "claims": _claims(mode=mode, window=window, qualified=qualified),
        "scope": qualification_scope(
            mode,
            None if mode == "continuous" else window.window_id,
        ),
    }


def run_full_clip_qualification(
    request: FullClipQualificationRequest,
) -> dict[str, Any]:
    preflight = preflight_full_clip_qualification(request)
    root = request.root
    output = request.output_path
    window = request.window
    gate = preflight["gate_contract"]
    motion = (root / DAD_DANCE_RELATIVE_PATH).resolve(strict=True)
    support_contract = load_teacher_support_contract(root)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FIXED_GPU)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    runtime_sources = evaluation._bind_evaluation_runtime_sources(root)  # noqa: SLF001
    if resolve_rsl_runtime_binding() != preflight["rsl_runtime"]:
        raise RuntimeError("RSL runtime changed after full-clip preflight")

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
        raise RuntimeError("full-clip qualification requires fixed visible CUDA device 0")
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
        fixed_anchor_index=window.anchor_q9,
    )
    env_cfg.seed = FIXED_SEED
    scheduled_task_audit = configure_scheduled_task_horizon(env_cfg, window)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    recorder: _FullClipTerminalRecorder | None = None
    try:
        wrapped = Native124SelectedV2CausalAdaptationWrapper(env, clip_actions=None)
        prime = prime_native124_selected_v2_causal_adaptation_environment(wrapped)
        command = env.command_manager.get_term("motion")
        if (
            evaluation._q9(command) != window.anchor_q9  # noqa: SLF001
            or int(env.common_step_counter) != 0
            or int(env._sim_step_counter) != 0
        ):
            raise RuntimeError("full-clip qualification prime anchor/counter drift")

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
            evaluation._q9(command) != window.anchor_q9  # noqa: SLF001
            or int(env.common_step_counter) != 0
            or int(env._sim_step_counter) != 0
            or int(env.max_episode_length) != window.transitions
        ):
            raise RuntimeError("runner construction changed scheduled reset/horizon")

        actor_hash_before = tensor_state_sha256(actor.state_dict())
        if actor_hash_before != ACTOR_STATE_SHA256:
            raise RuntimeError("actual RSL actor is not exact selected source")
        critic_hash_before = tensor_state_sha256(integration.critic.state_dict())
        optimizer_hash_before = safe_tree_sha256(
            runner.alg.optimizer.state_dict(),
            context="full-clip qualification optimizer before",
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

        recorder = _FullClipTerminalRecorder(
            env,
            velocity_limits,
            support_contract,
        )
        full_accumulator = evaluation.RolloutEvidenceAccumulator()
        burn_in_accumulator = evaluation.RolloutEvidenceAccumulator()
        scored_accumulator = evaluation.RolloutEvidenceAccumulator()
        parity_accumulator = ScheduledActorOnnxParityAccumulator(
            window,
            gate["actor_onnx_parity_max_absolute_error"],
        )
        semantics_accumulator = _ScheduledActionSemanticsAccumulator(window.transitions)
        attempted = 0
        first_actor_q9: int | None = None
        first_done: dict[str, Any] | None = None
        action_contract_failure: dict[str, Any] | None = None
        q9_discontinuity_count = 0

        def add_evidence(transition: int, evidence: evaluation.StepEvidence) -> None:
            full_accumulator.add(evidence)
            if transition < window.burn_in_transitions:
                burn_in_accumulator.add(evidence)
            else:
                scored_accumulator.add(evidence)

        for transition in range(window.transitions):
            q9_before = evaluation._q9(command)  # noqa: SLF001
            expected_q9 = window.anchor_q9 + transition
            if q9_before != expected_q9:
                q9_discontinuity_count += 1
                raise RuntimeError("full-clip q9 sequence discontinuity")
            if first_actor_q9 is None:
                first_actor_q9 = q9_before
            actor_observation = observations["actor"].detach().cpu().contiguous().numpy().copy()
            rsl_action_tensor = evaluation._deterministic_actor_action(  # noqa: SLF001
                actor,
                observations,
            )
            rsl_action = rsl_action_tensor.detach().cpu().contiguous().numpy().copy()[0]
            onnx_action = onnx_policy.run(actor_observation)
            per_step_parity_error = float(
                np.max(np.abs(rsl_action.astype(np.float64) - onnx_action.astype(np.float64)))
            )
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
                parity_max_absolute_error=per_step_parity_error,
            )
            observations, _, dones, extras = wrapped.step(rsl_action_tensor)
            attempted += 1
            done = bool(int(dones[0].detach().cpu().item()))
            terminal = recorder.finish(done=done)
            if done:
                if terminal is None:
                    raise RuntimeError("full-clip terminal lacks pre-reset evidence")
                evidence = terminal.pop("evidence")
                action_observation = terminal.pop("action_observation")
                semantics = action_observation["semantics"]
                if type(evidence) is not evaluation.StepEvidence:
                    raise RuntimeError("full-clip terminal evidence type drift")
                add_evidence(transition, evidence)
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
                if action_observation["failure"] is not None:
                    action_contract_failure = build_partial_action_contract_failure(
                        transition=transition,
                        q9_before=q9_before,
                        q9_after=evaluation._q9(command),  # noqa: SLF001
                        window=window,
                        environment_done=True,
                        parity_max_absolute_error=per_step_parity_error,
                        action_observation=action_observation,
                        step_evidence=evidence,
                    )
                break
            if terminal is not None or evaluation._extra_termination_names(extras):  # noqa: SLF001
                raise RuntimeError("full-clip nonterminal contains termination evidence")
            if evaluation._q9(command) != q9_before + 1:  # noqa: SLF001
                q9_discontinuity_count += 1
                raise RuntimeError("full-clip q9 post-step discontinuity")
            evidence = evaluation._step_evidence(env, velocity_limits)  # noqa: SLF001
            add_evidence(transition, evidence)
            action_observation = capture_action_semantics_or_expected_failure(
                env,
                rsl_action,
                support_contract,
            )
            semantics_accumulator.add(action_observation["semantics"])
            if action_observation["failure"] is not None:
                action_contract_failure = build_partial_action_contract_failure(
                    transition=transition,
                    q9_before=q9_before,
                    q9_after=evaluation._q9(command),  # noqa: SLF001
                    window=window,
                    environment_done=False,
                    parity_max_absolute_error=per_step_parity_error,
                    action_observation=action_observation,
                    step_evidence=evidence,
                )
                break

        integration.assert_frozen_invariants()
        actor_hash_after = tensor_state_sha256(actor.state_dict())
        critic_hash_after = tensor_state_sha256(integration.critic.state_dict())
        optimizer_hash_after = safe_tree_sha256(
            runner.alg.optimizer.state_dict(),
            context="full-clip qualification optimizer after",
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
            raise RuntimeError("training state changed during full-clip qualification")
        if first_actor_q9 != window.anchor_q9:
            raise RuntimeError("selected source actor did not begin at pinned anchor")
        if resolve_rsl_runtime_binding() != preflight["rsl_runtime"]:
            raise RuntimeError("RSL runtime changed during full-clip qualification")

        rollout_summary = full_accumulator.report()
        burn_in_summary = burn_in_accumulator.report()
        scored_summary = scored_accumulator.report()
        parity = parity_accumulator.report()
        action_semantics = semantics_accumulator.report()
        if action_contract_failure is not None:
            action_contract_failure["accumulated_prefix"] = {
                "attempted_transitions_including_failure": attempted,
                "per_step_actor_onnx_parity": parity,
                "rollout_safety_and_reward": rollout_summary,
                "runtime_action_semantics": action_semantics,
            }
        expected_final = bool(
            first_done is not None
            and first_done.get("transition") == window.transitions - 1
            and first_done.get("q9_before") == window.last_q9
            and first_done.get("termination_names") == ["time_out"]
            and first_done.get("is_timeout") is True
            and first_done.get("is_terminated") is False
        )
        unexpected_done_before_final_count = int(
            first_done is not None and first_done.get("transition") != window.transitions - 1
        )
        termination_count = int(not expected_final)
        support_summary = nominal._support_nominal_summary(  # noqa: SLF001
            rollout_summary=rollout_summary,
            termination_count=termination_count,
            q9_discontinuity_count=q9_discontinuity_count,
            action_semantics=action_semantics,
        )
        partition_counts = {
            "burn_in_transition_count": burn_in_summary["transition_count"],
            "scored_transition_count": scored_summary["transition_count"],
            "unexpected_done_before_final_count": unexpected_done_before_final_count,
        }
        reset_seam = {
            "prime_q9": window.anchor_q9,
            "first_deterministic_actor_q9": first_actor_q9,
            "fixed_warmup_or_action_substitution_steps": 0,
            "action_substitution": False,
            "first_action": "original_selected_pt_deterministic_actor_mean",
            "reset_buffer_proof": reset_buffer_proof,
        }
        qualification = assess_full_clip_qualification(
            mode=request.mode,
            window=window,
            gate=gate,
            reset_seam=reset_seam,
            first_done=first_done,
            attempted=attempted,
            partition_counts=partition_counts,
            support_summary=support_summary,
            rollout_summary=rollout_summary,
            parity=parity,
            action_semantics=action_semantics,
            source_identity=source_identity,
            frozen_state=frozen_state,
        )
        passed = qualification["qualified_requested_mode"] is True
        report = {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "kind": QUALIFICATION_KIND,
            "qualified_requested_mode": passed,
            "mode": request.mode,
            "window_id": window.window_id,
            "verdict": (
                "selected_source_action_contract_failure_quarantined"
                if action_contract_failure is not None
                else (
                    f"selected_source_{request.mode}_{window.window_id}_qualified"
                    if passed
                    else f"selected_source_{request.mode}_{window.window_id}_rejected"
                )
            ),
            "scope": qualification_scope(request.mode, request.phase_window_id),
            "claims": qualification["claims"],
            "preflight": preflight,
            "runtime_sources": runtime_sources,
            "scheduled_task_audit": scheduled_task_audit,
            "prime": prime,
            "source_identity": source_identity,
            "reset_seam": reset_seam,
            "rollout": {
                "attempted_transitions_including_done": attempted,
                "first_done": first_done,
                "partition_counts": partition_counts,
                "summary_all_transitions": rollout_summary,
                "summary_burn_in": burn_in_summary,
                "summary_scored": scored_summary,
                "support_nominal_summary_all_transitions": support_summary,
            },
            "per_step_actor_onnx_parity": parity,
            "runtime_action_semantics": action_semantics,
            "partial_action_contract_failure": action_contract_failure,
            "frozen_training_state_proof": frozen_state,
            "qualification": qualification,
            "old_evidence": {
                "canonical_500_step_qualifier_modified": False,
                "warm_evaluator_modified": False,
                "existing_reports_reclassified_or_superseded": False,
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


def write_full_clip_qualification_new(
    request: FullClipQualificationRequest,
    report: Mapping[str, Any],
) -> Path:
    if type(request) is not FullClipQualificationRequest:
        raise TypeError("request must be exact FullClipQualificationRequest")
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
    request: FullClipQualificationRequest,
) -> dict[str, Any]:
    window = request.window
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "kind": QUALIFICATION_KIND,
        "qualified_requested_mode": False,
        "mode": request.mode,
        "window_id": window.window_id,
        "verdict": "selected_source_full_clip_qualification_runtime_error",
        "error": f"{type(error).__name__}: {error}",
        "request": {
            "repository_root": str(request.repository_root),
            "output": str(request.output),
        },
        "scope": qualification_scope(request.mode, request.phase_window_id),
        "claims": _claims(mode=request.mode, window=window, qualified=False),
        "safety": {
            "training_performed": False,
            "hardware_or_network_commands_performed": False,
            "teacher_labels_admitted": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
        },
    }


__all__ = [
    "FullClipQualificationRequest",
    "MODES",
    "PHASE_WINDOW_IDS",
    "QUALIFICATION_KIND",
    "ScheduledActorOnnxParityAccumulator",
    "assess_full_clip_qualification",
    "configure_scheduled_task_horizon",
    "failure_report",
    "full_clip_gate_contract",
    "preflight_full_clip_qualification",
    "qualification_scope",
    "resolve_qualification_window",
    "run_full_clip_qualification",
    "write_full_clip_qualification_new",
]
