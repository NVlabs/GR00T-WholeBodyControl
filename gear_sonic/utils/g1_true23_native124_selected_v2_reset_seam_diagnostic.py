"""Non-qualifying deterministic diagnostic for the Stage-1 training reset seam.

Unlike the qualification evaluator, this diagnostic applies the deterministic
warm actor immediately after the target-only prime at q9=9.  It deliberately
performs no fixed warmup and no action substitution.  Its only purpose is to
measure the reset/action history distribution used by PPO training.
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
    audit_native124_selected_v2_ankle_task_env_cfg,
)
from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
    CAUSAL_HISTORY_ANCHOR_INDEX,
    DAD_DANCE_RELATIVE_PATH,
)
from gear_sonic.scripts.train_g1_true23_native124_selected_v2_ankle import (
    FIXED_GPU,
    FIXED_SEED,
    resolve_rsl_runtime_binding,
)
from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
    safe_tree_sha256,
    sha256_file,
    tensor_state_sha256,
)
from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import (
    load_composite_mjlab_contract,
)
import gear_sonic.utils.g1_true23_native124_selected_v2_ankle_evaluation as evaluation

DIAGNOSTIC_KIND = "training_reset_seam_diagnostic"
DIAGNOSTIC_SCHEMA_VERSION = 1
MAX_TRANSITIONS = 500
DEVICE = evaluation.DEVICE


@dataclass(frozen=True)
class TrainingResetSeamDiagnosticRequest:
    """Hash-bound request sharing the existing evaluator's path contract."""

    repository_root: Path
    warm_checkpoint: Path
    expected_warm_sha256: str
    output: Path

    def __post_init__(self) -> None:
        if any(not isinstance(value, Path) for value in (self.repository_root, self.warm_checkpoint, self.output)):
            raise TypeError("diagnostic paths must be pathlib.Path values")
        evaluation._require_lower_sha256(  # noqa: SLF001
            self.expected_warm_sha256,
            "expected_warm_sha256",
        )

    def as_warm_evaluation_request(self) -> evaluation.WarmEvaluationRequest:
        return evaluation.WarmEvaluationRequest(
            repository_root=self.repository_root,
            warm_checkpoint=self.warm_checkpoint,
            expected_warm_sha256=self.expected_warm_sha256,
            output=self.output,
        )

    @property
    def root(self) -> Path:
        return self.as_warm_evaluation_request().root

    @property
    def checkpoint(self) -> Path:
        return self.as_warm_evaluation_request().checkpoint

    @property
    def output_path(self) -> Path:
        return self.as_warm_evaluation_request().output_path


def diagnostic_scope() -> dict[str, Any]:
    """Return the permanent non-qualification boundary."""

    return {
        "classification": DIAGNOSTIC_KIND,
        "qualification_performed": False,
        "candidate_decision_emitted": False,
        "promotion_decision_emitted": False,
        "deployment_decision_emitted": False,
    }


def preflight_training_reset_seam_diagnostic(
    request: TrainingResetSeamDiagnosticRequest,
) -> dict[str, Any]:
    """Bind exact existing evaluator inputs without constructing a simulator."""

    if type(request) is not TrainingResetSeamDiagnosticRequest:
        raise TypeError("request must be exact TrainingResetSeamDiagnosticRequest")
    base = evaluation.preflight_warm_evaluation(request.as_warm_evaluation_request())
    source_path = Path(__file__).resolve(strict=True)
    evaluator_path = Path(evaluation.__file__).resolve(strict=True)
    return {
        "schema": "g1_true23_native124_selected_v2_reset_seam_preflight_v1",
        "kind": DIAGNOSTIC_KIND,
        "ready": True,
        "warm_checkpoint": base["warm_checkpoint"],
        "selected_source": base["selected_source"],
        "motion": base["motion"],
        "output": base["output"],
        "runner_agent_config_sha256": base["runner_agent_config_sha256"],
        "rsl_runtime": base["rsl_runtime"],
        "diagnostic_sources": {
            "reset_seam_utility": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
            "existing_warm_evaluator": {
                "path": str(evaluator_path),
                "sha256": sha256_file(evaluator_path),
            },
        },
        "fixed": {
            "seed": FIXED_SEED,
            "device": DEVICE,
            "num_envs": 1,
            "prime_q9": CAUSAL_HISTORY_ANCHOR_INDEX,
            "fixed_warmup_steps": 0,
            "action_substitution": False,
            "first_action": "deterministic_warm_actor_mean",
            "max_transitions": MAX_TRANSITIONS,
        },
        "diagnostic_scope": diagnostic_scope(),
        "safety": {
            "simulator_constructed": False,
            "simulator_steps": 0,
            "training_updates": 0,
            "hardware_authorized": False,
            "network_used": False,
        },
    }


def _finite_single_batch_xyz(value: Any, body_count: int, context: str) -> np.ndarray:
    if (
        type(value) is not torch.Tensor
        or value.shape != (1, body_count, 3)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{context} must be finite floating [1,{body_count},3]")
    return value.detach().cpu().contiguous().numpy().copy()[0]


def capture_exact_ee_position_errors(raw_env: Any) -> list[dict[str, Any]]:
    """Capture the exact named-body z-error used by ``ee_body_pos``."""

    termination = raw_env.cfg.terminations.get("ee_body_pos")
    if termination is None or not isinstance(termination.params, Mapping):
        raise ValueError("diagnostic requires configured ee_body_pos termination")
    names = termination.params.get("body_names")
    threshold = termination.params.get("threshold")
    if (
        type(names) is not tuple
        or not names
        or len(set(names)) != len(names)
        or any(type(name) is not str or not name for name in names)
    ):
        raise ValueError("ee_body_pos body_names must be a unique nonempty tuple")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (float, int))
        or not math.isfinite(float(threshold))
        or float(threshold) <= 0.0
    ):
        raise ValueError("ee_body_pos threshold must be finite positive")
    threshold = float(threshold)

    command_name = termination.params.get("command_name")
    if type(command_name) is not str or not command_name:
        raise ValueError("ee_body_pos command_name must be nonempty string")
    command = raw_env.command_manager.get_term(command_name)
    command_body_names = tuple(command.cfg.body_names)
    if any(command_body_names.count(name) != 1 for name in names):
        raise ValueError("ee_body_pos body is not unique in motion command")
    reference = _finite_single_batch_xyz(
        command.body_pos_relative_w,
        len(command_body_names),
        "reference body positions",
    )
    measured = _finite_single_batch_xyz(
        command.robot_body_pos_w,
        len(command_body_names),
        "measured body positions",
    )

    result = []
    for name in names:
        index = command_body_names.index(name)
        error = measured[index] - reference[index]
        absolute_z = float(abs(error[2]))
        result.append(
            {
                "name": name,
                "command_body_index": index,
                "reference_position_w_m": [float(item) for item in reference[index]],
                "measured_position_w_m": [float(item) for item in measured[index]],
                "error_measured_minus_reference_m": [float(item) for item in error],
                "error_norm_m": float(np.linalg.norm(error.astype(np.float64))),
                "absolute_z_error_m": absolute_z,
                "termination_threshold_m": threshold,
                "z_termination_breached": absolute_z > threshold,
            }
        )
    return result


def prove_reset_actor_observation(
    observations: Mapping[str, Any],
    reset_diagnostics: Any,
) -> dict[str, Any]:
    """Prove virtual-q9 reset history and zero previous action at actor input."""

    actor_observation = observations.get("actor")
    reset_mask = getattr(reset_diagnostics, "reset_virtual_torso_mask", None)
    previous = getattr(
        reset_diagnostics,
        "previous_effective_selected_raw_hardware",
        None,
    )
    if (
        type(reset_mask) is not torch.Tensor
        or reset_mask.shape != (1,)
        or reset_mask.dtype != torch.bool
        or not bool(reset_mask[0].item())
        or type(previous) is not torch.Tensor
        or previous.shape != (1, 23)
        or previous.dtype != torch.float32
        or torch.count_nonzero(previous).item() != 0
        or type(actor_observation) is not torch.Tensor
        or actor_observation.shape != (1, 124)
        or actor_observation.dtype != torch.float32
        or not bool(torch.isfinite(actor_observation).all())
        or torch.count_nonzero(actor_observation[:, -23:]).item() != 0
    ):
        raise RuntimeError("reset-seam virtual torso/zero-prior actor contract drift")
    return {
        "reset_virtual_torso_mask": True,
        "previous_effective_selected_raw_hardware_is_zero": True,
        "actor_previous_action_slice": [-23, None],
        "actor_previous_action_slice_is_zero": True,
    }


class _TerminalPreResetEeRecorder:
    """Capture terminal evidence before MJLab autoreset mutates state."""

    def __init__(self, raw_env: Any, velocity_limits: np.ndarray) -> None:
        self.raw_env = raw_env
        self.velocity_limits = velocity_limits
        self._original_reset = raw_env._reset_idx
        self._armed: dict[str, Any] | None = None
        self.captured: dict[str, Any] | None = None

        def observed_reset(_env: Any, env_ids: torch.Tensor | None = None) -> None:
            if self._armed is not None and int(_env.common_step_counter) > 0:
                if self.captured is not None:
                    raise RuntimeError("reset-seam recorder captured more than once")
                if env_ids is None or int(env_ids.numel()) != 1 or int(env_ids.detach().cpu().item()) != 0:
                    raise RuntimeError("reset-seam recorder expected only environment zero")
                self.captured = {
                    **self._armed,
                    "episode_length_pre_reset": int(_env.episode_length_buf[0].detach().cpu().item()),
                    "termination_names": evaluation._termination_names(_env),  # noqa: SLF001
                    "is_timeout": bool(_env.termination_manager.time_outs[0].detach().cpu().item()),
                    "is_terminated": bool(_env.termination_manager.terminated[0].detach().cpu().item()),
                    "ee_body_position_errors": capture_exact_ee_position_errors(_env),
                    "evidence": evaluation._step_evidence(  # noqa: SLF001
                        _env,
                        self.velocity_limits,
                    ),
                }
            self._original_reset(env_ids)

        raw_env._reset_idx = MethodType(observed_reset, raw_env)

    def arm(self, *, transition: int, q9_before: int) -> None:
        if self._armed is not None:
            raise RuntimeError("reset-seam recorder already armed")
        self._armed = {
            "transition": transition,
            "q9_before": q9_before,
        }
        self.captured = None

    def finish(self, *, done: bool) -> dict[str, Any] | None:
        if self._armed is None:
            raise RuntimeError("reset-seam recorder was not armed")
        captured = self.captured
        self._armed = None
        self.captured = None
        if done != (captured is not None):
            raise RuntimeError("reset-seam terminal capture disagrees with wrapper done")
        return captured

    def restore(self) -> None:
        self.raw_env._reset_idx = self._original_reset


def run_training_reset_seam_diagnostic(
    request: TrainingResetSeamDiagnosticRequest,
) -> dict[str, Any]:
    """Replay deterministic actor immediately from q9=9 until first done."""

    preflight = preflight_training_reset_seam_diagnostic(request)
    root = request.root
    checkpoint = request.checkpoint
    output = request.output_path
    motion = (root / DAD_DANCE_RELATIVE_PATH).resolve(strict=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FIXED_GPU)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    runtime_sources = evaluation._bind_evaluation_runtime_sources(root)  # noqa: SLF001
    if resolve_rsl_runtime_binding() != preflight["rsl_runtime"]:
        raise RuntimeError("RSL runtime changed after reset-seam preflight")

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
        raise RuntimeError("reset-seam diagnostic requires exactly fixed visible CUDA device 0")
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
    agent_cfg = evaluation._evaluation_agent_config(root, motion, output)  # noqa: SLF001
    env_cfg = make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=str(motion),
        num_envs=1,
        play=False,
    )
    env_cfg.seed = FIXED_SEED
    task_audit = audit_native124_selected_v2_ankle_task_env_cfg(env_cfg)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    recorder: _TerminalPreResetEeRecorder | None = None
    try:
        wrapped = Native124SelectedV2CausalAdaptationWrapper(env, clip_actions=None)
        prime = prime_native124_selected_v2_causal_adaptation_environment(wrapped)
        command = env.command_manager.get_term("motion")
        prime_q9 = evaluation._q9(command)  # noqa: SLF001
        if (
            prime_q9 != CAUSAL_HISTORY_ANCHOR_INDEX
            or int(env.common_step_counter) != 0
            or int(env._sim_step_counter) != 0
        ):
            raise RuntimeError("reset-seam diagnostic did not prime without physics at q9=9")
        observations = wrapped.get_observations()
        reset_buffer_proof = prove_reset_actor_observation(
            observations,
            wrapped.diagnostics,
        )

        # Match training runner construction without consuming a simulator action.
        torch.manual_seed(FIXED_SEED)
        torch.cuda.manual_seed_all(FIXED_SEED)
        runner = MjlabOnPolicyRunner(wrapped, copy.deepcopy(agent_cfg), None, DEVICE)
        integration = evaluation.configure_hash_locked_warm_evaluation_runner(
            runner,
            repository_root=root,
            warm_checkpoint=checkpoint,
            expected_warm_sha256=request.expected_warm_sha256,
        )
        actor = integration.actor
        actor.eval()
        integration.assert_frozen_invariants()

        actor_hash_before = tensor_state_sha256(actor.state_dict())
        critic_hash_before = tensor_state_sha256(integration.critic.state_dict())
        optimizer_hash_before = safe_tree_sha256(
            runner.alg.optimizer.state_dict(),
            context="reset-seam optimizer before diagnostic",
        )
        recorder = _TerminalPreResetEeRecorder(env, velocity_limits)
        accumulator = evaluation.RolloutEvidenceAccumulator()
        first_done: dict[str, Any] | None = None
        deterministic_actor_checks = 0
        attempted = 0
        first_actor_q9: int | None = None

        for transition in range(MAX_TRANSITIONS):
            q9_before = evaluation._q9(command)  # noqa: SLF001
            if first_actor_q9 is None:
                first_actor_q9 = q9_before
            action = evaluation._deterministic_actor_action(  # noqa: SLF001
                actor,
                observations,
            )
            deterministic_actor_checks += 1
            recorder.arm(transition=transition, q9_before=q9_before)
            observations, _, dones, extras = wrapped.step(action)
            attempted += 1
            done = bool(int(dones[0].detach().cpu().item()))
            terminal = recorder.finish(done=done)
            if done:
                if terminal is None:
                    raise RuntimeError("reset-seam terminal lacks pre-reset evidence")
                evidence = terminal.pop("evidence")
                if type(evidence) is not evaluation.StepEvidence:
                    raise RuntimeError("reset-seam terminal evidence type drift")
                accumulator.add(evidence)
                names = evaluation._extra_termination_names(extras)  # noqa: SLF001
                if names != terminal["termination_names"]:
                    raise RuntimeError("reset-seam terminal extras disagree with pre-reset terms")
                first_done = {
                    "policy_transition": transition,
                    "q9_before": q9_before,
                    "q10_proof_before": q9_before + 1,
                    "q9_after_autoreset": evaluation._q9(command),  # noqa: SLF001
                    **terminal,
                    "evidence": evidence.to_dict(),
                }
                break
            if terminal is not None or evaluation._extra_termination_names(extras):  # noqa: SLF001
                raise RuntimeError("reset-seam nonterminal contains termination evidence")
            if evaluation._q9(command) != q9_before + 1:  # noqa: SLF001
                raise RuntimeError("reset-seam q9 discontinuity")
            accumulator.add(evaluation._step_evidence(env, velocity_limits))  # noqa: SLF001

        integration.assert_frozen_invariants()
        actor_hash_after = tensor_state_sha256(actor.state_dict())
        critic_hash_after = tensor_state_sha256(integration.critic.state_dict())
        optimizer_hash_after = safe_tree_sha256(
            runner.alg.optimizer.state_dict(),
            context="reset-seam optimizer after diagnostic",
        )
        storage_step = getattr(getattr(runner.alg, "storage", None), "step", None)
        frozen = {
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
            frozen["actor_unchanged"]
            and frozen["critic_unchanged"]
            and frozen["optimizer_unchanged"]
            and integration.optimizer_steps == 0
            and runner.current_learning_iteration == 0
            and storage_step == 0
        ):
            raise RuntimeError("training state changed during reset-seam diagnostic")
        if first_actor_q9 != CAUSAL_HISTORY_ANCHOR_INDEX:
            raise RuntimeError("reset-seam actor was not applied immediately at q9=9")
        if resolve_rsl_runtime_binding() != preflight["rsl_runtime"]:
            raise RuntimeError("RSL runtime changed during reset-seam diagnostic")

        report = {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "kind": DIAGNOSTIC_KIND,
            "completed": True,
            "diagnostic_scope": diagnostic_scope(),
            "preflight": preflight,
            "runtime_sources": runtime_sources,
            "task_audit": task_audit,
            "prime": prime,
            "reset_seam": {
                "prime_q9": prime_q9,
                "first_deterministic_actor_q9": first_actor_q9,
                "fixed_warmup_steps": 0,
                "action_substitution": False,
                "first_action": "deterministic_warm_actor_mean",
                "reset_buffer_proof": reset_buffer_proof,
            },
            "rollout": {
                "max_transitions": MAX_TRANSITIONS,
                "attempted_transitions_including_done": attempted,
                "deterministic_actor_checks": deterministic_actor_checks,
                "first_done": first_done,
                "summary_including_terminal_transition": accumulator.report(),
            },
            "frozen_training_state_proof": frozen,
            "safety": {
                "simulator_only": True,
                "training_performed": False,
                "hardware_or_network_commands_performed": False,
                "deployment_authorized": False,
            },
        }
        return evaluation._json_safe(report)  # noqa: SLF001
    finally:
        if recorder is not None:
            recorder.restore()
        env.close()


def write_training_reset_seam_diagnostic_new(
    request: TrainingResetSeamDiagnosticRequest,
    report: Mapping[str, Any],
) -> Path:
    """Publish one diagnostic report with exclusive creation."""

    if type(request) is not TrainingResetSeamDiagnosticRequest:
        raise TypeError("request must be exact TrainingResetSeamDiagnosticRequest")
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
    request: TrainingResetSeamDiagnosticRequest,
) -> dict[str, Any]:
    """Return a non-qualifying failure artifact."""

    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "kind": DIAGNOSTIC_KIND,
        "completed": False,
        "diagnostic_scope": diagnostic_scope(),
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
        },
    }


__all__ = [
    "DEVICE",
    "DIAGNOSTIC_KIND",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "MAX_TRANSITIONS",
    "TrainingResetSeamDiagnosticRequest",
    "capture_exact_ee_position_errors",
    "diagnostic_scope",
    "failure_report",
    "preflight_training_reset_seam_diagnostic",
    "prove_reset_actor_observation",
    "run_training_reset_seam_diagnostic",
    "write_training_reset_seam_diagnostic_new",
]
