"""Fail-closed mixed-controller recovery diagnostic for the scale-2 SONIC student."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
from gear_sonic.envs.mjlab.sonic_true23_student_qualification import (
    audit_sonic_true23_student_qualification_env_cfg,
    make_sonic_true23_student_qualification_env_cfg,
)
from gear_sonic.scripts import train_g1_true23_sonic_survival_length_score as scale2
from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs
from gear_sonic.utils import (
    g1_true23_sonic_student_closed_loop_qualification as student,
    g1_true23_sonic_student_teacher_recovery as legacy,
)
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import (
    compose_checkpoint21204_teacher_action,
    load_composite_mjlab_contract,
)
import gear_sonic.utils.g1_true23_native124_selected_v2_ankle_evaluation as evaluation

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path("gear_sonic/config/sim_validation/g1_true23_sonic_scale2_teacher_recovery_v1.json")
CONTRACT_SHA256 = "40471389054f57278e46485a674e0f393d8a8af290d6b3797d8a38aec58a25dc"
CONTRACT_V2_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_scale2_teacher_recovery_v2.json"
)
CONTRACT_V2_SHA256 = "c2f2598312a4dd3f71592b305c33e58f8f23fc401f5e8e49b7dcfe91c3070a54"
CONTRACT_V3_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_scale2_teacher_recovery_v3.json"
)
CONTRACT_V3_SHA256 = "59d6b8e5ab528aad32b5f06e162b3a3d8b8ce46a89003b97ed0abab9137cd631"
CONTRACT_V4_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_scale2_teacher_recovery_v4.json"
)
CONTRACT_V4_SHA256 = "97f801bc467110e834eeb5c368d8193bf26ca10f853af6c338ce1b9cdfd7e8a5"
MODES = ("q250", "q225", "q200", "q175", "q9")
MODE_STUDENT_TRANSITIONS = {"q250": 241, "q225": 216, "q200": 191, "q175": 166, "q9": 0}
TOTAL_TRANSITIONS = 510
ANCHOR_Q9 = 9
LAST_Q9 = 518
STRICT_RAW_ABS_MAX = 10.0
ACTION_DIM = 23


@dataclass(frozen=True)
class Scale2RecoveryWindow:
    mode: str
    student_transitions: int
    teacher_transitions: int

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unsupported scale2 recovery mode: {self.mode}")
        if self.student_transitions != MODE_STUDENT_TRANSITIONS[self.mode]:
            raise ValueError("scale2 recovery student transition count mismatch")
        if self.student_transitions + self.teacher_transitions != TOTAL_TRANSITIONS:
            raise ValueError("scale2 recovery partition mismatch")

    @property
    def teacher_first_q9(self) -> int:
        return ANCHOR_Q9 + self.student_transitions

    @property
    def student_last_q9(self) -> int:
        return self.teacher_first_q9 - 1

    def controller(self, transition: int) -> str:
        if (
            isinstance(transition, bool)
            or not isinstance(transition, int)
            or not 0 <= transition < TOTAL_TRANSITIONS
        ):
            raise ValueError("scale2 recovery transition outside window")
        return "student" if transition < self.student_transitions else "teacher"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "anchor_q9": ANCHOR_Q9,
            "last_q9": LAST_Q9,
            "transitions": TOTAL_TRANSITIONS,
            "student_transition_count": self.student_transitions,
            "student_first_q9": ANCHOR_Q9,
            "student_last_q9": self.student_last_q9,
            "teacher_transition_count": self.teacher_transitions,
            "teacher_first_q9": self.teacher_first_q9,
            "teacher_last_q9": LAST_Q9,
        }


def resolve_window(mode: str) -> Scale2RecoveryWindow:
    try:
        student_transitions = MODE_STUDENT_TRANSITIONS[mode]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unsupported scale2 recovery mode: {mode}") from error
    return Scale2RecoveryWindow(mode, student_transitions, TOTAL_TRANSITIONS - student_transitions)


@dataclass(frozen=True)
class Scale2RecoveryRequest:
    repository_root: Path
    output: Path
    mode: str
    runtime_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path) or not isinstance(self.output, Path):
            raise TypeError("scale2 recovery paths must be Path values")
        if self.runtime_seed is not None and (
            isinstance(self.runtime_seed, bool)
            or not isinstance(self.runtime_seed, int)
            or not 0 <= self.runtime_seed < 2**31
        ):
            raise ValueError("scale2 recovery runtime seed must be uint31")
        resolve_window(self.mode)

    @property
    def root(self) -> Path:
        root = self.repository_root.expanduser().resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("scale2 recovery repository root invalid")
        return root

    @property
    def output_path(self) -> Path:
        return evaluation._evaluation_output_path(self.root, self.output.expanduser())  # noqa: SLF001

    @property
    def window(self) -> Scale2RecoveryWindow:
        return resolve_window(self.mode)

    @property
    def seed(self) -> int:
        return fs.FIXED_SEED if self.runtime_seed is None else self.runtime_seed


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def load_contract(root: Path, mode: str | None = None) -> Mapping[str, Any]:
    q200 = mode == "q200"
    q175 = mode == "q175"
    q9 = mode == "q9"
    if q9:
        relative = CONTRACT_V4_RELATIVE_PATH
        expected_sha = CONTRACT_V4_SHA256
    elif q175:
        relative = CONTRACT_V3_RELATIVE_PATH
        expected_sha = CONTRACT_V3_SHA256
    elif q200:
        relative = CONTRACT_V2_RELATIVE_PATH
        expected_sha = CONTRACT_V2_SHA256
    else:
        relative = CONTRACT_RELATIVE_PATH
        expected_sha = CONTRACT_SHA256
    path = (root / relative).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != expected_sha:
        raise ValueError("scale2 recovery contract mismatch")
    contract = fs._strict_json(path, "scale2 recovery contract")
    required_zero = [
        "nonfinite_count",
        "q9_discontinuity_count",
        "raw_clip_required_count",
        "action_semantics_mismatch_count",
        "teacher_parity_violation_count",
        "teacher_composite_mismatch_count",
        "selected_state_mismatch_count",
        "hard_safety_violation_count",
        "soft_safety_warning_count",
    ]
    common_valid = bool(
        contract.get("source_student", {}).get("checkpoint_sha256") == scale2.SOURCE_CHECKPOINT_SHA256
        and contract.get("source_student", {}).get("policy_state_sha256") == scale2.SOURCE_POLICY_SHA256
        and contract.get("gates", {}).get("required_zero_counts") == required_zero
    )
    if q9:
        window = contract.get("window", {})
        evidence = contract.get("evidence_basis", {})
        valid = bool(
            contract.get("schema_version") == 4
            and contract.get("kind") == "g1_true23_selected_teacher_actuated_full_episode_contract_v1"
            and window.get("mode") == "q9"
            and window.get("student_transitions") == 0
            and window.get("teacher_transitions") == 510
            and window.get("teacher_first_q9") == 9
            and window.get("selected_previous_action_exact_zero_at_reset") is True
            and evidence.get("q175_failure_q9") == 281
            and evidence.get("q175_teacher_actions_before_failure") == 107
            and evidence.get("q175_terminal_body") == "right_wrist_roll_rubber_hand"
            and evidence.get("selected_teacher_nominal500_qualified") is True
        )
        for name in ("q175_failure", "selected_teacher_nominal500"):
            artifact = (root / str(evidence[f"{name}_relative_path"])).resolve(strict=True)
            valid = bool(
                valid and not artifact.is_symlink() and sha256_file(artifact) == evidence[f"{name}_sha256"]
            )
    elif q175:
        window = contract.get("window", {})
        evidence = contract.get("evidence_basis", {})
        valid = bool(
            contract.get("schema_version") == 3
            and contract.get("kind") == "g1_true23_sonic_scale2_teacher_recovery_contract_v3"
            and window.get("mode") == "q175"
            and window.get("student_transitions") == 166
            and window.get("teacher_transitions") == 344
            and window.get("teacher_first_q9") == 175
            and evidence.get("q200_failure_q9") == 214
            and evidence.get("q200_teacher_actions_before_failure") == 15
            and evidence.get("q200_terminal_body") == "left_wrist_roll_rubber_hand"
            and evidence.get("q175_margin_before_earliest_failure_frames") == 39
            and evidence.get("q175_teacher_actions_before_earliest_failure") == 40
        )
        artifact = (root / str(evidence["q200_failure_relative_path"])).resolve(strict=True)
        valid = bool(
            valid and not artifact.is_symlink() and sha256_file(artifact) == evidence["q200_failure_sha256"]
        )
    elif q200:
        window = contract.get("window", {})
        evidence = contract.get("evidence_basis", {})
        valid = bool(
            contract.get("schema_version") == 2
            and contract.get("kind") == "g1_true23_sonic_scale2_teacher_recovery_contract_v2"
            and window.get("mode") == "q200"
            and window.get("student_transitions") == 191
            and window.get("teacher_transitions") == 319
            and window.get("teacher_first_q9") == 200
            and evidence.get("earliest_noisy_failure_q9") == 214
            and evidence.get("q200_margin_before_earliest_failure_frames") == 14
        )
        for name in ("q250_campaign_failure", "clean_cross_seed_probe"):
            artifact = (root / str(evidence[f"{name}_relative_path"])).resolve(strict=True)
            valid = valid and not artifact.is_symlink() and sha256_file(artifact) == evidence[f"{name}_sha256"]
    else:
        modes = contract.get("window", {}).get("modes", {})
        valid = bool(
            contract.get("kind") == "g1_true23_sonic_scale2_teacher_recovery_contract_v1"
            and contract.get("seed") == fs.FIXED_SEED
            and list(modes) == ["q250", "q225"]
            and modes.get("q250", {}).get("student_transitions") == 241
            and modes.get("q225", {}).get("student_transitions") == 216
        )
    if not common_valid or not valid:
        raise ValueError("scale2 recovery contract semantic mismatch")
    return contract


def _legacy_preflight_request(root: Path) -> legacy.RecoveryRequest:
    return legacy.RecoveryRequest(
        repository_root=root,
        candidate_manifest=root / legacy.CURRENT_CANDIDATE_MANIFEST_RELATIVE_PATH,
        expected_candidate_manifest_sha256=legacy.CURRENT_CANDIDATE_MANIFEST_SHA256,
        output=root / "artifacts/g1_true23/.scale2_recovery_preflight_unused.json",
        mode="cutoff50",
    )


def _new_source_binding(root: Path) -> dict[str, Any]:
    relatives = (
        CONTRACT_RELATIVE_PATH,
        CONTRACT_V2_RELATIVE_PATH,
        CONTRACT_V3_RELATIVE_PATH,
        CONTRACT_V4_RELATIVE_PATH,
        Path("gear_sonic/utils/g1_true23_sonic_scale2_teacher_recovery.py"),
        Path("gear_sonic/scripts/diagnose_g1_true23_sonic_scale2_teacher_recovery.py"),
        Path("gear_sonic/scripts/train_g1_true23_sonic_survival_length_score.py"),
    )
    files = []
    for relative in relatives:
        path = (root / relative).resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"scale2 recovery source invalid: {relative.as_posix()}")
        files.append({"path": relative.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"files": files, "binding_sha256": _canonical_sha256({"files": files})}


def _preflight_internal(request: Scale2RecoveryRequest) -> dict[str, Any]:
    root = request.root
    contract = load_contract(root, request.mode)
    contract_sha256 = {
        "q9": CONTRACT_V4_SHA256,
        "q175": CONTRACT_V3_SHA256,
        "q200": CONTRACT_V2_SHA256,
    }.get(request.mode, CONTRACT_SHA256)
    base = legacy._preflight_internal(_legacy_preflight_request(root))  # noqa: SLF001
    if base.get("ready") is not True:
        raise RuntimeError("scale2 recovery base preflight not ready")
    if sha256_file(scale2.SOURCE_CHECKPOINT_PATH) != scale2.SOURCE_CHECKPOINT_SHA256:
        raise ValueError("scale2 recovery checkpoint SHA mismatch")
    policy = scale2._load_source_policy()  # noqa: SLF001
    observed = inspect_true23_policy_state({"policy_state_dict": policy}, reference_profile=fs.REFERENCE_PROFILE)
    if observed != scale2.SOURCE_POLICY_SHA256:
        raise ValueError("scale2 recovery policy identity mismatch")
    return {
        "ready": True,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "base": base,
        "policy_state": policy,
        "policy_state_sha256": observed,
        "new_sources": _new_source_binding(root),
    }


def preflight(request: Scale2RecoveryRequest) -> dict[str, Any]:
    try:
        value = _preflight_internal(request)
        base = value["base"]
        student_base = base["student"]
        teacher = base["teacher"]
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_scale2_teacher_recovery_preflight_v1",
            "ready": True,
            "mode": request.mode,
            "window": request.window.to_dict(),
            "contract_sha256": value["contract_sha256"],
            "student": {
                "checkpoint_sha256": scale2.SOURCE_CHECKPOINT_SHA256,
                "policy_state_sha256": scale2.SOURCE_POLICY_SHA256,
                "fixed_inputs": student_base["fixed_inputs"],
            },
            "teacher": {
                "checkpoint_sha256": teacher.checkpoint_sha256,
                "actor_state_sha256": teacher.actor_state_sha256,
                "onnx_sha256": teacher.onnx_sha256,
            },
            "legacy_sources_binding_sha256": base["sources"]["binding_sha256"],
            "new_sources": value["new_sources"],
            "external_sources_binding_sha256": base["external_sources"]["binding_sha256"],
            "physical_assets_manifest_sha256": base["assets"]["manifest_sha256"],
            "simulator_constructed": False,
            "simulator_steps": 0,
            "student_queries": 0,
            "teacher_queries": 0,
            "teacher_labels_admitted": 0,
            "training_performed": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_scale2_teacher_recovery_preflight_v1",
            "ready": False,
            "mode": request.mode,
            "error": {"type": type(error).__name__, "message": str(error)},
            "simulator_constructed": False,
            "simulator_steps": 0,
            "student_queries": 0,
            "teacher_queries": 0,
            "teacher_labels_admitted": 0,
            "training_performed": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }


def _support_fallback() -> dict[str, Any]:
    return {
        "minimum_base_height_m": -math.inf,
        "maximum_base_tilt_rad": math.inf,
        "maximum_joint_velocity_ratio": math.inf,
        "maximum_tracking_rmse_rad": math.inf,
        "maximum_plain_sonic_raw_native_abs": math.inf,
        "hard_safety_violation_count": 1,
        "soft_safety_warning_count": 1,
        "raw_clip_required_count": 1,
        "action_semantics_mismatch_count": 1,
    }


def _with_rollout_safety_counts(support: Mapping[str, Any], rollout_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **support,
        "hard_safety_violation_count": int(rollout_summary["hard_safety_violation_count"]),
        "soft_safety_warning_count": int(rollout_summary["soft_safety_warning_count"]),
    }


def assess(
    *,
    window: Scale2RecoveryWindow,
    started: int,
    completed: int,
    student_queries: int,
    teacher_queries: int,
    controller_counts: Mapping[str, int],
    first_done: Mapping[str, Any] | None,
    history_checks: int,
    history_shifts: int,
    support: Mapping[str, Any],
    teacher_parity: Mapping[str, Any],
    teacher_composite: Mapping[str, Any],
    selected_state: Mapping[str, Any],
    bindings_unchanged: bool,
    partial_failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected_final_timeout = bool(
        first_done is not None
        and first_done.get("transition") == TOTAL_TRANSITIONS - 1
        and first_done.get("q9_before") == LAST_Q9
        and first_done.get("termination_names") == ["time_out"]
        and first_done.get("is_timeout") is True
        and first_done.get("is_terminated") is False
        and first_done.get("q9_after_autoreset") == ANCHOR_Q9
    )
    zero_counts = {
        "nonfinite_count": int(support.get("nonfinite_count", 1)),
        "q9_discontinuity_count": int(support.get("q9_discontinuity_count", 1)),
        "raw_clip_required_count": int(support.get("raw_clip_required_count", 1)),
        "action_semantics_mismatch_count": int(support.get("action_semantics_mismatch_count", 1)),
        "teacher_parity_violation_count": int(teacher_parity.get("violation_count", 1)),
        "teacher_composite_mismatch_count": int(teacher_composite.get("mismatch_count", 1)),
        "selected_state_mismatch_count": int(selected_state.get("mismatch_count", 1)),
        "hard_safety_violation_count": int(support.get("hard_safety_violation_count", 1)),
        "soft_safety_warning_count": int(support.get("soft_safety_warning_count", 1)),
    }
    gates = {
        "exact_counts": bool(
            started == completed == student_queries == teacher_queries == TOTAL_TRANSITIONS
            and controller_counts == {"student": window.student_transitions, "teacher": window.teacher_transitions}
        ),
        "exact_final_timeout": expected_final_timeout,
        "history": history_checks == TOTAL_TRANSITIONS and history_shifts == TOTAL_TRANSITIONS - 1,
        "handoff": selected_state.get("handoff_observed") is True,
        "all_required_zero_counts": all(value == 0 for value in zero_counts.values()),
        "physical_support": bool(
            float(support.get("minimum_base_height_m", -math.inf)) >= 0.45
            and float(support.get("maximum_base_tilt_rad", math.inf)) <= 1.0
            and float(support.get("maximum_joint_velocity_ratio", math.inf)) <= 1.0
            and float(support.get("maximum_tracking_rmse_rad", math.inf)) <= 0.75
            and float(support.get("maximum_plain_sonic_raw_native_abs", math.inf)) < STRICT_RAW_ABS_MAX
        ),
        "bindings_unchanged": bindings_unchanged,
        "no_partial_failure": partial_failure is None,
    }
    passed = all(gates.values())
    return {
        "passed": passed,
        "verdict": "scale2_teacher_recovery_passed" if passed else "scale2_teacher_recovery_failed",
        "gates": gates,
        "required_zero_counts": zero_counts,
        "teacher_labels_admitted": 0,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def run(request: Scale2RecoveryRequest) -> dict[str, Any]:
    internal = _preflight_internal(request)
    root = request.root
    window = request.window
    base = internal["base"]
    student_base = base["student"]
    motion = Path(student_base["fixed_inputs"]["motion_path"])
    topology_contract = fs.load_task_space_ppo_contract(root)
    topology = (root / topology_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
        strict=True
    )
    runtime_seed = request.seed
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(student.FIXED_GPU),
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
        }
    )
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("scale2 recovery requires fixed visible CUDA device 0")
    random.seed(runtime_seed)
    np.random.seed(runtime_seed % (2**32))
    torch.manual_seed(runtime_seed)
    torch.cuda.manual_seed_all(runtime_seed)
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    actor = scale2._actor(internal["policy_state"], topology)  # noqa: SLF001
    actor.eval()
    composite_contract = load_composite_mjlab_contract(root)
    velocity_limits = np.asarray(
        composite_contract["nominal_gate"]["velocity_limit_hardware_radps"], dtype=np.float32
    )
    cfg = make_sonic_true23_student_qualification_env_cfg(
        motion_file=str(motion), num_envs=1, anchor_q9=ANCHOR_Q9, transitions=TOTAL_TRANSITIONS
    )
    cfg.seed = runtime_seed
    task_audit = audit_sonic_true23_student_qualification_env_cfg(
        cfg, expected_anchor_q9=ANCHOR_Q9, expected_transitions=TOTAL_TRANSITIONS
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=student.DEVICE)
    recorder: legacy._RecoveryTerminalRecorder | None = None  # noqa: SLF001
    teacher: legacy._ExactTeacherPair | None = None  # noqa: SLF001
    started = 0
    completed = 0
    student_queries = 0
    controller_counts = {"student": 0, "teacher": 0}
    partial_failure: dict[str, Any] | None = None
    cleanup_errors: list[str] = []
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        if wrapped.clip_actions is not None or wrapped.max_episode_length != TOTAL_TRANSITIONS:
            raise RuntimeError("scale2 recovery wrapper mismatch")
        prime = prime_sonic_true23_training_environment(wrapped)
        observations = wrapped.get_observations()
        command = env.command_manager.get_term("motion")
        if evaluation._q9(command) != ANCHOR_Q9:  # noqa: SLF001
            raise RuntimeError("scale2 recovery prime q9 mismatch")
        selected_state = legacy._Selected124State(env)  # noqa: SLF001
        reset_seam = legacy._reset_seam(env, observations, selected_state)  # noqa: SLF001
        teacher = legacy._ExactTeacherPair(base["teacher"], student.DEVICE)  # noqa: SLF001
        recorder = legacy._RecoveryTerminalRecorder(env, velocity_limits)  # noqa: SLF001
        rollout = evaluation.RolloutEvidenceAccumulator()
        action_accumulator = student._ActionSemanticsAccumulator(TOTAL_TRANSITIONS)  # noqa: SLF001
        composite_accumulator = legacy._CompositeAccumulator(window.teacher_transitions)  # noqa: SLF001
        previous_history: Mapping[str, torch.Tensor] | None = None
        history_checks = 0
        history_shifts = 0
        q9_discontinuity_count = 0
        nonfinite_count = 0
        selected_state_mismatch_count = 0
        selected_round_trip_max = 0.0
        handoff_observed = False
        first_done: dict[str, Any] | None = None
        autoreset_history: dict[str, Any] | None = None
        for transition in range(TOTAL_TRANSITIONS):
            q9 = evaluation._q9(command)  # noqa: SLF001
            expected_q9 = ANCHOR_Q9 + transition
            if q9 != expected_q9:
                q9_discontinuity_count += 1
                partial_failure = legacy._compact_partial_failure(  # noqa: SLF001
                    stage="pre_action_q9_continuity",
                    transition=transition,
                    q9=q9,
                    error=RuntimeError(f"expected q9 {expected_q9}, observed {q9}"),
                )
                break
            controller = window.controller(transition)
            episode_before = int(env.episode_length_buf[0].detach().cpu().item())
            common_before = int(env.common_step_counter)
            sim_before = int(env._sim_step_counter)
            command_counter_before = int(command.command_counter[0].detach().cpu().item())
            try:
                student._policy_history_proof(env, observations, transition)  # noqa: SLF001
                snapshot = student._policy_history_snapshot(env)  # noqa: SLF001
                if previous_history is not None:
                    student._assert_history_shift(previous_history, snapshot, transition)  # noqa: SLF001
                    history_shifts += 1
                previous_history = snapshot
                history_checks += 1
                with torch.inference_mode():
                    student_tensor = actor(observations, stochastic_output=False)
                if student_tensor.shape != (1, ACTION_DIM) or not bool(torch.isfinite(student_tensor).all()):
                    raise RuntimeError("scale2 student action nonfinite or wrong shape")
                student_raw = student_tensor.detach().cpu().numpy()[0].astype(np.float32, copy=True)
                student_queries += 1
                selected124, pre_step_torso = selected_state.build()
                teacher_raw, _ = teacher.infer(selected124)
                teacher_composite = compose_checkpoint21204_teacher_action(teacher_raw, repository_root=root)
                teacher_plain = teacher_composite.teacher_action_native
                behavior_raw = student_raw if controller == "student" else teacher_plain
                if float(np.max(np.abs(behavior_raw))) >= STRICT_RAW_ABS_MAX:
                    raise RuntimeError("scale2 recovery behavior requires forbidden raw clipping")
            except Exception as error:
                nonfinite_count += int("finite" in str(error).lower())
                partial_failure = legacy._compact_partial_failure(  # noqa: SLF001
                    stage="dual_policy_pre_action_inference", transition=transition, q9=q9, error=error
                )
                break
            action_tensor = torch.from_numpy(behavior_raw.reshape(1, ACTION_DIM)).to(
                device=student.DEVICE, dtype=torch.float32
            )
            recorder.arm(transition=transition, q9_before=q9, controller=controller, behavior_raw=behavior_raw)
            started += 1
            try:
                observations, _, dones, extras = wrapped.step(action_tensor)
            except Exception as error:
                partial_failure = legacy._compact_partial_failure(  # noqa: SLF001
                    stage="environment_step", transition=transition, q9=q9, error=error
                )
                break
            completed += 1
            controller_counts[controller] += 1
            done = bool(int(dones[0].detach().cpu().item()))
            terminal = recorder.finish(done=done)
            if done:
                if terminal is None or terminal.get("capture_errors"):
                    partial_failure = legacy._compact_partial_failure(  # noqa: SLF001
                        stage="terminal_pre_autoreset_capture",
                        transition=transition,
                        q9=q9,
                        error=RuntimeError("scale2 recovery terminal capture incomplete"),
                    )
                    break
                semantics = terminal["action_semantics"]
                evidence = terminal["step_evidence"]
            else:
                if terminal is not None or evaluation._extra_termination_names(extras):  # noqa: SLF001
                    raise RuntimeError("scale2 recovery nonterminal contains termination evidence")
                semantics = student._action_semantics(env, behavior_raw)  # noqa: SLF001
                evidence = evaluation._step_evidence(env, velocity_limits)  # noqa: SLF001
            action_accumulator.add(semantics)
            rollout.add(evidence)
            chain = legacy._mapping(semantics.get("chain"), "scale2 recovery action chain")  # noqa: SLF001
            final_target = np.asarray(chain["final_target_hardware"], dtype=np.float32)
            if controller == "teacher":
                composite_accumulator.add(semantics, teacher_composite)
            if transition == window.student_transitions:
                handoff_observed = bool(
                    controller == "teacher"
                    and selected_state.update_count == window.student_transitions
                    and q9 == window.teacher_first_q9
                )
            if done:
                autoreset_history = student._policy_history_proof(env, observations, 0)  # noqa: SLF001
                selected_autoreset = selected_state.synchronize_autoreset()
                first_done = legacy._terminal_compact(  # noqa: SLF001
                    terminal,
                    evaluation._q9(command),  # noqa: SLF001
                    autoreset_history,
                    selected_autoreset,
                )
                break
            q9_after = evaluation._q9(command)  # noqa: SLF001
            if (
                q9_after != q9 + 1
                or int(env.episode_length_buf[0].detach().cpu().item()) != episode_before + 1
                or int(env.common_step_counter) != common_before + 1
                or int(env._sim_step_counter) <= sim_before
                or int(command.command_counter[0].detach().cpu().item()) != command_counter_before
                or student._command_resampled(command)  # noqa: SLF001
            ):
                q9_discontinuity_count += 1
                partial_failure = legacy._compact_partial_failure(  # noqa: SLF001
                    stage="post_action_continuity",
                    transition=transition,
                    q9=q9,
                    error=RuntimeError("scale2 recovery simulator continuity drift"),
                )
                break
            state_proof = selected_state.update_nonterminal(pre_step_torso, final_target)
            selected_round_trip_max = max(
                selected_round_trip_max, float(state_proof["maximum_target_round_trip_absolute_error"])
            )
            selected_state_mismatch_count += int(state_proof["passed"] is not True)
        rollout_summary = rollout.report()
        action_report = action_accumulator.report()
        teacher_parity = teacher.report()
        teacher_composite_report = composite_accumulator.report()
        expected_final_timeout = bool(
            first_done is not None
            and first_done.get("transition") == TOTAL_TRANSITIONS - 1
            and first_done.get("q9_before") == LAST_Q9
            and first_done.get("termination_names") == ["time_out"]
        )
        support = (
            _with_rollout_safety_counts(
                student._support_summary(  # noqa: SLF001
                    rollout_summary,
                    expected_final_timeout=expected_final_timeout,
                    q9_discontinuity_count=q9_discontinuity_count,
                    nonfinite_count=nonfinite_count,
                    action_semantics=action_report,
                ),
                rollout_summary,
            )
            if rollout_summary["transition_count"]
            else _support_fallback()
        )
        frozen_after = student._snapshot_preflight_bound_files(base["frozen_specs"])  # noqa: SLF001
        frozen_models = student._frozen_input_evidence(base["frozen_before"], frozen_after)  # noqa: SLF001
        bindings_unchanged = bool(
            legacy.executed_recovery_source_binding(root) == base["sources"]
            and legacy.external_runtime_source_binding(root) == base["external_sources"]
            and student.physical_model_asset_binding(root) == base["assets"]
            and _new_source_binding(root) == internal["new_sources"]
            and frozen_models.get("all_preflight_bound_inputs_unchanged") is True
            and load_contract(root, request.mode) == internal["contract"]
        )
        selected_report = {
            "build_count": selected_state.build_count,
            "nonterminal_update_count": selected_state.update_count,
            "mismatch_count": selected_state_mismatch_count,
            "maximum_target_round_trip_absolute_error": selected_round_trip_max,
            "reset_previous_selected_raw_is_exact_zero": selected_state.reset_previous_zero,
            "handoff_observed": handoff_observed,
            "student_teacher_handoff_resets_state": False,
        }
        qualification = assess(
            window=window,
            started=started,
            completed=completed,
            student_queries=student_queries,
            teacher_queries=teacher.query_count,
            controller_counts=controller_counts,
            first_done=first_done,
            history_checks=history_checks,
            history_shifts=history_shifts,
            support=support,
            teacher_parity=teacher_parity,
            teacher_composite=teacher_composite_report,
            selected_state=selected_report,
            bindings_unchanged=bindings_unchanged,
            partial_failure=partial_failure,
        )
        after_policy = inspect_true23_policy_state(
            {"policy_state_dict": actor.export_true23_policy_state()}, reference_profile=fs.REFERENCE_PROFILE
        )
        if after_policy != scale2.SOURCE_POLICY_SHA256:
            raise RuntimeError("scale2 recovery student policy mutated")
        report = {
            "schema_version": 1,
            "kind": "g1_true23_sonic_scale2_teacher_recovery_diagnostic_v1",
            "contract_sha256": internal["contract_sha256"],
            "mode": window.mode,
            "effective_seed": runtime_seed,
            "window": window.to_dict(),
            "student": {
                "checkpoint_sha256": scale2.SOURCE_CHECKPOINT_SHA256,
                "policy_state_sha256": scale2.SOURCE_POLICY_SHA256,
                "query_count": student_queries,
            },
            "teacher": {
                "checkpoint_sha256": base["teacher"].checkpoint_sha256,
                "actor_state_sha256": base["teacher"].actor_state_sha256,
                "onnx_sha256": base["teacher"].onnx_sha256,
                "parity": teacher_parity,
                "query_count": teacher.query_count,
            },
            "task_audit": task_audit,
            "prime": prime,
            "reset_seam": reset_seam,
            "simulator_step_calls_started": started,
            "completed_transitions": completed,
            "controller_counts": controller_counts,
            "first_done": first_done,
            "history": {
                "check_count": history_checks,
                "shift_count": history_shifts,
                "autoreset": autoreset_history,
            },
            "rollout": rollout_summary,
            "support_summary": support,
            "action_semantics": action_report,
            "teacher_composite_action_chain": teacher_composite_report,
            "selected124_external_state": selected_report,
            "bindings_unchanged": bindings_unchanged,
            "partial_failure": partial_failure,
            "qualification": qualification,
            "teacher_labels_admitted": 0,
            "training_arrays_present": False,
            "training_performed": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
        legacy._assert_publication_boundary(report)  # noqa: SLF001
        return evaluation._json_safe(report)  # noqa: SLF001
    finally:
        if recorder is not None:
            try:
                recorder.restore()
            except Exception as error:
                cleanup_errors.append(f"recorder_restore:{type(error).__name__}:{error}")
        try:
            env.close()
        except Exception as error:
            cleanup_errors.append(f"environment_close:{type(error).__name__}:{error}")
        if cleanup_errors:
            raise RuntimeError(";".join(cleanup_errors))


__all__ = [
    "CONTRACT_SHA256",
    "MODES",
    "Scale2RecoveryRequest",
    "Scale2RecoveryWindow",
    "assess",
    "load_contract",
    "preflight",
    "resolve_window",
    "run",
]
