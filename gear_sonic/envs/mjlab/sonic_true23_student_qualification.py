"""Nominal closed-loop environment for one raw-native23 SONIC student.

This module is additive and simulator-only.  It starts from the audited
selected-V2 DadDance task, then replaces only its selected-teacher action
boundary with the exact V11 raw-native SONIC boundary and removes the
synthetic native124 actor observation.  The student evaluator owns inference;
the environment applies V2 exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from gear_sonic.envs.mjlab.native124_selected_v2_ankle_task import (
    CONTROL_DT_S,
    DISABLED_NOMINAL_EVENTS,
    audit_native124_selected_v2_ankle_task_env_cfg,
    make_native124_selected_v2_ankle_task_env_cfg,
)
from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
    DAD_DANCE_FRAME_COUNT,
    Native124SelectedV2CausalMotionCommandCfg,
    causal_episode_last_q9,
    resolve_causal_reset_anchor_index,
)
from gear_sonic.envs.mjlab.sonic_true23 import (
    _MJLAB_IMPORT_ERROR,
    SONIC_TRUE23_CRITIC_DIM,
    SONIC_TRUE23_POLICY_DIM,
    SONIC_TRUE23_TOKENIZER_DIM,
    TARGET_DOF,
    native_actions_to_hardware_targets,
    padded_previous_action,
)
from gear_sonic.envs.mjlab.sonic_true23_causal_history_safe_target_v11 import (
    SafeTargetNativeIl23JointPositionAction,
    SafeTargetNativeIl23JointPositionActionCfg,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
    safe_target_transform_contract,
)

STUDENT_QUALIFICATION_OBSERVATION_GROUPS = ("tokenizer", "policy", "critic")
_BASE_AUDIT_STAMP = "_sonic_student_qualification_base_audit"


def _exact_episode_length_s(control_dt: float, transitions: int) -> float:
    """Choose a duration whose ManagerBasedRlEnv ceil resolves exactly."""

    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("student qualification control_dt must be finite and positive")
    if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions <= 0:
        raise ValueError("student qualification transitions must be positive integer")
    duration = math.nextafter(transitions * control_dt, -math.inf)
    if math.ceil(duration / control_dt) != transitions:
        raise RuntimeError("student exact episode duration does not resolve to requested transitions")
    return duration


if _MJLAB_IMPORT_ERROR is None:

    @dataclass(kw_only=True)
    class DiagnosticSafeTargetNativeIl23JointPositionActionCfg(SafeTargetNativeIl23JointPositionActionCfg):
        """Build exact V11 action plus read-only qualification diagnostics."""

        def build(
            self,
            env: Any,
        ) -> "DiagnosticSafeTargetNativeIl23JointPositionAction":
            return DiagnosticSafeTargetNativeIl23JointPositionAction(self, env)

    class DiagnosticSafeTargetNativeIl23JointPositionAction(SafeTargetNativeIl23JointPositionAction):
        """Expose the raw->candidate->V2 chain without changing V11 control."""

        cfg: DiagnosticSafeTargetNativeIl23JointPositionActionCfg

        def __init__(
            self,
            cfg: DiagnosticSafeTargetNativeIl23JointPositionActionCfg,
            env: Any,
        ) -> None:
            super().__init__(cfg, env)
            default = torch.as_tensor(
                SAFE_TARGET_DEFAULT_Q_HARDWARE,
                dtype=torch.float32,
                device=self.device,
            )
            self._candidate_target_hardware = default.expand(
                self.num_envs,
                -1,
            ).clone()
            self._raw_clip_mask_native = torch.zeros_like(
                self._raw_actions,
                dtype=torch.bool,
            )

        @property
        def plain_sonic_raw_action_native(self) -> torch.Tensor:
            return self._raw_actions

        @property
        def candidate_target_hardware(self) -> torch.Tensor:
            return self._candidate_target_hardware

        @property
        def final_target_hardware(self) -> torch.Tensor:
            return self._processed_actions

        @property
        def raw_clip_mask_native(self) -> torch.Tensor:
            return self._raw_clip_mask_native

        def process_actions(self, actions: torch.Tensor) -> None:
            # Candidate is the unprojected plain SONIC linear target.  V11
            # then clips raw at +/-10 and applies asymmetric safe tanh once.
            self._candidate_target_hardware[:] = native_actions_to_hardware_targets(
                actions,
                self._default_q_hardware[0],
            )
            self._raw_clip_mask_native[:] = torch.abs(actions) >= SAFE_TARGET_RAW_ACTION_CLIP
            super().process_actions(actions)

        def reset(
            self,
            env_ids: torch.Tensor | slice | None = None,
        ) -> None:
            if env_ids is None:
                env_ids = slice(None)
            super().reset(env_ids)
            default = torch.as_tensor(
                SAFE_TARGET_DEFAULT_Q_HARDWARE,
                dtype=torch.float32,
                device=self.device,
            )
            self._candidate_target_hardware[env_ids] = default
            self._raw_clip_mask_native[env_ids] = False

else:

    class DiagnosticSafeTargetNativeIl23JointPositionActionCfg:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR

    class DiagnosticSafeTargetNativeIl23JointPositionAction:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR


def make_sonic_true23_student_qualification_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
    anchor_q9: int,
    transitions: int,
) -> Any:
    """Build one fixed-window, corruption-preserving student environment."""

    if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions <= 0:
        raise ValueError("student qualification transitions must be positive integer")
    reset_anchor = resolve_causal_reset_anchor_index(anchor_q9)
    causal_episode_last_q9(
        reset_anchor_index=reset_anchor,
        episode_steps=transitions,
        causal_max_anchor=DAD_DANCE_FRAME_COUNT - 2,
    )
    cfg = make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        play=False,
        fixed_anchor_index=reset_anchor,
    )
    base_audit = audit_native124_selected_v2_ankle_task_env_cfg(cfg)
    base_groups = dict(cfg.observations)
    if tuple(base_groups) != (*STUDENT_QUALIFICATION_OBSERVATION_GROUPS, "actor"):
        raise ValueError("selected nominal observation group order drift")
    if (
        base_groups["tokenizer"].enable_corruption is not True
        or base_groups["policy"].enable_corruption is not True
        or base_groups["actor"].enable_corruption is not False
        or base_groups["policy"].terms["previous_action"].func is not padded_previous_action
    ):
        raise ValueError("selected nominal observation semantics drift")

    setattr(cfg, _BASE_AUDIT_STAMP, dict(base_audit))
    cfg.actions["joint_pos"] = DiagnosticSafeTargetNativeIl23JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
    )
    cfg.observations.pop("actor")
    if any(cfg.observations[name] is not base_groups[name] for name in STUDENT_QUALIFICATION_OBSERVATION_GROUPS):
        raise RuntimeError("student environment replaced a SONIC observation group")
    control_dt = float(cfg.sim.mujoco.timestep) * int(cfg.decimation)
    cfg.episode_length_s = _exact_episode_length_s(control_dt, transitions)
    audit_sonic_true23_student_qualification_env_cfg(
        cfg,
        expected_anchor_q9=reset_anchor,
        expected_transitions=transitions,
    )
    return cfg


def audit_sonic_true23_student_qualification_env_cfg(
    cfg: Any,
    *,
    expected_anchor_q9: int,
    expected_transitions: int,
) -> dict[str, Any]:
    """Fail closed on student boundary, history, horizon, or nominal drift."""

    reset_anchor = resolve_causal_reset_anchor_index(expected_anchor_q9)
    if (
        isinstance(expected_transitions, bool)
        or not isinstance(expected_transitions, int)
        or expected_transitions <= 0
    ):
        raise ValueError("expected_transitions must be a positive integer")
    last_q9 = causal_episode_last_q9(
        reset_anchor_index=reset_anchor,
        episode_steps=expected_transitions,
        causal_max_anchor=DAD_DANCE_FRAME_COUNT - 2,
    )
    base_audit = getattr(cfg, _BASE_AUDIT_STAMP, None)
    if not isinstance(base_audit, Mapping) or base_audit.get("schema") != (
        "g1_true23_native124_selected_v2_ankle_task_env_v1"
    ):
        raise ValueError("student environment lacks pre-mutation selected-task audit")
    if (
        base_audit.get("domain_randomization") is not False
        or base_audit.get("reset_anchor_q9") != reset_anchor
        or any(name in cfg.events for name in DISABLED_NOMINAL_EVENTS)
    ):
        raise ValueError("student qualification nominal event contract drift")

    action_cfg = cfg.actions.get("joint_pos")
    if type(action_cfg) is not DiagnosticSafeTargetNativeIl23JointPositionActionCfg:
        raise ValueError("student environment lacks exact diagnostic V11 action cfg")
    if action_cfg.entity_name != "robot" or tuple(action_cfg.actuator_names) != (".*",):
        raise ValueError("student action entity/actuator selection drift")
    if tuple(cfg.observations) != STUDENT_QUALIFICATION_OBSERVATION_GROUPS:
        raise ValueError("student environment observation groups/order drift")
    tokenizer = cfg.observations["tokenizer"]
    policy = cfg.observations["policy"]
    critic = cfg.observations["critic"]
    if (
        tokenizer.enable_corruption is not True
        or policy.enable_corruption is not True
        or policy.concatenate_terms is not True
        or int(policy.history_length) != 10
        or policy.flatten_history_dim is not True
        or policy.terms["previous_action"].func is not padded_previous_action
        or getattr(critic, "concatenate_terms", None) is not True
    ):
        raise ValueError("student SONIC observation/history contract drift")
    command = cfg.commands.get("motion")
    if type(command) is not Native124SelectedV2CausalMotionCommandCfg:
        raise ValueError("student environment lacks exact fixed-anchor causal command")
    if (
        command.fixed_anchor_index != reset_anchor
        or command.sampling_mode != "start"
        or command.pose_range != {}
        or command.velocity_range != {}
        or tuple(command.joint_position_range) != (0.0, 0.0)
    ):
        raise ValueError("student fixed-anchor command contract drift")
    control_dt = float(cfg.sim.mujoco.timestep) * int(cfg.decimation)
    expected_episode_length_s = _exact_episode_length_s(control_dt, expected_transitions)
    resolved_episode_steps = math.ceil(float(cfg.episode_length_s) / control_dt)
    if (
        abs(control_dt - CONTROL_DT_S) > 1.0e-12
        or float(cfg.episode_length_s) != expected_episode_length_s
        or resolved_episode_steps != expected_transitions
    ):
        raise ValueError("student control rate/horizon drift")
    termination = cfg.terminations.get("v2_raw_clip")
    if termination is None:
        raise ValueError("student environment lacks strict raw-clip termination")

    return {
        "schema": "g1_true23_sonic_student_qualification_env_v1",
        "base_selected_v2_nominal_audit": dict(base_audit),
        "base_audited_before_student_mutation": True,
        "action_cfg": f"{type(action_cfg).__module__}:{type(action_cfg).__qualname__}",
        "action_input": "plain_sonic_raw_native23",
        "safe_target_transform": safe_target_transform_contract(),
        "safe_target_transform_application_count": 1,
        "wrapper_action_clip": None,
        "reset_anchor_q9": reset_anchor,
        "last_action_q9": last_q9,
        "transitions": expected_transitions,
        "control_dt_s": control_dt,
        "episode_length_s": float(cfg.episode_length_s),
        "resolved_max_episode_length": resolved_episode_steps,
        "tokenizer_observation_dim": SONIC_TRUE23_TOKENIZER_DIM,
        "policy_history_dim": SONIC_TRUE23_POLICY_DIM,
        "critic_observation_dim": SONIC_TRUE23_CRITIC_DIM,
        "tokenizer_corruption_enabled": True,
        "policy_corruption_enabled": True,
        "synthetic_actor_observation_present": False,
        "domain_randomization": False,
        "raw_clip_termination": f"any_abs_raw_native_greater_than_or_equal_{SAFE_TARGET_RAW_ACTION_CLIP:g}",
        "student_action_substitution": False,
        "teacher_action_present": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def capture_student_action_chain(raw_env: Any) -> dict[str, torch.Tensor]:
    """Return cloned exact runtime links for one already-processed action."""

    term = raw_env.action_manager.get_term("joint_pos")
    if type(term) is not DiagnosticSafeTargetNativeIl23JointPositionAction:
        raise TypeError("runtime lacks exact diagnostic student action term")
    values = {
        "raw_native": term.plain_sonic_raw_action_native,
        "candidate_target_hardware": term.candidate_target_hardware,
        "safe_native": term.safe_native_action,
        "final_target_hardware": term.final_target_hardware,
        "raw_clip_mask_native": term.raw_clip_mask_native,
    }
    for name, value in values.items():
        if (
            type(value) is not torch.Tensor
            or value.shape != (raw_env.num_envs, TARGET_DOF)
            or (name == "raw_clip_mask_native" and value.dtype != torch.bool)
            or (name != "raw_clip_mask_native" and value.dtype != torch.float32)
            or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
        ):
            raise ValueError(f"student runtime action link drift: {name}")
    return {name: value.detach().clone() for name, value in values.items()}


__all__ = [
    "DiagnosticSafeTargetNativeIl23JointPositionAction",
    "DiagnosticSafeTargetNativeIl23JointPositionActionCfg",
    "STUDENT_QUALIFICATION_OBSERVATION_GROUPS",
    "audit_sonic_true23_student_qualification_env_cfg",
    "capture_student_action_chain",
    "make_sonic_true23_student_qualification_env_cfg",
]
