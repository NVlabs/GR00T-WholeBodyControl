"""Safe causal multi-motion task for teleop-v14 acquisition."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from gear_sonic.envs.mjlab.native124_selected_v2_ankle_task import DISABLED_NOMINAL_EVENTS
from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    make_causal_history_recovery_env_cfg as _make_causal_history_recovery_env_cfg,
)
from gear_sonic.envs.mjlab.sonic_true23_causal_history_safe_target_v11 import (
    SafeTargetNativeIl23JointPositionActionCfg,
)
from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
    EE_TERMINATION_BODY_NAMES,
    RIGHT_WRIST_BARRIER_DENOMINATOR_FLOOR,
    RIGHT_WRIST_BARRIER_ONSET_M,
    RIGHT_WRIST_BARRIER_WEIGHT,
    RIGHT_WRIST_BODY_NAME,
    RIGHT_WRIST_TERMINATION_M,
    WORST_EE_NORMALIZATION_M,
    WORST_EE_WEIGHT,
    right_wrist_prethreshold_barrier,
    worst_ee_z_normalized_squared,
)

ALIVE_WEIGHT = 5.0
NON_TIMEOUT_TERMINATION_WEIGHT = -5000.0


def corpus_clip_time_out(env: Any, command_name: str = "motion") -> torch.Tensor:
    """End episode before command can cross into another concatenated clip."""

    command = env.command_manager.get_term(command_name)
    clip_stop = getattr(command, "_env_clip_stop", None)
    if not isinstance(clip_stop, torch.Tensor) or clip_stop.shape != (env.num_envs,):
        raise ValueError("multi-motion command lacks per-environment clip stop")
    if clip_stop.dtype != torch.long:
        raise ValueError("multi-motion clip stop must be int64")
    return command.time_steps >= clip_stop - 1


def make_causal_multimotion_v14_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
    play: bool = False,
) -> Any:
    """Build exact causal H10/V11 task with clip-contained reset semantics."""

    from mjlab.envs import mdp
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg

    cfg = _make_causal_history_recovery_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        play=play,
    )
    cfg.actions["joint_pos"] = SafeTargetNativeIl23JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
    )
    for event_name in DISABLED_NOMINAL_EVENTS:
        cfg.events.pop(event_name, None)
    cfg.terminations["corpus_clip_time_out"] = TerminationTermCfg(
        func=corpus_clip_time_out,
        time_out=True,
        params={"command_name": "motion"},
    )
    cfg.rewards["alive"] = RewardTermCfg(func=mdp.is_alive, weight=ALIVE_WEIGHT)
    cfg.rewards["non_timeout_termination"] = RewardTermCfg(
        func=mdp.is_terminated,
        weight=NON_TIMEOUT_TERMINATION_WEIGHT,
    )
    cfg.rewards["worst_ee_z_normalized_squared"] = RewardTermCfg(
        func=worst_ee_z_normalized_squared,
        weight=WORST_EE_WEIGHT,
        params={
            "command_name": "motion",
            "body_names": EE_TERMINATION_BODY_NAMES,
            "normalization_m": WORST_EE_NORMALIZATION_M,
        },
    )
    cfg.rewards["right_wrist_prethreshold_barrier"] = RewardTermCfg(
        func=right_wrist_prethreshold_barrier,
        weight=RIGHT_WRIST_BARRIER_WEIGHT,
        params={
            "command_name": "motion",
            "body_name": RIGHT_WRIST_BODY_NAME,
            "onset_m": RIGHT_WRIST_BARRIER_ONSET_M,
            "termination_m": RIGHT_WRIST_TERMINATION_M,
            "denominator_floor": RIGHT_WRIST_BARRIER_DENOMINATOR_FLOOR,
        },
    )
    return cfg


def causal_multimotion_v14_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_multimotion_v14",
        "history_length": 10,
        "action_boundary": "raw_native23_to_safe_target_v11_once",
        "clip_boundary": "full_environment_timeout_reset_before_crossing",
        "artificial_action_substitution": False,
        "ee_termination_threshold_m": 0.25,
        "alive_weight": ALIVE_WEIGHT,
        "non_timeout_termination_weight": NON_TIMEOUT_TERMINATION_WEIGHT,
        "disabled_events": list(DISABLED_NOMINAL_EVENTS),
        "deployment_ready": False,
        "hardware_authorized": False,
    }


__all__: Sequence[str] = (
    "causal_multimotion_v14_contract",
    "corpus_clip_time_out",
    "make_causal_multimotion_v14_env_cfg",
)
