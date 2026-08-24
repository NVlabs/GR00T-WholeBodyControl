"""Neutral stand acquisition stage for causal true23 recovery."""

from __future__ import annotations

from typing import Any

from gear_sonic.envs.mjlab.sonic_true23_causal_history_acquisition import (
    make_causal_history_acquisition_env_cfg as _make_acquisition_env_cfg,
)

ALIVE_WEIGHT = 5.0
TRACKING_WEIGHT_MULTIPLIER = 2.0
_TRACKING_REWARDS = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
)


def make_causal_history_stand_acquisition_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
    play: bool = False,
) -> Any:
    """Build strict stand acquisition with track-first survival shaping."""

    from mjlab.envs import mdp
    from mjlab.managers.reward_manager import RewardTermCfg

    cfg = _make_acquisition_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        play=play,
    )
    cfg.rewards["alive"] = RewardTermCfg(func=mdp.is_alive, weight=ALIVE_WEIGHT)
    for name in _TRACKING_REWARDS:
        cfg.rewards[name].weight *= TRACKING_WEIGHT_MULTIPLIER
    return cfg


def causal_stand_acquisition_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_stand_acquisition_v3",
        "motion_scope": "neutral_stand_only",
        "alive_reward": {
            "function": "mjlab.envs.mdp.is_alive",
            "weight": ALIVE_WEIGHT,
        },
        "tracking_weight_multiplier": TRACKING_WEIGHT_MULTIPLIER,
        "action_target_reference_penalty_preserved": True,
        "action_target_soft_limit_barrier_preserved": True,
        "actual_joint_soft_limit_penalty_preserved": True,
        "ee_body_pos_termination_threshold_m": 0.25,
        "interval_pushes_enabled": False,
        "disturbance_finetune_required_before_promotion": True,
    }
