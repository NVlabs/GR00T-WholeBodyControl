"""Acquisition-stage recovery for the exact true23 causal-history task.

The first causal run exposed a genuine early-termination incentive: tracking
had no non-timeout termination cost, so PPO reduced its negative return by
ending episodes sooner.  This isolated task keeps the causal observation and
0.25 m end-effector termination contracts unchanged, adds MJLab's standard
non-timeout termination penalty, and removes interval pushes only while the
policy acquires causal tracking.
"""

from __future__ import annotations

from typing import Any

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    make_causal_history_recovery_env_cfg as _make_causal_history_recovery_env_cfg,
)

NON_TIMEOUT_TERMINATION_WEIGHT = -200.0
EE_BODY_POS_TERMINATION_THRESHOLD_M = 0.25


def make_causal_history_acquisition_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
    play: bool = False,
) -> Any:
    """Build causal acquisition without weakening any fall termination."""

    from mjlab.envs import mdp
    from mjlab.managers.reward_manager import RewardTermCfg

    cfg = _make_causal_history_recovery_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        play=play,
    )
    threshold = float(cfg.terminations["ee_body_pos"].params["threshold"])
    if threshold != EE_BODY_POS_TERMINATION_THRESHOLD_M:
        raise ValueError(
            "causal acquisition may not weaken the 0.25 m EE termination"
        )
    cfg.rewards["non_timeout_termination"] = RewardTermCfg(
        func=mdp.is_terminated,
        weight=NON_TIMEOUT_TERMINATION_WEIGHT,
    )
    cfg.events.pop("push_robot", None)
    return cfg


def causal_acquisition_contract() -> dict[str, object]:
    """Return the source-bindable stage contract."""

    return {
        "schema": "g1_true23_causal_history_acquisition_v2",
        "restart_from_approved_initialization": True,
        "collapsed_model_250_reused": False,
        "critic_reused": False,
        "optimizer_reused": False,
        "non_timeout_termination": {
            "function": "mjlab.envs.mdp.is_terminated",
            "weight": NON_TIMEOUT_TERMINATION_WEIGHT,
        },
        "ee_body_pos_termination_threshold_m": (
            EE_BODY_POS_TERMINATION_THRESHOLD_M
        ),
        "ee_termination_weakened": False,
        "interval_pushes_enabled": False,
        "stage_role": "causal_tracking_acquisition",
        "disturbance_finetune_required_before_promotion": True,
    }
