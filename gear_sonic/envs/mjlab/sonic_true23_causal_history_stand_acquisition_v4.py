"""Fail-closed causal stand acquisition after the v3 runtime audit."""

from __future__ import annotations

from typing import Any

import torch

from gear_sonic.envs.mjlab.sonic_true23_causal_history_stand_acquisition import (
    make_causal_history_stand_acquisition_env_cfg as _make_stand_env_cfg,
)

NON_TIMEOUT_TERMINATION_WEIGHT = -5000.0
EXPECTED_STEP_DT_S = 0.02
TERMINAL_COST_PER_EVENT = (
    NON_TIMEOUT_TERMINATION_WEIGHT * EXPECTED_STEP_DT_S
)
FIXED_CAUSAL_START_ANCHOR = 9
NEUTRAL_FRAME_COUNT = 600
MAXIMUM_EPISODE_STEPS = 500


def fixed_start_reference_exhaustion_guard(
    env: Any,
    command_name: str = "motion",
) -> torch.Tensor:
    """Raise before any hidden in-episode reference resample can occur."""

    command = env.command_manager.get_term(command_name)
    time_steps = getattr(command, "time_steps", None)
    maximum_anchor = getattr(command, "_causal_max_anchor", None)
    if not isinstance(time_steps, torch.Tensor) or not isinstance(
        maximum_anchor, int
    ):
        raise ValueError("v4 causal command lacks exhaustion-audit state")
    if torch.any(time_steps >= maximum_anchor):
        raise RuntimeError(
            "v4 fixed-start reference exhausted before episode reset"
        )
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)


def make_causal_history_stand_acquisition_v4_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
    play: bool = False,
) -> Any:
    """Build fixed-start/no-RSI stand acquisition with a real death cost."""

    from mjlab.managers.reward_manager import RewardTermCfg

    cfg = _make_stand_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        play=play,
    )
    control_dt = float(cfg.sim.mujoco.timestep) * int(cfg.decimation)
    if abs(control_dt - EXPECTED_STEP_DT_S) > 1.0e-12:
        raise ValueError("v4 terminal-cost proof requires exact 0.02 s control dt")
    if int(round(float(cfg.episode_length_s) / control_dt)) != (
        MAXIMUM_EPISODE_STEPS
    ):
        raise ValueError("v4 stand acquisition requires exact 500-step episodes")
    if FIXED_CAUSAL_START_ANCHOR + MAXIMUM_EPISODE_STEPS >= (
        NEUTRAL_FRAME_COUNT - 1
    ):
        raise ValueError("v4 neutral motion cannot cover one complete episode")

    command = cfg.commands["motion"]
    command.sampling_mode = "start"
    command.pose_range = {}
    command.velocity_range = {}
    command.joint_position_range = (0.0, 0.0)
    cfg.rewards["non_timeout_termination"].weight = (
        NON_TIMEOUT_TERMINATION_WEIGHT
    )
    cfg.rewards["fixed_start_reference_exhaustion_guard"] = RewardTermCfg(
        func=fixed_start_reference_exhaustion_guard,
        weight=1.0,
        params={"command_name": "motion"},
    )
    return cfg


def causal_stand_acquisition_v4_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_stand_acquisition_v4",
        "motion_scope": "neutral_stand_only",
        "sampling_mode": "start",
        "fixed_causal_start_anchor": FIXED_CAUSAL_START_ANCHOR,
        "maximum_episode_steps": MAXIMUM_EPISODE_STEPS,
        "neutral_frame_count": NEUTRAL_FRAME_COUNT,
        "minimum_unused_tail_frames": (
            NEUTRAL_FRAME_COUNT
            - 1
            - FIXED_CAUSAL_START_ANCHOR
            - MAXIMUM_EPISODE_STEPS
        ),
        "in_episode_reference_resample_permitted": False,
        "runtime_reference_exhaustion_guard": True,
        "reset_pose_range": {},
        "reset_velocity_range": {},
        "reset_joint_position_range_rad": [0.0, 0.0],
        "non_timeout_termination": {
            "function": "mjlab.envs.mdp.is_terminated",
            "weight": NON_TIMEOUT_TERMINATION_WEIGHT,
            "control_dt_s": EXPECTED_STEP_DT_S,
            "cost_per_event": TERMINAL_COST_PER_EVENT,
        },
        "alive_reward_weight": 5.0,
        "tracking_weight_multiplier": 2.0,
        "ee_body_pos_termination_threshold_m": 0.25,
        "action_target_reference_penalty_preserved": True,
        "action_target_soft_limit_barrier_preserved": True,
        "actual_joint_soft_limit_penalty_preserved": True,
        "interval_pushes_enabled": False,
        "disturbance_finetune_required_before_promotion": True,
    }
