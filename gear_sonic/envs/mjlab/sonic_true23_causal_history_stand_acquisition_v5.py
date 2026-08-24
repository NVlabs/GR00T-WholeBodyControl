"""Nominal causal stand acquisition after the v4 incentive gate failed."""

from __future__ import annotations

from typing import Any

import torch

from gear_sonic.envs.mjlab.sonic_true23_causal_history_stand_acquisition_v4 import (
    make_causal_history_stand_acquisition_v4_env_cfg as _make_v4_env_cfg,
)
from gear_sonic.envs.mjlab.sonic_true23_low_latency_recovery import (
    action_target_soft_limit_barrier,
)

ALIVE_WEIGHT = 20.0
TARGET_BARRIER_WEIGHT = -10.0
TARGET_BARRIER_MAX_NORMALIZED_PENALTY = 0.25
EXPECTED_STEP_DT_S = 0.02
MAXIMUM_TARGET_BARRIER_COST_PER_STEP = (
    TARGET_BARRIER_WEIGHT
    * TARGET_BARRIER_MAX_NORMALIZED_PENALTY
    * EXPECTED_STEP_DT_S
)
ALIVE_REWARD_PER_STEP = ALIVE_WEIGHT * EXPECTED_STEP_DT_S
_ACQUISITION_DISABLED_DOMAIN_RANDOMIZATION = (
    "base_com",
    "encoder_bias",
    "foot_friction",
)


def capped_action_target_soft_limit_barrier(
    env: Any,
    action_name: str = "joint_pos",
    entity_name: str = "robot",
    margin_fraction: float = 0.025,
    maximum_normalized_penalty: float = TARGET_BARRIER_MAX_NORMALIZED_PENALTY,
) -> torch.Tensor:
    """Keep the inner-limit gradient but bound its acquisition return cost."""

    if maximum_normalized_penalty != TARGET_BARRIER_MAX_NORMALIZED_PENALTY:
        raise ValueError("v5 target barrier cap differs from audited contract")
    penalty = action_target_soft_limit_barrier(
        env,
        action_name=action_name,
        entity_name=entity_name,
        margin_fraction=margin_fraction,
    )
    return torch.clamp(penalty, max=maximum_normalized_penalty)


def make_causal_history_stand_acquisition_v5_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
    play: bool = False,
) -> Any:
    """Build nominal stand acquisition with bounded safety shaping."""

    cfg = _make_v4_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        play=play,
    )
    cfg.rewards["alive"].weight = ALIVE_WEIGHT
    barrier = cfg.rewards["action_target_soft_limit_barrier"]
    barrier.func = capped_action_target_soft_limit_barrier
    barrier.weight = TARGET_BARRIER_WEIGHT
    barrier.params = {
        **barrier.params,
        "maximum_normalized_penalty": (
            TARGET_BARRIER_MAX_NORMALIZED_PENALTY
        ),
    }
    for name in _ACQUISITION_DISABLED_DOMAIN_RANDOMIZATION:
        cfg.events.pop(name, None)
    return cfg


def causal_stand_acquisition_v5_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_stand_acquisition_v5",
        "stage_role": "nominal_neutral_stand_acquisition",
        "alive_reward": {
            "weight": ALIVE_WEIGHT,
            "control_dt_s": EXPECTED_STEP_DT_S,
            "reward_per_nonterminal_step": ALIVE_REWARD_PER_STEP,
        },
        "target_soft_limit_barrier": {
            "weight": TARGET_BARRIER_WEIGHT,
            "inner_margin_fraction_of_soft_span": 0.025,
            "maximum_normalized_penalty": (
                TARGET_BARRIER_MAX_NORMALIZED_PENALTY
            ),
            "maximum_cost_per_step": MAXIMUM_TARGET_BARRIER_COST_PER_STEP,
            "physical_soft_limit_gate_weakened": False,
        },
        "alive_to_maximum_target_barrier_step_ratio": (
            ALIVE_REWARD_PER_STEP
            / abs(MAXIMUM_TARGET_BARRIER_COST_PER_STEP)
        ),
        "actual_joint_soft_limit_penalty_weight": -20.0,
        "action_target_reference_penalty_weight": -2.0,
        "non_timeout_terminal_cost_per_event": -100.0,
        "ee_body_pos_termination_threshold_m": 0.25,
        "sampling_mode": "start",
        "maximum_episode_steps": 500,
        "maximum_pre_timeout_anchor": 508,
        "maximum_valid_anchor": 598,
        "runtime_reference_exhaustion_guard": True,
        "reset_pose_range": {},
        "reset_velocity_range": {},
        "reset_joint_position_range_rad": [0.0, 0.0],
        "disabled_domain_randomization_for_acquisition": list(
            _ACQUISITION_DISABLED_DOMAIN_RANDOMIZATION
        ),
        "domain_randomization_finetune_required": True,
        "push_disturbance_finetune_required": True,
        "transition_and_dance_finetune_required": True,
        "deployment_ready": False,
    }
