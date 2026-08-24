"""Isolated low-latency true23 recovery task additions.

Base observation/action semantics remain unchanged.  Recovery only adds
training rewards that keep commanded position targets near the FK-consistent
reference and inside an inner soft-limit margin.
"""

from __future__ import annotations

from typing import Any

import torch

from gear_sonic.envs.mjlab.sonic_true23 import (
    make_sonic_true23_tracking_env_cfg,
)
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    REFERENCE_PROFILE_LOW_LATENCY,
    TARGET_DOF,
)

TARGET_REFERENCE_WEIGHT = -2.0
TARGET_SOFT_LIMIT_WEIGHT = -10.0
ACTUAL_SOFT_LIMIT_WEIGHT = -20.0
ACTION_RATE_WEIGHT = -0.2
TARGET_INNER_MARGIN_FRACTION = 0.025


def _processed_target(env: Any, action_name: str) -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    target = getattr(term, "processed_action", None)
    if not isinstance(target, torch.Tensor) or target.shape != (
        env.num_envs,
        TARGET_DOF,
    ):
        raise ValueError("recovery action term lacks exact [env,23] position target")
    if not torch.isfinite(target).all():
        raise ValueError("recovery action target contains NaN or Inf")
    return target


def action_target_reference_l2(
    env: Any,
    command_name: str = "motion",
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Penalize target/reference error in normalized deployment action units."""

    command = env.command_manager.get_term(command_name)
    reference = command.joint_pos
    target = _processed_target(env, action_name)
    if reference.shape != target.shape or not torch.isfinite(reference).all():
        raise ValueError("recovery reference joint target must be finite [env,23]")
    scale = torch.as_tensor(
        HARDWARE_23_ACTION_SCALE,
        dtype=target.dtype,
        device=target.device,
    )
    return torch.mean(torch.square((target - reference) / scale), dim=1)


def action_target_soft_limit_barrier(
    env: Any,
    action_name: str = "joint_pos",
    entity_name: str = "robot",
    margin_fraction: float = TARGET_INNER_MARGIN_FRACTION,
) -> torch.Tensor:
    """Penalize targets before they reach MJLab's 0.9-factor soft limits."""

    if not 0.0 < margin_fraction < 0.1:
        raise ValueError("target soft-limit margin_fraction must be in (0,0.1)")
    target = _processed_target(env, action_name)
    entity = env.scene[entity_name]
    limits = entity.data.soft_joint_pos_limits
    if not isinstance(limits, torch.Tensor) or limits.shape != (
        env.num_envs,
        TARGET_DOF,
        2,
    ):
        raise ValueError("recovery robot must expose exact [env,23,2] soft limits")
    span = limits[..., 1] - limits[..., 0]
    margin = span * margin_fraction
    inner_low = limits[..., 0] + margin
    inner_high = limits[..., 1] - margin
    violation = torch.relu(inner_low - target) + torch.relu(target - inner_high)
    return torch.mean(torch.square(violation / margin.clamp_min(1.0e-6)), dim=1)


def make_low_latency_recovery_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
    play: bool = False,
) -> Any:
    """Return exact low-latency task plus recovery-only reward shaping."""

    from mjlab.managers.reward_manager import RewardTermCfg

    cfg = make_sonic_true23_tracking_env_cfg(
        motion_file=motion_file,
        reference_profile=REFERENCE_PROFILE_LOW_LATENCY,
        num_envs=num_envs,
        play=play,
    )
    cfg.rewards["action_target_reference_l2"] = RewardTermCfg(
        func=action_target_reference_l2,
        weight=TARGET_REFERENCE_WEIGHT,
        params={"command_name": "motion", "action_name": "joint_pos"},
    )
    cfg.rewards["action_target_soft_limit_barrier"] = RewardTermCfg(
        func=action_target_soft_limit_barrier,
        weight=TARGET_SOFT_LIMIT_WEIGHT,
        params={
            "action_name": "joint_pos",
            "entity_name": "robot",
            "margin_fraction": TARGET_INNER_MARGIN_FRACTION,
        },
    )
    cfg.rewards["joint_limit"].weight = ACTUAL_SOFT_LIMIT_WEIGHT
    cfg.rewards["action_rate_l2"].weight = ACTION_RATE_WEIGHT
    return cfg


def recovery_reward_contract() -> dict[str, object]:
    return {
        "action_target_reference_l2": {
            "weight": TARGET_REFERENCE_WEIGHT,
            "normalization": "hardware_action_scale",
        },
        "action_target_soft_limit_barrier": {
            "weight": TARGET_SOFT_LIMIT_WEIGHT,
            "inner_margin_fraction_of_soft_span": TARGET_INNER_MARGIN_FRACTION,
        },
        "actual_joint_soft_limit": {"weight": ACTUAL_SOFT_LIMIT_WEIGHT},
        "action_rate_l2": {"weight": ACTION_RATE_WEIGHT},
    }
