"""Disturbance and domain-randomized true23 causal stand fine-tuning v9."""

from __future__ import annotations

import copy
from typing import Any

import torch

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    make_causal_history_recovery_env_cfg as _make_base_env_cfg,
)
from gear_sonic.envs.mjlab.sonic_true23_causal_history_stand_acquisition_v5 import (
    make_causal_history_stand_acquisition_v5_env_cfg as _make_v5_env_cfg,
)
from gear_sonic.envs.mjlab.sonic_true23_low_latency_recovery import (
    action_target_soft_limit_barrier,
)

TARGET_BARRIER_WEIGHT = -50.0
TARGET_BARRIER_MARGIN_FRACTION = 0.025
DOMINANT_HW10_BARRIER_WEIGHT = -25.0
DOMINANT_HARDWARE_INDEX = 10
DOMINANT_NATIVE_INDEX = 16
_RESTORED_EVENTS = ("push_robot", "base_com", "encoder_bias", "foot_friction")


def dominant_hw10_target_soft_limit_barrier(
    env: Any,
    action_name: str = "joint_pos",
    entity_name: str = "robot",
    margin_fraction: float = TARGET_BARRIER_MARGIN_FRACTION,
) -> torch.Tensor:
    """Log and strengthen dominant right-ankle-pitch target contribution."""

    term = env.action_manager.get_term(action_name)
    target = term.processed_action
    limits = env.scene[entity_name].data.soft_joint_pos_limits
    if target.shape != (env.num_envs, 23) or limits.shape != (env.num_envs, 23, 2):
        raise ValueError("v9 dominant barrier requires exact hardware [env,23]")
    span = limits[:, DOMINANT_HARDWARE_INDEX, 1] - limits[
        :, DOMINANT_HARDWARE_INDEX, 0
    ]
    margin = span * margin_fraction
    low = limits[:, DOMINANT_HARDWARE_INDEX, 0] + margin
    high = limits[:, DOMINANT_HARDWARE_INDEX, 1] - margin
    value = target[:, DOMINANT_HARDWARE_INDEX]
    violation = torch.relu(low - value) + torch.relu(value - high)
    return torch.square(violation / margin.clamp_min(1.0e-6))


def make_causal_history_disturbance_v9_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
    play: bool = False,
) -> Any:
    """Restore exact base disturbances while preserving v5 physical gates."""

    cfg = _make_v5_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        play=play,
    )
    base_cfg = _make_base_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        play=play,
    )
    for name in _RESTORED_EVENTS:
        if name not in base_cfg.events:
            if play and name == "push_robot":
                continue
            raise ValueError(f"v9 base task lacks required event: {name}")
        cfg.events[name] = copy.deepcopy(base_cfg.events[name])

    barrier = cfg.rewards["action_target_soft_limit_barrier"]
    barrier.func = action_target_soft_limit_barrier
    barrier.weight = TARGET_BARRIER_WEIGHT
    barrier.params = {
        "action_name": "joint_pos",
        "entity_name": "robot",
        "margin_fraction": TARGET_BARRIER_MARGIN_FRACTION,
    }
    from mjlab.managers.reward_manager import RewardTermCfg

    cfg.rewards["action_target_soft_limit_barrier_hw10"] = RewardTermCfg(
        func=dominant_hw10_target_soft_limit_barrier,
        weight=DOMINANT_HW10_BARRIER_WEIGHT,
        params={
            "action_name": "joint_pos",
            "entity_name": "robot",
            "margin_fraction": TARGET_BARRIER_MARGIN_FRACTION,
        },
    )
    audit_causal_history_disturbance_v9_env_cfg(cfg, require_push=not play)
    return cfg


def _callable_name(value: Any) -> str:
    return f"{value.__module__}:{value.__qualname__}"


def _entity_names(entity: Any, attribute: str) -> list[str]:
    value = getattr(entity, attribute)
    if isinstance(value, str):
        return [value]
    return list(value)


def audit_causal_history_disturbance_v9_env_cfg(
    cfg: Any,
    *,
    require_push: bool = True,
) -> dict[str, object]:
    """Validate executed event semantics and unchanged physical gates."""

    required = set(_RESTORED_EVENTS if require_push else _RESTORED_EVENTS[1:])
    if not required.issubset(cfg.events):
        raise ValueError("v9 executed environment omits required events")
    push = cfg.events.get("push_robot")
    if require_push:
        if (
            push.mode != "interval"
            or tuple(push.interval_range_s) != (1.0, 3.0)
            or push.params["velocity_range"]
            != {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78)}
        ):
            raise ValueError("v9 push_robot differs from exact base task")
    base_com = cfg.events["base_com"]
    encoder_bias = cfg.events["encoder_bias"]
    friction = cfg.events["foot_friction"]
    if (
        base_com.mode != "startup"
        or base_com.params["operation"] != "add"
        or base_com.params["ranges"]
        != {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}
        or _entity_names(base_com.params["asset_cfg"], "body_names")
        != ["torso_link"]
    ):
        raise ValueError("v9 base COM randomization differs from exact base task")
    if (
        encoder_bias.mode != "startup"
        or tuple(encoder_bias.params["bias_range"]) != (-0.01, 0.01)
    ):
        raise ValueError("v9 encoder bias randomization differs from exact base task")
    if (
        friction.mode != "startup"
        or friction.params["operation"] != "abs"
        or tuple(friction.params["ranges"]) != (0.3, 1.2)
        or friction.params["shared_random"] is not True
        or _entity_names(friction.params["asset_cfg"], "geom_names")
        != [r"^(left|right)_foot[1-7]_collision$"]
    ):
        raise ValueError("v9 foot friction randomization differs from exact base task")

    command = cfg.commands["motion"]
    control_dt = float(cfg.sim.mujoco.timestep) * int(cfg.decimation)
    barrier = cfg.rewards["action_target_soft_limit_barrier"]
    dominant_barrier = cfg.rewards["action_target_soft_limit_barrier_hw10"]
    physical = {
        "control_dt_s": control_dt,
        "episode_steps": int(round(float(cfg.episode_length_s) / control_dt)),
        "ee_body_pos_termination_threshold_m": float(
            cfg.terminations["ee_body_pos"].params["threshold"]
        ),
        "non_timeout_termination_weight": float(
            cfg.rewards["non_timeout_termination"].weight
        ),
        "actual_joint_limit_penalty_weight": float(cfg.rewards["joint_limit"].weight),
        "fixed_reference_guard_weight": float(
            cfg.rewards["fixed_start_reference_exhaustion_guard"].weight
        ),
        "sampling_mode": command.sampling_mode,
        "pose_range": command.pose_range,
        "velocity_range": command.velocity_range,
        "joint_position_range": list(command.joint_position_range),
    }
    expected_physical = {
        "control_dt_s": 0.02,
        "episode_steps": 500,
        "ee_body_pos_termination_threshold_m": 0.25,
        "non_timeout_termination_weight": -5000.0,
        "actual_joint_limit_penalty_weight": -20.0,
        "fixed_reference_guard_weight": 1.0,
        "sampling_mode": "start",
        "pose_range": {},
        "velocity_range": {},
        "joint_position_range": [0.0, 0.0],
    }
    if physical != expected_physical:
        raise ValueError("v9 changed a physical/termination/reference gate")
    if (
        barrier.func is not action_target_soft_limit_barrier
        or float(barrier.weight) != TARGET_BARRIER_WEIGHT
        or barrier.params
        != {
            "action_name": "joint_pos",
            "entity_name": "robot",
            "margin_fraction": TARGET_BARRIER_MARGIN_FRACTION,
        }
    ):
        raise ValueError("v9 target soft-limit barrier contract differs")
    if (
        dominant_barrier.func is not dominant_hw10_target_soft_limit_barrier
        or float(dominant_barrier.weight) != DOMINANT_HW10_BARRIER_WEIGHT
    ):
        raise ValueError("v9 dominant hardware-10 barrier contract differs")

    events: dict[str, object] = {
        "base_com": {
            "function": _callable_name(base_com.func),
            "mode": base_com.mode,
            "operation": base_com.params["operation"],
            "ranges_m": {
                "x": list(base_com.params["ranges"][0]),
                "y": list(base_com.params["ranges"][1]),
                "z": list(base_com.params["ranges"][2]),
            },
            "body_names": _entity_names(base_com.params["asset_cfg"], "body_names"),
        },
        "encoder_bias": {
            "function": _callable_name(encoder_bias.func),
            "mode": encoder_bias.mode,
            "bias_range_rad": list(encoder_bias.params["bias_range"]),
        },
        "foot_friction": {
            "function": _callable_name(friction.func),
            "mode": friction.mode,
            "operation": friction.params["operation"],
            "range": list(friction.params["ranges"]),
            "shared_random": friction.params["shared_random"],
            "geom_names": _entity_names(friction.params["asset_cfg"], "geom_names"),
        },
    }
    if require_push:
        events["push_robot"] = {
            "function": _callable_name(push.func),
            "mode": push.mode,
            "interval_range_s": list(push.interval_range_s),
            "velocity_range": push.params["velocity_range"],
        }
    return {
        "schema": "g1_true23_causal_history_disturbance_env_v9",
        "events": events,
        "target_soft_limit_barrier": {
            "function": _callable_name(barrier.func),
            "weight": float(barrier.weight),
            "uncapped": True,
            "margin_fraction": TARGET_BARRIER_MARGIN_FRACTION,
        },
        "dominant_target_barrier": {
            "function": _callable_name(dominant_barrier.func),
            "weight": float(dominant_barrier.weight),
            "hardware_index": DOMINANT_HARDWARE_INDEX,
            "native_policy_output_index": DOMINANT_NATIVE_INDEX,
            "joint_name": "right_ankle_pitch_joint",
        },
        "physical_gates": physical,
    }


def causal_history_disturbance_v9_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_history_disturbance_v9",
        "restart_from_model0": True,
        "restored_exact_base_events": list(_RESTORED_EVENTS),
        "target_soft_limit_barrier_weight": TARGET_BARRIER_WEIGHT,
        "target_soft_limit_barrier_uncapped": True,
        "dominant_hw10_barrier_weight": DOMINANT_HW10_BARRIER_WEIGHT,
        "dominant_hw10_maps_to_native16": True,
        "v8_final_affine_and_cumulative_kl_boundary_retained": True,
        "physical_termination_and_reference_gates_unchanged": True,
        "deployment_ready": False,
    }
