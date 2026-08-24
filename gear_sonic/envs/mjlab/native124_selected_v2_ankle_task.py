"""Nominal DadDance task for four-row selected-21204 ankle adaptation.

This is an offline/simulator-only acquisition task.  It layers task-space and
physical objectives over the exact causal q9/q10 environment and the sole
selected-21204 -> SONIC V2 action term.  It deliberately removes joint-target
imitation: the derivative must learn the dynamics of the target transform, not
copy an unsafe or unrepresentable reference setpoint.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from gear_sonic.envs.mjlab.native124_selected_v2_ankle_adaptation import (
    Selected21204HardwareToSonicV2JointPositionActionCfg,
)
from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
    DAD_DANCE_FRAME_COUNT,
    DAD_DANCE_SHA256,
    make_native124_selected_v2_causal_adaptation_env_cfg as _make_causal_adaptation_env_cfg,
    resolve_causal_reset_anchor_index,
)
from gear_sonic.envs.mjlab.sonic_true23 import (
    SONIC_TRUE23_CRITIC_DIM,
    SONIC_TRUE23_POLICY_DIM,
)
from gear_sonic.envs.mjlab.sonic_true23_causal_history_safe_target_v11 import (
    EVALUATOR_ALIGNED_RECOVERY_WEIGHT,
    evaluator_aligned_recovery_metric,
)
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    ISAACLAB_TO_MUJOCO_DOF,
    NATIVE_IL23_JOINT_NAMES,
    TARGET_DOF,
)
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTION_SCALE_HARDWARE as SELECTED_21204_ACTION_SCALE_HARDWARE,
)

CONTROL_DT_S = 0.02
EPISODE_STEPS = 500
EPISODE_LENGTH_S = CONTROL_DT_S * EPISODE_STEPS


def nominal_ankle_episode_steps_for_anchor(fixed_anchor_index: int | None) -> int:
    """Return the 500-step nominal cap without losing the final q10 proof."""

    reset_anchor = resolve_causal_reset_anchor_index(fixed_anchor_index)
    causal_safe_steps = DAD_DANCE_FRAME_COUNT - reset_anchor - 1
    return min(EPISODE_STEPS, causal_safe_steps)


ANKLE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
EE_TERMINATION_BODY_NAMES = (
    *ANKLE_BODY_NAMES,
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
)
ANKLE_SITE_NAMES = ("left_foot", "right_foot")
ANKLE_HARDWARE_INDICES = (4, 5, 10, 11)
ANKLE_NATIVE_INDICES = tuple(ISAACLAB_TO_MUJOCO_DOF[index] for index in ANKLE_HARDWARE_INDICES)
ANKLE_JOINT_NAMES = tuple(HARDWARE_23_JOINT_NAMES[index] for index in ANKLE_HARDWARE_INDICES)

FOOT_CONTACT_SENSOR_NAME = "feet_ground_contact"
DISABLED_NOMINAL_EVENTS = (
    "push_robot",
    "base_com",
    "encoder_bias",
    "foot_friction",
)

# Selected actor's source task used these physical costs and ankle-position
# scale.  Causal v9 established the stronger uncapped target barrier and v4
# established the death cost needed to prevent early-termination optimization.
ANKLE_POSITION_WEIGHT = 1.5
ANKLE_POSITION_STD_M = 0.15
ANKLE_ORIENTATION_WEIGHT = 1.0
ANKLE_ORIENTATION_STD_RAD = 0.25
ROOT_POSITION_WEIGHT = 0.5
ROOT_POSITION_STD_M = 0.3
ROOT_ORIENTATION_WEIGHT = 0.5
ROOT_ORIENTATION_STD_RAD = 0.4
TORQUE_WEIGHT = -2.0e-3
ACTUATOR_SATURATION_WEIGHT = -5.0
ACTUATOR_SATURATION_THRESHOLD_RATIO = 0.9
TARGET_SOFT_LIMIT_WEIGHT = -50.0
TARGET_SOFT_LIMIT_MARGIN_FRACTION = 0.025
ACTUAL_SOFT_LIMIT_WEIGHT = -20.0
ACTION_RATE_WEIGHT = -0.2
V2_ANKLE_PROJECTION_WEIGHT = -0.5
FOOT_SLIP_WEIGHT = -0.25
SOFT_LANDING_WEIGHT = -1.0e-3
ALIVE_WEIGHT = 5.0
NON_TIMEOUT_TERMINATION_WEIGHT = -5000.0


def _selected_action_term(env: Any, action_name: str) -> Any:
    term = env.action_manager.get_term(action_name)
    required = (
        "candidate_target_hardware",
        "final_target_hardware",
        "raw_clip_mask_native",
    )
    if any(not hasattr(term, name) for name in required):
        raise ValueError("ankle task requires selected-21204 SONIC V2 action term")
    return term


def v2_ankle_projection_l2(
    env: Any,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Penalize ankle target displacement introduced by V2 in selected units."""

    term = _selected_action_term(env, action_name)
    candidate = term.candidate_target_hardware
    final = term.final_target_hardware
    if (
        not isinstance(candidate, torch.Tensor)
        or not isinstance(final, torch.Tensor)
        or candidate.shape != (env.num_envs, TARGET_DOF)
        or final.shape != candidate.shape
        or candidate.dtype != torch.float32
        or final.dtype != torch.float32
        or not torch.isfinite(candidate).all()
        or not torch.isfinite(final).all()
    ):
        raise ValueError("V2 action diagnostics must be finite float32 [env,23]")
    indices = torch.as_tensor(
        ANKLE_HARDWARE_INDICES,
        dtype=torch.long,
        device=candidate.device,
    )
    scale = torch.as_tensor(
        SELECTED_21204_ACTION_SCALE_HARDWARE,
        dtype=torch.float32,
        device=candidate.device,
    ).index_select(0, indices)
    projection = candidate.index_select(1, indices) - final.index_select(1, indices)
    return torch.mean(torch.square(projection / scale), dim=1)


def v2_raw_clip_violation(
    env: Any,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Fail closed when any plain SONIC native action reaches ``abs(raw)>=10``."""

    term = _selected_action_term(env, action_name)
    mask = term.raw_clip_mask_native
    if not isinstance(mask, torch.Tensor) or mask.shape != (env.num_envs, TARGET_DOF) or mask.dtype != torch.bool:
        raise ValueError("V2 raw clip mask must be bool [env,23]")
    return torch.any(mask, dim=1)


def always_on_feet_slip(
    env: Any,
    sensor_name: str,
    asset_cfg: Any,
) -> torch.Tensor:
    """Penalize actual contacting-foot XY velocity; no command/contact labels."""

    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_name]
    found = sensor.data.found
    velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    expected = (env.num_envs, len(ANKLE_SITE_NAMES))
    if (
        not isinstance(found, torch.Tensor)
        or found.shape != expected
        or velocity.shape != (*expected, 2)
        or not torch.isfinite(found).all()
        or not torch.isfinite(velocity).all()
    ):
        raise ValueError("foot slip requires finite two-foot contact and site velocity")
    in_contact = found > 0
    return torch.sum(torch.sum(torch.square(velocity), dim=-1) * in_contact, dim=1)


def causal_proof_frame_guard(
    env: Any,
    command_name: str = "motion",
) -> torch.Tensor:
    """Raise before q9 loses its real q10 forward-difference proof frame."""

    command = env.command_manager.get_term(command_name)
    time_steps = getattr(command, "time_steps", None)
    motion = getattr(command, "motion", None)
    total = getattr(motion, "time_step_total", None)
    if (
        not isinstance(time_steps, torch.Tensor)
        or not isinstance(total, int)
        or time_steps.shape != (env.num_envs,)
    ):
        raise ValueError("causal proof guard requires command time-step state")
    if torch.any(time_steps + 1 >= total):
        raise RuntimeError("causal q9 reference lacks a real q10 proof frame")
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)


def make_native124_selected_v2_ankle_task_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
    play: bool = False,
    fixed_anchor_index: int | None = None,
) -> Any:
    """Build nominal, fixed-start DadDance acquisition for four ankle rows."""

    from mjlab.envs import mdp as env_mdp
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    import src.tasks.tracking.mdp as tracking_mdp
    import src.tasks.velocity.mdp as velocity_mdp

    cfg = _make_causal_adaptation_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        play=play,
        fixed_anchor_index=fixed_anchor_index,
    )
    if not isinstance(
        cfg.actions.get("joint_pos"),
        Selected21204HardwareToSonicV2JointPositionActionCfg,
    ):
        raise ValueError("ankle task base lacks selected-21204 SONIC V2 action")

    control_dt = float(cfg.sim.mujoco.timestep) * int(cfg.decimation)
    if abs(control_dt - CONTROL_DT_S) > 1.0e-12:
        raise ValueError("ankle task requires exact 50 Hz control")
    episode_steps = nominal_ankle_episode_steps_for_anchor(fixed_anchor_index)
    cfg.episode_length_s = episode_steps * CONTROL_DT_S

    command = cfg.commands["motion"]
    command.sampling_mode = "start"
    command.pose_range = {}
    command.velocity_range = {}
    command.joint_position_range = (0.0, 0.0)
    for name in DISABLED_NOMINAL_EVENTS:
        cfg.events.pop(name, None)

    sensors = tuple(cfg.scene.sensors or ())
    if any(sensor.name == FOOT_CONTACT_SENSOR_NAME for sensor in sensors):
        raise ValueError("ankle task foot-contact sensor name already exists")
    cfg.scene.sensors = sensors + (
        ContactSensorCfg(
            name=FOOT_CONTACT_SENSOR_NAME,
            primary=ContactMatch(
                mode="subtree",
                pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
            history_length=4,
        ),
    )

    # Joint-reference action imitation is intentionally invalid for this task.
    if cfg.rewards.pop("action_target_reference_l2", None) is None:
        raise ValueError("causal recovery joint-target term unexpectedly absent")

    ankle_pos = copy.deepcopy(cfg.rewards["motion_body_pos"])
    ankle_pos.weight = ANKLE_POSITION_WEIGHT
    ankle_pos.params = {
        "command_name": "motion",
        "std": ANKLE_POSITION_STD_M,
        "body_names": ANKLE_BODY_NAMES,
    }
    cfg.rewards["motion_ankle_pos"] = ankle_pos

    ankle_ori = copy.deepcopy(cfg.rewards["motion_body_ori"])
    ankle_ori.weight = ANKLE_ORIENTATION_WEIGHT
    ankle_ori.params = {
        "command_name": "motion",
        "std": ANKLE_ORIENTATION_STD_RAD,
        "body_names": ANKLE_BODY_NAMES,
    }
    cfg.rewards["motion_ankle_ori"] = ankle_ori

    root_pos = cfg.rewards["motion_global_root_pos"]
    root_pos.weight = ROOT_POSITION_WEIGHT
    root_pos.params = {
        "command_name": "motion",
        "std": ROOT_POSITION_STD_M,
    }
    root_ori = cfg.rewards["motion_global_root_ori"]
    root_ori.weight = ROOT_ORIENTATION_WEIGHT
    root_ori.params = {
        "command_name": "motion",
        "std": ROOT_ORIENTATION_STD_RAD,
    }

    actuator_cfg = SceneEntityCfg("robot", actuator_names=(".*",))
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=tracking_mdp.joint_torques_l2,
        weight=TORQUE_WEIGHT,
        params={"asset_cfg": actuator_cfg},
    )
    cfg.rewards["actuator_saturation"] = RewardTermCfg(
        func=tracking_mdp.actuator_saturation_cost,
        weight=ACTUATOR_SATURATION_WEIGHT,
        params={
            "asset_cfg": copy.deepcopy(actuator_cfg),
            "threshold_ratio": ACTUATOR_SATURATION_THRESHOLD_RATIO,
        },
    )
    target_barrier = cfg.rewards["action_target_soft_limit_barrier"]
    target_barrier.weight = TARGET_SOFT_LIMIT_WEIGHT
    target_barrier.params = {
        "action_name": "joint_pos",
        "entity_name": "robot",
        "margin_fraction": TARGET_SOFT_LIMIT_MARGIN_FRACTION,
    }
    cfg.rewards["joint_limit"].weight = ACTUAL_SOFT_LIMIT_WEIGHT
    cfg.rewards["action_rate_l2"].weight = ACTION_RATE_WEIGHT
    cfg.rewards["v2_ankle_projection_l2"] = RewardTermCfg(
        func=v2_ankle_projection_l2,
        weight=V2_ANKLE_PROJECTION_WEIGHT,
        params={"action_name": "joint_pos"},
    )
    cfg.rewards["evaluator_aligned_recovery"] = RewardTermCfg(
        func=evaluator_aligned_recovery_metric,
        weight=EVALUATOR_ALIGNED_RECOVERY_WEIGHT,
        params={
            "command_name": "motion",
            "action_name": "joint_pos",
            "entity_name": "robot",
        },
    )

    foot_cfg = SceneEntityCfg("robot", site_names=ANKLE_SITE_NAMES)
    cfg.rewards["feet_slip"] = RewardTermCfg(
        func=always_on_feet_slip,
        weight=FOOT_SLIP_WEIGHT,
        params={
            "sensor_name": FOOT_CONTACT_SENSOR_NAME,
            "asset_cfg": foot_cfg,
        },
    )
    cfg.rewards["soft_landing"] = RewardTermCfg(
        func=velocity_mdp.soft_landing,
        weight=SOFT_LANDING_WEIGHT,
        params={
            "sensor_name": FOOT_CONTACT_SENSOR_NAME,
            "command_name": None,
        },
    )
    cfg.rewards["alive"] = RewardTermCfg(func=env_mdp.is_alive, weight=ALIVE_WEIGHT)
    cfg.rewards["non_timeout_termination"] = RewardTermCfg(
        func=env_mdp.is_terminated,
        weight=NON_TIMEOUT_TERMINATION_WEIGHT,
    )
    cfg.rewards["causal_proof_frame_guard"] = RewardTermCfg(
        func=causal_proof_frame_guard,
        weight=1.0,
        params={"command_name": "motion"},
    )
    cfg.terminations["v2_raw_clip"] = TerminationTermCfg(
        func=v2_raw_clip_violation,
        params={"action_name": "joint_pos"},
    )

    audit_native124_selected_v2_ankle_task_env_cfg(cfg)
    return cfg


def _entity_names(entity: Any, attribute: str) -> tuple[str, ...]:
    value = getattr(entity, attribute)
    return (value,) if isinstance(value, str) else tuple(value)


def audit_native124_selected_v2_ankle_task_env_cfg(cfg: Any) -> dict[str, object]:
    """Fail closed on objective, safety, causal, or contact-config drift."""

    action_cfg = cfg.actions.get("joint_pos")
    if not isinstance(
        action_cfg,
        Selected21204HardwareToSonicV2JointPositionActionCfg,
    ):
        raise ValueError("ankle task action contract drift")
    if "action_target_reference_l2" in cfg.rewards:
        raise ValueError("ankle task may not imitate reference joint targets")
    forbidden = {"motion_joint_pos", "motion_leg_joint_pos", "motion_joint_vel"}
    if forbidden.intersection(cfg.rewards):
        raise ValueError("ankle task contains forbidden joint-label reward")
    if any(name in cfg.events for name in DISABLED_NOMINAL_EVENTS):
        raise ValueError("stage-1 nominal task contains domain randomization")

    command = cfg.commands["motion"]
    if (
        command.sampling_mode != "start"
        or command.pose_range != {}
        or command.velocity_range != {}
        or tuple(command.joint_position_range) != (0.0, 0.0)
    ):
        raise ValueError("stage-1 DadDance reset contract drift")
    control_dt = float(cfg.sim.mujoco.timestep) * int(cfg.decimation)
    fixed_anchor_index = getattr(command, "fixed_anchor_index", None)
    episode_steps = nominal_ankle_episode_steps_for_anchor(fixed_anchor_index)
    expected_episode_length_s = episode_steps * CONTROL_DT_S
    if (
        abs(control_dt - CONTROL_DT_S) > 1.0e-12
        or abs(float(cfg.episode_length_s) - expected_episode_length_s) > 1.0e-12
    ):
        raise ValueError("stage-1 control horizon drift")

    sensor_matches = [
        sensor for sensor in tuple(cfg.scene.sensors or ()) if sensor.name == FOOT_CONTACT_SENSOR_NAME
    ]
    if len(sensor_matches) != 1:
        raise ValueError("stage-1 task requires one exact foot-contact sensor")
    sensor = sensor_matches[0]
    if (
        sensor.reduce != "netforce"
        or tuple(sensor.fields) != ("found", "force")
        or sensor.track_air_time is not True
        or int(sensor.history_length) != 4
        or sensor.primary.mode != "subtree"
        or sensor.primary.pattern != r"^(left_ankle_roll_link|right_ankle_roll_link)$"
        or sensor.primary.entity != "robot"
        or sensor.secondary.mode != "body"
        or sensor.secondary.pattern != "terrain"
    ):
        raise ValueError("stage-1 foot-contact sensor contract drift")

    expected_rewards = {
        "motion_ankle_pos": (ANKLE_POSITION_WEIGHT, ANKLE_POSITION_STD_M),
        "motion_ankle_ori": (
            ANKLE_ORIENTATION_WEIGHT,
            ANKLE_ORIENTATION_STD_RAD,
        ),
        "motion_global_root_pos": (ROOT_POSITION_WEIGHT, ROOT_POSITION_STD_M),
        "motion_global_root_ori": (
            ROOT_ORIENTATION_WEIGHT,
            ROOT_ORIENTATION_STD_RAD,
        ),
        "joint_torques_l2": (TORQUE_WEIGHT, None),
        "actuator_saturation": (ACTUATOR_SATURATION_WEIGHT, None),
        "action_target_soft_limit_barrier": (TARGET_SOFT_LIMIT_WEIGHT, None),
        "joint_limit": (ACTUAL_SOFT_LIMIT_WEIGHT, None),
        "action_rate_l2": (ACTION_RATE_WEIGHT, None),
        "v2_ankle_projection_l2": (V2_ANKLE_PROJECTION_WEIGHT, None),
        "evaluator_aligned_recovery": (
            EVALUATOR_ALIGNED_RECOVERY_WEIGHT,
            None,
        ),
        "feet_slip": (FOOT_SLIP_WEIGHT, None),
        "soft_landing": (SOFT_LANDING_WEIGHT, None),
        "alive": (ALIVE_WEIGHT, None),
        "non_timeout_termination": (NON_TIMEOUT_TERMINATION_WEIGHT, None),
    }
    for name, (weight, std) in expected_rewards.items():
        term = cfg.rewards.get(name)
        if term is None or float(term.weight) != weight:
            raise ValueError(f"stage-1 reward drift: {name}")
        if std is not None and float(term.params.get("std")) != std:
            raise ValueError(f"stage-1 reward std drift: {name}")
    for name in ("motion_ankle_pos", "motion_ankle_ori"):
        if tuple(cfg.rewards[name].params["body_names"]) != ANKLE_BODY_NAMES:
            raise ValueError(f"stage-1 ankle body selection drift: {name}")
    if float(cfg.rewards["actuator_saturation"].params["threshold_ratio"]) != ACTUATOR_SATURATION_THRESHOLD_RATIO:
        raise ValueError("stage-1 actuator saturation threshold drift")
    if cfg.rewards["action_target_soft_limit_barrier"].params != {
        "action_name": "joint_pos",
        "entity_name": "robot",
        "margin_fraction": TARGET_SOFT_LIMIT_MARGIN_FRACTION,
    }:
        raise ValueError("stage-1 target soft-limit margin drift")
    if (
        cfg.rewards["feet_slip"].func is not always_on_feet_slip
        or cfg.rewards["v2_ankle_projection_l2"].func is not v2_ankle_projection_l2
        or cfg.rewards["evaluator_aligned_recovery"].func is not evaluator_aligned_recovery_metric
        or cfg.terminations["v2_raw_clip"].func is not v2_raw_clip_violation
    ):
        raise ValueError("stage-1 custom objective function drift")
    if _entity_names(cfg.rewards["feet_slip"].params["asset_cfg"], "site_names") != ANKLE_SITE_NAMES:
        raise ValueError("stage-1 foot-site selection drift")

    terminations = cfg.terminations
    physical_thresholds = {
        "anchor_pos": 0.25,
        "anchor_ori": 0.8,
        "ee_body_pos": 0.25,
    }
    for name, threshold in physical_thresholds.items():
        if float(terminations[name].params["threshold"]) != threshold:
            raise ValueError(f"stage-1 physical termination drift: {name}")
    if tuple(terminations["ee_body_pos"].params["body_names"]) != (EE_TERMINATION_BODY_NAMES):
        raise ValueError("stage-1 EE termination body selection drift")
    if "v2_raw_clip" not in terminations:
        raise ValueError("stage-1 lacks fail-closed raw-clip termination")

    return {
        "schema": "g1_true23_native124_selected_v2_ankle_task_env_v1",
        "motion_sha256": DAD_DANCE_SHA256,
        "base": "native124_selected_v2_causal_adaptation",
        "control_dt_s": control_dt,
        "reset_anchor_q9": resolve_causal_reset_anchor_index(fixed_anchor_index),
        "episode_steps": episode_steps,
        "sampling_mode": "start",
        "critic_observation_dim": SONIC_TRUE23_CRITIC_DIM,
        "sonic_policy_history_dim": SONIC_TRUE23_POLICY_DIM,
        "ankle_hardware_indices": list(ANKLE_HARDWARE_INDICES),
        "ankle_native_indices": list(ANKLE_NATIVE_INDICES),
        "ankle_joint_names": list(ANKLE_JOINT_NAMES),
        "joint_target_imitation": False,
        "contact_label_supervision": False,
        "actual_contact_only": True,
        "domain_randomization": False,
        "raw_clip_termination": "abs_plain_sonic_raw_native_greater_than_or_equal_10",
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def native124_selected_v2_ankle_task_contract() -> dict[str, object]:
    """Static stage-1 contract; no training, support, or deployment claim."""

    return {
        "schema": "g1_true23_native124_selected_v2_ankle_task_v1",
        "motion": {
            "name": "B_DadDance.npz",
            "sha256": DAD_DANCE_SHA256,
            "sampling_mode": "start",
            "contact_labels_present": False,
        },
        "causal": {
            "reference": "q9",
            "forward_difference_proof": "q10",
            "actor_observation_dim": 124,
            "privileged_critic_dim": SONIC_TRUE23_CRITIC_DIM,
            "sonic_history_dim_preserved": SONIC_TRUE23_POLICY_DIM,
        },
        "trainable_output_rows": {
            "hardware_indices": list(ANKLE_HARDWARE_INDICES),
            "native_indices": list(ANKLE_NATIVE_INDICES),
            "joint_names": list(ANKLE_JOINT_NAMES),
        },
        "rewards": {
            "motion_ankle_pos": {
                "weight": ANKLE_POSITION_WEIGHT,
                "std_m": ANKLE_POSITION_STD_M,
            },
            "motion_ankle_ori": {
                "weight": ANKLE_ORIENTATION_WEIGHT,
                "std_rad": ANKLE_ORIENTATION_STD_RAD,
            },
            "motion_global_root_pos": {
                "weight": ROOT_POSITION_WEIGHT,
                "std_m": ROOT_POSITION_STD_M,
                "causal_reference": "q9_pelvis",
            },
            "motion_global_root_ori": {
                "weight": ROOT_ORIENTATION_WEIGHT,
                "std_rad": ROOT_ORIENTATION_STD_RAD,
                "causal_reference": "q9_pelvis",
            },
            "joint_torques_l2": {"weight": TORQUE_WEIGHT},
            "actuator_saturation": {
                "weight": ACTUATOR_SATURATION_WEIGHT,
                "threshold_ratio": ACTUATOR_SATURATION_THRESHOLD_RATIO,
            },
            "target_soft_limit": {
                "weight": TARGET_SOFT_LIMIT_WEIGHT,
                "inner_margin_fraction": TARGET_SOFT_LIMIT_MARGIN_FRACTION,
            },
            "actual_soft_limit": {"weight": ACTUAL_SOFT_LIMIT_WEIGHT},
            "action_rate": {"weight": ACTION_RATE_WEIGHT},
            "v2_ankle_projection": {"weight": V2_ANKLE_PROJECTION_WEIGHT},
            "evaluator_aligned_recovery": {
                "weight": EVALUATOR_ALIGNED_RECOVERY_WEIGHT,
                "metric": "pelvis_tilt+pelvis_height_error+applied_target_tracking_rmse",
            },
            "actual_contact_foot_slip": {"weight": FOOT_SLIP_WEIGHT},
            "actual_contact_soft_landing": {"weight": SOFT_LANDING_WEIGHT},
            "alive": {"weight": ALIVE_WEIGHT},
            "non_timeout_termination": {
                "weight": NON_TIMEOUT_TERMINATION_WEIGHT,
                "cost_per_event_at_50hz": (NON_TIMEOUT_TERMINATION_WEIGHT * CONTROL_DT_S),
            },
        },
        "forbidden_objectives": [
            "action_target_reference_l2",
            "motion_joint_pos",
            "motion_leg_joint_pos",
            "motion_joint_vel",
            "reference_contact_label_matching",
        ],
        "nominal_disabled_events": list(DISABLED_NOMINAL_EVENTS),
        "preserved_terminations": {
            "anchor_pos_m": 0.25,
            "anchor_ori_rad": 0.8,
            "ee_body_pos_z_m": 0.25,
            "v2_raw_clip": "any_abs_plain_sonic_raw_native_greater_than_or_equal_10",
        },
        "runner_action_clip_required": None,
        "offline_or_simulator_only": True,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


if len(set(ANKLE_HARDWARE_INDICES)) != 4 or len(set(ANKLE_NATIVE_INDICES)) != 4:
    raise AssertionError("ankle row contract must contain four unique rows")
if tuple(NATIVE_IL23_JOINT_NAMES[index] for index in ANKLE_NATIVE_INDICES) != ANKLE_JOINT_NAMES:
    raise AssertionError("ankle hardware/native row mapping drift")
