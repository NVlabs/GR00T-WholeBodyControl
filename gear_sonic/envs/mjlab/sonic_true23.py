"""Exact SONIC true23 observation and action semantics for MJLab 1.2.

Stock ``Unitree-G1-23Dof-Tracking`` remains a useful teacher task.  This
module builds a separate task whose policy boundary matches the released
SONIC encoder/decoder contract:

* ``tokenizer``: 268 values (one routing bit plus the exact 267 encoder input)
* ``policy``: term-major H10 padded-IL29 proprioception, 930 values
* ``critic``: the stock privileged MJLab tracking observation
* action: native PhysX/IsaacLab 23 order at the policy boundary, converted to
  hardware/MuJoCo order inside the action term

MJLab and Unitree's task package are optional at import time so tensor-contract
tests can run before the simulator environment is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import torch

from gear_sonic.utils.g1_23dof_contract import (
    DEFAULT_REFERENCE_PROFILE,
    DEPLOYMENT_HISTORY_LENGTH,
    HARDWARE_23_ACTION_SCALE,
    HARDWARE_23_JOINT_NAMES,
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_TO_CANONICAL_IL29,
    REFERENCE_PROFILES,
    SOURCE_DOF,
    SOURCE_IL29_EXCLUDED_INDICES,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
    TELEOP_ENCODER_INPUT_TERM_DIMS,
)

SONIC_TRUE23_TASK_ID = "Unitree-G1-23Dof-SONIC-True23-Tracking"
SONIC_TRUE23_TOKENIZER_TERM_DIMS: Mapping[str, int] = {
    "encoder_index": 1,
    "command_multi_future_lower_body": 240,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "motion_anchor_ori_b": 6,
}
SONIC_TRUE23_POLICY_TERM_DIMS: Mapping[str, int] = {
    "base_ang_vel": 3 * DEPLOYMENT_HISTORY_LENGTH,
    "joint_pos_rel": SOURCE_DOF * DEPLOYMENT_HISTORY_LENGTH,
    "joint_vel": SOURCE_DOF * DEPLOYMENT_HISTORY_LENGTH,
    "previous_action": SOURCE_DOF * DEPLOYMENT_HISTORY_LENGTH,
    "projected_gravity": 3 * DEPLOYMENT_HISTORY_LENGTH,
}
SONIC_TRUE23_TOKENIZER_DIM = sum(SONIC_TRUE23_TOKENIZER_TERM_DIMS.values())
SONIC_TRUE23_POLICY_DIM = sum(SONIC_TRUE23_POLICY_TERM_DIMS.values())
# Stock Unitree task critic: command46 + anchor3 + anchor orientation6 +
# 14 body positions42 + 14 body orientations84 + base velocities6 +
# q23 + dq23 + action23.
SONIC_TRUE23_CRITIC_DIM = 256

if SONIC_TRUE23_TOKENIZER_DIM != TELEOP_ENCODER_INPUT_DIM + 1:
    raise AssertionError("SONIC tokenizer dimension contract drift")
if tuple(SONIC_TRUE23_TOKENIZER_TERM_DIMS.values())[1:] != (TELEOP_ENCODER_INPUT_TERM_DIMS):
    raise AssertionError("SONIC tokenizer term contract drift")
if SONIC_TRUE23_POLICY_DIM != 930:
    raise AssertionError("SONIC H10 proprioception dimension contract drift")

# KNEES_BENT_KEYFRAME from Unitree's MJLab asset, in hardware/MuJoCo order.
# This is also the default-q contract used by deployment and sim2sim.
SONIC_HARDWARE_DEFAULT_Q = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
)
_VR_BODY_NAMES = (
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
    "torso_link",
)
_VR_BODY_OFFSETS = (
    (0.18, -0.025, 0.0),
    (0.18, 0.025, 0.0),
    (0.0, 0.0, 0.35),
)
_REQUIRED_REFERENCE_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_rubber_hand",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_rubber_hand",
)
_TERMINATION_END_EFFECTOR_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
)


def _require_last_dim(value: torch.Tensor, size: int, context: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise ValueError(f"{context} must be a batched tensor")
    if value.shape[-1] != size:
        raise ValueError(f"{context} must have last dimension {size}, got {value.shape[-1]}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{context} contains NaN or Inf")


def _clamp_reference_start_indices_(
    time_steps: torch.Tensor,
    env_ids: torch.Tensor,
    maximum: int,
) -> None:
    """Clamp selected starts with write-back after advanced indexing."""

    if maximum < 0:
        raise ValueError("reference maximum start must be non-negative")
    time_steps[env_ids] = time_steps[env_ids].clamp(max=maximum)


def hardware_to_native_il23(value: torch.Tensor) -> torch.Tensor:
    """Convert compact hardware/MuJoCo order to native PhysX/IsaacLab order."""

    _require_last_dim(value, TARGET_DOF, "hardware true23 tensor")
    indices = torch.as_tensor(
        MUJOCO_TO_ISAACLAB_DOF,
        dtype=torch.long,
        device=value.device,
    )
    return value.index_select(-1, indices)


def native_il23_to_hardware(value: torch.Tensor) -> torch.Tensor:
    """Convert native PhysX/IsaacLab order to compact hardware/MuJoCo order."""

    _require_last_dim(value, TARGET_DOF, "native true23 tensor")
    indices = torch.as_tensor(
        ISAACLAB_TO_MUJOCO_DOF,
        dtype=torch.long,
        device=value.device,
    )
    return value.index_select(-1, indices)


def pad_native_il23_to_canonical_il29(value: torch.Tensor) -> torch.Tensor:
    """Scatter native 23 values into canonical IL29; six absent slots stay zero."""

    _require_last_dim(value, TARGET_DOF, "native true23 tensor")
    result = value.new_zeros((*value.shape[:-1], SOURCE_DOF))
    indices = torch.as_tensor(
        NATIVE_IL23_TO_CANONICAL_IL29,
        dtype=torch.long,
        device=value.device,
    )
    result.index_copy_(-1, indices, value)
    return result


def native_actions_to_hardware_targets(
    native_action: torch.Tensor,
    default_q_hardware: torch.Tensor | Sequence[float] = SONIC_HARDWARE_DEFAULT_Q,
) -> torch.Tensor:
    """Apply exact SONIC order conversion, scale, and default-position offset."""

    hardware_action = native_il23_to_hardware(native_action)
    default_q = torch.as_tensor(
        default_q_hardware,
        dtype=native_action.dtype,
        device=native_action.device,
    )
    scale = torch.as_tensor(
        HARDWARE_23_ACTION_SCALE,
        dtype=native_action.dtype,
        device=native_action.device,
    )
    if default_q.shape != (TARGET_DOF,):
        raise ValueError("default_q_hardware must contain 23 values")
    return default_q + hardware_action * scale


def flatten_term_major_policy_history(
    *,
    base_ang_vel: torch.Tensor,
    joint_pos_rel: torch.Tensor,
    joint_vel: torch.Tensor,
    previous_action: torch.Tensor,
    projected_gravity: torch.Tensor,
) -> torch.Tensor:
    """Flatten five oldest-to-newest histories in released term-major order."""

    terms = (
        ("base_ang_vel", base_ang_vel, 3),
        ("joint_pos_rel", joint_pos_rel, SOURCE_DOF),
        ("joint_vel", joint_vel, SOURCE_DOF),
        ("previous_action", previous_action, SOURCE_DOF),
        ("projected_gravity", projected_gravity, 3),
    )
    flattened: list[torch.Tensor] = []
    for name, value, width in terms:
        if value.ndim != 3 or value.shape[1:] != (
            DEPLOYMENT_HISTORY_LENGTH,
            width,
        ):
            raise ValueError(f"{name} must have shape [batch, {DEPLOYMENT_HISTORY_LENGTH}, {width}]")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
        flattened.append(value.flatten(start_dim=1))
    result = torch.cat(flattened, dim=-1)
    if result.shape[-1] != SONIC_TRUE23_POLICY_DIM:
        raise AssertionError("term-major policy history shape drift")
    for value in (joint_pos_rel, joint_vel, previous_action):
        if torch.count_nonzero(value[..., list(SOURCE_IL29_EXCLUDED_INDICES)]):
            raise ValueError("missing canonical IL29 slot became non-zero")
    return result


def prime_sonic_true23_training_environment(
    wrapped_env: Any,
    *,
    max_reset_attempts: int = 32,
) -> dict[str, Any]:
    """Refresh reset-time targets without a fake simulation transition.

    ``RslRlVecEnvWrapper`` performs the final explicit reset.  Upstream reset
    does not refresh ``MotionCommand.body_*_relative_w``; its first step would
    therefore terminate against stale targets.  The custom command owns a
    target-only refresh that changes no physics, command clock, curriculum, or
    RNG state.  Reject rare randomized terminal starts as a full batch, then
    rebuild observation histories from one coherent frame.
    """

    if isinstance(max_reset_attempts, bool) or not isinstance(max_reset_attempts, int) or max_reset_attempts <= 0:
        raise ValueError("max_reset_attempts must be a positive integer")
    env = getattr(wrapped_env, "unwrapped", None)
    if env is None or env is wrapped_env:
        raise TypeError("true23 target refresh requires a constructed RslRlVecEnvWrapper")
    common_step_counter = int(env.common_step_counter)
    simulation_step_counter = int(env._sim_step_counter)
    if common_step_counter != 0 or simulation_step_counter != 0:
        raise RuntimeError("true23 target refresh must run before any simulation step")

    all_env_ids = torch.arange(
        env.num_envs,
        dtype=torch.long,
        device=env.device,
    )
    rejected_counts: list[dict[str, int]] = []
    prime_log: Mapping[str, Any] | None = None
    maximum_error = 0.0
    termination_counts: dict[str, int] = {}

    for attempt in range(1, max_reset_attempts + 1):
        if attempt > 1:
            env.reset(env_ids=all_env_ids)
        command = _motion_command(env, "motion")
        refresh = getattr(
            command,
            "refresh_relative_body_targets_after_reset",
            None,
        )
        if not callable(refresh):
            raise TypeError("true23 motion command lacks target-only reset refresh")

        robot = _robot(env)
        unchanged_tensors = {
            "time_steps": command.time_steps.detach().clone(),
            "time_left": command.time_left.detach().clone(),
            "command_counter": command.command_counter.detach().clone(),
            "episode_length": env.episode_length_buf.detach().clone(),
            "joint_pos": robot.data.joint_pos.detach().clone(),
            "joint_vel": robot.data.joint_vel.detach().clone(),
        }
        refresh()

        for name, before in unchanged_tensors.items():
            if name == "episode_length":
                after = env.episode_length_buf
            elif name in {"joint_pos", "joint_vel"}:
                after = getattr(robot.data, name)
            else:
                after = getattr(command, name)
            if not torch.equal(after, before):
                raise RuntimeError(f"true23 target-only refresh mutated {name}")
        if (
            int(env.common_step_counter) != common_step_counter
            or int(env._sim_step_counter) != simulation_step_counter
        ):
            raise RuntimeError("true23 target-only refresh advanced simulation counters")

        body_names = tuple(command.cfg.body_names)
        if any(name not in body_names for name in _TERMINATION_END_EFFECTOR_NAMES):
            raise ValueError("motion command omits termination end effector")
        body_indices = [body_names.index(name) for name in _TERMINATION_END_EFFECTOR_NAMES]
        error = torch.abs(
            command.body_pos_relative_w[:, body_indices, -1] - command.robot_body_pos_w[:, body_indices, -1]
        )
        maximum_error = float(error.max().detach().cpu())

        bad = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        termination_counts = {}
        for term_name in env.termination_manager.active_terms:
            term_cfg = env.termination_manager.get_term_cfg(term_name)
            if term_cfg.time_out:
                continue
            term_value = term_cfg.func(env, **term_cfg.params)
            if (
                not isinstance(term_value, torch.Tensor)
                or term_value.shape != (env.num_envs,)
                or term_value.dtype != torch.bool
            ):
                raise TypeError(f"termination {term_name!r} returned invalid shape/type")
            termination_counts[term_name] = int(torch.count_nonzero(term_value).detach().cpu())
            bad |= term_value
        if torch.any(bad):
            rejected_counts.append(dict(termination_counts))
            continue

        env.observation_manager.reset(all_env_ids)
        observations = env.observation_manager.compute(update_history=True)
        env.obs_buf = observations
        for group_name, expected_dim in (
            ("tokenizer", SONIC_TRUE23_TOKENIZER_DIM),
            ("policy", SONIC_TRUE23_POLICY_DIM),
            ("critic", SONIC_TRUE23_CRITIC_DIM),
        ):
            value = observations[group_name]
            _require_last_dim(
                value,
                expected_dim,
                f"primed {group_name} observation",
            )

        extras = env.extras
        if not isinstance(extras, dict):
            raise TypeError("MJLab true23 priming extras must be a dictionary")
        prime_log = extras.get("log")
        if prime_log is not None and not isinstance(prime_log, Mapping):
            raise TypeError("MJLab true23 priming log must be a mapping")
        # Reset-time entries are stale, but reward terms require the runner log
        # sink to exist before the first simulator transition.
        extras["log"] = {}
        termination_cfg = env.cfg.terminations["ee_body_pos"]
        threshold = float(termination_cfg.params["threshold"])
        return {
            "physics_steps": 0,
            "target_refreshes": attempt,
            "full_batch_reset_retries": attempt - 1,
            "max_reset_attempts": max_reset_attempts,
            "rejected_termination_counts": rejected_counts,
            "initial_termination_counts": termination_counts,
            "discarded_prime_log_entries": (len(prime_log) if prime_log is not None else 0),
            "post_prime_max_ee_z_error": maximum_error,
            "ee_z_termination_threshold": threshold,
            "common_step_counter": common_step_counter,
            "simulation_step_counter": simulation_step_counter,
        }

    raise RuntimeError(
        "true23 target refresh could not sample a nonterminal initial batch "
        f"after {max_reset_attempts} attempts; "
        f"last_termination_counts={termination_counts}, "
        f"max_ee_z_error={maximum_error}"
    )


def _robot(env: Any) -> Any:
    return env.scene["robot"]


def _motion_command(env: Any, command_name: str) -> Any:
    command = env.command_manager.get_term(command_name)
    if command is None:
        raise ValueError(f"motion command {command_name!r} is unavailable")
    return command


def _quat_normalize(quaternion: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm <= torch.finfo(quaternion.dtype).eps):
        raise ValueError("zero-norm quaternion")
    return quaternion / norm


def _quat_inverse(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = _quat_normalize(quaternion)
    result = quaternion.clone()
    result[..., 1:] *= -1
    return result


def _quat_multiply_raw(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    w1, x1, y1, z1 = left.unbind(dim=-1)
    w2, x2, y2, z2 = right.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _quat_normalize(
        _quat_multiply_raw(
            _quat_normalize(left),
            _quat_normalize(right),
        )
    )


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = _quat_normalize(quaternion)
    vector_quaternion = torch.cat(
        (torch.zeros_like(vector[..., :1]), vector),
        dim=-1,
    )
    return _quat_multiply_raw(
        _quat_multiply_raw(quaternion, vector_quaternion),
        _quat_inverse(quaternion),
    )[..., 1:]


def _rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = _quat_normalize(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    matrix = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape((*quaternion.shape[:-1], 3, 3))
    return matrix[..., :2].reshape((*quaternion.shape[:-1], 6))


def encoder_index(env: Any) -> torch.Tensor:
    """Single active teleop encoder route; not part of encoder's 267 values."""

    return torch.ones((env.num_envs, 1), dtype=torch.float32, device=env.device)


def command_multi_future_lower_body(
    env: Any,
    command_name: str,
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
) -> torch.Tensor:
    """Return 10 lower-body frames: positions120 followed by velocities120."""

    command = _motion_command(env, command_name)
    try:
        profile = REFERENCE_PROFILES[reference_profile]
    except KeyError as exc:
        raise ValueError(f"unsupported true23 reference profile: {reference_profile}") from exc
    offsets = torch.arange(
        0,
        profile.future_frame_step * 10,
        profile.future_frame_step,
        dtype=torch.long,
        device=command.time_steps.device,
    )
    indexes = command.time_steps[:, None] + offsets[None, :]
    # One proof frame beyond the last sampled frame must remain available.
    proof_indexes = indexes[:, -1] + 1
    total = int(command.motion.time_step_total)
    if torch.any(proof_indexes >= total):
        raise ValueError("motion reference lacks genuine future/proof frame for SONIC horizon")
    joint_pos = command.motion.joint_pos[indexes, :12].reshape(env.num_envs, -1)
    joint_vel = command.motion.joint_vel[indexes, :12].reshape(env.num_envs, -1)
    result = torch.cat((joint_pos, joint_vel), dim=-1)
    _require_last_dim(result, 240, "multi-future lower-body command")
    return result


def _reference_vr_points(command: Any) -> tuple[torch.Tensor, torch.Tensor]:
    body_names = tuple(command.cfg.body_names)
    if any(name not in body_names for name in _VR_BODY_NAMES):
        raise ValueError("motion command omits required SONIC VR reference body")
    body_indices = [body_names.index(name) for name in _VR_BODY_NAMES]
    body_pos = command.body_pos_w[:, body_indices]
    body_quat = command.body_quat_w[:, body_indices]
    offsets = torch.as_tensor(
        _VR_BODY_OFFSETS,
        dtype=body_pos.dtype,
        device=body_pos.device,
    ).expand(body_pos.shape[0], -1, -1)
    return body_pos + _quat_apply(body_quat, offsets), body_quat


def vr_3point_local_target(env: Any, command_name: str) -> torch.Tensor:
    """Reference wrists/torso points in current reference pelvis frame."""

    command = _motion_command(env, command_name)
    body_pos, _ = _reference_vr_points(command)
    anchor_quat = command.anchor_quat_w[:, None, :].expand(-1, 3, -1)
    local = _quat_apply(
        _quat_inverse(anchor_quat),
        body_pos - command.anchor_pos_w[:, None, :],
    )
    return local.reshape(env.num_envs, 9)


def vr_3point_local_orn_target(env: Any, command_name: str) -> torch.Tensor:
    """Reference wrist/torso quaternions in current reference pelvis frame."""

    command = _motion_command(env, command_name)
    _, body_quat = _reference_vr_points(command)
    anchor_quat = command.anchor_quat_w[:, None, :].expand(-1, 3, -1)
    local = _quat_multiply(_quat_inverse(anchor_quat), body_quat)
    return local.reshape(env.num_envs, 12)


def motion_anchor_ori_b(env: Any, command_name: str) -> torch.Tensor:
    """Reference pelvis orientation relative to measured robot pelvis."""

    command = _motion_command(env, command_name)
    relative = _quat_multiply(
        _quat_inverse(command.robot_anchor_quat_w),
        command.anchor_quat_w,
    )
    return _rotation_6d(relative)


def padded_joint_pos_rel(env: Any) -> torch.Tensor:
    robot = _robot(env)
    q_hardware = robot.data.joint_pos
    default = torch.as_tensor(
        SONIC_HARDWARE_DEFAULT_Q,
        dtype=q_hardware.dtype,
        device=q_hardware.device,
    )
    return pad_native_il23_to_canonical_il29(hardware_to_native_il23(q_hardware - default))


def padded_joint_vel(env: Any) -> torch.Tensor:
    return pad_native_il23_to_canonical_il29(hardware_to_native_il23(_robot(env).data.joint_vel))


def padded_previous_action(env: Any) -> torch.Tensor:
    # V11 exposes the action actually applied after its differentiable safety
    # transform.  Earlier action terms have no such property and retain their
    # original raw native-action history semantics.
    term = env.action_manager.get_term("joint_pos")
    action = getattr(term, "safe_native_action", env.action_manager.action)
    return pad_native_il23_to_canonical_il29(action)


def base_ang_vel(env: Any, sensor_name: str = "robot/imu_ang_vel") -> torch.Tensor:
    sensor = env.scene[sensor_name]
    value = sensor.data
    _require_last_dim(value, 3, "MJLab angular velocity sensor")
    return value


def projected_gravity(env: Any) -> torch.Tensor:
    value = _robot(env).data.projected_gravity_b
    _require_last_dim(value, 3, "MJLab projected gravity")
    return value


_MJLAB_IMPORT_ERROR: Exception | None = None
try:
    from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
    from mjlab.utils.lab_api.math import (
        quat_apply as _mjlab_quat_apply,
        quat_inv as _mjlab_quat_inv,
        quat_mul as _mjlab_quat_mul,
        yaw_quat as _mjlab_yaw_quat,
    )
    from src.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
except (ImportError, ModuleNotFoundError) as exc:
    _MJLAB_IMPORT_ERROR = exc
    ActionTerm = object  # type: ignore[assignment,misc]
    ActionTermCfg = object  # type: ignore[assignment,misc]
    MotionCommand = object  # type: ignore[assignment,misc]
    MotionCommandCfg = object  # type: ignore[assignment,misc]


if _MJLAB_IMPORT_ERROR is None:

    @dataclass(kw_only=True)
    class NativeIl23JointPositionActionCfg(ActionTermCfg):
        """Native-IL23 policy action converted internally to hardware order."""

        entity_name: str = "robot"
        actuator_names: tuple[str, ...] = (".*",)

        def build(self, env: Any) -> "NativeIl23JointPositionAction":
            return NativeIl23JointPositionAction(self, env)

    class NativeIl23JointPositionAction(ActionTerm):
        """Exact true23 position action with native raw-action history."""

        cfg: NativeIl23JointPositionActionCfg

        def __init__(
            self,
            cfg: NativeIl23JointPositionActionCfg,
            env: Any,
        ) -> None:
            super().__init__(cfg=cfg, env=env)
            target_ids, target_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
            if tuple(target_names) != HARDWARE_23_JOINT_NAMES:
                raise ValueError(
                    "MJLab actuated joint order differs from exact hardware true23 "
                    f"contract: {tuple(target_names)!r}"
                )
            self._target_ids = torch.as_tensor(
                target_ids,
                dtype=torch.long,
                device=self.device,
            )
            self._raw_actions = torch.zeros(
                (self.num_envs, TARGET_DOF),
                dtype=torch.float32,
                device=self.device,
            )
            self._processed_actions = torch.zeros_like(self._raw_actions)
            default = self._entity.data.default_joint_pos[:, self._target_ids].clone()
            expected = torch.as_tensor(
                SONIC_HARDWARE_DEFAULT_Q,
                dtype=default.dtype,
                device=default.device,
            )
            if not torch.allclose(default, expected.expand_as(default), atol=1e-6):
                raise ValueError("MJLab default joint positions differ from SONIC KNEES_BENT deployment offset")
            self._default_q_hardware = default

        @property
        def action_dim(self) -> int:
            return TARGET_DOF

        @property
        def raw_action(self) -> torch.Tensor:
            return self._raw_actions

        @property
        def processed_action(self) -> torch.Tensor:
            return self._processed_actions

        def process_actions(self, actions: torch.Tensor) -> None:
            _require_last_dim(actions, TARGET_DOF, "SONIC native action")
            self._raw_actions[:] = actions
            self._processed_actions[:] = native_actions_to_hardware_targets(
                actions,
                self._default_q_hardware[0],
            )

        def apply_actions(self) -> None:
            encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
            self._entity.set_joint_position_target(
                self._processed_actions - encoder_bias,
                joint_ids=self._target_ids,
            )

        def reset(
            self,
            env_ids: torch.Tensor | slice | None = None,
        ) -> None:
            if env_ids is None:
                env_ids = slice(None)
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = self._default_q_hardware[env_ids]

    @dataclass(kw_only=True)
    class SonicTrue23MotionCommandCfg(MotionCommandCfg):
        """Unitree motion command with a reserved SONIC future horizon."""

        reference_profile: str = DEFAULT_REFERENCE_PROFILE

        def build(self, env: Any) -> "SonicTrue23MotionCommand":
            return SonicTrue23MotionCommand(self, env)

    class SonicTrue23MotionCommand(MotionCommand):
        """Stock motion tracking plus a genuine, non-clamped future horizon."""

        cfg: SonicTrue23MotionCommandCfg

        def __init__(self, cfg: SonicTrue23MotionCommandCfg, env: Any) -> None:
            super().__init__(cfg, env)
            try:
                profile = REFERENCE_PROFILES[cfg.reference_profile]
            except KeyError as exc:
                raise ValueError(f"unsupported true23 reference profile: {cfg.reference_profile}") from exc
            self._sonic_proof_offset = profile.future_frame_step * 9 + 1
            if self.motion.time_step_total <= self._sonic_proof_offset:
                raise ValueError(
                    f"SONIC motion requires at least {self._sonic_proof_offset + 1} frames for reference horizon"
                )
            self._sonic_max_start = self.motion.time_step_total - self._sonic_proof_offset - 1

        def refresh_relative_body_targets_after_reset(self) -> None:
            """Refresh cached targets without advancing state or simulation."""

            body_count = len(self.cfg.body_names)
            anchor_pos_w = self.anchor_pos_w[:, None, :].repeat(
                1,
                body_count,
                1,
            )
            anchor_quat_w = self.anchor_quat_w[:, None, :].repeat(
                1,
                body_count,
                1,
            )
            robot_anchor_pos_w = self.robot_anchor_pos_w[:, None, :].repeat(
                1,
                body_count,
                1,
            )
            robot_anchor_quat_w = self.robot_anchor_quat_w[
                :,
                None,
                :,
            ].repeat(
                1,
                body_count,
                1,
            )

            delta_pos_w = robot_anchor_pos_w
            delta_pos_w[..., 2] = anchor_pos_w[..., 2]
            delta_ori_w = _mjlab_yaw_quat(
                _mjlab_quat_mul(
                    robot_anchor_quat_w,
                    _mjlab_quat_inv(anchor_quat_w),
                )
            )
            refreshed_quat = _mjlab_quat_mul(
                delta_ori_w,
                self.body_quat_w,
            )
            refreshed_pos = delta_pos_w + _mjlab_quat_apply(
                delta_ori_w,
                self.body_pos_w - anchor_pos_w,
            )
            if (
                refreshed_pos.shape != self.body_pos_relative_w.shape
                or refreshed_quat.shape != self.body_quat_relative_w.shape
                or not torch.isfinite(refreshed_pos).all()
                or not torch.isfinite(refreshed_quat).all()
            ):
                raise RuntimeError("true23 reset target refresh produced invalid tensors")
            self.body_pos_relative_w.copy_(refreshed_pos)
            self.body_quat_relative_w.copy_(refreshed_quat)

        def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
            self.time_steps[env_ids] = torch.randint(
                0,
                self._sonic_max_start + 1,
                (len(env_ids),),
                device=self.device,
            )
            self.metrics["sampling_entropy"][:] = 1.0
            self.metrics["sampling_top1_prob"][:] = 1.0 / max(
                self._sonic_max_start + 1,
                1,
            )
            self.metrics["sampling_top1_bin"][:] = 0.5

        def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
            super()._adaptive_sampling(env_ids)
            _clamp_reference_start_indices_(
                self.time_steps,
                env_ids,
                self._sonic_max_start,
            )

        def _update_command(self) -> None:
            exhausted = self.time_steps >= self._sonic_max_start
            if torch.any(exhausted):
                # Stock update increments then resamples at motion end.
                self.time_steps[exhausted] = self.motion.time_step_total - 1
            super()._update_command()

else:

    class NativeIl23JointPositionActionCfg:  # type: ignore[no-redef]
        """Unavailable placeholder when MJLab is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR

    class NativeIl23JointPositionAction:  # type: ignore[no-redef]
        """Unavailable placeholder when MJLab is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR

    class SonicTrue23MotionCommandCfg:
        """Unavailable placeholder when Unitree MJLab sources are absent."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR

    class SonicTrue23MotionCommand:
        """Unavailable placeholder when Unitree MJLab sources are absent."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR


def _require_mjlab() -> None:
    if _MJLAB_IMPORT_ERROR is not None:
        raise RuntimeError(
            "MJLab 1.2 and Unitree unitree_rl_mjlab task sources are required"
        ) from _MJLAB_IMPORT_ERROR


def _motion_cfg_kwargs(base_cfg: Any) -> dict[str, Any]:
    return {field.name: getattr(base_cfg, field.name) for field in fields(base_cfg)}


def _sonic_motion_cfg(base_cfg: Any, reference_profile: str) -> Any:
    """Create importable horizon-safe command config."""

    kwargs = _motion_cfg_kwargs(base_cfg)
    kwargs.update(
        {
            "anchor_body_name": "pelvis",
            "body_names": _REQUIRED_REFERENCE_BODY_NAMES,
            "reference_profile": reference_profile,
        }
    )
    return SonicTrue23MotionCommandCfg(**kwargs)


def make_sonic_true23_tracking_env_cfg(
    *,
    motion_file: str = "",
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
    num_envs: int | None = None,
    play: bool = False,
) -> Any:
    """Build Unitree MJLab task with exact SONIC policy boundary."""

    _require_mjlab()
    if reference_profile not in REFERENCE_PROFILES:
        raise ValueError(f"unsupported true23 reference profile: {reference_profile}")

    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.utils.noise import UniformNoiseCfg
    from src.assets.robots.unitree_g1.g1_23dof_constants import (
        KNEES_BENT_KEYFRAME,
    )
    from src.tasks.tracking.config.g1_23dof.env_cfgs import (
        unitree_g1_23dof_flat_tracking_env_cfg,
    )

    cfg = unitree_g1_23dof_flat_tracking_env_cfg(
        has_state_estimation=True,
        play=play,
    )
    if num_envs is not None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        cfg.scene.num_envs = num_envs

    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.init_state = KNEES_BENT_KEYFRAME

    base_motion_cfg = cfg.commands["motion"]
    base_motion_cfg.motion_file = motion_file
    cfg.commands["motion"] = _sonic_motion_cfg(
        base_motion_cfg,
        reference_profile,
    )
    cfg.actions = {
        "joint_pos": NativeIl23JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
        )
    }

    tokenizer = ObservationGroupCfg(
        terms={
            "encoder_index": ObservationTermCfg(func=encoder_index),
            "command_multi_future_lower_body": ObservationTermCfg(
                func=command_multi_future_lower_body,
                params={
                    "command_name": "motion",
                    "reference_profile": reference_profile,
                },
            ),
            "vr_3point_local_target": ObservationTermCfg(
                func=vr_3point_local_target,
                params={"command_name": "motion"},
            ),
            "vr_3point_local_orn_target": ObservationTermCfg(
                func=vr_3point_local_orn_target,
                params={"command_name": "motion"},
            ),
            "motion_anchor_ori_b": ObservationTermCfg(
                func=motion_anchor_ori_b,
                params={"command_name": "motion"},
                noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            ),
        },
        concatenate_terms=True,
        enable_corruption=not play,
        nan_policy="error",
    )
    policy = ObservationGroupCfg(
        terms={
            "base_ang_vel": ObservationTermCfg(
                func=base_ang_vel,
                params={"sensor_name": "robot/imu_ang_vel"},
                noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            ),
            "joint_pos_rel": ObservationTermCfg(func=padded_joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=padded_joint_vel),
            "previous_action": ObservationTermCfg(func=padded_previous_action),
            "projected_gravity": ObservationTermCfg(
                func=projected_gravity,
                noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            ),
        },
        concatenate_terms=True,
        enable_corruption=not play,
        history_length=DEPLOYMENT_HISTORY_LENGTH,
        flatten_history_dim=True,
        nan_policy="error",
    )
    critic = cfg.observations["critic"]
    cfg.observations = {
        "tokenizer": tokenizer,
        "policy": policy,
        "critic": critic,
    }
    return cfg


def register_sonic_true23_tracking_task(
    *,
    rl_cfg: Any,
    runner_cls: type | None,
    motion_file: str = "",
    task_id: str = SONIC_TRUE23_TASK_ID,
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
) -> str:
    """Register custom task after caller supplies custom exact-policy runner."""

    _require_mjlab()
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    if task_id in list_tasks():
        raise ValueError(f"task is already registered: {task_id}")
    register_mjlab_task(
        task_id=task_id,
        env_cfg=make_sonic_true23_tracking_env_cfg(
            motion_file=motion_file,
            reference_profile=reference_profile,
            play=False,
        ),
        play_env_cfg=make_sonic_true23_tracking_env_cfg(
            motion_file=motion_file,
            reference_profile=reference_profile,
            play=True,
        ),
        rl_cfg=rl_cfg,
        runner_cls=runner_cls,
    )
    return task_id
