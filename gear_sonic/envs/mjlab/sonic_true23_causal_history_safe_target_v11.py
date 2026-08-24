"""V11 causal true23 task with the deployment target transform in-loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gear_sonic.envs.mjlab.sonic_true23 import (
    _MJLAB_IMPORT_ERROR,
    TARGET_DOF,
    NativeIl23JointPositionAction,
    NativeIl23JointPositionActionCfg,
    _motion_command,
    _require_last_dim,
)
from gear_sonic.envs.mjlab.sonic_true23_causal_history_disturbance_v9 import (
    audit_causal_history_disturbance_v9_env_cfg,
    make_causal_history_disturbance_v9_env_cfg,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_SOFT_LOWER_HARDWARE,
    SAFE_TARGET_SOFT_UPPER_HARDWARE,
    safe_target_transform_contract,
    safe_target_transform_torch,
)

EVALUATOR_ALIGNED_RECOVERY_WEIGHT = -25.0


def evaluator_aligned_recovery_metric(
    env: Any,
    command_name: str = "motion",
    action_name: str = "joint_pos",
    entity_name: str = "robot",
) -> torch.Tensor:
    """Match the promotion recovery signal continuously during training.

    This is the exact sum used by the MuJoCo gate: pelvis tilt, pelvis-height
    error, and RMS joint tracking error against the actually applied target.
    Continuous application makes every randomized push reward rapid return.
    """

    command = _motion_command(env, command_name)
    robot = env.scene[entity_name]
    action = env.action_manager.get_term(action_name)
    target = action.processed_action - robot.data.encoder_bias
    if target.shape != (env.num_envs, TARGET_DOF):
        raise ValueError("V11 recovery target must be exact hardware [env,23]")
    gravity = robot.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0))
    height_error = torch.abs(
        command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]
    )
    tracking_rmse = torch.sqrt(
        torch.mean(torch.square(target - robot.data.joint_pos), dim=-1)
    )
    return tilt + height_error + tracking_rmse


if _MJLAB_IMPORT_ERROR is None:

    @dataclass(kw_only=True)
    class SafeTargetNativeIl23JointPositionActionCfg(
        NativeIl23JointPositionActionCfg
    ):
        """Build the exact V11 differentiable safe-target action term."""

        def build(self, env: Any) -> "SafeTargetNativeIl23JointPositionAction":
            return SafeTargetNativeIl23JointPositionAction(self, env)


    class SafeTargetNativeIl23JointPositionAction(
        NativeIl23JointPositionAction
    ):
        """Apply asymmetric tanh limits and expose applied-action history."""

        cfg: SafeTargetNativeIl23JointPositionActionCfg

        def __init__(
            self,
            cfg: SafeTargetNativeIl23JointPositionActionCfg,
            env: Any,
        ) -> None:
            super().__init__(cfg, env)
            actual_limits = self._entity.data.soft_joint_pos_limits[
                :, self._target_ids, :
            ]
            expected_limits = torch.stack(
                (
                    torch.as_tensor(
                        SAFE_TARGET_SOFT_LOWER_HARDWARE,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.as_tensor(
                        SAFE_TARGET_SOFT_UPPER_HARDWARE,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                ),
                dim=-1,
            )
            if not torch.allclose(
                actual_limits,
                expected_limits.expand_as(actual_limits),
                atol=1.0e-5,
                rtol=0.0,
            ):
                raise ValueError("MJLab soft limits differ from V11 bound constants")
            self._safe_native_actions = torch.zeros_like(self._raw_actions)

        @property
        def safe_native_action(self) -> torch.Tensor:
            return self._safe_native_actions

        def process_actions(self, actions: torch.Tensor) -> None:
            _require_last_dim(actions, TARGET_DOF, "V11 raw native action")
            self._raw_actions[:] = actions
            safe_native, target_unbiased = safe_target_transform_torch(actions)
            self._safe_native_actions[:] = safe_native
            self._processed_actions[:] = target_unbiased

        def reset(
            self,
            env_ids: torch.Tensor | slice | None = None,
        ) -> None:
            if env_ids is None:
                env_ids = slice(None)
            self._raw_actions[env_ids] = 0.0
            self._safe_native_actions[env_ids] = 0.0
            default = torch.as_tensor(
                SAFE_TARGET_DEFAULT_Q_HARDWARE,
                dtype=torch.float32,
                device=self.device,
            )
            self._processed_actions[env_ids] = default

else:

    class SafeTargetNativeIl23JointPositionActionCfg:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR

    class SafeTargetNativeIl23JointPositionAction:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR


def make_causal_history_safe_target_v11_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
    play: bool = False,
) -> Any:
    """Build V9 push/DR physics with the V11 target transform in-loop."""

    cfg = make_causal_history_disturbance_v9_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        play=play,
    )
    cfg.actions["joint_pos"] = SafeTargetNativeIl23JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
    )
    from mjlab.managers.reward_manager import RewardTermCfg

    cfg.rewards["evaluator_aligned_rapid_recovery"] = RewardTermCfg(
        func=evaluator_aligned_recovery_metric,
        weight=EVALUATOR_ALIGNED_RECOVERY_WEIGHT,
        params={
            "command_name": "motion",
            "action_name": "joint_pos",
            "entity_name": "robot",
        },
    )
    audit_causal_history_safe_target_v11_env_cfg(cfg, require_push=not play)
    return cfg


def audit_causal_history_safe_target_v11_env_cfg(
    cfg: Any,
    *,
    require_push: bool = True,
) -> dict[str, object]:
    """Fail closed unless V9 physics and V11 action semantics both execute."""

    v9 = audit_causal_history_disturbance_v9_env_cfg(
        cfg,
        require_push=require_push,
    )
    action_cfg = cfg.actions.get("joint_pos")
    if not isinstance(action_cfg, SafeTargetNativeIl23JointPositionActionCfg):
        raise ValueError("V11 executed environment lacks safe-target action cfg")
    if action_cfg.entity_name != "robot" or tuple(action_cfg.actuator_names) != (
        ".*",
    ):
        raise ValueError("V11 action selection differs from exact true23 contract")
    recovery = cfg.rewards.get("evaluator_aligned_rapid_recovery")
    if (
        recovery is None
        or recovery.func is not evaluator_aligned_recovery_metric
        or float(recovery.weight) != EVALUATOR_ALIGNED_RECOVERY_WEIGHT
        or recovery.params
        != {
            "command_name": "motion",
            "action_name": "joint_pos",
            "entity_name": "robot",
        }
    ):
        raise ValueError("V11 evaluator-aligned recovery reward differs")
    return {
        "schema": "g1_true23_causal_history_safe_target_env_v11",
        "v9_environment": v9,
        "action_cfg": (
            f"{type(action_cfg).__module__}:{type(action_cfg).__qualname__}"
        ),
        "target_transform": safe_target_transform_contract(),
        "history_uses_applied_safe_native_action": True,
        "encoder_bias_applied_after_unbiased_target": True,
        "evaluator_aligned_recovery": {
            "function": (
                f"{recovery.func.__module__}:{recovery.func.__qualname__}"
            ),
            "weight": float(recovery.weight),
            "metric": "tilt+abs(pelvis_height-reference)+target_tracking_rmse",
            "continuous_under_interval_pushes": True,
        },
    }


def causal_history_safe_target_v11_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_causal_history_safe_target_v11",
        "restart_from_model0": True,
        "v9_push_domain_randomization_and_physical_gates_unchanged": True,
        "safe_target_transform": safe_target_transform_contract(),
        "target_transform_trained_in_loop": True,
        "previous_action_is_applied_safe_native_action": True,
        "evaluator_aligned_rapid_recovery_weight": (
            EVALUATOR_ALIGNED_RECOVERY_WEIGHT
        ),
        "post_hoc_clamp_relabel": False,
        "deployment_ready": False,
    }
