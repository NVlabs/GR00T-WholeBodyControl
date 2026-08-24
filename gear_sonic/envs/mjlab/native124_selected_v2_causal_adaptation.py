"""Exact causal native124 boundary for selected-21204 -> SONIC V2 PPO.

This module is simulator-only.  It keeps the proven SONIC causal recovery
environment and privileged critic, but exposes an additional 124-value actor
group with the timing used by the checkpoint-21204 shadow collector.  The
ActionManager boundary remains selected-checkpoint raw hardware order and the
V2 safe-target transform is applied exactly once by the dedicated action term.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.native124_selected_v2_ankle_adaptation import (
    SELECTED_21204_HOME_Q_HARDWARE,
    Selected21204HardwareToSonicV2JointPositionAction,
    Selected21204HardwareToSonicV2JointPositionActionCfg,
)
from gear_sonic.envs.mjlab.sonic_true23 import (
    _MJLAB_IMPORT_ERROR,
    SONIC_TRUE23_CRITIC_DIM,
    SONIC_TRUE23_POLICY_DIM,
    prime_sonic_true23_training_environment,
)
from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    CAUSAL_HISTORY_ANCHOR_INDEX,
    CausalHistoryMotionCommand,
    CausalHistoryMotionCommandCfg,
    make_causal_history_recovery_env_cfg,
)
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    TARGET_DOF,
)

NATIVE124_OBSERVATION_DIM = 124
CONTROL_HZ = 50.0
CONTROL_DT_S = 1.0 / CONTROL_HZ
TORSO_BODY_NAME = "torso_link"
PELVIS_BODY_NAME = "pelvis"
DAD_DANCE_FRAME_COUNT = 2090
DAD_DANCE_SHA256 = "a4962f1e4df45ca70ada473a962b52527b17ac667dfb17ef3cd37d4ed21c3bfb"
DAD_DANCE_RELATIVE_PATH = Path("artifacts/g1_native124_multimotion/scaling_all61/feasible_v1/npz/B_DadDance.npz")


def resolve_causal_reset_anchor_index(
    fixed_anchor_index: int | None,
    *,
    causal_min_anchor: int = CAUSAL_HISTORY_ANCHOR_INDEX,
    causal_max_anchor: int = DAD_DANCE_FRAME_COUNT - 2,
) -> int:
    """Return one validated deterministic q9 reset anchor."""

    if (
        isinstance(causal_min_anchor, bool)
        or not isinstance(causal_min_anchor, int)
        or isinstance(causal_max_anchor, bool)
        or not isinstance(causal_max_anchor, int)
        or causal_min_anchor < 0
        or causal_max_anchor < causal_min_anchor
    ):
        raise ValueError("causal anchor bounds are invalid")
    if fixed_anchor_index is None:
        return causal_min_anchor
    if isinstance(fixed_anchor_index, bool) or not isinstance(fixed_anchor_index, int):
        raise ValueError("fixed_anchor_index must be an integer or None")
    if not causal_min_anchor <= fixed_anchor_index <= causal_max_anchor:
        raise ValueError(
            "fixed_anchor_index must preserve real q0..q10 history and q10 proof: "
            f"expected {causal_min_anchor}..{causal_max_anchor}, got {fixed_anchor_index}"
        )
    return fixed_anchor_index


def causal_episode_last_q9(
    *,
    reset_anchor_index: int,
    episode_steps: int,
    causal_max_anchor: int = DAD_DANCE_FRAME_COUNT - 2,
) -> int:
    """Validate an episode window and return its final action-bearing q9."""

    reset_anchor = resolve_causal_reset_anchor_index(
        reset_anchor_index,
        causal_max_anchor=causal_max_anchor,
    )
    if isinstance(episode_steps, bool) or not isinstance(episode_steps, int) or episode_steps <= 0:
        raise ValueError("episode_steps must be a positive integer")
    last_q9 = reset_anchor + episode_steps - 1
    if last_q9 > causal_max_anchor:
        raise ValueError(f"episode can overrun DadDance q10 proof: last q9 {last_q9} exceeds {causal_max_anchor}")
    return last_q9


try:
    from mjlab.rl import RslRlVecEnvWrapper
    from tensordict import TensorDict
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - WSL integration path.
    _WRAPPER_IMPORT_ERROR: Exception | None = exc
    RslRlVecEnvWrapper = object  # type: ignore[assignment,misc]
    TensorDict = Any  # type: ignore[assignment,misc]
else:
    _WRAPPER_IMPORT_ERROR = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dad_dance_motion_file(path: str | Path) -> Path:
    """Hash-lock the only motion admitted by the initial adaptation task."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.name != DAD_DANCE_RELATIVE_PATH.name:
        raise ValueError("causal adaptation requires the exact B_DadDance.npz file")
    actual = _sha256(resolved)
    if actual != DAD_DANCE_SHA256:
        raise ValueError(f"DadDance motion SHA256 mismatch: expected {DAD_DANCE_SHA256}, got {actual}")
    with np.load(resolved, allow_pickle=False) as data:
        required = {
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        if not required.issubset(data.files):
            raise ValueError("DadDance motion array contract is incomplete")
        joint_pos = data["joint_pos"]
        if joint_pos.shape != (DAD_DANCE_FRAME_COUNT, TARGET_DOF):
            raise ValueError("DadDance joint_pos shape drift")
        if any(not np.isfinite(data[name]).all() for name in required):
            raise ValueError("DadDance motion contains NaN or Inf")
    return resolved


def _require_float32_batch(
    value: torch.Tensor,
    width: int,
    context: str,
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[1] != width
        or (batch_size is not None and value.shape[0] != batch_size)
    ):
        rows = "batch" if batch_size is None else str(batch_size)
        raise ValueError(f"{context} must have shape [{rows},{width}]")
    if value.dtype != torch.float32:
        raise ValueError(f"{context} must be float32")
    if device is not None and value.device != device:
        raise ValueError(f"{context} device mismatch")
    if not torch.isfinite(value).all():
        raise ValueError(f"{context} contains NaN or Inf")
    return value


def _require_indices(
    q9_indices: torch.Tensor,
    *,
    batch_size: int,
    frame_count: int,
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(q9_indices, torch.Tensor)
        or q9_indices.shape != (batch_size,)
        or q9_indices.dtype != torch.long
        or q9_indices.device != device
    ):
        raise ValueError("q9 indices must be device-local int64 [batch]")
    if torch.any(q9_indices < CAUSAL_HISTORY_ANCHOR_INDEX):
        raise ValueError("q9 index lacks the required causal history")
    if torch.any(q9_indices + 1 >= frame_count):
        raise ValueError("q9 index lacks a q10 proof frame")
    return q9_indices


def causal_forward_difference_at_indices(
    motion_joint_pos: torch.Tensor,
    q9_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return q9, q10, and exact 50 Hz forward-difference qd reference."""

    if (
        not isinstance(motion_joint_pos, torch.Tensor)
        or motion_joint_pos.ndim != 2
        or motion_joint_pos.shape[1] != TARGET_DOF
        or motion_joint_pos.dtype != torch.float32
        or not torch.isfinite(motion_joint_pos).all()
    ):
        raise ValueError("motion joint_pos must be finite float32 [frames,23]")
    if not isinstance(q9_indices, torch.Tensor) or q9_indices.ndim != 1:
        raise ValueError("q9 indices must be a one-dimensional tensor")
    q9_indices = _require_indices(
        q9_indices,
        batch_size=q9_indices.shape[0],
        frame_count=motion_joint_pos.shape[0],
        device=motion_joint_pos.device,
    )
    q9 = motion_joint_pos.index_select(0, q9_indices)
    q10 = motion_joint_pos.index_select(0, q9_indices + 1)
    qd = (q10 - q9) * CONTROL_HZ
    return q9, q10, qd


def _normalize_quaternion(value: torch.Tensor, context: str) -> torch.Tensor:
    _require_float32_batch(value, 4, context)
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if torch.any(norm <= torch.finfo(value.dtype).eps):
        raise ValueError(f"{context} contains a zero-norm quaternion")
    if torch.any(torch.abs(norm - 1.0) > 1.0e-3):
        raise ValueError(f"{context} must contain unit WXYZ quaternions")
    return value / norm


def _quaternion_inverse(value: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    result[:, 1:] *= -1.0
    return result


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
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


def _yaw_quaternion(value: torch.Tensor) -> torch.Tensor:
    value = _normalize_quaternion(value, "yaw-source quaternion")
    w, x, y, z = value.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def virtual_q9_torso_quaternion(
    current_robot_torso_q10_wxyz: torch.Tensor,
    reference_torso_q9_wxyz: torch.Tensor,
    reference_torso_q10_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct reset q9 torso using the causal command's yaw alignment."""

    current = _normalize_quaternion(
        current_robot_torso_q10_wxyz,
        "current robot torso q10 quaternion",
    )
    reference_q9 = _normalize_quaternion(
        reference_torso_q9_wxyz,
        "reference torso q9 quaternion",
    )
    reference_q10 = _normalize_quaternion(
        reference_torso_q10_wxyz,
        "reference torso q10 quaternion",
    )
    if current.shape != reference_q9.shape or current.shape != reference_q10.shape:
        raise ValueError("virtual torso quaternion batches differ")
    alignment = _yaw_quaternion(_quaternion_multiply(current, _quaternion_inverse(reference_q10)))
    result = _quaternion_multiply(alignment, reference_q9)
    return _normalize_quaternion(result, "virtual robot torso q9 quaternion")


def relative_torso_rotation_6d(
    robot_torso_q9_wxyz: torch.Tensor,
    reference_torso_q9_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Match the checkpoint-21204 shadow's float64 quaternion-to-6D path."""

    _normalize_quaternion(
        robot_torso_q9_wxyz,
        "robot torso q9 quaternion",
    )
    _normalize_quaternion(
        reference_torso_q9_wxyz,
        "reference torso q9 quaternion",
    )
    # The durable shadow casts the original float32 samples to float64 before
    # normalization.  Normalizing in float32 first creates a small but
    # accumulating closed-loop action drift, so preserve that exact order.
    robot = robot_torso_q9_wxyz.to(dtype=torch.float64)
    robot = robot / torch.linalg.vector_norm(robot, dim=-1, keepdim=True)
    reference = reference_torso_q9_wxyz.to(dtype=torch.float64)
    reference = reference / torch.linalg.vector_norm(reference, dim=-1, keepdim=True)
    if robot.shape != reference.shape:
        raise ValueError("relative torso quaternion batches differ")
    relative = _quaternion_multiply(_quaternion_inverse(robot), reference)
    relative = relative / torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
    w, x, y, z = relative.unbind(dim=-1)
    matrix = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape((-1, 3, 3))
    return matrix[:, :, :2].reshape((-1, 6)).to(dtype=torch.float32)


def build_native124_selected_v2_causal_actor(
    *,
    reference_q9_hardware: torch.Tensor,
    reference_q10_hardware: torch.Tensor,
    robot_torso_q9_wxyz: torch.Tensor,
    reference_torso_q9_wxyz: torch.Tensor,
    base_angular_velocity_q10: torch.Tensor,
    joint_pos_biased_q10_hardware: torch.Tensor,
    joint_velocity_q10_hardware: torch.Tensor,
    previous_effective_selected_raw_hardware: torch.Tensor,
) -> torch.Tensor:
    """Build exact float32 ``[B,124]`` selected-checkpoint actor input."""

    values = (
        (reference_q9_hardware, TARGET_DOF, "reference q9 hardware"),
        (reference_q10_hardware, TARGET_DOF, "reference q10 hardware"),
        (robot_torso_q9_wxyz, 4, "robot torso q9 quaternion"),
        (reference_torso_q9_wxyz, 4, "reference torso q9 quaternion"),
        (base_angular_velocity_q10, 3, "base angular velocity q10"),
        (joint_pos_biased_q10_hardware, TARGET_DOF, "biased joint position q10"),
        (joint_velocity_q10_hardware, TARGET_DOF, "joint velocity q10"),
        (
            previous_effective_selected_raw_hardware,
            TARGET_DOF,
            "previous effective selected raw action",
        ),
    )
    first = values[0][0]
    _require_float32_batch(first, values[0][1], values[0][2])
    batch_size = first.shape[0]
    device = first.device
    for value, width, context in values[1:]:
        _require_float32_batch(
            value,
            width,
            context,
            batch_size=batch_size,
            device=device,
        )
    home = torch.as_tensor(
        SELECTED_21204_HOME_Q_HARDWARE,
        dtype=torch.float32,
        device=device,
    )
    result = torch.cat(
        (
            reference_q9_hardware,
            (reference_q10_hardware - reference_q9_hardware) * CONTROL_HZ,
            relative_torso_rotation_6d(
                robot_torso_q9_wxyz,
                reference_torso_q9_wxyz,
            ),
            base_angular_velocity_q10,
            joint_pos_biased_q10_hardware - home,
            joint_velocity_q10_hardware,
            previous_effective_selected_raw_hardware,
        ),
        dim=-1,
    )
    return _require_float32_batch(
        result,
        NATIVE124_OBSERVATION_DIM,
        "causal native124 actor observation",
        batch_size=batch_size,
        device=device,
    )


def merge_causal_history_after_step(
    *,
    done_mask: torch.Tensor,
    pre_step_robot_torso_wxyz: torch.Tensor,
    reset_virtual_robot_torso_q9_wxyz: torch.Tensor,
    effective_selected_raw_action_hardware: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join normal and autoreset rows without replacing the sampled action."""

    if not isinstance(done_mask, torch.Tensor) or done_mask.ndim != 1 or done_mask.dtype != torch.bool:
        raise ValueError("done mask must be bool [batch]")
    batch_size = done_mask.shape[0]
    device = done_mask.device
    _require_float32_batch(
        pre_step_robot_torso_wxyz,
        4,
        "pre-step robot torso",
        batch_size=batch_size,
        device=device,
    )
    _require_float32_batch(
        reset_virtual_robot_torso_q9_wxyz,
        4,
        "reset virtual robot torso q9",
        batch_size=batch_size,
        device=device,
    )
    _require_float32_batch(
        effective_selected_raw_action_hardware,
        TARGET_DOF,
        "effective selected raw action",
        batch_size=batch_size,
        device=device,
    )
    buffered_torso = torch.where(
        done_mask[:, None],
        reset_virtual_robot_torso_q9_wxyz,
        pre_step_robot_torso_wxyz,
    )
    previous_action = torch.where(
        done_mask[:, None],
        torch.zeros_like(effective_selected_raw_action_hardware),
        effective_selected_raw_action_hardware,
    )
    return buffered_torso, previous_action


def _native124_actor_placeholder(env: Any) -> torch.Tensor:
    """Reserve the exact actor space; the wrapper replaces this tensor."""

    return torch.zeros(
        (env.num_envs, NATIVE124_OBSERVATION_DIM),
        dtype=torch.float32,
        device=env.device,
    )


if _MJLAB_IMPORT_ERROR is None:

    @dataclass(kw_only=True)
    class Native124SelectedV2CausalMotionCommandCfg(CausalHistoryMotionCommandCfg):
        """Causal SONIC command with checkpoint-21204 qd semantics."""

        fixed_anchor_index: int | None = None

        def build(self, env: Any) -> "Native124SelectedV2CausalMotionCommand":
            return Native124SelectedV2CausalMotionCommand(self, env)

    class Native124SelectedV2CausalMotionCommand(CausalHistoryMotionCommand):
        cfg: Native124SelectedV2CausalMotionCommandCfg

        def __init__(self, cfg: Native124SelectedV2CausalMotionCommandCfg, env: Any) -> None:
            super().__init__(cfg, env)
            fixed = cfg.fixed_anchor_index
            self._fixed_anchor_index = (
                None
                if fixed is None
                else resolve_causal_reset_anchor_index(
                    fixed,
                    causal_min_anchor=self._causal_min_anchor,
                    causal_max_anchor=self._causal_max_anchor,
                )
            )

        @property
        def reset_anchor_index(self) -> int:
            """q9 selected before reset writes the phase-specific q10 robot state."""

            return self._causal_min_anchor if self._fixed_anchor_index is None else self._fixed_anchor_index

        def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
            if self._fixed_anchor_index is None:
                super()._uniform_sampling(env_ids)
                return
            self.time_steps[env_ids] = self._fixed_anchor_index
            span = self._causal_max_anchor - self._causal_min_anchor + 1
            fixed_bin = (self._fixed_anchor_index - self._causal_min_anchor) / max(span, 1)
            self.metrics["sampling_entropy"][:] = 0.0
            self.metrics["sampling_top1_prob"][:] = 1.0
            self.metrics["sampling_top1_bin"][:] = fixed_bin

        @property
        def joint_vel(self) -> torch.Tensor:
            _, _, qd = causal_forward_difference_at_indices(
                self.motion.joint_pos,
                self.time_steps,
            )
            return qd

else:

    class Native124SelectedV2CausalMotionCommandCfg:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR

    class Native124SelectedV2CausalMotionCommand:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR


def make_native124_selected_v2_causal_adaptation_env_cfg(
    *,
    motion_file: str | Path,
    num_envs: int,
    play: bool = False,
    fixed_anchor_index: int | None = None,
) -> Any:
    """Build the single-DadDance causal recovery task for ankle adaptation."""

    if _MJLAB_IMPORT_ERROR is not None:
        raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    reset_anchor_index = resolve_causal_reset_anchor_index(fixed_anchor_index)
    motion_path = validate_dad_dance_motion_file(motion_file)
    cfg = make_causal_history_recovery_env_cfg(
        motion_file=str(motion_path),
        num_envs=num_envs,
        play=play,
    )

    base_command = cfg.commands["motion"]
    command_kwargs = {field.name: getattr(base_command, field.name) for field in fields(base_command)}
    command_kwargs["sampling_mode"] = "start"
    command_kwargs["fixed_anchor_index"] = fixed_anchor_index
    cfg.commands["motion"] = Native124SelectedV2CausalMotionCommandCfg(**command_kwargs)
    cfg.actions["joint_pos"] = Selected21204HardwareToSonicV2JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
    )

    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )

    if "actor" in cfg.observations:
        raise RuntimeError("causal SONIC base unexpectedly already exposes an actor group")
    critic = cfg.observations["critic"]
    actor = ObservationGroupCfg(
        terms={"native124_selected_v2_causal": ObservationTermCfg(func=_native124_actor_placeholder)},
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="error",
    )
    cfg.observations = {**cfg.observations, "actor": actor}
    if cfg.observations["critic"] is not critic:
        raise RuntimeError("causal privileged critic configuration was replaced")

    # Timeout must happen before the selected q9 could lose its q10 proof.
    safe_episode_steps = DAD_DANCE_FRAME_COUNT - reset_anchor_index - 1
    cfg.episode_length_s = min(
        float(cfg.episode_length_s),
        safe_episode_steps * CONTROL_DT_S,
    )
    return cfg


@dataclass(frozen=True)
class Native124CausalWrapperDiagnostics:
    q9_indices: torch.Tensor
    q10_proof_indices: torch.Tensor
    reset_virtual_torso_mask: torch.Tensor
    buffered_robot_torso_q9_wxyz: torch.Tensor
    previous_effective_selected_raw_hardware: torch.Tensor
    actor_observation: torch.Tensor


if _WRAPPER_IMPORT_ERROR is None:

    class Native124SelectedV2CausalAdaptationWrapper(RslRlVecEnvWrapper):
        """Replace only ``actor`` with causal native124 observations."""

        def __init__(self, env: Any, clip_actions: float | None = None) -> None:
            if clip_actions is not None:
                raise ValueError("causal selected-V2 wrapper requires clip_actions=None")
            self._buffered_robot_torso_q9_wxyz: torch.Tensor | None = None
            self._previous_effective_selected_raw_hardware: torch.Tensor | None = None
            self._reset_virtual_torso_mask: torch.Tensor | None = None
            self._last_q9_indices: torch.Tensor | None = None
            self._last_episode_lengths: torch.Tensor | None = None
            self._last_command_counters: torch.Tensor | None = None
            self._last_actor_observation: torch.Tensor | None = None
            super().__init__(env, clip_actions=None)
            self._validate_static_contract()
            self._refresh_after_full_reset()

        def _command(self) -> Native124SelectedV2CausalMotionCommand:
            command = self.unwrapped.command_manager.get_term("motion")
            if type(command) is not Native124SelectedV2CausalMotionCommand:
                raise TypeError("environment lacks exact causal selected-V2 command")
            return command

        def _action_term(
            self,
        ) -> Selected21204HardwareToSonicV2JointPositionAction:
            action = self.unwrapped.action_manager.get_term("joint_pos")
            if type(action) is not Selected21204HardwareToSonicV2JointPositionAction:
                raise TypeError("environment lacks exact selected-21204 V2 action term")
            return action

        def _validate_static_contract(self) -> None:
            if self.clip_actions is not None or self.num_actions != TARGET_DOF:
                raise ValueError("causal wrapper action boundary drift")
            if abs(float(self.unwrapped.step_dt) - CONTROL_DT_S) > 1.0e-12:
                raise ValueError("causal wrapper requires exact 50 Hz control")
            actor_cfg = self.cfg.observations.get("actor")
            if actor_cfg is None or actor_cfg.enable_corruption is not False:
                raise ValueError("native124 actor observation corruption must be disabled")
            command = self._command()
            action = self._action_term()
            del action
            if command.cfg.sampling_mode != "start":
                raise ValueError("causal adaptation must start-sample DadDance")
            if command.motion.time_step_total != DAD_DANCE_FRAME_COUNT:
                raise ValueError("causal adaptation DadDance frame-count drift")
            if command._causal_max_anchor != DAD_DANCE_FRAME_COUNT - 2:  # noqa: SLF001
                raise ValueError("causal command does not reserve q10 proof")
            causal_episode_last_q9(
                reset_anchor_index=command.reset_anchor_index,
                episode_steps=self.max_episode_length,
                causal_max_anchor=command._causal_max_anchor,  # noqa: SLF001
            )
            robot = self.unwrapped.scene["robot"]
            if tuple(robot.joint_names) != tuple(HARDWARE_23_JOINT_NAMES):
                raise ValueError("robot joints are not compact hardware-23 order")
            body_names = tuple(command.cfg.body_names)
            if body_names.count(TORSO_BODY_NAME) != 1:
                raise ValueError("motion command must contain exactly one torso_link")
            if command.cfg.anchor_body_name == TORSO_BODY_NAME:
                raise ValueError("SONIC causal pelvis anchor unexpectedly became torso")

        def _reference_torso(
            self,
            q9_indices: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            command = self._command()
            q9_indices = _require_indices(
                q9_indices,
                batch_size=self.num_envs,
                frame_count=command.motion.time_step_total,
                device=self.device,
            )
            body_names = tuple(command.cfg.body_names)
            torso_index = body_names.index(TORSO_BODY_NAME)
            reference = command.motion.body_quat_w
            q9 = reference[q9_indices, torso_index]
            q10 = reference[q9_indices + 1, torso_index]
            _require_float32_batch(
                q9,
                4,
                "reference torso q9",
                batch_size=self.num_envs,
                device=self.device,
            )
            _require_float32_batch(
                q10,
                4,
                "reference torso q10",
                batch_size=self.num_envs,
                device=self.device,
            )
            return q9, q10

        def _current_robot_torso(self) -> torch.Tensor:
            command = self._command()
            torso_index = tuple(command.cfg.body_names).index(TORSO_BODY_NAME)
            value = command.robot_body_quat_w[:, torso_index].clone()
            return _require_float32_batch(
                value,
                4,
                "current robot torso",
                batch_size=self.num_envs,
                device=self.device,
            )

        def _current_indices(self) -> torch.Tensor:
            command = self._command()
            return _require_indices(
                command.time_steps,
                batch_size=self.num_envs,
                frame_count=command.motion.time_step_total,
                device=self.device,
            ).clone()

        def _synchronize_reset_buffers(self) -> None:
            q9_indices = self._current_indices()
            reference_q9, reference_q10 = self._reference_torso(q9_indices)
            self._buffered_robot_torso_q9_wxyz = virtual_q9_torso_quaternion(
                self._current_robot_torso(),
                reference_q9,
                reference_q10,
            )
            self._previous_effective_selected_raw_hardware = torch.zeros(
                (self.num_envs, TARGET_DOF),
                dtype=torch.float32,
                device=self.device,
            )
            self._reset_virtual_torso_mask = torch.ones(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            self._last_q9_indices = q9_indices
            self._last_episode_lengths = self.episode_length_buf.clone()
            self._last_command_counters = self._command().command_counter.clone()
            self._last_actor_observation = None

        def _refresh_after_full_reset(self) -> None:
            command = self._command()
            command.refresh_relative_body_targets_after_reset()
            all_env_ids = torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            )
            self.unwrapped.observation_manager.reset(all_env_ids)
            self.unwrapped.obs_buf = self.unwrapped.observation_manager.compute(update_history=True)
            self._synchronize_reset_buffers()

        def _actor_observation(self) -> torch.Tensor:
            if (
                self._buffered_robot_torso_q9_wxyz is None
                or self._previous_effective_selected_raw_hardware is None
                or self._last_q9_indices is None
            ):
                raise RuntimeError("causal wrapper buffers are not initialized")
            q9_indices = self._current_indices()
            if not torch.equal(q9_indices, self._last_q9_indices):
                raise RuntimeError("command changed outside causal wrapper step/reset")
            command = self._command()
            reference_q9, reference_q10, _ = causal_forward_difference_at_indices(
                command.motion.joint_pos,
                q9_indices,
            )
            reference_torso_q9, _ = self._reference_torso(q9_indices)
            robot = self.unwrapped.scene["robot"]
            biased_q = getattr(robot.data, "joint_pos_biased", None)
            if not isinstance(biased_q, torch.Tensor):
                raise ValueError("causal actor requires robot.data.joint_pos_biased")
            try:
                gyro = self.unwrapped.scene["robot/imu_ang_vel"].data
            except (KeyError, TypeError) as error:
                raise ValueError("causal actor requires robot/imu_ang_vel") from error
            actor = build_native124_selected_v2_causal_actor(
                reference_q9_hardware=reference_q9,
                reference_q10_hardware=reference_q10,
                robot_torso_q9_wxyz=self._buffered_robot_torso_q9_wxyz,
                reference_torso_q9_wxyz=reference_torso_q9,
                base_angular_velocity_q10=gyro,
                joint_pos_biased_q10_hardware=biased_q,
                joint_velocity_q10_hardware=robot.data.joint_vel,
                previous_effective_selected_raw_hardware=(self._previous_effective_selected_raw_hardware),
            )
            self._last_actor_observation = actor.detach().clone()
            return actor

        def _replace_actor(self, observations: TensorDict) -> TensorDict:
            if "actor" not in observations or "critic" not in observations:
                raise KeyError("wrapped observation lacks actor or privileged critic")
            critic_before = observations["critic"]
            observations.set("actor", self._actor_observation())
            if not torch.equal(observations["critic"], critic_before):
                raise RuntimeError("causal wrapper mutated privileged critic")
            return observations

        @property
        def diagnostics(self) -> Native124CausalWrapperDiagnostics:
            if (
                self._last_q9_indices is None
                or self._reset_virtual_torso_mask is None
                or self._buffered_robot_torso_q9_wxyz is None
                or self._previous_effective_selected_raw_hardware is None
                or self._last_actor_observation is None
            ):
                raise RuntimeError("causal diagnostics require an observation")
            return Native124CausalWrapperDiagnostics(
                q9_indices=self._last_q9_indices.detach().clone(),
                q10_proof_indices=self._last_q9_indices.detach().clone() + 1,
                reset_virtual_torso_mask=(self._reset_virtual_torso_mask.detach().clone()),
                buffered_robot_torso_q9_wxyz=(self._buffered_robot_torso_q9_wxyz.detach().clone()),
                previous_effective_selected_raw_hardware=(
                    self._previous_effective_selected_raw_hardware.detach().clone()
                ),
                actor_observation=self._last_actor_observation.detach().clone(),
            )

        def get_observations(self) -> TensorDict:
            return self._replace_actor(super().get_observations())

        def reset(self) -> tuple[TensorDict, dict]:
            _, extras = super().reset()
            self._refresh_after_full_reset()
            observations = TensorDict(
                self.unwrapped.obs_buf,
                batch_size=[self.num_envs],
            )
            return self._replace_actor(observations), extras

        def step(
            self,
            actions: torch.Tensor,
        ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
            _require_float32_batch(
                actions,
                TARGET_DOF,
                "selected-21204 sampled action",
                batch_size=self.num_envs,
                device=self.device,
            )
            if (
                self._last_q9_indices is None
                or self._last_episode_lengths is None
                or self._last_command_counters is None
            ):
                raise RuntimeError("causal wrapper buffers are not initialized")
            before_q9 = self._current_indices()
            if not torch.equal(before_q9, self._last_q9_indices):
                raise RuntimeError("pre-step q9 discontinuity")
            before_episode = self.episode_length_buf.clone()
            before_counter = self._command().command_counter.clone()
            if not torch.equal(before_episode, self._last_episode_lengths) or not torch.equal(
                before_counter, self._last_command_counters
            ):
                raise RuntimeError("episode or command changed outside wrapper")
            pre_step_torso = self._current_robot_torso()

            observations, rewards, dones, extras = super().step(actions)
            if dones.shape != (self.num_envs,) or dones.dtype != torch.long:
                raise ValueError("wrapped dones must be int64 [batch]")
            done_mask = dones.to(dtype=torch.bool)
            after_q9 = self._current_indices()
            after_episode = self.episode_length_buf.clone()
            after_counter = self._command().command_counter.clone()
            advancing = ~done_mask
            if torch.any(after_q9[advancing] != before_q9[advancing] + 1):
                raise RuntimeError("non-reset q9 discontinuity or command resample")
            if torch.any(after_episode[advancing] != before_episode[advancing] + 1):
                raise RuntimeError("non-reset episode counter discontinuity")
            if torch.any(after_counter[advancing] != before_counter[advancing]):
                raise RuntimeError("non-reset command counter changed")
            if torch.any(after_q9[done_mask] != self._command().reset_anchor_index):
                raise RuntimeError("reset q9 did not return to deterministic reset anchor")

            reference_q9, reference_q10 = self._reference_torso(after_q9)
            virtual_torso = virtual_q9_torso_quaternion(
                self._current_robot_torso(),
                reference_q9,
                reference_q10,
            )
            effective_previous = self._action_term().effective_selected_raw_action_hardware
            _require_float32_batch(
                effective_previous,
                TARGET_DOF,
                "effective selected previous action",
                batch_size=self.num_envs,
                device=self.device,
            )
            (
                self._buffered_robot_torso_q9_wxyz,
                self._previous_effective_selected_raw_hardware,
            ) = merge_causal_history_after_step(
                done_mask=done_mask,
                pre_step_robot_torso_wxyz=pre_step_torso,
                reset_virtual_robot_torso_q9_wxyz=virtual_torso,
                effective_selected_raw_action_hardware=effective_previous,
            )
            self._reset_virtual_torso_mask = done_mask.clone()
            self._last_q9_indices = after_q9
            self._last_episode_lengths = after_episode
            self._last_command_counters = after_counter
            self._last_actor_observation = None
            return (
                self._replace_actor(observations),
                rewards,
                dones,
                extras,
            )

else:

    class Native124SelectedV2CausalAdaptationWrapper:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab RSL wrapper is required") from _WRAPPER_IMPORT_ERROR


def prime_native124_selected_v2_causal_adaptation_environment(
    wrapped_env: Any,
    *,
    max_reset_attempts: int = 32,
) -> dict[str, Any]:
    """Prime SONIC reset targets, then synchronize the native124 reset row."""

    if type(wrapped_env) is not Native124SelectedV2CausalAdaptationWrapper:
        raise TypeError("prime requires exact causal selected-V2 wrapper")
    report = prime_sonic_true23_training_environment(
        wrapped_env,
        max_reset_attempts=max_reset_attempts,
    )
    wrapped_env._synchronize_reset_buffers()  # noqa: SLF001
    observations = wrapped_env.get_observations()
    actor = observations["actor"]
    critic = observations["critic"]
    policy = observations["policy"]
    _require_float32_batch(
        actor,
        NATIVE124_OBSERVATION_DIM,
        "primed causal actor",
        batch_size=wrapped_env.num_envs,
        device=wrapped_env.device,
    )
    _require_float32_batch(
        critic,
        SONIC_TRUE23_CRITIC_DIM,
        "primed privileged critic",
        batch_size=wrapped_env.num_envs,
        device=wrapped_env.device,
    )
    _require_float32_batch(
        policy,
        SONIC_TRUE23_POLICY_DIM,
        "primed SONIC H10 policy history",
        batch_size=wrapped_env.num_envs,
        device=wrapped_env.device,
    )
    return {
        **report,
        "actor_observation_dim": NATIVE124_OBSERVATION_DIM,
        "critic_observation_dim": SONIC_TRUE23_CRITIC_DIM,
        "policy_history_dim": SONIC_TRUE23_POLICY_DIM,
        "actor_reset_previous_action": "zero",
        "action_substitution": False,
        "wrapper_action_clip": None,
    }


def causal_adaptation_contract() -> dict[str, object]:
    return {
        "schema": "g1_true23_native124_selected_v2_causal_adaptation_v1",
        "motion": {
            "relative_path": DAD_DANCE_RELATIVE_PATH.as_posix(),
            "sha256": DAD_DANCE_SHA256,
            "frame_count": DAD_DANCE_FRAME_COUNT,
            "sampling_mode": "start",
        },
        "actor": {
            "dimension": NATIVE124_OBSERVATION_DIM,
            "order": (
                "reference_q9_hardware_23",
                "forward_difference_q9_q10_hardware_23",
                "relative_torso_q9_rotation_6d",
                "measured_gyro_q10_3",
                "measured_biased_q10_minus_selected_home_23",
                "measured_joint_velocity_q10_23",
                "previous_effective_selected_raw_hardware_23",
            ),
            "corruption": False,
        },
        "critic": "unchanged_sonic_privileged_256",
        "sonic_policy_history": "unchanged_930",
        "action": "selected21204_raw_hardware_to_sonic_v2_once",
        "wrapper_action_clip": None,
        "autoreset": {
            "torso": "virtual_q9_from_current_q10_and_reference_q9_q10_yaw_alignment",
            "previous_action": "zero",
            "sampled_action_substitution": False,
        },
        "simulator_only": True,
        "hardware_authorized": False,
    }
