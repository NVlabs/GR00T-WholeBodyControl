"""Span-safe balanced multi-motion command for the stock native124 G1 task."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any, Literal

import torch

from gear_sonic.utils.g1_23dof_incremental_corpus import (
    EPISODE_FRAMES,
    CorpusCatalog,
    load_catalog,
)

OBSERVATION_DIM = 124
ACTION_DIM = 23

try:
    from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - runtime-only dependency
    _IMPORT_ERROR: Exception | None = error
    MotionCommand = object  # type: ignore[assignment,misc]
    MotionCommandCfg = object  # type: ignore[assignment,misc]
else:
    _IMPORT_ERROR = None


def weighted_span_indices(weights: torch.Tensor, count: int) -> torch.Tensor:
    if weights.ndim != 1 or count < 0 or weights.numel() == 0:
        raise ValueError("weights must be a non-empty vector and count non-negative")
    if not torch.isfinite(weights).all() or bool((weights < 0).any()) or float(weights.sum()) <= 0.0:
        raise ValueError("span weights must be finite, non-negative, and have positive sum")
    return torch.multinomial(weights / weights.sum(), count, replacement=True)


def lead_reference_steps(
    time_steps: torch.Tensor, window_stops: torch.Tensor, lead_frames: int
) -> torch.Tensor:
    """Advance actor joint commands without crossing assigned clip windows."""
    if lead_frames < 0 or time_steps.shape != window_stops.shape:
        raise ValueError("lead_frames must be non-negative and step tensors shape-matched")
    return torch.minimum(time_steps + lead_frames, window_stops - 1)


def deterministic_window_offsets(
    env_ids: torch.Tensor, sample_counts: torch.Tensor, widths: torch.Tensor, seed: int
) -> torch.Tensor:
    """Return reset-order-independent phase offsets for paired policy gates."""
    if env_ids.shape != sample_counts.shape or env_ids.shape != widths.shape:
        raise ValueError("deterministic offset tensors must be shape-matched")
    if bool((widths <= 0).any()):
        raise ValueError("deterministic offset widths must be positive")
    keys = (
        (env_ids.long() + 1) * 1_103_515_245
        + (sample_counts.long() + 1) * 12_345
        + int(seed) * 2_654_435_761
    )
    return torch.remainder(keys, widths)


def blend_reference_actions(
    policy_actions: torch.Tensor,
    joint_reference: torch.Tensor,
    action_scale: torch.Tensor | float,
    action_offset: torch.Tensor | float,
    blend: float,
) -> torch.Tensor:
    """Blend learned raw actions toward raw PD targets for reference joints."""
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("reference action blend must be within [0, 1]")
    reference_actions = (joint_reference - action_offset) / action_scale
    return torch.lerp(policy_actions, reference_actions, blend)


def blend_reference_leg_actions(
    policy_actions: torch.Tensor,
    joint_reference: torch.Tensor,
    action_scale: torch.Tensor | float,
    action_offset: torch.Tensor | float,
    blend: float,
) -> torch.Tensor:
    """Apply reference blend only to twelve leg actions."""
    result = policy_actions.clone()
    scale = action_scale[:, :12] if isinstance(action_scale, torch.Tensor) else action_scale
    offset = action_offset[:, :12] if isinstance(action_offset, torch.Tensor) else action_offset
    result[:, :12] = blend_reference_actions(
        policy_actions[:, :12], joint_reference[:, :12], scale, offset, blend
    )
    return result


if _IMPORT_ERROR is None:

    @dataclass(kw_only=True)
    class Native124MultiMotionCommandCfg(MotionCommandCfg):
        sidecar_path: str
        episode_frames: int = EPISODE_FRAMES
        only_clip: str | None = None
        clip_by_env: bool = False
        command_lead_frames: int = 0
        deterministic_start_seed: int | None = None
        sampling_mode: Literal["uniform"] = "uniform"

        def build(self, env: Any) -> "Native124MultiMotionCommand":
            return Native124MultiMotionCommand(self, env)


    class Native124MultiMotionCommand(MotionCommand):
        cfg: Native124MultiMotionCommandCfg

        def __init__(self, cfg: Native124MultiMotionCommandCfg, env: Any) -> None:
            self.catalog = load_catalog(Path(cfg.sidecar_path))
            if Path(cfg.motion_file).expanduser().resolve() != self.catalog.corpus_path:
                raise ValueError("motion_file differs from sidecar corpus")
            if cfg.episode_frames != self.catalog.episode_frames:
                raise ValueError("episode length differs from corpus")
            if cfg.sampling_mode != "uniform":
                raise ValueError("multi-motion command requires uniform sampling hook")
            if not math.isclose(float(env.step_dt), 1.0 / self.catalog.fps, abs_tol=1e-12):
                raise ValueError("multi-motion command requires 50 Hz control")
            if int(env.max_episode_length) != cfg.episode_frames:
                raise ValueError("environment and corpus episode lengths differ")
            if cfg.command_lead_frames < 0 or cfg.command_lead_frames >= cfg.episode_frames:
                raise ValueError("command lead must fit within episode window")
            super().__init__(cfg, env)
            weights = [span.weight for span in self.catalog.spans]
            if cfg.only_clip is not None:
                if cfg.only_clip not in {span.name for span in self.catalog.spans}:
                    raise ValueError(f"unknown requested clip: {cfg.only_clip}")
                weights = [span.weight if span.name == cfg.only_clip else 0.0 for span in self.catalog.spans]
            if cfg.only_clip is not None and cfg.clip_by_env:
                raise ValueError("only_clip and clip_by_env are mutually exclusive")
            self._span_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self.span_ids = torch.full_like(self.time_steps, -1)
            self.window_stops = torch.full_like(self.time_steps, -1)
            self._span_resampled = torch.zeros_like(self.time_steps, dtype=torch.bool)
            self._deterministic_sample_counts = torch.zeros_like(self.time_steps)

        @property
        def command(self) -> torch.Tensor:
            if bool((self.window_stops <= self.time_steps).any()):
                command_steps = self.time_steps
            else:
                command_steps = lead_reference_steps(
                    self.time_steps, self.window_stops, self.cfg.command_lead_frames
                )
            return torch.cat(
                (self.motion.joint_pos[command_steps], self.motion.joint_vel[command_steps]),
                dim=1,
            )

        def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
            if self.cfg.clip_by_env:
                selected = env_ids % len(self.catalog.spans)
            else:
                selected = weighted_span_indices(self._span_weights, len(env_ids))
            spans = [self.catalog.spans[index] for index in selected.cpu().tolist()]
            lows = torch.tensor([span.start for span in spans], dtype=torch.long, device=self.device)
            highs = torch.tensor(
                [span.stop - self.cfg.episode_frames for span in spans], dtype=torch.long, device=self.device
            )
            widths = highs - lows + 1
            if self.cfg.deterministic_start_seed is None:
                offsets = torch.floor(torch.rand(len(env_ids), device=self.device) * widths.float()).long()
            else:
                counts = self._deterministic_sample_counts[env_ids]
                offsets = deterministic_window_offsets(
                    env_ids, counts, widths, self.cfg.deterministic_start_seed
                )
                self._deterministic_sample_counts[env_ids] += 1
            starts = lows + offsets
            self.time_steps[env_ids] = starts
            self.window_stops[env_ids] = starts + self.cfg.episode_frames
            self.span_ids[env_ids] = selected
            self._span_resampled[env_ids] = self._env.episode_length_buf[env_ids] != 0
            probabilities = self._span_weights / self._span_weights.sum()
            entropy = -(probabilities * probabilities.log()).sum()
            entropy /= math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
            self.metrics["sampling_entropy"][env_ids] = entropy
            self.metrics["sampling_top1_prob"][env_ids] = probabilities.max()
            self.metrics["sampling_top1_bin"][env_ids] = selected.float() / max(len(probabilities) - 1, 1)

        def _update_command(self) -> None:
            skip_advance = (self._env.episode_length_buf == 0) | self._span_resampled
            expected = self.time_steps + (~skip_advance).long()
            if bool((expected >= self.window_stops).any()):
                raise RuntimeError("reference would cross its assigned clip window")
            self.time_steps[skip_advance] -= 1
            super()._update_command()
            if not torch.equal(self.time_steps, expected):
                raise RuntimeError("stock command changed span-safe reference timing")
            self._span_resampled.zero_()

else:

    class Native124MultiMotionCommandCfg:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab runtime unavailable") from _IMPORT_ERROR


def make_native124_multi_motion_env_cfg(
    sidecar_path: Path,
    *,
    ee_termination_threshold: float = 0.25,
    anchor_termination_threshold: float = 0.25,
    orientation_termination_threshold: float = 0.8,
    enable_pushes: bool = True,
    only_clip: str | None = None,
    clip_by_env: bool = False,
    ee_reward_weight: float = 0.0,
    ee_reward_std: float = 0.15,
    root_reward_weight: float = 0.5,
    root_reward_std: float = 0.3,
    command_lead_frames: int = 0,
    deterministic_start_seed: int | None = None,
) -> Any:
    if _IMPORT_ERROR is not None:
        raise RuntimeError("MJLab runtime unavailable") from _IMPORT_ERROR
    from src.tasks.tracking.config.g1_23dof.env_cfgs import unitree_g1_23dof_flat_tracking_env_cfg

    catalog: CorpusCatalog = load_catalog(sidecar_path)
    cfg = unitree_g1_23dof_flat_tracking_env_cfg(
        has_state_estimation=False,
        torque_penalty_weight=-2.0e-3,
        actuator_saturation_penalty_weight=-5.0,
        actuator_saturation_threshold_ratio=0.9,
        joint_limit_penalty_weight=-100.0,
        joint_tracking_weight=2.0,
        leg_joint_tracking_weight=4.0,
        joint_velocity_tracking_weight=0.5,
        action_rate_penalty_weight=-3.0e-2,
        body_position_tracking_std=0.15,
        body_orientation_tracking_std=0.25,
    )
    base = cfg.commands["motion"]
    command_kwargs = {field.name: getattr(base, field.name) for field in fields(base)}
    command_kwargs.update(
        motion_file=str(catalog.corpus_path),
        sidecar_path=str(sidecar_path.resolve()),
        episode_frames=catalog.episode_frames,
        only_clip=only_clip,
        clip_by_env=clip_by_env,
        command_lead_frames=command_lead_frames,
        deterministic_start_seed=deterministic_start_seed,
        sampling_mode="uniform",
    )
    cfg.commands["motion"] = Native124MultiMotionCommandCfg(**command_kwargs)
    reward_values = (ee_reward_weight, ee_reward_std, root_reward_weight, root_reward_std)
    if any(not math.isfinite(value) for value in reward_values) or min(ee_reward_std, root_reward_std) <= 0.0:
        raise ValueError("reward weights/stds must be finite and stds positive")
    cfg.rewards["motion_global_root_pos"].weight = root_reward_weight
    cfg.rewards["motion_global_root_pos"].params["std"] = root_reward_std
    if ee_reward_weight != 0.0:
        ee_reward = copy.deepcopy(cfg.rewards["motion_body_pos"])
        ee_reward.weight = ee_reward_weight
        ee_reward.params["std"] = ee_reward_std
        ee_reward.params["body_names"] = (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_roll_rubber_hand",
            "right_wrist_roll_rubber_hand",
        )
        cfg.rewards["motion_ee_body_pos"] = ee_reward
    thresholds = (
        ee_termination_threshold,
        anchor_termination_threshold,
        orientation_termination_threshold,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("termination thresholds must be finite and positive")
    cfg.terminations["ee_body_pos"].params["threshold"] = ee_termination_threshold
    cfg.terminations["anchor_pos"].params["threshold"] = anchor_termination_threshold
    cfg.terminations["anchor_ori"].params["threshold"] = orientation_termination_threshold
    if not enable_pushes:
        cfg.events.pop("push_robot", None)
    terms = tuple(cfg.observations["actor"].terms)
    expected = ("command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel", "actions")
    if terms != expected:
        raise RuntimeError("native124 observation contract drifted")
    return cfg
