"""Exact causal-history true23 semantics for 50 Hz Pico teleoperation.

Released low-latency weights remain architecture initialization only.  This
profile changes the 240-value lower-body term from future samples to ten
oldest-to-anchor samples.  It therefore has a new semantic profile/hash and
must be retrained; shape equality never authorizes relabeling.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from typing import Any

import torch

from gear_sonic.envs.mjlab.sonic_true23 import (
    SonicTrue23MotionCommand,
    SonicTrue23MotionCommandCfg,
    _motion_command,
    _quat_inverse,
    _quat_multiply,
    _require_last_dim,
    _rotation_6d,
)
from gear_sonic.envs.mjlab.sonic_true23_low_latency_recovery import (
    make_low_latency_recovery_env_cfg,
)
from gear_sonic.utils.g1_23dof_contract import (
    CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES,
    REFERENCE_PROFILE_LOW_LATENCY,
)

CAUSAL_HISTORY_PROFILE = "true23_causal_step1_history_0p02s_v1"
CAUSAL_HISTORY_FRAME_COUNT = 10
CAUSAL_HISTORY_ANCHOR_INDEX = 9
CAUSAL_HISTORY_PROOF_INDEX = 10
CAUSAL_HISTORY_FRAME_OFFSETS_S = tuple(
    (index - CAUSAL_HISTORY_ANCHOR_INDEX) * 0.02
    for index in range(CAUSAL_HISTORY_FRAME_COUNT)
)
CAUSAL_HISTORY_ANCHOR_AGE_S = 0.02
CAUSAL_HISTORY_PROOF_OFFSET_S = 0.02


def causal_history_profile_contract() -> dict[str, Any]:
    body = {
        "schema": "g1_true23_causal_history_profile_v1",
        "profile": CAUSAL_HISTORY_PROFILE,
        "source_sample_rate_hz": 50,
        "source_sample_period_s": 0.02,
        "architecture_initialization_profile": REFERENCE_PROFILE_LOW_LATENCY,
        "encoder_input_dim": 267,
        "lower_body_term_dim": 240,
        "lower_body_term_name": "causal_history_lower_body",
        "lower_body_il29_indices_in_encoder_order": list(
            CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES
        ),
        "lower_body_order": "mujoco_hardware_left6_then_right6",
        "position_frame_count": CAUSAL_HISTORY_FRAME_COUNT,
        "position_order": "oldest_to_anchor",
        "position_offsets_from_anchor_s": list(CAUSAL_HISTORY_FRAME_OFFSETS_S),
        "velocity_order": "oldest_to_anchor",
        "velocity_definition": "forward_difference_q_i_to_q_i_plus_1_over_0p02s",
        "anchor_frame": "q9",
        "proof_frame": "q10",
        "proof_frame_offset_from_anchor_s": CAUSAL_HISTORY_PROOF_OFFSET_S,
        "anchor_age_at_emission_s": CAUSAL_HISTORY_ANCHOR_AGE_S,
        "reference_channels_anchor": "q9",
        "reference_channels": [
            "lower_body_positions_and_velocities",
            "vr_3point_local_target",
            "vr_3point_local_orientation_target",
            "reference_pelvis_orientation",
            "buffered_robot_anchor_orientation",
            "reward_and_tracking_target",
        ],
        "control_and_proprioception_frame": "q10_current",
        "future_samples_relative_to_emission": False,
        "repeated_or_synthetic_future_frames": False,
        "released_profile_relabel_permitted": False,
        "retraining_required": True,
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {**body, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def causal_history_lower_body(
    env: Any,
    command_name: str = "motion",
) -> torch.Tensor:
    """Positions q0..q9 plus proven forward dq using q1..q10."""

    command = _motion_command(env, command_name)
    anchor = command.time_steps
    offsets = torch.arange(
        -CAUSAL_HISTORY_ANCHOR_INDEX,
        1,
        dtype=torch.long,
        device=anchor.device,
    )
    indexes = anchor[:, None] + offsets[None, :]
    proof_indexes = indexes + 1
    total = int(command.motion.time_step_total)
    if torch.any(indexes < 0):
        raise ValueError("causal history lacks q0 oldest frame")
    if torch.any(proof_indexes[:, -1] >= total):
        raise ValueError("causal history lacks q10 forward-difference proof frame")
    positions = command.motion.joint_pos[indexes, :12]
    next_positions = command.motion.joint_pos[proof_indexes, :12]
    velocities = (next_positions - positions) / 0.02
    result = torch.cat(
        (positions.reshape(env.num_envs, -1), velocities.reshape(env.num_envs, -1)),
        dim=-1,
    )
    _require_last_dim(result, 240, "causal-history lower-body command")
    return result


def causal_motion_anchor_ori_b(
    env: Any,
    command_name: str = "motion",
) -> torch.Tensor:
    """q9 reference pelvis relative to robot pelvis buffered at q9."""

    command = _motion_command(env, command_name)
    buffered = command.causal_robot_anchor_quat_w
    relative = _quat_multiply(
        _quat_inverse(buffered),
        command.anchor_quat_w,
    )
    return _rotation_6d(relative)


try:
    from mjlab.utils.lab_api.math import (
        quat_apply,
        quat_from_euler_xyz,
        quat_inv,
        quat_mul,
        sample_uniform,
        yaw_quat,
    )
except ImportError as exc:  # pragma: no cover - import-only contract tests.
    _CAUSAL_MJLAB_IMPORT_ERROR: ImportError | None = exc
else:
    _CAUSAL_MJLAB_IMPORT_ERROR = None


if _CAUSAL_MJLAB_IMPORT_ERROR is None:

    @dataclass(kw_only=True)
    class CausalHistoryMotionCommandCfg(SonicTrue23MotionCommandCfg):
        """Motion command whose target is q9 while robot/proprio is q10."""

        def build(self, env: Any) -> "CausalHistoryMotionCommand":
            return CausalHistoryMotionCommand(self, env)

    class CausalHistoryMotionCommand(SonicTrue23MotionCommand):
        cfg: CausalHistoryMotionCommandCfg

        def __init__(self, cfg: CausalHistoryMotionCommandCfg, env: Any) -> None:
            super().__init__(cfg, env)
            self._causal_min_anchor = CAUSAL_HISTORY_ANCHOR_INDEX
            self._causal_max_anchor = self.motion.time_step_total - 2
            if self._causal_max_anchor < self._causal_min_anchor:
                raise ValueError("causal history motion lacks q0..q10 frames")
            # Replace future-horizon bound inherited from released step1.
            self._sonic_max_start = self._causal_max_anchor
            # Manager construction probes observation shapes before first reset.
            # Seed a valid real q0..q10 window; reset later resamples normally.
            self.time_steps.fill_(self._causal_min_anchor)
            self._causal_robot_anchor_pos_w = torch.zeros(
                (self.num_envs, 3), dtype=torch.float32, device=self.device
            )
            self._causal_robot_anchor_quat_w = torch.zeros(
                (self.num_envs, 4), dtype=torch.float32, device=self.device
            )
            self._causal_robot_anchor_quat_w[:, 0] = 1.0
            self._causal_last_current_anchor_pos_w = self._causal_robot_anchor_pos_w.clone()
            self._causal_last_current_anchor_quat_w = self._causal_robot_anchor_quat_w.clone()
            self._causal_resampled = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._force_causal_start = False

        @property
        def causal_robot_anchor_pos_w(self) -> torch.Tensor:
            return self._causal_robot_anchor_pos_w

        @property
        def causal_robot_anchor_quat_w(self) -> torch.Tensor:
            return self._causal_robot_anchor_quat_w

        def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
            if self._force_causal_start:
                self.time_steps[env_ids] = self._causal_min_anchor
            else:
                self.time_steps[env_ids] = torch.randint(
                    self._causal_min_anchor,
                    self._causal_max_anchor + 1,
                    (len(env_ids),),
                    device=self.device,
                )
            span = self._causal_max_anchor - self._causal_min_anchor + 1
            self.metrics["sampling_entropy"][:] = 1.0
            self.metrics["sampling_top1_prob"][:] = 1.0 / max(span, 1)
            self.metrics["sampling_top1_bin"][:] = 0.5

        def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
            super()._adaptive_sampling(env_ids)
            self.time_steps[env_ids] = self.time_steps[env_ids].clamp(
                min=self._causal_min_anchor,
                max=self._causal_max_anchor,
            )

        def _resample_command(self, env_ids: torch.Tensor) -> None:
            sampling_mode = self.cfg.sampling_mode
            if sampling_mode == "start":
                self._force_causal_start = True
                self.cfg.sampling_mode = "uniform"
            try:
                if self.cfg.sampling_mode == "uniform":
                    self._uniform_sampling(env_ids)
                else:
                    if self.cfg.sampling_mode != "adaptive":
                        raise ValueError("unsupported causal sampling mode")
                    self._adaptive_sampling(env_ids)
            finally:
                self.cfg.sampling_mode = sampling_mode
                self._force_causal_start = False

            proof = self.time_steps + 1
            root_pos = self.motion.body_pos_w[proof, 0].clone()
            root_pos += self._env.scene.env_origins
            root_ori = self.motion.body_quat_w[proof, 0].clone()
            root_lin_vel = self.motion.body_lin_vel_w[proof, 0].clone()
            root_ang_vel = self.motion.body_ang_vel_w[proof, 0].clone()

            pose_ranges = torch.tensor(
                [
                    self.cfg.pose_range.get(key, (0.0, 0.0))
                    for key in ("x", "y", "z", "roll", "pitch", "yaw")
                ],
                device=self.device,
            )
            pose_random = sample_uniform(
                pose_ranges[:, 0],
                pose_ranges[:, 1],
                (len(env_ids), 6),
                device=self.device,
            )
            root_pos[env_ids] += pose_random[:, :3]
            root_ori[env_ids] = quat_mul(
                quat_from_euler_xyz(
                    pose_random[:, 3], pose_random[:, 4], pose_random[:, 5]
                ),
                root_ori[env_ids],
            )
            velocity_ranges = torch.tensor(
                [
                    self.cfg.velocity_range.get(key, (0.0, 0.0))
                    for key in ("x", "y", "z", "roll", "pitch", "yaw")
                ],
                device=self.device,
            )
            velocity_random = sample_uniform(
                velocity_ranges[:, 0],
                velocity_ranges[:, 1],
                (len(env_ids), 6),
                device=self.device,
            )
            root_lin_vel[env_ids] += velocity_random[:, :3]
            root_ang_vel[env_ids] += velocity_random[:, 3:]

            joint_pos = self.motion.joint_pos[proof].clone()
            joint_vel = self.motion.joint_vel[proof].clone()
            joint_pos += sample_uniform(
                lower=self.cfg.joint_position_range[0],
                upper=self.cfg.joint_position_range[1],
                size=joint_pos.shape,
                device=joint_pos.device,
            )
            limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos[env_ids] = torch.clip(
                joint_pos[env_ids], limits[..., 0], limits[..., 1]
            )
            self.robot.write_joint_state_to_sim(
                joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
            )
            root_state = torch.cat(
                (
                    root_pos[env_ids],
                    root_ori[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ),
                dim=-1,
            )
            self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
            self.robot.clear_state(env_ids=env_ids)
            self._causal_resampled[env_ids] = True

        def _virtual_anchor_at_q9(
            self,
            env_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            anchor = self.time_steps[env_ids]
            proof = anchor + 1
            reference_anchor_pos = self.motion.body_pos_w[
                anchor, self.motion_anchor_body_index
            ] + self._env.scene.env_origins[env_ids]
            reference_anchor_quat = self.motion.body_quat_w[
                anchor, self.motion_anchor_body_index
            ]
            reference_proof_pos = self.motion.body_pos_w[
                proof, self.motion_anchor_body_index
            ] + self._env.scene.env_origins[env_ids]
            reference_proof_quat = self.motion.body_quat_w[
                proof, self.motion_anchor_body_index
            ]
            current_pos = self.robot_anchor_pos_w[env_ids]
            current_quat = self.robot_anchor_quat_w[env_ids]
            delta_quat = yaw_quat(quat_mul(current_quat, quat_inv(reference_proof_quat)))
            buffered_pos = current_pos + quat_apply(
                delta_quat,
                reference_anchor_pos - reference_proof_pos,
            )
            buffered_quat = quat_mul(delta_quat, reference_anchor_quat)
            return buffered_pos, buffered_quat

        def _refresh_targets_from_causal_anchor(self) -> None:
            body_count = len(self.cfg.body_names)
            reference_pos = self.anchor_pos_w[:, None, :].repeat(1, body_count, 1)
            reference_quat = self.anchor_quat_w[:, None, :].repeat(1, body_count, 1)
            robot_pos = self._causal_robot_anchor_pos_w[:, None, :].repeat(
                1, body_count, 1
            )
            robot_quat = self._causal_robot_anchor_quat_w[:, None, :].repeat(
                1, body_count, 1
            )
            delta_pos = robot_pos.clone()
            delta_pos[..., 2] = reference_pos[..., 2]
            delta_quat = yaw_quat(quat_mul(robot_quat, quat_inv(reference_quat)))
            self.body_quat_relative_w.copy_(quat_mul(delta_quat, self.body_quat_w))
            self.body_pos_relative_w.copy_(
                delta_pos + quat_apply(delta_quat, self.body_pos_w - reference_pos)
            )

        def refresh_relative_body_targets_after_reset(self) -> None:
            all_env_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
            buffered_pos, buffered_quat = self._virtual_anchor_at_q9(all_env_ids)
            self._causal_robot_anchor_pos_w.copy_(buffered_pos)
            self._causal_robot_anchor_quat_w.copy_(buffered_quat)
            self._causal_last_current_anchor_pos_w.copy_(self.robot_anchor_pos_w)
            self._causal_last_current_anchor_quat_w.copy_(self.robot_anchor_quat_w)
            self._refresh_targets_from_causal_anchor()
            self._causal_resampled.zero_()

        def _update_command(self) -> None:
            current_pos = self.robot_anchor_pos_w.clone()
            current_quat = self.robot_anchor_quat_w.clone()
            resampled = self._causal_resampled.clone()
            advancing = ~resampled
            self.time_steps[advancing] += 1
            exhausted = advancing & (self.time_steps > self._causal_max_anchor)
            exhausted_ids = exhausted.nonzero(as_tuple=False).flatten()
            if len(exhausted_ids) > 0:
                self._resample_command(exhausted_ids)
                resampled[exhausted_ids] = True
                advancing[exhausted_ids] = False

            if torch.any(advancing):
                self._causal_robot_anchor_pos_w[advancing] = (
                    self._causal_last_current_anchor_pos_w[advancing]
                )
                self._causal_robot_anchor_quat_w[advancing] = (
                    self._causal_last_current_anchor_quat_w[advancing]
                )
            resampled_ids = resampled.nonzero(as_tuple=False).flatten()
            if len(resampled_ids) > 0:
                buffered_pos, buffered_quat = self._virtual_anchor_at_q9(resampled_ids)
                self._causal_robot_anchor_pos_w[resampled_ids] = buffered_pos
                self._causal_robot_anchor_quat_w[resampled_ids] = buffered_quat

            self._refresh_targets_from_causal_anchor()
            self._causal_last_current_anchor_pos_w.copy_(current_pos)
            self._causal_last_current_anchor_quat_w.copy_(current_quat)
            self._causal_resampled.zero_()
            if self.cfg.sampling_mode == "adaptive":
                self.bin_failed_count = (
                    self.cfg.adaptive_alpha * self._current_bin_failed
                    + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
                )
                self._current_bin_failed.zero_()

else:

    class CausalHistoryMotionCommandCfg:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("causal task requires MJLab") from _CAUSAL_MJLAB_IMPORT_ERROR

    class CausalHistoryMotionCommand:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("causal task requires MJLab") from _CAUSAL_MJLAB_IMPORT_ERROR


def make_causal_history_recovery_env_cfg(
    *,
    motion_file: str,
    num_envs: int,
    play: bool = False,
) -> Any:
    """Build recovery task with a new causal semantic profile."""

    cfg = make_low_latency_recovery_env_cfg(
        motion_file=motion_file,
        num_envs=num_envs,
        play=play,
    )
    base_command = cfg.commands["motion"]
    command_kwargs = {
        field.name: getattr(base_command, field.name) for field in fields(base_command)
    }
    cfg.commands["motion"] = CausalHistoryMotionCommandCfg(**command_kwargs)
    tokenizer_terms = cfg.observations["tokenizer"].terms
    lower_body = tokenizer_terms["command_multi_future_lower_body"]
    lower_body.func = causal_history_lower_body
    lower_body.params = {"command_name": "motion"}
    cfg.observations["tokenizer"].terms = {
        "encoder_index": tokenizer_terms["encoder_index"],
        "causal_history_lower_body": lower_body,
        "vr_3point_local_target": tokenizer_terms["vr_3point_local_target"],
        "vr_3point_local_orn_target": tokenizer_terms[
            "vr_3point_local_orn_target"
        ],
        "motion_anchor_ori_b": tokenizer_terms["motion_anchor_ori_b"],
    }
    anchor_orientation = cfg.observations["tokenizer"].terms["motion_anchor_ori_b"]
    anchor_orientation.func = causal_motion_anchor_ori_b
    anchor_orientation.params = {"command_name": "motion"}
    return cfg
