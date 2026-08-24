"""Read-only MJLab transition join for iteration-21204 shadow inference.

This collector observes a SONIC transition.  It never supplies an action to
MJLab and never promotes the selected native124 output to a training label.
The one-step join is deliberate: the robot torso measured before ``step`` is
the buffered q9 torso anchor for the causal observation returned after it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any, Protocol

import numpy as np
import torch

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTION_DIM,
    OBSERVATION_DIM,
    build_checkpoint21204_observation,
    checkpoint21204_raw_action_to_hardware_targets,
    hardware_targets_to_checkpoint21204_raw_action,
)

ENCODER_DIM = 267
TOKENIZER_DIM = ENCODER_DIM + 1
PROPRIOCEPTION_DIM = 930
TOKEN_DIM = 64
DECODER_INPUT_DIM = TOKEN_DIM + PROPRIOCEPTION_DIM
TORSO_BODY_NAME = "torso_link"
CONTROL_HZ = 50.0


class ShadowTeacher(Protocol):
    """Minimal selected-policy surface accepted by the shadow collector."""

    def run(self, observation: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class MjlabShadowBeforeStep:
    """Immutable-by-convention copy of one pre-step SONIC state."""

    collector_identity: int
    sequence: int
    num_envs: int
    command_identity: int
    command_q9_indices: np.ndarray
    command_counters: np.ndarray
    episode_lengths: np.ndarray
    robot_torso_quaternion_wxyz: np.ndarray
    encoder_input_267: np.ndarray
    proprioception_h10_930: np.ndarray
    student_raw_native_action_23: np.ndarray
    encoded_token_64: np.ndarray | None
    decoder_input_994: np.ndarray | None
    continuity_from_previous_step: np.ndarray


def _raw_environment(env: Any) -> Any:
    raw = getattr(env, "unwrapped", None)
    return raw if raw is not None and raw is not env else env


def _num_envs(env: Any) -> int:
    value = getattr(env, "num_envs", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError("MJLab environment num_envs must be a positive integer")
    return value


def _observation_group(observations: Any, name: str) -> Any:
    try:
        return observations[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"MJLab observations lack {name!r} group") from error


def _float32_batch(value: Any, rows: int, width: int, context: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or value.shape != (rows, width):
            raise ValueError(f"{context} must be float32 [{rows},{width}]")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{context} contains NaN or Inf")
        return value.detach().to(device="cpu").contiguous().numpy().copy()
    if isinstance(value, np.ndarray):
        if value.dtype != np.float32 or value.shape != (rows, width):
            raise ValueError(f"{context} must be float32 [{rows},{width}]")
        if not np.isfinite(value).all():
            raise ValueError(f"{context} contains NaN or Inf")
        return np.ascontiguousarray(value).copy()
    raise TypeError(f"{context} must be a torch.Tensor or numpy.ndarray")


def _integer_vector(value: Any, rows: int, context: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.shape[0] != rows or value.dtype == torch.bool:
            raise ValueError(f"{context} must be integer [{rows}]")
        if value.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError(f"{context} must be integer [{rows}]")
        return value.detach().to(device="cpu", dtype=torch.int64).numpy().copy()
    if isinstance(value, np.ndarray):
        if value.shape != (rows,) or value.dtype.kind not in "iu":
            raise ValueError(f"{context} must be integer [{rows}]")
        return value.astype(np.int64, copy=True)
    raise TypeError(f"{context} must be a torch.Tensor or numpy.ndarray")


def _done_vector(value: Any, rows: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.shape != (rows,) or value.dtype not in {
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError(f"wrapped MJLab dones must be bool/integer [{rows}]")
        result = value.detach().to(device="cpu", dtype=torch.int64).numpy()
    elif isinstance(value, np.ndarray):
        if value.shape != (rows,) or value.dtype.kind not in "biu":
            raise ValueError(f"wrapped MJLab dones must be bool/integer [{rows}]")
        result = value.astype(np.int64, copy=False)
    else:
        raise TypeError("wrapped MJLab dones must be a torch.Tensor or numpy.ndarray")
    if not np.isin(result, (0, 1)).all():
        raise ValueError("wrapped MJLab dones must contain only zero or one")
    return result.astype(bool, copy=True)


def _motion_command(env: Any) -> Any:
    manager = getattr(env, "command_manager", None)
    getter = getattr(manager, "get_term", None)
    if not callable(getter):
        raise TypeError("MJLab environment lacks command manager")
    command = getter("motion")
    if command is None:
        raise ValueError("MJLab environment lacks motion command")
    return command


def _robot(env: Any) -> Any:
    try:
        return env.scene["robot"]
    except (KeyError, TypeError) as error:
        raise ValueError("MJLab scene lacks robot entity") from error


def _validate_robot_joint_order(robot: Any) -> None:
    if tuple(getattr(robot, "joint_names", ())) != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError("MJLab robot joint order differs from hardware compact 23")


def _torso_quaternion(env: Any, rows: int) -> np.ndarray:
    robot = _robot(env)
    _validate_robot_joint_order(robot)
    names = tuple(getattr(robot, "body_names", ()))
    if names.count(TORSO_BODY_NAME) != 1:
        raise ValueError("MJLab robot must contain exactly one torso_link")
    values = getattr(robot.data, "body_link_quat_w", None)
    if not isinstance(values, torch.Tensor) or values.ndim != 3:
        raise ValueError("MJLab robot body_link_quat_w must be a batched tensor")
    torso = values[:, names.index(TORSO_BODY_NAME), :]
    result = _float32_batch(torso, rows, 4, "robot torso quaternion")
    _validate_quaternions(result, "robot torso quaternion")
    return result


def _validate_quaternions(values: np.ndarray, context: str) -> None:
    norms = np.linalg.norm(values.astype(np.float64), axis=-1)
    if np.any(norms <= np.finfo(np.float32).eps) or np.any(np.abs(norms - 1.0) > 1.0e-3):
        raise ValueError(f"{context} must contain unit WXYZ quaternions")


def _relative_rotation_6d(
    robot_wxyz: np.ndarray,
    reference_wxyz: np.ndarray,
) -> np.ndarray:
    _validate_quaternions(robot_wxyz[None, :], "robot q9 torso quaternion")
    _validate_quaternions(reference_wxyz[None, :], "reference q9 torso quaternion")
    robot = robot_wxyz.astype(np.float64)
    reference = reference_wxyz.astype(np.float64)
    robot /= np.linalg.norm(robot)
    reference /= np.linalg.norm(reference)
    inverse = robot.copy()
    inverse[1:] *= -1.0
    w1, x1, y1, z1 = inverse
    w2, x2, y2, z2 = reference
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    matrix = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return matrix[:, :2].reshape(6).astype(np.float32)


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _extract_student_observation(
    observations: Any,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = _float32_batch(
        _observation_group(observations, "tokenizer"),
        rows,
        TOKENIZER_DIM,
        "MJLab tokenizer observation",
    )
    if not np.array_equal(tokenizer[:, 0], np.ones(rows, dtype=np.float32)):
        raise ValueError("MJLab tokenizer encoder route must equal one")
    policy = _float32_batch(
        _observation_group(observations, "policy"),
        rows,
        PROPRIOCEPTION_DIM,
        "MJLab policy observation",
    )
    return tokenizer[:, 1:].copy(), policy


def _resolve_decoder_inputs(
    *,
    rows: int,
    proprioception: np.ndarray,
    encoded_token_64: Any | None,
    decoder_input_994: Any | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    token = (
        None
        if encoded_token_64 is None
        else _float32_batch(encoded_token_64, rows, TOKEN_DIM, "encoded SONIC token")
    )
    decoder = (
        None
        if decoder_input_994 is None
        else _float32_batch(
            decoder_input_994,
            rows,
            DECODER_INPUT_DIM,
            "SONIC decoder input",
        )
    )
    if token is None and decoder is None:
        return None, None
    if decoder is None:
        assert token is not None
        return token, np.concatenate((token, proprioception), axis=-1)
    if token is None:
        token = decoder[:, :TOKEN_DIM].copy()
    expected = np.concatenate((token, proprioception), axis=-1)
    if not np.array_equal(decoder, expected):
        raise ValueError("SONIC decoder input must equal token64 concatenated with proprioception930")
    return token, decoder


def _encoder_bias_randomization_active(env: Any, robot: Any, rows: int) -> bool:
    bias = getattr(robot.data, "encoder_bias", None)
    if bias is None:
        return True
    values = _float32_batch(bias, rows, ACTION_DIM, "MJLab encoder bias")
    cfg = getattr(env, "cfg", None)
    events = getattr(cfg, "events", None)
    if not isinstance(events, Mapping):
        return True
    return events.get("encoder_bias") is not None or bool(np.count_nonzero(values))


def _measured_joint_position(env: Any, rows: int) -> np.ndarray:
    robot = _robot(env)
    biased = getattr(robot.data, "joint_pos_biased", None)
    if biased is not None:
        return _float32_batch(biased, rows, ACTION_DIM, "biased MJLab joint position")
    if _encoder_bias_randomization_active(env, robot, rows):
        raise ValueError(
            "joint_pos_biased is required while encoder-bias domain randomization is active or unknown"
        )
    return _float32_batch(robot.data.joint_pos, rows, ACTION_DIM, "MJLab joint position")


def _action_term(env: Any) -> Any:
    manager = getattr(env, "action_manager", None)
    getter = getattr(manager, "get_term", None)
    if not callable(getter):
        raise TypeError("MJLab environment lacks action manager")
    term = getter("joint_pos")
    if term is None:
        raise ValueError("MJLab environment lacks joint_pos action term")
    return term


def _motion_reference(
    command: Any,
    q9_indices: np.ndarray,
    rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion = getattr(command, "motion", None)
    joint_pos = getattr(motion, "joint_pos", None)
    body_quat = getattr(motion, "body_quat_w", None)
    if (
        not isinstance(joint_pos, torch.Tensor)
        or joint_pos.ndim != 2
        or joint_pos.shape[1] != ACTION_DIM
        or joint_pos.dtype != torch.float32
    ):
        raise ValueError("MJLab motion joint_pos must be float32 [frames,23]")
    body_names = tuple(getattr(getattr(command, "cfg", None), "body_names", ()))
    if body_names.count(TORSO_BODY_NAME) != 1:
        raise ValueError("MJLab motion command must contain exactly one torso_link")
    if (
        not isinstance(body_quat, torch.Tensor)
        or body_quat.ndim != 3
        or body_quat.shape[1] != len(body_names)
        or body_quat.shape[2] != 4
        or body_quat.dtype != torch.float32
    ):
        raise ValueError("MJLab motion body_quat_w contract mismatch")
    if np.any(q9_indices < 0) or np.any(q9_indices + 1 >= joint_pos.shape[0]):
        raise ValueError("MJLab causal q9 reference lacks q10 proof frame")
    index = torch.as_tensor(q9_indices, dtype=torch.long, device=joint_pos.device)
    proof = index + 1
    q9 = _float32_batch(joint_pos[index], rows, ACTION_DIM, "q9 motion joint position")
    q10 = _float32_batch(joint_pos[proof], rows, ACTION_DIM, "q10 motion joint position")
    torso = _float32_batch(
        body_quat[index, body_names.index(TORSO_BODY_NAME)],
        rows,
        4,
        "q9 reference torso quaternion",
    )
    _validate_quaternions(torso, "q9 reference torso quaternion")
    return q9, (q10 - q9) * np.float32(CONTROL_HZ), torso


class Native124Checkpoint21204MjlabShadowCollector:
    """Observe SONIC transitions and run selected native124 policy in shadow."""

    def __init__(self, teacher: ShadowTeacher) -> None:
        if not callable(getattr(teacher, "run", None)):
            raise TypeError("shadow teacher must expose run(observation)")
        self._teacher = teacher
        self._sequence = 0
        self._pending: MjlabShadowBeforeStep | None = None
        self._last_after_q9: np.ndarray | None = None
        self._last_after_episode_lengths: np.ndarray | None = None
        self._ready: np.ndarray | None = None

    def before_step(
        self,
        env: Any,
        observations: Any,
        student_raw_native_action: Any,
        *,
        encoded_token_64: Any | None = None,
        decoder_input_994: Any | None = None,
    ) -> MjlabShadowBeforeStep:
        """Copy pre-step state without changing environment or caller tensors."""

        if self._pending is not None:
            raise RuntimeError("shadow collector already has an unmatched before_step")
        raw = _raw_environment(env)
        rows = _num_envs(raw)
        command = _motion_command(raw)
        q9 = _integer_vector(command.time_steps, rows, "MJLab command time_steps")
        command_counters = _integer_vector(
            getattr(command, "command_counter", None),
            rows,
            "MJLab command_counter",
        )
        episode_lengths = _integer_vector(
            raw.episode_length_buf,
            rows,
            "MJLab episode_length_buf",
        )
        encoder, proprioception = _extract_student_observation(observations, rows)
        student_action = _float32_batch(
            student_raw_native_action,
            rows,
            ACTION_DIM,
            "student raw native action",
        )
        token, decoder = _resolve_decoder_inputs(
            rows=rows,
            proprioception=proprioception,
            encoded_token_64=encoded_token_64,
            decoder_input_994=decoder_input_994,
        )
        if (
            self._last_after_q9 is None
            or self._last_after_episode_lengths is None
            or self._ready is None
            or self._last_after_q9.shape != q9.shape
        ):
            continuity = np.zeros(rows, dtype=bool)
        else:
            continuity = (
                self._ready & (q9 == self._last_after_q9) & (episode_lengths == self._last_after_episode_lengths)
            )
        snapshot = MjlabShadowBeforeStep(
            collector_identity=id(self),
            sequence=self._sequence,
            num_envs=rows,
            command_identity=id(command),
            command_q9_indices=q9,
            command_counters=command_counters,
            episode_lengths=episode_lengths,
            robot_torso_quaternion_wxyz=_torso_quaternion(raw, rows),
            encoder_input_267=encoder,
            proprioception_h10_930=proprioception,
            student_raw_native_action_23=student_action,
            encoded_token_64=token,
            decoder_input_994=decoder,
            continuity_from_previous_step=continuity,
        )
        self._pending = snapshot
        return snapshot

    def after_step(
        self,
        env: Any,
        step_result: Any,
        before: MjlabShadowBeforeStep,
    ) -> dict[str, Any]:
        """Join post-step q10 state and return shadow candidates or quarantine."""

        if self._pending is None or before is not self._pending:
            raise RuntimeError("after_step must receive this collector's unmatched snapshot")
        try:
            result = self._after_step(env, step_result, before)
        except Exception:
            self._last_after_q9 = None
            self._last_after_episode_lengths = None
            self._ready = None
            raise
        finally:
            self._pending = None
            self._sequence += 1
        return result

    def _after_step(
        self,
        env: Any,
        step_result: Any,
        before: MjlabShadowBeforeStep,
    ) -> dict[str, Any]:
        if not isinstance(step_result, tuple) or len(step_result) != 4:
            raise TypeError("wrapped MJLab step result must be (observations, rewards, dones, extras)")
        post_observations, _rewards, done_value, _extras = step_result
        raw = _raw_environment(env)
        rows = _num_envs(raw)
        if rows != before.num_envs or before.collector_identity != id(self):
            raise ValueError("MJLab batch or shadow collector identity changed across step")
        command = _motion_command(raw)
        if id(command) != before.command_identity:
            raise ValueError("MJLab motion command object changed across step")

        after_q9 = _integer_vector(command.time_steps, rows, "post-step command time_steps")
        after_command_counters = _integer_vector(
            getattr(command, "command_counter", None),
            rows,
            "post-step command_counter",
        )
        after_episode = _integer_vector(
            raw.episode_length_buf,
            rows,
            "post-step episode_length_buf",
        )
        dones = _done_vector(done_value, rows)
        post_encoder, post_proprioception = _extract_student_observation(post_observations, rows)
        measured_q = _measured_joint_position(raw, rows)
        robot = _robot(raw)
        measured_qd = _float32_batch(
            robot.data.joint_vel,
            rows,
            ACTION_DIM,
            "MJLab joint velocity",
        )
        try:
            angular_velocity_value = raw.scene["robot/imu_ang_vel"].data
        except (KeyError, TypeError) as error:
            raise ValueError("MJLab scene lacks robot/imu_ang_vel sensor") from error
        angular_velocity = _float32_batch(
            angular_velocity_value,
            rows,
            3,
            "MJLab IMU angular velocity",
        )
        action_term = _action_term(raw)
        actual_raw_action = _float32_batch(
            getattr(action_term, "raw_action", None),
            rows,
            ACTION_DIM,
            "actual SONIC raw native action",
        )
        applied_target = _float32_batch(
            getattr(action_term, "processed_action", None),
            rows,
            ACTION_DIM,
            "actual SONIC processed hardware target",
        )
        reference_q, reference_qd, reference_torso = _motion_reference(command, after_q9, rows)

        progression = after_q9 == before.command_q9_indices + 1
        command_counter_unchanged = after_command_counters == before.command_counters
        episode_progression = after_episode == before.episode_lengths + 1
        resampled_value = getattr(command, "_causal_resampled", None)
        if resampled_value is None:
            resampled = np.zeros(rows, dtype=bool)
        else:
            resampled = _done_vector(resampled_value, rows)
        action_matches = np.all(
            actual_raw_action == before.student_raw_native_action_23,
            axis=-1,
        )

        self._last_after_q9 = after_q9.copy()
        self._last_after_episode_lengths = after_episode.copy()
        self._ready = (~dones) & progression & command_counter_unchanged & episode_progression & (~resampled)

        records: list[dict[str, Any]] = []
        for env_id in range(rows):
            reasons: list[str] = []
            if not before.continuity_from_previous_step[env_id]:
                reasons.append("first_reset_or_resynchronization_frame")
            if dones[env_id]:
                reasons.append("terminated_or_truncated")
            if not progression[env_id]:
                reasons.append("command_noncontiguous_or_resampled")
            if not command_counter_unchanged[env_id]:
                reasons.append("command_resampled")
            if not episode_progression[env_id]:
                reasons.append("episode_counter_reset_or_noncontiguous")
            if resampled[env_id]:
                reasons.append("causal_command_resampled")
            if not action_matches[env_id]:
                reasons.append("student_action_clipped_or_mismatched")
            records.append(
                self._record(
                    env_id=env_id,
                    before=before,
                    after_q9=after_q9,
                    after_command_counters=after_command_counters,
                    after_episode=after_episode,
                    post_encoder=post_encoder,
                    post_proprioception=post_proprioception,
                    reference_q=reference_q,
                    reference_qd=reference_qd,
                    reference_torso=reference_torso,
                    measured_q=measured_q,
                    measured_qd=measured_qd,
                    angular_velocity=angular_velocity,
                    actual_raw_action=actual_raw_action,
                    applied_target=applied_target,
                    reasons=reasons,
                )
            )
        return {
            "schema_version": 1,
            "kind": "g1_true23_native124_21204_mjlab_shadow_transition_batch_v1",
            "sequence": before.sequence,
            "diagnostic_only": True,
            "robot_commands_performed_by_collector": False,
            "actuation_permitted": False,
            "deployment_ready": False,
            "teacher_labels_admitted": False,
            "promotion_eligible": False,
            "causal_distribution_shift_requires_support_gate": True,
            "record_count": rows,
            "candidate_count": sum(record["verdict"] == "shadow_candidate" for record in records),
            "quarantine_count": sum(record["verdict"] == "quarantine" for record in records),
            "records": records,
        }

    def _record(
        self,
        *,
        env_id: int,
        before: MjlabShadowBeforeStep,
        after_q9: np.ndarray,
        after_command_counters: np.ndarray,
        after_episode: np.ndarray,
        post_encoder: np.ndarray,
        post_proprioception: np.ndarray,
        reference_q: np.ndarray,
        reference_qd: np.ndarray,
        reference_torso: np.ndarray,
        measured_q: np.ndarray,
        measured_qd: np.ndarray,
        angular_velocity: np.ndarray,
        actual_raw_action: np.ndarray,
        applied_target: np.ndarray,
        reasons: list[str],
    ) -> dict[str, Any]:
        student_transition = {
            "encoder_input_267": before.encoder_input_267[env_id].tolist(),
            "encoder_input_sha256": _sha256(before.encoder_input_267[env_id]),
            "proprioception_h10_930": before.proprioception_h10_930[env_id].tolist(),
            "proprioception_sha256": _sha256(before.proprioception_h10_930[env_id]),
            "requested_raw_native_action_23": before.student_raw_native_action_23[env_id].tolist(),
            "actual_raw_native_action_23": actual_raw_action[env_id].tolist(),
            "encoded_token_64": (
                None if before.encoded_token_64 is None else before.encoded_token_64[env_id].tolist()
            ),
            "decoder_input_994": (
                None if before.decoder_input_994 is None else before.decoder_input_994[env_id].tolist()
            ),
        }
        next_state = {
            "encoder_input_267": post_encoder[env_id].tolist(),
            "encoder_input_sha256": _sha256(post_encoder[env_id]),
            "proprioception_h10_930": post_proprioception[env_id].tolist(),
            "proprioception_sha256": _sha256(post_proprioception[env_id]),
        }
        base = {
            "env_id": env_id,
            "verdict": "quarantine" if reasons else "shadow_candidate",
            "quarantine_reasons": reasons,
            "diagnostic_only": True,
            "robot_commands_performed_by_collector": False,
            "actuation_permitted": False,
            "deployment_ready": False,
            "teacher_label_admitted": False,
            "training_label_eligible": False,
            "promotion_eligible": False,
            "support_gate_required": True,
            "q9_reference_index_before": int(before.command_q9_indices[env_id]),
            "q10_proof_index_before": int(before.command_q9_indices[env_id] + 1),
            "q9_reference_index_after": int(after_q9[env_id]),
            "q10_proof_index_after": int(after_q9[env_id] + 1),
            "episode_length_before": int(before.episode_lengths[env_id]),
            "episode_length_after": int(after_episode[env_id]),
            "command_counter_before": int(before.command_counters[env_id]),
            "command_counter_after": int(after_command_counters[env_id]),
            "student_transition": student_transition,
            "student_next_state": next_state,
            "previous_applied_target_hardware": applied_target[env_id].tolist(),
            "selected_observation_124": None,
            "selected_observation_sha256": None,
            "raw_tracker_action_hardware": None,
            "candidate_target_hardware": None,
        }
        if reasons:
            return base
        previous_raw = hardware_targets_to_checkpoint21204_raw_action(applied_target[env_id])
        rotation = _relative_rotation_6d(
            before.robot_torso_quaternion_wxyz[env_id],
            reference_torso[env_id],
        )
        observation = build_checkpoint21204_observation(
            q_ref_hardware=reference_q[env_id],
            qd_ref_hardware=reference_qd[env_id],
            torso_motion_anchor_ori_b=rotation,
            base_angular_velocity=angular_velocity[env_id],
            q_measured_hardware=measured_q[env_id],
            qd_measured_hardware=measured_qd[env_id],
            previous_applied_raw_action_hardware=previous_raw,
        )
        action = self._teacher.run(observation)
        if (
            not isinstance(action, np.ndarray)
            or action.dtype != np.float32
            or action.shape != (ACTION_DIM,)
            or not np.isfinite(action).all()
        ):
            raise RuntimeError("shadow teacher action must be finite float32 [23]")
        target = checkpoint21204_raw_action_to_hardware_targets(action)
        base.update(
            {
                "selected_observation_124": observation[0].tolist(),
                "selected_observation_sha256": _sha256(observation),
                "previous_applied_raw_action_hardware": previous_raw.tolist(),
                "raw_tracker_action_hardware": action.tolist(),
                "candidate_target_hardware": target.tolist(),
            }
        )
        return base


assert TOKENIZER_DIM == 268
assert DECODER_INPUT_DIM == 994
assert OBSERVATION_DIM == 124
