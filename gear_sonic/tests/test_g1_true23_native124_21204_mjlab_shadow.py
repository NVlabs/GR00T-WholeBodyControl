from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from gear_sonic.utils import g1_true23_native124_21204_mjlab_shadow as shadow
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTION_SCALE_HARDWARE,
    HOME_Q_HARDWARE,
)


class _Teacher:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.action = np.linspace(-0.5, 0.5, 23, dtype=np.float32)

    def run(self, observation: np.ndarray) -> np.ndarray:
        self.calls.append(observation.copy())
        return self.action.copy()


class _Manager:
    def __init__(self, term: Any) -> None:
        self.term = term

    def get_term(self, name: str) -> Any:
        return self.term if name in {"motion", "joint_pos"} else None


class _RobotData:
    def __init__(self, num_envs: int) -> None:
        self.body_link_quat_w = torch.zeros(num_envs, 2, 4, dtype=torch.float32)
        self.body_link_quat_w[..., 0] = 1.0
        self.joint_pos = torch.as_tensor(HOME_Q_HARDWARE).repeat(num_envs, 1)
        self.joint_pos_biased = self.joint_pos.clone()
        self.joint_vel = torch.zeros(num_envs, 23, dtype=torch.float32)
        self.encoder_bias = torch.zeros(num_envs, 23, dtype=torch.float32)


class _Environment:
    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        frame = torch.arange(80, dtype=torch.float32)[:, None]
        joint = torch.arange(23, dtype=torch.float32)[None, :]
        joint_pos = frame * 0.01 + joint * 0.001
        body_quat = torch.zeros(80, 2, 4, dtype=torch.float32)
        body_quat[..., 0] = 1.0
        motion = SimpleNamespace(joint_pos=joint_pos, body_quat_w=body_quat)
        self.command = SimpleNamespace(
            time_steps=torch.arange(9, 9 + num_envs, dtype=torch.long),
            command_counter=torch.ones(num_envs, dtype=torch.long),
            motion=motion,
            cfg=SimpleNamespace(body_names=("pelvis", "torso_link")),
            _causal_resampled=torch.zeros(num_envs, dtype=torch.bool),
        )
        self.command_manager = _Manager(self.command)
        self.action = SimpleNamespace(
            raw_action=torch.zeros(num_envs, 23, dtype=torch.float32),
            processed_action=torch.as_tensor(HOME_Q_HARDWARE).repeat(num_envs, 1),
        )
        self.action_manager = _Manager(self.action)
        self.robot = SimpleNamespace(
            body_names=("pelvis", "torso_link"),
            joint_names=tuple(shadow.HARDWARE_23_JOINT_NAMES),
            data=_RobotData(num_envs),
        )
        self.imu = SimpleNamespace(data=torch.zeros(num_envs, 3, dtype=torch.float32))
        self.scene = {"robot": self.robot, "robot/imu_ang_vel": self.imu}
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.cfg = SimpleNamespace(events={})


def _observations(num_envs: int, offset: float = 0.0) -> dict[str, torch.Tensor]:
    tokenizer = torch.arange(num_envs * 268, dtype=torch.float32).reshape(num_envs, 268)
    tokenizer += offset
    tokenizer[:, 0] = 1.0
    policy = torch.arange(num_envs * 930, dtype=torch.float32).reshape(num_envs, 930)
    policy = policy * 0.001 + offset
    return {"tokenizer": tokenizer, "policy": policy}


def _advance(
    env: _Environment,
    action: torch.Tensor,
    *,
    increment: int = 1,
    dones: torch.Tensor | None = None,
    observation_offset: float = 1.0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, object]]:
    env.command.time_steps += increment
    env.episode_length_buf += 1
    env.action.raw_action.copy_(action)
    selected_previous_raw = torch.linspace(-0.3, 0.3, 23).repeat(env.num_envs, 1)
    target = torch.as_tensor(HOME_Q_HARDWARE) + torch.as_tensor(ACTION_SCALE_HARDWARE) * selected_previous_raw
    env.action.processed_action.copy_(target)
    env.robot.data.joint_pos += 0.02
    env.robot.data.joint_pos_biased = env.robot.data.joint_pos + env.robot.data.encoder_bias
    env.robot.data.joint_vel[:] = torch.linspace(-1.0, 1.0, 23)
    env.imu.data[:] = torch.tensor((0.25, -0.5, 0.75))
    half = np.sqrt(0.5)
    env.robot.data.body_link_quat_w[:, 1] = torch.tensor((half, 0.0, 0.0, half))
    if dones is None:
        dones = torch.zeros(env.num_envs, dtype=torch.long)
    return (
        _observations(env.num_envs, observation_offset),
        torch.zeros(env.num_envs),
        dones,
        {},
    )


def _prime_one_quarantined_transition(
    collector: shadow.Native124Checkpoint21204MjlabShadowCollector,
    env: _Environment,
    observations: dict[str, torch.Tensor],
) -> dict[str, Any]:
    action = torch.linspace(-0.2, 0.2, 23).repeat(env.num_envs, 1)
    before = collector.before_step(env, observations, action)
    result = _advance(env, action)
    return collector.after_step(env, result, before)


def test_first_frame_is_quarantined_without_teacher_inference_or_input_mutation() -> None:
    teacher = _Teacher()
    collector = shadow.Native124Checkpoint21204MjlabShadowCollector(teacher)
    env = _Environment()
    observations = _observations(env.num_envs)
    action = torch.linspace(-0.2, 0.2, 23).repeat(env.num_envs, 1)
    token = torch.linspace(-1.0, 1.0, 64).repeat(env.num_envs, 1)
    original_observations = {name: value.clone() for name, value in observations.items()}
    original_action = action.clone()
    original_token = token.clone()
    original_command = env.command.time_steps.clone()
    original_torso = env.robot.data.body_link_quat_w.clone()

    before = collector.before_step(env, observations, action, encoded_token_64=token)

    assert torch.equal(env.command.time_steps, original_command)
    assert torch.equal(env.robot.data.body_link_quat_w, original_torso)
    assert all(torch.equal(observations[name], value) for name, value in original_observations.items())
    assert torch.equal(action, original_action)
    assert torch.equal(token, original_token)
    np.testing.assert_array_equal(before.encoder_input_267, observations["tokenizer"][:, 1:].numpy())
    np.testing.assert_array_equal(before.proprioception_h10_930, observations["policy"].numpy())
    assert before.decoder_input_994 is not None
    np.testing.assert_array_equal(before.decoder_input_994[:, :64], token.numpy())
    np.testing.assert_array_equal(before.decoder_input_994[:, 64:], observations["policy"].numpy())

    step_result = _advance(env, action)
    state_before_after_call = (
        env.command.time_steps.clone(),
        env.robot.data.joint_pos_biased.clone(),
        env.action.processed_action.clone(),
    )
    batch = collector.after_step(env, step_result, before)
    state_after_after_call = (
        env.command.time_steps,
        env.robot.data.joint_pos_biased,
        env.action.processed_action,
    )

    assert all(torch.equal(left, right) for left, right in zip(state_before_after_call, state_after_after_call))
    assert batch["candidate_count"] == 0
    assert batch["quarantine_count"] == env.num_envs
    assert teacher.calls == []
    for record in batch["records"]:
        assert record["verdict"] == "quarantine"
        assert record["quarantine_reasons"] == ["first_reset_or_resynchronization_frame"]
        assert record["raw_tracker_action_hardware"] is None
        assert record["candidate_target_hardware"] is None
        assert record["teacher_label_admitted"] is False


def test_contiguous_second_transition_builds_exact_selected_observation_and_student_tuple() -> None:
    teacher = _Teacher()
    collector = shadow.Native124Checkpoint21204MjlabShadowCollector(teacher)
    env = _Environment()
    observations = _observations(env.num_envs)
    first = _prime_one_quarantined_transition(collector, env, observations)
    assert first["candidate_count"] == 0
    pre_observations = _observations(env.num_envs, 1.0)
    action = torch.linspace(-0.15, 0.15, 23).repeat(env.num_envs, 1)
    token = torch.linspace(0.0, 0.63, 64).repeat(env.num_envs, 1)
    before = collector.before_step(env, pre_observations, action, encoded_token_64=token)
    pre_torso = before.robot_torso_quaternion_wxyz.copy()
    q9_before = before.command_q9_indices.copy()
    step_result = _advance(env, action, observation_offset=2.0)
    batch = collector.after_step(env, step_result, before)

    assert batch["candidate_count"] == env.num_envs
    assert batch["quarantine_count"] == 0
    assert len(teacher.calls) == env.num_envs
    record = batch["records"][0]
    assert record["verdict"] == "shadow_candidate"
    assert record["q9_reference_index_after"] == int(q9_before[0] + 1)
    assert record["q10_proof_index_after"] == int(q9_before[0] + 2)
    np.testing.assert_array_equal(
        np.asarray(record["student_transition"]["encoder_input_267"], dtype=np.float32),
        pre_observations["tokenizer"][0, 1:].numpy(),
    )
    np.testing.assert_array_equal(
        np.asarray(record["student_next_state"]["encoder_input_267"], dtype=np.float32),
        step_result[0]["tokenizer"][0, 1:].numpy(),
    )
    observation = np.asarray(record["selected_observation_124"], dtype=np.float32)
    q9 = record["q9_reference_index_after"]
    expected_q9 = env.command.motion.joint_pos[q9].numpy()
    expected_q10 = env.command.motion.joint_pos[q9 + 1].numpy()
    np.testing.assert_array_equal(observation[:23], expected_q9)
    np.testing.assert_allclose(observation[23:46], (expected_q10 - expected_q9) * 50.0, atol=0.0)
    assert pre_torso[0, 0] == pytest.approx(np.sqrt(0.5), abs=1.0e-7)
    np.testing.assert_allclose(observation[46:52], (0.0, 1.0, -1.0, 0.0, 0.0, 0.0), atol=2.0e-7)
    np.testing.assert_array_equal(observation[52:55], (0.25, -0.5, 0.75))
    np.testing.assert_allclose(
        observation[55:78],
        env.robot.data.joint_pos_biased[0].numpy() - HOME_Q_HARDWARE,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(observation[78:101], env.robot.data.joint_vel[0].numpy())
    np.testing.assert_allclose(observation[101:124], np.linspace(-0.3, 0.3, 23), atol=1.0e-7)
    np.testing.assert_array_equal(
        np.asarray(record["raw_tracker_action_hardware"], dtype=np.float32),
        teacher.action,
    )
    np.testing.assert_allclose(
        np.asarray(record["candidate_target_hardware"], dtype=np.float32),
        HOME_Q_HARDWARE + ACTION_SCALE_HARDWARE * teacher.action,
        atol=1.0e-7,
    )
    assert record["teacher_label_admitted"] is False
    assert record["training_label_eligible"] is False
    assert batch["robot_commands_performed_by_collector"] is False


@pytest.mark.parametrize(
    ("failure", "reason"),
    (
        ("done", "terminated_or_truncated"),
        ("jump", "command_noncontiguous_or_resampled"),
        ("resample_counter", "command_resampled"),
        ("episode_reset", "episode_counter_reset_or_noncontiguous"),
        ("action_mismatch", "student_action_clipped_or_mismatched"),
    ),
)
def test_done_reset_resample_and_action_mismatch_are_quarantined(
    failure: str,
    reason: str,
) -> None:
    teacher = _Teacher()
    collector = shadow.Native124Checkpoint21204MjlabShadowCollector(teacher)
    env = _Environment()
    _prime_one_quarantined_transition(collector, env, _observations(env.num_envs))
    action = torch.linspace(-0.1, 0.1, 23).repeat(env.num_envs, 1)
    before = collector.before_step(env, _observations(env.num_envs, 1.0), action)
    step_result = _advance(
        env,
        action,
        increment=2 if failure == "jump" else 1,
        dones=torch.ones(env.num_envs, dtype=torch.long) if failure == "done" else None,
    )
    if failure == "episode_reset":
        env.episode_length_buf.zero_()
    if failure == "resample_counter":
        env.command.command_counter.add_(1)
    if failure == "action_mismatch":
        env.action.raw_action.add_(1.0)
    batch = collector.after_step(env, step_result, before)

    assert batch["candidate_count"] == 0
    assert teacher.calls == []
    assert all(reason in record["quarantine_reasons"] for record in batch["records"])
    assert all(record["raw_tracker_action_hardware"] is None for record in batch["records"])


def test_discontinuity_forces_one_resynchronization_transition_before_labels() -> None:
    teacher = _Teacher()
    collector = shadow.Native124Checkpoint21204MjlabShadowCollector(teacher)
    env = _Environment(num_envs=1)
    observations = _observations(1)
    _prime_one_quarantined_transition(collector, env, observations)
    env.command.time_steps += 7
    action = torch.zeros(1, 23)
    before = collector.before_step(env, _observations(1, 1.0), action)
    second = collector.after_step(env, _advance(env, action, observation_offset=2.0), before)
    assert second["records"][0]["quarantine_reasons"] == ["first_reset_or_resynchronization_frame"]
    before = collector.before_step(env, _observations(1, 2.0), action)
    third = collector.after_step(env, _advance(env, action, observation_offset=3.0), before)
    assert third["candidate_count"] == 1
    assert len(teacher.calls) == 1


def test_decoder_input_must_match_token_plus_exact_h10_proprioception() -> None:
    collector = shadow.Native124Checkpoint21204MjlabShadowCollector(_Teacher())
    env = _Environment(num_envs=1)
    observations = _observations(1)
    action = torch.zeros(1, 23)
    token = torch.zeros(1, 64)
    decoder = torch.cat((token, observations["policy"]), dim=-1)
    decoder[0, -1] += 1.0
    original = decoder.clone()
    with pytest.raises(ValueError, match="token64 concatenated"):
        collector.before_step(
            env,
            observations,
            action,
            encoded_token_64=token,
            decoder_input_994=decoder,
        )
    assert torch.equal(decoder, original)


def test_joint_position_fallback_is_rejected_when_encoder_bias_dr_is_active() -> None:
    collector = shadow.Native124Checkpoint21204MjlabShadowCollector(_Teacher())
    env = _Environment(num_envs=1)
    observations = _observations(1)
    action = torch.zeros(1, 23)
    before = collector.before_step(env, observations, action)
    step_result = _advance(env, action)
    del env.robot.data.joint_pos_biased
    env.cfg.events["encoder_bias"] = object()
    with pytest.raises(ValueError, match="joint_pos_biased is required"):
        collector.after_step(env, step_result, before)


@pytest.mark.parametrize(
    ("field", "bad", "match"),
    (
        ("tokenizer", torch.zeros(2, 267), "tokenizer observation"),
        ("policy", torch.full((2, 930), torch.nan), "NaN or Inf"),
        ("action", torch.zeros(2, 22), "student raw native action"),
    ),
)
def test_before_step_rejects_shape_and_finiteness_drift(
    field: str,
    bad: torch.Tensor,
    match: str,
) -> None:
    collector = shadow.Native124Checkpoint21204MjlabShadowCollector(_Teacher())
    env = _Environment()
    observations = _observations(2)
    action = torch.zeros(2, 23)
    if field == "action":
        action = bad
    else:
        observations[field] = bad
    with pytest.raises((TypeError, ValueError), match=match):
        collector.before_step(env, observations, action)


def test_shadow_module_has_no_transport_or_filesystem_write_surface() -> None:
    source = Path(shadow.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    forbidden_imports = ("unitree", "dds", "zmq", "socket", "subprocess")
    assert not any(token in name.lower() for name in imports for token in forbidden_imports)
    assert not {"open", "write_text", "write_bytes", "save", "savetxt"}.intersection(calls)


def test_snapshot_copies_are_not_aliases_of_caller_values() -> None:
    collector = shadow.Native124Checkpoint21204MjlabShadowCollector(_Teacher())
    env = _Environment(num_envs=1)
    observations = _observations(1)
    action = torch.zeros(1, 23)
    before = collector.before_step(env, observations, action)
    frozen = copy.deepcopy(before)
    observations["tokenizer"].add_(10.0)
    observations["policy"].add_(20.0)
    action.add_(30.0)
    env.robot.data.body_link_quat_w[:, 1] = torch.tensor((0.0, 0.0, 0.0, 1.0))
    np.testing.assert_array_equal(before.encoder_input_267, frozen.encoder_input_267)
    np.testing.assert_array_equal(before.proprioception_h10_930, frozen.proprioception_h10_930)
    np.testing.assert_array_equal(before.student_raw_native_action_23, frozen.student_raw_native_action_23)
    np.testing.assert_array_equal(
        before.robot_torso_quaternion_wxyz,
        frozen.robot_torso_quaternion_wxyz,
    )
