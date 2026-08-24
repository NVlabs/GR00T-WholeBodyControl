"""Tensor-level tests for exact SONIC semantics at the MJLab boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23 import (
    SONIC_HARDWARE_DEFAULT_Q,
    SONIC_TRUE23_POLICY_DIM,
    SONIC_TRUE23_TOKENIZER_DIM,
    _clamp_reference_start_indices_,
    command_multi_future_lower_body,
    flatten_term_major_policy_history,
    hardware_to_native_il23,
    motion_anchor_ori_b,
    native_actions_to_hardware_targets,
    native_il23_to_hardware,
    pad_native_il23_to_canonical_il29,
    prime_sonic_true23_training_environment,
    vr_3point_local_orn_target,
    vr_3point_local_target,
)
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    ISAACLAB_TO_MUJOCO_DOF,
    NATIVE_IL23_TO_CANONICAL_IL29,
    SOURCE_IL29_EXCLUDED_INDICES,
)


def test_contract_dimensions_are_exact() -> None:
    assert SONIC_TRUE23_TOKENIZER_DIM == 268
    assert SONIC_TRUE23_POLICY_DIM == 930


def test_native_hardware_permutations_and_targets_are_exact() -> None:
    native = torch.arange(23, dtype=torch.float32).unsqueeze(0)
    hardware = native_il23_to_hardware(native)
    assert hardware.tolist()[0] == [float(native[0, index]) for index in ISAACLAB_TO_MUJOCO_DOF]
    assert torch.equal(hardware_to_native_il23(hardware), native)

    targets = native_actions_to_hardware_targets(native)
    expected = torch.tensor(SONIC_HARDWARE_DEFAULT_Q) + hardware[0] * torch.tensor(HARDWARE_23_ACTION_SCALE)
    assert torch.equal(targets[0], expected)


def test_padded_il29_has_only_native_slots() -> None:
    native = torch.arange(1, 24, dtype=torch.float32).unsqueeze(0)
    padded = pad_native_il23_to_canonical_il29(native)
    assert padded.shape == (1, 29)
    for native_index, canonical_index in enumerate(NATIVE_IL23_TO_CANONICAL_IL29):
        assert padded[0, canonical_index] == native[0, native_index]
    assert torch.count_nonzero(padded[0, list(SOURCE_IL29_EXCLUDED_INDICES)]) == 0


def test_term_major_history_order_and_phantom_slots() -> None:
    batch = 2
    base = torch.arange(batch * 10 * 3, dtype=torch.float32).reshape(batch, 10, 3)
    q = torch.zeros(batch, 10, 29)
    dq = torch.zeros_like(q)
    action = torch.zeros_like(q)
    q[..., 0] = torch.arange(10)
    dq[..., 1] = torch.arange(10) + 20
    action[..., 2] = torch.arange(10) + 40
    gravity = torch.arange(batch * 10 * 3, dtype=torch.float32).reshape(batch, 10, 3)
    flat = flatten_term_major_policy_history(
        base_ang_vel=base,
        joint_pos_rel=q,
        joint_vel=dq,
        previous_action=action,
        projected_gravity=gravity,
    )
    assert flat.shape == (batch, 930)
    assert torch.equal(flat[:, :30], base.flatten(start_dim=1))
    assert torch.equal(flat[:, 30:320], q.flatten(start_dim=1))
    assert torch.equal(flat[:, 320:610], dq.flatten(start_dim=1))
    assert torch.equal(flat[:, 610:900], action.flatten(start_dim=1))
    assert torch.equal(flat[:, 900:], gravity.flatten(start_dim=1))

    q[..., SOURCE_IL29_EXCLUDED_INDICES[0]] = 1.0
    with pytest.raises(ValueError, match="missing canonical"):
        flatten_term_major_policy_history(
            base_ang_vel=base,
            joint_pos_rel=q,
            joint_vel=dq,
            previous_action=action,
            projected_gravity=gravity,
        )


def _fake_reference_env(time_steps: torch.Tensor) -> SimpleNamespace:
    total = 80
    joint_pos = torch.arange(total * 23, dtype=torch.float32).reshape(total, 23)
    joint_vel = joint_pos + 10000
    body_names = (
        "pelvis",
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand",
        "torso_link",
    )
    batch = time_steps.shape[0]
    body_pos = torch.zeros(batch, len(body_names), 3)
    body_pos[:, 1] = torch.tensor((1.0, 0.0, 0.0))
    body_pos[:, 2] = torch.tensor((0.0, 1.0, 0.0))
    body_pos[:, 3] = torch.tensor((0.0, 0.0, 1.0))
    body_quat = torch.zeros(batch, len(body_names), 4)
    body_quat[..., 0] = 1.0
    identity = torch.zeros(batch, 4)
    identity[:, 0] = 1.0
    command = SimpleNamespace(
        time_steps=time_steps,
        motion=SimpleNamespace(
            time_step_total=total,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        ),
        cfg=SimpleNamespace(body_names=body_names),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        anchor_pos_w=torch.zeros(batch, 3),
        anchor_quat_w=identity,
        robot_anchor_quat_w=identity,
    )
    return SimpleNamespace(
        num_envs=batch,
        command_manager=SimpleNamespace(get_term=lambda name: command if name == "motion" else None),
    )


def test_tokenizer_reference_terms_match_exact_layout() -> None:
    env = _fake_reference_env(torch.tensor((2, 4)))
    command = env.command_manager.get_term("motion")
    value = command_multi_future_lower_body(env, "motion")
    indexes = command.time_steps[:, None] + torch.arange(0, 50, 5)
    expected_pos = command.motion.joint_pos[indexes, :12].reshape(2, 120)
    expected_vel = command.motion.joint_vel[indexes, :12].reshape(2, 120)
    assert torch.equal(value[:, :120], expected_pos)
    assert torch.equal(value[:, 120:], expected_vel)

    position = vr_3point_local_target(env, "motion")
    orientation = vr_3point_local_orn_target(env, "motion")
    anchor_orientation = motion_anchor_ori_b(env, "motion")
    assert position.shape == (2, 9)
    assert torch.allclose(
        position[0],
        torch.tensor(
            (
                1.18,
                -0.025,
                0.0,
                0.18,
                1.025,
                0.0,
                0.0,
                0.0,
                1.35,
            )
        ),
    )
    assert torch.equal(
        orientation,
        torch.tensor((1.0, 0.0, 0.0, 0.0) * 3).repeat(2, 1),
    )
    assert torch.equal(
        anchor_orientation,
        torch.tensor((1.0, 0.0, 0.0, 1.0, 0.0, 0.0)).repeat(2, 1),
    )


def test_reference_near_end_is_rejected_not_clamped() -> None:
    env = _fake_reference_env(torch.tensor((34,)))
    with pytest.raises(ValueError, match="genuine future"):
        command_multi_future_lower_body(env, "motion")


def test_adaptive_reference_clamp_writes_back_advanced_indices() -> None:
    starts = torch.tensor((2, 714, 4, 700), dtype=torch.long)
    env_ids = torch.tensor((1, 3), dtype=torch.long)

    _clamp_reference_start_indices_(starts, env_ids, 668)

    assert torch.equal(starts, torch.tensor((2, 668, 4, 668)))


def test_training_prime_refreshes_post_wrapper_without_simulation_step() -> None:
    body_names = (
        "pelvis",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand",
    )
    target = torch.zeros(2, len(body_names), 3)
    measured = target.clone()
    measured[..., -1] = 0.1
    refresh_calls = 0

    def refresh_targets() -> None:
        nonlocal refresh_calls
        refresh_calls += 1

    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=body_names),
        body_pos_relative_w=target,
        robot_body_pos_w=measured,
        time_steps=torch.tensor((4, 8), dtype=torch.long),
        time_left=torch.ones(2),
        command_counter=torch.ones(2, dtype=torch.long),
        refresh_relative_body_targets_after_reset=refresh_targets,
    )

    class FakeObservationManager:
        def __init__(self) -> None:
            self.reset_ids: torch.Tensor | None = None

        def reset(self, env_ids: torch.Tensor) -> None:
            self.reset_ids = env_ids.clone()

        def compute(self, *, update_history: bool) -> dict[str, torch.Tensor]:
            assert update_history is True
            return {
                "tokenizer": torch.zeros(2, 268),
                "policy": torch.zeros(2, 930),
                "critic": torch.zeros(2, 256),
            }

    ee_cfg = SimpleNamespace(
        time_out=False,
        params={"threshold": 0.25},
        func=lambda _env, threshold: (
            torch.abs(command.body_pos_relative_w[:, 1:, -1] - command.robot_body_pos_w[:, 1:, -1]).amax(dim=1)
            > threshold
        ),
    )

    class FakeEnvironment:
        num_envs = 2
        device = "cpu"
        common_step_counter = 0
        _sim_step_counter = 0
        episode_length_buf = torch.zeros(2, dtype=torch.long)
        cfg = SimpleNamespace(terminations={"ee_body_pos": ee_cfg})
        command_manager = SimpleNamespace(get_term=lambda name: command if name == "motion" else None)
        termination_manager = SimpleNamespace(
            active_terms=("ee_body_pos",),
            get_term_cfg=lambda name: ee_cfg if name == "ee_body_pos" else None,
        )

        def __init__(self) -> None:
            self.prime_log = {"Episode_Termination/ee_body_pos": 1}
            self.extras = {
                "log": self.prime_log,
                "time_outs": torch.zeros(2, dtype=torch.bool),
            }
            self.observation_manager = FakeObservationManager()
            self.scene = {
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        joint_pos=torch.zeros(2, 23),
                        joint_vel=torch.zeros(2, 23),
                    )
                )
            }

    env = FakeEnvironment()
    wrapped = SimpleNamespace(unwrapped=env)
    report = prime_sonic_true23_training_environment(wrapped)

    assert report["physics_steps"] == 0
    assert report["target_refreshes"] == 1
    assert report["full_batch_reset_retries"] == 0
    assert report["initial_termination_counts"] == {
        "ee_body_pos": 0,
    }
    assert report["discarded_prime_log_entries"] == 1
    assert report["post_prime_max_ee_z_error"] == pytest.approx(0.1)
    assert refresh_calls == 1
    assert torch.equal(
        env.observation_manager.reset_ids,
        torch.tensor((0, 1)),
    )
    assert env.obs_buf["policy"].shape == (2, 930)
    assert env.extras["log"] == {}
    assert env.extras["log"] is not env.prime_log
    assert env.extras["time_outs"].shape == (2,)
    assert env.prime_log == {"Episode_Termination/ee_body_pos": 1}


def test_training_prime_rejects_pre_wrapper_call() -> None:
    raw_env = SimpleNamespace(
        unwrapped=None,
    )
    raw_env.unwrapped = raw_env

    with pytest.raises(TypeError, match="RslRlVecEnvWrapper"):
        prime_sonic_true23_training_environment(raw_env)
