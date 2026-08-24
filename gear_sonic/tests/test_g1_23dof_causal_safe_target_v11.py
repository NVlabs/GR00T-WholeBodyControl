from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23 import padded_previous_action
from gear_sonic.envs.mjlab.sonic_true23_causal_history_safe_target_v11 import (
    causal_history_safe_target_v11_contract,
    evaluator_aligned_recovery_metric,
)
from gear_sonic.scripts.train_g1_23dof_mjlab_causal_safe_target_v11 import (
    causal_safe_target_v11_training_contract,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_GUARANTEED_GUARD_RAD,
    SAFE_TARGET_INNER_LOWER_HARDWARE,
    SAFE_TARGET_INNER_UPPER_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
    SAFE_TARGET_SOFT_LOWER_HARDWARE,
    SAFE_TARGET_SOFT_UPPER_HARDWARE,
    safe_target_transform_contract,
    safe_target_transform_numpy,
    safe_target_transform_torch,
)


def test_v11_zero_preserving_and_differentiable_at_zero() -> None:
    raw = torch.zeros((2, 23), dtype=torch.float32, requires_grad=True)
    safe, target = safe_target_transform_torch(raw)
    assert torch.equal(safe, torch.zeros_like(safe))
    assert torch.equal(
        target,
        torch.tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE).expand_as(target),
    )
    safe.sum().backward()
    assert torch.allclose(raw.grad, torch.ones_like(raw), atol=2.0e-6, rtol=0.0)


def test_v11_extremes_guarantee_post_bias_soft_limit_guard() -> None:
    raw = np.stack(
        (
            np.full(23, -1000.0, dtype=np.float32),
            np.full(23, 1000.0, dtype=np.float32),
        )
    )
    _, target = safe_target_transform_numpy(raw)
    low = np.asarray(SAFE_TARGET_SOFT_LOWER_HARDWARE, dtype=np.float32)
    high = np.asarray(SAFE_TARGET_SOFT_UPPER_HARDWARE, dtype=np.float32)
    # Applied target is unbiased target minus encoder bias.  Check both ends
    # of the declared +/-0.01 rad bias envelope.
    applied_low = target[0] - np.float32(0.01)
    applied_high = target[1] + np.float32(0.01)
    assert np.all(applied_low >= low + SAFE_TARGET_GUARANTEED_GUARD_RAD - 2e-7)
    assert np.all(applied_high <= high - SAFE_TARGET_GUARANTEED_GUARD_RAD + 2e-7)
    assert np.all(
        target[0]
        >= np.asarray(SAFE_TARGET_INNER_LOWER_HARDWARE, dtype=np.float32)
        - np.float32(3e-7)
    )
    assert np.all(
        target[1]
        <= np.asarray(SAFE_TARGET_INNER_UPPER_HARDWARE, dtype=np.float32)
        + np.float32(3e-7)
    )


def test_v11_torch_and_numpy_implementations_match() -> None:
    rng = np.random.default_rng(20260803)
    raw = rng.normal(size=(17, 23)).astype(np.float32)
    safe_np, target_np = safe_target_transform_numpy(raw)
    safe_t, target_t = safe_target_transform_torch(torch.from_numpy(raw))
    np.testing.assert_allclose(safe_t.numpy(), safe_np, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(target_t.numpy(), target_np, atol=2e-7, rtol=2e-7)


def test_v11_transform_clamps_raw_action_before_tanh() -> None:
    boundary = np.stack(
        (
            np.full(23, -SAFE_TARGET_RAW_ACTION_CLIP, dtype=np.float32),
            np.full(23, SAFE_TARGET_RAW_ACTION_CLIP, dtype=np.float32),
        )
    )
    beyond = boundary * np.float32(100.0)

    safe_boundary_np, target_boundary_np = safe_target_transform_numpy(boundary)
    safe_beyond_np, target_beyond_np = safe_target_transform_numpy(beyond)
    np.testing.assert_array_equal(safe_beyond_np, safe_boundary_np)
    np.testing.assert_array_equal(target_beyond_np, target_boundary_np)

    safe_boundary_t, target_boundary_t = safe_target_transform_torch(
        torch.from_numpy(boundary)
    )
    safe_beyond_t, target_beyond_t = safe_target_transform_torch(
        torch.from_numpy(beyond)
    )
    assert torch.equal(safe_beyond_t, safe_boundary_t)
    assert torch.equal(target_beyond_t, target_boundary_t)


def test_v11_previous_action_uses_applied_safe_action() -> None:
    safe = torch.arange(23, dtype=torch.float32).reshape(1, 23)
    raw = torch.full_like(safe, -99.0)
    term = SimpleNamespace(safe_native_action=safe)
    manager = SimpleNamespace(
        action=raw,
        get_term=lambda name: term if name == "joint_pos" else None,
    )
    result = padded_previous_action(SimpleNamespace(action_manager=manager))
    assert result.shape == (1, 29)
    assert not torch.any(result == -99.0)


def test_v11_recovery_reward_matches_evaluator_signal() -> None:
    target = torch.full((1, 23), 0.1, dtype=torch.float32)
    action_term = SimpleNamespace(processed_action=target)
    command = SimpleNamespace(
        anchor_pos_w=torch.tensor([[0.0, 0.0, 0.8]]),
        robot_anchor_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            encoder_bias=torch.zeros((1, 23)),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            joint_pos=torch.zeros((1, 23)),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        scene={"robot": robot},
        action_manager=SimpleNamespace(get_term=lambda _name: action_term),
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )
    metric = evaluator_aligned_recovery_metric(env)
    assert torch.allclose(metric, torch.tensor([0.15]), atol=1.0e-6, rtol=0.0)


def test_v11_contract_is_fail_closed_training_claim() -> None:
    transform = safe_target_transform_contract()
    task = causal_history_safe_target_v11_contract()
    training = causal_safe_target_v11_training_contract()
    assert transform["margin_rad"] == 0.012
    assert transform["guaranteed_post_bias_guard_rad"] == 0.0019
    assert transform["nominal_post_bias_guard_rad"] == 0.002
    assert len(transform["constants_sha256"]) == 64
    assert len(transform["formula_sha256"]) == 64
    assert task["target_transform_trained_in_loop"] is True
    assert task["post_hoc_clamp_relabel"] is False
    assert task["evaluator_aligned_rapid_recovery_weight"] == -25.0
    assert training["restart_from_model0"] is True
    assert training["planned_accepted_updates"] == 50
    assert training["deployment_ready"] is False
    assert transform["schema"] == "g1_true23_safe_target_transform_v2"
    assert transform["kind"] == "asymmetric_zero_preserving_tanh_raw_clip_v2"
    assert transform["raw_action_clip"] == 10.0
    assert transform["raw_action_clip_order"] == (
        "native_isaaclab_23_before_permutation"
    )
