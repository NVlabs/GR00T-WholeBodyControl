from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab import native124_selected_v2_ankle_adaptation as action_module
from gear_sonic.envs.mjlab.native124_selected_v2_ankle_adaptation import (
    SELECTED_21204_V2_ACTION_CONSTANTS_SHA256,
    Selected21204HardwareToSonicV2JointPositionAction,
    Selected21204HardwareToSonicV2JointPositionActionCfg,
    Selected21204ToSonicV2ActionCore,
    encoder_biased_hardware_target,
    selected21204_raw_hardware_to_sonic_v2_torch,
    selected21204_v2_action_contract,
    validate_selected21204_v2_action_constants,
)
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_torch,
)
from gear_sonic.utils.g1_true23_teacher_support import (
    compose_checkpoint21204_teacher_action,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "raw",
    (
        np.zeros(23, dtype=np.float32),
        np.linspace(-0.75, 0.75, 23, dtype=np.float32),
        np.random.default_rng(20260809)
        .normal(
            0.0,
            0.65,
            size=23,
        )
        .astype(np.float32),
    ),
)
def test_torch_action_chain_matches_frozen_composite_helper(raw: np.ndarray) -> None:
    expected = compose_checkpoint21204_teacher_action(
        raw,
        repository_root=REPOSITORY_ROOT,
    )
    actual = selected21204_raw_hardware_to_sonic_v2_torch(torch.from_numpy(raw).unsqueeze(0))

    np.testing.assert_array_equal(
        actual.selected_raw_action_hardware.numpy()[0],
        expected.teacher_raw_action_hardware,
    )
    np.testing.assert_allclose(
        actual.candidate_target_hardware.numpy()[0],
        expected.teacher_candidate_target_hardware,
        atol=1.0e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        actual.plain_sonic_raw_action_native.numpy()[0],
        expected.teacher_action_native,
        atol=2.0e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        actual.safe_native_action.numpy()[0],
        expected.teacher_applied_safe_action_native,
        atol=2.0e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        actual.final_target_hardware.numpy()[0],
        expected.teacher_target_hardware,
        atol=2.0e-7,
        rtol=0.0,
    )
    expected_effective = (
        expected.teacher_target_hardware - action_module.SELECTED_21204_HOME_Q_HARDWARE
    ) / action_module.SELECTED_21204_ACTION_SCALE_HARDWARE
    np.testing.assert_allclose(
        actual.effective_selected_raw_action_hardware.numpy()[0],
        expected_effective,
        atol=5.0e-7,
        rtol=0.0,
    )
    assert not torch.any(actual.raw_clip_mask_native)


def test_v2_executes_once_and_final_target_is_not_double_transformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = Selected21204ToSonicV2ActionCore(num_envs=1, device="cpu")
    raw = torch.linspace(-0.4, 0.8, 23, dtype=torch.float32).unsqueeze(0)
    real_transform = action_module.safe_target_transform_torch
    call_count = 0

    def counted(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal call_count
        call_count += 1
        return real_transform(value)

    monkeypatch.setattr(action_module, "safe_target_transform_torch", counted)
    core.process_actions(raw)
    assert call_count == 1

    diagnostics = core.diagnostics
    safe_once, target_once = real_transform(diagnostics.plain_sonic_raw_action_native)
    _, target_twice = real_transform(safe_once)
    assert torch.allclose(
        diagnostics.final_target_hardware,
        target_once,
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.allclose(
        diagnostics.final_target_hardware,
        target_twice,
        atol=1.0e-6,
        rtol=0.0,
    )


def test_action_core_preserves_action_manager_raw_hardware_semantics() -> None:
    core = Selected21204ToSonicV2ActionCore(num_envs=2, device="cpu")
    raw = torch.stack(
        (
            torch.arange(23, dtype=torch.float32) / 20.0,
            -torch.arange(23, dtype=torch.float32) / 30.0,
        )
    )
    core.process_actions(raw)

    assert torch.equal(core.raw_action, raw)
    diagnostics = core.diagnostics
    assert torch.equal(diagnostics.selected_raw_action_hardware, raw)
    assert not torch.equal(diagnostics.plain_sonic_raw_action_native, raw)

    diagnostics.selected_raw_action_hardware.fill_(999.0)
    diagnostics.final_target_hardware.fill_(999.0)
    assert torch.equal(core.raw_action, raw)
    assert not torch.any(core.processed_action == 999.0)


def test_effective_selected_prior_reconstructs_applied_target_not_actor_raw() -> None:
    core = Selected21204ToSonicV2ActionCore(num_envs=2, device="cpu")
    raw = torch.stack(
        (
            torch.zeros(23, dtype=torch.float32),
            torch.linspace(-0.8, 0.9, 23, dtype=torch.float32),
        )
    )
    core.process_actions(raw)
    effective = core.effective_selected_raw_action_hardware
    selected_home = torch.as_tensor(
        action_module.SELECTED_21204_HOME_Q_HARDWARE,
        dtype=torch.float32,
    )
    selected_scale = torch.as_tensor(
        action_module.SELECTED_21204_ACTION_SCALE_HARDWARE,
        dtype=torch.float32,
    )
    expected_effective = (core.processed_action - selected_home) / selected_scale
    reconstructed = selected_home + selected_scale * effective

    assert torch.isfinite(effective).all()
    assert torch.equal(effective, expected_effective)
    assert torch.allclose(
        reconstructed,
        core.processed_action,
        atol=1.0e-7,
        rtol=0.0,
    )
    assert not torch.allclose(effective, raw, atol=1.0e-6, rtol=0.0)
    assert torch.equal(core.raw_action, raw)
    assert torch.equal(
        core.diagnostics.effective_selected_raw_action_hardware,
        effective,
    )


def test_hardware_to_native_order_is_explicit_before_v2() -> None:
    raw = torch.linspace(-1.0, 1.0, 23, dtype=torch.float32).unsqueeze(0)
    diagnostics = selected21204_raw_hardware_to_sonic_v2_torch(raw)

    candidate = (
        torch.as_tensor(
            action_module.SELECTED_21204_HOME_Q_HARDWARE,
            dtype=torch.float32,
        )
        + torch.as_tensor(
            action_module.SELECTED_21204_ACTION_SCALE_HARDWARE,
            dtype=torch.float32,
        )
        * raw[0]
    )
    plain_hardware = (
        candidate - torch.as_tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=torch.float32)
    ) / torch.as_tensor(HARDWARE_23_ACTION_SCALE, dtype=torch.float32)
    expected_native = plain_hardware[torch.as_tensor(MUJOCO_TO_ISAACLAB_DOF, dtype=torch.long)]
    assert torch.equal(
        diagnostics.plain_sonic_raw_action_native[0],
        expected_native,
    )
    assert torch.equal(
        diagnostics.plain_sonic_raw_action_native[
            0,
            torch.as_tensor(ISAACLAB_TO_MUJOCO_DOF, dtype=torch.long),
        ],
        plain_hardware,
    )


def test_encoder_bias_is_applied_once_after_final_hardware_target() -> None:
    core = Selected21204ToSonicV2ActionCore(num_envs=2, device="cpu")
    raw = torch.stack(
        (
            torch.linspace(-0.5, 0.5, 23, dtype=torch.float32),
            torch.linspace(0.8, -0.2, 23, dtype=torch.float32),
        )
    )
    bias = torch.stack(
        (
            torch.linspace(-0.01, 0.01, 23, dtype=torch.float32),
            torch.linspace(0.01, -0.01, 23, dtype=torch.float32),
        )
    )
    core.process_actions(raw)

    applied = core.applied_target(bias)
    expected = core.processed_action - bias
    assert torch.equal(applied, expected)
    assert torch.equal(
        encoder_biased_hardware_target(core.processed_action, bias),
        expected,
    )


@pytest.mark.parametrize(
    "bad",
    (
        torch.zeros(23, dtype=torch.float32),
        torch.zeros((1, 22), dtype=torch.float32),
        torch.zeros((1, 23), dtype=torch.float64),
        torch.full((1, 23), float("nan"), dtype=torch.float32),
        torch.full((1, 23), float("inf"), dtype=torch.float32),
    ),
)
def test_action_chain_fails_closed_on_shape_dtype_or_nonfinite(
    bad: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        selected21204_raw_hardware_to_sonic_v2_torch(bad)


def test_core_rejects_wrong_batch_and_exposes_actual_clip_mask() -> None:
    core = Selected21204ToSonicV2ActionCore(num_envs=2, device="cpu")
    with pytest.raises(ValueError, match=r"shape \[2,23\]"):
        core.process_actions(torch.zeros((1, 23), dtype=torch.float32))

    large = torch.full((2, 23), 100.0, dtype=torch.float32)
    core.process_actions(large)
    diagnostics = core.diagnostics
    assert torch.any(diagnostics.raw_clip_mask_native)
    expected_safe, expected_target = safe_target_transform_torch(diagnostics.plain_sonic_raw_action_native)
    assert torch.equal(diagnostics.safe_native_action, expected_safe)
    assert torch.equal(diagnostics.final_target_hardware, expected_target)


def test_clip_mask_includes_strict_support_boundary() -> None:
    plain_native = torch.zeros((3, 23), dtype=torch.float32)
    plain_native[0, 0] = 10.0
    plain_native[1, 0] = -10.0
    plain_native[2, 0] = torch.nextafter(
        torch.tensor(10.0, dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
    )
    plain_hardware = plain_native[:, torch.as_tensor(ISAACLAB_TO_MUJOCO_DOF, dtype=torch.long)]
    candidate = (
        torch.as_tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=torch.float32)
        + torch.as_tensor(HARDWARE_23_ACTION_SCALE, dtype=torch.float32) * plain_hardware
    )
    selected_raw_hardware = (
        candidate
        - torch.as_tensor(
            action_module.SELECTED_21204_HOME_Q_HARDWARE,
            dtype=torch.float32,
        )
    ) / torch.as_tensor(
        action_module.SELECTED_21204_ACTION_SCALE_HARDWARE,
        dtype=torch.float32,
    )
    diagnostics = selected21204_raw_hardware_to_sonic_v2_torch(selected_raw_hardware)

    # Reconstructed boundary can move by one float32 ULP through affine
    # inversion.  Prove mask's exact threshold directly as well as end-to-end.
    reconstructed = diagnostics.plain_sonic_raw_action_native[:, 0]
    assert reconstructed[0] >= 10.0
    assert reconstructed[1] <= -10.0
    assert abs(float(reconstructed[2])) < 10.0
    assert diagnostics.raw_clip_mask_native[:, 0].tolist() == [True, True, False]


def test_constant_and_joint_order_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert validate_selected21204_v2_action_constants() == SELECTED_21204_V2_ACTION_CONSTANTS_SHA256

    drifted_scale = action_module.SELECTED_21204_ACTION_SCALE_HARDWARE.copy()
    drifted_scale[4] += np.float32(0.001)
    monkeypatch.setattr(
        action_module,
        "SELECTED_21204_ACTION_SCALE_HARDWARE",
        drifted_scale,
    )
    with pytest.raises(RuntimeError, match="constants SHA256 mismatch"):
        validate_selected21204_v2_action_constants()
    monkeypatch.undo()

    drifted_order = list(action_module.ISAACLAB_TO_MUJOCO_DOF)
    drifted_order[0], drifted_order[1] = drifted_order[1], drifted_order[0]
    monkeypatch.setattr(
        action_module,
        "ISAACLAB_TO_MUJOCO_DOF",
        tuple(drifted_order),
    )
    with pytest.raises(RuntimeError, match="permutations are not inverse"):
        validate_selected21204_v2_action_constants()


def test_contract_records_single_v2_and_non_deployment_boundary() -> None:
    contract = selected21204_v2_action_contract()
    assert contract["safe_target_v2_application_count"] == 1
    assert contract["action_manager_raw_preserved"] is True
    assert contract["sonic_previous_action_history"] == ("applied_safe_native_action")
    assert contract["native124_previous_action_history"] == ("effective_selected_raw_action_hardware")
    assert contract["hardware_authorized"] is False
    assert contract["deployment_ready"] is False


def test_real_mjlab_native124_stock_home_constructs_selected_v2_action_term() -> None:
    """WSL-only construction proof; creates environment but takes no step."""

    if action_module._MJLAB_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab runtime unavailable")

    from mjlab.envs import ManagerBasedRlEnv

    from gear_sonic.envs.mjlab.native124_multi_motion import (
        make_native124_multi_motion_env_cfg,
    )

    sidecar = REPOSITORY_ROOT / "artifacts/g1_native124_multimotion/novel_tired_one_leg_jump/corpus.spans.json"
    cfg = make_native124_multi_motion_env_cfg(
        sidecar,
        enable_pushes=False,
    )
    cfg.scene.num_envs = 1
    cfg.seed = 20260809
    cfg.actions["joint_pos"] = Selected21204HardwareToSonicV2JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
    )

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        term = env.action_manager.get_term("joint_pos")
        assert type(term) is Selected21204HardwareToSonicV2JointPositionAction
        robot = env.scene["robot"]
        target_ids, target_names = robot.find_joints_by_actuator_names((".*",))
        assert tuple(target_names) == action_module.HARDWARE_23_JOINT_NAMES
        assert term._target_ids.tolist() == list(target_ids)  # noqa: SLF001

        stock_default = robot.data.default_joint_pos[:, term._target_ids]  # noqa: SLF001
        selected_home = torch.as_tensor(
            action_module.SELECTED_21204_HOME_Q_HARDWARE,
            dtype=torch.float32,
            device=stock_default.device,
        )
        sonic_default = torch.as_tensor(
            SAFE_TARGET_DEFAULT_Q_HARDWARE,
            dtype=torch.float32,
            device=stock_default.device,
        )
        assert torch.allclose(
            stock_default,
            selected_home.expand_as(stock_default),
            atol=1.0e-6,
            rtol=0.0,
        )
        assert not torch.allclose(
            stock_default,
            sonic_default.expand_as(stock_default),
            atol=1.0e-6,
            rtol=0.0,
        )
        assert torch.equal(
            term.raw_action,
            torch.zeros((1, 23), dtype=torch.float32, device=term.device),
        )
        assert int(env.common_step_counter) == 0
    finally:
        env.close()
