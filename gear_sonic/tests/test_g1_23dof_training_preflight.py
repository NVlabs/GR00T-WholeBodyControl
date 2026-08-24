from pathlib import Path
import sys
import types

import pytest

from gear_sonic.scripts.preflight_g1_23dof_training import (
    LOW_LATENCY_EXPERIMENT,
    _audit_runtime,
    _compose_config,
    _policy_term_order,
    audit_true23_training,
)

REPO_ROOT = Path(__file__).parents[2]
INITIAL_CHECKPOINT = REPO_ROOT / "sonic_release/g1_23dof_rev_1_0_init.pt"
LOW_LATENCY_INITIAL_CHECKPOINT = (
    REPO_ROOT
    / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"
)


def test_policy_configclass_preserves_deployment_proprioception_order():
    declaration = _policy_term_order(REPO_ROOT)
    relevant = tuple(
        name
        for name in declaration
        if name in {"base_ang_vel", "joint_pos", "joint_vel", "actions", "gravity_dir"}
    )
    assert relevant == (
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
        "gravity_dir",
    )


def test_preflight_rejects_missing_checkpoint_after_config_and_asset_checks():
    cfg = _compose_config(REPO_ROOT, "does-not-exist.pt")
    report = audit_true23_training(
        cfg,
        repo_root=REPO_ROOT,
        require_runtime=False,
        require_motion_data=False,
    )
    assert report.ok is False
    assert report.errors == (
        f"checkpoint missing: {(REPO_ROOT / 'does-not-exist.pt').resolve()}",
    )
    assert report.details["action_dof"] == 23
    assert report.details["policy_proprioception_order"] == [
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
        "gravity_dir",
    ]
    assert Path(report.details["urdf_path"]).is_file()
    assert Path(report.details["mjcf_path"]).is_file()


def test_preflight_rejects_generic_previous_action_in_padded_slots():
    cfg = _compose_config(REPO_ROOT, "does-not-exist.pt")
    cfg.manager_env.observations.policy.actions.func = (
        "gear_sonic.envs.manager_env.mdp:last_action"
    )
    report = audit_true23_training(
        cfg,
        repo_root=REPO_ROOT,
        require_runtime=False,
        require_motion_data=False,
    )
    assert any("g1_23dof_padded_last_action" in error for error in report.errors)


def test_smoke_composition_forces_one_env_and_sample_motion_without_promotion():
    cfg = _compose_config(
        REPO_ROOT,
        "does-not-exist.pt",
        smoke=True,
        motion_file="sample_data/robot_filtered",
    )
    report = audit_true23_training(
        cfg,
        repo_root=REPO_ROOT,
        require_runtime=False,
        require_motion_data=False,
        smoke=True,
    )
    assert cfg.num_envs == 1
    assert cfg.use_wandb is False
    assert cfg.manager_env.config.terrain_type == "plane"
    assert cfg.manager_env.commands.motion.motion_lib_cfg.motion_file == (
        "sample_data/robot_filtered"
    )
    assert report.details["training_mode"] == "smoke"
    assert report.details["promotion_allowed"] is False
    assert not any("num_envs" in error for error in report.errors)


def test_undersized_gpu_is_error_for_full_training_but_warning_for_smoke(monkeypatch):
    import importlib.util

    import torch

    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab.__version__ = "2.3.2"
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setattr(sys, "version_info", (3, 11, 0))
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda _name: object(),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: types.SimpleNamespace(
            name="test 8 GiB GPU",
            total_memory=8 * 1024**3,
        ),
    )

    full_errors, full_warnings, full_details = [], [], {}
    _audit_runtime(
        full_errors,
        full_warnings,
        full_details,
        smoke=False,
    )
    assert any("full true23 fine-tuning/validation" in error for error in full_errors)
    assert full_warnings == []

    smoke_errors, smoke_warnings, smoke_details = [], [], {}
    _audit_runtime(
        smoke_errors,
        smoke_warnings,
        smoke_details,
        smoke=True,
    )
    assert smoke_errors == []
    assert any("promotion forbidden" in warning for warning in smoke_warnings)


@pytest.mark.skipif(
    not INITIAL_CHECKPOINT.is_file(),
    reason="downloaded true23 initialization checkpoint unavailable",
)
def test_checked_in_true23_inputs_pass_offline_except_optional_motion_download():
    cfg = _compose_config(REPO_ROOT, str(INITIAL_CHECKPOINT))
    report = audit_true23_training(
        cfg,
        repo_root=REPO_ROOT,
        require_runtime=False,
        require_motion_data=False,
    )
    assert report.errors == ()
    assert report.ok is True
    assert report.details["checkpoint_stage"] == "checkpoint_initialization"
    assert report.details["checkpoint_output_dof"] == 23
    assert len(report.details["policy_state_sha256"]) == 64


@pytest.mark.skipif(
    not LOW_LATENCY_INITIAL_CHECKPOINT.is_file(),
    reason="downloaded low-latency true23 initialization unavailable",
)
def test_low_latency_init_matches_only_low_latency_training_profile():
    low_latency_cfg = _compose_config(
        REPO_ROOT,
        str(LOW_LATENCY_INITIAL_CHECKPOINT),
        experiment=LOW_LATENCY_EXPERIMENT,
    )
    low_latency_report = audit_true23_training(
        low_latency_cfg,
        repo_root=REPO_ROOT,
        require_runtime=False,
        require_motion_data=False,
    )
    assert low_latency_report.errors == ()
    assert low_latency_report.details["reference_profile"] == (
        "released_low_latency_step1_0p02s"
    )

    normal_cfg = _compose_config(
        REPO_ROOT,
        str(LOW_LATENCY_INITIAL_CHECKPOINT),
    )
    normal_report = audit_true23_training(
        normal_cfg,
        repo_root=REPO_ROOT,
        require_runtime=False,
        require_motion_data=False,
    )
    assert any(
        "reference_profile differs" in error
        for error in normal_report.errors
    )


def test_isaac_urdf_path_is_resolved_from_module_not_process_cwd():
    source = (
        REPO_ROOT / "gear_sonic/envs/manager_env/robots/g1_23dof.py"
    ).read_text(encoding="utf-8")
    assert "Path(__file__).resolve().parents[3]" in source
    assert 'asset_path=str(ASSET_DIR / "g1_23dof_rev_1_0.urdf")' in source
