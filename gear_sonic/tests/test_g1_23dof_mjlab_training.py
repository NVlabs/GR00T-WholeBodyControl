"""Fail-closed tests for exact-policy true23 MJLab training checkpoints."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.scripts import inspect_g1_23dof_mjlab_training as cli
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    load_safe_true23_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import MINIMUM_TRAINING_UPDATES
from gear_sonic.utils.g1_23dof_mjlab_training import (
    MJLAB_CHECKPOINT_HEADER,
    MJLAB_COMMIT,
    MJLAB_VERSION,
    MUJOCO_VERSION,
    MUJOCO_WARP_COMMIT,
    MUJOCO_WARP_VERSION,
    PYTHON_VERSION_POLICY,
    TORCH_VERSION,
    UNITREE_RL_MJLAB_COMMIT,
    WARP_LANG_VERSION,
    build_file_manifest,
    build_mjlab_training_checkpoint,
    build_mjlab_training_lineage,
    load_mjlab_training_checkpoint,
    restore_mjlab_training_checkpoint,
    save_mjlab_training_checkpoint,
    validate_mjlab_training_lineage,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INITIAL_CHECKPOINT = (
    _REPO_ROOT / "sonic_release/g1_23dof_rev_1_0_init.pt"
)


class _StateModule:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self._state = {
            key: value.detach().clone() for key, value in state.items()
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state

    def load_state_dict(
        self,
        state: dict[str, torch.Tensor],
        *,
        strict: bool,
    ) -> None:
        assert strict is True
        if set(state) != set(self._state):
            raise ValueError("state keys differ")
        self._state = {
            key: value.detach().clone() for key, value in state.items()
        }


class _OptimizerSink:
    def __init__(self) -> None:
        self.loaded_state: dict | None = None

    def load_state_dict(self, state: dict) -> None:
        self.loaded_state = state


@pytest.fixture(scope="module")
def training_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict:
    temp_dir = tmp_path_factory.mktemp("mjlab_lineage")
    source_file = temp_dir / "exact_runner.py"
    asset_file = temp_dir / "g1.xml"
    dataset_file = temp_dir / "motion.npz"
    source_file.write_text("EXACT_TRUE23 = True\n", encoding="utf-8")
    asset_file.write_text("<mujoco/>\n", encoding="utf-8")
    dataset_file.write_bytes(b"exact-motion-dataset")
    source_manifest = build_file_manifest(
        {"runner/exact_runner.py": source_file},
        kind="source_files",
    )
    asset_manifest = build_file_manifest(
        {"assets/g1_23dof.xml": asset_file},
        kind="robot_assets",
    )
    dataset_manifest = build_file_manifest(
        {"motions/true23.npz": dataset_file},
        kind="motion_dataset",
    )
    lineage = build_mjlab_training_lineage(
        _INITIAL_CHECKPOINT,
        resolved_config={
            "task": "Unitree-G1-23Dof-Tracking-No-State-Estimation",
            "num_envs": 128,
            "headless": True,
            "output_dim": 23,
        },
        source_manifest=source_manifest,
        asset_manifest=asset_manifest,
        dataset_manifest=dataset_manifest,
    )
    initialization = load_safe_true23_checkpoint(_INITIAL_CHECKPOINT)
    trained_policy = dict(initialization["policy_state_dict"])
    output_bias_key = "actor_module.decoders.g1_dyn.module.12.bias"
    trained_policy[output_bias_key] = trained_policy[output_bias_key].clone()
    trained_policy[output_bias_key][0] += 1.0e-4
    critic_state = {
        "obs_normalizer.count": torch.tensor(0, dtype=torch.long),
        "value.0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "value.0.bias": torch.zeros(3),
        "value.2.weight": torch.ones(1, 3),
        "value.2.bias": torch.zeros(1),
    }
    optimizer_state = {
        "state": {
            0: {
                "step": torch.tensor(50.0),
                "exp_avg": torch.zeros(1),
            }
        },
        "param_groups": [
            {
                "params": list(range(len(trained_policy))),
                "lr": 3.0e-4,
                "eps": 1.0e-5,
            }
        ],
    }
    resume_path = temp_dir / "exact_true23_resume.pt"
    trainer_state = {
        "completed_update_count": MINIMUM_TRAINING_UPDATES,
        "current_learning_iteration": MINIMUM_TRAINING_UPDATES,
        "env_common_step_counter": 153_600,
    }
    save_mjlab_training_checkpoint(
        resume_path,
        policy_state_dict=trained_policy,
        critic_state_dict=critic_state,
        optimizer_state_dict=optimizer_state,
        update_count=MINIMUM_TRAINING_UPDATES,
        trainer_state=trainer_state,
        lineage=lineage,
    )
    return {
        "lineage": lineage,
        "initial_policy": initialization["policy_state_dict"],
        "trained_policy": trained_policy,
        "critic_state": critic_state,
        "optimizer_state": optimizer_state,
        "trainer_state": trainer_state,
        "resume_path": resume_path,
    }


def test_lineage_pins_runtime_and_exact_joint_materials(
    training_bundle: dict,
) -> None:
    lineage = training_bundle["lineage"]
    components = lineage["runtime_pins"]["components"]
    assert components["unitree_rl_mjlab"]["commit"] == (
        UNITREE_RL_MJLAB_COMMIT
    )
    assert components["mjlab"] == {
        "repository": "https://github.com/mujocolab/mjlab.git",
        "version": MJLAB_VERSION,
        "commit": MJLAB_COMMIT,
    }
    assert components["mujoco_warp"]["version"] == MUJOCO_WARP_VERSION
    assert components["mujoco_warp"]["commit"] == MUJOCO_WARP_COMMIT
    assert components["mujoco_warp"]["commit_authoritative"] is True
    assert components["mujoco_warp"]["unitree_setup_declared_version"] == (
        "3.5.0"
    )
    stable = components["stable_training_environment"]
    assert stable == {
        "python_version_policy": PYTHON_VERSION_POLICY,
        "torch": TORCH_VERSION,
        "mujoco": MUJOCO_VERSION,
        "warp_lang": WARP_LANG_VERSION,
        "resolution": (
            "stable_pypi_workaround_for_unavailable_official_lock_dev_build"
        ),
        "fully_frozen_official_mjlab_lock": False,
    }
    assert lineage["materials"]["joint_semantics"]["payload"][
        "hardware_joint_ids"
    ] == list(range(13)) + list(range(15, 20)) + list(range(22, 27))
    assert lineage["materials"]["joint_semantics"]["payload"][
        "canonical_il29_missing_indices"
    ] == [5, 8, 25, 26, 27, 28]
    assert lineage["stock_rsl_posthoc_conversion"] is False
    assert lineage["deployment_ready"] is False
    assert lineage["promotion_eligible"] is False


def test_resume_is_weights_only_safe_and_never_promoted(
    training_bundle: dict,
) -> None:
    path = training_bundle["resume_path"]
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw[MJLAB_CHECKPOINT_HEADER]["weights_only_safe"] is True
    checkpoint = load_mjlab_training_checkpoint(
        path,
        expected_lineage_sha256=training_bundle["lineage"][
            "lineage_sha256"
        ],
        minimum_update_count=MINIMUM_TRAINING_UPDATES,
    )
    assert checkpoint["update_count"] == MINIMUM_TRAINING_UPDATES
    assert checkpoint["optimizer_state_dict"]["state"][0]["step"].item() == 50
    assert checkpoint["trainer_state"] == training_bundle["trainer_state"]
    assert checkpoint["critic_state_sha256"]
    assert (
        checkpoint["critic_state_dict"]["obs_normalizer.count"].ndim == 0
    )
    assert checkpoint["critic_state_dict"][
        "obs_normalizer.count"
    ].dtype is torch.int64
    assert checkpoint["training_gate"][
        "simulation_candidate_review_allowed"
    ] is True
    assert checkpoint["training_gate"]["deployment_ready"] is False
    assert checkpoint["training_gate"]["promotion_eligible"] is False


def test_update_gate_requires_50_real_updates_and_changed_policy(
    training_bundle: dict,
) -> None:
    before_threshold = build_mjlab_training_checkpoint(
        policy_state_dict=training_bundle["trained_policy"],
        critic_state_dict=training_bundle["critic_state"],
        optimizer_state_dict=training_bundle["optimizer_state"],
        update_count=MINIMUM_TRAINING_UPDATES - 1,
        trainer_state={
            "completed_update_count": MINIMUM_TRAINING_UPDATES - 1,
            "current_learning_iteration": MINIMUM_TRAINING_UPDATES - 1,
            "env_common_step_counter": 150_528,
        },
        lineage=training_bundle["lineage"],
    )
    assert before_threshold["training_gate"]["minimum_updates_reached"] is False
    assert before_threshold["training_gate"][
        "simulation_candidate_review_allowed"
    ] is False

    unchanged = build_mjlab_training_checkpoint(
        policy_state_dict=training_bundle["initial_policy"],
        critic_state_dict=training_bundle["critic_state"],
        optimizer_state_dict=training_bundle["optimizer_state"],
        update_count=MINIMUM_TRAINING_UPDATES,
        trainer_state=training_bundle["trainer_state"],
        lineage=training_bundle["lineage"],
    )
    assert unchanged["training_gate"][
        "policy_changed_from_initialization"
    ] is False
    assert unchanged["training_gate"][
        "simulation_candidate_review_allowed"
    ] is False


def test_stock_or_partial_policy_cannot_be_relabelled(
    training_bundle: dict,
) -> None:
    stock_actor = {
        "mlp.0.weight": torch.zeros(128, 64),
        "mlp.0.bias": torch.zeros(128),
        "mlp.2.weight": torch.zeros(23, 128),
        "mlp.2.bias": torch.zeros(23),
    }
    with pytest.raises(ValueError, match="exact warm-start"):
        build_mjlab_training_checkpoint(
            policy_state_dict=stock_actor,
            critic_state_dict=training_bundle["critic_state"],
            optimizer_state_dict=training_bundle["optimizer_state"],
            update_count=50,
            trainer_state=training_bundle["trainer_state"],
            lineage=training_bundle["lineage"],
        )

    partial = dict(training_bundle["trained_policy"])
    partial.pop("actor_module.encoders.teleop.module.0.bias")
    with pytest.raises(ValueError, match="exact warm-start"):
        build_mjlab_training_checkpoint(
            policy_state_dict=partial,
            critic_state_dict=training_bundle["critic_state"],
            optimizer_state_dict=training_bundle["optimizer_state"],
            update_count=50,
            trainer_state=training_bundle["trainer_state"],
            lineage=training_bundle["lineage"],
        )


def test_lineage_pin_or_material_tamper_is_rejected(
    training_bundle: dict,
) -> None:
    tampered_pin = copy.deepcopy(training_bundle["lineage"])
    tampered_pin["runtime_pins"]["components"]["mjlab"]["commit"] = "0" * 40
    with pytest.raises(ValueError, match="runtime pins"):
        validate_mjlab_training_lineage(tampered_pin)

    tampered_dataset = copy.deepcopy(training_bundle["lineage"])
    tampered_dataset["materials"]["motion_dataset"]["files"][0][
        "sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="aggregate"):
        validate_mjlab_training_lineage(tampered_dataset)


def test_optimizer_state_must_remain_weights_only_safe(
    training_bundle: dict,
) -> None:
    unsafe_optimizer = {
        "state": {0: {"unsafe": SimpleNamespace(value=1)}},
        "param_groups": [{"params": [0]}],
    }
    with pytest.raises(ValueError, match="weights-only-unsafe"):
        build_mjlab_training_checkpoint(
            policy_state_dict=training_bundle["trained_policy"],
            critic_state_dict=training_bundle["critic_state"],
            optimizer_state_dict=unsafe_optimizer,
            update_count=50,
            trainer_state=training_bundle["trainer_state"],
            lineage=training_bundle["lineage"],
        )


def test_critic_and_trainer_counters_are_required(
    training_bundle: dict,
) -> None:
    with pytest.raises(ValueError, match="critic_state_dict"):
        build_mjlab_training_checkpoint(
            policy_state_dict=training_bundle["trained_policy"],
            critic_state_dict={},
            optimizer_state_dict=training_bundle["optimizer_state"],
            update_count=50,
            trainer_state=training_bundle["trainer_state"],
            lineage=training_bundle["lineage"],
        )

    nonfinite_critic = dict(training_bundle["critic_state"])
    nonfinite_critic["value.2.bias"] = torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="NaN or infinity"):
        build_mjlab_training_checkpoint(
            policy_state_dict=training_bundle["trained_policy"],
            critic_state_dict=nonfinite_critic,
            optimizer_state_dict=training_bundle["optimizer_state"],
            update_count=50,
            trainer_state=training_bundle["trainer_state"],
            lineage=training_bundle["lineage"],
        )

    mismatched_state = dict(training_bundle["trainer_state"])
    mismatched_state["completed_update_count"] = 49
    with pytest.raises(ValueError, match="must equal"):
        build_mjlab_training_checkpoint(
            policy_state_dict=training_bundle["trained_policy"],
            critic_state_dict=training_bundle["critic_state"],
            optimizer_state_dict=training_bundle["optimizer_state"],
            update_count=50,
            trainer_state=mismatched_state,
            lineage=training_bundle["lineage"],
        )


def test_restore_recovers_actor_critic_optimizer_and_trainer_state(
    training_bundle: dict,
) -> None:
    actor = _StateModule(training_bundle["initial_policy"])
    blank_critic = {
        key: torch.zeros_like(value)
        for key, value in training_bundle["critic_state"].items()
    }
    critic = _StateModule(blank_critic)
    optimizer = _OptimizerSink()

    trainer_state = restore_mjlab_training_checkpoint(
        training_bundle["resume_path"],
        policy_module=actor,
        critic_module=critic,
        optimizer=optimizer,
        expected_lineage=training_bundle["lineage"],
        minimum_update_count=50,
    )

    assert trainer_state == training_bundle["trainer_state"]
    assert optimizer.loaded_state is not None
    for key, expected in training_bundle["trained_policy"].items():
        assert torch.equal(actor.state_dict()[key], expected)
    for key, expected in training_bundle["critic_state"].items():
        assert torch.equal(critic.state_dict()[key], expected)


def test_resume_rollback_and_promotion_filename_are_rejected(
    training_bundle: dict,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="older than required"):
        load_mjlab_training_checkpoint(
            training_bundle["resume_path"],
            minimum_update_count=MINIMUM_TRAINING_UPDATES + 1,
        )
    with pytest.raises(ValueError, match="promotion filename"):
        save_mjlab_training_checkpoint(
            tmp_path / "model.promotion.pt",
            policy_state_dict=training_bundle["trained_policy"],
            critic_state_dict=training_bundle["critic_state"],
            optimizer_state_dict=training_bundle["optimizer_state"],
            update_count=50,
            trainer_state=training_bundle["trainer_state"],
            lineage=training_bundle["lineage"],
        )


def test_inspection_cli_reports_training_only_boundary(
    training_bundle: dict,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        cli.main(
            [
                "--checkpoint",
                str(training_bundle["resume_path"]),
                "--minimum-update-count",
                "50",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "checkpoint role: training_resume_only" in output
    assert "simulation candidate review allowed: true" in output
    assert "deployment ready: false" in output
    assert "promotion eligible: false" in output
