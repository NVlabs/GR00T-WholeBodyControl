"""Adversarial checks for true23 retraining and warm-start lineage."""

import copy
from functools import lru_cache
from pathlib import Path

from omegaconf import open_dict
import pytest

from gear_sonic.scripts import preflight_g1_23dof_training as preflight
from gear_sonic.utils.g1_23dof_artifact import (
    MOTION_DATASET_SOURCE_ARCHIVE,
    build_training_material_evidence,
    make_training_checkpoint_records,
    validate_training_material_evidence,
)
from gear_sonic.utils.g1_23dof_contract import (
    NORMAL_INITIAL_POLICY_STATE_SHA256,
    NORMAL_RELEASE_SHA256,
    OBS_LAYOUT_PADDED_IL29,
    REFERENCE_PROFILE_LOW_LATENCY,
    REFERENCE_PROFILE_NORMAL,
    reference_profile_contract,
)


@lru_cache(maxsize=1)
def _training_material() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    config = preflight._compose_config(
        repo_root,
        "sonic_release/g1_23dof_rev_1_0_init.pt",
        experiment=preflight.EXPERIMENT,
    )
    return build_training_material_evidence(config, repo_root=repo_root)


def _training_record_args() -> dict:
    return {
        "global_step": 50,
        "history_length": 10,
        "observation_layout": OBS_LAYOUT_PADDED_IL29,
        "policy_state_sha256": "a" * 64,
        "reference_profile": REFERENCE_PROFILE_NORMAL,
        "source_family": "sonic_release",
        "source_revision": None,
        "source_checkpoint_sha256": NORMAL_RELEASE_SHA256,
        "initial_policy_state_sha256": (
            NORMAL_INITIAL_POLICY_STATE_SHA256
        ),
        "motion_dataset": {
            "schema_version": 1,
            "source_archive": dict(MOTION_DATASET_SOURCE_ARCHIVE),
            "processed": {
                "root_relpath": (
                    "data/motion_lib_bones_seed/robot_filtered"
                ),
                "file_count": 1,
                "total_bytes": 1,
                "manifest_sha256": "f" * 64,
            },
        },
        "training_material": _training_material(),
        "training_start_global_step": 0,
    }


def test_training_lineage_rejects_fewer_than_minimum_updates() -> None:
    args = _training_record_args()
    args["global_step"] = 49

    with pytest.raises(ValueError, match="fewer than 50 policy updates"):
        make_training_checkpoint_records(**args)


def test_training_lineage_rejects_unchanged_initialized_policy() -> None:
    args = _training_record_args()
    args["policy_state_sha256"] = args["initial_policy_state_sha256"]

    with pytest.raises(ValueError, match="unchanged from initialization"):
        make_training_checkpoint_records(**args)


def test_normal_release_cannot_be_relabelled_low_latency() -> None:
    args = _training_record_args()
    args["reference_profile"] = REFERENCE_PROFILE_LOW_LATENCY

    with pytest.raises(ValueError, match="not an approved warm-start release"):
        make_training_checkpoint_records(**args)


def test_training_material_rejects_reward_drift_and_snapshot_tamper() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = preflight._compose_config(
        repo_root,
        "sonic_release/g1_23dof_rev_1_0_init.pt",
        experiment=preflight.EXPERIMENT,
    )
    with open_dict(config):
        config.manager_env.rewards.feet_acc.weight = 123.0
    with pytest.raises(ValueError, match="approved true23"):
        build_training_material_evidence(config, repo_root=repo_root)

    tampered = copy.deepcopy(_training_material())
    tampered["resolved_config"]["manager_env"]["rewards"]["feet_acc"][
        "weight"
    ] = 123.0
    with pytest.raises(ValueError, match="resolved_config_sha256"):
        validate_training_material_evidence(tampered)


def test_training_material_rejects_runtime_manifest_drift() -> None:
    tampered = copy.deepcopy(_training_material())
    tampered["runtime_source"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest_sha256"):
        validate_training_material_evidence(tampered)


def test_preflight_rejects_initialization_report_not_bound_to_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "init.pt"
    checkpoint_path.write_bytes(b"safe-test-placeholder")
    policy_sha256 = "c" * 64
    config = {
        "checkpoint": checkpoint_path.name,
        "resume": False,
        "true23_warm_start": {
            "source_family": "sonic_release",
            "source_hf_revision": None,
            "source_checkpoint_sha256": NORMAL_RELEASE_SHA256,
            "initialization_only": True,
        },
        "manager_env": {
            "config": {
                "robot": {
                    "embodiment": {
                        "reference_profile": REFERENCE_PROFILE_NORMAL,
                        "reference_contract": reference_profile_contract(
                            REFERENCE_PROFILE_NORMAL
                        ),
                    }
                }
            }
        },
    }
    checkpoint = {
        "g1_23dof_metadata": {
            "reference_profile": REFERENCE_PROFILE_NORMAL,
        },
        "g1_23dof_initialization_report": {
            "source_family": "sonic_release",
            "source_revision": "wrong-revision",
            "source_checkpoint_sha256": NORMAL_RELEASE_SHA256,
            "reference_profile": REFERENCE_PROFILE_NORMAL,
            "initial_policy_state_sha256": policy_sha256,
        },
    }
    monkeypatch.setattr(
        preflight,
        "load_safe_true23_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        preflight,
        "validate_artifact_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        preflight,
        "inspect_true23_policy_state",
        lambda _checkpoint: policy_sha256,
    )
    monkeypatch.setattr(
        preflight,
        "checkpoint_stage",
        lambda _checkpoint: "checkpoint_initialization",
    )
    errors: list[str] = []
    details: dict = {}

    preflight._audit_checkpoint(config, tmp_path, errors, details)

    assert errors == [
        "checkpoint contract failed: initialization checkpoint source "
        "provenance differs from training config"
    ]
