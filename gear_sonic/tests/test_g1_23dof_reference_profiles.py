from functools import lru_cache
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
import torch

from gear_sonic.scripts import init_g1_23dof_checkpoint as initializer
from gear_sonic.trl.utils.g1_23dof_checkpoint import initialize_checkpoint
from gear_sonic.utils.g1_23dof_artifact import (
    MOTION_DATASET_SOURCE_ARCHIVE,
    _contract_descriptor,
    build_training_material_evidence,
    decoder_layer_dims_for_profile,
    make_training_checkpoint_records,
    true23_reference_profile_from_config,
    validate_training_checkpoint_records,
)
from gear_sonic.utils.g1_23dof_contract import (
    APPROVED_WARM_START_RELEASES,
    ARTIFACT_SCHEMA_VERSION,
    LOW_LATENCY_INITIAL_POLICY_STATE_SHA256,
    NORMAL_INITIAL_POLICY_STATE_SHA256,
    NORMAL_RELEASE_SHA256,
    OBS_LAYOUT_PADDED_IL29,
    REFERENCE_PROFILE_LOW_LATENCY,
    REFERENCE_PROFILE_NORMAL,
    REFERENCE_PROFILES,
    make_artifact_metadata,
    reference_profile_contract,
    validate_artifact_contract,
)

TEST_MOTION_DATASET = {
    "schema_version": 1,
    "source_archive": dict(MOTION_DATASET_SOURCE_ARCHIVE),
    "processed": {
        "root_relpath": "data/motion_lib_bones_seed/robot_filtered",
        "file_count": 1,
        "total_bytes": 1,
        "manifest_sha256": "f" * 64,
    },
}


@lru_cache(maxsize=1)
def _low_latency_training_material():
    from gear_sonic.scripts.preflight_g1_23dof_training import (
        LOW_LATENCY_EXPERIMENT,
        _compose_config,
    )

    repo_root = Path(__file__).resolve().parents[2]
    config = _compose_config(
        repo_root,
        "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
        experiment=LOW_LATENCY_EXPERIMENT,
    )
    return build_training_material_evidence(config, repo_root=repo_root)


def _source_policy() -> dict[str, torch.Tensor]:
    return {
        "actor_module.encoders.teleop.module.0.weight": torch.ones(2, 267),
        "actor_module.decoders.g1_dyn.module.0.weight": torch.ones(4, 994),
        "actor_module.decoders.g1_dyn.module.12.weight": torch.ones(29, 4),
        "actor_module.decoders.g1_dyn.module.12.bias": torch.zeros(29),
        "std": torch.ones(29),
    }


def _compose(experiment: str):
    config_dir = str((Path(__file__).parents[1] / "config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        return compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/all_modes/" + experiment,
            ],
        )


def test_reference_profiles_are_immutable_and_offsets_are_exact():
    normal = reference_profile_contract(REFERENCE_PROFILE_NORMAL)
    low_latency = reference_profile_contract(REFERENCE_PROFILE_LOW_LATENCY)

    assert normal["future_frame_offsets_s"] == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]
    assert low_latency["future_frame_offsets_s"] == [
        0.0,
        0.02,
        0.04,
        0.06,
        0.08,
        0.1,
        0.12,
        0.14,
        0.16,
        0.18,
    ]
    assert normal["future_frame_step"] == 5
    assert low_latency["future_frame_step"] == 1
    with pytest.raises(TypeError):
        REFERENCE_PROFILES["fake"] = REFERENCE_PROFILES[REFERENCE_PROFILE_NORMAL]
    assert (
        APPROVED_WARM_START_RELEASES[NORMAL_RELEASE_SHA256][
            "initial_policy_state_sha256"
        ]
        == NORMAL_INITIAL_POLICY_STATE_SHA256
    )
    assert (
        APPROVED_WARM_START_RELEASES[
            initializer.LOW_LATENCY_SONIC_RELEASE_SHA256
        ]["initial_policy_state_sha256"]
        == LOW_LATENCY_INITIAL_POLICY_STATE_SHA256
    )


@pytest.mark.parametrize(
    "profile",
    (REFERENCE_PROFILE_NORMAL, REFERENCE_PROFILE_LOW_LATENCY),
)
def test_artifact_metadata_binds_exact_reference_contract(profile):
    metadata = make_artifact_metadata(
        history_length=10,
        observation_layout=OBS_LAYOUT_PADDED_IL29,
        checkpoint_stage="trained",
        reference_profile=profile,
    )
    assert metadata["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert metadata["reference_profile"] == profile
    assert metadata["reference_contract"] == reference_profile_contract(profile)
    validate_artifact_contract(
        metadata,
        decoder_input_dim=994,
        decoder_output_dim=23,
        require_deployment_ready=False,
    )

    metadata["reference_contract"]["future_frame_step"] = (
        1 if profile == REFERENCE_PROFILE_NORMAL else 5
    )
    with pytest.raises(ValueError, match="immutable temporal profile"):
        validate_artifact_contract(
            metadata,
            decoder_input_dim=994,
            decoder_output_dim=23,
            require_deployment_ready=False,
        )


def test_low_latency_initializer_and_training_records_preserve_profile():
    initialized, report = initialize_checkpoint(
        {"policy_state_dict": _source_policy()},
        reference_profile=REFERENCE_PROFILE_LOW_LATENCY,
        source_checkpoint_sha256=initializer.LOW_LATENCY_SONIC_RELEASE_SHA256,
        source_revision=initializer.LOW_LATENCY_SONIC_RELEASE_HF_REVISION,
        source_family="sonic_low_latency",
    )
    metadata = initialized["g1_23dof_metadata"]
    assert metadata["reference_profile"] == REFERENCE_PROFILE_LOW_LATENCY
    assert report["reference_profile"] == REFERENCE_PROFILE_LOW_LATENCY
    assert (
        report["source_checkpoint_sha256"]
        == initializer.LOW_LATENCY_SONIC_RELEASE_SHA256
    )
    assert report["source_revision"] == initializer.LOW_LATENCY_SONIC_RELEASE_HF_REVISION

    trained_metadata, evidence = make_training_checkpoint_records(
        global_step=50,
        history_length=10,
        observation_layout=OBS_LAYOUT_PADDED_IL29,
        policy_state_sha256="a" * 64,
        reference_profile=REFERENCE_PROFILE_LOW_LATENCY,
        source_family="sonic_low_latency",
        source_revision=initializer.LOW_LATENCY_SONIC_RELEASE_HF_REVISION,
        source_checkpoint_sha256=initializer.LOW_LATENCY_SONIC_RELEASE_SHA256,
        initial_policy_state_sha256=(
            LOW_LATENCY_INITIAL_POLICY_STATE_SHA256
        ),
        motion_dataset=TEST_MOTION_DATASET,
        training_material=_low_latency_training_material(),
    )
    validate_training_checkpoint_records(
        {
            "g1_23dof_metadata": trained_metadata,
            "g1_23dof_training_evidence": evidence,
        },
        global_step=50,
        policy_state_sha256="a" * 64,
    )
    assert evidence["reference_contract"]["horizon_s"] == 0.18


def test_normal_and_low_latency_hydra_profiles_do_not_change_policy_shapes():
    normal = _compose("sonic_g1_23dof_rev_1_0_warm_start")
    low_latency = _compose(
        "sonic_g1_23dof_rev_1_0_low_latency_warm_start"
    )

    assert true23_reference_profile_from_config(normal) == REFERENCE_PROFILE_NORMAL
    assert (
        true23_reference_profile_from_config(low_latency)
        == REFERENCE_PROFILE_LOW_LATENCY
    )
    assert normal.manager_env.commands.motion.dt_future_ref_frames == 0.1
    assert low_latency.manager_env.commands.motion.dt_future_ref_frames == 0.02
    assert normal.manager_env.commands.motion.num_future_frames == 10
    assert low_latency.manager_env.commands.motion.num_future_frames == 10
    assert _contract_descriptor(REFERENCE_PROFILE_NORMAL)["decoder_input_dim"] == 994
    assert (
        _contract_descriptor(REFERENCE_PROFILE_LOW_LATENCY)["decoder_input_dim"]
        == 994
    )
    assert decoder_layer_dims_for_profile(REFERENCE_PROFILE_NORMAL) == (
        994,
        2048,
        2048,
        1024,
        1024,
        512,
        512,
        23,
    )
    assert decoder_layer_dims_for_profile(REFERENCE_PROFILE_LOW_LATENCY) == (
        994,
        4096,
        4096,
        2048,
        2048,
        1024,
        1024,
        512,
        512,
        23,
    )
    assert list(low_latency.algo.config.actor.backbone.encoders) == ["teleop"]
    assert list(low_latency.algo.config.actor.backbone.decoders) == ["g1_dyn"]


def test_pinned_low_latency_release_cannot_be_relabelled(monkeypatch, tmp_path):
    source = tmp_path / "last.pt"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        initializer,
        "_sha256",
        lambda _path: initializer.LOW_LATENCY_SONIC_RELEASE_SHA256,
    )
    monkeypatch.setattr(
        initializer.torch,
        "load",
        lambda *_args, **_kwargs: {"policy_state_dict": _source_policy()},
    )

    _, digest, release = initializer._load_pinned_legacy_release(source)

    assert digest == initializer.LOW_LATENCY_SONIC_RELEASE_SHA256
    assert release["reference_profile"] == REFERENCE_PROFILE_LOW_LATENCY
    assert (
        release["source_revision"]
        == initializer.LOW_LATENCY_SONIC_RELEASE_HF_REVISION
    )
