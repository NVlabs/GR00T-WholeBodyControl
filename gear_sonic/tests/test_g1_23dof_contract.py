from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import torch

from gear_sonic.trl.utils.g1_23dof_checkpoint import (
    initialize_checkpoint,
    initialize_policy_state_dict,
)
from gear_sonic.utils.g1_23dof_contract import (
    DECODER_OUTPUT_LAYOUT,
    EXCLUDED_HARDWARE_JOINT_IDS,
    HARDWARE_23_ACTION_SCALE,
    HARDWARE_23_JOINT_NAMES,
    HARDWARE_JOINT_IDS,
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_ACTION_SCALE,
    NATIVE_IL23_JOINT_NAMES,
    NATIVE_IL23_TO_CANONICAL_IL29,
    OBS_LAYOUT_CHECKPOINT_INIT_29,
    OBS_LAYOUT_NATIVE_23,
    SIM_VALIDATION_SCHEMA_VERSION,
    SOURCE_IL29_EXCLUDED_INDICES,
    SOURCE_IL29_JOINT_NAMES,
    decoder_input_keep_indices_29_to_23,
    decoder_shape,
    expand_il23_to_il29,
    make_artifact_metadata,
    validate_artifact_contract,
)


def _source_policy(history_length=10):
    input_dim = decoder_shape(history_length, OBS_LAYOUT_CHECKPOINT_INIT_29).input_dim
    return {
        "actor_module.encoders.teleop.module.0.weight": torch.ones(2, 267),
        "actor_module.encoders.g1.module.0.weight": torch.ones(2, 640),
        "actor_module.decoders.g1_kin.module.8.weight": torch.ones(640, 2),
        "actor_module.decoders.g1_dyn.module.0.weight": torch.arange(input_dim, dtype=torch.float32).repeat(4, 1),
        "actor_module.decoders.g1_dyn.module.12.weight": torch.arange(29 * 4, dtype=torch.float32).reshape(29, 4),
        "actor_module.decoders.g1_dyn.module.12.bias": torch.arange(29, dtype=torch.float32),
        "std": torch.arange(29, dtype=torch.float32),
    }


def _validated_evidence():
    digest = "a" * 64
    evidence = {
        "schema_version": SIM_VALIDATION_SCHEMA_VERSION,
        "computed_pass": True,
        "report_sha256": digest,
        "report_payload_sha256": digest,
        "checkpoint_sha256": digest,
        "required_scenarios": ["nominal", "disturbance_50", "disturbance_100"],
        "run_count": 3,
    }
    hashes = {
        key: digest
        for key in (
            "checkpoint_sha256",
            "policy_state_sha256",
            "encoder_state_sha256",
            "decoder_state_sha256",
            "encoder_onnx_sha256",
            "decoder_onnx_sha256",
            "sim_report_sha256",
            "sim_report_payload_sha256",
            "contract_sha256",
            "robot_asset_sha256",
            "robot_config_sha256",
            "sim_config_sha256",
            "encoder_config_sha256",
            "decoder_config_sha256",
            "policy_config_sha256",
            "encoder_embedded_metadata_sha256",
            "decoder_embedded_metadata_sha256",
        )
    }
    return evidence, hashes


def test_joint_id_sets_and_compact_permutations_are_exact():
    assert HARDWARE_JOINT_IDS == tuple(range(13)) + tuple(range(15, 20)) + tuple(range(22, 27))
    assert EXCLUDED_HARDWARE_JOINT_IDS == (13, 14, 20, 21, 27, 28)
    assert len(set(HARDWARE_JOINT_IDS)) == 23
    assert sorted(HARDWARE_JOINT_IDS + EXCLUDED_HARDWARE_JOINT_IDS) == list(range(29))
    assert [ISAACLAB_TO_MUJOCO_DOF[i] for i in MUJOCO_TO_ISAACLAB_DOF] == list(range(23))
    assert [MUJOCO_TO_ISAACLAB_DOF[i] for i in ISAACLAB_TO_MUJOCO_DOF] == list(range(23))
    assert (
        tuple(SOURCE_IL29_JOINT_NAMES[index] for index in NATIVE_IL23_TO_CANONICAL_IL29) == NATIVE_IL23_JOINT_NAMES
    )
    assert tuple(NATIVE_IL23_ACTION_SCALE[index] for index in ISAACLAB_TO_MUJOCO_DOF) == HARDWARE_23_ACTION_SCALE
    assert tuple(NATIVE_IL23_JOINT_NAMES[index] for index in ISAACLAB_TO_MUJOCO_DOF) == HARDWARE_23_JOINT_NAMES
    assert HARDWARE_23_ACTION_SCALE[4:6] == (0.44, 0.44)


def test_checkpoint_padding_expands_only_excluded_slots():
    compact = tuple(float(index + 1) for index in range(23))
    expanded = expand_il23_to_il29(compact, excluded_values=(101, 102, 103, 104, 105, 106))
    assert tuple(expanded[index] for index in NATIVE_IL23_TO_CANONICAL_IL29) == compact
    assert tuple(expanded[index] for index in SOURCE_IL29_EXCLUDED_INDICES) == (
        101,
        102,
        103,
        104,
        105,
        106,
    )


def test_decoder_shapes_cover_padded_warm_start_and_native_layouts():
    assert decoder_shape(10, OBS_LAYOUT_CHECKPOINT_INIT_29).input_dim == 994
    assert decoder_shape(4, OBS_LAYOUT_CHECKPOINT_INIT_29).input_dim == 436
    assert decoder_shape(10, OBS_LAYOUT_NATIVE_23).input_dim == 814
    assert decoder_shape(4, OBS_LAYOUT_NATIVE_23).input_dim == 364
    keep = decoder_input_keep_indices_29_to_23(10)
    assert len(keep) == 814
    assert len(set(keep)) == len(keep)
    assert max(keep) < 994


def test_warm_start_keeps_29_slot_input_but_creates_true_23_head():
    initialized, report = initialize_policy_state_dict(_source_policy())
    assert initialized["actor_module.decoders.g1_dyn.module.0.weight"].shape == (4, 994)
    assert initialized["actor_module.decoders.g1_dyn.module.12.weight"].shape == (23, 4)
    assert initialized["actor_module.decoders.g1_dyn.module.12.bias"].tolist() == list(
        NATIVE_IL23_TO_CANONICAL_IL29
    )
    assert initialized["std"].tolist() == list(NATIVE_IL23_TO_CANONICAL_IL29)
    assert not any("encoders.g1." in key for key in initialized)
    assert not any("decoders.g1_kin." in key for key in initialized)
    assert report["initialization_only"] is True


def test_native_initializer_selects_semantic_decoder_columns():
    source = _source_policy()
    source_input = source["actor_module.decoders.g1_dyn.module.0.weight"].clone()
    initialized, _ = initialize_policy_state_dict(source, target_observation_layout=OBS_LAYOUT_NATIVE_23)
    keep = decoder_input_keep_indices_29_to_23(10)
    expected = source_input.index_select(1, torch.tensor(keep))
    assert initialized["actor_module.decoders.g1_dyn.module.0.weight"].shape == (4, 814)
    assert torch.equal(initialized["actor_module.decoders.g1_dyn.module.0.weight"], expected)


def test_checkpoint_initializer_rejects_unknown_policy_branches():
    source = _source_policy()
    source["actor_module.decoders.future_robot.module.0.weight"] = torch.ones(2, 2)
    with pytest.raises(ValueError, match="outside the true23 transfer allowlist"):
        initialize_policy_state_dict(source)


def test_checkpoint_initializer_rejects_semantically_ambiguous_pre23_head():
    source = _source_policy()
    native_rows = torch.tensor(NATIVE_IL23_TO_CANONICAL_IL29)
    for key in (
        "actor_module.decoders.g1_dyn.module.12.weight",
        "actor_module.decoders.g1_dyn.module.12.bias",
        "std",
    ):
        source[key] = source[key].index_select(0, native_rows)
    with pytest.raises(ValueError, match="released 29-row"):
        initialize_policy_state_dict(source)


def test_checkpoint_initializer_drops_shape_coupled_training_state():
    checkpoint = {
        "policy_state_dict": _source_policy(),
        "value_state_dict": {"bad": torch.ones(1)},
        "optimizer_state_dict": {"bad": 1},
        "lr_scheduler_state_dict": {"bad": 1},
        "env_state_dict": {"bad": 1},
        "state": object(),
        "args": object(),
        "unknown_resume_state": {"must": "not survive"},
    }
    initialized, _ = initialize_checkpoint(checkpoint)
    for key in (
        "value_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "env_state_dict",
        "state",
        "args",
        "unknown_resume_state",
    ):
        assert key not in initialized
    assert initialized["g1_23dof_metadata"]["deployment_ready"] is False


def test_artifact_gate_rejects_initialization_and_accepts_validated_padded_model():
    warm_start = make_artifact_metadata(
        history_length=10,
        observation_layout=OBS_LAYOUT_CHECKPOINT_INIT_29,
        checkpoint_stage="checkpoint_initialization",
    )
    with pytest.raises(ValueError, match="checkpoint_stage"):
        validate_artifact_contract(warm_start, decoder_input_dim=994, decoder_output_dim=23)

    evidence, hashes = _validated_evidence()
    trained = make_artifact_metadata(
        history_length=10,
        observation_layout=OBS_LAYOUT_CHECKPOINT_INIT_29,
        checkpoint_stage="trained",
        deployment_ready=True,
        sim_validation_passed=True,
        simulation_evidence=evidence,
        artifact_hashes=hashes,
    )
    validate_artifact_contract(trained, decoder_input_dim=994, decoder_output_dim=23)
    with pytest.raises(ValueError, match="expected 23"):
        validate_artifact_contract(trained, decoder_input_dim=994, decoder_output_dim=29)

    wrong_history = dict(trained)
    wrong_history["history_length"] = 4
    wrong_history["decoder_input_dim"] = 436
    with pytest.raises(ValueError, match="exactly 10"):
        validate_artifact_contract(
            wrong_history,
            decoder_input_dim=436,
            decoder_output_dim=23,
        )


def test_vendored_rev_1_0_assets_are_native_23_not_dummy_29():
    asset_dir = Path(__file__).parents[1] / "data" / "robots" / "g1"
    urdf = ET.parse(asset_dir / "g1_23dof_rev_1_0.urdf").getroot()
    movable = [joint for joint in urdf.findall("joint") if joint.attrib["type"] == "revolute"]
    assert urdf.attrib["name"] == "g1_23dof_rev_1_0"
    assert len(movable) == 23

    mjcf = ET.parse(asset_dir / "g1_23dof_rev_1_0.xml").getroot()
    motors = mjcf.find("actuator").findall("motor")
    assert mjcf.attrib["model"] == "g1_23dof_rev_1_0"
    assert len(motors) == 23
    assert all("dummy" not in body.attrib.get("name", "") for body in mjcf.findall(".//body"))


def test_native_il23_order_matches_reduced_urdf_breadth_first_tree():
    asset_dir = Path(__file__).parents[1] / "data" / "robots" / "g1"
    urdf = ET.parse(asset_dir / "g1_23dof_rev_1_0.urdf").getroot()
    children_by_parent = {}
    for joint in urdf.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        children_by_parent.setdefault(parent, []).append(joint)

    queue = ["pelvis"]
    movable_breadth_first = []
    while queue:
        parent = queue.pop(0)
        for joint in children_by_parent.get(parent, []):
            queue.append(joint.find("child").attrib["link"])
            if joint.attrib["type"] != "fixed":
                movable_breadth_first.append(joint.attrib["name"])

    assert tuple(movable_breadth_first) == NATIVE_IL23_JOINT_NAMES


def test_g1_23dof_warm_start_hydra_config_composes():
    from hydra import compose, initialize_config_dir

    config_dir = str((Path(__file__).parents[1] / "config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(
            config_name="base",
            overrides=["+exp=manager/universal_token/all_modes/sonic_g1_23dof_rev_1_0_warm_start"],
        )

    assert cfg.manager_env.config.robot.type == "g1_23dof_rev_1_0"
    assert cfg.manager_env.config.robot.embodiment.required_mode_machine == 4
    assert cfg.manager_env.config.robot.embodiment.observation_layout == ("canonical_il29_fixed_slots_v1")
    assert cfg.manager_env.config.robot.embodiment.decoder_output_layout == DECODER_OUTPUT_LAYOUT
    assert list(cfg.algo.config.actor.backbone.encoders) == ["teleop"]
    assert list(cfg.algo.config.actor.backbone.decoders) == ["g1_dyn"]
    assert cfg.manager_env.commands.motion.motion_lib_cfg.asset.assetFileName == ("g1_23dof_rev_1_0.xml")
    for term_name in ("joint_pos", "joint_vel", "actions"):
        term = cfg.manager_env.observations.policy[term_name]
        assert "noise" not in term or term.noise is None
