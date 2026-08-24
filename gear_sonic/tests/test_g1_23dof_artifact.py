import copy
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
import onnx
import pytest
import torch

from gear_sonic.scripts.run_g1_23dof_sim_validation import (
    TRACE_KIND,
    TRACE_SCHEMA_VERSION,
    deterministic_disturbance_vector,
    recompute_trace_metrics,
)
from gear_sonic.utils import g1_23dof_artifact as artifact_module
from gear_sonic.utils.g1_23dof_artifact import (
    _ROBOT_ASSET_PATH,
    _ROBOT_CONFIG_PATH,
    _SIM_VALIDATION_CONFIG_PATH,
    MOTION_DATASET_SOURCE_ARCHIVE,
    build_training_material_evidence,
    build_true23_decoder,
    build_true23_policy_pair,
    canonical_json_bytes,
    export_validated_true23_artifact,
    is_true23_training_config,
    make_training_checkpoint_records,
    sha256_bytes,
    sha256_file,
    validate_encoder_onnx_structure,
    validate_onnx_structure,
    validate_runtime_config_snapshot,
    validate_simulation_report,
    verify_validated_true23_artifact,
)
from gear_sonic.utils.g1_23dof_checkpoint_io import build_safe_promotion_checkpoint
from gear_sonic.utils.g1_23dof_contract import (
    DECODER_OUTPUT_LAYOUT,
    DEPLOYMENT_DECODER_INPUT_DIM,
    NORMAL_INITIAL_POLICY_STATE_SHA256,
    NORMAL_RELEASE_SHA256,
    OBS_LAYOUT_PADDED_IL29,
    REFERENCE_PROFILE_NORMAL,
    TARGET_DOF,
    make_artifact_metadata,
    reference_profile_contract,
)
from gear_sonic.utils.inference_helpers import available_universal_token_encoders

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
TEST_MATERIAL_PROVENANCE = {
    "schema_version": 1,
    "runtime_source": {
        "schema_version": 1,
        "file_count": 1,
        "total_bytes": 1,
        "manifest_sha256": "e" * 64,
        "files": [
            {
                "relpath": "gear_sonic/test.py",
                "size_bytes": 1,
                "sha256": "d" * 64,
            }
        ],
    },
    "motion_dataset": TEST_MOTION_DATASET,
}


@lru_cache(maxsize=1)
def _normal_training_material():
    from gear_sonic.scripts.preflight_g1_23dof_training import (
        EXPERIMENT,
        _compose_config,
    )

    repo_root = Path(__file__).resolve().parents[2]
    config = _compose_config(
        repo_root,
        "sonic_release/g1_23dof_rev_1_0_init.pt",
        experiment=EXPERIMENT,
    )
    return build_training_material_evidence(config, repo_root=repo_root)


def _write_trained_checkpoint(path: Path):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    policy_state = {"std": torch.full((23,), 0.2)}

    def add_mlp(prefix: str, dimensions: tuple[int, ...]) -> None:
        for position, (input_dim, output_dim) in enumerate(zip(dimensions, dimensions[1:])):
            index = position * 2
            policy_state[f"{prefix}{index}.weight"] = (
                torch.randn(output_dim, input_dim, generator=generator) * 0.01
            )
            policy_state[f"{prefix}{index}.bias"] = torch.randn(output_dim, generator=generator) * 0.01

    add_mlp(
        "actor_module.encoders.teleop.module.",
        (267, 2048, 1024, 512, 512, 64),
    )
    add_mlp(
        "actor_module.decoders.g1_dyn.module.",
        (994, 2048, 2048, 1024, 1024, 512, 512, 23),
    )
    _, _, policy_state_sha256 = build_true23_policy_pair({"policy_state_dict": policy_state})
    metadata, training_evidence = make_training_checkpoint_records(
        global_step=1200,
        history_length=10,
        observation_layout=OBS_LAYOUT_PADDED_IL29,
        policy_state_sha256=policy_state_sha256,
        source_family="sonic_release",
        source_revision=None,
        source_checkpoint_sha256=NORMAL_RELEASE_SHA256,
        initial_policy_state_sha256=NORMAL_INITIAL_POLICY_STATE_SHA256,
        motion_dataset=TEST_MOTION_DATASET,
        training_material=_normal_training_material(),
    )
    checkpoint = build_safe_promotion_checkpoint(
        policy_state_dict=policy_state,
        global_step=1200,
        metadata=metadata,
        training_evidence=training_evidence,
    )
    torch.save(checkpoint, path)
    return checkpoint


@pytest.fixture
def approved_sim_producer(tmp_path, monkeypatch):
    runner_path = tmp_path / "approved_sim_runner.py"
    runner_path.write_text("# deterministic test-only runner identity\n", encoding="utf-8")
    runner_sha256 = sha256_file(runner_path)
    config = json.loads(
        json.dumps(artifact_module._sim_validation_config())  # noqa: SLF001
    )
    config["producer"]["runner_sha256"] = runner_sha256
    config["producer"]["promotion_enabled"] = True
    monkeypatch.setattr(artifact_module, "_SIM_VALIDATION_RUNNER_PATH", runner_path)
    monkeypatch.setattr(artifact_module, "_sim_validation_config", lambda: config)
    material_provenance = copy.deepcopy(TEST_MATERIAL_PROVENANCE)
    material_provenance["runtime_source"] = _normal_training_material()[
        "runtime_source"
    ]
    monkeypatch.setattr(
        artifact_module,
        "simulation_material_provenance",
        lambda *_args, **_kwargs: copy.deepcopy(material_provenance),
    )
    return runner_sha256


def _simulation_report(checkpoint_path: Path, runner_sha256: str):
    runtime_config = _approved_test_runtime_config()
    runtime_hashes = validate_runtime_config_snapshot(runtime_config)
    material_provenance = copy.deepcopy(TEST_MATERIAL_PROVENANCE)
    material_provenance["runtime_source"] = _normal_training_material()[
        "runtime_source"
    ]
    runs = []
    for scenario, scale, recovery in (
        ("nominal", 0.0, 0.0),
        ("disturbance_50", 0.5, 0.8),
        ("disturbance_100", 1.0, 1.6),
    ):
        for seed in (1729, 2718, 3141):
            runs.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "episodes": 22,
                    "steps": 5500,
                    "disturbance_scale": scale,
                    "termination_count": 0,
                    "nonfinite_count": 0,
                    "soft_limit_violation_count": 0,
                    "phantom_observation_max_abs": 0.0,
                    "max_recovery_time_s": recovery,
                    "action_saturation_fraction": 0.0,
                    "mpjpe_m": 0.08,
                }
            )
    return {
        "schema_version": 2,
        "kind": "g1_23dof_sim_validation",
        "robot_model": "g1_23dof_rev_1_0",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "producer": {
            "kind": "gear_sonic_true23_isaaclab_disturbance_validation",
            "version": 1,
            "runner_sha256": runner_sha256,
        },
        "reference_profile": REFERENCE_PROFILE_NORMAL,
        "reference_contract": reference_profile_contract(
            REFERENCE_PROFILE_NORMAL
        ),
        "observation_layout": "canonical_il29_fixed_slots_v1",
        "history_length": 10,
        "decoder_input_dim": 994,
        "decoder_output_dim": 23,
        "decoder_output_layout": DECODER_OUTPUT_LAYOUT,
        "runtime_config": {
            "resolved": runtime_config,
            **runtime_hashes,
        },
        "simulator": {
            "name": "IsaacLab",
            "version": "test",
            "asset_sha256": sha256_file(_ROBOT_ASSET_PATH),
            "robot_config_sha256": sha256_file(_ROBOT_CONFIG_PATH),
            "config_sha256": sha256_file(_SIM_VALIDATION_CONFIG_PATH),
            "runtime_config_sha256": runtime_hashes["resolved_config_sha256"],
        },
        "material_provenance": material_provenance,
        "trace_manifest_sha256": "0" * 64,
        "runs": runs,
    }


def _approved_test_runtime_config(
    experiment: str = "sonic_g1_23dof_rev_1_0_warm_start",
):
    config_dir = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/all_modes/" + experiment,
                "num_envs=22",
                "headless=true",
            ],
        )
    config = OmegaConf.merge(
        config,
        config.eval_overrides,
        {"num_envs": 22, "headless": True},
    )
    with open_dict(config):
        for event in config.manager_env.config.get("train_only_events", []):
            config.manager_env.events.pop(event, None)
        for termination in config.manager_env.config.get(
            "train_only_terminations", []
        ):
            config.manager_env.terminations.pop(termination, None)

    def resolved(path):
        value = OmegaConf.select(config, path)
        return (
            OmegaConf.to_container(value, resolve=True)
            if OmegaConf.is_config(value)
            else value
        )

    manager_keys = tuple(
        key
        for key in config.manager_env.config
        if key not in {"save_rendering_dir", "experiment_dir", "obs"}
    )
    return {
        "use_manager_env": resolved("use_manager_env"),
        "headless": resolved("headless"),
        "num_envs": resolved("num_envs"),
        "sim_type": resolved("sim_type"),
        "force_flat_terrain": resolved("force_flat_terrain"),
        "multi_gpu": resolved("multi_gpu"),
        "num_gpus": resolved("num_gpus"),
        "seed": resolved("seed"),
        "global_rank": resolved("global_rank"),
        "actor_prop_history_length": resolved("actor_prop_history_length"),
        "actor_actions_history_length": resolved("actor_actions_history_length"),
        "manager_env": {
            "_target_": resolved("manager_env._target_"),
            "config": {
                key: resolved(f"manager_env.config.{key}") for key in manager_keys
            },
            **{
                key: resolved(f"manager_env.{key}")
                for key in (
                    "commands",
                    "actions",
                    "observations",
                    "rewards",
                    "terminations",
                    "events",
                    "curriculum",
                    "recorders",
                )
            },
        },
        "algo": {
            "config": {
                "actor": resolved("algo.config.actor"),
                "module_dim": resolved("algo.config.module_dim") or {},
                "distill_only": resolved("algo.config.distill_only") or False,
            }
        },
        "trainer": {"schedule_dict": resolved("trainer.schedule_dict") or {}},
    }


def _materialize_test_traces(path: Path, report: dict) -> None:
    if "runs" not in report or "producer" not in report or "simulator" not in report:
        return
    config = artifact_module._sim_validation_config()  # noqa: SLF001
    trace_dir = path.with_name(f"{path.stem}.traces")
    trace_dir.mkdir()
    manifest = []
    for run in report["runs"]:
        episodes = run["episodes"]
        saturated_count = 3 if run["action_saturation_fraction"] > 0.1 else 0
        recovery_start = int(
            config["disturbance_schedule"]["apply_step"]
            + run["max_recovery_time_s"] * config["control_hz"]
        )
        records = []
        for step in range(config["minimum_coverage"]["steps_per_episode"]):
            if step == config["disturbance_schedule"]["apply_step"]:
                deltas = [
                    deterministic_disturbance_vector(
                        seed=run["seed"],
                        episode_index=episode,
                        scale=run["disturbance_scale"],
                        config=config,
                    )
                    for episode in range(episodes)
                ]
            else:
                deltas = [[0.0] * 6 for _ in range(episodes)]
            recovery_value = (
                0.0
                if run["disturbance_scale"] == 0.0 or step < config["disturbance_schedule"]["apply_step"]
                or step >= recovery_start
                else 1.0
            )
            records.append(
                {
                    "step": step,
                    "disturbance_delta": deltas,
                    "terminated": [False] * episodes,
                    "timed_out": [False] * episodes,
                    "nonfinite": [False] * episodes,
                    "soft_limit_violation": [False] * episodes,
                    "phantom_observation_max_abs": [0.0] * episodes,
                    "recovery_metric": [recovery_value] * episodes,
                    "action_saturated_count": [saturated_count] * episodes,
                    "action_count": [23] * episodes,
                    "mpjpe_m": [run["mpjpe_m"]] * episodes,
                }
            )
        run.update(
            recompute_trace_metrics(
                records,
                episodes=episodes,
                disturbance_scale=run["disturbance_scale"],
                config=config,
            )
        )
        trace = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "kind": TRACE_KIND,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "producer": report["producer"],
            "simulator": report["simulator"],
            "material_provenance": report["material_provenance"],
            "scenario": run["scenario"],
            "seed": run["seed"],
            "disturbance_scale": run["disturbance_scale"],
            "control_hz": config["control_hz"],
            "episodes": episodes,
            "steps_per_episode": config["minimum_coverage"]["steps_per_episode"],
            "records": records,
        }
        trace_bytes = canonical_json_bytes(trace)
        relative = f"{trace_dir.name}/{run['scenario']}-seed-{run['seed']}.json"
        trace_path = path.parent / relative
        trace_path.write_bytes(trace_bytes)
        trace_record = {
            "file": relative,
            "sha256": sha256_bytes(trace_bytes),
            "payload_sha256": sha256_bytes(canonical_json_bytes(trace)),
            "record_count": len(records),
        }
        run["trace"] = trace_record
        manifest.append(
            {
                "scenario": run["scenario"],
                "seed": run["seed"],
                **trace_record,
            }
        )
    report["trace_manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))


def _write_report(path: Path, report) -> None:
    _materialize_test_traces(path, report)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_export_is_static_validated_and_deterministic(
    tmp_path,
    approved_sim_producer,
):
    checkpoint_path = tmp_path / "trained.pt"
    _write_trained_checkpoint(checkpoint_path)
    report_path = tmp_path / "sim-report.json"
    _write_report(
        report_path,
        _simulation_report(checkpoint_path, approved_sim_producer),
    )

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_encoder, first_decoder, first_metadata, metadata = export_validated_true23_artifact(
        checkpoint_path,
        report_path,
        first_dir / "model.onnx",
    )
    second_encoder, second_decoder, second_metadata, _ = export_validated_true23_artifact(
        checkpoint_path,
        report_path,
        second_dir / "model.onnx",
    )

    assert sha256_file(first_encoder) == sha256_file(second_encoder)
    assert sha256_file(first_decoder) == sha256_file(second_decoder)
    assert first_metadata.read_bytes() == second_metadata.read_bytes()
    assert metadata["checkpoint_stage"] == "trained"
    assert metadata["sim_validation_passed"] is True
    assert metadata["simulation_evidence"]["computed_pass"] is True
    assert metadata["history_length"] == 10
    assert metadata["decoder_input_dim"] == DEPLOYMENT_DECODER_INPUT_DIM
    assert metadata["decoder_output_dim"] == TARGET_DOF

    encoder_model = onnx.load(first_encoder)
    validate_encoder_onnx_structure(encoder_model)
    assert [dimension.dim_value for dimension in encoder_model.graph.input[0].type.tensor_type.shape.dim] == [
        1,
        267,
    ]
    assert [dimension.dim_value for dimension in encoder_model.graph.output[0].type.tensor_type.shape.dim] == [
        1,
        64,
    ]

    model = onnx.load(first_decoder)
    assert [dimension.dim_value for dimension in model.graph.input[0].type.tensor_type.shape.dim] == [
        1,
        994,
    ]
    assert [dimension.dim_value for dimension in model.graph.output[0].type.tensor_type.shape.dim] == [1, 23]
    verify_validated_true23_artifact(
        first_encoder,
        first_decoder,
        first_metadata,
        checkpoint_path=checkpoint_path,
        simulation_report_path=report_path,
    )
    model.graph.input[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    with pytest.raises(ValueError, match="dynamic"):
        validate_onnx_structure(model)


def test_export_rejects_initialization_only_checkpoint_even_with_global_step(
    tmp_path,
    approved_sim_producer,
):
    checkpoint_path = tmp_path / "initialization.pt"
    checkpoint = _write_trained_checkpoint(checkpoint_path)
    checkpoint["g1_23dof_metadata"] = make_artifact_metadata(
        history_length=10,
        observation_layout=OBS_LAYOUT_PADDED_IL29,
        checkpoint_stage="checkpoint_initialization",
    )
    checkpoint.pop("g1_23dof_training_evidence")
    torch.save(checkpoint, checkpoint_path)
    report_path = tmp_path / "sim-report.json"
    _write_report(
        report_path,
        _simulation_report(checkpoint_path, approved_sim_producer),
    )

    with pytest.raises(ValueError, match="checkpoint_stage|training-produced"):
        export_validated_true23_artifact(
            checkpoint_path,
            report_path,
            tmp_path / "blocked.onnx",
        )
    assert not (tmp_path / "blocked.onnx").exists()


def test_export_rejects_weights_changed_after_training_marker(
    tmp_path,
    approved_sim_producer,
):
    checkpoint_path = tmp_path / "trained.pt"
    checkpoint = _write_trained_checkpoint(checkpoint_path)
    checkpoint["policy_state_dict"]["actor_module.decoders.g1_dyn.module.2.bias"][0] += 1.0
    torch.save(checkpoint, checkpoint_path)
    report_path = tmp_path / "sim-report.json"
    _write_report(
        report_path,
        _simulation_report(checkpoint_path, approved_sim_producer),
    )

    with pytest.raises(ValueError, match="training_evidence"):
        export_validated_true23_artifact(
            checkpoint_path,
            report_path,
            tmp_path / "changed.onnx",
        )


def test_export_rejects_boolean_only_or_failed_simulation_evidence(
    tmp_path,
    approved_sim_producer,
):
    checkpoint_path = tmp_path / "trained.pt"
    _write_trained_checkpoint(checkpoint_path)

    boolean_report_path = tmp_path / "boolean.json"
    _write_report(boolean_report_path, {"passed": True})
    with pytest.raises(ValueError, match="keys mismatch"):
        export_validated_true23_artifact(
            checkpoint_path,
            boolean_report_path,
            tmp_path / "boolean.onnx",
        )

    failed_report = _simulation_report(checkpoint_path, approved_sim_producer)
    failed_report["runs"][8]["action_saturation_fraction"] = 0.11
    failed_report_path = tmp_path / "failed.json"
    _write_report(failed_report_path, failed_report)
    with pytest.raises(ValueError, match="saturation"):
        export_validated_true23_artifact(
            checkpoint_path,
            failed_report_path,
            tmp_path / "failed.onnx",
        )
    assert not (tmp_path / "failed.onnx").exists()


def test_export_rejects_wrong_checkpoint_binding_and_wrong_decoder_shape(
    tmp_path,
    approved_sim_producer,
):
    checkpoint_path = tmp_path / "trained.pt"
    _write_trained_checkpoint(checkpoint_path)
    report = _simulation_report(checkpoint_path, approved_sim_producer)
    report["checkpoint_sha256"] = "c" * 64
    report_path = tmp_path / "wrong-binding.json"
    _write_report(report_path, report)
    with pytest.raises(ValueError, match="different checkpoint"):
        export_validated_true23_artifact(
            checkpoint_path,
            report_path,
            tmp_path / "wrong-binding.onnx",
        )

    toy_policy = {
        "actor_module.encoders.teleop.module.0.weight": torch.randn(64, 267),
        "actor_module.encoders.teleop.module.0.bias": torch.randn(64),
        "actor_module.decoders.g1_dyn.module.0.weight": torch.randn(8, 994),
        "actor_module.decoders.g1_dyn.module.0.bias": torch.randn(8),
        "actor_module.decoders.g1_dyn.module.2.weight": torch.randn(23, 8),
        "actor_module.decoders.g1_dyn.module.2.bias": torch.randn(23),
        "std": torch.full((23,), 0.2),
    }
    with pytest.raises(ValueError, match="linear layer indices"):
        build_true23_decoder({"policy_state_dict": toy_policy})


def test_verifier_rejects_sidecar_tampering(tmp_path, approved_sim_producer):
    checkpoint_path = tmp_path / "trained.pt"
    _write_trained_checkpoint(checkpoint_path)
    report_path = tmp_path / "sim-report.json"
    _write_report(
        report_path,
        _simulation_report(checkpoint_path, approved_sim_producer),
    )
    encoder_path, decoder_path, metadata_path, _ = export_validated_true23_artifact(
        checkpoint_path,
        report_path,
        tmp_path / "model.onnx",
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["mode_machine"] = 5
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        verify_validated_true23_artifact(
            encoder_path,
            decoder_path,
            metadata_path,
        )


def test_export_rejects_output_path_aliases_before_loading_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="distinct paths"):
        export_validated_true23_artifact(
            tmp_path / "missing.pt",
            tmp_path / "missing-report.json",
            tmp_path / "pair",
            metadata_path=tmp_path / "pair_encoder.onnx",
        )


def test_sim_promotion_is_unavailable_without_checked_in_producer(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.bin"
    checkpoint_path.write_bytes(b"checkpoint")
    report = _simulation_report(checkpoint_path, "e" * 64)
    report_path = tmp_path / "blocked-report.json"
    _write_report(report_path, report)
    with pytest.raises(ValueError, match="promotion unavailable"):
        validate_simulation_report(
            report,
            checkpoint_sha256=sha256_file(checkpoint_path),
            report_sha256="a" * 64,
            report_payload_sha256="b" * 64,
            report_path=report_path,
            checkpoint_motion_dataset=TEST_MOTION_DATASET,
        )


def test_training_records_and_encoder_filter_are_fail_closed():
    metadata, evidence = make_training_checkpoint_records(
        global_step=50,
        history_length=10,
        observation_layout=OBS_LAYOUT_PADDED_IL29,
        policy_state_sha256="d" * 64,
        source_family="sonic_release",
        source_revision=None,
        source_checkpoint_sha256=NORMAL_RELEASE_SHA256,
        initial_policy_state_sha256=NORMAL_INITIAL_POLICY_STATE_SHA256,
        motion_dataset=TEST_MOTION_DATASET,
        training_material=_normal_training_material(),
    )
    assert metadata["checkpoint_stage"] == "trained"
    assert evidence["global_step"] == 50
    assert evidence["weights_only_initialization"] is False

    module = SimpleNamespace(encoders_to_iterate=["teleop"])
    assert available_universal_token_encoders(module) == ("teleop",)
    manager_config = {"robot": {"type": "g1_23dof_rev_1_0"}}
    full_config = {
        "manager_env": {
            "config": {
                "robot": {
                    "embodiment": {
                        "model": "g1_23dof_rev_1_0",
                    }
                }
            }
        }
    }
    assert is_true23_training_config(manager_config)
    assert is_true23_training_config(full_config)
    assert not is_true23_training_config({"robot": {"type": "g1_model_12_dex"}})


def test_normal_and_low_latency_runtime_hashes_are_separately_approved():
    normal = _approved_test_runtime_config()
    low_latency = _approved_test_runtime_config(
        "sonic_g1_23dof_rev_1_0_low_latency_warm_start"
    )

    normal_hashes = validate_runtime_config_snapshot(normal)
    low_latency_hashes = validate_runtime_config_snapshot(low_latency)
    assert normal_hashes["material_config_sha256"] != low_latency_hashes[
        "material_config_sha256"
    ]

    profile_swapped = copy.deepcopy(low_latency)
    profile_swapped["manager_env"]["config"]["robot"]["embodiment"][
        "reference_profile"
    ] = REFERENCE_PROFILE_NORMAL
    profile_swapped["manager_env"]["config"]["robot"]["embodiment"][
        "reference_contract"
    ] = reference_profile_contract(REFERENCE_PROFILE_NORMAL)
    with pytest.raises(ValueError, match="approved true23 runtime contract"):
        validate_runtime_config_snapshot(profile_swapped)
