from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gear_sonic.scripts.train_g1_true23_native124_selected_v2_ankle import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM,
    FIXED_GPU,
    FIXED_LEARNING_RATE,
    FIXED_SEED,
    HIDDEN_DIMS,
    RSL_DISTRIBUTION_NAME,
    RSL_EXPECTED_DIST_INFO_DIRNAME,
    RSL_EXPECTED_FILE_SHA256,
    RSL_EXPECTED_TREE_FILE_COUNT,
    RSL_EXPECTED_TREE_MANIFEST_SHA256,
    RSL_EXPECTED_TREE_TOTAL_BYTES,
    RSL_EXPECTED_VERSION,
    RSL_RUNTIME_MODULE_FILES,
    SMOKE_ITERATIONS,
    SMOKE_NUM_ENVS,
    TRAIN_CHECKPOINT_INTERVAL,
    TRAIN_MAX_ITERATIONS,
    TRAIN_MAX_NUM_ENVS,
    Stage1LaunchPlan,
    _best_effort_stop_logger,
    _canonical_sha256,
    _optimizer_step_audit,
    _parser,
    _plan_from_args,
    _validate_content_hash,
    _validate_rsl_runtime_paths,
    _validated_stage1_checkpoint_paths,
    _write_json_exclusive,
    build_stage1_material_manifest,
    expected_stage1_checkpoint_filenames,
    expected_stage1_optimizer_steps,
    preflight,
    resolve_rsl_runtime_binding,
    resolved_training_config,
    stage1_agent_config,
)
from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
    ANKLE_HARDWARE_ROWS,
    SELECTED_ACTOR_STATE_SHA256,
    SELECTED_CHECKPOINT_SHA256,
)
from gear_sonic.utils.g1_23dof_native124_21204_adapter import ONNX_SHA256

REPO_ROOT = Path(__file__).resolve().parents[2]
MOTION = (
    REPO_ROOT
    / "artifacts"
    / "g1_native124_multimotion"
    / "scaling_all61"
    / "feasible_v1"
    / "npz"
    / "B_DadDance.npz"
)


def _plan(tmp_path: Path, mode: str = "smoke") -> Stage1LaunchPlan:
    if mode == "smoke":
        num_envs = SMOKE_NUM_ENVS
        iterations = SMOKE_ITERATIONS
    else:
        num_envs = 32
        iterations = 20
    return Stage1LaunchPlan(
        mode=mode,  # type: ignore[arg-type]
        motion_file=MOTION,
        run_dir=tmp_path / f"{mode}_run",
        num_envs=num_envs,
        iterations=iterations,
    )


def _synthetic_runtime_binding() -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "g1_true23_native124_selected_v2_rsl_runtime_v1",
        "distribution_name": RSL_DISTRIBUTION_NAME,
        "version": RSL_EXPECTED_VERSION,
        "package_root": "/synthetic/site-packages/rsl_rl",
        "distribution_metadata_root": "/synthetic/site-packages/rsl_rl_lib-5.0.1.dist-info",
        "executed_source_manifest": {
            "schema_version": 2,
            "kind": "source_files",
            "file_count": RSL_EXPECTED_TREE_FILE_COUNT,
            "total_bytes": RSL_EXPECTED_TREE_TOTAL_BYTES,
            "manifest_sha256": RSL_EXPECTED_TREE_MANIFEST_SHA256,
            "files": [],
        },
        "expected_file_sha256": dict(RSL_EXPECTED_FILE_SHA256),
    }
    value["runtime_binding_sha256"] = _canonical_sha256(value)
    return value


@pytest.mark.parametrize("mode", ["smoke", "train"])
def test_agent_config_is_exact_fixed_four_ankle_rsl_boundary(tmp_path: Path, mode: str) -> None:
    plan = _plan(tmp_path, mode)
    cfg = stage1_agent_config(plan)

    assert cfg["seed"] == FIXED_SEED
    assert cfg["max_iterations"] == plan.iterations
    assert cfg["save_interval"] == (plan.iterations if mode == "smoke" else TRAIN_CHECKPOINT_INTERVAL)
    assert cfg["obs_groups"] == {"actor": ["actor"], "critic": ["critic"]}
    assert cfg["clip_actions"] is None
    assert cfg["logger"] == "tensorboard"
    assert cfg["upload_model"] is False
    assert cfg["resume"] is False
    assert cfg["actor"] == {
        "hidden_dims": HIDDEN_DIMS,
        "activation": "elu",
        "obs_normalization": True,
        "cnn_cfg": None,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        "class_name": "MLPModel",
    }
    assert cfg["critic"] == {
        "hidden_dims": HIDDEN_DIMS,
        "activation": "elu",
        "obs_normalization": True,
        "cnn_cfg": None,
        "distribution_cfg": None,
        "class_name": "MLPModel",
    }
    algorithm = cfg["algorithm"]
    assert algorithm["class_name"] == "PPO"
    assert algorithm["learning_rate"] == FIXED_LEARNING_RATE
    assert algorithm["schedule"] == "fixed"
    assert algorithm["desired_kl"] is None
    assert algorithm["rnd_cfg"] is None
    assert algorithm["symmetry_cfg"] is None


def test_resolved_config_binds_selected_actor_fresh_critic_and_no_dr(tmp_path: Path) -> None:
    runtime = _synthetic_runtime_binding()
    resolved = resolved_training_config(_plan(tmp_path), runtime)
    model = resolved["model"]

    assert model["actor_observation_dim"] == ACTOR_OBSERVATION_DIM
    assert model["critic_observation_dim"] == CRITIC_OBSERVATION_DIM
    assert model["action_dim"] == ACTION_DIM
    assert model["source_checkpoint_sha256"] == SELECTED_CHECKPOINT_SHA256
    assert model["source_actor_state_sha256"] == SELECTED_ACTOR_STATE_SHA256
    assert model["source_onnx_sha256"] == ONNX_SHA256
    assert model["selected_load"] == {
        "actor": True,
        "critic": False,
        "iteration": False,
        "optimizer": False,
        "rnd": False,
    }
    assert model["trainable_actor_hardware_rows"] == list(ANKLE_HARDWARE_ROWS)
    assert model["critic_initialization"] == "fresh_privileged_256"
    assert model["optimizer_initialization"] == "fresh_adam_iteration_0"
    assert resolved["random_initial_episode_length"] is False
    assert resolved["rsl_runtime"] == runtime
    assert resolved["checkpointing"]["initial_checkpoint"] == "initial_model.pt"
    assert resolved["checkpointing"]["first_updated_checkpoint"] == "model_0.pt"
    assert resolved["checkpointing"]["final_checkpoint"] == "model_1.pt"
    assert resolved["checkpointing"]["save_interval_outer_ppo_iterations"] == 2
    assert resolved["checkpointing"]["periodic_checkpoint_iterations"] == [0]
    assert resolved["checkpointing"]["expected_checkpoint_files"] == [
        "initial_model.pt",
        "model_0.pt",
        "model_1.pt",
    ]
    assert resolved["safety"] == {
        "simulator_only": True,
        "hardware_authorized": False,
        "robot_network_commands": False,
        "external_network_calls": False,
        "upload_model": False,
        "domain_randomization": False,
        "push_disturbances": False,
        "deployment_ready": False,
    }
    claimed = resolved["resolved_config_sha256"]
    copy_without_hash = dict(resolved)
    copy_without_hash.pop("resolved_config_sha256")
    assert claimed == _canonical_sha256(copy_without_hash)
    assert _validate_content_hash(resolved, "resolved_config_sha256") == claimed


def test_material_manifest_hash_binds_resolved_config_and_exact_inputs(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    runtime = _synthetic_runtime_binding()
    resolved = resolved_training_config(plan, runtime)
    manifest = build_stage1_material_manifest(plan, resolved, runtime)

    assert manifest["resolved_config_sha256"] == resolved["resolved_config_sha256"]
    assert manifest["rsl_runtime"] == runtime
    assert manifest["source_files"]["file_count"] >= 20
    assert manifest["robot_assets"]["file_count"] >= 20
    bound = {item["logical_path"]: item for item in manifest["bound_inputs"]["files"]}
    assert bound["motions/B_DadDance.npz"]["sha256"] == resolved["motion_sha256"]
    assert bound["selected/model21204_alpha25.pt"]["sha256"] == SELECTED_CHECKPOINT_SHA256
    assert bound["selected/policy.onnx"]["sha256"] == ONNX_SHA256
    assert _validate_content_hash(manifest, "material_manifest_sha256") == manifest["material_manifest_sha256"]

    tampered = copy.deepcopy(resolved)
    tampered["num_envs"] += 1
    with pytest.raises(ValueError, match="resolved_config_sha256 mismatch"):
        build_stage1_material_manifest(plan, tampered, runtime)


def test_rsl_runtime_path_version_and_sha_drift_fail_closed(tmp_path: Path) -> None:
    package_root = tmp_path / "site-packages" / "rsl_rl"
    module_files: dict[str, Path] = {}
    for module_name, relative in RSL_RUNTIME_MODULE_FILES.items():
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic {module_name}\n", encoding="utf-8")
        module_files[module_name] = path
    metadata_root = tmp_path / "site-packages" / RSL_EXPECTED_DIST_INFO_DIRNAME
    metadata_root.mkdir()
    (metadata_root / "METADATA").write_text("Version: 5.0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="RSL version mismatch"):
        _validate_rsl_runtime_paths(
            version="5.0.0",
            package_root=package_root,
            distribution_metadata_root=metadata_root,
            module_files=module_files,
        )
    drifted_path = dict(module_files)
    drifted_path["rsl_rl.algorithms.ppo"] = module_files["rsl_rl.models.mlp_model"]
    with pytest.raises(ValueError, match="RSL module path drift"):
        _validate_rsl_runtime_paths(
            version=RSL_EXPECTED_VERSION,
            package_root=package_root,
            distribution_metadata_root=metadata_root,
            module_files=drifted_path,
        )
    with pytest.raises(ValueError, match="RSL installed source SHA drift"):
        _validate_rsl_runtime_paths(
            version=RSL_EXPECTED_VERSION,
            package_root=package_root,
            distribution_metadata_root=metadata_root,
            module_files=module_files,
        )


def test_preflight_is_read_only_and_accepts_hash_locked_local_materials(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    runtime = _synthetic_runtime_binding()
    report = preflight(plan, rsl_runtime_binding=runtime)

    assert report["ready"] is True, report["problems"]
    assert report["problems"] == []
    assert report["rsl_runtime"] == runtime
    assert report["source"]["checkpoint_sha256"] == SELECTED_CHECKPOINT_SHA256
    assert report["source"]["actor_state_sha256"] == SELECTED_ACTOR_STATE_SHA256
    assert report["source"]["onnx_sha256"] == ONNX_SHA256
    assert report["safety"] == {
        "simulator_constructed": False,
        "simulator_steps": 0,
        "training_updates": 0,
        "hardware_authorized": False,
        "network_used": False,
    }
    assert not plan.run_dir.exists()


def test_preflight_rejects_existing_run_directory_without_mutation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan.run_dir.mkdir()
    marker = plan.run_dir / "owner.txt"
    marker.write_text("user\n", encoding="utf-8")

    report = preflight(plan, rsl_runtime_binding=_synthetic_runtime_binding())

    assert report["ready"] is False
    assert any("overwrite forbidden" in problem for problem in report["problems"])
    assert marker.read_text(encoding="utf-8") == "user\n"


def test_exclusive_json_writer_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "bound.json"
    _write_json_exclusive(output, {"schema": "first", "value": 1})
    before = output.read_bytes()
    assert json.loads(before) == {"schema": "first", "value": 1}

    with pytest.raises(FileExistsError):
        _write_json_exclusive(output, {"schema": "second", "value": 2})
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "argv",
    [
        ["smoke", "--num-envs", "8"],
        ["train", "--learning-rate", "0.0001"],
        ["train", "--seed", "1"],
        ["train", "--resume", "model.pt"],
        ["train", "--upload"],
        ["train", "--iterations", "1"],
        ["train", "--iterations", "4"],
        ["train", "--iterations", "6"],
        ["train", "--iterations", str(TRAIN_MAX_ITERATIONS + 1)],
        ["train", "--num-envs", str(TRAIN_MAX_NUM_ENVS + 1)],
    ],
)
def test_cli_rejects_unsupported_or_unbounded_training_controls(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(argv)


def test_cli_profiles_resolve_only_fixed_seed_gpu_and_learning_contract(tmp_path: Path) -> None:
    smoke_args = _parser().parse_args(["smoke", "--run-dir", str(tmp_path / "smoke")])
    smoke = _plan_from_args(smoke_args)
    assert smoke.mode == "smoke"
    assert smoke.num_envs == SMOKE_NUM_ENVS
    assert smoke.iterations == SMOKE_ITERATIONS

    train_args = _parser().parse_args(
        [
            "train",
            "--run-dir",
            str(tmp_path / "train"),
            "--num-envs",
            "16",
            "--iterations",
            "25",
        ]
    )
    train = _plan_from_args(train_args)
    assert train.mode == "train"
    assert train.num_envs == 16
    assert train.iterations == 25
    resolved = resolved_training_config(train, _synthetic_runtime_binding())
    assert resolved["seed"] == FIXED_SEED
    assert resolved["gpu"] == FIXED_GPU
    assert resolved["agent"]["algorithm"]["learning_rate"] == FIXED_LEARNING_RATE


def test_launch_plan_rejects_smoke_drift_and_unbounded_train(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="smoke plan dimensions are fixed"):
        Stage1LaunchPlan("smoke", MOTION, tmp_path / "bad_smoke", 8, 2)
    with pytest.raises(ValueError, match="train num_envs"):
        Stage1LaunchPlan("train", MOTION, tmp_path / "bad_envs", TRAIN_MAX_NUM_ENVS + 1, 2)
    with pytest.raises(ValueError, match="train iterations"):
        Stage1LaunchPlan("train", MOTION, tmp_path / "bad_iterations", 1, 1)
    with pytest.raises(ValueError, match="multiple of 5"):
        Stage1LaunchPlan("train", MOTION, tmp_path / "bad_cadence", 1, 6)


def test_train_checkpoint_contract_is_exact_and_rejects_missing_or_extra(tmp_path: Path) -> None:
    plan = Stage1LaunchPlan("train", MOTION, tmp_path / "run", 32, 20)
    expected = (
        "initial_model.pt",
        "model_0.pt",
        "model_5.pt",
        "model_10.pt",
        "model_15.pt",
        "model_19.pt",
    )
    assert plan.checkpoint_interval == TRAIN_CHECKPOINT_INTERVAL
    assert expected_stage1_checkpoint_filenames(plan) == expected
    resolved = resolved_training_config(plan, _synthetic_runtime_binding())
    assert resolved["agent"]["save_interval"] == TRAIN_CHECKPOINT_INTERVAL
    assert resolved["checkpointing"]["save_interval_outer_ppo_iterations"] == 5
    assert resolved["checkpointing"]["periodic_checkpoint_iterations"] == [0, 5, 10, 15]
    assert resolved["checkpointing"]["expected_checkpoint_files"] == list(expected)

    plan.run_dir.mkdir()
    for filename in expected:
        (plan.run_dir / filename).write_bytes(filename.encode("ascii"))
    assert tuple(path.name for path in _validated_stage1_checkpoint_paths(plan.run_dir, plan)) == expected

    extra = plan.run_dir / "model_7.pt"
    extra.write_bytes(b"extra")
    with pytest.raises(RuntimeError, match=r"unexpected=\['model_7.pt'\]"):
        _validated_stage1_checkpoint_paths(plan.run_dir, plan)
    extra.unlink()
    (plan.run_dir / "model_10.pt").unlink()
    with pytest.raises(RuntimeError, match=r"missing=\['model_10.pt'\]"):
        _validated_stage1_checkpoint_paths(plan.run_dir, plan)


@pytest.mark.parametrize("mode,expected", [("smoke", 2), ("train", 240)])
def test_optimizer_step_audit_requires_exact_epoch_minibatch_product(
    tmp_path: Path,
    mode: str,
    expected: int,
) -> None:
    plan = _plan(tmp_path, mode)
    assert expected_stage1_optimizer_steps(plan) == expected

    class Integration:
        prior_completed_optimizer_steps = 0
        optimizer_steps = expected
        completed_optimizer_steps_total = expected

    assert _optimizer_step_audit(plan, Integration()) == {
        "expected_session_optimizer_steps": expected,
        "prior_completed_optimizer_steps": 0,
        "actual_session_optimizer_steps": expected,
        "actual_completed_optimizer_steps_total": expected,
    }
    Integration.optimizer_steps -= 1
    Integration.completed_optimizer_steps_total -= 1
    with pytest.raises(RuntimeError, match="optimizer-step count drift"):
        _optimizer_step_audit(plan, Integration())


def test_failed_logger_close_is_best_effort_and_never_raises() -> None:
    class Logger:
        writer = object()
        calls = 0

        def stop_logging_writer(self) -> None:
            self.calls += 1
            raise RuntimeError("logger close failed")

    class Runner:
        logger = Logger()

    _best_effort_stop_logger(Runner())
    assert Runner.logger.calls == 1


def test_pinned_rsl_constructs_exact_launcher_actor_and_critic(tmp_path: Path) -> None:
    try:
        from mjlab.rl.runner import MjlabOnPolicyRunner
        from tensordict import TensorDict
        import torch

        from gear_sonic.trl.mjlab.native124_selected_v2_ankle_rsl import (
            configure_selected_v2_ankle_rsl_runner,
        )
    except ImportError:
        pytest.skip("pinned WSL RSL-RL runtime required")

    runtime = resolve_rsl_runtime_binding()
    assert runtime["version"] == RSL_EXPECTED_VERSION
    assert runtime["expected_file_sha256"] == RSL_EXPECTED_FILE_SHA256
    assert runtime["executed_source_manifest"]["file_count"] == len(RSL_EXPECTED_FILE_SHA256)
    assert runtime["executed_source_manifest"]["manifest_sha256"] == RSL_EXPECTED_TREE_MANIFEST_SHA256
    assert _validate_content_hash(runtime, "runtime_binding_sha256") == runtime["runtime_binding_sha256"]

    observations = TensorDict(
        {
            "actor": torch.zeros(2, ACTOR_OBSERVATION_DIM),
            "critic": torch.zeros(2, CRITIC_OBSERVATION_DIM),
        },
        batch_size=[2],
    )

    class FakeEnv:
        num_actions = ACTION_DIM
        num_envs = 2
        cfg: dict[str, object] = {}

        def get_observations(self) -> TensorDict:
            return observations

    plan = _plan(tmp_path, "train")
    cfg = copy.deepcopy(stage1_agent_config(plan))
    runner = MjlabOnPolicyRunner(FakeEnv(), cfg, None, "cpu")
    algorithm = runner.alg
    actor = algorithm.actor
    critic = algorithm.critic

    assert runner.cfg["obs_groups"] == {"actor": ["actor"], "critic": ["critic"]}
    assert runner.cfg["save_interval"] == TRAIN_CHECKPOINT_INTERVAL
    assert expected_stage1_checkpoint_filenames(plan) == (
        "initial_model.pt",
        "model_0.pt",
        "model_5.pt",
        "model_10.pt",
        "model_15.pt",
        "model_19.pt",
    )
    assert actor.obs_dim == ACTOR_OBSERVATION_DIM
    assert actor.obs_groups == ["actor"]
    assert actor.mlp[0].in_features == ACTOR_OBSERVATION_DIM
    assert actor.mlp[-1].out_features == ACTION_DIM
    assert critic.obs_dim == CRITIC_OBSERVATION_DIM
    assert critic.obs_groups == ["critic"]
    assert critic.mlp[0].in_features == CRITIC_OBSERVATION_DIM
    assert critic.mlp[-1].out_features == 1
    assert algorithm.learning_rate == FIXED_LEARNING_RATE
    assert algorithm.schedule == "fixed"
    assert algorithm.rnd is None
    integration = configure_selected_v2_ankle_rsl_runner(
        runner,
        repository_root=REPO_ROOT,
    )
    assert integration.actor_observation_group == "actor"
    assert integration.actor.obs_groups == ["actor"]
    assert integration.critic.obs_groups == ["critic"]
