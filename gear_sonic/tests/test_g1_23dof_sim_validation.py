import copy
import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
import pytest
import torch

from gear_sonic.scripts import run_g1_23dof_sim_validation as runner
from gear_sonic.utils import g1_23dof_artifact as artifact
from gear_sonic.utils.g1_23dof_checkpoint_io import build_safe_promotion_checkpoint
from gear_sonic.utils.g1_23dof_contract import (
    OBS_LAYOUT_PADDED_IL29,
    make_artifact_metadata,
)


def _approved_test_runtime_config():
    config_dir = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/all_modes/"
                "sonic_g1_23dof_rev_1_0_warm_start",
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


class FakeIsaacLabBackend:
    num_envs = 22
    control_hz = 50
    simulator_name = "IsaacLab"
    simulator_version = "fake-test-runtime"

    def __init__(self):
        self.step_index = 0
        self.disturbances = []
        self.seed = None
        self.resolved_config = _approved_test_runtime_config()

    def start_run(self, seed: int) -> None:
        self.seed = seed
        self.step_index = 0

    def apply_velocity_delta(self, deltas):
        assert len(deltas) == self.num_envs
        assert all(len(delta) == 6 for delta in deltas)
        self.disturbances.append((self.seed, deltas))

    def step(self):
        # 0.4 s recovery: steps 50..69 remain above baseline+margin.
        recovering = 50 <= self.step_index < 70
        telemetry = {
            "terminated": [False] * self.num_envs,
            "timed_out": [False] * self.num_envs,
            "nonfinite": [False] * self.num_envs,
            "soft_limit_violation": [False] * self.num_envs,
            "phantom_observation_max_abs": [0.0] * self.num_envs,
            "recovery_metric": [1.0 if recovering else 0.0] * self.num_envs,
            "action_saturated_count": [0] * self.num_envs,
            "action_count": [23] * self.num_envs,
            "mpjpe_m": [0.05] * self.num_envs,
        }
        self.step_index += 1
        return telemetry


def _write_safe_test_checkpoint(path: Path) -> None:
    checkpoint = build_safe_promotion_checkpoint(
        policy_state_dict={"test.weight": torch.zeros(1)},
        global_step=1,
        metadata=make_artifact_metadata(
            history_length=10,
            observation_layout=OBS_LAYOUT_PADDED_IL29,
            checkpoint_stage="trained",
        ),
        training_evidence={
            "motion_dataset": {
                "schema_version": 1,
                "source_archive": dict(
                    artifact.MOTION_DATASET_SOURCE_ARCHIVE
                ),
                "processed": {
                    "root_relpath": (
                        "data/motion_lib_bones_seed/robot_filtered"
                    ),
                    "file_count": 1,
                    "total_bytes": 1,
                    "manifest_sha256": "f" * 64,
                },
            }
        },
    )
    torch.save(checkpoint, path)


def _approve_runner(monkeypatch):
    config = json.loads(json.dumps(artifact._sim_validation_config()))  # noqa: SLF001
    config["producer"]["runner_sha256"] = artifact.sha256_file(runner._RUNNER_PATH)  # noqa: SLF001
    config["producer"]["promotion_enabled"] = True
    monkeypatch.setattr(artifact, "_sim_validation_config", lambda: config)
    return config


def _fake_checkpoint_output():
    return {
        "performed": True,
        "before_simulator_launch": True,
        "device": "cpu",
        "dtype": "float32",
        "encoder_input_shape": [1, 267],
        "token_shape": [1, 64],
        "decoder_input_shape": [1, 994],
        "action_shape": [1, 23],
        "policy_state_sha256": "a" * 64,
        "token_sha256": "b" * 64,
        "action_sha256": "c" * 64,
        "action_values": [0.0] * 23,
    }


def _generate(tmp_path: Path):
    checkpoint = tmp_path / "trained.pt"
    _write_safe_test_checkpoint(checkpoint)
    report_path = tmp_path / "sim-report.json"
    backend = FakeIsaacLabBackend()
    report = runner._generate_non_promotable_test_report(  # noqa: SLF001
        backend,
        checkpoint_path=checkpoint,
        output_path=report_path,
    )
    return backend, checkpoint, report_path, report


def _validate_synthetic_trace(report_path: Path, report, index: int = 0):
    run = report["runs"][index]
    return artifact._validate_raw_trace(  # noqa: SLF001
        run["trace"],
        report_path=report_path,
        scenario=run["scenario"],
        seed=run["seed"],
        episodes=run["episodes"],
        disturbance_scale=run["disturbance_scale"],
        checkpoint_sha256=report["checkpoint_sha256"],
        producer=report["producer"],
        simulator=report["simulator"],
        material_provenance=report["material_provenance"],
        config=runner._validation_config(),  # noqa: SLF001
        context=f"synthetic runs[{index}]",
    )


def test_fake_backend_is_rejected_by_official_evidence_producer(tmp_path):
    checkpoint = tmp_path / "trained.pt"
    _write_safe_test_checkpoint(checkpoint)
    report_path = tmp_path / "official-report.json"
    with pytest.raises(TypeError, match="exact verified IsaacLabBackend"):
        runner.generate_validation_report(
            FakeIsaacLabBackend(),
            checkpoint_path=checkpoint,
            output_path=report_path,
        )
    assert not report_path.exists()
    assert not report_path.with_name("official-report.traces").exists()


def test_official_evidence_requires_exact_isaaclab_version():
    assert (
        runner._require_supported_isaaclab_version("2.3.2")  # noqa: SLF001
        == "2.3.2"
    )
    with pytest.raises(RuntimeError, match="requires Isaac Lab 2.3.2"):
        runner._require_supported_isaaclab_version("2.3.1")  # noqa: SLF001
    with pytest.raises(RuntimeError, match="requires Isaac Lab 2.3.2"):
        runner._require_supported_isaaclab_version(None)  # noqa: SLF001


def test_fake_backend_produces_only_non_promotable_test_fixture(tmp_path, monkeypatch):
    backend, checkpoint, report_path, report = _generate(tmp_path)
    assert report["kind"] == runner.NON_PROMOTABLE_TEST_REPORT_KIND
    assert report["producer"]["kind"] == runner.NON_PROMOTABLE_TEST_PRODUCER_KIND
    assert report["simulator"]["name"] == "SyntheticTestBackend"
    assert len(report["runs"]) == 9
    assert len(backend.disturbances) == 9
    assert sum(run["episodes"] for run in report["runs"]) == 198
    assert all(run["steps"] == 5500 for run in report["runs"])
    assert all(run["termination_count"] == 0 for run in report["runs"])
    assert all(run["max_recovery_time_s"] <= 0.4 for run in report["runs"])
    assert len(list(report_path.with_name("sim-report.traces").glob("*.json"))) == 9

    _approve_runner(monkeypatch)
    report_bytes = report_path.read_bytes()
    with pytest.raises(ValueError, match="simulation report kind must be"):
        artifact.validate_simulation_report(
            artifact.load_strict_json(report_path),
            checkpoint_sha256=artifact.sha256_file(checkpoint),
            report_sha256=artifact.sha256_bytes(report_bytes),
            report_payload_sha256=artifact.sha256_bytes(
                artifact.canonical_json_bytes(artifact.load_strict_json(report_path))
            ),
            report_path=report_path,
            checkpoint_motion_dataset={
                "schema_version": 1,
                "source_archive": dict(
                    artifact.MOTION_DATASET_SOURCE_ARCHIVE
                ),
                "processed": {
                    "root_relpath": (
                        "data/motion_lib_bones_seed/robot_filtered"
                    ),
                    "file_count": 1,
                    "total_bytes": 1,
                    "manifest_sha256": "f" * 64,
                },
            },
        )


def test_trace_tamper_and_missing_trace_are_rejected(tmp_path):
    _backend, _checkpoint, report_path, report = _generate(tmp_path)
    first_trace = report_path.parent / report["runs"][0]["trace"]["file"]
    first_trace.write_bytes(first_trace.read_bytes() + b" ")
    with pytest.raises(ValueError, match="trace SHA-256 mismatch"):
        _validate_synthetic_trace(report_path, report)

    first_trace.unlink()
    with pytest.raises(ValueError, match="trace file is missing"):
        _validate_synthetic_trace(report_path, report)


def test_trace_path_traversal_and_external_symlink_are_rejected(tmp_path):
    traversal_dir = tmp_path / "traversal"
    traversal_dir.mkdir()
    _backend, _checkpoint, report_path, report = _generate(traversal_dir)
    report["runs"][0]["trace"]["file"] = "../outside.json"
    with pytest.raises(ValueError, match="trace.file must be"):
        _validate_synthetic_trace(report_path, report)

    symlink_dir = tmp_path / "symlink"
    symlink_dir.mkdir()
    _backend, checkpoint, report_path, report = _generate(symlink_dir)
    trace_path = report_path.parent / report["runs"][0]["trace"]["file"]
    external_trace = tmp_path / "external-trace.json"
    trace_path.replace(external_trace)
    try:
        trace_path.symlink_to(external_trace)
    except OSError:
        pytest.skip("creating symlinks is unavailable on this Windows host")
    with pytest.raises(ValueError, match="escapes report directory"):
        _validate_synthetic_trace(report_path, report)


def test_cli_dry_run_and_validate_only(tmp_path, monkeypatch, capsys):
    _backend, checkpoint, report_path, _report = _generate(tmp_path)
    dry_output = tmp_path / "future-report.json"
    monkeypatch.setattr(
        runner,
        "_checkpoint_output_dry_run",
        lambda _checkpoint: _fake_checkpoint_output(),
    )
    assert (
        runner.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(dry_output),
                "--dry-run",
            ]
        )
        == 0
    )
    dry_payload = json.loads(capsys.readouterr().out)
    assert dry_payload["will_launch_isaaclab"] is False
    assert dry_payload["episodes_per_seed"] == 22
    assert dry_payload["steps_per_episode"] == 250
    output = dry_payload["checkpoint_output"]
    assert output["performed"] is True
    assert output["before_simulator_launch"] is True
    assert output["encoder_input_shape"] == [1, 267]
    assert output["token_shape"] == [1, 64]
    assert output["decoder_input_shape"] == [1, 994]
    assert output["action_shape"] == [1, 23]
    assert len(output["action_values"]) == 23

    _approve_runner(monkeypatch)
    with pytest.raises(ValueError, match="simulation report kind must be"):
        runner.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(report_path),
                "--validate-only",
            ]
        )


def test_checkpoint_output_dry_run_requires_exact_finite_true23_pair(monkeypatch):
    class Encoder(torch.nn.Module):
        def forward(self, value):
            return torch.zeros((value.shape[0], 64), dtype=torch.float32)

    class Decoder(torch.nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            self.output_dim = output_dim

        def forward(self, value):
            assert value.shape == (1, 994)
            return torch.arange(
                self.output_dim,
                dtype=torch.float32,
            ).reshape(1, self.output_dim)

    monkeypatch.setattr(
        runner,
        "build_true23_policy_pair",
        lambda _checkpoint: (Encoder(), Decoder(23), "a" * 64),
    )
    monkeypatch.setattr(
        runner,
        "validate_training_checkpoint_records",
        lambda *_args, **_kwargs: None,
    )
    result = runner._checkpoint_output_dry_run({"state": {"global_step": 50}})
    assert result["action_shape"] == [1, 23]
    assert result["action_values"] == [float(value) for value in range(23)]

    monkeypatch.setattr(
        runner,
        "build_true23_policy_pair",
        lambda _checkpoint: (Encoder(), Decoder(29), "a" * 64),
    )
    with pytest.raises(ValueError, match=r"float32 \[1,23\]"):
        runner._checkpoint_output_dry_run({"state": {"global_step": 50}})


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("manager_env", "config", "sim_dt"), 0.01),
        (
            (
                "manager_env",
                "terminations",
                "anchor_pos",
                "params",
                "threshold",
            ),
            9.0,
        ),
        (
            ("manager_env", "observations", "policy", "joint_pos", "func"),
            "gear_sonic.envs.manager_env.mdp:joint_pos_rel",
        ),
        (
            (
                "algo",
                "config",
                "actor",
                "backbone",
                "decoders",
                "g1_dyn",
                "params",
                "module_config_dict",
                "layer_config",
                "hidden_dims",
            ),
            [8],
        ),
    ),
)
def test_material_runtime_config_mutations_are_rejected(path, value):
    config = copy.deepcopy(_approved_test_runtime_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="approved true23 runtime contract"):
        artifact.validate_runtime_config_snapshot(config)


def test_cli_rejects_all_arbitrary_eval_overrides(tmp_path):
    checkpoint = tmp_path / "trained.pt"
    _write_safe_test_checkpoint(checkpoint)
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(tmp_path / "report.json"),
                "--dry-run",
                "--eval-arg",
                "++manager_env.config.sim_dt=0.01",
            ]
        )


def test_cli_rejects_unsafe_resume_checkpoint_before_launch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"not a safe promotion checkpoint")
    report = tmp_path / "report.json"

    def unexpected_launch(*_args, **_kwargs):
        raise AssertionError("unsafe checkpoint must be rejected before subprocess launch")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_launch)
    with pytest.raises(ValueError, match="safe weights-only true23 artifact"):
        runner.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(report),
            ]
        )


def test_normal_cli_uses_only_fixed_hydra_overrides(tmp_path, monkeypatch):
    checkpoint = tmp_path / "trained.pt"
    _write_safe_test_checkpoint(checkpoint)
    report = tmp_path / "report.json"
    captured = {}
    events = []

    def tracked_output_dry_run(checkpoint_payload):
        assert runner.checkpoint_stage(checkpoint_payload) == "trained"
        events.append("output_dry_run")
        return _fake_checkpoint_output()

    def fake_run(command, *, cwd, env, check):
        assert events == ["output_dry_run"]
        events.append("simulator_launch")
        captured.update(command=command, cwd=cwd, env=env, check=check)
        Path(env[runner.OUTPUT_ENV_VAR]).write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner, "_checkpoint_output_dry_run", tracked_output_dry_run)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert (
        runner.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(report),
            ]
        )
        == 0
    )
    hydra_args = captured["command"][2:]
    assert events == ["output_dry_run", "simulator_launch"]
    assert hydra_args == [
        f"checkpoint={checkpoint.resolve().as_posix()}",
        "+num_envs=22",
        "+headless=true",
        "+run_eval_loop=false",
        "+eval_callbacks=[]",
        "+true23_sim_validation=true",
    ]
