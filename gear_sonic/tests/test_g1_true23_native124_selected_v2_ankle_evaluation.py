from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
    ACTION_DIM,
    ANKLE_HARDWARE_ROWS,
    CRITIC_OBSERVATION_DIM,
    OBSERVATION_DIM,
    SelectedNative124Actor,
    load_selected_v2_ankle_adaptation,
    sha256_file,
)
import gear_sonic.utils.g1_true23_native124_selected_v2_ankle_evaluation as evaluation

REPO_ROOT = Path(__file__).resolve().parents[2]


def _step_evidence(*, reward: float = 1.0, count: int = 0) -> evaluation.StepEvidence:
    return evaluation.StepEvidence(
        reward=float(reward),
        scalars={key: float(index + 1) for index, key in enumerate(evaluation._SAFETY_SCALAR_KEYS)},
        counts={key: count for key in evaluation._SAFETY_COUNT_KEYS},
        reward_rates={"alive": 5.0, "motion_ankle_pos": 1.25},
    )


def _actor_state(actor: SelectedNative124Actor) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous().clone() for key, value in actor.state_dict().items()}


def test_warm_evaluation_request_requires_lower_hash_and_exclusive_evidence_path(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "artifacts" / "g1_true23"
    evidence.mkdir(parents=True)
    checkpoint = tmp_path / "warm.pt"
    checkpoint.write_bytes(b"warm")
    request = evaluation.WarmEvaluationRequest(
        repository_root=tmp_path,
        warm_checkpoint=checkpoint,
        expected_warm_sha256="0" * 64,
        output=Path("artifacts/g1_true23/report.json"),
    )
    assert request.output_path == evidence / "report.json"
    with pytest.raises(ValueError, match="64 lowercase"):
        evaluation.WarmEvaluationRequest(
            repository_root=tmp_path,
            warm_checkpoint=checkpoint,
            expected_warm_sha256="A" * 64,
            output=Path("artifacts/g1_true23/report.json"),
        )
    outside = evaluation.WarmEvaluationRequest(
        repository_root=tmp_path,
        warm_checkpoint=checkpoint,
        expected_warm_sha256="0" * 64,
        output=Path("outside.json"),
    )
    with pytest.raises(ValueError, match="must stay under"):
        _ = outside.output_path


def test_actor_delta_accepts_only_trainable_output_rows() -> None:
    source = _actor_state(SelectedNative124Actor())
    adapted = {key: value.clone() for key, value in source.items()}
    adapted["mlp.6.weight"][list(ANKLE_HARDWARE_ROWS), :3] += 0.125
    adapted["mlp.6.bias"][list(ANKLE_HARDWARE_ROWS)] -= 0.25

    report = evaluation.compare_actor_to_selected_source(adapted, source)

    assert report["frozen_state_byte_exact"] is True
    assert report["trainable_rows_changed"] is True
    assert report["trainable_changed_element_count"] == 16
    assert report["trainable_maximum_absolute_delta"] == pytest.approx(0.25)
    bad = {key: value.clone() for key, value in adapted.items()}
    bad["mlp.6.bias"][0] += 1.0
    with pytest.raises(ValueError, match="changed frozen actor rows"):
        evaluation.compare_actor_to_selected_source(bad, source)
    bad_full = {key: value.clone() for key, value in adapted.items()}
    bad_full["mlp.0.bias"][0] += 1.0
    with pytest.raises(ValueError, match="changed frozen actor tensor"):
        evaluation.compare_actor_to_selected_source(bad_full, source)


def test_rollout_accumulator_tracks_finite_task_and_safety_evidence() -> None:
    accumulator = evaluation.RolloutEvidenceAccumulator()
    accumulator.add(_step_evidence(reward=1.0, count=0))
    accumulator.add(_step_evidence(reward=3.0, count=2))

    report = accumulator.report()

    assert report["transition_count"] == 2
    assert report["reward"]["mean"] == 2.0
    assert report["safety_count_totals"]["raw_clip_coordinate_count"] == 2
    assert report["hard_safety_violation_count"] == 12
    assert report["soft_safety_warning_count"] == 4
    assert report["reward_rate_by_term"]["alive"]["mean"] == 5.0
    with pytest.raises(ValueError, match="reward term set changed"):
        accumulator.add(
            evaluation.StepEvidence(
                reward=0.0,
                scalars={key: 0.0 for key in evaluation._SAFETY_SCALAR_KEYS},
                counts={key: 0 for key in evaluation._SAFETY_COUNT_KEYS},
                reward_rates={"different": 0.0},
            )
        )
    with pytest.raises(ValueError, match="finite float"):
        _step_evidence(reward=float("nan"))


def test_qualification_requires_exact_full_timeout_and_nonzero_adaptation() -> None:
    accumulator = evaluation.RolloutEvidenceAccumulator()
    accumulator.add(_step_evidence(reward=1.0, count=0))
    first_done = {
        "policy_transition": evaluation.EXPECTED_TIMEOUT_POLICY_TRANSITION,
        "q9_before": evaluation.EXPECTED_TIMEOUT_Q9,
        "episode_length_pre_reset": evaluation.EPISODE_STEPS,
        "termination_names": ["time_out"],
        "is_timeout": True,
        "is_terminated": False,
    }
    result = evaluation._qualification(
        first_done=first_done,
        policy_summary=accumulator.report(),
        actor_delta={"trainable_rows_changed": True, "frozen_state_byte_exact": True},
        attempted=evaluation.EXPECTED_POLICY_TRANSITIONS_ATTEMPTED,
        deterministic_actor_checks=evaluation.EXPECTED_POLICY_TRANSITIONS_ATTEMPTED,
    )
    assert result["candidate_gate_passed"] is True
    assert result["strict_nominal_gate_passed"] is True
    early = dict(first_done)
    early["q9_before"] = 58
    early_result = evaluation._qualification(
        first_done=early,
        policy_summary=accumulator.report(),
        actor_delta={"trainable_rows_changed": True, "frozen_state_byte_exact": True},
        attempted=evaluation.EXPECTED_POLICY_TRANSITIONS_ATTEMPTED,
        deterministic_actor_checks=evaluation.EXPECTED_POLICY_TRANSITIONS_ATTEMPTED,
    )
    assert early_result["candidate_gate_passed"] is False
    assert early_result["survived_beyond_selected_source_q9_58_failure"] is False


def test_report_writer_is_exclusive_and_failure_report_is_quarantined(tmp_path: Path) -> None:
    evidence = tmp_path / "artifacts" / "g1_true23"
    evidence.mkdir(parents=True)
    checkpoint = tmp_path / "warm.pt"
    checkpoint.write_bytes(b"warm")
    request = evaluation.WarmEvaluationRequest(
        repository_root=tmp_path,
        warm_checkpoint=checkpoint,
        expected_warm_sha256="0" * 64,
        output=Path("artifacts/g1_true23/report.json"),
    )
    report = evaluation.failure_report(RuntimeError("boom"), request)

    output = evaluation.write_warm_evaluation_report_new(request, report)

    assert output.is_file()
    assert '"candidate_quarantined": true' in output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluation.write_warm_evaluation_report_new(request, report)


def test_preflight_rejects_wrong_warm_hash_before_simulator(tmp_path: Path) -> None:
    checkpoint = tmp_path / "warm.pt"
    checkpoint.write_bytes(b"not a warm checkpoint")
    request = evaluation.WarmEvaluationRequest(
        repository_root=REPO_ROOT,
        warm_checkpoint=checkpoint,
        expected_warm_sha256="0" * 64,
        output=Path(f"artifacts/g1_true23/{tmp_path.name}_never_written.json"),
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluation.preflight_warm_evaluation(request)


def test_evaluator_runner_uses_exact_public_smoke_profile_for_inference_only() -> None:
    motion = REPO_ROOT / evaluation.DAD_DANCE_RELATIVE_PATH
    output = REPO_ROOT / "artifacts/g1_true23/unused_evaluation_report.json"

    cfg = evaluation._evaluation_agent_config(REPO_ROOT, motion, output)

    assert cfg["run_name"] == f"smoke_seed{evaluation.FIXED_SEED}"
    assert cfg["num_steps_per_env"] == 8
    assert cfg["max_iterations"] == evaluation.SMOKE_ITERATIONS
    assert cfg["algorithm"]["learning_rate"] == evaluation.FIXED_LEARNING_RATE
    assert cfg["algorithm"]["schedule"] == "fixed"
    assert cfg["obs_groups"] == {"actor": ["actor"], "critic": ["critic"]}


def _build_actual_runner(seed: int):
    pytest.importorskip("mjlab")
    pytest.importorskip("rsl_rl")
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict

    torch.manual_seed(seed)
    observations = TensorDict(
        {
            "actor": torch.randn(2, OBSERVATION_DIM),
            "critic": torch.randn(2, CRITIC_OBSERVATION_DIM),
        },
        batch_size=[2],
    )
    groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = MLPModel(
        observations,
        groups,
        "actor",
        ACTION_DIM,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    critic = MLPModel(
        observations,
        groups,
        "critic",
        1,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )
    storage = RolloutStorage("rl", 2, 1, observations, [ACTION_DIM], "cpu")
    algorithm = PPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=evaluation.FIXED_LEARNING_RATE,
        schedule="fixed",
        desired_kl=None,
        device="cpu",
    )
    runner = object.__new__(MjlabOnPolicyRunner)
    runner.alg = algorithm
    runner.current_learning_iteration = 0
    runner.completed_update_count = 0
    runner._training_state_poisoned = False
    return runner


def test_actual_rsl_hash_locked_warm_load_without_simulator(tmp_path: Path) -> None:
    source_runner = _build_actual_runner(410)
    from gear_sonic.trl.mjlab.native124_selected_v2_ankle_rsl import (
        configure_selected_v2_ankle_rsl_runner,
    )

    source_integration = configure_selected_v2_ankle_rsl_runner(
        source_runner,
        repository_root=REPO_ROOT,
    )
    with torch.no_grad():
        source_integration.actor.mlp[6].bias[list(ANKLE_HARDWARE_ROWS)] += 1.0e-4
    warm = tmp_path / "warm.pt"
    publication = source_integration.save_warm_restart(warm)
    output_suffix = hashlib.sha256(str(tmp_path).encode("utf-8")).hexdigest()[:16]
    request = evaluation.WarmEvaluationRequest(
        repository_root=REPO_ROOT,
        warm_checkpoint=warm,
        expected_warm_sha256=publication.sha256,
        output=Path(f"artifacts/g1_true23/warm_eval_preflight_{output_suffix}.json"),
    )
    preflight = evaluation.preflight_warm_evaluation(request)
    assert preflight["ready"] is True
    assert preflight["fixed"]["runner_profile"] == "smoke_for_inference_only"
    assert preflight["rsl_runtime"] == evaluation.resolve_rsl_runtime_binding()
    target_runner = _build_actual_runner(411)

    loaded = evaluation.configure_hash_locked_warm_evaluation_runner(
        target_runner,
        repository_root=REPO_ROOT,
        warm_checkpoint=warm,
        expected_warm_sha256=publication.sha256,
    )

    source_bundle = load_selected_v2_ankle_adaptation(
        loaded.lineage.checkpoint_path,
        learning_rate=evaluation.FIXED_LEARNING_RATE,
        device="cpu",
    )
    delta = evaluation.compare_actor_to_selected_source(
        loaded.actor.state_dict(),
        source_bundle.actor.state_dict(),
    )
    assert delta["trainable_rows_changed"] is True
    assert target_runner.alg.storage.step == 0
    assert target_runner.alg.optimizer.state == {}
    assert sha256_file(warm) == publication.sha256


def test_real_cuda_one_warmup_transition_captures_strict_metric_evidence() -> None:
    pytest.importorskip("mjlab")
    if not torch.cuda.is_available():
        pytest.skip("CUDA MJLab runtime required")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sim import TorchArray
    from mjlab.utils.torch import configure_torch_backends

    from gear_sonic.envs.mjlab.native124_selected_v2_ankle_task import (
        make_native124_selected_v2_ankle_task_env_cfg,
    )
    from gear_sonic.envs.mjlab.native124_selected_v2_causal_adaptation import (
        Native124SelectedV2CausalAdaptationWrapper,
        prime_native124_selected_v2_causal_adaptation_environment,
    )

    torch.manual_seed(evaluation.FIXED_SEED)
    torch.cuda.manual_seed_all(evaluation.FIXED_SEED)
    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    motion = REPO_ROOT / evaluation.DAD_DANCE_RELATIVE_PATH
    cfg = make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=str(motion),
        num_envs=1,
        play=False,
    )
    cfg.seed = evaluation.FIXED_SEED
    env = ManagerBasedRlEnv(cfg=cfg, device=evaluation.DEVICE)
    try:
        wrapped = Native124SelectedV2CausalAdaptationWrapper(env, clip_actions=None)
        prime_native124_selected_v2_causal_adaptation_environment(wrapped)
        assert env.common_step_counter == 0
        force_range = env.sim.model.actuator_forcerange
        assert type(force_range) is TorchArray
        assert force_range.shape == (1, ACTION_DIM, 2)
        proof = evaluation.prove_warmup_action_equivalence(device=evaluation.DEVICE)
        action = torch.as_tensor(
            proof["selected_raw_action_hardware"],
            dtype=torch.float32,
            device=evaluation.DEVICE,
        ).reshape(1, ACTION_DIM)
        _, rewards, dones, _ = wrapped.step(action)
        assert rewards.shape == (1,)
        assert int(dones[0].detach().cpu().item()) == 0
        assert env.common_step_counter == 1
        velocity_contract = evaluation.load_composite_mjlab_contract(REPO_ROOT)
        velocity_limits = np.asarray(
            velocity_contract["nominal_gate"]["velocity_limit_hardware_radps"],
            dtype=np.float32,
        )

        evidence = evaluation._step_evidence(env, velocity_limits)

        assert evidence.scalars["maximum_actuator_force_ratio"] >= 0.0
        assert evidence.counts["actuator_force_hard_limit_coordinate_count"] >= 0
        assert evidence.counts["raw_clip_coordinate_count"] == 0
        assert all(np.isfinite(value) for value in evidence.scalars.values())
        assert all(np.isfinite(value) for value in evidence.reward_rates.values())
    finally:
        env.close()
