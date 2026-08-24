from __future__ import annotations

from pathlib import Path

import pytest
import torch

try:
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict
except ImportError:
    pytest.skip("installed WSL MJLab/RSL-RL runtime required", allow_module_level=True)

from gear_sonic.trl.mjlab.native124_selected_v2_ankle_rsl import (
    CRITIC_OBSERVATION_DIM,
    PARITY_ATOL,
    SELECTED_LOAD_CONFIG,
    configure_selected_v2_ankle_rsl_runner,
)
from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
    ACTION_DIM,
    ANKLE_HARDWARE_ROWS,
    LEFT_KNEE_EVIDENCE_GATE,
    LEFT_KNEE_HARDWARE_ROW,
    OBSERVATION_DIM,
    AnkleRowConfig,
    FreshAdam,
    sha256_file,
    tensor_state_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTOR_GROUP = "native124_causal"


def _build_actual_runner(seed: int = 123) -> tuple[MjlabOnPolicyRunner, TensorDict]:
    torch.manual_seed(seed)
    observations = TensorDict(
        {
            ACTOR_GROUP: torch.randn(8, OBSERVATION_DIM),
            "critic": torch.randn(8, CRITIC_OBSERVATION_DIM),
        },
        batch_size=[8],
    )
    observation_groups = {
        "actor": [ACTOR_GROUP],
        "critic": ["critic"],
    }
    actor = MLPModel(
        observations,
        observation_groups,
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
        observation_groups,
        "critic",
        1,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )
    storage = RolloutStorage("rl", 8, 1, observations, [ACTION_DIM], "cpu")
    algorithm = PPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-4,
        schedule="fixed",
        desired_kl=None,
        device="cpu",
    )
    runner = object.__new__(MjlabOnPolicyRunner)
    runner.alg = algorithm
    runner.current_learning_iteration = 73
    runner.completed_update_count = 73
    runner._training_state_poisoned = False
    return runner, observations


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes(order="C")


def _frozen_rows(rows: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(index for index in range(ACTION_DIM) if index not in rows)


def test_actual_rsl_actor_only_load_keeps_fresh_256_critic_and_resets_optimizer() -> None:
    runner, observations = _build_actual_runner()
    critic_before = {key: value.detach().clone() for key, value in runner.alg.critic.state_dict().items()}
    source_optimizer = runner.alg.optimizer

    integration = configure_selected_v2_ankle_rsl_runner(
        runner,
        repository_root=REPO_ROOT,
    )

    assert integration.parity["passed"] is True
    assert integration.parity["case_count"] == 5
    assert integration.parity["max_absolute_error"] <= PARITY_ATOL
    assert integration.parity["post_sample_mask_passed"] is True
    assert integration.parity["post_sample_mask_max_absolute_error"] <= PARITY_ATOL
    assert integration.actor.obs_dim == OBSERVATION_DIM
    assert integration.actor.obs_groups == [ACTOR_GROUP]
    assert integration.critic.obs_dim == CRITIC_OBSERVATION_DIM
    assert integration.critic.obs_groups == ["critic"]
    assert integration.fresh_critic_state_sha256 == tensor_state_sha256(critic_before)
    assert integration.fresh_critic_state_sha256 != integration.lineage.critic_state_sha256
    assert all(torch.equal(integration.critic.state_dict()[key], value) for key, value in critic_before.items())
    assert runner.alg.optimizer is not source_optimizer
    assert type(runner.alg.optimizer) is FreshAdam
    assert runner.alg.optimizer.state == {}
    assert len(runner.alg.optimizer.param_groups[0]["params"]) == 10
    assert runner.current_learning_iteration == 0
    assert runner.completed_update_count == 0
    assert SELECTED_LOAD_CONFIG == {
        "actor": True,
        "critic": False,
        "iteration": False,
        "optimizer": False,
        "rnd": False,
    }

    normalizer_before = {
        key: _bytes(value) for key, value in integration.actor.obs_normalizer.state_dict().items()
    }
    runner.alg.train_mode()
    integration.actor.update_normalization(observations)
    integration.actor.obs_normalizer.update(observations[ACTOR_GROUP])
    assert integration.actor.obs_normalization is False
    assert integration.actor.obs_normalizer.training is False
    assert {
        key: _bytes(value) for key, value in integration.actor.obs_normalizer.state_dict().items()
    } == normalizer_before
    integration.assert_frozen_invariants()


def test_actual_rsl_stochastic_noise_and_synthetic_update_touch_only_ankle_rows() -> None:
    runner, observations = _build_actual_runner(seed=456)
    integration = configure_selected_v2_ankle_rsl_runner(
        runner,
        repository_root=REPO_ROOT,
    )
    actor = integration.actor
    critic = integration.critic
    frozen = _frozen_rows(ANKLE_HARDWARE_ROWS)
    actor_state_before = {key: _bytes(value) for key, value in actor.state_dict().items()}
    actor_output_weight_before = actor.mlp[6].weight.detach().clone()
    actor_output_bias_before = actor.mlp[6].bias.detach().clone()
    critic_weight_before = critic.mlp[6].weight.detach().clone()

    deterministic = actor(observations, stochastic_output=False)
    torch.manual_seed(991)
    sampled = actor(observations, stochastic_output=True)
    old_mean = actor.output_mean.detach().clone()
    old_std = actor.output_std.detach().clone()
    old_log_prob_per_coordinate = torch.distributions.Normal(old_mean, old_std).log_prob(sampled)
    assert torch.equal(sampled[:, list(frozen)], deterministic[:, list(frozen)])
    ankle_delta = sampled[:, list(ANKLE_HARDWARE_ROWS)] - deterministic[:, list(ANKLE_HARDWARE_ROWS)]
    assert torch.count_nonzero(ankle_delta) > 0

    runner.alg.optimizer.zero_grad()
    loss = actor(observations, stochastic_output=False).sum() + critic(observations).sum()
    loss.backward()
    assert torch.count_nonzero(actor.mlp[6].weight.grad[list(frozen)]) == 0
    assert torch.count_nonzero(actor.mlp[6].bias.grad[list(frozen)]) == 0
    runner.alg.optimizer.step()

    assert integration.optimizer_steps == 1
    assert not torch.equal(critic.mlp[6].weight, critic_weight_before)
    assert not torch.equal(
        actor.mlp[6].weight[list(ANKLE_HARDWARE_ROWS)],
        actor_output_weight_before[list(ANKLE_HARDWARE_ROWS)],
    )
    assert _bytes(actor.mlp[6].weight[list(frozen)]) == _bytes(actor_output_weight_before[list(frozen)])
    assert _bytes(actor.mlp[6].bias[list(frozen)]) == _bytes(actor_output_bias_before[list(frozen)])
    for key, expected in actor_state_before.items():
        if key not in {"mlp.6.weight", "mlp.6.bias"}:
            assert _bytes(actor.state_dict()[key]) == expected

    actor(observations, stochastic_output=True)
    new_mean = actor.output_mean.detach().clone()
    new_std = actor.output_std.detach().clone()
    new_log_prob_per_coordinate = torch.distributions.Normal(new_mean, new_std).log_prob(sampled)
    assert torch.equal(new_mean[:, list(frozen)], old_mean[:, list(frozen)])
    assert torch.equal(new_std[:, list(frozen)], old_std[:, list(frozen)])
    assert torch.equal(
        new_log_prob_per_coordinate[:, list(frozen)],
        old_log_prob_per_coordinate[:, list(frozen)],
    )
    integration.assert_frozen_invariants()


def test_actual_rsl_ppo_runs_one_synthetic_storage_update_without_env_rollout() -> None:
    runner, observations = _build_actual_runner(seed=610)
    integration = configure_selected_v2_ankle_rsl_runner(
        runner,
        repository_root=REPO_ROOT,
    )
    frozen = _frozen_rows(ANKLE_HARDWARE_ROWS)
    actor_normalizer_before = {
        key: _bytes(value) for key, value in integration.actor.obs_normalizer.state_dict().items()
    }
    actor_frozen_weight_before = integration.actor.mlp[6].weight[list(frozen)].detach().clone()

    actions = runner.alg.act(observations)
    assert torch.equal(
        actions[:, list(frozen)],
        integration.actor.output_mean[:, list(frozen)],
    )
    runner.alg.process_env_step(
        observations,
        torch.linspace(-0.5, 0.5, 8),
        torch.zeros(8, dtype=torch.bool),
        {},
    )
    runner.alg.compute_returns(observations)
    losses = runner.alg.update()

    assert set(losses) == {"entropy", "surrogate", "value"}
    assert integration.optimizer_steps == 1
    assert runner.alg.optimizer.state
    assert torch.equal(
        integration.actor.mlp[6].weight[list(frozen)],
        actor_frozen_weight_before,
    )
    assert {
        key: _bytes(value) for key, value in integration.actor.obs_normalizer.state_dict().items()
    } == actor_normalizer_before
    integration.assert_frozen_invariants()


def test_actual_rsl_warm_restart_omits_optimizer_and_reloads_with_fresh_adam(
    tmp_path: Path,
) -> None:
    runner, observations = _build_actual_runner(seed=777)
    integration = configure_selected_v2_ankle_rsl_runner(
        runner,
        repository_root=REPO_ROOT,
    )
    runner.alg.optimizer.zero_grad()
    (integration.actor(observations).sum() + integration.critic(observations).sum()).backward()
    runner.alg.optimizer.step()
    actor_saved = {key: value.detach().clone() for key, value in integration.actor.state_dict().items()}
    critic_saved = {key: value.detach().clone() for key, value in integration.critic.state_dict().items()}
    output = tmp_path / "rsl_ankle_warm.pt"
    publication = runner.save(str(output))
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert set(payload) == {"actor_state_dict", "critic_state_dict", "metadata"}
    assert "optimizer_state_dict" not in payload
    assert publication.sha256 == sha256_file(output)
    with pytest.raises(RuntimeError, match="stock RSL load is forbidden"):
        runner.load(str(output))

    resumed_runner, _ = _build_actual_runner(seed=778)
    resumed = configure_selected_v2_ankle_rsl_runner(
        resumed_runner,
        repository_root=REPO_ROOT,
        restart_path=output,
        expected_restart_sha256=publication.sha256,
    )
    assert resumed.prior_completed_optimizer_steps == 1
    assert resumed.optimizer_steps == 0
    assert resumed.completed_optimizer_steps_total == 1
    assert resumed_runner.current_learning_iteration == 0
    assert resumed_runner.alg.optimizer.state == {}
    assert type(resumed_runner.alg.optimizer) is FreshAdam
    assert all(torch.equal(resumed.actor.state_dict()[key], value) for key, value in actor_saved.items())
    assert all(torch.equal(resumed.critic.state_dict()[key], value) for key, value in critic_saved.items())

    with pytest.raises(ValueError, match="warm restart SHA-256 mismatch"):
        bad_runner, _ = _build_actual_runner(seed=779)
        configure_selected_v2_ankle_rsl_runner(
            bad_runner,
            repository_root=REPO_ROOT,
            restart_path=output,
            expected_restart_sha256="0" * 64,
        )


def test_actual_rsl_left_knee_gate_changes_sample_mask_and_frozen_drift_poisons() -> None:
    runner, observations = _build_actual_runner(seed=880)
    config = AnkleRowConfig(
        include_left_knee=True,
        left_knee_gate=LEFT_KNEE_EVIDENCE_GATE,
    )
    integration = configure_selected_v2_ankle_rsl_runner(
        runner,
        repository_root=REPO_ROOT,
        config=config,
    )
    deterministic = integration.actor(observations)
    sampled = integration.actor(observations, stochastic_output=True)
    rows = (LEFT_KNEE_HARDWARE_ROW, *ANKLE_HARDWARE_ROWS)
    frozen = _frozen_rows(rows)
    assert torch.equal(sampled[:, list(frozen)], deterministic[:, list(frozen)])
    assert torch.count_nonzero(sampled[:, list(rows)] - deterministic[:, list(rows)]) > 0

    frozen_row = frozen[0]
    original = integration.actor.mlp[6].bias[frozen_row].detach().clone()
    with torch.no_grad():
        integration.actor.mlp[6].bias[frozen_row].add_(1.0)
    with pytest.raises(RuntimeError, match="frozen actor output bias rows changed"):
        runner.alg.optimizer.step()
    assert torch.equal(integration.actor.mlp[6].bias[frozen_row], original)
    assert runner._native124_selected_v2_ankle_poisoned is True
    assert runner._training_state_poisoned is True
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.alg.optimizer.step()
