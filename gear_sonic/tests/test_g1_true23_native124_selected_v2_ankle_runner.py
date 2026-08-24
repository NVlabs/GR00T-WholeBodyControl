from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
    ACTION_DIM,
    ANKLE_HARDWARE_ROWS,
    BROAD_ACTOR_STATE_SHA256,
    BROAD_CRITIC_STATE_SHA256,
    BROAD_OPTIMIZER_STATE_SHA256,
    CRITIC_OBSERVATION_DIM,
    LEFT_KNEE_EVIDENCE_GATE,
    LEFT_KNEE_HARDWARE_ROW,
    OBSERVATION_DIM,
    SELECTED_ACTOR_STATE_SHA256,
    SELECTED_CHECKPOINT_ITERATION,
    SELECTED_CHECKPOINT_SHA256,
    AnkleRowConfig,
    load_selected_v2_ankle_adaptation,
    safe_tree_sha256,
    sha256_file,
    tensor_state_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTED_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "g1_native124_multimotion"
    / "scaling_all61"
    / "feasible_v1"
    / "interpolated"
    / "model21204_alpha25.pt"
)
BROAD_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "g1_native124_multimotion"
    / "scaling_all61"
    / "feasible_v1"
    / "train_125_lr5e7"
    / "model_21204.pt"
)


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes(order="C")


def _frozen_rows(rows: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(index for index in range(ACTION_DIM) if index not in rows)


def test_selected_checkpoint_proves_interpolated_actor_and_inherited_broad_state() -> None:
    assert sha256_file(SELECTED_CHECKPOINT) == SELECTED_CHECKPOINT_SHA256
    selected = torch.load(SELECTED_CHECKPOINT, map_location="cpu", weights_only=True)
    broad = torch.load(BROAD_CHECKPOINT, map_location="cpu", weights_only=True)

    assert tensor_state_sha256(selected["actor_state_dict"]) == SELECTED_ACTOR_STATE_SHA256
    assert tensor_state_sha256(broad["actor_state_dict"]) == BROAD_ACTOR_STATE_SHA256
    assert SELECTED_ACTOR_STATE_SHA256 != BROAD_ACTOR_STATE_SHA256
    assert any(
        not torch.equal(selected["actor_state_dict"][key], broad["actor_state_dict"][key])
        for key in selected["actor_state_dict"]
    )
    assert tensor_state_sha256(selected["critic_state_dict"]) == BROAD_CRITIC_STATE_SHA256
    assert all(
        torch.equal(selected["critic_state_dict"][key], broad["critic_state_dict"][key])
        for key in selected["critic_state_dict"]
    )
    assert (
        safe_tree_sha256(selected["optimizer_state_dict"], context="selected optimizer")
        == BROAD_OPTIMIZER_STATE_SHA256
    )
    assert (
        safe_tree_sha256(broad["optimizer_state_dict"], context="broad optimizer") == BROAD_OPTIMIZER_STATE_SHA256
    )


def test_loader_uses_exact_topology_and_always_fresh_adam() -> None:
    bundle = load_selected_v2_ankle_adaptation(SELECTED_CHECKPOINT, learning_rate=2.5e-5, load_critic=True)

    assert tuple(bundle.actor.mlp[0].weight.shape) == (512, OBSERVATION_DIM)
    assert tuple(bundle.actor.mlp[2].weight.shape) == (256, 512)
    assert tuple(bundle.actor.mlp[4].weight.shape) == (128, 256)
    assert tuple(bundle.actor.mlp[6].weight.shape) == (ACTION_DIM, 128)
    assert bundle.critic is not None
    assert tuple(bundle.critic.mlp[0].weight.shape) == (512, CRITIC_OBSERVATION_DIM)
    assert tuple(bundle.critic.mlp[6].weight.shape) == (1, 128)
    assert isinstance(bundle.optimizer, torch.optim.Adam)
    assert bundle.optimizer.state == {}
    assert bundle.iteration == 0
    assert bundle.prior_completed_steps == 0
    assert bundle.lineage.source_iteration == SELECTED_CHECKPOINT_ITERATION
    assert bundle.lineage.metadata()["actor_differs_from_broad"] is True
    assert bundle.lineage.metadata()["critic_matches_broad"] is True
    assert bundle.lineage.metadata()["optimizer_matches_broad"] is True
    assert bundle.lineage.metadata()["source_optimizer_loaded"] is False
    assert bundle.optimizer.param_groups[0]["weight_decay"] == 0.0
    selected = torch.load(SELECTED_CHECKPOINT, map_location="cpu", weights_only=True)
    with pytest.raises(RuntimeError, match="optimizer state loading is forbidden"):
        bundle.optimizer.load_state_dict(selected["optimizer_state_dict"])


def test_only_four_hardware_ankle_rows_receive_actor_gradients_or_updates() -> None:
    bundle = load_selected_v2_ankle_adaptation(SELECTED_CHECKPOINT, learning_rate=1.0e-4, load_critic=False)
    output = bundle.actor.mlp[6]
    weight_before = output.weight.detach().clone()
    bias_before = output.bias.detach().clone()
    fully_frozen_before = {
        name: _bytes(parameter)
        for name, parameter in bundle.actor.state_dict().items()
        if not name.startswith("mlp.6.")
    }

    bundle.zero_grad()
    bundle.actor(torch.randn(8, OBSERVATION_DIM)).sum().backward()
    frozen = _frozen_rows(ANKLE_HARDWARE_ROWS)
    assert torch.count_nonzero(output.weight.grad[list(frozen)]) == 0
    assert torch.count_nonzero(output.bias.grad[list(frozen)]) == 0
    assert torch.count_nonzero(output.weight.grad[list(ANKLE_HARDWARE_ROWS)]) > 0
    assert torch.count_nonzero(output.bias.grad[list(ANKLE_HARDWARE_ROWS)]) > 0

    bundle.step()

    assert bundle.iteration == 1
    assert _bytes(output.weight[list(frozen)]) == _bytes(weight_before[list(frozen)])
    assert _bytes(output.bias[list(frozen)]) == _bytes(bias_before[list(frozen)])
    assert not torch.equal(output.weight[list(ANKLE_HARDWARE_ROWS)], weight_before[list(ANKLE_HARDWARE_ROWS)])
    assert not torch.equal(output.bias[list(ANKLE_HARDWARE_ROWS)], bias_before[list(ANKLE_HARDWARE_ROWS)])
    for name, expected in fully_frozen_before.items():
        assert _bytes(bundle.actor.state_dict()[name]) == expected
    bundle.assert_frozen_invariants()


def test_observation_normalizer_trunk_and_std_stay_frozen_in_train_mode() -> None:
    bundle = load_selected_v2_ankle_adaptation(SELECTED_CHECKPOINT, learning_rate=1.0e-4, load_critic=True)
    assert bundle.critic is not None
    actor_normalizer_before = {
        key: _bytes(value) for key, value in bundle.actor.obs_normalizer.state_dict().items()
    }
    critic_normalizer_before = {
        key: _bytes(value) for key, value in bundle.critic.obs_normalizer.state_dict().items()
    }
    std_before = _bytes(bundle.actor.distribution.std_param)

    bundle.train()
    bundle.actor.obs_normalizer.update(torch.randn(32, OBSERVATION_DIM))
    bundle.critic.obs_normalizer.update(torch.randn(32, CRITIC_OBSERVATION_DIM))

    assert bundle.actor.obs_normalizer.training is False
    assert bundle.actor.obs_normalizer.until == 0
    assert bundle.actor.obs_normalizer.updates_enabled is False
    assert bundle.critic.obs_normalizer.training is False
    assert bundle.critic.obs_normalizer.until == 0
    assert bundle.critic.obs_normalizer.updates_enabled is False
    assert {
        key: _bytes(value) for key, value in bundle.actor.obs_normalizer.state_dict().items()
    } == actor_normalizer_before
    assert {
        key: _bytes(value) for key, value in bundle.critic.obs_normalizer.state_dict().items()
    } == critic_normalizer_before
    assert _bytes(bundle.actor.distribution.std_param) == std_before
    bundle.assert_frozen_invariants()


def test_left_knee_requires_explicit_gate_and_then_becomes_fifth_row() -> None:
    with pytest.raises(ValueError, match="explicit diagnosed evidence gate"):
        AnkleRowConfig(include_left_knee=True)
    with pytest.raises(ValueError, match="forbidden"):
        AnkleRowConfig(left_knee_gate=LEFT_KNEE_EVIDENCE_GATE)

    config = AnkleRowConfig(
        include_left_knee=True,
        left_knee_gate=LEFT_KNEE_EVIDENCE_GATE,
    )
    assert config.trainable_hardware_rows == (
        LEFT_KNEE_HARDWARE_ROW,
        *ANKLE_HARDWARE_ROWS,
    )
    bundle = load_selected_v2_ankle_adaptation(
        SELECTED_CHECKPOINT,
        learning_rate=1.0e-4,
        config=config,
    )
    bundle.zero_grad()
    bundle.actor(torch.zeros(2, OBSERVATION_DIM)).sum().backward()
    assert bundle.actor.mlp[6].bias.grad[LEFT_KNEE_HARDWARE_ROW] == 2


def test_direct_frozen_row_mutation_is_restored_and_poisons_bundle() -> None:
    bundle = load_selected_v2_ankle_adaptation(SELECTED_CHECKPOINT, learning_rate=1.0e-4)
    frozen_row = _frozen_rows(ANKLE_HARDWARE_ROWS)[0]
    original = bundle.actor.mlp[6].bias[frozen_row].detach().clone()
    with torch.no_grad():
        bundle.actor.mlp[6].bias[frozen_row].add_(1.0)

    with pytest.raises(RuntimeError, match="frozen actor output bias rows changed"):
        bundle.step()
    assert torch.equal(bundle.actor.mlp[6].bias[frozen_row], original)
    with pytest.raises(RuntimeError, match="poisoned"):
        bundle.step()


def test_warm_restart_preserves_adapted_weights_but_resets_adam_and_iteration(
    tmp_path: Path,
) -> None:
    bundle = load_selected_v2_ankle_adaptation(SELECTED_CHECKPOINT, learning_rate=7.5e-5, load_critic=True)
    assert bundle.critic is not None
    bundle.zero_grad()
    loss = bundle.actor(torch.randn(4, OBSERVATION_DIM)).sum()
    loss = loss + bundle.critic(torch.randn(4, CRITIC_OBSERVATION_DIM)).sum()
    loss.backward()
    bundle.step()
    actor_saved = {key: value.detach().clone() for key, value in bundle.actor.state_dict().items()}
    critic_saved = {key: value.detach().clone() for key, value in bundle.critic.state_dict().items()}
    output = tmp_path / "ankle_warm_restart.pt"
    publication = bundle.save_warm_restart(output)

    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert set(payload) == {"actor_state_dict", "critic_state_dict", "metadata"}
    assert "optimizer_state_dict" not in payload
    assert publication.sha256 == sha256_file(output)
    resumed = load_selected_v2_ankle_adaptation(
        SELECTED_CHECKPOINT,
        learning_rate=7.5e-5,
        load_critic=True,
        restart_path=output,
        expected_restart_sha256=publication.sha256,
    )

    assert resumed.iteration == 0
    assert resumed.prior_completed_steps == 1
    assert resumed.completed_steps_total == 1
    assert resumed.optimizer.state == {}
    assert all(torch.equal(resumed.actor.state_dict()[key], value) for key, value in actor_saved.items())
    assert resumed.critic is not None
    assert all(torch.equal(resumed.critic.state_dict()[key], value) for key, value in critic_saved.items())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bundle.save_warm_restart(output)


def test_warm_restart_rejects_optimizer_state_config_drift_and_frozen_row_drift(
    tmp_path: Path,
) -> None:
    bundle = load_selected_v2_ankle_adaptation(SELECTED_CHECKPOINT, learning_rate=1.0e-4)
    clean_payload = bundle.build_warm_restart()

    optimizer_path = tmp_path / "optimizer_injected.pt"
    optimizer_payload = dict(clean_payload)
    optimizer_payload["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    torch.save(optimizer_payload, optimizer_path)
    with pytest.raises(ValueError, match="unexpected=.*optimizer_state_dict"):
        load_selected_v2_ankle_adaptation(
            SELECTED_CHECKPOINT,
            learning_rate=1.0e-4,
            restart_path=optimizer_path,
            expected_restart_sha256=sha256_file(optimizer_path),
        )

    config_path = tmp_path / "config_drift.pt"
    torch.save(clean_payload, config_path)
    left_knee = AnkleRowConfig(
        include_left_knee=True,
        left_knee_gate=LEFT_KNEE_EVIDENCE_GATE,
    )
    with pytest.raises(ValueError, match="metadata/config/optimizer contract drift"):
        load_selected_v2_ankle_adaptation(
            SELECTED_CHECKPOINT,
            learning_rate=1.0e-4,
            config=left_knee,
            restart_path=config_path,
            expected_restart_sha256=sha256_file(config_path),
        )
    with pytest.raises(ValueError, match="metadata/config/optimizer contract drift"):
        load_selected_v2_ankle_adaptation(
            SELECTED_CHECKPOINT,
            learning_rate=2.0e-4,
            restart_path=config_path,
            expected_restart_sha256=sha256_file(config_path),
        )

    frozen_path = tmp_path / "frozen_drift.pt"
    frozen_payload = bundle.build_warm_restart()
    frozen_row = _frozen_rows(ANKLE_HARDWARE_ROWS)[0]
    frozen_payload["actor_state_dict"]["mlp.6.bias"][frozen_row] += 1.0
    torch.save(frozen_payload, frozen_path)
    with pytest.raises(ValueError, match="changed frozen actor rows"):
        load_selected_v2_ankle_adaptation(
            SELECTED_CHECKPOINT,
            learning_rate=1.0e-4,
            restart_path=frozen_path,
            expected_restart_sha256=sha256_file(frozen_path),
        )


def test_warm_restart_requires_and_checks_external_content_hash(tmp_path: Path) -> None:
    bundle = load_selected_v2_ankle_adaptation(SELECTED_CHECKPOINT, learning_rate=1.0e-4)
    output = tmp_path / "restart.pt"
    publication = bundle.save_warm_restart(output)

    with pytest.raises(ValueError, match="must be supplied together"):
        load_selected_v2_ankle_adaptation(
            SELECTED_CHECKPOINT,
            learning_rate=1.0e-4,
            restart_path=output,
        )
    wrong_hash = "0" * 64
    assert wrong_hash != publication.sha256
    with pytest.raises(ValueError, match="warm restart SHA-256 mismatch"):
        load_selected_v2_ankle_adaptation(
            SELECTED_CHECKPOINT,
            learning_rate=1.0e-4,
            restart_path=output,
            expected_restart_sha256=wrong_hash,
        )


def test_selected_checkpoint_hash_mismatch_fails_before_loading(tmp_path: Path) -> None:
    corrupt = tmp_path / "model21204_alpha25.pt"
    corrupt.write_bytes(SELECTED_CHECKPOINT.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_selected_v2_ankle_adaptation(corrupt, learning_rate=1.0e-4)
