from __future__ import annotations

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from gear_sonic.scripts import train_g1_true23_sonic_task_space_ppo_full_support as cli
from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as subject

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _passing_rollout_record() -> dict[str, Any]:
    counts: list[dict[str, int]] = []
    for q9 in range(9, 169):
        if q9 <= 88:
            count = 128
        elif q9 <= 125:
            count = 112
        elif q9 <= 132:
            count = 96
        elif q9 <= 153:
            count = 64
        elif q9 <= 162:
            count = 32
        else:
            count = 16
        counts.append({"q9": q9, "count": count})
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "g1_true23_sonic_task_space_full_support_rollout_evidence_v1",
        "contract_sha256": subject.CONTRACT_SHA256,
        "trace_sha256": subject.TRACE_SHA256,
        "run_materials_sha256": "a" * 64,
        "first_episode_q9_histogram": counts,
        "total_inserted_transitions": 20_480,
        "pre_adam_state": {
            "storage_step": 160,
            "executed_training_transitions": 20_480,
            "optimizer_step_count": 0,
            "optimizer_state_entry_count": 0,
            "completed_update_count": 0,
            "current_learning_iteration": 0,
            "critic_observation_normalizer_sample_count": 20_480,
            "policy_state_before_sha256": subject.INITIAL_OVERLAY_POLICY_STATE_SHA256,
            "policy_state_after_sha256": subject.INITIAL_OVERLAY_POLICY_STATE_SHA256,
            "policy_state_unchanged": True,
            "frozen_actor_before_sha256": subject.INITIAL_FROZEN_ACTOR_STATE_SHA256,
            "frozen_actor_after_sha256": subject.INITIAL_FROZEN_ACTOR_STATE_SHA256,
            "frozen_actor_state_unchanged": True,
            "trainable_actor_before_sha256": subject.INITIAL_TRAINABLE_ACTOR_STATE_SHA256,
            "trainable_actor_after_sha256": subject.INITIAL_TRAINABLE_ACTOR_STATE_SHA256,
            "trainable_actor_state_unchanged": True,
            "std_before_sha256": "4" * 64,
            "std_after_sha256": "4" * 64,
            "critic_mlp_before_sha256": "1" * 64,
            "critic_mlp_after_sha256": "1" * 64,
            "critic_mlp_state_unchanged": True,
            "critic_normalizer_before_sha256": "2" * 64,
            "critic_normalizer_after_sha256": "3" * 64,
            "returns_finite": True,
            "advantages_finite": True,
        },
        "first_episode_reward_and_termination_evidence": {
            "right_wrist_barrier_active_count": 10,
            "right_wrist_barrier_raw_sum": 1.0,
            "right_wrist_barrier_weighted_sum": -0.5,
            "worst_ee_raw_sum": 5.0,
            "ee_body_pos_terminal_count": 4,
        },
        "rng_state_hashes": {
            "pre_rollout_pre_probe": {"torch_cpu_state_sha256": "5" * 64},
            "pre_rollout": {"torch_cpu_state_sha256": "5" * 64},
            "post_rollout_pre_probe": {"torch_cpu_state_sha256": "6" * 64},
            "post_probe": {"torch_cpu_state_sha256": "6" * 64},
        },
        "pre_optimizer_memory": {
            "device_name": subject.EXPECTED_CUDA_DEVICE_NAME,
            "device_total_memory_bytes": subject.EXPECTED_CUDA_TOTAL_MEMORY_BYTES,
            "mini_batch_size": subject.MINI_BATCH_SIZE,
            "free_cuda_bytes_before_update": subject.MINIMUM_FREE_CUDA_BYTES,
            "minimum_required_free_cuda_bytes": subject.MINIMUM_FREE_CUDA_BYTES,
            "cuda_cache_released": True,
        },
        "nonfinite_count": 0,
        "q9_discontinuity_count": 0,
        "action_semantics_mismatch_count": 0,
        "optimizer_steps_at_publication": 0,
        "training_updates_at_publication": 0,
        "failed_model5_loaded": False,
        "failed_model5_resumed": False,
    }
    record["coverage_assessment"] = subject.assess_full_support_rollout_evidence(record)
    return record


def _evaluation(
    update: int,
    completed: int,
    episode_return: float,
    *,
    policy_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "g1_true23_sonic_task_space_ppo_full_support_evaluation_v1",
        "update_count": update,
        "evaluation_seed": subject.FIXED_SEED,
        "controller": "deterministic_actor_mean",
        "policy_state_sha256": policy_hash
        or (subject.INITIAL_OVERLAY_POLICY_STATE_SHA256 if update == 0 else "b" * 64),
        "completed_transitions": completed,
        "terminal_q9": 9 + completed - 1,
        "termination_names": ["ee_body_pos"],
        "episode_return": episode_return,
        "nonfinite_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": 0,
        "q9_discontinuity_count": 0,
    }


def test_contract_is_hash_bound_and_declares_only_one_full_rollout_update() -> None:
    contract = subject.load_full_support_contract(REPOSITORY_ROOT)
    assert contract["diagnostic_evidence"]["trace_sha256"] == subject.TRACE_SHA256
    assert contract["failed_parent_pilot"]["model5_is_failed_evidence_only"] is True
    assert contract["actor_initialization"]["failed_model5_loaded"] is False
    assert contract["actor_initialization"]["initial_fresh_critic_state_sha256"] == (
        subject.INITIAL_CRITIC_STATE_SHA256
    )
    assert contract["single_update"] == {
        **contract["single_update"],
        "num_envs": 128,
        "num_steps_per_env": 160,
        "training_transitions": 20_480,
        "maximum_updates": 1,
        "num_learning_epochs": 2,
        "num_mini_batches": 4,
        "mini_batch_size": 5_120,
        "optimizer_steps": 8,
    }
    assert contract["pre_optimizer_coverage_gate"]["required_minimum_survivors_at_q9"] == {
        "126": 96,
        "133": 64,
        "154": 32,
        "163": 16,
    }
    assert contract["evaluation_gates"]["candidate_minimum_terminal_q9"] == 164
    assert contract["boundaries"]["hardware_authorized"] is False


def test_rollout_gate_requires_direct_hash_rng_reward_memory_and_survivor_evidence() -> None:
    record = _passing_rollout_record()
    assert subject.assess_full_support_rollout_evidence(record)["gate_passed"] is True
    subject.validate_full_support_rollout_evidence(record)

    mutations = []
    bad = copy.deepcopy(record)
    bad["pre_adam_state"]["policy_state_after_sha256"] = "9" * 64
    mutations.append(bad)
    bad = copy.deepcopy(record)
    bad["pre_adam_state"]["critic_normalizer_after_sha256"] = "2" * 64
    mutations.append(bad)
    bad = copy.deepcopy(record)
    bad["rng_state_hashes"]["post_probe"] = {"torch_cpu_state_sha256": "7" * 64}
    mutations.append(bad)
    bad = copy.deepcopy(record)
    bad["action_semantics_mismatch_count"] = 1
    mutations.append(bad)
    bad = copy.deepcopy(record)
    bad["first_episode_q9_histogram"][126 - 9]["count"] = 95
    mutations.append(bad)
    bad = copy.deepcopy(record)
    bad["first_episode_reward_and_termination_evidence"]["right_wrist_barrier_active_count"] = 0
    mutations.append(bad)
    bad = copy.deepcopy(record)
    bad["pre_optimizer_memory"]["free_cuda_bytes_before_update"] = subject.MINIMUM_FREE_CUDA_BYTES - 1
    mutations.append(bad)

    for mutated in mutations:
        assessment = subject.assess_full_support_rollout_evidence(mutated)
        assert assessment["gate_passed"] is False
        assert assessment["optimizer_permitted"] is False


def test_tensor_state_snapshot_does_not_alias_live_normalizer() -> None:
    live = {"obs_normalizer.count": torch.tensor(0, dtype=torch.int64)}
    snapshot = subject._clone_tensor_state(live)
    live["obs_normalizer.count"].fill_(20_480)
    assert int(snapshot["obs_normalizer.count"].item()) == 0
    assert snapshot["obs_normalizer.count"].device.type == "cpu"


def test_reward_compute_and_finish_seams_have_no_host_materialization() -> None:
    for method in (
        subject._FullSupportRewardEvidenceRecorder._capture,
        subject._FullSupportRewardEvidenceRecorder.finish,
    ):
        source = inspect.getsource(method)
        assert ".cpu(" not in source
        assert ".item(" not in source
        assert ".tolist(" not in source
        assert "bool(" not in source


def test_evaluation_gate_uses_current_baseline_reward_floor_and_q164() -> None:
    baseline = _evaluation(0, 155, -122.0)
    partial = subject.assess_full_support_evaluations([baseline])
    assert partial["stop"] is False
    assert partial["candidate_selected"] is False

    passed = subject.assess_full_support_evaluations([baseline, _evaluation(1, 156, -130.0)])
    assert passed["passed"] is True
    assert passed["candidate_selected"] is True
    assert passed["candidate_reward_floor"] == -172.0

    short = subject.assess_full_support_evaluations([baseline, _evaluation(1, 155, -120.0)])
    assert short["candidate_selected"] is False
    catastrophic = subject.assess_full_support_evaluations([baseline, _evaluation(1, 156, -172.01)])
    assert catastrophic["candidate_selected"] is False
    unchanged = subject.assess_full_support_evaluations(
        [
            baseline,
            _evaluation(
                1,
                156,
                -120.0,
                policy_hash=subject.INITIAL_OVERLAY_POLICY_STATE_SHA256,
            ),
        ]
    )
    assert unchanged["candidate_selected"] is False


class _FakeState:
    def __init__(self, value: float = 0.0) -> None:
        self.value = torch.tensor([value])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": self.value}


class _FakeScheduleRunner:
    def __init__(self, root: Path, *, coverage_passes: bool) -> None:
        self.root = root
        self.coverage_passes = coverage_passes
        self.completed_update_count = 0
        self.current_learning_iteration = 0
        self._optimizer_step_count = 0
        self._executed_training_transitions = 0
        self._phase = "fresh"
        self._rollout_evidence_sha256 = None
        self._run_materials_sha256 = "a" * 64
        self._policy_state_adapter = _FakeState()
        self.alg = SimpleNamespace(critic=_FakeState(), optimizer=_FakeState())
        self.storage_step = 0
        self.collect_calls = 0
        self.optimize_calls = 0

    def _storage_step(self) -> int:
        return self.storage_step

    def _assert_execution_boundary(self) -> None:
        return None

    def save_current_checkpoint(self) -> Path:
        path = self.root / f"checkpoint_{self.completed_update_count}.pt"
        path.write_bytes(str(self.completed_update_count).encode())
        return path

    def collect_full_support_rollout(self) -> dict[str, Any]:
        self.collect_calls += 1
        self._executed_training_transitions = 20_480
        self.storage_step = 160
        self._phase = "rollout_ready" if self.coverage_passes else "rollout_failed"
        self._rollout_evidence_sha256 = "c" * 64
        return {"coverage_assessment": {"gate_passed": self.coverage_passes}}

    def optimize_collected_rollout(self) -> dict[str, float]:
        self.optimize_calls += 1
        self.completed_update_count = 1
        self.current_learning_iteration = 1
        self._optimizer_step_count = 8
        self.storage_step = 0
        self._phase = "updated"
        self._policy_state_adapter.value.fill_(1.0)
        return {"value": 1.0, "surrogate": 2.0, "entropy": 3.0}


@pytest.mark.parametrize("coverage_passes", [False, True])
def test_schedule_never_optimizes_failed_coverage_and_selects_only_passing_update1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, coverage_passes: bool
) -> None:
    runner = _FakeScheduleRunner(tmp_path, coverage_passes=coverage_passes)

    def fake_load(_path: Path, **_kwargs: Any) -> dict[str, Any]:
        return {"update_count": runner.completed_update_count}

    def fake_validate(value: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        update = value["update_count"]
        return {
            "update_count": update,
            "policy_state_sha256": (subject.INITIAL_OVERLAY_POLICY_STATE_SHA256 if update == 0 else "b" * 64),
        }

    monkeypatch.setattr(subject.torch, "load", fake_load)
    monkeypatch.setattr(subject, "validate_full_support_checkpoint", fake_validate)

    def evaluator(_runner: Any, update: int) -> dict[str, Any]:
        return _evaluation(update, 155 if update == 0 else 156, -122.0)

    result = subject.execute_full_support_schedule(runner, evaluator)
    assert runner.collect_calls == 1
    if coverage_passes:
        assert runner.optimize_calls == 1
        assert result["candidate"]["update_count"] == 1
        assert result["optimizer_step_count"] == 8
    else:
        assert runner.optimize_calls == 0
        assert result["candidate"] is None
        assert result["optimizer_step_count"] == 0


def test_coupled_learn_and_cli_hyperparameter_overrides_are_forbidden() -> None:
    runner = object.__new__(subject.SonicTaskSpacePpoFullSupportRunner)
    with pytest.raises(RuntimeError, match="forbids coupled learn"):
        runner.learn(1, False)
    parser = cli._parser()
    parsed = parser.parse_args(["experiment", "--repository-root", str(REPOSITORY_ROOT), "--run-dir", "run"])
    assert set(vars(parsed)) == {"command", "repository_root", "run_dir"}
    for forbidden in ("--seed", "--num-envs", "--num-steps", "--checkpoint"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "experiment",
                    "--run-dir",
                    "run",
                    forbidden,
                    "1",
                ]
            )
