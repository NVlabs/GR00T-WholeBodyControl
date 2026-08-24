from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from gear_sonic.trl.mjlab.supported_idle_checkpoint import TensorSpec
from gear_sonic.trl.mjlab.supported_idle_runner import (
    NUM_STEPS_PER_ENV,
    SupportedIdleTrainingController,
    derive_session_seed,
)


class _Module:
    def __init__(self) -> None:
        self.loaded: list[dict[str, int]] = []
        self.fail_load = False

    def state_dict(self) -> dict[str, int]:
        return {"weight": 1}

    def load_state_dict(self, value: dict[str, int], *, strict: bool) -> None:
        assert strict is True
        if self.fail_load:
            raise RuntimeError("module load failed")
        self.loaded.append(value)


class _Optimizer:
    def __init__(self) -> None:
        self.loaded: list[dict[str, Any]] = []
        self.fail_load = False

    def state_dict(self) -> dict[str, Any]:
        return {"state": {}, "param_groups": []}

    def load_state_dict(self, value: dict[str, Any]) -> None:
        if self.fail_load:
            raise RuntimeError("optimizer load failed")
        self.loaded.append(value)


class _Alg:
    def __init__(self) -> None:
        self.actor, self.critic, self.optimizer = _Module(), _Module(), _Optimizer()
        self.reset_calls = 0
        self.update_calls = 0
        self.fail_reset = False
        self.fail_on_update: int | None = None

    def reset_optimizer(self) -> None:
        if self.fail_reset:
            raise RuntimeError("reset failed")
        self.reset_calls += 1

    def act(self, obs: int) -> int:
        return obs

    def process_env_step(self, *_args: Any) -> None:
        return None

    def compute_returns(self, _obs: int) -> None:
        return None

    def update(self) -> None:
        self.update_calls += 1
        if self.fail_on_update == self.update_calls:
            raise RuntimeError("update failed")


class _Env:
    def __init__(self) -> None:
        self.common_step_counter = 9
        self.loaded_command: dict[str, int] | None = None
        self.step_calls = 0
        self.fail_on_step: int | None = None

    def get_observations(self) -> int:
        return 1

    def step(self, action: int) -> tuple[int, int, bool, dict[str, Any]]:
        self.step_calls += 1
        if self.fail_on_step == self.step_calls:
            raise RuntimeError("rollout failed")
        self.common_step_counter += 1
        return action, 0, False, {}

    def command_state_dict(self) -> dict[str, int]:
        return {"time_steps": 3}

    def load_command_state_dict(self, state: dict[str, int]) -> None:
        self.loaded_command = state


def _lineage(source_kind: str = "dad_dance_seed") -> dict[str, Any]:
    return {
        "plan_payload_sha256": "a" * 64,
        "plan_file_sha256": "b" * 64,
        "authorization_file_sha256": "c" * 64,
        "job_id": "static_model3500",
        "source_checkpoint_sha256": "d" * 64,
        "source_kind": source_kind,
        "corpus_sha256": "e" * 64,
        "sidecar_sha256": "f" * 64,
        "runtime_file_sha256": {"runner.py": "1" * 64},
        "package_sha256": {"torch": "2" * 64},
    }


def _semantics(source_kind: str) -> dict[str, Any]:
    initialization = {
        "dad_dance_seed": "seed_actor_critic_only",
        "qualified_phase_parent": "phase_parent_actor_critic_only",
    }[source_kind]
    return {
        "initialization": initialization,
        "source_optimizer_loaded": False,
        "optimizer_reset": True,
        "resume_loads_optimizer": True,
    }


def _controller(
    tmp_path: Path,
    *,
    planned: int = 300,
    source_kind: str = "dad_dance_seed",
    save_error: BaseException | None = None,
    initialized: bool = True,
) -> tuple[SupportedIdleTrainingController, _Alg, _Env, dict[str, list[Any]]]:
    alg, env = _Alg(), _Env()
    captured: dict[str, list[Any]] = {"builds": [], "paths": []}

    def build(**kwargs: Any) -> dict[str, Any]:
        captured["builds"].append(kwargs)
        return kwargs

    def save(path: Path, *_args: Any, **_kwargs: Any) -> None:
        captured["paths"].append(path)
        if save_error is not None:
            raise save_error

    controller = SupportedIdleTrainingController(
        alg=alg,
        env=env,
        checkpoint_dir=tmp_path,
        command_schema={"time_steps": TensorSpec((), torch.int64)},
        lineage=_lineage(source_kind),
        optimizer_semantics=_semantics(source_kind),
        planned_updates=planned,
        checkpoint_builder=build,
        checkpoint_saver=save,
        rng_capture=lambda: {"rng": 1},
        rng_restore=lambda _state: None,
        testing_only=True,
    )
    if initialized:
        mode = {
            "dad_dance_seed": "seed_actor_critic_only",
            "qualified_phase_parent": "phase_parent_actor_critic_only",
        }[source_kind]
        controller.initialize_from_source(_source(), mode=mode)
    return controller, alg, env, captured


def _source() -> dict[str, Any]:
    return {"actor_state_dict": {"weight": 2}, "critic_state_dict": {"weight": 3}}


def _resume(source_kind: str = "dad_dance_seed") -> dict[str, Any]:
    return {
        **_source(),
        "optimizer_state_dict": {"state": {}},
        "rng_state": {"rng": 2},
        "trainer_state": {
            "completed_updates": 6,
            "current_learning_iteration": 6,
            "env_common_step_counter": 44,
        },
        "completed_updates": 6,
        "iter": 5,
        "command_state": {"time_steps": 8},
        "lineage": _lineage(source_kind),
        "optimizer_semantics": _semantics(source_kind),
    }


def test_two_updates_collect_48_steps_and_save_once_model_2(tmp_path: Path) -> None:
    controller, alg, env, captured = _controller(tmp_path)
    controller.learn(2)
    assert NUM_STEPS_PER_ENV == 24
    assert env.step_calls == 48 and alg.update_calls == 2
    assert controller.completed_updates == 2
    assert len(captured["builds"]) == len(captured["paths"]) == 1
    assert captured["paths"] == [tmp_path / "model_2.pt"]


def test_no_replay_starts_at_completed_counter(tmp_path: Path) -> None:
    controller, _alg, env, captured = _controller(tmp_path)
    controller.completed_updates = controller.current_learning_iteration = 4
    controller.learn(2)
    assert env.step_calls == 48 and controller.completed_updates == 6
    assert captured["builds"][0]["completed_updates"] == 6


def test_successful_session_is_sealed_against_second_learn(tmp_path: Path) -> None:
    controller, _alg, env, captured = _controller(tmp_path)
    controller.learn(1)
    with pytest.raises(RuntimeError, match="session already completed"):
        controller.learn(1)
    assert env.step_calls == 24 and len(captured["paths"]) == 1


def test_second_rollout_failure_keeps_one_completed_and_saves_nothing(tmp_path: Path) -> None:
    controller, _alg, env, captured = _controller(tmp_path)
    env.fail_on_step = 30
    with pytest.raises(RuntimeError, match="rollout failed"):
        controller.learn(2)
    assert env.step_calls == 30 and controller.completed_updates == 1
    assert captured["builds"] == captured["paths"] == []
    with pytest.raises(RuntimeError, match="poisoned"):
        controller.learn(1)


def test_second_update_failure_keeps_one_completed_and_saves_nothing(tmp_path: Path) -> None:
    controller, alg, env, captured = _controller(tmp_path)
    alg.fail_on_update = 2
    with pytest.raises(RuntimeError, match="update failed"):
        controller.learn(2)
    assert env.step_calls == 48 and controller.completed_updates == 1
    assert captured["builds"] == captured["paths"] == []
    with pytest.raises(RuntimeError, match="poisoned"):
        controller.save_checkpoint()


def test_save_failure_after_update_poisoned(tmp_path: Path) -> None:
    controller, _alg, env, _captured = _controller(tmp_path, save_error=RuntimeError("disk failed"))
    with pytest.raises(RuntimeError, match="disk failed"):
        controller.learn(1)
    assert env.step_calls == 24 and controller.completed_updates == 1
    with pytest.raises(RuntimeError, match="poisoned"):
        controller.save_checkpoint()


def test_caps_reject_before_rollout(tmp_path: Path) -> None:
    controller, _alg, env, _captured = _controller(tmp_path)
    with pytest.raises(ValueError, match="cap"):
        controller.learn(251)
    assert env.step_calls == 0
    planned, _alg, planned_env, _captured = _controller(tmp_path, planned=2)
    with pytest.raises(ValueError, match="cap"):
        planned.learn(3)
    assert planned_env.step_calls == 0


def test_seed_and_phase_parent_reset_without_optimizer_load(tmp_path: Path) -> None:
    for source_kind, mode in (
        ("dad_dance_seed", "seed_actor_critic_only"),
        ("qualified_phase_parent", "phase_parent_actor_critic_only"),
    ):
        controller, alg, _env, _captured = _controller(tmp_path, source_kind=source_kind, initialized=False)
        controller.initialize_from_source(_source(), mode=mode)
        assert alg.reset_calls == 1 and not alg.optimizer.loaded


def test_partial_initialization_and_resume_hook_failures_poison(tmp_path: Path) -> None:
    controller, alg, _env, _captured = _controller(tmp_path, initialized=False)
    alg.critic.fail_load = True
    with pytest.raises(RuntimeError, match="module load failed"):
        controller.initialize_from_source(_source(), mode="seed_actor_critic_only")
    with pytest.raises(RuntimeError, match="poisoned"):
        controller.learn(1)

    resumed, resumed_alg, _env, _captured = _controller(tmp_path, initialized=False)
    resumed_alg.optimizer.fail_load = True
    with pytest.raises(RuntimeError, match="optimizer load failed"):
        resumed.resume_same_job(_resume())
    with pytest.raises(RuntimeError, match="poisoned"):
        resumed.learn(1)


def test_same_job_restores_full_state(tmp_path: Path) -> None:
    controller, alg, env, _captured = _controller(tmp_path, initialized=False)
    controller.resume_same_job(_resume())
    assert controller.completed_updates == 6 and alg.optimizer.loaded
    assert env.common_step_counter == 44 and env.loaded_command == {"time_steps": 8}


def test_same_job_rejects_lineage_mismatch_before_mutation(tmp_path: Path) -> None:
    controller, alg, _env, _captured = _controller(tmp_path, initialized=False)
    checkpoint = _resume()
    checkpoint["lineage"]["job_id"] = "other_job"
    with pytest.raises(ValueError, match="lineage/optimizer"):
        controller.resume_same_job(checkpoint)
    assert not alg.actor.loaded and not alg.optimizer.loaded


def test_invalid_lineage_and_seed_inputs_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        derive_session_seed("A" * 64, "job", 0)
    assert 0 <= derive_session_seed("a" * 64, "job", 3) <= 0xFFFFFFFF
    lineage = _lineage()
    lineage["corpus_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA-256"):
        SupportedIdleTrainingController(
            alg=_Alg(),
            env=_Env(),
            checkpoint_dir=tmp_path,
            command_schema={"time_steps": TensorSpec((), torch.int64)},
            lineage=lineage,
            optimizer_semantics=_semantics("dad_dance_seed"),
            planned_updates=1,
        )


def test_custom_persistence_hooks_require_testing_only(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="testing_only"):
        SupportedIdleTrainingController(
            alg=_Alg(),
            env=_Env(),
            checkpoint_dir=tmp_path,
            command_schema={"time_steps": TensorSpec((), torch.int64)},
            lineage=_lineage(),
            optimizer_semantics=_semantics("dad_dance_seed"),
            planned_updates=1,
            checkpoint_saver=lambda *_args, **_kwargs: None,
        )


def test_test_scaffold_must_initialize_before_learning_or_saving(tmp_path: Path) -> None:
    controller, _alg, env, captured = _controller(tmp_path, initialized=False)
    with pytest.raises(RuntimeError, match="not initialized"):
        controller.learn(1)
    with pytest.raises(RuntimeError, match="not initialized"):
        controller.save_checkpoint()
    assert env.step_calls == 0
    assert captured["builds"] == captured["paths"] == []


def test_default_controller_has_no_production_initialization_route(tmp_path: Path) -> None:
    alg, env = _Alg(), _Env()
    controller = SupportedIdleTrainingController(
        alg=alg,
        env=env,
        checkpoint_dir=tmp_path,
        command_schema={"time_steps": TensorSpec((), torch.int64)},
        lineage=_lineage(),
        optimizer_semantics=_semantics("dad_dance_seed"),
        planned_updates=1,
    )
    with pytest.raises(RuntimeError, match="testing-only"):
        controller.initialize_from_source(object(), mode="seed_actor_critic_only")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="testing-only"):
        controller.resume_same_job(object())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="not initialized"):
        controller.learn(1)
    with pytest.raises(RuntimeError, match="not initialized"):
        controller.save_checkpoint()
    assert not controller._initialized
    assert alg.reset_calls == 0 and not alg.actor.loaded and not alg.optimizer.loaded
    assert env.step_calls == 0


def test_testing_only_requires_exact_bool(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="testing_only must be bool"):
        SupportedIdleTrainingController(
            alg=_Alg(),
            env=_Env(),
            checkpoint_dir=tmp_path,
            command_schema={"time_steps": TensorSpec((), torch.int64)},
            lineage=_lineage(),
            optimizer_semantics=_semantics("dad_dance_seed"),
            planned_updates=1,
            testing_only=1,  # type: ignore[arg-type]
        )
