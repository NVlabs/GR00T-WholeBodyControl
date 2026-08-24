"""Focused tests for exact true23 MJLab runner state and counters."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from gear_sonic.trl.mjlab import runner as runner_module
from gear_sonic.trl.mjlab.runner import (
    True23MjlabOnPolicyRunner,
    _environment_common_step_counter,
    _validate_resolved_config_binding,
)


class _FakePolicy:
    output_std = torch.ones(23)


class _FakeStorage:
    def __init__(self) -> None:
        self.step = 0
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1
        self.step = 0


class _FakeAlgorithm:
    def __init__(self) -> None:
        self.rnd = None
        self.learning_rate = 3.0e-4
        self.update_calls = 0
        self.rollout_calls = 0
        self.return_calls = 0
        self._policy = _FakePolicy()
        self.storage = _FakeStorage()

    def train_mode(self) -> None:
        pass

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        self.rollout_calls += 1
        return torch.zeros(obs.shape[0], 23)

    def process_env_step(self, *_args: Any) -> None:
        pass

    def compute_returns(self, _obs: torch.Tensor) -> None:
        self.return_calls += 1

    def update(self) -> dict[str, float]:
        self.update_calls += 1
        return {"surrogate": 0.0}

    def get_policy(self) -> _FakePolicy:
        return self._policy


class _FakeEnvironment:
    device = "cpu"
    max_episode_length = 100

    def __init__(self, common_step_counter: int = 0) -> None:
        self.unwrapped = SimpleNamespace(
            common_step_counter=common_step_counter
        )
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)

    def get_observations(self) -> torch.Tensor:
        return torch.zeros(2, 1)

    def step(
        self,
        _actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.unwrapped.common_step_counter += 1
        return (
            torch.zeros(2, 1),
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            {},
        )


class _FakeLogger:
    def __init__(self) -> None:
        self.logged_iterations: list[int] = []
        self.started = False
        self.stopped = False

    def init_logging_writer(self) -> None:
        self.started = True

    def process_env_step(self, *_args: Any) -> None:
        pass

    def log(self, **kwargs: Any) -> None:
        self.logged_iterations.append(kwargs["it"])

    def stop_logging_writer(self) -> None:
        self.stopped = True


class _TinyPolicyAdapter:
    def __init__(self) -> None:
        self.weight = torch.tensor([1.0])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": self.weight.detach().clone()}

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        assert strict is True
        self.weight = state_dict["weight"].detach().clone()


def _bare_runner(tmp_path: Path) -> True23MjlabOnPolicyRunner:
    instance = object.__new__(True23MjlabOnPolicyRunner)
    instance.env = _FakeEnvironment()
    instance.device = "cpu"
    instance.alg = _FakeAlgorithm()
    instance.logger = _FakeLogger()
    instance.cfg = {
        "save_interval": 2,
        "num_steps_per_env": 1,
        "check_for_nan": False,
    }
    instance.is_distributed = False
    instance.completed_update_count = 0
    instance.current_learning_iteration = 0
    instance.checkpoint_dir = tmp_path
    instance.write_initial_checkpoint = True
    instance._last_checkpoint_path = None
    instance._last_checkpoint_update_count = None
    instance._training_state_poisoned = False
    return instance


def _attach_recording_save(
    instance: True23MjlabOnPolicyRunner,
) -> list[int]:
    saved: list[int] = []

    def record(path: str, infos: dict | None = None) -> None:
        assert infos is None
        output = Path(path).resolve()
        saved.append(instance.completed_update_count)
        instance._last_checkpoint_path = output
        instance._last_checkpoint_update_count = (
            instance.completed_update_count
        )

    instance.save = record  # type: ignore[method-assign]
    return saved


def test_registered_runner_signature_requires_explicit_lineage_inputs() -> None:
    parameters = inspect.signature(
        True23MjlabOnPolicyRunner.__init__
    ).parameters

    assert list(parameters)[:5] == [
        "self",
        "env",
        "train_cfg",
        "log_dir",
        "device",
    ]
    for name in (
        "warm_start_checkpoint_path",
        "resolved_config",
        "source_manifest",
        "asset_manifest",
        "dataset_manifest",
    ):
        assert parameters[name].default is inspect.Parameter.empty
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_resolved_agent_config_must_equal_executed_train_cfg() -> None:
    train_cfg = {
        "seed": 42,
        "max_iterations": 50,
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "actor": {"class_name": "ExactTrue23"},
    }
    resolved = {
        "agent": {
            "seed": 42,
            "max_iterations": 50,
            "actor": {"class_name": "ExactTrue23"},
        }
    }

    _validate_resolved_config_binding(resolved, train_cfg)
    resolved["agent"]["max_iterations"] = 49
    with pytest.raises(ValueError, match="differs from executed"):
        _validate_resolved_config_binding(resolved, train_cfg)


def test_learn_counts_completed_updates_and_saves_periodic_and_final(
    tmp_path: Path,
) -> None:
    runner = _bare_runner(tmp_path)
    saved = _attach_recording_save(runner)

    runner.learn(3)

    assert runner.completed_update_count == 3
    assert runner.current_learning_iteration == 3
    assert runner.alg.update_calls == 3
    assert runner.alg.rollout_calls == 3
    assert runner.alg.return_calls == 3
    assert runner.env.unwrapped.common_step_counter == 3
    assert runner.logger.logged_iterations == [0, 1, 2]
    assert saved == [0, 2, 3]
    assert runner.logger.started is True
    assert runner.logger.stopped is True


def test_model_zero_precedes_first_learning_rollout_and_update(
    tmp_path: Path,
) -> None:
    runner = _bare_runner(tmp_path)
    runner.cfg["save_interval"] = 10
    events: list[str] = []
    get_observations = runner.env.get_observations
    act = runner.alg.act
    update = runner.alg.update

    def record_observation() -> torch.Tensor:
        events.append("observation")
        return get_observations()

    def record_act(obs: torch.Tensor) -> torch.Tensor:
        events.append("act")
        return act(obs)

    def record_update() -> dict[str, float]:
        events.append("update")
        return update()

    def record_save(path: str, infos: dict | None = None) -> None:
        assert infos is None
        events.append(f"save_{runner.completed_update_count}")
        runner._last_checkpoint_path = Path(path).resolve()
        runner._last_checkpoint_update_count = runner.completed_update_count

    runner.env.get_observations = record_observation  # type: ignore[method-assign]
    runner.alg.act = record_act  # type: ignore[method-assign]
    runner.alg.update = record_update  # type: ignore[method-assign]
    runner.save = record_save  # type: ignore[method-assign]

    runner.learn(1)

    assert events == ["save_0", "observation", "act", "update", "save_1"]


def test_episode_log_is_consumed_once_and_next_step_keeps_fresh_sink(
    tmp_path: Path,
) -> None:
    runner = _bare_runner(tmp_path)
    runner.cfg["num_steps_per_env"] = 2
    _attach_recording_save(runner)
    episode_log = {"Episode_Termination/ee_body_pos": 1}
    time_outs = torch.zeros(2, dtype=torch.bool)
    extras = {"log": episode_log, "time_outs": time_outs}
    seen_logs: list[dict[str, int] | None] = []
    step_count = 0

    def step_with_persistent_extras(
        _actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        nonlocal step_count
        step_count += 1
        if step_count == 2:
            # Real MJLab reward terms, including ``soft_landing``, write to
            # this shared sink before any episode reset can recreate it.
            extras["log"]["Metrics/landing_force_mean"] = 2
        runner.env.unwrapped.common_step_counter += 1
        return (
            torch.zeros(2, 1),
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            extras,
        )

    def record_logger_step(
        _rewards: torch.Tensor,
        _dones: torch.Tensor,
        step_extras: dict,
        _intrinsic_rewards: torch.Tensor | None,
    ) -> None:
        seen_logs.append(step_extras.get("log"))
        assert step_extras["time_outs"] is time_outs

    runner.env.step = step_with_persistent_extras  # type: ignore[method-assign]
    runner.logger.process_env_step = record_logger_step  # type: ignore[method-assign]

    runner.learn(1)

    assert seen_logs == [
        episode_log,
        {"Metrics/landing_force_mean": 2},
    ]
    assert seen_logs[0] is episode_log
    assert seen_logs[1] is not episode_log
    assert episode_log == {"Episode_Termination/ee_body_pos": 1}
    assert extras["log"] == {}
    assert extras["log"] is not seen_logs[1]
    assert extras["time_outs"] is time_outs


def test_resumed_learning_has_no_stock_off_by_one(
    tmp_path: Path,
) -> None:
    runner = _bare_runner(tmp_path)
    runner.completed_update_count = 4
    runner.current_learning_iteration = 4
    saved = _attach_recording_save(runner)

    runner.learn(2)

    assert runner.completed_update_count == 6
    assert runner.current_learning_iteration == 6
    assert runner.logger.logged_iterations == [4, 5]
    assert saved == [6]


def test_save_binds_complete_state_and_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _bare_runner(tmp_path)
    runner.completed_update_count = 7
    runner.current_learning_iteration = 7
    runner.env.unwrapped.common_step_counter = 91
    runner._training_lineage = {"lineage_sha256": "a" * 64}
    runner._policy_state_adapter = _TinyPolicyAdapter()
    runner.alg.critic = torch.nn.Linear(2, 1)
    runner.alg.optimizer = torch.optim.Adam(
        [
            *runner.alg.critic.parameters(),
            runner._policy_state_adapter.weight,
        ],
        lr=3.0e-4,
    )
    captured: dict[str, Any] = {}

    def fake_save(output_path: Path, **kwargs: Any) -> Path:
        captured["output_path"] = output_path
        captured.update(kwargs)
        return Path(output_path)

    monkeypatch.setattr(
        runner_module,
        "save_mjlab_training_checkpoint",
        fake_save,
    )
    output = tmp_path / "model_7.pt"

    runner.save(str(output))

    assert captured["update_count"] == 7
    assert captured["trainer_state"] == {
        "completed_update_count": 7,
        "current_learning_iteration": 7,
        "env_common_step_counter": 91,
        "algorithm_learning_rate": 3.0e-4,
    }
    assert captured["lineage"] == runner._training_lineage
    assert captured["overwrite"] is False
    assert set(captured["critic_state_dict"]) == {"weight", "bias"}
    assert set(captured["optimizer_state_dict"]) == {
        "state",
        "param_groups",
    }


def test_load_restores_all_counters_and_env_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _bare_runner(tmp_path)
    runner._training_lineage = {"lineage_sha256": "b" * 64}
    runner._lineage_sha256 = "b" * 64
    runner._policy_state_adapter = _TinyPolicyAdapter()
    runner.alg.critic = torch.nn.Linear(2, 1)
    runner.alg.optimizer = torch.optim.Adam(
        runner.alg.critic.parameters(),
        lr=3.0e-4,
    )
    captured: dict[str, Any] = {}

    def fake_restore(
        path: str,
        **kwargs: Any,
    ) -> dict[str, int | float]:
        captured["path"] = path
        captured.update(kwargs)
        kwargs["optimizer"].param_groups[0]["lr"] = 1.0e-4
        return {
            "completed_update_count": 12,
            "current_learning_iteration": 12,
            "env_common_step_counter": 345,
            "algorithm_learning_rate": 1.0e-4,
        }

    monkeypatch.setattr(
        runner_module,
        "restore_mjlab_training_checkpoint",
        fake_restore,
    )
    checkpoint = tmp_path / "model_12.pt"

    result = runner.load(str(checkpoint))

    assert runner.completed_update_count == 12
    assert runner.current_learning_iteration == 12
    assert runner.env.unwrapped.common_step_counter == 345
    assert runner.alg.learning_rate == 1.0e-4
    assert captured["expected_lineage"] == runner._training_lineage
    assert captured["minimum_update_count"] == 0
    assert captured["map_location"] == "cpu"
    assert result["lineage_sha256"] == "b" * 64


def test_failed_restore_rolls_back_every_mutable_runner_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _bare_runner(tmp_path)
    runner.completed_update_count = 3
    runner.current_learning_iteration = 3
    runner.env.unwrapped.common_step_counter = 27
    runner._training_lineage = {"lineage_sha256": "c" * 64}
    runner._lineage_sha256 = "c" * 64
    runner._policy_state_adapter = _TinyPolicyAdapter()
    runner.alg.critic = torch.nn.Linear(2, 1)
    runner.alg.optimizer = torch.optim.Adam(
        runner.alg.critic.parameters(),
        lr=3.0e-4,
    )
    policy_before = runner._policy_state_adapter.weight.clone()
    critic_before = {
        key: value.clone()
        for key, value in runner.alg.critic.state_dict().items()
    }

    def broken_restore(_path: str, **kwargs: Any) -> dict[str, int]:
        kwargs["policy_module"].weight.add_(5.0)
        with torch.no_grad():
            kwargs["critic_module"].weight.add_(5.0)
        kwargs["optimizer"].param_groups[0]["lr"] = 0.9
        raise ValueError("simulated corrupt resume")

    monkeypatch.setattr(
        runner_module,
        "restore_mjlab_training_checkpoint",
        broken_restore,
    )

    with pytest.raises(ValueError, match="corrupt resume"):
        runner.load(str(tmp_path / "corrupt.pt"))

    assert torch.equal(
        runner._policy_state_adapter.weight,
        policy_before,
    )
    for key, value in runner.alg.critic.state_dict().items():
        assert torch.equal(value, critic_before[key])
    assert runner.alg.optimizer.param_groups[0]["lr"] == 3.0e-4
    assert runner.completed_update_count == 3
    assert runner.current_learning_iteration == 3
    assert runner.env.unwrapped.common_step_counter == 27


def test_interrupted_restore_rolls_back_and_does_not_unpoison_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _bare_runner(tmp_path)
    runner._training_lineage = {"lineage_sha256": "d" * 64}
    runner._lineage_sha256 = "d" * 64
    runner._policy_state_adapter = _TinyPolicyAdapter()
    runner.alg.critic = torch.nn.Linear(2, 1)
    runner.alg.optimizer = torch.optim.Adam(
        runner.alg.critic.parameters(),
        lr=3.0e-4,
    )
    policy_before = runner._policy_state_adapter.weight.clone()
    critic_before = {
        key: value.clone()
        for key, value in runner.alg.critic.state_dict().items()
    }

    def interrupted_restore(_path: str, **kwargs: Any) -> dict[str, int]:
        kwargs["policy_module"].weight.add_(5.0)
        with torch.no_grad():
            kwargs["critic_module"].weight.add_(5.0)
        kwargs["optimizer"].param_groups[0]["lr"] = 0.9
        raise KeyboardInterrupt

    monkeypatch.setattr(
        runner_module,
        "restore_mjlab_training_checkpoint",
        interrupted_restore,
    )

    with pytest.raises(KeyboardInterrupt):
        runner.load(str(tmp_path / "interrupted.pt"))

    assert torch.equal(
        runner._policy_state_adapter.weight,
        policy_before,
    )
    for key, value in runner.alg.critic.state_dict().items():
        assert torch.equal(value, critic_before[key])
    assert runner.alg.optimizer.param_groups[0]["lr"] == 3.0e-4
    assert runner._training_state_poisoned is False


def test_restore_rollback_failure_leaves_runner_poisoned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _bare_runner(tmp_path)
    runner._training_lineage = {"lineage_sha256": "e" * 64}
    runner._lineage_sha256 = "e" * 64
    runner._policy_state_adapter = _TinyPolicyAdapter()
    runner.alg.critic = torch.nn.Linear(2, 1)
    runner.alg.optimizer = torch.optim.Adam(
        runner.alg.critic.parameters(),
        lr=3.0e-4,
    )

    def broken_restore(_path: str, **_kwargs: Any) -> dict[str, int]:
        raise ValueError("corrupt checkpoint")

    def broken_rollback(
        _state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        assert strict is True
        raise RuntimeError("rollback failed")

    monkeypatch.setattr(
        runner_module,
        "restore_mjlab_training_checkpoint",
        broken_restore,
    )
    runner._policy_state_adapter.load_state_dict = broken_rollback  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="discard this runner"):
        runner.load(str(tmp_path / "corrupt.pt"))

    assert runner._training_state_poisoned is True


def test_failed_ppo_iteration_poison_requires_exact_reload(
    tmp_path: Path,
) -> None:
    runner = _bare_runner(tmp_path)
    runner.completed_update_count = 1
    runner.current_learning_iteration = 1

    def fail_mid_update() -> dict[str, float]:
        runner.alg.update_calls += 1
        raise RuntimeError("mid-minibatch failure")

    runner.alg.update = fail_mid_update  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="mid-minibatch"):
        runner.learn(1)

    assert runner.completed_update_count == 1
    assert runner.current_learning_iteration == 1
    assert runner._training_state_poisoned is True
    with pytest.raises(RuntimeError, match="partially mutated"):
        runner.learn(1)
    with pytest.raises(RuntimeError, match="partially mutated"):
        True23MjlabOnPolicyRunner.save(
            runner,
            str(tmp_path / "unsafe.pt"),
        )


def test_poisoned_runner_recovers_only_recorded_older_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _bare_runner(tmp_path)
    runner.cfg["save_interval"] = 10
    runner.completed_update_count = 2
    runner.current_learning_iteration = 2
    checkpoint = (tmp_path / "model_2.pt").resolve()
    runner._last_checkpoint_path = checkpoint
    runner._last_checkpoint_update_count = 2
    outcomes: list[dict[str, float] | BaseException] = [
        {"surrogate": 0.0},
        RuntimeError("mid-minibatch failure"),
    ]

    def update_then_fail() -> dict[str, float]:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            runner.alg.storage.step = 1
            raise outcome
        return outcome

    runner.alg.update = update_then_fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="mid-minibatch"):
        runner.learn(2)

    assert runner.completed_update_count == 3
    assert runner._training_state_poisoned is True
    runner._training_lineage = {"lineage_sha256": "f" * 64}
    runner._lineage_sha256 = "f" * 64
    runner._policy_state_adapter = _TinyPolicyAdapter()
    runner.alg.critic = torch.nn.Linear(2, 1)
    runner.alg.optimizer = torch.optim.Adam(
        runner.alg.critic.parameters(),
        lr=3.0e-4,
    )
    captured: dict[str, Any] = {}

    def restore_recorded(
        path: str,
        **kwargs: Any,
    ) -> dict[str, int | float]:
        captured["path"] = path
        captured.update(kwargs)
        return {
            "completed_update_count": 2,
            "current_learning_iteration": 2,
            "env_common_step_counter": 17,
            "algorithm_learning_rate": 3.0e-4,
        }

    monkeypatch.setattr(
        runner_module,
        "restore_mjlab_training_checkpoint",
        restore_recorded,
    )

    runner.load(str(checkpoint))

    assert captured["minimum_update_count"] == 2
    assert runner.completed_update_count == 2
    assert runner.current_learning_iteration == 2
    assert runner.env.unwrapped.common_step_counter == 17
    assert runner.alg.storage.clear_calls == 1
    assert runner.alg.storage.step == 0
    assert runner._training_state_poisoned is False


def test_poisoned_runner_rejects_nonrecorded_checkpoint(
    tmp_path: Path,
) -> None:
    runner = _bare_runner(tmp_path)
    runner.completed_update_count = 3
    runner.current_learning_iteration = 3
    runner._training_state_poisoned = True
    runner._last_checkpoint_path = (tmp_path / "model_2.pt").resolve()
    runner._last_checkpoint_update_count = 2

    with pytest.raises(RuntimeError, match="only its recorded last exact"):
        runner.load(str(tmp_path / "model_3.pt"))


def test_poisoned_runner_requires_recorded_checkpoint_count_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _bare_runner(tmp_path)
    runner.completed_update_count = 3
    runner.current_learning_iteration = 3
    runner._training_state_poisoned = True
    checkpoint = (tmp_path / "model_2.pt").resolve()
    runner._last_checkpoint_path = checkpoint
    runner._last_checkpoint_update_count = 2
    runner._training_lineage = {"lineage_sha256": "0" * 64}
    runner._lineage_sha256 = "0" * 64
    runner._policy_state_adapter = _TinyPolicyAdapter()
    runner.alg.critic = torch.nn.Linear(2, 1)
    runner.alg.optimizer = torch.optim.Adam(
        runner.alg.critic.parameters(),
        lr=3.0e-4,
    )

    def restore_wrong_count(
        _path: str,
        **_kwargs: Any,
    ) -> dict[str, int | float]:
        return {
            "completed_update_count": 3,
            "current_learning_iteration": 3,
            "env_common_step_counter": 19,
            "algorithm_learning_rate": 3.0e-4,
        }

    monkeypatch.setattr(
        runner_module,
        "restore_mjlab_training_checkpoint",
        restore_wrong_count,
    )

    with pytest.raises(ValueError, match="recorded last exact update_count"):
        runner.load(str(checkpoint))

    assert runner.completed_update_count == 3
    assert runner.current_learning_iteration == 3
    assert runner._training_state_poisoned is True


@pytest.mark.parametrize("value", [-1, 1.5, True, torch.ones(2)])
def test_environment_counter_rejects_unsafe_values(value: Any) -> None:
    env = _FakeEnvironment()
    env.unwrapped.common_step_counter = value

    with pytest.raises(ValueError, match="common_step_counter"):
        _environment_common_step_counter(env)


def test_stock_rsl_exports_are_always_forbidden(tmp_path: Path) -> None:
    runner = _bare_runner(tmp_path)

    with pytest.raises(RuntimeError, match="JIT export is forbidden"):
        runner.export_policy_to_jit(str(tmp_path))
    with pytest.raises(RuntimeError, match="ONNX export is forbidden"):
        runner.export_policy_to_onnx(str(tmp_path))
