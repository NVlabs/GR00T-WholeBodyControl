"""Fail-closed MJLab runner for direct exact-SONIC true-23 PPO training.

The stock RSL-RL checkpoint and export paths do not describe the split SONIC
encoder/decoder deployment contract.  This runner therefore owns the learning
loop counters and writes only the weights-only-safe, lineage-bound resume
schema from :mod:`gear_sonic.utils.g1_23dof_mjlab_training`.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math
import numbers
import os
from pathlib import Path
import time
from typing import Any

import torch

from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state
from gear_sonic.utils.g1_23dof_mjlab_training import (
    build_mjlab_training_lineage,
    restore_mjlab_training_checkpoint,
    save_mjlab_training_checkpoint,
    validate_mjlab_training_lineage,
)

try:
    from mjlab.rl.runner import MjlabOnPolicyRunner as _MjlabOnPolicyRunner
except ImportError as exc:  # Unit tests and checkpoint inspection need no MJLab.
    _MJLAB_IMPORT_ERROR: ImportError | None = exc

    class _MjlabOnPolicyRunner:  # type: ignore[no-redef]
        """Import-only placeholder used when the pinned MJLab runtime is absent."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "True23MjlabOnPolicyRunner requires the pinned MJLab runtime"
            ) from _MJLAB_IMPORT_ERROR

else:
    _MJLAB_IMPORT_ERROR = None


def _require_nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{context} must be an integer >= 0")
    result = int(value)
    if result < 0:
        raise ValueError(f"{context} must be an integer >= 0")
    return result


def _validate_resolved_config_binding(
    resolved_config: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
) -> None:
    """Bind the declared agent config to the dictionary RSL actually executes."""

    resolved_agent = resolved_config.get("agent")
    if not isinstance(resolved_agent, Mapping):
        raise ValueError(
            "resolved_config.agent must contain the executed RSL config"
        )
    executed_agent = copy.deepcopy(dict(train_cfg))
    for key in ("resume", "load_run", "load_checkpoint"):
        executed_agent.pop(key, None)
    if dict(resolved_agent) != executed_agent:
        raise ValueError(
            "resolved_config.agent differs from executed train_cfg"
        )


def _require_learning_rate(value: Any, context: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{context} tensor must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{context} must be a finite float > 0")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{context} must be a finite float > 0")
    return result


def _environment_common_step_counter(env: Any) -> int:
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is None or not hasattr(unwrapped, "common_step_counter"):
        raise TypeError(
            "true23 MJLab environment must expose "
            "unwrapped.common_step_counter"
        )
    value = unwrapped.common_step_counter
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("env common_step_counter tensor must be scalar")
        value = value.detach().cpu().item()
    return _require_nonnegative_integer(value, "env common_step_counter")


def _set_environment_common_step_counter(env: Any, value: Any) -> None:
    counter = _require_nonnegative_integer(
        value,
        "restored env common_step_counter",
    )
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is None or not hasattr(unwrapped, "common_step_counter"):
        raise TypeError(
            "true23 MJLab environment must expose "
            "unwrapped.common_step_counter"
        )
    current = unwrapped.common_step_counter
    if isinstance(current, torch.Tensor):
        if current.numel() != 1:
            raise ValueError("env common_step_counter tensor must be scalar")
        with torch.no_grad():
            current.fill_(counter)
    else:
        unwrapped.common_step_counter = counter


class _ExactPolicyStateAdapter:
    """Expose an RSL actor through the exact deployment checkpoint namespace."""

    def __init__(self, actor: Any) -> None:
        if not hasattr(actor, "export_true23_policy_state"):
            raise TypeError(
                "true23 runner actor must implement "
                "export_true23_policy_state()"
            )
        if not hasattr(actor, "core") or not hasattr(actor, "distribution"):
            raise TypeError("true23 runner actor lacks exact core/distribution")
        self.actor = actor

    def state_dict(self) -> dict[str, torch.Tensor]:
        state = self.actor.export_true23_policy_state()
        if not isinstance(state, Mapping):
            raise TypeError("export_true23_policy_state() must return a mapping")
        copied: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                raise TypeError(
                    "exact true23 policy state must map strings to tensors"
                )
            copied[key] = value.detach().cpu().contiguous().clone()
        return copied

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        if strict is not True:
            raise ValueError("true23 policy restore must be strict")
        current = self.state_dict()
        if set(state_dict) != set(current):
            missing = sorted(set(current) - set(state_dict))
            unknown = sorted(set(state_dict) - set(current))
            raise ValueError(
                "true23 policy state keys mismatch; "
                f"missing={missing}, unknown={unknown}"
            )
        if "std" not in state_dict:
            raise ValueError("true23 policy state lacks direct action std")
        network_state = {
            key: value for key, value in state_dict.items() if key != "std"
        }
        self.actor.core.load_state_dict(network_state, strict=True)
        target_std = self.actor.distribution.std_param
        restored_std = state_dict["std"]
        if tuple(restored_std.shape) != tuple(target_std.shape):
            raise ValueError("true23 restored action std shape mismatch")
        with torch.no_grad():
            target_std.copy_(
                restored_std.to(
                    device=target_std.device,
                    dtype=target_std.dtype,
                )
            )


class True23MjlabOnPolicyRunner(_MjlabOnPolicyRunner):
    """Train and resume the exact split SONIC policy in pinned MJLab.

    The first four parameters match ``MjlabOnPolicyRunner`` so this class can
    be supplied directly as ``runner_cls`` to
    ``register_sonic_true23_tracking_task``.  A training launcher must also
    pass every lineage keyword explicitly.
    """

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
        *,
        warm_start_checkpoint_path: str | Path,
        resolved_config: Mapping[str, Any],
        source_manifest: Mapping[str, Any],
        asset_manifest: Mapping[str, Any],
        dataset_manifest: Mapping[str, Any],
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        if _MJLAB_IMPORT_ERROR is not None:
            raise RuntimeError(
                "True23MjlabOnPolicyRunner requires the pinned MJLab runtime"
            ) from _MJLAB_IMPORT_ERROR
        if not isinstance(train_cfg, dict):
            raise TypeError("train_cfg must be a dictionary")
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError as exc:
            raise ValueError("WORLD_SIZE must be an integer") from exc
        if world_size != 1:
            raise ValueError(
                "true23 lineage checkpointing currently requires one process"
            )

        warm_start_path = Path(
            warm_start_checkpoint_path
        ).expanduser().resolve()
        actor_cfg = train_cfg.get("actor")
        if not isinstance(actor_cfg, Mapping):
            raise ValueError("train_cfg.actor must be a mapping")
        configured_warm_start = actor_cfg.get("warm_start_path")
        if not isinstance(configured_warm_start, str):
            raise ValueError("train_cfg.actor.warm_start_path must be explicit")
        if Path(configured_warm_start).expanduser().resolve() != warm_start_path:
            raise ValueError(
                "runner warm start differs from actor warm_start_path"
            )
        _validate_resolved_config_binding(resolved_config, train_cfg)

        # Bind immutable inputs before actor/critic construction or any PPO
        # optimizer work performed by this runner.  Environment setup may
        # already have observed or primed the simulation.
        lineage = build_mjlab_training_lineage(
            warm_start_path,
            resolved_config=resolved_config,
            source_manifest=source_manifest,
            asset_manifest=asset_manifest,
            dataset_manifest=dataset_manifest,
        )
        self._training_lineage = validate_mjlab_training_lineage(lineage)
        self._lineage_sha256 = self._training_lineage["lineage_sha256"]

        if checkpoint_dir is None:
            if log_dir is None:
                raise ValueError(
                    "checkpoint_dir or log_dir is required for true23 training"
                )
            checkpoint_dir = log_dir
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.registry_name = registry_name
        self._last_checkpoint_path: Path | None = None
        self._last_checkpoint_update_count: int | None = None
        self._training_state_poisoned = False

        # MJLab/RSL-RL mutate nested configuration during construction.
        super().__init__(
            env,
            copy.deepcopy(train_cfg),
            log_dir,
            device,
        )
        if getattr(self, "is_distributed", False):
            raise ValueError(
                "true23 lineage checkpointing currently requires one process"
            )
        if bool(getattr(self.alg, "rnd", False)):
            raise ValueError(
                "true23 exact resume does not support auxiliary RND state"
            )
        self.completed_update_count = 0
        self.current_learning_iteration = 0
        self._policy_state_adapter = _ExactPolicyStateAdapter(
            self.alg.get_policy()
        )
        self._validate_initial_policy()
        _environment_common_step_counter(self.env)
        self._require_counter_coherence()

    @property
    def lineage_sha256(self) -> str:
        return self._lineage_sha256

    @property
    def training_lineage(self) -> dict[str, Any]:
        return copy.deepcopy(self._training_lineage)

    def _validate_initial_policy(self) -> None:
        reference_profile = self._training_lineage["warm_start"][
            "reference_profile"
        ]
        state = self._policy_state_adapter.state_dict()
        actual_hash = inspect_true23_policy_state(
            {"policy_state_dict": state},
            reference_profile=reference_profile,
        )
        expected_hash = self._training_lineage["warm_start"][
            "initial_policy_state_sha256"
        ]
        if actual_hash != expected_hash:
            raise ValueError(
                "constructed MJLab actor differs from immutable warm start"
            )

    def _require_counter_coherence(self) -> int:
        completed = _require_nonnegative_integer(
            self.completed_update_count,
            "completed_update_count",
        )
        current = _require_nonnegative_integer(
            self.current_learning_iteration,
            "current_learning_iteration",
        )
        if completed != current:
            raise RuntimeError(
                "runner update counters diverged; refusing checkpoint"
            )
        return completed

    def _trainer_state(self) -> dict[str, int | float]:
        completed = self._require_counter_coherence()
        return {
            "completed_update_count": completed,
            "current_learning_iteration": completed,
            "env_common_step_counter": _environment_common_step_counter(
                self.env
            ),
            "algorithm_learning_rate": self._algorithm_learning_rate(),
        }

    def _algorithm_learning_rate(self) -> float:
        learning_rate = _require_learning_rate(
            self.alg.learning_rate,
            "algorithm learning_rate",
        )
        param_groups = getattr(self.alg.optimizer, "param_groups", None)
        if not isinstance(param_groups, list) or not param_groups:
            raise ValueError("optimizer must expose non-empty param_groups")
        for index, group in enumerate(param_groups):
            if not isinstance(group, Mapping) or "lr" not in group:
                raise ValueError(
                    f"optimizer param_groups[{index}] lacks learning rate"
                )
            group_learning_rate = _require_learning_rate(
                group["lr"],
                f"optimizer param_groups[{index}].lr",
            )
            if group_learning_rate != learning_rate:
                raise ValueError(
                    "algorithm and optimizer learning rates diverged"
                )
        return learning_rate

    def _require_checkpointable(self) -> None:
        if bool(getattr(self, "_training_state_poisoned", False)):
            raise RuntimeError(
                "runner state was partially mutated by a failed PPO "
                "iteration; strictly load the last exact checkpoint before "
                "saving or continuing"
            )

    def save(self, path: str, infos: dict | None = None) -> None:
        """Write one complete, atomic, weights-only-safe exact resume."""

        self._require_checkpointable()
        if infos is not None:
            raise ValueError(
                "stock RSL infos are forbidden in exact true23 checkpoints"
            )
        completed = self._require_counter_coherence()
        output = Path(path).expanduser().resolve()
        saved = save_mjlab_training_checkpoint(
            output,
            policy_state_dict=self._policy_state_adapter.state_dict(),
            critic_state_dict=self.alg.critic.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(),
            update_count=completed,
            trainer_state=self._trainer_state(),
            lineage=self._training_lineage,
            overwrite=False,
        )
        self._last_checkpoint_path = saved
        self._last_checkpoint_update_count = completed

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict[str, Any]:
        """Restore exact actor, critic, optimizer, counters, and env curriculum."""

        if load_cfg is not None:
            raise ValueError("partial true23 resume is forbidden")
        if strict is not True:
            raise ValueError("true23 resume must be strict")

        requested_path = Path(path).expanduser().resolve()
        completed_before = self._require_counter_coherence()
        poisoned_before = bool(
            getattr(self, "_training_state_poisoned", False)
        )
        recovery_update_count: int | None = None
        minimum_update_count = completed_before
        if poisoned_before:
            recorded_path = getattr(self, "_last_checkpoint_path", None)
            recorded_count = getattr(
                self,
                "_last_checkpoint_update_count",
                None,
            )
            if recorded_path is None or recorded_count is None:
                raise RuntimeError(
                    "poisoned true23 runner has no recorded exact checkpoint; "
                    "discard it and resume in a new runner"
                )
            if requested_path != Path(recorded_path).expanduser().resolve():
                raise RuntimeError(
                    "poisoned true23 runner may load only its recorded last "
                    "exact checkpoint"
                )
            recovery_update_count = _require_nonnegative_integer(
                recorded_count,
                "last checkpoint update_count",
            )
            if recovery_update_count > completed_before:
                raise RuntimeError(
                    "last checkpoint update_count exceeds runner counter"
                )
            minimum_update_count = recovery_update_count

        policy_before = self._policy_state_adapter.state_dict()
        critic_before = copy.deepcopy(self.alg.critic.state_dict())
        optimizer_before = copy.deepcopy(self.alg.optimizer.state_dict())
        current_before = self.current_learning_iteration
        env_counter_before = _environment_common_step_counter(self.env)
        learning_rate_before = self.alg.learning_rate
        self._training_state_poisoned = True
        try:
            trainer_state = restore_mjlab_training_checkpoint(
                path,
                policy_module=self._policy_state_adapter,
                critic_module=self.alg.critic,
                optimizer=self.alg.optimizer,
                expected_lineage=self._training_lineage,
                minimum_update_count=minimum_update_count,
                map_location=map_location or "cpu",
            )
            completed = _require_nonnegative_integer(
                trainer_state["completed_update_count"],
                "restored completed_update_count",
            )
            current = _require_nonnegative_integer(
                trainer_state["current_learning_iteration"],
                "restored current_learning_iteration",
            )
            if completed != current:
                raise ValueError("restored true23 update counters diverge")
            if (
                recovery_update_count is not None
                and completed != recovery_update_count
            ):
                raise ValueError(
                    "restored checkpoint differs from recorded last exact "
                    "update_count"
                )
            restored_learning_rate = _require_learning_rate(
                trainer_state["algorithm_learning_rate"],
                "restored algorithm_learning_rate",
            )
            for index, group in enumerate(self.alg.optimizer.param_groups):
                group_learning_rate = _require_learning_rate(
                    group["lr"],
                    f"restored optimizer param_groups[{index}].lr",
                )
                if group_learning_rate != restored_learning_rate:
                    raise ValueError(
                        "restored algorithm and optimizer learning rates "
                        "diverge"
                    )
            self.alg.learning_rate = restored_learning_rate
            _set_environment_common_step_counter(
                self.env,
                trainer_state["env_common_step_counter"],
            )
            self.completed_update_count = completed
            self.current_learning_iteration = current
            self._require_counter_coherence()
            if recovery_update_count is not None:
                storage = getattr(self.alg, "storage", None)
                clear_storage = getattr(storage, "clear", None)
                if not callable(clear_storage):
                    raise RuntimeError(
                        "poisoned true23 recovery requires pinned RSL rollout "
                        "storage.clear(); discard runner and resume in a new "
                        "process"
                    )
                clear_storage()
                storage_step = _require_nonnegative_integer(
                    getattr(storage, "step", None),
                    "RSL rollout storage step",
                )
                if storage_step != 0:
                    raise RuntimeError(
                        "RSL rollout storage did not reset after poisoned "
                        "true23 recovery"
                    )
        except BaseException:
            try:
                self._policy_state_adapter.load_state_dict(
                    policy_before,
                    strict=True,
                )
                self.alg.critic.load_state_dict(critic_before, strict=True)
                self.alg.optimizer.load_state_dict(optimizer_before)
                self.completed_update_count = completed_before
                self.current_learning_iteration = current_before
                self.alg.learning_rate = learning_rate_before
                _set_environment_common_step_counter(
                    self.env,
                    env_counter_before,
                )
            except BaseException as rollback_error:
                self._training_state_poisoned = True
                raise RuntimeError(
                    "true23 checkpoint restore rollback failed; discard this "
                    "runner and resume in a new process"
                ) from rollback_error
            self._training_state_poisoned = poisoned_before
            raise

        self._training_state_poisoned = False
        self._last_checkpoint_path = requested_path
        self._last_checkpoint_update_count = completed
        return {
            "trainer_state": dict(trainer_state),
            "lineage_sha256": self._lineage_sha256,
        }

    def _numbered_checkpoint_path(self, update_count: int) -> Path:
        count = _require_nonnegative_integer(
            update_count,
            "checkpoint update_count",
        )
        return self.checkpoint_dir / f"model_{count}.pt"

    def _save_numbered_checkpoint(self) -> None:
        completed = self._require_counter_coherence()
        output = self._numbered_checkpoint_path(completed)
        if (
            self._last_checkpoint_path == output
            and self._last_checkpoint_update_count == completed
        ):
            return
        self.save(str(output))

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run PPO; counters count successful outer ``alg.update()`` calls."""

        iterations = _require_nonnegative_integer(
            num_learning_iterations,
            "num_learning_iterations",
        )
        if type(init_at_random_ep_len) is not bool:
            raise TypeError("init_at_random_ep_len must be bool")
        save_interval = _require_nonnegative_integer(
            self.cfg["save_interval"],
            "save_interval",
        )
        if save_interval == 0:
            raise ValueError("save_interval must be greater than zero")
        self._require_checkpointable()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        # ``model_0`` means zero completed outer PPO iterations. Environment
        # setup may already have refreshed observations; save before this
        # learning call collects its first rollout or updates.
        if self.completed_update_count == 0:
            self._save_numbered_checkpoint()

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if getattr(self, "is_distributed", False):
            raise RuntimeError("true23 runner does not support distributed PPO")
        self.logger.init_logging_writer()

        start_update_count = self._require_counter_coherence()
        total_update_count = start_update_count + iterations
        for update_index in range(start_update_count, total_update_count):
            start = time.time()
            try:
                with torch.inference_mode():
                    for _ in range(self.cfg["num_steps_per_env"]):
                        actions = self.alg.act(obs)
                        obs, rewards, dones, extras = self.env.step(
                            actions.to(self.env.device)
                        )
                        if self.cfg.get("check_for_nan", True):
                            from rsl_rl.utils import check_nan

                            check_nan(obs, rewards, dones)
                        obs = obs.to(self.device)
                        rewards = rewards.to(self.device)
                        dones = dones.to(self.device)
                        self.alg.process_env_step(
                            obs,
                            rewards,
                            dones,
                            extras,
                        )
                        intrinsic_rewards = (
                            self.alg.intrinsic_rewards
                            if bool(getattr(self.alg, "rnd", False))
                            else None
                        )
                        self.logger.process_env_step(
                            rewards,
                            dones,
                            extras,
                            intrinsic_rewards,
                        )
                        # MJLab retains the last reset log in ``env.extras``.
                        # RSL stores the log object by reference, so replace
                        # the shared sink after processing.  Popping it makes
                        # per-step reward metrics such as ``soft_landing``
                        # fail on their next ``env.extras["log"]`` write;
                        # clearing it in-place would erase queued metrics.
                        extras["log"] = {}

                    stop = time.time()
                    collect_time = stop - start
                    start = stop
                    self.alg.compute_returns(obs)

                loss_dict = self.alg.update()
            except BaseException:
                # RSL performs multiple optimizer steps inside one alg.update.
                # An exception can therefore leave weights/optimizer/storage
                # partially mutated even though no PPO iteration completed.
                self._training_state_poisoned = True
                raise
            learn_time = time.time() - start

            # ``update_count`` means completed outer PPO iterations: one
            # successful alg.update() call, regardless of its internal
            # epoch/minibatch optimizer-step count.
            completed = update_index + 1
            self.completed_update_count = completed
            self.current_learning_iteration = completed

            self.logger.log(
                it=update_index,
                start_it=start_update_count,
                total_it=total_update_count,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )

            if completed % save_interval == 0:
                self._save_numbered_checkpoint()

        self._save_numbered_checkpoint()
        self.logger.stop_logging_writer()

    def export_policy_to_jit(
        self,
        path: str,
        filename: str = "policy.pt",
    ) -> None:
        del path, filename
        raise RuntimeError(
            "stock RSL JIT export is forbidden for true23; use the "
            "hash-bound SONIC pair exporter"
        )

    def export_policy_to_onnx(
        self,
        path: str,
        filename: str = "policy.onnx",
        verbose: bool = False,
    ) -> None:
        del path, filename, verbose
        raise RuntimeError(
            "stock RSL ONNX export is forbidden for true23; use the "
            "hash-bound SONIC pair exporter"
        )
