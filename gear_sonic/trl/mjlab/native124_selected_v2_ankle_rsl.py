"""Bind selected native124 ankle adaptation to the installed RSL-RL PPO.

The configurator mutates an already constructed ``MjlabOnPolicyRunner`` only.
It uses RSL's own ``PPO``, actor, critic, and optimizer surfaces; it does not
implement a rollout loop or a second PPO.  Selected checkpoint loading is
actor-only.  The causal privileged critic is a fresh 256-input model and the
selected checkpoint's inherited broad critic/Adam are never loaded.

Fixed synthetic 124-vectors prove model-boundary parity only.  They make no
claim about causal q9/q10 observation parity or environment semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import math
import os
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch import nn

from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
    _ACTOR_FULLY_FROZEN_KEYS,
    ACTION_DIM,
    ACTOR_STATE_SCHEMA,
    NORMALIZATION_EPSILON,
    OBSERVATION_DIM,
    AnkleRowConfig,
    CheckpointPublication,
    FreshAdam,
    SourceLineage,
    TensorSpec,
    _exact_mapping_keys,
    _existing_regular_file,
    _gradient_mask_hooks,
    _load_safe_mapping,
    _load_selected_source,
    _require_sha256,
    _same_tensor_bytes,
    _validated_tensor_state,
    safe_tree_sha256,
    sha256_file,
    tensor_state_sha256,
)
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ONNX_SHA256,
    Native124Checkpoint21204Policy,
    load_checkpoint21204_binding,
)

CRITIC_OBSERVATION_DIM = 256
PARITY_ATOL = 1.0e-5
PARITY_RTOL = 1.0e-5
SELECTED_LOAD_CONFIG = {
    "actor": True,
    "critic": False,
    "iteration": False,
    "optimizer": False,
    "rnd": False,
}
WARM_LOAD_CONFIG = {
    "actor": True,
    "critic": True,
    "iteration": False,
    "optimizer": False,
    "rnd": False,
}
SNAPSHOT_KIND = "g1_true23_native124_selected_v2_ankle_rsl256_warm_restart"
SNAPSHOT_SCHEMA_VERSION = 1
_SNAPSHOT_ROOT_KEYS = {"actor_state_dict", "critic_state_dict", "metadata"}

CRITIC_STATE_SCHEMA = {
    "obs_normalizer._mean": TensorSpec((1, CRITIC_OBSERVATION_DIM), torch.float32),
    "obs_normalizer._var": TensorSpec((1, CRITIC_OBSERVATION_DIM), torch.float32),
    "obs_normalizer._std": TensorSpec((1, CRITIC_OBSERVATION_DIM), torch.float32),
    "obs_normalizer.count": TensorSpec((), torch.int64),
    "mlp.0.weight": TensorSpec((512, CRITIC_OBSERVATION_DIM), torch.float32),
    "mlp.0.bias": TensorSpec((512,), torch.float32),
    "mlp.2.weight": TensorSpec((256, 512), torch.float32),
    "mlp.2.bias": TensorSpec((256,), torch.float32),
    "mlp.4.weight": TensorSpec((128, 256), torch.float32),
    "mlp.4.bias": TensorSpec((128,), torch.float32),
    "mlp.6.weight": TensorSpec((1, 128), torch.float32),
    "mlp.6.bias": TensorSpec((1,), torch.float32),
}


def _mro_contains(value: Any, module_name: str, class_name: str) -> bool:
    return any(base.__module__ == module_name and base.__name__ == class_name for base in type(value).__mro__)


def _require_actual_rsl_objects(runner: Any) -> tuple[Any, Any, Any]:
    if not _mro_contains(runner, "mjlab.rl.runner", "MjlabOnPolicyRunner"):
        raise TypeError("runner must inherit installed mjlab.rl.runner.MjlabOnPolicyRunner")
    alg = getattr(runner, "alg", None)
    if not _mro_contains(alg, "rsl_rl.algorithms.ppo", "PPO"):
        raise TypeError("runner.alg must be installed rsl_rl.algorithms.PPO")
    actor = getattr(alg, "actor", None)
    critic = getattr(alg, "critic", None)
    if not _mro_contains(actor, "rsl_rl.models.mlp_model", "MLPModel"):
        raise TypeError("runner.alg.actor must be installed rsl_rl.models.MLPModel")
    if not _mro_contains(critic, "rsl_rl.models.mlp_model", "MLPModel"):
        raise TypeError("runner.alg.critic must be installed rsl_rl.models.MLPModel")
    if getattr(alg, "rnd", None) is not None:
        raise ValueError("native124 ankle adaptation requires RND disabled")
    return alg, actor, critic


def _require_linear(module: Any, input_dim: int, output_dim: int, context: str) -> None:
    if type(module) is not nn.Linear or module.in_features != input_dim or module.out_features != output_dim:
        raise ValueError(f"{context} must be Linear({input_dim}, {output_dim})")


def _validate_actual_actor(actor: Any) -> tuple[str, dict[str, torch.Tensor]]:
    groups = getattr(actor, "obs_groups", None)
    if type(groups) is not list or len(groups) != 1 or type(groups[0]) is not str or not groups[0]:
        raise ValueError("actual RSL actor must consume exactly one named 124-value observation group")
    if getattr(actor, "obs_dim", None) != OBSERVATION_DIM:
        raise ValueError("actual RSL actor observation dimension must be 124")
    if getattr(actor, "obs_normalization", None) is not True:
        raise ValueError("actual RSL actor must be constructed with observation normalization")
    normalizer = getattr(actor, "obs_normalizer", None)
    if not _mro_contains(normalizer, "rsl_rl.modules.normalization", "EmpiricalNormalization"):
        raise TypeError("actual RSL actor normalizer must be EmpiricalNormalization")
    if getattr(normalizer, "eps", None) != NORMALIZATION_EPSILON:
        raise ValueError("actual RSL actor normalization epsilon mismatch")
    mlp = getattr(actor, "mlp", None)
    if not isinstance(mlp, nn.Sequential) or len(mlp) != 7:
        raise ValueError("actual RSL actor MLP topology mismatch")
    _require_linear(mlp[0], OBSERVATION_DIM, 512, "actor.mlp[0]")
    _require_linear(mlp[2], 512, 256, "actor.mlp[2]")
    _require_linear(mlp[4], 256, 128, "actor.mlp[4]")
    _require_linear(mlp[6], 128, ACTION_DIM, "actor.mlp[6]")
    if any(type(mlp[index]) is not nn.ELU for index in (1, 3, 5)):
        raise ValueError("actual RSL actor activation must be ELU")
    distribution = getattr(actor, "distribution", None)
    if not _mro_contains(distribution, "rsl_rl.modules.distribution", "GaussianDistribution"):
        raise TypeError("actual RSL actor distribution must be GaussianDistribution")
    if getattr(distribution, "std_type", None) != "scalar" or not hasattr(distribution, "std_param"):
        raise ValueError("actual RSL actor must use scalar-space std_param")
    state = _validated_tensor_state(actor.state_dict(), ACTOR_STATE_SCHEMA, context="actual actor")
    return groups[0], state


def _validate_actual_critic(critic: Any) -> dict[str, torch.Tensor]:
    if getattr(critic, "obs_groups", None) != ["critic"]:
        raise ValueError("actual RSL critic must consume only privileged 'critic' group")
    if getattr(critic, "obs_dim", None) != CRITIC_OBSERVATION_DIM:
        raise ValueError("actual RSL privileged critic observation dimension must be 256")
    if getattr(critic, "obs_normalization", None) is not True:
        raise ValueError("actual RSL critic must be constructed with observation normalization")
    if getattr(critic, "distribution", None) is not None:
        raise ValueError("actual RSL critic must be deterministic")
    normalizer = getattr(critic, "obs_normalizer", None)
    if not _mro_contains(normalizer, "rsl_rl.modules.normalization", "EmpiricalNormalization"):
        raise TypeError("actual RSL critic normalizer must be EmpiricalNormalization")
    mlp = getattr(critic, "mlp", None)
    if not isinstance(mlp, nn.Sequential) or len(mlp) != 7:
        raise ValueError("actual RSL critic MLP topology mismatch")
    _require_linear(mlp[0], CRITIC_OBSERVATION_DIM, 512, "critic.mlp[0]")
    _require_linear(mlp[2], 512, 256, "critic.mlp[2]")
    _require_linear(mlp[4], 256, 128, "critic.mlp[4]")
    _require_linear(mlp[6], 128, 1, "critic.mlp[6]")
    if any(type(mlp[index]) is not nn.ELU for index in (1, 3, 5)):
        raise ValueError("actual RSL critic activation must be ELU")
    return _validated_tensor_state(critic.state_dict(), CRITIC_STATE_SCHEMA, context="actual critic")


def _fixed_parity_probes() -> tuple[tuple[str, np.ndarray], ...]:
    ramp = np.linspace(-1.0, 1.0, OBSERVATION_DIM, dtype=np.float32)[None, :]
    impulse_first = np.zeros((1, OBSERVATION_DIM), dtype=np.float32)
    impulse_first[0, 0] = 1.0
    impulse_middle = np.zeros((1, OBSERVATION_DIM), dtype=np.float32)
    impulse_middle[0, OBSERVATION_DIM // 2] = -0.75
    impulse_last = np.zeros((1, OBSERVATION_DIM), dtype=np.float32)
    impulse_last[0, -1] = 0.5
    return (
        ("zero", np.zeros((1, OBSERVATION_DIM), dtype=np.float32)),
        ("ramp", ramp),
        ("impulse_first", impulse_first),
        ("impulse_middle", impulse_middle),
        ("impulse_last", impulse_last),
    )


def _verify_deterministic_parity(
    actor: Any,
    actor_observation_group: str,
    policy: Native124Checkpoint21204Policy,
) -> dict[str, Any]:
    device = next(actor.parameters()).device
    maximum = 0.0
    worst_case = ""
    with torch.inference_mode():
        for name, probe in _fixed_parity_probes():
            actual = (
                actor(
                    {actor_observation_group: torch.from_numpy(probe).to(device)},
                    stochastic_output=False,
                )
                .detach()
                .cpu()
                .numpy()
            )
            expected = policy.run(probe)[None, :]
            error = float(np.max(np.abs(actual - expected)))
            if error > maximum:
                maximum = error
                worst_case = name
            if not np.allclose(actual, expected, atol=PARITY_ATOL, rtol=PARITY_RTOL):
                raise ValueError(f"actual RSL actor deterministic parity failed at {name}: max_abs={error:.9g}")
    return {
        "atol": PARITY_ATOL,
        "case_count": len(_fixed_parity_probes()),
        "max_absolute_error": maximum,
        "onnx_sha256": ONNX_SHA256,
        "passed": True,
        "probe_names": [name for name, _ in _fixed_parity_probes()],
        "rtol": PARITY_RTOL,
        "worst_case": worst_case,
    }


def _install_frozen_normalizer(actor: Any) -> Any:
    normalizer = actor.obs_normalizer
    count = int(normalizer.count.detach().cpu().item())
    actor.obs_normalization = False
    normalizer.until = count

    def keep_eval(module: Any, _mode: bool = True) -> Any:
        nn.Module.train(module, False)
        return module

    normalizer.train = MethodType(keep_eval, normalizer)
    normalizer.eval()
    return normalizer.train.__func__


def _install_masked_distribution_sample(distribution: Any, rows: tuple[int, ...]) -> Any:
    original_sample = distribution.sample

    def masked_sample(module: Any) -> torch.Tensor:
        sampled = original_sample()
        result = module.mean.clone()
        result[..., list(rows)] = sampled[..., list(rows)]
        return result

    distribution.sample = MethodType(masked_sample, distribution)
    distribution._native124_trainable_sample_rows = rows
    return distribution.sample.__func__


def _validate_restart_frozen_actor(
    actor_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    config: AnkleRowConfig,
) -> None:
    for key in _ACTOR_FULLY_FROZEN_KEYS:
        if not _same_tensor_bytes(actor_state[key], source_state[key]):
            raise ValueError(f"RSL warm restart changed frozen actor tensor: {key}")
    frozen_rows = tuple(index for index in range(ACTION_DIM) if index not in config.trainable_hardware_rows)
    index = torch.tensor(frozen_rows, dtype=torch.long)
    for key in ("mlp.6.weight", "mlp.6.bias"):
        if not _same_tensor_bytes(
            actor_state[key].index_select(0, index), source_state[key].index_select(0, index)
        ):
            raise ValueError(f"RSL warm restart changed frozen actor rows: {key}")


class _ActualRslFrozenGuard:
    def __init__(
        self,
        actor: Any,
        config: AnkleRowConfig,
        normalizer_train_function: Any,
        distribution_sample_function: Any,
    ) -> None:
        self.actor = actor
        self.config = config
        self.normalizer_train_function = normalizer_train_function
        self.distribution_sample_function = distribution_sample_function
        self.trainable_rows = config.trainable_hardware_rows
        self.frozen_rows = tuple(index for index in range(ACTION_DIM) if index not in self.trainable_rows)
        state = actor.state_dict()
        self.full = {key: state[key].detach().clone() for key in _ACTOR_FULLY_FROZEN_KEYS}
        index = torch.tensor(self.frozen_rows, dtype=torch.long, device=actor.mlp[6].weight.device)
        self.output_weight = actor.mlp[6].weight.detach().index_select(0, index).clone()
        self.output_bias = actor.mlp[6].bias.detach().index_select(0, index).clone()

    def assert_intact(self) -> None:
        normalizer = self.actor.obs_normalizer
        distribution = self.actor.distribution
        if (
            self.actor.obs_normalization is not False
            or normalizer.training
            or normalizer.until != int(normalizer.count.detach().cpu().item())
            or normalizer.train.__func__ is not self.normalizer_train_function
        ):
            raise RuntimeError("actual RSL actor normalizer freeze/update contract drift")
        if (
            distribution.sample.__func__ is not self.distribution_sample_function
            or getattr(distribution, "_native124_trainable_sample_rows", None) != self.trainable_rows
        ):
            raise RuntimeError("actual RSL actor stochastic sample-mask contract drift")
        state = self.actor.state_dict()
        for key, expected in self.full.items():
            if not _same_tensor_bytes(state[key], expected):
                raise RuntimeError(f"actual RSL frozen actor tensor changed: {key}")
        index = torch.tensor(self.frozen_rows, dtype=torch.long, device=self.actor.mlp[6].weight.device)
        if not _same_tensor_bytes(self.actor.mlp[6].weight.detach().index_select(0, index), self.output_weight):
            raise RuntimeError("actual RSL frozen actor output weight rows changed")
        if not _same_tensor_bytes(self.actor.mlp[6].bias.detach().index_select(0, index), self.output_bias):
            raise RuntimeError("actual RSL frozen actor output bias rows changed")

    def restore(self) -> None:
        with torch.no_grad():
            state = self.actor.state_dict()
            for key, expected in self.full.items():
                state[key].copy_(expected)
            index = torch.tensor(self.frozen_rows, dtype=torch.long, device=self.actor.mlp[6].weight.device)
            self.actor.mlp[6].weight.index_copy_(0, index, self.output_weight.to(self.actor.mlp[6].weight.device))
            self.actor.mlp[6].bias.index_copy_(0, index, self.output_bias.to(self.actor.mlp[6].bias.device))
        self.actor.obs_normalization = False
        self.actor.obs_normalizer.until = int(self.actor.obs_normalizer.count.detach().cpu().item())
        self.actor.obs_normalizer.eval()


class RslSelectedV2AnkleIntegration:
    """Guard and warm-checkpoint surface attached to one actual RSL runner."""

    def __init__(
        self,
        *,
        runner: Any,
        config: AnkleRowConfig,
        lineage: SourceLineage,
        actor_observation_group: str,
        parity: Mapping[str, Any],
        fresh_critic_state_sha256: str,
        prior_completed_optimizer_steps: int,
        normalizer_train_function: Any,
        distribution_sample_function: Any,
        gradient_handles: tuple[Any, ...],
    ) -> None:
        self.runner = runner
        self.alg = runner.alg
        self.actor = self.alg.actor
        self.critic = self.alg.critic
        self.config = config
        self.lineage = lineage
        self.actor_observation_group = actor_observation_group
        self.parity = dict(parity)
        self.fresh_critic_state_sha256 = fresh_critic_state_sha256
        self.prior_completed_optimizer_steps = prior_completed_optimizer_steps
        self.optimizer_steps = 0
        self._poisoned = False
        self._gradient_handles = gradient_handles
        self._guard = _ActualRslFrozenGuard(
            self.actor,
            config,
            normalizer_train_function,
            distribution_sample_function,
        )
        self._optimizer_pre_handle = self.alg.optimizer.register_step_pre_hook(self._optimizer_pre_step)
        self._optimizer_post_handle = self.alg.optimizer.register_step_post_hook(self._optimizer_post_step)
        self.assert_frozen_invariants()

    @property
    def completed_optimizer_steps_total(self) -> int:
        return self.prior_completed_optimizer_steps + self.optimizer_steps

    def _mark_poisoned(self) -> None:
        self._poisoned = True
        self.runner._native124_selected_v2_ankle_poisoned = True
        if hasattr(self.runner, "_training_state_poisoned"):
            self.runner._training_state_poisoned = True

    def _require_healthy(self) -> None:
        if self._poisoned or bool(getattr(self.runner, "_native124_selected_v2_ankle_poisoned", False)):
            raise RuntimeError("actual RSL ankle adaptation state is poisoned; discard runner")

    def _expected_optimizer_parameters(self) -> list[nn.Parameter]:
        return [
            self.actor.mlp[6].weight,
            self.actor.mlp[6].bias,
            *list(self.critic.parameters()),
        ]

    def _assert_optimizer_contract(self) -> None:
        optimizer = self.alg.optimizer
        if type(optimizer) is not FreshAdam or len(optimizer.param_groups) != 1:
            raise RuntimeError("actual RSL PPO fresh-Adam contract drift")
        group = optimizer.param_groups[0]
        if group.get("weight_decay") != 0.0 or group.get("lr") != self.alg.learning_rate:
            raise RuntimeError("actual RSL PPO Adam LR/weight-decay contract drift")
        actual = group.get("params")
        expected = self._expected_optimizer_parameters()
        if (
            type(actual) is not list
            or len(actual) != len(expected)
            or any(left is not right for left, right in zip(actual, expected))
        ):
            raise RuntimeError("actual RSL PPO optimizer parameter membership/order drift")
        for name, parameter in self.actor.named_parameters():
            should_train = name in {"mlp.6.weight", "mlp.6.bias"}
            if parameter.requires_grad is not should_train:
                raise RuntimeError(f"actual RSL actor requires_grad drift: {name}")
        for parameter in self.critic.parameters():
            if not parameter.requires_grad:
                raise RuntimeError("actual RSL critic must remain trainable")
        for parameter in (self.actor.mlp[6].weight, self.actor.mlp[6].bias):
            state = optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                moment = state.get(key)
                if moment is not None and bool((moment[list(self._guard.frozen_rows)] != 0).any()):
                    raise RuntimeError(f"actual RSL PPO {key} contains frozen-row state")

    def _optimizer_pre_step(
        self, _optimizer: torch.optim.Optimizer, _args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> None:
        self._require_healthy()
        try:
            self._assert_optimizer_contract()
            self._guard.assert_intact()
        except Exception:
            self._guard.restore()
            self._mark_poisoned()
            raise

    def _optimizer_post_step(
        self, _optimizer: torch.optim.Optimizer, _args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> None:
        try:
            self._assert_optimizer_contract()
            self._guard.assert_intact()
            self.optimizer_steps += 1
        except Exception:
            self._guard.restore()
            self._mark_poisoned()
            raise

    def assert_frozen_invariants(self) -> None:
        self._require_healthy()
        self._assert_optimizer_contract()
        self._guard.assert_intact()

    def _metadata(
        self,
        actor_state_sha256: str,
        critic_state_sha256: str,
    ) -> dict[str, Any]:
        return {
            "integration": {
                "actor_observation_dim": OBSERVATION_DIM,
                "actor_observation_group": self.actor_observation_group,
                "critic_observation_dim": CRITIC_OBSERVATION_DIM,
                "critic_observation_groups": ["critic"],
                "critic_source": "fresh_causal_privileged_256_never_selected_broad",
                "row_config": self.config.metadata(),
                "selected_load_config": dict(SELECTED_LOAD_CONFIG),
                "warm_load_config": dict(WARM_LOAD_CONFIG),
            },
            "kind": SNAPSHOT_KIND,
            "optimizer_contract": {
                "class": "torch.optim.Adam",
                "completed_optimizer_steps_total": self.completed_optimizer_steps_total,
                "iteration_resets_on_load": True,
                "learning_rate": float(self.alg.learning_rate),
                "load_state_dict": "forbidden",
                "resume_state_saved": False,
                "source_state_loaded": False,
                "weight_decay": 0.0,
            },
            "parity_contract": {
                "atol": PARITY_ATOL,
                "case_count": len(_fixed_parity_probes()),
                "onnx_sha256": ONNX_SHA256,
                "probe_names": [name for name, _ in _fixed_parity_probes()],
                "rtol": PARITY_RTOL,
            },
            "payload": {
                "actor_state_sha256": actor_state_sha256,
                "critic_state_sha256": critic_state_sha256,
            },
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "source": self.lineage.metadata(),
        }

    def build_warm_restart(self) -> dict[str, Any]:
        self.assert_frozen_invariants()
        actor_state = _validated_tensor_state(
            self.actor.state_dict(), ACTOR_STATE_SCHEMA, context="RSL warm actor"
        )
        critic_state = _validated_tensor_state(
            self.critic.state_dict(), CRITIC_STATE_SCHEMA, context="RSL warm critic"
        )
        return {
            "actor_state_dict": actor_state,
            "critic_state_dict": critic_state,
            "metadata": self._metadata(tensor_state_sha256(actor_state), tensor_state_sha256(critic_state)),
        }

    def save_warm_restart(self, path: str | Path) -> CheckpointPublication:
        output = Path(path).expanduser().resolve(strict=False)
        if output.suffix.lower() != ".pt":
            raise ValueError("actual RSL warm restart path must end in .pt")
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to overwrite actual RSL warm restart: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        torch.save(self.build_warm_restart(), buffer)
        content = buffer.getvalue()
        descriptor: int | None = None
        created = False
        try:
            descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            created = True
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            if created and os.path.lexists(output):
                output.unlink()
            raise
        return CheckpointPublication(output, hashlib.sha256(content).hexdigest())


def _load_warm_restart(
    path: str | Path,
    *,
    expected_sha256: str,
    source_actor_state: Mapping[str, torch.Tensor],
    expected_metadata_without_payload: Mapping[str, Any],
    config: AnkleRowConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    restart = _existing_regular_file(path, "actual RSL warm restart")
    actual_hash = sha256_file(restart)
    if actual_hash != _require_sha256(expected_sha256, "expected_restart_sha256"):
        raise ValueError(
            f"actual RSL warm restart SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}"
        )
    payload = _exact_mapping_keys(
        _load_safe_mapping(restart, "actual RSL warm restart"),
        _SNAPSHOT_ROOT_KEYS,
        "actual RSL warm restart",
    )
    if sha256_file(restart) != actual_hash:
        raise RuntimeError("actual RSL warm restart changed while loading")
    actor_state = _validated_tensor_state(
        payload["actor_state_dict"], ACTOR_STATE_SCHEMA, context="RSL restart actor"
    )
    critic_state = _validated_tensor_state(
        payload["critic_state_dict"], CRITIC_STATE_SCHEMA, context="RSL restart critic"
    )
    _validate_restart_frozen_actor(actor_state, source_actor_state, config)
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("actual RSL warm restart metadata must be mapping")
    expected_metadata = dict(expected_metadata_without_payload)
    optimizer_contract = metadata.get("optimizer_contract")
    if not isinstance(optimizer_contract, Mapping):
        raise ValueError("actual RSL warm restart optimizer metadata must be mapping")
    completed = optimizer_contract.get("completed_optimizer_steps_total")
    if type(completed) is not int or completed < 0:
        raise ValueError("actual RSL warm restart completed optimizer steps invalid")
    expected_optimizer = dict(expected_metadata["optimizer_contract"])
    expected_optimizer["completed_optimizer_steps_total"] = completed
    expected_metadata["optimizer_contract"] = expected_optimizer
    expected_metadata["payload"] = {
        "actor_state_sha256": tensor_state_sha256(actor_state),
        "critic_state_sha256": tensor_state_sha256(critic_state),
    }
    if metadata != expected_metadata:
        raise ValueError("actual RSL warm restart metadata/config/payload drift")
    return actor_state, critic_state, completed


def _install_runner_checkpoint_boundary(runner: Any, integration: RslSelectedV2AnkleIntegration) -> None:
    def warm_save(_runner: Any, path: str, infos: dict[str, Any] | None = None) -> CheckpointPublication:
        if infos is not None:
            raise ValueError("stock RSL infos are forbidden in ankle warm restarts")
        return integration.save_warm_restart(path)

    def reject_stock_load(_runner: Any, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "stock RSL load is forbidden after ankle configuration; construct a fresh runner and "
            "configure it with the hash-bound warm restart"
        )

    runner.save = MethodType(warm_save, runner)
    runner.load = MethodType(reject_stock_load, runner)


def configure_selected_v2_ankle_rsl_runner(
    runner: Any,
    *,
    repository_root: str | Path,
    config: AnkleRowConfig | None = None,
    restart_path: str | Path | None = None,
    expected_restart_sha256: str | None = None,
) -> RslSelectedV2AnkleIntegration:
    """Configure one constructed actual RSL PPO for targeted ankle adaptation."""

    if (restart_path is None) != (expected_restart_sha256 is None):
        raise ValueError("restart_path and expected_restart_sha256 must be supplied together")
    selected_config = AnkleRowConfig() if config is None else config
    if type(selected_config) is not AnkleRowConfig:
        raise TypeError("config must be exact AnkleRowConfig")
    alg, actor, critic = _require_actual_rsl_objects(runner)
    actor_group, _initial_actor_state = _validate_actual_actor(actor)
    fresh_critic_state = _validate_actual_critic(critic)
    fresh_critic_hash = tensor_state_sha256(fresh_critic_state)
    source_optimizer = alg.optimizer
    source_optimizer_state = source_optimizer.state_dict()
    source_optimizer_hash = safe_tree_sha256(source_optimizer_state, context="pre-configuration RSL optimizer")
    storage_step = getattr(getattr(alg, "storage", None), "step", 0)
    if type(storage_step) is not int or storage_step != 0:
        raise ValueError("actual RSL rollout storage must be empty before configuration")
    if isinstance(getattr(alg, "learning_rate", None), bool) or not isinstance(
        getattr(alg, "learning_rate", None), (float, int)
    ):
        raise ValueError("actual RSL PPO learning_rate must be numeric")
    learning_rate = float(alg.learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("actual RSL PPO learning_rate must be finite and positive")

    binding = load_checkpoint21204_binding(repository_root)
    source_actor_state, _selected_critic_state, lineage = _load_selected_source(binding.checkpoint_path)
    selected_policy = Native124Checkpoint21204Policy(binding)
    load_iteration = alg.load({"actor_state_dict": source_actor_state}, dict(SELECTED_LOAD_CONFIG), True)
    if load_iteration is not False:
        raise RuntimeError("actual RSL selected actor-only load changed iteration contract")
    loaded_actor_state = _validated_tensor_state(
        actor.state_dict(), ACTOR_STATE_SCHEMA, context="loaded actual RSL actor"
    )
    if tensor_state_sha256(loaded_actor_state) != lineage.actor_state_sha256:
        raise RuntimeError("actual RSL selected actor load changed checkpoint tensors")
    if tensor_state_sha256(_validate_actual_critic(critic)) != fresh_critic_hash:
        raise RuntimeError("selected actor-only load changed fresh causal critic")
    if fresh_critic_hash == lineage.critic_state_sha256:
        raise RuntimeError("fresh causal critic unexpectedly equals selected inherited broad critic")
    if (
        alg.optimizer is not source_optimizer
        or safe_tree_sha256(alg.optimizer.state_dict(), context="post-actor-load RSL optimizer")
        != source_optimizer_hash
    ):
        raise RuntimeError("selected actor-only load changed source optimizer")
    parity = _verify_deterministic_parity(actor, actor_group, selected_policy)

    prior_completed_steps = 0
    warm_actor_state: dict[str, torch.Tensor] | None = None
    warm_critic_state: dict[str, torch.Tensor] | None = None
    if restart_path is not None:
        if expected_restart_sha256 is None:  # pragma: no cover - guarded above
            raise RuntimeError("internal actual RSL warm restart hash drift")
        expected_stub = {
            "integration": {
                "actor_observation_dim": OBSERVATION_DIM,
                "actor_observation_group": actor_group,
                "critic_observation_dim": CRITIC_OBSERVATION_DIM,
                "critic_observation_groups": ["critic"],
                "critic_source": "fresh_causal_privileged_256_never_selected_broad",
                "row_config": selected_config.metadata(),
                "selected_load_config": dict(SELECTED_LOAD_CONFIG),
                "warm_load_config": dict(WARM_LOAD_CONFIG),
            },
            "kind": SNAPSHOT_KIND,
            "optimizer_contract": {
                "class": "torch.optim.Adam",
                "completed_optimizer_steps_total": 0,
                "iteration_resets_on_load": True,
                "learning_rate": learning_rate,
                "load_state_dict": "forbidden",
                "resume_state_saved": False,
                "source_state_loaded": False,
                "weight_decay": 0.0,
            },
            "parity_contract": {
                "atol": PARITY_ATOL,
                "case_count": len(_fixed_parity_probes()),
                "onnx_sha256": ONNX_SHA256,
                "probe_names": [name for name, _ in _fixed_parity_probes()],
                "rtol": PARITY_RTOL,
            },
            "payload": {},
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "source": lineage.metadata(),
        }
        warm_actor_state, warm_critic_state, prior_completed_steps = _load_warm_restart(
            restart_path,
            expected_sha256=expected_restart_sha256,
            source_actor_state=source_actor_state,
            expected_metadata_without_payload=expected_stub,
            config=selected_config,
        )
        warm_iteration = alg.load(
            {
                "actor_state_dict": warm_actor_state,
                "critic_state_dict": warm_critic_state,
            },
            dict(WARM_LOAD_CONFIG),
            True,
        )
        if warm_iteration is not False:
            raise RuntimeError("actual RSL warm load changed iteration contract")

    actor.requires_grad_(False)
    actor.mlp[6].weight.requires_grad_(True)
    actor.mlp[6].bias.requires_grad_(True)
    critic.requires_grad_(True)
    normalizer_train_function = _install_frozen_normalizer(actor)
    distribution_sample_function = _install_masked_distribution_sample(
        actor.distribution, selected_config.trainable_hardware_rows
    )
    if restart_path is None:
        post_mask_parity = _verify_deterministic_parity(actor, actor_group, selected_policy)
        parity = dict(parity)
        parity["post_sample_mask_max_absolute_error"] = post_mask_parity["max_absolute_error"]
        parity["post_sample_mask_passed"] = post_mask_parity["passed"]
    gradient_handles = _gradient_mask_hooks(actor, selected_config.trainable_hardware_rows)
    optimizer_parameters = [actor.mlp[6].weight, actor.mlp[6].bias, *list(critic.parameters())]
    alg.optimizer = FreshAdam(optimizer_parameters, lr=learning_rate, weight_decay=0.0)
    if alg.optimizer.state:
        raise RuntimeError("actual RSL fresh Adam unexpectedly contains state")
    runner.current_learning_iteration = 0
    if hasattr(runner, "completed_update_count"):
        runner.completed_update_count = 0
    runner._native124_selected_v2_ankle_poisoned = False
    if hasattr(runner, "_training_state_poisoned"):
        runner._training_state_poisoned = False
    integration = RslSelectedV2AnkleIntegration(
        runner=runner,
        config=selected_config,
        lineage=lineage,
        actor_observation_group=actor_group,
        parity=parity,
        fresh_critic_state_sha256=fresh_critic_hash,
        prior_completed_optimizer_steps=prior_completed_steps,
        normalizer_train_function=normalizer_train_function,
        distribution_sample_function=distribution_sample_function,
        gradient_handles=gradient_handles,
    )
    runner._native124_selected_v2_ankle_integration = integration
    _install_runner_checkpoint_boundary(runner, integration)
    return integration


__all__ = [
    "CRITIC_OBSERVATION_DIM",
    "PARITY_ATOL",
    "PARITY_RTOL",
    "RslSelectedV2AnkleIntegration",
    "SELECTED_LOAD_CONFIG",
    "WARM_LOAD_CONFIG",
    "configure_selected_v2_ankle_rsl_runner",
]
