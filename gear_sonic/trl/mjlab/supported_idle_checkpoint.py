"""CPU-only, fail-closed checkpoint format for supported-idle PPO.

This module deliberately knows nothing about MJLab environments or runners.
It serializes only PyTorch weights-only-safe values and makes checkpoint
publication atomic within one directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = 1
CHECKPOINT_KIND = "g1_true23_native124_supported_idle_resume"
HEADER_KEY = "g1_true23_native124_supported_idle_checkpoint"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class TensorSpec:
    """Exact CPU tensor contract for a runner-owned state value."""

    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class CheckpointPublication:
    """Immutable identity returned after exclusive checkpoint publication."""

    path: Path
    sha256: str


ACTOR_STATE_SCHEMA = {
    "obs_normalizer._mean": TensorSpec((1, 124), torch.float32),
    "obs_normalizer._var": TensorSpec((1, 124), torch.float32),
    "obs_normalizer._std": TensorSpec((1, 124), torch.float32),
    "obs_normalizer.count": TensorSpec((), torch.int64),
    "distribution.std_param": TensorSpec((23,), torch.float32),
    "mlp.0.weight": TensorSpec((512, 124), torch.float32),
    "mlp.0.bias": TensorSpec((512,), torch.float32),
    "mlp.2.weight": TensorSpec((256, 512), torch.float32),
    "mlp.2.bias": TensorSpec((256,), torch.float32),
    "mlp.4.weight": TensorSpec((128, 256), torch.float32),
    "mlp.4.bias": TensorSpec((128,), torch.float32),
    "mlp.6.weight": TensorSpec((23, 128), torch.float32),
    "mlp.6.bias": TensorSpec((23,), torch.float32),
}
CRITIC_STATE_SCHEMA = {
    "obs_normalizer._mean": TensorSpec((1, 256), torch.float32),
    "obs_normalizer._var": TensorSpec((1, 256), torch.float32),
    "obs_normalizer._std": TensorSpec((1, 256), torch.float32),
    "obs_normalizer.count": TensorSpec((), torch.int64),
    "mlp.0.weight": TensorSpec((512, 256), torch.float32),
    "mlp.0.bias": TensorSpec((512,), torch.float32),
    "mlp.2.weight": TensorSpec((256, 512), torch.float32),
    "mlp.2.bias": TensorSpec((256,), torch.float32),
    "mlp.4.weight": TensorSpec((128, 256), torch.float32),
    "mlp.4.bias": TensorSpec((128,), torch.float32),
    "mlp.6.weight": TensorSpec((1, 128), torch.float32),
    "mlp.6.bias": TensorSpec((1,), torch.float32),
}
OPTIMIZER_PARAMETER_SPECS = (
    TensorSpec((23,), torch.float32),
    TensorSpec((512, 124), torch.float32),
    TensorSpec((512,), torch.float32),
    TensorSpec((256, 512), torch.float32),
    TensorSpec((256,), torch.float32),
    TensorSpec((128, 256), torch.float32),
    TensorSpec((128,), torch.float32),
    TensorSpec((23, 128), torch.float32),
    TensorSpec((23,), torch.float32),
    TensorSpec((512, 256), torch.float32),
    TensorSpec((512,), torch.float32),
    TensorSpec((256, 512), torch.float32),
    TensorSpec((256,), torch.float32),
    TensorSpec((128, 256), torch.float32),
    TensorSpec((128,), torch.float32),
    TensorSpec((1, 128), torch.float32),
    TensorSpec((1,), torch.float32),
)


def _require_sha256(value: Any, context: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in _HEX for character in value):
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def _require_nonnegative_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} must be integer >= 0")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _copy_safe(value: Any, context: str) -> Any:
    """Deep-copy only values supported by ``torch.load(weights_only=True)``."""

    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} contains non-finite float")
        return value
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError(f"{context} tensor must use strided layout")
        copied = value.detach().cpu().contiguous().clone()
        if (copied.is_floating_point() or copied.is_complex()) and not bool(torch.isfinite(copied).all()):
            raise ValueError(f"{context} tensor contains non-finite values")
        return copied
    if value_type in {list, tuple}:
        return [_copy_safe(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str | int, Any] = {}
        for key, item in value.items():
            if type(key) not in {str, int} or (type(key) is str and not key):
                raise ValueError(f"{context} has unsafe key")
            result[key] = _copy_safe(item, f"{context}[{key!r}]")
        return result
    raise ValueError(f"{context} contains weights-only-unsafe {type(value).__qualname__}")


def _validate_hash_map(value: Any, context: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{context} must be non-empty mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise ValueError(f"{context} keys must be non-empty strings")
        result[key] = _require_sha256(item, f"{context}.{key}")
    return result


def validate_lineage(value: Any) -> dict[str, Any]:
    """Validate immutable plan/auth/job/material provenance."""

    if not isinstance(value, Mapping):
        raise ValueError("lineage must be mapping")
    expected = {
        "plan_payload_sha256",
        "plan_file_sha256",
        "authorization_file_sha256",
        "job_id",
        "source_checkpoint_sha256",
        "source_kind",
        "corpus_sha256",
        "sidecar_sha256",
        "runtime_file_sha256",
        "package_sha256",
    }
    _exact_keys(value, expected, "lineage")
    if type(value["job_id"]) is not str or not value["job_id"]:
        raise ValueError("lineage.job_id must be non-empty string")
    if value["source_kind"] not in {
        "dad_dance_seed",
        "qualified_phase_parent",
    }:
        raise ValueError("lineage.source_kind unsupported")
    return {
        "plan_payload_sha256": _require_sha256(value["plan_payload_sha256"], "lineage.plan_payload_sha256"),
        "plan_file_sha256": _require_sha256(value["plan_file_sha256"], "lineage.plan_file_sha256"),
        "authorization_file_sha256": _require_sha256(
            value["authorization_file_sha256"], "lineage.authorization_file_sha256"
        ),
        "job_id": value["job_id"],
        "source_checkpoint_sha256": _require_sha256(
            value["source_checkpoint_sha256"], "lineage.source_checkpoint_sha256"
        ),
        "source_kind": value["source_kind"],
        "corpus_sha256": _require_sha256(value["corpus_sha256"], "lineage.corpus_sha256"),
        "sidecar_sha256": _require_sha256(value["sidecar_sha256"], "lineage.sidecar_sha256"),
        "runtime_file_sha256": _validate_hash_map(value["runtime_file_sha256"], "lineage.runtime_file_sha256"),
        "package_sha256": _validate_hash_map(value["package_sha256"], "lineage.package_sha256"),
    }


def validate_optimizer_semantics(value: Any) -> dict[str, Any]:
    """Record exact distinction between initialization and later resume."""

    if not isinstance(value, Mapping):
        raise ValueError("optimizer_semantics must be mapping")
    expected = {"initialization", "source_optimizer_loaded", "optimizer_reset", "resume_loads_optimizer"}
    _exact_keys(value, expected, "optimizer_semantics")
    initialization = value["initialization"]
    expected_values = {
        "seed_actor_critic_only": (False, True, True),
        "phase_parent_actor_critic_only": (False, True, True),
    }
    if initialization not in expected_values:
        raise ValueError("optimizer_semantics.initialization unsupported")
    actual = (
        value["source_optimizer_loaded"],
        value["optimizer_reset"],
        value["resume_loads_optimizer"],
    )
    if actual != expected_values[initialization] or any(type(item) is not bool for item in actual):
        raise ValueError("optimizer_semantics contradicts initialization")
    return dict(value)


def _validate_optimizer_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("optimizer_state_dict must be mapping")
    _exact_keys(value, {"state", "param_groups"}, "optimizer_state_dict")
    state = value["state"]
    if not isinstance(state, Mapping):
        raise ValueError("optimizer_state_dict.state must be mapping")
    copied_state: dict[int, dict[str, Any]] = {}
    for parameter_id, parameter_state in state.items():
        if type(parameter_id) is not int or parameter_id < 0:
            raise ValueError("optimizer_state_dict.state keys must be non-negative integers")
        if not isinstance(parameter_state, Mapping):
            raise ValueError("optimizer_state_dict.state values must be mappings")
        copied = _copy_safe(parameter_state, f"optimizer_state_dict.state[{parameter_id}]")
        assert isinstance(copied, Mapping)
        if any(type(key) is not str or not key for key in copied):
            raise ValueError("optimizer_state_dict.state fields must be non-empty strings")
        copied_state[parameter_id] = dict(copied)
    groups = value["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("optimizer_state_dict.param_groups must contain one native PPO group")
    copied_groups: list[dict[str, Any]] = []
    expected_group_keys = {
        "amsgrad",
        "betas",
        "capturable",
        "decoupled_weight_decay",
        "differentiable",
        "eps",
        "foreach",
        "fused",
        "lr",
        "maximize",
        "params",
        "weight_decay",
    }
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"optimizer_state_dict.param_groups[{index}] invalid")
        _exact_keys(group, expected_group_keys, f"optimizer_state_dict.param_groups[{index}]")
        params = group["params"]
        if not isinstance(params, list) or params != list(range(len(OPTIMIZER_PARAMETER_SPECS))):
            raise ValueError(f"optimizer_state_dict.param_groups[{index}].params invalid")
        for flag in (
            "amsgrad",
            "capturable",
            "decoupled_weight_decay",
            "differentiable",
            "maximize",
        ):
            if type(group[flag]) is not bool or group[flag] is not False:
                raise ValueError(f"optimizer_state_dict.param_groups[{index}].{flag} must be false")
        if group["foreach"] is not None or group["fused"] is not None:
            raise ValueError(f"optimizer_state_dict.param_groups[{index}] foreach/fused must be None")
        betas = group["betas"]
        if type(betas) not in {list, tuple} or tuple(betas) != (0.9, 0.999):
            raise ValueError(f"optimizer_state_dict.param_groups[{index}].betas mismatch")
        if type(group["eps"]) is not float or group["eps"] != 1.0e-8:
            raise ValueError(f"optimizer_state_dict.param_groups[{index}].eps mismatch")
        weight_decay = group["weight_decay"]
        if type(weight_decay) not in {int, float} or not math.isfinite(float(weight_decay)) or weight_decay != 0:
            raise ValueError(f"optimizer_state_dict.param_groups[{index}].weight_decay mismatch")
        learning_rate = group["lr"]
        if type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(f"optimizer_state_dict.param_groups[{index}].lr invalid")
        copied = _copy_safe(group, f"optimizer_state_dict.param_groups[{index}]")
        assert isinstance(copied, Mapping)
        if any(type(key) is not str or not key for key in copied):
            raise ValueError("optimizer_state_dict.param_groups fields must be non-empty strings")
        copied_groups.append(dict(copied))
    return {"state": copied_state, "param_groups": copied_groups}


def _validate_completed_adam_state(optimizer: Mapping[str, Any]) -> None:
    expected_ids = set(range(len(OPTIMIZER_PARAMETER_SPECS)))
    state = optimizer["state"]
    if set(state) != expected_ids:
        raise ValueError("completed checkpoint optimizer state must cover native PPO parameters exactly")
    for parameter_id, spec in enumerate(OPTIMIZER_PARAMETER_SPECS):
        parameter_state = state[parameter_id]
        _exact_keys(parameter_state, {"step", "exp_avg", "exp_avg_sq"}, f"optimizer state {parameter_id}")
        step = parameter_state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.device.type != "cpu"
            or step.ndim != 0
            or step.dtype != torch.float32
            or not bool(torch.isfinite(step))
        ):
            raise ValueError(f"optimizer state {parameter_id}.step invalid")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = parameter_state[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != spec.dtype
                or tuple(tensor.shape) != spec.shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"optimizer state {parameter_id}.{name} invalid")


def validate_tensor_state(
    value: Any,
    schema: Mapping[str, TensorSpec],
    *,
    context: str,
) -> dict[str, torch.Tensor]:
    """Validate exact tensor keys, CPU dtype/shape, and finite float values."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be mapping")
    if not schema:
        raise ValueError(f"{context} schema must not be empty")
    _exact_keys(value, set(schema), context)
    result: dict[str, torch.Tensor] = {}
    for name, spec in schema.items():
        if type(name) is not str or not name or not isinstance(spec, TensorSpec):
            raise ValueError(f"{context} schema invalid")
        if any(type(dimension) is not int or dimension < 0 for dimension in spec.shape):
            raise ValueError(f"{context}.{name} schema shape invalid")
        tensor = value[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{context}.{name} must be tensor")
        if tensor.layout != torch.strided or tensor.device.type != "cpu":
            raise ValueError(f"{context}.{name} must be CPU strided tensor")
        if tensor.dtype != spec.dtype or tuple(tensor.shape) != spec.shape:
            raise ValueError(f"{context}.{name} dtype/shape mismatch")
        copied = _copy_safe(tensor, f"{context}.{name}")
        assert isinstance(copied, torch.Tensor)
        result[name] = copied
    return result


def capture_rng_state(*, cuda_states: Mapping[str, torch.Tensor] | None = None) -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and caller-provided CUDA RNG state."""

    py_version, py_words, py_gauss = random.getstate()
    np_name, np_words, np_pos, np_has_gauss, np_cached = np.random.get_state()
    return validate_rng_state(
        {
            "execution_mode": "cuda_full" if cuda_states else "cpu_only",
            "cuda_devices": sorted((cuda_states or {}).keys()),
            "python": {"version": py_version, "words": list(py_words), "gauss_next": py_gauss},
            "numpy": {
                "bit_generator": np_name,
                "words": torch.from_numpy(np.asarray(np_words, dtype=np.uint32).copy()),
                "position": np_pos,
                "has_gauss": np_has_gauss,
                "cached_gaussian": float(np_cached),
            },
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": dict(cuda_states or {}),
        }
    )


def validate_rng_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("rng_state must be mapping")
    _exact_keys(
        value,
        {"execution_mode", "cuda_devices", "python", "numpy", "torch_cpu", "torch_cuda"},
        "rng_state",
    )
    execution_mode = value["execution_mode"]
    if execution_mode not in {"cpu_only", "cuda_full"}:
        raise ValueError("rng_state.execution_mode unsupported")
    cuda_devices = value["cuda_devices"]
    if not isinstance(cuda_devices, list) or any(type(device) is not str for device in cuda_devices):
        raise ValueError("rng_state.cuda_devices must be string list")
    if cuda_devices != sorted(set(cuda_devices)):
        raise ValueError("rng_state.cuda_devices must be sorted unique")
    python_state = value["python"]
    if not isinstance(python_state, Mapping):
        raise ValueError("rng_state.python must be mapping")
    _exact_keys(python_state, {"version", "words", "gauss_next"}, "rng_state.python")
    version = _require_nonnegative_int(python_state["version"], "rng_state.python.version")
    if version != 3:
        raise ValueError("rng_state.python version must be 3")
    words = python_state["words"]
    if not isinstance(words, list) or len(words) != 625:
        raise ValueError("rng_state.python.words must contain 625 integers")
    if any(type(item) is not int or item < 0 or item > 0xFFFFFFFF for item in words[:-1]):
        raise ValueError("rng_state.python.words contains invalid MT19937 word")
    if type(words[-1]) is not int or not 0 <= words[-1] <= 624:
        raise ValueError("rng_state.python index invalid")
    gauss_next = python_state["gauss_next"]
    if gauss_next is not None and (type(gauss_next) is not float or not math.isfinite(gauss_next)):
        raise ValueError("rng_state.python.gauss_next invalid")
    numpy_state = value["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise ValueError("rng_state.numpy must be mapping")
    _exact_keys(
        numpy_state, {"bit_generator", "words", "position", "has_gauss", "cached_gaussian"}, "rng_state.numpy"
    )
    if numpy_state["bit_generator"] != "MT19937":
        raise ValueError("rng_state.numpy bit generator must be MT19937")
    np_words = validate_tensor_state(
        {"words": numpy_state["words"]}, {"words": TensorSpec((624,), torch.uint32)}, context="rng_state.numpy"
    )["words"]
    np_position = _require_nonnegative_int(numpy_state["position"], "rng_state.numpy.position")
    if np_position > 624 or type(numpy_state["has_gauss"]) is not int or numpy_state["has_gauss"] not in {0, 1}:
        raise ValueError("rng_state.numpy position/has_gauss invalid")
    cached = numpy_state["cached_gaussian"]
    if type(cached) is not float or not math.isfinite(cached):
        raise ValueError("rng_state.numpy.cached_gaussian invalid")
    torch_cpu = value["torch_cpu"]
    if (
        not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
        or torch_cpu.numel() == 0
    ):
        raise ValueError("rng_state.torch_cpu must be non-empty uint8 vector")
    torch_cpu = _copy_safe(torch_cpu, "rng_state.torch_cpu")
    assert isinstance(torch_cpu, torch.Tensor)
    try:
        torch.Generator(device="cpu").set_state(torch_cpu)
    except RuntimeError as error:
        raise ValueError("rng_state.torch_cpu is not valid CPU generator state") from error
    cuda = value["torch_cuda"]
    if not isinstance(cuda, Mapping):
        raise ValueError("rng_state.torch_cuda must be mapping")
    cuda_result: dict[str, torch.Tensor] = {}
    for device, state in cuda.items():
        if type(device) is not str or not device.startswith("cuda:"):
            raise ValueError("rng_state.torch_cuda device key invalid")
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() == 0
        ):
            raise ValueError("rng_state.torch_cuda state must be non-empty uint8 vector")
        copied = _copy_safe(state, f"rng_state.torch_cuda.{device}")
        assert isinstance(copied, torch.Tensor)
        cuda_result[device] = copied
    if sorted(cuda_result) != cuda_devices:
        raise ValueError("rng_state CUDA devices do not match CUDA state")
    if execution_mode == "cpu_only" and cuda_devices:
        raise ValueError("cpu_only RNG state must not include CUDA state")
    if execution_mode == "cuda_full" and not cuda_devices:
        raise ValueError("cuda_full RNG state requires CUDA device state")
    return {
        "execution_mode": execution_mode,
        "cuda_devices": list(cuda_devices),
        "python": {"version": version, "words": list(words), "gauss_next": gauss_next},
        "numpy": {
            "bit_generator": "MT19937",
            "words": np_words,
            "position": np_position,
            "has_gauss": numpy_state["has_gauss"],
            "cached_gaussian": cached,
        },
        "torch_cpu": torch_cpu,
        "torch_cuda": cuda_result,
    }


def restore_rng_state(value: Any) -> None:
    """Restore validated CPU RNG state. CUDA restoration stays runner-owned."""

    state = validate_rng_state(value)
    if state["execution_mode"] != "cpu_only":
        raise RuntimeError("CUDA RNG restore must be performed by the GPU runner")
    random.setstate((state["python"]["version"], tuple(state["python"]["words"]), state["python"]["gauss_next"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["words"].numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])


def build_checkpoint(
    *,
    actor_state_dict: Mapping[str, torch.Tensor],
    critic_state_dict: Mapping[str, torch.Tensor],
    optimizer_state_dict: Mapping[str, Any],
    completed_updates: int,
    env_common_step_counter: int,
    lineage: Mapping[str, Any],
    optimizer_semantics: Mapping[str, Any],
    rng_state: Mapping[str, Any],
    command_state: Mapping[str, torch.Tensor],
    command_schema: Mapping[str, TensorSpec],
) -> dict[str, Any]:
    completed = _require_nonnegative_int(completed_updates, "completed_updates")
    environment_counter = _require_nonnegative_int(env_common_step_counter, "env_common_step_counter")
    return validate_checkpoint(
        {
            HEADER_KEY: {"schema_version": SCHEMA_VERSION, "kind": CHECKPOINT_KIND},
            "actor_state_dict": validate_tensor_state(
                actor_state_dict, ACTOR_STATE_SCHEMA, context="actor_state_dict"
            ),
            "critic_state_dict": validate_tensor_state(
                critic_state_dict, CRITIC_STATE_SCHEMA, context="critic_state_dict"
            ),
            "optimizer_state_dict": _validate_optimizer_state(optimizer_state_dict),
            "completed_updates": completed,
            "iter": completed - 1,
            "trainer_state": {
                "completed_updates": completed,
                "current_learning_iteration": completed,
                "env_common_step_counter": environment_counter,
            },
            "lineage": validate_lineage(lineage),
            "optimizer_semantics": validate_optimizer_semantics(optimizer_semantics),
            "rng_state": validate_rng_state(rng_state),
            "command_state": validate_tensor_state(command_state, command_schema, context="command_state"),
        },
        command_schema=command_schema,
    )


def validate_checkpoint(value: Any, *, command_schema: Mapping[str, TensorSpec]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint must be mapping")
    expected = {
        HEADER_KEY,
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "completed_updates",
        "iter",
        "trainer_state",
        "lineage",
        "optimizer_semantics",
        "rng_state",
        "command_state",
    }

    _exact_keys(value, expected, "checkpoint")
    header = value[HEADER_KEY]
    if not isinstance(header, Mapping) or set(header) != {"schema_version", "kind"}:
        raise ValueError("checkpoint header mismatch")
    if type(header["schema_version"]) is not int or header["schema_version"] != SCHEMA_VERSION:
        raise ValueError("checkpoint header mismatch")
    if header["kind"] != CHECKPOINT_KIND:
        raise ValueError("checkpoint header mismatch")
    completed = _require_nonnegative_int(value["completed_updates"], "completed_updates")
    if value["iter"] != completed - 1 or type(value["iter"]) is not int:
        raise ValueError("checkpoint iter must equal completed_updates - 1")
    trainer_state = value["trainer_state"]
    if not isinstance(trainer_state, Mapping):
        raise ValueError("trainer_state must be mapping")
    _exact_keys(
        trainer_state,
        {"completed_updates", "current_learning_iteration", "env_common_step_counter"},
        "trainer_state",
    )
    if (
        _require_nonnegative_int(trainer_state["completed_updates"], "trainer_state.completed_updates")
        != completed
        or _require_nonnegative_int(
            trainer_state["current_learning_iteration"], "trainer_state.current_learning_iteration"
        )
        != completed
    ):
        raise ValueError("trainer_state update counters mismatch")
    environment_counter = _require_nonnegative_int(
        trainer_state["env_common_step_counter"], "trainer_state.env_common_step_counter"
    )
    optimizer = _validate_optimizer_state(value["optimizer_state_dict"])
    lineage = validate_lineage(value["lineage"])
    optimizer_semantics = validate_optimizer_semantics(value["optimizer_semantics"])
    expected_initialization = {
        "dad_dance_seed": "seed_actor_critic_only",
        "qualified_phase_parent": "phase_parent_actor_critic_only",
    }[lineage["source_kind"]]
    if optimizer_semantics["initialization"] != expected_initialization:
        raise ValueError("lineage.source_kind contradicts optimizer_semantics")
    if completed == 0:
        if optimizer["state"]:
            raise ValueError("fresh checkpoint optimizer state must be empty")
    else:
        _validate_completed_adam_state(optimizer)
    return {
        HEADER_KEY: dict(header),
        "actor_state_dict": validate_tensor_state(
            value["actor_state_dict"], ACTOR_STATE_SCHEMA, context="actor_state_dict"
        ),
        "critic_state_dict": validate_tensor_state(
            value["critic_state_dict"], CRITIC_STATE_SCHEMA, context="critic_state_dict"
        ),
        "optimizer_state_dict": optimizer,
        "completed_updates": completed,
        "iter": completed - 1,
        "trainer_state": {
            "completed_updates": completed,
            "current_learning_iteration": completed,
            "env_common_step_counter": environment_counter,
        },
        "lineage": lineage,
        "optimizer_semantics": optimizer_semantics,
        "rng_state": validate_rng_state(value["rng_state"]),
        "command_state": validate_tensor_state(value["command_state"], command_schema, context="command_state"),
    }


def _exact_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.layout == right.layout
            and left.device == right.device
            and tuple(left.shape) == tuple(right.shape)
            and bool(torch.equal(left, right))
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_exact_value_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_value_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def save_checkpoint_exclusive(
    path: str | Path, checkpoint: Mapping[str, Any], *, command_schema: Mapping[str, TensorSpec]
) -> CheckpointPublication:
    """Reload-verify, fsync, then atomically publish. Existing path always fails."""

    # Standard Python cannot fsync a directory on Windows. Refuse before
    # creating a temporary file: returning after an undurable rename would
    # falsely claim crash-safe publication.
    if os.name == "nt":
        raise OSError("atomic checkpoint publication requires directory fsync support")
    output = Path(path).expanduser()
    if output.is_symlink():
        raise ValueError("checkpoint output must not be symlink")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"checkpoint parent must already exist: {output.parent}")
    normalized = validate_checkpoint(checkpoint, command_schema=command_schema)
    temporary: Path | None = None
    owned_identity: tuple[int, int] | None = None

    def owns(path_value: Path) -> bool:
        if owned_identity is None:
            return False
        try:
            stat = os.lstat(path_value)
        except FileNotFoundError:
            return False
        return (stat.st_dev, stat.st_ino) == owned_identity

    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            stat = os.fstat(handle.fileno())
            owned_identity = (stat.st_dev, stat.st_ino)
            torch.save(normalized, handle)
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
            snapshot = handle.read()
            loaded = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
            verified = validate_checkpoint(loaded, command_schema=command_schema)
            if not _exact_value_equal(normalized, verified):
                raise ValueError("checkpoint reload differs from normalized checkpoint")
            published_sha256 = hashlib.sha256(snapshot).hexdigest()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
            if not owns(temporary):
                raise RuntimeError("checkpoint temporary pathname changed before publication")
            # Link publication is exclusive: unlike check-then-replace, it cannot
            # overwrite a checkpoint created by another process after our check.
            try:
                os.link(temporary, output, follow_symlinks=False)
            except FileExistsError:
                raise FileExistsError(f"refusing to overwrite checkpoint: {output}") from None
            if not owns(output):
                raise RuntimeError("published checkpoint inode differs from owned temporary")
            output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
            if output_sha256 != published_sha256 or not owns(output):
                if owns(output):
                    output.unlink()
                    directory_fd = os.open(output.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                raise RuntimeError("published checkpoint bytes changed during publication")
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if not owns(temporary):
            raise RuntimeError("checkpoint temporary pathname changed after publication")
        temporary.unlink()
        temporary = None
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and owns(temporary):
            temporary.unlink()
    return CheckpointPublication(path=output, sha256=published_sha256)


def load_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_lineage: Mapping[str, Any],
    expected_rng_execution_mode: str,
    expected_cuda_devices: tuple[str, ...],
    command_schema: Mapping[str, TensorSpec],
) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError("checkpoint must not be symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("checkpoint must be regular file")
    expected = _require_sha256(expected_sha256, "expected_sha256")
    snapshot = resolved.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != expected:
        raise ValueError("checkpoint SHA-256 mismatch")
    try:
        loaded = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("checkpoint is not weights-only-safe") from error
    validated = validate_checkpoint(loaded, command_schema=command_schema)
    if not _exact_value_equal(validated["lineage"], validate_lineage(expected_lineage)):
        raise ValueError("checkpoint lineage differs from expected lineage")
    if expected_rng_execution_mode not in {"cpu_only", "cuda_full"}:
        raise ValueError("expected_rng_execution_mode unsupported")
    if (
        any(type(device) is not str for device in expected_cuda_devices)
        or tuple(sorted(set(expected_cuda_devices))) != expected_cuda_devices
    ):
        raise ValueError("expected_cuda_devices must be sorted unique tuple")
    if (
        validated["rng_state"]["execution_mode"] != expected_rng_execution_mode
        or tuple(validated["rng_state"]["cuda_devices"]) != expected_cuda_devices
    ):
        raise ValueError("checkpoint RNG execution/device contract mismatch")
    return validated
