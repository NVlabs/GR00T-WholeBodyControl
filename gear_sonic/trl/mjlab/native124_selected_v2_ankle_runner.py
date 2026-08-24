"""Fail-closed loader and row mask for selected native124 ankle adaptation.

This module is CPU/MJLab independent.  It reconstructs the exact RSL-RL v5
``124 -> 512 -> 256 -> 128 -> 23`` actor from the selected alpha25 checkpoint,
discards the source Adam state, and permits actor updates only on the four
hardware ankle rows.  A fifth left-knee row requires an explicit evidence gate.

Snapshots are warm restarts: adapted weights survive, while Adam state is never
serialized or loaded and the local adaptation iteration always restarts at zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
from typing import Any

import torch
from torch import nn

OBSERVATION_DIM = 124
CRITIC_OBSERVATION_DIM = 256
ACTION_DIM = 23
HIDDEN_DIMS = (512, 256, 128)
NORMALIZATION_EPSILON = 1.0e-2

SELECTED_CHECKPOINT_ITERATION = 21204
SELECTED_CHECKPOINT_SHA256 = "9cb0a06db441b8ceb51404b45ba25a81bd4120114aa6b97d6f660cac3f742f81"
SELECTED_ACTOR_STATE_SHA256 = "17302f7076cb480fe4ffc253e7b8228fcbaa033ccb3bf7aac1ed34940b8648ec"
BROAD_ACTOR_STATE_SHA256 = "dd25ab6abcec895a2ee117bcce0575bd6f6e77afb419a10f8ad0bef4f5117304"
BROAD_CRITIC_STATE_SHA256 = "652052e0fe1ec8de7aad975ab2c75e180a98d7abb553d6266381a19ccd860d56"
BROAD_OPTIMIZER_STATE_SHA256 = "d2f890fe5136be0e1a940dadb30b965ac92c63e526e7890d8affbf0caa334612"

ANKLE_HARDWARE_ROWS = (4, 5, 10, 11)
LEFT_KNEE_HARDWARE_ROW = 3
LEFT_KNEE_EVIDENCE_GATE = "diagnosed-left-knee-limit-evidence-v1"

SNAPSHOT_KIND = "g1_true23_native124_selected_v2_ankle_warm_restart"
SNAPSHOT_SCHEMA_VERSION = 1

_SOURCE_ROOT_KEYS = {
    "actor_state_dict",
    "critic_state_dict",
    "infos",
    "iter",
    "optimizer_state_dict",
}
_SNAPSHOT_ROOT_KEYS = {"actor_state_dict", "critic_state_dict", "metadata"}


@dataclass(frozen=True)
class TensorSpec:
    """Exact state-dict tensor contract."""

    shape: tuple[int, ...]
    dtype: torch.dtype


ACTOR_STATE_SCHEMA = {
    "obs_normalizer._mean": TensorSpec((1, OBSERVATION_DIM), torch.float32),
    "obs_normalizer._var": TensorSpec((1, OBSERVATION_DIM), torch.float32),
    "obs_normalizer._std": TensorSpec((1, OBSERVATION_DIM), torch.float32),
    "obs_normalizer.count": TensorSpec((), torch.int64),
    "distribution.std_param": TensorSpec((ACTION_DIM,), torch.float32),
    "mlp.0.weight": TensorSpec((512, OBSERVATION_DIM), torch.float32),
    "mlp.0.bias": TensorSpec((512,), torch.float32),
    "mlp.2.weight": TensorSpec((256, 512), torch.float32),
    "mlp.2.bias": TensorSpec((256,), torch.float32),
    "mlp.4.weight": TensorSpec((128, 256), torch.float32),
    "mlp.4.bias": TensorSpec((128,), torch.float32),
    "mlp.6.weight": TensorSpec((ACTION_DIM, 128), torch.float32),
    "mlp.6.bias": TensorSpec((ACTION_DIM,), torch.float32),
}
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

_ACTOR_FULLY_FROZEN_KEYS = (
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
    "obs_normalizer.count",
    "distribution.std_param",
    "mlp.0.weight",
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
    "mlp.4.weight",
    "mlp.4.bias",
)
_CRITIC_FROZEN_KEYS = (
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
    "obs_normalizer.count",
)


@dataclass(frozen=True)
class AnkleRowConfig:
    """Actor output-row contract; left knee is opt-in and evidence-gated."""

    include_left_knee: bool = False
    left_knee_gate: str | None = None

    def __post_init__(self) -> None:
        if type(self.include_left_knee) is not bool:
            raise TypeError("include_left_knee must be bool")
        if self.include_left_knee:
            if self.left_knee_gate != LEFT_KNEE_EVIDENCE_GATE:
                raise ValueError("left knee row requires the explicit diagnosed evidence gate")
        elif self.left_knee_gate is not None:
            raise ValueError("left_knee_gate is forbidden when left knee is disabled")

    @property
    def trainable_hardware_rows(self) -> tuple[int, ...]:
        if self.include_left_knee:
            return (LEFT_KNEE_HARDWARE_ROW, *ANKLE_HARDWARE_ROWS)
        return ANKLE_HARDWARE_ROWS

    def metadata(self) -> dict[str, Any]:
        return {
            "include_left_knee": self.include_left_knee,
            "left_knee_gate": self.left_knee_gate,
            "trainable_hardware_rows": list(self.trainable_hardware_rows),
        }


@dataclass(frozen=True)
class SourceLineage:
    """Content proof for alpha25 actor versus inherited broad critic/Adam."""

    checkpoint_path: Path
    checkpoint_sha256: str
    source_iteration: int
    actor_state_sha256: str
    broad_actor_state_sha256: str
    critic_state_sha256: str
    optimizer_state_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "actor_differs_from_broad": self.actor_state_sha256 != self.broad_actor_state_sha256,
            "actor_state_sha256": self.actor_state_sha256,
            "broad_actor_state_sha256": self.broad_actor_state_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "critic_matches_broad": self.critic_state_sha256 == BROAD_CRITIC_STATE_SHA256,
            "critic_state_sha256": self.critic_state_sha256,
            "optimizer_matches_broad": self.optimizer_state_sha256 == BROAD_OPTIMIZER_STATE_SHA256,
            "optimizer_state_sha256": self.optimizer_state_sha256,
            "source_iteration": self.source_iteration,
            "source_optimizer_loaded": False,
        }


@dataclass(frozen=True)
class CheckpointPublication:
    """Identity of one exclusively published warm-restart snapshot."""

    path: Path
    sha256: str


class FrozenEmpiricalNormalizer(nn.Module):
    """RSL-RL v5 state names with update path permanently disabled."""

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.eps = NORMALIZATION_EPSILON
        self.until = 0
        self.updates_enabled = False
        self.register_buffer("_mean", torch.zeros(1, observation_dim))
        self.register_buffer("_var", torch.ones(1, observation_dim))
        self.register_buffer("_std", torch.ones(1, observation_dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))
        super().train(False)

    def train(self, mode: bool = True) -> FrozenEmpiricalNormalizer:
        """Ignore parent train-mode propagation so RSL ``update`` stays off."""

        super().train(False)
        return self

    def update(self, _observation: torch.Tensor) -> None:
        """Deliberate no-op: selected normalization statistics are immutable."""

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self._mean) / (self._std + self.eps)


class _SavedDistributionState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.std_param = nn.Parameter(torch.ones(ACTION_DIM), requires_grad=False)


class SelectedNative124Actor(nn.Module):
    """Exact deterministic selected actor topology and RSL state names."""

    def __init__(self) -> None:
        super().__init__()
        self.obs_normalizer = FrozenEmpiricalNormalizer(OBSERVATION_DIM)
        self.distribution = _SavedDistributionState()
        self.mlp = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, HIDDEN_DIMS[0]),
            nn.ELU(),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ELU(),
            nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2]),
            nn.ELU(),
            nn.Linear(HIDDEN_DIMS[2], ACTION_DIM),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(observation))


class SelectedNative124Critic(nn.Module):
    """Optional inherited broad critic with exact RSL state names."""

    def __init__(self) -> None:
        super().__init__()
        self.obs_normalizer = FrozenEmpiricalNormalizer(CRITIC_OBSERVATION_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(CRITIC_OBSERVATION_DIM, HIDDEN_DIMS[0]),
            nn.ELU(),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ELU(),
            nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2]),
            nn.ELU(),
            nn.Linear(HIDDEN_DIMS[2], 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(observation))


class FreshAdam(torch.optim.Adam):
    """Adam whose source/resume state-loading surface is deliberately closed."""

    def load_state_dict(self, _state_dict: Mapping[str, Any]) -> None:
        raise RuntimeError("optimizer state loading is forbidden; construct a fresh Adam")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def _existing_regular_file(path: str | Path, context: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{context} must not be symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{context} does not exist: {candidate}") from error
    if not resolved.is_file():
        raise ValueError(f"{context} must be regular file: {resolved}")
    return resolved


def _exact_mapping_keys(value: Any, expected: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be mapping")
    if any(type(key) is not str or not key for key in value):
        raise ValueError(f"{context} keys must be non-empty strings")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    return value


def _validated_tensor_state(
    value: Any,
    schema: Mapping[str, TensorSpec],
    *,
    context: str,
) -> dict[str, torch.Tensor]:
    state = _exact_mapping_keys(value, set(schema), context)
    result: dict[str, torch.Tensor] = {}
    for key, spec in schema.items():
        tensor = state[key]
        if type(tensor) is not torch.Tensor:
            raise ValueError(f"{context}[{key!r}] must be exact torch.Tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"{context}[{key!r}] must use strided layout")
        if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
            raise ValueError(
                f"{context}[{key!r}] contract mismatch: expected "
                f"{spec.shape}/{spec.dtype}, got {tuple(tensor.shape)}/{tensor.dtype}"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{context}[{key!r}] contains non-finite values")
        result[key] = tensor.detach().cpu().contiguous().clone()

    count = int(result["obs_normalizer.count"].item())
    variance = result["obs_normalizer._var"]
    standard_deviation = result["obs_normalizer._std"]
    if count < 0 or bool((variance < 0).any()) or bool((standard_deviation < 0).any()):
        raise ValueError(f"{context} observation normalizer state invalid")
    if not torch.allclose(standard_deviation.square(), variance, rtol=1.0e-5, atol=1.0e-8):
        raise ValueError(f"{context} observation std/variance mismatch")
    return result


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Stable content hash used by existing native124 export reports."""

    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tree_hash_update(digest: Any, value: Any, context: str) -> None:
    if value is None:
        digest.update(b"N")
    elif type(value) is bool:
        digest.update(b"B1" if value else b"B0")
    elif type(value) is int:
        digest.update(b"I" + str(value).encode("ascii") + b"\0")
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} contains non-finite float")
        digest.update(b"F" + struct.pack(">d", value))
    elif type(value) is str:
        encoded = value.encode("utf-8")
        digest.update(b"S" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError(f"{context} tensor must use strided layout")
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{context} tensor contains non-finite values")
        digest.update(b"T")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0" + tensor.numpy().tobytes(order="C"))
    elif type(value) in {list, tuple}:
        digest.update(b"L" if type(value) is list else b"Q")
        digest.update(len(value).to_bytes(8, "big"))
        for index, item in enumerate(value):
            _tree_hash_update(digest, item, f"{context}[{index}]")
    elif isinstance(value, Mapping):
        digest.update(b"M" + len(value).to_bytes(8, "big"))

        def sort_key(key: Any) -> tuple[int, str]:
            if type(key) is int:
                return (0, str(key))
            if type(key) is str:
                return (1, key)
            raise ValueError(f"{context} has unsupported mapping key")

        for key in sorted(value, key=sort_key):
            _tree_hash_update(digest, key, f"{context}.key")
            _tree_hash_update(digest, value[key], f"{context}[{key!r}]")
    else:
        raise ValueError(f"{context} contains weights-only-unsafe {type(value).__qualname__}")


def safe_tree_sha256(value: Any, *, context: str) -> str:
    """Stable hash for nested weights-only-safe optimizer data."""

    digest = hashlib.sha256()
    _tree_hash_update(digest, value, context)
    return digest.hexdigest()


def _validate_interpolation_info(value: Any) -> None:
    infos = _exact_mapping_keys(value, {"actor_interpolation", "env_state"}, "infos")
    interpolation = _exact_mapping_keys(
        infos["actor_interpolation"], {"pilot", "broad", "broad_weight"}, "actor_interpolation"
    )
    env_state = _exact_mapping_keys(infos["env_state"], {"common_step_counter"}, "env_state")
    if interpolation["broad_weight"] != 0.25:
        raise ValueError("selected actor broad interpolation weight mismatch")
    pilot = str(interpolation["pilot"]).replace("\\", "/")
    broad = str(interpolation["broad"]).replace("\\", "/")
    if not pilot.endswith("retention_polish_run_2048/model_21124.pt"):
        raise ValueError("selected actor pilot lineage mismatch")
    if not broad.endswith("feasible_v1/train_125_lr5e7/model_21204.pt") or pilot == broad:
        raise ValueError("selected actor broad lineage mismatch")
    if env_state["common_step_counter"] != 509928:
        raise ValueError("selected checkpoint environment counter mismatch")


def _load_safe_mapping(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"{context} is not compatible with weights-only loading") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} root must be mapping")
    return value


def _load_selected_source(
    checkpoint_path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], SourceLineage]:
    path = _existing_regular_file(checkpoint_path, "selected checkpoint")
    first_hash = sha256_file(path)
    if first_hash != SELECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"selected checkpoint SHA-256 mismatch: expected {SELECTED_CHECKPOINT_SHA256}, got {first_hash}"
        )
    payload = _exact_mapping_keys(
        _load_safe_mapping(path, "selected checkpoint"), _SOURCE_ROOT_KEYS, "selected checkpoint"
    )
    if sha256_file(path) != first_hash:
        raise RuntimeError("selected checkpoint changed while loading")
    if payload["iter"] != SELECTED_CHECKPOINT_ITERATION or type(payload["iter"]) is not int:
        raise ValueError("selected checkpoint iteration mismatch")
    _validate_interpolation_info(payload["infos"])

    actor_state = _validated_tensor_state(
        payload["actor_state_dict"], ACTOR_STATE_SCHEMA, context="actor_state_dict"
    )
    critic_state = _validated_tensor_state(
        payload["critic_state_dict"], CRITIC_STATE_SCHEMA, context="critic_state_dict"
    )
    actor_hash = tensor_state_sha256(actor_state)
    critic_hash = tensor_state_sha256(critic_state)
    optimizer_hash = safe_tree_sha256(payload["optimizer_state_dict"], context="optimizer_state_dict")
    if actor_hash != SELECTED_ACTOR_STATE_SHA256 or actor_hash == BROAD_ACTOR_STATE_SHA256:
        raise ValueError("selected actor content/interpolation proof mismatch")
    if critic_hash != BROAD_CRITIC_STATE_SHA256:
        raise ValueError("selected critic is not exact inherited broad critic")
    if optimizer_hash != BROAD_OPTIMIZER_STATE_SHA256:
        raise ValueError("selected optimizer is not exact inherited broad optimizer")

    lineage = SourceLineage(
        checkpoint_path=path,
        checkpoint_sha256=first_hash,
        source_iteration=SELECTED_CHECKPOINT_ITERATION,
        actor_state_sha256=actor_hash,
        broad_actor_state_sha256=BROAD_ACTOR_STATE_SHA256,
        critic_state_sha256=critic_hash,
        optimizer_state_sha256=optimizer_hash,
    )
    return actor_state, critic_state, lineage


def _tensor_bytes(tensor: torch.Tensor) -> tuple[str, tuple[int, ...], bytes]:
    cpu = tensor.detach().cpu().contiguous()
    return str(cpu.dtype), tuple(cpu.shape), cpu.numpy().tobytes(order="C")


def _same_tensor_bytes(left: torch.Tensor, right: torch.Tensor) -> bool:
    return _tensor_bytes(left) == _tensor_bytes(right)


class _FrozenStateGuard:
    def __init__(
        self,
        actor: SelectedNative124Actor,
        critic: SelectedNative124Critic | None,
        trainable_rows: tuple[int, ...],
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.trainable_rows = trainable_rows
        self.frozen_rows = tuple(index for index in range(ACTION_DIM) if index not in trainable_rows)
        actor_state = actor.state_dict()
        self.actor_full = {key: actor_state[key].detach().clone() for key in _ACTOR_FULLY_FROZEN_KEYS}
        frozen_index = torch.tensor(self.frozen_rows, dtype=torch.long, device=actor.mlp[6].weight.device)
        self.output_weight = actor.mlp[6].weight.detach().index_select(0, frozen_index).clone()
        self.output_bias = actor.mlp[6].bias.detach().index_select(0, frozen_index).clone()
        self.critic_full: dict[str, torch.Tensor] = {}
        if critic is not None:
            critic_state = critic.state_dict()
            self.critic_full = {key: critic_state[key].detach().clone() for key in _CRITIC_FROZEN_KEYS}

    def _normalizer_flags_intact(self) -> bool:
        normalizers = [self.actor.obs_normalizer]
        if self.critic is not None:
            normalizers.append(self.critic.obs_normalizer)
        return all(
            not normalizer.training and normalizer.until == 0 and normalizer.updates_enabled is False
            for normalizer in normalizers
        )

    def assert_intact(self) -> None:
        if not self._normalizer_flags_intact():
            raise RuntimeError("frozen observation-normalizer update flag drift")
        actor_state = self.actor.state_dict()
        for key, expected in self.actor_full.items():
            if not _same_tensor_bytes(actor_state[key], expected):
                raise RuntimeError(f"frozen actor tensor changed: {key}")
        frozen_index = torch.tensor(self.frozen_rows, dtype=torch.long, device=self.actor.mlp[6].weight.device)
        weight = self.actor.mlp[6].weight.detach().index_select(0, frozen_index)
        bias = self.actor.mlp[6].bias.detach().index_select(0, frozen_index)
        if not _same_tensor_bytes(weight, self.output_weight):
            raise RuntimeError("frozen actor output weight rows changed")
        if not _same_tensor_bytes(bias, self.output_bias):
            raise RuntimeError("frozen actor output bias rows changed")
        if self.critic is not None:
            critic_state = self.critic.state_dict()
            for key, expected in self.critic_full.items():
                if not _same_tensor_bytes(critic_state[key], expected):
                    raise RuntimeError(f"frozen critic tensor changed: {key}")

    def restore(self) -> None:
        with torch.no_grad():
            actor_state = self.actor.state_dict()
            for key, expected in self.actor_full.items():
                actor_state[key].copy_(expected)
            frozen_index = torch.tensor(self.frozen_rows, dtype=torch.long, device=self.actor.mlp[6].weight.device)
            self.actor.mlp[6].weight.index_copy_(
                0, frozen_index, self.output_weight.to(self.actor.mlp[6].weight.device)
            )
            self.actor.mlp[6].bias.index_copy_(0, frozen_index, self.output_bias.to(self.actor.mlp[6].bias.device))
            if self.critic is not None:
                critic_state = self.critic.state_dict()
                for key, expected in self.critic_full.items():
                    critic_state[key].copy_(expected)
        self.actor.obs_normalizer.eval()
        if self.critic is not None:
            self.critic.obs_normalizer.eval()


def _validate_restart_frozen_state(
    actor_state: Mapping[str, torch.Tensor],
    critic_state: Mapping[str, torch.Tensor] | None,
    source_actor_state: Mapping[str, torch.Tensor],
    source_critic_state: Mapping[str, torch.Tensor],
    config: AnkleRowConfig,
) -> None:
    for key in _ACTOR_FULLY_FROZEN_KEYS:
        if not _same_tensor_bytes(actor_state[key], source_actor_state[key]):
            raise ValueError(f"warm restart changed frozen actor tensor: {key}")
    frozen_rows = tuple(index for index in range(ACTION_DIM) if index not in config.trainable_hardware_rows)
    index = torch.tensor(frozen_rows, dtype=torch.long)
    for key in ("mlp.6.weight", "mlp.6.bias"):
        if not _same_tensor_bytes(
            actor_state[key].index_select(0, index), source_actor_state[key].index_select(0, index)
        ):
            raise ValueError(f"warm restart changed frozen actor rows: {key}")
    if critic_state is not None:
        for key in _CRITIC_FROZEN_KEYS:
            if not _same_tensor_bytes(critic_state[key], source_critic_state[key]):
                raise ValueError(f"warm restart changed frozen critic tensor: {key}")


class SelectedV2AnkleAdaptation:
    """Fresh-Adam adaptation bundle with automatic pre/post-step invariants."""

    def __init__(
        self,
        *,
        actor: SelectedNative124Actor,
        critic: SelectedNative124Critic | None,
        optimizer: FreshAdam,
        config: AnkleRowConfig,
        lineage: SourceLineage,
        learning_rate: float,
        prior_completed_steps: int,
        gradient_handles: tuple[Any, ...],
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.config = config
        self.lineage = lineage
        self.learning_rate = learning_rate
        self.prior_completed_steps = prior_completed_steps
        self.iteration = 0
        self._gradient_handles = gradient_handles
        self._poisoned = False
        self._guard = _FrozenStateGuard(actor, critic, config.trainable_hardware_rows)
        self._optimizer_pre_handle = optimizer.register_step_pre_hook(self._optimizer_pre_step)
        self._optimizer_post_handle = optimizer.register_step_post_hook(self._optimizer_post_step)
        self.assert_frozen_invariants()

    @property
    def completed_steps_total(self) -> int:
        return self.prior_completed_steps + self.iteration

    def _require_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError("ankle adaptation bundle is poisoned; discard and reload")

    def _assert_optimizer_contract(self) -> None:
        if type(self.optimizer) is not FreshAdam:
            raise RuntimeError("optimizer class drift")
        if len(self.optimizer.param_groups) != 1:
            raise RuntimeError("optimizer parameter-group drift")
        group = self.optimizer.param_groups[0]
        if group.get("weight_decay") != 0.0 or group.get("lr") != self.learning_rate:
            raise RuntimeError("fresh Adam learning-rate/weight-decay contract drift")
        expected_parameters = [self.actor.mlp[6].weight, self.actor.mlp[6].bias]
        if self.critic is not None:
            expected_parameters.extend(self.critic.mlp.parameters())
        actual_parameters = group.get("params")
        if (
            type(actual_parameters) is not list
            or len(actual_parameters) != len(expected_parameters)
            or any(actual is not expected for actual, expected in zip(actual_parameters, expected_parameters))
        ):
            raise RuntimeError("optimizer parameter membership/order drift")

        frozen_rows = self._guard.frozen_rows
        for parameter in (self.actor.mlp[6].weight, self.actor.mlp[6].bias):
            state = self.optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                moment = state.get(key)
                if moment is not None and bool((moment[list(frozen_rows)] != 0).any()):
                    raise RuntimeError(f"optimizer {key} contains frozen-row state")

    def _optimizer_pre_step(
        self, _optimizer: torch.optim.Optimizer, _args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> None:
        self._require_healthy()
        try:
            self._assert_optimizer_contract()
            self._guard.assert_intact()
        except Exception:
            self._guard.restore()
            self._poisoned = True
            raise

    def _optimizer_post_step(
        self, _optimizer: torch.optim.Optimizer, _args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> None:
        try:
            self._assert_optimizer_contract()
            self._guard.assert_intact()
            self.iteration += 1
        except Exception:
            self._guard.restore()
            self._poisoned = True
            raise

    def train(self) -> None:
        self._require_healthy()
        self.actor.train()
        self.actor.obs_normalizer.eval()
        if self.critic is not None:
            self.critic.train()
            self.critic.obs_normalizer.eval()
        self.assert_frozen_invariants()

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self._require_healthy()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Run Adam; registered hooks enforce byte-exact frozen state."""

        self._require_healthy()
        self.optimizer.step()

    def assert_frozen_invariants(self) -> None:
        self._require_healthy()
        self._assert_optimizer_contract()
        self._guard.assert_intact()

    def _snapshot_metadata(self) -> dict[str, Any]:
        return {
            "adaptation": {
                "completed_steps_total": self.completed_steps_total,
                "critic_loaded": self.critic is not None,
                "row_config": self.config.metadata(),
            },
            "kind": SNAPSHOT_KIND,
            "optimizer_contract": {
                "class": "torch.optim.Adam",
                "iteration_resets_on_load": True,
                "learning_rate": self.learning_rate,
                "load_state_dict": "forbidden",
                "resume_state_saved": False,
                "source_state_loaded": False,
                "weight_decay": 0.0,
            },
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "source": self.lineage.metadata(),
        }

    def build_warm_restart(self) -> dict[str, Any]:
        self.assert_frozen_invariants()
        return {
            "actor_state_dict": {
                key: value.detach().cpu().contiguous().clone() for key, value in self.actor.state_dict().items()
            },
            "critic_state_dict": None
            if self.critic is None
            else {
                key: value.detach().cpu().contiguous().clone() for key, value in self.critic.state_dict().items()
            },
            "metadata": self._snapshot_metadata(),
        }

    def save_warm_restart(self, path: str | Path) -> CheckpointPublication:
        """Publish one no-overwrite weights-only-safe snapshot."""

        output = Path(path).expanduser().resolve(strict=False)
        if output.suffix.lower() != ".pt":
            raise ValueError("warm restart path must end in .pt")
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to overwrite warm restart: {output}")
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


def _gradient_mask_hooks(actor: SelectedNative124Actor, rows: tuple[int, ...]) -> tuple[Any, ...]:
    weight_mask = torch.zeros_like(actor.mlp[6].weight)
    bias_mask = torch.zeros_like(actor.mlp[6].bias)
    weight_mask[list(rows)] = 1
    bias_mask[list(rows)] = 1
    weight_handle = actor.mlp[6].weight.register_hook(lambda gradient: gradient * weight_mask)
    bias_handle = actor.mlp[6].bias.register_hook(lambda gradient: gradient * bias_mask)
    return weight_handle, bias_handle


def _validate_learning_rate(value: float) -> float:
    if type(value) not in {float, int} or type(value) is bool:
        raise TypeError("learning_rate must be finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("learning_rate must be finite and positive")
    return result


def _expected_snapshot_metadata(
    *,
    lineage: SourceLineage,
    config: AnkleRowConfig,
    load_critic: bool,
    learning_rate: float,
    completed_steps_total: int,
) -> dict[str, Any]:
    return {
        "adaptation": {
            "completed_steps_total": completed_steps_total,
            "critic_loaded": load_critic,
            "row_config": config.metadata(),
        },
        "kind": SNAPSHOT_KIND,
        "optimizer_contract": {
            "class": "torch.optim.Adam",
            "iteration_resets_on_load": True,
            "learning_rate": learning_rate,
            "load_state_dict": "forbidden",
            "resume_state_saved": False,
            "source_state_loaded": False,
            "weight_decay": 0.0,
        },
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source": lineage.metadata(),
    }


def _load_restart_states(
    restart_path: str | Path,
    *,
    expected_restart_sha256: str,
    source_actor_state: Mapping[str, torch.Tensor],
    source_critic_state: Mapping[str, torch.Tensor],
    lineage: SourceLineage,
    config: AnkleRowConfig,
    load_critic: bool,
    learning_rate: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, int]:
    path = _existing_regular_file(restart_path, "warm restart")
    first_hash = sha256_file(path)
    if first_hash != _require_sha256(expected_restart_sha256, "expected_restart_sha256"):
        raise ValueError(f"warm restart SHA-256 mismatch: expected {expected_restart_sha256}, got {first_hash}")
    payload = _exact_mapping_keys(_load_safe_mapping(path, "warm restart"), _SNAPSHOT_ROOT_KEYS, "warm restart")
    if sha256_file(path) != first_hash:
        raise RuntimeError("warm restart changed while loading")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("warm restart metadata must be mapping")
    adaptation = metadata.get("adaptation")
    if not isinstance(adaptation, Mapping):
        raise ValueError("warm restart adaptation metadata must be mapping")
    completed = adaptation.get("completed_steps_total")
    if type(completed) is not int or completed < 0:
        raise ValueError("warm restart completed_steps_total must be integer >= 0")
    expected_metadata = _expected_snapshot_metadata(
        lineage=lineage,
        config=config,
        load_critic=load_critic,
        learning_rate=learning_rate,
        completed_steps_total=completed,
    )
    if metadata != expected_metadata:
        raise ValueError("warm restart metadata/config/optimizer contract drift")

    actor_state = _validated_tensor_state(
        payload["actor_state_dict"], ACTOR_STATE_SCHEMA, context="warm actor_state_dict"
    )
    critic_state: dict[str, torch.Tensor] | None
    if load_critic:
        critic_state = _validated_tensor_state(
            payload["critic_state_dict"],
            CRITIC_STATE_SCHEMA,
            context="warm critic_state_dict",
        )
    else:
        if payload["critic_state_dict"] is not None:
            raise ValueError("critic state present when critic loading is disabled")
        critic_state = None
    _validate_restart_frozen_state(
        actor_state,
        critic_state,
        source_actor_state,
        source_critic_state,
        config,
    )
    return actor_state, critic_state, completed


def load_selected_v2_ankle_adaptation(
    checkpoint_path: str | Path,
    *,
    learning_rate: float,
    config: AnkleRowConfig | None = None,
    load_critic: bool = False,
    restart_path: str | Path | None = None,
    expected_restart_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> SelectedV2AnkleAdaptation:
    """Load selected actor, optional critic, and one always-fresh Adam.

    The selected checkpoint's optimizer is validated only as evidence of the
    inherited broad-state mismatch.  It is never passed to ``Adam.load_state_dict``.
    Warm restarts likewise contain no optimizer state and reset ``iteration``.
    """

    if type(load_critic) is not bool:
        raise TypeError("load_critic must be bool")
    selected_config = AnkleRowConfig() if config is None else config
    if type(selected_config) is not AnkleRowConfig:
        raise TypeError("config must be exact AnkleRowConfig")
    if (restart_path is None) != (expected_restart_sha256 is None):
        raise ValueError("restart_path and expected_restart_sha256 must be supplied together")
    lr = _validate_learning_rate(learning_rate)
    source_actor_state, source_critic_state, lineage = _load_selected_source(checkpoint_path)
    actor_state = source_actor_state
    critic_state: dict[str, torch.Tensor] | None = source_critic_state if load_critic else None
    prior_completed_steps = 0
    if restart_path is not None:
        if expected_restart_sha256 is None:  # pragma: no cover - guarded above
            raise RuntimeError("internal warm-restart hash contract drift")
        actor_state, critic_state, prior_completed_steps = _load_restart_states(
            restart_path,
            expected_restart_sha256=expected_restart_sha256,
            source_actor_state=source_actor_state,
            source_critic_state=source_critic_state,
            lineage=lineage,
            config=selected_config,
            load_critic=load_critic,
            learning_rate=lr,
        )

    target_device = torch.device(device)
    actor = SelectedNative124Actor()
    actor.load_state_dict(actor_state, strict=True)
    actor.to(target_device)
    actor.requires_grad_(False)
    actor.mlp[6].weight.requires_grad_(True)
    actor.mlp[6].bias.requires_grad_(True)
    actor.train()
    actor.obs_normalizer.eval()

    critic: SelectedNative124Critic | None = None
    optimizer_parameters: list[nn.Parameter] = [actor.mlp[6].weight, actor.mlp[6].bias]
    if load_critic:
        if critic_state is None:  # pragma: no cover - guarded above
            raise RuntimeError("internal critic-state contract drift")
        critic = SelectedNative124Critic()
        critic.load_state_dict(critic_state, strict=True)
        critic.to(target_device)
        critic.requires_grad_(False)
        for parameter in critic.mlp.parameters():
            parameter.requires_grad_(True)
        critic.train()
        critic.obs_normalizer.eval()
        optimizer_parameters.extend(critic.mlp.parameters())

    gradient_handles = _gradient_mask_hooks(actor, selected_config.trainable_hardware_rows)
    optimizer = FreshAdam(optimizer_parameters, lr=lr, weight_decay=0.0)
    if optimizer.state:
        raise RuntimeError("fresh Adam unexpectedly contains state")
    return SelectedV2AnkleAdaptation(
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        config=selected_config,
        lineage=lineage,
        learning_rate=lr,
        prior_completed_steps=prior_completed_steps,
        gradient_handles=gradient_handles,
    )


__all__ = [
    "ACTION_DIM",
    "ANKLE_HARDWARE_ROWS",
    "AnkleRowConfig",
    "BROAD_ACTOR_STATE_SHA256",
    "BROAD_CRITIC_STATE_SHA256",
    "BROAD_OPTIMIZER_STATE_SHA256",
    "CheckpointPublication",
    "CRITIC_OBSERVATION_DIM",
    "FreshAdam",
    "LEFT_KNEE_EVIDENCE_GATE",
    "LEFT_KNEE_HARDWARE_ROW",
    "OBSERVATION_DIM",
    "SELECTED_ACTOR_STATE_SHA256",
    "SELECTED_CHECKPOINT_ITERATION",
    "SELECTED_CHECKPOINT_SHA256",
    "SelectedNative124Actor",
    "SelectedNative124Critic",
    "SelectedV2AnkleAdaptation",
    "load_selected_v2_ankle_adaptation",
    "safe_tree_sha256",
    "sha256_file",
    "tensor_state_sha256",
]
