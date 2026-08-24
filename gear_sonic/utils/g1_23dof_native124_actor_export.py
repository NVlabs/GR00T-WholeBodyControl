"""Strict standalone runtime/export for the native Unitree G1 124-to-23 actor.

The loader reconstructs the deterministic RSL-RL actor directly from its
``actor_state_dict``.  It deliberately does not import MJLab, create an
environment, or load optimizer/critic state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn

OBSERVATION_DIM = 124
ACTION_DIM = 23
HIDDEN_DIMS = (512, 256, 128)
NORMALIZATION_EPSILON = 1.0e-2
ONNX_OPSET_VERSION = 18
ONNX_INPUT_NAME = "obs"
ONNX_OUTPUT_NAME = "actions"
PARITY_ATOL = 1.0e-5
PARITY_RTOL = 1.0e-5
PARITY_SEED = 11500
EXPORT_SCHEMA_VERSION = 1
EXPORT_KIND = "unitree_g1_23dof_native124_actor_export"

_FLOAT32 = torch.float32
_INT64 = torch.int64
_EXPECTED_ACTOR_TENSORS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "obs_normalizer._mean": ((1, OBSERVATION_DIM), _FLOAT32),
    "obs_normalizer._var": ((1, OBSERVATION_DIM), _FLOAT32),
    "obs_normalizer._std": ((1, OBSERVATION_DIM), _FLOAT32),
    "obs_normalizer.count": ((), _INT64),
    "distribution.std_param": ((ACTION_DIM,), _FLOAT32),
    "mlp.0.weight": ((512, OBSERVATION_DIM), _FLOAT32),
    "mlp.0.bias": ((512,), _FLOAT32),
    "mlp.2.weight": ((256, 512), _FLOAT32),
    "mlp.2.bias": ((256,), _FLOAT32),
    "mlp.4.weight": ((128, 256), _FLOAT32),
    "mlp.4.bias": ((128,), _FLOAT32),
    "mlp.6.weight": ((ACTION_DIM, 128), _FLOAT32),
    "mlp.6.bias": ((ACTION_DIM,), _FLOAT32),
}
_ONNX_STATE_INITIALIZERS = (
    "obs_normalizer._mean",
    "mlp.0.weight",
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
    "mlp.4.weight",
    "mlp.4.bias",
    "mlp.6.weight",
    "mlp.6.bias",
)
_EXPECTED_ONNX_OPS = (
    "Sub",
    "Div",
    "Gemm",
    "Elu",
    "Gemm",
    "Elu",
    "Gemm",
    "Elu",
    "Gemm",
)


class _SavedEmpiricalNormalizer(nn.Module):
    """RSL-RL v5 empirical normalizer with its complete saved state."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, OBSERVATION_DIM))
        self.register_buffer("_var", torch.ones(1, OBSERVATION_DIM))
        self.register_buffer("_std", torch.ones(1, OBSERVATION_DIM))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self._mean) / (self._std + NORMALIZATION_EPSILON)


class _SavedDistributionState(nn.Module):
    """Retain the saved stochastic scale while exposing deterministic means."""

    def __init__(self) -> None:
        super().__init__()
        self.std_param = nn.Parameter(torch.ones(ACTION_DIM), requires_grad=False)


class Native124Actor(nn.Module):
    """Exact deterministic RSL-RL G1 actor: normalize, then ELU MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.obs_normalizer = _SavedEmpiricalNormalizer()
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


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _existing_regular_file(path: str | Path, description: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{description} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{description} does not exist: {candidate}") from error
    if not resolved.is_file():
        raise ValueError(f"{description} must be a regular file: {resolved}")
    return resolved


def _new_explicit_path(path: str | Path, description: str, suffix: str) -> Path:
    candidate = Path(path).expanduser().resolve(strict=False)
    if candidate.suffix.lower() != suffix:
        raise ValueError(f"{description} must end in {suffix}: {candidate}")
    if os.path.lexists(candidate):
        raise FileExistsError(f"refusing to overwrite {description}: {candidate}")
    return candidate


def _validate_sha256(value: str, description: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be 64 lowercase hexadecimal characters")
    return value


def _validated_actor_state(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError("actor_state_dict must be a mapping")
    if any(type(key) is not str or not key for key in value):
        raise ValueError("actor_state_dict keys must be non-empty strings")
    actual_keys = set(value)
    expected_keys = set(_EXPECTED_ACTOR_TENSORS)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(f"actor_state_dict key mismatch: missing={missing}, unexpected={unexpected}")

    state: dict[str, torch.Tensor] = {}
    for key, (expected_shape, expected_dtype) in _EXPECTED_ACTOR_TENSORS.items():
        tensor = value[key]
        if type(tensor) is not torch.Tensor:
            raise ValueError(f"actor_state_dict[{key!r}] must be an exact torch.Tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"actor_state_dict[{key!r}] must use strided layout")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"actor_state_dict[{key!r}] shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"actor_state_dict[{key!r}] dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"actor_state_dict[{key!r}] contains non-finite values")
        state[key] = tensor.detach().cpu().contiguous().clone()

    if int(state["obs_normalizer.count"].item()) < 0:
        raise ValueError("obs_normalizer.count must be non-negative")
    variance = state["obs_normalizer._var"]
    standard_deviation = state["obs_normalizer._std"]
    if bool((variance < 0).any()) or bool((standard_deviation <= 0).any()):
        raise ValueError("observation variance/std must be non-negative/positive")
    if not torch.allclose(standard_deviation.square(), variance, rtol=1.0e-5, atol=1.0e-8):
        raise ValueError("observation std is inconsistent with saved variance")
    return state


def _tensor_manifest(state: Mapping[str, torch.Tensor]) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    descriptors: list[dict[str, Any]] = []
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        raw = tensor.numpy().tobytes(order="C")
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(dtype.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(raw)
        descriptors.append(
            {
                "dtype": dtype,
                "key": key,
                "parameter_count": tensor.numel(),
                "shape": shape,
            }
        )
    return digest.hexdigest(), descriptors


def load_native124_actor(
    checkpoint_path: str | Path,
) -> tuple[Native124Actor, dict[str, Any]]:
    """Safely load and strictly reconstruct one native RSL-RL actor."""

    checkpoint = _existing_regular_file(checkpoint_path, "checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("checkpoint is not compatible with safe torch weights-only loading") from error
    if sha256_file(checkpoint) != checkpoint_hash:
        raise RuntimeError("checkpoint changed while it was being loaded")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    if any(type(key) is not str or not key for key in payload):
        raise ValueError("checkpoint root keys must be non-empty strings")
    if "model_state_dict" in payload:
        raise ValueError("legacy or ambiguous model_state_dict checkpoints are unsupported")
    if "actor_state_dict" not in payload:
        raise ValueError("checkpoint has no actor_state_dict")
    iteration = payload.get("iter")
    if type(iteration) is not int or iteration < 0:
        raise ValueError("checkpoint iter must be a non-negative integer")

    state = _validated_actor_state(payload["actor_state_dict"])
    actor = Native124Actor()
    actor.load_state_dict(state, strict=True)
    actor.eval()
    actor.requires_grad_(False)
    reconstructed = actor.state_dict()
    if any(not torch.equal(reconstructed[key], tensor) for key, tensor in state.items()):
        raise RuntimeError("actor reconstruction changed saved tensor values")

    state_hash, descriptors = _tensor_manifest(state)
    lineage = {
        "actor_state": {
            "parameter_count": sum(tensor.numel() for tensor in state.values()),
            "sha256": state_hash,
            "tensor_count": len(state),
            "tensors": descriptors,
        },
        "checkpoint": {
            "iteration": iteration,
            "path": str(checkpoint),
            "root_keys": sorted(payload),
            "sha256": checkpoint_hash,
        },
        "safe_load": {
            "map_location": "cpu",
            "torch_weights_only": True,
        },
    }
    return actor, lineage


def run_native124_actor(actor: Native124Actor, observation: np.ndarray) -> np.ndarray:
    """Run strict CPU float32 inference and return one ``[1,23]`` action."""

    if type(actor) is not Native124Actor:
        raise TypeError("actor must be an exact Native124Actor")
    if type(observation) is not np.ndarray:
        raise TypeError("observation must be an exact numpy.ndarray")
    if observation.dtype != np.float32:
        raise ValueError("observation dtype must be float32")
    if observation.shape != (1, OBSERVATION_DIM):
        raise ValueError(f"observation shape must be (1, {OBSERVATION_DIM})")
    if not np.isfinite(observation).all():
        raise ValueError("observation contains non-finite values")
    contiguous = np.ascontiguousarray(observation)
    with torch.inference_mode():
        action = actor(torch.from_numpy(contiguous)).detach().cpu().numpy().copy()
    if action.dtype != np.float32 or action.shape != (1, ACTION_DIM):
        raise RuntimeError("actor returned an invalid action contract")
    if not np.isfinite(action).all():
        raise RuntimeError("actor returned non-finite actions")
    return action


def deterministic_parity_probes(
    actor: Native124Actor,
    *,
    seed: int = PARITY_SEED,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Build deterministic boundary/adversarial and seeded-random inputs."""

    if type(actor) is not Native124Actor:
        raise TypeError("actor must be an exact Native124Actor")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    mean = actor.obs_normalizer._mean.detach().cpu().numpy()
    denominator = actor.obs_normalizer._std.detach().cpu().numpy() + np.float32(NORMALIZATION_EPSILON)
    normalized: list[tuple[str, np.ndarray]] = [
        ("adversarial_zero", np.zeros((1, OBSERVATION_DIM), dtype=np.float32)),
        ("adversarial_plus_one", np.ones((1, OBSERVATION_DIM), dtype=np.float32)),
        ("adversarial_minus_one", -np.ones((1, OBSERVATION_DIM), dtype=np.float32)),
        (
            "adversarial_alternating_extrema",
            np.tile(np.asarray([-32.0, 32.0], dtype=np.float32), OBSERVATION_DIM // 2)[None, :],
        ),
        (
            "adversarial_ramp",
            np.linspace(-32.0, 32.0, OBSERVATION_DIM, dtype=np.float32)[None, :],
        ),
        (
            "adversarial_reverse_ramp",
            np.linspace(32.0, -32.0, OBSERVATION_DIM, dtype=np.float32)[None, :],
        ),
    ]
    boundary_indices = (0, 22, 23, 45, 46, 51, 52, 54, 55, 77, 78, 100, 101, 123)
    for index in boundary_indices:
        for sign in (-1.0, 1.0):
            impulse = np.zeros((1, OBSERVATION_DIM), dtype=np.float32)
            impulse[0, index] = np.float32(sign * 32.0)
            normalized.append((f"adversarial_boundary_{index}_{sign:+.0f}", impulse))

    rng = np.random.default_rng(seed)
    for scale in (0.01, 0.1, 1.0, 4.0, 16.0, 32.0):
        for sample_index in range(8):
            sample = rng.normal(0.0, scale, size=(1, OBSERVATION_DIM)).astype(np.float32)
            normalized.append((f"random_scale_{scale:g}_{sample_index}", sample))

    return tuple(
        (
            name,
            np.ascontiguousarray(mean + denominator * standardized, dtype=np.float32),
        )
        for name, standardized in normalized
    )


def _onnx_shape(value_info: Any) -> list[int]:
    dimensions = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise ValueError(f"ONNX tensor {value_info.name!r} has a dynamic dimension")
        dimensions.append(dimension.dim_value)
    return dimensions


def _load_validated_onnx(
    actor: Native124Actor,
    onnx_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    try:
        import onnx
        from onnx import TensorProto, numpy_helper
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError("onnx and onnxruntime are required for actor export") from error

    path = _existing_regular_file(onnx_path, "ONNX actor")
    actual_hash = sha256_file(path)
    if expected_sha256 is not None:
        expected_hash = _validate_sha256(expected_sha256, "expected ONNX SHA256")
        if actual_hash != expected_hash:
            raise ValueError(f"ONNX SHA256 mismatch: expected {expected_hash}, got {actual_hash}")

    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    standard_opsets = [item.version for item in model.opset_import if item.domain in ("", "ai.onnx")]
    foreign_opsets = [item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")]
    if standard_opsets != [ONNX_OPSET_VERSION] or foreign_opsets:
        raise ValueError(f"ONNX opset mismatch: standard={standard_opsets}, foreign={foreign_opsets}")
    inputs = [(item.name, _onnx_shape(item), item.type.tensor_type.elem_type) for item in model.graph.input]
    outputs = [(item.name, _onnx_shape(item), item.type.tensor_type.elem_type) for item in model.graph.output]
    expected_inputs = [(ONNX_INPUT_NAME, [1, OBSERVATION_DIM], TensorProto.FLOAT)]
    expected_outputs = [(ONNX_OUTPUT_NAME, [1, ACTION_DIM], TensorProto.FLOAT)]
    if inputs != expected_inputs or outputs != expected_outputs:
        raise ValueError(f"ONNX actor contract mismatch: inputs={inputs}, outputs={outputs}")
    operation_types = tuple(node.op_type for node in model.graph.node)
    if operation_types != _EXPECTED_ONNX_OPS:
        raise ValueError(f"ONNX operation graph mismatch: {operation_types}")

    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if len(initializers) != len(_ONNX_STATE_INITIALIZERS) + 1:
        raise ValueError(f"ONNX initializer count mismatch: {len(initializers)}")
    actor_state = actor.state_dict()
    for key in _ONNX_STATE_INITIALIZERS:
        if key not in initializers:
            raise ValueError(f"ONNX actor is missing initializer {key!r}")
        expected = actor_state[key].detach().cpu().numpy()
        if not np.array_equal(initializers[key], expected):
            raise ValueError(f"ONNX initializer differs from checkpoint: {key}")
    division_nodes = [node for node in model.graph.node if node.op_type == "Div"]
    if len(division_nodes) != 1 or len(division_nodes[0].input) != 2:
        raise ValueError("ONNX actor must contain one binary Div normalization node")
    divisor_name = division_nodes[0].input[1]
    expected_divisor = (actor.obs_normalizer._std.detach().cpu() + NORMALIZATION_EPSILON).numpy()
    if divisor_name not in initializers or not np.array_equal(initializers[divisor_name], expected_divisor):
        raise ValueError("ONNX normalization divisor differs from checkpoint std + 0.01")
    expected_initializer_names = set(_ONNX_STATE_INITIALIZERS) | {divisor_name}
    if set(initializers) != expected_initializer_names:
        raise ValueError(
            "ONNX initializer names mismatch: "
            f"expected={sorted(expected_initializer_names)}, got={sorted(initializers)}"
        )

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    runtime_inputs = [(item.name, item.shape, item.type) for item in session.get_inputs()]
    runtime_outputs = [(item.name, item.shape, item.type) for item in session.get_outputs()]
    if runtime_inputs != [(ONNX_INPUT_NAME, [1, OBSERVATION_DIM], "tensor(float)")]:
        raise ValueError(f"ONNX Runtime input contract mismatch: {runtime_inputs}")
    if runtime_outputs != [(ONNX_OUTPUT_NAME, [1, ACTION_DIM], "tensor(float)")]:
        raise ValueError(f"ONNX Runtime output contract mismatch: {runtime_outputs}")
    validation = {
        "checkpoint_initializers_exact": True,
        "checker_full_check": True,
        "input": {"dtype": "float32", "name": ONNX_INPUT_NAME, "shape": [1, OBSERVATION_DIM]},
        "normalizer_divisor_exact": True,
        "onnx_opset": ONNX_OPSET_VERSION,
        "operation_types": list(operation_types),
        "output": {"dtype": "float32", "name": ONNX_OUTPUT_NAME, "shape": [1, ACTION_DIM]},
        "path": str(path),
        "sha256": actual_hash,
        "shape_inference_strict": True,
    }
    return session, validation


def _run_onnx(session: Any, observation: np.ndarray) -> np.ndarray:
    action = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: observation})[0]
    if action.dtype != np.float32 or action.shape != (1, ACTION_DIM):
        raise RuntimeError("ONNX actor returned an invalid action contract")
    if not np.isfinite(action).all():
        raise RuntimeError("ONNX actor returned non-finite actions")
    return action


def _compare_runners(
    probes: Sequence[tuple[str, np.ndarray]],
    actual_runner: Callable[[np.ndarray], np.ndarray],
    expected_runner: Callable[[np.ndarray], np.ndarray],
    *,
    actual_name: str,
    expected_name: str,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    max_absolute_error = 0.0
    max_relative_error = 0.0
    worst_absolute_case = ""
    first_failure: str | None = None
    for name, observation in probes:
        actual = actual_runner(observation)
        expected = expected_runner(observation)
        difference = np.abs(actual - expected)
        case_absolute = float(difference.max())
        case_relative = float(np.max(difference / np.maximum(np.abs(expected), np.float32(atol))))
        if case_absolute > max_absolute_error:
            max_absolute_error = case_absolute
            worst_absolute_case = name
        max_relative_error = max(max_relative_error, case_relative)
        if first_failure is None and not np.allclose(actual, expected, atol=atol, rtol=rtol):
            first_failure = name
    if first_failure is not None:
        raise ValueError(
            f"{actual_name} does not match {expected_name} at probe {first_failure!r}; "
            f"max_abs={max_absolute_error:.9g}, atol={atol:.9g}, rtol={rtol:.9g}"
        )
    return {
        "actual": actual_name,
        "atol": atol,
        "case_count": len(probes),
        "expected": expected_name,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "passed": True,
        "rtol": rtol,
        "worst_absolute_case": worst_absolute_case,
    }


def verify_actor_onnx_parity(
    actor: Native124Actor,
    onnx_path: str | Path,
    *,
    expected_sha256: str | None = None,
    probes: Sequence[tuple[str, np.ndarray]] | None = None,
    atol: float = PARITY_ATOL,
    rtol: float = PARITY_RTOL,
) -> dict[str, Any]:
    """Validate exact initializers and strict numerical actor/ONNX parity."""

    if not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0:
        raise ValueError("parity tolerances must be finite and non-negative")
    selected_probes = tuple(probes or deterministic_parity_probes(actor))
    if not selected_probes:
        raise ValueError("at least one parity probe is required")
    session, validation = _load_validated_onnx(actor, onnx_path, expected_sha256=expected_sha256)
    comparison = _compare_runners(
        selected_probes,
        lambda observation: _run_onnx(session, observation),
        lambda observation: run_native124_actor(actor, observation),
        actual_name="onnxruntime_cpu",
        expected_name="checkpoint_torch_cpu",
        atol=atol,
        rtol=rtol,
    )
    return {"comparison": comparison, "onnx": validation}


def verify_onnx_pair_parity(
    actor: Native124Actor,
    actual_onnx_path: str | Path,
    expected_onnx_path: str | Path,
    *,
    expected_onnx_sha256: str,
    probes: Sequence[tuple[str, np.ndarray]] | None = None,
    atol: float = PARITY_ATOL,
    rtol: float = PARITY_RTOL,
) -> dict[str, Any]:
    """Compare two structurally checkpoint-exact ONNX actors."""

    selected_probes = tuple(probes or deterministic_parity_probes(actor))
    actual_session, actual_validation = _load_validated_onnx(actor, actual_onnx_path)
    expected_session, expected_validation = _load_validated_onnx(
        actor,
        expected_onnx_path,
        expected_sha256=expected_onnx_sha256,
    )
    comparison = _compare_runners(
        selected_probes,
        lambda observation: _run_onnx(actual_session, observation),
        lambda observation: _run_onnx(expected_session, observation),
        actual_name="exported_onnxruntime_cpu",
        expected_name="reference_onnxruntime_cpu",
        atol=atol,
        rtol=rtol,
    )
    return {
        "actual_onnx": actual_validation,
        "comparison": comparison,
        "expected_onnx": expected_validation,
    }


def _write_new_file(path: Path, content: bytes) -> None:
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite output: {path}") from error
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if created and os.path.lexists(path):
            path.unlink()
        raise


def _export_temporary_onnx(actor: Native124Actor, path: Path) -> None:
    sample = torch.zeros(1, OBSERVATION_DIM, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            actor,
            sample,
            str(path),
            input_names=[ONNX_INPUT_NAME],
            output_names=[ONNX_OUTPUT_NAME],
            opset_version=ONNX_OPSET_VERSION,
            do_constant_folding=True,
            dynamo=False,
            export_params=True,
        )


def export_native124_actor(
    checkpoint_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    reference_onnx_path: str | Path | None = None,
    expected_reference_sha256: str | None = None,
    parity_seed: int = PARITY_SEED,
) -> dict[str, Any]:
    """Export one new static actor and lineage report; never overwrite files."""

    output = _new_explicit_path(output_path, "ONNX output", ".onnx")
    report_output = _new_explicit_path(report_path, "lineage report", ".json")
    if output == report_output:
        raise ValueError("ONNX output and lineage report paths must differ")
    has_reference = reference_onnx_path is not None
    if has_reference != (expected_reference_sha256 is not None):
        raise ValueError("reference_onnx_path and expected_reference_sha256 must be supplied together")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output) or os.path.lexists(report_output):
        raise FileExistsError("refusing to overwrite ONNX output or lineage report")

    actor, lineage = load_native124_actor(checkpoint_path)
    probes = deterministic_parity_probes(actor, seed=parity_seed)
    reference_parity = None
    if reference_onnx_path is not None and expected_reference_sha256 is not None:
        reference_parity = verify_actor_onnx_parity(
            actor,
            reference_onnx_path,
            expected_sha256=expected_reference_sha256,
            probes=probes,
        )

    with tempfile.TemporaryDirectory(prefix=".native124-export-", dir=output.parent) as directory:
        temporary_onnx = Path(directory) / "actor.onnx"
        _export_temporary_onnx(actor, temporary_onnx)
        exported_parity = verify_actor_onnx_parity(actor, temporary_onnx, probes=probes)
        reference_to_export_parity = None
        if reference_onnx_path is not None and expected_reference_sha256 is not None:
            reference_to_export_parity = verify_onnx_pair_parity(
                actor,
                temporary_onnx,
                reference_onnx_path,
                expected_onnx_sha256=expected_reference_sha256,
                probes=probes,
            )
        output_bytes = temporary_onnx.read_bytes()
        output_hash = hashlib.sha256(output_bytes).hexdigest()

    checkpoint_path_resolved = Path(lineage["checkpoint"]["path"])
    if sha256_file(checkpoint_path_resolved) != lineage["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint changed before export publication")
    if os.path.lexists(output) or os.path.lexists(report_output):
        raise FileExistsError("refusing to overwrite ONNX output or lineage report")

    exported_parity["onnx"]["path"] = str(output)
    if reference_to_export_parity is not None:
        reference_to_export_parity["actual_onnx"]["path"] = str(output)
    report: dict[str, Any] = {
        "actor_contract": {
            "activation": "ELU",
            "action_dim": ACTION_DIM,
            "hidden_dims": list(HIDDEN_DIMS),
            "input": {"dtype": "float32", "name": ONNX_INPUT_NAME, "shape": [1, 124]},
            "normalization_epsilon": NORMALIZATION_EPSILON,
            "observation_dim": OBSERVATION_DIM,
            "output": {"dtype": "float32", "name": ONNX_OUTPUT_NAME, "shape": [1, 23]},
        },
        "checkpoint_lineage": lineage,
        "export": {
            "no_overwrite": True,
            "onnx_opset": ONNX_OPSET_VERSION,
            "output_path": str(output),
            "output_sha256": output_hash,
        },
        "kind": EXPORT_KIND,
        "parity": {
            "exported_onnx_vs_checkpoint": exported_parity,
            "exported_onnx_vs_reference": reference_to_export_parity,
            "probe_suite": {
                "adversarial_case_count": sum(name.startswith("adversarial_") for name, _ in probes),
                "case_count": len(probes),
                "random_case_count": sum(name.startswith("random_") for name, _ in probes),
                "seed": parity_seed,
            },
            "reference_onnx_vs_checkpoint": reference_parity,
        },
        "runtime": {
            "device": "cpu",
            "environment_constructed": False,
            "mjlab_imported": False,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        },
        "schema_version": EXPORT_SCHEMA_VERSION,
    }
    report_bytes = _canonical_json_bytes(report)
    _write_new_file(output, output_bytes)
    try:
        _write_new_file(report_output, report_bytes)
    except Exception:
        if output.is_file() and sha256_file(output) == output_hash:
            output.unlink()
        raise
    if sha256_file(output) != output_hash:
        raise RuntimeError("published ONNX hash differs from verified temporary export")
    return report
