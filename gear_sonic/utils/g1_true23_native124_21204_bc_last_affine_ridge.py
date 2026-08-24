"""Deterministic offline BC adaptation for the true23 SONIC decoder.

The admitted input is one immutable 510-row, teacher-controlled DadDance
bootstrap artifact.  The causal encoder and decoder trunk remain frozen.  A
purged blocked diagnostic selects ridge regularization for a residual update
to ``layers.8`` only.  Passing this module's gates creates an offline
simulator candidate, never support/DAgger evidence or a deployable policy.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import copy
import ctypes
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes, sha256_file
from gear_sonic.utils.g1_true23_native124_21204_bootstrap_mjlab import (
    ARRAY_SPECS,
    load_bootstrap_training_candidate,
)

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_native124_21204_bc_last_affine_ridge_contract_v1"
MANIFEST_KIND = "g1_true23_native124_21204_bc_last_affine_ridge_manifest_v1"
FAILURE_MANIFEST_KIND = "g1_true23_native124_21204_bc_last_affine_ridge_failure_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_native124_21204_bc_last_affine_ridge_v1.json"
)
CONTRACT_SHA256 = "d41d5d05f90c8fef0d88ba89bd4795c8deacafac57893dc8118e70b66db1087f"

TOTAL_ROWS = 510
RESET_PREFIX_ROWS = 10
ACTION_DIM = 23
DECODER_DIM = 994
HIDDEN_DIM = 512
FOLD_COUNT = 5
FOLD_SIZE = 102
PURGE_ROWS = 10
FINAL_WEIGHT_NAME = "layers.8.weight"
FINAL_BIAS_NAME = "layers.8.bias"
HIDDEN_OUTPUT_NAME = "/Mul_7_output_0"
EXPECTED_NODE_OPS = tuple(op for _ in range(8) for op in ("Gemm", "Sigmoid", "Mul")) + ("Gemm",)
RUNTIME_SOURCE_RELATIVE_PATHS = (
    Path("gear_sonic/utils/g1_true23_native124_21204_bc_last_affine_ridge.py"),
    Path("gear_sonic/scripts/fit_g1_true23_native124_21204_bc_last_affine_ridge.py"),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_sha256(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest values must be finite")
        return value
    if value is None or type(value) in (str, int, bool):
        return value
    raise TypeError(f"unsupported manifest value: {type(value).__name__}")


def _regular_file(root: Path, entry: Mapping[str, Any], context: str) -> Path:
    relative = entry.get("path")
    expected = _require_sha256(entry.get("sha256"), f"{context} sha256")
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ValueError(f"{context} path must be repository-relative")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    try:
        path = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{context} is missing: {relative}") from error
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"{context} must be a regular repository file")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{context} SHA256 mismatch: expected {expected}, got {actual}")
    return path


def load_offline_bc_contract(
    repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Load exact fit contract; any byte or semantic drift fails closed."""

    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    path = root / CONTRACT_RELATIVE_PATH
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("offline BC contract must be a regular repository file")
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != CONTRACT_SHA256:
        raise ValueError(f"offline BC contract SHA256 mismatch: expected {CONTRACT_SHA256}, got {actual}")
    try:
        contract = _mapping(json.loads(payload.decode("utf-8")), "offline BC contract")
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("offline BC contract must be strict UTF-8 JSON") from error
    _validate_contract(contract)
    return contract


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("role") != "deterministic_offline_bootstrap_bc_simulator_candidate_only"
        or contract.get("seed") != 20260805
        or contract.get("randomness_used") is not False
    ):
        raise ValueError("offline BC contract identity mismatch")
    rows = _mapping(contract.get("rows"), "rows")
    expected_rows = {
        "total": TOTAL_ROWS,
        "reset_prefix": RESET_PREFIX_ROWS,
        "real_h10": TOTAL_ROWS - RESET_PREFIX_ROWS,
        "q9_first": 9,
        "q9_last_action": 518,
        "fold_count": FOLD_COUNT,
        "fold_size": FOLD_SIZE,
        "purge_rows_each_side": PURGE_ROWS,
    }
    if any(rows.get(name) != value for name, value in expected_rows.items()):
        raise ValueError("offline BC row/fold contract mismatch")
    model = _mapping(contract.get("model"), "model")
    expected_model = {
        "decoder_input_name": "obs_dict",
        "decoder_input_shape": [1, DECODER_DIM],
        "decoder_output_name": "action",
        "decoder_output_shape": [1, ACTION_DIM],
        "hidden_output_name": HIDDEN_OUTPUT_NAME,
        "hidden_width": HIDDEN_DIM,
        "final_weight_name": FINAL_WEIGHT_NAME,
        "final_weight_shape": [ACTION_DIM, HIDDEN_DIM],
        "final_bias_name": FINAL_BIAS_NAME,
        "final_bias_shape": [ACTION_DIM],
        "onnx_opset": 13,
        "source_checkpoint_present": False,
        "optimizer_state": None,
        "resume_capable": False,
    }
    if any(model.get(name) != value for name, value in expected_model.items()):
        raise ValueError("offline BC model/ABI contract mismatch")
    fit = _mapping(contract.get("fit"), "fit")
    if (
        fit.get("dtype") != "float64"
        or fit.get("cpu_threads") != 1
        or fit.get("pca_variance_fraction") != 0.999
        or fit.get("pca_min_rank") != 8
        or fit.get("pca_max_rank") != 64
        or fit.get("bias_regularized") is not True
        or fit.get("pca_centering_used_for_basis_only") is not True
        or fit.get("ridge_features_use_uncentered_hidden_projection") is not True
        or fit.get("label_array") != "teacher_label_raw_native23"
        or fit.get("input_array") != "decoder994"
        or fit.get("label_semantics") != "pre_safe_transform_plain_sonic_raw_native23"
        or fit.get("v2_transform_application_count") != 1
    ):
        raise ValueError("offline BC fit/label contract mismatch")
    lambdas = fit.get("lambda_grid")
    if (
        not isinstance(lambdas, list)
        or len(lambdas) != 15
        or any(type(value) not in (int, float) or float(value) <= 0 for value in lambdas)
        or [float(value) for value in lambdas] != sorted(float(value) for value in lambdas)
    ):
        raise ValueError("offline BC lambda grid mismatch")
    scopes = _mapping(contract.get("gate_scopes"), "gate_scopes")
    if (
        scopes.get("oof_is_not_subject_to_resubstitution_thresholds") is not True
        or scopes.get("neither_scope_is_generalization_or_closed_loop_evidence") is not True
    ):
        raise ValueError("offline BC gate-scope contract mismatch")
    boundaries = _mapping(contract.get("boundaries"), "boundaries")
    required_true = {
        "offline_behavior_cloning_only",
        "simulator_candidate_only_if_all_gates_pass",
        "teacher_controlled_data",
    }
    required_false = {
        "reset_prefix_support_admitted",
        "support_qualification_performed",
        "support_admitted",
        "on_policy_data",
        "dagger_data",
        "promotion_eligible",
        "deployment_ready",
        "hardware_authorized",
        "robot_or_network_commands_permitted",
    }
    if any(boundaries.get(name) is not True for name in required_true) or any(
        boundaries.get(name) is not False for name in required_false
    ):
        raise ValueError("offline BC permanent boundary mismatch")


def _canonical_manifest(path: Path, context: str) -> Mapping[str, Any]:
    payload = path.read_bytes()
    try:
        body = _mapping(json.loads(payload.decode("utf-8")), context)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} must be strict UTF-8 JSON") from error
    if payload != canonical_json_bytes(body):
        raise ValueError(f"{context} must use canonical JSON bytes")
    return body


def _validate_bootstrap_manifest_binding(
    body: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    inputs = _mapping(contract["inputs"], "inputs")
    expected_payload = inputs["bootstrap_manifest"]["payload_sha256"]
    if body.get("manifest_payload_sha256") != expected_payload:
        raise ValueError("bootstrap manifest payload identity drift")
    artifact = _mapping(body.get("artifact"), "bootstrap artifact")
    qualification = _mapping(body.get("qualification"), "bootstrap qualification")
    boundaries = _mapping(body.get("boundaries"), "bootstrap boundaries")
    rows = _mapping(body.get("rows"), "bootstrap rows")
    if (
        artifact.get("npz_sha256") != inputs["bootstrap_npz"]["sha256"]
        or artifact.get("classification") != "bootstrap_bc_training_candidate"
        or artifact.get("strict_loader_admissible") is not True
        or qualification.get("whole_run_quarantined") is not False
        or qualification.get("bootstrap_bc_eligible_rows") != TOTAL_ROWS
        or rows.get("total") != TOTAL_ROWS
        or rows.get("reset_prefix_bootstrap_only") != RESET_PREFIX_ROWS
        or rows.get("real_h10") != TOTAL_ROWS - RESET_PREFIX_ROWS
    ):
        raise ValueError("bootstrap manifest is not exact admitted 510-row candidate")
    required_false = {
        "student_policy_present",
        "student_action_present",
        "support_qualification_performed",
        "support_admitted",
        "on_policy_data",
        "dagger_data",
        "promotion_eligible",
        "deployment_ready",
        "hardware_authorized",
        "robot_or_network_commands_performed",
    }
    if boundaries.get("teacher_controlled") is not True or any(
        boundaries.get(name) is not False for name in required_false
    ):
        raise ValueError("bootstrap manifest boundary drift")
    materials = _mapping(body.get("materials"), "bootstrap materials")
    preflight = _mapping(materials.get("preflight"), "bootstrap materials.preflight")
    executed = _mapping(preflight.get("executed_sources"), "bootstrap executed_sources")
    if executed.get("binding_sha256") != inputs["bootstrap_executed_source_binding_sha256"]:
        raise ValueError("bootstrap executed-source binding drift")


def _tensor_shape(value_info: Any) -> list[int]:
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type != 1:
        raise ValueError("ONNX ABI tensors must be float32")
    result: list[int] = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise ValueError("ONNX ABI must use static dimensions")
        result.append(int(dimension.dim_value))
    return result


def _validate_encoder_abi(model: Any, contract: Mapping[str, Any]) -> None:
    spec = _mapping(contract["model"], "model")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("encoder must have exactly one input and output")
    if (
        model.graph.input[0].name != spec["encoder_input_name"]
        or _tensor_shape(model.graph.input[0]) != spec["encoder_input_shape"]
        or model.graph.output[0].name != spec["encoder_output_name"]
        or _tensor_shape(model.graph.output[0]) != spec["encoder_output_shape"]
    ):
        raise ValueError("encoder ABI drift")


def _initializer_map(model: Any) -> dict[str, Any]:
    result = {value.name: value for value in model.graph.initializer}
    if len(result) != len(model.graph.initializer):
        raise ValueError("ONNX initializer names must be unique")
    return result


def _validate_source_decoder_abi(model: Any, contract: Mapping[str, Any]) -> None:
    from onnx import numpy_helper

    spec = _mapping(contract["model"], "model")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("source decoder must have exactly one input and output")
    if (
        model.graph.input[0].name != spec["decoder_input_name"]
        or _tensor_shape(model.graph.input[0]) != spec["decoder_input_shape"]
        or model.graph.output[0].name != spec["decoder_output_name"]
        or _tensor_shape(model.graph.output[0]) != spec["decoder_output_shape"]
    ):
        raise ValueError("source decoder ABI drift")
    opsets = {item.domain: int(item.version) for item in model.opset_import}
    if opsets.get("") != spec["onnx_opset"]:
        raise ValueError("source decoder opset drift")
    if tuple(node.op_type for node in model.graph.node) != EXPECTED_NODE_OPS:
        raise ValueError("source decoder node topology drift")
    if HIDDEN_OUTPUT_NAME not in set(model.graph.node[-2].output):
        raise ValueError("source decoder final hidden output drift")
    final = model.graph.node[-1]
    if (
        final.op_type != "Gemm"
        or list(final.input) != [HIDDEN_OUTPUT_NAME, FINAL_WEIGHT_NAME, FINAL_BIAS_NAME]
        or list(final.output) != ["action"]
    ):
        raise ValueError("source decoder final affine topology drift")
    initializers = _initializer_map(model)
    if FINAL_WEIGHT_NAME not in initializers or FINAL_BIAS_NAME not in initializers:
        raise ValueError("source decoder final affine initializers missing")
    weight = numpy_helper.to_array(initializers[FINAL_WEIGHT_NAME])
    bias = numpy_helper.to_array(initializers[FINAL_BIAS_NAME])
    if weight.shape != (ACTION_DIM, HIDDEN_DIM) or bias.shape != (ACTION_DIM,):
        raise ValueError("source decoder final affine shape drift")
    if weight.dtype != np.float32 or bias.dtype != np.float32:
        raise ValueError("source decoder final affine dtype drift")


def _model_static_binding(model: Any) -> Mapping[str, Any]:
    initializers = _initializer_map(model)
    return {
        "ir_version": int(model.ir_version),
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "domain": model.domain,
        "model_version": int(model.model_version),
        "doc_string": model.doc_string,
        "opset_sha256": _sha256_bytes(
            b"".join(item.SerializeToString(deterministic=True) for item in model.opset_import)
        ),
        "input_sha256": _sha256_bytes(
            b"".join(item.SerializeToString(deterministic=True) for item in model.graph.input)
        ),
        "output_sha256": _sha256_bytes(
            b"".join(item.SerializeToString(deterministic=True) for item in model.graph.output)
        ),
        "node_sha256": _sha256_bytes(
            b"".join(item.SerializeToString(deterministic=True) for item in model.graph.node)
        ),
        "metadata_sha256": _sha256_bytes(
            b"".join(item.SerializeToString(deterministic=True) for item in model.metadata_props)
        ),
        "initializer_names": list(initializers),
        "initializer_sha256": {
            name: _sha256_bytes(value.SerializeToString(deterministic=True))
            for name, value in initializers.items()
        },
    }


def _canonical_initializer_state_binding(
    model: Any,
    names: set[str] | None = None,
) -> Mapping[str, Any]:
    """Hash canonical tensor values, independent of protobuf tensor encoding."""

    from onnx import numpy_helper

    initializers = _initializer_map(model)
    selected = set(initializers) if names is None else set(names)
    if not selected or not selected.issubset(initializers):
        raise ValueError("canonical initializer-state selection is invalid")
    entries = []
    for name in sorted(selected):
        array = np.ascontiguousarray(numpy_helper.to_array(initializers[name]))
        entries.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "value_sha256": _sha256_bytes(array.tobytes(order="C")),
            }
        )
    return {
        "algorithm": "canonical_sorted_onnx_initializer_value_v1",
        "tensor_count": len(entries),
        "entries": entries,
        "state_sha256": _sha256_bytes(canonical_json_bytes(entries)),
    }


def _runtime_source_binding(root: Path) -> Mapping[str, Any]:
    entries = []
    for relative in RUNTIME_SOURCE_RELATIVE_PATHS:
        candidate = root / relative
        if candidate.is_symlink():
            raise ValueError(f"offline BC runtime source must not be symlink: {relative}")
        path = candidate.resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"offline BC runtime source missing/invalid: {relative}")
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "files": entries,
        "binding_sha256": _sha256_bytes(canonical_json_bytes(entries)),
    }


def _configuration_text(function: Any) -> str:
    stream = io.StringIO()
    with redirect_stdout(stream):
        function()
    return stream.getvalue().replace("\r\n", "\n")


def _openblas_control(package: Any, prefix: str) -> tuple[Any, Any, Any, Path]:
    package_root = Path(package.__file__).resolve().parent
    candidates = sorted(
        {
            *package_root.glob(".libs/*openblas*"),
            *package_root.parent.glob(f"{package.__name__}.libs/*openblas*"),
        }
    )
    libraries = [path for path in candidates if path.is_file()]
    if len(libraries) != 1:
        raise RuntimeError(f"expected one {package.__name__} OpenBLAS library, got {libraries}")
    path = libraries[0]
    library = ctypes.CDLL(str(path))
    symbol_pairs = (
        (f"{prefix}_openblas_set_num_threads", f"{prefix}_openblas_get_num_threads"),
        ("openblas_set_num_threads64_", "openblas_get_num_threads64_"),
        ("openblas_set_num_threads", "openblas_get_num_threads"),
    )
    for set_name, get_name in symbol_pairs:
        try:
            setter = getattr(library, set_name)
            getter = getattr(library, get_name)
        except AttributeError:
            continue
        setter.argtypes = [ctypes.c_int]
        setter.restype = None
        getter.argtypes = []
        getter.restype = ctypes.c_int
        return setter, getter, (set_name, get_name), path
    raise RuntimeError(f"OpenBLAS thread-control symbols unavailable in {path}")


@contextmanager
def _single_thread_numeric_runtime() -> Any:
    """Set NumPy and SciPy OpenBLAS runtimes to one thread, then restore."""

    import scipy

    controls = (
        ("numpy", *_openblas_control(np, "")),
        ("scipy", *_openblas_control(scipy, "scipy")),
    )
    previous: list[tuple[Any, int]] = []
    evidence = []
    try:
        for name, setter, getter, symbols, path in controls:
            before = int(getter())
            previous.append((setter, before))
            setter(1)
            active = int(getter())
            if active != 1:
                raise RuntimeError(f"{name} OpenBLAS refused single-thread setting")
            evidence.append(
                {
                    "runtime": name,
                    "library_path": str(path),
                    "library_sha256": sha256_file(path),
                    "library_size_bytes": path.stat().st_size,
                    "set_symbol": symbols[0],
                    "get_symbol": symbols[1],
                    "threads_before": before,
                    "threads_during_fit": active,
                }
            )
        numpy_config = _configuration_text(np.show_config)
        scipy_config = _configuration_text(scipy.show_config)
        yield {
            "determinism_scope": "same_bound_runtime_only_cross_runtime_determinism_not_claimed",
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "numpy_show_config_sha256": _sha256_bytes(numpy_config.encode("utf-8")),
            "numpy_show_config": numpy_config,
            "scipy_show_config_sha256": _sha256_bytes(scipy_config.encode("utf-8")),
            "scipy_show_config": scipy_config,
            "openblas": evidence,
        }
    finally:
        for setter, threads in reversed(previous):
            setter(threads)


def assert_only_final_affine_changed(source: Any, adapted: Any) -> None:
    """Prove graph/ABI/trunk identity; permit only final initializer changes."""

    source_binding = _model_static_binding(source)
    adapted_binding = _model_static_binding(adapted)
    for name in (
        "ir_version",
        "producer_name",
        "producer_version",
        "domain",
        "model_version",
        "doc_string",
        "opset_sha256",
        "input_sha256",
        "output_sha256",
        "node_sha256",
        "metadata_sha256",
        "initializer_names",
    ):
        if source_binding[name] != adapted_binding[name]:
            raise ValueError(f"adapted decoder changed protected model field: {name}")
    allowed = {FINAL_WEIGHT_NAME, FINAL_BIAS_NAME}
    for name, expected in source_binding["initializer_sha256"].items():
        if name not in allowed and adapted_binding["initializer_sha256"].get(name) != expected:
            raise ValueError(f"adapted decoder trunk initializer drift: {name}")
    if all(
        adapted_binding["initializer_sha256"].get(name) == source_binding["initializer_sha256"].get(name)
        for name in allowed
    ):
        raise ValueError("adapted decoder did not change final affine")


def preflight_offline_bc(
    repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Verify all immutable inputs without fitting or publishing."""

    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    contract = load_offline_bc_contract(root)
    inputs = _mapping(contract["inputs"], "inputs")
    paths: dict[str, Path] = {}
    for name in (
        "bootstrap_npz",
        "bootstrap_manifest",
        "bootstrap_contract",
        "support_contract",
        "composite_contract",
        "causal_encoder",
        "source_decoder",
        "causal_metadata",
        "teacher_checkpoint",
        "teacher_onnx",
    ):
        paths[name] = _regular_file(root, _mapping(inputs[name], f"inputs.{name}"), name)
    bound_sources = inputs.get("bound_source_files")
    if not isinstance(bound_sources, list) or len(bound_sources) != 3:
        raise ValueError("bound source-file list drift")
    for index, entry in enumerate(bound_sources):
        _regular_file(root, _mapping(entry, f"bound source {index}"), f"bound source {index}")
    bootstrap_manifest = _canonical_manifest(paths["bootstrap_manifest"], "bootstrap manifest")
    _validate_bootstrap_manifest_binding(bootstrap_manifest, contract)

    metadata = _canonical_manifest(paths["causal_metadata"], "causal metadata")
    hashes = _mapping(metadata.get("hashes"), "causal metadata hashes")
    source_decoder = _mapping(inputs["source_decoder"], "source_decoder")
    causal_encoder = _mapping(inputs["causal_encoder"], "causal_encoder")
    expected_metadata_hashes = {
        "decoder_onnx_sha256": source_decoder["sha256"],
        "decoder_state_sha256": source_decoder["state_sha256"],
        "encoder_onnx_sha256": causal_encoder["sha256"],
        "encoder_state_sha256": causal_encoder["state_sha256"],
        "policy_state_sha256": source_decoder["policy_state_sha256"],
        "lineage_sha256": source_decoder["lineage_sha256"],
    }
    if any(hashes.get(name) != value for name, value in expected_metadata_hashes.items()):
        raise ValueError("causal metadata lineage drift")
    missing_source_checkpoint = paths["causal_metadata"].parent / "causal_model_250.pt"
    if missing_source_checkpoint.exists() or contract["model"]["source_checkpoint_present"] is not False:
        raise ValueError("causal source-checkpoint absence contract drift")

    import onnx

    encoder_model = onnx.load(paths["causal_encoder"], load_external_data=False)
    decoder_model = onnx.load(paths["source_decoder"], load_external_data=False)
    onnx.checker.check_model(encoder_model, full_check=True)
    onnx.checker.check_model(decoder_model, full_check=True)
    _validate_encoder_abi(encoder_model, contract)
    _validate_source_decoder_abi(decoder_model, contract)
    return {
        "contract_sha256": CONTRACT_SHA256,
        "bootstrap_npz_sha256": inputs["bootstrap_npz"]["sha256"],
        "bootstrap_manifest_sha256": inputs["bootstrap_manifest"]["sha256"],
        "bootstrap_manifest_payload_sha256": inputs["bootstrap_manifest"]["payload_sha256"],
        "bootstrap_contract_sha256": inputs["bootstrap_contract"]["sha256"],
        "support_contract_sha256": inputs["support_contract"]["sha256"],
        "composite_contract_sha256": inputs["composite_contract"]["sha256"],
        "causal_encoder_sha256": inputs["causal_encoder"]["sha256"],
        "source_decoder_sha256": inputs["source_decoder"]["sha256"],
        "teacher_checkpoint_sha256": inputs["teacher_checkpoint"]["sha256"],
        "teacher_actor_state_sha256": inputs["teacher_checkpoint"]["actor_state_sha256"],
        "teacher_onnx_sha256": inputs["teacher_onnx"]["sha256"],
        "bootstrap_executed_source_binding_sha256": inputs["bootstrap_executed_source_binding_sha256"],
        "source_checkpoint_present": False,
        "optimizer_state": None,
        "resume_capable": False,
        "decoder_static_binding": _model_static_binding(decoder_model),
        "source_decoder_canonical_initializer_state": _canonical_initializer_state_binding(decoder_model),
        "source_final_affine_canonical_state": _canonical_initializer_state_binding(
            decoder_model,
            {FINAL_WEIGHT_NAME, FINAL_BIAS_NAME},
        ),
        "offline_bc_runtime_sources": _runtime_source_binding(root),
    }


def _validated_fit_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(arrays) != set(ARRAY_SPECS):
        missing = sorted(set(ARRAY_SPECS) - set(arrays))
        extra = sorted(set(arrays) - set(ARRAY_SPECS))
        raise ValueError(f"bootstrap array schema mismatch; missing={missing}, extra={extra}")
    result: dict[str, np.ndarray] = {}
    for name, (dtype, shape) in ARRAY_SPECS.items():
        value = arrays[name]
        if type(value) is not np.ndarray or value.dtype != dtype or value.shape != shape:
            raise ValueError(f"{name} must be {dtype} with shape {shape}")
        if np.issubdtype(value.dtype, np.floating) and not bool(np.isfinite(value).all()):
            raise ValueError(f"{name} contains nonfinite values")
        result[name] = value
    row = np.arange(TOTAL_ROWS, dtype=np.int64)
    if (
        not np.array_equal(result["row_index"], row)
        or not np.array_equal(result["q9_reference_index"], row + 9)
        or not np.array_equal(result["reset_prefix"], row < RESET_PREFIX_ROWS)
        or not np.array_equal(result["steady_history"], row >= RESET_PREFIX_ROWS)
    ):
        raise ValueError("bootstrap row/reset indexing drift")
    expected_decoder = np.concatenate((result["token64"], result["proprio930"]), axis=1)
    if not np.array_equal(result["decoder994"], expected_decoder):
        raise ValueError("decoder994 is not exact token64+proprio930 concat")
    return result


def admit_offline_bc_inputs(
    repository_root: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any], Mapping[str, Any]]:
    """Strictly admit published bootstrap pair after full existing replay gates."""

    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    contract = load_offline_bc_contract(root)
    preflight = preflight_offline_bc(root)
    inputs = contract["inputs"]
    arrays, manifest = load_bootstrap_training_candidate(
        root / inputs["bootstrap_npz"]["path"],
        root / inputs["bootstrap_manifest"]["path"],
        repository_root=root,
    )
    return _validated_fit_arrays(arrays), manifest, preflight


@dataclass(frozen=True)
class Projection:
    mean: np.ndarray
    basis: np.ndarray
    centered_score_rms: float
    rank: int
    explained_fraction: float
    singular_values: np.ndarray

    def transform(self, hidden: np.ndarray) -> np.ndarray:
        values = np.asarray(hidden, dtype=np.float64)
        return values @ self.basis.T


def canonical_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-driver SVD with canonical vector signs."""

    from scipy.linalg import svd

    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or not bool(np.isfinite(value).all()):
        raise ValueError("SVD input must be finite rank-2 float64")
    u, singular, vt = svd(
        value,
        full_matrices=False,
        check_finite=True,
        overwrite_a=False,
        lapack_driver="gesvd",
    )
    for index in range(vt.shape[0]):
        pivot = int(np.argmax(np.abs(vt[index])))
        if vt[index, pivot] < 0.0:
            vt[index] *= -1.0
            u[:, index] *= -1.0
    return u, singular, vt


def fit_projection(
    hidden: np.ndarray,
    *,
    variance_fraction: float = 0.999,
    minimum_rank: int = 8,
    maximum_rank: int = 64,
) -> Projection:
    values = np.asarray(hidden, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= maximum_rank:
        raise ValueError("hidden matrix has insufficient rows")
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    _, singular, vt = canonical_svd(centered)
    energy = singular * singular
    total = float(np.sum(energy, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("hidden trajectory has zero/nonfinite variance")
    cumulative = np.cumsum(energy, dtype=np.float64) / total
    rank = int(np.searchsorted(cumulative, variance_fraction, side="left") + 1)
    if rank < minimum_rank or rank > maximum_rank:
        raise ValueError(f"PCA rank gate failed: required {minimum_rank}..{maximum_rank}, got {rank}")
    centered_score_rms = math.sqrt(total / (values.shape[0] * rank))
    if not math.isfinite(centered_score_rms) or centered_score_rms <= 0.0:
        raise ValueError("centered PCA energy diagnostic is invalid")
    return Projection(
        mean=mean,
        basis=vt[:rank].copy(),
        centered_score_rms=centered_score_rms,
        rank=rank,
        explained_fraction=float(cumulative[rank - 1]),
        singular_values=singular[:rank].copy(),
    )


def contiguous_purged_folds(
    *,
    total_rows: int = TOTAL_ROWS,
    fold_count: int = FOLD_COUNT,
    purge_rows: int = PURGE_ROWS,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if total_rows <= 0 or fold_count <= 1 or total_rows % fold_count:
        raise ValueError("rows must divide evenly into at least two folds")
    if purge_rows < 0:
        raise ValueError("purge_rows must be nonnegative")
    block = total_rows // fold_count
    rows = np.arange(total_rows, dtype=np.int64)
    result = []
    for fold in range(fold_count):
        start = fold * block
        stop = start + block
        validation = rows[start:stop]
        forbidden_start = max(0, start - purge_rows)
        forbidden_stop = min(total_rows, stop + purge_rows)
        training = rows[(rows < forbidden_start) | (rows >= forbidden_stop)]
        if not training.size or np.intersect1d(training, validation).size:
            raise ValueError("invalid purged fold")
        if np.any(np.abs(training[:, None] - validation[None, :]) <= purge_rows):
            raise ValueError("purged fold leaks adjacent history rows")
        result.append((training, validation))
    return tuple(result)


def _ridge_residual(
    scores: np.ndarray,
    residual: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    z = np.asarray(scores, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    if z.ndim != 2 or r.ndim != 2 or z.shape[0] != r.shape[0]:
        raise ValueError("ridge score/residual shape mismatch")
    if ridge_lambda <= 0.0 or not math.isfinite(ridge_lambda):
        raise ValueError("ridge lambda must be positive and finite")
    design = np.concatenate((z, np.ones((z.shape[0], 1), dtype=np.float64)), axis=1)
    gram = (design.T @ design) / design.shape[0]
    system = gram + ridge_lambda * np.eye(design.shape[1], dtype=np.float64)
    condition = float(np.linalg.cond(system))
    rhs = (design.T @ r) / design.shape[0]
    solution = np.linalg.solve(system, rhs)
    coefficient = solution[:-1].T
    intercept = solution[-1]
    return coefficient, intercept, condition


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    row_rmse = np.sqrt(np.mean(error * error, axis=1, dtype=np.float64))
    return {
        "rmse": float(np.sqrt(np.mean(error * error, dtype=np.float64))),
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "max_abs": float(np.max(np.abs(error))),
        "p95_row_rmse": float(np.quantile(row_rmse, 0.95)),
        "p99_row_rmse": float(np.quantile(row_rmse, 0.99)),
        "max_row_rmse": float(np.max(row_rmse)),
        "per_dof_rmse": np.sqrt(np.mean(error * error, axis=0, dtype=np.float64)).tolist(),
    }


def _resubstitution_output_gates(
    prediction: np.ndarray,
    target: np.ndarray,
    frozen_prediction: np.ndarray,
    gates: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[str]]:
    """Apply only gates defined for final all-row refit/export outputs."""

    output = np.asarray(prediction, dtype=np.float64)
    label = np.asarray(target, dtype=np.float64)
    frozen = np.asarray(frozen_prediction, dtype=np.float64)
    all_metrics = _metrics(output, label)
    prefix_metrics = _metrics(output[:RESET_PREFIX_ROWS], label[:RESET_PREFIX_ROWS])
    h10_metrics = _metrics(output[RESET_PREFIX_ROWS:], label[RESET_PREFIX_ROWS:])
    frozen_metrics = _metrics(frozen, label)
    issues: list[str] = []
    if all_metrics["rmse"] > float(gates["maximum_all_row_rmse"]):
        issues.append("all_row_rmse_gate_failed")
    if prefix_metrics["rmse"] > float(gates["maximum_reset_prefix_rmse"]):
        issues.append("reset_prefix_rmse_gate_failed")
    if h10_metrics["rmse"] > float(gates["maximum_real_h10_rmse"]):
        issues.append("real_h10_rmse_gate_failed")
    if all_metrics["p95_row_rmse"] > float(gates["maximum_p95_row_rmse"]):
        issues.append("p95_row_rmse_gate_failed")
    if all_metrics["max_abs"] > float(gates["maximum_absolute_error"]):
        issues.append("maximum_absolute_error_gate_failed")
    tolerance = float(gates["per_dof_training_rmse_regression_tolerance"])
    regressed = np.flatnonzero(
        np.asarray(all_metrics["per_dof_rmse"]) > np.asarray(frozen_metrics["per_dof_rmse"]) + tolerance
    ).astype(int)
    if regressed.size:
        issues.append("per_dof_training_rmse_regression")
    maximum_prediction = float(np.max(np.abs(output)))
    if maximum_prediction >= float(gates["plain_raw_absolute_strict_max"]):
        issues.append("plain_raw_absolute_gate_failed")
    report = {
        "scope": "resubstitution_only_not_heldout_or_generalization",
        "all": all_metrics,
        "reset_prefix_10": prefix_metrics,
        "real_h10_500": h10_metrics,
        "frozen_decoder_all": frozen_metrics,
        "maximum_abs_prediction": maximum_prediction,
        "per_dof_regression_zero_based_indices": regressed.tolist(),
        "gate_issues": issues,
        "passed": not issues,
    }
    return report, issues


@dataclass(frozen=True)
class RidgeFit:
    delta_weight: np.ndarray
    delta_bias: np.ndarray
    prediction: np.ndarray
    report: Mapping[str, Any]


def fit_last_affine_residual(
    hidden: np.ndarray,
    frozen_prediction: np.ndarray,
    target: np.ndarray,
    contract: Mapping[str, Any],
) -> RidgeFit:
    """Fit exact PCA-ridge residual and evaluate every offline gate."""

    h = np.asarray(hidden, dtype=np.float64)
    y0 = np.asarray(frozen_prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if h.shape != (TOTAL_ROWS, HIDDEN_DIM):
        raise ValueError(f"hidden must be [{TOTAL_ROWS},{HIDDEN_DIM}]")
    if y0.shape != (TOTAL_ROWS, ACTION_DIM) or y.shape != (TOTAL_ROWS, ACTION_DIM):
        raise ValueError(f"predictions/targets must be [{TOTAL_ROWS},{ACTION_DIM}]")
    if not all(bool(np.isfinite(value).all()) for value in (h, y0, y)):
        raise ValueError("fit inputs contain nonfinite values")
    fit = _mapping(contract["fit"], "fit")
    gates = _mapping(contract["gates"], "gates")
    lambdas = [float(value) for value in fit["lambda_grid"]]
    residual = y - y0
    folds = contiguous_purged_folds()
    predictions = {value: np.empty_like(y) for value in lambdas}
    fold_materials: list[dict[str, Any]] = []
    frozen_sse = 0.0
    mean_sse = 0.0
    for fold_index, (training, validation) in enumerate(folds):
        projection = fit_projection(
            h[training],
            variance_fraction=float(fit["pca_variance_fraction"]),
            minimum_rank=int(fit["pca_min_rank"]),
            maximum_rank=int(fit["pca_max_rank"]),
        )
        train_scores = projection.transform(h[training])
        validation_scores = projection.transform(h[validation])
        conditions: dict[str, float] = {}
        for ridge_lambda in lambdas:
            coefficient, intercept, condition = _ridge_residual(
                train_scores,
                residual[training],
                ridge_lambda,
            )
            predictions[ridge_lambda][validation] = y0[validation] + (
                validation_scores @ coefficient.T + intercept
            )
            conditions[format(ridge_lambda, ".17g")] = condition
        frozen_error = y0[validation] - y[validation]
        mean_error = y[training].mean(axis=0, dtype=np.float64) - y[validation]
        frozen_sse += float(np.sum(frozen_error * frozen_error, dtype=np.float64))
        mean_sse += float(np.sum(mean_error * mean_error, dtype=np.float64))
        fold_materials.append(
            {
                "fold": fold_index,
                "validation_first": int(validation[0]),
                "validation_last": int(validation[-1]),
                "training_row_count": int(training.size),
                "purge_rows_each_side": PURGE_ROWS,
                "pca_rank": projection.rank,
                "pca_explained_fraction": projection.explained_fraction,
                "ridge_condition_by_lambda": conditions,
            }
        )
    pooled_sse = {value: float(np.sum((predictions[value] - y) ** 2, dtype=np.float64)) for value in lambdas}
    minimum_sse = min(pooled_sse.values())
    eligible = [value for value in lambdas if pooled_sse[value] <= 1.01 * minimum_sse]
    selected_lambda = max(eligible)
    cv_prediction = predictions[selected_lambda]
    denominator = TOTAL_ROWS * ACTION_DIM
    cv_rmse = math.sqrt(pooled_sse[selected_lambda] / denominator)
    frozen_cv_rmse = math.sqrt(frozen_sse / denominator)
    mean_cv_rmse = math.sqrt(mean_sse / denominator)
    selected_key = format(selected_lambda, ".17g")
    every_fold_beats = True
    selected_folds: list[dict[str, Any]] = []
    for material, (_, validation) in zip(fold_materials, folds, strict=True):
        adapted_metrics = _metrics(cv_prediction[validation], y[validation])
        frozen_metrics = _metrics(y0[validation], y[validation])
        beats = adapted_metrics["rmse"] < frozen_metrics["rmse"]
        every_fold_beats = every_fold_beats and beats
        selected_folds.append(
            {
                **material,
                "selected_ridge_condition": material["ridge_condition_by_lambda"][selected_key],
                "adapted_metrics": adapted_metrics,
                "frozen_metrics": frozen_metrics,
                "beats_frozen_decoder": beats,
            }
        )

    projection = fit_projection(
        h,
        variance_fraction=float(fit["pca_variance_fraction"]),
        minimum_rank=int(fit["pca_min_rank"]),
        maximum_rank=int(fit["pca_max_rank"]),
    )
    scores = projection.transform(h)
    coefficient, intercept, final_condition = _ridge_residual(
        scores,
        residual,
        selected_lambda,
    )
    delta_weight = coefficient @ projection.basis
    delta_bias = intercept
    prediction = y0 + h @ delta_weight.T + delta_bias

    final_gate_report, final_gate_issues = _resubstitution_output_gates(
        prediction,
        y,
        y0,
        gates,
    )
    oof_threshold_comparison, _ = _resubstitution_output_gates(
        cv_prediction,
        y,
        y0,
        gates,
    )
    issues: list[str] = []
    if projection.rank < int(gates["pca_rank_min"]) or projection.rank > int(gates["pca_rank_max"]):
        issues.append("pca_rank_gate_failed")
    if final_condition > float(gates["maximum_ridge_condition_number"]):
        issues.append("ridge_condition_gate_failed")
    ratio = float(gates["blocked_pooled_rmse_ratio_to_both_baselines"])
    if cv_rmse > ratio * frozen_cv_rmse or cv_rmse > ratio * mean_cv_rmse:
        issues.append("blocked_pooled_rmse_gate_failed")
    if bool(gates["every_block_beats_frozen_decoder"]) and not every_fold_beats:
        issues.append("blocked_fold_regression")
    issues.extend(final_gate_issues)

    report = {
        "solver": {
            "dtype": "float64",
            "randomness_used": False,
            "seed_recorded_not_consumed": 20260805,
            "svd_driver": "scipy_lapack_gesvd",
            "svd_sign_rule": "largest_absolute_loading_positive",
            "pca_variance_fraction": float(fit["pca_variance_fraction"]),
            "pca_rank": projection.rank,
            "pca_explained_fraction": projection.explained_fraction,
            "pca_centered_score_rms_diagnostic_only": projection.centered_score_rms,
            "ridge_design": "uncentered_hidden_projection_plus_ones",
            "bias_regularized": True,
            "selected_lambda": selected_lambda,
            "lambda_selection": fit["lambda_selection"],
            "final_ridge_condition": final_condition,
            "delta_weight_frobenius": float(np.linalg.norm(delta_weight)),
            "delta_weight_spectral": float(np.linalg.norm(delta_weight, ord=2)),
            "delta_bias_l2": float(np.linalg.norm(delta_bias)),
        },
        "blocked_diagnostic": {
            "claim": "purged_contiguous_regularization_diagnostic_not_heldout_generalization",
            "resubstitution_thresholds_do_not_define_oof_gate": True,
            "fold_count": FOLD_COUNT,
            "fold_size": FOLD_SIZE,
            "purge_rows_each_side": PURGE_ROWS,
            "selected_pooled_rmse": cv_rmse,
            "frozen_decoder_pooled_rmse": frozen_cv_rmse,
            "train_fold_teacher_mean_pooled_rmse": mean_cv_rmse,
            "pooled_sse_by_lambda": [{"lambda": value, "sse": pooled_sse[value]} for value in lambdas],
            "folds": selected_folds,
            "selected_oof_metrics": _metrics(cv_prediction, y),
            "selected_oof_per_dof_regression_zero_based_indices": oof_threshold_comparison[
                "per_dof_regression_zero_based_indices"
            ],
            "oof_meets_resubstitution_thresholds": oof_threshold_comparison["passed"],
            "oof_resubstitution_threshold_comparison_issues": oof_threshold_comparison["gate_issues"],
        },
        "final_resubstitution": final_gate_report,
        "gate_issues": sorted(set(issues)),
        "offline_fit_gates_passed": not issues,
    }
    return RidgeFit(
        delta_weight=delta_weight,
        delta_bias=delta_bias,
        prediction=prediction,
        report=_json_safe(report),
    )


def _cpu_ort_session(model_bytes: bytes) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model_bytes,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def extract_frozen_decoder_hidden(
    source_model: Any,
    decoder994: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Run exact static source decoder, temporarily exposing final hidden state."""

    from onnx import TensorProto, helper
    import onnxruntime as ort

    probe = copy.deepcopy(source_model)
    if HIDDEN_OUTPUT_NAME in {value.name for value in probe.graph.output}:
        raise ValueError("source decoder unexpectedly exports hidden output")
    probe.graph.output.extend(
        [helper.make_tensor_value_info(HIDDEN_OUTPUT_NAME, TensorProto.FLOAT, [1, HIDDEN_DIM])]
    )
    session = _cpu_ort_session(probe.SerializeToString(deterministic=True))
    source_prediction = []
    hidden = []
    for row in np.asarray(decoder994, dtype=np.float32):
        action, feature = session.run(
            ["action", HIDDEN_OUTPUT_NAME],
            {"obs_dict": row.reshape(1, DECODER_DIM)},
        )
        source_prediction.append(action[0])
        hidden.append(feature[0])
    prediction = np.asarray(source_prediction, dtype=np.float32)
    features = np.asarray(hidden, dtype=np.float32)
    if prediction.shape != (TOTAL_ROWS, ACTION_DIM) or features.shape != (
        TOTAL_ROWS,
        HIDDEN_DIM,
    ):
        raise RuntimeError("source decoder probe shape drift")
    if not bool(np.isfinite(prediction).all()) or not bool(np.isfinite(features).all()):
        raise RuntimeError("source decoder probe returned nonfinite output")
    return (
        features.astype(np.float64),
        prediction.astype(np.float64),
        {
            "provider": session.get_providers()[0],
            "onnxruntime_version": ort.__version__,
            "ort_intra_op_threads": 1,
            "ort_inter_op_threads": 1,
            "ort_execution_mode": "ORT_SEQUENTIAL",
            "ort_graph_optimization_level": "ORT_DISABLE_ALL",
            "input_dtype": "float32",
            "fit_dtype": "float64",
            "row_count": TOTAL_ROWS,
        },
    )


def export_final_affine_model(
    source_model: Any,
    weight: np.ndarray,
    bias: np.ndarray,
) -> tuple[Any, bytes]:
    """Replace only final affine tensors and produce deterministic ONNX bytes."""

    import onnx
    from onnx import numpy_helper

    new_weight = np.asarray(weight, dtype=np.float32)
    new_bias = np.asarray(bias, dtype=np.float32)
    if new_weight.shape != (ACTION_DIM, HIDDEN_DIM) or new_bias.shape != (ACTION_DIM,):
        raise ValueError("adapted final affine shape mismatch")
    if not bool(np.isfinite(new_weight).all()) or not bool(np.isfinite(new_bias).all()):
        raise ValueError("adapted final affine contains nonfinite values")
    adapted = copy.deepcopy(source_model)
    replacements = {
        FINAL_WEIGHT_NAME: numpy_helper.from_array(new_weight, name=FINAL_WEIGHT_NAME),
        FINAL_BIAS_NAME: numpy_helper.from_array(new_bias, name=FINAL_BIAS_NAME),
    }
    seen: set[str] = set()
    for index, initializer in enumerate(adapted.graph.initializer):
        if initializer.name in replacements:
            adapted.graph.initializer[index].CopyFrom(replacements[initializer.name])
            seen.add(initializer.name)
    if seen != set(replacements):
        raise ValueError("source decoder final affine initializer missing")
    onnx.checker.check_model(adapted, full_check=True)
    assert_only_final_affine_changed(source_model, adapted)
    first = adapted.SerializeToString(deterministic=True)
    second = adapted.SerializeToString(deterministic=True)
    if first != second:
        raise RuntimeError("adapted ONNX serialization is not byte deterministic")
    return adapted, first


@dataclass(frozen=True)
class OfflineBCOutcome:
    passed: bool
    report: Mapping[str, Any]
    decoder_bytes: bytes | None


def run_offline_bc_fit(
    repository_root: str | Path | None = None,
) -> OfflineBCOutcome:
    """Run complete CPU-only fit.  Failed gates return no decoder bytes."""

    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    contract = load_offline_bc_contract(root)
    preflight_before = preflight_offline_bc(root)
    arrays, bootstrap_manifest, preflight = admit_offline_bc_inputs(root)
    if canonical_json_bytes(preflight) != canonical_json_bytes(preflight_before):
        raise RuntimeError("immutable input/runtime-source drift during admission")
    source_path = root / contract["inputs"]["source_decoder"]["path"]

    import onnx
    from onnx import numpy_helper

    source_model = onnx.load(source_path, load_external_data=False)
    _validate_source_decoder_abi(source_model, contract)
    hidden, frozen_prediction, runtime = extract_frozen_decoder_hidden(
        source_model,
        arrays["decoder994"],
    )
    with _single_thread_numeric_runtime() as numeric_runtime:
        fit = fit_last_affine_residual(
            hidden,
            frozen_prediction,
            arrays["teacher_label_raw_native23"],
            contract,
        )
        repeat_fit = fit_last_affine_residual(
            hidden,
            frozen_prediction,
            arrays["teacher_label_raw_native23"],
            contract,
        )
    if (
        fit.delta_weight.tobytes() != repeat_fit.delta_weight.tobytes()
        or fit.delta_bias.tobytes() != repeat_fit.delta_bias.tobytes()
        or fit.prediction.tobytes() != repeat_fit.prediction.tobytes()
        or canonical_json_bytes(fit.report) != canonical_json_bytes(repeat_fit.report)
    ):
        raise RuntimeError("repeated same-runtime fit is not byte deterministic")
    initializers = _initializer_map(source_model)
    source_weight = numpy_helper.to_array(initializers[FINAL_WEIGHT_NAME]).astype(
        np.float64,
        copy=False,
    )
    source_bias = numpy_helper.to_array(initializers[FINAL_BIAS_NAME]).astype(
        np.float64,
        copy=False,
    )
    adapted_weight = (source_weight + fit.delta_weight).astype(np.float32)
    adapted_bias = (source_bias + fit.delta_bias).astype(np.float32)
    repeated_weight = (source_weight + repeat_fit.delta_weight).astype(np.float32)
    repeated_bias = (source_bias + repeat_fit.delta_bias).astype(np.float32)
    if adapted_weight.tobytes() != repeated_weight.tobytes() or adapted_bias.tobytes() != repeated_bias.tobytes():
        raise RuntimeError("repeated same-runtime adapted head is not byte deterministic")
    source_weight_norm = float(np.linalg.norm(source_weight))
    source_bias_norm = float(np.linalg.norm(source_bias))
    adaptation_magnitude = {
        "scope": "evidence_only_not_a_post_hoc_trust_gate",
        "delta_weight_frobenius": float(np.linalg.norm(fit.delta_weight)),
        "source_weight_frobenius": source_weight_norm,
        "delta_to_source_weight_frobenius_ratio": (float(np.linalg.norm(fit.delta_weight)) / source_weight_norm),
        "delta_bias_l2": float(np.linalg.norm(fit.delta_bias)),
        "source_bias_l2": source_bias_norm,
        "delta_to_source_bias_l2_ratio": float(np.linalg.norm(fit.delta_bias)) / source_bias_norm,
        "warning": ("large last-affine displacement observed; offline fit cannot establish closed-loop trust"),
        "requires_closed_loop_simulator_evidence": True,
    }
    issues = list(fit.report["gate_issues"])
    decoder_bytes: bytes | None = None
    export_report: dict[str, Any] = {
        "attempted": False,
        "passed": False,
        "only_final_affine_changed": False,
        "encoder_unchanged": True,
        "decoder_trunk_unchanged": True,
        "changed_initializer_names": [FINAL_WEIGHT_NAME, FINAL_BIAS_NAME],
        "abi": {
            "input_name": "obs_dict",
            "input_shape": [1, DECODER_DIM],
            "input_dtype": "float32",
            "output_name": "action",
            "output_shape": [1, ACTION_DIM],
            "output_dtype": "float32",
            "dynamic_axes": False,
            "opset": 13,
        },
        "action_semantics": {
            "output": "pre_safe_transform_plain_sonic_raw_native23",
            "action_order": "native_physx_il23_bfs_v1",
            "v2_transform_application_count": 1,
            "wrapper_action_clip": None,
        },
        "onnx_embedded_metadata_updated": False,
        "onnx_embedded_metadata_unchanged": True,
        "adapted_lineage_record_location": "external_hash_bound_manifest_only",
    }
    if not issues:
        adapted_model, candidate_bytes = export_final_affine_model(
            source_model,
            adapted_weight,
            adapted_bias,
        )
        _, repeated_candidate_bytes = export_final_affine_model(
            source_model,
            repeated_weight,
            repeated_bias,
        )
        if candidate_bytes != repeated_candidate_bytes:
            raise RuntimeError("repeated same-runtime adapted ONNX is not byte deterministic")
        export_report["attempted"] = True
        assert_only_final_affine_changed(source_model, adapted_model)
        export_report["only_final_affine_changed"] = True
        session = _cpu_ort_session(candidate_bytes)
        exported = np.concatenate(
            [
                session.run(["action"], {"obs_dict": row.reshape(1, DECODER_DIM)})[0]
                for row in arrays["decoder994"]
            ],
            axis=0,
        ).astype(np.float64)
        expected_float32_affine = hidden @ adapted_weight.astype(np.float64).T + adapted_bias.astype(np.float64)
        parity = float(np.max(np.abs(exported - expected_float32_affine)))
        exported_gate_report, exported_gate_issues = _resubstitution_output_gates(
            exported,
            arrays["teacher_label_raw_native23"],
            frozen_prediction,
            contract["gates"],
        )
        decoder_state = _canonical_initializer_state_binding(adapted_model)
        final_state = _canonical_initializer_state_binding(
            adapted_model,
            {FINAL_WEIGHT_NAME, FINAL_BIAS_NAME},
        )
        export_report.update(
            {
                "provider": session.get_providers()[0],
                "reference_max_abs_error": parity,
                "candidate_decoder_sha256": _sha256_bytes(candidate_bytes),
                "candidate_decoder_size_bytes": len(candidate_bytes),
                "static_binding": _model_static_binding(adapted_model),
                "candidate_decoder_canonical_initializer_state": decoder_state,
                "candidate_decoder_canonical_state_sha256": decoder_state["state_sha256"],
                "candidate_final_affine_canonical_state": final_state,
                "candidate_final_affine_state_sha256": final_state["state_sha256"],
                "actual_float32_ort_resubstitution_gates": exported_gate_report,
                "blocked_oof_gates_recomputed_on_export": False,
                "blocked_oof_gate_reason": (
                    "blocked OOF predictions come from fold-specific fits, not final exported refit"
                ),
                "same_runtime_repeat_fit_head_byte_identical": True,
                "same_runtime_repeat_fit_onnx_byte_identical": True,
                "cross_runtime_byte_determinism_claimed": False,
            }
        )
        if parity > float(contract["gates"]["maximum_export_reference_absolute_error"]):
            issues.append("export_reference_parity_gate_failed")
        if not bool(np.isfinite(exported).all()):
            issues.append("export_nonfinite_output")
        issues.extend(f"exported_float32_{issue}" for issue in exported_gate_issues)
        export_report["passed"] = not issues
        if not issues:
            decoder_bytes = candidate_bytes

    preflight_after = preflight_offline_bc(root)
    if canonical_json_bytes(preflight_after) != canonical_json_bytes(preflight_before):
        raise RuntimeError("immutable input/runtime-source drift during offline fit")
    fit_report = dict(fit.report)
    fit_report["adaptation_magnitude"] = adaptation_magnitude
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND if not issues else FAILURE_MANIFEST_KIND,
        "classification": (
            "offline_bc_eligible_for_closed_loop_simulator_experiment"
            if not issues
            else "offline_bc_diagnostic_failed_gates"
        ),
        "contract": {
            "path": CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": CONTRACT_SHA256,
        },
        "lineage": {
            **dict(preflight),
            "bootstrap_manifest_kind": bootstrap_manifest["kind"],
            "source_checkpoint_present": False,
            "optimizer_state": None,
            "resume_capable": False,
        },
        "runtime": {
            **dict(runtime),
            "onnx_version": onnx.__version__,
            "numeric_runtime": numeric_runtime,
            "runtime_sources": preflight_before["offline_bc_runtime_sources"],
            "immutable_inputs_and_runtime_sources_rechecked_after_fit": True,
            "no_simulator_or_gpu_used": True,
            "no_robot_hardware_or_network_commands_performed": True,
        },
        "fit": fit_report,
        "export": export_report,
        "gate_issues": sorted(set(issues)),
        "eligible_for_closed_loop_simulator_experiment": not issues,
        "claims": {
            "blocked_diagnostic_is_independent_heldout_evidence": False,
            "all_510_fit_is_generalization_evidence": False,
            "closed_loop_simulator_qualified": False,
            "full_clip_qualified": False,
        },
        "boundaries": {
            "offline_behavior_cloning_only": True,
            "teacher_controlled_training_data": True,
            "reset_prefix_support_admitted": False,
            "support_qualification_performed": False,
            "support_admitted": False,
            "on_policy_data": False,
            "dagger_data": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
            "robot_or_network_commands_permitted": False,
        },
    }
    return OfflineBCOutcome(
        passed=not issues,
        report=_json_safe(report),
        decoder_bytes=decoder_bytes,
    )


@dataclass(frozen=True)
class OfflineBCRequest:
    repository_root: Path
    output_prefix: Path

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path) or not isinstance(
            self.output_prefix,
            Path,
        ):
            raise TypeError("offline BC request paths must be pathlib.Path values")

    @property
    def root(self) -> Path:
        return self.repository_root.expanduser().resolve(strict=True)

    @property
    def prefix(self) -> Path:
        raw = self.output_prefix.expanduser()
        value = raw if raw.is_absolute() else self.root / raw
        result = value.resolve(strict=False)
        evidence_root = (self.root / "artifacts/g1_true23").resolve(strict=True)
        if result.parent != evidence_root or result.name in ("", ".", ".."):
            raise ValueError("offline BC output prefix must be a direct child of artifacts/g1_true23")
        if result.suffix:
            raise ValueError("offline BC output prefix must not have a suffix")
        return result

    @property
    def decoder_path(self) -> Path:
        return Path(f"{self.prefix}.decoder.onnx")

    @property
    def manifest_path(self) -> Path:
        return Path(f"{self.prefix}.manifest.json")


def _write_temporary(parent: Path, payload: bytes, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=parent,
        prefix=".offline-bc-ridge-",
        suffix=suffix,
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _publish_link_new(temporary: Path, final: Path, context: str) -> None:
    try:
        os.link(temporary, final)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {context}: {final}") from error


def _assert_outcome_lineage_current(
    request: OfflineBCRequest,
    outcome: OfflineBCOutcome,
) -> None:
    """Rehash immutable parents, runtime sources, and candidate immediately before publish."""

    lineage = outcome.report.get("lineage")
    if lineage is None:
        if outcome.passed:
            raise ValueError("passing offline BC outcome lacks lineage")
        return
    current = preflight_offline_bc(request.root)
    bound = _mapping(lineage, "outcome lineage")
    for name, value in current.items():
        if name not in bound or canonical_json_bytes(bound[name]) != canonical_json_bytes(value):
            raise RuntimeError(f"offline BC parent/runtime source drift before publish: {name}")
    runtime = _mapping(outcome.report.get("runtime"), "outcome runtime")
    numeric = _mapping(runtime.get("numeric_runtime"), "outcome numeric runtime")
    import onnx
    import onnxruntime as ort
    import scipy

    current_versions = {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "numpy_show_config_sha256": _sha256_bytes(_configuration_text(np.show_config).encode("utf-8")),
        "scipy_show_config_sha256": _sha256_bytes(_configuration_text(scipy.show_config).encode("utf-8")),
    }
    if any(numeric.get(name) != value for name, value in current_versions.items()):
        raise RuntimeError("offline BC numeric runtime drift before publish")
    if runtime.get("onnx_version") != onnx.__version__ or runtime.get("onnxruntime_version") != ort.__version__:
        raise RuntimeError("offline BC ONNX/ORT runtime drift before publish")
    libraries = numeric.get("openblas")
    if not isinstance(libraries, list) or len(libraries) != 2:
        raise ValueError("offline BC numeric runtime OpenBLAS binding missing")
    for entry in libraries:
        binding = _mapping(entry, "OpenBLAS binding")
        path = Path(str(binding.get("library_path")))
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != binding.get("library_size_bytes")
            or sha256_file(path) != binding.get("library_sha256")
        ):
            raise RuntimeError("offline BC OpenBLAS library drift before publish")
    if not outcome.passed:
        return
    if outcome.decoder_bytes is None:
        raise ValueError("passing offline BC outcome lacks decoder bytes")
    export = _mapping(outcome.report.get("export"), "outcome export")
    if export.get("candidate_decoder_sha256") != _sha256_bytes(outcome.decoder_bytes):
        raise ValueError("candidate decoder hash differs from fit report")

    contract = load_offline_bc_contract(request.root)
    source = onnx.load(
        request.root / contract["inputs"]["source_decoder"]["path"],
        load_external_data=False,
    )
    adapted = onnx.load_from_string(outcome.decoder_bytes)
    _validate_source_decoder_abi(adapted, contract)
    assert_only_final_affine_changed(source, adapted)
    decoder_state = _canonical_initializer_state_binding(adapted)
    final_state = _canonical_initializer_state_binding(
        adapted,
        {FINAL_WEIGHT_NAME, FINAL_BIAS_NAME},
    )
    if (
        export.get("candidate_decoder_canonical_state_sha256") != decoder_state["state_sha256"]
        or export.get("candidate_final_affine_state_sha256") != final_state["state_sha256"]
    ):
        raise ValueError("candidate canonical decoder/head state hash drift")


def publish_offline_bc_outcome_new(
    request: OfflineBCRequest,
    outcome: OfflineBCOutcome,
) -> tuple[Path | None, Path, Mapping[str, Any]]:
    """Atomically publish decoder+manifest on pass, manifest only on failure."""

    if type(request) is not OfflineBCRequest or type(outcome) is not OfflineBCOutcome:
        raise TypeError("exact OfflineBCRequest and OfflineBCOutcome required")
    _assert_outcome_lineage_current(request, outcome)
    decoder = request.decoder_path
    manifest = request.manifest_path
    if manifest.parent.is_symlink() or not manifest.parent.is_dir():
        raise ValueError("offline BC output directory must be regular and existing")
    for path, context in ((decoder, "offline BC decoder"), (manifest, "offline BC manifest")):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite {context}: {path}")
    if outcome.passed != (outcome.decoder_bytes is not None):
        raise ValueError("offline BC outcome decoder/pass mismatch")
    body = copy.deepcopy(dict(outcome.report))
    body["artifact"] = {
        "decoder_filename": decoder.name if outcome.passed else None,
        "decoder_sha256": (_sha256_bytes(outcome.decoder_bytes) if outcome.decoder_bytes is not None else None),
        "decoder_size_bytes": (len(outcome.decoder_bytes) if outcome.decoder_bytes is not None else None),
        "publishable_decoder": outcome.passed,
        "manifest_filename": manifest.name,
        "publication_protocol": (
            "decoder_hardlink_then_manifest_hardlink_commit"
            if outcome.passed
            else "diagnostic_manifest_hardlink_commit_no_decoder"
        ),
        "overwrite_permitted": False,
    }
    body["manifest_payload_sha256"] = _sha256_bytes(canonical_json_bytes(body))
    encoded = canonical_json_bytes(_json_safe(body))
    temporary_decoder: Path | None = None
    temporary_manifest: Path | None = None
    decoder_published = False
    manifest_published = False
    try:
        if outcome.decoder_bytes is not None:
            temporary_decoder = _write_temporary(decoder.parent, outcome.decoder_bytes, ".onnx.tmp")
            if sha256_file(temporary_decoder) != body["artifact"]["decoder_sha256"]:
                raise RuntimeError("temporary decoder hash drift")
        temporary_manifest = _write_temporary(manifest.parent, encoded, ".json.tmp")
        if temporary_decoder is not None:
            _publish_link_new(temporary_decoder, decoder, "offline BC decoder")
            decoder_published = True
        _publish_link_new(temporary_manifest, manifest, "offline BC manifest")
        manifest_published = True
        return (decoder if outcome.passed else None), manifest, body
    except BaseException:
        if manifest_published:
            manifest.unlink(missing_ok=True)
        if decoder_published:
            decoder.unlink(missing_ok=True)
        raise
    finally:
        if temporary_decoder is not None:
            temporary_decoder.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)


def failure_outcome(error: BaseException, preflight: Mapping[str, Any] | None = None) -> OfflineBCOutcome:
    """Create quarantine-only report for structural/runtime failure."""

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": FAILURE_MANIFEST_KIND,
        "classification": "offline_bc_structural_or_runtime_failure",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": CONTRACT_SHA256,
        },
        "error": f"{type(error).__name__}: {error}",
        "preflight": None if preflight is None else _json_safe(preflight),
        "gate_issues": ["structural_or_runtime_failure"],
        "eligible_for_closed_loop_simulator_experiment": False,
        "claims": {
            "closed_loop_simulator_qualified": False,
            "full_clip_qualified": False,
        },
        "boundaries": {
            "offline_behavior_cloning_only": True,
            "support_qualification_performed": False,
            "support_admitted": False,
            "on_policy_data": False,
            "dagger_data": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
            "robot_or_network_commands_permitted": False,
        },
    }
    return OfflineBCOutcome(passed=False, report=report, decoder_bytes=None)


__all__ = [
    "CONTRACT_RELATIVE_PATH",
    "CONTRACT_SHA256",
    "OfflineBCOutcome",
    "OfflineBCRequest",
    "Projection",
    "RidgeFit",
    "admit_offline_bc_inputs",
    "assert_only_final_affine_changed",
    "canonical_svd",
    "contiguous_purged_folds",
    "export_final_affine_model",
    "extract_frozen_decoder_hidden",
    "failure_outcome",
    "fit_last_affine_residual",
    "fit_projection",
    "load_offline_bc_contract",
    "preflight_offline_bc",
    "publish_offline_bc_outcome_new",
    "run_offline_bc_fit",
]
