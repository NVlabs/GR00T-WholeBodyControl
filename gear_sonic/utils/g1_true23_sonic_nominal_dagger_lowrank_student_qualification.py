"""Closed-loop adapter for the low-rank last-block SONIC student."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from gear_sonic.utils import (
    g1_true23_sonic_nominal_dagger_lowrank_last_block_bc_v2 as fit,
    g1_true23_sonic_student_closed_loop_qualification as student,
)

CANDIDATE_MANIFEST_RELATIVE_PATH = Path(
    "artifacts/g1_true23/sonic_nominal_dagger_lowrank_last_block_bc_v2.manifest.json"
)
CANDIDATE_MANIFEST_SHA256 = "52c0806c0a5b5c4697198dc6ee4fbf2f4d5d0630c3fbf3ce84c5de6c7d5a8e79"
CANDIDATE_DECODER_SHA256 = "d391c44bb29d0c0d9dd84df123d62b309be97970da906ce7c1c2a7bc62a7ba20"
ALLOWED_RUNTIME_SEEDS = (20260805, 611723381, 835868017, 921108064)
INPUT_NAMES = (
    "nominal_npz",
    "nominal_manifest",
    "cutoff50_npz",
    "cutoff50_manifest",
    "cutoff140_npz",
    "cutoff140_manifest",
    "base_decoder",
    "base_manifest",
    "parent_base_decoder",
    "bootstrap_npz",
    "bootstrap_manifest",
    "round2_decoder",
    "round2_manifest",
    "round2_failed_student_report",
    "rejected_v1_manifest",
)
EXTRA_NAMES = tuple(f"lowrank_{name}" for name in INPUT_NAMES)
CHANGED_INITIALIZERS = (
    "layers.7.bias",
    "layers.7.weight",
    "layers.8.bias",
    "layers.8.weight",
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def _validator(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    fit.validate_candidate_manifest_fields(manifest, contract)


def _validate_lowrank_candidate_decoder(
    *,
    source_decoder: Path,
    candidate_decoder: Path,
    candidate_decoder_sha256: str,
    decoder994: np.ndarray,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    student._require_sha256(candidate_decoder_sha256, "candidate_decoder_sha256")  # noqa: SLF001
    source = student._regular_file(  # noqa: SLF001
        source_decoder, student.CAUSAL_DECODER_SHA256, "low-rank source decoder proof"
    )
    candidate = student._regular_file(  # noqa: SLF001
        candidate_decoder, candidate_decoder_sha256, "low-rank candidate decoder proof"
    )
    source_model = onnx.load(source, load_external_data=False)
    candidate_model = onnx.load(candidate, load_external_data=False)
    onnx.checker.check_model(source_model, full_check=True)
    onnx.checker.check_model(candidate_model, full_check=True)
    expected_input = ("obs_dict", (1, student.DECODER_DIM), 1)
    expected_output = ("action", (1, student.ACTION_DIM), 1)
    for label, model in (("source", source_model), ("candidate", candidate_model)):
        if (
            len(model.graph.input) != 1
            or student._onnx_value_signature(model.graph.input[0]) != expected_input  # noqa: SLF001
            or len(model.graph.output) != 1
            or student._onnx_value_signature(model.graph.output[0]) != expected_output  # noqa: SLF001
            or [(entry.domain, int(entry.version)) for entry in model.opset_import] != [("", 13)]
        ):
            raise ValueError(f"{label} low-rank decoder ABI drift")
    if [node.SerializeToString() for node in source_model.graph.node] != [
        node.SerializeToString() for node in candidate_model.graph.node
    ]:
        raise ValueError("low-rank candidate graph differs from source")
    source_arrays = student._initializer_arrays(source_model)  # noqa: SLF001
    candidate_arrays = student._initializer_arrays(candidate_model)  # noqa: SLF001
    if tuple(source_arrays) != tuple(candidate_arrays):
        raise ValueError("low-rank candidate initializer order drift")
    changed: list[str] = []
    for name, source_value in source_arrays.items():
        candidate_value = candidate_arrays[name]
        if source_value.shape != candidate_value.shape or source_value.dtype != candidate_value.dtype:
            raise ValueError(f"low-rank candidate initializer ABI drift: {name}")
        if source_value.tobytes(order="C") != candidate_value.tobytes(order="C"):
            changed.append(name)
    if tuple(sorted(changed)) != CHANGED_INITIALIZERS:
        raise ValueError(f"low-rank candidate changed unexpected tensors: {sorted(changed)}")
    expected_shapes = {
        "layers.7.weight": (512, 512),
        "layers.7.bias": (512,),
        "layers.8.weight": (23, 512),
        "layers.8.bias": (23,),
    }
    if any(candidate_arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise ValueError("low-rank candidate trainable tensor shape drift")
    session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("low-rank candidate decoder provider mismatch")
    ort_output = np.concatenate(
        [session.run(["action"], {"obs_dict": row.reshape(1, student.DECODER_DIM)})[0] for row in decoder994],
        axis=0,
    )
    torch_output = student._torch_decoder_output(decoder994, candidate_arrays)  # noqa: SLF001
    error = np.abs(ort_output.astype(np.float64) - torch_output.astype(np.float64))
    maximum = float(np.max(error))
    violations = int(np.count_nonzero(error > student.MODEL_PARITY_MAX_ABSOLUTE_ERROR))
    if (
        ort_output.dtype != np.float32
        or ort_output.shape != (student.TOTAL_ROWS, student.ACTION_DIM)
        or not bool(np.isfinite(ort_output).all())
        or violations
        or maximum > student.MODEL_PARITY_MAX_ABSOLUTE_ERROR
    ):
        raise ValueError("low-rank candidate Torch/ORT all-510 parity failed")
    source_binding = student._initializer_binding(source_arrays)  # noqa: SLF001
    candidate_binding = student._initializer_binding(candidate_arrays)  # noqa: SLF001
    frozen_names = sorted(set(source_arrays) - set(CHANGED_INITIALIZERS))
    frozen_descriptor = {name: source_binding[name] for name in frozen_names}
    return {
        "source_decoder_sha256": student.CAUSAL_DECODER_SHA256,
        "candidate_decoder_sha256": candidate_decoder_sha256,
        "static_float32_abi": True,
        "opset": 13,
        "changed_initializer_names": list(CHANGED_INITIALIZERS),
        "only_layers7_and8_changed": True,
        "frozen_initializer_count": len(frozen_names),
        "frozen_layers0_through6_binding_sha256": student.sha256_bytes(
            student.canonical_json_bytes(frozen_descriptor)
        ),
        "candidate_initializer_binding_sha256": student.sha256_bytes(
            student.canonical_json_bytes(candidate_binding)
        ),
        "torch_ort_parity": {
            "check_rows": student.TOTAL_ROWS,
            "check_coordinates": int(error.size),
            "maximum_absolute_error": maximum,
            "p99_absolute_error": float(np.quantile(error, 0.99)),
            "violation_count": violations,
            "threshold": student.MODEL_PARITY_MAX_ABSOLUTE_ERROR,
            "passed": True,
        },
        "output_semantics": "plain_sonic_raw_native23_pre_v2",
        "safe_target_transform_embedded": False,
    }


@contextmanager
def _adapter_scope(root: Path, seed: int) -> Iterator[None]:
    if seed not in ALLOWED_RUNTIME_SEEDS:
        raise ValueError("runtime seed is not sealed")
    original_names = student.FROZEN_INPUT_NAMES
    extended = original_names + EXTRA_NAMES
    original_builder = student._preflight_bound_file_specs  # noqa: SLF001
    saved = {
        "FIXED_SEED": student.FIXED_SEED,
        "BC_SCHEMA_VERSION": student.BC_SCHEMA_VERSION,
        "BC_MANIFEST_KIND": student.BC_MANIFEST_KIND,
        "BC_CONTRACT_RELATIVE_PATH": student.BC_CONTRACT_RELATIVE_PATH,
        "BC_CONTRACT_SHA256": student.BC_CONTRACT_SHA256,
        "FROZEN_INPUT_NAMES": student.FROZEN_INPUT_NAMES,
        "load_offline_bc_contract": student.load_offline_bc_contract,
        "_validate_candidate_manifest_fields": student._validate_candidate_manifest_fields,  # noqa: SLF001
        "_preflight_bound_file_specs": student._preflight_bound_file_specs,  # noqa: SLF001
        "validate_candidate_decoder": student.validate_candidate_decoder,
    }

    def specs(
        request: student.StudentQualificationRequest, preflight: Mapping[str, Any]
    ) -> dict[str, dict[str, str]]:
        current = student.FROZEN_INPUT_NAMES
        try:
            student.FROZEN_INPUT_NAMES = original_names
            result = original_builder(request, preflight)
        finally:
            student.FROZEN_INPUT_NAMES = current
        inputs = _mapping(fit.load_contract(root)["inputs"], "fit inputs")
        for output_name, input_name in zip(EXTRA_NAMES, INPUT_NAMES, strict=True):
            entry = _mapping(inputs[input_name], input_name)
            result[output_name] = {
                "path": str((root / str(entry["path"])).resolve(strict=True)),
                "expected_sha256": str(entry["sha256"]),
            }
        if tuple(result) != extended:
            raise RuntimeError("low-rank frozen input inventory drift")
        return result

    try:
        student.FIXED_SEED = seed
        student.BC_SCHEMA_VERSION = 1
        student.BC_MANIFEST_KIND = fit.MANIFEST_KIND
        student.BC_CONTRACT_RELATIVE_PATH = fit.CONTRACT_RELATIVE_PATH
        student.BC_CONTRACT_SHA256 = fit.CONTRACT_SHA256
        student.FROZEN_INPUT_NAMES = extended
        student.load_offline_bc_contract = fit.load_contract
        student._validate_candidate_manifest_fields = _validator  # noqa: SLF001
        student._preflight_bound_file_specs = specs  # noqa: SLF001
        student.validate_candidate_decoder = _validate_lowrank_candidate_decoder
        yield
    finally:
        for name, value in saved.items():
            setattr(student, name, value)


def run_qualification(*, repository_root: Path, output: Path, runtime_seed: int) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    manifest = (root / CANDIDATE_MANIFEST_RELATIVE_PATH).resolve(strict=True)
    if fit.base.nominal_fit.sha256_file(manifest) != CANDIDATE_MANIFEST_SHA256:
        raise ValueError("low-rank candidate manifest hash mismatch")
    request = student.StudentQualificationRequest(
        repository_root=root,
        candidate_manifest=manifest,
        expected_candidate_manifest_sha256=CANDIDATE_MANIFEST_SHA256,
        output=output,
        mode="initial510",
    )
    if os.path.lexists(request.output_path):
        raise FileExistsError("qualification output exists")
    with _adapter_scope(root, runtime_seed):
        report = dict(student.run_student_qualification(request))
        report["lowrank_nominal_dagger_candidate"] = {
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "candidate_decoder_sha256": CANDIDATE_DECODER_SHA256,
            "runtime_seed": runtime_seed,
            "heldout_seed": runtime_seed in (835868017, 921108064),
            "teacher_substitution_used": False,
            "support_qualified": False,
        }
        student.write_student_qualification_new(request, report)
    return report


__all__ = ["ALLOWED_RUNTIME_SEEDS", "CANDIDATE_MANIFEST_SHA256", "run_qualification"]
