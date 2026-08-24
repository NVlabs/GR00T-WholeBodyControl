"""Closed-loop adapter for targeted seed835 counterexample SONIC candidate."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from gear_sonic.utils import (
    g1_true23_sonic_seed835_counterexample_bc_last_affine as fit,
    g1_true23_sonic_student_closed_loop_qualification as student,
)

CANDIDATE_MANIFEST_RELATIVE_PATH = Path(
    "artifacts/g1_true23/sonic_seed835_counterexample_bc_last_affine_v1.manifest.json"
)
CANDIDATE_MANIFEST_SHA256 = "e8194a759a829368a9635493fab079cb30e38d3d74780e13ad743ea8b2a517fe"
CANDIDATE_DECODER_SHA256 = "812123cdea6c500f769223f06903c663c1e39debacdbd9d03d16ecb480b39745"
ALLOWED_RUNTIME_SEEDS = (20260805, 611723381, 835868017, 921108064)
INPUT_NAMES = (
    "nominal_npz",
    "nominal_manifest",
    "intervention_npz",
    "intervention_manifest",
    "seed835_teacher_npz",
    "seed835_teacher_manifest",
    "counterexample_npz",
    "counterexample_manifest",
    "base_decoder",
    "base_manifest",
    "bootstrap_npz",
    "bootstrap_manifest",
)
EXTRA_NAMES = tuple(f"seed835_counterexample_{name}" for name in INPUT_NAMES)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def _validator(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    fit.validate_candidate_manifest_fields(manifest, contract)


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
            raise RuntimeError("counterexample candidate frozen input inventory drift")
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
        yield
    finally:
        for name, value in saved.items():
            setattr(student, name, value)


def run_qualification(*, repository_root: Path, output: Path, runtime_seed: int) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    manifest = (root / CANDIDATE_MANIFEST_RELATIVE_PATH).resolve(strict=True)
    if fit.nominal_fit.sha256_file(manifest) != CANDIDATE_MANIFEST_SHA256:
        raise ValueError("counterexample candidate manifest hash mismatch")
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
        report["seed835_counterexample_candidate"] = {
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "candidate_decoder_sha256": CANDIDATE_DECODER_SHA256,
            "runtime_seed": runtime_seed,
            "teacher_substitution_used": False,
            "support_qualified": False,
            "hardware_authorized": False,
        }
        student.write_student_qualification_new(request, report)
    return report


__all__ = ["ALLOWED_RUNTIME_SEEDS", "CANDIDATE_MANIFEST_SHA256", "run_qualification"]
