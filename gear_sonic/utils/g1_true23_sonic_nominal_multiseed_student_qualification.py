"""Closed-loop qualification adapter for the nominal multi-seed SONIC student."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from gear_sonic.utils import (
    g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge as fit,
    g1_true23_sonic_student_closed_loop_qualification as student,
)

CANDIDATE_MANIFEST_RELATIVE_PATH = Path(
    "artifacts/g1_true23/sonic_nominal_multiseed_bc_last_affine_ridge_v1.manifest.json"
)
CANDIDATE_MANIFEST_SHA256 = "359e2526e9d8b8fb009dca3841b831798555046922feb93425257aac0f16dca9"
ALLOWED_RUNTIME_SEEDS = (20260805, 611723381, 835868017, 921108064)
EXTRA_FROZEN_INPUT_NAMES = (
    "nominal_dataset_npz",
    "nominal_dataset_manifest",
    "nominal_dataset_contract",
    "nominal_base_decoder",
    "nominal_base_manifest",
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _candidate_validator(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    fit.validate_candidate_manifest_fields(manifest, contract)


@contextmanager
def _adapter_scope(root: Path, runtime_seed: int) -> Iterator[None]:
    if runtime_seed not in ALLOWED_RUNTIME_SEEDS:
        raise ValueError("runtime seed is not in the sealed qualification set")
    original_names = student.FROZEN_INPUT_NAMES
    extended_names = original_names + EXTRA_FROZEN_INPUT_NAMES
    original_spec_builder = student._preflight_bound_file_specs  # noqa: SLF001
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

    def extended_spec_builder(
        request: student.StudentQualificationRequest,
        preflight: Mapping[str, Any],
    ) -> dict[str, dict[str, str]]:
        current = student.FROZEN_INPUT_NAMES
        try:
            student.FROZEN_INPUT_NAMES = original_names
            specs = original_spec_builder(request, preflight)
        finally:
            student.FROZEN_INPUT_NAMES = current
        contract = fit.load_contract(root)
        inputs = _mapping(contract["inputs"], "nominal fit inputs")
        for output_name, input_name in zip(
            EXTRA_FROZEN_INPUT_NAMES,
            ("dataset_npz", "dataset_manifest", "dataset_contract", "base_decoder", "base_manifest"),
            strict=True,
        ):
            entry = _mapping(inputs[input_name], input_name)
            specs[output_name] = {
                "path": str((root / str(entry["path"])).resolve(strict=True)),
                "expected_sha256": str(entry["sha256"]),
            }
        if tuple(specs) != extended_names:
            raise RuntimeError("nominal student frozen input inventory drift")
        return specs

    try:
        student.FIXED_SEED = runtime_seed
        student.BC_SCHEMA_VERSION = fit.SCHEMA_VERSION
        student.BC_MANIFEST_KIND = fit.MANIFEST_KIND
        student.BC_CONTRACT_RELATIVE_PATH = fit.CONTRACT_RELATIVE_PATH
        student.BC_CONTRACT_SHA256 = fit.CONTRACT_SHA256
        student.FROZEN_INPUT_NAMES = extended_names
        student.load_offline_bc_contract = fit.load_contract
        student._validate_candidate_manifest_fields = _candidate_validator  # noqa: SLF001
        student._preflight_bound_file_specs = extended_spec_builder  # noqa: SLF001
        yield
    finally:
        for name, value in saved.items():
            setattr(student, name, value)


def run_nominal_multiseed_student_qualification(
    *, repository_root: Path, output: Path, runtime_seed: int
) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    manifest = (root / CANDIDATE_MANIFEST_RELATIVE_PATH).resolve(strict=True)
    if fit.sha256_file(manifest) != CANDIDATE_MANIFEST_SHA256:
        raise ValueError("nominal multi-seed candidate manifest hash mismatch")
    request = student.StudentQualificationRequest(
        repository_root=root,
        candidate_manifest=manifest,
        expected_candidate_manifest_sha256=CANDIDATE_MANIFEST_SHA256,
        mode="initial510",
        output=output,
    )
    if os.path.lexists(request.output_path):
        raise FileExistsError("nominal multi-seed student qualification output exists")
    with _adapter_scope(root, runtime_seed):
        report = dict(student.run_student_qualification(request))
        report["nominal_multiseed_candidate"] = {
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "candidate_decoder_sha256": "91014f0cc37899ae795cc09e6a5a3c653ff6c587cf5d10cf22f41cbad280d544",
            "runtime_seed": runtime_seed,
            "seed_role": (
                "historical_baseline_seed"
                if runtime_seed == 20260805
                else "known_failed_seed"
                if runtime_seed == 611723381
                else "collection_label_heldout_seed"
            ),
            "heldout_labels_used_for_fit": False,
        }
        student.write_student_qualification_new(request, report)
    return report


__all__ = [
    "ALLOWED_RUNTIME_SEEDS",
    "CANDIDATE_MANIFEST_SHA256",
    "run_nominal_multiseed_student_qualification",
]
