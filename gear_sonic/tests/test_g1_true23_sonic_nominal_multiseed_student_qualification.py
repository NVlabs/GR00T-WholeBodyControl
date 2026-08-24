from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gear_sonic.scripts.qualify_g1_true23_sonic_nominal_multiseed_student import _parser
from gear_sonic.utils import (
    g1_true23_sonic_nominal_multiseed_bc_last_affine_ridge as fit,
    g1_true23_sonic_nominal_multiseed_student_qualification as adapter,
    g1_true23_sonic_student_closed_loop_qualification as student,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _manifest() -> dict:
    return json.loads((REPO_ROOT / adapter.CANDIDATE_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))


def test_candidate_manifest_exact_and_tamper_rejected() -> None:
    contract = fit.load_contract(REPO_ROOT)
    manifest = _manifest()
    fit.validate_candidate_manifest_fields(manifest, contract)
    bad = copy.deepcopy(manifest)
    bad["boundaries"]["hardware_authorized"] = True
    with pytest.raises(ValueError):
        fit.validate_candidate_manifest_fields(bad, contract)


def test_adapter_scope_patches_and_restores_all_globals() -> None:
    old_seed = student.FIXED_SEED
    old_names = student.FROZEN_INPUT_NAMES
    with adapter._adapter_scope(REPO_ROOT, 835868017):  # noqa: SLF001
        assert student.FIXED_SEED == 835868017
        assert student.BC_MANIFEST_KIND == fit.MANIFEST_KIND
        assert student.BC_CONTRACT_SHA256 == fit.CONTRACT_SHA256
        assert student.FROZEN_INPUT_NAMES == old_names + adapter.EXTRA_FROZEN_INPUT_NAMES
    assert student.FIXED_SEED == old_seed
    assert student.FROZEN_INPUT_NAMES == old_names


def test_real_adapter_preflight_ready_without_simulator() -> None:
    request = student.StudentQualificationRequest(
        repository_root=REPO_ROOT,
        candidate_manifest=REPO_ROOT / adapter.CANDIDATE_MANIFEST_RELATIVE_PATH,
        expected_candidate_manifest_sha256=adapter.CANDIDATE_MANIFEST_SHA256,
        output=Path("artifacts/g1_true23/unused_nominal_multiseed_student_test.json"),
        mode="initial510",
    )
    with adapter._adapter_scope(REPO_ROOT, 921108064):  # noqa: SLF001
        report = student.preflight_student_qualification(request)
        assert report["ready"] is True
        assert report["fixed_runtime"]["seed"] == 921108064
        assert tuple(report["frozen_input_files"]) == student.FROZEN_INPUT_NAMES
        assert len(report["frozen_input_files"]) == 22
        assert report["safety"]["simulator_constructed"] is False
        assert report["safety"]["simulator_steps"] == 0


def test_cli_seeds_are_fixed_and_output_required() -> None:
    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--runtime-seed", "123", "--output", "x.json"])
    args = parser.parse_args(
        [
            "--runtime-seed",
            str(adapter.ALLOWED_RUNTIME_SEEDS[0]),
            "--output",
            "artifacts/g1_true23/test.json",
        ]
    )
    assert args.runtime_seed == adapter.ALLOWED_RUNTIME_SEEDS[0]
