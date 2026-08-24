from __future__ import annotations

from pathlib import Path

from gear_sonic.utils import (
    g1_true23_sonic_seed835_counterexample_round3_rank256_student_qualification as adapter,
    g1_true23_sonic_student_closed_loop_qualification as student,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_adapter_scope_binds_rank256_candidate_and_restores() -> None:
    saved = (student.FIXED_SEED, student.BC_MANIFEST_KIND, student.FROZEN_INPUT_NAMES)
    with adapter._adapter_scope(REPOSITORY_ROOT, 835868017):  # noqa: SLF001
        assert student.FIXED_SEED == 835868017
        assert student.BC_MANIFEST_KIND.endswith("round3_rank256_bc_last_affine_manifest_v1")
        assert len(student.FROZEN_INPUT_NAMES) == len(saved[2]) + len(adapter.EXTRA_NAMES)
    assert (student.FIXED_SEED, student.BC_MANIFEST_KIND, student.FROZEN_INPUT_NAMES) == saved


def test_candidate_hashes_match() -> None:
    manifest = REPOSITORY_ROOT / adapter.CANDIDATE_MANIFEST_RELATIVE_PATH
    decoder = manifest.with_name("sonic_seed835_counterexample_round3_rank256_bc_last_affine_v1.decoder.onnx")
    assert adapter.fit.nominal_fit.sha256_file(manifest) == adapter.CANDIDATE_MANIFEST_SHA256
    assert adapter.fit.nominal_fit.sha256_file(decoder) == adapter.CANDIDATE_DECODER_SHA256
