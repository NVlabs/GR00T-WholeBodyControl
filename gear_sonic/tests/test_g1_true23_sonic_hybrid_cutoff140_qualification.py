from __future__ import annotations

from pathlib import Path

import pytest

from gear_sonic.scripts.qualify_g1_true23_sonic_hybrid_cutoff140 import _parser
from gear_sonic.utils import (
    g1_true23_sonic_hybrid_cutoff140_qualification as hybrid,
    g1_true23_sonic_student_closed_loop_qualification as student,
    g1_true23_sonic_student_teacher_recovery as recovery,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_contract_binds_prior_pass_and_exact_partition() -> None:
    contract = hybrid.load_contract(REPO_ROOT)
    assert contract["runtime"]["student_transitions"] == 140
    assert contract["runtime"]["teacher_transitions"] == 370
    assert contract["runtime"]["teacher_first_q9"] == 149
    assert contract["boundaries"]["mixed_controller_teleop_candidate"] is True
    assert contract["boundaries"]["pure_student_qualified"] is False
    assert contract["boundaries"]["hardware_authorized"] is False


def test_qualification_scope_patches_seed_sources_and_restores() -> None:
    old_seed = student.FIXED_SEED
    old_sources = recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS
    with hybrid._qualification_scope(REPO_ROOT, 835868017):  # noqa: SLF001
        assert student.FIXED_SEED == 835868017
        for path in hybrid.SOURCE_PATHS:
            assert path in recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS
    assert student.FIXED_SEED == old_seed
    assert recovery.RECOVERY_EXECUTED_SOURCE_RELATIVE_PATHS == old_sources


def test_cli_rejects_unsealed_seed_and_requires_output() -> None:
    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--runtime-seed", "123", "--output", "x.json"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--runtime-seed", str(hybrid.ALLOWED_RUNTIME_SEEDS[0])])
