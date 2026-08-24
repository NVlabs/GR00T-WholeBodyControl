"""Run exact selected teacher from q9=9 on heldout seed835."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from gear_sonic.utils import g1_true23_native124_selected_source_nominal_qualification as nominal

RUNTIME_SEED = 835868017
PARENT_FAILURE_RELATIVE_PATH = Path("artifacts/g1_true23/sonic_hybrid_cutoff50.seed835868017.v1.json")
PARENT_FAILURE_SHA256 = "89e7809a7c8c6c599f465ceb3f542b398b73b78f23fdd3e9787b8257eb78d9a6"


@contextmanager
def _seed_scope() -> Iterator[None]:
    previous = nominal.FIXED_SEED
    try:
        nominal.FIXED_SEED = RUNTIME_SEED
        yield
    finally:
        nominal.FIXED_SEED = previous


def run(*, repository_root: Path, output: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    failure = (root / PARENT_FAILURE_RELATIVE_PATH).resolve(strict=True)
    if nominal.sha256_file(failure) != PARENT_FAILURE_SHA256:
        raise ValueError("heldout835 cutoff50 failure evidence drift")
    request = nominal.SelectedSourceNominalQualificationRequest(root, output)
    with _seed_scope():
        report = dict(nominal.run_selected_source_nominal_qualification(request))
        report["heldout_seed_diagnostic"] = {
            "runtime_seed": RUNTIME_SEED,
            "controller": "exact_selected21204_teacher_from_q9_9",
            "parent_failure": {
                "path": PARENT_FAILURE_RELATIVE_PATH.as_posix(),
                "sha256": PARENT_FAILURE_SHA256,
            },
            "simulator_only": True,
            "hardware_authorized": False,
        }
        nominal.write_selected_source_nominal_qualification_new(request, report)
    return report


__all__ = ["PARENT_FAILURE_SHA256", "RUNTIME_SEED", "run"]
