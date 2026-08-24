"""Round4 seed835 failure-prefix collector using generic round3 machinery."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from gear_sonic.utils import (
    g1_true23_sonic_seed835_counterexample_round3_rank256_student_qualification as adapter,
    g1_true23_sonic_seed835_failure_prefix_dagger_round3 as base,
)

CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_seed835_failure_prefix_dagger_round4_v1.json"
)
CONTRACT_SHA256 = "14cdfda19ab9001eafaba76ab5b7428332f78261d856301eec14dc9cc279b68a"
CONTRACT_KIND = "g1_true23_sonic_seed835_failure_prefix_dagger_round4_contract_v1"
MANIFEST_KIND = "g1_true23_sonic_seed835_failure_prefix_dagger_round4_manifest_v1"
ROWS = 486
SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/utils/g1_true23_sonic_seed835_failure_prefix_dagger_round3.py"),
    Path("gear_sonic/utils/g1_true23_sonic_seed835_failure_prefix_dagger_round4.py"),
    Path("gear_sonic/scripts/collect_g1_true23_sonic_seed835_failure_prefix_dagger_round4.py"),
)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be mapping")
    return value


def load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if base.recovery.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("round4 failure-prefix contract hash mismatch")
    body = _mapping(json.loads(path.read_text(encoding="utf-8")), "contract")
    rows = _mapping(body.get("rows"), "rows")
    admission = _mapping(body.get("admission"), "admission")
    if (
        body.get("schema_version") != 1
        or body.get("kind") != CONTRACT_KIND
        or rows != {"total": 486, "student_on_policy": 486, "teacher_actuated": 0, "q9_first": 9, "q9_last": 494}
        or admission.get("expected_terminal_transition") != 485
        or admission.get("expected_terminal_q9") != 494
        or admission.get("expected_termination_names") != ["ee_body_pos"]
        or body.get("boundaries", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("round4 failure-prefix contract semantic drift")
    inputs = _mapping(body.get("inputs"), "inputs")
    for name, raw_entry in inputs.items():
        entry = _mapping(raw_entry, name)
        candidate = (root / str(entry["path"])).resolve(strict=True)
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or base.recovery.sha256_file(candidate) != entry["sha256"]
        ):
            raise ValueError(f"round4 failure-prefix input mismatch: {name}")
    parent = _mapping(
        json.loads((root / str(inputs["student_failure"]["path"])).read_text(encoding="utf-8")),
        "parent failure",
    )
    first_done = _mapping(parent.get("first_done"), "parent first_done")
    if (
        parent.get("verdict") != "student_qualification_failed"
        or parent.get("attempted_transitions") != ROWS
        or first_done.get("transition") != 485
        or first_done.get("q9_before") != 494
        or first_done.get("termination_names") != ["ee_body_pos"]
    ):
        raise ValueError("round4 failure-prefix parent evidence drift")
    return body


@contextmanager
def _scope() -> Iterator[None]:
    saved = {
        "adapter": base.adapter,
        "CONTRACT_RELATIVE_PATH": base.CONTRACT_RELATIVE_PATH,
        "CONTRACT_SHA256": base.CONTRACT_SHA256,
        "CONTRACT_KIND": base.CONTRACT_KIND,
        "MANIFEST_KIND": base.MANIFEST_KIND,
        "ROWS": base.ROWS,
        "SOURCE_PATHS": base.SOURCE_PATHS,
        "load_contract": base.load_contract,
    }
    try:
        base.adapter = adapter
        base.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
        base.CONTRACT_SHA256 = CONTRACT_SHA256
        base.CONTRACT_KIND = CONTRACT_KIND
        base.MANIFEST_KIND = MANIFEST_KIND
        base.ROWS = ROWS
        base.SOURCE_PATHS = SOURCE_PATHS
        base.load_contract = load_contract
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def preflight(request: base.collection_base.CollectionRequest) -> Mapping[str, Any]:
    with _scope():
        return base.preflight(request)


def collect(
    request: base.collection_base.CollectionRequest,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    with _scope():
        return base.collect(request)


def publish(
    request: base.collection_base.CollectionRequest,
    arrays: Mapping[str, np.ndarray],
    materials: Mapping[str, Any],
) -> tuple[Path, Path, Mapping[str, Any]]:
    with _scope():
        return base.publish(request, arrays, materials)


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    with _scope():
        base._validate_arrays(arrays)  # noqa: SLF001


__all__ = ["MANIFEST_KIND", "ROWS", "collect", "load_contract", "preflight", "publish"]
