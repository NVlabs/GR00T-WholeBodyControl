"""Versioned round2 refit with a whole-recovery improvement gate calibrated to correlated rows."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from gear_sonic.utils import g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge as base

CONTRACT_KIND = "g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge_contract_v2"
MANIFEST_KIND = "g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge_manifest_v2"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_nominal_dagger_round2_bc_last_affine_ridge_v2.json"
)
CONTRACT_SHA256 = "e003f8c2db18cd205204fd07c7f6c213c322ba467a0423353b33f8cfffadd152"


@contextmanager
def _version_scope() -> Iterator[None]:
    saved = {
        "CONTRACT_KIND": base.CONTRACT_KIND,
        "MANIFEST_KIND": base.MANIFEST_KIND,
        "CONTRACT_RELATIVE_PATH": base.CONTRACT_RELATIVE_PATH,
        "CONTRACT_SHA256": base.CONTRACT_SHA256,
    }
    try:
        base.CONTRACT_KIND = CONTRACT_KIND
        base.MANIFEST_KIND = MANIFEST_KIND
        base.CONTRACT_RELATIVE_PATH = CONTRACT_RELATIVE_PATH
        base.CONTRACT_SHA256 = CONTRACT_SHA256
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def load_contract(root: Path) -> Mapping[str, Any]:
    with _version_scope():
        return base.load_contract(root)


def preflight(request: Any) -> Mapping[str, Any]:
    with _version_scope():
        return base.preflight(request)


def run_fit(request: Any) -> Any:
    with _version_scope():
        return base.run_fit(request)


def validate_candidate_manifest_fields(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    with _version_scope():
        base.validate_candidate_manifest_fields(manifest, contract)


__all__ = [
    "CONTRACT_SHA256",
    "MANIFEST_KIND",
    "load_contract",
    "preflight",
    "run_fit",
    "validate_candidate_manifest_fields",
]
