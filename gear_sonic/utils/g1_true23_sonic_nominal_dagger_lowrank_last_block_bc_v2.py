"""Versioned low-rank fit with a five-percent whole-recovery improvement gate."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from gear_sonic.utils import g1_true23_sonic_nominal_dagger_lowrank_last_block_bc as base

CONTRACT_KIND = "g1_true23_sonic_nominal_dagger_lowrank_last_block_bc_contract_v2"
MANIFEST_KIND = "g1_true23_sonic_nominal_dagger_lowrank_last_block_bc_manifest_v2"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_nominal_dagger_lowrank_last_block_bc_v2.json"
)
CONTRACT_SHA256 = "a49783aeea3469f331e4271c2f528988d565486cac918cf007e5a4d30c3d9cb2"


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
