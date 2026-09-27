from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from gear_sonic.data.unitree_conversion.contracts import INSPIRE_GATE_REASONS
from gear_sonic.data.unitree_conversion.pipeline import (
    run_dex3_pipeline,
    run_inspire_diagnostics,
)
from gear_sonic.data.unitree_conversion.provenance import load_source_lock_snapshot

LOCK = Path("gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml")
RUN_NETWORK_SMOKE = os.environ.get("UNITREE_SONIC_RUN_NETWORK_SMOKE") == "1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(
    not RUN_NETWORK_SMOKE,
    reason="set UNITREE_SONIC_RUN_NETWORK_SMOKE=1 to run the pinned five-plus-five cohort",
)
def test_fixed_five_plus_five_network_smoke_and_resume(tmp_path: Path) -> None:
    snapshot = load_source_lock_snapshot(LOCK)
    cache = Path(os.environ.get("UNITREE_SONIC_SMOKE_CACHE", tmp_path / "cache"))
    output = Path(os.environ.get("UNITREE_SONIC_SMOKE_OUTPUT", tmp_path / "output"))

    inspire_report = run_inspire_diagnostics(
        lock=snapshot,
        output_root=output / "diagnostic-only",
        smoke=True,
        cache_dir=cache,
    )
    assert inspire_report.source_episode_ids == (0, 152, 304, 456, 608)
    assert inspire_report.status_counts == {"blocked_unverified": 5}
    assert inspire_report.gate_reasons == frozenset(INSPIRE_GATE_REASONS)
    assert inspire_report.encoder_invocation_count == 0
    assert inspire_report.target_dataset_paths == ()

    dex_report = run_dex3_pipeline(
        lock=snapshot,
        output_root=output,
        smoke=True,
        cache_dir=cache,
        resume=True,
    )
    assert dex_report.source_episode_ids == (0, 78, 155, 233, 310)
    assert dex_report.validated_episode_count == 5
    assert dex_report.failed_episode_count == 0
    assert dex_report.status_counts == {"validated": 5}
    assert len(dex_report.target_dataset_paths) == 1

    dataset = dex_report.target_dataset_paths[0]
    manifest = dataset / "source-manifest.json"
    checksums = dataset / "dataset-checksums.sha256"
    before = (_sha256(manifest), _sha256(checksums))

    resumed_report = run_dex3_pipeline(
        lock=snapshot,
        output_root=output,
        smoke=True,
        cache_dir=cache,
        resume=True,
    )
    assert resumed_report.source_episode_ids == (0, 78, 155, 233, 310)
    assert resumed_report.validated_episode_count == 5
    assert resumed_report.failed_episode_count == 0
    assert resumed_report.status_counts == {"reused": 5}
    assert resumed_report.encoder_invocation_count == 0
    assert resumed_report.target_dataset_paths == (dataset,)
    assert (_sha256(manifest), _sha256(checksums)) == before
