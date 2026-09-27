"""Convert pinned Unitree Dex3 data or run fail-closed Inspire diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Literal

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.data.unitree_conversion.pipeline import (
    PipelineReport,
    run_dex3_pipeline,
    run_inspire_diagnostics,
)
from gear_sonic.data.unitree_conversion.provenance import (
    load_source_lock_snapshot,
)


@dataclass(frozen=True)
class ConvertConfig:
    """Explicit, revision-locked conversion configuration."""

    source_lock: Path = Path("gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml")
    output_root: Path = Path("outputs/unitree_sonic_conversion")
    kind: Literal["dex3", "inspire"] = "dex3"
    smoke: bool = True
    cache_dir: Path | None = None
    resume: bool = True
    workers: int = 1


def _validate_config(config: ConvertConfig, snapshot=None):
    if type(config.workers) is not int or config.workers < 1:
        raise ValueError("workers must be an integer of at least one")
    if snapshot is None:
        snapshot = load_source_lock_snapshot(config.source_lock)
    if snapshot.lock.scope == "smoke" and not config.smoke:
        raise ValueError("a scope: smoke source lock requires --smoke")
    return snapshot


def run(config: ConvertConfig) -> tuple[PipelineReport, Path]:
    """Execute one source-specific mode and persist its canonical report."""
    snapshot = load_source_lock_snapshot(config.source_lock)
    snapshot = _validate_config(config, snapshot)
    if config.kind == "dex3":
        report_root = config.output_root
        report = run_dex3_pipeline(
            lock=snapshot,
            output_root=report_root,
            smoke=config.smoke,
            cache_dir=config.cache_dir,
            resume=config.resume,
        )
    else:
        report_root = config.output_root / "diagnostic-only"
        report = run_inspire_diagnostics(
            lock=snapshot,
            output_root=report_root,
            smoke=config.smoke,
            cache_dir=config.cache_dir,
        )
    report_path = report.write_json(report_root / "pipeline-report.json")
    return report, report_path


def main() -> None:
    import tyro

    config = tyro.cli(ConvertConfig)
    report, report_path = run(config)
    print(json.dumps({"report_path": str(report_path)}, sort_keys=True))
    if not report.succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
