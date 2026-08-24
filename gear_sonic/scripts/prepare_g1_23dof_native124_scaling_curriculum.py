"""Build retention-safe staged curricula from the native124 zero-shot scorecard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _weight(row: dict[str, Any], *, first_stage: bool) -> float:
    score = float(row["completion_score"])
    if bool(row["trained"]):
        return 4.0 if first_stage else 12.0
    if score == 100.0:
        return 1.0
    if score >= 80.0:
        return 2.0 if first_stage else 3.0
    if score >= 40.0:
        return 3.0 if first_stage else 5.0
    if score > 0.0:
        return 4.0 if first_stage else 7.0
    return 6.0 if first_stage else 10.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--census-npz", type=Path, required=True)
    parser.add_argument("--override-npz", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scorecard = json.loads(args.scorecard.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    census_npz = args.census_npz.resolve(strict=True)
    override_npz = args.override_npz.resolve(strict=True) if args.override_npz else None
    pilot_root = census_npz.parent.parent
    repo_root = pilot_root.parent.parent
    proven_paths = {
        "B_DadDance": repo_root
        / "external_dependencies/unitree_rl_mjlab/src/assets/motions/g1_23dof/B_DadDance.npz",
        "B_HandsUp": pilot_root / "motions_true23/B_HandsUp.npz",
        "J_Dance0_StepTouch": pilot_root / "motions_true23/J_Dance0_StepTouch.npz",
        "B_BowKarate": pilot_root / "motions_true23/B_BowKarate_adaptive3_safe_root.npz",
        "B_ForwardKarate": pilot_root / "motions_true23/B_ForwardKarate_adaptive3.npz",
        "J_Dance2_Salsa": pilot_root / "motions_true23/J_Dance2_Salsa_adaptive3_safe_root.npz",
    }
    proven_paths = {name: path.resolve(strict=True) for name, path in proven_paths.items()}

    def entry(row: dict[str, Any], *, first_stage: bool) -> dict[str, Any]:
        name = str(row["clip"])
        override = override_npz / f"{name}.npz" if override_npz is not None else None
        if override is not None and override.exists():
            path = str(override.resolve(strict=True))
        elif name in proven_paths:
            path = str(proven_paths[name])
        else:
            path = str((census_npz / f"{name}.npz").resolve(strict=True))
        return {"name": name, "path": path, "weight": _weight(row, first_stage=first_stage)}

    rows = scorecard["results"]
    stage_a_rows = [
        row for row in rows if bool(row["trained"]) or float(row["completion_score"]) >= 80.0
    ]
    manifests = {
        "stage_a_manifest.json": {
            "description": "proven retention plus zero-shot perfect/near motions",
            "motions": [entry(row, first_stage=True) for row in stage_a_rows],
        },
        "stage_b_all61_manifest.json": {
            "description": "all 61 motions, failure-weighted with proven retention",
            "motions": [entry(row, first_stage=False) for row in rows],
        },
    }
    for filename, manifest in manifests.items():
        path = args.output / filename
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{len(manifest['motions'])} clips -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
