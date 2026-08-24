"""Project failing lower-body phases toward feasible true-23 support poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

HOME_LEGS = np.asarray(
    (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0),
    dtype=np.float64,
)
CORE_CLIPS = {
    "B_DadDance",
    "B_HandsUp",
    "J_Dance0_StepTouch",
    "B_BowKarate",
    "B_ForwardKarate",
    "J_Dance2_Salsa",
}


def _clusters(phases: list[int], gap: int) -> list[list[int]]:
    groups: list[list[int]] = []
    for phase in sorted(phases):
        if not groups or phase - groups[-1][-1] > gap:
            groups.append([phase])
        else:
            groups[-1].append(phase)
    return groups


def _cosine_mask(length: int, centers: list[int], plateau: int, taper: int) -> np.ndarray:
    mask = np.zeros(length, dtype=np.float64)
    for center in centers:
        distance = np.abs(np.arange(length) - center)
        local = np.zeros(length, dtype=np.float64)
        local[distance <= plateau] = 1.0
        transition = (distance > plateau) & (distance < plateau + taper)
        phase = (distance[transition] - plateau) / taper
        local[transition] = 0.5 + 0.5 * np.cos(np.pi * phase)
        mask = np.maximum(mask, local)
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--override-input", type=Path)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--plateau", type=int, default=90)
    parser.add_argument("--taper", type=int, default=150)
    parser.add_argument("--cluster-gap", type=int, default=250)
    parser.add_argument("--completion-threshold", type=float, default=0.8)
    parser.add_argument("--stabilize-root", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.strength <= 1.0 or not 0.0 <= args.completion_threshold <= 1.0:
        raise ValueError("strength and completion threshold must be within [0, 1]")
    if min(args.plateau, args.taper) <= 0 or args.cluster_gap < 0:
        raise ValueError("plateau/taper must be positive and cluster gap non-negative")

    sources = sorted(args.input.resolve(strict=True).glob("*.csv"))
    if not sources:
        raise FileNotFoundError("no true-23 CSV files found")
    override = args.override_input.resolve(strict=True) if args.override_input else None
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for base_source in sources:
        name = base_source.stem
        override_source = override / base_source.name if override else None
        source = override_source if override_source is not None and override_source.exists() else base_source
        destination = args.output / base_source.name
        report_path = args.gates / f"{name}.json"
        if name in CORE_CLIPS or not report_path.exists():
            shutil.copy2(source, destination)
            results.append({"clip": name, "modified": False, "reason": "core_or_missing_gate"})
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rate = float(report["completion_rate"])
        failed = [
            int(phase)
            for failure, phase in zip(
                report["first_failure_steps"], report["terminal_reference_steps"], strict=True
            )
            if int(failure) >= 0
        ]
        if rate >= args.completion_threshold or not failed:
            shutil.copy2(source, destination)
            results.append({"clip": name, "modified": False, "reason": "above_threshold"})
            continue

        motion = np.loadtxt(source, delimiter=",", dtype=np.float64)
        centers_50hz = [int(round(float(np.median(group)))) for group in _clusters(failed, args.cluster_gap)]
        centers = [min(int(round(center * 60.0 / 50.0)), len(motion) - 1) for center in centers_50hz]
        mask = _cosine_mask(len(motion), centers, args.plateau, args.taper)
        blend = (args.strength * mask)[:, None]
        original_legs = motion[:, 7:19].copy()
        motion[:, 7:19] = original_legs * (1.0 - blend) + HOME_LEGS * blend
        if args.stabilize_root:
            root_rpy = Rotation.from_quat(motion[:, 3:7]).as_euler("xyz")
            root_rpy[:, :2] *= 1.0 - blend
            motion[:, 3:7] = Rotation.from_euler("xyz", root_rpy).as_quat()
            motion[:, 2] = motion[:, 2] * (1.0 - blend[:, 0]) + 0.8 * blend[:, 0]
        np.savetxt(destination, motion, delimiter=",", fmt="%.9f")
        results.append(
            {
                "clip": name,
                "modified": True,
                "completion_rate": rate,
                "centers_50hz": centers_50hz,
                "centers_csv_60hz": centers,
                "max_leg_delta_rad": float(np.abs(motion[:, 7:19] - original_legs).max()),
            }
        )

    summary = {
        "schema": "g1_true23_failure_phase_projection_v1",
        "strength": args.strength,
        "plateau_csv_frames": args.plateau,
        "taper_csv_frames": args.taper,
        "stabilize_root": args.stabilize_root,
        "clip_count": len(results),
        "modified_count": sum(bool(item["modified"]) for item in results),
        "results": results,
    }
    (args.output / "projection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{summary['modified_count']}/{len(results)} clips projected -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
