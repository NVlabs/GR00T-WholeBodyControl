"""Build full-clip retention plus exact failure-phase replay windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ARRAY_NAMES = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
CORE_CLIPS = {
    "B_DadDance",
    "B_HandsUp",
    "J_Dance0_StepTouch",
    "B_BowKarate",
    "B_ForwardKarate",
    "J_Dance2_Salsa",
}


def _full_weight(rate: float, core: bool, core_weight: float) -> float:
    if core:
        return core_weight
    if rate >= 1.0:
        return 1.0
    if rate >= 0.8:
        return 2.0
    if rate >= 0.5:
        return 3.0
    if rate > 0.0:
        return 4.0
    return 6.0


def _clusters(phases: list[int], gap: int) -> list[list[int]]:
    groups: list[list[int]] = []
    for phase in sorted(phases):
        if not groups or phase - groups[-1][-1] > gap:
            groups.append([phase])
        else:
            groups[-1].append(phase)
    return sorted(groups, key=lambda group: (-len(group), int(np.median(group))))


def _write_window(source: Path, output: Path, center: int) -> tuple[int, int]:
    with np.load(source, allow_pickle=False) as data:
        frames = int(data["joint_pos"].shape[0])
        if frames < 500:
            raise ValueError("phase window source must contain at least 500 frames")
        start = min(max(center - 300, 0), frames - 500)
        stop = start + 500
        payload = {
            name: np.asarray(data[name][start:stop], dtype=np.float32) for name in ARRAY_NAMES
        }
        fps = np.asarray(data["fps"], dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, fps=fps, **payload)
    return start, stop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-windows-per-clip", type=int, default=2)
    parser.add_argument("--cluster-gap", type=int, default=250)
    parser.add_argument("--core-weight", type=float, default=12.0)
    parser.add_argument("--core-window-weight", type=float, default=12.0)
    args = parser.parse_args()
    if (
        args.max_windows_per_clip <= 0
        or args.cluster_gap < 0
        or args.core_weight <= 0.0
        or args.core_window_weight <= 0.0
    ):
        raise ValueError("window count and weights must be positive; cluster gap non-negative")

    catalog = json.loads(args.spans.resolve(strict=True).read_text(encoding="utf-8"))
    source_by_name = {
        str(source["name"]): Path(source["path"]).resolve(strict=True)
        for source in catalog["sources"]
    }
    args.output.mkdir(parents=True, exist_ok=True)
    windows_dir = args.output / "windows"
    motions: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []

    for span in catalog["spans"]:
        name = str(span["name"])
        source = source_by_name[name]
        report_path = (args.gates / f"{name}.json").resolve(strict=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rate = float(report["completion_rate"])
        motions.append(
            {
                "name": name,
                "path": str(source),
                "weight": _full_weight(rate, name in CORE_CLIPS, args.core_weight),
            }
        )
        failed_phases = [
            int(phase)
            for failure, phase in zip(
                report["first_failure_steps"],
                report["terminal_reference_steps"],
                strict=True,
            )
            if int(failure) >= 0
        ]
        with np.load(source, allow_pickle=False) as data:
            frames = int(data["joint_pos"].shape[0])
        if not failed_phases or frames < 500:
            continue
        for window_id, group in enumerate(
            _clusters(failed_phases, args.cluster_gap)[: args.max_windows_per_clip]
        ):
            center = int(round(float(np.median(group))))
            window_name = f"{name}__failure_{window_id + 1}"
            output_path = windows_dir / f"{window_name}.npz"
            start, stop = _write_window(source, output_path, center)
            weight = (
                args.core_window_weight
                if name in CORE_CLIPS
                else max(4.0, 8.0 * (1.0 - rate))
            )
            motions.append({"name": window_name, "path": str(output_path.resolve()), "weight": weight})
            windows.append(
                {
                    "name": window_name,
                    "source_clip": name,
                    "failure_samples": len(group),
                    "failure_phases": group,
                    "start": start,
                    "stop": stop,
                    "weight": weight,
                }
            )

    manifest = {
        "schema": "g1_true23_phase_local_replay_manifest_v1",
        "description": "all full clips retained plus clustered failure-phase windows",
        "motions": motions,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "g1_true23_phase_local_replay_summary_v1",
        "full_clip_count": len(catalog["spans"]),
        "window_count": len(windows),
        "windows": windows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{len(catalog['spans'])} full clips + {len(windows)} failure windows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
