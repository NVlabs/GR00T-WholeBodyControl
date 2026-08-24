"""Aggregate per-clip native124 gates into persistent JSON, CSV, and Markdown scorecards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gear_sonic.scripts.batch_evaluate_g1_23dof_native124_multi_motion import TRAINED_CLIPS, _category


def _row(report: dict[str, Any]) -> dict[str, Any]:
    steps = [500 if step < 0 else step for step in report["first_failure_steps"]]
    completed = int(report["completed_rollouts"])
    rollouts = int(report["rollouts"])
    completion = 100.0 * completed / rollouts
    return {
        "clip": report["clip"],
        "category": _category(report["clip"]),
        "trained": report["clip"] in TRAINED_CLIPS,
        "completed": completed,
        "rollouts": rollouts,
        "completion_score": completion,
        "survival_score": 100.0 * sum(steps) / (500.0 * len(steps)),
        "anchor_ori_failures": int(report["failure_counts"]["anchor_ori"]),
        "anchor_pos_failures": int(report["failure_counts"]["anchor_pos"]),
        "ee_body_pos_failures": int(report["failure_counts"]["ee_body_pos"]),
        "scaling_priority": "pass" if completion == 100.0 else "near" if completion >= 80.0 else "scale",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = json.loads(args.spans.read_text(encoding="utf-8"))
    expected = [span["name"] for span in catalog["spans"]]
    rows = []
    missing = []
    for clip in expected:
        path = args.gates / f"{clip}.json"
        if not path.exists():
            missing.append(clip)
            continue
        rows.append(_row(json.loads(path.read_text(encoding="utf-8"))))
    rows.sort(key=lambda item: (item["category"], item["clip"]))

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "clip_count": len(group),
            "perfect_count": sum(item["completion_score"] == 100.0 for item in group),
            "mean_completion_score": sum(item["completion_score"] for item in group) / len(group),
            "mean_survival_score": sum(item["survival_score"] for item in group) / len(group),
        }

    summary = {
        "schema": "g1_true23_native124_zero_shot_scorecard_v1",
        "definitions": {
            "completion_score": "percent of 10 strict 500-step rollouts completing without termination",
            "survival_score": "mean percent of the 500-step horizon survived",
        },
        "missing": missing,
        "all": aggregate(rows),
        "karate": aggregate([item for item in rows if item["category"] == "karate"]),
        "untrained": aggregate([item for item in rows if not item["trained"]]),
        "results": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scorecard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "scorecard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# G1 native124 zero-shot motion scorecard",
        "",
        f"Evaluated {len(rows)}/{len(expected)} clips. Completion is the percent of strict "
        "500-step rollouts that finish.",
        "",
        "| Motion | Group | Trained | Completion | Survival | Priority |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in sorted(rows, key=lambda row: (row["category"] != "karate", row["clip"])):
        lines.append(
            f"| {item['clip']} | {item['category']} | {'yes' if item['trained'] else 'no'} | "
            f"{item['completion_score']:.0f} | {item['survival_score']:.1f} | {item['scaling_priority']} |"
        )
    (args.output / "scorecard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("all", "karate", "untrained", "missing")}, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
