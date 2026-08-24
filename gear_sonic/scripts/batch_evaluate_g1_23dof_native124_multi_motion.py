"""Run sparse strict native124 zero-shot gates over every corpus clip."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

TRAINED_CLIPS = {
    "B_BowKarate",
    "B_DadDance",
    "B_ForwardKarate",
    "B_HandsUp",
    "J_Dance0_StepTouch",
    "J_Dance2_Salsa",
}


def _category(name: str) -> str:
    if "Karate" in name or name.startswith("M_"):
        return "karate"
    named_dances = {
        "B_DadDance",
        "B_LongDance",
        "B_SpiralDance",
        "B_StretchDance",
        "B_WiggleDance",
    }
    if "Dance" in name or name in named_dances:
        return "dance"
    return "bonus"


def _run_clip(
    evaluator: Path,
    spans: Path,
    checkpoint: Path,
    output_dir: Path,
    clip: str,
    rollouts: int,
) -> dict[str, Any]:
    report_path = output_dir / f"{clip}.json"
    log_path = output_dir / f"{clip}.log"
    command = [
        sys.executable,
        str(evaluator),
        "--spans",
        str(spans),
        "--checkpoint",
        str(checkpoint),
        "--clip",
        clip,
        "--output",
        str(report_path),
        "--rollouts",
        str(rollouts),
    ]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("ORT_NUM_THREADS", "1")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False)
    if not report_path.exists():
        return {"clip": clip, "error": f"evaluator exit {result.returncode}; no report", "log": str(log_path)}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    steps = [500 if step < 0 else step for step in report["first_failure_steps"]]
    return {
        "clip": clip,
        "category": _category(clip),
        "trained": clip in TRAINED_CLIPS,
        "completed": int(report["completed_rollouts"]),
        "rollouts": int(report["rollouts"]),
        "completion_score": 100.0 * float(report["completed_rollouts"]) / float(report["rollouts"]),
        "survival_score": 100.0 * sum(steps) / (500.0 * len(steps)),
        "failure_counts": report["failure_counts"],
        "report": str(report_path),
        "log": str(log_path),
        "evaluator_exit": result.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("gear_sonic/scripts/evaluate_g1_23dof_native124_multi_motion.py"),
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.rollouts <= 0:
        raise ValueError("workers and rollouts must be positive")

    spans = args.spans.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    evaluator = args.evaluator.resolve(strict=True)
    catalog = json.loads(spans.read_text(encoding="utf-8"))
    clips = [str(span["name"]) for span in catalog["spans"]]
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_clip,
                evaluator,
                spans,
                checkpoint,
                args.output,
                clip,
                args.rollouts,
            ): clip
            for clip in clips
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            score = row.get("completion_score", "error")
            print(f"[{index}/{len(clips)}] {row['clip']}: {score}", flush=True)

    rows.sort(key=lambda row: row["clip"])
    valid = [row for row in rows if "error" not in row]
    summary = {
        "schema": "g1_true23_native124_zero_shot_census_v1",
        "checkpoint": str(checkpoint),
        "spans": str(spans),
        "rollouts_per_clip": args.rollouts,
        "steps_per_rollout": 500,
        "clip_count": len(clips),
        "evaluated_count": len(valid),
        "perfect_count": sum(row["completed"] == row["rollouts"] for row in valid),
        "karate": {
            "clip_count": sum(row["category"] == "karate" for row in valid),
            "perfect_count": sum(
                row["category"] == "karate" and row["completed"] == row["rollouts"] for row in valid
            ),
        },
        "untrained": {
            "clip_count": sum(not row["trained"] for row in valid),
            "perfect_count": sum(not row["trained"] and row["completed"] == row["rollouts"] for row in valid),
        },
        "results": rows,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")
    return 0 if len(valid) == len(clips) else 2


if __name__ == "__main__":
    raise SystemExit(main())
