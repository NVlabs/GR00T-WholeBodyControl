"""Score converted clips by how hard they are to track, and emit a curriculum.

A decoder that has just been reshaped from 29 to 23 outputs cannot follow fast
root translation or large end-effector excursions on the first iterations. Given
a corpus of arbitrary motion it terminates immediately and learns to end
episodes rather than to track, which is what the anchor_pos and ee_body_pos
termination counts showed.

Difficulty here is deliberately measured against the terminations that actually
fire: root (anchor) speed and vertical end-effector excursion. Clips are ranked
so an easy subset can be trained first and harder motion introduced later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Body indices in the converted npz: 0 is the pelvis (anchor). Hands and feet
# are the bodies the ee_body_pos termination watches.
ANCHOR_BODY = 0
END_EFFECTOR_BODIES = (7, 13, 19, 24 - 1)


def score_clip(path: Path) -> dict[str, float] | None:
    try:
        with np.load(path) as data:
            root = np.asarray(data["body_pos_w"])[:, ANCHOR_BODY, :]
            bodies = np.asarray(data["body_pos_w"])
            joint_vel = np.asarray(data["joint_vel"])
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            frames = int(root.shape[0])
    except Exception:  # noqa: BLE001
        return None
    if frames < 32:
        return None

    dt = 1.0 / fps
    root_speed = np.linalg.norm(np.diff(root, axis=0), axis=-1) / dt
    horizontal = np.linalg.norm(root[:, :2] - root[0, :2], axis=-1)

    ee = bodies[:, list(END_EFFECTOR_BODIES), 2]
    ee_excursion = np.abs(ee - ee[0]).max()

    return {
        "frames": float(frames),
        "duration_s": frames / fps,
        "root_speed_mean": float(root_speed.mean()),
        "root_speed_max": float(root_speed.max()),
        "root_travel_m": float(horizontal.max()),
        "ee_excursion_m": float(ee_excursion),
        "joint_vel_max": float(np.abs(joint_vel).max()),
        # Ranking blends the two signals the terminations actually watch.
        "difficulty": float(
            root_speed.mean() * 2.0 + ee_excursion + np.abs(joint_vel).max() * 0.05
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = sorted(args.input.glob("*.npz"))
    if args.limit:
        paths = paths[: args.limit]

    scored: dict[str, dict[str, float]] = {}
    for index, path in enumerate(paths):
        result = score_clip(path)
        if result is not None:
            scored[path.name] = result
        if (index + 1) % 500 == 0:
            print(f"  scored {index + 1}/{len(paths)}", flush=True)

    if not scored:
        raise SystemExit("no clips could be scored")

    ranked = sorted(scored.items(), key=lambda kv: kv[1]["difficulty"])
    difficulties = np.array([v["difficulty"] for _, v in ranked])
    speeds = np.array([v["root_speed_mean"] for _, v in ranked])

    args.output.write_text(
        json.dumps(
            {
                "kind": "g1_true23_motion_difficulty_v1",
                "clip_count": len(ranked),
                "difficulty_percentiles": {
                    str(p): float(np.percentile(difficulties, p))
                    for p in (10, 25, 50, 75, 90)
                },
                "root_speed_percentiles": {
                    str(p): float(np.percentile(speeds, p))
                    for p in (10, 25, 50, 75, 90)
                },
                "ranked": [{"name": n, **v} for n, v in ranked],
            },
            indent=1,
        )
    )

    print(f"\nscored {len(ranked)} clips -> {args.output}")
    print(f"{'percentile':<12}{'difficulty':>12}{'root_speed':>12}")
    for p in (10, 25, 50, 75, 90):
        print(
            f"{p:<12}{np.percentile(difficulties, p):>12.3f}"
            f"{np.percentile(speeds, p):>12.3f}"
        )
    print("\neasiest 3:")
    for name, value in ranked[:3]:
        print(f"  {value['difficulty']:6.2f}  speed={value['root_speed_mean']:.3f}  {name}")
    print("hardest 3:")
    for name, value in ranked[-3:]:
        print(f"  {value['difficulty']:6.2f}  speed={value['root_speed_mean']:.3f}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
