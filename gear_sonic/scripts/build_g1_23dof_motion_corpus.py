"""Materialize many converted clips into one corpus npz plus a span sidecar.

The stock mjlab ``MotionLoader`` reads a single npz, so a corpus has to be
presented as one file. Concatenation alone is not enough: episode starts are
sampled uniformly over the whole timeline, so a window can run off the end of
one clip and into the next, presenting a reference that teleports. The sidecar
records where every clip begins and ends so the sampler can be constrained to
stay inside one.

Clips shorter than the episode length are dropped: they can never host a full
episode, and keeping them would only add boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_multi_motion import MOTION_ARRAY_NAMES

DEFAULT_EPISODE_FRAMES = 500


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spans", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--episode-frames", type=int, default=DEFAULT_EPISODE_FRAMES)
    parser.add_argument("--expected-fps", type=float, default=50.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = sorted(args.input.glob("*.npz"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no npz clips under {args.input}")

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in MOTION_ARRAY_NAMES}
    spans: list[dict[str, object]] = []
    cursor = 0
    dropped_short = 0
    dropped_bad = 0

    for path in paths:
        try:
            with np.load(path) as data:
                if any(name not in data for name in MOTION_ARRAY_NAMES):
                    dropped_bad += 1
                    continue
                frames = int(data["joint_pos"].shape[0])
                if frames < args.episode_frames:
                    dropped_short += 1
                    continue
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                if abs(fps - args.expected_fps) > 1e-6:
                    dropped_bad += 1
                    continue
                arrays = {name: np.asarray(data[name]) for name in MOTION_ARRAY_NAMES}
                if not all(np.all(np.isfinite(v)) for v in arrays.values()):
                    dropped_bad += 1
                    continue
        except Exception:  # noqa: BLE001
            dropped_bad += 1
            continue
        for name in MOTION_ARRAY_NAMES:
            chunks[name].append(arrays[name])
        spans.append({"name": path.stem, "start": cursor, "length": frames})
        cursor += frames

    if not spans:
        raise SystemExit(
            f"no clip reached {args.episode_frames} frames; "
            f"dropped_short={dropped_short} dropped_bad={dropped_bad}"
        )

    joined = {
        name: np.concatenate(values, axis=0).astype(np.float32)
        for name, values in chunks.items()
    }
    joined["fps"] = np.asarray([args.expected_fps], dtype=np.float64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **joined)
    temporary.replace(args.output)

    sidecar = args.spans or args.output.with_suffix(".spans.json")
    sidecar.write_text(
        json.dumps(
            {
                "kind": "g1_true23_motion_corpus_spans_v1",
                "corpus": str(args.output),
                "episode_frames": args.episode_frames,
                "fps": args.expected_fps,
                "clip_count": len(spans),
                "total_frames": cursor,
                "dropped_short": dropped_short,
                "dropped_bad": dropped_bad,
                "spans": spans,
            },
            indent=1,
        )
    )

    hours = cursor / args.expected_fps / 3600.0
    print(
        f"[OK] {len(spans)} clips -> {args.output}\n"
        f"     {cursor} frames ({hours:.2f} h at {args.expected_fps:g} fps)\n"
        f"     dropped: {dropped_short} short (<{args.episode_frames}), "
        f"{dropped_bad} unusable\n"
        f"     spans -> {sidecar}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
