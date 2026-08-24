"""Build and validate span-contained native124 multi-motion corpora."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "g1_true23_native124_incremental_corpus_v1"
FPS = 50.0
EPISODE_FRAMES = 500
ARRAY_NAMES = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
POSE_ARRAYS = ("joint_pos", "body_pos_w", "body_quat_w")
VELOCITY_ARRAYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")


@dataclass(frozen=True)
class MotionSpan:
    name: str
    start: int
    stop: int
    original_length: int
    weight: float

    @property
    def stored_length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class CorpusCatalog:
    corpus_path: Path
    corpus_sha256: str
    frame_count: int
    spans: tuple[MotionSpan, ...]
    episode_frames: int = EPISODE_FRAMES
    fps: float = FPS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(ARRAY_NAMES) - set(data.files))
        if missing:
            raise ValueError(f"{path.name} missing arrays: {missing}")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        arrays = {name: np.asarray(data[name], dtype=np.float32) for name in ARRAY_NAMES}
    if fps != FPS:
        raise ValueError(f"{path.name} must be exactly {FPS:g} Hz, got {fps:g}")
    frames = arrays["joint_pos"].shape[0]
    if arrays["joint_pos"].shape != (frames, 23) or arrays["joint_vel"].shape != (frames, 23):
        raise ValueError(f"{path.name} is not a native 23-DoF motion")
    if frames < 2 or any(value.shape[0] != frames for value in arrays.values()):
        raise ValueError(f"{path.name} has inconsistent frame counts")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"{path.name} contains NaN or Inf")
    return arrays


def _terminal_hold(arrays: Mapping[str, np.ndarray], minimum_frames: int) -> dict[str, np.ndarray]:
    frames = arrays["joint_pos"].shape[0]
    if frames >= minimum_frames:
        return {name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()}
    repeat = minimum_frames - frames
    result: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if name in POSE_ARRAYS:
            tail = np.repeat(value[-1:], repeat, axis=0)
        else:
            tail = np.zeros((repeat, *value.shape[1:]), dtype=np.float32)
        result[name] = np.concatenate((value, tail), axis=0).astype(np.float32)
    return result


def build_corpus(
    entries: Sequence[tuple[str, Path, float]],
    output_path: Path,
    sidecar_path: Path,
    *,
    episode_frames: int = EPISODE_FRAMES,
) -> CorpusCatalog:
    if not entries:
        raise ValueError("at least one motion is required")
    if episode_frames <= 0:
        raise ValueError("episode_frames must be positive")
    names = [name for name, _, _ in entries]
    if len(set(names)) != len(names):
        raise ValueError("motion names must be unique")

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in ARRAY_NAMES}
    spans: list[MotionSpan] = []
    sources: list[dict[str, Any]] = []
    cursor = 0
    for name, source, weight in entries:
        source = source.expanduser().resolve(strict=True)
        if not name or not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("motion name and weight must be positive")
        raw = _load_motion(source)
        original_length = raw["joint_pos"].shape[0]
        held = _terminal_hold(raw, episode_frames)
        stored_length = held["joint_pos"].shape[0]
        for array_name in ARRAY_NAMES:
            chunks[array_name].append(held[array_name])
        spans.append(MotionSpan(name, cursor, cursor + stored_length, original_length, float(weight)))
        sources.append(
            {
                "name": name,
                "path": str(source),
                "sha256": sha256_file(source),
                "original_length": original_length,
                "stored_length": stored_length,
                "terminal_hold_frames": stored_length - original_length,
                "weight": float(weight),
            }
        )
        cursor += stored_length

    joined = {name: np.concatenate(values, axis=0).astype(np.float32) for name, values in chunks.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, fps=np.asarray([FPS], dtype=np.float64), **joined)
    temporary.replace(output_path)
    corpus_hash = sha256_file(output_path)
    payload = {
        "schema": SCHEMA,
        "fps": FPS,
        "episode_frames": episode_frames,
        "corpus": {
            "path": output_path.name,
            "sha256": corpus_hash,
            "frame_count": cursor,
        },
        "sources": sources,
        "spans": [
            {
                "name": span.name,
                "start": span.start,
                "stop": span.stop,
                "original_length": span.original_length,
                "weight": span.weight,
            }
            for span in spans
        ],
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return load_catalog(sidecar_path)


def load_catalog(sidecar_path: Path) -> CorpusCatalog:
    sidecar_path = sidecar_path.expanduser().resolve(strict=True)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or float(payload.get("fps", 0.0)) != FPS:
        raise ValueError("unsupported corpus sidecar")
    episode_frames = int(payload.get("episode_frames", 0))
    if episode_frames <= 0:
        raise ValueError("invalid episode frame count")
    corpus_meta = payload["corpus"]
    corpus_path = (sidecar_path.parent / corpus_meta["path"]).resolve(strict=True)
    expected_hash = str(corpus_meta["sha256"])
    if sha256_file(corpus_path) != expected_hash:
        raise ValueError("corpus hash differs from sidecar")
    frame_count = int(corpus_meta["frame_count"])
    spans = tuple(
        MotionSpan(
            name=str(raw["name"]),
            start=int(raw["start"]),
            stop=int(raw["stop"]),
            original_length=int(raw["original_length"]),
            weight=float(raw["weight"]),
        )
        for raw in payload["spans"]
    )
    cursor = 0
    for span in spans:
        if span.start != cursor or span.stop <= span.start or span.stored_length < episode_frames:
            raise ValueError("corpus spans are not contiguous episode-safe windows")
        if span.original_length <= 0 or span.original_length > span.stored_length or span.weight <= 0.0:
            raise ValueError("invalid corpus span metadata")
        cursor = span.stop
    if cursor != frame_count:
        raise ValueError("span coverage differs from corpus frame count")
    arrays = _load_motion(corpus_path)
    if arrays["joint_pos"].shape[0] != frame_count:
        raise ValueError("corpus arrays differ from sidecar frame count")
    return CorpusCatalog(corpus_path, expected_hash, frame_count, spans, episode_frames, FPS)


def window_start_bounds(span: MotionSpan, episode_frames: int = EPISODE_FRAMES) -> tuple[int, int]:
    upper = span.stop - episode_frames
    if upper < span.start:
        raise ValueError("span cannot contain an episode")
    return span.start, upper
