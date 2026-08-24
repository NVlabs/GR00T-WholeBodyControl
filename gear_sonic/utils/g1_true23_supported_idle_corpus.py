"""Build and verify the exact Step1A supported-idle native motion corpus.

The stock MJLab ``MotionLoader`` accepts one NPZ.  This module materializes the
two pinned Step1A idle motions as four non-overlapping spans in one NPZ:

* one 500-frame static-frame-0 span per source clip;
* one trajectory span per source clip, with terminal hold padding only where
  needed to reach the 500-step training horizon.

Outputs remain diagnostic material.  Building this corpus does not authorize
training or deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = 1
KIND = "g1_true23_supported_idle_native_corpus_v1"
FPS = 50.0
EPISODE_FRAMES = 500
BODY_COUNT = 24

STATIC_KIND = "static_frame0"
TRAJECTORY_KIND = "trajectory"
STATIC_TRANSFORM = "static_frame0_repeat_zero_velocity_v1"
TRAJECTORY_TRANSFORM = "trajectory_terminal_hold_zero_velocity_v1"

FPS_ARRAY = "fps"
POSE_ARRAY_NAMES = ("joint_pos", "body_pos_w", "body_quat_w")
VELOCITY_ARRAY_NAMES = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")
MOTION_ARRAY_NAMES = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
NPZ_ARRAY_NAMES = (FPS_ARRAY, *MOTION_ARRAY_NAMES)

_ARRAY_TRAILING_SHAPES: dict[str, tuple[int, ...]] = {
    "joint_pos": (23,),
    "joint_vel": (23,),
    "body_pos_w": (BODY_COUNT, 3),
    "body_quat_w": (BODY_COUNT, 4),
    "body_lin_vel_w": (BODY_COUNT, 3),
    "body_ang_vel_w": (BODY_COUNT, 3),
}


@dataclass(frozen=True)
class PinnedClip:
    clip_id: str
    sha256: str
    frame_count: int


CHANGE_IDLE = PinnedClip(
    clip_id="idle__220713__change_idle_left_a021",
    sha256="2b1f3fa0a47f2d301d7dafd1826e2833b7902c13c7ed8c975f2d01bd9e6ef703",
    frame_count=362,
)
HANDS_ON_BACK = PinnedClip(
    clip_id="idle__220721__hands_on_back_loop_a036m",
    sha256="03465b7668773c6390b9424b39bd6cdb8adda1bd89dc07a440fda06d661570b2",
    frame_count=547,
)
PINNED_CLIPS = (CHANGE_IDLE, HANDS_ON_BACK)


def sha256_file(path: str | Path) -> str:
    """Return lowercase SHA-256 for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_sidecar_path(corpus_path: str | Path) -> Path:
    """Return ``<corpus-stem>.spans.json`` beside a corpus NPZ."""

    corpus = Path(corpus_path)
    return corpus.with_suffix(".spans.json")


def _regular_source(path: str | Path, description: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{description} may not be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is not a regular file: {resolved}")
    return resolved


def _new_output(path: str | Path, suffix: str, description: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise FileExistsError(f"refusing to overwrite {description}: {candidate}")
    resolved = candidate.resolve()
    if resolved.suffix.lower() != suffix:
        raise ValueError(f"{description} must end in {suffix}")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite {description}: {resolved}")
    return resolved


def _validate_quaternions(values: np.ndarray, description: str) -> None:
    norms = np.linalg.norm(values.astype(np.float64), axis=-1)
    if not np.allclose(norms, 1.0, atol=1.0e-5, rtol=0.0):
        worst = float(np.max(np.abs(norms - 1.0)))
        raise ValueError(f"{description} contains non-unit quaternions; max norm error={worst}")


def _load_motion(path: Path, pinned: PinnedClip) -> dict[str, np.ndarray]:
    actual_hash = sha256_file(path)
    if actual_hash != pinned.sha256:
        raise ValueError(f"{pinned.clip_id} SHA-256 mismatch: expected {pinned.sha256}, got {actual_hash}")

    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(NPZ_ARRAY_NAMES):
                raise ValueError(f"{pinned.clip_id} arrays differ from exact schema: {sorted(archive.files)}")
            arrays = {name: np.array(archive[name], copy=True) for name in NPZ_ARRAY_NAMES}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and pinned.clip_id in str(error):
            raise
        raise ValueError(f"cannot load {pinned.clip_id} as a safe NPZ") from error

    fps = arrays[FPS_ARRAY]
    if fps.shape != (1,) or fps.dtype != np.dtype(np.float64):
        raise ValueError(f"{pinned.clip_id} fps must be float64 shape (1,)")
    if not np.isfinite(fps).all() or float(fps[0]) != FPS:
        raise ValueError(f"{pinned.clip_id} fps must be exactly {FPS:g}")

    for name in MOTION_ARRAY_NAMES:
        value = arrays[name]
        expected_shape = (pinned.frame_count, *_ARRAY_TRAILING_SHAPES[name])
        if value.shape != expected_shape:
            raise ValueError(
                f"{pinned.clip_id} {name} shape mismatch: expected {expected_shape}, got {value.shape}"
            )
        if value.dtype != np.dtype(np.float32):
            raise ValueError(f"{pinned.clip_id} {name} must be float32")
        if not np.isfinite(value).all():
            raise ValueError(f"{pinned.clip_id} {name} contains NaN or Inf")

    _validate_quaternions(arrays["body_quat_w"], f"{pinned.clip_id} body_quat_w")
    return arrays


def _repeat_frame(value: np.ndarray, frame: int, count: int) -> np.ndarray:
    return np.repeat(value[frame : frame + 1], count, axis=0)


def _static_arrays(source: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays = {name: _repeat_frame(source[name], 0, EPISODE_FRAMES) for name in POSE_ARRAY_NAMES}
    arrays.update(
        {
            name: np.zeros(
                (EPISODE_FRAMES, *source[name].shape[1:]),
                dtype=np.float32,
            )
            for name in VELOCITY_ARRAY_NAMES
        }
    )
    return arrays


def _trajectory_arrays(
    source: Mapping[str, np.ndarray], source_frame_count: int
) -> tuple[dict[str, np.ndarray], int]:
    repeat_count = max(EPISODE_FRAMES - source_frame_count, 0)
    arrays: dict[str, np.ndarray] = {}
    for name in MOTION_ARRAY_NAMES:
        original = source[name]
        if repeat_count == 0:
            arrays[name] = np.array(original, copy=True)
            continue
        suffix = (
            _repeat_frame(original, source_frame_count - 1, repeat_count)
            if name in POSE_ARRAY_NAMES
            else np.zeros((repeat_count, *original.shape[1:]), dtype=np.float32)
        )
        arrays[name] = np.concatenate((original, suffix), axis=0)
    return arrays, repeat_count


def _span(
    *,
    pinned: PinnedClip,
    kind: str,
    start: int,
    stored_length: int,
    original_length: int,
    terminal_repeat_count: int,
    transform: str,
) -> dict[str, Any]:
    return {
        "id": f"{pinned.clip_id}::{kind}",
        "source_clip_id": pinned.clip_id,
        "kind": kind,
        "start": start,
        "stop": start + stored_length,
        "stored_length": stored_length,
        "original_length": original_length,
        "source_frame_count": pinned.frame_count,
        "terminal_repeat_count": terminal_repeat_count,
        "transform": transform,
    }


def _expected_spans() -> list[dict[str, Any]]:
    descriptions = (
        (CHANGE_IDLE, STATIC_KIND, EPISODE_FRAMES, 1, EPISODE_FRAMES - 1, STATIC_TRANSFORM),
        (HANDS_ON_BACK, STATIC_KIND, EPISODE_FRAMES, 1, EPISODE_FRAMES - 1, STATIC_TRANSFORM),
        (
            CHANGE_IDLE,
            TRAJECTORY_KIND,
            EPISODE_FRAMES,
            CHANGE_IDLE.frame_count,
            EPISODE_FRAMES - CHANGE_IDLE.frame_count,
            TRAJECTORY_TRANSFORM,
        ),
        (
            HANDS_ON_BACK,
            TRAJECTORY_KIND,
            HANDS_ON_BACK.frame_count,
            HANDS_ON_BACK.frame_count,
            0,
            TRAJECTORY_TRANSFORM,
        ),
    )
    spans: list[dict[str, Any]] = []
    cursor = 0
    for pinned, kind, stored, original, repeated, transform in descriptions:
        spans.append(
            _span(
                pinned=pinned,
                kind=kind,
                start=cursor,
                stored_length=stored,
                original_length=original,
                terminal_repeat_count=repeated,
                transform=transform,
            )
        )
        cursor += stored
    return spans


EXPECTED_SPANS = tuple(_expected_spans())
TOTAL_FRAMES = EXPECTED_SPANS[-1]["stop"]


def _array_contract(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(arrays[name].shape),
            "dtype": arrays[name].dtype.name,
        }
        for name in NPZ_ARRAY_NAMES
    }


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load supported-idle sidecar: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("supported-idle sidecar must be a JSON object")
    return payload


def _expect_exact_keys(value: Mapping[str, Any], expected: set[str], description: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{description} keys mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _load_corpus_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(NPZ_ARRAY_NAMES):
                raise ValueError(f"corpus arrays differ from exact schema: {sorted(archive.files)}")
            return {name: np.array(archive[name], copy=True) for name in NPZ_ARRAY_NAMES}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "corpus arrays" in str(error):
            raise
        raise ValueError(f"cannot load supported-idle corpus: {path}") from error


def _validate_corpus_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    fps = arrays[FPS_ARRAY]
    if fps.shape != (1,) or fps.dtype != np.dtype(np.float64):
        raise ValueError("corpus fps must be float64 shape (1,)")
    if not np.isfinite(fps).all() or float(fps[0]) != FPS:
        raise ValueError(f"corpus fps must be exactly {FPS:g}")

    for name in MOTION_ARRAY_NAMES:
        value = arrays[name]
        expected_shape = (TOTAL_FRAMES, *_ARRAY_TRAILING_SHAPES[name])
        if value.shape != expected_shape:
            raise ValueError(f"corpus {name} shape mismatch: expected {expected_shape}, got {value.shape}")
        if value.dtype != np.dtype(np.float32):
            raise ValueError(f"corpus {name} must be float32")
        if not np.isfinite(value).all():
            raise ValueError(f"corpus {name} contains NaN or Inf")
    _validate_quaternions(arrays["body_quat_w"], "corpus body_quat_w")


def _slice(arrays: Mapping[str, np.ndarray], span: Mapping[str, Any], name: str) -> np.ndarray:
    return arrays[name][int(span["start"]) : int(span["stop"])]


def _validate_materialized_semantics(arrays: Mapping[str, np.ndarray]) -> None:
    change_static, hands_static, change_trajectory, hands_trajectory = EXPECTED_SPANS

    for static_span, trajectory_span in (
        (change_static, change_trajectory),
        (hands_static, hands_trajectory),
    ):
        for name in POSE_ARRAY_NAMES:
            static = _slice(arrays, static_span, name)
            if not np.array_equal(static, np.repeat(static[:1], EPISODE_FRAMES, axis=0)):
                raise ValueError(f"{static_span['id']} {name} does not repeat frame 0 exactly")
            trajectory = _slice(arrays, trajectory_span, name)
            if not np.array_equal(static[0], trajectory[0]):
                raise ValueError(f"{static_span['id']} {name} differs from trajectory frame 0")
        for name in VELOCITY_ARRAY_NAMES:
            if np.count_nonzero(_slice(arrays, static_span, name)) != 0:
                raise ValueError(f"{static_span['id']} {name} must be exactly zero")

    change_original = CHANGE_IDLE.frame_count
    for name in POSE_ARRAY_NAMES:
        trajectory = _slice(arrays, change_trajectory, name)
        expected_suffix = np.repeat(
            trajectory[change_original - 1 : change_original],
            EPISODE_FRAMES - change_original,
            axis=0,
        )
        if not np.array_equal(trajectory[change_original:], expected_suffix):
            raise ValueError(f"{change_trajectory['id']} {name} terminal hold mismatch")
    for name in VELOCITY_ARRAY_NAMES:
        trajectory = _slice(arrays, change_trajectory, name)
        if np.count_nonzero(trajectory[change_original:]) != 0:
            raise ValueError(f"{change_trajectory['id']} {name} terminal hold must be zero")

    for name in MOTION_ARRAY_NAMES:
        if _slice(arrays, hands_trajectory, name).shape[0] != HANDS_ON_BACK.frame_count:
            raise ValueError(f"{hands_trajectory['id']} {name} was truncated or padded")


def _validate_against_sources(arrays: Mapping[str, np.ndarray], source_paths: Mapping[str, str | Path]) -> None:
    if set(source_paths) != {clip.clip_id for clip in PINNED_CLIPS}:
        raise ValueError("source_paths must contain exactly the two pinned clip IDs")
    loaded = {
        pinned.clip_id: _load_motion(
            _regular_source(source_paths[pinned.clip_id], f"{pinned.clip_id} source"), pinned
        )
        for pinned in PINNED_CLIPS
    }
    span_by_id = {span["id"]: span for span in EXPECTED_SPANS}
    for pinned in PINNED_CLIPS:
        source = loaded[pinned.clip_id]
        static = span_by_id[f"{pinned.clip_id}::{STATIC_KIND}"]
        trajectory = span_by_id[f"{pinned.clip_id}::{TRAJECTORY_KIND}"]
        for name in POSE_ARRAY_NAMES:
            if not np.array_equal(_slice(arrays, static, name)[0], source[name][0]):
                raise ValueError(f"{static['id']} {name} does not match pinned source frame 0")
        for name in MOTION_ARRAY_NAMES:
            materialized = _slice(arrays, trajectory, name)[: pinned.frame_count]
            if not np.array_equal(materialized, source[name]):
                raise ValueError(f"{trajectory['id']} {name} changed an original source frame")


def validate_supported_idle_corpus(
    corpus_path: str | Path,
    sidecar_path: str | Path | None = None,
    *,
    source_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Fail closed unless corpus bytes, sidecar, and optional sources all agree."""

    corpus = _regular_source(corpus_path, "supported-idle corpus")
    sidecar = _regular_source(
        sidecar_path if sidecar_path is not None else default_sidecar_path(corpus),
        "supported-idle sidecar",
    )
    if corpus.parent != sidecar.parent:
        raise ValueError("supported-idle corpus and sidecar must share one directory")

    payload = _strict_json(sidecar)
    _expect_exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "fps",
            "episode_frames",
            "diagnostic_only",
            "training_authorized",
            "corpus",
            "sources",
            "spans",
        },
        "sidecar",
    )
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(payload["schema_version"], bool):
        raise ValueError("supported-idle sidecar schema_version mismatch")
    if payload["kind"] != KIND:
        raise ValueError("supported-idle sidecar schema or kind mismatch")
    if payload["fps"] != FPS or isinstance(payload["fps"], bool):
        raise ValueError("supported-idle sidecar fps mismatch")
    if payload["episode_frames"] != EPISODE_FRAMES or isinstance(payload["episode_frames"], bool):
        raise ValueError("supported-idle sidecar episode_frames mismatch")
    if payload["diagnostic_only"] is not True or payload["training_authorized"] is not False:
        raise ValueError("supported-idle corpus must remain diagnostic-only and unauthorized")

    corpus_meta = payload["corpus"]
    if not isinstance(corpus_meta, dict):
        raise ValueError("sidecar corpus entry must be an object")
    _expect_exact_keys(
        corpus_meta,
        {"path", "sha256", "frame_count", "arrays"},
        "sidecar corpus",
    )
    if corpus_meta["path"] != corpus.name:
        raise ValueError("sidecar corpus path must be its basename beside the sidecar")
    if corpus_meta["frame_count"] != TOTAL_FRAMES or isinstance(corpus_meta["frame_count"], bool):
        raise ValueError("sidecar corpus total_frames mismatch")
    if corpus_meta["sha256"] != sha256_file(corpus):
        raise ValueError("sidecar corpus SHA-256 mismatch")

    arrays = _load_corpus_arrays(corpus)
    _validate_corpus_arrays(arrays)
    expected_array_contract = _array_contract(arrays)
    if corpus_meta["arrays"] != expected_array_contract:
        raise ValueError("sidecar corpus array shapes or dtypes mismatch")

    sources = payload["sources"]
    if not isinstance(sources, list) or len(sources) != len(PINNED_CLIPS):
        raise ValueError("sidecar must contain exactly two pinned sources")
    for entry, pinned in zip(sources, PINNED_CLIPS, strict=True):
        if not isinstance(entry, dict):
            raise ValueError("sidecar source entries must be objects")
        _expect_exact_keys(
            entry,
            {"clip_id", "path", "sha256", "source_frame_count"},
            f"source {pinned.clip_id}",
        )
        if (
            entry["clip_id"] != pinned.clip_id
            or entry["sha256"] != pinned.sha256
            or entry["source_frame_count"] != pinned.frame_count
            or isinstance(entry["source_frame_count"], bool)
            or not isinstance(entry["path"], str)
            or not entry["path"]
        ):
            raise ValueError(f"sidecar source identity mismatch for {pinned.clip_id}")

    spans = payload["spans"]
    if spans != list(EXPECTED_SPANS):
        raise ValueError("sidecar spans differ from exact supported-idle layout")

    _validate_materialized_semantics(arrays)
    if source_paths is not None:
        _validate_against_sources(arrays, source_paths)
    return payload


def _write_temporary_npz(parent: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    descriptor, name = tempfile.mkstemp(dir=parent, prefix=".supported-idle-", suffix=".npz.tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_temporary_bytes(parent: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(dir=parent, prefix=".supported-idle-", suffix=".json.tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_new(temporary: Path, output: Path, description: str) -> None:
    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {description}: {output}") from error


def build_supported_idle_corpus(
    *,
    change_motion: str | Path,
    hands_motion: str | Path,
    corpus_path: str | Path,
    sidecar_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build, publish, reopen, and exactly verify the four-span corpus."""

    corpus = _new_output(corpus_path, ".npz", "supported-idle corpus")
    sidecar = _new_output(
        sidecar_path if sidecar_path is not None else default_sidecar_path(corpus),
        ".json",
        "supported-idle sidecar",
    )
    if corpus.parent != sidecar.parent:
        raise ValueError("supported-idle corpus and sidecar must share one directory")
    if corpus == sidecar:
        raise ValueError("supported-idle corpus and sidecar paths must differ")
    corpus.parent.mkdir(parents=True, exist_ok=True)

    source_paths = {
        CHANGE_IDLE.clip_id: _regular_source(change_motion, "change-idle source"),
        HANDS_ON_BACK.clip_id: _regular_source(hands_motion, "hands-on-back source"),
    }
    if corpus in source_paths.values() or sidecar in source_paths.values():
        raise ValueError("output paths may not alias pinned source motions")
    source_arrays = {pinned.clip_id: _load_motion(source_paths[pinned.clip_id], pinned) for pinned in PINNED_CLIPS}

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in MOTION_ARRAY_NAMES}
    for pinned in PINNED_CLIPS:
        static = _static_arrays(source_arrays[pinned.clip_id])
        for name in MOTION_ARRAY_NAMES:
            chunks[name].append(static[name])
    for pinned in PINNED_CLIPS:
        trajectory, repeat_count = _trajectory_arrays(source_arrays[pinned.clip_id], pinned.frame_count)
        expected_repeat = max(EPISODE_FRAMES - pinned.frame_count, 0)
        if repeat_count != expected_repeat:
            raise RuntimeError("internal terminal-repeat calculation drift")
        for name in MOTION_ARRAY_NAMES:
            chunks[name].append(trajectory[name])

    corpus_arrays: dict[str, np.ndarray] = {
        FPS_ARRAY: np.asarray([FPS], dtype=np.float64),
        **{
            name: np.concatenate(chunks[name], axis=0).astype(np.float32, copy=False)
            for name in MOTION_ARRAY_NAMES
        },
    }
    _validate_corpus_arrays(corpus_arrays)
    _validate_materialized_semantics(corpus_arrays)

    temporary_corpus: Path | None = None
    temporary_sidecar: Path | None = None
    corpus_created = False
    sidecar_created = False
    try:
        temporary_corpus = _write_temporary_npz(corpus.parent, corpus_arrays)
        reopened = _load_corpus_arrays(temporary_corpus)
        _validate_corpus_arrays(reopened)
        _validate_materialized_semantics(reopened)
        _validate_against_sources(reopened, source_paths)
        if any(not np.array_equal(reopened[name], corpus_arrays[name]) for name in NPZ_ARRAY_NAMES):
            raise RuntimeError("reopened temporary corpus differs from materialized arrays")

        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "fps": FPS,
            "episode_frames": EPISODE_FRAMES,
            "diagnostic_only": True,
            "training_authorized": False,
            "corpus": {
                "path": corpus.name,
                "sha256": sha256_file(temporary_corpus),
                "frame_count": TOTAL_FRAMES,
                "arrays": _array_contract(reopened),
            },
            "sources": [
                {
                    "clip_id": pinned.clip_id,
                    "path": source_paths[pinned.clip_id].as_posix(),
                    "sha256": pinned.sha256,
                    "source_frame_count": pinned.frame_count,
                }
                for pinned in PINNED_CLIPS
            ],
            "spans": list(EXPECTED_SPANS),
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        temporary_sidecar = _write_temporary_bytes(sidecar.parent, encoded)

        _publish_new(temporary_corpus, corpus, "supported-idle corpus")
        corpus_created = True
        _publish_new(temporary_sidecar, sidecar, "supported-idle sidecar")
        sidecar_created = True

        verified = validate_supported_idle_corpus(
            corpus,
            sidecar,
            source_paths=source_paths,
        )
        if verified != payload:
            raise RuntimeError("persisted supported-idle sidecar differs from build result")
        return verified
    except BaseException:
        if sidecar_created:
            sidecar.unlink(missing_ok=True)
        if corpus_created:
            corpus.unlink(missing_ok=True)
        raise
    finally:
        if temporary_sidecar is not None:
            temporary_sidecar.unlink(missing_ok=True)
        if temporary_corpus is not None:
            temporary_corpus.unlink(missing_ok=True)
