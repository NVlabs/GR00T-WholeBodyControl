"""Span-safe supported-idle corpus command for the stock native124 task.

Catalog validation is pure CPU code.  MJLab imports are optional until the
runtime command or environment-config builder is used.  Vendor files and the
stock task configuration are never modified.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Literal

import numpy as np
import torch

from gear_sonic.utils.g1_true23_supported_idle_corpus import (
    EXPECTED_SPANS as PINNED_EXPECTED_SPANS,
    PINNED_CLIPS,
    validate_supported_idle_corpus,
)

SCHEMA_VERSION = 1
SIDECAR_KIND = "g1_true23_supported_idle_native_corpus_v1"
EPISODE_FRAMES = 500
FPS = 50.0
OBSERVATION_DIM = 124
ACTION_DIM = 23
SPAN_COUNT = 4
Phase = Literal["static", "trajectory"]
SpanKind = Literal["static_frame0", "trajectory"]
StartMode = Literal["start_only", "uniform_window"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_KEYS = {
    "schema_version",
    "kind",
    "episode_frames",
    "fps",
    "diagnostic_only",
    "training_authorized",
    "corpus",
    "sources",
    "spans",
}
_CORPUS_KEYS = {"path", "sha256", "frame_count", "arrays"}
_SOURCE_KEYS = {"clip_id", "path", "sha256", "source_frame_count"}
_SPAN_KEYS = {
    "id",
    "source_clip_id",
    "kind",
    "start",
    "stop",
    "stored_length",
    "original_length",
    "source_frame_count",
    "terminal_repeat_count",
    "transform",
}
_ARRAY_KEYS = {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}
_STATIC_TRANSFORM = "static_frame0_repeat_zero_velocity_v1"
_TRAJECTORY_TRANSFORM = "trajectory_terminal_hold_zero_velocity_v1"


@dataclass(frozen=True)
class CorpusArraySpec:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class SupportedIdleSource:
    clip_id: str
    path: str
    sha256: str
    source_frame_count: int


@dataclass(frozen=True)
class SupportedIdleSpan:
    id: str
    source_clip_id: str
    kind: SpanKind
    start: int
    stop: int
    stored_length: int
    original_length: int
    source_frame_count: int
    terminal_repeat_count: int
    transform: str


@dataclass(frozen=True)
class SupportedIdleCatalog:
    sidecar_path: Path
    corpus_path: Path
    corpus_sha256: str
    frame_count: int
    arrays: Mapping[str, CorpusArraySpec]
    sources: tuple[SupportedIdleSource, ...]
    spans: tuple[SupportedIdleSpan, ...]
    episode_frames: int = EPISODE_FRAMES
    fps: float = FPS

    def spans_for_phase(self, phase: Phase) -> tuple[SupportedIdleSpan, ...]:
        kind = _phase_kind(phase)
        return tuple(span for span in self.spans if span.kind == kind)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if set(value) != keys:
        raise ValueError(
            f"{name} keys mismatch: missing={sorted(keys - set(value))}, unexpected={sorted(set(value) - keys)}"
        )
    return value


def _exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if type(value) is not str or not value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must be a non-empty control-free string")
    return value


def _sha256(value: Any, name: str) -> str:
    text = _nonempty_string(value, name)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be lowercase SHA256")
    return text


def _phase_kind(phase: Phase | str) -> SpanKind:
    if phase == "static":
        return "static_frame0"
    if phase == "trajectory":
        return "trajectory"
    raise ValueError("phase must be 'static' or 'trajectory'")


def _resolve_corpus_path(sidecar: Path, value: Any) -> Path:
    text = _nonempty_string(value, "corpus.path")
    if "\\" in text:
        raise ValueError("corpus.path must use relative POSIX syntax")
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts or relative.name != text:
        raise ValueError("corpus.path must be one basename beside the sidecar")
    candidate = sidecar.parent / relative.name
    if candidate.is_symlink():
        raise ValueError("corpus must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"corpus does not exist: {candidate}") from error
    if not resolved.is_file():
        raise ValueError(f"corpus must be a regular file: {resolved}")
    return resolved


def _parse_array_specs(value: Any, frame_count: int) -> dict[str, CorpusArraySpec]:
    arrays = _mapping(value, "corpus.arrays", _ARRAY_KEYS)
    expected = {
        "fps": ((1,), "float64"),
        "joint_pos": ((frame_count, ACTION_DIM), "float32"),
        "joint_vel": ((frame_count, ACTION_DIM), "float32"),
        "body_pos_w": ((frame_count, 24, 3), "float32"),
        "body_quat_w": ((frame_count, 24, 4), "float32"),
        "body_lin_vel_w": ((frame_count, 24, 3), "float32"),
        "body_ang_vel_w": ((frame_count, 24, 3), "float32"),
    }
    result: dict[str, CorpusArraySpec] = {}
    for name, (expected_shape, expected_dtype) in expected.items():
        item = _mapping(arrays[name], f"corpus.arrays.{name}", {"shape", "dtype"})
        shape_value = item["shape"]
        if not isinstance(shape_value, list) or any(type(dim) is not int for dim in shape_value):
            raise ValueError(f"corpus.arrays.{name}.shape must be an integer list")
        shape = tuple(shape_value)
        dtype = _nonempty_string(item["dtype"], f"corpus.arrays.{name}.dtype")
        if shape != expected_shape or dtype != expected_dtype:
            raise ValueError(
                f"corpus.arrays.{name} mismatch: expected {expected_shape}/{expected_dtype}, got {shape}/{dtype}"
            )
        result[name] = CorpusArraySpec(shape=shape, dtype=dtype)
    return result


def _parse_sources(value: Any) -> tuple[SupportedIdleSource, ...]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("sources must contain exactly two entries")
    sources = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f"sources[{index}]", _SOURCE_KEYS)
        sources.append(
            SupportedIdleSource(
                clip_id=_nonempty_string(item["clip_id"], f"sources[{index}].clip_id"),
                path=_nonempty_string(item["path"], f"sources[{index}].path"),
                sha256=_sha256(item["sha256"], f"sources[{index}].sha256"),
                source_frame_count=_exact_int(
                    item["source_frame_count"],
                    f"sources[{index}].source_frame_count",
                    minimum=1,
                ),
            )
        )
    clip_ids = [source.clip_id for source in sources]
    if clip_ids != sorted(clip_ids) or len(set(clip_ids)) != 2:
        raise ValueError("sources must have two unique clip_ids in sorted order")
    for source, pinned in zip(sources, PINNED_CLIPS, strict=True):
        if (
            source.clip_id != pinned.clip_id
            or source.sha256 != pinned.sha256
            or source.source_frame_count != pinned.frame_count
        ):
            raise ValueError(f"source identity differs from pinned clip {pinned.clip_id!r}")
    return tuple(sources)


def _parse_spans(
    value: Any,
    *,
    frame_count: int,
    sources: Sequence[SupportedIdleSource],
) -> tuple[SupportedIdleSpan, ...]:
    if not isinstance(value, list) or len(value) != SPAN_COUNT:
        raise ValueError(f"spans must contain exactly {SPAN_COUNT} entries")
    source_by_id = {source.clip_id: source for source in sources}
    result = []
    cursor = 0
    for index, raw in enumerate(value):
        item = _mapping(raw, f"spans[{index}]", _SPAN_KEYS)
        kind = item["kind"]
        if kind not in ("static_frame0", "trajectory"):
            raise ValueError(f"spans[{index}].kind is unsupported: {kind!r}")
        source_clip_id = _nonempty_string(item["source_clip_id"], f"spans[{index}].source_clip_id")
        if source_clip_id not in source_by_id:
            raise ValueError(f"spans[{index}] references unknown source_clip_id")
        start = _exact_int(item["start"], f"spans[{index}].start")
        stop = _exact_int(item["stop"], f"spans[{index}].stop", minimum=1)
        stored_length = _exact_int(item["stored_length"], f"spans[{index}].stored_length", minimum=1)
        original_length = _exact_int(item["original_length"], f"spans[{index}].original_length", minimum=1)
        source_frame_count = _exact_int(
            item["source_frame_count"],
            f"spans[{index}].source_frame_count",
            minimum=1,
        )
        repeat_count = _exact_int(item["terminal_repeat_count"], f"spans[{index}].terminal_repeat_count")
        transform = _nonempty_string(item["transform"], f"spans[{index}].transform")
        span_id = _nonempty_string(item["id"], f"spans[{index}].id")
        if start != cursor or stop - start != stored_length:
            raise ValueError(f"spans[{index}] is non-contiguous or has inconsistent length")
        if stored_length < EPISODE_FRAMES:
            raise ValueError(f"spans[{index}] cannot host a {EPISODE_FRAMES}-frame episode")
        if original_length + repeat_count != stored_length:
            raise ValueError(f"spans[{index}] repeat count does not explain stored length")
        source = source_by_id[source_clip_id]
        if source_frame_count != source.source_frame_count:
            raise ValueError(f"spans[{index}] source_frame_count differs from source record")
        if span_id != f"{source_clip_id}::{kind}":
            raise ValueError(f"spans[{index}].id does not bind source and kind")
        if kind == "static_frame0":
            if original_length != 1 or transform != _STATIC_TRANSFORM:
                raise ValueError(f"spans[{index}] has invalid static transform lineage")
        elif original_length != source_frame_count or transform != _TRAJECTORY_TRANSFORM:
            raise ValueError(f"spans[{index}] has invalid trajectory transform lineage")
        result.append(
            SupportedIdleSpan(
                id=span_id,
                source_clip_id=source_clip_id,
                kind=kind,
                start=start,
                stop=stop,
                stored_length=stored_length,
                original_length=original_length,
                source_frame_count=source_frame_count,
                terminal_repeat_count=repeat_count,
                transform=transform,
            )
        )
        cursor = stop
    if cursor != frame_count:
        raise ValueError("span coverage does not equal corpus.frame_count")
    if len({span.id for span in result}) != SPAN_COUNT:
        raise ValueError("span ids must be unique")
    per_source = Counter((span.source_clip_id, span.kind) for span in result)
    expected_pairs = {(source.clip_id, kind) for source in sources for kind in ("static_frame0", "trajectory")}
    if set(per_source) != expected_pairs or any(count != 1 for count in per_source.values()):
        raise ValueError("each source must have exactly one static and one trajectory span")
    materialized = [
        {field.name: getattr(span, field.name) for field in fields(SupportedIdleSpan)} for span in result
    ]
    if materialized != list(PINNED_EXPECTED_SPANS):
        raise ValueError("spans differ from the exact pinned supported-idle layout")
    return tuple(result)


def _validate_npz(
    path: Path,
    specs: Mapping[str, CorpusArraySpec],
    spans: Sequence[SupportedIdleSpan],
) -> None:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != _ARRAY_KEYS or len(data.files) != len(_ARRAY_KEYS):
            raise ValueError("corpus NPZ array names differ from sidecar")
        arrays: dict[str, np.ndarray] = {}
        for name, spec in specs.items():
            array = np.asarray(data[name])
            if array.shape != spec.shape or array.dtype.name != spec.dtype:
                raise ValueError(f"corpus array {name!r} differs from sidecar metadata")
            if not np.isfinite(array).all():
                raise ValueError(f"corpus array {name!r} contains non-finite values")
            arrays[name] = array
        if float(arrays["fps"].reshape(-1)[0]) != FPS:
            raise ValueError(f"corpus fps must be exactly {FPS:g}")
        quaternion_norms = np.linalg.norm(arrays["body_quat_w"].astype(np.float64), axis=-1)
        if not np.allclose(quaternion_norms, 1.0, atol=1.0e-5, rtol=0.0):
            raise ValueError("corpus body_quat_w contains non-unit quaternions")
        for span in spans:
            if span.kind == "static_frame0":
                for name in ("joint_pos", "body_pos_w", "body_quat_w"):
                    values = arrays[name][span.start : span.stop]
                    if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
                        raise ValueError(f"static span {span.id!r} changes {name}")
                for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                    if np.any(arrays[name][span.start : span.stop] != 0.0):
                        raise ValueError(f"static span {span.id!r} has nonzero {name}")
            elif span.terminal_repeat_count:
                terminal = span.start + span.original_length - 1
                repeated = slice(terminal + 1, span.stop)
                for name in ("joint_pos", "body_pos_w", "body_quat_w"):
                    values = arrays[name][repeated]
                    if not np.array_equal(values, np.broadcast_to(arrays[name][terminal], values.shape)):
                        raise ValueError(f"trajectory hold {span.id!r} changes {name}")
                for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                    if np.any(arrays[name][repeated] != 0.0):
                        raise ValueError(f"trajectory hold {span.id!r} has nonzero {name}")


def load_supported_idle_catalog(
    sidecar_path: str | Path,
    *,
    expected_corpus_sha256: str,
) -> SupportedIdleCatalog:
    """Load a hash-bound supported-idle catalog and reject all schema drift."""

    caller_pinned_hash = _sha256(
        expected_corpus_sha256,
        "expected_corpus_sha256",
    )
    candidate = Path(sidecar_path).expanduser()
    if candidate.is_symlink():
        raise ValueError("sidecar must not be a symlink")
    try:
        sidecar = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"sidecar does not exist: {candidate}") from error
    if not sidecar.is_file():
        raise ValueError("sidecar must be a regular file")
    try:
        root = json.loads(sidecar.read_text(encoding="utf-8"), object_pairs_hook=_json_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sidecar must be strict UTF-8 JSON") from error
    data = _mapping(root, "sidecar", _TOP_KEYS)
    if data["schema_version"] != SCHEMA_VERSION or type(data["schema_version"]) is not int:
        raise ValueError(f"schema_version must be exact integer {SCHEMA_VERSION}")
    if data["kind"] != SIDECAR_KIND:
        raise ValueError(f"kind must be {SIDECAR_KIND!r}")
    if data["episode_frames"] != EPISODE_FRAMES or type(data["episode_frames"]) is not int:
        raise ValueError(f"episode_frames must be exact integer {EPISODE_FRAMES}")
    if type(data["fps"]) not in (int, float) or isinstance(data["fps"], bool) or float(data["fps"]) != FPS:
        raise ValueError(f"fps must be exactly {FPS:g}")
    if data["diagnostic_only"] is not True or data["training_authorized"] is not False:
        raise ValueError("sidecar must be diagnostic_only and training_authorized=false")

    corpus = _mapping(data["corpus"], "corpus", _CORPUS_KEYS)
    frame_count = _exact_int(corpus["frame_count"], "corpus.frame_count", minimum=1)
    specs = _parse_array_specs(corpus["arrays"], frame_count)
    sources = _parse_sources(data["sources"])
    spans = _parse_spans(data["spans"], frame_count=frame_count, sources=sources)
    corpus_path = _resolve_corpus_path(sidecar, corpus["path"])
    expected_hash = _sha256(corpus["sha256"], "corpus.sha256")
    if expected_hash != caller_pinned_hash:
        raise ValueError("sidecar corpus SHA256 differs from caller-pinned expected_corpus_sha256")
    actual_hash = _sha256_file(corpus_path)
    if actual_hash != caller_pinned_hash:
        raise ValueError(f"corpus SHA256 mismatch: expected {caller_pinned_hash}, got {actual_hash}")
    _validate_npz(corpus_path, specs, spans)
    validate_supported_idle_corpus(corpus_path, sidecar)
    if _sha256_file(corpus_path) != actual_hash:
        raise RuntimeError("corpus changed while catalog was being validated")
    return SupportedIdleCatalog(
        sidecar_path=sidecar,
        corpus_path=corpus_path,
        corpus_sha256=actual_hash,
        frame_count=frame_count,
        arrays=specs,
        sources=sources,
        spans=spans,
    )


def balanced_span_indices(
    catalog: SupportedIdleCatalog,
    phase: Phase,
    count: int,
    *,
    selection_offset: int = 0,
) -> tuple[int, ...]:
    """Return deterministic round-robin span indices balanced by source."""

    count = _exact_int(count, "count")
    selection_offset = _exact_int(selection_offset, "selection_offset")
    kind = _phase_kind(phase)
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, span in enumerate(catalog.spans):
        if span.kind == kind:
            by_source[span.source_clip_id].append(index)
    source_ids = [source.clip_id for source in catalog.sources if source.clip_id in by_source]
    if not source_ids:
        raise ValueError(f"catalog contains no spans for phase {phase!r}")
    result = []
    for position in range(selection_offset, selection_offset + count):
        source_index = position % len(source_ids)
        source_round = position // len(source_ids)
        choices = by_source[source_ids[source_index]]
        result.append(choices[source_round % len(choices)])
    return tuple(result)


def span_window_start_bounds(
    span: SupportedIdleSpan,
    *,
    episode_frames: int = EPISODE_FRAMES,
) -> tuple[int, int]:
    """Return inclusive valid window-start bounds for one span."""

    episode_frames = _exact_int(episode_frames, "episode_frames", minimum=1)
    upper = span.stop - episode_frames
    if upper < span.start:
        raise ValueError("span cannot contain the requested episode")
    return span.start, upper


def advance_span_time_steps(
    time_steps: torch.Tensor,
    window_stops: torch.Tensor,
    episode_length_buf: torch.Tensor,
) -> torch.Tensor:
    """Advance active envs once, while reset-at-zero envs remain on frame zero."""

    for tensor, name in (
        (time_steps, "time_steps"),
        (window_stops, "window_stops"),
        (episode_length_buf, "episode_length_buf"),
    ):
        if type(tensor) is not torch.Tensor or tensor.dtype != torch.long or tensor.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional torch.long tensor")
    if time_steps.shape != window_stops.shape or time_steps.shape != episode_length_buf.shape:
        raise ValueError("span time tensors must have identical shapes")
    if not (time_steps.device == window_stops.device == episode_length_buf.device):
        raise ValueError("span time tensors must share one device")
    if bool((episode_length_buf < 0).any()):
        raise ValueError("episode_length_buf must be non-negative")
    if bool((time_steps < 0).any()) or bool((time_steps >= window_stops).any()):
        raise RuntimeError("current reference frame is outside its assigned episode window")
    result = time_steps + (episode_length_buf != 0).to(dtype=torch.long)
    if bool((result >= window_stops).any()):
        raise RuntimeError("reference advance would cross its assigned episode window")
    return result


_MJLAB_IMPORT_ERROR: Exception | None = None
try:
    from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
except (ImportError, ModuleNotFoundError) as error:
    _MJLAB_IMPORT_ERROR = error
    MotionCommand = object  # type: ignore[assignment,misc]
    MotionCommandCfg = object  # type: ignore[assignment,misc]


if _MJLAB_IMPORT_ERROR is None:

    @dataclass(kw_only=True)
    class SupportedIdleMotionCommandCfg(MotionCommandCfg):
        """Stock motion command constrained to one supported-idle span window."""

        sidecar_path: str
        expected_corpus_sha256: str
        phase: Phase
        episode_frames: int = EPISODE_FRAMES
        start_mode: StartMode = "start_only"
        sampling_mode: Literal["uniform"] = "uniform"

        def build(self, env: Any) -> "SupportedIdleMotionCommand":
            return SupportedIdleMotionCommand(self, env)

    class SupportedIdleMotionCommand(MotionCommand):
        """Stock loader/reset/cache behavior with span-contained frame sampling."""

        cfg: SupportedIdleMotionCommandCfg

        def __init__(self, cfg: SupportedIdleMotionCommandCfg, env: Any) -> None:
            catalog = load_supported_idle_catalog(
                cfg.sidecar_path,
                expected_corpus_sha256=cfg.expected_corpus_sha256,
            )
            if cfg.episode_frames != EPISODE_FRAMES:
                raise ValueError(f"episode_frames must be {EPISODE_FRAMES}")
            _phase_kind(cfg.phase)
            if cfg.start_mode not in ("start_only", "uniform_window"):
                raise ValueError("start_mode must be 'start_only' or 'uniform_window'")
            if cfg.sampling_mode != "uniform":
                raise ValueError("supported-idle command reserves stock uniform sampling hook")
            if min(cfg.resampling_time_range) <= EPISODE_FRAMES / FPS:
                raise ValueError("command timer must not resample inside a 500-step episode")
            motion_path = Path(cfg.motion_file).expanduser().resolve(strict=True)
            if motion_path != catalog.corpus_path:
                raise ValueError("motion_file does not match the sidecar-bound corpus")
            if not math.isclose(float(env.step_dt), 1.0 / FPS, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("supported-idle command requires exact 50 Hz control")
            if int(env.max_episode_length) != EPISODE_FRAMES:
                raise ValueError("supported-idle command requires a 500-step episode")

            self.catalog = catalog
            self._balanced_selection_offset = 0
            super().__init__(cfg, env)
            if _sha256_file(catalog.corpus_path) != catalog.corpus_sha256:
                raise RuntimeError("corpus changed while stock MotionLoader was loading it")
            if self.motion.time_step_total != catalog.frame_count:
                raise RuntimeError("MotionLoader frame count differs from validated catalog")
            if tuple(self.motion.joint_pos.shape) != (catalog.frame_count, ACTION_DIM):
                raise RuntimeError("MotionLoader joint contract is not native 23-DoF")
            self.span_starts = torch.full_like(self.time_steps, -1)
            self.span_stops = torch.full_like(self.time_steps, -1)
            self.span_ids = torch.full_like(self.time_steps, -1)
            self.episode_starts = torch.full_like(self.time_steps, -1)
            self.episode_stops = torch.full_like(self.time_steps, -1)
            self._span_resampled = torch.zeros_like(self.time_steps, dtype=torch.bool)
            all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            self._assign_span_windows(all_env_ids, sample_windows=False, advance_balance=False)

        def _assign_span_windows(
            self,
            env_ids: torch.Tensor,
            *,
            sample_windows: bool,
            advance_balance: bool,
        ) -> None:
            indices = balanced_span_indices(
                self.catalog,
                self.cfg.phase,
                len(env_ids),
                selection_offset=self._balanced_selection_offset,
            )
            if advance_balance:
                self._balanced_selection_offset += len(env_ids)
            selected = [self.catalog.spans[index] for index in indices]
            span_starts = torch.tensor([span.start for span in selected], dtype=torch.long, device=self.device)
            span_stops = torch.tensor([span.stop for span in selected], dtype=torch.long, device=self.device)
            if sample_windows and self.cfg.start_mode == "uniform_window":
                max_offsets = span_stops - span_starts - self.cfg.episode_frames
                random_values = torch.rand(len(env_ids), dtype=torch.float32, device=self.device)
                offsets = torch.floor(random_values * (max_offsets + 1).float()).long()
                offsets = torch.minimum(offsets, max_offsets)
            else:
                offsets = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
            starts = span_starts + offsets
            stops = starts + self.cfg.episode_frames
            if bool((starts < span_starts).any()) or bool((stops > span_stops).any()):
                raise RuntimeError("sampled episode window escapes its assigned span")
            self.span_ids[env_ids] = torch.tensor(indices, dtype=torch.long, device=self.device)
            self.span_starts[env_ids] = span_starts
            self.span_stops[env_ids] = span_stops
            self.episode_starts[env_ids] = starts
            self.episode_stops[env_ids] = stops
            self.time_steps[env_ids] = starts

        def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
            """Hook called by the unchanged stock reset/write implementation."""

            self._assign_span_windows(env_ids, sample_windows=True, advance_balance=True)
            self._span_resampled[env_ids] = self._env.episode_length_buf[env_ids] != 0
            source_count = len(self.catalog.sources)
            self.metrics["sampling_entropy"][env_ids] = 1.0 if source_count > 1 else 0.0
            self.metrics["sampling_top1_prob"][env_ids] = 1.0 / source_count
            self.metrics["sampling_top1_bin"][env_ids] = self.span_ids[env_ids].float() / max(
                len(self.catalog.spans) - 1, 1
            )

        def _update_command(self) -> None:
            skip_advance = (self._env.episode_length_buf == 0) | self._span_resampled
            effective_episode_length = self._env.episode_length_buf.clone()
            effective_episode_length[skip_advance] = 0
            expected = advance_span_time_steps(
                self.time_steps,
                self.episode_stops,
                effective_episode_length,
            )
            self.time_steps[skip_advance] -= 1
            super()._update_command()
            if not torch.equal(self.time_steps, expected):
                raise RuntimeError("stock MotionCommand changed span-safe time-step semantics")
            self._span_resampled.zero_()

else:

    class SupportedIdleMotionCommandCfg:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR

    class SupportedIdleMotionCommand:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("MJLab 1.2 is required") from _MJLAB_IMPORT_ERROR


def _require_mjlab() -> None:
    if _MJLAB_IMPORT_ERROR is not None:
        raise RuntimeError("MJLab 1.2 and Unitree tracking sources are required") from _MJLAB_IMPORT_ERROR


def _assert_native_config_contract(cfg: Any) -> None:
    if not math.isclose(float(cfg.sim.mujoco.timestep), 0.005, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("stock native task physics timestep drifted")
    if cfg.decimation != 4 or not math.isclose(cfg.episode_length_s, 10.0):
        raise RuntimeError("supported-idle task requires 10 seconds at 50 Hz")
    actor_terms = tuple(cfg.observations["actor"].terms)
    expected_terms = (
        "command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    )
    if actor_terms != expected_terms or sum((46, 6, 3, 23, 23, 23)) != OBSERVATION_DIM:
        raise RuntimeError("stock no-state-estimation actor is not the native124 contract")
    if tuple(cfg.actions) != ("joint_pos",):
        raise RuntimeError("stock task action topology drifted from one native23 term")


def make_supported_idle_native_env_cfg(
    *,
    sidecar_path: str | Path,
    expected_corpus_sha256: str,
    phase: Phase,
    start_mode: StartMode = "start_only",
) -> Any:
    """Copy the stock 124-to-23 task and replace only its motion command."""

    catalog = load_supported_idle_catalog(
        sidecar_path,
        expected_corpus_sha256=expected_corpus_sha256,
    )
    _phase_kind(phase)
    if start_mode not in ("start_only", "uniform_window"):
        raise ValueError("start_mode must be 'start_only' or 'uniform_window'")
    _require_mjlab()
    from src.tasks.tracking.config.g1_23dof.env_cfgs import (
        unitree_g1_23dof_flat_tracking_env_cfg,
    )

    cfg = unitree_g1_23dof_flat_tracking_env_cfg(has_state_estimation=False)
    _assert_native_config_contract(cfg)
    base_command = cfg.commands["motion"]
    if not isinstance(base_command, MotionCommandCfg):
        raise RuntimeError("stock task motion command type drifted")
    command_kwargs = {field.name: getattr(base_command, field.name) for field in fields(base_command)}
    command_kwargs.update(
        {
            "episode_frames": EPISODE_FRAMES,
            "expected_corpus_sha256": catalog.corpus_sha256,
            "motion_file": str(catalog.corpus_path),
            "phase": phase,
            "sampling_mode": "uniform",
            "sidecar_path": str(catalog.sidecar_path),
            "start_mode": start_mode,
        }
    )
    cfg.commands["motion"] = SupportedIdleMotionCommandCfg(**command_kwargs)
    _assert_native_config_contract(cfg)
    return cfg


__all__ = [
    "ACTION_DIM",
    "EPISODE_FRAMES",
    "FPS",
    "OBSERVATION_DIM",
    "SIDECAR_KIND",
    "SCHEMA_VERSION",
    "CorpusArraySpec",
    "SupportedIdleCatalog",
    "SupportedIdleMotionCommand",
    "SupportedIdleMotionCommandCfg",
    "SupportedIdleSource",
    "SupportedIdleSpan",
    "advance_span_time_steps",
    "balanced_span_indices",
    "load_supported_idle_catalog",
    "make_supported_idle_native_env_cfg",
    "span_window_start_bounds",
]
