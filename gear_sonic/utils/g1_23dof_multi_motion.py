"""Sample across a corpus of mjlab motion clips instead of a single file.

The stock ``MotionLoader`` holds one clip as flat time-indexed tensors, so the
causal trainer can only ever see the motion passed as ``--motion-file``. A
policy trained that way tracks one clip; teleoperation needs coverage of many.

This module deliberately does not modify ``unitree_rl_mjlab``. It builds a
corpus view over many converted clips and exposes the same attributes the
stock loader does, so it can stand in without touching shared code that other
training runs may be using.

Clip boundaries matter: an episode that runs off the end of one clip and into
the next would present a discontinuous reference the policy cannot track. The
corpus therefore keeps per-clip offsets and clamps every sampled window to the
clip it started in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

MOTION_ARRAY_NAMES = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class ClipSpan:
    """Where one clip lives inside the concatenated corpus."""

    name: str
    start: int
    length: int

    @property
    def stop(self) -> int:
        return self.start + self.length


class MultiMotionCorpus:
    """Concatenated clips plus the bookkeeping to keep episodes inside one clip."""

    def __init__(
        self,
        motion_files: Sequence[Path],
        body_indexes: torch.Tensor,
        *,
        device: str = "cpu",
        min_frames: int = 1,
        expected_fps: float | None = 50.0,
    ) -> None:
        if not motion_files:
            raise ValueError("a motion corpus needs at least one clip")

        chunks: dict[str, list[np.ndarray]] = {n: [] for n in MOTION_ARRAY_NAMES}
        spans: list[ClipSpan] = []
        cursor = 0
        skipped: list[str] = []

        for path in motion_files:
            with np.load(path) as data:
                missing = [n for n in MOTION_ARRAY_NAMES if n not in data]
                if missing:
                    skipped.append(f"{path.name}: missing {missing}")
                    continue
                frames = int(data["joint_pos"].shape[0])
                if frames < min_frames:
                    skipped.append(f"{path.name}: {frames} frames")
                    continue
                if expected_fps is not None and "fps" in data:
                    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                    if abs(fps - expected_fps) > 1e-6:
                        skipped.append(f"{path.name}: fps {fps}")
                        continue
                for name in MOTION_ARRAY_NAMES:
                    chunks[name].append(np.asarray(data[name]))
            spans.append(ClipSpan(name=path.stem, start=cursor, length=frames))
            cursor += frames

        if not spans:
            raise ValueError("no usable clips in the corpus")

        joined = {
            name: torch.tensor(
                np.concatenate(values, axis=0), dtype=torch.float32, device=device
            )
            for name, values in chunks.items()
        }

        self.joint_pos = joined["joint_pos"]
        self.joint_vel = joined["joint_vel"]
        self._body_pos_w = joined["body_pos_w"]
        self._body_quat_w = joined["body_quat_w"]
        self._body_lin_vel_w = joined["body_lin_vel_w"]
        self._body_ang_vel_w = joined["body_ang_vel_w"]

        self._body_indexes = body_indexes
        self.body_pos_w = self._body_pos_w[:, body_indexes]
        self.body_quat_w = self._body_quat_w[:, body_indexes]
        self.body_lin_vel_w = self._body_lin_vel_w[:, body_indexes]
        self.body_ang_vel_w = self._body_ang_vel_w[:, body_indexes]

        self.time_step_total = int(self.joint_pos.shape[0])
        self.spans = tuple(spans)
        self.skipped = tuple(skipped)

        starts = torch.tensor([s.start for s in spans], dtype=torch.long, device=device)
        stops = torch.tensor([s.stop for s in spans], dtype=torch.long, device=device)
        self._clip_starts = starts
        self._clip_stops = stops
        # Frame -> clip index, so a sampled time step can be clamped to its clip.
        owner = torch.zeros(self.time_step_total, dtype=torch.long, device=device)
        for index, span in enumerate(spans):
            owner[span.start : span.stop] = index
        self._frame_owner = owner

    def __len__(self) -> int:
        return len(self.spans)

    def sample_start_indices(
        self, count: int, *, episode_length: int, generator=None
    ) -> torch.Tensor:
        """Random start frames whose full episode fits inside a single clip."""
        eligible = [
            span for span in self.spans if span.length >= episode_length
        ]
        if not eligible:
            raise ValueError(
                f"no clip is at least {episode_length} frames; "
                f"longest is {max(s.length for s in self.spans)}"
            )
        device = self.joint_pos.device
        choice = torch.randint(
            len(eligible), (count,), generator=generator, device=device
        )
        starts = torch.tensor(
            [s.start for s in eligible], dtype=torch.long, device=device
        )[choice]
        room = torch.tensor(
            [s.length - episode_length for s in eligible],
            dtype=torch.long,
            device=device,
        )[choice]
        jitter = (
            torch.rand(count, generator=generator, device=device) * (room + 1)
        ).long()
        return starts + torch.minimum(jitter, room)

    def clamp_to_clip(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Clamp frame indices so none crosses out of the clip it belongs to."""
        flat = time_steps.reshape(-1).clamp(0, self.time_step_total - 1)
        owner = self._frame_owner[flat]
        lo = self._clip_starts[owner]
        hi = self._clip_stops[owner] - 1
        return torch.minimum(torch.maximum(flat, lo), hi).reshape(time_steps.shape)


def load_corpus(
    motion_dir: Path,
    body_indexes: torch.Tensor,
    *,
    device: str = "cpu",
    limit: int | None = None,
    min_frames: int = 1,
) -> MultiMotionCorpus:
    paths = sorted(motion_dir.glob("*.npz"))
    if limit:
        paths = paths[:limit]
    return MultiMotionCorpus(
        paths, body_indexes, device=device, min_frames=min_frames
    )
