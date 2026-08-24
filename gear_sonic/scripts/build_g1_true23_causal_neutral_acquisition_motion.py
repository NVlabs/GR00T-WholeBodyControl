"""Extract the FK-valid neutral segment for causal stand acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_SOURCE = Path(
    "/root/.cache/g1_true23_mjlab/recovery/"
    "g1_true23_low_latency_stand_transition_dance_v2.npz"
)
DEFAULT_SOURCE_METADATA = DEFAULT_SOURCE.with_suffix(".json")
DEFAULT_OUTPUT = Path(
    "/root/.cache/g1_true23_mjlab/recovery/"
    "g1_true23_causal_neutral_acquisition_v3.npz"
)
NEUTRAL_FRAME_COUNT = 600


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source: Path, source_metadata: Path, output: Path) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    source_metadata = source_metadata.expanduser().resolve()
    output = output.expanduser().resolve()
    metadata_output = output.with_suffix(".json")
    if output.exists() or metadata_output.exists():
        raise FileExistsError("refusing to overwrite neutral acquisition motion")
    parent_metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    if parent_metadata["output"]["sha256"] != _sha256(source):
        raise ValueError("parent recovery motion hash mismatch")

    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if float(arrays["fps"].reshape(-1)[0]) != 50.0:
        raise ValueError("neutral acquisition source must be 50 Hz")
    frame_fields = tuple(name for name in arrays if name != "fps")
    if any(arrays[name].shape[0] < NEUTRAL_FRAME_COUNT for name in frame_fields):
        raise ValueError("neutral acquisition source is shorter than 600 frames")
    neutral = {"fps": arrays["fps"].copy()}
    neutral.update(
        {name: arrays[name][:NEUTRAL_FRAME_COUNT].copy() for name in frame_fields}
    )
    if any(not np.isfinite(value).all() for value in neutral.values()):
        raise ValueError("neutral acquisition motion contains NaN or Inf")
    for name in ("joint_pos", "body_pos_w", "body_quat_w"):
        if not np.allclose(neutral[name][0], neutral[name][-1], atol=1.0e-6):
            raise ValueError(f"neutral acquisition {name} is not loop-closed")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **neutral)
    metadata = {
        "schema": "g1_true23_low_latency_recovery_motion_v1",
        "simulator_only": True,
        "deployment_ready": False,
        "acquisition": {
            "schema": "g1_true23_causal_neutral_acquisition_motion_v3",
            "role": "neutral_stand_only",
            "parent_filename": source.name,
            "parent_sha256": _sha256(source),
            "source_frames_inclusive": [0, NEUTRAL_FRAME_COUNT - 1],
            "transitions_present": False,
            "dance_present": False,
        },
        "output": {
            "filename": output.name,
            "sha256": _sha256(output),
            "fps": 50.0,
            "frames": NEUTRAL_FRAME_COUNT,
            "duration_s": NEUTRAL_FRAME_COUNT / 50.0,
        },
        "segments_inclusive": {
            "neutral_only": [0, NEUTRAL_FRAME_COUNT - 1]
        },
        "kinematics": parent_metadata["kinematics"],
        "model": parent_metadata["model"],
        "source": {
            "filename": source.name,
            "sha256": _sha256(source),
            "frames": int(arrays["joint_pos"].shape[0]),
        },
        "safe_joint_projection": {
            "inherited_from_parent": True,
            "inner_margin_fraction_of_soft_span": 0.025,
            "soft_limit_factor": 0.9,
        },
    }
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, metadata_output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output, metadata = build(args.source, args.source_metadata, args.output)
    print(output)  # noqa: T201
    print(metadata)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
