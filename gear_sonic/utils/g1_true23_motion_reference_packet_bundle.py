"""Build offline causal true23 reference packets from a validated motion.

This bridges a known motion into the same saved/ZMQ packet ABI used by PICO
teleoperation.  It never opens DDS, hardware, network, or robot channels.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    motion_reference_terms,
    validate_reference_terms,
)

REQUIRED_MOTION_ARRAYS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_motion_reference_packet_bundle(
    motion: Mapping[str, np.ndarray],
    *,
    source_motion_sha256: str,
    transitions: int | None = None,
) -> dict[str, Any]:
    missing = REQUIRED_MOTION_ARRAYS - set(motion)
    if missing:
        raise ValueError(f"motion arrays missing: {sorted(missing)}")
    frame_count = int(np.asarray(motion["joint_pos"]).shape[0])
    if frame_count < 12:
        raise ValueError("motion needs at least 12 frames")
    if not isinstance(source_motion_sha256, str) or len(source_motion_sha256) != 64:
        raise ValueError("source motion SHA256 mismatch")
    maximum = frame_count - 11
    count = maximum if transitions is None else transitions
    if type(count) is not int or not 1 <= count <= maximum:
        raise ValueError("transition count exceeds causal motion range")

    packets = [motion_reference_terms(motion, 9 + transition) for transition in range(count)]
    summaries = [validate_reference_terms(packet) for packet in packets]
    for previous, current in zip(summaries, summaries[1:], strict=False):
        if (
            current["control_index"] != previous["control_index"] + 1
            or current["control_monotonic_ns"] != previous["control_monotonic_ns"] + 20_000_000
        ):
            raise ValueError("generated reference packets are not contiguous")

    semantic = [
        {
            "schema_version": 1,
            "kind": "g1_true23_motion_derived_pico_transport_probe_v1",
            "source_motion_sha256": source_motion_sha256,
            "anchor_source_frame_index": summary["anchor_index"],
            "raw_live_pico_packet": False,
            "live_headset_source_proven": False,
            "hardware_authorized": False,
        }
        for summary in summaries
    ]
    return {
        "robot_independent_reference_packets": packets,
        "semantic_packets": semantic,
    }


def load_motion_reference_packet_bundle(
    motion_path: Path,
    *,
    transitions: int | None = None,
) -> dict[str, Any]:
    resolved = motion_path.resolve(strict=True)
    with np.load(resolved, allow_pickle=False) as archive:
        motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    return build_motion_reference_packet_bundle(
        motion,
        source_motion_sha256=sha256_file(resolved),
        transitions=transitions,
    )


def write_exclusive_packet_bundle(path: Path, bundle: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(bundle, stream, separators=(",", ":"), sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


__all__ = (
    "build_motion_reference_packet_bundle",
    "load_motion_reference_packet_bundle",
    "sha256_file",
    "write_exclusive_packet_bundle",
)
