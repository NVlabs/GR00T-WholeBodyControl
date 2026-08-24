from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import validate_reference_terms
from gear_sonic.utils.g1_true23_motion_reference_packet_bundle import (
    build_motion_reference_packet_bundle,
    write_exclusive_packet_bundle,
)


def _motion(frames: int = 14) -> dict[str, np.ndarray]:
    joint = np.zeros((frames, 23), dtype=np.float32)
    joint[:, 0] = np.arange(frames, dtype=np.float32) * 0.001
    body_pos = np.zeros((frames, 24, 3), dtype=np.float32)
    body_quat = np.zeros((frames, 24, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    return {
        "joint_pos": joint,
        "joint_vel": np.zeros_like(joint),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": np.zeros_like(body_pos),
        "body_ang_vel_w": np.zeros_like(body_pos),
    }


def test_packet_bundle_matches_causal_transport_abi() -> None:
    bundle = build_motion_reference_packet_bundle(_motion(), source_motion_sha256="a" * 64)
    packets = bundle["robot_independent_reference_packets"]
    assert len(packets) == 3
    summaries = [validate_reference_terms(packet) for packet in packets]
    assert [item["anchor_index"] for item in summaries] == [9, 10, 11]
    assert [item["control_index"] for item in summaries] == [10, 11, 12]
    assert len(bundle["semantic_packets"]) == len(packets)
    assert all(item["hardware_authorized"] is False for item in bundle["semantic_packets"])


def test_packet_bundle_rejects_out_of_range_transition_count() -> None:
    with pytest.raises(ValueError, match="causal motion range"):
        build_motion_reference_packet_bundle(_motion(), source_motion_sha256="a" * 64, transitions=4)


def test_packet_bundle_write_is_exclusive(tmp_path: Path) -> None:
    bundle = build_motion_reference_packet_bundle(_motion(), source_motion_sha256="a" * 64)
    output = tmp_path / "packets.json"
    write_exclusive_packet_bundle(output, bundle)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert len(loaded["robot_independent_reference_packets"]) == 3
    with pytest.raises(FileExistsError):
        write_exclusive_packet_bundle(output, bundle)
