from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_contract import ISAACLAB_TO_MUJOCO_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_pico_fullbody_motion import (
    TARGET_ANKLE_BODY_HEIGHT_M,
    build_pico_fullbody_motion,
)

ROOT = Path(__file__).resolve().parents[2]


def _bundle(clip: str, filename: str) -> dict[str, object]:
    path = ROOT / "artifacts/g1_true23/pico_saved_clip_replay_v1" / clip / filename
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("clip", "filename", "packet_count", "height_range"),
    (
        ("upright", "causal_packets_neutral_calibrated_v1.json", 41, (0.79, 0.81)),
        ("standing_1806", "causal_packets_neutral_calibrated_v1.json", 38, (0.79, 0.81)),
        ("crouch", "causal_packets_external_upright_calibration_v1.json", 62, (0.33, 0.40)),
    ),
)
def test_real_pico_clip_becomes_exact_fullbody_true23_motion(
    clip: str, filename: str, packet_count: int, height_range: tuple[float, float]
) -> None:
    bundle = _bundle(clip, filename)
    arrays, report = build_pico_fullbody_motion(
        repository_root=ROOT,
        packet_bundle=bundle,
        minimum_frames=1024,
    )
    assert arrays["joint_pos"].shape == (1024, 23)
    assert arrays["joint_vel"].shape == (1024, 23)
    assert arrays["body_pos_w"].shape == (1024, 24, 3)
    assert arrays["body_quat_w"].shape == (1024, 24, 4)
    assert arrays["body_lin_vel_w"].shape == (1024, 24, 3)
    assert arrays["body_ang_vel_w"].shape == (1024, 24, 3)
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert report["source_packet_count"] == packet_count
    assert report["source_motion_frames"] == packet_count + 10
    assert report["physical_dof"] == 23
    assert report["source_29dof_physics_used"] is False
    assert height_range[0] <= report["minimum_root_height_m"] <= height_range[1]
    assert height_range[0] <= report["maximum_root_height_m"] <= height_range[1]

    packets = bundle["robot_independent_reference_packets"]
    native_to_hardware = np.asarray(ISAACLAB_TO_MUJOCO_DOF, dtype=np.int64)
    for index, packet in enumerate(packets):
        expected = np.asarray(packet["q_ref23_native"], dtype=np.float32)[native_to_hardware]
        assert np.array_equal(arrays["joint_pos"][9 + index], expected)
    source_frames = packet_count + 10
    assert np.all(arrays["joint_pos"][source_frames:] == arrays["joint_pos"][source_frames - 1])
    assert np.count_nonzero(arrays["joint_vel"][source_frames:]) == 0
    ankle_z = arrays["body_pos_w"][:source_frames, (6, 12), 2]
    assert np.allclose(np.min(ankle_z, axis=1), TARGET_ANKLE_BODY_HEIGHT_M, atol=1.0e-6)


def test_fullbody_builder_rejects_noncontiguous_packet_sequence() -> None:
    bundle = _bundle("upright", "causal_packets_neutral_calibrated_v1.json")
    damaged = copy.deepcopy(bundle)
    damaged["robot_independent_reference_packets"][1]["pico_anchor_source_frame_index"] += 1
    damaged["robot_independent_reference_packets"][1]["control_source_frame_index"] += 1
    with pytest.raises(ValueError, match="contiguous 50 Hz"):
        build_pico_fullbody_motion(
            repository_root=ROOT,
            packet_bundle=damaged,
            minimum_frames=1024,
        )


def test_fullbody_builder_rejects_too_short_requested_storage() -> None:
    bundle = _bundle("crouch", "causal_packets_external_upright_calibration_v1.json")
    with pytest.raises(ValueError, match="cannot truncate"):
        build_pico_fullbody_motion(
            repository_root=ROOT,
            packet_bundle=bundle,
            minimum_frames=20,
        )


def test_crouch_projection_changes_only_unreachable_native23_targets() -> None:
    bundle = _bundle("crouch", "causal_packets_external_upright_calibration_v1.json")
    arrays, report = build_pico_fullbody_motion(
        repository_root=ROOT,
        packet_bundle=bundle,
        minimum_frames=72,
        reachable_raw_abs=9.5,
    )
    projection = report["reference_projection"]
    assert projection["enabled"] is True
    assert projection["source_packet_values_preserved"] is False
    assert projection["changed_frame_joint_pairs"] > 0
    assert projection["changed_frames"] > 0
    assert set(projection["changed_joint_names"]) == {
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_elbow_joint",
        "right_knee_joint",
    }
    raw = np.full(23, np.float32(9.5), dtype=np.float32)
    _, positive = safe_target_transform_numpy(raw)
    _, negative = safe_target_transform_numpy(-raw)
    assert np.all(arrays["joint_pos"] >= np.minimum(negative, positive) - 1.0e-6)
    assert np.all(arrays["joint_pos"] <= np.maximum(negative, positive) + 1.0e-6)


@pytest.mark.parametrize("value", (0.0, 10.0, 11.0))
def test_reachable_projection_rejects_invalid_raw_bound(value: float) -> None:
    bundle = _bundle("upright", "causal_packets_neutral_calibrated_v1.json")
    with pytest.raises(ValueError, match="reachable raw bound"):
        build_pico_fullbody_motion(
            repository_root=ROOT,
            packet_bundle=bundle,
            reachable_raw_abs=value,
        )
