from __future__ import annotations

import math
from pathlib import Path

import pytest

from gear_sonic.tests.test_g1_23dof_pico_retargeted_producer import _capture
from gear_sonic.utils import g1_23dof_xr24_soma_adapter as adapter
from gear_sonic.utils.g1_23dof_contract import (
    REFERENCE_PROFILE_NORMAL,
    SOURCE_IL29_JOINT_NAMES,
)
from gear_sonic.utils.g1_23dof_pico_retargeted_producer import (
    RETARGET_REPLAY_STATUS_EXACT_NONPROMOTABLE,
    SOMA_MJ29_TO_CANONICAL_IL29,
    validate_retarget_trace_contract,
)
from gear_sonic.utils.g1_23dof_semantic_reference import (
    SOURCE_SAMPLE_PERIOD_NS,
    build_stream_reference_window,
)


def test_coordinate_adapter_rotates_xrobot_openxr_world_without_role_fill() -> None:
    result = adapter.pico_unity_pose_to_soma(
        22,
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
    )
    assert result == [-1.0, 2.0, -3.0, 0.0, 1.0, 0.0, 0.0]

    pelvis = adapter.pico_unity_pose_to_soma(
        0,
        [0.25, 1.0, -0.5, 0.0, 0.0, 0.0, 1.0],
    )
    assert pelvis[:3] == [-0.25, 1.0, 0.5]
    assert math.isclose(abs(pelvis[5]), 0.0, abs_tol=1.0e-12)
    assert math.isclose(abs(pelvis[4]), 1.0, abs_tol=1.0e-12)


def test_neutral_calibration_maps_role_frames_and_pelvis_origin() -> None:
    class Skeleton:
        joint_names = [name for _, name in adapter.XR24_ROLE_TO_SOMA_JOINT]
        num_joints = len(joint_names)

        def joint_index(self, name: str) -> int:
            return self.joint_names.index(name)

    skeleton = Skeleton()
    target = [[float(index), 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] for index in range(skeleton.num_joints)]
    neutral = _neutral_standing_poses()
    calibration = adapter.build_xr24_soma_neutral_calibration(
        [neutral for _ in range(10)],
        skeleton=skeleton,
        target_neutral_global=target,
    )
    pelvis = adapter.apply_xr24_soma_neutral_calibration(
        0,
        neutral[0],
        calibration,
    )
    assert pelvis[:3] == pytest.approx(target[0][:3])
    assert pelvis[3:] == pytest.approx(target[0][3:])
    for role_index, soma_name in adapter.XR24_ROLE_TO_SOMA_JOINT:
        mapped = adapter.apply_xr24_soma_neutral_calibration(
            role_index,
            neutral[role_index],
            calibration,
        )
        assert mapped[3:] == pytest.approx(target[skeleton.joint_index(soma_name)][3:])
    assert len(calibration["calibration_sha256"]) == 64


def _neutral_standing_poses() -> list[list[float]]:
    poses = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(24)]
    poses[0][:3] = [0.0, 0.95, 0.0]
    poses[1][:3] = [-0.10, 0.90, 0.0]
    poses[2][:3] = [0.10, 0.90, 0.0]
    poses[4][:3] = [-0.10, 0.50, 0.0]
    poses[5][:3] = [0.10, 0.50, 0.0]
    poses[7][:3] = [-0.10, 0.08, 0.0]
    poses[8][:3] = [0.10, 0.08, 0.0]
    poses[10][:3] = [-0.10, 0.02, 0.10]
    poses[11][:3] = [0.10, 0.02, 0.10]
    poses[15][:3] = [0.0, 1.55, 0.0]
    return poses


def test_neutral_standing_gate_accepts_extended_legs() -> None:
    report = adapter.assess_xr24_neutral_standing(_neutral_standing_poses())
    assert report["pass"] is True
    assert report["metrics"]["left_knee_angle_deg"] == pytest.approx(180.0)


def test_neutral_standing_gate_rejects_seated_acquisition() -> None:
    poses = _neutral_standing_poses()
    poses[0][1] = 0.45
    poses[1][:3] = [-0.10, 0.42, 0.0]
    poses[2][:3] = [0.10, 0.42, 0.0]
    poses[4][:3] = [-0.10, 0.50, 0.35]
    poses[5][:3] = [0.10, 0.50, 0.35]
    report = adapter.assess_xr24_neutral_standing(poses)
    assert report["pass"] is False
    assert report["checks"]["pelvis_above_knees_m"] is False
    with pytest.raises(ValueError, match="not neutral standing"):
        adapter.require_xr24_neutral_standing(poses)


def test_exact_50hz_resample_has_real_contiguous_raw_brackets() -> None:
    capture = _capture(49)
    samples = adapter.resample_raw_capture_50hz(capture)

    assert len(samples) == 49
    assert samples[0]["raw_bracket_indices"] == [0, 1]
    assert samples[0]["raw_interpolation_alpha"] == 0.0
    assert samples[-1]["raw_bracket_indices"] == [47, 48]
    assert samples[-1]["raw_interpolation_alpha"] == 1.0
    for previous, current in zip(samples[:-1], samples[1:], strict=True):
        assert current["source_frame_index"] == previous["source_frame_index"] + 1
        assert current["reference_monotonic_ns"] == previous["reference_monotonic_ns"] + SOURCE_SAMPLE_PERIOD_NS
        left, right = current["raw_bracket_indices"]
        assert right == left + 1


def _synthetic_mj29(sample_count: int) -> list[list[float]]:
    rows = []
    for frame_index in range(sample_count):
        il29 = [0.0] * 29
        for joint_index in range(12):
            il29[joint_index] = 0.1 + frame_index * (joint_index + 1) * 1.0e-4
        rows.append([il29[target] for target in SOMA_MJ29_TO_CANONICAL_IL29])
    return rows


def test_exact_backend_trace_state_stays_nonpromotable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _capture(49)
    timing = adapter.resample_raw_capture_50hz(capture)
    rows = _synthetic_mj29(len(timing))

    monkeypatch.setattr(
        adapter,
        "_execute_soma",
        lambda *_args, **_kwargs: (
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, *row] for row in rows],
            timing,
            object(),
        ),
    )
    trace = adapter.build_retarget_trace(
        capture,
        soma_source_root=Path("unused"),
        profile=REFERENCE_PROFILE_NORMAL,
    )
    summary = validate_retarget_trace_contract(
        capture,
        trace,
        profile=REFERENCE_PROFILE_NORMAL,
    )

    assert trace["certification"] == {
        "exact_backend_replay_performed": True,
        "raw_capture_replayed": True,
        "full_il29_verified": True,
        "promotion_eligible": False,
        "status": RETARGET_REPLAY_STATUS_EXACT_NONPROMOTABLE,
    }
    assert summary["exact_backend_replay_performed"] is True
    assert summary["promotion_eligible"] is False
    assert trace["joint_order_il29"] == list(SOURCE_IL29_JOINT_NAMES)

    frames = adapter.semantic_frames_from_trace(trace)
    assert len(frames) == 48
    assert all(frame["complete_joint_mask_il29"] == [True] * 29 for frame in frames)
    assert any(abs(frame["joint_vel_il29"][0]) > 0.0 for frame in frames)
    window = build_stream_reference_window(
        frames,
        profile=REFERENCE_PROFILE_NORMAL,
        emitted_monotonic_ns=frames[-1]["capture_monotonic_ns"],
    )
    assert len(window["command_multi_future_lower_body"]) == 240


def test_full_encoder_terms_use_measured_robot_orientation() -> None:
    frames = adapter.semantic_frames_from_trace(
        adapter._build_retarget_trace_from_rows(  # noqa: SLF001
            _capture(49),
            mj29_rows=_synthetic_mj29(49),
            resampled=adapter.resample_raw_capture_50hz(_capture(49)),
            profile=REFERENCE_PROFILE_NORMAL,
        )
    )
    window = build_stream_reference_window(
        frames,
        profile=REFERENCE_PROFILE_NORMAL,
        emitted_monotonic_ns=frames[-1]["capture_monotonic_ns"],
    )
    body_term = {
        "source_frame_index": window["playback"]["frame_index"],
        "reference_monotonic_ns": window["playback"]["frame_monotonic_ns"],
        "vr_3point_local_target": [0.1] * 9,
        "vr_3point_local_orn_target": [0.0, 0.0, 0.0, 1.0] * 3,
        "reference_anchor_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    terms = adapter.complete_encoder_terms(
        reference_window=window,
        body_term=body_term,
        robot_anchor_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
    )

    assert len(terms["command_multi_future_lower_body"]) == 240
    assert len(terms["vr_3point_local_target"]) == 9
    assert len(terms["vr_3point_local_orn_target"]) == 12
    assert terms["motion_anchor_ori_b"] == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
