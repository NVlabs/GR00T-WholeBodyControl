from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from gear_sonic.utils import g1_23dof_semantic_reference as semantic

REPO_ROOT = Path(__file__).resolve().parents[2]
MOTION_DIR = (
    REPO_ROOT
    / "gear_sonic_deploy"
    / "reference"
    / "example"
    / "squat_001__A359"
)


def _load_csv(path: Path) -> list[list[float]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        return [[float(value) for value in row] for row in reader]


def _recorded_window(profile: str = semantic.PROFILE_TRUE23_STEP5) -> dict:
    epoch_ns = 1_000_000_000_000
    frame_index = 10
    frame_ns = epoch_ns + frame_index * semantic.SOURCE_SAMPLE_PERIOD_NS
    return semantic.build_recorded_reference_window(
        MOTION_DIR,
        profile=profile,
        playback_frame_index=frame_index,
        playback_epoch_monotonic_ns=epoch_ns,
        emitted_monotonic_ns=frame_ns + 5_000_000,
    )


def _stream_frames(
    *,
    source_kind: str = semantic.SOURCE_RETARGETED_DELAYED,
    frame_count: int = 47,
) -> list[dict]:
    start_index = 100
    start_ns = 10_000_000_000
    frames = []
    for offset in range(frame_count):
        frame_index = start_index + offset
        reference_ns = start_ns + offset * semantic.SOURCE_SAMPLE_PERIOD_NS
        positions = [
            offset * 0.001 + joint_index * 0.01
            for joint_index in range(semantic.SOURCE_DOF)
        ]
        frames.append(
            {
                "schema_version": semantic.SEMANTIC_REFERENCE_SCHEMA_VERSION,
                "kind": semantic.SEMANTIC_REFERENCE_FRAME_KIND,
                "source_kind": source_kind,
                "source_session_id": "semantic-session-1",
                "producer_sha256": "a" * 64,
                "source_frame_index": frame_index,
                "reference_monotonic_ns": reference_ns,
                "capture_monotonic_ns": (
                    reference_ns + 1_000_000
                    if source_kind == semantic.SOURCE_RETARGETED_DELAYED
                    else start_ns
                ),
                "sample_period_ns": semantic.SOURCE_SAMPLE_PERIOD_NS,
                "joint_order": semantic.JOINT_ORDER,
                "complete_joint_mask_il29": [True] * semantic.SOURCE_DOF,
                "joint_values_semantics": "full_reference_no_fill",
                "velocity_semantics": "source_50hz_forward_difference",
                "temporal_semantics": (
                    "measured_delayed_reference"
                    if source_kind == semantic.SOURCE_RETARGETED_DELAYED
                    else "precomputed_future"
                ),
                "joint_pos_il29": positions,
                "joint_vel_il29": [0.05] * semantic.SOURCE_DOF,
            }
        )
    return frames


def test_recorded_reference_builds_exact_true23_240_term() -> None:
    window = _recorded_window()
    summary = semantic.validate_semantic_reference_window(window, motion_dir=MOTION_DIR)

    assert summary == {
        "profile": semantic.PROFILE_TRUE23_STEP5,
        "source_kind": semantic.SOURCE_RECORDED_MOTION,
        "playback_frame_index": 10,
        "horizon_s": 0.9,
        "command_dim": 240,
        "terminal_clamped": False,
        "read_only": True,
    }
    assert window["future_frame_indices"] == list(range(10, 56, 5))
    assert window["future_frame_offsets_s"] == pytest.approx(
        [index / 10 for index in range(10)]
    )
    assert len(window["joint_pos_lower_body"]) == 120
    assert len(window["joint_vel_lower_body"]) == 120
    assert window["command_multi_future_lower_body"] == [
        *window["joint_pos_lower_body"],
        *window["joint_vel_lower_body"],
    ]
    assert any(abs(value) > 1.0e-6 for value in window["joint_pos_lower_body"])
    assert any(abs(value) > 1.0e-6 for value in window["joint_vel_lower_body"])
    assert semantic.true23_command_from_window(
        window,
        motion_dir=MOTION_DIR,
    ) == window["command_multi_future_lower_body"]

    positions = _load_csv(MOTION_DIR / "joint_pos.csv")
    expected_first = [
        positions[10][index] for index in semantic.LOWER_BODY_IL29_INDICES
    ]
    expected_last = [
        positions[55][index] for index in semantic.LOWER_BODY_IL29_INDICES
    ]
    assert window["joint_pos_lower_body"][:12] == expected_first
    assert window["joint_pos_lower_body"][-12:] == expected_last


def test_same_shape_low_latency_time_semantics_are_not_true23() -> None:
    window = _recorded_window(semantic.PROFILE_RELEASED_LOW_LATENCY_STEP1)
    semantic.validate_semantic_reference_window(window, motion_dir=MOTION_DIR)

    assert semantic.required_buffer_frames(
        semantic.PROFILE_RELEASED_LOW_LATENCY_STEP1
    ) == 11
    assert semantic.required_buffer_frames(semantic.PROFILE_TRUE23_STEP5) == 47
    assert window["future_frame_indices"] == list(range(10, 20))
    assert window["future_frame_offsets_s"][-1] == pytest.approx(0.18)
    assert len(window["command_multi_future_lower_body"]) == 240
    with pytest.raises(ValueError, match="same-shape low-latency"):
        semantic.true23_command_from_window(window, motion_dir=MOTION_DIR)


def test_checked_in_protocol_matches_bridge_and_current_pico_stays_blocked() -> None:
    protocol = json.loads(
        (
            REPO_ROOT
            / "gear_sonic"
            / "config"
            / "deployment"
            / "g1_true23_semantic_reference_protocol.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["source_sample_rate_hz"] == semantic.SOURCE_SAMPLE_RATE_HZ
    assert protocol["joint_order"] == semantic.JOINT_ORDER
    assert protocol["lower_body_il29_indices"] == list(
        semantic.LOWER_BODY_IL29_INDICES
    )
    for profile in (
        semantic.PROFILE_TRUE23_STEP5,
        semantic.PROFILE_RELEASED_LOW_LATENCY_STEP1,
    ):
        spec = protocol["temporal_profiles"][profile]
        assert spec["future_frame_step"] == semantic.future_frame_step(profile)
        assert spec["future_frame_offsets_s"] == pytest.approx(
            semantic.future_offsets_s(profile)
        )
        assert spec["required_buffer_frames"] == semantic.required_buffer_frames(
            profile
        )

    current_pico_source = (
        REPO_ROOT / "gear_sonic" / "scripts" / "pico_manager_thread_server.py"
    ).read_text(encoding="utf-8")
    assert "joint_pos = np.zeros(29)" in current_pico_source
    assert '"joint_vel": np.zeros((N, 29))' in current_pico_source
    assert "num_frames_to_send: int = 5" in current_pico_source
    assert protocol["live_evidence_bindings"][
        "single_timestamp_for_all_encoder_terms_allowed"
    ] is False


def test_recorded_reference_never_clamps_incomplete_terminal_horizon() -> None:
    positions = _load_csv(MOTION_DIR / "joint_pos.csv")
    # Last selected step5 frame would be final CSV row. Bridge requires one
    # additional real source frame to prove its velocity instead of clamping.
    playback_frame = (
        len(positions)
        - semantic.required_buffer_frames(semantic.PROFILE_TRUE23_STEP5)
        + 1
    )
    epoch_ns = 1_000_000_000_000
    with pytest.raises(ValueError, match="horizon plus velocity proof"):
        semantic.build_recorded_reference_window(
            MOTION_DIR,
            profile=semantic.PROFILE_TRUE23_STEP5,
            playback_frame_index=playback_frame,
            playback_epoch_monotonic_ns=epoch_ns,
            emitted_monotonic_ns=(
                epoch_ns + playback_frame * semantic.SOURCE_SAMPLE_PERIOD_NS
            ),
        )


@pytest.mark.parametrize(
    ("field", "index"),
    [
        ("joint_pos_lower_body", 37),
        ("joint_vel_lower_body", 84),
        ("command_multi_future_lower_body", 203),
    ],
)
def test_recorded_reference_tamper_fails_source_replay(field: str, index: int) -> None:
    window = _recorded_window()
    window[field][index] += 0.25
    with pytest.raises(ValueError):
        semantic.validate_semantic_reference_window(window, motion_dir=MOTION_DIR)


def test_delayed_stream_proves_oldest_is_playback_not_capture_now() -> None:
    frames = _stream_frames()
    final_selected_reference_ns = frames[-2]["reference_monotonic_ns"]
    proof_frame_reference_ns = frames[-1]["reference_monotonic_ns"]
    emitted_ns = proof_frame_reference_ns + 5_000_000
    window = semantic.build_stream_reference_window(
        frames,
        profile=semantic.PROFILE_TRUE23_STEP5,
        emitted_monotonic_ns=emitted_ns,
    )
    summary = semantic.validate_semantic_reference_window(window)

    assert summary["horizon_s"] == 0.9
    assert window["playback"]["emission_lag_ns"] == 925_000_000
    assert window["future_reference_monotonic_ns"][0] == frames[0][
        "reference_monotonic_ns"
    ]
    assert (
        window["future_reference_monotonic_ns"][-1]
        == final_selected_reference_ns
    )
    assert window["future_frame_indices"] == list(range(100, 146, 5))
    assert window["source"]["temporal_semantics"] == "measured_delayed_reference"


def test_planner_stream_may_materialize_real_future_before_emission() -> None:
    frames = _stream_frames(source_kind=semantic.SOURCE_PLANNER)
    emitted_ns = frames[0]["reference_monotonic_ns"] + 5_000_000
    window = semantic.build_stream_reference_window(
        frames,
        profile=semantic.PROFILE_TRUE23_STEP5,
        emitted_monotonic_ns=emitted_ns,
    )

    semantic.validate_semantic_reference_window(window)
    assert window["future_reference_monotonic_ns"][-1] > emitted_ns
    assert window["source"]["temporal_semantics"] == "precomputed_future"


def test_stream_rejects_raw_pico_zero_fill_and_incomplete_buffer() -> None:
    frames = _stream_frames()
    with pytest.raises(ValueError, match="at least 47"):
        semantic.build_stream_reference_window(
            frames[:-1],
            profile=semantic.PROFILE_TRUE23_STEP5,
            emitted_monotonic_ns=frames[-1]["reference_monotonic_ns"],
        )

    raw = copy.deepcopy(frames[0])
    raw["source_kind"] = "raw_pico_pose"
    raw["joint_pos_il29"] = [0.0] * semantic.SOURCE_DOF
    raw["joint_vel_il29"] = [0.0] * semantic.SOURCE_DOF
    with pytest.raises(ValueError, match="raw/tracker pose"):
        semantic.validate_semantic_reference_frame(raw)

    incomplete = copy.deepcopy(frames[0])
    incomplete["complete_joint_mask_il29"][7] = False
    with pytest.raises(ValueError, match="source contract"):
        semantic.validate_semantic_reference_frame(incomplete)


def test_stream_rejects_velocity_relabel_and_timestamp_phase_change() -> None:
    frames = _stream_frames()
    zero_velocity = copy.deepcopy(frames)
    zero_velocity[7]["joint_vel_il29"] = [0.0] * semantic.SOURCE_DOF
    with pytest.raises(ValueError, match="velocity"):
        semantic.build_stream_reference_window(
            zero_velocity,
            profile=semantic.PROFILE_TRUE23_STEP5,
            emitted_monotonic_ns=frames[-1]["reference_monotonic_ns"] + 5_000_000,
        )

    wrong_phase = copy.deepcopy(frames)
    wrong_phase[20]["reference_monotonic_ns"] += 1
    with pytest.raises(ValueError, match="contiguous 50 Hz"):
        semantic.build_stream_reference_window(
            wrong_phase,
            profile=semantic.PROFILE_TRUE23_STEP5,
            emitted_monotonic_ns=frames[-1]["reference_monotonic_ns"] + 5_000_000,
        )
