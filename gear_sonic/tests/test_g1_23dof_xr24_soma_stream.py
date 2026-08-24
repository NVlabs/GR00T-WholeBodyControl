from __future__ import annotations

import copy

import pytest

from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    causal_history_profile_contract,
)
from gear_sonic.tests.test_g1_23dof_pico_retargeted_producer import _capture
from gear_sonic.utils import g1_23dof_xr24_soma_stream as stream
from gear_sonic.utils.g1_23dof_contract import (
    REFERENCE_PROFILE_LOW_LATENCY,
    REFERENCE_PROFILE_NORMAL,
    reference_profile_contract,
)
from gear_sonic.utils.g1_23dof_live_shadow import build_encoder_observation
from gear_sonic.utils.g1_23dof_xr24_soma_adapter import (
    resample_raw_capture_50hz,
)
from gear_sonic.utils.g1_23dof_xr24_soma_stream import (
    CausalHistorySemanticProducer,
    IncrementalCaptureResampler,
    RollingSemanticProducer,
    causal_history_reference_terms,
    compare_rolling_to_batch,
    complete_causal_history_encoder_packet,
    complete_profile_bound_encoder_packet,
    measured_future_latency_proof,
    validate_causal_history_packet,
)


def _identity(capture: dict) -> dict:
    return {
        key: copy.deepcopy(capture[key])
        for key in (
            "schema_version",
            "kind",
            "session_id",
            "source",
            "authorization",
        )
    }


def test_measured_future_profiles_cannot_meet_60ms_freshness() -> None:
    normal = measured_future_latency_proof(REFERENCE_PROFILE_NORMAL)
    low_latency = measured_future_latency_proof(
        REFERENCE_PROFILE_LOW_LATENCY
    )

    assert normal["selected_horizon_ns"] == 900_000_000
    assert normal["intrinsic_minimum_measured_delay_ns"] == 920_000_000
    assert normal["checked_adapter_delay_ns"] == 940_000_000
    assert low_latency["selected_horizon_ns"] == 180_000_000
    assert low_latency["intrinsic_minimum_measured_delay_ns"] == 200_000_000
    assert low_latency["checked_adapter_delay_ns"] == 220_000_000
    assert normal["intrinsic_budget_feasible"] is False
    assert low_latency["intrinsic_budget_feasible"] is False


def test_incremental_resampler_matches_batch_without_future_samples() -> None:
    capture = _capture(17)
    incremental = IncrementalCaptureResampler(_identity(capture))
    streamed = [
        sample
        for frame in capture["frames"]
        for sample in incremental.push(frame)
    ]
    batch = resample_raw_capture_50hz(capture)

    assert len(streamed) == len(batch)
    for sample, batch_sample in zip(streamed, batch, strict=True):
        assert sample["source_frame_index"] == batch_sample["source_frame_index"]
        assert (
            sample["reference_monotonic_ns"]
            == batch_sample["reference_monotonic_ns"]
        )
        assert sample["body_poses"] == batch_sample["body_poses"]
        left, right = sample["raw_bracket_indices"]
        assert right == left + 1
        assert (
            capture["frames"][left]["capture_monotonic_ns"]
            <= sample["reference_monotonic_ns"]
        )
        assert (
            sample["reference_monotonic_ns"]
            <= capture["frames"][right]["capture_monotonic_ns"]
        )


def test_batch_comparison_never_promotes() -> None:
    batch = [[float(index)] * 36 for index in range(3)]
    rolling = [
        {"joint_root7_mj29": list(row)}
        for row in batch
    ]
    report = compare_rolling_to_batch(rolling, batch)

    assert report["exact_within_tolerance"] is True
    assert report["max_abs_error"] == 0.0
    assert report["promotion_eligible"] is False


def _rolling_sample(index: int) -> dict:
    reference_ns = 7_000_000_000 + index * 20_000_000
    return {
        "kind": stream.ROLLING_SOMA_KIND,
        "promotion_eligible": False,
        "source_frame_index": index,
        "reference_monotonic_ns": reference_ns,
        "capture_monotonic_ns": reference_ns,
        "compute_finished_monotonic_ns": reference_ns,
        "producer_sha256": "d" * 64,
        "joint_pos_il29": [
            0.1 + index * (joint + 1) * 1.0e-4
            for joint in range(29)
        ],
        "body_term": {
            "source_frame_index": index,
            "reference_monotonic_ns": reference_ns,
            "capture_monotonic_ns": reference_ns,
            "vr_3point_local_target": [0.1] * 9,
            "vr_3point_local_orn_target": [0.0, 0.0, 0.0, 1.0] * 3,
            "reference_anchor_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }


def test_causal_stream_contract_exactly_matches_training_contract() -> None:
    assert stream.causal_history_stream_contract() == (
        causal_history_profile_contract()
    )


def test_experimental_soma_profiles_are_distinct_and_nonpromotable() -> None:
    exact = stream.soma_rolling_retarget_contract(24)
    ik16 = stream.soma_rolling_retarget_contract(16)
    ik12 = stream.soma_rolling_retarget_contract(12)
    assert len(
        {
            exact["contract_sha256"],
            ik16["contract_sha256"],
            ik12["contract_sha256"],
        }
    ) == 3
    assert exact["exact_ik24_equivalence_required"] is True
    assert ik16["requires_sim_and_shadow_revalidation"] is True
    assert ik12["promotion_eligible"] is False
    with pytest.raises(ValueError, match="24, 16, or 12"):
        stream.soma_rolling_retarget_contract(8)


def test_causal_history_packet_is_exact_q0_q9_plus_q10_proof() -> None:
    producer = CausalHistorySemanticProducer(
        source_session_id="causal-live-capture"
    )
    packets = [
        packet
        for index in range(11)
        if (packet := producer.push(_rolling_sample(index))) is not None
    ]
    assert len(packets) == 1
    packet = packets[0]
    expected_positions = [
        0.1 + frame * (joint + 1) * 1.0e-4
        for frame in range(10)
        for joint in stream.CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES
    ]
    expected_velocities = [
        (joint + 1) * 0.005
        for _frame in range(10)
        for joint in stream.CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES
    ]
    assert packet["profile"] == stream.CAUSAL_HISTORY_PROFILE
    assert packet["position_source_frame_indices"] == list(range(10))
    assert packet["velocity_proof_source_frame_indices"] == list(range(1, 11))
    assert packet["anchor_source_frame_index"] == 9
    assert packet["proof_source_frame_index"] == 10
    assert packet["intrinsic_measurement_delay_ns"] == 20_000_000
    assert packet["causal_history_lower_body"][:120] == pytest.approx(
        expected_positions
    )
    assert packet["causal_history_lower_body"][120:] == pytest.approx(
        expected_velocities
    )
    expected_q_ref23 = [
        _rolling_sample(9)["joint_pos_il29"][index]
        for index in stream.NATIVE_IL23_TO_CANONICAL_IL29
    ]
    expected_qd_ref23 = [
        (_rolling_sample(10)["joint_pos_il29"][index] - value) * 50.0
        for index, value in zip(
            stream.NATIVE_IL23_TO_CANONICAL_IL29,
            expected_q_ref23,
            strict=True,
        )
    ]
    assert packet["q_ref23_native"] == pytest.approx(expected_q_ref23)
    assert packet["qd_ref23_native"] == pytest.approx(expected_qd_ref23)
    tampered = copy.deepcopy(packet)
    tampered["q_ref23_native"][5] += 0.01
    with pytest.raises(ValueError, match="not bound to q9/q10"):
        validate_causal_history_packet(tampered)
    assert packet["sdk_derivatives_consumed"] is False
    assert packet["positions_repeated_or_synthesized"] is False
    assert packet["anchor_body_term"]["source_frame_index"] == 9
    assert packet["promotion_eligible"] is False
    summary = validate_causal_history_packet(packet)
    assert summary["encoder_lower_body_dim"] == 240
    reference_terms = causal_history_reference_terms(packet)
    assert reference_terms["kind"] == (
        stream.CAUSAL_HISTORY_REFERENCE_TERMS_KIND
    )
    assert reference_terms["pico_anchor_source_frame_index"] == 9
    assert reference_terms["control_source_frame_index"] == 10
    assert reference_terms["q_ref23_native"] == pytest.approx(expected_q_ref23)
    assert reference_terms["qd_ref23_native"] == pytest.approx(expected_qd_ref23)
    assert "motion_anchor_ori_b" not in reference_terms
    assert "future_frame_offsets_s" not in reference_terms


def test_causal_history_rejects_canonical_il29_prefix_mislabeled_as_legs() -> None:
    producer = CausalHistorySemanticProducer(source_session_id="selector-test")
    packet = None
    for index in range(11):
        packet = producer.push(_rolling_sample(index))
    assert packet is not None

    bad = copy.deepcopy(packet)
    bad["proof_lower_body_position"] = list(bad["proof_joint_pos_il29"][:12])
    last_position = bad["causal_history_positions_lower_body"][-12:]
    bad["causal_history_lower_body"][-12:] = [
        (proof - anchor) * 50.0
        for anchor, proof in zip(
            last_position,
            bad["proof_lower_body_position"],
            strict=True,
        )
    ]
    with pytest.raises(ValueError, match="selector/order differs from encoder ABI"):
        validate_causal_history_packet(bad)


def test_causal_history_encoder_leg_order_is_mujoco_hardware_order() -> None:
    producer = CausalHistorySemanticProducer(source_session_id="encoder-order-test")
    packet = None
    for index in range(11):
        packet = producer.push(_rolling_sample(index))
    assert packet is not None

    assert stream.CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES == (
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
    )
    contract = stream.causal_history_stream_contract()
    assert contract["lower_body_il29_indices_in_encoder_order"] == list(
        stream.CAUSAL_ENCODER_LOWER_BODY_IL29_INDICES
    )
    assert contract["lower_body_order"] == "mujoco_hardware_left6_then_right6"

    completed = complete_causal_history_encoder_packet(
        semantic_packet=packet,
        robot_anchor_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        robot_anchor_monotonic_ns=packet["anchor_reference_monotonic_ns"],
        robot_anchor_source_contract=(
            stream.CAUSAL_HISTORY_ROBOT_ANCHOR_CONTRACT
        ),
    )
    assert len(completed["encoder_input"]) == 267
    assert completed["encoder_terms"]["pico_anchor_source_frame_index"] == 9
    assert completed["encoder_terms"]["control_source_frame_index"] == 10
    assert "future_frame_offsets_s" not in completed["encoder_terms"]
    assert "command_multi_future_lower_body" not in completed["encoder_terms"]
    with pytest.raises(ValueError, match="exact causal q9"):
        complete_causal_history_encoder_packet(
            semantic_packet=packet,
            robot_anchor_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
            robot_anchor_monotonic_ns=(
                packet["anchor_reference_monotonic_ns"] + 1
            ),
            robot_anchor_source_contract=(
                stream.CAUSAL_HISTORY_ROBOT_ANCHOR_CONTRACT
            ),
        )
    with pytest.raises(ValueError, match="no complete reference window"):
        complete_profile_bound_encoder_packet(
            semantic_packet=packet,
            expected_profile=REFERENCE_PROFILE_LOW_LATENCY,
            robot_anchor_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
            robot_anchor_monotonic_ns=packet["anchor_reference_monotonic_ns"],
        )


def test_causal_position_path_ignores_untrusted_sdk_derivatives() -> None:
    capture = _capture(17)
    poisoned = copy.deepcopy(capture)
    for frame in poisoned["frames"]:
        frame["body_velocities"][0][0] = 4.7846e6
        frame["body_accelerations"][0][2] = 5.8945e21
    assert resample_raw_capture_50hz(poisoned) == resample_raw_capture_50hz(
        capture
    )


def test_causal_history_rejects_noncontiguous_q_proof() -> None:
    producer = CausalHistorySemanticProducer(source_session_id="causal-gap")
    producer.push(_rolling_sample(0))
    with pytest.raises(ValueError, match="contiguous exact 50 Hz"):
        producer.push(_rolling_sample(2))


def test_low_latency_rolling_packet_is_exactly_profile_bound() -> None:
    producer = RollingSemanticProducer(
        source_session_id="rolling-low-latency",
        profile=REFERENCE_PROFILE_LOW_LATENCY,
    )
    packets = [
        packet
        for index in range(12)
        if (packet := producer.push(_rolling_sample(index))) is not None
    ]
    packet = packets[-1]
    window = packet["reference_window"]

    assert window is not None
    assert packet["measured_delay_ns"] == 220_000_000
    assert window["future_frame_indices"] == list(range(10))
    completed = complete_profile_bound_encoder_packet(
        semantic_packet=packet,
        expected_profile=REFERENCE_PROFILE_LOW_LATENCY,
        robot_anchor_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        robot_anchor_monotonic_ns=window["playback"]["frame_monotonic_ns"],
    )
    terms = completed["encoder_terms"]
    observation = build_encoder_observation(
        terms,
        expected_reference_contract=reference_profile_contract(
            REFERENCE_PROFILE_LOW_LATENCY
        ),
    )

    assert len(observation) == 267
    assert terms["future_frame_offsets_s"][-1] == pytest.approx(0.18)
    assert completed["promotion_eligible"] is False
    with pytest.raises(ValueError, match="exact semantic anchor"):
        complete_profile_bound_encoder_packet(
            semantic_packet=packet,
            expected_profile=REFERENCE_PROFILE_LOW_LATENCY,
            robot_anchor_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
            robot_anchor_monotonic_ns=(
                window["playback"]["frame_monotonic_ns"] + 1
            ),
        )
