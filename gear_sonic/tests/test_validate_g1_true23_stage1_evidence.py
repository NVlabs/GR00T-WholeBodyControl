from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import pytest

from gear_sonic.scripts.validate_g1_true23_stage1_evidence import (
    EvidenceError,
    _load_jsonl,
    validate_active,
    validate_publisher,
    validate_runtime_publisher,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_bytes(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records).encode(
            "utf-8"
        )
    )


def _fixture(tmp_path: Path) -> tuple[argparse.Namespace, list[dict[str, object]]]:
    binary = tmp_path / "g1_true23_active_gantry"
    shadow = tmp_path / "v12_promoted_shadow_20260803_200000.jsonl"
    promotion = tmp_path / "active_promotion.json"
    evidence = tmp_path / "v12_stage1_active_20260803_200100.jsonl"
    binary.write_bytes(b"reviewed-controller")
    shadow.write_bytes(b'{"passed":true}\n')
    promotion.write_bytes(b'{"authorized":true}\n')
    authorization_id = "gantry-20260803-test"

    def common(event: str, monotonic_ns: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "g1_true23_stage1_gantry_execution_evidence",
            "event": event,
            "authorization_id": authorization_id,
            "monotonic_ns": monotonic_ns,
        }

    start = common("session_start", 1)
    start.update(
        {
            "controller_binary_path": str(binary.resolve()),
            "controller_binary_sha256": _sha256(binary),
            "encoder_sha256": "1" * 64,
            "decoder_sha256": "2" * 64,
            "metadata_sha256": "3" * 64,
            "promotion_sha256": "4" * 64,
            "active_promotion_sha256": _sha256(promotion),
            "live_shadow_evidence_sha256": _sha256(shadow),
            "live_shadow_action_frames": 100,
            "network": "eth0",
            "pico_endpoint": "tcp://127.0.0.1:5557",
            "post_arm_duration_seconds": 20,
            "minimum_policy_command_frames": 100,
            "mutation_gate_open": False,
            "motion_mode_released": False,
            "lowcmd_publisher_created": False,
        }
    )
    artifact = common("artifact_gate_passed", 2)
    artifact.update(
        {
            "onnx_dry_run_passed": True,
            "artifact_bytes_reverified": True,
            "active_promotion_authorized": True,
            "live_shadow_evidence_validated": True,
        }
    )
    mutation = common("mutation_gate_open", 3)
    mutation.update(
        {
            "stable_advancing_lowstate_samples": 5,
            "required_mode_machine": 4,
            "crc_valid": True,
        }
    )
    motion = common("motion_mode_released", 4)
    motion["post_release_mode_name_empty"] = True
    publisher = common("lowcmd_publisher_created", 5)
    publisher.update({"topic": "rt/lowcmd", "writes_before_event": 0})
    ready = common("first_policy_ready_for_arm", 6)
    ready.update({"real_history_frames": 10, "policy_freshness_limit_ns": 100_000_000})
    command = common("first_armed_policy_command_written", 7)
    command.update(
        {
            "policy_command_frame": 1,
            "native_action_dof": 23,
            "feedforward_tau_zero": True,
        }
    )
    complete = common("session_complete", 8)
    complete.update(
        {
            "passed": True,
            "armed_transition_observed": True,
            "policy_command_frames": 100,
            "minimum_policy_command_frames": 100,
            "damping_frames_after_stop": 250,
            "required_damping_frames_after_stop": 250,
            "maximum_target_delta_from_state_rad": 0.1,
            "maximum_target_slew_rad": 0.0005,
            "maximum_abs_predicted_effort_nm": 10.0,
            "maximum_abs_feedforward_tau_nm": 0.0,
            "final_fault": "operator_stop",
            "stop_reason": "reviewed_post_arm_duration_complete",
            "post_arm_elapsed_ns": 20_000_000_000,
            "required_post_arm_duration_ns": 20_000_000_000,
            "inference_error": "",
            "writer_error": "",
            "publisher_write_failed": False,
        }
    )
    records = [
        start,
        artifact,
        mutation,
        motion,
        publisher,
        ready,
        command,
        complete,
    ]
    args = argparse.Namespace(
        evidence=evidence,
        binary=binary,
        shadow_evidence=shadow,
        active_promotion=promotion,
        authorization_id=authorization_id,
    )
    return args, records


def _publisher_fixture(
    tmp_path: Path,
) -> tuple[
    argparse.Namespace,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    publisher_source = tmp_path / "stream_g1_23dof_pico_causal_zmq.py"
    worker_source = tmp_path / "stream_g1_23dof_pico_raw_worker.py"
    xrt_module = tmp_path / "xrobotoolkit_sdk.so"
    publisher_source.write_bytes(b"frozen measured publisher\n")
    worker_source.write_bytes(b"frozen raw worker\n")
    xrt_module.write_bytes(b"frozen hardened xrt\n")
    run_id = "20260803_210000"
    publisher_evidence = tmp_path / f"v12_pico_causal_publisher_{run_id}.jsonl"
    shadow_evidence = tmp_path / f"v12_promoted_shadow_{run_id}.jsonl"
    session_id = "pico-causal-live-test"
    kind = "g1_true23_pico_causal_zmq_evidence"
    authorization = {
        "read_only": True,
        "dds_opened": False,
        "robot_channel_opened": False,
        "actuation_authorized": False,
        "robot_commands_published": False,
    }
    origin_ns = 10_000_000_000
    origin_body_ns = 20_000_000_000
    control_ns = origin_ns + 200_000_000
    completed_unix_ns = time.time_ns()
    started_unix_ns = completed_unix_ns - 1_000_000_000

    start = {
        "schema_version": 1,
        "kind": kind,
        "event": "session_start",
        "session_id": session_id,
        "started_monotonic_ns": origin_ns - 1_000_000_000,
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns_contract": "terminal_record_v1",
        "bind": "tcp://127.0.0.1:5557",
        "pinned_soma_source_root": "/root/.cache/g1_true23_soma/source",
        "publisher_sha256": _sha256(publisher_source),
        "requested_xrt_module_path": str(xrt_module.resolve()),
        "requested_xrt_module_sha256": _sha256(xrt_module),
        "capture_worker_sha256": _sha256(worker_source),
        "pico_client_apk_sha256": "e" * 64,
        "capture_queue_capacity": 64,
        "requested_packets": 1,
        "termination_contract": ("finite_session_complete_or_signal_session_stopped_v1"),
        "authorization": dict(authorization),
    }
    xrt = {
        "schema_version": 1,
        "kind": kind,
        "event": "xrt_binding_verified",
        "session_id": session_id,
        "imported_xrt_module_path": str(xrt_module.resolve()),
        "imported_xrt_module_sha256": _sha256(xrt_module),
        "matches_requested_binding": True,
        "authorization": dict(authorization),
    }
    prime = {
        "schema_version": 1,
        "kind": kind,
        "event": "solver_primed_before_stream_clock",
        "session_id": session_id,
        "prime_report": {},
        "prime_body_sample_sequence": 100,
        "semantic_sample_emitted": False,
    }
    pose_contract = "unique_or_bounded_bracketed_measured_xr24_at_tick_v1"
    interpolation_contract = (
        "linear_xyz_slerp_quaternion_between_consecutive_measured_xr24_v1"
    )
    derivative_contract = "soma_il29_q_50hz_forward_difference_dq_v1"
    config = {
        "schema_version": 1,
        "kind": kind,
        "event": "stream_runtime_configured",
        "session_id": session_id,
        "capture_mode": "continuous_worker_fifo_bounded_measured_tick_selector_v3",
        "pose_selection_contract": pose_contract,
        "xrt_worker_lifecycle": (
            "single_process_request_prime_then_continuous_stream_v2"
        ),
        "capture_queue_capacity": 64,
        "capture_queue_overflow_policy": "fail_closed",
        "raw_capture_accounting": ("every_raw_frame_ordered; selected_or_explicitly_superseded"),
        "startup_origin_selection": ("newest_validated_raw_frame_after_runtime_setup_v1"),
        "maximum_selected_source_age_ns": 20_000_000,
        "selected_source_age_clock": "local_capture_monotonic",
        "measured_interpolation_enabled": True,
        "measured_interpolation_contract": interpolation_contract,
        "maximum_interpolation_capture_span_ns": 80_000_000,
        "maximum_interpolation_left_bracket_age_ns": 80_000_000,
        "maximum_interpolation_body_source_span_ns": 80_000_000,
        "maximum_selected_body_source_timestamp_delta_ns": 80_000_000,
        "control_derivative_contract": derivative_contract,
        "control_derivative_period_ns": 20_000_000,
        "source_pose_timestamp_relabelled": False,
        "termination_contract": ("finite_session_complete_or_signal_session_stopped_v1"),
        "post_start_exact_contiguous_20ms_required": True,
        "publisher_max_age_ns": 80_000_000,
        "publisher_performance_target_ns": 40_000_000,
        "publisher_performance_target_is_safety_gate": False,
        "downstream_total_freshness_limit_ns": 100_000_000,
        "reserved_transport_and_inference_margin_ns": 20_000_000,
        "cyclic_gc_was_enabled": True,
        "cyclic_gc_collected_objects_before_stream": 0,
        "cyclic_gc_enabled_during_stream": False,
        "python_thread_switch_interval_s": 0.001,
        "authorization": dict(authorization),
    }
    origin = {
        "schema_version": 1,
        "kind": kind,
        "event": "stream_origin_selected",
        "session_id": session_id,
        "pose_selection_contract": pose_contract,
        "startup_raw_frame_count": 2,
        "startup_raw_frame_indices": [1, 2],
        "startup_superseded_raw_frame_count": 1,
        "startup_superseded_raw_frame_indices": [1],
        "startup_superseded_body_source_sequence_gap_count": 2,
        "selected_raw_frame_index": 2,
        "selected_capture_monotonic_ns": origin_ns,
        "selected_body_sample_sequence": 104,
        "selected_body_sample_timestamp_ns": origin_body_ns,
        "selected_capture_delta_ns": None,
        "selected_body_source_timestamp_delta_ns": None,
        "selected_capture_age_at_tick_ns": 0,
        "selected_source_age_clock": "local_capture_monotonic",
        "selected_body_source_sequence_gap_count": 0,
        "control_source_frame_index": 0,
        "control_monotonic_ns": origin_ns,
        "raw_bracket_indices": [2, 2],
        "raw_interpolation_alpha": 0.0,
        "source_pose_timestamp_relabelled": False,
        "positions_repeated_or_synthesized": False,
        "semantic_sample_emitted": False,
        "authorization": dict(authorization),
    }
    packet = {
        "schema_version": 1,
        "kind": kind,
        "event": "reference_packet_published",
        "session_id": session_id,
        "packet_index": 0,
        "control_source_frame_index": 10,
        "control_monotonic_ns": control_ns,
        "published_monotonic_ns": control_ns + 1_000_000,
        "publisher_age_ns": 1_000_000,
        "fresh_within_40ms": True,
        "fresh_within_60ms": True,
        "publisher_performance_target_ns": 40_000_000,
        "publisher_performance_target_met": True,
        "publisher_performance_target_is_safety_gate": False,
        "publisher_max_age_ns": 80_000_000,
        "within_publisher_age_budget": True,
        "pre_send_age_ns": 500_000,
        "pose_selection_contract": pose_contract,
        "selected_raw_frame_index": 22,
        "selected_body_sample_sequence": 125,
        "selected_body_sample_timestamp_ns": origin_body_ns + 200_000_000,
        "selected_capture_monotonic_ns": control_ns,
        "selected_source_delta_ns": 20_000_000,
        "selected_capture_delta_ns": 20_000_000,
        "selected_body_source_timestamp_delta_ns": 20_000_000,
        "maximum_selected_body_source_timestamp_delta_ns": 80_000_000,
        "selected_body_sequence_delta_from_previous_selected": 3,
        "selected_capture_age_at_tick_ns": 0,
        "selected_source_age_at_tick_ns": 0,
        "selected_source_age_clock": "local_capture_monotonic",
        "maximum_selected_source_age_ns": 20_000_000,
        "positions_interpolated_from_measured_xr24": False,
        "measured_interpolation_contract": None,
        "interpolation_capture_span_ns": 0,
        "interpolation_body_source_span_ns": 0,
        "interpolation_ready_delay_ns": 0,
        "selected_capture_wait_duration_ns": 1,
        "selected_capture_ipc_duration_ns": 1,
        "selected_capture_queue_depth_before_enqueue": 1,
        "selected_capture_queue_depth_after_dequeue": 0,
        "selected_capture_frame_index_delta": 1,
        "selected_raw_body_sample_sequence_delta": 2,
        "selected_body_source_sequence_gap_count": 1,
        "selected_mailbox_batch_size": 2,
        "selected_mailbox_coalesced_prior_frames": 1,
        "superseded_raw_frame_count": 1,
        "superseded_raw_frame_indices": [21],
        "raw_bracket_indices": [22, 22],
        "raw_interpolation_alpha": 0.0,
        "control_tick_period_ns": 20_000_000,
        "control_derivative_contract": derivative_contract,
        "control_derivative_period_ns": 20_000_000,
        "source_pose_timestamp_relabelled": False,
        "rolling_compute_duration_ns": 1,
        "target_build_duration_ns": 1,
        "solver_duration_ns": 1,
        "body_term_duration_ns": 1,
        "packet_validation_duration_ns": 1,
        "wire_encode_duration_ns": 1,
        "zmq_send_duration_ns": 500_000,
        "previous_evidence_write_duration_ns": None,
        "cyclic_gc_enabled": False,
        "post_start_contiguous_20ms": True,
        "packet_validation": {
            "profile": "true23_causal_step1_history_0p02s_v1",
            "anchor_source_frame_index": 9,
            "proof_source_frame_index": 10,
            "encoder_lower_body_dim": 240,
            "sdk_derivatives_consumed": False,
            "sha256": "a" * 64,
            "promotion_eligible": False,
        },
        "wire_sha256": "b" * 64,
        "sdk_derivatives_consumed": False,
        "positions_repeated_or_synthesized": False,
        "authorization": dict(authorization),
    }

    def duration(count: int, value: int) -> dict[str, object]:
        return {
            "count": count,
            "mean_ns": value if count else None,
            "max_ns": value if count else None,
        }

    selected = list(range(2, 23, 2))
    superseded = list(range(3, 22, 2))
    complete = {
        "schema_version": 1,
        "kind": kind,
        "event": "session_complete",
        "session_id": session_id,
        "requested_packets": 1,
        "completion_reason": "packet_target_reached",
        "stop_requested": False,
        "stop_signal_number": None,
        "stop_signal_name": None,
        "completed_unix_ns_contract": "terminal_record_v1",
        "completed_unix_ns": completed_unix_ns,
        "raw_frames": 22,
        "raw_frames_drained": 22,
        "retargeted_samples": 11,
        "published_packets": 1,
        "fresh_packets": 1,
        "all_packets_fresh_within_60ms": True,
        "all_packets_fresh_within_40ms": True,
        "packets_meeting_40ms_performance_target": 1,
        "publisher_performance_target_ns": 40_000_000,
        "publisher_performance_target_is_safety_gate": False,
        "all_packets_within_publisher_age_budget": True,
        "publisher_max_age_ns": 80_000_000,
        "capture_mode": "continuous_worker_fifo_bounded_measured_tick_selector_v3",
        "pose_selection": {
            "pose_selection_contract": pose_contract,
            "selected_frames": 11,
            "interpolated_control_frames": 0,
            "total_control_frames": 11,
            "selected_raw_frame_indices": selected,
            "superseded_raw_frames": 10,
            "superseded_raw_frame_indices": superseded,
            "selected_capture_delta_timing": duration(10, 20_000_000),
            "selected_body_source_timestamp_delta_timing": duration(10, 20_000_000),
            "max_capture_age_at_tick_ns": 0,
            "maximum_allowed_capture_age_at_tick_ns": 80_000_000,
            "positions_repeated_or_synthesized": False,
            "positions_interpolated_from_measured_xr24": False,
            "measured_interpolation_contract": interpolation_contract,
            "maximum_interpolation_capture_span_ns": 80_000_000,
            "maximum_interpolation_left_bracket_age_ns": 80_000_000,
            "maximum_interpolation_body_source_span_ns": 80_000_000,
            "interpolation_capture_span_timing": duration(0, 0),
            "interpolation_body_source_span_timing": duration(0, 0),
            "interpolation_ready_delay_timing": duration(0, 0),
            "interpolation_raw_brackets": [],
            "pending_raw_frames": 0,
            "pending_raw_frame_indices": [],
        },
        "background_capture": {
            "queue_capacity": 64,
            "captured_frames": 21,
            "max_queue_depth": 2,
            "queue_depth_at_snapshot": 0,
            "queued_raw_frame_indices": [],
            "capture_wait_mean_ns": 1,
            "capture_wait_max_ns": 1,
            "capture_ipc_mean_ns": 1,
            "capture_ipc_max_ns": 1,
            "source_sequence_gap_count": 1,
            "max_source_sequence_delta": 2,
            "mailbox_batch_count": 11,
            "mailbox_coalesced_prior_frames": 10,
            "queue_overflowed": False,
            "failure": None,
            "frames_silently_dropped": 0,
        },
        "stream_origin_raw_frame_index": 2,
        "startup_raw_frame_indices": [1, 2],
        "startup_superseded_raw_frame_count": 1,
        "startup_superseded_raw_frame_indices": [1],
        "captured_raw_frame_accounting": {
            "startup_superseded_raw_frame_indices": [1],
            "selector_selected_raw_frame_indices": selected,
            "selector_superseded_raw_frame_indices": superseded,
            "selector_pending_raw_frame_indices": [],
            "terminal_capture_queue_raw_frame_indices": [],
        },
        "captured_raw_frame_accounting_count": 22,
        "all_captured_raw_frames_explicitly_accounted": True,
        "all_raw_accounting_categories_disjoint": True,
        "total_body_source_sequence_gap_count": 3,
        "xrt_worker_lifecycle": (
            "single_process_request_prime_then_continuous_stream_v2"
        ),
        "capture_wait_timing": duration(22, 1),
        "capture_ipc_timing": duration(22, 1),
        "capture_queue_depth_after_dequeue_max": 1,
        "rolling_compute_timing": duration(11, 1),
        "target_build_timing": duration(11, 1),
        "solver_timing": duration(11, 1),
        "body_term_timing": duration(11, 1),
        "evidence_write_timing": duration(1, 1),
        "startup_frames_dropped_silently": 0,
        "semantic_frames_skipped_between_published_packets": 0,
        "raw_frames_silently_dropped": 0,
        "terminal_unprocessed_selected_samples": 0,
        "terminal_unprocessed_selected_raw_frame_indices": [],
        "terminal_unconsumed_capture_queue_depth": 0,
        "post_start_exact_contiguous_20ms": True,
        "cyclic_gc_was_enabled": True,
        "cyclic_gc_collected_objects_before_stream": 0,
        "cyclic_gc_enabled_during_stream": False,
        "cyclic_gc_restore_in_finally": True,
        "imported_xrt_module_path": str(xrt_module.resolve()),
        "imported_xrt_module_sha256": _sha256(xrt_module),
        "publisher_sha256": _sha256(publisher_source),
        "capture_worker_sha256": _sha256(worker_source),
        "pico_client_apk_sha256": "e" * 64,
        "authorization": dict(authorization),
    }
    records = [start, xrt, prime, config, origin, packet, complete]
    shadow_records = [
        {
            "event": "session_start",
            "started_monotonic_ns": start["started_monotonic_ns"] - 1,
        },
        {
            "event": "action_frame",
            "control_source_frame_index": 10,
            "control_monotonic_ns": control_ns,
        },
        {"event": "session_complete", "passed": True},
    ]
    _write_jsonl(publisher_evidence, records)
    _write_jsonl(shadow_evidence, shadow_records)
    args = argparse.Namespace(
        publisher_evidence=publisher_evidence,
        shadow_evidence=shadow_evidence,
        publisher_source=publisher_source,
        worker_source=worker_source,
        xrt_module=xrt_module,
        apk_sha256="e" * 64,
        minimum_packets=1,
        max_age_seconds=300.0,
    )
    return args, records, shadow_records


def _runtime_publisher_fixture(
    tmp_path: Path,
) -> tuple[
    argparse.Namespace,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    publisher_args, publisher_records, _ = _publisher_fixture(tmp_path)
    active_args, active_records = _fixture(tmp_path)
    active_times = [
        10_210_000_000,
        10_220_000_000,
        10_230_000_000,
        10_240_000_000,
        10_250_000_000,
        10_260_000_000,
        10_300_000_000,
        31_000_000_000,
    ]
    for record, monotonic_ns in zip(active_records, active_times, strict=True):
        record["monotonic_ns"] = monotonic_ns
    _write_jsonl(active_args.evidence, active_records)

    first_packet = publisher_records[5]
    packet_count = 1_015
    packets: list[dict[str, object]] = []
    first_control_ns = first_packet["control_monotonic_ns"]
    first_body_timestamp_ns = first_packet["selected_body_sample_timestamp_ns"]
    for index in range(packet_count):
        packet = copy.deepcopy(first_packet)
        control_ns = first_control_ns + index * 20_000_000
        packet.update(
            {
                "packet_index": index,
                "control_source_frame_index": 10 + index,
                "control_monotonic_ns": control_ns,
                "published_monotonic_ns": control_ns + 1_000_000,
                "selected_raw_frame_index": 22 + index,
                "selected_body_sample_sequence": 125 + index,
                "selected_body_sample_timestamp_ns": (first_body_timestamp_ns + index * 20_000_000),
                "selected_capture_monotonic_ns": control_ns,
                "previous_evidence_write_duration_ns": (None if index == 0 else 1),
            }
        )
        packet["packet_validation"]["anchor_source_frame_index"] = 9 + index
        packet["packet_validation"]["proof_source_frame_index"] = 10 + index
        if index:
            packet.update(
                {
                    "selected_body_sequence_delta_from_previous_selected": 1,
                    "selected_capture_queue_depth_before_enqueue": 0,
                    "selected_raw_body_sample_sequence_delta": 1,
                    "selected_body_source_sequence_gap_count": 0,
                    "selected_mailbox_batch_size": 1,
                    "selected_mailbox_coalesced_prior_frames": 0,
                    "superseded_raw_frame_count": 0,
                    "superseded_raw_frame_indices": [],
                    "raw_bracket_indices": [22 + index, 22 + index],
                }
            )
        packets.append(packet)

    terminal = publisher_records[-1]
    selected = [*range(2, 23, 2), *range(23, 22 + packet_count)]
    superseded = list(range(3, 22, 2))
    raw_frames = 21 + packet_count
    retargeted_samples = packet_count + 10
    terminal.update(
        {
            "event": "session_stopped",
            "requested_packets": 2_000,
            "completion_reason": "signal_requested",
            "stop_requested": True,
            "stop_signal_number": 15,
            "stop_signal_name": "SIGTERM",
            "completed_unix_ns": time.time_ns(),
            "raw_frames": raw_frames,
            "raw_frames_drained": raw_frames,
            "retargeted_samples": retargeted_samples,
            "published_packets": packet_count,
            "fresh_packets": packet_count,
            "packets_meeting_40ms_performance_target": packet_count,
            "captured_raw_frame_accounting_count": raw_frames,
            "capture_wait_timing": {
                "count": raw_frames,
                "mean_ns": 1,
                "max_ns": 1,
            },
            "capture_ipc_timing": {
                "count": raw_frames,
                "mean_ns": 1,
                "max_ns": 1,
            },
            "rolling_compute_timing": {
                "count": retargeted_samples,
                "mean_ns": 1,
                "max_ns": 1,
            },
            "target_build_timing": {
                "count": retargeted_samples,
                "mean_ns": 1,
                "max_ns": 1,
            },
            "solver_timing": {
                "count": retargeted_samples,
                "mean_ns": 1,
                "max_ns": 1,
            },
            "body_term_timing": {
                "count": retargeted_samples,
                "mean_ns": 1,
                "max_ns": 1,
            },
            "evidence_write_timing": {
                "count": packet_count,
                "mean_ns": 1,
                "max_ns": 1,
            },
        }
    )
    terminal["pose_selection"].update(
        {
            "selected_frames": retargeted_samples,
            "total_control_frames": retargeted_samples,
            "selected_raw_frame_indices": selected,
            "superseded_raw_frames": len(superseded),
            "superseded_raw_frame_indices": superseded,
            "selected_capture_delta_timing": {
                "count": retargeted_samples - 1,
                "mean_ns": 20_000_000,
                "max_ns": 20_000_000,
            },
            "selected_body_source_timestamp_delta_timing": {
                "count": retargeted_samples - 1,
                "mean_ns": 20_000_000,
                "max_ns": 20_000_000,
            },
        }
    )
    terminal["background_capture"].update(
        {
            "captured_frames": raw_frames - 1,
            "mailbox_batch_count": packet_count + 10,
            "mailbox_coalesced_prior_frames": 10,
        }
    )
    terminal["captured_raw_frame_accounting"].update(
        {
            "selector_selected_raw_frame_indices": selected,
            "selector_superseded_raw_frame_indices": superseded,
        }
    )
    publisher_records[0]["requested_packets"] = 2_000
    publisher_records = [
        *publisher_records[:5],
        *packets,
        terminal,
    ]
    runtime_publisher_evidence = tmp_path / "v12_stage1_pico_publisher_20260803_200100.jsonl"
    _write_jsonl(runtime_publisher_evidence, publisher_records)
    args = argparse.Namespace(
        publisher_evidence=runtime_publisher_evidence,
        active_evidence=active_args.evidence,
        publisher_source=publisher_args.publisher_source,
        worker_source=publisher_args.worker_source,
        xrt_module=publisher_args.xrt_module,
        apk_sha256=publisher_args.apk_sha256,
        binary=active_args.binary,
        shadow_evidence=active_args.shadow_evidence,
        active_promotion=active_args.active_promotion,
        authorization_id=active_args.authorization_id,
        minimum_packets=100,
        max_age_seconds=300.0,
    )
    return args, publisher_records, active_records


def test_active_execution_evidence_accepts_exact_success(tmp_path: Path) -> None:
    args, records = _fixture(tmp_path)
    _write_jsonl(args.evidence, records)
    result = validate_active(args)
    assert result["policy_command_frames"] == 100
    assert result["damping_frames_after_stop"] == 250


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stop_reason", "wireless_operator_stop"),
        ("post_arm_elapsed_ns", 19_999_999_999),
        ("maximum_target_slew_rad", 0.000_501),
        ("publisher_write_failed", True),
    ],
)
def test_active_execution_evidence_rejects_false_pass(tmp_path: Path, field: str, value: object) -> None:
    args, records = _fixture(tmp_path)
    records[-1][field] = value
    _write_jsonl(args.evidence, records)
    with pytest.raises(EvidenceError):
        validate_active(args)


def test_active_execution_evidence_rejects_extra_field(tmp_path: Path) -> None:
    args, records = _fixture(tmp_path)
    records[2]["unexpected"] = True
    _write_jsonl(args.evidence, records)
    with pytest.raises(EvidenceError):
        validate_active(args)


def test_jsonl_loader_rejects_duplicate_key(tmp_path: Path) -> None:
    evidence = tmp_path / "duplicate.jsonl"
    evidence.write_bytes(b'{"event":"a","event":"b"}\n')
    with pytest.raises(EvidenceError, match="duplicate JSON key"):
        _load_jsonl(evidence, "test evidence")


def test_publisher_evidence_accepts_exact_measured_q0_q10_run(
    tmp_path: Path,
) -> None:
    args, _, _ = _publisher_fixture(tmp_path)
    result = validate_publisher(args)
    assert result["published_packets"] == 1


def test_publisher_evidence_accepts_bounded_measured_bracket_interpolation(
    tmp_path: Path,
) -> None:
    args, records, _ = _publisher_fixture(tmp_path)
    packet = records[5]
    control_ns = packet["control_monotonic_ns"]
    packet.update(
        {
            "published_monotonic_ns": control_ns + 11_000_000,
            "publisher_age_ns": 11_000_000,
            "pre_send_age_ns": 10_500_000,
            "selected_raw_frame_index": None,
            "selected_body_sample_sequence": None,
            "selected_capture_monotonic_ns": control_ns + 10_000_000,
            "selected_capture_delta_ns": None,
            "selected_body_sequence_delta_from_previous_selected": None,
            "selected_capture_age_at_tick_ns": 20_000_000,
            "selected_source_age_at_tick_ns": 20_000_000,
            "selected_source_age_clock": (
                "local_capture_monotonic_left_bracket"
            ),
            "maximum_selected_source_age_ns": 40_000_000,
            "selected_capture_frame_index_delta": None,
            "selected_raw_body_sample_sequence_delta": None,
            "selected_body_source_sequence_gap_count": 0,
            "selected_mailbox_batch_size": 1,
            "selected_mailbox_coalesced_prior_frames": 0,
            "superseded_raw_frame_count": 0,
            "superseded_raw_frame_indices": [],
            "raw_bracket_indices": [20, 21],
            "raw_interpolation_alpha": 2.0 / 3.0,
            "positions_interpolated_from_measured_xr24": True,
            "measured_interpolation_contract": (
                "linear_xyz_slerp_quaternion_between_consecutive_measured_xr24_v1"
            ),
            "interpolation_capture_span_ns": 30_000_000,
            "interpolation_body_source_span_ns": 30_000_000,
            "interpolation_ready_delay_ns": 10_000_000,
        }
    )
    complete = records[-1]
    selected = list(range(2, 21, 2))
    superseded = list(range(3, 20, 2))
    complete.update(
        {
            "raw_frames": 21,
            "raw_frames_drained": 21,
            "captured_raw_frame_accounting_count": 21,
            "capture_wait_timing": {
                "count": 21,
                "mean_ns": 1,
                "max_ns": 1,
            },
            "capture_ipc_timing": {
                "count": 21,
                "mean_ns": 1,
                "max_ns": 1,
            },
        }
    )
    complete["pose_selection"].update(
        {
            "selected_frames": 10,
            "interpolated_control_frames": 1,
            "total_control_frames": 11,
            "selected_raw_frame_indices": selected,
            "superseded_raw_frames": 9,
            "superseded_raw_frame_indices": superseded,
            "selected_capture_delta_timing": {
                "count": 9,
                "mean_ns": 20_000_000,
                "max_ns": 20_000_000,
            },
            "selected_body_source_timestamp_delta_timing": {
                "count": 9,
                "mean_ns": 20_000_000,
                "max_ns": 20_000_000,
            },
            "max_capture_age_at_tick_ns": 20_000_000,
            "positions_interpolated_from_measured_xr24": True,
            "interpolation_capture_span_timing": {
                "count": 1,
                "mean_ns": 30_000_000,
                "max_ns": 30_000_000,
            },
            "interpolation_body_source_span_timing": {
                "count": 1,
                "mean_ns": 30_000_000,
                "max_ns": 30_000_000,
            },
            "interpolation_ready_delay_timing": {
                "count": 1,
                "mean_ns": 10_000_000,
                "max_ns": 10_000_000,
            },
            "interpolation_raw_brackets": [[20, 21]],
            "pending_raw_frames": 1,
            "pending_raw_frame_indices": [21],
        }
    )
    complete["background_capture"].update(
        {"captured_frames": 20, "mailbox_batch_count": 10}
    )
    complete["captured_raw_frame_accounting"].update(
        {
            "selector_selected_raw_frame_indices": selected,
            "selector_superseded_raw_frame_indices": superseded,
            "selector_pending_raw_frame_indices": [21],
        }
    )
    _write_jsonl(args.publisher_evidence, records)

    result = validate_publisher(args)

    assert result["published_packets"] == 1
    assert result["publisher_sha256"] == _sha256(args.publisher_source)


def test_publisher_evidence_rejects_body_source_delta_over_80ms(
    tmp_path: Path,
) -> None:
    args, records, _ = _publisher_fixture(tmp_path)
    records[5]["selected_body_source_timestamp_delta_ns"] = 80_000_001
    records[5]["selected_source_delta_ns"] = 60_000_001
    _write_jsonl(args.publisher_evidence, records)
    with pytest.raises(EvidenceError):
        validate_publisher(args)


def test_publisher_evidence_rejects_capture_age_over_one_tick(
    tmp_path: Path,
) -> None:
    args, records, _ = _publisher_fixture(tmp_path)
    records[5]["selected_capture_age_at_tick_ns"] = 20_000_001
    records[5]["selected_source_age_at_tick_ns"] = 20_000_001
    records[5]["selected_capture_monotonic_ns"] = records[5]["control_monotonic_ns"] - 20_000_001
    records[-1]["pose_selection"]["max_capture_age_at_tick_ns"] = 20_000_001
    _write_jsonl(args.publisher_evidence, records)
    with pytest.raises(EvidenceError):
        validate_publisher(args)


def test_publisher_evidence_rejects_overlapping_raw_accounting(
    tmp_path: Path,
) -> None:
    args, records, _ = _publisher_fixture(tmp_path)
    records[-1]["pose_selection"]["superseded_raw_frames"] = 1
    records[-1]["pose_selection"]["superseded_raw_frame_indices"] = [10]
    records[-1]["captured_raw_frame_accounting"]["selector_superseded_raw_frame_indices"] = [10]
    _write_jsonl(args.publisher_evidence, records)
    with pytest.raises(EvidenceError):
        validate_publisher(args)


def test_publisher_evidence_rejects_shadow_control_mismatch(
    tmp_path: Path,
) -> None:
    args, _, shadow_records = _publisher_fixture(tmp_path)
    shadow_records[1]["control_source_frame_index"] = 11
    _write_jsonl(args.shadow_evidence, shadow_records)
    with pytest.raises(EvidenceError):
        validate_publisher(args)


def test_publisher_evidence_rejects_changed_publisher_bytes(
    tmp_path: Path,
) -> None:
    args, _, _ = _publisher_fixture(tmp_path)
    args.publisher_source.write_bytes(b"changed after evidence\n")
    with pytest.raises(EvidenceError):
        validate_publisher(args)


def test_runtime_publisher_accepts_exact_sigterm_and_active_window(
    tmp_path: Path,
) -> None:
    args, _, _ = _runtime_publisher_fixture(tmp_path)
    result = validate_runtime_publisher(args)
    assert result["terminal_event"] == "session_stopped"
    assert result["published_packets"] == 1_015
    assert result["policy_command_frames"] == 100


def test_runtime_publisher_rejects_non_wrapper_signal(tmp_path: Path) -> None:
    args, publisher_records, _ = _runtime_publisher_fixture(tmp_path)
    publisher_records[-1]["stop_signal_number"] = 2
    publisher_records[-1]["stop_signal_name"] = "SIGINT"
    _write_jsonl(args.publisher_evidence, publisher_records)
    with pytest.raises(EvidenceError):
        validate_runtime_publisher(args)


def test_runtime_publisher_rejects_incomplete_active_window(
    tmp_path: Path,
) -> None:
    args, _, active_records = _runtime_publisher_fixture(tmp_path)
    active_records[6]["monotonic_ns"] = 11_000_000_000
    _write_jsonl(args.active_evidence, active_records)
    with pytest.raises(EvidenceError):
        validate_runtime_publisher(args)
