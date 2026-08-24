"""Strict byte/provenance gates for True23 Stage-1 gantry execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PAIR_SHADOW = re.compile(r"v12_promoted_shadow_(\d{8}_\d{6})\.jsonl\Z")
_PAIR_PUBLISHER = re.compile(r"v12_pico_causal_publisher_(\d{8}_\d{6})\.jsonl\Z")
_PAIR_STAGE1_PUBLISHER = re.compile(r"v12_stage1_pico_publisher_(\d{8}_\d{6})\.jsonl\Z")
_PAIR_STAGE1_ACTIVE = re.compile(r"v12_stage1_active_(\d{8}_\d{6})\.jsonl\Z")
_PUBLISHER_KIND = "g1_true23_pico_causal_zmq_evidence"
_ACTIVE_KIND = "g1_true23_stage1_gantry_execution_evidence"
_TERMINATION_CONTRACT = "finite_session_complete_or_signal_session_stopped_v1"
_AUTHORIZATION = {
    "read_only": True,
    "dds_opened": False,
    "robot_channel_opened": False,
    "actuation_authorized": False,
    "robot_commands_published": False,
}


class EvidenceError(ValueError):
    pass


def _reject(message: str) -> None:
    raise EvidenceError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_non_symlink(path: Path, role: str) -> Path:
    absolute = path.absolute()
    status = os.lstat(absolute)
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        _reject(f"{role} must be a regular non-symlink file")
    return absolute.resolve(strict=True)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _reject(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _reject(f"non-finite JSON constant: {value}")


def _load_jsonl(path: Path, role: str) -> tuple[Path, list[dict[str, Any]]]:
    resolved = _regular_non_symlink(path, role)
    size = resolved.stat().st_size
    if size <= 0 or size > 64 * 1024 * 1024:
        _reject(f"{role} is empty or oversized")
    data = resolved.read_bytes()
    if not data.endswith(b"\n") or b"\r" in data:
        _reject(f"{role} must use LF records and terminal LF")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(data[:-1].split(b"\n"), 1):
        if not line or len(line) > 2 * 1024 * 1024:
            _reject(f"{role} line {line_number} is empty or oversized")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_pairs_no_duplicates,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _reject(f"{role} line {line_number} is invalid JSON: {exc}")
        if not isinstance(value, dict):
            _reject(f"{role} line {line_number} must be one object")
        records.append(value)
    return resolved, records


def _exact(record: dict[str, Any], keys: set[str], role: str) -> None:
    if set(record) != keys:
        missing = sorted(keys - set(record))
        extra = sorted(set(record) - keys)
        _reject(f"{role} field set mismatch; missing={missing}, extra={extra}")


def _integer(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _reject(f"{role} must be integer")
    return value


def _number(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject(f"{role} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _reject(f"{role} must be finite")
    return result


def _common(record: dict[str, Any], *, kind: str, event: str) -> None:
    if record.get("schema_version") != 1 or record.get("kind") != kind or record.get("event") != event:
        _reject(f"{event} identity mismatch")


def _integer_list(value: Any, role: str) -> list[int]:
    if not isinstance(value, list):
        _reject(f"{role} must be integer array")
    result = [_integer(item, role) for item in value]
    if any(item < 0 for item in result):
        _reject(f"{role} contains negative index")
    return result


def _duration_summary(value: Any, role: str) -> tuple[int, float | None, int | None]:
    if not isinstance(value, dict):
        _reject(f"{role} must be object")
    _exact(value, {"count", "mean_ns", "max_ns"}, role)
    count = _integer(value.get("count"), f"{role}.count")
    if count < 0:
        _reject(f"{role}.count must be nonnegative")
    if count == 0:
        if value.get("mean_ns") is not None or value.get("max_ns") is not None:
            _reject(f"{role} empty summary must use null mean/max")
        return count, None, None
    mean_ns = _number(value.get("mean_ns"), f"{role}.mean_ns")
    max_ns = _integer(value.get("max_ns"), f"{role}.max_ns")
    if mean_ns < 0 or max_ns < 0 or mean_ns > max_ns:
        _reject(f"{role} duration summary is inconsistent")
    return count, mean_ns, max_ns


def validate_publisher(args: argparse.Namespace) -> dict[str, Any]:
    runtime = bool(getattr(args, "runtime", False))
    publisher_path, records = _load_jsonl(args.publisher_evidence, "publisher evidence")
    shadow_path: Path | None = None
    shadow_records: list[dict[str, Any]] = []
    active_path: Path | None = None
    active_records: list[dict[str, Any]] = []
    if runtime:
        active_path, active_records = _load_jsonl(args.active_evidence, "active execution evidence")
    else:
        shadow_path, shadow_records = _load_jsonl(args.shadow_evidence, "shadow evidence")
    publisher_source = _regular_non_symlink(args.publisher_source, "publisher source")
    worker_source = _regular_non_symlink(args.worker_source, "capture worker source")
    xrt_module = _regular_non_symlink(args.xrt_module, "XRT module")
    if runtime:
        assert active_path is not None
        active_match = _PAIR_STAGE1_ACTIVE.fullmatch(active_path.name)
        publisher_match = _PAIR_STAGE1_PUBLISHER.fullmatch(publisher_path.name)
        if active_match is None or publisher_match is None or active_match.group(1) != publisher_match.group(1):
            _reject("runtime publisher and active filenames are not one paired run")
    else:
        assert shadow_path is not None
        shadow_match = _PAIR_SHADOW.fullmatch(shadow_path.name)
        publisher_match = _PAIR_PUBLISHER.fullmatch(publisher_path.name)
        if shadow_match is None or publisher_match is None or shadow_match.group(1) != publisher_match.group(1):
            _reject("publisher and promoted-shadow filenames are not one paired run")
    if not _SHA256.fullmatch(args.apk_sha256):
        _reject("APK binding must be lowercase SHA-256")
    if args.minimum_packets <= 0 or args.max_age_seconds <= 0:
        _reject("publisher validation bounds must be positive")
    age_seconds = time.time() - publisher_path.stat().st_mtime
    if age_seconds < -5 or age_seconds > args.max_age_seconds:
        _reject("publisher evidence is not fresh")
    if len(records) < args.minimum_packets + 6:
        _reject("publisher evidence cannot prove minimum packet count")
    events = [record.get("event") for record in records]
    if events[:5] != [
        "session_start",
        "xrt_binding_verified",
        "solver_primed_before_stream_clock",
        "stream_runtime_configured",
        "stream_origin_selected",
    ]:
        _reject("publisher gate event order mismatch")
    terminal_event = "session_stopped" if runtime else "session_complete"
    if events[-1] != terminal_event or "session_failed" in events:
        _reject("publisher evidence lacks unique successful terminal event")
    packet_records = records[5:-1]
    if not packet_records or any(record.get("event") != "reference_packet_published" for record in packet_records):
        _reject("publisher evidence has trailing, missing, or reordered records")

    start = records[0]
    required_start = {
        "schema_version",
        "kind",
        "event",
        "session_id",
        "started_monotonic_ns",
        "started_unix_ns",
        "completed_unix_ns_contract",
        "bind",
        "pinned_soma_source_root",
        "publisher_sha256",
        "requested_xrt_module_path",
        "requested_xrt_module_sha256",
        "capture_worker_sha256",
        "pico_client_apk_sha256",
        "capture_queue_capacity",
        "requested_packets",
        "termination_contract",
        "authorization",
    }
    _exact(start, required_start, "publisher session_start")
    _common(start, kind=_PUBLISHER_KIND, event="session_start")
    session_id = start.get("session_id")
    expected_publisher_sha = _sha256(publisher_source)
    expected_worker_sha = _sha256(worker_source)
    expected_xrt_sha = _sha256(xrt_module)
    started_unix_ns = _integer(start.get("started_unix_ns"), "started_unix_ns")
    started_monotonic_ns = _integer(start.get("started_monotonic_ns"), "started_monotonic_ns")
    queue_capacity = _integer(start.get("capture_queue_capacity"), "capture_queue_capacity")
    requested_packets = _integer(start.get("requested_packets"), "requested_packets")
    if (
        not isinstance(session_id, str)
        or not session_id.startswith("pico-causal-live-")
        or started_monotonic_ns <= 0
        or started_unix_ns <= 0
        or start.get("completed_unix_ns_contract") != "terminal_record_v1"
        or start.get("bind") != "tcp://127.0.0.1:5557"
        or start.get("publisher_sha256") != expected_publisher_sha
        or start.get("capture_worker_sha256") != expected_worker_sha
        or start.get("requested_xrt_module_path") != str(xrt_module)
        or start.get("requested_xrt_module_sha256") != expected_xrt_sha
        or start.get("pico_client_apk_sha256") != args.apk_sha256
        or start.get("pinned_soma_source_root") != "/root/.cache/g1_true23_soma/source"
        or queue_capacity <= 0
        or requested_packets <= 0
        or start.get("termination_contract") != _TERMINATION_CONTRACT
        or start.get("authorization") != _AUTHORIZATION
    ):
        _reject("publisher session_start byte/binding contract mismatch")

    xrt = records[1]
    _exact(
        xrt,
        {
            "schema_version",
            "kind",
            "event",
            "session_id",
            "imported_xrt_module_path",
            "imported_xrt_module_sha256",
            "matches_requested_binding",
            "authorization",
        },
        "xrt_binding_verified",
    )
    _common(xrt, kind=_PUBLISHER_KIND, event="xrt_binding_verified")
    if (
        xrt.get("session_id") != session_id
        or xrt.get("imported_xrt_module_path") != str(xrt_module)
        or xrt.get("imported_xrt_module_sha256") != expected_xrt_sha
        or xrt.get("matches_requested_binding") is not True
        or xrt.get("authorization") != _AUTHORIZATION
    ):
        _reject("imported XRT byte binding mismatch")

    prime = records[2]
    _exact(
        prime,
        {
            "schema_version",
            "kind",
            "event",
            "session_id",
            "prime_report",
            "prime_body_sample_sequence",
            "semantic_sample_emitted",
        },
        "solver_primed_before_stream_clock",
    )
    _common(prime, kind=_PUBLISHER_KIND, event="solver_primed_before_stream_clock")
    if (
        prime.get("session_id") != session_id
        or not isinstance(prime.get("prime_report"), dict)
        or _integer(
            prime.get("prime_body_sample_sequence"),
            "prime_body_sample_sequence",
        )
        < 0
        or prime.get("semantic_sample_emitted") is not False
    ):
        _reject("publisher prime contract mismatch")

    config = records[3]
    _exact(
        config,
        {
            "schema_version",
            "kind",
            "event",
            "session_id",
            "capture_mode",
            "pose_selection_contract",
            "xrt_worker_lifecycle",
            "capture_queue_capacity",
            "capture_queue_overflow_policy",
            "raw_capture_accounting",
            "startup_origin_selection",
            "maximum_selected_source_age_ns",
            "selected_source_age_clock",
            "measured_interpolation_enabled",
            "measured_interpolation_contract",
            "maximum_interpolation_capture_span_ns",
            "maximum_interpolation_left_bracket_age_ns",
            "maximum_interpolation_body_source_span_ns",
            "maximum_selected_body_source_timestamp_delta_ns",
            "control_derivative_contract",
            "control_derivative_period_ns",
            "source_pose_timestamp_relabelled",
            "termination_contract",
            "post_start_exact_contiguous_20ms_required",
            "publisher_max_age_ns",
            "publisher_performance_target_ns",
            "publisher_performance_target_is_safety_gate",
            "downstream_total_freshness_limit_ns",
            "reserved_transport_and_inference_margin_ns",
            "cyclic_gc_was_enabled",
            "cyclic_gc_collected_objects_before_stream",
            "cyclic_gc_enabled_during_stream",
            "python_thread_switch_interval_s",
            "authorization",
        },
        "stream_runtime_configured",
    )
    _common(config, kind=_PUBLISHER_KIND, event="stream_runtime_configured")
    pose_contract = "unique_or_bounded_bracketed_measured_xr24_at_tick_v1"
    interpolation_contract = (
        "linear_xyz_slerp_quaternion_between_consecutive_measured_xr24_v1"
    )
    derivative_contract = "soma_il29_q_50hz_forward_difference_dq_v1"
    capture_mode = "continuous_worker_fifo_bounded_measured_tick_selector_v3"
    if (
        config.get("session_id") != session_id
        or config.get("capture_mode") != capture_mode
        or config.get("pose_selection_contract") != pose_contract
        or config.get("xrt_worker_lifecycle")
        != "single_process_request_prime_then_continuous_stream_v2"
        or config.get("capture_queue_capacity") != queue_capacity
        or config.get("capture_queue_overflow_policy") != "fail_closed"
        or config.get("raw_capture_accounting") != "every_raw_frame_ordered; selected_or_explicitly_superseded"
        or config.get("startup_origin_selection") != "newest_validated_raw_frame_after_runtime_setup_v1"
        or config.get("maximum_selected_source_age_ns") != 20_000_000
        or config.get("selected_source_age_clock") != "local_capture_monotonic"
        or config.get("measured_interpolation_enabled") is not True
        or config.get("measured_interpolation_contract")
        != interpolation_contract
        or config.get("maximum_interpolation_capture_span_ns") != 80_000_000
        or config.get("maximum_interpolation_left_bracket_age_ns")
        != 80_000_000
        or config.get("maximum_interpolation_body_source_span_ns")
        != 80_000_000
        or config.get("maximum_selected_body_source_timestamp_delta_ns") != 80_000_000
        or config.get("control_derivative_contract") != derivative_contract
        or config.get("control_derivative_period_ns") != 20_000_000
        or config.get("source_pose_timestamp_relabelled") is not False
        or config.get("termination_contract") != _TERMINATION_CONTRACT
        or config.get("post_start_exact_contiguous_20ms_required") is not True
        or config.get("publisher_max_age_ns") != 80_000_000
        or config.get("publisher_performance_target_ns") != 40_000_000
        or config.get("publisher_performance_target_is_safety_gate") is not False
        or config.get("downstream_total_freshness_limit_ns") != 100_000_000
        or config.get("reserved_transport_and_inference_margin_ns") != 20_000_000
        or not isinstance(config.get("cyclic_gc_was_enabled"), bool)
        or _integer(
            config.get("cyclic_gc_collected_objects_before_stream"),
            "cyclic_gc_collected_objects_before_stream",
        )
        < 0
        or config.get("cyclic_gc_enabled_during_stream") is not False
        or config.get("python_thread_switch_interval_s") != 0.001
        or config.get("authorization") != _AUTHORIZATION
    ):
        _reject("publisher runtime configuration contract mismatch")

    origin = records[4]
    _exact(
        origin,
        {
            "schema_version",
            "kind",
            "event",
            "session_id",
            "pose_selection_contract",
            "startup_raw_frame_count",
            "startup_raw_frame_indices",
            "startup_superseded_raw_frame_count",
            "startup_superseded_raw_frame_indices",
            "startup_superseded_body_source_sequence_gap_count",
            "selected_raw_frame_index",
            "selected_capture_monotonic_ns",
            "selected_body_sample_sequence",
            "selected_body_sample_timestamp_ns",
            "selected_capture_delta_ns",
            "selected_body_source_timestamp_delta_ns",
            "selected_capture_age_at_tick_ns",
            "selected_source_age_clock",
            "selected_body_source_sequence_gap_count",
            "control_source_frame_index",
            "control_monotonic_ns",
            "raw_bracket_indices",
            "raw_interpolation_alpha",
            "source_pose_timestamp_relabelled",
            "positions_repeated_or_synthesized",
            "semantic_sample_emitted",
            "authorization",
        },
        "stream_origin_selected",
    )
    _common(origin, kind=_PUBLISHER_KIND, event="stream_origin_selected")
    startup_indices = _integer_list(origin.get("startup_raw_frame_indices"), "startup_raw_frame_indices")
    startup_superseded = _integer_list(
        origin.get("startup_superseded_raw_frame_indices"),
        "startup_superseded_raw_frame_indices",
    )
    origin_raw = _integer(origin.get("selected_raw_frame_index"), "origin raw index")
    origin_capture_ns = _integer(origin.get("selected_capture_monotonic_ns"), "origin capture time")
    origin_body_sequence = _integer(origin.get("selected_body_sample_sequence"), "origin body sequence")
    origin_body_timestamp = _integer(origin.get("selected_body_sample_timestamp_ns"), "origin body timestamp")
    if (
        origin.get("session_id") != session_id
        or origin.get("pose_selection_contract") != pose_contract
        or not startup_indices
        or startup_indices != list(range(startup_indices[0], startup_indices[-1] + 1))
        or len(startup_indices) != _integer(origin.get("startup_raw_frame_count"), "startup_raw_frame_count")
        or startup_superseded != startup_indices[:-1]
        or len(startup_superseded)
        != _integer(
            origin.get("startup_superseded_raw_frame_count"),
            "startup_superseded_raw_frame_count",
        )
        or _integer(
            origin.get("startup_superseded_body_source_sequence_gap_count"),
            "startup superseded body gap count",
        )
        < 0
        or origin_raw != startup_indices[-1]
        or origin_capture_ns <= 0
        or origin_body_sequence < 0
        or origin_body_timestamp <= 0
        or origin.get("selected_capture_delta_ns") is not None
        or origin.get("selected_body_source_timestamp_delta_ns") is not None
        or origin.get("selected_capture_age_at_tick_ns") != 0
        or origin.get("selected_source_age_clock") != "local_capture_monotonic"
        or _integer(
            origin.get("selected_body_source_sequence_gap_count"),
            "origin selected body source sequence gap count",
        )
        < 0
        or origin.get("control_source_frame_index") != 0
        or origin.get("control_monotonic_ns") != origin_capture_ns
        or origin.get("raw_bracket_indices") != [origin_raw, origin_raw]
        or _number(origin.get("raw_interpolation_alpha"), "origin interpolation") != 0.0
        or origin.get("source_pose_timestamp_relabelled") is not False
        or origin.get("positions_repeated_or_synthesized") is not False
        or origin.get("semantic_sample_emitted") is not False
        or origin.get("authorization") != _AUTHORIZATION
    ):
        _reject("measured stream origin contract mismatch")

    packet_keys = {
        "schema_version",
        "kind",
        "event",
        "session_id",
        "packet_index",
        "control_source_frame_index",
        "control_monotonic_ns",
        "published_monotonic_ns",
        "publisher_age_ns",
        "fresh_within_40ms",
        "fresh_within_60ms",
        "publisher_performance_target_ns",
        "publisher_performance_target_met",
        "publisher_performance_target_is_safety_gate",
        "publisher_max_age_ns",
        "within_publisher_age_budget",
        "pre_send_age_ns",
        "pose_selection_contract",
        "selected_raw_frame_index",
        "selected_body_sample_sequence",
        "selected_body_sample_timestamp_ns",
        "selected_capture_monotonic_ns",
        "selected_source_delta_ns",
        "selected_capture_delta_ns",
        "selected_body_source_timestamp_delta_ns",
        "maximum_selected_body_source_timestamp_delta_ns",
        "selected_body_sequence_delta_from_previous_selected",
        "selected_capture_age_at_tick_ns",
        "selected_source_age_at_tick_ns",
        "selected_source_age_clock",
        "maximum_selected_source_age_ns",
        "positions_interpolated_from_measured_xr24",
        "measured_interpolation_contract",
        "interpolation_capture_span_ns",
        "interpolation_body_source_span_ns",
        "interpolation_ready_delay_ns",
        "selected_capture_wait_duration_ns",
        "selected_capture_ipc_duration_ns",
        "selected_capture_queue_depth_before_enqueue",
        "selected_capture_queue_depth_after_dequeue",
        "selected_capture_frame_index_delta",
        "selected_raw_body_sample_sequence_delta",
        "selected_body_source_sequence_gap_count",
        "selected_mailbox_batch_size",
        "selected_mailbox_coalesced_prior_frames",
        "superseded_raw_frame_count",
        "superseded_raw_frame_indices",
        "raw_bracket_indices",
        "raw_interpolation_alpha",
        "control_tick_period_ns",
        "control_derivative_contract",
        "control_derivative_period_ns",
        "source_pose_timestamp_relabelled",
        "rolling_compute_duration_ns",
        "target_build_duration_ns",
        "solver_duration_ns",
        "body_term_duration_ns",
        "packet_validation_duration_ns",
        "wire_encode_duration_ns",
        "zmq_send_duration_ns",
        "previous_evidence_write_duration_ns",
        "cyclic_gc_enabled",
        "post_start_contiguous_20ms",
        "packet_validation",
        "wire_sha256",
        "sdk_derivatives_consumed",
        "positions_repeated_or_synthesized",
        "authorization",
    }

    previous_frame: int | None = None
    previous_ns: int | None = None
    previous_published_ns: int | None = None
    previous_selected_raw: int | None = None
    previous_body_sequence: int | None = None
    previous_body_timestamp: int | None = None
    previous_control_body_timestamp: int | None = None
    previous_capture_ns: int | None = None
    selected_packet_indices: list[int] = []
    packet_selected_or_interpolated: list[int | None] = []
    interpolated_packet_count = 0
    packet_superseded_indices: list[int] = []
    packet_superseded_by_packet: list[list[int]] = []
    performance_target_count = 0
    fresh_within_60ms_count = 0
    for index, packet in enumerate(packet_records):
        _exact(packet, packet_keys, f"reference_packet_published[{index}]")
        _common(packet, kind=_PUBLISHER_KIND, event="reference_packet_published")
        frame = _integer(
            packet.get("control_source_frame_index"),
            "control_source_frame_index",
        )
        control_ns = _integer(packet.get("control_monotonic_ns"), "control_monotonic_ns")
        published_ns = _integer(packet.get("published_monotonic_ns"), "published_monotonic_ns")
        age_ns = _integer(packet.get("publisher_age_ns"), "publisher_age_ns")
        pre_send_age_ns = _integer(packet.get("pre_send_age_ns"), "pre_send_age_ns")
        send_duration_ns = _integer(packet.get("zmq_send_duration_ns"), "zmq_send_duration_ns")
        interpolated = packet.get(
            "positions_interpolated_from_measured_xr24"
        )
        if not isinstance(interpolated, bool):
            _reject(f"publisher packet {index} interpolation flag must be boolean")
        selected_raw_value = packet.get("selected_raw_frame_index")
        selected_raw = (
            None
            if interpolated and selected_raw_value is None
            else _integer(selected_raw_value, "selected raw index")
        )
        body_sequence_value = packet.get("selected_body_sample_sequence")
        body_sequence = (
            None
            if interpolated and body_sequence_value is None
            else _integer(body_sequence_value, "selected body sequence")
        )
        body_timestamp = _integer(
            packet.get("selected_body_sample_timestamp_ns"),
            "selected body timestamp",
        )
        capture_ns = _integer(
            packet.get("selected_capture_monotonic_ns"),
            "selected capture time",
        )
        capture_delta_value = packet.get("selected_capture_delta_ns")
        capture_delta = (
            None
            if interpolated and capture_delta_value is None
            else _integer(capture_delta_value, "selected capture delta")
        )
        body_delta = _integer(
            packet.get("selected_body_source_timestamp_delta_ns"),
            "selected body source timestamp delta",
        )
        body_sequence_delta_value = packet.get(
            "selected_body_sequence_delta_from_previous_selected"
        )
        body_sequence_delta = (
            None
            if interpolated and body_sequence_delta_value is None
            else _integer(body_sequence_delta_value, "selected body sequence delta")
        )
        capture_age = _integer(
            packet.get("selected_capture_age_at_tick_ns"),
            "selected capture age",
        )
        queue_before = _integer(
            packet.get("selected_capture_queue_depth_before_enqueue"),
            "queue before",
        )
        queue_after = _integer(
            packet.get("selected_capture_queue_depth_after_dequeue"),
            "queue after",
        )
        mailbox_batch_size = _integer(packet.get("selected_mailbox_batch_size"), "mailbox batch")
        mailbox_coalesced = _integer(
            packet.get("selected_mailbox_coalesced_prior_frames"),
            "mailbox coalesced",
        )
        raw_body_sequence_delta_value = packet.get(
            "selected_raw_body_sample_sequence_delta"
        )
        raw_body_sequence_delta = (
            None
            if interpolated and raw_body_sequence_delta_value is None
            else _integer(raw_body_sequence_delta_value, "raw body delta")
        )
        body_source_gap_count = _integer(
            packet.get("selected_body_source_sequence_gap_count"),
            "body gap",
        )
        superseded = _integer_list(
            packet.get("superseded_raw_frame_indices"),
            "superseded_raw_frame_indices",
        )
        previous_evidence_duration = packet.get("previous_evidence_write_duration_ns")
        validation = packet.get("packet_validation")
        if not isinstance(validation, dict):
            _reject(f"publisher packet {index} validation must be object")
        _exact(
            validation,
            {
                "profile",
                "anchor_source_frame_index",
                "proof_source_frame_index",
                "encoder_lower_body_dim",
                "sdk_derivatives_consumed",
                "sha256",
                "promotion_eligible",
            },
            f"reference_packet_published[{index}].packet_validation",
        )
        capture_frame_delta_value = packet.get(
            "selected_capture_frame_index_delta"
        )
        capture_frame_delta = (
            None
            if interpolated and capture_frame_delta_value is None
            else _integer(capture_frame_delta_value, "capture frame delta")
        )
        raw_brackets = _integer_list(
            packet.get("raw_bracket_indices"), "raw_bracket_indices"
        )
        interpolation_alpha = _number(
            packet.get("raw_interpolation_alpha"), "interpolation alpha"
        )
        interpolation_capture_span = _integer(
            packet.get("interpolation_capture_span_ns"),
            "interpolation capture span",
        )
        interpolation_body_span = _integer(
            packet.get("interpolation_body_source_span_ns"),
            "interpolation body span",
        )
        interpolation_ready_delay = _integer(
            packet.get("interpolation_ready_delay_ns"),
            "interpolation ready delay",
        )
        if interpolated:
            interpolation_fields_valid = (
                selected_raw is None
                and body_sequence is None
                and capture_delta is None
                and body_sequence_delta is None
                and capture_frame_delta is None
                and raw_body_sequence_delta is None
                and len(raw_brackets) == 2
                and raw_brackets[1] == raw_brackets[0] + 1
                and 0.0 < interpolation_alpha < 1.0
                and packet.get("measured_interpolation_contract")
                == interpolation_contract
                and 0 < interpolation_capture_span <= 80_000_000
                and 0 < interpolation_body_span <= 80_000_000
                and 0 <= interpolation_ready_delay < interpolation_capture_span
                and 0 < capture_age <= 40_000_000
                and packet.get("selected_source_age_clock")
                == "local_capture_monotonic_left_bracket"
                and packet.get("maximum_selected_source_age_ns")
                == 40_000_000
                and capture_ns - control_ns == interpolation_ready_delay
                and interpolation_capture_span - capture_age
                == interpolation_ready_delay
                and superseded == []
            )
        else:
            interpolation_fields_valid = (
                selected_raw is not None
                and body_sequence is not None
                and capture_delta is not None
                and body_sequence_delta is not None
                and capture_frame_delta == 1
                and raw_body_sequence_delta is not None
                and raw_brackets == [selected_raw, selected_raw]
                and interpolation_alpha == 0.0
                and packet.get("measured_interpolation_contract") is None
                and interpolation_capture_span == 0
                and interpolation_body_span == 0
                and interpolation_ready_delay == 0
                and 0 <= capture_age <= 20_000_000
                and packet.get("selected_source_age_clock")
                == "local_capture_monotonic"
                and packet.get("maximum_selected_source_age_ns")
                == 20_000_000
                and control_ns - capture_ns == capture_age
            )
        packet_selected_or_interpolated.append(selected_raw)
        if selected_raw is not None:
            selected_packet_indices.append(selected_raw)
        interpolated_packet_count += int(interpolated)
        packet_superseded_indices.extend(superseded)
        packet_superseded_by_packet.append(superseded)
        if (
            packet.get("session_id") != session_id
            or _integer(packet.get("packet_index"), "packet_index") != index
            or published_ns - control_ns != age_ns
            or not 0 <= age_ns <= 80_000_000
            or packet.get("fresh_within_40ms") is not (age_ns <= 40_000_000)
            or packet.get("fresh_within_60ms") is not (age_ns <= 60_000_000)
            or packet.get("publisher_performance_target_ns") != 40_000_000
            or packet.get("publisher_performance_target_met") is not (
                age_ns <= 40_000_000
            )
            or packet.get("publisher_performance_target_is_safety_gate") is not False
            or packet.get("publisher_max_age_ns") != 80_000_000
            or packet.get("within_publisher_age_budget") is not True
            or not 0 <= pre_send_age_ns <= age_ns
            or send_duration_ns < 0
            or pre_send_age_ns + send_duration_ns != age_ns
            or packet.get("pose_selection_contract") != pose_contract
            or not interpolation_fields_valid
            or (selected_raw is not None and selected_raw < 0)
            or (body_sequence is not None and body_sequence < 0)
            or body_timestamp <= 0
            or capture_ns <= 0
            or not 0 < body_delta <= 80_000_000
            or packet.get("maximum_selected_body_source_timestamp_delta_ns") != 80_000_000
            or packet.get("selected_source_delta_ns") != body_delta
            or packet.get("selected_source_age_at_tick_ns") != capture_age
            or _integer(
                packet.get("selected_capture_wait_duration_ns"),
                "capture wait",
            )
            < 0
            or _integer(
                packet.get("selected_capture_ipc_duration_ns"),
                "capture IPC",
            )
            < 0
            or not 0 <= queue_before < queue_capacity
            or not 0 <= queue_after <= queue_capacity
            or body_source_gap_count < 0
            or (
                raw_body_sequence_delta is not None
                and (
                    raw_body_sequence_delta < 1
                    or raw_body_sequence_delta != body_source_gap_count + 1
                )
            )
            or mailbox_batch_size <= 0
            or mailbox_coalesced != mailbox_batch_size - 1
            or len(superseded)
            != _integer(
                packet.get("superseded_raw_frame_count"),
                "superseded count",
            )
            or packet.get("control_tick_period_ns") != 20_000_000
            or packet.get("control_derivative_contract") != derivative_contract
            or packet.get("control_derivative_period_ns") != 20_000_000
            or packet.get("source_pose_timestamp_relabelled") is not False
            or any(
                _integer(packet.get(field), field) < 0
                for field in (
                    "rolling_compute_duration_ns",
                    "target_build_duration_ns",
                    "solver_duration_ns",
                    "body_term_duration_ns",
                    "packet_validation_duration_ns",
                    "wire_encode_duration_ns",
                )
            )
            or (index == 0 and previous_evidence_duration is not None)
            or (
                index > 0
                and (
                    previous_evidence_duration is None
                    or _integer(
                        previous_evidence_duration,
                        "previous evidence write duration",
                    )
                    < 0
                )
            )
            or packet.get("cyclic_gc_enabled") is not False
            or packet.get("post_start_contiguous_20ms") is not True
            or validation.get("profile") != "true23_causal_step1_history_0p02s_v1"
            or validation.get("anchor_source_frame_index") != frame - 1
            or validation.get("proof_source_frame_index") != frame
            or validation.get("encoder_lower_body_dim") != 240
            or validation.get("sdk_derivatives_consumed") is not False
            or not _SHA256.fullmatch(str(validation.get("sha256", "")))
            or validation.get("promotion_eligible") is not False
            or packet.get("sdk_derivatives_consumed") is not False
            or packet.get("positions_repeated_or_synthesized") is not False
            or not _SHA256.fullmatch(str(packet.get("wire_sha256", "")))
            or packet.get("authorization") != _AUTHORIZATION
            or (previous_frame is not None and frame != previous_frame + 1)
            or (previous_ns is not None and control_ns != previous_ns + 20_000_000)
            or (previous_published_ns is not None and published_ns <= previous_published_ns)
            or (
                previous_control_body_timestamp is not None
                and body_timestamp <= previous_control_body_timestamp
            )
            or len(set(superseded)) != len(superseded)
            or (
                selected_raw is not None
                and (
                    (previous_selected_raw is not None and selected_raw <= previous_selected_raw)
                    or (
                        previous_body_sequence is not None
                        and body_sequence - previous_body_sequence
                        != body_sequence_delta
                    )
                    or (
                        previous_body_timestamp is not None
                        and body_timestamp - previous_body_timestamp != body_delta
                    )
                    or (
                        previous_capture_ns is not None
                        and capture_ns - previous_capture_ns != capture_delta
                    )
                    or (
                        previous_selected_raw is not None
                        and superseded
                        != list(range(previous_selected_raw + 1, selected_raw))
                    )
                )
            )
            or (
                selected_raw is None
                and previous_selected_raw is not None
                and (
                    previous_body_timestamp is None
                    or raw_brackets[0] != previous_selected_raw
                    or body_timestamp - previous_body_timestamp != body_delta
                )
            )
        ):
            _reject(f"publisher packet {index} continuity/freshness contract failed")
        performance_target_count += int(age_ns <= 40_000_000)
        fresh_within_60ms_count += int(age_ns <= 60_000_000)
        previous_frame = frame
        previous_ns = control_ns
        previous_published_ns = published_ns
        previous_control_body_timestamp = body_timestamp
        if selected_raw is not None:
            previous_selected_raw = selected_raw
            previous_body_sequence = body_sequence
            previous_body_timestamp = body_timestamp
            previous_capture_ns = capture_ns

    if (
        packet_records[0].get("control_source_frame_index") != 10
        or packet_records[0].get("control_monotonic_ns") != origin_capture_ns + 10 * 20_000_000
    ):
        _reject("first published packet does not prove exact q0..q10 history")

    packet_pairs = [
        (
            _integer(record.get("control_source_frame_index"), "packet frame"),
            _integer(record.get("control_monotonic_ns"), "packet control time"),
        )
        for record in packet_records
    ]
    if runtime:
        expected_active_events = [
            "session_start",
            "artifact_gate_passed",
            "mutation_gate_open",
            "motion_mode_released",
            "lowcmd_publisher_created",
            "first_policy_ready_for_arm",
            "first_armed_policy_command_written",
            "session_complete",
        ]
        if [record.get("event") for record in active_records] != expected_active_events:
            _reject("paired active evidence event order is not exact")
        active_started_ns = _integer(active_records[0].get("monotonic_ns"), "active start monotonic time")
        policy_ready_ns = _integer(active_records[5].get("monotonic_ns"), "policy-ready monotonic time")
        first_command_ns = _integer(active_records[6].get("monotonic_ns"), "first-command monotonic time")
        required_active_ns = _integer(
            active_records[-1].get("required_post_arm_duration_ns"),
            "active required post-arm duration",
        )
        if (
            started_monotonic_ns >= active_started_ns
            or packet_records[0].get("published_monotonic_ns") > policy_ready_ns
            or packet_records[-1].get("published_monotonic_ns") < first_command_ns + required_active_ns
            or len(packet_records) < required_active_ns // 20_000_000 + 10
        ):
            _reject("runtime publisher does not cover the active policy window")
    else:
        shadow_events = [record.get("event") for record in shadow_records]
        if (
            not shadow_events
            or shadow_events[0] != "session_start"
            or shadow_events[-1] != "session_complete"
            or shadow_records[-1].get("passed") is not True
            or shadow_records[0].get("started_monotonic_ns", started_monotonic_ns + 1) > started_monotonic_ns
        ):
            _reject("paired promoted-shadow terminal/order contract failed")
        shadow_pairs = [
            (
                _integer(record.get("control_source_frame_index"), "shadow frame"),
                _integer(record.get("control_monotonic_ns"), "shadow control time"),
            )
            for record in shadow_records
            if record.get("event") in {"causal_warmup_frame", "action_frame"}
        ]
        if not shadow_pairs or len(shadow_pairs) < args.minimum_packets:
            _reject("paired shadow does not contain reviewed causal frame proof")
        try:
            shadow_offset = packet_pairs.index(shadow_pairs[0])
        except ValueError:
            _reject("paired shadow first causal frame is absent from publisher evidence")
        if packet_pairs[shadow_offset : shadow_offset + len(shadow_pairs)] != shadow_pairs:
            _reject("paired shadow causal frames are not one contiguous publisher slice")

    complete = records[-1]
    terminal_keys = {
        "schema_version",
        "kind",
        "event",
        "session_id",
        "requested_packets",
        "completion_reason",
        "stop_requested",
        "stop_signal_number",
        "stop_signal_name",
        "completed_unix_ns_contract",
        "completed_unix_ns",
        "raw_frames",
        "raw_frames_drained",
        "retargeted_samples",
        "published_packets",
        "fresh_packets",
        "all_packets_fresh_within_60ms",
        "all_packets_fresh_within_40ms",
        "packets_meeting_40ms_performance_target",
        "publisher_performance_target_ns",
        "publisher_performance_target_is_safety_gate",
        "all_packets_within_publisher_age_budget",
        "publisher_max_age_ns",
        "capture_mode",
        "pose_selection",
        "background_capture",
        "stream_origin_raw_frame_index",
        "startup_raw_frame_indices",
        "startup_superseded_raw_frame_count",
        "startup_superseded_raw_frame_indices",
        "captured_raw_frame_accounting",
        "captured_raw_frame_accounting_count",
        "all_captured_raw_frames_explicitly_accounted",
        "all_raw_accounting_categories_disjoint",
        "total_body_source_sequence_gap_count",
        "xrt_worker_lifecycle",
        "capture_wait_timing",
        "capture_ipc_timing",
        "capture_queue_depth_after_dequeue_max",
        "rolling_compute_timing",
        "target_build_timing",
        "solver_timing",
        "body_term_timing",
        "evidence_write_timing",
        "startup_frames_dropped_silently",
        "semantic_frames_skipped_between_published_packets",
        "raw_frames_silently_dropped",
        "terminal_unprocessed_selected_samples",
        "terminal_unprocessed_selected_raw_frame_indices",
        "terminal_unconsumed_capture_queue_depth",
        "post_start_exact_contiguous_20ms",
        "cyclic_gc_was_enabled",
        "cyclic_gc_collected_objects_before_stream",
        "cyclic_gc_enabled_during_stream",
        "cyclic_gc_restore_in_finally",
        "imported_xrt_module_path",
        "imported_xrt_module_sha256",
        "publisher_sha256",
        "capture_worker_sha256",
        "pico_client_apk_sha256",
        "authorization",
    }
    _exact(complete, terminal_keys, f"publisher {terminal_event}")
    _common(complete, kind=_PUBLISHER_KIND, event=terminal_event)
    packet_count = len(packet_records)
    raw_frames = _integer(complete.get("raw_frames"), "raw_frames")
    raw_frames_drained = _integer(complete.get("raw_frames_drained"), "raw_frames_drained")
    retargeted_samples = _integer(complete.get("retargeted_samples"), "retargeted_samples")
    if (
        complete.get("session_id") != session_id
        or complete.get("requested_packets") != requested_packets
        or _integer(complete.get("published_packets"), "published_packets") != packet_count
        or _integer(complete.get("fresh_packets"), "fresh_packets") != packet_count
        or complete.get("all_packets_fresh_within_40ms") is not (
            performance_target_count == packet_count
        )
        or complete.get("all_packets_fresh_within_60ms") is not (
            fresh_within_60ms_count == packet_count
        )
        or _integer(
            complete.get("packets_meeting_40ms_performance_target"),
            "packets_meeting_40ms_performance_target",
        )
        != performance_target_count
        or complete.get("publisher_performance_target_ns") != 40_000_000
        or complete.get("publisher_performance_target_is_safety_gate") is not False
        or complete.get("all_packets_within_publisher_age_budget") is not True
        or complete.get("publisher_max_age_ns") != 80_000_000
        or complete.get("capture_mode") != capture_mode
        or (not runtime and retargeted_samples != packet_count + 10)
        or (runtime and retargeted_samples not in {packet_count + 10, packet_count + 11})
        or (
            not runtime
            and (
                requested_packets != packet_count
                or complete.get("completion_reason") != "packet_target_reached"
                or complete.get("stop_requested") is not False
                or complete.get("stop_signal_number") is not None
                or complete.get("stop_signal_name") is not None
            )
        )
        or (
            runtime
            and (
                requested_packets <= packet_count
                or complete.get("completion_reason") != "signal_requested"
                or complete.get("stop_requested") is not True
                or complete.get("stop_signal_number") != 15
                or complete.get("stop_signal_name") != "SIGTERM"
            )
        )
        or raw_frames <= 0
        or not 0 < raw_frames_drained <= raw_frames
        or complete.get("imported_xrt_module_path") != str(xrt_module)
        or complete.get("imported_xrt_module_sha256") != expected_xrt_sha
        or complete.get("pico_client_apk_sha256") != args.apk_sha256
        or complete.get("publisher_sha256") != expected_publisher_sha
        or complete.get("capture_worker_sha256") != expected_worker_sha
        or complete.get("authorization") != _AUTHORIZATION
    ):
        _reject("publisher terminal count/freshness/byte binding mismatch")

    pose = complete.get("pose_selection")
    if not isinstance(pose, dict):
        _reject("terminal pose_selection must be object")
    _exact(
        pose,
        {
            "pose_selection_contract",
            "selected_frames",
            "interpolated_control_frames",
            "total_control_frames",
            "selected_raw_frame_indices",
            "superseded_raw_frames",
            "superseded_raw_frame_indices",
            "selected_capture_delta_timing",
            "selected_body_source_timestamp_delta_timing",
            "max_capture_age_at_tick_ns",
            "maximum_allowed_capture_age_at_tick_ns",
            "positions_repeated_or_synthesized",
            "positions_interpolated_from_measured_xr24",
            "measured_interpolation_contract",
            "maximum_interpolation_capture_span_ns",
            "maximum_interpolation_left_bracket_age_ns",
            "maximum_interpolation_body_source_span_ns",
            "interpolation_capture_span_timing",
            "interpolation_body_source_span_timing",
            "interpolation_ready_delay_timing",
            "interpolation_raw_brackets",
            "pending_raw_frames",
            "pending_raw_frame_indices",
        },
        "terminal pose_selection",
    )
    selected_frames = _integer(pose.get("selected_frames"), "selected_frames")
    interpolated_control_frames = _integer(
        pose.get("interpolated_control_frames"),
        "interpolated_control_frames",
    )
    total_control_frames = _integer(
        pose.get("total_control_frames"), "total_control_frames"
    )
    pose_selected = _integer_list(pose.get("selected_raw_frame_indices"), "selected_raw_frame_indices")
    pose_superseded = _integer_list(
        pose.get("superseded_raw_frame_indices"),
        "terminal superseded_raw_frame_indices",
    )
    pose_pending = _integer_list(pose.get("pending_raw_frame_indices"), "pending_raw_frame_indices")
    capture_delta_count, _, capture_delta_max = _duration_summary(
        pose.get("selected_capture_delta_timing"),
        "selected_capture_delta_timing",
    )
    body_delta_count, _, body_delta_max = _duration_summary(
        pose.get("selected_body_source_timestamp_delta_timing"),
        "selected_body_source_timestamp_delta_timing",
    )
    interpolation_capture_count, _, interpolation_capture_max = (
        _duration_summary(
            pose.get("interpolation_capture_span_timing"),
            "interpolation_capture_span_timing",
        )
    )
    interpolation_body_count, _, interpolation_body_max = _duration_summary(
        pose.get("interpolation_body_source_span_timing"),
        "interpolation_body_source_span_timing",
    )
    interpolation_ready_count, _, interpolation_ready_max = _duration_summary(
        pose.get("interpolation_ready_delay_timing"),
        "interpolation_ready_delay_timing",
    )
    interpolation_brackets_value = pose.get("interpolation_raw_brackets")
    if not isinstance(interpolation_brackets_value, list):
        _reject("interpolation_raw_brackets must be array")
    interpolation_brackets = [
        _integer_list(value, "interpolation_raw_bracket")
        for value in interpolation_brackets_value
    ]
    terminal_pending_samples = _integer(
        complete.get("terminal_unprocessed_selected_samples"),
        "terminal_unprocessed_selected_samples",
    )
    terminal_pending_selected = _integer_list(
        complete.get("terminal_unprocessed_selected_raw_frame_indices"),
        "terminal_unprocessed_selected_raw_frame_indices",
    )
    packet_capture_deltas = [
        _integer(packet.get("selected_capture_delta_ns"), "packet capture delta")
        for packet in packet_records
        if packet.get("selected_capture_delta_ns") is not None
    ]
    maximum_packet_capture_delta = max(packet_capture_deltas, default=0)
    maximum_packet_body_delta = max(
        _integer(
            packet.get("selected_body_source_timestamp_delta_ns"),
            "packet body source timestamp delta",
        )
        for packet in packet_records
    )
    maximum_packet_capture_age = max(
        _integer(
            packet.get("selected_capture_age_at_tick_ns"),
            "packet capture age",
        )
        for packet in packet_records
    )
    terminal_max_capture_age = _integer(pose.get("max_capture_age_at_tick_ns"), "max capture age")
    expected_pose_superseded = [
        raw_index
        for previous_raw, selected_raw in zip(pose_selected, pose_selected[1:], strict=False)
        for raw_index in range(previous_raw + 1, selected_raw)
    ]
    warmup_selected_count = (
        selected_frames
        - len(selected_packet_indices)
        - len(terminal_pending_selected)
    )
    if (
        pose.get("pose_selection_contract") != pose_contract
        or total_control_frames
        != selected_frames + interpolated_control_frames
        or interpolated_control_frames < 0
        or len(pose_selected) != selected_frames
        or len(set(pose_selected)) != selected_frames
        or pose_selected != sorted(pose_selected)
        or not pose_selected
        or pose_selected[0] != origin_raw
        or len(pose_superseded) != _integer(pose.get("superseded_raw_frames"), "superseded_raw_frames")
        or len(set(pose_superseded)) != len(pose_superseded)
        or pose_superseded != sorted(pose_superseded)
        or pose_superseded != expected_pose_superseded
        or len(pose_pending) != _integer(pose.get("pending_raw_frames"), "pending_raw_frames")
        or len(set(pose_pending)) != len(pose_pending)
        or pose_pending != sorted(pose_pending)
        or capture_delta_count != max(0, selected_frames - 1)
        or body_delta_count != max(0, selected_frames - 1)
        or (capture_delta_max is not None and capture_delta_max > 40_000_000)
        or capture_delta_max is None
        or capture_delta_max < maximum_packet_capture_delta
        or (body_delta_max is not None and body_delta_max > 80_000_000)
        or body_delta_max is None
        or body_delta_max < maximum_packet_body_delta
        or not 0 <= terminal_max_capture_age <= 80_000_000
        or terminal_max_capture_age < maximum_packet_capture_age
        or pose.get("maximum_allowed_capture_age_at_tick_ns") != 80_000_000
        or pose.get("positions_repeated_or_synthesized") is not False
        or pose.get("positions_interpolated_from_measured_xr24")
        is not (interpolated_control_frames > 0)
        or pose.get("measured_interpolation_contract")
        != interpolation_contract
        or pose.get("maximum_interpolation_capture_span_ns") != 80_000_000
        or pose.get("maximum_interpolation_left_bracket_age_ns")
        != 80_000_000
        or pose.get("maximum_interpolation_body_source_span_ns")
        != 80_000_000
        or interpolation_capture_count != interpolated_control_frames
        or interpolation_body_count != interpolated_control_frames
        or interpolation_ready_count != interpolated_control_frames
        or len(interpolation_brackets) != interpolated_control_frames
        or any(
            len(bracket) != 2 or bracket[1] != bracket[0] + 1
            for bracket in interpolation_brackets
        )
        or (
            interpolation_capture_max is not None
            and interpolation_capture_max > 80_000_000
        )
        or (
            interpolation_body_max is not None
            and interpolation_body_max > 80_000_000
        )
        or (
            interpolation_ready_max is not None
            and interpolation_ready_max >= 80_000_000
        )
        or total_control_frames != retargeted_samples + terminal_pending_samples
        or len(terminal_pending_selected) > terminal_pending_samples
        or warmup_selected_count < 1
        or pose_selected[
            warmup_selected_count : warmup_selected_count
            + len(selected_packet_indices)
        ]
        != selected_packet_indices
        or (
            terminal_pending_selected
            and terminal_pending_selected
            != pose_selected[-len(terminal_pending_selected) :]
        )
    ):
        _reject("terminal measured pose selection/accounting failed")
    previous_packet_raw = pose_selected[warmup_selected_count - 1]
    for index, (selected_raw, superseded) in enumerate(
        zip(
            packet_selected_or_interpolated,
            packet_superseded_by_packet,
            strict=True,
        )
    ):
        expected = (
            []
            if selected_raw is None
            else list(range(previous_packet_raw + 1, selected_raw))
        )
        if superseded != expected:
            _reject(f"publisher packet {index} does not exactly classify raw gaps")
        if selected_raw is None:
            bracket = _integer_list(
                packet_records[index].get("raw_bracket_indices"),
                "packet raw bracket",
            )
            if bracket[0] != previous_packet_raw:
                _reject(
                    f"publisher packet {index} interpolation left raw mismatch"
                )
        else:
            previous_packet_raw = selected_raw

    background = complete.get("background_capture")
    if not isinstance(background, dict):
        _reject("terminal background_capture must be object")
    _exact(
        background,
        {
            "queue_capacity",
            "captured_frames",
            "max_queue_depth",
            "queue_depth_at_snapshot",
            "queued_raw_frame_indices",
            "capture_wait_mean_ns",
            "capture_wait_max_ns",
            "capture_ipc_mean_ns",
            "capture_ipc_max_ns",
            "source_sequence_gap_count",
            "max_source_sequence_delta",
            "mailbox_batch_count",
            "mailbox_coalesced_prior_frames",
            "queue_overflowed",
            "failure",
            "frames_silently_dropped",
        },
        "terminal background_capture",
    )
    queued_indices = _integer_list(background.get("queued_raw_frame_indices"), "queued_raw_frame_indices")
    queue_depth = _integer(background.get("queue_depth_at_snapshot"), "queue_depth_at_snapshot")
    captured_frames = _integer(background.get("captured_frames"), "background captured_frames")
    max_queue_depth = _integer(background.get("max_queue_depth"), "max_queue_depth")
    background_gap_count = _integer(background.get("source_sequence_gap_count"), "source gap count")
    maximum_source_delta = _integer(background.get("max_source_sequence_delta"), "max source delta")
    mailbox_batch_count = _integer(background.get("mailbox_batch_count"), "mailbox batch count")
    background_mailbox_coalesced = _integer(
        background.get("mailbox_coalesced_prior_frames"),
        "mailbox coalesced",
    )
    if (
        background.get("queue_capacity") != queue_capacity
        or not 0 <= max_queue_depth <= queue_capacity
        or max_queue_depth < 1
        or max_queue_depth < queue_depth
        or max_queue_depth
        < max(
            _integer(
                packet.get("selected_capture_queue_depth_before_enqueue"),
                "packet queue before",
            )
            + 1
            for packet in packet_records
        )
        or queue_depth != len(queued_indices)
        or not 0 <= queue_depth <= queue_capacity
        or len(set(queued_indices)) != len(queued_indices)
        or queued_indices != sorted(queued_indices)
        or captured_frames != raw_frames - 1
        or captured_frames <= 0
        or background.get("queue_overflowed") is not False
        or background.get("failure") is not None
        or background.get("frames_silently_dropped") != 0
        or background_gap_count < 0
        or maximum_source_delta < 1
        or maximum_source_delta > background_gap_count + 1
        or not 1 <= mailbox_batch_count <= raw_frames_drained - 1
        or background_mailbox_coalesced != raw_frames_drained - 1 - mailbox_batch_count
    ):
        _reject("terminal background capture contract failed")
    for mean_field, max_field in (
        ("capture_wait_mean_ns", "capture_wait_max_ns"),
        ("capture_ipc_mean_ns", "capture_ipc_max_ns"),
    ):
        mean_value = _number(background.get(mean_field), mean_field)
        max_value = _number(background.get(max_field), max_field)
        if mean_value < 0 or max_value < 0 or mean_value > max_value:
            _reject(f"terminal {mean_field}/{max_field} are inconsistent")

    accounting = complete.get("captured_raw_frame_accounting")
    if not isinstance(accounting, dict):
        _reject("captured_raw_frame_accounting must be object")
    accounting_keys = {
        "startup_superseded_raw_frame_indices",
        "selector_selected_raw_frame_indices",
        "selector_superseded_raw_frame_indices",
        "selector_pending_raw_frame_indices",
        "terminal_capture_queue_raw_frame_indices",
    }
    _exact(accounting, accounting_keys, "captured_raw_frame_accounting")
    accounting_lists = {key: _integer_list(accounting.get(key), key) for key in accounting_keys}
    flattened = [item for values in accounting_lists.values() for item in values]
    if (
        any(values != sorted(values) or len(set(values)) != len(values) for values in accounting_lists.values())
        or not flattened
        or accounting_lists["startup_superseded_raw_frame_indices"] != startup_superseded
        or accounting_lists["selector_selected_raw_frame_indices"] != pose_selected
        or accounting_lists["selector_superseded_raw_frame_indices"] != pose_superseded
        or accounting_lists["selector_pending_raw_frame_indices"] != pose_pending
        or accounting_lists["terminal_capture_queue_raw_frame_indices"] != queued_indices
        or len(flattened) != raw_frames
        or len(set(flattened)) != raw_frames
        or sorted(flattened) != list(range(min(flattened), max(flattened) + 1))
        or complete.get("captured_raw_frame_accounting_count") != raw_frames
        or complete.get("all_captured_raw_frames_explicitly_accounted") is not True
        or complete.get("all_raw_accounting_categories_disjoint") is not True
        or complete.get("stream_origin_raw_frame_index") != origin_raw
        or complete.get("startup_raw_frame_indices") != startup_indices
        or complete.get("startup_superseded_raw_frame_count") != len(startup_superseded)
        or complete.get("startup_superseded_raw_frame_indices") != startup_superseded
        or raw_frames_drained != raw_frames - queue_depth
        or raw_frames_drained
        != len(startup_superseded) + selected_frames + len(pose_superseded) + len(pose_pending)
        or complete.get("terminal_unconsumed_capture_queue_depth") != queue_depth
    ):
        _reject("terminal raw-frame accounting is not exact/disjoint")

    timing_counts = {
        field: _duration_summary(complete.get(field), field)[0]
        for field in (
            "capture_wait_timing",
            "capture_ipc_timing",
            "rolling_compute_timing",
            "target_build_timing",
            "solver_timing",
            "body_term_timing",
            "evidence_write_timing",
        )
    }
    total_gap_count = _integer(
        complete.get("total_body_source_sequence_gap_count"),
        "total_body_source_sequence_gap_count",
    )
    startup_origin_gap_count = _integer(
        origin.get("startup_superseded_body_source_sequence_gap_count"),
        "startup superseded body gap count",
    ) + _integer(
        origin.get("selected_body_source_sequence_gap_count"),
        "origin selected body source sequence gap count",
    )
    inferred_first_delivery_gap = total_gap_count - background_gap_count
    queue_depth_after_dequeue_max = _integer(
        complete.get("capture_queue_depth_after_dequeue_max"),
        "capture_queue_depth_after_dequeue_max",
    )
    if (
        timing_counts["capture_wait_timing"] != raw_frames_drained
        or timing_counts["capture_ipc_timing"] != raw_frames_drained
        or timing_counts["rolling_compute_timing"] != retargeted_samples
        or timing_counts["target_build_timing"] != retargeted_samples
        or timing_counts["solver_timing"] != retargeted_samples
        or timing_counts["body_term_timing"] != retargeted_samples
        or timing_counts["evidence_write_timing"] != packet_count
        or not 0 <= queue_depth_after_dequeue_max <= queue_capacity
        or queue_depth_after_dequeue_max > max_queue_depth
        or queue_depth_after_dequeue_max
        < max(
            _integer(
                packet.get("selected_capture_queue_depth_after_dequeue"),
                "packet queue after",
            )
            for packet in packet_records
        )
        or total_gap_count < background_gap_count
        or total_gap_count < startup_origin_gap_count
        or not 0 <= inferred_first_delivery_gap <= startup_origin_gap_count
        or (len(startup_indices) == 1 and inferred_first_delivery_gap != startup_origin_gap_count)
        or sum(
            _integer(
                packet.get("selected_body_source_sequence_gap_count"),
                "packet selected body source sequence gap count",
            )
            for packet in packet_records
        )
        > total_gap_count
        or complete.get("xrt_worker_lifecycle")
        != "single_process_request_prime_then_continuous_stream_v2"
        or complete.get("startup_frames_dropped_silently") != 0
        or complete.get("semantic_frames_skipped_between_published_packets") != 0
        or complete.get("raw_frames_silently_dropped") != 0
        or complete.get("post_start_exact_contiguous_20ms") is not True
        or complete.get("cyclic_gc_was_enabled") != config.get("cyclic_gc_was_enabled")
        or complete.get("cyclic_gc_collected_objects_before_stream")
        != config.get("cyclic_gc_collected_objects_before_stream")
        or complete.get("cyclic_gc_enabled_during_stream") is not False
        or complete.get("cyclic_gc_restore_in_finally") is not True
    ):
        _reject("publisher terminal runtime/silent-drop contract failed")
    completed_unix_ns = _integer(complete.get("completed_unix_ns"), "completed_unix_ns")
    wall_age_ns = time.time_ns() - completed_unix_ns
    if (
        complete.get("completed_unix_ns_contract") != "terminal_record_v1"
        or completed_unix_ns < started_unix_ns
        or wall_age_ns < -5_000_000_000
        or wall_age_ns > int(args.max_age_seconds * 1e9)
    ):
        _reject("publisher terminal wall-clock evidence is not fresh")
    if packet_count < args.minimum_packets:
        _reject("publisher packet count is below reviewed minimum")
    result = {
        "publisher_evidence_sha256": _sha256(publisher_path),
        "publisher_sha256": expected_publisher_sha,
        "capture_worker_sha256": expected_worker_sha,
        "xrt_module_sha256": expected_xrt_sha,
        "pico_client_apk_sha256": args.apk_sha256,
        "published_packets": packet_count,
        "terminal_event": terminal_event,
    }
    if runtime:
        assert active_path is not None
        result["active_evidence_sha256"] = _sha256(active_path)
    else:
        assert shadow_path is not None
        result["shadow_evidence_sha256"] = _sha256(shadow_path)
    return result


def validate_active(args: argparse.Namespace) -> dict[str, Any]:
    evidence_path, records = _load_jsonl(args.evidence, "active execution evidence")
    binary = _regular_non_symlink(args.binary, "active controller binary")
    shadow = _regular_non_symlink(args.shadow_evidence, "shadow evidence")
    active_promotion = _regular_non_symlink(args.active_promotion, "active promotion")
    expected_events = [
        "session_start",
        "artifact_gate_passed",
        "mutation_gate_open",
        "motion_mode_released",
        "lowcmd_publisher_created",
        "first_policy_ready_for_arm",
        "first_armed_policy_command_written",
        "session_complete",
    ]
    events = [record.get("event") for record in records]
    if events != expected_events:
        _reject("active evidence event sequence is not exact successful sequence")
    authorization_id = records[0].get("authorization_id")
    previous_ns = 0
    for record, event in zip(records, expected_events, strict=True):
        _common(record, kind=_ACTIVE_KIND, event=event)
        if record.get("authorization_id") != authorization_id:
            _reject("active evidence authorization_id changed")
        monotonic_ns = _integer(record.get("monotonic_ns"), "monotonic_ns")
        if monotonic_ns <= previous_ns:
            _reject("active evidence monotonic time did not advance")
        previous_ns = monotonic_ns
    start = records[0]
    _exact(
        start,
        {
            "schema_version",
            "kind",
            "event",
            "authorization_id",
            "monotonic_ns",
            "controller_binary_path",
            "controller_binary_sha256",
            "encoder_sha256",
            "decoder_sha256",
            "metadata_sha256",
            "promotion_sha256",
            "active_promotion_sha256",
            "live_shadow_evidence_sha256",
            "live_shadow_action_frames",
            "network",
            "pico_endpoint",
            "post_arm_duration_seconds",
            "minimum_policy_command_frames",
            "mutation_gate_open",
            "motion_mode_released",
            "lowcmd_publisher_created",
        },
        "active session_start",
    )
    if (
        authorization_id != args.authorization_id
        or start.get("controller_binary_path") != str(binary)
        or start.get("controller_binary_sha256") != _sha256(binary)
        or start.get("active_promotion_sha256") != _sha256(active_promotion)
        or start.get("live_shadow_evidence_sha256") != _sha256(shadow)
        or _integer(start.get("live_shadow_action_frames"), "live_shadow_action_frames") < 100
        or start.get("network") != "eth0"
        or start.get("pico_endpoint") != "tcp://127.0.0.1:5557"
        or not (20 <= _integer(start.get("post_arm_duration_seconds"), "post_arm_duration_seconds") <= 30)
        or _integer(start.get("minimum_policy_command_frames"), "minimum_policy_command_frames") != 100
        or start.get("mutation_gate_open") is not False
        or start.get("motion_mode_released") is not False
        or start.get("lowcmd_publisher_created") is not False
    ):
        _reject("active session_start contract mismatch")
    for field in (
        "encoder_sha256",
        "decoder_sha256",
        "metadata_sha256",
        "promotion_sha256",
        "active_promotion_sha256",
        "live_shadow_evidence_sha256",
    ):
        if not _SHA256.fullmatch(str(start.get(field, ""))):
            _reject(f"active {field} must be lowercase SHA-256")

    artifact_gate = records[1]
    _exact(
        artifact_gate,
        {
            "schema_version",
            "kind",
            "event",
            "authorization_id",
            "monotonic_ns",
            "onnx_dry_run_passed",
            "artifact_bytes_reverified",
            "active_promotion_authorized",
            "live_shadow_evidence_validated",
        },
        "artifact_gate_passed",
    )
    if any(
        artifact_gate.get(field) is not True
        for field in (
            "onnx_dry_run_passed",
            "artifact_bytes_reverified",
            "active_promotion_authorized",
            "live_shadow_evidence_validated",
        )
    ):
        _reject("active artifact gate did not pass every condition")

    mutation_gate = records[2]
    _exact(
        mutation_gate,
        {
            "schema_version",
            "kind",
            "event",
            "authorization_id",
            "monotonic_ns",
            "stable_advancing_lowstate_samples",
            "required_mode_machine",
            "crc_valid",
        },
        "mutation_gate_open",
    )
    if (
        mutation_gate.get("stable_advancing_lowstate_samples") != 5
        or mutation_gate.get("required_mode_machine") != 4
        or mutation_gate.get("crc_valid") is not True
    ):
        _reject("active mutation gate contract mismatch")

    motion_gate = records[3]
    _exact(
        motion_gate,
        {
            "schema_version",
            "kind",
            "event",
            "authorization_id",
            "monotonic_ns",
            "post_release_mode_name_empty",
        },
        "motion_mode_released",
    )
    if motion_gate.get("post_release_mode_name_empty") is not True:
        _reject("motion-mode release was not proven")

    publisher_gate = records[4]
    _exact(
        publisher_gate,
        {
            "schema_version",
            "kind",
            "event",
            "authorization_id",
            "monotonic_ns",
            "topic",
            "writes_before_event",
        },
        "lowcmd_publisher_created",
    )
    if publisher_gate.get("topic") != "rt/lowcmd" or publisher_gate.get("writes_before_event") != 0:
        _reject("LowCmd publisher creation ordering mismatch")

    policy_ready = records[5]
    _exact(
        policy_ready,
        {
            "schema_version",
            "kind",
            "event",
            "authorization_id",
            "monotonic_ns",
            "real_history_frames",
            "policy_freshness_limit_ns",
        },
        "first_policy_ready_for_arm",
    )
    if (
        policy_ready.get("real_history_frames") != 10
        or policy_ready.get("policy_freshness_limit_ns") != 100_000_000
    ):
        _reject("first policy-ready contract mismatch")

    first_command = records[6]
    _exact(
        first_command,
        {
            "schema_version",
            "kind",
            "event",
            "authorization_id",
            "monotonic_ns",
            "policy_command_frame",
            "native_action_dof",
            "feedforward_tau_zero",
        },
        "first_armed_policy_command_written",
    )
    if (
        first_command.get("policy_command_frame") != 1
        or first_command.get("native_action_dof") != 23
        or first_command.get("feedforward_tau_zero") is not True
    ):
        _reject("first armed policy command contract mismatch")

    terminal = records[-1]
    _exact(
        terminal,
        {
            "schema_version",
            "kind",
            "event",
            "authorization_id",
            "monotonic_ns",
            "passed",
            "armed_transition_observed",
            "policy_command_frames",
            "minimum_policy_command_frames",
            "damping_frames_after_stop",
            "required_damping_frames_after_stop",
            "maximum_target_delta_from_state_rad",
            "maximum_target_slew_rad",
            "maximum_abs_predicted_effort_nm",
            "maximum_abs_feedforward_tau_nm",
            "final_fault",
            "stop_reason",
            "post_arm_elapsed_ns",
            "required_post_arm_duration_ns",
            "inference_error",
            "writer_error",
            "publisher_write_failed",
        },
        "active session_complete",
    )
    action_frames = _integer(terminal.get("policy_command_frames"), "policy_command_frames")
    damping_frames = _integer(terminal.get("damping_frames_after_stop"), "damping_frames_after_stop")
    required_duration_ns = _integer(
        terminal.get("required_post_arm_duration_ns"),
        "required_post_arm_duration_ns",
    )
    elapsed_ns = _integer(terminal.get("post_arm_elapsed_ns"), "post_arm_elapsed_ns")
    if (
        terminal.get("passed") is not True
        or terminal.get("armed_transition_observed") is not True
        or action_frames < 100
        or terminal.get("minimum_policy_command_frames") != 100
        or damping_frames < 250
        or terminal.get("required_damping_frames_after_stop") != 250
        or _number(
            terminal.get("maximum_abs_feedforward_tau_nm"),
            "maximum_abs_feedforward_tau_nm",
        )
        != 0.0
        or terminal.get("final_fault") != "operator_stop"
        or terminal.get("stop_reason") != "reviewed_post_arm_duration_complete"
        or required_duration_ns != start.get("post_arm_duration_seconds") * 1_000_000_000
        or elapsed_ns < required_duration_ns
        or terminal.get("inference_error") != ""
        or terminal.get("writer_error") != ""
        or terminal.get("publisher_write_failed") is not False
    ):
        _reject("active terminal does not prove successful actuation and safe stop")
    target_delta = _number(
        terminal.get("maximum_target_delta_from_state_rad"),
        "maximum_target_delta_from_state_rad",
    )
    target_slew = _number(terminal.get("maximum_target_slew_rad"), "maximum_target_slew_rad")
    predicted_effort = _number(
        terminal.get("maximum_abs_predicted_effort_nm"),
        "maximum_abs_predicted_effort_nm",
    )
    if (
        target_delta < 0
        or target_slew < 0
        or target_slew > 0.0005 + 1e-9
        or predicted_effort < 0
        or predicted_effort > 34.75 + 1e-6
    ):
        _reject("active target/slew/predicted-effort safety metrics failed")
    return {
        "active_evidence_sha256": _sha256(evidence_path),
        "authorization_id": authorization_id,
        "policy_command_frames": action_frames,
        "damping_frames_after_stop": damping_frames,
    }


def validate_runtime_publisher(args: argparse.Namespace) -> dict[str, Any]:
    active_result = validate_active(
        argparse.Namespace(
            evidence=args.active_evidence,
            binary=args.binary,
            shadow_evidence=args.shadow_evidence,
            active_promotion=args.active_promotion,
            authorization_id=args.authorization_id,
        )
    )
    runtime_args = argparse.Namespace(
        runtime=True,
        publisher_evidence=args.publisher_evidence,
        active_evidence=args.active_evidence,
        publisher_source=args.publisher_source,
        worker_source=args.worker_source,
        xrt_module=args.xrt_module,
        apk_sha256=args.apk_sha256,
        minimum_packets=args.minimum_packets,
        max_age_seconds=args.max_age_seconds,
    )
    publisher_result = validate_publisher(runtime_args)
    if publisher_result.get("active_evidence_sha256") != active_result.get("active_evidence_sha256"):
        _reject("active evidence changed between linked validations")
    return {**publisher_result, **active_result}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    publisher = subparsers.add_parser("publisher")
    publisher.add_argument("--publisher-evidence", type=Path, required=True)
    publisher.add_argument("--shadow-evidence", type=Path, required=True)
    publisher.add_argument("--publisher-source", type=Path, required=True)
    publisher.add_argument("--worker-source", type=Path, required=True)
    publisher.add_argument("--xrt-module", type=Path, required=True)
    publisher.add_argument("--apk-sha256", required=True)
    publisher.add_argument("--minimum-packets", type=int, default=100)
    publisher.add_argument("--max-age-seconds", type=float, default=300.0)
    runtime = subparsers.add_parser("publisher-runtime")
    runtime.add_argument("--publisher-evidence", type=Path, required=True)
    runtime.add_argument("--active-evidence", type=Path, required=True)
    runtime.add_argument("--publisher-source", type=Path, required=True)
    runtime.add_argument("--worker-source", type=Path, required=True)
    runtime.add_argument("--xrt-module", type=Path, required=True)
    runtime.add_argument("--apk-sha256", required=True)
    runtime.add_argument("--binary", type=Path, required=True)
    runtime.add_argument("--shadow-evidence", type=Path, required=True)
    runtime.add_argument("--active-promotion", type=Path, required=True)
    runtime.add_argument("--authorization-id", required=True)
    runtime.add_argument("--minimum-packets", type=int, default=100)
    runtime.add_argument("--max-age-seconds", type=float, default=300.0)
    active = subparsers.add_parser("active")
    active.add_argument("--evidence", type=Path, required=True)
    active.add_argument("--binary", type=Path, required=True)
    active.add_argument("--shadow-evidence", type=Path, required=True)
    active.add_argument("--active-promotion", type=Path, required=True)
    active.add_argument("--authorization-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "publisher":
            result = validate_publisher(args)
        elif args.command == "publisher-runtime":
            result = validate_runtime_publisher(args)
        else:
            result = validate_active(args)
    except (EvidenceError, OSError) as exc:
        print(f"[BLOCKED] Stage-1 evidence: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
