#!/usr/bin/env python3
"""Authorize stage-one gantry control from exact causal live-shadow evidence."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_23dof_contract import ROBOT_MODEL
from gear_sonic.utils.g1_true23_causal_promotion import (
    APPLIED_SAFE_NATIVE_ACTION,
    _read_strict_json,
    _regular_file,
    verify_causal_promotion,
    write_new_json,
)

SCHEMA_VERSION = 2
KIND = "g1_true23_causal_gantry_active_promotion"
AUTHORIZATION_PHRASE = "I_CONFIRM_G1_TRUE23_STAGE1_GANTRY"
EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_KIND = "g1_true23_integrated_live_shadow_evidence"
REFERENCE_PROFILE = "true23_causal_step1_history_0p02s_v1"
REFERENCE_CONTRACT_SHA256 = (
    "bd046467325fe7f7f585fd692f01223ed7a3b2742c51ed414072c98fe12806f7"
)
SAFE_TARGET_TRANSFORM_SHA256 = (
    "74f2277042da83e81ee8a37d90ba6e723bf6e0651ee9b9987ee7effc78fca516"
)
MIN_ACTION_FRAMES = 100
CAUSAL_WARMUP_FRAMES = 9
CONTROL_PERIOD_NS = 20_000_000
MAX_EVIDENCE_AGE_S = 300.0
_AUTHORIZATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_START_KEYS = {
    "schema_version",
    "kind",
    "event",
    "started_monotonic_ns",
    "reference_profile",
    "reference_contract_sha256",
    "artifact_class",
    "decoder_output_semantics",
    "external_safe_target_transform_applied",
    "encoder_sha256",
    "decoder_sha256",
    "metadata_sha256",
    "promotion_sha256",
    "network",
    "pico_endpoint",
    "requested_action_frames",
    "robot_mutation_authorized",
}
_LOWSTATE_KEYS = {
    "schema_version",
    "kind",
    "event",
    "mode_machine",
    "crc_rejects",
    "history_warmup_span_ns",
}
_WARMUP_KEYS = {
    "schema_version",
    "kind",
    "event",
    "control_source_frame_index",
    "control_monotonic_ns",
    "history_samples",
    "packet_age_ns",
    "sdk_derivatives_consumed",
}
_ACTION_KEYS = {
    "schema_version",
    "kind",
    "event",
    "action_frame_index",
    "control_source_frame_index",
    "pico_anchor_source_frame_index",
    "pico_anchor_monotonic_ns",
    "control_monotonic_ns",
    "received_monotonic_ns",
    "produced_monotonic_ns",
    "packet_age_ns",
    "end_to_end_age_ns",
    "lowstate_age_ns",
    "inference_ns",
    "native_action",
    "decoder_output_semantics",
    "external_safe_target_transform_applied",
    "normalized_max_abs",
    "target_position_min_margin_rad",
    "target_limit_violations",
    "slew_checked",
    "target_slew_ratio_max",
    "target_slew_violations",
    "sdk_derivatives_consumed",
    "accepted",
}
_COMPLETE_KEYS = {
    "schema_version",
    "kind",
    "event",
    "passed",
    "action_frames",
    "causal_warmup_frames",
    "maximum_normalized_abs",
    "minimum_target_position_margin_rad",
    "maximum_target_slew_ratio",
    "crc_rejects",
    "robot_mutation_authorized",
}


def _exact(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{context} fields do not match exact schema")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _load_jsonl(path: Path) -> tuple[Path, list[Mapping[str, Any]], str]:
    resolved = _regular_file(path, "live-shadow evidence")
    if resolved.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("live-shadow evidence exceeds 16 MiB")
    initial_sha256 = sha256_file(resolved)
    records: list[Mapping[str, Any]] = []

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"live-shadow evidence contains duplicate key: {key}")
            result[key] = value
        return result

    with resolved.open("rb") as stream:
        for index, line in enumerate(stream):
            if not line.endswith(b"\n"):
                raise ValueError("live-shadow JSONL line lacks newline")
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=reject_duplicates,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON token: {token}")
                    ),
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"live-shadow JSONL line {index} is invalid") from exc
            if not isinstance(value, Mapping):
                raise ValueError("live-shadow JSONL records must be objects")
            records.append(value)
    if sha256_file(resolved) != initial_sha256:
        raise ValueError("live-shadow evidence changed while loading")
    return resolved, records, initial_sha256


def _event_identity(record: Mapping[str, Any], event: str, context: str) -> None:
    if (
        record["schema_version"] != EVIDENCE_SCHEMA_VERSION
        or record["kind"] != EVIDENCE_KIND
        or record["event"] != event
    ):
        raise ValueError(f"{context} identity mismatch")


def _validate_control_frame(
    record: Mapping[str, Any],
    *,
    previous_frame: int | None,
    previous_time_ns: int | None,
    context: str,
) -> tuple[int, int]:
    frame = _integer(record["control_source_frame_index"], f"{context} frame", minimum=0)
    time_ns = _integer(record["control_monotonic_ns"], f"{context} time", minimum=1)
    if previous_frame is not None and frame != previous_frame + 1:
        raise ValueError("live-shadow control frames are not consecutive")
    if previous_time_ns is not None and time_ns != previous_time_ns + CONTROL_PERIOD_NS:
        raise ValueError("live-shadow control timestamps are not exact 20 ms")
    return frame, time_ns


def validate_causal_live_shadow_evidence(
    evidence_path: Path,
    *,
    promotion_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    metadata_path: Path,
) -> Mapping[str, Any]:
    """Validate promoted causal JSONL shadow PASS and return computed summary."""

    evidence, records, evidence_sha256 = _load_jsonl(evidence_path)
    if len(records) < 2 + CAUSAL_WARMUP_FRAMES + MIN_ACTION_FRAMES + 1:
        raise ValueError("live-shadow evidence is too short")
    start = _exact(records[0], _START_KEYS, "session_start")
    _event_identity(start, "session_start", "session_start")
    requested = _integer(
        start["requested_action_frames"], "requested_action_frames", minimum=MIN_ACTION_FRAMES
    )
    if requested > 100_000:
        raise ValueError("requested_action_frames exceeds producer bound")
    expected_start = {
        "reference_profile": REFERENCE_PROFILE,
        "reference_contract_sha256": REFERENCE_CONTRACT_SHA256,
        "artifact_class": "promoted_shadow",
        "decoder_output_semantics": APPLIED_SAFE_NATIVE_ACTION,
        "external_safe_target_transform_applied": False,
        "encoder_sha256": sha256_file(_regular_file(encoder_path, "encoder ONNX")),
        "decoder_sha256": sha256_file(_regular_file(decoder_path, "decoder ONNX")),
        "metadata_sha256": sha256_file(_regular_file(metadata_path, "metadata")),
        "promotion_sha256": sha256_file(_regular_file(promotion_path, "promotion")),
        "robot_mutation_authorized": False,
    }
    if any(start.get(key) != value for key, value in expected_start.items()):
        raise ValueError("live-shadow session_start artifact/safety binding mismatch")
    _integer(start["started_monotonic_ns"], "started_monotonic_ns", minimum=1)
    if not isinstance(start["network"], str) or not start["network"]:
        raise ValueError("live-shadow network is missing")
    if not isinstance(start["pico_endpoint"], str) or not start[
        "pico_endpoint"
    ].startswith(("tcp://", "ipc://")):
        raise ValueError("live-shadow PICO endpoint is invalid")

    lowstate = _exact(records[1], _LOWSTATE_KEYS, "lowstate_gate_open")
    _event_identity(lowstate, "lowstate_gate_open", "lowstate_gate_open")
    if (
        _integer(lowstate["mode_machine"], "mode_machine") != 4
        or _integer(lowstate["crc_rejects"], "crc_rejects") != 0
        or _integer(lowstate["history_warmup_span_ns"], "history_warmup_span_ns")
        != 40_000_000
    ):
        raise ValueError("live-shadow LowState mode/CRC/history gate failed")

    expected_record_count = 2 + CAUSAL_WARMUP_FRAMES + requested + 1
    if len(records) != expected_record_count:
        raise ValueError("live-shadow evidence event count is not exact")
    previous_frame: int | None = None
    previous_time_ns: int | None = None
    for index in range(CAUSAL_WARMUP_FRAMES):
        record = _exact(records[2 + index], _WARMUP_KEYS, f"warmup[{index}]")
        _event_identity(record, "causal_warmup_frame", f"warmup[{index}]")
        previous_frame, previous_time_ns = _validate_control_frame(
            record,
            previous_frame=previous_frame,
            previous_time_ns=previous_time_ns,
            context=f"warmup[{index}]",
        )
        packet_age = _integer(record["packet_age_ns"], "warmup packet_age_ns")
        if (
            _integer(record["history_samples"], "warmup history_samples")
            != index + 1
            or record["sdk_derivatives_consumed"] is not False
            or not -5_000_000 <= packet_age <= 100_000_000
        ):
            raise ValueError("live-shadow causal warmup gate failed")

    maximum_abs = 0.0
    minimum_margin = math.inf
    maximum_slew_ratio = 0.0
    for index in range(requested):
        record = _exact(
            records[2 + CAUSAL_WARMUP_FRAMES + index],
            _ACTION_KEYS,
            f"action_frame[{index}]",
        )
        _event_identity(record, "action_frame", f"action_frame[{index}]")
        previous_frame, previous_time_ns = _validate_control_frame(
            record,
            previous_frame=previous_frame,
            previous_time_ns=previous_time_ns,
            context=f"action_frame[{index}]",
        )
        if _integer(record["action_frame_index"], "action frame index") != index:
            raise ValueError("live-shadow action indices are not consecutive")
        anchor_frame = _integer(
            record["pico_anchor_source_frame_index"], "PICO anchor frame", minimum=0
        )
        anchor_ns = _integer(
            record["pico_anchor_monotonic_ns"], "PICO anchor time", minimum=1
        )
        received_ns = _integer(record["received_monotonic_ns"], "received time", minimum=1)
        produced_ns = _integer(record["produced_monotonic_ns"], "produced time", minimum=1)
        packet_age = _integer(record["packet_age_ns"], "packet age")
        end_to_end = _integer(record["end_to_end_age_ns"], "end-to-end age")
        lowstate_age = _integer(record["lowstate_age_ns"], "LowState age")
        inference_ns = _integer(record["inference_ns"], "inference time", minimum=0)
        if (
            anchor_frame + 1 != previous_frame
            or anchor_ns + CONTROL_PERIOD_NS != previous_time_ns
            or received_ns - previous_time_ns != packet_age
            or produced_ns - previous_time_ns != end_to_end
            or produced_ns < received_ns
            or not -5_000_000 <= packet_age <= 100_000_000
            or not -5_000_000 <= end_to_end <= 100_000_000
            or not -5_000_000 <= lowstate_age <= 40_000_000
            or inference_ns > 20_000_000
        ):
            raise ValueError("live-shadow timing/causal-anchor gate failed")
        action = record["native_action"]
        if not isinstance(action, list) or len(action) != 23:
            raise ValueError("live-shadow native action must contain 23 values")
        action_values = [
            _number(value, f"action_frame[{index}].native_action") for value in action
        ]
        normalized = _number(record["normalized_max_abs"], "normalized_max_abs")
        margin = _number(
            record["target_position_min_margin_rad"], "target position margin"
        )
        slew_ratio = _number(record["target_slew_ratio_max"], "target slew ratio")
        if (
            not math.isclose(normalized, max(abs(value) for value in action_values), abs_tol=1e-6)
            or not 0.0 <= normalized <= 20.0
            or margin < 0.0
            or _integer(record["target_limit_violations"], "target limit violations")
            != 0
            or record["slew_checked"] is not (index > 0)
            or not 0.0 <= slew_ratio <= 1.0
            or _integer(record["target_slew_violations"], "target slew violations")
            != 0
            or record["sdk_derivatives_consumed"] is not False
            or record["decoder_output_semantics"] != APPLIED_SAFE_NATIVE_ACTION
            or record["external_safe_target_transform_applied"] is not False
            or record["accepted"] is not True
        ):
            raise ValueError("live-shadow action safety gate failed")
        maximum_abs = max(maximum_abs, normalized)
        minimum_margin = min(minimum_margin, margin)
        maximum_slew_ratio = max(maximum_slew_ratio, slew_ratio)

    complete = _exact(records[-1], _COMPLETE_KEYS, "session_complete")
    _event_identity(complete, "session_complete", "session_complete")
    if (
        complete["passed"] is not True
        or _integer(complete["action_frames"], "completed action_frames") != requested
        or _integer(complete["causal_warmup_frames"], "completed warmup frames")
        != CAUSAL_WARMUP_FRAMES
        or _integer(complete["crc_rejects"], "completed CRC rejects") != 0
        or complete["robot_mutation_authorized"] is not False
        or not math.isclose(
            _number(complete["maximum_normalized_abs"], "maximum normalized abs"),
            maximum_abs,
            abs_tol=1e-6,
        )
        or not math.isclose(
            _number(
                complete["minimum_target_position_margin_rad"],
                "minimum target margin",
            ),
            minimum_margin,
            abs_tol=1e-6,
        )
        or not math.isclose(
            _number(complete["maximum_target_slew_ratio"], "maximum slew ratio"),
            maximum_slew_ratio,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("live-shadow terminal PASS summary mismatch")
    if sha256_file(evidence) != evidence_sha256:
        raise ValueError("live-shadow evidence changed during validation")
    return {
        "evidence_sha256": evidence_sha256,
        "action_frames": requested,
        "maximum_normalized_abs": maximum_abs,
        "minimum_target_position_margin_rad": minimum_margin,
        "maximum_target_slew_ratio": maximum_slew_ratio,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-onnx", type=Path, required=True)
    parser.add_argument("--decoder-onnx", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--live-shadow-evidence", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--gantry-authorize", required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _require_fresh_evidence(path: Path) -> None:
    evidence = _regular_file(path, "live-shadow evidence")
    age_s = time.time() - evidence.stat().st_mtime
    if age_s < -5.0 or age_s > MAX_EVIDENCE_AGE_S:
        raise ValueError(f"live-shadow evidence is not fresh (age={age_s:.3f}s)")


def _body(args: argparse.Namespace) -> dict[str, Any]:
    if args.gantry_authorize != AUTHORIZATION_PHRASE:
        raise ValueError("exact explicit gantry authorization phrase is required")
    if not _AUTHORIZATION_ID_RE.fullmatch(args.authorization_id):
        raise ValueError("authorization-id must be 8-128 safe identifier characters")
    input_paths = {
        "promotion": _regular_file(args.promotion, "promotion"),
        "checkpoint": _regular_file(args.checkpoint, "checkpoint"),
        "encoder": _regular_file(args.encoder_onnx, "encoder ONNX"),
        "decoder": _regular_file(args.decoder_onnx, "decoder ONNX"),
        "metadata": _regular_file(args.metadata, "metadata"),
        "full_report": _regular_file(args.full_report, "full report"),
        "live_shadow": _regular_file(
            args.live_shadow_evidence, "live-shadow evidence"
        ),
    }
    if len(set(input_paths.values())) != len(input_paths):
        raise ValueError("causal gantry authorizer inputs must be distinct")
    initial_hashes = {role: sha256_file(path) for role, path in input_paths.items()}
    promotion = verify_causal_promotion(
        input_paths["promotion"],
        checkpoint_path=input_paths["checkpoint"],
        encoder_path=input_paths["encoder"],
        decoder_path=input_paths["decoder"],
        metadata_path=input_paths["metadata"],
        full_report_path=input_paths["full_report"],
    )
    if (
        promotion.get("deployment_bytes_authorized") is not True
        or promotion.get("active_motor_control_authorized") is not False
        or promotion.get("gantry_or_rated_support_required") is not True
        or promotion.get("free_standing_authorized") is not False
    ):
        raise ValueError("base promotion is not exact causal gantry-stage source")
    live_summary = validate_causal_live_shadow_evidence(
        input_paths["live_shadow"],
        promotion_path=input_paths["promotion"],
        encoder_path=input_paths["encoder"],
        decoder_path=input_paths["decoder"],
        metadata_path=input_paths["metadata"],
    )
    source = promotion["source_artifact"]
    campaign = promotion["full_campaign_evidence"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "robot_model": ROBOT_MODEL,
        "decoder_output_dim": 23,
        "mode_machine": 4,
        "action_clip_value": 20.0,
        "deployment_ready": True,
        "active_motor_control_authorized": True,
        "gantry_authorized": True,
        "free_standing_authorized": False,
        "decoder_output_semantics": APPLIED_SAFE_NATIVE_ACTION,
        "previous_action_semantics": APPLIED_SAFE_NATIVE_ACTION,
        "external_safe_target_transform_allowed": False,
        "safe_target_transform_sha256": SAFE_TARGET_TRANSFORM_SHA256,
        "source_promotion_sha256": initial_hashes["promotion"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "lineage_sha256": source["lineage_sha256"],
        "policy_state_sha256": source["policy_state_sha256"],
        "encoder_onnx_sha256": source["encoder_onnx_sha256"],
        "decoder_onnx_sha256": source["decoder_onnx_sha256"],
        "metadata_sha256": source["metadata_sha256"],
        "full_campaign_aggregate_sha256": campaign["aggregate_report_sha256"],
        "full_campaign_shard_manifest_sha256": campaign["shard_manifest_sha256"],
        "live_shadow_evidence_sha256": live_summary["evidence_sha256"],
        "authorization_id": args.authorization_id,
    }
    if {
        role: sha256_file(path) for role, path in input_paths.items()
    } != initial_hashes:
        raise ValueError("causal gantry authorizer input changed during verification")
    return body


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.verify_only:
        _require_fresh_evidence(args.live_shadow_evidence)
    body = _body(args)
    if args.verify_only:
        output = _regular_file(args.output, "active gantry sidecar")
        actual = _read_strict_json(output, "active gantry sidecar")
        if actual != body:
            raise ValueError("active gantry sidecar differs from re-verified evidence")
        print(f"causal active gantry sidecar verified: {output}")  # noqa: T201
        return 0
    _require_fresh_evidence(args.live_shadow_evidence)
    output = write_new_json(args.output, body)
    print(f"causal active gantry sidecar created: {output}")  # noqa: T201
    print("free-standing authorization: false")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
