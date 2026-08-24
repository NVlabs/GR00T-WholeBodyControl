"""Benchmark exact-config rolling SOMA against pinned batch replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from gear_sonic.utils.g1_23dof_pico_retargeted_producer import (
    validate_raw_capture,
)
from gear_sonic.utils.g1_23dof_xr24_soma_adapter import (
    _execute_soma,
    _g1_reference_body_terms,
    resample_raw_capture_50hz,
)
from gear_sonic.utils.g1_23dof_xr24_soma_stream import (
    CausalHistorySemanticProducer,
    ExperimentalSomaRollingRetargeter,
    IncrementalCaptureResampler,
    PinnedSomaRollingRetargeter,
    all_released_profile_latency_proofs,
    causal_history_reference_terms,
    causal_history_stream_contract,
    compare_rolling_to_batch,
    validate_causal_history_packet,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, value: Any) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        resolved,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        resolved.unlink(missing_ok=True)
        raise


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[int | float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("distribution needs finite values")
    return {
        "sample_count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only exact-config rolling SOMA benchmark. Opens no XR, ZMQ, DDS, ADB, or robot channel."
        )
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--neutral-calibration-capture",
        type=Path,
        help="Separate real capture supplying the required first 10 neutral-standing frames.",
    )
    parser.add_argument(
        "--soma-source-root",
        type=Path,
        default=Path("/root/.cache/g1_true23_soma/source"),
    )
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--require-deadline",
        action="store_true",
        help="Exit 2 when equivalence or every-sample 20 ms deadline fails.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--causal-packets-output", type=Path)
    parser.add_argument(
        "--experimental-solver-iterations",
        type=int,
        choices=(12, 16),
        help=("Distinct non-promotable SOMA profile; never treated as pinned 24-iteration equivalence."),
    )
    return parser


def _compare_body_terms(
    rolling_rows: list[dict[str, Any]],
    batch_terms: list[dict[str, Any]],
    *,
    absolute_tolerance: float,
) -> dict[str, Any]:
    if len(rolling_rows) != len(batch_terms):
        raise ValueError("rolling and batch body-term counts differ")
    fields = (
        "vr_3point_local_target",
        "vr_3point_local_orn_target",
        "reference_anchor_quaternion_xyzw",
    )
    differences = [
        abs(float(rolling_value) - float(batch_value))
        for rolling_row, batch in zip(rolling_rows, batch_terms, strict=True)
        for field in fields
        for rolling_value, batch_value in zip(
            rolling_row["body_term"][field],
            batch[field],
            strict=True,
        )
    ]
    aligned = all(
        rolling["body_term"]["source_frame_index"] == batch["source_frame_index"]
        and rolling["body_term"]["reference_monotonic_ns"] == batch["reference_monotonic_ns"]
        for rolling, batch in zip(rolling_rows, batch_terms, strict=True)
    )
    maximum = max(differences, default=0.0)
    return {
        "sample_count": len(rolling_rows),
        "value_count": len(differences),
        "absolute_tolerance": absolute_tolerance,
        "timestamp_aligned": aligned,
        "max_abs_error": maximum,
        "exact_within_tolerance": aligned and maximum <= absolute_tolerance,
        "promotion_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    capture_path = args.capture.resolve()
    capture = _load_json(capture_path)
    capture_validation = validate_raw_capture(capture)
    calibration_path = (
        capture_path if args.neutral_calibration_capture is None else args.neutral_calibration_capture.resolve()
    )
    calibration_capture = _load_json(calibration_path)
    calibration_validation = validate_raw_capture(calibration_capture)
    calibration_samples = resample_raw_capture_50hz(calibration_capture)
    neutral_body_pose_frames = [sample["body_poses"] for sample in calibration_samples[:10]]
    identity = {
        key: capture[key]
        for key in (
            "schema_version",
            "kind",
            "session_id",
            "source",
            "authorization",
        )
    }
    resampler = IncrementalCaptureResampler(identity)
    streamed_samples = [sample for frame in capture["frames"] for sample in resampler.push(frame)]

    batch_started_ns = time.monotonic_ns()
    batch_rows, batch_timing, batch_pipeline = _execute_soma(
        capture,
        soma_source_root=args.soma_source_root.resolve(),
        neutral_body_pose_frames=neutral_body_pose_frames,
    )
    batch_body_terms = _g1_reference_body_terms(
        batch_rows,
        batch_timing,
        batch_pipeline,
    )
    batch_finished_ns = time.monotonic_ns()
    if len(streamed_samples) != len(batch_timing):
        raise RuntimeError("stream and batch resample counts differ")
    if any(
        streamed["source_frame_index"] != batch["source_frame_index"]
        or streamed["reference_monotonic_ns"] != batch["reference_monotonic_ns"]
        or streamed["body_poses"] != batch["body_poses"]
        for streamed, batch in zip(
            streamed_samples,
            batch_timing,
            strict=True,
        )
    ):
        raise RuntimeError("stream and batch 50 Hz pose samples differ")

    rolling = (
        ExperimentalSomaRollingRetargeter(
            soma_source_root=args.soma_source_root.resolve(),
            solver_iterations=args.experimental_solver_iterations,
        )
        if args.experimental_solver_iterations is not None
        else PinnedSomaRollingRetargeter(soma_source_root=args.soma_source_root.resolve())
    )
    rolling.calibrate(neutral_body_pose_frames)
    rolling_started_ns = time.monotonic_ns()
    rolling_rows = [rolling.push(sample) for sample in streamed_samples]
    rolling_finished_ns = time.monotonic_ns()
    equivalence = compare_rolling_to_batch(
        rolling_rows,
        batch_rows,
        absolute_tolerance=args.absolute_tolerance,
    )
    body_equivalence = _compare_body_terms(
        rolling_rows,
        batch_body_terms,
        absolute_tolerance=args.absolute_tolerance,
    )
    timing = rolling.timing_summary()
    causal_producer = CausalHistorySemanticProducer(source_session_id=capture["session_id"])
    causal_packets = [packet for row in rolling_rows if (packet := causal_producer.push(row)) is not None]
    causal_summaries = [validate_causal_history_packet(packet) for packet in causal_packets]
    causal_reference_packets = [causal_history_reference_terms(packet) for packet in causal_packets]
    if not causal_packets:
        raise RuntimeError("capture lacks 11 exact 50 Hz q0..q10 samples")
    lower_positions = [
        float(value) for packet in causal_packets for value in packet["causal_history_lower_body"][:120]
    ]
    lower_velocities = [
        float(value) for packet in causal_packets for value in packet["causal_history_lower_body"][120:]
    ]
    bracket_wait_ns = [
        int(sample["capture_monotonic_ns"]) - int(sample["reference_monotonic_ns"]) for sample in streamed_samples
    ]
    if not all(math.isfinite(value) for value in lower_positions + lower_velocities):
        raise RuntimeError("causal-history replay contains non-finite values")
    experimental = args.experimental_solver_iterations is not None
    continuous_50hz_proven = (
        not experimental
        and equivalence["exact_within_tolerance"]
        and body_equivalence["exact_within_tolerance"]
        and timing["deadline_met_every_sample"]
    )
    report = {
        "schema_version": 1,
        "kind": "g1_true23_xr24_pinned_soma_rolling_benchmark",
        "capture": str(capture_path),
        "capture_file_sha256": _sha256_file(capture_path),
        "capture_validation": capture_validation,
        "neutral_calibration_capture": str(calibration_path),
        "neutral_calibration_capture_file_sha256": _sha256_file(calibration_path),
        "neutral_calibration_capture_validation": calibration_validation,
        "soma_source_root": str(args.soma_source_root.resolve()),
        "pinned_runtime_report": rolling.runtime_report,
        "sample_count": len(streamed_samples),
        "batch_total_duration_ns": batch_finished_ns - batch_started_ns,
        "rolling_total_duration_ns": rolling_finished_ns - rolling_started_ns,
        "rolling_batch_equivalence": equivalence,
        "rolling_body_term_equivalence": body_equivalence,
        "rolling_steady_timing": timing,
        "retarget_contract": rolling.retarget_contract,
        "experimental_profile": experimental,
        "rolling_outputs_finite": all(
            math.isfinite(float(value)) for row in rolling_rows for value in row["joint_root7_mj29"]
        ),
        "rolling_joint_limits_verified": (rolling.joint_limit_verified_sample_count == len(rolling_rows)),
        "rolling_joint_limit_verified_sample_count": (rolling.joint_limit_verified_sample_count),
        "released_profile_latency_proofs": all_released_profile_latency_proofs(),
        "causal_history_replay": {
            "contract": causal_history_stream_contract(),
            "packet_count": len(causal_packets),
            "packet_summaries_sha256": _sha256(causal_summaries),
            "packets_sha256": _sha256(causal_packets),
            "reference_packet_count": len(causal_reference_packets),
            "reference_packets_sha256": _sha256(causal_reference_packets),
            "first_anchor_source_frame_index": causal_packets[0]["anchor_source_frame_index"],
            "last_anchor_source_frame_index": causal_packets[-1]["anchor_source_frame_index"],
            "first_control_source_frame_index": causal_packets[0]["proof_source_frame_index"],
            "last_control_source_frame_index": causal_packets[-1]["proof_source_frame_index"],
            "intrinsic_measurement_delay_ns": 20_000_000,
            "resample_bracket_wait_ns": _distribution(bracket_wait_ns),
            "lower_body_position_rad": _distribution(lower_positions),
            "lower_body_velocity_radps": _distribution(lower_velocities),
            "all_outputs_finite": True,
            "sdk_derivatives_consumed": False,
            "positions_repeated_or_synthesized": False,
            "complete_267_encoder_packet_count": 0,
            "complete_267_encoder_blocker": (
                "robot IMU quaternion history was not captured; each packet "
                "requires interpolation at exact q9 while control/proprioception "
                "uses q10"
            ),
            "promotion_eligible": False,
        },
        "continuous_50hz_live_proven": continuous_50hz_proven,
        "live_deployment_approved": False,
        "promotion_eligible": False,
        "authorization": {
            "read_only": True,
            "dds_opened": False,
            "robot_channel_opened": False,
            "actuation_authorized": False,
            "robot_commands_published": False,
        },
    }
    _write_exclusive(args.output, report)
    if args.causal_packets_output is not None:
        _write_exclusive(
            args.causal_packets_output,
            {
                "semantic_packets": causal_packets,
                "robot_independent_reference_packets": (causal_reference_packets),
            },
        )
    print(_json_bytes(report).decode("utf-8"), end="")
    return 2 if args.require_deadline and not continuous_50hz_proven else 0


if __name__ == "__main__":
    raise SystemExit(main())
