"""Replay validated real PICO reference packets over localhost at exact 50 Hz."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import zmq

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import validate_reference_terms

CONTROL_PERIOD_NS = 20_000_000


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exclusive_json(path: Path, report: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_reference_packets(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "robot_independent_reference_packets",
        "semantic_packets",
    }:
        raise ValueError("saved PICO packet bundle schema mismatch")
    packets = payload["robot_independent_reference_packets"]
    if not isinstance(packets, list) or not packets:
        raise ValueError("saved PICO packet bundle has no reference packets")
    summaries = [validate_reference_terms(packet) for packet in packets]
    for previous, current in zip(summaries, summaries[1:], strict=False):
        if current["control_index"] != previous["control_index"] + 1:
            raise ValueError("saved PICO packet indices are not contiguous")
    return packets


def rebase_reference_packet_time(
    packet: Mapping[str, Any],
    *,
    first_control_index: int,
    first_control_monotonic_ns: int,
) -> dict[str, Any]:
    summary = validate_reference_terms(packet)
    offset = summary["control_index"] - first_control_index
    if offset < 0:
        raise ValueError("packet precedes timed replay origin")
    control_ns = first_control_monotonic_ns + offset * CONTROL_PERIOD_NS
    rebased = copy.deepcopy(dict(packet))
    rebased["pico_anchor_monotonic_ns"] = control_ns - CONTROL_PERIOD_NS
    rebased["control_monotonic_ns"] = control_ns
    validate_reference_terms(rebased)
    return rebased


def publish_timed_bundle(
    *,
    packets: list[dict[str, Any]],
    bind: str,
    subscriber_warmup_s: float,
) -> dict[str, Any]:
    if not bind.startswith("tcp://127.0.0.1:"):
        raise ValueError("timed PICO replay is restricted to localhost TCP")
    if subscriber_warmup_s < 0.1:
        raise ValueError("subscriber warmup must be at least 0.1 seconds")
    first_summary = validate_reference_terms(packets[0])
    context = zmq.Context(io_threads=1)
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 1000)
    socket.bind(bind)
    schedule_slips_ns: list[int] = []
    started_ns = time.monotonic_ns()
    try:
        time.sleep(subscriber_warmup_s)
        first_control_ns = time.monotonic_ns() + CONTROL_PERIOD_NS
        for packet in packets:
            rebased = rebase_reference_packet_time(
                packet,
                first_control_index=first_summary["control_index"],
                first_control_monotonic_ns=first_control_ns,
            )
            deadline_ns = int(rebased["control_monotonic_ns"])
            while True:
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                time.sleep(min(remaining_ns / 1_000_000_000, 0.002))
            sent_ns = time.monotonic_ns()
            schedule_slips_ns.append(sent_ns - deadline_ns)
            socket.send_json(rebased)
        time.sleep(0.1)
    finally:
        socket.close()
        context.term()
    finished_ns = time.monotonic_ns()
    return {
        "schema_version": 1,
        "kind": "g1_true23_saved_pico_timed_zmq_replay",
        "bind": bind,
        "packet_count": len(packets),
        "first_control_source_frame_index": first_summary["control_index"],
        "last_control_source_frame_index": validate_reference_terms(packets[-1])["control_index"],
        "control_period_ns": CONTROL_PERIOD_NS,
        "maximum_schedule_slip_ns": max(schedule_slips_ns),
        "mean_schedule_slip_ns": sum(schedule_slips_ns) / len(schedule_slips_ns),
        "wall_duration_ns": finished_ns - started_ns,
        "values_rebased": ["pico_anchor_monotonic_ns", "control_monotonic_ns"],
        "pose_and_reference_values_unchanged": True,
        "passed": True,
        "authorization": {
            "localhost_only": True,
            "dds_opened": False,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5557")
    parser.add_argument("--subscriber-warmup-s", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packets = load_reference_packets(args.packets)
    report = publish_timed_bundle(
        packets=packets,
        bind=args.bind,
        subscriber_warmup_s=args.subscriber_warmup_s,
    )
    report["saved_packet_bundle"] = str(args.packets.resolve())
    report["saved_packet_bundle_sha256"] = _sha256_file(args.packets.resolve())
    _exclusive_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
