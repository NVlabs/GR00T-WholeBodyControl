"""Receive and validate PICO causal-reference transport without robot APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import zmq


_AUTHORIZATION = {
    "read_only": True,
    "dds_opened": False,
    "robot_channel_opened": False,
    "actuation_authorized": False,
    "robot_commands_published": False,
}
_KIND = "g1_true23_xr24_soma_causal_history_reference_terms"
_PROFILE = "true23_causal_step1_history_0p02s_v1"
_DERIVATIVE = "soma_il29_q_50hz_forward_difference_dq_v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _finite_vector(packet: dict[str, object], name: str, size: int) -> None:
    value = packet.get(name)
    if (
        not isinstance(value, list)
        or len(value) != size
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain {size} finite numbers")


def _validate(packet: dict[str, object]) -> tuple[int, int, int, int]:
    for name, size in (
        ("causal_history_lower_body", 240),
        ("vr_3point_local_target", 9),
        ("vr_3point_local_orn_target", 12),
        ("reference_anchor_quaternion_xyzw", 4),
        ("anchor_joint_pos_il29", 29),
        ("proof_joint_pos_il29", 29),
        ("q_ref23_native", 23),
        ("qd_ref23_native", 23),
    ):
        _finite_vector(packet, name, size)
    control_index = packet.get("control_source_frame_index")
    anchor_index = packet.get("pico_anchor_source_frame_index")
    control_ns = packet.get("control_monotonic_ns")
    anchor_ns = packet.get("pico_anchor_monotonic_ns")
    if (
        packet.get("schema_version") != 1
        or packet.get("kind") != _KIND
        or packet.get("reference_profile") != _PROFILE
        or not isinstance(packet.get("reference_contract_sha256"), str)
        or len(str(packet["reference_contract_sha256"])) != 64
        or not isinstance(control_index, int)
        or isinstance(control_index, bool)
        or not isinstance(anchor_index, int)
        or isinstance(anchor_index, bool)
        or anchor_index != control_index - 1
        or not isinstance(control_ns, int)
        or isinstance(control_ns, bool)
        or not isinstance(anchor_ns, int)
        or isinstance(anchor_ns, bool)
        or anchor_ns != control_ns - 20_000_000
        or packet.get("control_derivative_contract") != _DERIVATIVE
        or packet.get("sdk_derivatives_consumed") is not False
    ):
        raise ValueError("causal reference transport contract mismatch")
    return control_index, control_ns, anchor_index, anchor_ns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connect", default="tcp://127.0.0.1:5557")
    parser.add_argument("--packets", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.packets <= 0 or args.timeout_seconds <= 0:
        raise ValueError("positive packet count and timeout required")
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, max(1, round(args.timeout_seconds * 1000)))
    socket.connect(args.connect)
    previous_index: int | None = None
    previous_ns: int | None = None
    ages: list[int] = []
    started_ns = time.monotonic_ns()
    with args.evidence.open("x", encoding="utf-8", newline="\n") as evidence:
        def write(record: dict[str, object]) -> None:
            evidence.write(_canonical_bytes(record).decode("utf-8") + "\n")
            evidence.flush()

        write(
            {
                "schema_version": 1,
                "kind": "g1_true23_pico_transport_validation",
                "event": "session_start",
                "connect": args.connect,
                "requested_packets": args.packets,
                "started_monotonic_ns": started_ns,
                "authorization": dict(_AUTHORIZATION),
            }
        )
        for packet_index in range(args.packets):
            wire = socket.recv()
            received_ns = time.monotonic_ns()
            packet = json.loads(wire)
            if not isinstance(packet, dict):
                raise ValueError("transport record must be object")
            control_index, control_ns, anchor_index, anchor_ns = _validate(packet)
            if previous_index is not None and (
                control_index != previous_index + 1
                or control_ns != previous_ns + 20_000_000
            ):
                raise ValueError("transport lost exact contiguous 20 ms cadence")
            age_ns = received_ns - control_ns
            if not 0 <= age_ns <= 80_000_000:
                raise ValueError(f"transport packet outside 80 ms budget: {age_ns}")
            ages.append(age_ns)
            write(
                {
                    "schema_version": 1,
                    "kind": "g1_true23_pico_transport_validation",
                    "event": "packet_validated",
                    "packet_index": packet_index,
                    "control_source_frame_index": control_index,
                    "control_monotonic_ns": control_ns,
                    "pico_anchor_source_frame_index": anchor_index,
                    "pico_anchor_monotonic_ns": anchor_ns,
                    "received_monotonic_ns": received_ns,
                    "transport_age_ns": age_ns,
                    "wire_sha256": hashlib.sha256(wire).hexdigest(),
                    "authorization": dict(_AUTHORIZATION),
                }
            )
            previous_index = control_index
            previous_ns = control_ns
        write(
            {
                "schema_version": 1,
                "kind": "g1_true23_pico_transport_validation",
                "event": "session_complete",
                "validated_packets": len(ages),
                "exact_contiguous_20ms": True,
                "all_within_80ms": True,
                "packets_within_40ms": sum(age <= 40_000_000 for age in ages),
                "maximum_transport_age_ns": max(ages),
                "mean_transport_age_ns": sum(ages) / len(ages),
                "completed_monotonic_ns": time.monotonic_ns(),
                "authorization": dict(_AUTHORIZATION),
            }
        )
    socket.close()
    context.term()
    print(
        f"[PASS] validated {len(ages)} PICO causal packets; "
        f"max_age_ms={max(ages) / 1e6:.1f}; no robot channel opened",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
