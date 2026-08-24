"""Python-3.10 XR24 capture worker for the causal PICO shadow stream.

The hardened XRoboToolkit extension is CPython-3.10-only, while pinned SOMA
runs in Python 3.12.  This worker bridges only validated read-only raw frames
over prefixed stdout.  It opens no Unitree, DDS, or robot command surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any
import uuid

from gear_sonic.scripts.capture_g1_23dof_pico_raw import (
    _load_xrt,
    _next_advancing_snapshot,
    _require_sha256,
    _sdk_binary_path,
    _sha256_file,
)
from gear_sonic.utils.g1_23dof_pico_retargeted_producer import (
    ANKLE_ROLE_SEMANTICS,
    ANKLE_ROLE_TO_BODY_INDEX,
    ATOMIC_SNAPSHOT_CONTRACT,
    DERIVATIVE_LAYOUT_CONTRACT,
    POSITION_DERIVED_CONTROL_DERIVATIVE_CONTRACT,
    RAW_CAPTURE_KIND,
    RAW_CAPTURE_SCHEMA_VERSION,
    SOURCE_COHERENCE_CONTRACT,
    XRT_BODY_JOINT_NAMES,
    raw_frame_from_bodytracking_xrt_snapshot,
)

_PREFIX = "G1_TRUE23_XR24\t"
_AUTHORIZATION = {
    "read_only": True,
    "dds_opened": False,
    "robot_channel_opened": False,
    "actuation_authorized": False,
    "robot_commands_published": False,
}


def _emit(event: dict[str, Any]) -> None:
    payload = json.dumps(
        event, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    print(f"{_PREFIX}{payload}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-binary", type=Path, required=True)
    parser.add_argument("--expected-xrt-module-path", type=Path, required=True)
    parser.add_argument("--expected-xrt-module-sha256", required=True)
    parser.add_argument("--pico-client-apk-sha256", required=True)
    parser.add_argument("--frame-timeout-s", type=float, default=2.0)
    parser.add_argument("--session-id")
    parser.add_argument("--request-driven", action="store_true")
    return parser


def _verify_imported_xrt_binding(
    imported_path: Path,
    *,
    expected_path: Path,
    expected_sha256: str,
) -> dict[str, str]:
    """Bind the worker to the exact extension selected by its publisher."""

    imported = imported_path.resolve()
    expected = expected_path.resolve()
    expected_digest = _require_sha256(
        expected_sha256, "expected_xrt_module_sha256"
    )
    if imported != expected:
        raise RuntimeError(
            "imported XRT module path mismatch: "
            f"expected {expected}, imported {imported}"
        )
    if not imported.is_file():
        raise FileNotFoundError(f"imported XRT module missing: {imported}")
    imported_digest = _sha256_file(imported)
    if imported_digest != expected_digest:
        raise RuntimeError(
            "imported XRT module hash mismatch: "
            f"expected {expected_digest}, imported {imported_digest}"
        )
    return {"path": str(imported), "sha256": imported_digest}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    service_binary = args.service_binary.resolve()
    if not service_binary.is_file():
        raise FileNotFoundError(
            f"XRoboToolkit service binary missing: {service_binary}"
        )
    apk_hash = _require_sha256(
        args.pico_client_apk_sha256, "pico_client_apk_sha256"
    )
    if args.frame_timeout_s <= 0.0:
        raise ValueError("frame-timeout-s must be positive")

    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    xrt = _load_xrt()
    sdk_binary = _sdk_binary_path(xrt)
    xrt_binding = _verify_imported_xrt_binding(
        sdk_binary,
        expected_path=args.expected_xrt_module_path,
        expected_sha256=args.expected_xrt_module_sha256,
    )
    _emit({"event": "xrt_binding_verified", "binding": xrt_binding})
    xrt.init()
    try:
        previous_sequence = 0
        previous_timestamp_ns = 0
        previous_capture_ns = 0
        frame_index = 0
        identity: dict[str, Any] | None = None
        client_build: str | None = None
        body_timestamp_contract: str | None = None
        joint_timestamp_contract: str | None = None
        while not stop:
            if args.request_driven:
                request = sys.stdin.readline()
                if request == "" or request.strip() == "STOP":
                    break
                if request.strip() == "STREAM":
                    args.request_driven = False
                elif request.strip() != "NEXT":
                    raise ValueError(
                        "capture worker request must be NEXT, STREAM, or STOP"
                    )
            try:
                snapshot = _next_advancing_snapshot(
                    xrt,
                    previous_sequence=previous_sequence,
                    previous_timestamp_ns=previous_timestamp_ns,
                    timeout_s=min(float(args.frame_timeout_s), 0.5),
                )
            except TimeoutError as exc:
                if args.request_driven:
                    _emit({"event": "timeout", "failure": str(exc)})
                continue
            snapshot_client_build = str(snapshot["health_client_build"])
            snapshot_body_contract = str(snapshot["body_timestamp_contract"])
            snapshot_joint_contract = str(snapshot["joint_timestamp_contract"])
            if identity is None:
                client_build = snapshot_client_build
                body_timestamp_contract = snapshot_body_contract
                joint_timestamp_contract = snapshot_joint_contract
                identity = {
                    "schema_version": RAW_CAPTURE_SCHEMA_VERSION,
                    "kind": RAW_CAPTURE_KIND,
                    "session_id": args.session_id
                    or f"pico-xr24-live-{uuid.uuid4()}",
                    "source": {
                        "atomic_snapshot_contract": ATOMIC_SNAPSHOT_CONTRACT,
                        "xrobotoolkit_sdk_sha256": _sha256_file(sdk_binary),
                        "pc_service_sha256": _sha256_file(service_binary),
                        "pico_client_build": client_build,
                        "pico_client_apk_sha256": apk_hash,
                        "derivative_layout_contract": DERIVATIVE_LAYOUT_CONTRACT,
                        "sdk_derivatives_control_usable": False,
                        "control_derivative_contract": (
                            POSITION_DERIVED_CONTROL_DERIVATIVE_CONTRACT
                        ),
                        "source_coherence_contract": SOURCE_COHERENCE_CONTRACT,
                        "body_timestamp_contract": body_timestamp_contract,
                        "joint_timestamp_contract": joint_timestamp_contract,
                        "ankle_role_semantics": ANKLE_ROLE_SEMANTICS,
                        "ankle_role_indices": dict(ANKLE_ROLE_TO_BODY_INDEX),
                        "body_joint_order": list(XRT_BODY_JOINT_NAMES),
                    },
                    "authorization": dict(_AUTHORIZATION),
                }
                _emit({"event": "identity", "identity": identity})
            elif (
                snapshot_client_build != client_build
                or snapshot_body_contract != body_timestamp_contract
                or snapshot_joint_contract != joint_timestamp_contract
            ):
                raise RuntimeError("PICO hardened source contract changed live")

            capture_ns = max(time.monotonic_ns(), previous_capture_ns + 1)
            raw_frame = raw_frame_from_bodytracking_xrt_snapshot(
                snapshot,
                frame_index=frame_index,
                capture_monotonic_ns=capture_ns,
            )
            _emit({"event": "frame", "frame": raw_frame})
            previous_sequence = int(snapshot["sample_sequence"])
            previous_timestamp_ns = int(snapshot["timestamp_ns"])
            previous_capture_ns = capture_ns
            frame_index += 1
        return 0
    finally:
        xrt.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _emit({"event": "failure", "failure": str(exc)})
        print(f"[BLOCKED] XR24 capture worker: {exc}", file=sys.stderr)
        raise
