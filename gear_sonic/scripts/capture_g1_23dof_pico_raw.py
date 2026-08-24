"""Capture immutable hardened PICO XR24 frames for exact SOMA replay.

This tool opens only the local XRoboToolkit SDK.  It imports no Unitree SDK,
opens no DDS/ZMQ/robot channel, and never materializes policy commands.
Captured frames remain non-promotable until they are replayed through the
pinned NVIDIA SOMA backend and the XR24 coordinate/role adapter is approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from types import ModuleType
from typing import Any
import uuid

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
    validate_bodytracking_xrt_snapshot,
    validate_raw_capture,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AUTHORIZATION = {
    "read_only": True,
    "dds_opened": False,
    "robot_channel_opened": False,
    "actuation_authorized": False,
    "robot_commands_published": False,
}


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize capture evidence without importing training/Torch modules."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, context: str) -> str:
    normalized = value.casefold()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return normalized


def _next_advancing_snapshot(
    xrt: Any,
    *,
    previous_sequence: int,
    previous_timestamp_ns: int,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = "no snapshot received"
    while time.monotonic() < deadline:
        try:
            snapshot = dict(xrt.get_body_snapshot())
            validate_bodytracking_xrt_snapshot(snapshot)
            sequence = int(snapshot["sample_sequence"])
            timestamp_ns = int(snapshot["timestamp_ns"])
            if (
                sequence > previous_sequence
                and timestamp_ns > previous_timestamp_ns
            ):
                return snapshot
            last_error = (
                "body sample did not advance "
                f"(sequence={sequence}, timestamp_ns={timestamp_ns})"
            )
        except (KeyError, TypeError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(0.005)
    raise TimeoutError(
        f"no advancing valid hardened XR24 BodyTracking frame within "
        f"{timeout_s:.3f}s: {last_error}"
    )


def collect_raw_capture(
    xrt: Any,
    *,
    frame_count: int,
    frame_timeout_s: float,
    session_id: str,
    xrobotoolkit_sdk_sha256: str,
    pc_service_sha256: str,
    pico_client_apk_sha256: str,
) -> dict[str, Any]:
    """Collect advancing atomic frames and return a validated raw capture."""
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 2
    ):
        raise ValueError("frame_count must be an integer >= 2")
    if (
        isinstance(frame_timeout_s, bool)
        or not isinstance(frame_timeout_s, (int, float))
        or frame_timeout_s <= 0
    ):
        raise ValueError("frame_timeout_s must be positive")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be non-empty")

    source_hashes = {
        "xrobotoolkit_sdk_sha256": _require_sha256(
            xrobotoolkit_sdk_sha256,
            "xrobotoolkit_sdk_sha256",
        ),
        "pc_service_sha256": _require_sha256(
            pc_service_sha256,
            "pc_service_sha256",
        ),
        "pico_client_apk_sha256": _require_sha256(
            pico_client_apk_sha256,
            "pico_client_apk_sha256",
        ),
    }
    frames: list[dict[str, Any]] = []
    previous_sequence = 0
    previous_timestamp_ns = 0
    previous_capture_ns = 0
    client_build: str | None = None
    body_timestamp_contract: str | None = None
    joint_timestamp_contract: str | None = None
    for frame_index in range(frame_count):
        snapshot = _next_advancing_snapshot(
            xrt,
            previous_sequence=previous_sequence,
            previous_timestamp_ns=previous_timestamp_ns,
            timeout_s=float(frame_timeout_s),
        )
        snapshot_client_build = str(snapshot["health_client_build"])
        if client_build is None:
            client_build = snapshot_client_build
        elif snapshot_client_build != client_build:
            raise ValueError("PICO hardened client build changed during capture")
        snapshot_body_timestamp_contract = str(
            snapshot["body_timestamp_contract"]
        )
        snapshot_joint_timestamp_contract = str(
            snapshot["joint_timestamp_contract"]
        )
        if body_timestamp_contract is None:
            body_timestamp_contract = snapshot_body_timestamp_contract
            joint_timestamp_contract = snapshot_joint_timestamp_contract
        elif (
            snapshot_body_timestamp_contract != body_timestamp_contract
            or snapshot_joint_timestamp_contract != joint_timestamp_contract
        ):
            raise ValueError("PICO timestamp contract changed during capture")
        capture_ns = max(time.monotonic_ns(), previous_capture_ns + 1)
        frame = raw_frame_from_bodytracking_xrt_snapshot(
            snapshot,
            frame_index=frame_index,
            capture_monotonic_ns=capture_ns,
        )
        frames.append(frame)
        previous_sequence = int(snapshot["sample_sequence"])
        previous_timestamp_ns = int(snapshot["timestamp_ns"])
        previous_capture_ns = capture_ns

    capture = {
        "schema_version": RAW_CAPTURE_SCHEMA_VERSION,
        "kind": RAW_CAPTURE_KIND,
        "session_id": session_id.strip(),
        "source": {
            "atomic_snapshot_contract": ATOMIC_SNAPSHOT_CONTRACT,
            **source_hashes,
            "pico_client_build": client_build,
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
        "frames": frames,
        "authorization": dict(_AUTHORIZATION),
    }
    validate_raw_capture(capture)
    return capture


def _load_xrt() -> ModuleType:
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise RuntimeError(
            "xrobotoolkit_sdk unavailable; run inside the configured WSL "
            "XRoboToolkit environment"
        ) from exc
    if not callable(getattr(xrt, "init", None)):
        raise RuntimeError("xrobotoolkit_sdk.init is unavailable")
    if not callable(getattr(xrt, "get_body_snapshot", None)):
        raise RuntimeError(
            "hardened xrobotoolkit_sdk.get_body_snapshot is unavailable"
        )
    if not callable(getattr(xrt, "close", None)):
        raise RuntimeError("xrobotoolkit_sdk.close is unavailable")
    return xrt


def _sdk_binary_path(xrt: ModuleType) -> Path:
    module_file = getattr(xrt, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError("cannot resolve xrobotoolkit_sdk binary path")
    path = Path(module_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"xrobotoolkit_sdk binary missing: {path}")
    return path


def _write_exclusive(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only hardened PICO XR24 capture for exact offline SOMA replay."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pico-client-apk-sha256", required=True)
    parser.add_argument(
        "--service-binary",
        type=Path,
        default=Path("/opt/apps/roboticsservice/RoboticsServiceProcess"),
    )
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--frame-timeout-s", type=float, default=5.0)
    parser.add_argument("--session-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    service_binary = args.service_binary.resolve()
    if not service_binary.is_file():
        raise FileNotFoundError(
            f"XRoboToolkit service binary missing: {service_binary}"
        )
    if args.output.resolve().exists():
        raise FileExistsError(f"refusing to overwrite capture: {args.output}")

    xrt = _load_xrt()
    sdk_binary = _sdk_binary_path(xrt)
    xrt.init()
    try:
        capture = collect_raw_capture(
            xrt,
            frame_count=args.frames,
            frame_timeout_s=args.frame_timeout_s,
            session_id=args.session_id or f"pico-xr24-{uuid.uuid4()}",
            xrobotoolkit_sdk_sha256=_sha256_file(sdk_binary),
            pc_service_sha256=_sha256_file(service_binary),
            pico_client_apk_sha256=args.pico_client_apk_sha256,
        )
        payload = canonical_json_bytes(capture)
        _write_exclusive(args.output, payload)
        summary = validate_raw_capture(capture)
        print(
            canonical_json_bytes(
                {
                    "status": "captured_read_only_non_promotable",
                    "output": str(args.output.resolve()),
                    "capture_sha256": summary["sha256"],
                    "frame_count": summary["frame_count"],
                    "authorization": dict(_AUTHORIZATION),
                }
            ).decode("utf-8")
        )
        return 0
    finally:
        xrt.close()


if __name__ == "__main__":
    raise SystemExit(main())
