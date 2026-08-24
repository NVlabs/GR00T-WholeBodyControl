"""Probe PICO XR24 tracking health without exposing poses or robot surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from gear_sonic.scripts.capture_g1_23dof_pico_raw import _load_xrt, _sdk_binary_path

EXPECTED_XRT_SHA256 = "34eeb4484fb68e860ef4c7a1617022e020084a041b98226aea80f6eb93483de1"
EXPECTED_SERVICE_SHA256 = "8654b4f3552e36e1223f6589491ebe6c82002a07a09520fae7f257465ce0bbbc"
HEALTH_KEYS = (
    "health_available",
    "health_valid",
    "health_supported",
    "health_connect_state_result",
    "health_connected_band_count",
    "health_tracker_count",
    "health_unique_tracker_count",
    "health_calibrated",
    "health_calibration_result",
    "health_is_tracking",
    "health_tracking_state_code",
    "health_body_data_result",
    "health_body_state_result",
    "health_body_error_code",
    "health_body_role_count",
    "health_client_build",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def health_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {key: snapshot.get(key) for key in HEALTH_KEYS}


def tracking_health_passes(health: Mapping[str, Any]) -> bool:
    return bool(
        health.get("health_available") is True
        and health.get("health_valid") is True
        and health.get("health_supported") is True
        and health.get("health_connect_state_result") == 0
        and isinstance(health.get("health_connected_band_count"), int)
        and health["health_connected_band_count"] >= 2
        and isinstance(health.get("health_unique_tracker_count"), int)
        and health["health_unique_tracker_count"] >= 2
        and health.get("health_calibrated") is True
        and health.get("health_calibration_result") == 0
        and health.get("health_is_tracking") is True
        and health.get("health_tracking_state_code") == 0
        and health.get("health_body_data_result") == 0
        and health.get("health_body_state_result") == 0
        and health.get("health_body_error_code") == 0
        and health.get("health_body_role_count") == 24
    )


def _exclusive_json(path: Path, report: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument(
        "--service-binary",
        type=Path,
        default=Path("/opt/apps/roboticsservice/RoboticsServiceProcess"),
    )
    args = parser.parse_args(argv)
    if args.duration_seconds <= 0.0 or not 0.005 <= args.poll_seconds <= 1.0:
        raise ValueError("health probe timing mismatch")
    service = args.service_binary.resolve(strict=True)
    service_sha256 = _sha256_file(service)
    if service_sha256 != EXPECTED_SERVICE_SHA256:
        raise ValueError("XR service SHA256 mismatch")

    xrt = _load_xrt()
    binding = _sdk_binary_path(xrt)
    binding_sha256 = _sha256_file(binding)
    if binding_sha256 != EXPECTED_XRT_SHA256:
        raise ValueError("XRT binding SHA256 mismatch")

    latest: dict[str, Any] | None = None
    snapshot_calls = 0
    exception_counts: dict[str, int] = {}
    xrt.init()
    try:
        deadline = time.monotonic() + args.duration_seconds
        while time.monotonic() < deadline:
            try:
                latest = health_view(dict(xrt.get_body_snapshot()))
                snapshot_calls += 1
                if tracking_health_passes(latest):
                    break
            except Exception as error:  # SDK boundary; report type only.
                name = type(error).__name__
                exception_counts[name] = exception_counts.get(name, 0) + 1
            time.sleep(args.poll_seconds)
    finally:
        xrt.close()

    passed = latest is not None and tracking_health_passes(latest)
    report = {
        "schema_version": 1,
        "kind": "g1_true23_pico_tracking_health_probe_v1",
        "xrt_binding_path": str(binding),
        "xrt_binding_sha256": binding_sha256,
        "service_binary_path": str(service),
        "service_binary_sha256": service_sha256,
        "duration_seconds": args.duration_seconds,
        "snapshot_calls": snapshot_calls,
        "exception_counts": exception_counts,
        "latest_health": latest,
        "passed": passed,
        "authorization": {
            "read_only": True,
            "poses_published": False,
            "dds_opened": False,
            "robot_channel_opened": False,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }
    _exclusive_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
