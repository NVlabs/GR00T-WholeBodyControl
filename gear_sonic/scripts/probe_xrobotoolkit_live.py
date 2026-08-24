"""Print one XRoboToolkit SDK availability/snapshot diagnostic."""

from __future__ import annotations

import json

import xrobotoolkit_sdk as xrt


def _call(name: str) -> object:
    function = getattr(xrt, name, None)
    if not callable(function):
        return "unavailable"
    try:
        return function()
    except Exception as exc:  # diagnostic must report every SDK surface
        return f"ERROR: {exc!r}"


def main() -> int:
    xrt.init()
    snapshot = _call("get_body_snapshot")
    if isinstance(snapshot, dict):
        snapshot = {
            key: snapshot.get(key)
            for key in (
                "available",
                "timestamp_ns",
                "sample_sequence",
                "health_available",
                "health_supported",
                "health_valid",
                "health_schema_version",
                "health_client_build",
                "health_sample_sequence",
                "health_body_role_count",
                "health_tracker_count",
                "health_unique_tracker_count",
                "health_calibrated",
                "health_is_tracking",
                "health_body_data_result",
                "health_body_error_code",
            )
        }
    result = {
        "is_body_data_available": _call("is_body_data_available"),
        "get_body_timestamp_ns": _call("get_body_timestamp_ns"),
        "get_body_snapshot": snapshot,
    }
    print(json.dumps(result, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
