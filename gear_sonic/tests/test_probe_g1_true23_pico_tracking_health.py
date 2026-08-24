from __future__ import annotations

from gear_sonic.scripts.probe_g1_true23_pico_tracking_health import (
    HEALTH_KEYS,
    health_view,
    tracking_health_passes,
)


def _healthy() -> dict[str, object]:
    return {
        "health_available": True,
        "health_valid": True,
        "health_supported": True,
        "health_connect_state_result": 0,
        "health_connected_band_count": 2,
        "health_tracker_count": 2,
        "health_unique_tracker_count": 2,
        "health_calibrated": True,
        "health_calibration_result": 0,
        "health_is_tracking": True,
        "health_tracking_state_code": 0,
        "health_body_data_result": 0,
        "health_body_state_result": 0,
        "health_body_error_code": 0,
        "health_body_role_count": 24,
        "health_client_build": "xrobotoolkit-pico-health-v1",
    }


def test_health_view_does_not_expose_pose_arrays() -> None:
    source = {**_healthy(), "body_poses": [[1.0] * 7], "secret": "not exposed"}
    view = health_view(source)
    assert tuple(view) == HEALTH_KEYS
    assert "body_poses" not in view
    assert "secret" not in view
    assert tracking_health_passes(view) is True


def test_tracking_health_gate_fails_closed() -> None:
    for key, bad in (
        ("health_available", False),
        ("health_connected_band_count", 1),
        ("health_unique_tracker_count", 1),
        ("health_calibrated", False),
        ("health_is_tracking", False),
        ("health_body_role_count", 23),
    ):
        health = _healthy()
        health[key] = bad
        assert tracking_health_passes(health) is False
