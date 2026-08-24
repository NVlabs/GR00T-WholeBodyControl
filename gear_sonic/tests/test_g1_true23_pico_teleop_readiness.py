from __future__ import annotations

import json
from pathlib import Path

from gear_sonic.utils.g1_true23_pico_teleop_readiness import audit_pico_teleop_readiness

ROOT = Path(__file__).resolve().parents[2]


def test_current_readiness_fails_only_at_live_headset_gate() -> None:
    report = audit_pico_teleop_readiness(
        repository_root=ROOT,
        health_report_path=(
            ROOT / "artifacts/g1_true23/pico_internet_fullbody_teleop_v1/live_pico_walk.health.v1.json"
        ),
    )
    assert report["physical_dof"] == 23
    assert report["source_29dof_physics_used"] is False
    assert report["simulator_motion_and_transport_ready"] is True
    assert report["live_headset_health_ready"] is False
    assert report["live_headset_simulator_qualified"] is False
    assert report["blocking_gate"] == "live_pico_tracking_health"
    assert report["robot_teleop_authorized"] is False
    assert report["hardware_authorized"] is False


def test_health_boundary_tamper_is_rejected(tmp_path: Path) -> None:
    source = json.loads(
        (ROOT / "artifacts/g1_true23/pico_internet_fullbody_teleop_v1/live_pico_walk.health.v1.json").read_text(
            encoding="utf-8"
        )
    )
    source["authorization"]["hardware_authorized"] = True
    damaged = tmp_path / "health.json"
    damaged.write_text(json.dumps(source), encoding="utf-8")
    try:
        audit_pico_teleop_readiness(repository_root=ROOT, health_report_path=damaged)
    except ValueError as error:
        assert "boundary mismatch" in str(error)
    else:
        raise AssertionError("hardware-authorized health evidence was accepted")
