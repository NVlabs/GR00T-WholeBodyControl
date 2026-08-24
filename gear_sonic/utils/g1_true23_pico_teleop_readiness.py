"""Fail-closed readiness audit for authentic PICO true23 teleoperation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from gear_sonic.utils.g1_true23_pico_sonic_mode_registry import load_native23_mode_profile

PROFILE_NAME = "pico_internet_fullbody_walk"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def audit_pico_teleop_readiness(
    *,
    repository_root: Path,
    health_report_path: Path,
    live_consumer_report_path: Path | None = None,
) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    profile = load_native23_mode_profile(root, PROFILE_NAME)
    health_path = health_report_path.resolve(strict=True)
    health = _mapping(json.loads(health_path.read_text(encoding="utf-8")), "health report")
    health_auth = _mapping(health.get("authorization"), "health authorization")
    if (
        health.get("kind") != "g1_true23_pico_tracking_health_probe_v1"
        or health.get("xrt_binding_sha256") != "34eeb4484fb68e860ef4c7a1617022e020084a041b98226aea80f6eb93483de1"
        or health.get("service_binary_sha256")
        != "8654b4f3552e36e1223f6589491ebe6c82002a07a09520fae7f257465ce0bbbc"
        or health_auth.get("read_only") is not True
        or health_auth.get("poses_published") is not False
        or health_auth.get("dds_opened") is not False
        or health_auth.get("robot_channel_opened") is not False
        or health_auth.get("hardware_authorized") is not False
        or health_auth.get("robot_commands_published") is not False
    ):
        raise ValueError("health report boundary mismatch")

    live_report_path: Path | None = None
    live_report_sha256: str | None = None
    live_simulator_qualified = False
    live_completed_transitions = 0
    if live_consumer_report_path is not None:
        live_report_path = live_consumer_report_path.resolve(strict=True)
        live_report_sha256 = sha256_file(live_report_path)
        live_report = _mapping(json.loads(live_report_path.read_text(encoding="utf-8")), "live consumer report")
        live_auth = _mapping(live_report.get("authorization"), "live consumer authorization")
        completed = live_report.get("completed_transitions")
        if type(completed) is not int:
            raise ValueError("live consumer transition count mismatch")
        live_completed_transitions = completed
        live_simulator_qualified = bool(
            live_report.get("passed") is True
            and live_report.get("kind") == "g1_true23_clean_mujoco_teleop_session"
            and live_report.get("native23_profile") == PROFILE_NAME
            and live_report.get("decoder_sha256") == profile.decoder_sha256
            and live_report.get("physical_dof") == 23
            and live_report.get("decoder_output_dof") == 23
            and live_report.get("source_29dof_physics_used") is False
            and live_report.get("live_transport_proven") is True
            and completed >= 500
            and isinstance(live_report.get("maximum_reference_age_ns"), int)
            and live_report["maximum_reference_age_ns"] <= 100_000_000
            and live_auth.get("simulator_only") is True
            and live_auth.get("dds_opened") is False
            and live_auth.get("hardware_authorized") is False
            and live_auth.get("robot_commands_published") is False
        )

    simulator_motion_and_transport_ready = bool(
        profile.live_transport_proven
        and profile.live_headset_source_proven is False
        and profile.simulator_modes == ("saved", "zmq")
        and len(profile.evidence_paths) >= 8
    )
    headset_health_ready = health.get("passed") is True
    return {
        "schema_version": 1,
        "kind": "g1_true23_authentic_pico_teleop_readiness_v1",
        "profile": PROFILE_NAME,
        "physical_dof": 23,
        "decoder_output_dof": 23,
        "source_29dof_physics_used": False,
        "decoder_sha256": profile.decoder_sha256,
        "profile_evidence_count": len(profile.evidence_paths),
        "simulator_motion_and_transport_ready": simulator_motion_and_transport_ready,
        "headset_health_report": str(health_path),
        "headset_health_report_sha256": sha256_file(health_path),
        "live_headset_health_ready": headset_health_ready,
        "live_consumer_report": None if live_report_path is None else str(live_report_path),
        "live_consumer_report_sha256": live_report_sha256,
        "live_completed_transitions": live_completed_transitions,
        "live_headset_simulator_qualified": live_simulator_qualified,
        "simulator_qualification_complete": bool(
            simulator_motion_and_transport_ready and headset_health_ready and live_simulator_qualified
        ),
        "robot_teleop_authorized": False,
        "hardware_authorized": False,
        "dds_opened": False,
        "robot_commands_published": False,
        "blocking_gate": (
            None
            if live_simulator_qualified
            else (
                "live_pico_tracking_health"
                if not headset_health_ready
                else "live_pico_true23_mujoco_500_transition_session"
            )
        ),
    }


__all__ = ("PROFILE_NAME", "audit_pico_teleop_readiness", "sha256_file")
