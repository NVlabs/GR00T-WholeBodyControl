"""Strict simulator-only registry for proven native-23 Pico and SONIC modes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

REGISTRY_RELATIVE_PATH = Path("gear_sonic/config/sim_validation/g1_true23_pico_sonic_mode_registry_v1.json")
REGISTRY_KIND = "g1_true23_pico_sonic_mode_registry_v1"
PROFILE_NAMES = (
    "pico_fullbody",
    "pico_internet_fullbody_walk",
    "sonic_hand_crawl",
    "sonic_elbow_crawl",
)
LIVE_TRANSPORT_PROFILES = ("pico_fullbody", "pico_internet_fullbody_walk")


@dataclass(frozen=True)
class Native23ModeProfile:
    name: str
    purpose: str
    encoder_path: Path
    encoder_sha256: str
    decoder_path: Path
    decoder_sha256: str
    simulator_modes: tuple[str, ...]
    minimum_base_height_m: float
    maximum_base_tilt_rad: float
    evidence_paths: tuple[Path, ...]
    live_transport_proven: bool
    live_headset_source_proven: bool
    sustained_deep_crouch_authorized: bool


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


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys mismatch")


def _regular_hash_bound(root: Path, value: Mapping[str, Any], context: str) -> Path:
    _exact_keys(value, {"path", "sha256"}, context)
    relative = value["path"]
    expected = value["sha256"]
    if not isinstance(relative, str) or not relative or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{context} path/hash mismatch")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} is missing or not a regular file")
    if sha256_file(path) != expected:
        raise ValueError(f"{context} SHA256 mismatch")
    return path


def load_native23_mode_profile(repository_root: Path, profile_name: str) -> Native23ModeProfile:
    root = repository_root.resolve()
    registry_path = root / REGISTRY_RELATIVE_PATH
    registry = _mapping(json.loads(registry_path.read_text(encoding="utf-8")), "mode registry")
    _exact_keys(
        registry,
        {
            "schema_version",
            "kind",
            "physical_dof",
            "decoder_output_dof",
            "source_29dof_physics_used",
            "hardware_authorized",
            "encoder",
            "transport_evidence",
            "profiles",
            "excluded_modes",
        },
        "mode registry",
    )
    if (
        type(registry["schema_version"]) is not int
        or registry["schema_version"] != 1
        or registry["kind"] != REGISTRY_KIND
        or type(registry["physical_dof"]) is not int
        or registry["physical_dof"] != 23
        or type(registry["decoder_output_dof"]) is not int
        or registry["decoder_output_dof"] != 23
        or registry["source_29dof_physics_used"] is not False
        or registry["hardware_authorized"] is not False
    ):
        raise ValueError("mode registry native23 boundary mismatch")
    encoder = _mapping(registry["encoder"], "encoder")
    encoder_path = _regular_hash_bound(root, encoder, "encoder")
    transport = _mapping(registry["transport_evidence"], "transport evidence")
    _exact_keys(
        transport,
        {
            "kind",
            "consumer",
            "publisher",
            "real_saved_pico_values",
            "timestamps_rebased",
            "live_headset_source_proven",
        },
        "transport evidence",
    )
    if (
        transport["kind"] != "timed_saved_pico_packets_over_localhost_zmq"
        or transport["real_saved_pico_values"] is not True
        or transport["timestamps_rebased"] is not True
        or transport["live_headset_source_proven"] is not False
    ):
        raise ValueError("transport evidence claim mismatch")
    consumer_path = _regular_hash_bound(
        root, _mapping(transport["consumer"], "transport consumer"), "transport consumer"
    )
    publisher_path = _regular_hash_bound(
        root, _mapping(transport["publisher"], "transport publisher"), "transport publisher"
    )
    consumer = _mapping(json.loads(consumer_path.read_text(encoding="utf-8")), "transport consumer report")
    publisher = _mapping(json.loads(publisher_path.read_text(encoding="utf-8")), "transport publisher report")
    if (
        consumer.get("passed") is not True
        or consumer.get("physical_dof") != 23
        or consumer.get("decoder_output_dof") != 23
        or consumer.get("source_29dof_physics_used") is not False
        or consumer.get("native23_profile") != "pico_fullbody"
        or consumer.get("live_transport_proven") is not True
        or consumer.get("completed_transitions") != 41
        or publisher.get("passed") is not True
        or publisher.get("packet_count") != 41
        or publisher.get("pose_and_reference_values_unchanged") is not True
        or _mapping(publisher.get("authorization"), "transport publisher authorization").get("hardware_authorized")
        is not False
    ):
        raise ValueError("transport evidence report mismatch")
    profiles = _mapping(registry["profiles"], "profiles")
    if tuple(profiles) != PROFILE_NAMES:
        raise ValueError("mode registry profile order/names mismatch")
    if profile_name not in PROFILE_NAMES:
        raise ValueError(f"unknown native23 mode profile: {profile_name}")
    profile = _mapping(profiles[profile_name], f"profile {profile_name}")
    profile_keys = {
        "purpose",
        "decoder",
        "simulator_modes",
        "minimum_base_height_m",
        "maximum_base_tilt_rad",
        "evidence",
        "live_transport_proven",
        "live_headset_source_proven",
        "sustained_deep_crouch_authorized",
    }
    if profile_name == "pico_internet_fullbody_walk":
        profile_keys.add("profile_transport_evidence")
    _exact_keys(profile, profile_keys, f"profile {profile_name}")
    if not isinstance(profile["purpose"], str) or not profile["purpose"]:
        raise ValueError("profile purpose mismatch")
    decoder = _mapping(profile["decoder"], "decoder")
    decoder_path = _regular_hash_bound(root, decoder, "decoder")
    modes = profile["simulator_modes"]
    if (
        not isinstance(modes, list)
        or not modes
        or any(not isinstance(mode, str) or not mode for mode in modes)
        or len(set(modes)) != len(modes)
    ):
        raise ValueError("profile simulator modes mismatch")
    evidence = profile["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("profile evidence mismatch")
    evidence_paths: list[Path] = []
    for index, item_value in enumerate(evidence):
        item = _mapping(item_value, f"evidence {index}")
        _exact_keys(item, {"path", "sha256", "minimum_completed_transitions"}, f"evidence {index}")
        minimum = item["minimum_completed_transitions"]
        if type(minimum) is not int or minimum <= 0:
            raise ValueError("evidence minimum transition mismatch")
        path = _regular_hash_bound(root, {"path": item["path"], "sha256": item["sha256"]}, f"evidence {index}")
        report = _mapping(json.loads(path.read_text(encoding="utf-8")), f"evidence report {index}")
        authorization = _mapping(report.get("authorization"), f"evidence authorization {index}")
        if (
            report.get("passed") is not True
            or type(report.get("physical_dof")) is not int
            or report["physical_dof"] != 23
            or type(report.get("decoder_output_dof")) is not int
            or report["decoder_output_dof"] != 23
            or report.get("source_29dof_physics_used") is not False
            or authorization.get("hardware_authorized") is not False
            or report.get("decoder_sha256") != decoder["sha256"]
            or type(report.get("completed_transitions")) is not int
            or report["completed_transitions"] < minimum
        ):
            raise ValueError(f"evidence report {index} does not prove native23 mode")
        if report.get("kind") == "g1_true23_clean_mujoco_teleop_session" and (
            report.get("native23_profile") != profile_name
            or report.get("safety_fallback_enabled") is not False
            or report.get("gain_profile") != "released_retained"
        ):
            raise ValueError(f"evidence report {index} teleop profile mismatch")
        evidence_paths.append(path)
    if type(profile["live_transport_proven"]) is not bool:
        raise ValueError("live transport claim must be bool")
    if profile["live_headset_source_proven"] is not False:
        raise ValueError("live headset source must remain unproven")
    if profile_name in LIVE_TRANSPORT_PROFILES and profile["live_transport_proven"] is not True:
        raise ValueError("Pico transport evidence missing")
    if profile_name not in LIVE_TRANSPORT_PROFILES and profile["live_transport_proven"] is not False:
        raise ValueError("library profile cannot claim live transport")
    if profile_name == "pico_internet_fullbody_walk":
        transport_bindings = profile["profile_transport_evidence"]
        if not isinstance(transport_bindings, list) or len(transport_bindings) != 2:
            raise ValueError("profile transport evidence rows mismatch")
        for transport_index, transport_value in enumerate(transport_bindings):
            context = f"profile transport evidence {transport_index}"
            transport_binding = _mapping(transport_value, context)
            _exact_keys(
                transport_binding,
                {"expected_packet_count", "packet_bundle", "consumer", "publisher"},
                context,
            )
            expected_count = transport_binding["expected_packet_count"]
            if type(expected_count) is not int or expected_count <= 0:
                raise ValueError("profile transport packet count mismatch")
            packet_path = _regular_hash_bound(
                root,
                _mapping(transport_binding["packet_bundle"], "profile packet bundle"),
                "profile packet bundle",
            )
            consumer_report_path = _regular_hash_bound(
                root,
                _mapping(transport_binding["consumer"], "profile transport consumer"),
                "profile transport consumer",
            )
            publisher_report_path = _regular_hash_bound(
                root,
                _mapping(transport_binding["publisher"], "profile transport publisher"),
                "profile transport publisher",
            )
            packet_bundle = _mapping(json.loads(packet_path.read_text(encoding="utf-8")), "profile packet bundle")
            packet_rows = packet_bundle.get("robot_independent_reference_packets")
            consumer_report = _mapping(
                json.loads(consumer_report_path.read_text(encoding="utf-8")),
                "profile transport consumer report",
            )
            publisher_report = _mapping(
                json.loads(publisher_report_path.read_text(encoding="utf-8")),
                "profile transport publisher report",
            )
            consumer_auth = _mapping(
                consumer_report.get("authorization"), "profile transport consumer authorization"
            )
            publisher_auth = _mapping(
                publisher_report.get("authorization"), "profile transport publisher authorization"
            )
            if (
                not isinstance(packet_rows, list)
                or len(packet_rows) != expected_count
                or consumer_report.get("passed") is not True
                or consumer_report.get("native23_profile") != profile_name
                or consumer_report.get("decoder_sha256") != decoder["sha256"]
                or consumer_report.get("completed_transitions") != expected_count
                or consumer_report.get("live_transport_proven") is not True
                or not isinstance(consumer_report.get("maximum_reference_age_ns"), int)
                or consumer_report["maximum_reference_age_ns"] > 100_000_000
                or consumer_auth.get("hardware_authorized") is not False
                or publisher_report.get("passed") is not True
                or publisher_report.get("packet_count") != expected_count
                or publisher_report.get("pose_and_reference_values_unchanged") is not True
                or publisher_report.get("saved_packet_bundle_sha256") != sha256_file(packet_path)
                or publisher_auth.get("hardware_authorized") is not False
            ):
                raise ValueError("profile transport evidence report mismatch")
    minimum_height = profile["minimum_base_height_m"]
    maximum_tilt = profile["maximum_base_tilt_rad"]
    if (
        type(minimum_height) is not float
        or not 0.12 <= minimum_height <= 0.45
        or type(maximum_tilt) is not float
        or not 0.5 <= maximum_tilt <= 2.2
    ):
        raise ValueError("profile physical gate mismatch")
    if profile["sustained_deep_crouch_authorized"] is not False:
        raise ValueError("sustained deep crouch must remain disabled")
    return Native23ModeProfile(
        name=profile_name,
        purpose=profile["purpose"],
        encoder_path=encoder_path,
        encoder_sha256=encoder["sha256"],
        decoder_path=decoder_path,
        decoder_sha256=decoder["sha256"],
        simulator_modes=tuple(modes),
        minimum_base_height_m=minimum_height,
        maximum_base_tilt_rad=maximum_tilt,
        evidence_paths=tuple(evidence_paths),
        live_transport_proven=profile["live_transport_proven"],
        live_headset_source_proven=False,
        sustained_deep_crouch_authorized=False,
    )
