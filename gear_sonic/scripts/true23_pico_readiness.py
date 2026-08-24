#!/usr/bin/env python3
"""Fail-closed, non-actuating true23/PICO readiness inventory.

This command may query ADB package state and ICMP reachability. It never
imports Unitree SDK code, opens DDS, starts a robot process, or writes a robot
command. Human output goes to stderr; one complete JSON document goes to
stdout.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

HARDENED_PICO_PACKAGE = "com.xrobotoolkit.client.hardened"
REPORT_KIND = "g1_true23_pico_readiness"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    evidence: Mapping[str, Any]
    required: bool = True


@dataclass(frozen=True)
class Inputs:
    repo_root: Path
    adb: Path | None
    adb_serial: str | None
    pico_host: str | None
    robot_host: str | None
    checkpoint: Path | None
    simulation_report: Path | None
    encoder_onnx: Path | None
    decoder_onnx: Path | None
    metadata: Path | None
    shadow_binary: Path | None
    live_shadow_evidence: Path | None
    live_shadow_producer: Path | None
    timeout_s: float = 2.0
    now_utc: datetime | None = None


CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


def _pass(name: str, detail: str, **evidence: Any) -> Check:
    return Check(name, "PASS", detail, evidence)


def _fail(name: str, detail: str, **evidence: Any) -> Check:
    return Check(name, "FAIL", detail, evidence)


def _error(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return text[:500]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(list(command), 127, "", str(exc))


def _discover_adb(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    found = shutil.which("adb")
    if found:
        return Path(found).resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates = sorted(
            Path(local_app_data).glob(
                "Microsoft/WinGet/Packages/Google.PlatformTools_*/platform-tools/adb.exe"
            )
        )
        if candidates:
            return candidates[-1].resolve()
    return None


def _check_pico_adb(inputs: Inputs, runner: CommandRunner) -> Check:
    adb = _discover_adb(inputs.adb)
    if adb is None or not adb.is_file():
        return _fail("pico_adb_apk", "ADB executable unavailable")
    devices = runner((str(adb), "devices", "-l"), inputs.timeout_s)
    if devices.returncode:
        return _fail(
            "pico_adb_apk",
            f"`adb devices -l` failed: {_error(RuntimeError(devices.stderr))}",
            adb=str(adb),
        )
    records: dict[str, str] = {}
    for line in devices.stdout.splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and not line.startswith("List of devices"):
            records[fields[0]] = fields[1]
    authorized = sorted(serial for serial, state in records.items() if state == "device")
    if inputs.adb_serial is not None:
        if records.get(inputs.adb_serial) != "device":
            return _fail(
                "pico_adb_apk",
                f"requested PICO serial {inputs.adb_serial!r} is not authorized",
                adb=str(adb),
                device_states=records,
            )
        serial = inputs.adb_serial
    elif len(authorized) != 1:
        return _fail(
            "pico_adb_apk",
            f"need exactly one authorized ADB device; found {len(authorized)}",
            adb=str(adb),
            device_states=records,
        )
    else:
        serial = authorized[0]
    package = runner(
        (
            str(adb),
            "-s",
            serial,
            "shell",
            "pm",
            "path",
            HARDENED_PICO_PACKAGE,
        ),
        inputs.timeout_s,
    )
    installed = (
        package.returncode == 0
        and any(line.strip().startswith("package:") for line in package.stdout.splitlines())
    )
    if not installed:
        return _fail(
            "pico_adb_apk",
            f"{HARDENED_PICO_PACKAGE} not confirmed on authorized PICO",
            adb=str(adb),
            serial=serial,
        )
    return _pass(
        "pico_adb_apk",
        "one authorized ADB device exposes hardened XRoboToolkit package",
        adb=str(adb),
        serial=serial,
        package=HARDENED_PICO_PACKAGE,
    )


def _check_reachability(
    name: str,
    host: str | None,
    timeout_s: float,
    runner: CommandRunner,
) -> Check:
    if host is None:
        return _fail(name, "private IPv4 host not configured")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return _fail(name, f"invalid IPv4 address: {host!r}")
    if address.version != 4 or not address.is_private:
        return _fail(name, f"only explicit private IPv4 targets are allowed: {host!r}")
    if platform.system() == "Windows":
        command = ("ping", "-n", "1", "-w", str(max(1, round(timeout_s * 1000))), host)
    else:
        command = ("ping", "-c", "1", "-W", str(max(1, math.ceil(timeout_s))), host)
    result = runner(command, timeout_s + 1.0)
    if result.returncode:
        return _fail(name, f"ICMP probe failed for {host}", host=host, icmp_only=True)
    return _pass(
        name,
        f"{host} answered one ICMP probe; this does not prove application or DDS health",
        host=host,
        icmp_only=True,
    )


def _check_checkpoint(path: Path | None) -> Check:
    if path is None or not path.is_file():
        return _fail("trained_checkpoint", f"checkpoint missing: {path}")
    checkpoint_sha256 = _sha256(path)
    try:
        from gear_sonic.utils.g1_23dof_artifact import (
            inspect_true23_policy_state,
            validate_training_checkpoint_records,
        )
        from gear_sonic.utils.g1_23dof_checkpoint_io import (
            checkpoint_stage,
            extract_global_step,
            load_safe_true23_checkpoint,
        )
        from gear_sonic.utils.g1_23dof_contract import validate_artifact_contract

        checkpoint = load_safe_true23_checkpoint(
            path,
            map_location="cpu",
            allow_legacy_initialization=True,
        )
        metadata = checkpoint.get("g1_23dof_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("g1_23dof_metadata missing")
        validate_artifact_contract(
            metadata,
            decoder_input_dim=994,
            decoder_output_dim=23,
            require_deployment_ready=False,
        )
        stage = checkpoint_stage(checkpoint)
        policy_sha256 = inspect_true23_policy_state(checkpoint)
        if stage == "checkpoint_initialization":
            return _fail(
                "trained_checkpoint",
                "valid native true23 initialization checkpoint only; genuine retraining required",
                path=str(path.resolve()),
                sha256=checkpoint_sha256,
                checkpoint_stage=stage,
                policy_state_sha256=policy_sha256,
            )
        if stage != "trained":
            raise ValueError(f"unsupported checkpoint_stage {stage!r}")
        evidence = checkpoint.get("g1_23dof_training_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("training evidence missing")
        global_step = extract_global_step(checkpoint)
        if evidence.get("global_step") != global_step:
            raise ValueError("state.global_step disagrees with training evidence")
        validate_training_checkpoint_records(
            checkpoint,
            global_step=global_step,
            policy_state_sha256=policy_sha256,
        )
    except Exception as exc:
        return _fail(
            "trained_checkpoint",
            f"checkpoint rejected: {_error(exc)}",
            path=str(path.resolve()),
            sha256=checkpoint_sha256,
        )
    return _pass(
        "trained_checkpoint",
        f"native true23 trained checkpoint valid at global_step={global_step}",
        path=str(path.resolve()),
        sha256=checkpoint_sha256,
        checkpoint_stage="trained",
        global_step=global_step,
        policy_state_sha256=policy_sha256,
    )


def _check_simulation(path: Path | None, checkpoint: Path | None) -> Check:
    if path is None or not path.is_file():
        return _fail("simulation_evidence", f"simulation report missing: {path}")
    if checkpoint is None or not checkpoint.is_file():
        return _fail("simulation_evidence", "checkpoint missing; cannot verify report binding")
    try:
        from gear_sonic.utils.g1_23dof_artifact import (
            canonical_json_bytes,
            load_strict_json,
            sha256_bytes,
            validate_simulation_report,
        )
        from gear_sonic.utils.g1_23dof_checkpoint_io import (
            load_safe_true23_checkpoint,
        )

        raw = path.read_bytes()
        report = load_strict_json(path)
        checkpoint_payload = load_safe_true23_checkpoint(
            checkpoint,
            map_location="cpu",
        )
        reference_profile = checkpoint_payload["g1_23dof_metadata"][
            "reference_profile"
        ]
        summary = validate_simulation_report(
            report,
            checkpoint_sha256=_sha256(checkpoint),
            report_sha256=sha256_bytes(raw),
            report_payload_sha256=sha256_bytes(canonical_json_bytes(report)),
            report_path=path,
            reference_profile=reference_profile,
            checkpoint_motion_dataset=checkpoint_payload[
                "g1_23dof_training_evidence"
            ]["motion_dataset"],
        )
    except Exception as exc:
        return _fail(
            "simulation_evidence",
            f"raw simulation evidence rejected: {_error(exc)}",
            path=str(path.resolve()),
            sha256=_sha256(path),
        )
    return _pass(
        "simulation_evidence",
        "raw nominal/50%/100% disturbance traces recomputed and checkpoint-bound",
        path=str(path.resolve()),
        sha256=_sha256(path),
        total_episodes=summary["total_episodes"],
        total_steps=summary["total_steps"],
    )


def _check_artifact_pair(inputs: Inputs) -> Check:
    paths = (inputs.encoder_onnx, inputs.decoder_onnx, inputs.metadata)
    if any(path is None or not path.is_file() for path in paths):
        return _fail(
            "paired_onnx",
            "encoder, decoder, or metadata sidecar missing",
            paths=[str(path) for path in paths],
        )
    try:
        from gear_sonic.utils.g1_23dof_artifact import verify_validated_true23_artifact

        metadata = verify_validated_true23_artifact(
            inputs.encoder_onnx,
            inputs.decoder_onnx,
            inputs.metadata,
            checkpoint_path=inputs.checkpoint,
            simulation_report_path=inputs.simulation_report,
        )
    except Exception as exc:
        return _fail("paired_onnx", f"paired artifact rejected: {_error(exc)}")
    return _pass(
        "paired_onnx",
        "paired static ONNX, embedded metadata, hashes, bindings, and chained dry-run valid",
        encoder_sha256=_sha256(inputs.encoder_onnx),
        decoder_sha256=_sha256(inputs.decoder_onnx),
        metadata_sha256=_sha256(inputs.metadata),
        global_step=metadata["training_evidence"]["global_step"],
    )


def _binary_format(path: Path) -> str | None:
    magic = path.read_bytes()[:4]
    if magic == b"\x7fELF":
        return "ELF"
    if magic[:2] == b"MZ":
        return "PE"
    return None


def _check_shadow_binary(path: Path | None) -> Check:
    if path is None or not path.is_file():
        return _fail("shadow_binary", f"shadow binary missing: {path}")
    try:
        binary_format = _binary_format(path)
        if binary_format is None or path.stat().st_size < 1024:
            raise ValueError("file is not a materialized ELF/PE executable")
        digest = _sha256(path)
    except Exception as exc:
        return _fail("shadow_binary", f"shadow binary rejected: {_error(exc)}")
    return _pass(
        "shadow_binary",
        "shadow-only executable available; reporter did not execute or open DDS",
        path=str(path.resolve()),
        sha256=digest,
        format=binary_format,
    )


def _validate_live_evidence(inputs: Inputs, now_utc: datetime) -> Check:
    evidence_path = inputs.live_shadow_evidence
    producer_path = inputs.live_shadow_producer
    if producer_path is None or not producer_path.is_file():
        return _fail(
            "integrated_live_shadow",
            "approved integrated live-shadow producer missing; independent PICO/LowState "
            "checks cannot prove live policy observation/inference",
        )
    if evidence_path is None or not evidence_path.is_file():
        return _fail(
            "integrated_live_shadow",
            "fresh integrated live-shadow evidence missing",
            producer_sha256=_sha256(producer_path),
        )
    try:
        from gear_sonic.utils.g1_23dof_live_shadow import (
            validate_live_shadow_evidence,
        )
    except ImportError as exc:
        return _fail(
            "integrated_live_shadow",
            "approved integrated live-shadow validator unavailable; "
            f"weaker file/boolean evidence is never accepted: {_error(exc)}",
        )
    try:
        from gear_sonic.utils.g1_23dof_artifact import load_strict_json

        artifact_paths = {
            "checkpoint": inputs.checkpoint,
            "simulation_report": inputs.simulation_report,
            "encoder_onnx": inputs.encoder_onnx,
            "decoder_onnx": inputs.decoder_onnx,
            "metadata": inputs.metadata,
        }
        if any(path is None or not path.is_file() for path in artifact_paths.values()):
            raise ValueError(
                "all five trained artifact/evidence files are required for live binding"
            )
        summary = validate_live_shadow_evidence(
            load_strict_json(evidence_path),
            producer_path=producer_path,
            artifact_paths=artifact_paths,
            repo_root=inputs.repo_root,
            now_utc=now_utc,
        )
        if not isinstance(summary, Mapping):
            raise ValueError("live-shadow validator did not return computed summary")
    except Exception as exc:
        return _fail(
            "integrated_live_shadow",
            f"integrated live-shadow evidence rejected: {_error(exc)}",
            path=str(evidence_path.resolve()),
        )
    return _pass(
        "integrated_live_shadow",
        "approved validator recomputed fresh common-window PICO semantics, "
        "CRC-valid mode-4 LowState, canonical H10 fixed slots, and paired "
        "267->64->994->23 inference/output bindings",
        path=str(evidence_path.resolve()),
        sha256=_sha256(evidence_path),
        producer_sha256=_sha256(producer_path),
        summary=dict(summary),
    )


def collect(inputs: Inputs, runner: CommandRunner = _run) -> Mapping[str, Any]:
    now_utc = inputs.now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    checks = [
        _pass(
            "read_only_scope",
            "ADB package query + ICMP only; no DDS, robot SDK, process launch, or robot command",
            dds_opened=False,
            robot_command_sent=False,
        ),
        _check_pico_adb(inputs, runner),
        _check_reachability("pico_network", inputs.pico_host, inputs.timeout_s, runner),
        _check_reachability("robot_network", inputs.robot_host, inputs.timeout_s, runner),
        _check_checkpoint(inputs.checkpoint),
        _check_simulation(inputs.simulation_report, inputs.checkpoint),
        _check_artifact_pair(inputs),
        _check_shadow_binary(inputs.shadow_binary),
        _validate_live_evidence(inputs, now_utc),
    ]
    blockers = [f"{check.name}: {check.detail}" for check in checks if check.required and check.status != "PASS"]
    return {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "generated_at_utc": now_utc.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": str(inputs.repo_root.resolve()),
        "shadow_readiness": "READY" if not blockers else "NO-GO",
        "actuation_decision": "NO-GO",
        "robot_command_authorized": False,
        "gantry_test_authorized": False,
        "no_go_reasons": blockers + [
            "read-only readiness command never authorizes robot commands or gantry testing"
        ],
        "checks": [asdict(check) for check in checks],
    }


def _human(report: Mapping[str, Any]) -> str:
    lines = [
        f"True23/PICO shadow readiness: {report['shadow_readiness']}",
        "Robot actuation: NO-GO (read-only report; no command authority)",
    ]
    lines.extend(
        f"[{check['status']}] {check['name']}: {check['detail']}"
        for check in report["checks"]
    )
    lines.append("NO-GO reasons:")
    lines.extend(f"- {reason}" for reason in report["no_go_reasons"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Read-only true23/PICO readiness report (JSON stdout, human stderr)"
    )
    parser.add_argument("--adb", type=Path)
    parser.add_argument("--adb-serial")
    parser.add_argument("--pico-host", default=os.environ.get("PICO_IP"))
    parser.add_argument("--robot-host", default=os.environ.get("G1_ROBOT_IP"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "sonic_release/g1_23dof_rev_1_0_init.pt",
    )
    parser.add_argument("--simulation-report", type=Path)
    parser.add_argument("--encoder-onnx", type=Path)
    parser.add_argument("--decoder-onnx", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--shadow-binary",
        type=Path,
        default=repo_root / "gear_sonic_deploy/target/release/g1_true23_shadow_gate",
    )
    parser.add_argument("--live-shadow-evidence", type=Path)
    parser.add_argument(
        "--live-shadow-producer",
        type=Path,
        default=repo_root / "gear_sonic_deploy/target/release/g1_true23_live_shadow",
    )
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout must be positive and finite")
    inputs = Inputs(
        repo_root=Path(__file__).resolve().parents[2],
        adb=args.adb,
        adb_serial=args.adb_serial,
        pico_host=args.pico_host,
        robot_host=args.robot_host,
        checkpoint=args.checkpoint,
        simulation_report=args.simulation_report,
        encoder_onnx=args.encoder_onnx,
        decoder_onnx=args.decoder_onnx,
        metadata=args.metadata,
        shadow_binary=args.shadow_binary,
        live_shadow_evidence=args.live_shadow_evidence,
        live_shadow_producer=args.live_shadow_producer,
        timeout_s=args.timeout,
    )
    report = collect(inputs)
    if not args.json_only:
        print(_human(report), file=sys.stderr)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["shadow_readiness"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
