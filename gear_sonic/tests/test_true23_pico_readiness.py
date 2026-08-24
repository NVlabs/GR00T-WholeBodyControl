from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from gear_sonic.scripts import true23_pico_readiness as readiness
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes
from gear_sonic.utils.g1_23dof_contract import HARDWARE_JOINT_IDS
from gear_sonic.utils.g1_23dof_live_shadow import (
    LIVE_EVIDENCE_KIND,
    LIVE_PRODUCER_KIND,
)


def _inputs(tmp_path: Path, **overrides):
    values = {
        "repo_root": tmp_path,
        "adb": None,
        "adb_serial": None,
        "pico_host": None,
        "robot_host": None,
        "checkpoint": None,
        "simulation_report": None,
        "encoder_onnx": None,
        "decoder_onnx": None,
        "metadata": None,
        "shadow_binary": None,
        "live_shadow_evidence": None,
        "live_shadow_producer": None,
        "timeout_s": 0.1,
        "now_utc": datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return readiness.Inputs(**values)


def test_adb_probe_only_queries_devices_and_hardened_package(tmp_path):
    adb = tmp_path / "adb.exe"
    adb.write_bytes(b"MZ")
    commands = []

    def run(command, timeout_s):
        commands.append((tuple(command), timeout_s))
        if command[1:] == ("devices", "-l"):
            return subprocess.CompletedProcess(
                command,
                0,
                "List of devices attached\nPICO123 device product:pico\n",
                "",
            )
        assert command[1:] == (
            "-s",
            "PICO123",
            "shell",
            "pm",
            "path",
            readiness.HARDENED_PICO_PACKAGE,
        )
        return subprocess.CompletedProcess(
            command,
            0,
            "package:/data/app/hardened/base.apk\n",
            "",
        )

    result = readiness._check_pico_adb(_inputs(tmp_path, adb=adb), run)

    assert result.status == "PASS"
    assert result.evidence["serial"] == "PICO123"
    assert len(commands) == 2
    flat = " ".join(token for command, _timeout in commands for token in command)
    assert "install" not in flat
    assert "am start" not in flat
    assert "force-stop" not in flat


def test_network_probe_requires_explicit_private_ipv4_without_running_command():
    commands = []

    def run(command, timeout_s):
        commands.append((command, timeout_s))
        raise AssertionError("runner must not be called")

    assert readiness._check_reachability("pico_network", None, 1.0, run).status == "FAIL"
    assert (
        readiness._check_reachability(
            "robot_network",
            "8.8.8.8",
            1.0,
            run,
        ).status
        == "FAIL"
    )
    assert commands == []


def test_checked_in_initialization_checkpoint_is_explicit_no_go():
    repo_root = Path(__file__).resolve().parents[2]
    result = readiness._check_checkpoint(
        repo_root / "sonic_release/g1_23dof_rev_1_0_init.pt"
    )

    assert result.status == "FAIL"
    assert result.evidence["checkpoint_stage"] == "checkpoint_initialization"
    assert "genuine retraining required" in result.detail


def _materialize_live_fixture(tmp_path: Path):
    paths = {}
    for name in (
        "checkpoint",
        "simulation_report",
        "encoder_onnx",
        "decoder_onnx",
        "metadata",
    ):
        path = tmp_path / name
        path.write_bytes(f"{name}-content".encode())
        paths[name] = path
    producer = tmp_path / "g1_true23_integrated_shadow_probe"
    producer.write_bytes(b"\x7fELF" + b"\0" * 2048)
    evidence_path = tmp_path / "live.json"
    now = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)
    pico_samples = []
    lowstate_samples = []
    inference_samples = []
    for index in range(5):
        monotonic_ns = 1_100_000_000 + index * 100_000_000
        body_source_ns = 10_000 + index
        pico_samples.append(
            {
                "monotonic_ns": monotonic_ns,
                "body_source_ns": body_source_ns,
                "body_pose": [0.0] * 168,
                "tracker_ids": ["left-ankle", "right-ankle"],
                "tracking_state": "BT_VALID",
                "calibrated": True,
                "left_source_ns": 20_000 + index,
                "left_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "left_tracking_bits": 3,
                "right_source_ns": 30_000 + index,
                "right_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "right_tracking_bits": 3,
            }
        )
        lowstate_samples.append(
            {
                "monotonic_ns": monotonic_ns + 1,
                "tick": 100 + index,
                "mode_machine": 4,
                "crc_expected": 1234 + index,
                "crc_computed": 1234 + index,
                "hardware_joint_ids": list(HARDWARE_JOINT_IDS),
                "q": [0.0] * 23,
                "dq": [0.0] * 23,
            }
        )
        inference_samples.append(
            {
                "monotonic_ns": monotonic_ns + 2,
                "pico_body_source_ns": body_source_ns,
                "observation": [0.0] * 267,
                "token": [0.0] * 64,
                "action": [0.0] * 23,
            }
        )
    evidence = {
        "schema_version": 1,
        "kind": LIVE_EVIDENCE_KIND,
        "captured_at_utc": now.isoformat().replace("+00:00", "Z"),
        "producer": {
            "kind": LIVE_PRODUCER_KIND,
            "version": 1,
            "filename": producer.name,
            "sha256": readiness._sha256(producer),
        },
        "artifact_hashes": {
            name: readiness._sha256(path) for name, path in paths.items()
        },
        "window": {
            "start_monotonic_ns": 1_000_000_000,
            "end_monotonic_ns": 2_000_000_000,
        },
        "pico_samples": pico_samples,
        "lowstate_samples": lowstate_samples,
        "inference_samples": inference_samples,
    }
    evidence_path.write_bytes(canonical_json_bytes(evidence))
    inputs = _inputs(
        tmp_path,
        **paths,
        live_shadow_evidence=evidence_path,
        live_shadow_producer=producer,
        now_utc=now,
    )
    return inputs, evidence


def _install_live_validator(monkeypatch, validator):
    module = ModuleType("gear_sonic.utils.g1_23dof_live_shadow")
    module.validate_live_shadow_evidence = validator
    monkeypatch.setitem(
        sys.modules,
        "gear_sonic.utils.g1_23dof_live_shadow",
        module,
    )


def test_integrated_live_evidence_uses_only_approved_public_validator(
    tmp_path,
    monkeypatch,
):
    inputs, _evidence = _materialize_live_fixture(tmp_path)
    calls = []

    def validator(evidence, *, producer_path, artifact_paths, repo_root, now_utc):
        calls.append(
            (evidence, producer_path, artifact_paths, repo_root, now_utc)
        )
        return {"computed_pass": True, "sample_count": 5}

    _install_live_validator(monkeypatch, validator)

    result = readiness._validate_live_evidence(inputs, inputs.now_utc)

    assert result.status == "PASS"
    assert "267->64->994->23" in result.detail
    assert len(calls) == 1
    assert set(calls[0][2]) == {
        "checkpoint",
        "simulation_report",
        "encoder_onnx",
        "decoder_onnx",
        "metadata",
    }


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda evidence: evidence.update(
                {
                    "captured_at_utc": (
                        datetime(2026, 7, 30, tzinfo=timezone.utc)
                        - timedelta(minutes=2)
                    ).isoformat()
                }
            ),
            "stale or future-dated",
        ),
        (
            lambda evidence: evidence["lowstate_samples"][2].update(
                {"mode_machine": 5}
            ),
            "mode_machine is not 4",
        ),
        (
            lambda evidence: evidence["inference_samples"][1].update(
                {"action": [0.0] * 22}
            ),
            "must contain 23 values",
        ),
        (
            lambda evidence: evidence["artifact_hashes"].update(
                {"encoder_onnx": "0" * 64}
            ),
            "artifact hash mismatch",
        ),
    ],
)
def test_integrated_live_evidence_fails_closed(
    tmp_path,
    monkeypatch,
    mutate,
    reason,
):
    inputs, evidence = _materialize_live_fixture(tmp_path)
    mutate(evidence)
    inputs.live_shadow_evidence.write_bytes(canonical_json_bytes(evidence))

    def reject(*_args, **_kwargs):
        raise ValueError(reason)

    _install_live_validator(monkeypatch, reject)

    result = readiness._validate_live_evidence(inputs, inputs.now_utc)

    assert result.status == "FAIL"
    assert reason in result.detail


def test_hand_authored_pass_boolean_is_not_live_evidence(tmp_path, monkeypatch):
    inputs, _evidence = _materialize_live_fixture(tmp_path)
    inputs.live_shadow_evidence.write_bytes(canonical_json_bytes({"passed": True}))

    def reject(evidence, **_kwargs):
        if set(evidence) != {
            "schema_version",
            "kind",
            "captured_at_utc",
            "producer",
            "artifact_hashes",
            "window",
            "pico_samples",
            "lowstate_samples",
            "inference_samples",
        }:
            raise ValueError("live evidence must contain exact keys")
        return {"computed_pass": True}

    _install_live_validator(monkeypatch, reject)

    result = readiness._validate_live_evidence(inputs, inputs.now_utc)

    assert result.status == "FAIL"
    assert "exact keys" in result.detail


def test_collect_is_always_non_actuating_and_lists_explicit_blockers(tmp_path):
    commands = []

    def run(command, timeout_s):
        commands.append((command, timeout_s))
        raise AssertionError("no command should run without configured ADB/hosts")

    report = readiness.collect(
        _inputs(tmp_path, adb=tmp_path / "missing-adb"),
        runner=run,
    )

    assert report["shadow_readiness"] == "NO-GO"
    assert report["actuation_decision"] == "NO-GO"
    assert report["robot_command_authorized"] is False
    assert report["gantry_test_authorized"] is False
    assert any("trained_checkpoint" in reason for reason in report["no_go_reasons"])
    assert any("integrated_live_shadow" in reason for reason in report["no_go_reasons"])
    assert commands == []
