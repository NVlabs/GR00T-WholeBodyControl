#!/usr/bin/env python3
"""Non-actuating preflight for PICO whole-body teleoperation on Unitree G1."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time

SONIC_ENCODER_SHA256 = "60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b"
SONIC_DECODER_SHA256 = "c4ac2e74045e7cbfb568f15e6bf47ea7ce023df7a94322af50be223e0a628bab"
SONIC_OBSERVATION_CONFIG_SHA256 = "582b9a273a3d69fbf49ae59b39295a3be2b4a295e195ef4cf674b5e2571c90ab"
SONIC_PLANNER_SHA256 = "39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea"

G1_ACTIVE_MOTOR_COUNT = 29
HG_LOWSTATE_MOTOR_SLOTS = 35
TRUE23_HARDWARE_JOINT_IDS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    17,
    18,
    19,
    22,
    23,
    24,
    25,
    26,
)
TRUE23_REQUIRED_MODE_MACHINE = 4
_LOWSTATE_MIN_ADVANCING_TRANSITIONS = 3
_LOWSTATE_MIN_ADVANCE_RATE_HZ = 5.0
_LOWSTATE_MAX_ADVANCE_GAP_S = 0.5
_UINT32_MODULUS = 1 << 32
_MAX_ACTIVE_MOTOR_POSITION_RAD = 4.0 * math.pi
_MAX_ACTIVE_MOTOR_VELOCITY_RAD_S = 100.0
_MAX_ACTIVE_MOTOR_TORQUE_NM = 1000.0
_MIN_IMU_QUATERNION_NORM = 0.5
_MAX_IMU_QUATERNION_NORM = 1.5
_SONIC_COMPATIBLE_G1_MODE_MACHINES = {
    2: "29-DoF beta",
    5: "29-DoF rev1",
}
_SONIC_INCOMPATIBLE_G1_MODE_MACHINES = {
    1: "23-DoF beta",
    3: "29-DoF locked-waist beta",
    4: "23-DoF rev1",
    6: "29-DoF locked-waist rev1",
    9: "dual-arm",
}


@dataclass
class Result:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class _LowStateSample:
    arrival_s: float
    tick: int
    mode_machine: int


class Checks:
    def __init__(self) -> None:
        self.results: list[Result] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.results.append(Result(name=name, status=status, detail=detail))

    def pass_(self, name: str, detail: str) -> None:
        self.add(name, "PASS", detail)

    def warn(self, name: str, detail: str) -> None:
        self.add(name, "WARN", detail)

    def fail(self, name: str, detail: str) -> None:
        self.add(name, "FAIL", detail)

    @property
    def failed(self) -> bool:
        return any(result.status == "FAIL" for result in self.results)


def _run(command: list[str], timeout_s: float = 5.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"timed out after {timeout_s}s",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, stdout="", stderr=str(exc))


def _process_uses_executable(executable: Path) -> bool:
    expected = executable.resolve()
    for proc_exe in Path("/proc").glob("[0-9]*/exe"):
        try:
            if proc_exe.resolve() == expected:
                return True
        except OSError:
            continue
    return False


def _load_shared_library(checks: Checks, label: str, library: str) -> None:
    try:
        ctypes.CDLL(library)
    except OSError as exc:
        checks.fail(label, f"cannot load {library}: {exc}")
    else:
        checks.pass_(label, f"{library} loadable")


def _ubuntu_version() -> str | None:
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return None
    match = re.search(r'^VERSION_ID="?([^"\n]+)"?$', os_release.read_text(), re.MULTILINE)
    return match.group(1) if match else None


def _wsl_mirrored() -> bool | None:
    if "microsoft" not in platform.release().lower():
        return None
    active_mode = _run(["wslinfo", "--networking-mode"])
    if active_mode.returncode == 0:
        mode_lines = [line.strip().casefold() for line in active_mode.stdout.splitlines() if line.strip()]
        return bool(mode_lines and mode_lines[-1] == "mirrored")

    # Older WSL releases lack wslinfo. In that case only, inspect the current
    # Windows user's configuration as the best available compatibility check.
    windows_profile = _run(["cmd.exe", "/d", "/c", "echo %USERPROFILE%"])
    if windows_profile.returncode:
        return False
    profile_path = _run(["wslpath", "-u", windows_profile.stdout.strip()])
    if profile_path.returncode:
        return False
    config = Path(profile_path.stdout.strip()) / ".wslconfig"
    try:
        text = config.read_text(errors="replace")
    except OSError:
        return False
    return bool(re.search(r"^\s*networkingMode\s*=\s*mirrored\s*$", text, re.I | re.M))


def _interface_ipv4(interface: str) -> list[str]:
    result = _run(["ip", "-4", "-o", "addr", "show", "dev", interface])
    if result.returncode:
        return []
    return re.findall(r"\binet\s+([0-9.]+)/", result.stdout)


def _auto_robot_interface() -> tuple[str | None, list[str]]:
    result = _run(["ip", "-4", "-o", "addr", "show"])
    for line in result.stdout.splitlines():
        match = re.search(r"^\d+:\s+(\S+)\s+inet\s+(192\.168\.123\.\d+)/", line)
        if match:
            return match.group(1), [match.group(2)]
    return None, []


def _tensorrt_version(root: Path) -> tuple[str | None, str]:
    candidates = [
        root / "include" / "NvInferVersion.h",
        root / "include" / "x86_64-linux-gnu" / "NvInferVersion.h",
        root / "include" / "aarch64-linux-gnu" / "NvInferVersion.h",
    ]
    header = next((candidate for candidate in candidates if candidate.is_file()), None)
    if header is None:
        return None, f"NvInferVersion.h not found below {root}"
    text = header.read_text(errors="replace")
    defines = dict(
        re.findall(
            r"^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z0-9_]+)",
            text,
            re.MULTILINE,
        )
    )
    values: dict[str, str] = {}
    for key in ("MAJOR", "MINOR", "PATCH", "BUILD"):
        value = defines.get(f"NV_TENSORRT_{key}")
        seen: set[str] = set()
        while value is not None and not value.isdigit() and value not in seen:
            seen.add(value)
            value = defines.get(value)
        if value is not None and value.isdigit():
            values[key] = value
    if "MAJOR" not in values or "MINOR" not in values:
        return None, f"version macros missing in {header}"
    version = ".".join(values[key] for key in ("MAJOR", "MINOR", "PATCH") if key in values)
    return version, str(header)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_model(
    checks: Checks,
    path: Path,
    label: str,
    expected_sha256: str | None = None,
) -> None:
    if not path.is_file():
        checks.fail(label, f"missing: {path}")
    elif _is_lfs_pointer(path):
        checks.fail(label, f"Git LFS pointer, not model content: {path}")
    elif path.stat().st_size < 1024:
        checks.fail(label, f"unexpectedly small ({path.stat().st_size} bytes): {path}")
    elif expected_sha256 and (actual_sha256 := _sha256(path)) != expected_sha256:
        checks.fail(
            label,
            f"SHA256 mismatch: {actual_sha256}; expected {expected_sha256}",
        )
    else:
        checksum = f"; SHA256={expected_sha256}" if expected_sha256 else ""
        checks.pass_(
            label,
            f"{path} ({path.stat().st_size / 1_000_000:.1f} MB){checksum}",
        )


def _check_runtime_asset(checks: Checks, path: Path, label: str) -> None:
    if not path.is_file():
        checks.fail(label, f"missing: {path}")
    elif _is_lfs_pointer(path) or path.stat().st_size < 1024:
        checks.fail(label, f"missing Git LFS content: {path}")
    else:
        checks.pass_(label, f"{path} ({path.stat().st_size / 1_000_000:.1f} MB)")


def _finite_sequence(value, expected_size: int, label: str) -> tuple[float, ...]:
    try:
        values = tuple(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a numeric sequence") from exc
    if len(values) != expected_size:
        raise ValueError(f"{label} must contain {expected_size} values; got {len(values)}")

    converted = []
    for index, item in enumerate(values):
        try:
            scalar = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label}[{index}] is not numeric") from exc
        if not math.isfinite(scalar):
            raise ValueError(f"{label}[{index}] is non-finite")
        converted.append(scalar)
    return tuple(converted)


def _validate_lowstate_message(
    message,
    arrival_s: float,
    active_motor_ids: tuple[int, ...] = tuple(range(G1_ACTIVE_MOTOR_COUNT)),
) -> _LowStateSample:
    try:
        validated_arrival_s = float(arrival_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("callback arrival time is not numeric") from exc
    if not math.isfinite(validated_arrival_s) or validated_arrival_s < 0.0:
        raise ValueError(f"callback arrival time must be finite and nonnegative; got {arrival_s}")

    try:
        tick = operator.index(message.tick)
    except (AttributeError, TypeError) as exc:
        raise ValueError("tick is missing or not an integer") from exc
    if not 0 <= tick < _UINT32_MODULUS:
        raise ValueError(f"tick must be uint32; got {tick}")

    try:
        mode_machine = operator.index(message.mode_machine)
    except (AttributeError, TypeError) as exc:
        raise ValueError("mode_machine is missing or not an integer") from exc
    if not 0 <= mode_machine <= 0xFF:
        raise ValueError(f"mode_machine must be uint8; got {mode_machine}")

    try:
        motor_states = tuple(message.motor_state)
    except (AttributeError, TypeError) as exc:
        raise ValueError("motor_state is missing or not a sequence") from exc
    if len(motor_states) != HG_LOWSTATE_MOTOR_SLOTS:
        raise ValueError(
            "motor_state must use exact HG 35-slot representation "
            f"({len(active_motor_ids)} required body-index slots); got {len(motor_states)}"
        )
    if (
        not active_motor_ids
        or len(set(active_motor_ids)) != len(active_motor_ids)
        or any(
            isinstance(motor_index, bool)
            or not isinstance(motor_index, int)
            or not 0 <= motor_index < G1_ACTIVE_MOTOR_COUNT
            for motor_index in active_motor_ids
        )
    ):
        raise ValueError("active_motor_ids must be unique G1 body-index slots")
    for motor_index in active_motor_ids:
        motor_state = motor_states[motor_index]
        for field, limit in (
            ("q", _MAX_ACTIVE_MOTOR_POSITION_RAD),
            ("dq", _MAX_ACTIVE_MOTOR_VELOCITY_RAD_S),
            ("tau_est", _MAX_ACTIVE_MOTOR_TORQUE_NM),
        ):
            try:
                value = float(getattr(motor_state, field))
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"motor_state[{motor_index}].{field} is missing or not numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"motor_state[{motor_index}].{field} is non-finite")
            if abs(value) > limit:
                raise ValueError(f"motor_state[{motor_index}].{field} magnitude {abs(value):g} exceeds {limit:g}")

    try:
        imu_state = message.imu_state
    except AttributeError as exc:
        raise ValueError("imu_state is missing") from exc
    quaternion = _finite_sequence(
        getattr(imu_state, "quaternion", None),
        4,
        "imu_state.quaternion",
    )
    quaternion_norm = math.sqrt(sum(component * component for component in quaternion))
    if not _MIN_IMU_QUATERNION_NORM <= quaternion_norm <= _MAX_IMU_QUATERNION_NORM:
        raise ValueError(
            f"imu_state.quaternion norm {quaternion_norm:g} outside "
            f"[{_MIN_IMU_QUATERNION_NORM:g}, {_MAX_IMU_QUATERNION_NORM:g}]"
        )
    _finite_sequence(getattr(imu_state, "gyroscope", None), 3, "imu_state.gyroscope")

    return _LowStateSample(
        arrival_s=validated_arrival_s,
        tick=tick,
        mode_machine=mode_machine,
    )


def _assess_lowstate_samples(
    samples: list[_LowStateSample],
    invalid_count: int,
    invalid_details: list[str],
    interface: str,
    observation_start_s: float,
    observation_end_s: float,
    policy_profile: str = "released29",
) -> tuple[bool, str]:
    if policy_profile not in {"released29", "true23"}:
        raise ValueError(f"unsupported policy profile: {policy_profile}")
    window_s = max(0.0, observation_end_s - observation_start_s)
    callbacks = len(samples) + invalid_count
    if invalid_count:
        first_error = invalid_details[0] if invalid_details else "unknown validation error"
        return (
            False,
            f"{invalid_count}/{callbacks} invalid rt/lowstate callback(s) on {interface}; "
            f"first error: {first_error}",
        )
    if not samples:
        return False, f"zero rt/lowstate samples on {interface} after {window_s:.2f}s"

    ordered = sorted(samples, key=lambda sample: sample.arrival_s)
    advancing_samples = [ordered[0]]
    previous = ordered[0]
    duplicate_count = 0
    gaps = []
    for current in ordered[1:]:
        delta = (current.tick - previous.tick) % _UINT32_MODULUS
        if delta == 0:
            # CycloneDDS can deliver the same source sample more than once.
            # Ignore duplicates without refreshing the last advancing-source
            # timestamp, so repeated delivery cannot mask a stalled robot.
            duplicate_count += 1
            continue
        if delta >= _UINT32_MODULUS // 2:
            return (
                False,
                f"regressing rt/lowstate tick {previous.tick}->{current.tick} on {interface}",
            )
        gap_s = current.arrival_s - previous.arrival_s
        if gap_s < 0.0:
            return False, f"non-monotonic callback arrival time on {interface}"
        gaps.append(gap_s)
        advancing_samples.append(current)
        previous = current

    transitions = len(advancing_samples) - 1
    if transitions < _LOWSTATE_MIN_ADVANCING_TRANSITIONS:
        return (
            False,
            f"only {transitions} advancing rt/lowstate transition(s) on {interface} "
            f"during {window_s:.2f}s; need at least {_LOWSTATE_MIN_ADVANCING_TRANSITIONS}",
        )

    active_span_s = advancing_samples[-1].arrival_s - advancing_samples[0].arrival_s
    if active_span_s <= 0.0:
        return False, f"advancing rt/lowstate ticks had no measurable callback interval on {interface}"
    advance_rate_hz = transitions / active_span_s
    max_gap_s = max(gaps)
    final_age_s = max(0.0, observation_end_s - advancing_samples[-1].arrival_s)
    motor_detail = (
        "23 required body-index slots finite"
        if policy_profile == "true23"
        else "29 body-index slots finite"
    )
    evidence = (
        f"{len(advancing_samples)} advancing samples from {len(ordered)} callback(s) "
        f"in {window_s:.2f}s on {interface}; "
        f"tick {advancing_samples[0].tick}->{advancing_samples[-1].tick}; "
        f"rate={advance_rate_hz:.1f}Hz; max_gap={max_gap_s * 1000.0:.1f}ms; "
        f"final_age={final_age_s * 1000.0:.1f}ms; "
        f"duplicates={duplicate_count}; "
        f"mode_machine={ordered[-1].mode_machine}; "
        f"motor_state=35 slots/{motor_detail}; "
        "IMU quaternion=4, gyro=3 finite"
    )
    failures = []
    observed_mode_machines = {sample.mode_machine for sample in advancing_samples}
    if len(observed_mode_machines) != 1:
        failures.append(
            "mode_machine changed during probe: "
            + ", ".join(str(mode) for mode in sorted(observed_mode_machines))
        )
    else:
        mode_machine = next(iter(observed_mode_machines))
        if policy_profile == "true23":
            if mode_machine != TRUE23_REQUIRED_MODE_MACHINE:
                failures.append(
                    "true23 profile requires 23-DoF rev1 "
                    f"mode_machine={TRUE23_REQUIRED_MODE_MACHINE}; observed {mode_machine}"
                )
        else:
            if mode_machine in _SONIC_INCOMPATIBLE_G1_MODE_MACHINES:
                embodiment = _SONIC_INCOMPATIBLE_G1_MODE_MACHINES[mode_machine]
                failures.append(
                    f"detected {embodiment} (mode_machine={mode_machine}); "
                    "released SONIC requires a full 29-DoF G1"
                )
            elif mode_machine not in _SONIC_COMPATIBLE_G1_MODE_MACHINES:
                failures.append(
                    f"unknown G1 embodiment mode_machine={mode_machine}; "
                    "only full 29-DoF modes 2 and 5 are accepted"
                )
    if advance_rate_hz < _LOWSTATE_MIN_ADVANCE_RATE_HZ:
        failures.append(f"rate below {_LOWSTATE_MIN_ADVANCE_RATE_HZ:.1f}Hz")
    if max_gap_s > _LOWSTATE_MAX_ADVANCE_GAP_S:
        failures.append(f"gap exceeds {_LOWSTATE_MAX_ADVANCE_GAP_S:.1f}s")
    if final_age_s > _LOWSTATE_MAX_ADVANCE_GAP_S:
        failures.append(f"latest sample older than {_LOWSTATE_MAX_ADVANCE_GAP_S:.1f}s")
    if failures:
        return False, f"{evidence}; {'; '.join(failures)}"
    return True, evidence


def _load_lowstate_api():
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    return ChannelFactoryInitialize, ChannelSubscriber, LowState_


def _probe_lowstate(
    checks: Checks,
    interface: str,
    duration_s: float,
    *,
    policy_profile: str = "released29",
) -> None:
    try:
        channel_factory_initialize, channel_subscriber, lowstate_type = _load_lowstate_api()
    except ImportError as exc:
        checks.fail("Unitree LowState", f"unitree_sdk2py unavailable: {exc}")
        return

    lock = threading.Lock()
    samples: list[_LowStateSample] = []
    invalid_count = 0
    invalid_details: list[str] = []
    collecting = False

    def callback(message) -> None:
        nonlocal invalid_count
        arrival_s = time.monotonic()
        with lock:
            if not collecting:
                return
            try:
                active_motor_ids = (
                    TRUE23_HARDWARE_JOINT_IDS
                    if policy_profile == "true23"
                    else tuple(range(G1_ACTIVE_MOTOR_COUNT))
                )
                sample = _validate_lowstate_message(
                    message,
                    arrival_s,
                    active_motor_ids,
                )
            except Exception as exc:
                invalid_count += 1
                if len(invalid_details) < 3:
                    invalid_details.append(str(exc))
                return
            samples.append(sample)

    subscriber = None
    probe_error = None
    observation_start_s = time.monotonic()
    observation_end_s = observation_start_s
    try:
        channel_factory_initialize(0, interface)
        subscriber = channel_subscriber("rt/lowstate", lowstate_type)
        subscriber.Init(callback, 10)
        with lock:
            observation_start_s = time.monotonic()
            collecting = True
        deadline = observation_start_s + duration_s
        while (now := time.monotonic()) < deadline:
            time.sleep(min(0.05, deadline - now))
    except Exception as exc:
        probe_error = f"DDS probe failed on {interface}: {exc}"
    finally:
        with lock:
            observation_end_s = time.monotonic()
            collecting = False
        if subscriber is not None:
            try:
                subscriber.Close()
            except Exception as exc:
                cleanup_error = f"DDS subscriber cleanup failed on {interface}: {exc}"
                probe_error = f"{probe_error}; {cleanup_error}" if probe_error else cleanup_error

    with lock:
        observed_samples = list(samples)
        observed_invalid_count = invalid_count
        observed_invalid_details = list(invalid_details)
    if probe_error:
        checks.fail("Unitree LowState", probe_error)
        return

    passed, detail = _assess_lowstate_samples(
        observed_samples,
        observed_invalid_count,
        observed_invalid_details,
        interface,
        observation_start_s,
        observation_end_s,
        policy_profile,
    )
    if passed:
        checks.pass_("Unitree LowState", detail)
    else:
        checks.fail("Unitree LowState", detail)


def _assess_pico_tracker_health(
    snapshot: dict[str, object],
    *,
    previous_sequence: int | None = None,
) -> tuple[bool, str]:
    if snapshot.get("health_supported") is not True:
        return (
            False,
            "hardened PICO body-tracker health protocol missing; "
            "stock XRoboToolkit v1.1.1 cannot prove live ankle trackers",
        )
    if snapshot.get("health_available") is not True:
        return False, "body-tracker health telemetry unavailable; select Full body"

    def integer(key: str) -> int:
        value = snapshot[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        return value

    try:
        schema = integer("health_schema_version")
        sequence = integer("health_sample_sequence")
        timestamp_ns = integer("health_timestamp_ns")
        calibration_result = integer("health_calibration_result")
        tracking_mode = integer("health_tracking_mode")
        connect_result = integer("health_connect_state_result")
        tracker_count = integer("health_tracker_count")
        unique_count = integer("health_unique_tracker_count")
        body_state_result = integer("health_body_state_result")
        tracking_state_code = integer("health_tracking_state_code")
        body_state_code = integer("health_body_state_code")
        body_error_code = integer("health_body_error_code")
        connected_band_count = integer("health_connected_band_count")
        body_data_result = integer("health_body_data_result")
        body_role_count = integer("health_body_role_count")
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"body-tracker health telemetry malformed: {exc}"

    if schema != 1:
        return False, f"unsupported body-tracker health schema {schema}"
    if sequence <= 0 or timestamp_ns <= 0:
        return False, "body-tracker health sequence/timestamp is not positive"
    if calibration_result != 0:
        return False, f"tracker calibration query failed with code {calibration_result}"
    if snapshot.get("health_calibrated") is not True:
        return False, "PICO motion trackers are not calibrated"
    if tracking_mode != 0:
        return False, f"tracker runtime mode is {tracking_mode}, expected BodyTracking (0)"
    if connect_result != 0:
        return False, f"tracker connection query failed with code {connect_result}"
    if tracker_count < 2 or unique_count < 2:
        return (
            False,
            f"only {tracker_count} tracker(s), {unique_count} unique; at least two required",
        )
    if body_state_result != 0 or tracking_state_code != 0:
        return (
            False,
            "body state query failed "
            f"(result={body_state_result}, trackingState={tracking_state_code})",
        )
    if snapshot.get("health_is_tracking") is not True:
        return False, "PICO BodyTracking reports tracking lost"
    if body_state_code != 1:
        return (
            False,
            f"PICO BodyTracking is not BT_VALID "
            f"(state={body_state_code}, error={body_error_code})",
        )
    if connected_band_count < 2:
        return False, f"BodyTracking reports only {connected_band_count} connected band(s)"
    if body_data_result != 0:
        return False, f"body data query failed with code {body_data_result}"
    if body_role_count != 24:
        return False, f"body data has {body_role_count} roles, expected 24"
    if snapshot.get("health_valid") is not True:
        return False, "hardened PICO health gate reports invalid"
    if previous_sequence is not None and sequence <= previous_sequence:
        return False, "body-tracker health sequence did not advance"

    return (
        True,
        f"BT_VALID; calibrated; {tracker_count} connected, "
        f"{unique_count} unique; sequence {sequence} advancing",
    )


def _assess_pico_controller_tracking_health(
    snapshot: dict[str, object],
    *,
    previous_sequence: int | None = None,
) -> tuple[bool, str]:
    if snapshot.get("health_supported") is not True:
        return False, "hardened PICO controller-tracking health protocol missing"
    if snapshot.get("health_available") is not True:
        return False, "controller-tracking health telemetry unavailable; select Controller"

    def integer(key: str) -> int:
        value = snapshot[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        return value

    try:
        schema = integer("health_schema_version")
        sequence = integer("health_sample_sequence")
        timestamp_ns = integer("health_timestamp_ns")
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"controller-tracking health telemetry malformed: {exc}"

    if schema != 1:
        return False, f"unsupported controller-tracking health schema {schema}"
    if sequence <= 0 or timestamp_ns <= 0:
        return False, "controller-tracking health sequence/timestamp is not positive"

    for side in ("left", "right"):
        if snapshot.get(f"health_{side}_device_valid") is not True:
            return False, f"{side} controller device is invalid"
        if snapshot.get(f"health_{side}_is_tracked_available") is not True:
            return False, f"{side} controller tracked-state query is unavailable"
        if snapshot.get(f"health_{side}_is_tracked") is not True:
            return False, f"{side} controller reports tracking lost"
        if snapshot.get(f"health_{side}_tracking_state_available") is not True:
            return False, f"{side} controller tracking-state query is unavailable"
        try:
            tracking_state = integer(f"health_{side}_tracking_state")
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"{side} controller tracking state malformed: {exc}"
        if tracking_state & 3 != 3:
            return (
                False,
                f"{side} controller lacks position+rotation tracking "
                f"(state={tracking_state})",
            )
        if snapshot.get(f"health_{side}_valid") is not True:
            return False, f"{side} controller health gate reports invalid"

    if snapshot.get("health_valid") is not True:
        return False, "hardened PICO controller health gate reports invalid"
    if previous_sequence is not None and sequence <= previous_sequence:
        return False, "controller-tracking health sequence did not advance"

    return True, f"both controllers tracked in position+rotation; sequence {sequence} advancing"


def _wait_for_pico_body_snapshot(xrt: object, timeout_s: float) -> dict[str, object]:
    """Wait through service/client startup until Body arrives or timeout expires."""
    deadline = time.monotonic() + timeout_s
    snapshot = dict(xrt.get_body_snapshot())
    while not bool(snapshot.get("available")) and time.monotonic() < deadline:
        # xrt.init() resets the sticky protocol marker. An initial
        # health_supported=False snapshot is therefore normal before the first
        # hardened-client packet and must not be mistaken for a stock client.
        time.sleep(0.05)
        snapshot = dict(xrt.get_body_snapshot())
    return snapshot


def _probe_pico(checks: Checks, timeout_s: float) -> None:
    service = Path("/opt/apps/roboticsservice/runService.sh")
    service_binary = Path("/opt/apps/roboticsservice/RoboticsServiceProcess")
    if not service.is_file():
        checks.fail("PICO live tracking", f"XRoboToolkit service missing: {service}")
        return
    if not service_binary.is_file():
        checks.fail("PICO live tracking", f"XRoboToolkit binary missing: {service_binary}")
        return
    try:
        import numpy as np
        import xrobotoolkit_sdk as xrt

        from gear_sonic.utils.teleop.input_readers import validate_body_poses
    except ImportError as exc:
        checks.fail("PICO live tracking", f"XRoboToolkit Python SDK unavailable: {exc}")
        return

    try:
        if not _process_uses_executable(service_binary):
            subprocess.Popen(
                ["bash", str(service)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            service_deadline = time.monotonic() + min(timeout_s, 3.0)
            while time.monotonic() < service_deadline and not _process_uses_executable(service_binary):
                time.sleep(0.05)
            if not _process_uses_executable(service_binary):
                checks.fail("PICO live tracking", "XRoboToolkit service did not start")
                return

        xrt.init()
        if not callable(getattr(xrt, "get_body_snapshot", None)):
            checks.fail("PICO live tracking", "SDK getter missing: get_body_snapshot")
            return
        body_snapshot = _wait_for_pico_body_snapshot(xrt, timeout_s)
        if not bool(body_snapshot.get("available")):
            health_ok, health_detail = _assess_pico_tracker_health(body_snapshot)
            if health_ok:
                checks.fail(
                    "PICO tracker health",
                    "health reports valid but matching Body data is unavailable",
                )
            else:
                checks.fail("PICO tracker health", health_detail)
            checks.fail(
                "PICO live tracking",
                f"no body data within {timeout_s:.1f}s; check app IP, Send, Full body, and trackers",
            )
            return

        first_stamp = int(body_snapshot["timestamp_ns"])
        try:
            first_health_sequence = int(body_snapshot["health_sample_sequence"])
        except (KeyError, TypeError, ValueError):
            first_health_sequence = 0
        latest_stamp = first_stamp
        latest_body_snapshot = body_snapshot
        advancing_deadline = time.monotonic() + timeout_s
        latest_health_sequence = first_health_sequence
        while time.monotonic() < advancing_deadline and (
            latest_stamp <= first_stamp
            or latest_health_sequence <= first_health_sequence
        ):
            time.sleep(0.01)
            latest_body_snapshot = dict(xrt.get_body_snapshot())
            latest_stamp = int(latest_body_snapshot["timestamp_ns"])
            try:
                latest_health_sequence = int(
                    latest_body_snapshot["health_sample_sequence"]
                )
            except (KeyError, TypeError, ValueError):
                latest_health_sequence = 0
        poses = np.asarray(latest_body_snapshot["poses"])
        body_stream_valid = False
        if first_stamp <= 0 or latest_stamp <= first_stamp:
            checks.fail("PICO live tracking", "positive device timestamp did not advance")
        elif body_reason := validate_body_poses(poses):
            checks.fail("PICO live tracking", body_reason)
        else:
            body_stream_valid = True
            checks.pass_(
                "PICO live tracking",
                "advancing finite 24-joint Body API stream; "
                f"shape={poses.shape}; HEAD/body-source timestamp advanced",
            )

        health_ok, health_detail = _assess_pico_tracker_health(
            latest_body_snapshot,
            previous_sequence=first_health_sequence,
        )
        if health_ok:
            checks.pass_("PICO tracker health", health_detail)
        else:
            checks.fail("PICO tracker health", health_detail)

        controller_stream_valid = False
        final_body_stream_valid = False
        final_health_ok = False
        coherent_packet = False
        controller_getters = ("get_controller_snapshot",)
        missing = [name for name in controller_getters if not callable(getattr(xrt, name, None))]
        if missing:
            checks.fail("PICO controllers", f"SDK getters missing: {', '.join(missing)}")
        else:
            first_controller_snapshot = dict(xrt.get_controller_snapshot())
            first_controller_stamps = {
                side: int(first_controller_snapshot[f"{side}_timestamp_ns"]) for side in ("left", "right")
            }
            try:
                first_controller_health_sequence = int(
                    first_controller_snapshot["health_sample_sequence"]
                )
            except (KeyError, TypeError, ValueError):
                first_controller_health_sequence = 0
            latest_controller_stamps = dict(first_controller_stamps)
            latest_controller_snapshot = first_controller_snapshot
            latest_controller_health_sequence = first_controller_health_sequence
            coherent_body_snapshot = latest_body_snapshot
            controller_deadline = time.monotonic() + timeout_s
            while time.monotonic() < controller_deadline:
                time.sleep(0.01)
                coherent_body_snapshot = dict(xrt.get_body_snapshot())
                latest_controller_snapshot = dict(xrt.get_controller_snapshot())
                latest_controller_stamps = {
                    side: int(latest_controller_snapshot[f"{side}_timestamp_ns"]) for side in ("left", "right")
                }
                try:
                    latest_controller_health_sequence = int(
                        latest_controller_snapshot["health_sample_sequence"]
                    )
                    final_body_stamp = int(coherent_body_snapshot["timestamp_ns"])
                    final_body_health_sequence = int(
                        coherent_body_snapshot["health_sample_sequence"]
                    )
                    body_packet_timestamp_ns = int(
                        coherent_body_snapshot["health_timestamp_ns"]
                    )
                    controller_packet_timestamp_ns = int(
                        latest_controller_snapshot["health_timestamp_ns"]
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                coherent_packet = (
                    body_packet_timestamp_ns > 0
                    and body_packet_timestamp_ns == controller_packet_timestamp_ns
                )
                if (
                    coherent_packet
                    and all(
                        latest_controller_stamps[side]
                        > first_controller_stamps[side]
                        for side in ("left", "right")
                    )
                    and latest_controller_health_sequence
                    > first_controller_health_sequence
                    and final_body_stamp > latest_stamp
                    and final_body_health_sequence > latest_health_sequence
                ):
                    break

            controller_health_ok, controller_health_detail = (
                _assess_pico_controller_tracking_health(
                    latest_controller_snapshot,
                    previous_sequence=first_controller_health_sequence,
                )
            )
            if controller_health_ok:
                checks.pass_("PICO controller health", controller_health_detail)
            else:
                checks.fail("PICO controller health", controller_health_detail)

            final_health_ok, final_health_detail = _assess_pico_tracker_health(
                coherent_body_snapshot,
                previous_sequence=latest_health_sequence,
            )
            try:
                final_body_stamp = int(coherent_body_snapshot["timestamp_ns"])
                final_body_poses = np.asarray(coherent_body_snapshot["poses"])
            except (KeyError, TypeError, ValueError) as exc:
                final_body_reason = f"final Body value conversion failed: {exc}"
            else:
                if not bool(coherent_body_snapshot.get("available")):
                    final_body_reason = "final Body data is unavailable"
                elif final_body_stamp <= latest_stamp:
                    final_body_reason = "final Body timestamp did not advance"
                else:
                    final_body_reason = validate_body_poses(final_body_poses)
                    final_body_stream_valid = not final_body_reason

            if not coherent_packet:
                checks.fail(
                    "PICO coherent tracking frame",
                    "body and controller health did not arrive in the same source packet",
                )
            elif not final_health_ok:
                checks.fail("PICO coherent tracking frame", final_health_detail)
            elif not final_body_stream_valid:
                checks.fail("PICO coherent tracking frame", final_body_reason)
            else:
                checks.pass_(
                    "PICO coherent tracking frame",
                    "same-packet body/controller health stayed valid after sequential probes",
                )

            try:
                left_pose = np.asarray(latest_controller_snapshot["left_pose"], dtype=float)
                right_pose = np.asarray(latest_controller_snapshot["right_pose"], dtype=float)
                left_axis = np.asarray(latest_controller_snapshot["left_thumbstick"], dtype=float)
                right_axis = np.asarray(latest_controller_snapshot["right_thumbstick"], dtype=float)
                analog_values = np.asarray(
                    [
                        latest_controller_snapshot["left_trigger_value"],
                        latest_controller_snapshot["right_trigger_value"],
                        latest_controller_snapshot["left_squeeze_value"],
                        latest_controller_snapshot["right_squeeze_value"],
                    ],
                    dtype=float,
                )
                button_values = np.asarray(
                    [
                        latest_controller_snapshot["right_primary_click"],
                        latest_controller_snapshot["right_secondary_click"],
                        latest_controller_snapshot["left_primary_click"],
                        latest_controller_snapshot["left_secondary_click"],
                        latest_controller_snapshot["left_thumbstick_click"],
                        latest_controller_snapshot["right_thumbstick_click"],
                        latest_controller_snapshot["left_menu_button"],
                        latest_controller_snapshot["right_menu_button"],
                    ],
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError) as exc:
                checks.fail("PICO controllers", f"controller value conversion failed: {exc}")
            else:
                controller_poses = (left_pose, right_pose)
                controller_axes = (left_axis, right_axis)
                if not all(
                    latest_controller_stamps[side] > first_controller_stamps[side]
                    and latest_controller_stamps[side] > 0
                    for side in ("left", "right")
                ):
                    checks.fail(
                        "PICO controllers",
                        "coherent paired-controller frame timestamp did not advance",
                    )
                elif any(pose.shape != (7,) for pose in controller_poses):
                    checks.fail(
                        "PICO controllers",
                        f"unexpected controller pose shapes: {left_pose.shape}, {right_pose.shape}",
                    )
                elif any(axis.shape != (2,) for axis in controller_axes):
                    checks.fail(
                        "PICO controllers",
                        f"unexpected controller axis shapes: {left_axis.shape}, {right_axis.shape}",
                    )
                elif not all(np.isfinite(pose).all() for pose in controller_poses):
                    checks.fail("PICO controllers", "controller pose contains NaN or Inf")
                elif not all(np.isfinite(axis).all() for axis in controller_axes):
                    checks.fail("PICO controllers", "controller axis contains NaN or Inf")
                elif not np.isfinite(analog_values).all() or not np.isfinite(button_values).all():
                    checks.fail("PICO controllers", "controller input contains NaN or Inf")
                elif any(np.any(np.abs(pose[:3]) > 10.0) for pose in controller_poses):
                    checks.fail("PICO controllers", "controller position is outside safe bounds")
                elif not all(0.5 <= float(np.linalg.norm(pose[3:])) <= 1.5 for pose in controller_poses):
                    checks.fail(
                        "PICO controllers",
                        "one or both controller poses are not tracked (invalid quaternion)",
                    )
                elif any(np.any(np.abs(axis) > 1.25) for axis in controller_axes):
                    checks.fail("PICO controllers", "controller axis is outside expected range")
                elif np.any(analog_values < -0.05) or np.any(analog_values > 1.05):
                    checks.fail("PICO controllers", "controller trigger/grip is outside expected range")
                elif np.any(button_values < -0.05) or np.any(button_values > 1.05):
                    checks.fail("PICO controllers", "controller button is outside expected range")
                elif not controller_health_ok:
                    checks.fail(
                        "PICO controllers",
                        "numeric controller stream present but authoritative tracking health failed",
                    )
                elif not coherent_packet:
                    checks.fail(
                        "PICO controllers",
                        "controller stream was not coherent with a same-packet Body frame",
                    )
                else:
                    controller_stream_valid = True
                    checks.pass_(
                        "PICO controllers",
                        "advancing paired-controller frames are authoritatively tracked "
                        "with finite in-range poses and inputs",
                    )

        if (
            body_stream_valid
            and health_ok
            and final_body_stream_valid
            and final_health_ok
            and controller_stream_valid
            and coherent_packet
        ):
            checks.pass_(
                "PICO full-body safety gate",
                "same-packet Body/controller freshness, authoritative controller "
                "position+rotation tracking, BT_VALID state, calibration, successful "
                "body APIs, all 24 calculated roles, and live tracker counts passed; "
                "raw Motion mode is mutually exclusive and non-gating",
            )
    except Exception as exc:
        checks.fail("PICO live tracking", f"probe failed: {exc}")
    finally:
        try:
            xrt.close()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Non-actuating PICO 4 Ultra + Unitree G1 whole-body preflight")
    parser.add_argument(
        "--policy-profile",
        choices=("released29", "true23"),
        default="released29",
        help=(
            "released29 validates NVIDIA's released 29-DoF policy; true23 "
            "validates a trained, simulation-bound native 23-DoF artifact pair"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run dependency/asset diagnostics only; explicitly skip live PICO and G1 gates",
    )
    parser.add_argument(
        "--pico-only",
        action="store_true",
        help=(
            "Run dependency/asset checks plus live PICO full-body diagnostics; "
            "skip G1 network and LowState and do not assert robot readiness"
        ),
    )
    parser.add_argument("--robot-interface", help="Local robot-facing interface, e.g. eth0")
    parser.add_argument(
        "--robot-dof",
        type=int,
        choices=[23, 29],
        required=True,
        help="Physical G1 body DOF; must match --policy-profile",
    )
    parser.add_argument("--checkpoint", type=Path, help="True23 trained checkpoint")
    parser.add_argument(
        "--simulation-report",
        type=Path,
        help="True23 hash-bound IsaacLab simulation report",
    )
    parser.add_argument("--encoder-onnx", type=Path, help="True23 teleop encoder ONNX")
    parser.add_argument("--decoder-onnx", type=Path, help="True23 decoder ONNX")
    parser.add_argument("--metadata", type=Path, help="True23 pair metadata sidecar")
    parser.add_argument(
        "--probe-lowstate",
        action="store_true",
        help="Subscribe read-only to rt/lowstate on the robot interface",
    )
    parser.add_argument(
        "--lowstate-seconds",
        type=float,
        default=3.0,
        help="LowState probe window (default: 3.0)",
    )
    parser.add_argument(
        "--live-pico",
        action="store_true",
        help="Start XRoboToolkit service and verify live body/controller data",
    )
    parser.add_argument(
        "--pico-timeout",
        type=float,
        default=10.0,
        help="PICO live-data wait (default: 10.0)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def _validate_cli_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    true23_path_flags = {
        "--checkpoint": args.checkpoint,
        "--simulation-report": args.simulation_report,
        "--encoder-onnx": args.encoder_onnx,
        "--decoder-onnx": args.decoder_onnx,
        "--metadata": args.metadata,
    }
    if args.policy_profile == "true23":
        if args.robot_dof != 23:
            parser.error("--policy-profile true23 requires --robot-dof 23")
        missing = [flag for flag, value in true23_path_flags.items() if value is None]
        if missing:
            parser.error(
                "--policy-profile true23 requires " + ", ".join(missing)
            )
    else:
        if args.robot_dof != 29:
            parser.error("--policy-profile released29 requires --robot-dof 29")
        supplied = [flag for flag, value in true23_path_flags.items() if value is not None]
        if supplied:
            parser.error(
                "true23 artifact paths require --policy-profile true23: "
                + ", ".join(supplied)
            )

    if args.pico_only:
        conflicts = []
        if args.offline:
            conflicts.append("--offline")
        if args.robot_interface is not None:
            conflicts.append("--robot-interface")
        if args.probe_lowstate:
            conflicts.append("--probe-lowstate")
        if conflicts:
            parser.error("--pico-only cannot be combined with " + ", ".join(conflicts))
        args.live_pico = True
    elif args.offline:
        if args.live_pico or args.probe_lowstate:
            parser.error("--offline cannot be combined with live hardware probes")
    else:
        missing_live_gates = []
        if not args.robot_interface:
            missing_live_gates.append("--robot-interface")
        if not args.probe_lowstate:
            missing_live_gates.append("--probe-lowstate")
        if not args.live_pico:
            missing_live_gates.append("--live-pico")
        if missing_live_gates:
            parser.error(
                "live preflight requires "
                + ", ".join(missing_live_gates)
                + "; use --offline only for non-hardware diagnostics"
            )
    return args


def _run_requested_live_probes(
    checks: Checks,
    args: argparse.Namespace,
    interface: str | None,
) -> None:
    if args.pico_only:
        checks.warn(
            "PICO-only scope",
            "live PICO full-body diagnostic only; G1 robot interface and Unitree LowState "
            "intentionally skipped; no robot commands sent; result is not G1/robot-ready",
        )
    if args.pico_only:
        if args.live_pico:
            _probe_pico(checks, args.pico_timeout)
        return

    if args.probe_lowstate and not interface:
        checks.fail("Unitree LowState", "no robot interface available")
        if args.live_pico:
            _probe_pico(checks, args.pico_timeout)
        return

    if args.probe_lowstate and args.live_pico:
        # Full live readiness requires evidence from overlapping observation
        # windows. Separate result sinks avoid concurrent mutation and preserve
        # deterministic output order after both read-only probes finish.
        lowstate_checks = Checks()
        pico_checks = Checks()
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="pico-g1-readonly-preflight",
        ) as executor:
            if args.policy_profile == "true23":
                lowstate_future = executor.submit(
                    _probe_lowstate,
                    lowstate_checks,
                    interface,
                    args.lowstate_seconds,
                    policy_profile=args.policy_profile,
                )
            else:
                lowstate_future = executor.submit(
                    _probe_lowstate,
                    lowstate_checks,
                    interface,
                    args.lowstate_seconds,
                )
            pico_future = executor.submit(
                _probe_pico,
                pico_checks,
                args.pico_timeout,
            )
            try:
                lowstate_future.result()
            except Exception as exc:
                lowstate_checks.fail(
                    "Unitree LowState",
                    f"unexpected read-only probe failure: {exc}",
                )
            try:
                pico_future.result()
            except Exception as exc:
                pico_checks.fail(
                    "PICO live tracking",
                    f"unexpected read-only probe failure: {exc}",
                )
        checks.results.extend(lowstate_checks.results)
        checks.results.extend(pico_checks.results)
        return

    if args.probe_lowstate:
        if args.policy_profile == "true23":
            _probe_lowstate(
                checks,
                interface,
                args.lowstate_seconds,
                policy_profile=args.policy_profile,
            )
        else:
            _probe_lowstate(checks, interface, args.lowstate_seconds)
    if args.live_pico:
        _probe_pico(checks, args.pico_timeout)


def _check_policy_artifacts(
    checks: Checks,
    args: argparse.Namespace,
    deploy: Path,
) -> None:
    if args.policy_profile == "released29":
        _check_model(
            checks,
            deploy / "policy/low_latency/model_encoder.onnx",
            "SONIC encoder",
            SONIC_ENCODER_SHA256,
        )
        _check_model(
            checks,
            deploy / "policy/low_latency/model_decoder.onnx",
            "SONIC decoder",
            SONIC_DECODER_SHA256,
        )
        _check_model(
            checks,
            deploy / "policy/low_latency/observation_config.yaml",
            "SONIC observation config",
            SONIC_OBSERVATION_CONFIG_SHA256,
        )
        _check_model(
            checks,
            deploy / "planner/target_vel/V2/planner_sonic.onnx",
            "SONIC planner",
            SONIC_PLANNER_SHA256,
        )
        return

    try:
        from gear_sonic.utils.g1_23dof_artifact import (
            verify_validated_true23_artifact,
        )

        metadata = verify_validated_true23_artifact(
            args.encoder_onnx,
            args.decoder_onnx,
            args.metadata,
            checkpoint_path=args.checkpoint,
            simulation_report_path=args.simulation_report,
        )
    except Exception as exc:
        checks.fail(
            "True23 trained artifact pair",
            f"public verifier rejected checkpoint/simulation/ONNX binding: {exc}",
        )
    else:
        checks.pass_(
            "True23 trained artifact pair",
            "public verifier accepted trained checkpoint, raw simulation evidence, "
            "paired ONNX hashes/metadata, static shapes, and finite chained dry-run; "
            f"global_step={metadata['training_evidence']['global_step']}",
        )

    # Artifact verification plus independent PICO/LowState probes is not an
    # integrated live policy test. Keep real control closed until a read-only
    # runtime proves same-window PICO observation -> encoder -> decoder output
    # against advancing mode_machine==4 telemetry.
    checks.fail(
        "True23 integrated live inference",
        "missing approved integrated read-only producer proving same-window PICO "
        "observation [1,267] -> token [1,64] -> finite action [1,23] while "
        "advancing CRC-valid mode_machine=4 LowState; robot commands remain prohibited",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = _validate_cli_args(parser, parser.parse_args(argv))
    if (
        not math.isfinite(args.lowstate_seconds)
        or not math.isfinite(args.pico_timeout)
        or args.lowstate_seconds <= 0.0
        or args.pico_timeout <= 0.0
    ):
        raise SystemExit("probe timeouts must be positive and finite")

    checks = Checks()
    repo_root = Path(__file__).resolve().parents[2]
    machine = platform.machine().lower()

    ubuntu = _ubuntu_version()
    if ubuntu == "22.04":
        checks.pass_("Operating system", "Ubuntu 22.04")
    else:
        checks.fail("Operating system", f"Ubuntu 22.04 required; detected {ubuntu or platform.system()}")

    mirrored = _wsl_mirrored()
    if mirrored is True:
        checks.pass_("WSL networking", "mirrored mode active")
    elif mirrored is False:
        checks.fail("WSL networking", "WSL detected without networkingMode=mirrored")
    else:
        checks.pass_("WSL networking", "native Linux host")

    if sys.version_info[:2] == (3, 10):
        checks.pass_("Python", platform.python_version())
    else:
        checks.fail("Python", f"Python 3.10 required; detected {platform.python_version()}")

    if sys.prefix != sys.base_prefix:
        checks.pass_("Teleop venv", sys.prefix)
    else:
        checks.warn("Teleop venv", "not running from .venv_teleop")

    nvidia = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if nvidia.returncode == 0 and nvidia.stdout.strip():
        checks.pass_("NVIDIA GPU", nvidia.stdout.strip())
    else:
        checks.fail("NVIDIA GPU", nvidia.stderr.strip() or "nvidia-smi failed")

    try:
        import torch
    except Exception as exc:
        checks.fail("PyTorch CUDA", f"torch import failed: {exc}")
    else:
        torch_version = str(torch.__version__)
        torch_cuda = str(torch.version.cuda or "")
        torch_cuda_available = bool(torch.cuda.is_available())
        if machine not in ("aarch64", "arm64"):
            if torch_version.startswith("2.9.1") and torch_cuda.startswith("12.8") and torch_cuda_available:
                checks.pass_(
                    "PyTorch CUDA",
                    f"torch={torch_version}; compiled CUDA={torch_cuda}; GPU available",
                )
            else:
                checks.fail(
                    "PyTorch CUDA",
                    "desktop profile requires torch 2.9.1 CUDA 12.8 with an "
                    f"available GPU; detected torch={torch_version}, "
                    f"compiled CUDA={torch_cuda or 'none'}, "
                    f"available={torch_cuda_available}",
                )
        elif torch_cuda.startswith("12.") and torch_cuda_available:
            checks.pass_(
                "PyTorch CUDA",
                f"torch={torch_version}; compiled CUDA={torch_cuda}; GPU available",
            )
        else:
            checks.fail(
                "PyTorch CUDA",
                f"CUDA 12.x torch with an available GPU required; "
                f"detected torch={torch_version}, compiled CUDA={torch_cuda or 'none'}, "
                f"available={torch_cuda_available}",
            )

    nvcc = _run(["bash", "-lc", "command -v nvcc && nvcc --version"])
    cuda_match = re.search(r"\brelease\s+(\d+)\.(\d+)", nvcc.stdout)
    cuda_detail = next(
        (line.strip() for line in nvcc.stdout.splitlines() if "release" in line),
        nvcc.stdout.strip(),
    )
    if nvcc.returncode == 0 and cuda_match and cuda_match.group(1) == "12":
        checks.pass_("CUDA toolkit", cuda_detail)
    elif nvcc.returncode == 0 and nvcc.stdout.strip():
        checks.fail("CUDA toolkit", f"CUDA 12.x required; detected {cuda_detail}")
    else:
        checks.fail("CUDA toolkit", "nvcc unavailable")

    tensorrt_root = Path(os.environ.get("TensorRT_ROOT", Path.home() / "TensorRT")).expanduser()
    version, detail = _tensorrt_version(tensorrt_root)
    expected = "10.7" if machine in ("aarch64", "arm64") else "10.13"
    if version and version.startswith(f"{expected}."):
        checks.pass_("TensorRT", f"{version} ({detail})")
    elif version:
        checks.fail("TensorRT", f"{expected}.x required on {machine}; detected {version} ({detail})")
    else:
        checks.fail("TensorRT", detail)
    _load_shared_library(checks, "TensorRT runtime", "libnvinfer.so")

    onnx_root = Path("/opt/onnxruntime")
    onnx_version_file = onnx_root / "VERSION_NUMBER"
    try:
        onnx_version = onnx_version_file.read_text().strip()
    except OSError:
        checks.fail("ONNX Runtime", f"missing: {onnx_version_file}")
    else:
        if onnx_version == "1.16.3":
            checks.pass_("ONNX Runtime", f"{onnx_version} ({onnx_root})")
        else:
            checks.fail("ONNX Runtime", f"1.16.3 required; detected {onnx_version}")
    _load_shared_library(
        checks,
        "ONNX Runtime library",
        str(onnx_root / "lib/libonnxruntime.so"),
    )

    service = Path("/opt/apps/roboticsservice/runService.sh")
    if service.is_file():
        checks.pass_("XRoboToolkit service", str(service))
    else:
        checks.fail("XRoboToolkit service", f"missing: {service}")

    try:
        __import__("xrobotoolkit_sdk")
    except Exception as exc:
        checks.fail("XRoboToolkit Python SDK", f"import failed: {exc}")
    else:
        checks.pass_("XRoboToolkit Python SDK", "native extension imported")

    architecture_dir = "aarch64" if machine in ("aarch64", "arm64") else "x86_64"
    native_xrt_root = repo_root / "external_dependencies" / "XRoboToolkit-PC-Service-Pybind_X86_and_ARM64" / "lib"
    native_xrt = (
        native_xrt_root / "aarch64/libPXREARobotSDK.so"
        if architecture_dir == "aarch64"
        else native_xrt_root / "libPXREARobotSDK.so"
    )
    if native_xrt.is_file() and native_xrt.stat().st_size > 1_000_000:
        checks.pass_("XRoboToolkit native library", str(native_xrt))
    else:
        checks.fail("XRoboToolkit native library", f"missing/LFS pointer: {native_xrt}")

    sim_mesh_root = repo_root / "gear_sonic/data/robot_model/model_data/g1/meshes"
    sim_meshes = sorted(sim_mesh_root.glob("*.[Ss][Tt][Ll]"))
    invalid_sim_meshes = [path.name for path in sim_meshes if path.stat().st_size < 1_024 or _is_lfs_pointer(path)]
    if len(sim_meshes) < 60:
        checks.fail(
            "G1 simulation meshes",
            f"expected at least 60 STL assets under {sim_mesh_root}; found {len(sim_meshes)}",
        )
    elif invalid_sim_meshes:
        checks.fail(
            "G1 simulation meshes",
            f"missing/LFS assets: {', '.join(invalid_sim_meshes[:8])}",
        )
    else:
        checks.pass_("G1 simulation meshes", f"{len(sim_meshes)} materialized STL assets")

    deploy = repo_root / "gear_sonic_deploy"
    unitree_lib_root = deploy / "thirdparty/unitree_sdk2"
    _check_runtime_asset(
        checks,
        unitree_lib_root / f"lib/{architecture_dir}/libunitree_sdk2.a",
        "Unitree SDK library",
    )
    _check_runtime_asset(
        checks,
        unitree_lib_root / f"thirdparty/lib/{architecture_dir}/libddsc.so",
        "CycloneDDS C library",
    )
    _check_runtime_asset(
        checks,
        unitree_lib_root / f"thirdparty/lib/{architecture_dir}/libddscxx.so",
        "CycloneDDS C++ library",
    )
    reference_root = deploy / "reference/example"
    required_motion_files = {
        "body_ang_vel.csv",
        "body_lin_vel.csv",
        "body_pos.csv",
        "body_quat.csv",
        "joint_pos.csv",
        "joint_vel.csv",
    }
    motion_folders = (
        [path for path in reference_root.iterdir() if path.is_dir()] if reference_root.is_dir() else []
    )
    incomplete_folders = []
    for folder in motion_folders:
        valid_files = {
            path.name for path in folder.glob("*.csv") if path.stat().st_size >= 1024 and not _is_lfs_pointer(path)
        }
        if not required_motion_files.issubset(valid_files):
            incomplete_folders.append(folder.name)
    if not motion_folders:
        checks.fail("Reference motion", f"no motion folders under {reference_root}")
    elif incomplete_folders:
        checks.fail(
            "Reference motion",
            f"incomplete/LFS motion folders: {', '.join(incomplete_folders)}",
        )
    else:
        checks.pass_("Reference motion", f"{len(motion_folders)} complete six-CSV motion sets")

    _check_policy_artifacts(checks, args, deploy)

    if args.policy_profile == "released29":
        checks.pass_(
            "G1 embodiment assertion",
            "operator supplied 29-DoF; independently verify robot label/config before real control",
        )
    else:
        checks.pass_(
            "G1 embodiment assertion",
            "operator supplied native 23-DoF rev1 profile; live LowState must independently "
            "confirm mode_machine=4 and exact required slots",
        )

    interface = None
    addresses: list[str] = []
    if not args.pico_only:
        interface = args.robot_interface
        if interface:
            addresses = _interface_ipv4(interface)
        else:
            interface, addresses = _auto_robot_interface()
        if interface and any(address.startswith("192.168.123.") for address in addresses):
            checks.pass_("Robot network", f"{interface}: {', '.join(addresses)}")
        elif args.robot_interface:
            checks.fail(
                "Robot network",
                f"{args.robot_interface} lacks local 192.168.123.x IPv4; found {addresses or 'none'}",
            )
        else:
            checks.warn("Robot network", "no local 192.168.123.x interface connected")

    _run_requested_live_probes(checks, args, interface)

    if args.json:
        print(json.dumps([asdict(result) for result in checks.results], indent=2))
    else:
        for result in checks.results:
            print(f"[{result.status}] {result.name}: {result.detail}")
        failures = sum(result.status == "FAIL" for result in checks.results)
        warnings = sum(result.status == "WARN" for result in checks.results)
        print(f"\nSummary: {failures} failure(s), {warnings} warning(s)")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
