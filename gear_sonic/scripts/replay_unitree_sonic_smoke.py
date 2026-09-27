#!/usr/bin/env python3
"""Replay converted Unitree SONIC episodes through deployment and MuJoCo.

Heavy runtime dependencies (MuJoCo, ZMQ, PyArrow, and the simulator stack) are
loaded only after CLI/provenance validation. Importing this module is therefore
safe in the lightweight dataset-conversion environment.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Protocol

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from gear_sonic.data.unitree_conversion.replay import (
    AuthenticatedDataset,
    ProvenanceError,
    ReplayActionFrame,
    ReplayMetricsCollector,
    ReplayReport,
    ReplaySample,
    SourceEpisodeRef,
    aggregate_replay_reports,
    authenticate_dataset,
    collector_from_mujoco_model,
    guarded_authenticated_read,
    max_mujoco_contact_force,
    resolve_source_episode_ids,
    run_replay_schedule,
    stop_owned_process,
)

REPLAY_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "action.motion_token",
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPLOY_BINARY = REPOSITORY_ROOT / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
DEFAULT_POLICY_FILE = REPOSITORY_ROOT / "gear_sonic_deploy/policy/low_latency/model_decoder.onnx"
DEFAULT_MOTION_DATA = REPOSITORY_ROOT / "gear_sonic_deploy/reference/example"
DEFAULT_PLANNER_FILE = REPOSITORY_ROOT / "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
DEFAULT_OBSERVATION_CONFIG = REPOSITORY_ROOT / "gear_sonic_deploy/policy/low_latency/observation_config.yaml"
DEFAULT_ENCODER_FILE = REPOSITORY_ROOT / "gear_sonic_deploy/policy/low_latency/model_encoder.onnx"
DEX3_DATASET_DIRECTORY = "unitreerobotics--G1_Dex3_Pouring_Dataset"
PINNED_SONIC_MODEL_REVISION = "9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2"
PINNED_ENCODER_URI = f"hf://models/nvidia/GEAR-SONIC@{PINNED_SONIC_MODEL_REVISION}/low_latency/model_encoder.onnx"
PINNED_OBSERVATION_CONFIG_URI = (
    f"hf://models/nvidia/GEAR-SONIC@{PINNED_SONIC_MODEL_REVISION}/low_latency/observation_config.yaml"
)
PINNED_ENCODER = (45933505, "60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b")
PINNED_OBSERVATION_CONFIG = (
    3258,
    "582b9a273a3d69fbf49ae59b39295a3be2b4a295e195ef4cf674b5e2571c90ab",
)
CONTROL_PERIOD_S = 1.0 / 50.0
UNITREE_BRIDGE_CHANNEL_ATTRIBUTES = (
    "low_state_puber",
    "odo_state_puber",
    "torso_imu_puber",
    "left_hand_state_puber",
    "right_hand_state_puber",
    "low_cmd_suber",
    "left_hand_cmd_suber",
    "right_hand_cmd_suber",
    "wireless_controller_puber",
)


class TimelineError(ValueError):
    """Target episode ordering or 50 Hz timeline is invalid."""


class TargetValidationError(ValueError):
    """Target action fields or schema cannot be replayed safely."""


def classify_replay_error(error: BaseException) -> str:
    if isinstance(error, ProvenanceError):
        return "provenance_error"
    if isinstance(error, TimelineError):
        return "timeline_error"
    if isinstance(error, TargetValidationError):
        return "target_validation_error"
    return "replay_acceptance_error"


def artifact_identity(path: str | Path) -> dict[str, object]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ProvenanceError(f"artifact must be a regular non-symlink file: {candidate}")
    resolved = candidate.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def verify_pinned_encoder_artifacts(
    encoder: str | Path,
    observation_config: str | Path,
    *,
    _expected: Mapping[str, tuple[int, str]] | None = None,
) -> dict[str, dict[str, object]]:
    expected = _expected or {
        "encoder": PINNED_ENCODER,
        "observation_config": PINNED_OBSERVATION_CONFIG,
    }
    result = {
        "encoder": artifact_identity(encoder),
        "observation_config": artifact_identity(observation_config),
    }
    for name, identity in result.items():
        expected_size, expected_sha256 = expected[name]
        if identity["size"] != expected_size or identity["sha256"] != expected_sha256:
            raise ProvenanceError(f"pinned {name} size or SHA-256 mismatch")
    return result


def resolve_replay_dataset_root(path: str | Path) -> tuple[Path, Path]:
    requested = Path(path).expanduser().resolve(strict=True)
    if not requested.is_dir():
        raise ProvenanceError("replay dataset root must be a directory")
    markers = ("source-manifest.json", "dataset-checksums.sha256")
    if all((requested / marker).is_file() for marker in markers):
        return requested, requested.parent / f"{requested.name}-replay-report.json"
    candidates = sorted(
        candidate.resolve()
        for candidate in requested.rglob(DEX3_DATASET_DIRECTORY)
        if candidate.is_dir() and all((candidate / marker).is_file() for marker in markers)
    )
    if len(candidates) != 1:
        raise ProvenanceError(
            f"output root must contain exactly one unique {DEX3_DATASET_DIRECTORY} child; got {len(candidates)}"
        )
    return candidates[0], requested / "unitree-sonic-replay-report.json"


def replay_protocol(sim_timestep_s: float) -> dict[str, object]:
    if not math.isfinite(sim_timestep_s) or sim_timestep_s <= 0.0:
        raise ValueError("sim_timestep_s must be finite and positive")
    return {
        "control_dt_s": CONTROL_PERIOD_S,
        "sim_timestep_s": sim_timestep_s,
        "exclude_before_s": 1.0,
        "final_hold_s": 1.0,
        "seed": 0,
        "warmup_s": 2.0,
    }


@dataclass(frozen=True)
class DeploymentConfig:
    binary: Path
    network_interface: str
    policy_file: Path
    motion_data_path: Path
    planner_file: Path
    observation_config: Path
    encoder_file: Path
    zmq_host: str = "127.0.0.1"
    action_port: int = 5556
    state_port: int = 5557

    def __post_init__(self) -> None:
        if not self.network_interface:
            raise ValueError("network_interface must not be empty")
        if not self.zmq_host:
            raise ValueError("zmq_host must not be empty")
        for name in ("action_port", "state_port"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise ValueError(f"{name} must be an integer TCP port")
        if self.action_port == self.state_port:
            raise ValueError("action_port and state_port must differ")


class ReplayRuntime(Protocol):
    collector: ReplayMetricsCollector

    def prime(
        self,
        frame: ReplayActionFrame,
        *,
        timeout_s: float,
        pace: Callable[[float], None],
    ) -> None: ...

    def publish(
        self,
        frame: ReplayActionFrame,
        phase: str,
        source_index: int,
        time_s: float,
    ) -> None: ...

    def step(self, time_s: float) -> ReplaySample: ...

    def assert_action_ack(
        self,
        frame: ReplayActionFrame,
        phase: str,
        source_index: int,
        time_s: float,
    ) -> None: ...

    def stop_control(self) -> None: ...

    def close(self) -> None: ...


class DeadlinePacer:
    """Pace simulator steps against an absolute monotonic timeline."""

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._monotonic = monotonic
        self._sleep = sleep
        self._start = monotonic()
        self._period_s: float | None = None
        self._step_count = 0

    def __call__(self, period_s: float) -> None:
        if not math.isfinite(period_s) or period_s <= 0.0:
            raise ValueError("pacing period must be finite and positive")
        if self._period_s is None:
            self._period_s = period_s
        elif period_s != self._period_s:
            raise ValueError("pacing period cannot change during replay")
        self._step_count += 1
        deadline = self._start + self._step_count * period_s
        remaining = deadline - self._monotonic()
        if remaining > 0.0:
            self._sleep(remaining)


def build_deployment_command(config: DeploymentConfig) -> list[str]:
    """Build the deployed low-latency invocation with no alternate input path."""

    return [
        str(config.binary),
        config.network_interface,
        str(config.policy_file),
        str(config.motion_data_path),
        "--planner-file",
        str(config.planner_file),
        "--obs-config",
        str(config.observation_config),
        "--encoder-file",
        str(config.encoder_file),
        "--input-type",
        "zmq_manager",
        "--output-type",
        "zmq",
        "--zmq-host",
        config.zmq_host,
        "--zmq-port",
        str(config.action_port),
        "--zmq-topic",
        "pose",
        "--zmq-out-port",
        str(config.state_port),
        "--zmq-out-topic",
        "g1_debug",
        "--disable-crc-check",
    ]


def build_execution_provenance(
    config: DeploymentConfig,
    dataset: AuthenticatedDataset,
    *,
    runtime_identities: Mapping[str, Mapping[str, object]],
    _pinned_expected: Mapping[str, tuple[int, str]] | None = None,
) -> dict[str, object]:
    pinned = verify_pinned_encoder_artifacts(
        config.encoder_file,
        config.observation_config,
        _expected=_pinned_expected,
    )
    expected_conversion_identity = {
        "encoder_sha256": pinned["encoder"]["sha256"],
        "encoder_config_sha256": pinned["observation_config"]["sha256"],
    }
    for name, expected_sha256 in expected_conversion_identity.items():
        if dataset.conversion_identity.get(name) != expected_sha256:
            raise ProvenanceError(
                f"dataset conversion identity {name} does not match the pinned deployment artifact"
            )
    pinned["encoder"]["immutable_uri"] = PINNED_ENCODER_URI
    pinned["observation_config"]["immutable_uri"] = PINNED_OBSERVATION_CONFIG_URI
    return {
        "command": build_deployment_command(config),
        "artifacts": {
            "binary": artifact_identity(config.binary),
            "decoder": artifact_identity(config.policy_file),
            "planner": artifact_identity(config.planner_file),
            **pinned,
        },
        "dataset": {
            "root": str(dataset.root),
            "source_manifest_sha256": dataset.source_manifest_sha256,
            "dataset_checksums_sha256": dataset.dataset_checksums_sha256,
            "conversion_identity": dict(dataset.conversion_identity),
        },
        "runtime": {name: dict(identity) for name, identity in sorted(runtime_identities.items())},
    }


def frames_from_columns(
    columns: Mapping[str, Sequence[object]],
    *,
    target_episode_index: int,
    expected_length: int,
) -> tuple[ReplayActionFrame, ...]:
    """Validate one target-local Parquet episode and construct wire actions."""

    missing = [name for name in REPLAY_COLUMNS if name not in columns]
    if missing:
        raise TargetValidationError(f"replay dataset columns are missing: {missing}")
    if isinstance(target_episode_index, bool) or not isinstance(target_episode_index, int):
        raise TargetValidationError("target_episode_index must be an integer")
    if isinstance(expected_length, bool) or not isinstance(expected_length, int) or expected_length <= 0:
        raise TimelineError("expected_length must be a positive integer")
    if any(len(columns[name]) != expected_length for name in REPLAY_COLUMNS):
        raise TimelineError("replay columns do not match the immutable episode length")

    episode_indices = np.asarray(columns["episode_index"])
    frame_indices = np.asarray(columns["frame_index"])
    timestamps = np.asarray(columns["timestamp"])
    if episode_indices.dtype.kind not in "iu" or not np.array_equal(
        episode_indices, np.full(expected_length, target_episode_index, dtype=episode_indices.dtype)
    ):
        raise TimelineError("episode_index does not match the provenance-resolved target index")
    if frame_indices.dtype.kind not in "iu" or not np.array_equal(
        frame_indices, np.arange(expected_length, dtype=frame_indices.dtype)
    ):
        raise TimelineError("frame_index must be the exact contiguous local timeline")
    expected_timestamps = np.asarray(
        [np.float32(index / 50.0) for index in range(expected_length)], dtype=np.float32
    )
    try:
        actual_timestamps = timestamps.astype(np.float32)
    except (TypeError, ValueError) as error:
        raise TimelineError("timestamp must contain numeric 50 Hz values") from error
    if not np.all(np.isfinite(actual_timestamps)) or not np.array_equal(actual_timestamps, expected_timestamps):
        raise TimelineError("timestamp must equal float32(frame_index / 50) exactly")

    frames: list[ReplayActionFrame] = []
    for index in range(expected_length):
        try:
            frames.append(
                ReplayActionFrame(
                    motion_token=np.asarray(columns["action.motion_token"][index], dtype=np.float32),
                    left_hand=np.asarray(columns["teleop.left_hand_joints"][index], dtype=np.float32),
                    right_hand=np.asarray(columns["teleop.right_hand_joints"][index], dtype=np.float32),
                )
            )
        except (TypeError, ValueError) as error:
            raise TargetValidationError(f"target action frame {index} is invalid: {error}") from error
    return tuple(frames)


def _arrow_column_values(table: object, name: str) -> list[object]:
    try:
        column = table.column(name)
        if hasattr(column, "combine_chunks"):
            column = column.combine_chunks()
        return column.to_pylist()
    except (AttributeError, KeyError) as error:
        raise TargetValidationError(f"replay Parquet is missing or cannot decode column {name}") from error


def _read_authenticated_replay_table(
    parquet_path: Path,
    expected_digest: str,
    reader: Callable[[Path], object],
) -> object:
    try:
        return guarded_authenticated_read(parquet_path, expected_digest, reader)
    except ProvenanceError:
        raise
    except Exception as error:
        raise TargetValidationError("replay Parquet cannot be decoded") from error


def load_episode_frames(
    dataset_root: Path,
    episode: SourceEpisodeRef,
    *,
    authenticated: AuthenticatedDataset | None = None,
) -> tuple[ReplayActionFrame, ...]:
    """Load a provenance-resolved local episode. Heavy readers stay lazy."""

    import pyarrow.parquet as pq

    from gear_sonic.data.exporter import Gr00tDatasetMetadata

    verified = authenticate_dataset(dataset_root) if authenticated is None else authenticated
    root = dataset_root.resolve(strict=True)
    if verified.root != root:
        raise ProvenanceError("authenticated dataset root does not match the requested replay root")
    metadata = Gr00tDatasetMetadata(repo_id="tmp/unitree_sonic_replay", root=root)
    relative = Path(metadata.get_data_file_path(episode.target_episode_index))
    parquet_path = (root / relative).resolve(strict=True)
    try:
        parquet_path.relative_to(root)
    except ValueError as error:
        raise ProvenanceError("resolved replay Parquet escapes the dataset root") from error
    if parquet_path.is_symlink() or not parquet_path.is_file():
        raise ProvenanceError("resolved replay Parquet is missing or unsafe")
    relative_text = parquet_path.relative_to(root).as_posix()
    expected_digest = verified.artifact_sha256.get(relative_text)
    if expected_digest is None:
        raise ProvenanceError("resolved replay Parquet is absent from the authenticated checksum tree")
    table = _read_authenticated_replay_table(
        parquet_path,
        expected_digest,
        lambda path: pq.read_table(path, columns=list(REPLAY_COLUMNS)),
    )
    columns = {name: _arrow_column_values(table, name) for name in REPLAY_COLUMNS}
    return frames_from_columns(
        columns,
        target_episode_index=episode.target_episode_index,
        expected_length=episode.episode_length,
    )


def quaternion_wxyz_to_roll_pitch(quaternion: np.ndarray) -> tuple[float, float]:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("root quaternion must be a finite WXYZ vector")
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("root quaternion must have nonzero norm")
    w, x, y, z = value / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def select_robot_actuator_joint_ids(
    model: object,
    robot_joint_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair exactly one actuator with each loaded body/hand robot joint."""

    joints = np.asarray(robot_joint_ids)
    if joints.ndim != 1 or joints.size == 0 or joints.dtype.kind not in "iu":
        raise ValueError("robot_joint_ids must be a non-empty 1D integer array")
    joints = joints.astype(np.int64, copy=False)
    if np.any(joints < 0) or np.unique(joints).size != joints.size:
        raise ValueError("robot_joint_ids must contain unique nonnegative IDs")
    transmissions = np.asarray(getattr(model, "actuator_trnid"))
    if transmissions.ndim != 2 or transmissions.shape[1] < 1:
        raise ValueError("loaded model actuator_trnid must have shape (A, >=1)")
    transmission_joints = transmissions[:, 0]
    actuator_ids: list[int] = []
    for joint_id in joints:
        matches = np.flatnonzero(transmission_joints == joint_id)
        if matches.size != 1:
            raise ValueError(f"loaded robot joint {joint_id} must map to exactly one actuator; got {matches.size}")
        actuator_ids.append(int(matches[0]))
    return np.asarray(actuator_ids, dtype=np.int64), joints.copy()


def send_repeated_command(
    payload: bytes,
    *,
    send: Callable[[bytes], None],
    attempts: int = 3,
    interval_s: float = 0.02,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Bound PUB/SUB slow-joiner risk with a fixed number of idempotent sends."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("command payload must be non-empty bytes")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        raise ValueError("command attempts must be a positive integer")
    if not math.isfinite(interval_s) or interval_s < 0.0:
        raise ValueError("command interval_s must be finite and nonnegative")
    for attempt in range(attempts):
        send(payload)
        if attempt + 1 < attempts:
            sleep(interval_s)


def validate_state_ack(
    message: object,
    frame: ReplayActionFrame,
    *,
    last_index: int,
) -> int | None:
    """Validate a fresh same-tick token/hand echo and controller output."""

    if not isinstance(message, Mapping):
        raise RuntimeError("g1_debug acknowledgement must be a mapping")
    required = {
        "control_loop_type",
        "index",
        "token_state",
        "left_hand_q_measured",
        "right_hand_q_measured",
        "last_action",
    }
    if not required.issubset(message):
        raise RuntimeError("g1_debug acknowledgement is missing controller fields")
    index = message["index"]
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or int(index) < 0:
        raise RuntimeError("g1_debug acknowledgement index is invalid")
    arrays: dict[str, np.ndarray] = {}
    for name, shape in (
        ("token_state", (64,)),
        ("left_hand_q_measured", (7,)),
        ("right_hand_q_measured", (7,)),
        ("last_action", (29,)),
    ):
        value = np.asarray(message[name])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise RuntimeError(f"g1_debug acknowledgement {name} is invalid")
        arrays[name] = value
    if message["control_loop_type"] != "cpp" or int(index) <= last_index:
        return None
    if not (
        np.array_equal(arrays["token_state"].astype(np.float32), frame.motion_token)
        and np.array_equal(arrays["left_hand_q_measured"].astype(np.float32), frame.left_hand)
        and np.array_equal(arrays["right_hand_q_measured"].astype(np.float32), frame.right_hand)
    ):
        return None
    return int(index)


def poll_until_acknowledged(
    poll: Callable[[], bool],
    *,
    timeout_s: float,
    poll_period_s: float = 0.0005,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll asynchronous telemetry until the caller-provided deadline."""

    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("acknowledgement timeout_s must be finite and positive")
    if not math.isfinite(poll_period_s) or poll_period_s <= 0.0:
        raise ValueError("acknowledgement poll_period_s must be finite and positive")
    deadline = monotonic() + timeout_s
    while True:
        if poll():
            return True
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            return False
        sleep(min(poll_period_s, remaining))


def prime_controller(
    frame: ReplayActionFrame,
    *,
    process: object,
    sim_dt: float,
    timeout_s: float,
    publish_action: Callable[[ReplayActionFrame], None],
    publish_start: Callable[[], None],
    step: Callable[[], None],
    acknowledged: Callable[[ReplayActionFrame], bool],
    monotonic: Callable[[], float] = time.monotonic,
    pace: Callable[[float], None] = time.sleep,
) -> int:
    """Prime through the 3 s C++ INIT state until a real action echo arrives."""

    if not math.isfinite(sim_dt) or sim_dt <= 0.0:
        raise ValueError("priming sim_dt must be finite and positive")
    if not math.isfinite(timeout_s) or timeout_s <= 3.0:
        raise ValueError("priming timeout_s must be finite and greater than the 3 s INIT")
    steps_per_control = round(CONTROL_PERIOD_S / sim_dt)
    if steps_per_control <= 0 or not math.isclose(
        steps_per_control * sim_dt,
        CONTROL_PERIOD_S,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("priming sim_dt must divide the 20 ms control period")
    minimum_steps = math.ceil(3.0 / sim_dt)
    deadline = monotonic() + timeout_s
    step_count = 0
    while True:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"owned deployment exited before acknowledgement with code {return_code}")
        if step_count % steps_per_control == 0:
            publish_action(frame)
            publish_start()
        step()
        step_count += 1
        pace(sim_dt)
        has_ack = acknowledged(frame)
        if step_count >= minimum_steps and has_ack:
            return step_count
        if monotonic() >= deadline:
            raise TimeoutError(f"controller acknowledgement timed out after {timeout_s:g} seconds")


def close_unitree_bridge_channels(bridge: object | None) -> None:
    """Idempotently close every DDS channel owned by the simulator bridge."""

    if bridge is None:
        return
    first_error: Exception | None = None
    for attribute in UNITREE_BRIDGE_CHANNEL_ATTRIBUTES:
        channel = getattr(bridge, attribute, None)
        if channel is None:
            continue
        setattr(bridge, attribute, None)
        try:
            channel.Close()
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _require_floating_root(env: object) -> None:
    if getattr(env, "use_floating_root_link", None) is not True:
        raise ValueError("replay requires the loaded simulator model to have a floating root")


def execute_owned_replay(
    frames: Sequence[ReplayActionFrame],
    *,
    deployment_command: Sequence[str],
    runtime: ReplayRuntime,
    process_factory: Callable[[Sequence[str]], object] = subprocess.Popen,
    readiness_timeout_s: float = 20.0,
    pace: Callable[[float], None] | None = None,
) -> ReplayReport:
    """Run one episode and clean exactly the runtime and child created for it."""

    if not deployment_command or any(not isinstance(item, str) or not item for item in deployment_command):
        raise ValueError("deployment_command must be a non-empty string sequence")
    if not frames:
        raise ValueError("replay requires at least one action frame")
    process: object | None = None
    control_started = False
    report: ReplayReport | None = None
    primary_error: BaseException | None = None
    try:
        process = process_factory(list(deployment_command))
        attach = getattr(runtime, "attach_process", None)
        if callable(attach):
            attach(process)
        control_started = True
        prime_pace = DeadlinePacer() if pace is None else pace
        runtime.prime(
            frames[0],
            timeout_s=readiness_timeout_s,
            pace=prime_pace,
        )
        report = run_replay_schedule(
            frames,
            collector=runtime.collector,
            publish=runtime.publish,
            step=runtime.step,
            acknowledge=runtime.assert_action_ack,
            pace=DeadlinePacer() if pace is None else pace,
        )
    except BaseException as error:
        primary_error = error

    cleanup_errors: list[str] = []
    if control_started:
        try:
            runtime.stop_control()
        except BaseException as error:
            cleanup_errors.append(f"cleanup stop_control failed: {type(error).__name__}: {error}")
    if process is not None:
        try:
            stop_owned_process(process)
        except BaseException as error:
            cleanup_errors.append(f"cleanup owned process failed: {type(error).__name__}: {error}")
    try:
        runtime.close()
    except BaseException as error:
        cleanup_errors.append(f"cleanup runtime.close failed: {type(error).__name__}: {error}")

    if primary_error is not None:
        if cleanup_errors:
            setattr(primary_error, "cleanup_errors", tuple(cleanup_errors))
        raise primary_error
    if cleanup_errors:
        error = RuntimeError("; ".join(cleanup_errors))
        setattr(error, "cleanup_errors", tuple(cleanup_errors))
        raise error
    if report is None:
        raise RuntimeError("replay completed without a report")
    return report


@dataclass(frozen=True)
class RuntimeDependencies:
    """Late-bound heavy runtime components, injectable for failure testing."""

    mujoco: object
    zmq: object
    simulator_factory: Callable[..., object]
    config_factory: Callable[..., object]
    pack_action: Callable[..., bytes]
    build_command: Callable[..., bytes]
    state_subscriber_factory: Callable[..., object] | None = None
    wbc_config_path: Path | None = None


def _load_runtime_dependencies() -> RuntimeDependencies:
    import mujoco
    import zmq

    from gear_sonic.scripts.run_vla_inference import pack_latent_action_message
    from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
    from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

    return RuntimeDependencies(
        mujoco=mujoco,
        zmq=zmq,
        simulator_factory=BaseSimulator,
        config_factory=SimLoopConfig,
        pack_action=pack_latent_action_message,
        build_command=build_command_message,
        state_subscriber_factory=ZMQStateSubscriber,
        wbc_config_path=(REPOSITORY_ROOT / "gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"),
    )


class MujocoZmqRuntime:
    """Concrete deterministic headless simulator and protocol-v4 transport."""

    def __init__(
        self,
        *,
        network_interface: str,
        zmq_host: str,
        action_port: int,
        state_port: int,
        _dependencies: RuntimeDependencies | None = None,
    ):
        self._simulator: object | None = None
        self._context: object | None = None
        self._publisher: object | None = None
        self._state_subscriber: object | None = None
        self._process: object | None = None
        self._closed = False
        dependencies = _dependencies or _load_runtime_dependencies()
        self._mujoco = dependencies.mujoco
        self._pack = dependencies.pack_action
        self._build_command = dependencies.build_command
        self._host = zmq_host
        self._state_port = state_port
        self._expected_frame: ReplayActionFrame | None = None
        self._expected_ack_received = False
        self._last_state_index = -1
        self._controller_acknowledged = False
        self.runtime_identities: dict[str, dict[str, object]] = {}
        try:
            random.seed(0)
            np.random.seed(0)
            config = dependencies.config_factory(
                interface=network_interface,
                sim_frequency=200,
                enable_onscreen=False,
                enable_offscreen=False,
                enable_image_publish=False,
                with_hands=True,
                verbose=False,
            )
            wbc_config = config.load_wbc_yaml()
            wbc_config["ENV_NAME"] = "default"
            wbc_config["ENABLE_ELASTIC_BAND"] = False
            scene_value = wbc_config.get("ROBOT_SCENE")
            if scene_value is not None:
                scene_path = Path(scene_value)
                if not scene_path.is_absolute():
                    scene_path = REPOSITORY_ROOT / scene_path
                self.runtime_identities["mujoco_scene"] = artifact_identity(scene_path)
            if dependencies.wbc_config_path is not None:
                self.runtime_identities["wbc_config"] = artifact_identity(dependencies.wbc_config_path)
            self._simulator = dependencies.simulator_factory(
                wbc_config,
                env_name="default",
                onscreen=False,
                offscreen=False,
                enable_image_publish=False,
            )

            env = self._simulator.sim_env
            _require_floating_root(env)
            model = env.mj_model
            data = env.mj_data
            # MuJoCo qpos0 supplies the standing pelvis height. The loaded WBC
            # configuration supplies the exact nominal 29-body-joint posture.
            data.qpos[:] = model.qpos0
            body_joint_ids = np.asarray(env.body_joint_index, dtype=np.int64)
            body_qpos_addresses = np.asarray(model.jnt_qposadr, dtype=np.int64)[body_joint_ids]
            nominal = np.asarray(wbc_config["DEFAULT_DOF_ANGLES"], dtype=np.float64)
            if nominal.shape != body_joint_ids.shape:
                raise ValueError("loaded simulator nominal posture does not match its body joints")
            data.qpos[body_qpos_addresses] = nominal
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            data.ctrl[:] = 0.0
            data.time = 0.0
            self._mujoco.mj_forward(model, data)
            self._initial_qpos = np.asarray(data.qpos, dtype=np.float64).copy()

            robot_joint_ids = np.concatenate(
                (
                    np.asarray(env.body_joint_index, dtype=np.int64),
                    np.asarray(env.left_hand_index, dtype=np.int64),
                    np.asarray(env.right_hand_index, dtype=np.int64),
                )
            )
            actuator_ids, joint_ids = select_robot_actuator_joint_ids(model, robot_joint_ids)
            self._actuator_ids = actuator_ids
            self._joint_ids = joint_ids
            self._qpos_addresses = np.asarray(model.jnt_qposadr, dtype=np.int64)[joint_ids]
            self._qvel_addresses = np.asarray(model.jnt_dofadr, dtype=np.int64)[joint_ids]
            self.collector = collector_from_mujoco_model(
                model,
                actuator_ids=actuator_ids,
                joint_ids=joint_ids,
                exclude_before_s=1.0,
                mujoco_module=self._mujoco,
            )

            if dependencies.state_subscriber_factory is not None:
                self._state_subscriber = dependencies.state_subscriber_factory(
                    host=zmq_host,
                    port=state_port,
                    topic="g1_debug",
                )
            self._context = dependencies.zmq.Context()
            self._publisher = self._context.socket(dependencies.zmq.PUB)
            self._publisher.setsockopt(dependencies.zmq.LINGER, 0)
            self._publisher.bind(f"tcp://{zmq_host}:{action_port}")
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def attach_process(self, process: object) -> None:
        self._process = process

    def _send(self, payload: bytes) -> None:
        if self._process is not None:
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(f"owned deployment exited during replay with code {return_code}")
        self._publisher.send(payload)

    def _send_start(self) -> None:
        self._send(self._build_command(start=True, stop=False, planner=False))

    def _poll_ack(self, frame: ReplayActionFrame) -> bool:
        if self._state_subscriber is None:
            raise RuntimeError("g1_debug state subscriber is unavailable")
        bridge = getattr(self._simulator, "unitree_bridge", None)
        if bridge is None or not all(
            getattr(bridge, attribute, False) is True
            for attribute in (
                "low_cmd_received",
                "left_hand_cmd_received",
                "right_hand_cmd_received",
            )
        ):
            return False
        message = self._state_subscriber.get_msg(clear=True)
        if message is None:
            return False
        acknowledged_index = validate_state_ack(
            message,
            frame,
            last_index=self._last_state_index,
        )
        if acknowledged_index is None:
            return False
        self._last_state_index = acknowledged_index
        self._controller_acknowledged = True
        return True

    def _reset_simulator_to_nominal(self) -> None:
        env = self._simulator.sim_env
        data = env.mj_data
        data.qpos[:] = self._initial_qpos
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.ctrl[:] = 0.0
        data.time = 0.0
        self._mujoco.mj_forward(env.mj_model, data)

    def _step_unmeasured(self) -> None:
        env = self._simulator.sim_env
        if not self._controller_acknowledged:
            self._reset_simulator_to_nominal()
        env.sim_step()

    def prime(
        self,
        frame: ReplayActionFrame,
        *,
        timeout_s: float,
        pace: Callable[[float], None],
    ) -> None:
        if self._process is None:
            raise RuntimeError("owned deployment process must be attached before priming")
        prime_controller(
            frame,
            process=self._process,
            sim_dt=self.collector.sim_dt,
            timeout_s=timeout_s,
            publish_action=lambda value: self.publish(value, "prime", 0, 0.0),
            publish_start=self._send_start,
            step=self._step_unmeasured,
            acknowledged=self._poll_ack,
            pace=pace,
        )
        # Priming advances dynamics while the C++ policy transitions out of
        # INIT.  Measured warm-up must nevertheless begin at the exact
        # deterministic nominal state required by the replay contract.
        self._reset_simulator_to_nominal()

    def publish(
        self,
        frame: ReplayActionFrame,
        phase: str,
        source_index: int,
        time_s: float,
    ) -> None:
        del phase, time_s
        self._expected_frame = frame
        self._expected_ack_received = False
        self._send(
            self._pack(
                motion_token=frame.motion_token,
                frame_index=np.asarray([source_index], dtype=np.int64),
                left_hand_joints=frame.left_hand,
                right_hand_joints=frame.right_hand,
            )
        )

    def step(self, time_s: float) -> ReplaySample:
        env = self._simulator.sim_env
        env.sim_step()
        if self._expected_frame is not None and self._poll_ack(self._expected_frame):
            self._expected_ack_received = True
        data = env.mj_data
        roll, pitch = quaternion_wxyz_to_roll_pitch(np.asarray(data.qpos[3:7]))
        warning_fault = any(int(warning.number) > 0 for warning in data.warning)
        process_fault = self._process is not None and self._process.poll() is not None
        return ReplaySample(
            time_s=time_s,
            root_height_m=float(data.qpos[2]),
            root_roll_rad=roll,
            root_pitch_rad=pitch,
            joint_positions=np.asarray(data.qpos[self._qpos_addresses], dtype=np.float64).copy(),
            joint_velocities=np.asarray(data.qvel[self._qvel_addresses], dtype=np.float64).copy(),
            torques=np.asarray(data.actuator_force[self._actuator_ids], dtype=np.float64).copy(),
            contact_force_n=max_mujoco_contact_force(
                env.mj_model,
                data,
                mujoco_module=self._mujoco,
            ),
            controller_fault=warning_fault or process_fault or bool(getattr(env, "fall", False)),
        )

    def assert_action_ack(
        self,
        frame: ReplayActionFrame,
        phase: str,
        source_index: int,
        time_s: float,
    ) -> None:
        del phase, source_index, time_s
        if self._expected_frame is frame and not self._expected_ack_received:
            self._expected_ack_received = poll_until_acknowledged(
                lambda: self._poll_ack(frame),
                timeout_s=CONTROL_PERIOD_S,
            )
        if self._expected_frame is not frame or not self._expected_ack_received:
            raise RuntimeError("continuous g1_debug controller acknowledgement was missed")

    def stop_control(self) -> None:
        if self._process is None or self._process.poll() is None:
            self._publisher.send(self._build_command(start=False, stop=True, planner=False))
            time.sleep(0.01)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        for resource, operation in (
            (self._state_subscriber, lambda value: value.close()),
            (self._publisher, lambda value: value.close(linger=0)),
            (self._context, lambda value: value.term()),
            (
                getattr(self._simulator, "unitree_bridge", None),
                close_unitree_bridge_channels,
            ),
            (self._simulator, lambda value: value.close()),
        ):
            if resource is None:
                continue
            try:
                operation(resource)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def write_json_report(path: str | Path, report: Mapping[str, object]) -> None:
    """Write a canonical JSON report by fsync + atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def json_safe(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                return "NaN"
            return "Infinity" if value > 0.0 else "-Infinity"
        return value

    payload = (
        json.dumps(json_safe(report), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _existing_file(path: str | Path, argument: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise ValueError(f"{argument} is not an existing file: {value}")
    return value


def _existing_directory(path: str | Path, argument: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_dir():
        raise ValueError(f"{argument} is not an existing directory: {value}")
    return value


def _resolve_report_path(
    requested: str | Path | None,
    dataset_root: str | Path,
    default: str | Path,
) -> Path:
    root = Path(dataset_root).resolve(strict=True)
    destination = Path(default if requested is None else requested).expanduser().resolve()
    if destination == root or root in destination.parents:
        raise ProvenanceError("replay report must be written outside the authenticated dataset tree")
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-episode-ids", type=int, nargs="+", required=True)
    parser.add_argument("--deploy-binary", default=str(DEFAULT_DEPLOY_BINARY))
    parser.add_argument("--network-interface", default="lo")
    parser.add_argument("--policy-file", default=str(DEFAULT_POLICY_FILE))
    parser.add_argument("--motion-data-path", default=str(DEFAULT_MOTION_DATA))
    parser.add_argument("--planner-file", default=str(DEFAULT_PLANNER_FILE))
    parser.add_argument("--observation-config", default=str(DEFAULT_OBSERVATION_CONFIG))
    parser.add_argument("--encoder-file", default=str(DEFAULT_ENCODER_FILE))
    parser.add_argument("--zmq-host", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, default=5556)
    parser.add_argument("--state-port", type=int, default=5557)
    parser.add_argument("--readiness-timeout-s", type=float, default=60.0)
    parser.add_argument("--output-report")
    parsed = parser.parse_args(argv)
    if not 1 <= len(parsed.source_episode_ids) <= 5:
        parser.error("smoke replay requires between one and five source episode IDs")
    return parsed


def _run(args: argparse.Namespace) -> int:
    requested_root = _existing_directory(args.dataset_root, "--dataset-root")
    dataset_root, default_output_report = resolve_replay_dataset_root(requested_root)
    output_report = _resolve_report_path(args.output_report, dataset_root, default_output_report)
    authenticated = authenticate_dataset(dataset_root)
    episodes = resolve_source_episode_ids(
        dataset_root,
        args.source_episode_ids,
        authenticated=authenticated,
    )
    config = DeploymentConfig(
        binary=_existing_file(args.deploy_binary, "--deploy-binary"),
        network_interface=args.network_interface,
        policy_file=_existing_file(args.policy_file, "--policy-file"),
        motion_data_path=_existing_directory(args.motion_data_path, "--motion-data-path"),
        planner_file=_existing_file(args.planner_file, "--planner-file"),
        observation_config=_existing_file(args.observation_config, "--observation-config"),
        encoder_file=_existing_file(args.encoder_file, "--encoder-file"),
        zmq_host=args.zmq_host,
        action_port=args.action_port,
        state_port=args.state_port,
    )
    command = build_deployment_command(config)
    execution_provenance = dict(
        build_execution_provenance(
            config,
            authenticated,
            runtime_identities={},
        )
    )
    episode_reports: list[ReplayReport] = []
    serialized: list[dict[str, object]] = []
    runtime_identities: dict[str, dict[str, object]] = {}
    sim_timestep_s: float | None = None
    for episode in episodes:
        runtime: object | None = None
        try:
            frames = load_episode_frames(dataset_root, episode, authenticated=authenticated)
            runtime = MujocoZmqRuntime(
                network_interface=config.network_interface,
                zmq_host=config.zmq_host,
                action_port=config.action_port,
                state_port=config.state_port,
            )
            current_sim_timestep = float(runtime.collector.sim_dt)
            if sim_timestep_s is None:
                sim_timestep_s = current_sim_timestep
            elif current_sim_timestep != sim_timestep_s:
                raise RuntimeError("simulator timestep changed within the replay cohort")
            current_identities = {
                name: dict(identity) for name, identity in sorted(runtime.runtime_identities.items())
            }
            if runtime_identities and current_identities != runtime_identities:
                raise ProvenanceError("MuJoCo scene or WBC configuration changed within the replay cohort")
            runtime_identities = current_identities
            report = execute_owned_replay(
                frames,
                deployment_command=command,
                runtime=runtime,
                readiness_timeout_s=args.readiness_timeout_s,
            )
            episode_reports.append(report)
            serialized.append(
                {
                    "source_episode_id": episode.source_episode_id,
                    "target_episode_index": episode.target_episode_index,
                    "status": "accepted" if report.accepted else "replay_acceptance_error",
                    "report": asdict(report),
                }
            )
        except Exception as error:
            close_runtime = getattr(runtime, "close", None)
            if callable(close_runtime):
                try:
                    close_runtime()
                except Exception as cleanup_error:
                    cleanup_errors = list(getattr(error, "cleanup_errors", ()))
                    cleanup_errors.append(
                        f"cleanup runtime.close failed: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    setattr(error, "cleanup_errors", tuple(cleanup_errors))
            failure: dict[str, object] = {
                "source_episode_id": episode.source_episode_id,
                "target_episode_index": episode.target_episode_index,
                "status": classify_replay_error(error),
                "error": f"{type(error).__name__}: {error}",
            }
            cleanup_errors = getattr(error, "cleanup_errors", ())
            if cleanup_errors:
                failure["cleanup_errors"] = list(cleanup_errors)
            serialized.append(failure)
    cohort = aggregate_replay_reports(episode_reports) if episode_reports else None
    complete = len(episode_reports) == len(episodes)
    accepted = complete and cohort is not None and cohort.accepted
    execution_provenance["runtime"] = runtime_identities
    protocol = (
        replay_protocol(sim_timestep_s)
        if sim_timestep_s is not None
        else {
            "control_dt_s": CONTROL_PERIOD_S,
            "sim_timestep_s": None,
            "exclude_before_s": 1.0,
            "final_hold_s": 1.0,
            "seed": 0,
            "warmup_s": 2.0,
        }
    )
    write_json_report(
        output_report,
        {
            "accepted": accepted,
            "cohort": asdict(cohort) if cohort is not None else None,
            "episode_reports": serialized,
            "execution_provenance": execution_provenance,
            "protocol": protocol,
            "source_episode_ids": [episode.source_episode_id for episode in episodes],
        },
    )
    return 0 if accepted else 1


def _isolated_child_command(
    args: argparse.Namespace,
    source_episode_id: int,
    output_report: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset-root",
        str(args.dataset_root),
        "--source-episode-ids",
        str(source_episode_id),
        "--deploy-binary",
        str(args.deploy_binary),
        "--network-interface",
        str(args.network_interface),
        "--policy-file",
        str(args.policy_file),
        "--motion-data-path",
        str(args.motion_data_path),
        "--planner-file",
        str(args.planner_file),
        "--observation-config",
        str(args.observation_config),
        "--encoder-file",
        str(args.encoder_file),
        "--zmq-host",
        str(args.zmq_host),
        "--action-port",
        str(args.action_port),
        "--state-port",
        str(args.state_port),
        "--readiness-timeout-s",
        str(args.readiness_timeout_s),
        "--output-report",
        str(output_report),
    ]


def _deserialize_replay_report(value: object) -> ReplayReport:
    if not isinstance(value, Mapping):
        raise ProvenanceError("isolated child replay report payload is invalid")
    expected_fields = {field.name for field in fields(ReplayReport)}
    if set(value) != expected_fields:
        raise ProvenanceError("isolated child replay report fields are invalid")
    payload = dict(value)
    for key, replacement in (("NaN", math.nan), ("Infinity", math.inf), ("-Infinity", -math.inf)):
        for field_name, field_value in tuple(payload.items()):
            if field_value == key:
                payload[field_name] = replacement
    gate_failures = payload.get("gate_failures")
    if not isinstance(gate_failures, list) or not all(isinstance(item, str) for item in gate_failures):
        raise ProvenanceError("isolated child replay gate failures are invalid")
    payload["gate_failures"] = tuple(gate_failures)
    try:
        return ReplayReport(**payload)
    except (TypeError, ValueError) as error:
        raise ProvenanceError("isolated child replay report values are invalid") from error


def run_isolated_cohort(
    args: argparse.Namespace,
    *,
    _process_runner: Callable[..., object] = subprocess.run,
) -> int:
    """Replay plural cohorts with one fresh DDS/MuJoCo process per episode."""

    if len(args.source_episode_ids) < 2:
        raise ValueError("isolated cohort replay requires at least two episode IDs")
    requested_root = _existing_directory(args.dataset_root, "--dataset-root")
    dataset_root, default_output_report = resolve_replay_dataset_root(requested_root)
    output_report = _resolve_report_path(args.output_report, dataset_root, default_output_report)
    authenticated = authenticate_dataset(dataset_root)
    episodes = resolve_source_episode_ids(
        dataset_root,
        args.source_episode_ids,
        authenticated=authenticated,
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)

    serialized: list[dict[str, object]] = []
    episode_reports: list[ReplayReport] = []
    execution_provenance: Mapping[str, object] | None = None
    child_protocol: Mapping[str, object] | None = None
    child_acceptance: list[bool] = []
    with tempfile.TemporaryDirectory(
        prefix=".unitree-sonic-replay-",
        dir=output_report.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        for position, episode in enumerate(episodes):
            child_output = temporary_root / f"episode-{position:06d}.json"
            completed = _process_runner(
                _isolated_child_command(args, episode.source_episode_id, child_output),
                check=False,
            )
            return_code = getattr(completed, "returncode", None)
            if return_code not in (0, 1) or not child_output.is_file():
                raise RuntimeError(
                    f"isolated replay child {episode.source_episode_id} failed before producing a report"
                )
            try:
                child = json.loads(child_output.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ProvenanceError("isolated replay child report is unreadable") from error
            if not isinstance(child, Mapping):
                raise ProvenanceError("isolated replay child report must be a mapping")
            if child.get("source_episode_ids") != [episode.source_episode_id]:
                raise ProvenanceError("isolated replay child reported the wrong source episode")
            child_episodes = child.get("episode_reports")
            if not isinstance(child_episodes, list) or len(child_episodes) != 1:
                raise ProvenanceError("isolated replay child must report exactly one episode")
            child_episode = child_episodes[0]
            if not isinstance(child_episode, Mapping):
                raise ProvenanceError("isolated replay child episode report must be a mapping")
            if child_episode.get("source_episode_id") != episode.source_episode_id:
                raise ProvenanceError("isolated replay child episode identity is inconsistent")
            serialized.append(dict(child_episode))
            if "report" in child_episode:
                episode_reports.append(_deserialize_replay_report(child_episode["report"]))

            current_provenance = child.get("execution_provenance")
            current_protocol = child.get("protocol")
            if not isinstance(current_provenance, Mapping) or not isinstance(current_protocol, Mapping):
                raise ProvenanceError("isolated replay child omitted execution provenance or protocol")
            if execution_provenance is None:
                execution_provenance = dict(current_provenance)
                child_protocol = dict(current_protocol)
            elif current_provenance != execution_provenance or current_protocol != child_protocol:
                raise ProvenanceError("isolated replay child provenance changed within the cohort")
            child_acceptance.append(child.get("accepted") is True and return_code == 0)

    cohort = aggregate_replay_reports(episode_reports) if episode_reports else None
    accepted = (
        len(episode_reports) == len(episodes) and all(child_acceptance) and cohort is not None and cohort.accepted
    )
    protocol = dict(child_protocol or {})
    protocol["episode_process_isolation"] = True
    write_json_report(
        output_report,
        {
            "accepted": accepted,
            "cohort": asdict(cohort) if cohort is not None else None,
            "episode_reports": serialized,
            "execution_provenance": dict(execution_provenance or {}),
            "protocol": protocol,
            "source_episode_ids": [episode.source_episode_id for episode in episodes],
        },
    )
    return 0 if accepted else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if len(args.source_episode_ids) > 1:
            return run_isolated_cohort(args)
        return _run(args)
    except (OSError, ValueError) as error:
        print(f"replay setup failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
