from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.data.unitree_conversion.replay import (
    ProvenanceError,
    ReplayActionFrame,
    ReplayMetricsCollector,
    ReplaySample,
    aggregate_replay_reports,
    authenticate_dataset,
    build_replay_schedule,
    collector_from_mujoco_model,
    guarded_authenticated_read,
    max_mujoco_contact_force,
    resolve_source_episode_ids,
    run_replay_schedule,
    stop_owned_process,
    wait_for_readiness,
)
from gear_sonic.scripts import replay_unitree_sonic_smoke as replay_cli
from gear_sonic.scripts.replay_unitree_sonic_smoke import (
    DeadlinePacer,
    DeploymentConfig,
    MujocoZmqRuntime,
    RuntimeDependencies,
    TargetValidationError,
    TimelineError,
    _parse_args,
    artifact_identity,
    build_deployment_command,
    build_execution_provenance,
    classify_replay_error,
    close_unitree_bridge_channels,
    execute_owned_replay,
    frames_from_columns,
    poll_until_acknowledged,
    prime_controller,
    quaternion_wxyz_to_roll_pitch,
    replay_protocol,
    resolve_replay_dataset_root,
    select_robot_actuator_joint_ids,
    send_repeated_command,
    validate_state_ack,
    verify_pinned_encoder_artifacts,
    write_json_report,
)


def sample(
    *,
    time_s: float = 1.0,
    root_height: float = 0.75,
    root_roll: float = 0.0,
    root_pitch: float = 0.0,
    positions: tuple[float, float] = (0.0, 0.0),
    velocities: tuple[float, float] = (0.0, 0.0),
    torques: tuple[float, float] = (0.0, 0.0),
    contact_force: float = 0.0,
    controller_fault: bool = False,
) -> ReplaySample:
    return ReplaySample(
        time_s=time_s,
        root_height_m=root_height,
        root_roll_rad=root_roll,
        root_pitch_rad=root_pitch,
        joint_positions=np.array(positions, dtype=np.float64),
        joint_velocities=np.array(velocities, dtype=np.float64),
        torques=np.array(torques, dtype=np.float64),
        contact_force_n=contact_force,
        controller_fault=controller_fault,
    )


def collector(*, mass: float = 1.0) -> ReplayMetricsCollector:
    return ReplayMetricsCollector(
        sim_dt=0.005,
        exclude_before_s=1.0,
        robot_mass_kg=mass,
        effort_limits=np.array([10.0, 20.0]),
        joint_limits=np.array([[-1.0, 1.0], [-2.0, 2.0]]),
        effort_limit_source="loaded_model.actuator_forcerange",
    )


def test_metrics_exclude_only_first_second_of_two_second_warmup() -> None:
    metrics = collector(mass=35.0)
    metrics.add(sample(time_s=0.5, torques=(100.0, 100.0), contact_force=99_999.0))
    metrics.add(sample(time_s=1.0, torques=(5.0, 10.0), contact_force=100.0))

    report = metrics.finalize()

    assert report.included_sample_count == 1
    assert report.torque_ratio_max == 0.5
    assert report.contact_force_max == 100.0


def test_metrics_record_linear_p50_p95_p99_and_max() -> None:
    metrics = collector(mass=100.0)
    for index, value in enumerate((0.0, 1.0, 2.0, 3.0, 4.0)):
        metrics.add(
            sample(
                time_s=1.0 + index * 0.005,
                root_roll=value / 10.0,
                torques=(value * 10.0, 0.0),
            )
        )

    report = metrics.finalize()

    assert report.root_roll_abs_p50 == pytest.approx(0.2)
    assert report.root_roll_abs_p95 == pytest.approx(0.38)
    assert report.root_roll_abs_p99 == pytest.approx(0.396)
    assert report.root_roll_abs_max == pytest.approx(0.4)
    assert report.torque_ratio_p50 == pytest.approx(2.0)
    assert report.torque_ratio_p95 == pytest.approx(3.8)
    assert report.torque_ratio_p99 == pytest.approx(3.96)
    assert report.torque_ratio_max == pytest.approx(4.0)


def test_all_acceptance_thresholds_are_inclusive() -> None:
    metrics = collector(mass=1.0)
    metrics.add(
        sample(
            time_s=1.0,
            root_height=0.45,
            root_roll=0.7,
            root_pitch=-0.7,
            positions=(-1.0, 2.0),
            velocities=(-50.0, 50.0),
            torques=(-10.5, 21.0),
            contact_force=98.1,
        )
    )
    metrics.add(sample(time_s=1.005, root_height=1.05))

    report = metrics.finalize()

    assert report.accepted is True
    assert report.gate_failures == ()
    assert report.root_height_min == 0.45
    assert report.root_height_max == 1.05


@pytest.mark.parametrize(
    ("overrides", "failure"),
    [
        ({"root_height": 0.449}, "root_height"),
        ({"root_height": 1.051}, "root_height"),
        ({"root_roll": 0.701}, "root_roll"),
        ({"root_pitch": -0.701}, "root_pitch"),
        ({"positions": (-1.021, 0.0)}, "joint_limit_overshoot"),
        ({"velocities": (50.001, 0.0)}, "joint_velocity"),
        ({"torques": (10.501, 0.0)}, "torque_ratio"),
        ({"contact_force": 98.101}, "contact_force"),
        ({"controller_fault": True}, "controller_fault"),
        ({"root_height": math.nan}, "nonfinite"),
        ({"contact_force": -0.001}, "contact_force"),
    ],
)
def test_each_acceptance_gate_fails_immediately_beyond_boundary(
    overrides: dict[str, object],
    failure: str,
) -> None:
    metrics = collector(mass=1.0)
    metrics.add(sample(**overrides))

    report = metrics.finalize()

    assert report.accepted is False
    assert failure in report.gate_failures


@pytest.mark.parametrize(
    ("overrides", "failure"),
    [
        ({"root_height": math.nextafter(0.45, -math.inf)}, "root_height"),
        ({"root_height": math.nextafter(1.05, math.inf)}, "root_height"),
        ({"root_roll": math.nextafter(0.7, math.inf)}, "root_roll"),
        ({"root_pitch": -math.nextafter(0.7, math.inf)}, "root_pitch"),
        ({"velocities": (math.nextafter(50.0, math.inf), 0.0)}, "joint_velocity"),
        ({"torques": (math.nextafter(10.5, math.inf), 0.0)}, "torque_ratio"),
        ({"contact_force": math.nextafter(10.0 * 9.81, math.inf)}, "contact_force"),
    ],
)
def test_gate_comparisons_reject_one_ulp_beyond_limit(
    overrides: dict[str, object],
    failure: str,
) -> None:
    metrics = collector(mass=1.0)
    metrics.add(sample(**overrides))

    assert failure in metrics.finalize().gate_failures


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sim_dt", 0.0, "sim_dt"),
        ("sim_dt", math.inf, "sim_dt"),
        ("exclude_before_s", -0.1, "exclude_before_s"),
        ("robot_mass_kg", 0.0, "robot_mass_kg"),
        ("effort_limits", np.array([10.0, 0.0]), "effort_limits"),
        ("effort_limits", np.array([[10.0, 20.0]]), "effort_limits"),
        ("joint_limits", np.array([[-1.0, 1.0]]), "joint_limits"),
        ("joint_limits", np.array([[1.0, -1.0], [-2.0, 2.0]]), "joint_limits"),
    ],
)
def test_collector_rejects_invalid_physics_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "sim_dt": 0.005,
        "exclude_before_s": 1.0,
        "robot_mass_kg": 35.0,
        "effort_limits": np.array([10.0, 20.0]),
        "joint_limits": np.array([[-1.0, 1.0], [-2.0, 2.0]]),
        "effort_limit_source": "model",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        ReplayMetricsCollector(**kwargs)


def test_collector_rejects_wrong_sample_shapes_and_time_order() -> None:
    metrics = collector()
    bad = sample()
    bad.joint_positions = np.zeros(3)
    with pytest.raises(ValueError, match="joint_positions"):
        metrics.add(bad)

    metrics.add(sample(time_s=1.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        metrics.add(sample(time_s=1.0))


def test_cohort_uses_extrema_not_mean() -> None:
    reports = []
    for torque, height in ((0.4, 0.8), (1.06, 0.7), (0.3, 0.6)):
        metrics = collector(mass=100.0)
        metrics.add(sample(root_height=height, torques=(10.0 * torque, 0.0)))
        reports.append(metrics.finalize())

    cohort = aggregate_replay_reports(reports)

    assert cohort.torque_ratio_max == 1.06
    assert cohort.root_height_min == 0.6
    assert cohort.accepted is False


@pytest.mark.parametrize("field", ["sim_dt", "effort_limit_source", "robot_mass_kg"])
def test_cohort_rejects_mixed_physics_contracts(field: str) -> None:
    first = collector(mass=100.0)
    first.add(sample())
    second = collector(mass=100.0)
    second.add(sample())
    first_report = first.finalize()
    second_values = dict(second.finalize().__dict__)
    second_values[field] = {
        "sim_dt": 0.01,
        "effort_limit_source": "different.model",
        "robot_mass_kg": 101.0,
    }[field]

    with pytest.raises(ValueError, match="same replay physics contract"):
        aggregate_replay_reports([first_report, type(first_report)(**second_values)])


def test_mujoco_contact_force_uses_l2_of_all_six_components() -> None:
    forces = (np.array([3.0, 4.0, 0.0, 0.0, 0.0, 12.0]), np.ones(6))

    class FakeMujoco:
        @staticmethod
        def mj_contactForce(model: object, data: object, index: int, output: np.ndarray) -> None:
            del model, data
            output[:] = forces[index]

    data = SimpleNamespace(ncon=2)

    assert max_mujoco_contact_force(object(), data, mujoco_module=FakeMujoco) == 13.0


def test_collector_loads_mass_effort_and_joint_limits_from_mujoco_model() -> None:
    model = SimpleNamespace(
        opt=SimpleNamespace(timestep=0.005),
        actuator_forcerange=np.zeros((2, 2)),
        jnt_range=np.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]]),
        jnt_actfrcrange=np.array([[-1.0, 1.0], [-10.0, 8.0], [-19.0, 20.0]]),
        jnt_actfrclimited=np.array([0, 1, 1]),
    )

    class FakeMujoco:
        @staticmethod
        def mj_getTotalmass(value: object) -> float:
            assert value is model
            return 35.0

    metrics = collector_from_mujoco_model(
        model,
        actuator_ids=np.array([0, 1]),
        joint_ids=np.array([1, 2]),
        mujoco_module=FakeMujoco,
    )

    assert metrics.robot_mass_kg == 35.0
    np.testing.assert_array_equal(metrics.effort_limits, [10.0, 20.0])
    np.testing.assert_array_equal(metrics.joint_limits, [[-2.0, 2.0], [-3.0, 3.0]])
    assert metrics.effort_limit_source == "mujoco.jnt_actfrcrange"


def test_collector_rejects_robot_joint_without_actuator_force_limit() -> None:
    model = SimpleNamespace(
        opt=SimpleNamespace(timestep=0.005),
        actuator_forcerange=np.zeros((2, 2)),
        jnt_range=np.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]]),
        jnt_actfrcrange=np.array([[-1.0, 1.0], [-10.0, 8.0], [-19.0, 20.0]]),
        jnt_actfrclimited=np.array([0, 1, 0]),
    )

    with pytest.raises(ValueError, match="jnt_actfrclimited"):
        collector_from_mujoco_model(
            model,
            actuator_ids=np.array([0, 1]),
            joint_ids=np.array([1, 2]),
            mujoco_module=SimpleNamespace(mj_getTotalmass=lambda value: 35.0),
        )


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "source-manifest.json").write_bytes(payload)
    artifacts = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "dataset-checksums.sha256"
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n"
        for path in artifacts
    ]
    (root / "dataset-checksums.sha256").write_text("".join(lines), encoding="utf-8")


def test_source_episode_resolution_never_assumes_target_local_index(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        {
            "source_episode_ids": [10, 20],
            "episode_lengths": [3, 4],
            "stages": [{"source_episode_id": 10}, {"source_episode_id": 20}],
        },
    )

    resolved = resolve_source_episode_ids(tmp_path, [20, 10])

    assert [(item.source_episode_id, item.target_episode_index, item.episode_length) for item in resolved] == [
        (20, 1, 4),
        (10, 0, 3),
    ]


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "length", "stage", "missing", "checksum", "negative_manifest", "negative_request"],
)
def test_source_episode_resolution_fails_closed_on_bad_provenance(
    mutation: str,
    tmp_path: Path,
) -> None:
    manifest = {
        "source_episode_ids": [10, 20],
        "episode_lengths": [3, 4],
        "stages": [{"source_episode_id": 10}, {"source_episode_id": 20}],
    }
    requested = [10]
    if mutation == "duplicate":
        manifest["source_episode_ids"] = [10, 10]
    elif mutation == "length":
        manifest["episode_lengths"] = [3]
    elif mutation == "stage":
        manifest["stages"] = [{"source_episode_id": 10}, {"source_episode_id": 99}]
    elif mutation == "missing":
        requested = [99]
    elif mutation == "negative_manifest":
        manifest["source_episode_ids"] = [-1, 20]
        manifest["stages"] = [{"source_episode_id": -1}, {"source_episode_id": 20}]
    elif mutation == "negative_request":
        requested = [-1]
    _write_manifest(tmp_path, manifest)
    if mutation == "checksum":
        (tmp_path / "source-manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="provenance|source episode"):
        resolve_source_episode_ids(tmp_path, requested)


def test_dataset_authentication_covers_every_artifact_and_detects_mutation(tmp_path: Path) -> None:
    data = tmp_path / "data/chunk-000"
    data.mkdir(parents=True)
    parquet = data / "episode_000000.parquet"
    parquet.write_bytes(b"immutable parquet")
    _write_manifest(
        tmp_path,
        {
            "source_episode_ids": [10],
            "episode_lengths": [3],
            "stages": [{"source_episode_id": 10}],
        },
    )

    authenticated = authenticate_dataset(tmp_path)
    expected = authenticated.artifact_sha256["data/chunk-000/episode_000000.parquet"]
    assert guarded_authenticated_read(parquet, expected, lambda path: path.read_bytes()) == b"immutable parquet"

    with pytest.raises(ValueError, match="provenance"):
        guarded_authenticated_read(
            parquet,
            expected,
            lambda path: path.write_bytes(b"mutated") or b"unreachable",
        )

    (tmp_path / "unlisted.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        authenticate_dataset(tmp_path)


def action_frame(value: float) -> ReplayActionFrame:
    return ReplayActionFrame(
        motion_token=np.full(64, value, dtype=np.float32),
        left_hand=np.full(7, value, dtype=np.float32),
        right_hand=np.full(7, value, dtype=np.float32),
    )


def test_schedule_has_exact_warmup_trajectory_hold_counts_and_timing() -> None:
    schedule = build_replay_schedule([action_frame(0.0), action_frame(1.0), action_frame(2.0)])

    assert len(schedule) == 100 + 3 + 50
    assert [item.phase for item in schedule[:100]] == ["warmup"] * 100
    assert [item.source_frame_index for item in schedule[:100]] == [0] * 100
    assert [item.time_s for item in schedule[:100]] == [index / 50.0 for index in range(100)]
    assert [item.source_frame_index for item in schedule[100:103]] == [0, 1, 2]
    assert [item.time_s for item in schedule[100:103]] == [2.0, 2.02, 2.04]
    assert [item.phase for item in schedule[-50:]] == ["hold"] * 50
    assert [item.source_frame_index for item in schedule[-50:]] == [2] * 50
    assert schedule[-1].time_s == 3.04


def test_schedule_execution_samples_every_sim_step() -> None:
    frames = [action_frame(0.0), action_frame(1.0)]
    metrics = collector(mass=100.0)
    published: list[tuple[str, int, float]] = []
    step_times: list[float] = []

    def publish(frame: ReplayActionFrame, phase: str, source_index: int, time_s: float) -> None:
        del frame
        published.append((phase, source_index, time_s))

    def step(time_s: float) -> ReplaySample:
        step_times.append(time_s)
        return sample(time_s=time_s)

    report = run_replay_schedule(
        frames,
        collector=metrics,
        publish=publish,
        step=step,
        pace=lambda delay: None,
    )

    assert len(published) == 100 + 2 + 50
    assert len(step_times) == (100 + 2 + 50) * 4
    assert step_times[0] == 0.005
    assert step_times[-1] == pytest.approx(3.04)
    assert report.duration_s == pytest.approx(3.04)


def test_schedule_requires_one_acknowledgement_per_50hz_command() -> None:
    metrics = collector(mass=100.0)
    acknowledgements: list[tuple[str, int]] = []

    run_replay_schedule(
        [action_frame(0.0)],
        collector=metrics,
        publish=lambda *args: None,
        step=lambda time_s: sample(time_s=time_s),
        acknowledge=lambda frame, phase, source_index, time_s: acknowledgements.append((phase, source_index)),
        pace=lambda delay: None,
    )

    assert len(acknowledgements) == 151
    assert acknowledgements[0] == ("warmup", 0)
    assert acknowledgements[-1] == ("hold", 0)


class FakeProcess:
    def __init__(self, *, exit_code: int | None = None, timeout_count: int = 0) -> None:
        self.exit_code = exit_code
        self.timeout_count = timeout_count
        self.events: list[object] = []

    def poll(self) -> int | None:
        return self.exit_code

    def send_signal(self, value: int) -> None:
        self.events.append(("signal", value))

    def wait(self, timeout: float) -> int:
        self.events.append(("wait", timeout))
        if self.timeout_count:
            self.timeout_count -= 1
            raise subprocess.TimeoutExpired("owned", timeout)
        self.exit_code = 0
        return 0

    def terminate(self) -> None:
        self.events.append("terminate")

    def kill(self) -> None:
        self.events.append("kill")
        self.exit_code = -9


def test_owned_process_cleanup_is_sigint_then_five_second_wait() -> None:
    process = FakeProcess()

    stop_owned_process(process)

    assert process.events == [("signal", signal.SIGINT), ("wait", 5.0)]


def test_owned_process_cleanup_terminates_only_after_timeout() -> None:
    process = FakeProcess(timeout_count=1)

    stop_owned_process(process)

    assert process.events == [
        ("signal", signal.SIGINT),
        ("wait", 5.0),
        "terminate",
        ("wait", 5.0),
    ]


def test_owned_process_cleanup_kills_only_after_terminate_timeout() -> None:
    process = FakeProcess(timeout_count=2)

    stop_owned_process(process)

    assert process.events == [
        ("signal", signal.SIGINT),
        ("wait", 5.0),
        "terminate",
        ("wait", 5.0),
        "kill",
        ("wait", 5.0),
    ]


def test_readiness_is_bounded_and_detects_early_process_failure() -> None:
    process = FakeProcess()
    clock = [0.0]

    def monotonic() -> float:
        return clock[0]

    def sleep(delay: float) -> None:
        clock[0] += delay

    with pytest.raises(TimeoutError, match="readiness"):
        wait_for_readiness(
            process,
            probe=lambda: False,
            timeout_s=0.2,
            monotonic=monotonic,
            sleep=sleep,
        )

    process.exit_code = 7
    with pytest.raises(RuntimeError, match="code 7"):
        wait_for_readiness(
            process,
            probe=lambda: False,
            timeout_s=1.0,
            monotonic=monotonic,
            sleep=sleep,
        )


def test_deployment_command_forces_deployed_zmq_manager_path(tmp_path: Path) -> None:
    config = DeploymentConfig(
        binary=tmp_path / "g1_deploy_onnx_ref",
        network_interface="lo",
        policy_file=tmp_path / "policy.onnx",
        motion_data_path=tmp_path / "motions",
        planner_file=tmp_path / "planner.onnx",
        observation_config=tmp_path / "observation_config.yaml",
        encoder_file=tmp_path / "encoder.onnx",
        zmq_host="127.0.0.1",
        action_port=6001,
        state_port=6002,
    )

    command = build_deployment_command(config)

    assert command[:4] == [
        str(config.binary),
        "lo",
        str(config.policy_file),
        str(config.motion_data_path),
    ]
    assert command[command.index("--input-type") + 1] == "zmq_manager"
    assert command[command.index("--output-type") + 1] == "zmq"
    assert command[command.index("--zmq-port") + 1] == "6001"
    assert command[command.index("--zmq-out-port") + 1] == "6002"


def test_task14_plural_cli_uses_repo_defaults_and_dataset_report() -> None:
    args = _parse_args(
        [
            "--dataset-root",
            "/converted",
            "--source-episode-ids",
            "0",
            "78",
            "155",
            "233",
            "310",
        ]
    )

    assert args.source_episode_ids == [0, 78, 155, 233, 310]
    assert args.deploy_binary.endswith("gear_sonic_deploy/target/release/g1_deploy_onnx_ref")
    assert args.policy_file.endswith("gear_sonic_deploy/policy/low_latency/model_decoder.onnx")
    assert args.observation_config.endswith("gear_sonic_deploy/policy/low_latency/observation_config.yaml")
    assert args.encoder_file.endswith("gear_sonic_deploy/policy/low_latency/model_encoder.onnx")
    assert args.readiness_timeout_s == 60.0
    assert args.output_report is None


def test_output_root_resolves_unique_dex3_child_and_report_stays_outside_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "unitreerobotics--G1_Dex3_Pouring_Dataset"
    dataset.mkdir()
    _write_manifest(
        dataset,
        {
            "source_episode_ids": [0],
            "episode_lengths": [2],
            "stages": [{"source_episode_id": 0}],
        },
    )

    resolved, outer_report = resolve_replay_dataset_root(tmp_path)
    direct, sibling_report = resolve_replay_dataset_root(dataset)

    assert resolved == dataset.resolve()
    assert direct == dataset.resolve()
    assert outer_report == tmp_path.resolve() / "unitree-sonic-replay-report.json"
    assert sibling_report == dataset.resolve().parent / f"{dataset.name}-replay-report.json"
    assert dataset.resolve() not in outer_report.parents
    assert dataset.resolve() not in sibling_report.parents


def test_output_root_fails_closed_when_dex3_child_is_ambiguous(tmp_path: Path) -> None:
    for parent in (tmp_path / "one", tmp_path / "two"):
        dataset = parent / "unitreerobotics--G1_Dex3_Pouring_Dataset"
        dataset.mkdir(parents=True)
        _write_manifest(
            dataset,
            {
                "source_episode_ids": [0],
                "episode_lengths": [2],
                "stages": [{"source_episode_id": 0}],
            },
        )

    with pytest.raises(ProvenanceError, match="unique"):
        resolve_replay_dataset_root(tmp_path)


def test_artifact_and_execution_provenance_report_exact_bytes_and_command(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_manifest(
        dataset,
        {
            "conversion_identity": {
                "encoder_sha256": hashlib.sha256(b"encoder").hexdigest(),
                "encoder_config_sha256": hashlib.sha256(b"config").hexdigest(),
            },
            "source_episode_ids": [0],
            "episode_lengths": [2],
            "stages": [{"source_episode_id": 0}],
        },
    )
    authenticated = authenticate_dataset(dataset)
    files = {}
    for name in ("binary", "decoder", "planner", "encoder", "config", "scene", "wbc"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    motions = tmp_path / "motions"
    motions.mkdir()
    config = DeploymentConfig(
        binary=files["binary"],
        network_interface="lo",
        policy_file=files["decoder"],
        motion_data_path=motions,
        planner_file=files["planner"],
        observation_config=files["config"],
        encoder_file=files["encoder"],
    )

    provenance = build_execution_provenance(
        config,
        authenticated,
        runtime_identities={
            "mujoco_scene": artifact_identity(files["scene"]),
            "wbc_config": artifact_identity(files["wbc"]),
        },
        _pinned_expected={
            "encoder": (len(b"encoder"), hashlib.sha256(b"encoder").hexdigest()),
            "observation_config": (len(b"config"), hashlib.sha256(b"config").hexdigest()),
        },
    )

    assert provenance["command"] == build_deployment_command(config)
    assert provenance["artifacts"]["binary"]["sha256"] == hashlib.sha256(b"binary").hexdigest()
    assert provenance["artifacts"]["decoder"]["size"] == len(b"decoder")
    assert provenance["dataset"]["source_manifest_sha256"] == authenticated.source_manifest_sha256
    assert provenance["dataset"]["dataset_checksums_sha256"] == authenticated.dataset_checksums_sha256
    assert provenance["dataset"]["conversion_identity"] == {
        "encoder_sha256": hashlib.sha256(b"encoder").hexdigest(),
        "encoder_config_sha256": hashlib.sha256(b"config").hexdigest(),
    }
    assert provenance["runtime"] == {
        "mujoco_scene": artifact_identity(files["scene"]),
        "wbc_config": artifact_identity(files["wbc"]),
    }


def test_execution_provenance_rejects_dataset_encoder_identity_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_manifest(
        dataset,
        {
            "conversion_identity": {
                "encoder_sha256": "0" * 64,
                "encoder_config_sha256": hashlib.sha256(b"config").hexdigest(),
            },
            "source_episode_ids": [0],
            "episode_lengths": [2],
            "stages": [{"source_episode_id": 0}],
        },
    )
    files = {}
    for name in ("binary", "decoder", "planner", "encoder", "config"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    motions = tmp_path / "motions"
    motions.mkdir()
    config = DeploymentConfig(
        binary=files["binary"],
        network_interface="lo",
        policy_file=files["decoder"],
        motion_data_path=motions,
        planner_file=files["planner"],
        observation_config=files["config"],
        encoder_file=files["encoder"],
    )

    with pytest.raises(ProvenanceError, match="conversion identity.*encoder_sha256"):
        build_execution_provenance(
            config,
            authenticate_dataset(dataset),
            runtime_identities={},
            _pinned_expected={
                "encoder": (len(b"encoder"), hashlib.sha256(b"encoder").hexdigest()),
                "observation_config": (
                    len(b"config"),
                    hashlib.sha256(b"config").hexdigest(),
                ),
            },
        )


def test_encoder_and_config_must_match_pinned_size_and_sha(tmp_path: Path) -> None:
    encoder = tmp_path / "encoder.onnx"
    config = tmp_path / "observation_config.yaml"
    encoder.write_bytes(b"encoder")
    config.write_bytes(b"config")
    expected = {
        "encoder": (len(b"encoder"), hashlib.sha256(b"encoder").hexdigest()),
        "observation_config": (len(b"config"), hashlib.sha256(b"config").hexdigest()),
    }

    verified = verify_pinned_encoder_artifacts(encoder, config, _expected=expected)
    assert verified["encoder"] == artifact_identity(encoder)

    encoder.write_bytes(b"tampered")
    with pytest.raises(ProvenanceError, match="encoder"):
        verify_pinned_encoder_artifacts(encoder, config, _expected=expected)


def test_error_classification_distinguishes_all_replay_failure_stages() -> None:
    assert classify_replay_error(ProvenanceError("bad tree")) == "provenance_error"
    assert classify_replay_error(TimelineError("bad timestamp")) == "timeline_error"
    assert classify_replay_error(TargetValidationError("bad token")) == "target_validation_error"
    assert classify_replay_error(RuntimeError("inactive controller")) == "replay_acceptance_error"


def test_protocol_report_contains_control_and_simulator_timesteps() -> None:
    protocol = replay_protocol(0.005)

    assert protocol["control_dt_s"] == 0.02
    assert protocol["sim_timestep_s"] == 0.005


def test_run_reports_resolved_dataset_provenance_protocol_and_failure_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "unitreerobotics--G1_Dex3_Pouring_Dataset"
    dataset.mkdir()
    _write_manifest(
        dataset,
        {
            "source_episode_ids": [10, 11],
            "episode_lengths": [2, 2],
            "stages": [{"source_episode_id": 10}, {"source_episode_id": 11}],
        },
    )
    files: dict[str, Path] = {}
    for name in ("binary", "decoder", "planner", "encoder", "config"):
        files[name] = tmp_path / name
        files[name].write_bytes(name.encode())
    motions = tmp_path / "motions"
    motions.mkdir()
    args = SimpleNamespace(
        dataset_root=str(tmp_path),
        source_episode_ids=[10, 11],
        deploy_binary=str(files["binary"]),
        network_interface="lo",
        policy_file=str(files["decoder"]),
        motion_data_path=str(motions),
        planner_file=str(files["planner"]),
        observation_config=str(files["config"]),
        encoder_file=str(files["encoder"]),
        zmq_host="127.0.0.1",
        action_port=5556,
        state_port=5557,
        readiness_timeout_s=3.1,
        output_report=None,
    )
    authenticated = authenticate_dataset(dataset)
    episodes = (
        replay_cli.SourceEpisodeRef(10, 0, 2, {"source_episode_id": 10}),
        replay_cli.SourceEpisodeRef(11, 1, 2, {"source_episode_id": 11}),
    )
    accepted_collector = collector(mass=100.0)
    accepted_collector.add(sample(time_s=1.0))
    accepted_report = accepted_collector.finalize()
    loaded: list[tuple[Path, int, object]] = []

    authentication_calls: list[Path] = []

    def authenticate(root: Path) -> object:
        authentication_calls.append(Path(root))
        return authenticated

    def resolve(root: Path, ids: object, *, authenticated: object) -> object:
        assert Path(root) == dataset.resolve()
        assert ids == [10, 11]
        assert authenticated is not None
        return episodes

    monkeypatch.setattr(replay_cli, "authenticate_dataset", authenticate)
    monkeypatch.setattr(replay_cli, "resolve_source_episode_ids", resolve)

    def load(root: Path, episode: object, *, authenticated: object) -> tuple[ReplayActionFrame, ...]:
        loaded.append((root, episode.source_episode_id, authenticated))
        if episode.source_episode_id == 11:
            raise TimelineError("bad timeline")
        return (action_frame(0.0),)

    class Runtime:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.collector = SimpleNamespace(sim_dt=0.005)
            self.runtime_identities = {"mujoco_scene": {"sha256": "scene"}}

    monkeypatch.setattr(replay_cli, "load_episode_frames", load)
    monkeypatch.setattr(replay_cli, "MujocoZmqRuntime", Runtime)
    monkeypatch.setattr(replay_cli, "execute_owned_replay", lambda *args, **kwargs: accepted_report)
    monkeypatch.setattr(
        replay_cli,
        "build_execution_provenance",
        lambda config, dataset, runtime_identities: {
            "command": build_deployment_command(config),
            "dataset_root": str(dataset.root),
            "runtime": dict(runtime_identities),
        },
    )

    assert replay_cli._run(args) == 1

    output = tmp_path / "unitree-sonic-replay-report.json"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["execution_provenance"]["dataset_root"] == str(dataset.resolve())
    assert report["execution_provenance"]["runtime"] == {"mujoco_scene": {"sha256": "scene"}}
    assert report["protocol"]["control_dt_s"] == 0.02
    assert report["protocol"]["sim_timestep_s"] == 0.005
    assert [item["status"] for item in report["episode_reports"]] == [
        "accepted",
        "timeline_error",
    ]
    assert loaded == [
        (dataset.resolve(), 10, authenticated),
        (dataset.resolve(), 11, authenticated),
    ]
    assert authentication_calls == [dataset.resolve()]
    assert not (dataset / "unitree-sonic-replay-report.json").exists()


def test_plural_replay_isolates_each_episode_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "unitreerobotics--G1_Dex3_Pouring_Dataset"
    dataset.mkdir()
    _write_manifest(
        dataset,
        {
            "source_episode_ids": [10, 11],
            "episode_lengths": [2, 2],
            "stages": [{"source_episode_id": 10}, {"source_episode_id": 11}],
        },
    )
    accepted_collector = collector(mass=100.0)
    accepted_collector.add(sample(time_s=1.0))
    accepted_report = accepted_collector.finalize()
    args = SimpleNamespace(
        dataset_root=str(tmp_path),
        source_episode_ids=[10, 11],
        deploy_binary="deploy",
        network_interface="lo",
        policy_file="decoder",
        motion_data_path="motions",
        planner_file="planner",
        observation_config="config",
        encoder_file="encoder",
        zmq_host="127.0.0.1",
        action_port=5556,
        state_port=5557,
        readiness_timeout_s=3.1,
        output_report=None,
    )
    commands: list[list[str]] = []

    def run_child(command: list[str], *, check: bool) -> object:
        assert check is False
        commands.append(command)
        source_id = int(command[command.index("--source-episode-ids") + 1])
        output = Path(command[command.index("--output-report") + 1])
        write_json_report(
            output,
            {
                "accepted": True,
                "cohort": asdict(aggregate_replay_reports([accepted_report])),
                "episode_reports": [
                    {
                        "source_episode_id": source_id,
                        "target_episode_index": source_id - 10,
                        "status": "accepted",
                        "report": asdict(accepted_report),
                    }
                ],
                "execution_provenance": {"runtime": {"scene": "same"}},
                "protocol": replay_protocol(0.005),
                "source_episode_ids": [source_id],
            },
        )
        return SimpleNamespace(returncode=0)

    assert replay_cli.run_isolated_cohort(args, _process_runner=run_child) == 0

    assert [command[command.index("--source-episode-ids") + 1] for command in commands] == ["10", "11"]
    assert len({command[command.index("--output-report") + 1] for command in commands}) == 2
    merged = json.loads((tmp_path / "unitree-sonic-replay-report.json").read_text())
    assert merged["accepted"] is True
    assert merged["source_episode_ids"] == [10, 11]
    assert [item["source_episode_id"] for item in merged["episode_reports"]] == [10, 11]
    assert merged["cohort"]["episode_count"] == 2
    assert merged["protocol"]["episode_process_isolation"] is True


def test_run_rejects_report_path_inside_authenticated_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "unitreerobotics--G1_Dex3_Pouring_Dataset"
    dataset.mkdir()
    _write_manifest(
        dataset,
        {
            "source_episode_ids": [0],
            "episode_lengths": [2],
            "stages": [{"source_episode_id": 0}],
        },
    )
    monkeypatch.setattr(replay_cli, "resolve_source_episode_ids", lambda root, ids: ())
    args = SimpleNamespace(
        dataset_root=str(dataset),
        source_episode_ids=[0],
        deploy_binary=str(tmp_path / "unused"),
        output_report=str(dataset / "report.json"),
    )

    with pytest.raises(ProvenanceError, match="outside"):
        replay_cli._resolve_report_path(args.output_report, dataset, tmp_path / "default.json")


def test_exact_filename_cli_bootstraps_repo_when_pythonpath_has_only_external_packages() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    external_paths = [
        path for path in sys.path if "site-packages" in path and repository_root not in Path(path).parents
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(external_paths)

    result = subprocess.run(
        [
            sys.executable,
            "gear_sonic/scripts/replay_unitree_sonic_smoke.py",
            "--help",
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--source-episode-ids" in result.stdout


def test_frames_are_loaded_from_target_index_with_exact_local_timeline() -> None:
    columns = {
        "episode_index": [4, 4],
        "frame_index": [0, 1],
        "timestamp": [0.0, 0.02],
        "action.motion_token": [np.zeros(64), np.ones(64)],
        "teleop.left_hand_joints": [np.zeros(7), np.ones(7)],
        "teleop.right_hand_joints": [np.ones(7), np.zeros(7)],
    }

    frames = frames_from_columns(columns, target_episode_index=4, expected_length=2)

    assert len(frames) == 2
    assert all(frame.motion_token.dtype == np.float32 for frame in frames)
    np.testing.assert_array_equal(frames[1].motion_token, np.ones(64, dtype=np.float32))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("episode_index", [4, 3], "episode_index"),
        ("frame_index", [0, 2], "frame_index"),
        ("timestamp", [0.0, 0.021], "timestamp"),
    ],
)
def test_frame_loading_fails_closed_on_local_timeline_mismatch(
    field: str,
    value: list[float] | list[int],
    message: str,
) -> None:
    columns = {
        "episode_index": [4, 4],
        "frame_index": [0, 1],
        "timestamp": [0.0, 0.02],
        "action.motion_token": [np.zeros(64), np.ones(64)],
        "teleop.left_hand_joints": [np.zeros(7), np.ones(7)],
        "teleop.right_hand_joints": [np.ones(7), np.zeros(7)],
    }
    columns[field] = value

    with pytest.raises(ValueError, match=message):
        frames_from_columns(columns, target_episode_index=4, expected_length=2)


def test_wxyz_roll_pitch_conversion_is_explicit() -> None:
    half = math.sqrt(0.5)

    roll, pitch = quaternion_wxyz_to_roll_pitch(np.array([half, half, 0.0, 0.0]))

    assert roll == pytest.approx(math.pi / 2)
    assert pitch == pytest.approx(0.0)


def test_robot_actuator_selection_ignores_free_base_and_preserves_robot_order() -> None:
    model = SimpleNamespace(
        actuator_trnid=np.array(
            [
                [0, 0],  # free-base/general actuator: not a robot joint
                [3, 0],
                [1, 0],
                [2, 0],
                [-1, 0],
            ]
        )
    )

    actuator_ids, joint_ids = select_robot_actuator_joint_ids(model, np.array([1, 2, 3]))

    np.testing.assert_array_equal(actuator_ids, [2, 3, 1])
    np.testing.assert_array_equal(joint_ids, [1, 2, 3])


def test_start_command_repetition_is_bounded_and_deterministic() -> None:
    payloads: list[bytes] = []
    delays: list[float] = []

    send_repeated_command(
        b"start-v4",
        send=payloads.append,
        attempts=3,
        interval_s=0.02,
        sleep=delays.append,
    )

    assert payloads == [b"start-v4"] * 3
    assert delays == [0.02, 0.02]


def test_g1_debug_ack_requires_new_token_hands_and_controller_action() -> None:
    frame = action_frame(0.25)
    message = {
        "control_loop_type": "cpp",
        "index": 8,
        "token_state": frame.motion_token.astype(np.float64),
        "left_hand_q_measured": frame.left_hand.astype(np.float64),
        "right_hand_q_measured": frame.right_hand.astype(np.float64),
        "last_left_hand_action": np.zeros(7),
        "last_right_hand_action": np.zeros(7),
        "last_action": np.zeros(29),
    }

    assert validate_state_ack(message, frame, last_index=7) == 8
    assert validate_state_ack(message, frame, last_index=8) is None
    message["token_state"] = np.zeros(64)
    assert validate_state_ack(message, frame, last_index=7) is None
    del message["last_action"]
    with pytest.raises(RuntimeError, match="acknowledgement"):
        validate_state_ack(message, frame, last_index=7)


def test_g1_debug_ack_uses_current_tick_hand_buffers_not_stale_logged_actions() -> None:
    frame = action_frame(0.25)
    message = {
        "control_loop_type": "cpp",
        "index": 8,
        "token_state": frame.motion_token.astype(np.float64),
        "left_hand_q_measured": frame.left_hand.astype(np.float64),
        "right_hand_q_measured": frame.right_hand.astype(np.float64),
        "last_left_hand_action": np.full(7, -0.5),
        "last_right_hand_action": np.full(7, 0.5),
        "last_action": np.zeros(29),
    }

    assert validate_state_ack(message, frame, last_index=7) == 8
    message["left_hand_q_measured"] = np.zeros(7)
    assert validate_state_ack(message, frame, last_index=7) is None


def test_runtime_ack_also_requires_all_three_simulator_command_receipts() -> None:
    frame = action_frame(0.25)
    message = {
        "control_loop_type": "cpp",
        "index": 8,
        "token_state": frame.motion_token.astype(np.float64),
        "left_hand_q_measured": frame.left_hand.astype(np.float64),
        "right_hand_q_measured": frame.right_hand.astype(np.float64),
        "last_action": np.zeros(29),
    }

    class Subscriber:
        @staticmethod
        def get_msg(*, clear: bool) -> object:
            assert clear is True
            return message

    bridge = SimpleNamespace(
        low_cmd_received=True,
        left_hand_cmd_received=True,
        right_hand_cmd_received=False,
    )
    runtime = object.__new__(MujocoZmqRuntime)
    runtime._state_subscriber = Subscriber()
    runtime._last_state_index = 7
    runtime._simulator = SimpleNamespace(unitree_bridge=bridge)

    assert runtime._poll_ack(frame) is False
    assert runtime._last_state_index == 7
    bridge.right_hand_cmd_received = True
    assert runtime._poll_ack(frame) is True
    assert runtime._last_state_index == 8


def test_command_boundary_checks_ack_that_arrived_during_final_pace() -> None:
    frame = action_frame(0.25)
    runtime = object.__new__(MujocoZmqRuntime)
    runtime._expected_frame = frame
    runtime._expected_ack_received = False
    runtime.collector = SimpleNamespace(sim_dt=0.005)
    polls: list[ReplayActionFrame] = []
    runtime._poll_ack = lambda value: polls.append(value) or True

    runtime.assert_action_ack(frame, "warmup", 0, 0.0)

    assert polls == [frame]
    assert runtime._expected_ack_received is True


def test_command_boundary_allows_one_50hz_period_for_async_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = action_frame(0.25)
    runtime = object.__new__(MujocoZmqRuntime)
    runtime._expected_frame = frame
    runtime._expected_ack_received = False
    runtime.collector = SimpleNamespace(sim_dt=0.005)
    timeouts: list[float] = []

    def poll_with_timeout(poll: object, *, timeout_s: float) -> bool:
        del poll
        timeouts.append(timeout_s)
        return True

    monkeypatch.setattr(replay_cli, "poll_until_acknowledged", poll_with_timeout)

    runtime.assert_action_ack(frame, "trajectory", 7, 0.14)

    assert timeouts == pytest.approx([0.02])


def test_command_ack_wait_is_bounded_to_one_simulator_step() -> None:
    clock = [10.0]
    polls: list[float] = []

    def poll() -> bool:
        polls.append(clock[0])
        return len(polls) == 3

    acknowledged = poll_until_acknowledged(
        poll,
        timeout_s=0.005,
        poll_period_s=0.001,
        monotonic=lambda: clock[0],
        sleep=lambda delay: clock.__setitem__(0, clock[0] + delay),
    )

    assert acknowledged is True
    assert polls == pytest.approx([10.0, 10.001, 10.002])
    assert clock[0] == pytest.approx(10.002)


def test_command_ack_wait_fails_at_exact_deadline() -> None:
    clock = [20.0]

    acknowledged = poll_until_acknowledged(
        lambda: False,
        timeout_s=0.005,
        poll_period_s=0.001,
        monotonic=lambda: clock[0],
        sleep=lambda delay: clock.__setitem__(0, clock[0] + delay),
    )

    assert acknowledged is False
    assert clock[0] == pytest.approx(20.005)


def test_unitree_bridge_creates_command_locks_before_subscribers_can_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gear_sonic.utils.mujoco_sim import unitree_sdk2py_bridge as bridge_module

    class ImmediatePublisher:
        def __init__(self, *args: object) -> None:
            del args

        def Init(self) -> None:
            return None

    class ImmediateSubscriber:
        def __init__(self, *args: object) -> None:
            del args

        def Init(self, handler: object, queue_depth: int) -> None:
            assert queue_depth == 1
            handler(object())

    monkeypatch.setattr(bridge_module, "ChannelPublisher", ImmediatePublisher)
    monkeypatch.setattr(bridge_module, "ChannelSubscriber", ImmediateSubscriber)

    bridge = bridge_module.UnitreeSdk2Bridge(
        {
            "ROBOT_TYPE": "g1_29dof",
            "NUM_MOTORS": 29,
            "NUM_HAND_MOTORS": 7,
            "USE_SENSOR": False,
        }
    )

    assert bridge.low_cmd_lock is not None
    assert bridge.left_hand_cmd_lock is not None
    assert bridge.right_hand_cmd_lock is not None


def test_default_mujoco_env_defines_disabled_elastic_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gear_sonic.utils.mujoco_sim import base_sim

    robot = SimpleNamespace(
        NUM_JOINTS=29,
        NUM_HAND_JOINTS=7,
        MOTOR_EFFORT_LIMIT_LIST=[1.0] * 43,
    )
    monkeypatch.setattr(base_sim, "Robot", lambda config: robot)
    monkeypatch.setattr(base_sim.DefaultEnv, "init_scene", lambda self: None)

    env = base_sim.DefaultEnv(
        {"SIMULATE_DT": 0.005},
        onscreen=False,
        offscreen=False,
        enable_image_publish=False,
    )

    assert env.elastic_band is None


def test_runtime_holds_nominal_state_only_until_first_controller_ack() -> None:
    events: list[object] = []
    data = SimpleNamespace(
        qpos=np.array([9.0, 9.0]),
        qvel=np.array([4.0]),
        qacc=np.array([5.0]),
        ctrl=np.array([6.0]),
        time=7.0,
    )
    bridge = SimpleNamespace(low_cmd_received=False)

    class Env:
        mj_data = data
        mj_model = object()

        @staticmethod
        def sim_step() -> None:
            events.append(
                (
                    data.qpos.copy(),
                    data.qvel.copy(),
                    data.qacc.copy(),
                    data.ctrl.copy(),
                    data.time,
                )
            )

    runtime = object.__new__(MujocoZmqRuntime)
    runtime._simulator = SimpleNamespace(sim_env=Env(), unitree_bridge=bridge)
    runtime._initial_qpos = np.array([1.0, 2.0])
    runtime._controller_acknowledged = False
    runtime._mujoco = SimpleNamespace(mj_forward=lambda model, value: events.append("forward"))

    runtime._step_unmeasured()
    runtime._controller_acknowledged = True
    data.qpos[:] = 8.0
    runtime._step_unmeasured()

    assert events[0] == "forward"
    np.testing.assert_array_equal(events[1][0], [1.0, 2.0])
    np.testing.assert_array_equal(events[1][1], [0.0])
    np.testing.assert_array_equal(events[1][2], [0.0])
    np.testing.assert_array_equal(events[1][3], [0.0])
    assert events[1][4] == 0.0
    np.testing.assert_array_equal(events[2][0], [8.0, 8.0])


def test_runtime_restores_exact_nominal_state_after_prime_before_measured_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = SimpleNamespace(
        qpos=np.array([9.0, 9.0]),
        qvel=np.array([4.0]),
        qacc=np.array([5.0]),
        ctrl=np.array([6.0]),
        time=7.0,
    )
    forward_calls: list[object] = []
    runtime = object.__new__(MujocoZmqRuntime)
    runtime._process = object()
    runtime.collector = SimpleNamespace(sim_dt=0.005)
    runtime._simulator = SimpleNamespace(
        sim_env=SimpleNamespace(mj_model="model", mj_data=data),
    )
    runtime._initial_qpos = np.array([1.0, 2.0])
    runtime._mujoco = SimpleNamespace(mj_forward=lambda model, value: forward_calls.append((model, value.time)))
    runtime._controller_acknowledged = False

    def complete_prime(*args: object, **kwargs: object) -> int:
        del args, kwargs
        data.qpos[:] = 8.0
        data.qvel[:] = 3.0
        data.qacc[:] = 2.0
        data.ctrl[:] = 1.0
        data.time = 3.0
        runtime._controller_acknowledged = True
        return 600

    monkeypatch.setattr(replay_cli, "prime_controller", complete_prime)

    runtime.prime(action_frame(0.0), timeout_s=60.0, pace=lambda delay: None)

    np.testing.assert_array_equal(data.qpos, [1.0, 2.0])
    np.testing.assert_array_equal(data.qvel, [0.0])
    np.testing.assert_array_equal(data.qacc, [0.0])
    np.testing.assert_array_equal(data.ctrl, [0.0])
    assert data.time == 0.0
    assert forward_calls == [("model", 0.0)]


def test_authenticated_parquet_decode_failure_is_target_validation_error(tmp_path: Path) -> None:
    parquet = tmp_path / "episode.parquet"
    parquet.write_bytes(b"invalid parquet")
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()

    with pytest.raises(TargetValidationError, match="cannot be decoded"):
        replay_cli._read_authenticated_replay_table(
            parquet,
            digest,
            lambda path: (_ for _ in ()).throw(OSError("decode failed")),
        )


def test_priming_waits_through_three_second_cpp_init_and_requires_real_ack() -> None:
    frame = action_frame(0.0)
    process = FakeProcess()
    clock = [0.0]
    counts = {"action": 0, "start": 0, "step": 0, "ack": 0}

    def pace(delay: float) -> None:
        clock[0] += delay

    steps = prime_controller(
        frame,
        process=process,
        sim_dt=0.005,
        timeout_s=3.1,
        publish_action=lambda value: counts.__setitem__("action", counts["action"] + 1),
        publish_start=lambda: counts.__setitem__("start", counts["start"] + 1),
        step=lambda: counts.__setitem__("step", counts["step"] + 1),
        acknowledged=lambda value: counts.__setitem__("ack", counts["ack"] + 1) or True,
        monotonic=lambda: clock[0],
        pace=pace,
    )

    assert steps == 600
    assert counts == {"action": 150, "start": 150, "step": 600, "ack": 600}
    assert clock[0] == pytest.approx(3.0)


def test_priming_fails_bounded_when_controller_never_acknowledges() -> None:
    clock = [0.0]

    with pytest.raises(TimeoutError, match="acknowledgement"):
        prime_controller(
            action_frame(0.0),
            process=FakeProcess(),
            sim_dt=0.005,
            timeout_s=3.01,
            publish_action=lambda value: None,
            publish_start=lambda: None,
            step=lambda: None,
            acknowledged=lambda value: False,
            monotonic=lambda: clock[0],
            pace=lambda delay: clock.__setitem__(0, clock[0] + delay),
        )


def test_deadline_pacer_compensates_step_work_to_keep_exact_frequency() -> None:
    clock = [10.0]
    delays: list[float] = []

    def monotonic() -> float:
        return clock[0]

    def sleep(delay: float) -> None:
        delays.append(delay)
        clock[0] += delay

    pacer = DeadlinePacer(monotonic=monotonic, sleep=sleep)
    clock[0] += 0.001  # first simulator step work
    pacer(0.005)
    clock[0] += 0.002  # second simulator step work
    pacer(0.005)

    assert delays == pytest.approx([0.004, 0.003])
    assert clock[0] == pytest.approx(10.01)


class FakeReplayRuntime:
    def __init__(self, metrics: ReplayMetricsCollector, *, publish_error: Exception | None = None):
        self.collector = metrics
        self.publish_error = publish_error
        self.events: list[object] = []

    def readiness_probe(self) -> bool:
        self.events.append("probe")
        return True

    def prime(self, frame: ReplayActionFrame, *, timeout_s: float, pace: object) -> None:
        del frame, timeout_s, pace
        self.events.append("prime")

    def publish(self, frame: ReplayActionFrame, phase: str, source_index: int, time_s: float) -> None:
        del frame, source_index, time_s
        self.events.append(("publish", phase))
        if self.publish_error is not None:
            raise self.publish_error

    def step(self, time_s: float) -> ReplaySample:
        return sample(time_s=time_s)

    def assert_action_ack(
        self,
        frame: ReplayActionFrame,
        phase: str,
        source_index: int,
        time_s: float,
    ) -> None:
        del frame, source_index, time_s
        self.events.append(("ack", phase))

    def stop_control(self) -> None:
        self.events.append("stop")

    def close(self) -> None:
        self.events.append("close")


def test_owned_replay_cleans_runtime_and_only_its_child_on_failure() -> None:
    runtime = FakeReplayRuntime(collector(mass=100.0), publish_error=RuntimeError("publish failed"))
    process = FakeProcess()

    with pytest.raises(RuntimeError, match="publish failed"):
        execute_owned_replay(
            [action_frame(0.0)],
            deployment_command=["owned-deploy"],
            runtime=runtime,
            process_factory=lambda command: process,
            readiness_timeout_s=1.0,
            pace=lambda delay: None,
        )

    assert runtime.events == ["prime", ("publish", "warmup"), "stop", "close"]
    assert process.events == [("signal", signal.SIGINT), ("wait", 5.0)]


def test_owned_replay_stops_child_before_closing_dds_runtime() -> None:
    cleanup_order: list[str] = []

    class OrderedRuntime(FakeReplayRuntime):
        def stop_control(self) -> None:
            cleanup_order.append("stop_control")

        def close(self) -> None:
            cleanup_order.append("close_runtime")

    class OrderedProcess(FakeProcess):
        def send_signal(self, value: int) -> None:
            cleanup_order.append("signal_child")
            super().send_signal(value)

        def wait(self, timeout: float) -> int:
            cleanup_order.append("wait_child")
            return super().wait(timeout)

    runtime = OrderedRuntime(collector(mass=100.0), publish_error=RuntimeError("publish failed"))

    with pytest.raises(RuntimeError, match="publish failed"):
        execute_owned_replay(
            [action_frame(0.0)],
            deployment_command=["owned-deploy"],
            runtime=runtime,
            process_factory=lambda command: OrderedProcess(),
            readiness_timeout_s=3.1,
            pace=lambda delay: None,
        )

    assert cleanup_order == [
        "stop_control",
        "signal_child",
        "wait_child",
        "close_runtime",
    ]


def test_owned_replay_fails_and_cleans_up_when_continuous_ack_is_missed() -> None:
    class MissingAckRuntime(FakeReplayRuntime):
        def assert_action_ack(
            self,
            frame: ReplayActionFrame,
            phase: str,
            source_index: int,
            time_s: float,
        ) -> None:
            del frame, phase, source_index, time_s
            raise RuntimeError("controller acknowledgement missed")

    runtime = MissingAckRuntime(collector(mass=100.0))
    process = FakeProcess()

    with pytest.raises(RuntimeError, match="acknowledgement missed"):
        execute_owned_replay(
            [action_frame(0.0)],
            deployment_command=["owned-deploy"],
            runtime=runtime,
            process_factory=lambda command: process,
            readiness_timeout_s=3.1,
            pace=lambda delay: None,
        )

    assert runtime.events[-2:] == ["stop", "close"]
    assert process.events == [("signal", signal.SIGINT), ("wait", 5.0)]


def test_owned_replay_preserves_primary_failure_and_all_cleanup_evidence() -> None:
    class FailingCleanupRuntime(FakeReplayRuntime):
        def stop_control(self) -> None:
            raise RuntimeError("stop failed")

        def close(self) -> None:
            raise RuntimeError("close failed")

    runtime = FailingCleanupRuntime(
        collector(mass=100.0),
        publish_error=RuntimeError("publish failed"),
    )
    process = FakeProcess(timeout_count=3)

    with pytest.raises(RuntimeError, match="publish failed") as captured:
        execute_owned_replay(
            [action_frame(0.0)],
            deployment_command=["owned-deploy"],
            runtime=runtime,
            process_factory=lambda command: process,
            readiness_timeout_s=3.1,
            pace=lambda delay: None,
        )

    assert captured.value.cleanup_errors == (
        "cleanup stop_control failed: RuntimeError: stop failed",
        "cleanup owned process failed: TimeoutExpired: Command 'owned' timed out after 5.0 seconds",
        "cleanup runtime.close failed: RuntimeError: close failed",
    )


def test_mujoco_runtime_constructor_cleans_partial_simulator_and_zmq_resources() -> None:
    events: list[object] = []
    loaded_config: dict[str, object] = {}
    model = SimpleNamespace(
        nu=3,
        qpos0=np.zeros(10),
        jnt_qposadr=np.array([0, 7, 8, 9]),
        jnt_dofadr=np.array([0, 6, 7, 8]),
        actuator_trnid=np.array([[1, 0], [2, 0], [3, 0]]),
        actuator_forcerange=np.zeros((3, 2)),
        jnt_range=np.array([[-1.0, 1.0]] * 4),
        jnt_actfrcrange=np.array([[-1.0, 1.0], [-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]]),
        jnt_actfrclimited=np.array([0, 1, 1, 1]),
        opt=SimpleNamespace(timestep=0.005),
    )
    data = SimpleNamespace(
        qpos=np.zeros(10),
        qvel=np.zeros(9),
        qacc=np.zeros(9),
        ctrl=np.zeros(3),
        time=1.0,
    )
    env = SimpleNamespace(
        mj_model=model,
        mj_data=data,
        body_joint_index=np.array([1]),
        left_hand_index=np.array([2]),
        right_hand_index=np.array([3]),
        use_floating_root_link=True,
    )

    class FakeSimulator:
        sim_env = env

        def close(self) -> None:
            events.append("simulator.close")

    class FakeConfig:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        @staticmethod
        def load_wbc_yaml() -> dict[str, object]:
            loaded_config.update({"DEFAULT_DOF_ANGLES": [0.0], "ENABLE_ELASTIC_BAND": True})
            return loaded_config

    class FakeMujoco:
        @staticmethod
        def mj_forward(fake_model: object, fake_data: object) -> None:
            assert fake_model is model and fake_data is data

        @staticmethod
        def mj_getTotalmass(fake_model: object) -> float:
            assert fake_model is model
            return 35.0

    class FakeSocket:
        def setsockopt(self, option: object, value: object) -> None:
            del option, value

        def bind(self, endpoint: str) -> None:
            assert endpoint == "tcp://127.0.0.1:5556"
            raise RuntimeError("bind failed")

        def close(self, linger: int) -> None:
            events.append(("socket.close", linger))

    class FakeContext:
        @staticmethod
        def socket(socket_type: object) -> FakeSocket:
            assert socket_type == 1
            return FakeSocket()

        def term(self) -> None:
            events.append("context.term")

    class FakeStateSubscriber:
        def __init__(self, **kwargs: object) -> None:
            events.append(("state.create", kwargs))

        def close(self) -> None:
            events.append("state.close")

    fake_zmq = SimpleNamespace(PUB=1, LINGER=2, Context=FakeContext)
    dependencies = RuntimeDependencies(
        mujoco=FakeMujoco,
        zmq=fake_zmq,
        simulator_factory=lambda *args, **kwargs: FakeSimulator(),
        config_factory=FakeConfig,
        pack_action=lambda **kwargs: b"pose",
        build_command=lambda **kwargs: b"command",
        state_subscriber_factory=FakeStateSubscriber,
    )

    with pytest.raises(RuntimeError, match="bind failed"):
        MujocoZmqRuntime(
            network_interface="lo",
            zmq_host="127.0.0.1",
            action_port=5556,
            state_port=5557,
            _dependencies=dependencies,
        )

    assert events == [
        ("state.create", {"host": "127.0.0.1", "port": 5557, "topic": "g1_debug"}),
        "state.close",
        ("socket.close", 0),
        "context.term",
        "simulator.close",
    ]
    assert loaded_config["ENABLE_ELASTIC_BAND"] is False


def test_runtime_fails_closed_for_a_fixed_root_simulator() -> None:
    with pytest.raises(ValueError, match="floating root"):
        replay_cli._require_floating_root(SimpleNamespace(use_floating_root_link=False))


def test_unitree_bridge_channels_close_exactly_once_across_repeated_cleanup() -> None:
    channel_names = (
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
    events: list[str] = []

    class Channel:
        def __init__(self, name: str) -> None:
            self.name = name

        def Close(self) -> None:
            events.append(self.name)

    bridge = SimpleNamespace(**{name: Channel(name) for name in channel_names})

    close_unitree_bridge_channels(bridge)
    close_unitree_bridge_channels(bridge)

    assert events == list(channel_names)
    assert all(getattr(bridge, name) is None for name in channel_names)


def test_json_report_is_canonical_atomic_and_replaces_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "replay.json"
    output.parent.mkdir()
    output.write_text("stale", encoding="utf-8")

    write_json_report(output, {"z": 1, "a": [2, 3]})

    assert output.read_bytes() == b'{"a":[2,3],"z":1}\n'
    assert not list(output.parent.glob(".replay.json.*.tmp"))


def test_json_report_preserves_nonfinite_gate_evidence_as_explicit_strings(tmp_path: Path) -> None:
    output = tmp_path / "replay.json"

    write_json_report(output, {"max": math.inf, "min": -math.inf, "sample": math.nan})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "max": "Infinity",
        "min": "-Infinity",
        "sample": "NaN",
    }
