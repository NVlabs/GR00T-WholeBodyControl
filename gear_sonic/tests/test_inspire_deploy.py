"""Offline hand selection, shared-world IO, and owned-process cleanup."""

import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

from gear_sonic.utils.hand_control.inspire.config import DEFAULT_MAPPING_PATH
from gear_sonic.utils.hand_control.inspire.contract import HandStatus
from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.inspire.backend import create_remote_backend
from gear_sonic.utils.mujoco_sim.inspire.environment import configure_simulation
from gear_sonic.utils.mujoco_sim.inspire.gateway import InspireSimGateway
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from gear_sonic_deploy.scripts.inspire_hand import FeedbackMonitor, preflight, run

ROOT = Path(__file__).resolve().parents[2]


def test_sim_wrapper_inspire_initializes_dds_once(monkeypatch):
    from gear_sonic.scripts import run_sim_loop as entry
    from gear_sonic.utils.mujoco_sim import base_sim, simulator_factory

    calls = []

    def initialize(domain, interface=None):
        calls.append((domain, interface))
        if len(calls) > 1:
            raise RuntimeError("duplicate DDS domain initialization")

    class Scene:
        def __init__(self, *args, **kwargs):
            self.hand_controller = self.viewer = self.image_publish_process = None

        def set_unitree_bridge(self, bridge):
            self.unitree_bridge = bridge

    # Exercise the real wrapper -> factory -> BaseSimulator constructor path,
    # without creating a DDS domain, rendering a window or stepping physics.
    monkeypatch.setattr(simulator_factory, "ChannelFactoryInitialize", initialize)
    monkeypatch.setattr(base_sim, "ChannelFactoryInitialize", initialize)
    monkeypatch.setattr(base_sim, "DefaultEnv", Scene)
    monkeypatch.setattr(base_sim, "UnitreeSdk2Bridge", lambda config: object())
    config = SimLoopConfig(interface="sim").load_wbc_yaml()
    config.update(HAND_TYPE="inspire", USE_JOYSTICK=False)
    wrapper = entry.SimWrapper(None, "default", config)
    try:
        assert calls == [(config["DOMAIN_ID"], config["INTERFACE"])]
    finally:
        wrapper.sim.close()


def test_sim_wrapper_keeps_default_dex3_initialization(monkeypatch):
    from gear_sonic.scripts import run_sim_loop as entry

    events = []
    simulator = object()
    monkeypatch.setattr(entry, "init_channel", lambda config: events.append("channel"))

    def create(config, env_name, **kwargs):
        events.append("simulator")
        return simulator

    monkeypatch.setattr(entry.SimulatorFactory, "create_simulator", create)
    wrapper = entry.SimWrapper(None, "default", {})
    assert wrapper.sim is simulator
    assert events == ["channel", "simulator"]


@pytest.mark.parametrize(
    "mode, message",
    [
        ("sim", "simulation reset or hand gateway restarted"),
        ("real", "^hand gateway restarted during deployment$"),
    ],
)
def test_gateway_identity_change_still_stops_deployment(mode, message):
    import msgpack

    from gear_sonic.utils.hand_control.inspire.contract import InspireHandState

    row = InspireHandState(9, 0, 0, HandStatus.DISARMED, [0] * 6, [0] * 6).to_wire_dict()
    row.update(simulation=mode == "sim", operator_session=dict(enabled=True, instance_id="new"))
    monitor = FeedbackMonitor.__new__(FeedbackMonitor)
    monitor.mode, monitor.expected_instance = mode, None
    monitor.instance, monitor.sequence, monitor.last_received = "old", 8, 123.0
    monitor.socket = SimpleNamespace(
        poll=lambda timeout: True, recv=lambda: msgpack.packb(row, use_bin_type=True)
    )
    with pytest.raises(RuntimeError, match=message):
        monitor.poll()
    assert (monitor.instance, monitor.sequence, monitor.last_received) == ("old", 8, 123.0)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def hand_config(tmp_path):
    config = yaml.safe_load(DEFAULT_MAPPING_PATH.read_text())
    first, second = free_port(), free_port()
    while first == second:
        second = free_port()
    config["hardware"].update(
        command_endpoint=f"tcp://127.0.0.1:{first}",
        state_endpoint=f"tcp://127.0.0.1:{second}",
        operation_timeout_s=1,
    )
    path = tmp_path / "hand.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


class BodyBridge:
    use_sensor = False
    num_body_motor = 29
    num_hand_motor = 0
    joystick = None

    def __init__(self):
        self.low_cmd_lock = threading.Lock()
        self.low_cmd = SimpleNamespace(
            motor_cmd=[SimpleNamespace(q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0) for _ in range(29)]
        )

    def cmd_received(self):
        return True

    def PublishLowState(self, obs):
        self.obs = obs


def make_env(config_path):
    config = SimLoopConfig().load_wbc_yaml()
    config = configure_simulation(config, config_path)
    config["PRINT_SCENE_INFORMATION"] = False
    env = DefaultEnv(config)
    env.unitree_bridge = BodyBridge()
    mujoco.mj_forward(env.mj_model, env.mj_data)
    return env


@pytest.fixture
def live_sim(hand_config):
    # The simulator thread creates and owns all ZMQ sockets and MuJoCo state.
    shared, errors = {}, []
    ready, stop = threading.Event(), threading.Event()

    def worker():
        gateway = None
        try:
            env = make_env(hand_config)
            gateway = InspireSimGateway(env.mj_model, env.mj_data, hand_config)
            env.hand_controller = gateway
            shared.update(env=env, gateway=gateway)
            ready.set()
            while not stop.is_set():
                env.sim_step()
                stop.wait(0.005)
        except BaseException as exc:
            errors.append(exc)
            ready.set()
        finally:
            if gateway is not None:
                gateway.close()

    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(4)
    assert not errors
    try:
        yield shared
    finally:
        stop.set()
        thread.join(4)
        assert not thread.is_alive()
        assert not errors


def test_named_body_limits_and_shared_physics(hand_config):
    env = make_env(hand_config)
    gateway = InspireSimGateway(env.mj_model, env.mj_data, hand_config)
    env.hand_controller = gateway
    try:
        assert env.mj_model.nu == 41 and env.mj_model.nv > 47
        assert len(env.body_actuator_index) == 29
        for ids, qadr, dadr, aids in [
            (
                env.body_joint_index,
                env.body_qpos_index,
                env.body_dof_index,
                env.body_actuator_index,
            ),
            (
                env.left_hand_index,
                env.left_hand_qpos_index,
                env.left_hand_dof_index,
                env.left_hand_actuator_index,
            ),
            (
                env.right_hand_index,
                env.right_hand_qpos_index,
                env.right_hand_dof_index,
                env.right_hand_actuator_index,
            ),
        ]:
            np.testing.assert_array_equal(qadr, env.mj_model.jnt_qposadr[ids])
            np.testing.assert_array_equal(dadr, env.mj_model.jnt_dofadr[ids])
            np.testing.assert_array_equal(env.mj_model.actuator_trnid[aids, 0], ids)
        official = mujoco.MjModel.from_xml_path(
            str(ROOT / "gear_sonic/data/robots/g1/g1_29dof_with_hand.xml")
        )
        old_limits = SimLoopConfig().load_wbc_yaml()["motor_effort_limit_list"]
        for aid in env.body_actuator_index:
            name = env.mj_model.joint(env.mj_model.actuator_trnid[aid, 0]).name
            old_id = official.joint(name).id
            old_aid = np.flatnonzero(official.actuator_trnid[:, 0] == old_id).item()
            assert env.torque_limit[aid] == old_limits[old_aid]
        # A right-arm command and left index command enter the same mj_step.
        env.unitree_bridge.low_cmd.motor_cmd[22].tau = 0.3
        from gear_sonic.utils.hand_control.inspire.contract import InspireHandCommand

        gateway.controller.arm()
        now = time.monotonic_ns()
        gateway.controller.submit(
            InspireHandCommand(1, now, [0, 0, 0, 0.08, 0, 0], [0] * 6), now_ns=now
        )
        before = env.mj_data.qpos.copy()
        env.sim_step()
        assert env.mj_data.ctrl[env.body_actuator_index[22]] == pytest.approx(0.3)
        assert env.mj_data.ctrl[gateway.controller.layout.left.actuator_ids[3]] > 0
        assert not np.array_equal(before, env.mj_data.qpos)
        gateway.step(now_ns=now + 250_000_000)
        assert gateway.controller.guard.status == HandStatus.STALE
        assert np.all(env.mj_data.ctrl[env.left_hand_actuator_index] == 0)
        old_instance = gateway.meta["instance_id"]
        env.reset()
        assert gateway.controller.guard.last_command is None
        assert gateway.meta["instance_id"] != old_instance
        assert not gateway.meta["active"]
        assert np.all(env.mj_data.ctrl[env.left_hand_actuator_index] == 0)
    finally:
        gateway.close()


def test_sim_session_moves_and_revokes(hand_config, live_sim):
    hand = create_remote_backend(hand_config)
    hand.connect()
    try:
        state = hand.read_state()
        left, right = np.maximum(0, state.left_position), np.maximum(0, state.right_position)
        hand.set_target(left, right)
        for i in range(1, 31):
            time.sleep(0.02)
            target = left.copy()
            target[3] += 0.008 * i
            hand.set_target(target, right)
        assert hand.read_state().left_position[3] > state.left_position[3] + 0.01
        hand.stop()
        assert hand._session.rows["hand"][0]["status"] == "DISARMED"
        # Explicit new session after a normal close is allowed.
    finally:
        hand.close()
    hand.connect()
    hand.close()


def test_no_dex3_topics_in_inspire(monkeypatch):
    import gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge as bridge

    topics = []

    class Channel:
        def __init__(self, name, *_):
            topics.append(name)

        def Init(self, *_):
            pass

        def Write(self, *_):
            pass

    monkeypatch.setattr(bridge, "ChannelPublisher", Channel)
    monkeypatch.setattr(bridge, "ChannelSubscriber", Channel)
    config = SimLoopConfig().load_wbc_yaml()
    UnitreeSdk2Bridge(config)
    assert len([t for t in topics if "/dex3/" in t]) == 4
    topics.clear()
    config["HAND_TYPE"] = "inspire"
    isolated = UnitreeSdk2Bridge(config)
    assert not any("/dex3/" in t for t in topics)
    assert "rt/lowcmd" in topics and "rt/lowstate" in topics
    assert isolated.GetAction()[0].shape == (29,)


@pytest.mark.parametrize(
    "args",
    [
        ["--hand"],
        ["--hand", "bogus"],
        ["--hand", "--help"],
        ["--hand-config", "x"],
        ["--hand-typo", "inspire"],
        ["--logs-dir"],
        ["--logs-dir", "--help"],
    ],
)
def test_invalid_shell_options_fail_before_setup(args):
    result = subprocess.run(
        ["bash", str(ROOT / "gear_sonic_deploy/deploy.sh"), *args], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Building the project" not in result.stdout


@pytest.mark.parametrize("csv_logs", [False, True])
def test_shell_hand_routing_uses_no_setup_or_hardware(tmp_path, csv_logs):
    # Copy only the launcher, stub build/setup/exec, and capture final argv.
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    (deploy / "scripts").mkdir()
    (deploy / "scripts/setup_env.sh").write_text(":\n")
    (deploy / "deploy.sh").write_text((ROOT / "gear_sonic_deploy/deploy.sh").read_text())
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls"
    for name in ("just", "cmake", "python3"):
        file = bindir / name
        file.write_text(
            f"#!{sys.executable}\nimport json,os,sys\n"
            'with open(os.environ["TEST_CALLS"], "a") as output:\n'
            ' output.write(json.dumps([os.path.basename(sys.argv[0]), *sys.argv[1:]])+"\\n")\n'
        )
        file.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir) + ":" + os.environ["PATH"], TEST_CALLS=str(log))
    for hand, mode, remote in [
        (hand, mode, False) for hand in (None, "dex3", "inspire") for mode in ("sim", "real")
    ] + [("inspire", "real", True)]:
        for confirm in ("y", "n"):
            log.write_text("")
            # Explicit real interface avoids environment-dependent auto detection.
            args = [mode if mode == "sim" else "test-interface"]
            if hand:
                args.extend(["--hand", hand])
            if remote:
                args.append("--hand-remote")
            # Spaces must remain one argument through both the preview and exec.
            checkpoint = str(tmp_path / "model directory/model")
            args.extend(["--cp", checkpoint, "--input-type", "keyboard"])
            logs_dir = str(tmp_path / "body logs")
            if csv_logs:
                args.extend(["--enable-csv-logs", "--logs-dir", logs_dir])
            result = subprocess.run(
                ["bash", str(deploy / "deploy.sh"), *args],
                input=confirm + "\n",
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0, result.stderr
            calls = [json.loads(row) for row in log.read_text().splitlines()]
            body_calls = [row for row in calls if row[:2] == ["just", "run"] or "--" in row]
            if hand == "inspire":
                preflight_call = next(row for row in calls if "--check" in row)
                assert preflight_call[preflight_call.index("--mode") + 1] == mode
                assert ("--remote" in preflight_call) == remote
            else:
                assert not any("gear_sonic_deploy.scripts.inspire_hand" in row for row in calls)
            if confirm == "n":
                assert not body_calls
                continue
            assert len(body_calls) == 1
            actual = body_calls[0]
            assert actual[actual.index("--hand") + 1] == (hand or "dex3")
            assert checkpoint + "_decoder.onnx" in actual
            assert actual[actual.index("--encoder-file") + 1] == checkpoint + "_encoder.onnx"
            assert actual[actual.index("--input-type") + 1] == "keyboard"
            assert ("--disable-crc-check" in actual) == (mode == "sim")
            assert ("--enable-csv-logs" in actual) == csv_logs
            assert ("--logs-dir" in actual) == csv_logs
            if csv_logs:
                assert actual[actual.index("--logs-dir") + 1] == logs_dir
            preview = result.stdout.split("The following command will be executed:", 1)[1]
            command_line = next(line for line in preview.splitlines() if line.startswith("  "))
            assert shlex.split(command_line) == actual


def test_sim_preflight_never_resolves_serial(hand_config):
    assert preflight("sim", hand_config, "/missing/driver") is None
    with pytest.raises(ValueError, match="left_device"):
        preflight("real", hand_config, "/missing/driver")


def test_sim_launcher_leaves_existing_server_alive(hand_config, live_sim):
    result = run(
        "sim", hand_config, "/never/a/serial/driver", [sys.executable, "-c", "raise SystemExit(7)"]
    )
    assert result == 7
    hand = create_remote_backend(hand_config)
    hand.connect()
    hand.close()


def test_sim_launcher_rolls_back_failed_body(hand_config, live_sim):
    with pytest.raises(FileNotFoundError):
        run("sim", hand_config, "/never/a/driver", ["/missing/body-command"])
    assert live_sim["gateway"].meta["active"] is False


def test_launcher_signal_reaps_owned_body(hand_config, live_sim, tmp_path):
    pidfile = tmp_path / "pid"
    body = f'import os,time;open({str(pidfile)!r},"w").write(str(os.getpid()));time.sleep(30)'
    process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-m",
            "gear_sonic_deploy.scripts.inspire_hand",
            "--mode",
            "sim",
            "--config",
            str(hand_config),
            "--",
            sys.executable,
            "-c",
            body,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 4
        while not pidfile.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=8)
        assert process.returncode == 143
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert live_sim["gateway"].meta["active"] is False
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("body_fails", [False, True])
def test_real_launcher_owns_driver_and_restores_devices(hand_config, tmp_path, body_fails):
    from gear_sonic.tests.test_inspire_serial_e2e import SerialEmulator

    binary = os.environ.get("INSPIRE_TEST_GATEWAY")
    if not binary:
        pytest.skip("set INSPIRE_TEST_GATEWAY for real gateway PTY lifecycle checks")
    left, right = SerialEmulator(2), SerialEmulator(3)
    try:
        cfg = yaml.safe_load(hand_config.read_text())
        cfg["hardware"].update(
            left_device=left.path,
            right_device=right.path,
            left_id=2,
            right_id=3,
            lock_path=str(tmp_path / "writer.lock"),
        )
        hand_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        if body_fails:
            broken = tmp_path / "body-invalid-format"
            broken.write_text("not an executable format\n")
            broken.chmod(0o755)
            with pytest.raises(OSError, match="Exec format"):
                run("real", hand_config, binary, [str(broken)])
        else:
            assert (
                run(
                    "real",
                    hand_config,
                    binary,
                    [sys.executable, "-c", "import time;time.sleep(.15)"],
                )
                == 0
            )
        assert not left.writes and not right.writes
        assert left.speed == right.speed == [400] * 6
    finally:
        left.close()
        right.close()


def test_real_driver_early_exit_does_not_start_body(hand_config, tmp_path):
    cfg = yaml.safe_load(hand_config.read_text())
    cfg["hardware"].update(left_device="/unused/left", right_device="/unused/right")
    hand_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    driver = tmp_path / "driver"
    driver.write_text("#!/bin/sh\nexit 9\n")
    driver.chmod(0o755)
    marker = tmp_path / "body-started"
    with pytest.raises(RuntimeError, match="exited before readiness"):
        run(
            "real",
            hand_config,
            driver,
            [sys.executable, "-c", f'open({str(marker)!r},"w").close()'],
        )
    assert not marker.exists()


def test_gateway_ownership_shape_and_reset(hand_config):
    import uuid

    from gear_sonic.utils.hand_control.inspire.contract import InspireHandCommand

    env = make_env(hand_config)
    gateway = InspireSimGateway(env.mj_model, env.mj_data, hand_config)
    try:
        owner = uuid.uuid4().hex
        request = dict(
            schema="g1.operator.command.v1",
            operation="PREPARE",
            request_id=uuid.uuid4().hex,
            session_id=owner,
            instance_id=gateway.meta["instance_id"],
            generation=0,
        )
        gateway._receive(request, time.monotonic_ns())
        assert gateway.meta["ok"] and gateway.meta["active"]
        request.update(session_id=uuid.uuid4().hex, request_id=uuid.uuid4().hex, generation=1)
        gateway._receive(request, time.monotonic_ns())
        assert not gateway.meta["ok"] and gateway.meta["session_id"] == owner
        now = time.monotonic_ns()
        command = InspireHandCommand(1, now, [0] * 6, [0] * 6).to_wire_dict()
        command["operator_session_id"] = request["session_id"]
        gateway._receive(command, now)
        assert gateway.controller.guard.last_command is None
        command["operator_session_id"] = owner
        command["left_q"] = [0] * 7
        with pytest.raises(ValueError):
            gateway._receive(command, now)
        command["left_q"] = [0] * 6
        gateway._receive(command, now)
        assert gateway.controller.guard.last_command.sequence == 1
        gateway.reset()
        gateway._receive(command, now)
        assert gateway.controller.guard.last_command is None
        assert not gateway.meta["active"]
    finally:
        gateway.close()


def test_real_client_rejects_sim_gateway(hand_config, live_sim):
    from gear_sonic.utils.hand_control.inspire.client import InspireRealBackend

    hand = InspireRealBackend(hand_config)
    with pytest.raises(RuntimeError, match="mode differs"):
        hand.connect()
    assert hand._session is None
    assert not live_sim["gateway"].meta["active"]


def test_default_dex3_layout_preserved():
    config = SimLoopConfig().load_wbc_yaml()
    config["PRINT_SCENE_INFORMATION"] = False
    env = DefaultEnv(config)
    assert env.mj_model.nu == 43
    for joints, qadr, dadr, aids in [
        (env.body_joint_index, env.body_qpos_index, env.body_dof_index, env.body_actuator_index),
        (
            env.left_hand_index,
            env.left_hand_qpos_index,
            env.left_hand_dof_index,
            env.left_hand_actuator_index,
        ),
        (
            env.right_hand_index,
            env.right_hand_qpos_index,
            env.right_hand_dof_index,
            env.right_hand_actuator_index,
        ),
    ]:
        np.testing.assert_array_equal(qadr, joints + 6)
        np.testing.assert_array_equal(dadr, joints + 5)
        np.testing.assert_array_equal(aids, joints - 1)


def test_full_sim_entry_point_with_body_dds(hand_config, tmp_path):
    log = tmp_path / "full-sim.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "gear_sonic/scripts/run_sim_loop.py",
                "--hand",
                "inspire",
                "--hand-config",
                str(hand_config),
                "--no-enable-onscreen",
            ],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        hand = create_remote_backend(hand_config)
        try:
            # Pinocchio and DDS setup precede hand readiness; no body policy is run.
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                assert process.poll() is None, log.read_text()
                try:
                    hand.connect()
                    break
                except RuntimeError:
                    hand.close()
            assert hand._session is not None, log.read_text()
            assert hand.read_state().error is None
            assert hand._session.rows["hand"][0]["simulation"] is True
            hand.close()
        finally:
            hand.close()
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        assert process.returncode == 0, log.read_text()


def test_launcher_rejects_unowned_gateway(hand_config, live_sim):
    from gear_sonic.utils.hand_control.inspire.config import hardware_settings

    monitor = FeedbackMonitor(
        hardware_settings(hand_config), "sim", expected_instance="not-this-process"
    )
    try:
        deadline = time.monotonic() + 1
        with pytest.raises(RuntimeError, match="another hand gateway"):
            while time.monotonic() < deadline:
                monitor.poll()
                time.sleep(0.01)
    finally:
        monitor.close()


@pytest.mark.parametrize(
    "args",
    [
        ["sim", "--hand", "inspire", "--hand-remote"],
        ["real", "--hand-remote"],
        ["real", "--hand", "inspire", "--hand-remote", "--hand-driver", "/unused"],
    ],
)
def test_remote_shell_invalid_combinations(args):
    result = subprocess.run(
        ["bash", str(ROOT / "gear_sonic_deploy/deploy.sh"), *args], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Building the project" not in result.stdout


def test_remote_preflight_needs_no_local_serial_or_binary(hand_config):
    assert preflight("real", hand_config, "/missing/driver", remote=True) is None
    with pytest.raises(ValueError, match="real mode"):
        preflight("sim", hand_config, "/missing/driver", remote=True)


def test_remote_launcher_reuses_external_serial_bridge(hand_config, tmp_path):
    from gear_sonic.tests.test_inspire_serial_e2e import SerialEmulator
    from gear_sonic.utils.hand_control.inspire.client import create_backend
    from gear_sonic.utils.hand_control.inspire.config import driver_command

    binary = os.environ.get("INSPIRE_TEST_GATEWAY")
    if not binary:
        pytest.skip("set INSPIRE_TEST_GATEWAY for PTY bridge checks")
    left, right = SerialEmulator(2), SerialEmulator(3)
    process = hand = None
    try:
        cfg = yaml.safe_load(hand_config.read_text())
        cfg["hardware"].update(
            left_device=left.path,
            right_device=right.path,
            left_id=2,
            right_id=3,
            lock_path=str(tmp_path / "writer.lock"),
            operation_timeout_s=3,
        )
        hand_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        process = subprocess.Popen(
            driver_command(binary, hand_config),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=dict(os.environ, G1_RL_SESSION_CONTROL="1"),
        )
        # The computer config deliberately has no local serial devices.
        client_config = tmp_path / "client.yaml"
        cfg["hardware"].update(left_device=None, right_device=None)
        client_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        assert (
            run(
                "real",
                client_config,
                "/no/local/driver",
                [sys.executable, "-c", "raise SystemExit(7)"],
                remote=True,
            )
            == 7
        )
        assert process.poll() is None
        hand = create_backend(client_config)
        hand.connect()
        state = hand.read_state()
        hand.set_target(state.left_position, state.right_position)
        with pytest.raises(RuntimeError, match="DISARMED"):
            run(
                "real",
                client_config,
                "/no/local/driver",
                [sys.executable, "-c", "raise SystemExit(0)"],
                remote=True,
            )
        hand.close()
        hand = None
        assert process.poll() is None
        assert (
            run(
                "real",
                client_config,
                "/no/local/driver",
                [sys.executable, "-c", "raise SystemExit(0)"],
                remote=True,
            )
            == 0
        )
        assert process.poll() is None
    finally:
        if hand is not None:
            hand.close()
        if process is not None:
            process.terminate()
            process.communicate(timeout=5)
        left.close()
        right.close()
