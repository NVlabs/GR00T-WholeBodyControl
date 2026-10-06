"""Launch SONIC with the Inspire gateway; simulation is started separately."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

import msgpack
import zmq

from gear_sonic.utils.hand_control.inspire.config import (
    DEFAULT_MAPPING_PATH,
    driver_command,
    hardware_settings,
    load_mapping,
)
from gear_sonic.utils.hand_control.inspire.contract import InspireHandState

DEFAULT_DRIVER = (
    Path(__file__).resolve().parents[2] / "build/inspire-native/inspire_direct_serial_hand_gateway"
)


def preflight(mode, config_path, driver, *, remote=False):
    """Validate configuration without opening serial ports or sending commands."""
    load_mapping(config_path)
    hardware_settings(config_path)
    if remote:
        if mode != "real":
            raise ValueError("--remote requires real mode")
        return None
    if mode == "real":
        argv = driver_command(driver, config_path)
        if not Path(argv[0]).is_file() or not os.access(argv[0], os.X_OK):
            raise ValueError(f"build the Inspire driver first: {argv[0]}")
        return argv
    if mode != "sim":
        raise ValueError("mode must be sim or real")
    return None


class FeedbackMonitor:
    def __init__(self, settings, mode, expected_instance=None):
        self.expected_instance = expected_instance
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(settings["state_endpoint"])
        self.mode = mode
        self.last_received = None
        self.sequence = -1
        self.instance = None

    def poll(self):
        if not self.socket.poll(0):
            return None
        row = msgpack.unpackb(self.socket.recv(), raw=False)
        InspireHandState.from_wire_dict(row)
        if (row.get("simulation") is True) != (self.mode == "sim"):
            raise RuntimeError("hand gateway mode differs from deploy mode")
        meta = row.get("operator_session", {})
        if meta.get("enabled") is not True or not isinstance(meta.get("instance_id"), str):
            raise RuntimeError("gateway session control is not enabled")
        if self.expected_instance is not None and meta["instance_id"] != self.expected_instance:
            raise RuntimeError("feedback is from another hand gateway, not the owned driver")
        if self.instance is not None and self.instance != meta["instance_id"]:
            if self.mode == "sim":
                raise RuntimeError(
                    "simulation reset or hand gateway restarted during deployment; "
                    "check the simulator log for a fall or reset"
                )
            raise RuntimeError("hand gateway restarted during deployment")
        self.instance = meta["instance_id"]
        if row["sequence"] <= self.sequence:
            return None
        self.sequence = row["sequence"]
        self.last_received = time.monotonic()
        return row

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


def stop_process(process):
    """Only signal the process group created by this launcher, with a bound."""
    if process is None:
        return
    for sig, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 1)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
            break
        except subprocess.TimeoutExpired:
            continue
    process.wait()


def run(mode, config_path, driver, body_command, *, remote=False):
    driver_argv = preflight(mode, config_path, driver, remote=remote)
    if not body_command:
        raise ValueError("body command is required")
    if shutil.which(body_command[0]) is None:
        raise FileNotFoundError(f"body executable not found: {body_command[0]}")
    settings = hardware_settings(config_path)
    instance = uuid.uuid4().hex if driver_argv is not None else None
    monitor = FeedbackMonitor(settings, mode, instance)
    driver_process = body_process = None
    interrupted = []
    previous = {}

    def on_signal(number, _frame):
        interrupted.append(number)

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, on_signal)
        if driver_argv is not None:
            driver_process = subprocess.Popen(
                driver_argv,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=dict(os.environ, G1_RL_SESSION_CONTROL="1", G1_HAND_INSTANCE_ID=instance),
            )
        deadline = time.monotonic() + settings["operation_timeout_s"]
        while True:
            if interrupted:
                return 128 + interrupted[0]
            if driver_process is not None and driver_process.poll() is not None:
                raise RuntimeError(
                    f"Inspire driver exited before readiness: {driver_process.returncode}"
                )
            row = monitor.poll()
            if row is not None:
                meta = row["operator_session"]
                if row["status"] != "DISARMED" or meta.get("active") or meta.get("fault_latched"):
                    raise RuntimeError(
                        "hand gateway must be healthy and DISARMED before deployment"
                    )
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "no hand feedback; start the simulator or remote bridge and check its state endpoint"
                )
            time.sleep(0.01)
        print(f"Inspire {mode} ready; hand targets use the independent native6 API.", flush=True)
        body_process = subprocess.Popen(body_command, start_new_session=True)
        while not interrupted:
            result = body_process.poll()
            if result is not None:
                return result
            if driver_process is not None and driver_process.poll() is not None:
                raise RuntimeError("Inspire driver exited during deployment")
            monitor.poll()
            if time.monotonic() - monitor.last_received > settings["state_timeout_s"]:
                raise RuntimeError("hand gateway feedback stopped during deployment")
            time.sleep(0.01)
        return 128 + interrupted[0]
    finally:
        # Body exits through its normal stop path before shutting down owned hand IO.
        stop_process(body_process)
        stop_process(driver_process)
        monitor.close()
        for number, handler in previous.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sim", "real"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--driver", type=Path, help="local gateway executable")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="real only: connect to an independently started bridge",
    )
    parser.add_argument("--check", action="store_true", help="validate without launching processes")
    parser.add_argument("body_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.remote and args.driver is not None:
        parser.error("--remote cannot be combined with --driver")
    args.driver = args.driver or DEFAULT_DRIVER
    try:
        if args.check:
            preflight(args.mode, args.config, args.driver, remote=args.remote)
            return
        command = args.body_command
        if command and command[0] == "--":
            command = command[1:]
        raise SystemExit(run(args.mode, args.config, args.driver, command, remote=args.remote))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Inspire deployment: {exc}\n")


if __name__ == "__main__":
    main()
