#!/usr/bin/env python3
"""Read and display Unitree G1 upper-body state from DDS.

This tool is deliberately read-only: it creates DDS subscribers, but never
publishes a command to the robot.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any


def _import_unitree_sdk() -> tuple[Any, Any, Any, Any, Any]:
    """Import an installed SDK, or the checkout bundled with this repository."""
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, IMUState_, LowCmd_, LowState_
    except ImportError:
        repository_root = Path(__file__).resolve().parents[2]
        bundled_sdk = repository_root / "external_dependencies" / "unitree_sdk2_python"
        if bundled_sdk.is_dir():
            sys.path.insert(0, str(bundled_sdk))
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
                HandState_,
                IMUState_,
                LowCmd_,
                LowState_,
            )
        except ImportError as error:
            raise SystemExit(
                "unitree_sdk2py is unavailable. Install it, or initialize "
                "external_dependencies/unitree_sdk2_python (including cyclonedds)."
            ) from error
    return ChannelFactoryInitialize, ChannelSubscriber, LowState_, LowCmd_, (IMUState_, HandState_)


# Hardware motor indices used by the official G1 29-DoF SDK examples.
UPPER_BODY_JOINTS = (
    (12, "waist_yaw"),
    (13, "waist_roll"),
    (14, "waist_pitch"),
    (15, "left_shoulder_pitch"),
    (16, "left_shoulder_roll"),
    (17, "left_shoulder_yaw"),
    (18, "left_elbow"),
    (19, "left_wrist_roll"),
    (20, "left_wrist_pitch"),
    (21, "left_wrist_yaw"),
    (22, "right_shoulder_pitch"),
    (23, "right_shoulder_roll"),
    (24, "right_shoulder_yaw"),
    (25, "right_elbow"),
    (26, "right_wrist_roll"),
    (27, "right_wrist_pitch"),
    (28, "right_wrist_yaw"),
)


class LatestMessages:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: dict[str, tuple[float, Any]] = {}

    def callback(self, name: str):
        def save(message: Any) -> None:
            with self._lock:
                # The SDK can reuse callback objects, so keep an independent snapshot.
                self._messages[name] = (time.monotonic(), copy.deepcopy(message))

        return save

    def snapshot(self) -> dict[str, tuple[float, Any]]:
        with self._lock:
            return dict(self._messages)


def _number(value: Any) -> float | int:
    return value if isinstance(value, int) else round(float(value), 6)


def _motor_dict(index: int, name: str, state: Any) -> dict[str, Any]:
    return {
        "index": index,
        "joint": name,
        "mode": int(state.mode),
        "q_rad": _number(state.q),
        "dq_rad_s": _number(state.dq),
        "ddq_rad_s2": _number(state.ddq),
        "tau_est_nm": _number(state.tau_est),
        "temperature_c": [int(value) for value in state.temperature],
        "voltage_v": _number(state.vol),
        "sensor": [int(value) for value in state.sensor],
        "motorstate": int(state.motorstate),
        "reserve": [int(value) for value in state.reserve],
    }


def _motor_command_dict(index: int, name: str, command: Any) -> dict[str, Any]:
    return {
        "index": index,
        "joint": name,
        "mode": int(command.mode),
        "q_des_rad": _number(command.q),
        "dq_des_rad_s": _number(command.dq),
        "tau_ff_nm": _number(command.tau),
        "stiffness_kp": _number(command.kp),
        "damping_kd": _number(command.kd),
        "reserve": int(command.reserve),
    }


def _imu_dict(imu: Any) -> dict[str, Any]:
    return {
        "quaternion_wxyz": [_number(value) for value in imu.quaternion],
        "gyroscope_rad_s": [_number(value) for value in imu.gyroscope],
        "accelerometer_m_s2": [_number(value) for value in imu.accelerometer],
        "rpy_rad": [_number(value) for value in imu.rpy],
        "temperature_c": int(imu.temperature),
    }


def _hand_dict(hand: Any) -> dict[str, Any]:
    return {
        "motors": [
            _motor_dict(index, f"hand_motor_{index}", state)
            for index, state in enumerate(hand.motor_state)
        ],
        "press_sensor_state": [
            {
                "pressure": [_number(value) for value in state.pressure],
                "temperature": [_number(value) for value in state.temperature],
                "lost": int(state.lost),
                "reserve": int(state.reserve),
            }
            for state in hand.press_sensor_state
        ],
        "imu": _imu_dict(hand.imu_state),
        "power_v": _number(hand.power_v),
        "power_a": _number(hand.power_a),
        "system_v": _number(hand.system_v),
        "device_v": _number(hand.device_v),
        "error": [int(value) for value in hand.error],
    }


def make_snapshot(messages: dict[str, tuple[float, Any]]) -> dict[str, Any] | None:
    if "lowstate" not in messages:
        return None

    received_at, lowstate = messages["lowstate"]
    result: dict[str, Any] = {
        "topic": "rt/lowstate",
        "age_ms": round((time.monotonic() - received_at) * 1000.0, 2),
        "version": [int(value) for value in lowstate.version],
        "tick": int(lowstate.tick),
        "mode_pr": int(lowstate.mode_pr),
        "mode_machine": int(lowstate.mode_machine),
        "imu": _imu_dict(lowstate.imu_state),
        "wireless_remote": [int(value) for value in lowstate.wireless_remote],
        "reserve": [int(value) for value in lowstate.reserve],
        "crc": int(lowstate.crc),
        "upper_body_motors": [
            _motor_dict(index, name, lowstate.motor_state[index])
            for index, name in UPPER_BODY_JOINTS
        ],
    }

    for key in ("secondary_imu", "left_hand", "right_hand"):
        if key not in messages:
            continue
        timestamp, message = messages[key]
        value = _imu_dict(message) if key == "secondary_imu" else _hand_dict(message)
        result[key] = {
            "age_ms": round((time.monotonic() - timestamp) * 1000.0, 2),
            "state": value,
        }
    for key, topic in (("lowcmd", "rt/lowcmd"), ("arm_sdk", "rt/arm_sdk")):
        if key not in messages:
            continue
        timestamp, command = messages[key]
        result[key] = {
            "topic": topic,
            "age_ms": round((time.monotonic() - timestamp) * 1000.0, 2),
            "mode_pr": int(command.mode_pr),
            "mode_machine": int(command.mode_machine),
            "upper_body_commands": [
                _motor_command_dict(index, name, command.motor_cmd[index])
                for index, name in UPPER_BODY_JOINTS
            ],
        }
    return result


def print_table(snapshot: dict[str, Any], clear: bool) -> None:
    if clear and sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    print(
        f"rt/lowstate  age={snapshot['age_ms']:.1f} ms  tick={snapshot['tick']}  "
        f"mode_pr={snapshot['mode_pr']}  mode_machine={snapshot['mode_machine']}"
    )
    imu = snapshot["imu"]
    print(f"pelvis IMU  rpy(rad)={imu['rpy_rad']}  gyro(rad/s)={imu['gyroscope_rad_s']}")
    if "secondary_imu" in snapshot:
        torso = snapshot["secondary_imu"]
        print(
            f"torso IMU   age={torso['age_ms']:.1f} ms  "
            f"rpy(rad)={torso['state']['rpy_rad']}  "
            f"gyro(rad/s)={torso['state']['gyroscope_rad_s']}"
        )
    else:
        print("torso IMU   waiting for rt/secondary_imu")

    print("\nidx joint                     q[rad]   dq[rad/s]  tau[Nm]  temp[C]     V  state")
    print("--- ----------------------- --------- ---------- -------- -------- ----- ----------")
    for motor in snapshot["upper_body_motors"]:
        temperatures = "/".join(str(value) for value in motor["temperature_c"])
        print(
            f"{motor['index']:>3} {motor['joint']:<23} "
            f"{motor['q_rad']:>9.4f} {motor['dq_rad_s']:>10.4f} "
            f"{motor['tau_est_nm']:>8.3f} {temperatures:>8} "
            f"{motor['voltage_v']:>5.1f} {motor['motorstate']:#010x}"
        )

    for side in ("left_hand", "right_hand"):
        if side in snapshot:
            hand = snapshot[side]
            print(
                f"\n{side}: age={hand['age_ms']:.1f} ms, "
                f"motors={len(hand['state']['motors'])}, error={hand['state']['error']}"
            )
    for key in ("lowcmd", "arm_sdk"):
        if key not in snapshot:
            continue
        command = snapshot[key]
        print(
            f"\n{command['topic']}  age={command['age_ms']:.1f} ms  "
            "(commanded values, not measured feedback)"
        )
        print("idx joint                       kp       kd      q_des     dq_des     tau_ff")
        print("--- ----------------------- -------- -------- ---------- ---------- ----------")
        for motor in command["upper_body_commands"]:
            print(
                f"{motor['index']:>3} {motor['joint']:<23} "
                f"{motor['stiffness_kp']:>8.3f} {motor['damping_kd']:>8.3f} "
                f"{motor['q_des_rad']:>10.4f} {motor['dq_des_rad_s']:>10.4f} "
                f"{motor['tau_ff_nm']:>10.3f}"
            )
    print("\nCtrl-C to stop", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only DDS viewer for Unitree G1 upper-body state."
    )
    parser.add_argument(
        "interface",
        nargs="?",
        default="wlxfc23cd997021",
        help="network interface used for DDS, e.g. eth0 or enp3s0",
    )
    parser.add_argument("--rate", type=float, default=1.0, help="display refresh rate in Hz")
    parser.add_argument("--once", action="store_true", help="print one received sample and exit")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--hands",
        action="store_true",
        help="also subscribe to rt/dex3/left/state and rt/dex3/right/state",
    )
    parser.add_argument(
        "--commands",
        action="store_true",
        help="also inspect stiffness/damping commands on rt/lowcmd and rt/arm_sdk",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="seconds to wait for the first LowState"
    )
    args = parser.parse_args()
    if args.rate <= 0:
        parser.error("--rate must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    factory_initialize, subscriber_type, lowstate_type, lowcmd_type, extra_types = (
        _import_unitree_sdk()
    )
    imu_type, hand_type = extra_types

    if args.interface:
        factory_initialize(0, args.interface)
    else:
        factory_initialize(0)

    messages = LatestMessages()
    subscribers = []  # Keep subscribers alive for the lifetime of the process.
    subscriptions = [
        ("lowstate", "rt/lowstate", lowstate_type),
        ("secondary_imu", "rt/secondary_imu", imu_type),
    ]
    if args.hands:
        subscriptions.extend(
            [
                ("left_hand", "rt/dex3/left/state", hand_type),
                ("right_hand", "rt/dex3/right/state", hand_type),
            ]
        )
    if args.commands:
        subscriptions.extend(
            [
                ("lowcmd", "rt/lowcmd", lowcmd_type),
                ("arm_sdk", "rt/arm_sdk", lowcmd_type),
            ]
        )

    for name, topic, message_type in subscriptions:
        subscriber = subscriber_type(topic, message_type)
        subscriber.Init(messages.callback(name), 10)
        subscribers.append(subscriber)

    deadline = time.monotonic() + args.timeout
    while "lowstate" not in messages.snapshot():
        if time.monotonic() >= deadline:
            interface = args.interface or "DDS auto-detection"
            print(
                f"No rt/lowstate received in {args.timeout:g} s via {interface}. "
                "Check the robot connection and network interface.",
                file=sys.stderr,
            )
            return 1
        time.sleep(0.02)

    interval = 1.0 / args.rate
    try:
        while True:
            snapshot = make_snapshot(messages.snapshot())
            if snapshot is not None:
                if args.json:
                    print(json.dumps(snapshot, ensure_ascii=False), flush=True)
                else:
                    print_table(snapshot, clear=not args.once)
            if args.once:
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
