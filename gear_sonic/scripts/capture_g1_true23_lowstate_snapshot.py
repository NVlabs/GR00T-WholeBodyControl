"""Print one validated read-only G1 true23 LowState snapshot."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time

from gear_sonic.scripts.pico_g1_preflight import (
    TRUE23_HARDWARE_JOINT_IDS,
    _load_lowstate_api,
    _validate_lowstate_message,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive")

    initialize, subscriber_type, lowstate_type = _load_lowstate_api()
    ready = threading.Event()
    result: dict[str, object] = {}
    error: list[Exception] = []

    def callback(message: object) -> None:
        if ready.is_set():
            return
        try:
            arrival_s = time.monotonic()
            sample = _validate_lowstate_message(
                message,
                arrival_s,
                TRUE23_HARDWARE_JOINT_IDS,
            )
            motor_state = message.motor_state  # type: ignore[attr-defined]
            result.update(
                {
                    "schema_version": 1,
                    "kind": "g1_true23_validated_lowstate_snapshot",
                    "read_only": True,
                    "interface": args.interface,
                    "arrival_monotonic_ns": int(arrival_s * 1e9),
                    "tick": sample.tick,
                    "mode_machine": sample.mode_machine,
                    "hardware_motor_indices": list(TRUE23_HARDWARE_JOINT_IDS),
                    "q_rad": [
                        float(motor_state[index].q)
                        for index in TRUE23_HARDWARE_JOINT_IDS
                    ],
                    "dq_rad_s": [
                        float(motor_state[index].dq)
                        for index in TRUE23_HARDWARE_JOINT_IDS
                    ],
                    "robot_commands_published": False,
                }
            )
        except Exception as exc:  # pragma: no cover - live hardware path
            error.append(exc)
        finally:
            ready.set()

    route_probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    route_probe.connect(("192.168.123.161", 9))
    local_ip = route_probe.getsockname()[0]
    route_probe.close()
    membership_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
        socket.IPPROTO_UDP,
    )
    membership_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    membership_socket.bind(("", 0))
    membership = socket.inet_aton("239.255.0.1") + socket.inet_aton(local_ip)
    membership_socket.setsockopt(
        socket.IPPROTO_IP,
        socket.IP_ADD_MEMBERSHIP,
        membership,
    )

    initialize(0, args.interface)
    subscriber = subscriber_type("rt/lowstate", lowstate_type)
    try:
        subscriber.Init(callback, 10)
        if not ready.wait(args.timeout_seconds):
            raise TimeoutError("no validated rt/lowstate snapshot before timeout")
    finally:
        subscriber.Close()
        membership_socket.close()
    if error:
        raise error[0]
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
