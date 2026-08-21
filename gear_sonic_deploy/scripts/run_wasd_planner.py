#!/usr/bin/env python3
"""Launch Gear Sonic deploy and drive planner walking plus named actions.

This script keeps the deploy runtime unchanged and sends packed ZMQ messages to
the existing `zmq_manager` input. It is intended as a small operator tool for
continuous speed and heading control.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import util as ctypes_util
import json
import math
from pathlib import Path
import select
import subprocess
import struct
import sys
import termios
import time
import tty

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HEADER_SIZE = 1280
IDLE = 0
SLOW_WALK = 1
WALK = 2
RUN = 3
LEFT_PUNCH = 11
RIGHT_HOOK = 16

ACTIONS = {
    "1": ("normal", "neutral_kick_r", "neutral_kick_R_001__A543"),
    "2": ("normal", "macarena", "macarena_001__A545"),
    "3": ("normal", "tired_one_leg_jumping", "tired_one_leg_jumping_R_001__A359"),
    "4": ("normal", "dance_in_da_party", "dance_in_da_party_001__A464"),
    "5": ("planner", "left_jab", LEFT_PUNCH),
    "6": ("planner", "right_hook", RIGHT_HOOK),
}


class RawKeyboard:
    def __enter__(self) -> "RawKeyboard":
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def read_available(self) -> list[str]:
        keys: list[str] = []
        while select.select([sys.stdin], [], [], 0)[0]:
            keys.append(sys.stdin.read(1))
        return keys


class ZMQPublisher:
    ZMQ_PUB = 1

    def __init__(self, endpoint: str):
        lib_path = ctypes_util.find_library("zmq")
        if not lib_path:
            raise RuntimeError("Could not find libzmq. Install libzmq3-dev or run gear_sonic_deploy/scripts/install_deps.sh.")

        self._lib = ctypes.CDLL(lib_path)
        self._lib.zmq_ctx_new.restype = ctypes.c_void_p
        self._lib.zmq_socket.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.zmq_socket.restype = ctypes.c_void_p
        self._lib.zmq_bind.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.zmq_bind.restype = ctypes.c_int
        self._lib.zmq_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._lib.zmq_send.restype = ctypes.c_int
        self._lib.zmq_close.argtypes = [ctypes.c_void_p]
        self._lib.zmq_close.restype = ctypes.c_int
        self._lib.zmq_ctx_term.argtypes = [ctypes.c_void_p]
        self._lib.zmq_ctx_term.restype = ctypes.c_int
        self._lib.zmq_errno.restype = ctypes.c_int
        self._lib.zmq_strerror.argtypes = [ctypes.c_int]
        self._lib.zmq_strerror.restype = ctypes.c_char_p

        self._ctx = self._lib.zmq_ctx_new()
        if not self._ctx:
            raise RuntimeError("zmq_ctx_new failed")

        self._socket = self._lib.zmq_socket(self._ctx, self.ZMQ_PUB)
        if not self._socket:
            self._lib.zmq_ctx_term(self._ctx)
            raise RuntimeError("zmq_socket(PUB) failed")

        rc = self._lib.zmq_bind(self._socket, endpoint.encode("utf-8"))
        if rc != 0:
            err = self._lib.zmq_strerror(self._lib.zmq_errno()).decode("utf-8", errors="replace")
            self.close()
            raise RuntimeError(f"zmq_bind({endpoint}) failed: {err}")

    def send(self, data: bytes) -> None:
        buf = ctypes.create_string_buffer(data)
        rc = self._lib.zmq_send(self._socket, buf, len(data), 0)
        if rc < 0:
            err = self._lib.zmq_strerror(self._lib.zmq_errno()).decode("utf-8", errors="replace")
            raise RuntimeError(f"zmq_send failed: {err}")

    def close(self) -> None:
        if getattr(self, "_socket", None):
            self._lib.zmq_close(self._socket)
            self._socket = None
        if getattr(self, "_ctx", None):
            self._lib.zmq_ctx_term(self._ctx)
            self._ctx = None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def locomotion_mode(speed: float) -> int:
    abs_speed = abs(speed)
    if abs_speed < 1e-3:
        return IDLE
    if abs_speed <= 0.8:
        return SLOW_WALK
    if abs_speed <= 2.5:
        return WALK
    return RUN


def unit_from_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw), math.sin(yaw), 0.0]


def build_header(fields: list[dict], count: int = 1) -> bytes:
    header = {"v": 1, "endian": "le", "count": count, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def build_command_message(start: bool, stop: bool, planner: bool) -> bytes:
    return build_command(start=start, stop=stop, planner=planner)


def build_command(
    start: bool,
    stop: bool,
    planner: bool,
    motion_name: str | None = None,
) -> bytes:
    fields = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = b"".join(
        (
            struct.pack("B", int(start)),
            struct.pack("B", int(stop)),
            struct.pack("B", int(planner)),
        )
    )
    if motion_name is not None:
        encoded = motion_name.encode("utf-8") + b"\x00"
        fields.append({"name": "motion_name", "dtype": "u8", "shape": [len(encoded)]})
        payload += encoded
    return b"command" + build_header(fields) + payload


def build_planner_message(
    mode: int, movement: list[float], facing: list[float], speed: float, height: float
) -> bytes:
    fields = [
        {"name": "mode", "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing", "dtype": "f32", "shape": [3]},
        {"name": "speed", "dtype": "f32", "shape": [1]},
        {"name": "height", "dtype": "f32", "shape": [1]},
    ]
    payload = b"".join(
        (
            struct.pack("<i", int(mode)),
            struct.pack("<fff", float(movement[0]), float(movement[1]), float(movement[2])),
            struct.pack("<fff", float(facing[0]), float(facing[1]), float(facing[2])),
            struct.pack("<f", float(speed)),
            struct.pack("<f", float(height)),
        )
    )
    return b"planner" + build_header(fields) + payload


def build_deploy_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(REPO_ROOT / "gear_sonic_deploy" / "deploy.sh"),
        "--yes",
        "--input-type",
        "zmq_manager",
        "--zmq-host",
        args.zmq_host,
        "--zmq-port",
        str(args.port),
    ]
    if args.checkpoint:
        cmd.extend(["--cp", args.checkpoint])
    if args.obs_config:
        cmd.extend(["--obs-config", args.obs_config])
    if args.planner:
        cmd.extend(["--planner", args.planner])
    if args.motion_data:
        cmd.extend(["--motion-data", args.motion_data])
    if args.output_type:
        cmd.extend(["--output-type", args.output_type])
    cmd.append(args.interface)
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gear Sonic deploy with slow-walk planner controls and named actions."
    )
    parser.add_argument("interface", nargs="?", default="sim", help="deploy interface: sim, real, IP, or interface name")
    parser.add_argument("--no-launch", action="store_true", help="only publish ZMQ commands; do not start deploy.sh")
    parser.add_argument("--yes", action="store_true", help="skip the wrapper confirmation for non-sim deploy launches")
    parser.add_argument("--zmq-host", default="localhost", help="host passed to deploy and used by the ZMQ manager")
    parser.add_argument("--bind-host", default="*", help="host/interface for this script's ZMQ PUB socket")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ manager port")
    parser.add_argument("--rate", type=float, default=20.0, help="planner command publish rate in Hz")
    parser.add_argument("--walk-speed", type=float, default=0.4, help="initial SLOW_WALK speed in m/s")
    parser.add_argument("--min-speed", type=float, default=0.2, help="minimum SLOW_WALK speed in m/s")
    parser.add_argument("--max-speed", type=float, default=0.8, help="maximum SLOW_WALK speed in m/s")
    parser.add_argument("--speed-step", type=float, default=0.1, help="speed increment for 0/9 keys in m/s")
    parser.add_argument("--momentum-decay", type=float, default=0.98, help="movement momentum multiplier per tick when no movement key is pressed")
    parser.add_argument("--turn-step", type=float, default=0.12, help="yaw increment per A/D key press in radians")
    parser.add_argument("--checkpoint", help="forwarded to deploy.sh --cp")
    parser.add_argument("--obs-config", help="forwarded to deploy.sh --obs-config")
    parser.add_argument("--planner", help="forwarded to deploy.sh --planner")
    parser.add_argument("--motion-data", help="forwarded to deploy.sh --motion-data")
    parser.add_argument("--output-type", help="forwarded to deploy.sh --output-type")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    period = 1.0 / args.rate

    pub = ZMQPublisher(f"tcp://{args.bind_host}:{args.port}")
    time.sleep(0.5)

    deploy_proc: subprocess.Popen | None = None
    if not args.no_launch:
        if args.interface != "sim" and not args.yes:
            confirm = input(f"About to launch deploy on '{args.interface}'. Proceed? [y/N]: ")
            if confirm.lower() != "y":
                print("Cancelled.")
                return 0
        deploy_proc = subprocess.Popen(
            build_deploy_command(args),
            cwd=REPO_ROOT / "gear_sonic_deploy",
            stdin=subprocess.DEVNULL,
        )

    speed = 0.0
    walk_speed = clamp(args.walk_speed, args.min_speed, args.max_speed)
    yaw = 0.0
    movement = [0.0, 0.0, 0.0]
    momentum = 0.0
    planner_active = True
    selected_action: tuple[str, str, str | int] | None = None
    action_hold_until = 0.0
    running = True

    print("Planner walking: W/S forward/back, A/D adjust-turn, Q/E face turn, 9/0 speed, ,/. strafe, Space stop")
    print("Actions: 1 neutral_kick_r, 2 macarena, 3 tired_one_leg_jumping, 4 dance_in_da_party, 5 left_jab, 6 right_hook")
    print("Press an action number to preview it, Enter to execute. O stops deploy, Esc or Ctrl-C quits.")
    print(f"Publishing command/planner topics on tcp://{args.bind_host}:{args.port}")

    pub.send(build_command(start=True, stop=False, planner=True))
    last = time.monotonic()

    def request_planner() -> None:
        nonlocal planner_active, action_hold_until
        if not planner_active:
            pub.send(build_command(start=True, stop=False, planner=True))
        planner_active = True
        action_hold_until = 0.0

    try:
        with RawKeyboard() as keyboard:
            while running:
                now = time.monotonic()
                dt = now - last
                last = now

                saw_movement_key = False
                for key in keyboard.read_available():
                    if key in {"\x1b", "\x03"}:
                        running = False
                        continue
                    if key in {"\n", "\r"}:
                        if selected_action:
                            action_type, label, payload = selected_action
                            print(f"\nExecuting {label}")
                            speed = 0.0
                            momentum = 0.0
                            if action_type == "normal":
                                planner_active = False
                                pub.send(build_command(start=True, stop=False, planner=False, motion_name=str(payload)))
                            else:
                                planner_active = True
                                pub.send(build_command(start=True, stop=False, planner=True))
                                movement = unit_from_yaw(yaw)
                                action_hold_until = now + 1.0
                                pub.send(build_planner_message(int(payload), movement, unit_from_yaw(yaw), 0.0, -1.0))
                            selected_action = None
                        continue

                    key = key.lower()
                    if key in ACTIONS:
                        selected_action = ACTIONS[key]
                        print(f"\nSelected {key}: {selected_action[1]} (press Enter to execute)")
                    elif key == "w":
                        request_planner()
                        movement = unit_from_yaw(yaw)
                        speed = walk_speed
                        momentum = 1.0
                        saw_movement_key = True
                    elif key == "s":
                        request_planner()
                        facing = unit_from_yaw(yaw)
                        movement = [-facing[0], -facing[1], 0.0]
                        speed = walk_speed
                        momentum = 1.0
                        saw_movement_key = True
                    elif key == "a":
                        request_planner()
                        yaw += args.turn_step
                        movement = unit_from_yaw(yaw)
                        speed = walk_speed
                        momentum = 1.0
                        saw_movement_key = True
                    elif key == "d":
                        request_planner()
                        yaw -= args.turn_step
                        movement = unit_from_yaw(yaw)
                        speed = walk_speed
                        momentum = 1.0
                        saw_movement_key = True
                    elif key == "q":
                        yaw += math.pi / 6.0
                    elif key == "e":
                        yaw -= math.pi / 6.0
                    elif key == ",":
                        request_planner()
                        movement = [-math.sin(yaw), math.cos(yaw), 0.0]
                        speed = walk_speed
                        momentum = 1.0
                        saw_movement_key = True
                    elif key == ".":
                        request_planner()
                        movement = [math.sin(yaw), -math.cos(yaw), 0.0]
                        speed = walk_speed
                        momentum = 1.0
                        saw_movement_key = True
                    elif key == "9":
                        walk_speed = clamp(walk_speed - args.speed_step, args.min_speed, args.max_speed)
                        print(f"\nSLOW_WALK speed: {walk_speed:.2f} m/s")
                    elif key == "0":
                        walk_speed = clamp(walk_speed + args.speed_step, args.min_speed, args.max_speed)
                        print(f"\nSLOW_WALK speed: {walk_speed:.2f} m/s")
                    elif key == " ":
                        speed = 0.0
                        momentum = 0.0
                        request_planner()
                    elif key == "]":
                        pub.send(build_command(start=True, stop=False, planner=True))
                    elif key == "o":
                        pub.send(build_command(start=False, stop=True, planner=True))

                if not saw_movement_key:
                    momentum *= args.momentum_decay
                    if momentum < 0.1:
                        momentum = 0.0
                        speed = 0.0

                facing = unit_from_yaw(yaw)
                if planner_active:
                    if action_hold_until > now:
                        # Keep the one-shot planner action alive briefly.
                        pass
                    else:
                        mode = SLOW_WALK if momentum > 0.0 else IDLE
                        command_speed = speed if mode != IDLE else -1.0
                        pub.send(
                            build_planner_message(
                                mode=mode,
                                movement=movement if mode != IDLE else [0.0, 0.0, 0.0],
                                facing=facing,
                                speed=command_speed,
                                height=-1.0,
                            )
                        )
                else:
                    mode = -1
                    command_speed = 0.0

                selected = selected_action[1] if selected_action else "-"
                sys.stdout.write(
                    "\r"
                    f"walk={walk_speed:.2f} cmd={command_speed:.2f} yaw={yaw:+.2f} "
                    f"mode={mode} selected={selected}      "
                    )
                sys.stdout.flush()

                if deploy_proc and deploy_proc.poll() is not None:
                    print(f"\ndeploy.sh exited with code {deploy_proc.returncode}")
                    running = False

                time.sleep(max(0.0, period - (time.monotonic() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping planner command stream.")
        pub.send(build_planner_message(IDLE, [0.0, 0.0, 0.0], unit_from_yaw(yaw), -1.0, -1.0))
        pub.send(build_command(start=False, stop=True, planner=True))
        time.sleep(0.1)
        pub.close()
        if deploy_proc and deploy_proc.poll() is None:
            deploy_proc.terminate()
            try:
                deploy_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                deploy_proc.kill()
                deploy_proc.wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
