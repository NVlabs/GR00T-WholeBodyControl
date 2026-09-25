#!/usr/bin/env python3
"""Keyboard tool to switch the G1 arm compliance profile at runtime.

Publishes JSON commands for the C++ arm compliance layer (g1_deploy_onnx_ref
started with --arm-compliance). The current command is re-sent at --rate Hz so
the deploy side's watchdog can tell a live link from a dead one.

Keys (no Enter needed):
    0 / 1 / 2   P0 rigid / P1 soft / P2 compliant
    e or SPACE  ESTOP  (arms: Kp=0, Kd=8, latched)
    r           release ESTOP -> P0 (ramps over ~1 s)
    q           quit (stops sending; the robot HOLDS the last gains)

Usage (sim, same PC):
    python gear_sonic/scripts/arm_compliance_cli.py
    ./deploy.sh --input-type zmq_manager --arm-compliance sim

One-shot (e.g. from another script):
    python gear_sonic/scripts/arm_compliance_cli.py --send '{"profile": "P2"}' --duration 2
"""

import argparse
import json
import select
import sys
import termios
import time
import tty

import zmq

KEYMAP = {
    "0": {"profile": "P0"},
    "1": {"profile": "P1"},
    "2": {"profile": "P2"},
    "e": {"estop": True},
    " ": {"estop": True},
    "r": {"release_estop": True, "profile": "P0"},
}


def describe(cmd: dict) -> str:
    if cmd.get("estop"):
        return "ESTOP"
    name = cmd.get("profile", "custom")
    return f"{name} (release ESTOP)" if cmd.get("release_estop") else name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=5565, help="port to bind (default: 5565)")
    parser.add_argument("--topic", default="compliance", help="ZMQ topic (default: compliance)")
    parser.add_argument("--rate", type=float, default=10.0, help="resend rate in Hz (default: 10)")
    parser.add_argument("--initial", default="P0", choices=["P0", "P1", "P2"], help="profile sent at start")
    parser.add_argument("--send", help="send this JSON command instead of running interactively")
    parser.add_argument("--duration", type=float, default=1.0, help="with --send: seconds to keep sending")
    args = parser.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://*:{args.port}")
    period = 1.0 / max(args.rate, 1.0)

    def publish(cmd: dict) -> None:
        sock.send_string(f"{args.topic} {json.dumps(cmd)}")

    if args.send:
        cmd = json.loads(args.send)
        end = time.time() + args.duration
        while time.time() < end:
            publish(cmd)
            time.sleep(period)
        print(f"sent {describe(cmd)} for {args.duration:.1f} s")
        return

    current = {"profile": args.initial}
    print(__doc__.split("Usage")[0].strip())
    print(f"\nPublishing on tcp://*:{args.port} topic '{args.topic}' at {args.rate:.0f} Hz")
    print(f"current: {describe(current)}", flush=True)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        next_send = 0.0
        while True:
            now = time.time()
            if now >= next_send:
                publish(current)
                next_send = now + period
            ready, _, _ = select.select([sys.stdin], [], [], 0.01)
            if not ready:
                continue
            key = sys.stdin.read(1).lower()
            if key == "q":
                break
            if key in KEYMAP:
                current = dict(KEYMAP[key])
                publish(current)
                next_send = time.time() + period
                print(f"current: {describe(current)}", flush=True)
                # After a release, keep re-sending the plain profile (release is one-shot)
                if current.get("release_estop"):
                    current = {"profile": current["profile"]}
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("stopped sending — the robot keeps the last gains")


if __name__ == "__main__":
    main()
