#!/usr/bin/env python3
"""Keyboard tool to switch the G1 arm compliance profile at runtime.

Publishes JSON commands for the C++ arm compliance layer (g1_deploy_onnx_ref
started with --arm-compliance). The current command is re-sent at --rate Hz so
the deploy side's watchdog can tell a live link from a dead one.

Keys (no Enter needed):
    0-9, a-z    select a profile from the menu printed at start
                (built-ins first, then any profiles from --profiles FILE)
    e or SPACE  ESTOP  (latched)
    r           release ESTOP -> RIGID
    q           quit (stops sending; the robot HOLDS the last gains)

Use the SAME --profiles file here and in deploy (--compliance-profiles), so both
sides know the same profile names.

Examples:
    python gear_sonic/scripts/arm_compliance_cli.py
    python gear_sonic/scripts/arm_compliance_cli.py \\
        --profiles gear_sonic_deploy/arm_compliance/study_8sets.json
    python gear_sonic/scripts/arm_compliance_cli.py --send '{"profile": "HUG"}' --duration 2
"""

import argparse
import json
import select
import sys
import termios
import time
import tty

import zmq

# Must match BuiltinProfilesJson() in arm_compliance.hpp
BUILTIN_PROFILES = ["RIGID", "HANDSHAKE", "HUG", "FISTBUMP", "FISTBUMP_SOFTWRIST", "SOFT"]
# Keys available for profiles (SPACE, e, r and q are reserved)
PROFILE_KEYS = "0123456789abcdfghijklmnopstuvwxyz"


def load_profile_names(path):
    with open(path) as f:
        data = json.load(f)
    if "profiles" in data:
        data = data["profiles"]
    return [name for name in data if not name.startswith("_")]


def describe(cmd):
    if cmd.get("estop"):
        return "ESTOP"
    name = cmd.get("profile", "custom")
    return f"{name} (release ESTOP)" if cmd.get("release_estop") else name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=5565, help="port to bind (default: 5565)")
    parser.add_argument("--topic", default="compliance", help="ZMQ topic (default: compliance)")
    parser.add_argument("--rate", type=float, default=10.0, help="resend rate in Hz (default: 10)")
    parser.add_argument("--profiles", help="JSON file with extra profiles (same file as deploy)")
    parser.add_argument("--initial", default="RIGID", help="profile sent at start (default: RIGID)")
    parser.add_argument("--send", help="send this JSON command instead of running interactively")
    parser.add_argument("--duration", type=float, default=1.0, help="with --send: seconds to keep sending")
    args = parser.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://*:{args.port}")
    period = 1.0 / max(args.rate, 1.0)

    def publish(cmd):
        sock.send_string(f"{args.topic} {json.dumps(cmd)}")

    if args.send:
        cmd = json.loads(args.send)
        end = time.time() + args.duration
        while time.time() < end:
            publish(cmd)
            time.sleep(period)
        print(f"sent {describe(cmd)} for {args.duration:.1f} s")
        return

    names = list(BUILTIN_PROFILES)
    if args.profiles:
        for n in load_profile_names(args.profiles):
            if n not in names:
                names.append(n)
    if len(names) > len(PROFILE_KEYS):
        print(f"warning: only the first {len(PROFILE_KEYS)} profiles get a key")
    keymap = {PROFILE_KEYS[i]: {"profile": n} for i, n in enumerate(names[: len(PROFILE_KEYS)])}
    keymap[" "] = {"estop": True}
    keymap["e"] = {"estop": True}
    keymap["r"] = {"release_estop": True, "profile": "RIGID"}

    print("Arm compliance profiles:")
    for k, cmd in keymap.items():
        if "profile" in cmd and not cmd.get("release_estop"):
            print(f"  {k}  {cmd['profile']}")
    print("  e/SPACE  ESTOP      r  release ESTOP -> RIGID      q  quit")
    print(f"\nPublishing on tcp://*:{args.port} topic '{args.topic}' at {args.rate:.0f} Hz")

    current = {"profile": args.initial}
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
            key = sys.stdin.read(1)
            if key.lower() == "q":
                break
            key = key if key == " " else key.lower()
            if key in keymap:
                current = dict(keymap[key])
                publish(current)
                next_send = time.time() + period
                print(f"current: {describe(current)}", flush=True)
                # The release flag is one-shot; keep re-sending the plain profile.
                if current.get("release_estop"):
                    current = {"profile": current["profile"]}
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("stopped sending — the robot keeps the last gains")


if __name__ == "__main__":
    main()
