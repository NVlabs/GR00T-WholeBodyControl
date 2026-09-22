#!/usr/bin/env python3

import sys
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_


HAND_TOPICS = {
    "left": "rt/brainco/left/state",
    "right": "rt/brainco/right/state",
}

FINGER_NAMES = [
    "thumb",
    "thumb_aux",
    "index",
    "middle",
    "ring",
    "pinky",
]

SAMPLES_PER_HAND = 3
STATE_TIMEOUT_SEC = 3.0


def print_states(hand_name, msg, sample_num):
    print(f"\n{hand_name.upper()} HAND — sample {sample_num}/{SAMPLES_PER_HAND}")
    print(f"{'finger':<12}{'q':>10}{'dq':>10}{'tau_est':>12}")

    for i, finger_name in enumerate(FINGER_NAMES):
        state = msg.states[i]
        print(
            f"{finger_name:<12}"
            f"{state.q:>10.4f}"
            f"{state.dq:>10.4f}"
            f"{state.tau_est:>12.4f}"
        )

    print("-" * 44)


def check_hand(hand_name, topic):
    subscriber = ChannelSubscriber(topic, MotorStates_)
    subscriber.Init()

    print(f"\nChecking {hand_name} hand: {topic}")
    received_samples = 0
    deadline = time.monotonic() + STATE_TIMEOUT_SEC

    try:
        while received_samples < SAMPLES_PER_HAND and time.monotonic() < deadline:
            msg = subscriber.Read()

            if msg is None:
                time.sleep(0.01)
                continue

            if len(msg.states) < len(FINGER_NAMES):
                print(
                    f"{hand_name.upper()} HAND: incomplete state "
                    f"({len(msg.states)}/{len(FINGER_NAMES)} motors)"
                )
                continue

            received_samples += 1
            print_states(hand_name, msg, received_samples)

        if received_samples == SAMPLES_PER_HAND:
            print(f"OK for {hand_name} hand")
            return True

        print(
            f"NO STATE for {hand_name} hand "
            f"(received {received_samples}/{SAMPLES_PER_HAND} valid messages)"
        )
        return False

    finally:
        subscriber.Close()


def main():
    if len(sys.argv) > 1:
        interface = sys.argv[1]
        ChannelFactoryInitialize(0, interface)
        print(f"DDS interface: {interface}")
    else:
        ChannelFactoryInitialize()
        print("DDS interface: default")

    results = {}

    for hand_name, topic in HAND_TOPICS.items():
        results[hand_name] = check_hand(hand_name, topic)

    print("\nFinal result:")
    for hand_name, ok in results.items():
        print(f"{hand_name}: {'OK' if ok else 'NO STATE'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
