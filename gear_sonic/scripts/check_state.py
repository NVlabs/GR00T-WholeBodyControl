import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_


FINGER_NAMES = [
    "Thumb",
    "Thumb_aux",
    "Index",
    "Middle",
    "Ring",
    "Pinky",
]


def print_motor_states(msg):
    if msg is None:
        print("No data")
        return

    print("-" * 60)

    for i, state in enumerate(msg.states):
        name = FINGER_NAMES[i] if i < len(FINGER_NAMES) else f"finger_{i}"

        print(
            f"{i}: {name:10s} | "
            f"q={state.q:.3f} | "
            f"dq={state.dq:.3f} | "
            f"tau/current={state.tau_est:.3f} | "
            f"temp={state.temperature} | "
            f"lost={state.lost}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 echo_brainco_state.py <network_interface> <left|right>")
        print()
        print("Example:")
        print("  python3 echo_brainco_state.py enP8p1s0 left")
        sys.exit(1)

    network_interface = sys.argv[1]
    side = sys.argv[2]

    topic = f"rt/brainco/{side}/state"

    ChannelFactoryInitialize(0, network_interface)

    sub = ChannelSubscriber(topic, MotorStates_)
    sub.Init()

    print(f"Listening topic: {topic}")
    print(f"Network interface: {network_interface}")

    while True:
        msg = sub.Read(1.0)

        if msg is not None:
            print_motor_states(msg)
        else:
            print("No state data. Check brainco_hand_server, network interface, and hand connection.")

        time.sleep(1)