"""Standalone runner: drive brainco hands from a PICO headset, no arm IK.

Pipeline:
    xrobotoolkit_sdk hand tracking
        -> hand_state_to_unitree_keypoints  (synchronous, this process)
        -> shared (25, 3) keypoint arrays in unitree-hand frame
        -> BraincoController                (subprocess)
        -> rt/brainco/{left,right}/cmd      (DDS)

Run:
    python -m decoupled_wbc.control.teleop.main.run_brainco_teleop \
        --network-interface eth0

The 26-joint layout and OpenXR hand-joint pose convention are fixed by the
Khronos XR_EXT_hand_tracking spec, so there are no on-device knobs.
"""

import argparse
import logging
import time

import xrobotoolkit_sdk as xrt
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from decoupled_wbc.control.teleop.device.pico.brainco import (
    BraincoController,
    hand_state_to_unitree_keypoints,
    make_brainco_shared_arrays,
    push_keypoints_to_shared,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network-interface",
        type=str,
        default="",
        help="Network interface for DDS (e.g. eth0). Empty = SDK default.",
    )
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument(
        "--state-timeout-s",
        type=float,
        default=5.0,
        help="Fail fast if rt/brainco/{left,right}/state isn't publishing "
        "within this window. 0 disables the timeout.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Consecutive valid frames required before publishing retargeted "
        "commands. The hand stays in the safe open pose during warmup.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.network_interface:
        ChannelFactoryInitialize(0, args.network_interface)
    else:
        ChannelFactoryInitialize(0)

    xrt.init()
    print("xrobotoolkit SDK initialised.")

    left_arr, right_arr = make_brainco_shared_arrays()

    # BraincoController spawns its own subprocess; this main loop only feeds it.
    BraincoController(
        left_arr,
        right_arr,
        fps=args.fps,
        state_timeout_s=args.state_timeout_s,
        warmup_frames=args.warmup_frames,
    )

    print("Brainco teleop running. Ctrl-C to stop.")
    period = 1.0 / args.fps
    try:
        while True:
            t0 = time.time()

            left_state = (
                xrt.get_left_hand_tracking_state()
                if xrt.get_left_hand_is_active()
                else None
            )
            right_state = (
                xrt.get_right_hand_tracking_state()
                if xrt.get_right_hand_is_active()
                else None
            )

            push_keypoints_to_shared(
                left_arr, hand_state_to_unitree_keypoints(left_state)
            )
            push_keypoints_to_shared(
                right_arr, hand_state_to_unitree_keypoints(right_state)
            )

            elapsed = time.time() - t0
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        try:
            xrt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
