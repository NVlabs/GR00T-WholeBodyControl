"""Brainco hand integration for Gear-Sonic.

`BraincoController` is exported lazily so unit tests and dev environments that
don't have unitree_sdk2py installed can still import the bridge math.
"""

from decoupled_wbc.control.teleop.device.pico.brainco.brainco_bridge import (
    hand_state_to_unitree_keypoints,
    make_brainco_shared_arrays,
    push_keypoints_to_shared,
)

__all__ = [
    "BraincoController",
    "hand_state_to_unitree_keypoints",
    "make_brainco_shared_arrays",
    "push_keypoints_to_shared",
]


def __getattr__(name):
    if name == "BraincoController":
        from decoupled_wbc.control.teleop.device.pico.brainco.robot_hand_brainco import (
            BraincoController,
        )

        return BraincoController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
