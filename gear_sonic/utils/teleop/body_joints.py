# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The 24-joint body-tracking layout, as an enum instead of bare indices.

Every skeleton reaching this repo is in the ``XR_BD_body_tracking`` (ByteDance)
layout: Pico sends it natively, and a Quest skeleton is reduced to it upstream
in isaacteleop before it crosses the DeviceIO schema boundary. So one enum
covers both sources.

``IntEnum`` rather than ``Enum`` deliberately -- members are still ints, so they
index numpy arrays, compare against raw indices from the wire, and work in
``range()`` without conversion. That keeps the enum adoptable one call site at a
time rather than requiring a flag-day change.

It also removes the need for a parallel name table: ``BodyJoint(3).name`` is
``"SPINE1"``, so printing a joint no longer depends on a list staying
index-aligned with the wire format by hand.
"""

from enum import IntEnum


class BodyJoint(IntEnum):
    """Joint indices, index-aligned with ``BodyJointPico`` on the wire."""

    PELVIS = 0
    LEFT_HIP = 1
    RIGHT_HIP = 2
    SPINE1 = 3
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    SPINE2 = 6
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    SPINE3 = 9
    LEFT_FOOT = 10
    RIGHT_FOOT = 11
    NECK = 12
    LEFT_COLLAR = 13
    RIGHT_COLLAR = 14
    HEAD = 15
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    LEFT_HAND = 22
    RIGHT_HAND = 23


#: Number of joints in the layout. Derived, so it cannot drift from the enum.
NUM_BODY_JOINTS = len(BodyJoint)

#: The spine chain: pelvis up through the head.
AXIAL_JOINTS = (
    BodyJoint.PELVIS,
    BodyJoint.SPINE1,
    BodyJoint.SPINE2,
    BodyJoint.SPINE3,
    BodyJoint.NECK,
    BodyJoint.HEAD,
)

#: Left and right limb chains, shoulder/hip outward.
LEFT_LIMB_JOINTS = (
    BodyJoint.LEFT_HIP,
    BodyJoint.LEFT_KNEE,
    BodyJoint.LEFT_ANKLE,
    BodyJoint.LEFT_FOOT,
    BodyJoint.LEFT_COLLAR,
    BodyJoint.LEFT_SHOULDER,
    BodyJoint.LEFT_ELBOW,
    BodyJoint.LEFT_WRIST,
    BodyJoint.LEFT_HAND,
)

RIGHT_LIMB_JOINTS = (
    BodyJoint.RIGHT_HIP,
    BodyJoint.RIGHT_KNEE,
    BodyJoint.RIGHT_ANKLE,
    BodyJoint.RIGHT_FOOT,
    BodyJoint.RIGHT_COLLAR,
    BodyJoint.RIGHT_SHOULDER,
    BodyJoint.RIGHT_ELBOW,
    BodyJoint.RIGHT_WRIST,
    BodyJoint.RIGHT_HAND,
)

#: Legs. Meta's inside-out body tracking does not camera-track these; it infers
#: them from head and hand motion, so they carry less information about the
#: operator's actual pose. See IsaacTeleop docs/source/device/body_tracking.rst.
INFERRED_ON_QUEST = frozenset(
    {
        BodyJoint.LEFT_HIP,
        BodyJoint.RIGHT_HIP,
        BodyJoint.LEFT_KNEE,
        BodyJoint.RIGHT_KNEE,
        BodyJoint.LEFT_ANKLE,
        BodyJoint.RIGHT_ANKLE,
        BodyJoint.LEFT_FOOT,
        BodyJoint.RIGHT_FOOT,
    }
)

__all__ = [
    "BodyJoint",
    "NUM_BODY_JOINTS",
    "AXIAL_JOINTS",
    "LEFT_LIMB_JOINTS",
    "RIGHT_LIMB_JOINTS",
    "INFERRED_ON_QUEST",
]
