# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identify which headset produced a body-tracking skeleton, from the data itself.

CloudXR converts a Quest skeleton into the ByteDance 24-joint layout, so the
stream is device-agnostic in every obvious respect: joint order, positions,
validity flags, sample rate and quantisation are indistinguishable between a
PICO 4 Ultra and a Quest 3. What does differ is the per-joint orientation
convention -- which is exactly what quest_skeleton_correction.py corrects.

So classify on the relationship between the two: express each bone's direction
(from positions, device-agnostic) in its parent joint's own frame (from the
quaternions, device-specific). The result is a fixed per-device signature.

Measured on 68 recorded captures (34 paired poses, both headsets): 100%
accurate from a single frame, with the two classes ~60 deg apart and a worst
observed margin of 50 deg. Correcting a Quest capture makes it classify as
PICO, which is what confirms the signature tracks the orientation convention
rather than something incidental.

Two limits worth knowing. The references come from one operator on two
headsets, so an unusual body could in principle sit closer to the wrong class;
the margin threshold exists to make that a refusal rather than a wrong answer.
And this needs body tracking to be flowing -- it cannot answer before the
first frame arrives.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# (parent, child) for the first child of each joint that has one, in the
# ByteDance 24-joint layout. Joints with no child contribute no bone.
BONES = (
    (0, 1),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (22, 23),
)

# Mean unit bone direction in the parent's frame, averaged over 34 poses per
# device. Regenerate with the capture tooling if the tables ever change.
REFERENCE_SIGNATURES = {
    "pico": np.array(
        [
            [-0.569483, -0.800637, +0.186198],
            [-0.104800, -0.994174, -0.025200],
            [+0.108128, -0.994123, +0.005252],
            [-0.033233, +0.981681, -0.187613],
            [+0.034194, -0.995497, +0.088410],
            [-0.044084, -0.995554, +0.083238],
            [+0.042478, +0.997577, -0.055105],
            [-0.281420, -0.434914, -0.855367],
            [+0.223743, -0.440340, -0.869505],
            [+0.056893, +0.984098, +0.168266],
            [-0.116010, +0.848975, -0.515541],
            [-0.983870, +0.080962, +0.159514],
            [+0.982762, +0.149268, +0.109079],
            [-0.994924, -0.045515, +0.089743],
            [+0.992442, -0.040522, +0.115835],
            [-0.994757, +0.004251, -0.102177],
            [+0.996593, +0.004190, -0.082373],
            [-0.977105, -0.119880, +0.175766],
            [+0.228497, -0.926656, -0.298491],
        ]
    ),
    "quest": np.array(
        [
            [+0.303532, -0.062886, -0.950744],
            [-0.999896, -0.014420, -0.000000],
            [+0.999894, +0.014567, +0.000000],
            [-0.990283, +0.139066, +0.000001],
            [-1.000000, +0.000000, -0.000000],
            [+1.000000, -0.000000, +0.000000],
            [-0.981412, -0.191910, +0.000000],
            [+0.933474, -0.350636, -0.075374],
            [-0.933486, +0.350512, +0.075798],
            [-0.970922, +0.239380, +0.002763],
            [-1.000000, -0.000004, -0.000003],
            [-0.995125, -0.090052, +0.040200],
            [+0.995118, +0.090128, -0.040219],
            [-0.999998, -0.001785, -0.000012],
            [+0.999998, +0.001776, -0.000001],
            [-0.999043, +0.041080, +0.015019],
            [+0.999003, -0.042084, -0.014921],
            [-0.999185, -0.036112, +0.018030],
            [+0.471542, +0.854318, +0.218608],
        ]
    ),
}

# The worst margin measured across 68 captures was 50 deg. Refusing below 20
# keeps a wide safety band while still rejecting a skeleton that matches
# neither reference -- a third headset, or a degenerate frame.
MIN_MARGIN_DEG = 20.0


def signature(positions: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    """Bone directions expressed in each parent joint's own frame.

    Args:
        positions: ``(24, 3)`` joint positions.
        orientations: ``(24, 4)`` global joint orientations, xyzw.

    Returns:
        ``(len(BONES), 3)`` array; zero rows mark bones of negligible length.
    """
    positions = np.asarray(positions, dtype=np.float64)
    orientations = np.asarray(orientations, dtype=np.float64)

    out = np.zeros((len(BONES), 3), dtype=np.float64)
    for i, (parent, child) in enumerate(BONES):
        delta = positions[child] - positions[parent]
        length = np.linalg.norm(delta)
        if length < 1e-3:
            continue
        out[i] = R.from_quat(orientations[parent]).inv().apply(delta / length)
    return out


def _mean_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Mean angle between corresponding rows, ignoring zero-length ones."""
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    usable = (na > 1e-6) & (nb > 1e-6)
    if not usable.any():
        return float("nan")
    cos = np.sum(a[usable] * b[usable], axis=-1) / (na[usable] * nb[usable])
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).mean())


def classify(
    positions: np.ndarray,
    orientations: np.ndarray,
    min_margin_deg: float = MIN_MARGIN_DEG,
) -> Tuple[Optional[str], dict]:
    """Identify the source headset from one frame of body tracking.

    Args:
        positions: ``(24, 3)`` joint positions.
        orientations: ``(24, 4)`` global joint orientations, xyzw.
        min_margin_deg: Refuse to decide when the two candidates are closer
            than this.

    Returns:
        ``(source, scores)`` where source is ``"pico"``, ``"quest"`` or
        ``None`` when the frame matches neither clearly enough to act on.
        ``scores`` maps each candidate to its mean angle in degrees, lower
        being a better match.
    """
    sig = signature(positions, orientations)
    scores = {
        name: _mean_angle_deg(sig, ref) for name, ref in REFERENCE_SIGNATURES.items()
    }
    if any(np.isnan(v) for v in scores.values()):
        return None, scores

    best, second = sorted(scores, key=lambda k: scores[k])[:2]
    if scores[second] - scores[best] < min_margin_deg:
        return None, scores
    return best, scores
