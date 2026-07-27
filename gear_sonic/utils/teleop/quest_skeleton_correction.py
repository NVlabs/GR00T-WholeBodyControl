"""Per-joint orientation correction mapping Quest body pose onto Pico conventions.

The CloudXR client SDK converts the Meta Quest IOBT skeleton into the ByteDance
24-joint layout before it reaches this machine (see IsaacTeleop
docs/source/device/body_tracking.rst:209-225). The joint *order* and *positions*
that result are correct -- the LOVR body_tracking sample, which renders
positions only, looks anatomically right on both headsets. The joint
*orientations* are not, and orientation is what drives the robot:
``compute_from_body_poses()`` builds the entire SMPL pose from the 24
quaternions and uses positions only for root translation.

The correction below is the measured per-joint offset, in PARENT-RELATIVE
(local) space, taking Quest orientations to Pico orientations::

    corrected_local[j] = CORRECTION[j].inv() * quest_local[j]

Local space is used because a whole-body turn between captures cancels there,
and because it is the quantity SMPL retargeting actually consumes.

Derivation
----------
Measured from a held T-pose captured on both headsets with
``gear_sonic/scripts/capture_body_skeleton.py`` (Pico median orientation spread
0.18 deg, Quest 0.90 deg -- genuinely static). Validated held-out against an
independently captured A-pose: 18/23 joints land within 20 deg after correction
versus 12/23 before, median residual 16.3 -> 10.2 deg.

The dominant defect it fixes is LEFT_HIP, which arrives from the Quest very
nearly 180 deg flipped (164.8 deg raw -> 10.2 deg corrected) while RIGHT_HIP is
almost correct. That asymmetry alone inverts the left leg and is more than
enough to topple the robot.

Known limitations
-----------------
RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST and LEFT_ELBOW remain pose-dependent:
their offset changes between the T-pose and A-pose by more than a constant
correction can absorb, so they are only partially fixed. They may be genuine
pose-dependence in the SDK mapping, or they may be capture noise -- "arms down"
is far less reproducible than a T-pose. Resolving that needs isolated-limb
captures. Legs and spine are well corrected and are what matter for stability.

Bone lengths also differ between the two skeletons (Quest reports a generic
left-right-symmetric rig: 456.6 mm thighs and 456.5 mm shins on both sides,
versus the Pico's measured 312/336 mm). This correction does not address that;
it only affects position-derived paths such as the 3-point VR pose, not the
SMPL orientation path.
"""

# (x, y, z, w) per joint, index-aligned with BodyJointPico / XR_BD_body_tracking.
QUEST_TO_PICO_LOCAL = (
    (-0.329249449, +0.435571297, -0.668324360, +0.505188080),  #  0 PELVIS          root: yaw removed (arbitrary heading); roll/pitch only
    (+0.042770340, -0.113111553, +0.991518197, -0.047624995),  #  1 LEFT_HIP        good (held-out residual 10.2 deg)
    (-0.076064199, +0.013615989, +0.070794467, +0.994493331),  #  2 RIGHT_HIP       good (held-out residual 14.6 deg)
    (+0.009349257, -0.016966120, +0.007107493, +0.999787090),  #  3 SPINE1          good (held-out residual 1.7 deg)
    (-0.029114504, +0.077208175, -0.098904971, +0.991669829),  #  4 LEFT_KNEE       good (held-out residual 3.1 deg)
    (-0.027014580, -0.056637319, -0.090590280, +0.993909366),  #  5 RIGHT_KNEE      good (held-out residual 7.8 deg)
    (-0.053284053, +0.017199204, +0.050272914, +0.997164796),  #  6 SPINE2          good (held-out residual 7.3 deg)
    (-0.037025379, +0.130475083, -0.538314513, +0.831758895),  #  7 LEFT_ANKLE      good (held-out residual 8.6 deg)
    (-0.115769422, -0.326280426, +0.542933405, -0.765089434),  #  8 RIGHT_ANKLE     good (held-out residual 7.9 deg)
    (+0.002883699, -0.011674074, +0.058154231, +0.998235186),  #  9 SPINE3          good (held-out residual 7.0 deg)
    (+0.091067820, +0.071516442, -0.043370098, +0.992326098),  # 10 LEFT_FOOT       good (held-out residual 2.6 deg)
    (-0.094588156, -0.059183938, -0.067708594, -0.991446362),  # 11 RIGHT_FOOT      good (held-out residual 1.7 deg)
    (+0.055001297, +0.058610699, -0.193056869, +0.977889916),  # 12 NECK            good (held-out residual 12.5 deg)
    (-0.235879998, -0.528321579, +0.178786096, +0.795784184),  # 13 LEFT_COLLAR     good (held-out residual 19.2 deg)
    (+0.642506402, -0.136150777, +0.718590481, -0.228639912),  # 14 RIGHT_COLLAR    POOR - pose-dependent, needs more data (held-out residual 20.7 deg)
    (-0.040531221, -0.000378702, +0.197037691, +0.979557668),  # 15 HEAD            good (held-out residual 12.7 deg)
    (+0.542564271, -0.376575726, -0.113709381, +0.742216216),  # 16 LEFT_SHOULDER   good (held-out residual 12.6 deg)
    (-0.486047860, +0.214306909, +0.038159073, -0.846388747),  # 17 RIGHT_SHOULDER  POOR - pose-dependent, needs more data (held-out residual 99.8 deg)
    (-0.257206160, +0.045123666, -0.101958707, +0.959902739),  # 18 LEFT_ELBOW      POOR - pose-dependent, needs more data (held-out residual 28.0 deg)
    (-0.280705261, +0.002222197, -0.216600443, +0.935031479),  # 19 RIGHT_ELBOW     good (held-out residual 15.1 deg)
    (-0.857425385, -0.029287075, -0.042466697, +0.512016168),  # 20 LEFT_WRIST      POOR - pose-dependent, needs more data (held-out residual 50.8 deg)
    (+0.909894812, -0.003827321, -0.036566597, -0.413206567),  # 21 RIGHT_WRIST     POOR - pose-dependent, needs more data (held-out residual 37.9 deg)
    (-0.000000002, -0.000000001, -0.000000001, +1.000000000),  # 22 LEFT_HAND       good (held-out residual 0.0 deg)
    (+0.000000013, +0.000000002, +0.000000004, +1.000000000),  # 23 RIGHT_HAND      good (held-out residual 0.0 deg)
)

# Joints whose correction did not generalise to the held-out pose. Treat their
# output as unreliable until better data is captured.
LOW_CONFIDENCE_JOINTS = (17, 18, 20, 21)
