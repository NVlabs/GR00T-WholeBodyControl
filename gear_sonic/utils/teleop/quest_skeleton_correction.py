"""Per-joint orientation correction mapping Quest body pose onto Pico conventions.

The CloudXR client SDK converts the Meta Quest IOBT skeleton into the ByteDance
24-joint layout before it reaches this machine (see IsaacTeleop
docs/source/device/body_tracking.rst:209-225). Joint order and positions survive
that conversion -- the LOVR body_tracking sample, which renders positions only,
looks anatomically correct on both headsets. The orientations do not, and
orientation is what drives the robot: compute_from_body_poses() builds the
whole SMPL pose from the 24 quaternions and uses positions only for root
translation.

Applied as::

    corrected_local[j] = CORRECTION[j].inv() * local_rots[j]

Frame note (important)
----------------------
compute_from_body_poses post-multiplies every global rotation by Ry(180 deg)
*before* forming parent-relative rotations, so its local frame is the raw
skeleton's local frame conjugated by Ry180. The values below are already
conjugated to match, i.e. stored as Ry180^-1 * offset * Ry180. Fitting an
offset from raw captures and applying it without this conjugation makes the
result *worse than no correction at all*; that mistake cost a full round of
this analysis.

Joint 0 is special: it is not parent-relative, so its correction is the plain
global offset G_quest * G_pico^-1 with no conjugation.

Derivation
----------
Fitted across 8 held static poses captured on both headsets with
gear_sonic/scripts/capture_body_skeleton.py --batch (tpose, arms_forward,
arms_forward_palms_up, goalpost, arms_overhead, bend_forward, lean_right,
sitting). All 16 captures were static (max orientation spread 2.6 deg) and
100% valid.

Validated leave-one-pose-out -- fit on 7 poses, evaluate on the held-out 8th.
Mean SMPL joint error versus the Pico reference:

    raw (no correction)   599 mm
    previous T-pose table 319 mm
    this table            216 mm      <- 64% better than raw, 32% better than v1

v4 wins on all 8 held-out poses individually, so the improvement is not driven
by one lucky pose.

Known limitations
-----------------
LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_HAND remain pose-dependent: their measured offset changes
by more than 30 deg between poses, so a single constant correction cannot fix
them. Legs, spine and head are well corrected and are what matter for balance.

The root correction includes yaw. During capture the two headsets reported
consistently different pelvis facing (Pico ~6 deg, Quest ~88 deg, each stable to
within ~10 deg across all 8 poses), so the ~80 deg difference is a device frame
convention rather than how the operator stood. Removing yaw made absolute error
much worse (722 mm vs 190 mm). If a future session shows a different offset,
re-derive it -- and note the teleop heading calibration (A+B+X+Y) can absorb a
residual yaw error in practice.

Bone lengths also differ between the two skeletons (Quest reports a generic
left-right-symmetric rig: 456.6 mm thighs and 456.5 mm shins on both sides,
versus the Pico's measured 312/336 mm). This correction does not address that;
it affects position-derived paths such as the 3-point VR pose, not the SMPL
orientation path.
"""

# (x, y, z, w) per joint, index-aligned with BodyJointPico / XR_BD_body_tracking.
# Already conjugated into compute_from_body_poses()'s local frame -- see above.
QUEST_TO_PICO_LOCAL = (
    (+0.531511133, -0.499491897, +0.440970321, -0.523019059),  #  0 PELVIS          root: full offset incl. yaw (see module docstring)
    (-0.079295146, -0.002681245, -0.993418055, -0.082617546),  #  1 LEFT_HIP        fair (LOO 23 deg)
    (-0.004139763, -0.025636928, -0.045382868, +0.998632067),  #  2 RIGHT_HIP       fair (LOO 21 deg)
    (-0.100928027, -0.011654117, -0.012059629, +0.994752371),  #  3 SPINE1          good (LOO 10 deg)
    (-0.043516648, +0.039249879, +0.147524563, +0.987320744),  #  4 LEFT_KNEE       fair (LOO 15 deg)
    (-0.073065749, -0.039937393, +0.113134378, +0.990084347),  #  5 RIGHT_KNEE      good (LOO 14 deg)
    (+0.007002402, +0.036207204, +0.023412081, +0.999045484),  #  6 SPINE2          good (LOO 11 deg)
    (+0.110821106, +0.148205179, +0.609676504, +0.770745397),  #  7 LEFT_ANKLE      good (LOO 10 deg)
    (+0.004862611, +0.263269959, +0.604213870, +0.752057766),  #  8 RIGHT_ANKLE     good (LOO 6 deg)
    (+0.014192559, +0.004335149, -0.009036562, +0.999849048),  #  9 SPINE3          good (LOO 6 deg)
    (-0.084357315, +0.064515496, +0.008908455, +0.994304900),  # 10 LEFT_FOOT       good (LOO 2 deg)
    (-0.095860252, +0.064373343, -0.015002456, +0.993197770),  # 11 RIGHT_FOOT      good (LOO 4 deg)
    (-0.128984529, +0.028894101, +0.121059739, +0.983805195),  # 12 NECK            good (LOO 10 deg)
    (+0.140448531, -0.591286424, -0.109875850, +0.786499760),  # 13 LEFT_COLLAR     good (LOO 13 deg)
    (-0.706184600, -0.109284392, -0.677467829, -0.174348998),  # 14 RIGHT_COLLAR    good (LOO 11 deg)
    (+0.072674718, +0.009003647, -0.215775097, +0.973693189),  # 15 HEAD            good (LOO 11 deg)
    (-0.511029137, -0.114737517, +0.489203094, +0.697398635),  # 16 LEFT_SHOULDER   POOR - pose-dependent (LOO 44 deg)
    (+0.262921867, -0.586018338, +0.286641822, +0.710838284),  # 17 RIGHT_SHOULDER  POOR - pose-dependent (LOO 73 deg)
    (+0.030672656, +0.244672317, +0.108546398, +0.963022494),  # 18 LEFT_ELBOW      fair (LOO 27 deg)
    (+0.231834406, -0.134440203, +0.360043518, +0.893614741),  # 19 RIGHT_ELBOW     fair (LOO 29 deg)
    (+0.694337820, -0.011242862, -0.118138216, +0.709797120),  # 20 LEFT_WRIST      fair (LOO 29 deg)
    (-0.737611586, +0.012842591, +0.153320291, -0.657462626),  # 21 RIGHT_WRIST     fair (LOO 19 deg)
    (-0.000000001, -0.000000000, +0.000000005, +1.000000000),  # 22 LEFT_HAND       good (LOO 0 deg)
    (-0.990442079, -0.068628293, +0.102554220, +0.061622049),  # 23 RIGHT_HAND      POOR - pose-dependent (LOO 32 deg)
)

# Joints whose offset varies by >30 deg across poses; a constant correction
# cannot fully fix them. Treat their output as unreliable.
LOW_CONFIDENCE_JOINTS = (16, 17, 23)
