"""Pose-dependent shoulder correction for Quest body tracking.

The constant per-joint correction in :mod:`quest_skeleton_correction` fixes
legs, spine, collars and wrists well, but leaves both shoulders pose-dependent
(leave-one-out residual 50 and 69 deg). Several explanations were tested against
14 held poses and ruled out:

* not a collar/shoulder redistribution -- the composite is no more stable than
  the shoulder alone;
* not a function of arm elevation;
* not shoulder axial roll -- an elbow-at-side roll trio varied the offset by
  only 15-22 deg, no more than measurement noise;
* not recoverable from positions -- each device's shoulder quaternion already
  points exactly at its own elbow (0.0 deg bone-axis spread), so substituting
  positions for the quaternion is provably a no-op. Measured: 190 mm either way.

What the offset *is*: structured and repeatable enough to interpolate. Repeating
three poses in a second session reproduced the shoulder offset to ~21 deg,
versus ~6 deg for every other joint. That 21 deg is the noise floor; the
pose-dependence it has to explain is 50-69 deg, i.e. much larger than the noise.

So the offset is stored as samples and blended by arm direction at runtime.
Leave-one-pose-out, median residual:

    joint       constant   blended
    LEFT_SHOULDER   50.1      21.0   <- at the noise floor
    RIGHT_SHOULDER  68.7      28.8

Elbows are deliberately excluded: blending did not help them (37.4 -> 37.3 and
42.5 -> 35.3), and their apparent error is largely inherited from the shoulder
above them.

Each entry is (arm_direction, offset_quat) where arm_direction is the
shoulder->elbow unit vector expressed in the pelvis frame (computable at runtime
from Quest data alone) and offset_quat is xyzw, already conjugated into
compute_from_body_poses's Ry180 local frame.

Blending uses Gaussian weights over the angle between the query direction and
each sample direction. SIGMA_DEG was chosen by leave-one-pose-out. It is
deliberately a smooth blend rather than nearest-neighbour: a hard switch would
step the correction discontinuously and jerk the robot.
"""

# Angular bandwidth of the Gaussian blend, degrees.
SIGMA_DEG = 20.0

# If every sample is further than this from the query the arm is outside the
# captured range; fall back to the constant correction rather than extrapolate.
MAX_ANGLE_DEG = 75.0

LEFT_SHOULDER_SAMPLES = (
    ((+0.147477, -0.061718, -0.987138), (-0.544749388, -0.259887582, +0.060430457, +0.795018685)),
    ((+0.032614, +0.999019, +0.029967), (-0.657560707, +0.141311306, +0.535269064, +0.511010822)),
    ((+0.516562, +0.855640, +0.032305), (-0.547794642, -0.025698277, +0.591544086, +0.591046719)),
    ((-0.123292, +0.188873, -0.974231), (-0.337795780, -0.094034193, +0.323383837, +0.878905271)),
    ((-0.895029, +0.365039, -0.256260), (-0.654809093, +0.075876624, +0.064713369, +0.749186205)),
    ((+0.938159, -0.234008, -0.255145), (-0.297798697, -0.129987754, +0.685116585, +0.651946611)),
    ((+0.806046, -0.188978, -0.560871), (-0.313053084, -0.462832136, +0.640364294, +0.526989328)),
    ((+0.990036, +0.140786, +0.002985), (-0.473483171, -0.118383612, +0.704398991, +0.515384389)),
    ((+0.063957, +0.081188, -0.994645), (-0.291813016, -0.198118006, +0.193667239, +0.915471147)),
    ((+0.029940, +0.998276, +0.050484), (-0.726036315, +0.083503982, +0.441950593, +0.520171152)),
    ((+0.038713, +0.246734, -0.968310), (-0.348260678, -0.232197376, +0.303902333, +0.855828400)),
    ((+0.974203, +0.094661, -0.204863), (-0.315994693, -0.392142612, +0.738370849, +0.448530953)),
    ((+0.964418, +0.147682, -0.219290), (-0.255884248, -0.357201916, +0.800512180, +0.407566304)),
    ((+0.855254, -0.262019, -0.447087), (-0.264783061, -0.433489972, +0.435492797, +0.743183960)),
)

RIGHT_SHOULDER_SAMPLES = (
    ((+0.109338, +0.038352, +0.993265), (+0.488376155, +0.206907284, -0.057876109, -0.845770928)),
    ((+0.069576, +0.995393, +0.065973), (+0.346796614, -0.586911595, +0.627256752, +0.376584460)),
    ((+0.506714, +0.862108, +0.003127), (+0.407865225, -0.771962373, +0.313809910, +0.373153310)),
    ((-0.032500, +0.134683, +0.990356), (-0.279308070, -0.095051743, +0.216004276, +0.930749333)),
    ((-0.878743, +0.362497, +0.310494), (+0.240682860, +0.025290771, -0.904288059, -0.351703347)),
    ((+0.924848, -0.246402, +0.289728), (+0.496489132, -0.649847143, +0.008495412, +0.575434671)),
    ((+0.999388, -0.017151, +0.030496), (+0.433279537, -0.391256811, -0.183722692, +0.790843172)),
    ((+0.992380, +0.068197, +0.102624), (+0.485607387, -0.703419191, +0.194036888, +0.481390271)),
    ((+0.093756, +0.107250, +0.989802), (-0.239715010, -0.284965346, +0.320335451, +0.871043435)),
    ((+0.017209, +0.999680, -0.018530), (+0.235867887, -0.585300295, +0.716815089, +0.296590683)),
    ((+0.069834, +0.106106, +0.991900), (-0.313941382, -0.249433150, +0.319278766, +0.858653004)),
    ((+0.941982, +0.161810, +0.294087), (+0.426524883, -0.508628405, -0.141296969, +0.734444577)),
    ((+0.920881, +0.137236, +0.364890), (+0.474839694, -0.450662666, -0.125705204, +0.745405009)),
    ((+0.865074, -0.259895, +0.429071), (+0.189752203, -0.539680253, -0.288681720, +0.767725336)),
)

ADAPTIVE_JOINTS = {16: LEFT_SHOULDER_SAMPLES, 17: RIGHT_SHOULDER_SAMPLES}

# (joint, child) used to build the arm-direction feature at runtime.
FEATURE_BONE = {16: 18, 17: 19}
