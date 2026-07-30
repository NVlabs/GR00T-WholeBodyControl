"""Two-sided per-joint correction mapping Quest body pose onto Pico conventions.

The CloudXR client SDK converts the Meta Quest IOBT skeleton into the ByteDance
24-joint layout before it reaches this machine (IsaacTeleop
docs/source/device/body_tracking.rst:209-225). Joint order and positions survive
that conversion; the orientations do not, and orientation is what drives the
robot -- ``compute_from_body_poses()`` builds the entire SMPL pose from the 24
quaternions and uses positions only for root translation.

Why two-sided
-------------
Each joint needs BOTH a left and a right multiply::

    local_quest = A * local_pico * B          (fit)
    corrected   = LEFT.inv() * local * RIGHT   (applied; LEFT = A, RIGHT = B.inv())

A is the parent-frame rotation: it decides which axis a motion comes out about.
B is the child-frame relabelling: it decides the joint's rest orientation. They
do different jobs and one rotation cannot do both. Every earlier attempt used a
left multiply only, and each failed in a way that static metrics could not see:

* a single fitted rotation on the root kept the body upright but permuted the
  axes -- operator yaw became robot pitch, toppling the robot within ~30 deg of
  turn;
* setting that root to identity restored the axis mapping but left B
  uncorrected, so SMPL was told the body was tilted ~90 deg and the legs
  collapsed on connect;
* the spine kept a left-only correction and turned operator waist flexion into
  robot lateral bend -- measured 87 deg between the axis the operator rotated
  about and the axis the robot rotated about.

Separating A from B requires each joint to rotate about at least TWO independent
axes across the captures. With one axis the fit still matches every static pose
while reproducing motion about the wrong one, which is exactly how these bugs
survived every static metric. The torso battery was added for that reason;
SPINE1's axis diversity went from 0.08 to 0.58 and the pelvis from a fixed
heading to four.

Fitted across 26 held poses (main + shoulder + torso batteries) by alternating
SVD. Median static residual 8 deg.

Motion-axis error -- the angle between the axis the operator rotates about and
the axis the corrected result rotates about, over all pose pairs with >30 deg of
real motion. This is the metric that matters and the one previous versions got
wrong:

    joint        left-only   two-sided
    SPINE1            87.0         7.7
    LEFT_HIP          87.6         8.5
    RIGHT_HIP         85.9         7.7
    LEFT_KNEE        103.9         3.8
    RIGHT_KNEE        79.2         2.4
    LEFT_SHOULDER     50.4        11.4
    RIGHT_SHOULDER   107.6        12.8
    elbows        71 / 102    30 / 29
    wrists         27 / 17    14 / 14

Mean SMPL joint error vs the Pico reference: 629 mm raw, 168 mm with the
previous left-only table, 127 mm here.

Root safety
-----------
The pelvis A came out at 4.2 deg -- near identity -- from an unconstrained fit.
That independently confirms the physical argument that both devices agree on
gravity, so their world frames can differ only in heading. It also means the
axis mapping is safe. ``load_skeleton_correction()`` re-checks this at startup
and refuses to run if a future refit permutes the axes.

Known limitations
-----------------
RIGHT_HAND keeps a one-sided correction: the two-sided fit made its motion-axis
error worse (14 -> 64 deg) with no SMPL benefit.

SPINE2, SPINE3, ankles, feet, collars and LEFT_HAND barely rotate across the
captures (<15 deg range). Their corrections are fitted from statics only. That
carries no axis-mapping risk -- a joint that does not move has no motion axis to
get wrong -- but their values are less well determined than the rest.

Quest infers leg pose from vision rather than tracking it, so leg data is
inherently less reliable than the Pico's tracker-based legs. Measured residual
after correction is 6.9 deg median overall but 12.4 deg on leg-bending poses,
with RIGHT_HIP worst at 27 deg during a deep squat. Adequate for standing and
moderate motion; treat deep leg flexion as approximate.

Wrist range -- a separate, shared limitation
-------------------------------------------
The wrist battery showed that the commanded G1 wrist PITCH barely responds to
wrist flexion, on BOTH headsets. A full flex-to-extend sweep rotates the raw
wrist joint by 132 deg (Pico) and 104 deg (Quest), but the commanded pitch moves
only ~25 deg, and it is not even monotonic on the Pico.

The cause is in the retargeting, not in this table. In the arm configuration an
operator uses for this (forearms forward, elbows at 90), the flexion shows up
almost entirely in the Z euler component of the SMPL wrist rotation --
Pico left wrist Z runs -58 -> +3 -> +73 across the sweep, a 131 deg span, while
Y spans only 19 deg. ``pico_manager_thread_server.py`` takes Y as wrist pitch and
routes Z to wrist YAW, so operator flexion drives the robot's yaw channel and
pitch sees almost nothing.

That affects the Pico path identically and is a decision about shared, long-tuned
retargeting code, so it is deliberately not changed here. Fixing it means
choosing the wrist axis per arm configuration rather than assuming a fixed euler
component, since the euler decomposition is configuration-dependent.

Bone lengths also differ (Quest reports a generic left-right-symmetric rig:
456.6 mm thighs and 456.5 mm shins, versus the Pico's measured 312/336 mm). This
correction does not address that; it affects position-derived paths such as the
3-point VR pose, not the SMPL orientation path.
"""

# corrected_local[j] = QUEST_TO_PICO_LEFT[j].inv() * local_rots[j] * QUEST_TO_PICO_RIGHT[j]
# Both are (x, y, z, w), index-aligned with BodyJointPico / XR_BD_body_tracking,
# and already expressed in compute_from_body_poses()'s Ry180 local frame.
# The trailing comment is the measured motion-axis error for that joint.

QUEST_TO_PICO_LEFT = (
    (-0.027511403, +0.023796476, +0.003647535, +0.999331550),  #  0 PELVIS          axis 7d
    (-0.044720703, +0.754457857, +0.106548888, -0.646096537),  #  1 LEFT_HIP        axis 8d
    (-0.034870528, -0.688311460, +0.105131198, +0.716909207),  #  2 RIGHT_HIP       axis 8d
    (-0.297136514, -0.640326094, -0.232699328, +0.668986852),  #  3 SPINE1          axis 8d
    (-0.091837609, +0.753800306, -0.000469295, -0.650654080),  #  4 LEFT_KNEE       axis 4d
    (-0.066386953, -0.663670653, +0.001421598, +0.745071820),  #  5 RIGHT_KNEE      axis 2d
    (-0.231098605, +0.450775298, +0.236681421, +0.829082005),  #  6 SPINE2          no motion
    (-0.193325733, -0.292543903, +0.473174574, +0.808176372),  #  7 LEFT_ANKLE      no motion
    (+0.080666313, -0.130154103, -0.250168677, +0.956016992),  #  8 RIGHT_ANKLE     no motion
    (-0.193829968, +0.291095981, +0.063504749, +0.934697930),  #  9 SPINE3          no motion
    (+0.000777823, -0.000129299, -0.002315641, +0.999997008),  # 10 LEFT_FOOT       no motion
    (-0.000932713, -0.003214431, +0.000214648, +0.999994376),  # 11 RIGHT_FOOT      no motion
    (+0.689556031, +0.396941447, +0.409005923, -0.446837916),  # 12 NECK            axis 20d
    (+0.000000128, +0.000000065, +0.000000046, +1.000000000),  # 13 LEFT_COLLAR     no motion
    (-0.000000048, +0.000000009, -0.000000005, +1.000000000),  # 14 RIGHT_COLLAR    no motion
    (+0.170184319, -0.537314121, +0.001000802, +0.826032585),  # 15 HEAD            axis 41d
    (+0.685903511, +0.430567306, -0.286984845, -0.511652097),  # 16 LEFT_SHOULDER   axis 11d
    (+0.524406506, +0.313247779, +0.418056809, +0.672385418),  # 17 RIGHT_SHOULDER  axis 13d
    (-0.596378466, +0.201590155, -0.000568601, +0.776977355),  # 18 LEFT_ELBOW      axis 30d
    (+0.794222492, +0.180394750, +0.001498366, +0.580229370),  # 19 RIGHT_ELBOW     axis 29d
    (-0.673952121, +0.167754280, -0.061302910, +0.716860512),  # 20 LEFT_WRIST      refit on wrist battery
    (+0.680723217, -0.096405995, -0.189724830, +0.700946699),  # 21 RIGHT_WRIST     refit on wrist battery
    (-0.000000000, +0.000000000, +0.000000000, +1.000000000),  # 22 LEFT_HAND       no motion
    (-0.990442079, -0.068628293, +0.102554220, +0.061622049),  # 23 RIGHT_HAND      no motion  [one-sided]
)

QUEST_TO_PICO_RIGHT = (
    (-0.487294213, -0.519786601, -0.477663283, +0.514007809),  #  0 PELVIS          axis 7d
    (-0.764142299, -0.008799327, -0.642202808, -0.059872129),  #  1 LEFT_HIP        axis 8d
    (+0.016948408, -0.688923455, +0.013346544, +0.724513005),  #  2 RIGHT_HIP       axis 8d
    (-0.247250232, -0.643653605, -0.199053185, +0.696387241),  #  3 SPINE1          axis 8d
    (+0.041662636, -0.792326356, -0.045966340, +0.606935141),  #  4 LEFT_KNEE       axis 4d
    (-0.095325615, -0.655243992, -0.029304282, +0.748805446),  #  5 RIGHT_KNEE      axis 2d
    (-0.199329899, +0.444560502, +0.210752738, +0.847476746),  #  6 SPINE2          no motion
    (-0.471818974, -0.141432022, -0.225134456, +0.840653505),  #  7 LEFT_ANKLE      no motion
    (-0.109134215, -0.325263745, -0.767960190, +0.540860763),  #  8 RIGHT_ANKLE     no motion
    (-0.219801081, +0.295846962, +0.076974510, +0.926410808),  #  9 SPINE3          no motion
    (+0.091276303, -0.066270899, +0.001803263, +0.993616401),  # 10 LEFT_FOOT       no motion
    (+0.092176492, -0.067892502, +0.027139606, +0.993054653),  # 11 RIGHT_FOOT      no motion
    (-0.665550260, -0.364782028, -0.461041332, +0.459801929),  # 12 NECK            axis 20d
    (-0.176315143, +0.620992538, +0.115300653, +0.754974832),  # 13 LEFT_COLLAR     no motion
    (-0.597692152, -0.169738088, -0.765402789, +0.167665271),  # 14 RIGHT_COLLAR    no motion
    (+0.202281237, -0.513752587, +0.157357361, +0.818766903),  # 15 HEAD            axis 41d
    (-0.462741695, -0.268396624, +0.071467792, +0.841858498),  # 16 LEFT_SHOULDER   axis 11d
    (-0.848561982, -0.096878563, -0.239914256, -0.461517341),  # 17 RIGHT_SHOULDER  axis 13d
    (-0.577485645, +0.245245978, -0.004380891, +0.778681930),  # 18 LEFT_ELBOW      axis 30d
    (-0.755959322, -0.172070047, +0.052891017, -0.629380603),  # 19 RIGHT_ELBOW     axis 29d
    (-0.997600287, -0.031760763, -0.059256840, -0.016539312),  # 20 LEFT_WRIST      refit on wrist battery
    (-0.080226020, -0.080786116, +0.029673977, +0.993054301),  # 21 RIGHT_WRIST     refit on wrist battery
    (+0.000000001, +0.000000001, -0.000000000, +1.000000000),  # 22 LEFT_HAND       no motion
    (+0.000000000, +0.000000000, +0.000000000, +1.000000000),  # 23 RIGHT_HAND      no motion  [one-sided]
)

# Joints fitted from static poses only (<15 deg of rotation across all captures),
# so less well determined than the rest. No axis-mapping risk: they do not move.
STATIC_ONLY_JOINTS = (6, 7, 8, 9, 10, 11, 13, 14, 22)

# Quest infers legs from vision; degrades on deep flexion (see docstring).
INFERRED_LEG_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)


# ---------------------------------------------------------------------------
# Wrist bias, radians, added to the commanded G1 wrist joints as
# (roll, pitch, yaw) per side.
#
# Measured at `01_wrist_neutral` -- forearms forward, elbows 90, wrists straight
# and relaxed -- where all three channels should read the same on both headsets.
# They did not:
#
#     channel     Pico     Quest    bias applied
#     L roll     -33.6     -24.5        -9.37
#     L pitch     +7.2      +7.1       +11.44
#     L yaw      +15.2     +44.1       -29.88
#     R roll     +22.3     +19.7        +2.74
#     R pitch     +9.4      +9.4       +16.87
#     R yaw      -19.9     -48.2       +28.13
#
# YAW carried the visible error, not pitch. An earlier pitch-only bias was
# already exact (delta -0.0 and +0.1 deg) yet the robot's hands still sat
# visibly extended at neutral, because roughly 29 deg of error per side was
# arriving through the yaw channel. That is consistent with the routing noted
# under "Wrist range" below: operator wrist flexion largely lands in the yaw
# channel rather than pitch.
#
# The pitch entries here are unchanged in effect from the previous pitch-only
# table; roll and yaw are new.
#
# Pico is zero by intent, not oversight. Its teleop has been tuned by people
# over a long time, and its non-zero neutral may be correct for the robot's
# mechanical neutral rather than an error; this data cannot distinguish those.
# The Pico path is bit-identical to before.
#
# This is a BIAS at neutral. It does not change how much the wrist travels --
# see "Wrist range" for why the usable range is separately limited, and why
# that limit lives in shared retargeting code rather than here.
WRIST_BIAS_RAD = {
    # (roll, pitch, yaw) per side
    "pico": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),   # intentionally untouched
    "quest": ((-0.163531, +0.199678, -0.521456),
              (+0.047809, +0.294511, +0.490959)),
}
